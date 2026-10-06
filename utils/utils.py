import math
import os
import pickle

from math import trunc
from pathlib import Path
from typing import Any, Literal, Sequence, cast, overload

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from datasets import Dataset, load_from_disk
from numpy.typing import NDArray
from PIL import Image, ImageOps
from safetensors.torch import save_file
from syntheticpatterns import show_grid
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

#### PATCH PATTERN CLEANING #######
class Clean_patterns:
    """Clean noisy patch predictions from a (soft) permutation module 
    against a known codebook of valid patterns, and recover the inputs true 
    intensity values for any prediction that matches a input's patch after 
    thresholding. `cnn_patches` is thresholded and binarized per-patch, 
    patches that don't look confidently binary are discarded, of these 
    any pattern not matching `valid_patterns` are also discarded. Finally 
    surviving patches have their true intensities substituted with those 
    from `initial_tensor`.

    Note:
    -----
    It is assumed that the `initial_tensor` is composed of N unique patterns
    in a given sequence and `valid_patterns` is a codebook built from those same 
    N patterns using the same sequence.

    Attributes:
        cnn_patches (torch.Tensor): Decoder output patches [B, M, P, P] to clean 
        initial_tensor (torch.Tensor): The original, true-intensity input tensor the
            decoder permuted, shape (B, M, P, P), values in [0, 1]. Its M
            patches are all unique patterns and are assumed to correspond
            position-for-position with `valid_patterns` (see class-level
            note above).
        valid_patterns (torch.Tensor): Codebook of known valid binary patterns [1, N, P, P] with binary (0/1)
            intensities.
        threshold_ratio (None|float): if float, multiplies the max intenisty 
            of a given patche's pixel with this threshold. Any pixels intensity below this threshold is gets masked. 
        binary_error_ratio (float): Maximum allowed average distance from 0 or 1 that patch intensities 
            can diverge (How far away can the average pixel be from 0s or 1s in a given patch).
    """
    def __init__(self,
            cnn_patches:torch.Tensor,
            input_tensor:torch.Tensor,
            pattern_book:torch.Tensor,
            threshold_ratio:float|None=None, 
            binary_error_ratio:float=0.37
            ) -> None:
        """
        Args:
            cnn_patches (torch.Tensor): Decoder output patches [B, M, P, P] to clean 
            initial_tensor (torch.Tensor): The original, true-intensity input tensor the
                decoder permuted, shape (B, M, P, P), values in [0, 1]. Its M
                patches are all unique patterns and are assumed to correspond
                position-for-position with `valid_patterns` (see class-level
                note above).
            valid_patterns (torch.Tensor): Codebook of known valid binary patterns [1, N, P, P] with binary (0/1)
                intensities.
            threshold_ratio (float|None): if float, multiplies the max intenisty 
                of a given patche's pixel with this threshold. Any pixels intensity below this threshold is gets masked.
            binary_error_ratio (float): Maximum allowed average distance from 0 or 1 that patch intensities 
                can diverge (How far away can the average pixel be from 0s or 1s in a given patch).
        """
        self.cnn_patches:torch.Tensor=cnn_patches
        self.initial_tensor:torch.Tensor=input_tensor
        self.valid_patterns:torch.Tensor=pattern_book
        self.threshold_ratio:float|None=threshold_ratio
        self.binary_error_ratio:float=binary_error_ratio
        
        
    def scale_and_binarize_patches(self) -> torch.Tensor:
        """
        Binarizes each predicted patch of a tensor shape [B,N,P,P] and zeros out any that don't look
        confidently binary according to set thresholds.
        
        Patches where the average pixel intensities is further away from 1 or 0 than 
        `binary_error_ratio` are removed and the remaining are binarized in  `P`. 
        Optionnally if `threshold_ratio` a float, individual pixels within
        the remaining patches that fall below `threshold_ratio` of that patch's
        own max intensity are also zeroed out.

        Returns:
            torch.Tensor: tensor of shape [B, M, P, P] with binary values.
        """

        # maximum value inside each patch
        patch_max = self.cnn_patches.amax(dim=(-1, -2), keepdim=True)

        # pixels above a fraction of their own patch maximum
        active_mask:None|torch.Tensor = None

        if self.threshold_ratio is not None:
            active_mask = self.cnn_patches >= patch_max * self.threshold_ratio
        
        # how close pixels are to 0 or 1
        binary_error = torch.minimum(
            self.cnn_patches,
            1 - self.cnn_patches
        ).mean(dim=(-1, -2), keepdim=True)

        # keep patches where patch pixels are less noisy between each other (drop blurry patches)
        mask = binary_error <= self.binary_error_ratio

        # final patch mask, keep patches where pixels are not too distant from the max intensity
        # and with less average noise
        # get the minimum intensity per patch
        patch_min = self.cnn_patches.amin(dim=(-1, -2), keepdim=True)
        # scale each patch independently according to min and max, then binarize and mask
        # patches/P shape values that did not pass thresholding
        scaled = (self.cnn_patches - patch_min) / (
            patch_max - patch_min + 1e-8
        )

        binary = (scaled >= 0.5).float()

        binary = binary * mask.float()

        if active_mask is not None:
            binary = binary * active_mask
        
            
        return binary

    def match_and_fetch_intensities(self) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Match each binarized patch against the codebook and fetch the
        corresponding true intensity patch for any that match.

        Looks up each patch of `cnn_patches` and compares it each codebook pattern 
        as a single integer "fingerprint" by treating the flattened P*P binary pattern 
        as a bitstring. A matched patch's true intensities are then fetched from
        `initial_tensor` at the matched codebook index.
        
        Note:
        -----
        It assumes that the `initial_tensor` and `pattern_book` have the same
        patterns sequence.
        
        Also requires P*P <= 64, since each flattened binary pattern is
        packed into a single int64 via bit-shifting (`1 << arange(P*P)`)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                matched_intensities [B, M, P, P]: 
                    - True intensity patch for every matched prediction. All-zero for any
                      patch that didn't pass thresholding or matched matched any in the codebook.
                safe_indices [B, M]: 
                    - The matched codebook/position index for each patch, with unmatched
                      patch indices clamped to 0 (not necessarily the real pattern 0).
        """

        device = self.cnn_patches.device
        B, M, P, _ = self.cnn_patches.shape

        # Threshold and binarize decoder patches
        patches:torch.Tensor = self.scale_and_binarize_patches()
        
        # Ensure integer representation
        patches_long = patches.to(torch.long)
        valid_patterns_long = self.valid_patterns.to(torch.long).unsqueeze(0)

        # Flatten binary patterns
        patches_flat = patches_long.flatten(start_dim=2)          
        valid_flat = valid_patterns_long.flatten(start_dim=2)      

        # Encode binary patterns as integers
        weights = (1 << torch.arange(P * P, device=device, dtype=torch.int64))

        patch_keys = (patches_flat * weights).sum(dim=-1)       
        valid_keys = (valid_flat * weights).sum(dim=-1)            

        # Find matches
        matches = patch_keys.unsqueeze(-1) == valid_keys.unsqueeze(1)  

        valid_mask = matches.any(dim=-1)                           

        # Index of matched valid_pattern
        match_indices = matches.to(torch.int64).argmax(dim=-1)     

        # Mark invalid matches
        match_indices = match_indices.masked_fill(~valid_mask, -1)

        # Gather original intensity patches - clamp to 0 to avoid -1s 
        # later mask any 0th patch if not correct
        safe_indices = match_indices.clamp(min=0)

        matched_intensities = torch.gather(
            self.initial_tensor,
            dim=1,
            index=safe_indices[..., None, None].expand(-1, -1, P, P),
        )

        # Remove invalid patches (false 0s)
        matched_intensities = matched_intensities * valid_mask[..., None, None]
        
        return matched_intensities, safe_indices

    def clean(self)-> torch.Tensor:
        """
        wrapper that invokes `match_and_fetch_intensities` 
        and returns only the matched true intensity patches

        Returns:
            torch.Tensor [B,M,P,P]: 
                - True intensity patch for every matched prediction. All zero for any
                  patch that didn't pass thresholding or matched any in the codebook.
        """
        return self.match_and_fetch_intensities()[0]


####### DATALOADER & SAVE MODEL ############


def create_dataloaders(
    path: str,
    batch_size: int,
    split:list[str]=["train","test","val"],
    number_workers: int | None = None,
    prefetch_factor: int=4,
    shuffle:bool|None=True,
    pin_memory:bool=True,
    persistent_workers:bool =True,
    subset_dataset:None|int=None
) -> list[DataLoader[dict[str, torch.Tensor]]|list[str]]:
    """
    Build DataLoaders for each requested split of a dataset saved on the disk.
    Loads a `datasets`-style saved dataset from for each
    entry in `split`. Wraps each in a DataLoader. 
    Optionally subsets each split to a random subset of
    samples via a `SubsetRandomSampler`. Returns the DataLoaders plus the
    class label names as a single list.

    Args:
        path (str): Root directory containing one subdirectory per split (e.g.
            `path/train`, `path/test`, `path/val`)
        batch_size (int): Batch size to use in the DataLoader.
        split (list[str]): Names of the subdirectories under `path` to load, and the
            order in which their DataLoaders appear in the returned list.
        number_workers (int|None): Forwarded to ``DataLoader``. Number of DataLoader worker processes. If None,
            it is replaced with roughly half the available CPU cores (minimum 1).
        prefetch_factor (int): Forwarded to ``DataLoader``. Number of batches each worker prefetches ahead of
            time.
        shuffle (bool|None): Forwarded to ``DataLoader``. Whether to shuffle each split's DataLoader. Is ignored if using ``subset_dataset``
            as a subset is automatically shuffled.
        pin_memory (bool): Forwarded to ``DataLoader``. If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
        persistent_workers (bool): Forwarded to ``DataLoader``. Number of batches loaded in advance by each worker. Requires `number_workers > 0` to have any effect. 
        subset_dataset (None|int): Number of data samples to randomly select for training. For any other split, 20% of this number of samples is used. If None, each split uses the full dataset.

    Returns:
        list[DataLoader[dict[str, torch.Tensor]]|list[str]]: One DataLoader per entry in `split`,
        in the same order, followed by a final list[str] element containing
        the dataset's class label names

    Examples::

        train_loader, test_loader, validation_loader, class_names = create_dataloaders(
        path="./../dataset/",
        batch_size=36,
        split=["train", "test", "val"],
        number_workers=4,
        prefetch_factor=6,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        subset_dataset=6000,
        )
        # Get a single batch from the test loader
        batch = next(iter(test_loader))
        x_im = batch["image"]
        x_lab = batch["label"]

        # Print class names
        print(class_names)
        print(class_names[x_lab])
    """
    if number_workers is None:
        WORKERS=os.cpu_count()

        if WORKERS:
            number_workers=math.ceil(WORKERS/2) if WORKERS > 1 else 1
        else: 
            number_workers=1


    data_loaders:list[DataLoader[dict[str, torch.Tensor]]|list[str]]=[]
    class_names:list[str]|None=None

    for i in split:
        load_path=Path(path)/Path(i)

        try:
            loaded_set: Dataset = cast(Dataset, load_from_disk(load_path))
        except Exception as e:
            raise RuntimeError(f"Cannot load {load_path}, maybe a path error? Check error: {e}") from e
        
        class_names=cast(list[str], loaded_set.features["label"].names)

        sampler=None

        # Get a random subset
        if isinstance(subset_dataset, int):
            # use only 20% for train/validation
            if i != "train":
                subset_samples:int=trunc(subset_dataset*0.2)
            else:
                subset_samples=subset_dataset
            indices:torch.Tensor = torch.randperm(len(loaded_set))[:subset_samples]
            sampler = SubsetRandomSampler(cast(Sequence[int], indices))
            shuffle = None # sampler substitudes shuffle
            print(f"Used a subset with {subset_samples} random samples")



        data_loaders.append(DataLoader(
                                cast(torch.utils.data.Dataset, loaded_set),
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                num_workers= number_workers,
                                pin_memory=pin_memory,
                                prefetch_factor=prefetch_factor,
                                persistent_workers=persistent_workers,
                                sampler=sampler
                                )
        )

    if isinstance(class_names, list):

        data_loaders.append(class_names)
        return data_loaders
    else:
        raise ValueError(
            f"Could not determine class names for splits {split} under {path!r}. "
            f"Either `split` was empty, or the loaded dataset's "
            f"`features[\"label\"].names` was missing or not a list."
        )



def save_model(model:nn.Module, 
               safe_root:str|Path="models", 
               model_name:str="model_128_PizzaCNN", 
               compiled:bool=True, 
               stats:dict[str, Any] | None = None
               ) -> None:
    """
    Save a model's state dict and optionally training statistics to disk 
    by writing weights as a `.safetensors` file under `safe_root`/.

    Note:
    -----
    If the model was compiled using ``torch.compile(mymodel)``, it is important 
    to use ``compiled = True`` to save the model's uncompiled state dict. Otherwise 
    pytorch's `load_state_dict` will not load the model weights cleanly. 

    Args:
        model(nn.Module): Model
        safe_root (str|Path): Directory the model (and optional stats) are saved into;
            created if it doesn't already exist.
        model_name (str): Base filename (without extention). Will be safed under `safe_root/<model_name>.safetensors`. 

        compiled (bool): Whether `model` was compiled with `torch.compile()`.

        stats (dict[str, Any]): If `True`, saves a dictionnary (e.g. training metrics) as a pickled object. It's filename will be safed as (`<model_name>_statistics.pkl`).

    Returns:
        None. Only writes files to disk.
    """

    MODEL_PATH=Path(safe_root)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME=model_name+'.safetensors'

    MODEL_SAVE_PATH= MODEL_PATH/MODEL_NAME

    if compiled :
        model_uncompiled:nn.Module=model._orig_mod # type:ignore
        state=model_uncompiled.state_dict() 

        for k, v in state.items():
            state[k] = v.contiguous()

        save_file(state, MODEL_SAVE_PATH)

    else:
        save_file(model.state_dict(), MODEL_SAVE_PATH)

    if stats is not None:
        stat_name:str =model_name+'_statistics.pkl'
        with open(MODEL_PATH/stat_name, "wb") as f:  # "wb" = write binary
            pickle.dump(stats, f)
    print(f"Saved {model_name} at {MODEL_SAVE_PATH}")

    

############ LOSS CURVES AND ATTENTION PLOT #############

def batch_make_grid(x:torch.Tensor, pad:None|int=None, pad_value:int=0) -> torch.Tensor:
    """
    Reconstructs a sequence of patches into an image/grid.

    Takes a tensor shaped [B, N, P, P] (batch, Number, width/heights per patches per sample) and
    tiles them into an image/grid of shape [B, C, G*P, G*P] per batch, where
    G = sqrt(N).

    Args:
        x (torch.Tensor): Patch tensor of shape [B, N, P, P] — batch of B samples, each with
            N patches of size P x P. A channel dim of 1 is added internally.
        pad (None|int): Optional padding to add around each patch on all
            four sides before tiling. It visually separate patches in
            the grid. If None, no padding is applied.
        pad_value (int): Fill value used for the padding border.

    Returns:
        torch.Tensor: Grid image of shape [B, C, G*P, G*P], where C=1 and G = sqrt(N).
    """
    x = x.unsqueeze(2)          # [B, N, 1, P, P]

    B, N, C, P, _ = x.shape
    G = math.isqrt(N)     
    assert G * G == N  #  N should be a perfect square

    if pad!=None:
        # Pad each patch: (left, right, top, bottom)
        x = F.pad(x, (pad, pad, pad, pad), value=pad_value)

        P = P + 2 * pad

    # Arrange into grid
    x = x.view(B, G, G, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, C, G * P, G * P)

    return x

def insert_patch_gaps(img: NDArray[np.floating], patch_size:int=4, gap:int=1, gap_value:float=np.nan) -> NDArray:
    """
    Insert a gap around patches of a 2D image.

    Args:
        img (NDArray[np.floating]): 2D array of shape (H, W) to insert gaps into.
        patch_size (int): Side length of each square patch, in pixels.
        gap: Number of gap pixels inserted between adjacent patches.
        gap_value (float): Fill value used for the inserted gap pixels. Defaults to
            NaN, which most colormaps render as a distinct/transparent color.
    Note:
    -----
    `img` dtype needs to be a float

    Returns:
        np.ndarray: Array of patches containing the original patches with a `gap_value` filling the space between them.
    """
    H, W = img.shape
    nH = H + (H // patch_size - 1) * gap
    nW = W + (W // patch_size - 1) * gap

    assert np.issubdtype(img.dtype, np.floating), \
        f"img must have a floating-point dtype, got {img.dtype}"

    out: np.ndarray = np.full((nH, nW), gap_value, dtype=img.dtype)

    out_r = 0
    for r in range(0, H, patch_size):
        out_c = 0
        for c in range(0, W, patch_size):
            out[out_r:out_r+patch_size,
                out_c:out_c+patch_size] = img[r:r+patch_size, c:c+patch_size]
            out_c += patch_size + gap
        out_r += patch_size + gap

    return out


def show_attention(image:torch.Tensor, attn:torch.Tensor, idx:int=0, patch_size:int=4, gap:int=1,
                   alpha:float=0.45, cmap:str="jet", figsize:tuple[float,float]=(9,9), title:str|None=None, left_title:str="CNN output", right_title:str="Attention overlay") -> None:
    """
    Visualize an attention map overlaid on an image.

    Selects one sample from a batch, reassembles its patches into a full
    image via `batch_make_grid`, upsamples the per-patch attention weights
    to pixel resolution via nearest-neighbor interpolation, and displays the
    result as two side-by-side panels: the raw patch grid (left) and the
    image with the attention heatmap overlaid (right).

    Args:
        image (torch.Tensor): 
            Tensor of shape (B, N, P, P). A batch of square patch sequences (N patches of size
            P x P).
        attn ( torch.Tensor):
            Shape (B, 1, N) a per patch attention weights map.
        idx (int): 
            Batch index of the sample to visualize.
        patch_size (int):
            Side length of each square patch, in pixels.
        gap (int):
            Gap (in pixels) inserted between patches for visual separation,
            forwarded to `insert_patch_gaps`.
        alpha (float):
            Opacity of the attention heatmap overlay.
        cmap (str): 
            Colormap name used for the attention heatmap.
        figsize (tuple[float, float]):
            The plotted figure size for a single image. The the width is automatically doubled to adjust the
            two side-by-side panels.
        title (str | None):
            Optional figure title. Defaults to no title
        left_title : str
            Title for the (left) `img` panel.
        right_title (str):
            Title for the right attention overlay panel.

    Returns:
        None: Only displays the figure via `plt.show()`
    """
    # double the width to adjust size for a two panel figure.
    figsize=(figsize[0]*2,figsize[1])

    # make_grid expects [B,N,P,P]
    image=image[idx].unsqueeze(0).to(dtype=float).detach() # type:ignore
    img=batch_make_grid(image)[0,0].cpu().numpy()          
    # 
    attn=attn[idx].unsqueeze(0).to(dtype=float) # type:ignore

    

    # attention heatmap
    H, W = img.shape
    grid_h = H // patch_size
    grid_w = W // patch_size

    heatmap = F.interpolate(
        attn.view(1, 1, grid_h, grid_w),
        size=(H, W),
        mode="nearest"
    ).squeeze().detach().cpu().numpy()

    # add patch gaps to better visually distinglish the patch patterns
    img_vis = insert_patch_gaps(img, patch_size, gap)
    heatmap_vis = insert_patch_gaps(heatmap, patch_size, gap)

    # ----- plot -----
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Original image
    show_grid(image[0].detach(), show=False,ax=ax[0])
    ax[0].set_title(left_title)
    ax[0].axis("off")

    # Overlay
    ax[1].imshow(img_vis, cmap="Grays")
    ax[1].imshow(heatmap_vis, cmap=cmap, alpha=alpha)
    ax[1].set_title(right_title)
    ax[1].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


######## NODE SIZES AND LINKS ###########
class Calculate_nodes_links:
    """Builds a graph representation of the average patch pattern by calculating
    node sizes from the average attention per patch and link sizes from the number of 
    adjacent patches.

    Attributes:
        device (torch.device): 
            Device inferred from `cnn_patch`.

        available_pattern (torch.Tensor [N,P,P]): 
            Codebook of valid binary patterns cast to torch.float32
        
        cnn_patch (torch.Tensor, [B, M, P,P]): 
            Decoder output patches, detached and cast to torch.float32.
        
        num_patterns (int): Number of patterns in the codebook (N), read from
            `available_pattern`'s shape
        
        attn (torch.Tensor, [B, C, N]): 
            Attention weights for `cnn_patch`. Detached and cast to torch.float32.
    """
    def __init__(self,available_pattern, cnn_patch, attn) -> None:
        """
        Args:
            available_pattern (torch.Tensor [N,P,P]): Codebook of valid binary patterns.
            
            cnn_patch (torch.Tensor, [B, M, P,P]): Decoder output patches.

            attn (torch.Tensor, [B, C, N]): Attention weights for `cnn_patch` 

        """
        self.device:torch.device=cnn_patch.device
        self.available_pattern:torch.Tensor=available_pattern.to(device=self.device, dtype=torch.float)
        self.cnn_patch:torch.Tensor=cnn_patch.detach().to(device=self.device, dtype=torch.float)
        self.num_patterns:int=available_pattern.shape[0]
        self.attn:torch.Tensor=attn.detach().to(device=self.device, dtype=torch.float)

    def scale_and_binarize_patches_(self, threshold_ratio:float|None=None, binary_error_ratio:float=0.37
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Classify each predicted patch as confidently-binary. It only returns the raw
            binarized patches and a per-patch confidence flag.
        
        
            Args:
                threshold_ratio (float|None): float that multiplies the max intenisty of a given patche's 
                    value with this threshold. Any intensity below this threshold gets flagged to be
                    masked.

                binary_error_ratio (float): Maximum allowed average distance from 0 or 1 that patch intensities 
                        can diverge (how far away can the average pixel be from 0s or 1s in a given patch).
        
            Returns:
                tuple[torch.Tensor, torch.Tensor]:
                    binary [B,M,P,P]:
                        - Every patch binarized via a per-patch scaling, if a value is lower than 0.5 it becomes 0 else 1.
                    
                    confident_mask [B, M, 1,1] | [B,M,P,P]:
                        - True wherever a patch's `binary_error_ratio` (and optionnally a `threshold_ratio`) check passed 
            
        """
        patch_max = self.cnn_patch.amax(dim=(-1, -2), keepdim=True)
        patch_min = self.cnn_patch.amin(dim=(-1, -2), keepdim=True)

        # for then next version test patch_max instead of 1 to make it scale free, eg.: binary_error = torch.minimum(self.cnn_patch, 1 - patch.max)/(patch_max + 1e-8)
        binary_error = torch.minimum(self.cnn_patch, 1 - self.cnn_patch).mean(dim=(-1, -2), keepdim=True)
        binary_mask = binary_error <= binary_error_ratio

        if threshold_ratio is not None:
                    active_mask = self.cnn_patch >= patch_max * threshold_ratio
                    confident_mask=binary_mask & active_mask
        else:
            confident_mask=binary_mask

        scaled = (self.cnn_patch - patch_min) / (patch_max - patch_min + 1e-8)
        binary = (scaled >= 0.5).float()

        return binary, confident_mask

    def zero_duplicates(self, dupl_patches:torch.Tensor, valid_pattern_idx:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """Keep one highest attention patch per codebook ID per sample and zero the rest.

        Patches resolving to the same codebook ID are considered duplicates and therefore zeroed
        to avoid counting them multiple times per sample.

        Args:
            dupl_patches (torch.Tensor [B, M, A, P, P]): Pattern patches stacked with attention on dim 2
                where channel 0 = binarized patterns and channel 1 = attention expanded over P x P.
                N is the patch count per sample (the class docstring uses N for codebook size).
            
            valid_pattern_idx (torch.Tensor [B, M], int64): Codebook ID for each `dual_patches` patch.
            eps (float): float to multiply position, that will be added to each patch, should be a tiny value to make the addition negligable.

        Returns:
            torch.Tensor [B, M, A, P, P]: Same shape and dtype as `dupl_patches`. Gives back one patch (M) per codebook (N) ID with the maximum attention. Duplicates are zeroed out.
        """
        B,M,_,_,_=dupl_patches.shape


        attn_scalar = dupl_patches[..., 1, 0, 0]   # [B,M] attention score per patch
        # add position to their attention score
        # to make an ordering. Thus among equal attention, the later index wins (if eps is tiny).
        #
        position = torch.arange(M, device=self.device).float()
        atten_score = attn_scalar + (position*eps)

        
        # the full pattern codebook [N]
        num_patterns = self.available_pattern.shape[0]   
        # keep the attention for all patches per codebook ID, for duplicates, get the max attention only
        max_score_per_id = torch.zeros(B, num_patterns, device=self.device).scatter_reduce_(
        1, valid_pattern_idx, atten_score, reduce="amax", include_self=False
        )
        # get a bool mask to determine which patch index (M) had the most attention.
        is_winner = atten_score == torch.gather(max_score_per_id, 1, valid_pattern_idx)
        # Zero out non-winning duplicates and keep the highest scoring
        # patch for each pattern ID.
        dupl_patches_unique=dupl_patches * is_winner.view(B, M, 1, 1, 1)
        return dupl_patches_unique

    @overload
    def retrieve_exact_patches(self, binary_patches:torch.Tensor, return_with_no_match_idx: Literal[False] = False) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def retrieve_exact_patches(self, binary_patches:torch.Tensor, return_with_no_match_idx: Literal[True]) -> torch.Tensor: ...

    def retrieve_exact_patches(self, binary_patches: torch.Tensor, return_with_no_match_idx: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Match each binary patch to its exact codebook entry using integer bit-keys.

        Args:
            binary_patches (torch.Tensor [B, M, P, P]): Codebook to look up the patches with binary patterns.
            
            return_with_no_match_idx (bool): Selects the output convention (see Returns).

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                if `return_with_no_match_idx=False` (exact_idx, exact_mask):
                    - exact_idx [B, M] int64: codebook index of the matching pattern; 0 where nothing
                        matched. 0 indexes are mixed with no matches, use mask to distinguish.
                    - exact_mask [B, M] bool: True where the patch equals some codebook pattern exactly.
                
                if `return_with_no_match_idx=True` (exact_idx [B, M] int64): 
                    - with -1 where nothing matched instead of 0 (index 0 stays a genuine match). The mask is not returned.
        """
        P = self.available_pattern.shape[-1]
        patches_flat = binary_patches.to(torch.long).flatten(start_dim=2)                              # (B,M,P*P)
        valid_flat = self.available_pattern.to(torch.long).unsqueeze(0).flatten(start_dim=2)            # (1,N,P*P)

        weights = 1 << torch.arange(P * P, device=self.device, dtype=torch.int64)
        patch_keys = (patches_flat * weights).sum(-1)     # (B,M)
        valid_keys = (valid_flat * weights).sum(-1)        # (1,N)

        exact = patch_keys.unsqueeze(-1) == valid_keys.unsqueeze(1)   # (B,M,N)
        exact_mask:torch.Tensor = exact.any(-1)                                    # (B,M)

        if return_with_no_match_idx:
                    exact_idx:torch.Tensor = torch.where(exact_mask, exact.float().argmax(-1), torch.full_like(exact_mask, -1, dtype=torch.long))
                    return exact_idx
        else:
            exact_idx:torch.Tensor = exact.float().argmax(-1)

        
        return exact_idx ,exact_mask

    def expand_to_codebook(self, patches_unique:torch.Tensor, idx:torch.Tensor,  num_patterns:int=4096) -> torch.Tensor:
        """Scatters the patch patterns and their attention into the correct slot of the codebook length.
        
        Example: For a given patch [B,M,A,P,P], where A[0] is the binary pattern and A[1] is the corresponding attention,
        and for a codebook of length [N=4096,P,P], if M[0]A[0] corresponds to the pattern [252] in the codebook it will
        be put into a new tensor of length B,N,P,P where B[0]N[252]=B[0]M[0] 

        Note:
        -----
            If patches_unique do contain duplicates, they will be added together in [A], 
            attention becomes summed and may inflate attention for a given patch and patterns stop being binary. 
        Args:
            patches_unique (torch.Tensor [B, M, A, P, P]): Patch tensor without duplicates with patterns and
                attention stacked on dim 2 (0 = pattern, 1 = attention) with [M] the total number of
                patches in `self.cnn_patch`.
            idx (torch.Tensor [B, M], int64): The patch idx of shape [B,M] that match a given unique pattern [N] of the codebook
            num_patterns (int): Size of the output slot axis. ( = N = `self.available_pattern.shape[0]` ).

        Returns:
            torch.Tensor [B, num_patterns, A, P, P]: 
                Tensor of shape [B, num_patterns, A, P, P] containing the
                scattered patch patterns and attention [A]. The [M] patches from the
                input are placed into their corresponding codebook  slots of length 
                `num_patterns` according to idx. 
        """

        B, M, A, P, _ = patches_unique.shape
        
        dtype=patches_unique.dtype

        # create a our destination tensor 
        expanded = torch.zeros(B, num_patterns, A, P, P, device=self.device, dtype=dtype)
        # now create the idx used to scatter the patches to the codebook sequence 
        scatter_idx = idx.view(B, M, 1, 1, 1).expand(-1, -1, A, P, P)   # [B, M, A, P, P]
        # scatter each patch M to its position in N and add A together (patterns and attention).
        # eg. For a given attention score A[1] of patch M_5 = codebook N_0, M_3 = N_0,  M_0=N_1 ... M_356  ->  N_0=2, N_1=1, N_3...N_4096
        # Note: if binary patterns are duplicated, they will be added together and stop becoming {0:1}!
        expanded.scatter_add_(1, scatter_idx, patches_unique)
        return expanded

    def calculate_node_sizes(self, eps:float=1e-8):
        """Calculate the average normalized attention for each pattern found in the `cnn_patch` and return
         a tensor of shape [B,N,A,P,P] with N corresponding to the N index of the codebook `available_pattern`

        This function normalizes attention per sample from `cnn_patch`, "cleans", binarize the predicted patches and matches
        them to an exact codebook pattern (`available_pattern`) while it zeros out blanks/non-matches. 
        It only keeping the highest-attention pattern if there are duplicates. Finally the attention is then averaged over the entire batch. 

        Args:
            eps (float): Divide by zero guard for the per sample attention min-max normalization.

        Returns:
            torch.Tensor [`num_patterns`, A=2, P, P]: 
                - a tensor that encodes any pattern discovered in `cnn_patch` with their average attention per sample scattered
                  into the length [N] (=`num_patterns`) from `available_pattern` of shape [N,P,P].
                  A[0] = The binary pattern. Any patch that matched this pattern in `available_pattern` is binary, else the patch gets zeroed. 
                  A[1] = attention that was averaged across all batch samples.
        """
        B, _, M = self.attn.shape
        P = self.available_pattern.shape[-1]

        # Scaling per sample min-max to [0,1]. 
        # Cause raw attention values are tiny (softmax-like), 
        # so without this step a sample with uniformly low attention would look "weak" 
        # next to a sample with uniformly high attention
        a_min = self.attn.amin(dim=(-2, -1), keepdim=True)
        a_max = self.attn.amax(dim=(-2, -1), keepdim=True)
        attn_normalized = (self.attn - a_min) / (a_max - a_min + eps)
        # clip low vlues < 0.001 and zero them
        attn_normalized = torch.where(attn_normalized < 0.0001, 0, attn_normalized).view(B, M)
        # binarize each predicted patch to {0,1} so it matches the available patterns
        # we don't threshold here
        binary_patches, _ = self.scale_and_binarize_patches_()
        # For each of the M patches, finds their specific codebook pattern, if available 
        exact_idx, exact_mask = self.retrieve_exact_patches(binary_patches, return_with_no_match_idx=False)

        # we zero any patch that wasn't in the codebook before removing duplicates,
        # because if not, a blank's exact_idx == 0 and would count together with the 0th pattern slot
        binary_patches = binary_patches * exact_mask[:, :, None, None].float()
        attn_normalized = attn_normalized * exact_mask.float()

        # Now merge attention with patches: [B,M,A,P,P]
        stacked_attention_patches = torch.stack(
            (binary_patches, attn_normalized[:, :, None, None].expand_as(binary_patches)), dim=2
        )
        # zero duplicates and keep only the highest attention per duplicated patch
        unique_patterns = self.zero_duplicates(stacked_attention_patches, exact_idx)


        # get new idx now without duplicates:
        exact_idx, _ = self.retrieve_exact_patches(unique_patterns[..., 0, :, :])
        # scatter now the unique patch patterns and their attention into the correct slot of the codebook length
        expanded_patterns = self.expand_to_codebook(unique_patterns, exact_idx, num_patterns=self.num_patterns)

        content = expanded_patterns[:, :, 0] # [B, N, A, P, P] -> [B, N, P, P] patterns
        attention = expanded_patterns[:, :, 1]# same with the attention score

        # We collapse the batch dimention:
        # max patterns to keep them binary
        # sum attention and average it using the batch samples
        content_merged = content.amax(dim=0) # [N,P,P]
        attention_merged = attention.sum(dim=0) / B
        # stack them back together [N,A,P,P]
        sum_expanded = torch.stack([content_merged, attention_merged], dim=1)

        return sum_expanded

    @overload
    def calculate_adjacent_links(self,remove_self_pairs: bool = True,*,return_top_k: None = None) -> torch.Tensor: ...

    @overload
    def calculate_adjacent_links(self,remove_self_pairs: bool = True,*,return_top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def calculate_adjacent_links(self, remove_self_pairs:bool=True, return_top_k:None|int=None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count spatial grid neighbors between codebook patterns to build an edge weighted adjacency matrix.

        Args:
            remove_self_pairs (bool): If True, zero the diagonal of the returned adjacency matrix
                after symmetrization, dropping "pattern k next to itself" edges.
                If False return with the diagonal intact
                Note: adjacency matrix is build comparing first top/left and right/bottom before finally
                summing the transposed matrix to adjust for the left/top, bottom/right direction, thus
                diagonal (self.pairs) is counted twice

            return_top_k (int | None): If set, return only the k strongest k edges (as pattern ID
                pairs and their counts) instead of the full dense matrix.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                
                - if return_top_k=None returns adjacency [num_patterns, num_patterns] float:
                    Where adjacency[i, j] = number of times that pattern i and pattern j were neighbours
                    left,right,top,bottom anywhere in the batch.
                
                - if return_top_k=k returns topk_i, topk_j, topk_counts: 
                    Where topk_i/topk_j are the flattened index row/col of pattern IDs of the k 
                    largest entries and topk_counts are their values.
        """
        # binarize each predicted patch, no need to threshold 
        # Note: 
        # the binarize function is also used when calculating node sizes,
        # if this gets too expensive we could refactor the code to check if it had been already used   
        binary_patches, _ = self.scale_and_binarize_patches_()

        # Get indexes of the available patterns with index -1 for blanks 
        idx=self.retrieve_exact_patches(binary_patches, return_with_no_match_idx=True)

        B, M = idx.shape
        G = math.isqrt(M)
        assert G * G == M

        # Create a grid and offset it by one in all directions to get the neighbours
        grid = idx.view(B, G, G)

        left = grid[:, :, :-1].reshape(-1)
        right = grid[:, :, 1:].reshape(-1)

        top = grid[:, :-1, :].reshape(-1)
        bottom = grid[:, 1:, :].reshape(-1)

        # concat them together 
        k_neigh1 = torch.cat([left, top])
        k_neigh2 = torch.cat([right, bottom])

        # drop any edges where either side was unmatched (with index -1)
        # also bicount counts occurrences of integers, so arguments needs to be strictly non negative
        valid = (k_neigh1 >= 0) & (k_neigh2 >= 0)
        k_neigh1, k_neigh2 = k_neigh1[valid], k_neigh2[valid]

        # Make a 2D grid into 1D. Bincount only counts integer co occurence, so we transform every pair into a unique int 
        # multiplying by N gives and adding k2 gives us an unique value for each pair. 
        pair_idx = k_neigh1 * self.num_patterns + k_neigh2
        # with minlength ensuring that all N possible positions exist in the result (including pairs that occurred zero times.)
        counts = torch.bincount(pair_idx, minlength=self.num_patterns * self.num_patterns)
        # now we reshape the counts back into NxN, where the code pattern index i,j refers to the actual count of said pair
        # we only looked at left to right, top to bottom 
        # so we now transpose to have right to left, bottom to top counts as well
        # Note: self pairs in the diagonally are counted twice when adding the transposed matrix
        adjacency = counts.view(self.num_patterns, self.num_patterns).float()
        adjacency = adjacency + adjacency.T

        if remove_self_pairs: 
            adjacency.fill_diagonal_(0) 

        # only retrieve the top counts and their index
        if return_top_k:
            assert isinstance(return_top_k, int) and not isinstance(return_top_k, bool), \
            f"return_top_k must be an int, got {type(return_top_k)}"
            # To get the top counts, we reverse the previous logic
            # flatten to 1D again and get the top counts
            flat = adjacency.flatten()
            topk_counts, topk_idx = flat.topk(return_top_k)
            # topk_idx belongs to flat, so we convert back to codebook idx (using the reverse of k1*N+k2 we used to encode these flattened idx) 
            # The full int division = K1, remainder K2
            topk_i, topk_j = topk_idx // self.num_patterns, topk_idx % self.num_patterns
            # pattern ids AND their counts, not counts alone
            return topk_i, topk_j, topk_counts   

        return adjacency

    def get_nodes_and_links(self, previous_values:None|tuple[torch.Tensor,torch.Tensor]=None, samples:None|int=None, verbose:bool=True)->tuple[torch.Tensor, torch.Tensor]:
        """Compute node and link tensors and optionally merge them into a running accumulation.

        Calls `calculate_node_sizes()` and `calculate_adjacent_links()` to calculate pattern attention
        and neighbours. If adding `previous_values`, merges the new result into it and averages
        them by dividing with `samples`

        Args:
            previous_values (tuple[torch.Tensor, torch.Tensor] | None): 
                Prior output from a previous call to of this method. If None, skips merging

            samples (None|int=None):
                The number of samples used to divide summed attention to average them. 
                Eg. if run in a loop, average once at the end using the loop index. 
            verbose (bool=True):
                prints info that summed (not averaged) attention scores are being returned
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - [0] = nodes [num_patterns, A=2, P, P]: 
                    A[0] are the binary patterns 
                    A[1] is the the averaged attention clipped to 0 below 0.0001 when merged.
                - [1] = links [num_patterns, num_patterns]: 
                    summed pattern neighbours counts.
        """
        # calculate node sizes (attention per patch) and neigbourg patteren occurance counts 
        nodes_and_links = (self.calculate_node_sizes(), self.calculate_adjacent_links())

        if previous_values is not None:

            nodes_total = previous_values[0].clone()
            # [N,A,P,P] let's use max to add new binary patterns in A[0] (if no patch was found before the given index should all 0)
            nodes_total[:, 0] = torch.maximum(previous_values[0][:, 0], nodes_and_links[0][:, 0])
            # summ the attention at A[1]
            nodes_total[:, 1] = previous_values[0][:, 1] + nodes_and_links[0][:, 1]   # still raw
            
            # Add counts and put them back into a new tensor
            new_links = nodes_and_links[1] + previous_values[1]
            nodes_and_links = (nodes_total, new_links)

            if samples is not None:
                # div the attention by the amount of samples to average
                new_att= nodes_and_links[0][:, 1]/samples 
                # To avoid negligable attention, if smaller than 10-4 make it 0
                new_att=torch.where(new_att < 0.0001, 0, new_att)
                nodes_and_links[0][:, 1] = new_att

            elif samples is None and verbose: 
                print("number of samples was not given, thus raw attention values were returned")

        return nodes_and_links

####### Node Graph #######

class Create_Network_Graph:
    """
    Create an ig.Graph with `build_pattern_graph` using a tuple of tensors ([N,A,P,P],[N,N]) to visualize 
    nodes in A[0] with a size of A[1] where links between nodes are determined by counts between [N,N]. 
    Optionnally, use an already created graph object to revizualize it differently.
    
        Attributes:
            graph_object (ig.Graph | None): Use an existing graph to call methods downstream. Or create one with `build_pattern_graph`.
    """

    def __init__(self, graph_object: ig.Graph|None=None) -> None:
        """
        Args:
            graph_object (ig.Graph | None): Use an existing graph to call methods downstream. Or create one with `build_pattern_graph`.
        """
        self.graph_object=graph_object

    def build_pattern_graph(self, nodes:torch.Tensor, links:torch.Tensor, min_edge_weight:float=0, min_size_percent:float=0.0, n_biggest_nodes:None|int=None) -> None:
        """
        Build an igraph.Graph using a two tensors of shape [N,A,P,P] and [N,N] where nodes are determined by each slot in A[0]
        representing a specific unique binary pattern and sizes are determined by the value in A[1] representing attention values. 
        Links between nodes are the counts between [N,N] and the counts represent how often these patterns were neighbours. 
        Optionally drops low-attention and non-top-K patterns and stores the result in `self.graph_object`.

        Args:
            nodes (torch.Tensor [N, A, P, P]): 
                tensor with binary patterns and an attention score, where A[0] is binary pattern,
                and A[1] the attention score used to build the node size.
            
            links (torch.Tensor [N, N]): Pairwise adjacency counts. Represents the counts of how often
                two given patterns were counted as being adjacent to each other.
            
            min_edge_weight (float): Edges with weight strictly below this are dropped
            
            min_size_percent (float): Drop nodes whose attention is below this fraction of the max
                attention among existing patterns (eg. with 0.0 keep all, with 0.1 drop the bottom 10% of max).
            
            n_biggest_nodes (int | None): If set keep only the `n_biggest_nodes` with the highest attention (applied after the
                `min_size_percent` filter, so it further narrows whatever survived that step).

        Returns:
            None: 
            Stores the result in `self.graph_object` as an igraph.Graph object with vertex attributes "pattern_id", "size", "pattern" and edge attribute "weight".
        """

        # [N,A,P,P], Get all the available binary patterns from A[0]
        content = nodes[:,0]
        # Get their attention values [A1]                     
        size = nodes[:,1,0,0]             

        # Get only the patterns that exist, missing pattern should be all 0s
        exists = content.flatten(-2).sum(-1) > 0

        # apply the size floor only among patterns that exist
        if min_size_percent > 0 and exists.any():
            size_threshold = size[exists].max() * min_size_percent
            exists = exists & (size >= size_threshold)
        # optionally filter the top n_biggest_nodes patterns
        if n_biggest_nodes is not None:
            existing_ids = torch.where(exists)[0]
            # only truncate if there are more patterns than n_biggest_nodes and only get the indices
            if existing_ids.numel() > n_biggest_nodes:
                top_idx = torch.topk(
                    size[existing_ids],
                    k=n_biggest_nodes,
                    largest=True,
                ).indices

                top_ids = existing_ids[top_idx]
                # Create a mask
                new_exists = torch.zeros_like(exists)
                new_exists[top_ids] = True
                exists = new_exists

        node_ids = torch.where(exists)[0]

        # only keep edges above 0, and only between existing nodes
        # as it is an identity matrix, we first 0 one side, then the other dimention
        adj = links.clone()
        adj[~exists, :] = 0
        adj[:, ~exists] = 0
        # zero any weight based on a given min_edg threashold 
        adj[adj < min_edge_weight] = 0

        # Let's get rid of half the matrix (it's doubled, so we just keep one side (using triu )
        row_idx, col_idx = torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=True)
        weights = adj[row_idx, col_idx]

        # igraph vertices have to be indexed 0..n-1 so we remap surviving pattern IDs
        id_map = {int(pid): i for i, pid in enumerate(node_ids.tolist())}
        edges = [(id_map[int(s)], id_map[int(d)]) for s, d in zip(row_idx.tolist(), col_idx.tolist())]
        # Now create the vertexes
        self.graph_object = ig.Graph(n=len(node_ids), edges=edges)
        self.graph_object.vs["pattern_id"] = node_ids.tolist()
        self.graph_object.vs["size"] = size[node_ids].cpu().numpy()
        self.graph_object.es["weight"] = weights.cpu().numpy()
        # store patterns as well
        self.graph_object.vs["pattern"] = content[node_ids].cpu().numpy()   # (num_nodes, P, P) binary arrays


    def pattern_to_pil(self, pattern: np.ndarray, upscale:int=8)->Image.Image:
        """Render one binary P x P pattern as an upscaled, bordered grayscale PIL image.

        Args:
            pattern (np.ndarray, [P, P]): Binary pattern array with values of 0(= background) and 1 (1 = foreground).
            upscale (int): image scaling factor.
            border_width (int): Unused. Accepted but never read; the border below is hardcoded to
                1 px regardless of this value.

        Returns:
            Image.Image: pattern image with where foreground pixels are black (= 0) and background white (= 255) with a black border
        """
        # transofrm binary P,P to 0/255 grayscale then nearest-neighbor upscaling
        img = ((1 -pattern) * 255).astype(np.uint8)
        img = Image.fromarray(img, mode="L").resize(
            (pattern.shape[0] * upscale, pattern.shape[1] * upscale), Image.NEAREST # type: ignore
        )
        return ImageOps.expand(img, border=1, fill="black")

    def style_and_plot(self, 
                       min_size:float=15, 
                       max_size:float=50, 
                       layout_name:str=("star","fr","grid", "drl","random","tree","rt","sugiyama","graphopt")[0], 
                       width:int=1200, 
                       height:int=1200,
                       show_patterns:bool=False, 
                       image_scale:float=5, 
                       always_show_annotation:bool=False, 
                       title:None|str=None,
                       plot_bgcolor:str="lightgrey"
                       )->None:
        """
        Render a `self.graph_object` as an interactive Plotly figure and show it.

        Args:
            min_size (float): min size (px) of the lowest-attention node to show, will be scaled between min and max size.
            max_size (float): max size (px) of the lowest-attention node to show, will be scaled between min and max size.
            layout_name (str): igraph layout name. defaults to "star". Example layouts: "star","fr","grid", "drl","random","tree","rt","sugiyama","graphopt"
            width (int): Figure width in px.
            height (int): Figure height in px.
            show_patterns (bool): If True, hides the patterns coverted to images through :func:`pattern_to_pil`.
            image_scale (float): Image size as a multiple of the node's normalized size. Only used when `show_patterns=True`.
            always_show_annotation (bool): If True, permanently label each node with their pattern ID and size.
            title (str | None): Optionally, centered figure title.
            plot_bgcolor (str): plotly color name or hex for the plot background.

        Returns:
            None. The figure is displayed with `fig.show()`.

        Raises:
            ValueError: If `self.graph_object` is None. A graph with no nodes or no edges also
                raises a ValueError, from numpy's min()/max() on an empty array; that case is
                not caught here (see flag 2).
        """
        if self.graph_object is None:
            raise ValueError ("No graph object was given. Please use a graph object or create one with build_pattern_graph().")
        else: 
            graph=self.graph_object

        coords = np.array(graph.layout(layout_name).coords)
        xs, ys = coords[:, 0], coords[:, 1]

        sizes = np.array(graph.vs["size"])
        sizes_norm = min_size + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) * (max_size - min_size)

        weights = np.array(graph.es["weight"])

        for e in graph.es:
            x0, y0 = coords[e.source]
            x1, y1 = coords[e.target]

        fig = go.Figure()


        n_buckets = 5
        bucket_idx = np.digitize(weights, np.linspace(weights.min(), weights.max(), n_buckets))
        b:None|int=None
        for b in range(1, n_buckets + 1):
            mask = bucket_idx == b
            if not mask.any():
                continue
            bx, by = [], []
            for e, m in zip(graph.es, mask):
                if not m:
                    continue
                x0, y0 = coords[e.source]
                x1, y1 = coords[e.target]
                bx += [x0, x1, None]
                by += [y0, y1, None]
            w = 0.3 + (b / n_buckets) * 3
            a = 0.1 + (b / n_buckets) * 0.5

            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                    line=dict(width=w, color=f"rgba(70,130,180,{a})"),
                                    hoverinfo="none", showlegend=False)
                                    )

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes_norm, color="steelblue", line=dict(width=0.5, color="white")),
            text=[f"pattern {pid}<br>size={s:.2f}" for pid, s in zip(graph.vs["pattern_id"], sizes)],
            hoverinfo="text", showlegend=False,
        ))
        if always_show_annotation:
            for x, y, pid, s in zip(xs, ys, graph.vs["pattern_id"], sizes):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=f"pattern {pid}<br>size={s:.2f}",
                    showarrow=False,
                    font=dict(color="grey"),
                    align="left",
                    xshift=-50,
                    yshift=30,
                )
        assert b is not None
        w = 0.3 + (b / n_buckets) * 3
        a = 0.1 + (b / n_buckets) * 0.5

        # show patterns:
        if show_patterns:
            # hide the marker dots themselves so images replace them rather than overlapping
            fig.data[-1].marker.opacity = 0 # type: ignore

            x_range = xs.max() - xs.min()
            y_range = ys.max() - ys.min()
            px_to_data_x = x_range / width
            px_to_data_y = y_range / height

            for i, pid in enumerate(graph.vs["pattern_id"]):
                img = self.pattern_to_pil(graph.vs["pattern"][i])
                s = sizes_norm[i] * image_scale
                fig.add_layout_image(
                    dict(
                        source=img,
                        x=xs[i], y=ys[i],
                        xref="x", yref="y",
                        sizex=s * px_to_data_x, sizey=s * px_to_data_y,
                        xanchor="center", yanchor="middle",
                        layer="above",
                    )
                )


        fig.update_layout(
            width=width, height=height,
            plot_bgcolor=plot_bgcolor,

            title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            ) if title is not None else None,

            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.show()