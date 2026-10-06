import math

import torch
from torch import nn
import torch.nn.functional as F

##################################
class ConstrictWindow(nn.Module):
    """
    Returns a square sub window of patch indices from a bigger grid. 
    This is used to restrict the window size, and therefore,
    the amount of total patches the downstream module looks at during training. 
    The window is located towards the center. During training, the window is randomly 
    jittered around the grid center each call while during eval, the window no jitter occurs.
    """
    def __init__(self) -> None:
        super().__init__()

    def window_indices(self, grid_size:int, window_size:torch.Tensor|int, top_left:torch.Tensor, device) -> torch.Tensor:
        """
        Created the flattened indices of every patch inside a
        `window_size` x `window_size` square window, given the 
        computed window's top-left corner indices, 
        within a `grid_size` x `grid_size` grid.

        Args:
              grid_size (int): Side length of the full square patch grid.
              window_size (torch.Tensor|int): Side length of the window to select.
              top_left (torch.Tensor): tensor of the window's top-left
                        corner in grid coordinates (row, col).
              device (str): pytorch device.
        
            Returns:
                torch.Tensor: flattened indices of every patch inside the selected window.
        """
        r0, c0 = top_left
        rows = torch.arange(r0, r0 + window_size, device=device) # type: ignore
        cols = torch.arange(c0, c0 + window_size, device=device) # type: ignore
        gr, gc = torch.meshgrid(rows, cols, indexing="ij")
        return (gr * grid_size + gc).flatten()

    def sample_window_origin(self, grid_size:int, window_size:int|torch.Tensor, device:str|torch.device, jitter:int|torch.Tensor) -> torch.Tensor:
        """
        Samples a random top-left origin for a square window, 
        centered on the grid with ``grid_size`` x ``grid_size`` = total N patches.
        In addition it randomly offsets the center around `jitter` patches in each
        direction (row and column independently), clamped to avoid the window
        drifting outside the grid.

        Args:
            grid_size (int): Side length in patches of the full square grid
                in N patches (= sqrt(N)).
            window_size (torch.Tensor|int): Side length in patches of the
                newly calculated constricted window to place.
            device (str|torch.device): PyTorch device to build the origin
                tensor on.
            jitter (torch.Tensor|int): Maximum offset in patches applied to
                the centered origin, in each of the row and column directions
                independently. If 0, the origin is centered with no jitter.

        Returns:
            torch.Tensor: row, col index of the top-left origin of a centered windowsize.
        """
        center:torch.Tensor = torch.tensor(
            [(grid_size - window_size) // 2, (grid_size - window_size) // 2],
            device=device, dtype=torch.long
        )
        if jitter > 0:
            offset:torch.Tensor = torch.randint(-jitter, jitter + 1, (2,), device=device)  # type: ignore
        else:
            offset = torch.zeros(2, device=device, dtype=torch.long)

        origin:torch.Tensor = (center + offset).clamp(0, grid_size - window_size)  # type: ignore

        return origin

    def forward(self, X:torch.Tensor, window_size_:torch.Tensor|int=18, jitter_:torch.Tensor|int=3, grid_size_:int=64):
        """
        Calculates the index for the constricted window. During training
        jitter is added around the center location while absent during
        eval.

        Note:
        -----
        ``window_size_`` and ``jitter_`` accept either a ``torch.Tensor`` or a Python
        ``int`` (defaults to ``int``).``torch.compile`` treats ``int`` as a Python integers are treated 
        as static values. Consequently, if either parameter changes during training (e.g.,
        during annealing), passing it as an ``int`` may cause the compiled graph to be recompiled 
        for each new value. Passing changing values as tensors allows them to remain dynamic
        inputs to the compiled graph and can avoid unnecessary recompilation and improve performance

        Args:
            X (torch.Tensor):
                Input tensor of shape ``[B, N, P, P]``, where ``B`` is the batch
                size, ``N`` is the number of patches in the full grid, and ``P``
                is the spatial width and height of each patch.
            
            window_size_ (torch.Tensor|int): Side length, in patches, of the square window to select.

            jitter_ (torch.Tensor|int): Maximum offset in patches applied to
                the centered origin, in each of the row and column directions
                independently. If 0, the origin is centered with no jitter.
            
            grid_size_ (int): Side length in patches of the full square grid
                in ``N`` patches.  (Typically ``grid_size_`` = sqrt(N)).

        Returns:
            torch.Tensor: flattened indices of every
            patch inside the selected window.
        """

        # jitter should only happen during training, 
        # window size should be fixed during training
        jitter = jitter_ if self.training else 0

        top_left_ = self.sample_window_origin(grid_size=grid_size_, window_size=window_size_, device=X.device, jitter=jitter)
        idx = self.window_indices(grid_size=grid_size_, window_size=window_size_, top_left=top_left_, device=X.device)

        return idx

class AttnPoolCNN(nn.Module):
    """
    A CNN head that produces a single scalar prediction per sample 
    via an attention-weighted pooling.

    Patches are reassembled into a single spatial grid and passed through 
    two conv blocks before being pooled with a learned attention map and classified.

    GroupNorm is used after each conv layer. GroupNorm is
    used instead of BatchNorm because its running statistics become 
    unreliable with the used small batch sizes.

    
    Args:
        patch_size (int): length of each square input patch. Used as
            both kernel and stride size of the first conv block.
    
        hidden_features (int): Number of channels used throughout the conv
            stack and the pooled feature vector (defaults to 64).
        
        in_channels (int): Number of input channels (defaults to grayscale).
        
        out_features (int): Dims of the classifier output (defaults to 1).

    """
      
    def __init__(self, patch_size:int, hidden_features:int=64, in_channels:int=1, out_features:int=1) -> None:

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.GroupNorm(num_groups=8, num_channels=hidden_features),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_features),
            nn.ReLU(),
            )

        # nn.ReflectionPad2d(1) using this the center values will be reflected on the outer boundaries instead of hard 0s, could minimize corners artifacts
        self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),  # another local mixing layer
                nn.GroupNorm(num_groups=8, num_channels=hidden_features),
                nn.ReLU(),
                nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),  # optional pointwise refinement
                nn.GroupNorm(num_groups=8, num_channels=hidden_features),
                nn.ReLU(),
            ) 
        
        self.dropout = nn.Dropout(0.3)
        self.dropout2d=nn.Dropout2d(0.1)
        self.attn_score = nn.Conv2d(hidden_features, 1, kernel_size=1)
        self.classifier=nn.Linear(hidden_features, out_features) 


    def batch_make_grid(self, x:torch.Tensor, pad:int|None=None, pad_value:int=0) -> torch.Tensor:
        """
        For any batch, each sample is transformed from a sequence of flat patch sequences 
        into a single square spatial grid (like an image) for each sample for all given batches.

        Args:
            
            x (torch.Tensor): [B, N, P, P] batch of N patches, each PxP. N must be a
                perfect square, since patches are laid
                out into a GxG grid.
            
            pad (None|int): Each patch is padded by this amount on all four sides 
                with the ``pad_value`` before being placed into the grid.
            
            pad_value (int): Fill value used for padding when ``pad`` is set.
            
            Notes:
            -----
                eg. with pad_value=0
                    0000000
                    0011100
                    0000000

                eg. with pad_value=5
                    5555555
                    5511155
                    5555555

        Returns:
            torch.Tensor: (B, 1, G*P, G*P) grid, where G = sqrt(N) and P (with or without padding)
        """
        x = x.unsqueeze(2)          # [B, N, 1, P, P]

        B, N, C, P, _ = x.shape
        G = math.isqrt(N)      # if N is guaranteed to be a perfect square
        assert G * G == N

        if pad is not None:
            # pad each patch: (left, right, top, bottom)
            x = F.pad(x, (pad, pad, pad, pad), value=pad_value)

            P = P + 2 * pad

        # Arrange into grid
        x = x.view(B, G, G, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, G * P, G * P)

        return x


    def forward(self, X:torch.Tensor)-> tuple[torch.Tensor,torch.Tensor]:
        """
        Args:
            X: [B, N, P, P] batch of N patches to classify.

        Returns:
            tuple[logits, attn]:
            
            logits (torch.Tensor): [B,] classifier output
            attn (torch.Tensor): [B, 1, P*P] attention weights over spatial positions
        """          
        
        X = self.batch_make_grid(X)   # [B, 1, G*P, G*P] with G = sqrt(N)
        
        X = self.conv1(X)             # [B, hidden_features, G*P,G*P]
        X = self.conv2(X)             # [B, hidden_features, G*P,G*P]


        attn:torch.Tensor = self.attn_score(X)    # [B, 1, G*P // patch_size, G*P // patch_size]
        attn = attn.flatten(2)                    # [B, 1, (G*P // patch_size) ** 2]
        attn = attn.softmax(dim=-1)               # get attention weights (summing to 1 over all spatial positions)

        X = self.dropout2d(X)
        features = X.flatten(2)             

        pooled = (features * attn).sum(dim=-1)
        pooled = self.dropout(pooled)

        logits:torch.Tensor = self.classifier(pooled)         
        
        
        return logits.squeeze(-1), attn

