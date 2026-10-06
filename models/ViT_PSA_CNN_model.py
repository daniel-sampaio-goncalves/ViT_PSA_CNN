from typing import Literal, overload

import torch
from torch import nn

from models.classifier.attnpoolcnn import AttnPoolCNN, ConstrictWindow
from models.decoder.permutation_decoder import PermutationDecoder, Sinkhorn
from models.encoder.vit import PatchEmbedding, Transform_Patches, TransformerEncoderBlock

############################################################################################
class ViT_PSA_CNN(nn.Module): # Vision transofmer PermutationSpacialAttention CNN 
    """
    End-to-end model composed of a ViT encoder lacking positional embeddings, 
    a learned permutation decoder combining a bilinear symetry head with a Sinkhorn 
    normalization and a windowed attention pooled CNN head.

    The model should be used with two different usage modes, 
    controlled by `return_vit_class`. 
    The ViT should be pretrained using `return_vit_class=True`. It
    runs a patch shuffling and masking task and returns a
    classification label.

    Full pipeline using `return_vit_class=False` that encodes patches of a same sequence, 
    predicts a soft permutation via `PermutationDecoder` + `Sinkhorn`, 
    and applies it to reorder the input patches. These will be fed into `AttnPoolCNN`
    classifier by first restricting the CNN's patch window via the `ConstrictWindow` module.
    
    Args:
        Patch (int): Side length of each square input patch `P`.
        number_of_patches (int): Total number of patches per image `N`    
        embedding_dimension (int): ViT embedding dimension `D`.
        CNN_features (int): Number of hidden channels in the `AttnPoolCNN` head.
        nb_heads (int): Number of attention heads per transformer block.
        MLP_Size (int): Hidden size of each transformer block's MLP.
        mlp_dropout (float): Dropout probability inside each transformer block's MLP.
        embedding_dropout (float): Dropout probability applied to the patch +
            class-token embeddings before the transformer.
        num_transformer_layers (int): Number of stacked `TransformerEncoderBlock`s.
        num_class (int): Number of output class labels (dims) of the ViT-pretraining classifier head.
    """
    
    def __init__(self, 
                 Patch:int=4,
                 number_of_patches:int=4096,
                 embedding_dimension:int=768,
                 CNN_features:int=64,
                 nb_heads:int=12,
                 MLP_Size:int=3072,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_transformer_layers:int=12,
                 num_class:int=1
                 ) -> None:
        
        super().__init__()                              
        self.P=Patch
        self.num_patches = number_of_patches
        # Patch transformer for vit pretraining:
        self.transform_patch=Transform_Patches()
        # Patch embedings
        self.patch_embedding=PatchEmbedding(P=self.P,D=embedding_dimension)
        self.class_token=nn.Parameter(torch.randn(1,1,embedding_dimension)*0.02)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # transformer block:
        self.transformer=nn.Sequential(*[TransformerEncoderBlock(
                                            embedding_dim=embedding_dimension,num_heads=nb_heads,mlp_size=MLP_Size,mlp_dropout=mlp_dropout
                                            )
                                        for _ in range(num_transformer_layers)])
        
        self.classifier=nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dimension),
                                      nn.Linear(in_features=embedding_dimension,out_features=num_class))
        self.perm_decoder=PermutationDecoder(dim=embedding_dimension, proj=256)
        self.sinkhorn=Sinkhorn()
        self.attnpoolcnn=AttnPoolCNN(patch_size=Patch,  hidden_features=CNN_features)
        self.create_local_window=ConstrictWindow()

    @overload
    def forward(
        self,
        X: torch.Tensor,
        *,
        return_vit_class: Literal[True],
        return_attention: bool        
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        X: torch.Tensor,
        return_attention: Literal[False],
        return_vit_class: Literal[False]
    ) -> tuple[
        torch.Tensor,  # permuted_class
        torch.Tensor,  # soft_perm
        torch.Tensor,  # cnn_patches
        ]: ...

    @overload
    def forward(
        self,
        X: torch.Tensor,
        return_attention: Literal[True],
        return_vit_class: Literal[False],
    ) -> tuple[
        torch.Tensor,  # permuted_class
        torch.Tensor,  # attention
        torch.Tensor,  # perm_patches
        torch.Tensor,  # permutation_score
        torch.Tensor,  # soft_perm
        torch.Tensor,  # cnn_patches
        ]: ...

    def forward(self, 
                X:torch.Tensor, 
                return_attention:bool=False, 
                return_vit_class:bool=False, 
                mask_ratio:torch.Tensor|float=0.0,
                grid_size:int=64,
                cnn_window_size:int|torch.Tensor=18,
                cnn_jitter:int|torch.Tensor=0) -> torch.Tensor|tuple[torch.Tensor,torch.Tensor,torch.Tensor]|tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Args:
            X: (B, N, P, P) batch of N patches of size PxP each.
            return_attention (bool): If True (and not in ViT-pretraining mode),
                also return the intermediate permutation/attention
                tensors alongside the final prediction — useful for
                inspection, not needed for standard training.
            return_vit_class (bool): If True and the module is in training
                mode, runs ViT-pretraining mode instead of the full
                pipeline: applies `Transform_Patches`'s shuffle/mask
                pretext task and returns a classification from the
                [CLS] token alone, skipping the permutation decoder and
                CNN head entirely.
            mask_ratio (torch.Tensor|float): Passed through to `Transform_Patches`; only
                relevant when `return_vit_class=True` during training.
            grid_size (int): Side length of the full square patch grid (= sqrt(N)),
                passed through to `ConstrictWindow`.
            cnn_window_size (torch.Tensor|int): Side length of the centered window of
                patches the CNN head is restricted to. Expected to
                already be annealed by the caller if used with a
                schedule (see `ConstrictWindow.forward`'s note on
                torch.compile implications of int vs. tensor values
                here).
            cnn_jitter (torch.Tensor|int): Max random offset applied to the window's
                origin during training; ignored during eval. See
                `ConstrictWindow`.

        Returns:
            if `return_vit_class=True`: 
                ViT classification logits.
            
            Otherwise:
            if `return_attention=True`:
                (permuted_class, attention, perm_patches,
                permutation_score, soft_perm, cnn_patches).
            else:
                (permuted_class, soft_perm, cnn_patches).
        """
        B,N,P,_= X.shape
        # Phase 1 - Vit
        # create the patch and flattening using our class:
        if return_vit_class and self.training:
            X=self.transform_patch(X, mask_ratio=mask_ratio)

        x_flatten=self.patch_embedding(X)
        # class_token; now expand the class token to all Batches:
        cls=self.class_token
        cls=cls.expand(B,-1,-1) # -1 keeps the same
        # concat class token:
        x_flatten_cls_pos=torch.concat((cls,x_flatten),dim=1)
        # dropout
        x_flatten_cls_pos=self.embedding_dropout(x_flatten_cls_pos)
        # transformer:
        x_transformer=self.transformer(x_flatten_cls_pos)

        if return_vit_class:
            class_out=self.classifier(x_transformer[:,0]) # B, P, D - add patches back to classifier
            return class_out.squeeze(-1)
        
        # Phase 2 - Decoder and CNN
        # Bilinear symetry decoder
        permutation_score= self.perm_decoder(x_transformer[:, 1:]) 
        # Sinkhorn norm
        soft_perm = self.sinkhorn(permutation_score)

        # apply a per batch permutation via batched matrix multiplication.
        perm_patches = torch.einsum("bpq,bphw->bqhw", soft_perm,X)

        # create subset of the decoder matrix with a central constricted window
        cnn_window_idx=self.create_local_window(perm_patches, window_size_=cnn_window_size,jitter_=cnn_jitter, grid_size_=grid_size)
        cnn_patches=perm_patches[:, cnn_window_idx]
        # Attentionpool CNN
        permuted_class, attention=self.attnpoolcnn(cnn_patches)

        # Only return most of the outputs if wanted: 
        if return_attention:

            return permuted_class, attention, perm_patches, permutation_score, soft_perm, cnn_patches

        # the only outputs needed during training are class labels,  itself
        else:
            return permuted_class, soft_perm, cnn_patches