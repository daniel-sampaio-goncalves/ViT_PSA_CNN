import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

############################################
class Transform_Patches(nn.Module):
    """
    Apply stochastic patch permutation, masking, and intensity perturbations.
    For an input of shape ``[B, N, P, P]``, this module applies a random permutation to ``N`` and
        randomly masks a given % of the total ``N`` patches (zeros ``P``). In addition, changes any pattern intensity if 
        <0.5 to 0.1 and >0.1 to 1 per sample according to a certain probability. 
        There is a 10% chance that a given batch won't be masked at all
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X:torch.Tensor, mask_ratio:float=0.15, binarize_prob:float|None=0.1)->torch.Tensor:
        """
        Randomply Permutes ``N`` and masks a given % of ``N`` patches according to mask_ratio.
        Acoording to the probability binarize_prob, perturbes a sample ``[N]`` by mapping
        patch intensities below 0.5 to 0.1 and intensities above 0.5 to 1. In addition, there
        is 10% random chance that a given batch won't be masked.


        Args:
            X (torch.Tensor): input tensor of shape ``[B, N, P, P]``
            mask_ratio (float): % of total random N patches masked.
            binarize_prob (float|None): % of samples per batch that are perturbed. Changes patch intensities < 0.5 to 0.1 and >0.5 to 1
        
        Returns:
           torch.Tensor: transformed tensor with the same shape as the input of shape ``[B, N, P, P]``. 

        """

        B, N, P, _ = X.shape
        device = X.device

        

        # argsort of random noise gives a valid random permutation, vectorized across batch
        perm_idx:torch.Tensor  = torch.rand(B, N, device=device).argsort(dim=-1)  # [B, N]
        batch_idx:torch.Tensor = torch.arange(B).unsqueeze(1).expand(-1, N) 

        x_permuted:torch.Tensor = X[batch_idx,perm_idx]  # [B, N, P, P]

        # defaults to 10%, patch intensities get split to 1 and 0.1 
        if binarize_prob is not None:
            binarize_sample: torch.Tensor = torch.rand(B, device=device) < binarize_prob  # [B]
            # if higher than 0.5 then binarize, else make patterns'intensity 0.1; helps to encourage pattern semantics without relying on individual intensity 
            binarized: torch.Tensor = torch.where(
                x_permuted > 0.5,
                1.0,
                torch.where((x_permuted>0) & (x_permuted<0.5),
                            0.1,
                            x_permuted)
            )
            x_permuted = torch.where(binarize_sample.view(B, 1, 1, 1), binarized, x_permuted)

        # independent random mask per sample: True where we drop the patch
        keep_prob:torch.Tensor = torch.rand(B, N, device=device)

        # Do not mask every step, keep 10% always with all patches
        if torch.rand(()) < 0.9:
            mask:torch.Tensor = keep_prob < mask_ratio                     # [B, N], True = masked
            X_masked:torch.Tensor = x_permuted.masked_fill(mask.view(B, N, 1, 1), 0.0)
        else:
            X_masked = x_permuted

        return X_masked
    
class PatchEmbedding(nn.Module):
    """Projects 2D patches into a learnable embedding space


    Input:
        ``[B, N, P, P]``

    Output:
        ``[B, N, D]``

    Args:
        P  (int): Spatial size of input patches [P,P]. Defaults to 4.
        D  (int): Dimensions of output embeddings. Defaults to 768.
    """
    def __init__(self, P:int=4,D:int=768) -> None:
        super().__init__()

        self.patch_size:int=P
        self.linear_projection:nn.Linear =nn.Linear(in_features=P*P, out_features=D)
        self.flatten:nn.Flatten =nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x:torch.Tensor)->torch.Tensor:

        x_flattened = self.flatten(x)

        # asserts if flatten set is the same as user input
        assert x_flattened.shape[-1] == self.patch_size**2 , f"Exprected patch_size:{self.patch_size}**2=({self.patch_size**2}), however got ({x_flattened.shape[-1]})"
        x_embedded:torch.Tensor=self.linear_projection(x_flattened)

        return x_embedded

class Multiheadattentionblock(nn.Module):
    """
    Creates a multi-head self-attention block ("MSA block" for short).
    """
    def __init__(self, e_dim, n_head,attn_dropout:float=0) -> None:
        super().__init__()
        self.MSA:nn.MultiheadAttention =nn.MultiheadAttention(embed_dim=e_dim,
                                        num_heads=n_head,
                                        dropout=attn_dropout,
                                        batch_first=True)
        self.layer_norm:nn.LayerNorm = nn.LayerNorm(normalized_shape=e_dim)

    def forward(self, X) -> torch.Tensor:
        """
        Creates a multi-head self-attention block ("MSA block" for short).
        """
        x_norm  = self.layer_norm(X)
        x_MSA, _= self.MSA(query=x_norm,
                          key=x_norm, 
                          value=x_norm,
                          need_weights=False)
        return x_MSA

class MLPblock(nn.Module):
    """
    Creates a layer-normalized multilayer perceptron (MLP) block.
    """
    def __init__(self,emb_dim=768,mlp_size=3072, dropout_val=0.1) -> None:
        super().__init__()

        self.norm:nn.LayerNorm = nn.LayerNorm(normalized_shape=emb_dim)
        
        self.mlp:nn.Sequential = nn.Sequential(
                nn.Linear(in_features=emb_dim,
                        out_features=mlp_size),
                nn.GELU(), 
                nn.Dropout(p=dropout_val),
                nn.Linear(in_features=mlp_size,
                        out_features=emb_dim),
                nn.Dropout(p=dropout_val) # Dropout, when used, is applied after every dense layer.
            )

    def forward(self,X) -> torch.Tensor:
        """
        Creates a layer-normalized MLP block.
        """

        x=self.norm(X)
        x=self.mlp(x)
        
        return x
    
class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block consisting of a MSA block followed by a MLP block, 
    with residual connections.

    The block follows the structure:

        x -> MSA(x) + x -> MLP(x) + x

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of each patch embedding.
    num_heads : int
        Number of attention heads used by the MSA block.
    mlp_size : int
        Hidden dims of the MLP block.
    mlp_dropout : float
        Dropout probability for the MLP block.
    attn_dropout : float
        Dropout probability for the attention block.
    use_checkpoint : bool
        True: use checkpointing during training to reduce GPU memory usage (slower)

    Notes
    -----
        checkpoint() discards this block's intermediate activations after
        the forward pass instead of keeping them in memory. This saves up
        to 30% in memory cost. Checkpoint is only used during training. At eval there's no backward pass,
        so recomputation would just add overhead for nothing.
    """
    def __init__(self, embedding_dim:int=768,
                 num_heads:int=12, # Number of MSA Heads
                 mlp_size:int=3072, # MLP hidden dims
                 mlp_dropout:float=0.3, # dropout for dense layers, vit paper uses 0.1 mlp and 0 for attn
                 attn_dropout:float=0.3,
                 use_checkpoint:bool=True # Enable activation checkpointing to reduce training memory usage
                 ) -> None: 
        super().__init__()
        self.MSA_block  = Multiheadattentionblock(e_dim=embedding_dim,n_head=num_heads, attn_dropout=attn_dropout)
        self.MLP_block  = MLPblock(emb_dim=embedding_dim,mlp_size=mlp_size,dropout_val=mlp_dropout)
        self.use_checkpoint = use_checkpoint


    def forward(self, X):
       
        # Activation checkpointing: recompute the block during backward to reduce GPU memory usage only during training.
        if self.use_checkpoint and self.training:    
            return checkpoint(self._forward, X, use_reentrant=False, preserve_rng_state=True) # keep use reentrant false and rng true if using dropout, also keep an eye out for torch.compile bugs
        return self._forward(X)
    
    def _forward(self, X) ->torch.Tensor:

        # add a residual connection for the MSA block (add the input to the output)
        x_msa = self.MSA_block(X) + X

        # add the msa back to the MLP block result
        x = self.MLP_block(x_msa) + x_msa

        return x
    
