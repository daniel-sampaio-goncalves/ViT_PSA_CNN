import torch
from torch import nn

####################################
class PermutationDecoder(nn.Module):
    """
    Feeds latent encoder embeddings into a bilinear symmetry matrix to predict a permutation.

    The decoder projects each input into separate query and key
    representations and computes a learned bilinear compatibility score.
    The bilinear interaction allows the model to learn relationships between 
    patches that are complementary.

    scores[b, i, j] = q[b, i] @ W @ k[b, j]

    where ``W`` is a learned interaction matrix.

    Args:
        dim: Dims of the input encoder embeddings.
        proj: Dims of the query/key projections.
    """

    def __init__(self, dim=768, proj=256) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.q = nn.Sequential(
            nn.Linear(dim, proj),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(proj, proj)
        )

        # Query and key are designed deliberately as asymmetric
        # q and k should represent different features in the bilinear score

        self.k = nn.Sequential(
            nn.Linear(dim, proj),
            nn.SiLU(),  # using SiLU to break symetry
            nn.Linear(proj, proj),
            nn.GELU(), # add another layer to improve symetry break
            nn.Dropout(p=0.1),
            nn.Linear(proj, proj)
        )
        
        #self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.5)))
        
        # learned bilinear interaction matrix W, shape [proj, proj]
        # replaces the plain dot product q·k with q#####T W k
        # using a close to 0 approach to start similar as a plain cosine similarity/
        # dot product.

        # Learned bilinear interaction matrix W, shape [proj, proj].
        self.W = nn.Parameter(torch.eye(proj) + (torch.randn(proj, proj) * 0.01) ) 


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] patch embeddings from an encoder.

        Returns:
            scores: [B, N, N] bilinear compatibility matrix, where
            scores[b, i, j] is the compatibility of patch i (as query)
                with patch j (as key) for sample b. Fed into Sinkhorn
                downstream to produce a permutation.
        """
        # Mahalanobis-style similarity.
        x = self.norm(x)
        q = self.q(x)  # [B, N, proj]
        k = self.k(x)  # [B, N, proj]
        
        # We could also normalize here, and restrict the model, can be worth trying depends on training
        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        
        # bilinear: scores[b,i,j] = q[b,i,:] @ W @ k[b,j,:]
        qW = torch.matmul(q,self.W)
        scores = torch.matmul(qW, k.transpose(-2, -1))
        
        return scores


class Sinkhorn(nn.Module):
    """Convert score matrices into approximately doubly stochastic matrices."""

    def __init__(self) -> None:
        super().__init__()

    def sinkhorn(self, matrix:torch.Tensor,n_iter:int)->torch.Tensor:
        """Run the log space Sinkhorn loop.

        Args:
            matrix: a log score matrix of shape ``[..., N, M]``.
            n_iter: Number of alternating row/column normalization steps.
                
        Returns:
            torch.Tensor: Matrix with rows and columns each approximately summing to 1.
        """
        for _ in range(n_iter):
            # Row normalization
            matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
            # Column normalization
            matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
        return matrix.exp()
        
    
    def forward(self, decoder_matrix, n_iter=20)->torch.Tensor:
        """Apply iterative Sinkhorn normalization in log space.

        Args:
            decoder_matrix: Score matrix of shape ``[..., N, M]``.
            n_iter: Number of row/column normalization iterations.

        Returns:
            torch.Tensor: Approximately doubly stochastic matrix with the same shape as ``decoder_matrix``
        """
        return  self.sinkhorn(decoder_matrix.float(),n_iter)