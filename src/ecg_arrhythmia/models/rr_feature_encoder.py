import torch
from torch import nn


class RRFeatureEncoder(nn.Module):
    """
    MLP encoder for RR timing features.

    Expected input:
    - rr: (batch_size, sequence_length, rr_feature_dim)
      or (batch_size, rr_feature_dim)

    Output:
    - embedding: same leading dimensions, final dimension = embedding_dim
    """

    def __init__(
        self,
        rr_feature_dim: int = 2,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=rr_feature_dim, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=16, out_features=embedding_dim),
            nn.ReLU(),
        )

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        if rr.shape[-1] != 2:
            raise ValueError(
                "RRFeatureEncoder expects the final dimension to contain "
                f"2 RR features. Found shape {tuple(rr.shape)}"
            )

        return self.encoder(rr)
    
# One thing for future reference: nn.Linear cares about the last dimension.
# Everything before the last dimension is preserved.