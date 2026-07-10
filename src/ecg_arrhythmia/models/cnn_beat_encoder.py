import torch
from torch import nn


class CNNBeatEncoder(nn.Module):
    """
    CNN encoder for individual ECG beat windows.

    This model is not used to classify beats directly. Instead, it converts each
    beat window into a morphology embedding that can later be passed
    into the Transformer.

    Expected input:
    - x: (batch_size, 1, window_size)

    Output:
    - embedding: (batch_size, embedding_dim)
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # input: (batch_size, 1, 240)
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 32, 120)
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=False
            ),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 64, 60)
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 128, 30)
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            # Collapse the time dimension.
            nn.AdaptiveAvgPool1d(output_size=1),
            # output: (batch_size, 128, 1)
        )

        self.projection = nn.Linear(in_features=128, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Dimension must be size (batch_size, channel, window_size)"
                f"Found size {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(f"Expects one ECG channel, found {x.shape[1]}")

        # Pass the input through the feature extractor
        x = self.features(x)
        # shape: (batch_size, 128, 1)

        # Remove the final time dimension.
        x = x.squeeze(dim=-1)
        # Shape: (batch_size, 128)

        # Project to chosen embedding dimension
        x = self.projection(x)
        # Shape: (batch_size, embedding_dim)

        return x
