import torch
from torch import nn


class CNNBaselineV2(nn.Module):
    """
    Slightly stronger CNN baseline.

    Compared with CNNBaseline v1:
    - Uses more channels.
    - Adds BatchNorm1d after each convolution.
    - Adds Dropout before the classifier.
    - Keeps the model small enough for later edge deployment.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        # BatchNorm stabilises training by normalising the convolution outputs
        # before the activation function. The learned scale and shift parameters
        # let the model adjust each feature channel's distribution.

        self.features = nn.Sequential(
            # input: (batch_size, 1, 240)
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 32, 120)
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 64, 60)
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
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

        self.classifier = nn.Sequential(
            # Also adds dropout. Used for regularization, a way to combat overfitting.
            # Reduces the models tendency to rely on certain neurons too much
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (Batch_size, 1, 240)
        x = self.features(x)
        # output: (batch_size, 128, 1)
        x = x.squeeze(dim=-1)
        # output: (batch_size, 128)
        return self.classifier(x)
