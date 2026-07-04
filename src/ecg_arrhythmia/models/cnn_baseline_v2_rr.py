import torch
from torch import nn


class CNNBaselineV2RR(nn.Module):
    """
    CNN baseline using both ECG morphology and RR timing features.

    ECG windows are processed by the CNN branch.
    RR features are processed by a small dense branch.
    The two representations are concatenated before classification.
    """

    def __init__(
        self,
        num_classes: int = 4,
        rr_feature_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Same architecture as V2
        self.ecg_features = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(output_size=1),
            # output: (batch_size, 128, 1)
        )

        self.rr_features = nn.Sequential(
            # input: (batch_size, 2)
            nn.Linear(in_features=rr_feature_dim, out_features=16),
            # output: (batch_size, 16)
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128 + 16, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        # x input: (batch_size, 1, 240)
        x = self.ecg_features(x)
        # output: (batch_size, 128, 1)
        x = x.squeeze(dim=-1)
        # output: (batch_size, 128)

        # rr input: (batch_size, 2)
        rr = self.rr_features(rr)
        # output: (batch_size, 16)

        # Joins them along the feature dimension
        # x : (batch_size, 128)
        # rr: (batch_size, 16)
        # com:(batch_size, 144)
        # example
        combined = torch.cat([x, rr], dim=1)
        # output: (batch_size, 144)

        return self.classifier(combined)


## Combined example for future reference:
# x = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6],
# ]) shape = (2,3)

# rr = torch.tensor([
#     [10, 20],
#     [30, 40],
# ]) shape = (2,2)
# combine = torch.cat([x, rr], dim=1)
#
# com = torch.tensor([
#       [1,2,3,10,20],
#       [4,5,6,30,40]
# ]) shape = (2,5)
# for each beat, attach its RR timing features to its ECG morphology features
# So the final classifier predicts the label using both what the beat looks like
# and when the beat occurred relative to recent rhythm
