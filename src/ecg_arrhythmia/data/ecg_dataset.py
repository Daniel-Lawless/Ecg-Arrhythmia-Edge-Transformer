from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# PyTorch’s nn.CrossEntropyLoss() expects the
# target labels to be integer class indices,
# as a LongTensor:
LABEL_TO_INDEX = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}


class ECGDataset(Dataset):
    # Loads data.
    def __init__(self, split_path: Path):
        # Load the data from which ever split is chosen
        self.X = np.load(split_path / "X.npy")
        self.y = np.load(split_path / "y.npy")

        # Convert labels to its corresponding integer value
        self.y_indices = np.array([LABEL_TO_INDEX[label] for label in self.y])

    # Tells Pytorch how many samples exist.
    def __len__(self):
        return len(self.y_indices)

    # tell PyTorch how to fetch one sample
    # This is meant to return one sample only without the batch dimension
    # DataLoader adds the batch dimension later.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x_tensor shape would be (240,), y_tensor would be a scalar tensor: shape ()
        x_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_indices[index], dtype=torch.long)

        # conv1d expects x_tensor shape (channels, sequence length).
        # So we have to add another dimension at the start.
        # Shape is now (1, 240)
        x_tensor = x_tensor.unsqueeze(0)

        return x_tensor, y_tensor


# Similar to PyTorch models, this is the standard pattern for creating PyTorch datasets.
# __init__(): load and store the data.
# __len__(): tell PyTorch how many samples are in the dataset.
# __getitem__(): return one sample and its label.
