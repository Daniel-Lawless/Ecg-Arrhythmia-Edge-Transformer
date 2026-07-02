import torch
from torch import nn


# nn.Module is the base class for almost every NN in Pytorch.
# This says that our model is a pytorch model. This gives
# our model behaviour like mode.parameters, model.train,
# model.eval, model.state_dict() etc.
class CNNBaselineV1(nn.Module):
    def __init__(self, num_classes: int = 4):
        # creates the internal storage PyTorch uses to track
        # our model
        super().__init__()

        # sequential means pass our darat through these
        # layers one after another. This extracts our
        # features
        self.features = nn.Sequential(
            # input: (batch_size, 1, 240)
            # (batch_size, number_of_channels, window_length)
            # To keep the length the same after convolution
            # layers we use padding = (kernel_size - 1) / 2
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            # output: (batch_size, 16, 240)
            # ReLu breaks linearlity so the model can learn complex patterns
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output: (batch_size, 16, 120)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            # output: (batch_size, 32, 120)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # output: (batch_size, 32, 60)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # output: (batch_size, 64, 60)
            nn.ReLU(),
            # This says out of all the 64 learned features,
            # how strongly was the feature present across the window.
            nn.AdaptiveAvgPool1d(output_size=1),
            # output: (batch_size, 64, 1)
        )

        # Fully connected layer to score our classes.
        # This sends the 64 features extracted by the
        # feature extractor into num_classes nodes
        # of shape linear_function = input @ weights.T + bias
        # and it outputs logits for each class. These raw logits
        # can later be passed into softmax to calculate probabilites
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)
        # output shape (batch_size, num_classes). a prediction for each window
        # in batch_size

    # This defines the forward pass of data.
    # Assume one batch of ECG windows comes in with shape: (32, 1, 240)
    # So 32 windows, one ECG lead per window, 240 samples per window.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passes the windows through the CNN feature extractor.
        # This of course when the model has alredy been trained
        # and has its learned weights.
        x = self.features(x)
        # output: (batch_size, 64, 1). It has learned 64
        # features for each window in batch_size, each with length 1

        # Removes the last useless dimension. dim=-1 means the last dimension
        x = x.squeeze(dim=-1)
        # output: (batch_size, 64)
        return self.classifier(x)


# This is the standard pattern for defining PyTorch models:
# inherit from nn.Module so PyTorch treats this class as a model.
# call super().__init__() to initialise PyTorch's internal tracking.
# define the model layers in __init__, e.g. self.features and self.classifier.
# define how data moves through those layers in forward().
