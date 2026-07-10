import torch
from torch import nn

from ecg_arrhythmia.models.cnn_beat_encoder import CNNBeatEncoder
from ecg_arrhythmia.models.rr_feature_encoder import RRFeatureEncoder


class ECGSequenceTransformer(nn.Module):
    """
    CNN + RR + Transformer model for K-beat ECG sequences.

    Input:
    - x:  (batch_size, sequence_length, 1, window_size)
    - rr: (batch_size, sequence_length, rr_feature_dim)

    Output:
    - logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 4,
        ecg_embedding_dim: int = 128,
        rr_embedding_dim: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_sequence_length: int = 20,
    ) -> None:

        super().__init__()

        # Define our encoders.
        self.ecg_encoder = CNNBeatEncoder(embedding_dim=ecg_embedding_dim)
        self.rr_encoder = RRFeatureEncoder(
            rr_feature_dim=2, embedding_dim=rr_embedding_dim, dropout=dropout
        )

        # This will represent the size of the embedding going into the model
        combined_embedding_dim = ecg_embedding_dim + rr_embedding_dim

        # projects the in features to give us the number of features
        # the transformer expects
        self.input_projection = nn.Linear(
            in_features=combined_embedding_dim, out_features=d_model
        )

        # This creates a learnable positonal embedding
        # for the transformer. For instance with (1, 20, 128)
        # this would mean 1 batch place holder, 20 possible beat
        # positions, and 128 learned values per beat position
        # nn.Parameter says this should be learned during training
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_sequence_length, d_model)
        )

        # This does many things internally. It performs multi-head self-attention,
        # which computes attention scores, between the beats, adds layernorm,
        # adds feedforward MLP, and dropout. The input and output shape remains
        # the same.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # encoder_layer determines what each encoder layer looks like,
        # then this stacks this architecutre num_layers times
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_model, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected x with shape (batch_size, sequence_length, 1, window_size). "
                f"Found {tuple(x.shape)}"
            )

        if rr.ndim != 3:
            raise ValueError(
                "Expected rr with shape (batch_size, sequence_length, rr_feature_dim). "
                f"Found {tuple(rr.shape)}"
            )

        if rr.shape[-1] != 2:
            raise ValueError(
                "Expected rr final dimension to contain 2 RR features. "
                f"Found {rr.shape[-1]}"
            )

        batch_size, sequence_length, channels, window_size = x.shape

        if channels != 1:
            raise ValueError(f"Expected one ECG channel. Found {channels}")

        if rr.shape[0] != batch_size or rr.shape[1] != sequence_length:
            raise ValueError(
                "x and rr must have the same batch_size and sequence_length. "
                f"Found x={tuple(x.shape)}, rr={tuple(rr.shape)}"
            )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
                f"Found sequence_length={sequence_length}"
            )

        if sequence_length > self.position_embedding.shape[1]:
            raise ValueError(
                f"sequence_length={sequence_length} is larger than "
                f"max_sequence_length={self.position_embedding.shape[1]}"
            )

        # Flatten batch and sequence dimensions so the same CNN processes
        # every beat independently.
        x = x.reshape(batch_size * sequence_length, channels, window_size)
        # shape: (batch_size * sequence_length, 1, 240)

        # Then we get an embedding for each of those beats
        ecg_embeddings = self.ecg_encoder(x)
        # shape: (batch_size * sequence_length, ecg_embedding_dim)

        ecg_embeddings = ecg_embeddings.reshape(batch_size, sequence_length, -1)
        # shape: (batch_size, sequence_length, ecg_embedding_dim)

        # Now we have a sequences of CNN beat embeddings
        # [
        #   embedding for beat i-4,
        #   embedding for beat i-3,
        #   embedding for beat i-2,
        #   embedding for beat i-1,
        #   embedding for beat i
        # ], etc...

        # We don't have to flatten this because the rr_encoder using nn.Linear
        # layers.
        rr_embeddings = self.rr_encoder(rr)
        # shape: (batch_size, sequence_length, rr_embedding_dim)

        # Now we have a sequences of beat embeddings and rr embeddings
        beat_embeddings = torch.cat([ecg_embeddings, rr_embeddings], dim=-1)
        # shape: (batch_size, sequence_length, ecg_embedding_dim + rr_embedding_dim)
        # i.e.,
        # [
        #   beat 0: [ECG features + RR features],
        #   beat 1: [ECG features + RR features],
        #   beat 2: [ECG features + RR features],
        #   beat 3: [ECG features + RR features],
        #   beat 4: [ECG features + RR features],
        # ]

        # Projects it to have the number of features the transformer expects
        transformer_input = self.input_projection(beat_embeddings)
        # shape: (batch_size, sequence_length, d_model)

        # This adds postional information to each beat embedding.
        # takes only the first sequence_length positional embeddings and adds
        # them to each beats embedding. This tells the transformer where the beat
        # is in the sequence.
        transformer_input = (
            transformer_input + self.position_embedding[:, :sequence_length, :]
        )
        # shape: (batch_size, sequence_length, d_model)

        # Pass this input through the model.
        transformer_output = self.transformer(transformer_input)
        # shape: (batch_size, sequence_length, d_model)

        # Selects the final beat embedding, because the target label is for
        # the final beat in the causal K-beat sequence.
        final_token = transformer_output[:, -1, :]
        # shape: (batch_size, d_model)

        # For each sequence, we now have 128 learned values representing
        # the final beats ECG morphology and RR features, and includes
        # information from the previous beats and the rhythm/context
        # relationships across the sequence.

        # We then pass this into the classifer to make predictions
        logits = self.classifier(final_token)
        # shape: (batch_size, num_classes)

        return logits
