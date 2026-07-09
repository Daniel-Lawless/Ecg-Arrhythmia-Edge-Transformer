import torch

from ecg_arrhythmia.models.cnn_beat_encoder import CNNBeatEncoder


def test_cnn_beat_encoder_returns_expected_embedding_shape():
    # Create the model
    model = CNNBeatEncoder(embedding_dim=128)

    # Create a fake x with batch_size 10,
    # 1 ecg channel, and 240 amplitude values
    x = torch.randn(10, 1, 240)

    # Create the embedding
    embeddings = model(x)

    # Ensure it has the shape we expect.
    assert embeddings.shape == (10, 128)


def test_cnn_beat_encoder_supports_flattened_k_beat_batches():
    model = CNNBeatEncoder(embedding_dim=128)

    batch_size = 4
    sequence_length = 5
    window_size = 240

    x = torch.randn(batch_size, sequence_length, 1, window_size)

    # The transformer model will later do this reshape before calling the encoder.
    x_flat = x.reshape(batch_size * sequence_length, 1, window_size)

    embeddings = model(x_flat)

    assert embeddings.shape == (batch_size * sequence_length, 128)

    # Then it will reshape back to sequence form.
    sequence_embeddings = embeddings.reshape(batch_size, sequence_length, 128)

    assert sequence_embeddings.shape == (batch_size, sequence_length, 128)
