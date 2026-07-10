import pytest
import torch

from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer


def test_sequence_transformer_returns_expected_logits_shape():
    model = ECGSequenceTransformer(num_classes=4)

    batch_size = 8
    sequence_length = 5
    window_size = 240

    x = torch.randn(batch_size, sequence_length, 1, window_size)
    rr = torch.randn(batch_size, sequence_length, 2)

    logits = model(x, rr)

    # Ensure we get the correct shape, we have 4 classes, so there
    # should be 4 predictions
    assert logits.shape == (batch_size, 4)


def test_sequence_transformer_supports_different_sequence_lengths():
    model = ECGSequenceTransformer(num_classes=4, max_sequence_length=20)

    batch_size = 4
    # Should work the same way as long as the sequence length
    # is less than max_sequence_length
    sequence_length = 7
    window_size = 240

    x = torch.randn(batch_size, sequence_length, 1, window_size)
    rr = torch.randn(batch_size, sequence_length, 2)

    logits = model(x, rr)

    assert logits.shape == (batch_size, 4)


def test_sequence_transformer_rejects_zero_sequence_length():
    model = ECGSequenceTransformer(num_classes=4, max_sequence_length=20)

    batch_size = 4
    sequence_length = 0
    window_size = 240

    x = torch.randn(batch_size, sequence_length, 1, window_size)
    rr = torch.randn(batch_size, sequence_length, 2)

    with pytest.raises(ValueError, match="sequence_length must be positive"):
        model(x, rr)
