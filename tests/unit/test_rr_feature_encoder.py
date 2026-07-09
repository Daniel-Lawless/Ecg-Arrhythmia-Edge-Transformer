import torch

from ecg_arrhythmia.models.rr_feature_encoder import RRFeatureEncoder


def test_rr_feature_encoder_returns_expected_shape_for_single_beats():
    model = RRFeatureEncoder(rr_feature_dim=2, embedding_dim=16)

    rr = torch.randn(10, 2)

    embeddings = model(rr)

    assert embeddings.shape == (10, 16)


def test_rr_feature_encoder_returns_expected_shape_for_k_beat_sequences():
    model = RRFeatureEncoder(rr_feature_dim=2, embedding_dim=16)

    rr = torch.randn(4, 5, 2)

    embeddings = model(rr)

    assert embeddings.shape == (4, 5, 16)
