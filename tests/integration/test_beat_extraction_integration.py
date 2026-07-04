import numpy as np
import pytest

from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.preprocessing.beat_extraction import BEAT_SYMBOLS, extract_beats


@pytest.mark.integration
def test_extract_beats_real_mitdb_record():

    # Retrieve a record
    record_name = "100"
    signals, fields, annotation = load_record(record_name=record_name)

    # Select a signal channel
    signal, _ = select_signal_channel(
        signals=signals, fields=fields, preferred_lead="MLII"
    )

    if annotation.symbol is None:
        raise ValueError("Annotation symbols cannot be None")

    # Extract windows, corresponding labels, and RR features
    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation.sample,
        annotation_symbols=annotation.symbol,
    )

    # It should have returned some windows
    assert beats_matrix.shape[0] > 0

    # There should be 240 columns (amplitude values)
    assert beats_matrix.shape[1] == 240

    # Each window should be assigned a label and one RR feature row.
    assert beats_matrix.shape[0] == labels.shape[0]
    assert labels.shape[0] == rr_features.shape[0]

    # Each beat should have 2 RR features:
    # previous RR interval in seconds and previous RR/local mean RR ratio.
    assert rr_features.shape[1] == 2

    # RR intervals and ratios should be finite positive values.
    assert np.all(np.isfinite(rr_features))
    assert np.all(rr_features[:, 0] > 0)
    assert np.all(rr_features[:, 1] > 0)

    # Return each unique label
    unique_labels = np.unique(labels)

    # Each label must be one of the beat symbols used by extract_beats.
    for label in unique_labels:
        assert label in BEAT_SYMBOLS
