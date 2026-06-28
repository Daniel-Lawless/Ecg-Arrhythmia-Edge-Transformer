import numpy as np
import pytest

from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.preprocessing.beat_extraction import extract_beats


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

    # Extract windows and corresponding labels
    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation.sample,
        annotation_symbols=annotation.symbol,
    )

    # Valid beat annotations
    beat_symbols = {
        "N",
        "L",
        "R",
        "B",
        "A",
        "a",
        "J",
        "S",
        "V",
        "r",
        "F",
        "e",
        "j",
        "n",
        "E",
        "/",
        "f",
        "Q",
        "?",
    }

    # It should have returned some windows
    assert beats_matrix.shape[0] > 0

    # There should be 240 columns (amplitude values)
    assert beats_matrix.shape[1] == 240

    # Each window should be assigned a label.
    assert beats_matrix.shape[0] == labels.shape[0]

    # Return each unique label
    unique_labels = np.unique(labels)

    # each label must be one of the allowed beat symbols
    for label in unique_labels:
        assert label in beat_symbols
