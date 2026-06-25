from src.data.beat_extraction import extract_beats
from src.data.load_record import load_record, select_signal_channel
import numpy as np
import pytest

@pytest.mark.integration
def test_beat_window_size():

    # Retrive a record
    record_name = "100"
    signals, fields, annotation = load_record(record_name=record_name)

    # Select a signal channel
    signal, _ = select_signal_channel(
        signals=signals,
        fields=fields,
        preferred_lead="MLII"
    )

    if annotation.symbol is None:
        raise ValueError("Annotation symbols cannot be None")

    # Extract windows and corresponding labels
    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation.sample,
        annotation_symbols=annotation.symbol
    )

    # Non-beat annotations
    beat_symbols = {
        "N", "L", "R", "B",
        "A", "a", "J", "S",
        "V", "r", "F",
        "e", "j", "n", "E",
        "/", "f", "Q", "?"
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

def test_extracts_single_beat_window_correctly():
    # Gives a fake signal array of np.array([0,1,2,...,299])
    signal = np.arange(300)

    # Annotation symbol "N" is at signal index 100
    annotation_samples = np.array([100])
    annotation_symbols = ["N"]

    # Extract window and corresponding labels
    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # It should have created 1 window with 240 values
    assert beats_matrix.shape == (1, 240)
    # That one window should have been assigned 1 label
    assert labels.shape == (1,)
    # That one label should have been "N"
    assert labels[0] == "N"

    # sample_index = 100
    # start = 100 - 90 = 10
    # end = 100 + 150 = 250
    expected_beat = signal[10:250]

    # First row (Our only row) should be this expected beat
    assert np.array_equal(beats_matrix[0], expected_beat)