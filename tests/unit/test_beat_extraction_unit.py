from src.data.beat_extraction import extract_beats
import numpy as np

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

def test_ignores_non_beat_annotations():
    signal = np.arange(300)

    annotation_samples = np.array([100, 120])

    # "+" is not a heartbeat symbol
    annotation_symbols = ["N", "+"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    assert beats_matrix.shape == (1, 240)
    assert labels.shape == (1,)
    # "+" should have been filtered out, so only "N" remains
    assert labels[0] == "N"

def test_skips_beat_too_close_to_start():
    signal = np.arange(300)

    annotation_samples = np.array([50, 100])
    annotation_symbols = ["N", "V"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # sample 50 would require start = -40, so it should be skipped.
    # sample 100 is valid.
    assert beats_matrix.shape == (1, 240)
    assert labels.tolist() == ["V"]

def test_skips_beat_too_close_to_end():
    signal = np.arange(300)

    # 250
    annotation_samples = np.array([100, 250])
    annotation_symbols = ["N", "L"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols
    )

    # sample 250 would require end = 400, but len(signal) = 300.
    # So only sample 100 should remain.
    assert beats_matrix.shape == (1,240)
    assert labels.tolist() == ["N"]

def test_extracts_multiple_valid_beats_in_order():
    signal = np.arange(400)

    annotation_samples = np.array([100, 200])
    annotation_symbols = ["N", "V"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # Both annotations are valid beats, 
    # so both windows should be extracted.
    assert beats_matrix.shape == (2, 240)
    assert labels.tolist() == ["N", "V"]

    # For annotation 1: start = 100 - 90 = 10, end = 100 + 150 = 250
    # For annotation 2: start = 200 - 90 = 110, end = 200 + 150 = 350
    assert np.array_equal(beats_matrix[0], signal[10:250])
    assert np.array_equal(beats_matrix[1], signal[110:350])

def test_returns_empty_arrays_when_no_valid_beats_are_extracted():
    signal = np.arange(300)

    annotation_samples = np.array([50])
    annotation_symbols = ["N"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # This annotation will not be accepted since 50 - 90 = - 40
    # and thus should be skipped. So we should have no windows or labels
    assert beats_matrix.shape == (0, 240)
    assert labels.shape == (0,)

def test_returns_empty_arrays_when_all_annotations_are_non_beats():
    signal = np.arange(300)

    annotation_samples = np.array([100, 120])
    # These are not valid beats
    annotation_symbols = ["+", "~"]

    beats_matrix, labels = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # Hence we should not have any windows or labels.
    assert beats_matrix.shape == (0, 240)
    assert labels.shape == (0,)
