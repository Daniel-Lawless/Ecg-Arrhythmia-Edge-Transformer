import numpy as np

from ecg_arrhythmia.preprocessing.beat_extraction import extract_beats

SAMPLING_RATE = 360


def test_extracts_single_beat_window_correctly():
    # Gives a fake signal array of np.array([0,1,2,...,399])
    signal = np.arange(400)

    # The first annotation provides the previous beat needed for RR.
    # The second annotation symbol "N" is at signal index 200.
    annotation_samples = np.array([100, 200])
    annotation_symbols = ["N", "N"]

    # Extract window, corresponding labels, and RR features
    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # It should have created 1 window with 240 values
    assert beats_matrix.shape == (1, 240)
    # That one window should have been assigned 1 label
    assert labels.shape == (1,)
    # That one beat should have 2 RR features: prev_rr_seconds and rr_ratio
    assert rr_features.shape == (1, 2)
    # That one label should have been "N"
    assert labels[0] == "N"

    # sample_index = 200
    # start = 200 - 90 = 110
    # end = 200 + 150 = 350
    expected_beat = signal[110:350]

    # First row (Our only row) should be this expected beat
    assert np.array_equal(beats_matrix[0], expected_beat)

    # Previous RR = 200 - 100 = 100 samples
    expected_prev_rr_seconds = 100 / SAMPLING_RATE
    assert np.isclose(rr_features[0, 0], expected_prev_rr_seconds)

    # Since this is the first extracted RR value, the local mean is the same
    # as prev_rr_seconds, so the ratio should be approximately 1.
    assert np.isclose(rr_features[0, 1], 1.0)


def test_ignores_non_beat_annotations():
    signal = np.arange(400)

    annotation_samples = np.array([100, 200, 220])

    # "+" is not a heartbeat symbol
    annotation_symbols = ["N", "V", "+"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    assert beats_matrix.shape == (1, 240)
    assert labels.shape == (1,)
    assert rr_features.shape == (1, 2)
    # "+" should have been filtered out, so only "V" remains.
    # The first "N" is used as the previous beat for the RR interval.
    assert labels[0] == "V"


def test_skips_beat_too_close_to_start():
    signal = np.arange(300)

    annotation_samples = np.array([50, 100])
    annotation_symbols = ["N", "V"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # sample 50 would require start = -40, so it should be skipped.
    # sample 100 is valid.
    assert beats_matrix.shape == (1, 240)
    assert labels.tolist() == ["V"]
    assert rr_features.shape == (1, 2)


def test_skips_beat_too_close_to_end():
    signal = np.arange(350)

    annotation_samples = np.array([100, 200, 250])
    annotation_symbols = ["N", "V", "L"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # sample 250 would require end = 400, but len(signal) = 350.
    # So only sample 200 should remain. The first sample is only used
    # to calculate the previous RR interval.
    assert beats_matrix.shape == (1, 240)
    assert labels.tolist() == ["V"]
    assert rr_features.shape == (1, 2)


def test_extracts_multiple_valid_beats_in_order():
    signal = np.arange(500)

    annotation_samples = np.array([100, 200, 300])
    annotation_symbols = ["N", "V", "L"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # The first annotation is used to calculate the first RR interval.
    # The second and third annotations are valid beats,
    # so both windows should be extracted.
    assert beats_matrix.shape == (2, 240)
    assert labels.tolist() == ["V", "L"]
    assert rr_features.shape == (2, 2)

    # For annotation 2: start = 200 - 90 = 110, end = 200 + 150 = 350
    # For annotation 3: start = 300 - 90 = 210, end = 300 + 150 = 450
    assert np.array_equal(beats_matrix[0], signal[110:350])
    assert np.array_equal(beats_matrix[1], signal[210:450])

    expected_prev_rr_seconds = 100 / SAMPLING_RATE
    assert np.allclose(rr_features[:, 0], expected_prev_rr_seconds)


def test_returns_empty_arrays_when_no_valid_beats_are_extracted():
    signal = np.arange(300)

    annotation_samples = np.array([50])
    annotation_symbols = ["N"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # This annotation will not be accepted since 50 - 90 = -40,
    # and it also has no previous beat for an RR interval.
    # So we should have no windows, labels, or RR features.
    assert beats_matrix.shape == (0, 240)
    assert labels.shape == (0,)
    assert rr_features.shape == (0, 2)


def test_returns_empty_arrays_when_all_annotations_are_non_beats():
    signal = np.arange(300)

    annotation_samples = np.array([100, 120])
    # These are not valid beats
    annotation_symbols = ["+", "~"]

    beats_matrix, labels, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )

    # Hence we should not have any windows, labels, or RR features.
    assert beats_matrix.shape == (0, 240)
    assert labels.shape == (0,)
    assert rr_features.shape == (0, 2)
