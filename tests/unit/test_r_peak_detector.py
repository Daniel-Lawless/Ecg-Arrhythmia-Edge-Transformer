import numpy as np
import pytest

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector


class DummyRPeakDetector(RPeakDetector):
    def __init__(self, peak_indices: np.ndarray):
        self.peak_indices = peak_indices

    @property
    def name(self) -> str:
        return "dummy"

    def _detect(
        self,
        signal: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        return self.peak_indices


def test_detector_returns_valid_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([100, 300, 500]),
    )

    signal = np.zeros(600)

    peak_indices = detector.detect(
        signal=signal,
        sampling_rate=360,
    )

    np.testing.assert_array_equal(
        peak_indices,
        np.array([100, 300, 500]),
    )

    assert peak_indices.dtype == np.int64


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 500)),
        np.zeros((500, 1)),
    ],
)
def test_detector_rejects_non_one_dimensional_signal(signal):
    detector = DummyRPeakDetector(
        peak_indices=np.array([100]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=signal,
            sampling_rate=360,
        )


def test_detector_rejects_empty_signal():
    detector = DummyRPeakDetector(
        peak_indices=np.array([], dtype=np.int64),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.array([]),
            sampling_rate=360,
        )


def test_detector_rejects_non_finite_signal_values():
    detector = DummyRPeakDetector(
        peak_indices=np.array([100]),
    )

    signal = np.array([0.0, np.nan, 1.0])

    with pytest.raises(ValueError):
        detector.detect(
            signal=signal,
            sampling_rate=360,
        )


@pytest.mark.parametrize(
    "sampling_rate",
    [0, -1, np.inf, np.nan],
)
def test_detector_rejects_invalid_sampling_rate(sampling_rate):
    detector = DummyRPeakDetector(
        peak_indices=np.array([100]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.zeros(500),
            sampling_rate=sampling_rate,
        )


def test_detector_allows_no_detected_peaks():
    detector = DummyRPeakDetector(
        peak_indices=np.array([], dtype=np.int64),
    )

    peak_indices = detector.detect(
        signal=np.zeros(500),
        sampling_rate=360,
    )

    assert peak_indices.shape == (0,)
    assert peak_indices.dtype == np.int64


def test_detector_rejects_negative_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([-1, 100]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.zeros(500),
            sampling_rate=360,
        )


def test_detector_rejects_out_of_range_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([100, 500]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.zeros(500),
            sampling_rate=360,
        )


def test_detector_rejects_duplicate_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([100, 100, 300]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.zeros(500),
            sampling_rate=360,
        )


def test_detector_rejects_unsorted_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([300, 100, 500]),
    )

    with pytest.raises(ValueError):
        detector.detect(
            signal=np.zeros(600),
            sampling_rate=360,
        )


def test_detector_rejects_non_integer_peak_indices():
    detector = DummyRPeakDetector(
        peak_indices=np.array([100.5, 300.2]),
    )

    with pytest.raises(TypeError):
        detector.detect(
            signal=np.zeros(500),
            sampling_rate=360,
        )
