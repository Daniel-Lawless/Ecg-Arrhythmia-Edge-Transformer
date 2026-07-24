import numpy as np
from numpy.typing import NDArray
from wfdb import processing

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector


class XQRSDetector(RPeakDetector):
    """
    R-peak detector using WFDB's XQRS algorithm.

    XQRS applies bandpass filtering, moving-wave integration,
    adaptive thresholds, refractory-period checks, and missed-beat
    backsearch to locate QRS complexes.
    """

    def __init__(self, learn: bool = True) -> None:
        # Whether XQRS should learn its initial detection parameters
        # from the ECG signal before running the main detection.
        self.learn = learn

    @property
    def name(self) -> str:
        return "xqrs"

    def _detect(
        self,
        signal: NDArray[np.float64],
        sampling_rate: float,
    ) -> NDArray[np.int64]:
        """
        Detect R-peaks using WFDB's XQRS implementation.
        """

        peak_indices = processing.xqrs_detect(
            sig=signal,
            fs=sampling_rate,
            learn=self.learn,
            verbose=False,
        )

        return np.asarray(peak_indices, dtype=np.int64)


xqrs = XQRSDetector()
