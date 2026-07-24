from abc import abstractmethod

from neurokit2.ecg.ecg_clean import ecg_clean
from neurokit2.ecg.ecg_peaks import ecg_peaks
import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector


class NeuroKitRPeakDetector(RPeakDetector):
    """
    Base wrapper for R-peak detectors provided by NeuroKit2.

    A subclass selects an NeuroKit2 algorithm by setting
    ``_neurokit_method``. The same identifier is used both for the
    algorithm-specific cleaning step and for peak detection, mirroring
    NeuroKit2's own ``ecg_process`` pipeline.
    """

    # NeuroKit2 method identifier, for example "hamilton2002" or
    # "elgendi2010". Subclasses must provide a concrete value.
    _neurokit_method: str

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable detector identifier."""

    def _detect(
        self,
        signal: NDArray[np.float64],
        sampling_rate: float,
    ) -> NDArray[np.int64]:

        if not sampling_rate.is_integer():
            raise ValueError(
                "NeuroKit2 requires sampling_rate to be a whole number."
            )

        neurokit_sampling_rate = int(sampling_rate)

        # Apply the algorithm-specific preprocessing once.
        cleaned_signal = ecg_clean(
            signal,
            sampling_rate=neurokit_sampling_rate,
            method=self._neurokit_method,
        )

        # Detect R-peaks on the cleaned signal. ecg_peaks returns a
        # (signals, info) pair, the absolute sample indices are stored
        # in info under the "ECG_R_Peaks" key.
        _, info = ecg_peaks(
            cleaned_signal,
            sampling_rate=neurokit_sampling_rate,
            method=self._neurokit_method,
        )

        return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
