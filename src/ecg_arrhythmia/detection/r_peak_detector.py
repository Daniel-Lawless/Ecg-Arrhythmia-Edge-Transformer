from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class RPeakDetector(ABC):
    """
    Base interface for ECG R-peak detectors.

    Detectors accept a single ECG lead and return the absolute sample
    positions of the detected R-peaks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the detector."""
        raise NotImplementedError

    def detect(
        self,
        signal: NDArray[np.floating],
        sampling_rate: float,
    ) -> NDArray[np.int64]:
        """
        Detect R-peaks in a one-dimensional ECG signal.

        Parameters
        ----------
        signal:
            One-dimensional ECG signal containing a single lead.

        sampling_rate:
            Signal sampling frequency in Hz.

        Returns
        -------
        NDArray[np.int64]
            Strictly increasing absolute sample indices for the
            detected R-peaks.
        """

        validated_signal = self._validate_signal(signal)
        validated_sampling_rate = self._validate_sampling_rate(sampling_rate)

        peak_indices = self._detect(
            signal=validated_signal,
            sampling_rate=validated_sampling_rate,
        )

        return self._validate_peak_indices(
            peak_indices=peak_indices,
            signal_length=len(validated_signal),
        )

    # abstractmethod means subclasses of this class must implement
    # this method.
    @abstractmethod
    def _detect(
        self,
        signal: NDArray[np.float64],
        sampling_rate: float,
    ) -> NDArray[np.int64]:
        """
        Run the detector-specific R-peak detection algorithm.

        The implementation should return the absolute sample indices
        of the detected R-peaks.
        """
        raise NotImplementedError

    # We use staticmethod when the method does not use information
    # stored in the object, so no self or cls is required.
    @staticmethod
    def _validate_signal(
        signal: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        # Convert the input into a NumPy array if necessary and
        # normalise its dtype.
        signal_array = np.asarray(signal, dtype=np.float64)

        # The signal should be a 1D array of amplitude values.
        if signal_array.ndim != 1:
            raise ValueError(
                "ECG signal must be one-dimensional, "
                f"but received shape {signal_array.shape}."
            )

        if signal_array.size == 0:
            raise ValueError("ECG signal must not be empty.")

        if not np.all(np.isfinite(signal_array)):
            raise ValueError("ECG signal must contain only finite values.")

        return signal_array

    @staticmethod
    def _validate_sampling_rate(
        sampling_rate: float,
    ) -> float:
        if not np.isfinite(sampling_rate):
            raise ValueError("Sampling rate must be finite.")

        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be greater than zero.")

        return float(sampling_rate)

    @staticmethod
    def _validate_peak_indices(
        peak_indices: NDArray[np.int64],
        signal_length: int,
    ) -> NDArray[np.int64]:
        # Convert the detector output into a NumPy array.
        peak_array = np.asarray(peak_indices)

        if peak_array.ndim != 1:
            raise ValueError(
                "Detected peak indices must be one-dimensional, "
                f"but received shape {peak_array.shape}."
            )

        # Checks if the inputs are in the np integer family
        # i.e., int8, int16, int32 etc.
        if not np.issubdtype(peak_array.dtype, np.integer):
            raise TypeError("Detected peak indices must contain only integers.")

        # Converts all valid integer outputs to int64 for consistency
        peak_array = peak_array.astype(np.int64, copy=False)

        # An empty array is a valid result when no peaks are found.
        if peak_array.size == 0:
            return peak_array

        if np.any(peak_array < 0):
            raise ValueError("Detected peak indices must not be negative.")

        if np.any(peak_array >= signal_length):
            raise ValueError("Detected peak indices must be inside the ECG signal.")

        # np.diff subtracts each value from the value after it.
        # A zero difference means there is a duplicate, while a
        # negative difference means the indices are out of order.
        if np.any(np.diff(peak_array) <= 0):
            raise ValueError(
                "Detected peak indices must be strictly increasing "
                "and contain no duplicates."
            )

        return peak_array
