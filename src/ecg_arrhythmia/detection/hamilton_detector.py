from .neurokit_detector import NeuroKitRPeakDetector


class HamiltonDetector(NeuroKitRPeakDetector):
    """
    R-peak detector using NeuroKit2's implementation of the Hamilton
    (2002) QRS detection algorithm.S
    """

    _neurokit_method = "hamilton2002"

    @property
    def name(self) -> str:
        return "hamilton"
