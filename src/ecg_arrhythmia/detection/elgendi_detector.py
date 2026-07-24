from .neurokit_detector import NeuroKitRPeakDetector


class ElgendiDetector(NeuroKitRPeakDetector):
    """
    R-peak detector using NeuroKit2's implementation of the Elgendi et
    al. (2010) QRS detection algorithm.
    """

    _neurokit_method = "elgendi2010"

    @property
    def name(self) -> str:
        return "elgendi"
