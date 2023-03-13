from abc import ABC, abstractmethod
import numpy as np


class CustomColorChecker(ABC):
    """This class offers core functionality that any color checker class should
    have"""

    @abstractmethod
    def extract_reference_lab_colors_from_bgr_img(self) -> np.ndarray:
        """This class must be provided by any implementation of this
        abstract class."""
        pass
