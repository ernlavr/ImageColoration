from dataclasses import dataclass
import numpy as np

@dataclass
class Image():
    """[summary] Container of a single image's colored and grayed data
    """
    colorData : np.ndarray
    grayData : np.ndarray
    AB : np.ndarray
