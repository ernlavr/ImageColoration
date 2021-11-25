from dataclasses import dataclass
import numpy as np

@dataclass
class Image():
    """[summary] Container of a single image dataset
        To get an RGB from Gray and AB, use the following steps
        1. LAB = cv2.merge(Gray, AB)
        2. RGB = cv2.cvtColor(MERGED, cv2.COLOR_LAB2BGR)
    """
    
    RGB : np.ndarray
    Gray : np.ndarray
    AB : np.ndarray
