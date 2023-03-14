from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class Observable:
    """
    Data class for an observable used for plotting

    Args:
        true_data: Numpy array with the observable values for the truth dataset
        fake_data: Numpy array with the observable values for the generated dataset
        tex_label: Observable name in LaTeX for labels in plots
        bins: Numpy array with the bin boundaries used for histograms
        xscale: X axis scale, "linear" (default) or "log", optional
        yscale: Y axis scale, "linear" (default) or "log", optional
        unit: Unit of the observable or None, if dimensionless, optional
    """
    true_data: np.ndarray
    fake_data: np.ndarray
    tex_label: str
    bins: np.ndarray
    xscale: str = "linear"
    yscale: str = "linear"
    unit: Optional[str] = None
