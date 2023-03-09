from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class Observable:
    true_data: np.ndarray,
    fake_data: np.ndarray,
    tex_label: str
    bins: np.ndarray
    xscale: str = "linear"
    yscale: str = "linear"
    unit: Optional[str] = None
