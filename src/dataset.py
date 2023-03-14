from dataclasses import dataclass
import numpy as np

from .observable import Observable

@dataclass
class DiscriminatorData:
    label: str
    suffix: str
    dim: int
    train_true: np.ndarray
    train_fake: np.ndarray
    test_true: np.ndarray
    test_fake: np.ndarray
    val_true: np.ndarray
    val_fake: np.ndarray
    observables: list[Observable]
