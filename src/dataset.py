from dataclasses import dataclass
from torch.utils.data import Dataset

from .observable import Observable

@dataclass
class DiscriminatorData:
    label: str
    suffix: str
    dim: int
    train_true: Dataset
    train_fake: Dataset
    test_true: Dataset
    test_fake: Dataset
    val_true: Dataset
    val_fake: Dataset
    observables: list[Observable]
