from dataclasses import dataclass
import numpy as np

from .observable import Observable

@dataclass
class DiscriminatorData:
    """
    Class containing the training and testing data for a discriminator.

    Args:
        label: Name of the dataset used in plots
        suffix: Name of the dataset used as suffix in output file names
        dim: Dimension of the training data
        train_true: Training data (truth samples)
        train_fake: Training data (generated samples)
        test_true: Test data (truth samples)
        test_fake: Test data (generated samples)
        val_true: Validation data (truth samples)
        val_fake: Validation data (generated samples)
        observables: List observables for plotting
    """
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
