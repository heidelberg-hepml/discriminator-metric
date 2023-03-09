import numpy as np
import matplotlib.pyplot as plt

from .observable import Observable

class Plots:
    def __init__(
        self,
        observables: List[Observable],
        weights_true: np.ndarray,
        weights_fake: np.ndarray
    ):
        self.observables = observables
        self.weights_true = weights_true
        self.weights_fake = weights_fake

        #TODO: matplotlib settings

    def plot_roc(self, file: str):
        #true vs fake
        #fake vs fake
        pass

    def plot_weight_hist(self, file: str):
        pass

    def plot_observables(self, file: str):
        pass

    def plot_clustering(self, file: str):
        pass

    def plot_single_observable(self, observable: Observable):
        pass

    def plot_single_clustering(self, observable: Observable):
        pass
