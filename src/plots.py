import warnings
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        self.bayesian = len(self.weights_true.shape) == 2

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]


    def plot_roc(self, file: str):
        #true vs fake
        #fake vs fake
        fig, ax = plt.subplots(figsize=(4,3.5))

        pass


    def plot_weight_hist(self, file: str):
        pass


    def plot_observables(self, file: str):
        with PdfPages(file) as pp:
            for observable in self.observables:
                self.plot_single_observable(pp, observable)


    def plot_single_observable(self, file, observable: Observable):
        bins = observable.bins

        if self.bayesian:
            rw_hists = np.stack([
                np.histogram(
                    observable.fake_data,
                    bins = bins,
                    weights = self.weights_fake[:,i],
                    density = True
                )[0] for i in range(self.weights_fake.shape[1])
            ], axis=1)
            rw_mean = np.mean(rw_hists, axis=1)
            rw_std = np.std(rw_hists, axis=1)
        else:
            rw_mean = np.histogram(
                observable.fake_data,
                bins=bins,
                weights=self.weights_fake
            )
            rw_std = None
        true_hist, _ = np.histogram(observable.true_data, bins=bins, density=True)
        fake_hist, _ = np.histogram(observable.fake_data, bins=bins, density=True)

        Line = namedtuple(
            "Line",
            ["y", "y_err", "y_ref", "y_orig", "label", "color"],
            defaults = [None, None, None, None, None]
        )
        lines = [
            Line(
                y = true_hist,
                label = "Truth",
                color = self.colors[0],
            ),
            Line(
                y = fake_hist,
                y_ref = true_hist,
                label = "Gen",
                color = self.colors[1],
            ),
            Line(
                y = rw_mean,
                y_err = rw_std,
                y_ref = true_hist,
                y_orig = fake_hist,
                label = "Reweighted",
                color = self.colors[2],
            ),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            fig, axs = plt.subplots(
                2, 1,
                sharex = True,
                figsize = (6, 4.5),
                gridspec_kw = {"height_ratios": (4, 1, 1), "hspace": 0.00}
            )

            for line in lines:
                integral = np.sum((bins[1:] - bins[:-1]) * line.y)
                scale = 1 / integral if integral != 0. else 1.
                if line.y_ref is not None:
                    ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                    ref_scale = 1 / ref_integral if ref_integral != 0. else 1.

                self.hist_line(
                    bins,
                    line.y * scale,
                    line.y_err * scale if line.y_err is not None else None,
                    label=line.label,
                    color=line.color
                )

                ratio_panels = []
                if line.y_ref is not None:
                    ratio_panels.append((axs[1], line.y_ref))
                if line.y_orig is not None:
                    ratio_panels.append((axs[2], line.y_orig))

                for ax, y_ref in ratio_panels:
                    ratio = (line.y * scale) / (y_ref * ref_scale)
                    ratio_isnan = np.isnan(ratio)
                    if line.y_err is not None:
                        ratio_err = np.sqrt((line.y_err / line.y)**2)
                        ratio_err[ratio_isnan] = 0.
                    else:
                        ratio_err = None
                    ratio[ratio_isnan] = 1.
                    self.hist_line(bins, ratio, ratio_err, label=None, color=line.color)

            axs[0].legend(loc=legend_pos, frameon=False)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale)

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
            axs[1].set_yticks([0.8,1,1.2])
            axs[1].set_ylim([0.75,1.25])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

            axs[2].set_ylabel(r"$w$")

            unit = "" if observable.unit is None else f" [{unit}]"
            axs[0].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[0].set_xscale(xscale)
            axs[0].set_xlim(bins[0], bins[-1])
                
            plt.savefig(file, format="pdf")
            plt.close()


    def hist_line(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str,
        ls: str
    ):
        dup_last = lambda a: np.append(a, a[-1])

        axs[0].step(
            bins,
            dup_last(y),
            label = label,
            color = color,
            linewidth = 1.0,
            where = "post",
        )
        if y_err is not None:
            axs[0].step(
                bins,
                dup_last(y + y_err),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            axs[0].step(
                bins,
                dup_last(y - y_err),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            axs[0].fill_between(
                bins,
                dup_last(y - y_err),
                dup_last(y + y_err),
                facecolor = color,
                alpha = 0.3,
                step = "post"
            )


    def plot_clustering(
        self,
        file: str,
        low_cutoffs: list[float],
        high_cutoffs: list[float]
    ):
        with PdfPages(file) as pp:
            for observable in self.observables:
                self.plot_single_observable(pp, observable)


    def plot_single_clustering(
        self,
        observable: Observable,
        low_cutoffs: list[float],
        high_cutoffs: list[float]
    ):
        pass
