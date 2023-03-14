import warnings
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from .observable import Observable

class Plots:
    def __init__(
        self,
        observables: list[Observable],
        weights_true: np.ndarray,
        weights_fake: np.ndarray,
        title: str
    ):
        self.observables = observables
        self.bayesian = len(weights_true.shape) == 2
        self.true_mask = np.all(np.isfinite(
            weights_true if self.bayesian else weights_true[:,None]
        ), axis=1)
        self.fake_mask = np.all(np.isfinite(
            weights_fake if self.bayesian else weights_fake[:,None]
        ), axis=1)
        self.weights_true = weights_true[self.true_mask]
        self.weights_fake = weights_fake[self.fake_mask]
        self.title = title

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]


    def plot_losses(self, file: str, losses: dict):
        with PdfPages(file) as pdf:
            self.plot_single_loss(
                pdf,
                "loss",
                (losses["train_loss"], losses["val_loss"]),
                ("train", "val")
            )
            if self.bayesian:
                self.plot_single_loss(
                    pdf,
                    "BCE loss",
                    (losses["train_bce_loss"], losses["val_bce_loss"]),
                    ("train", "val")
                )
                self.plot_single_loss(
                    pdf,
                    "KL loss",
                    (losses["train_kl_loss"], losses["val_kl_loss"]),
                    ("train", "val")
                )
            self.plot_single_loss(
                pdf,
                "learning rate",
                (losses["lr"], ),
                (None, )
            )


    def plot_single_loss(
        self,
        pdf: PdfPages,
        ylabel: str,
        curves: tuple[np.ndarray],
        labels: tuple[str]
    ):
        fig, ax = plt.subplots(figsize=(4,3.5))
        for i, (curve, label) in enumerate(zip(curves, labels)):
            epochs = np.arange(1, len(curve)+1)
            ax.plot(epochs, curve, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.text(
            x = 0.95,
            y = 0.95,
            s = self.title,
            horizontalalignment = "right",
            verticalalignment = "top",
            transform = ax.transAxes
        )
        if any(label is not None for label in labels):
            ax.legend(loc="center right", frameon=False)
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()


    def plot_roc(self, file: str):
        scores = np.concatenate((self.weights_true, self.weights_fake), axis=0)
        labels = np.concatenate((
            np.ones_like(self.weights_true),
            np.zeros_like(self.weights_fake)
        ), axis=0)
        index = np.argsort(scores, axis=0)
        sorted_labels = np.take_along_axis(labels, index, axis=0)
        tpr = np.concatenate((
            np.zeros_like(self.weights_true[:1,...]),
            np.cumsum(sorted_labels, axis=0) / self.weights_true.shape[0]
        ), axis=0)
        fpr = np.concatenate((
            np.zeros_like(self.weights_fake[:1,...]),
            np.cumsum(1-sorted_labels, axis=0) / self.weights_fake.shape[0]
        ), axis=0)
        auc = np.trapz(x=fpr, y=tpr, axis=0)

        fig, ax = plt.subplots(figsize=(4,3.5))
        if self.bayesian:
            for i in range(self.weights_fake.shape[1]):
                ax.plot(fpr[:,i], tpr[:,i], alpha=0.3, color=self.colors[0])
        else:
            ax.plot(fpr, tpr, color=self.colors[0])
        ax.plot([0,1], [0,1], color="k", ls="dashed")

        ax.set_xlabel(r"$\epsilon_S$")
        ax.set_ylabel(r"$\epsilon_B$")
        ax.text(
            x = 0.05,
            y = 0.95,
            s = f"AUC = ${np.mean(auc):.3f} \\pm {np.std(auc):.3f}$"
                if self.bayesian else f"AUC = {auc:.3f}",
            horizontalalignment = "left",
            verticalalignment = "top",
            transform = ax.transAxes
        )
        ax.text(
            x = 0.95,
            y = 0.05,
            s = self.title,
            horizontalalignment = "right",
            verticalalignment = "bottom",
            transform = ax.transAxes
        )
        plt.savefig(file, bbox_inches="tight")
        plt.close()


    def plot_weight_hist(self, file: str):
        with PdfPages(file) as pdf:
            clean_array = lambda a: a[np.isfinite(a)]
            wmin = min(
                np.min(self.weights_true[self.weights_true != 0]),
                np.min(self.weights_fake[self.weights_fake != 0])
            ) 
            wmax = max(np.max(self.weights_true), np.max(self.weights_fake))
            self.plot_single_weight_hist(
                pdf,
                bins=np.linspace(0, 3, 50),
                xscale="linear",
                yscale="linear"
            )
            self.plot_single_weight_hist(
                pdf,
                bins=np.logspace(np.log10(wmin), np.log10(wmax), 50),
                xscale="log",
                yscale="log"
            )
            self.plot_single_weight_hist(
                pdf,
                bins=np.logspace(-2, 1, 50),
                xscale="log",
                yscale="log"
            )


    def plot_single_weight_hist(
        self,
        pdf: PdfPages,
        bins: np.ndarray,
        xscale: str,
        yscale: str
    ):
        weights_combined = np.concatenate((self.weights_true, self.weights_fake), axis=0)
        if self.bayesian:
            true_hists = np.stack([
                np.histogram(
                    self.weights_true[:,i] / np.mean(self.weights_true[:,i]),
                    bins=bins
                )[0] for i in range(self.weights_true.shape[1])
            ], axis=1)
            fake_hists = np.stack([
                np.histogram(
                    self.weights_fake[:,i] / np.mean(self.weights_fake[:,i]),
                    bins=bins
                )[0] for i in range(self.weights_fake.shape[1])
            ], axis=1)
            combined_hists = np.stack([
                np.histogram(
                    weights_combined[:,i] / np.mean(weights_combined[:,i]),
                    bins=bins
                )[0] for i in range(weights_combined.shape[1])
            ], axis=1)

            y_true = np.mean(true_hists, axis=1)
            y_true_err = np.std(true_hists, axis=1)
            y_fake = np.mean(fake_hists, axis=1)
            y_fake_err = np.std(fake_hists, axis=1)
            y_combined = np.mean(combined_hists, axis=1)
            y_combined_err = np.std(combined_hists, axis=1)

        else:
            y_true = np.histogram(self.weights_true, bins=bins)[0]
            y_true_err = None
            y_fake = np.histogram(self.weights_fake, bins=bins)[0]
            y_fake_err = None
            y_combined = np.histogram(weights_combined, bins=bins)[0]
            y_combined_err = None

        fig, ax = plt.subplots(figsize=(4, 3.5))
        self.hist_line(
            ax,
            bins,
            y_combined,
            y_combined_err,
            label = "Comb",
            color = self.colors[0]
        )
        self.hist_line(
            ax,
            bins,
            y_true,
            y_true_err,
            label = "Truth",
            color = self.colors[1]
        )
        self.hist_line(
            ax,
            bins,
            y_fake,
            y_fake_err,
            label = "Gen",
            color = self.colors[2]
        )
        ax.text(
            x = 0.95,
            y = 0.95,
            s = self.title,
            horizontalalignment = "right",
            verticalalignment = "top",
            transform = ax.transAxes
        )
        ax.set_xlabel("weight")
        ax.set_ylabel("normalized")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(bins[0], bins[-1])
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()


    def plot_observables(self, file: str):
        with PdfPages(file) as pdf:
            for observable in self.observables:
                self.plot_single_observable(pdf, observable)


    def plot_single_observable(self, pdf: PdfPages, observable: Observable):
        bins = observable.bins

        if self.bayesian:
            rw_hists = np.stack([
                np.histogram(
                    observable.fake_data[self.fake_mask],
                    bins = bins,
                    weights = self.weights_fake[:,i],
                    density = True
                )[0] for i in range(self.weights_fake.shape[1])
            ], axis=1)
            rw_mean = np.mean(rw_hists, axis=1)
            rw_std = np.std(rw_hists, axis=1)
        else:
            rw_mean = np.histogram(
                observable.fake_data[self.fake_mask],
                bins=bins,
                weights=self.weights_fake
            )[0]
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
                3, 1,
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
                    axs[0],
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
                    self.hist_line(ax, bins, ratio, ratio_err, label=None, color=line.color)

            axs[0].legend(frameon=False)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale)

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
            axs[1].set_yticks([0.8,1,1.2])
            axs[1].set_ylim([0.75,1.25])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

            axs[2].set_ylabel(r"$w$")
            unit = "" if observable.unit is None else f" [{observable.unit}]"
            axs[2].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[2].set_xscale(observable.xscale)
            axs[2].set_xlim(bins[0], bins[-1])

            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()


    def plot_clustering(
        self,
        file: str,
        low_cutoffs: list[float],
        high_cutoffs: list[float]
    ):
        with PdfPages(file) as pdf:
            for observable in self.observables:
                self.plot_single_clustering(pdf, observable, low_cutoffs, high_cutoffs)


    def plot_single_clustering(
        self,
        pdf: PdfPages,
        observable: Observable,
        low_cutoffs: list[float],
        high_cutoffs: list[float]
    ):
        pass


    def hist_line(
        self,
        ax: mpl.axes.Axes,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str
    ):
        dup_last = lambda a: np.append(a, a[-1])

        ax.step(
            bins,
            dup_last(y),
            label = label,
            color = color,
            linewidth = 1.0,
            where = "post",
        )
        if y_err is not None:
            ax.step(
                bins,
                dup_last(y + y_err),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            ax.step(
                bins,
                dup_last(y - y_err),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            ax.fill_between(
                bins,
                dup_last(y - y_err),
                dup_last(y + y_err),
                facecolor = color,
                alpha = 0.3,
                step = "post"
            )
