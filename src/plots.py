import warnings
from collections import namedtuple
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from .observable import Observable

Line = namedtuple(
    "Line",
    ["y", "y_err", "y_ref", "y_orig", "label", "color", "fill"],
    defaults = [None, None, None, None, None, False]
)

class Plots:
    """
    Implements the plotting pipeline to evaluate generative models and the corresponding
    discriminator weights.
    """
    def __init__(
        self,
        observables: list[Observable],
        weights_true: np.ndarray,
        weights_fake: np.ndarray,
        losses: dict,
        title: str,
        log_gen_weights: Optional[np.ndarray] = None
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.

        Args:
            observables: List of observables
            weights_true: Discriminator weights for truth samples
            weights_fake: Discriminator weights for generated samples
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            title: Title added in all the plots
            log_gen_weights: For Bayesian generators: sampled log weights
        """
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
        self.losses = losses
        self.title = title
        self.log_gen_weights = log_gen_weights

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]


    def plot_losses(self, file: str):
        """
        Makes plots of the losses (total loss and if bayesian, BCE loss and KL loss
        separately) and learning rate as a function of the epoch.

        Args:
            file: Output file name
        """
        with PdfPages(file) as pdf:
            self.plot_single_loss(
                pdf,
                "loss",
                (self.losses["train_loss"], self.losses["val_loss"]),
                ("train", "val")
            )
            if self.bayesian:
                self.plot_single_loss(
                    pdf,
                    "BCE loss",
                    (self.losses["train_bce_loss"], self.losses["val_bce_loss"]),
                    ("train", "val")
                )
                self.plot_single_loss(
                    pdf,
                    "KL loss",
                    (self.losses["train_kl_loss"], self.losses["val_kl_loss"]),
                    ("train", "val")
                )
            self.plot_single_loss(
                pdf,
                "learning rate",
                (self.losses["lr"], ),
                (None, ),
                "log"
            )


    def plot_single_loss(
        self,
        pdf: PdfPages,
        ylabel: str,
        curves: tuple[np.ndarray],
        labels: tuple[str],
        yscale: str = "linear"
    ):
        """
        Makes single loss plot.

        Args:
            pdf: Multipage PDF object
            ylabel: Y axis label
            curves: List of numpy arrays with the loss curves to be plotted
            labels: Labels of the loss curves
            yscale: Y axis scale, "linear" or "log"
        """
        fig, ax = plt.subplots(figsize=(4,3.5))
        for i, (curve, label) in enumerate(zip(curves, labels)):
            epochs = np.arange(1, len(curve)+1)
            ax.plot(epochs, curve, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        self.corner_text(ax, self.title, "right", "top")
        ax.set_yscale(yscale)
        if any(label is not None for label in labels):
            ax.legend(loc="center right", frameon=False)
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()


    def plot_roc(self, file: str):
        """
        Plots the ROC curve and computes the AUC. For a Bayesian network, one curve is plotted
        for each sample from the distribution over the trainable weights.

        Args:
            file: Output file name
        """
        scores = -np.concatenate((1/self.weights_true, self.weights_fake), axis=0)
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
        self.corner_text(
            ax,
            f"AUC = ${np.mean(auc):.3f} \\pm {np.std(auc):.3f}$"
                if self.bayesian else f"AUC = {auc:.3f}",
            "left",
            "top"
        )
        self.corner_text(ax, self.title, "right", "bottom")
        plt.savefig(file, bbox_inches="tight")
        plt.close()


    def plot_weight_hist(self, file: str):
        """
        Plots the weight histograms of the generated, truth and combined test data. Multiple
        histograms with different axis limits and scales are made.

        Args:
            file: Output file name
        """
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
        """
        Plots a single weight histogram.

        Args:
            pdf: Multipage PDF object
            bins: Numpy array with the bin boundaries
            xscale: X axis scale, "linear" or "log"
            yscale: Y axis scale, "linear" or "log"
        """
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

            y_true = np.median(true_hists, axis=1)
            y_true_err = np.stack((
                np.quantile(true_hists, 0.159, axis=1),
                np.quantile(true_hists, 0.841, axis=1)
            ), axis=0)
            y_fake = np.median(fake_hists, axis=1)
            y_fake_err = np.stack((
                np.quantile(fake_hists, 0.159, axis=1),
                np.quantile(fake_hists, 0.841, axis=1)
            ), axis=0)
            y_combined = np.median(combined_hists, axis=1)
            y_combined_err = np.stack((
                np.quantile(combined_hists, 0.159, axis=1),
                np.quantile(combined_hists, 0.841, axis=1)
            ), axis=0)

            #y_true = np.mean(true_hists, axis=1)
            #y_true_err = np.std(true_hists, axis=1)
            #y_fake = np.mean(fake_hists, axis=1)
            #y_fake_err = np.std(fake_hists, axis=1)
            #y_combined = np.mean(combined_hists, axis=1)
            #y_combined_err = np.std(combined_hists, axis=1)

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
            y_combined / np.sum(y_combined),
            y_combined_err / np.sum(y_combined) if y_combined_err is not None else None,
            label = "Comb",
            color = self.colors[0]
        )
        self.hist_line(
            ax,
            bins,
            y_true / np.sum(y_true),
            y_true_err / np.sum(y_true) if y_true_err is not None else None,
            label = "Truth",
            color = self.colors[1]
        )
        self.hist_line(
            ax,
            bins,
            y_fake / np.sum(y_fake),
            y_fake_err / np.sum(y_fake) if y_fake_err is not None else None,
            label = "Gen",
            color = self.colors[2]
        )
        self.corner_text(ax, self.title, "right", "top")
        ax.set_xlabel("weight")
        ax.set_ylabel("normalized")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(bins[0], bins[-1])
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()


    def plot_bgen_weights(self, file: str):
        """
        Plots 2d histogram of the error on the weights from a Bayesian generator network
        against the weights found by the discriminator.

        Args:
            file: Output file name
        """
        assert self.log_gen_weights is not None

        w_bgen = np.std(self.log_gen_weights, axis=1)
        if self.bayesian:
            w_bgen = np.repeat(w_bgen[:,None], self.weights_fake.shape[1], axis=1).flatten()
            w_disc = self.weights_fake.flatten()
        else:
            w_disc = self.weights_fake
        x_bins = np.linspace(0,4,30)
        y_bins = np.linspace(0,3,30)

        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.hist2d(
            w_bgen,
            w_disc,
            bins=(x_bins, y_bins),
            rasterized=True,
            norm = mpl.colors.LogNorm(),
            density=True,
            cmap="jet"
        )
        ax.set_xlabel(r"$\sigma(\log w_\text{gen})$")
        ax.set_ylabel(r"$w_\text{disc}$")

        plt.savefig(file, bbox_inches="tight")
        plt.close()


    def plot_observables(self, file: str):
        """
        Plots histograms of all the given observables. The truth, generated and reweighted
        distributions are shown. The ratio of the latter two to the truth is shown in a
        second panel. The third panel shows the marginalized discriminator weight.

        Args:
            file: Output file name
        """
        with PdfPages(file) as pdf:
            for observable in self.observables:
                self.plot_single_observable(pdf, observable)


    def plot_single_observable(self, pdf: PdfPages, observable: Observable):
        """
        Plots the histograms for a single observable

        Args:
            pdf: Multipage PDF object
            observable: Observable to be plotted
        """
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
            #rw_mean = np.mean(rw_hists, axis=1)
            #rw_std = np.std(rw_hists, axis=1)
            rw_mean = np.median(rw_hists, axis=1)
            rw_std = np.stack((
                np.quantile(rw_hists, 0.159, axis=1),
                np.quantile(rw_hists, 0.841, axis=1)
            ), axis=0)
        else:
            rw_mean = np.histogram(
                observable.fake_data[self.fake_mask],
                bins=bins,
                weights=self.weights_fake
            )[0]
            rw_std = None
        true_hist, _ = np.histogram(observable.true_data, bins=bins, density=True)
        fake_hist, _ = np.histogram(observable.fake_data, bins=bins, density=True)

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
        self.hist_plot(pdf, lines, bins, observable)


    def plot_clustering(
        self,
        file: str,
        lower_thresholds: list[float],
        upper_thresholds: list[float]
    ):
        """
        Plots the clustering histograms for all observables. A subset of samples is selected
        by imposing thresholds for the weights. Then, the histograms of these subsets are
        plotted.

        Args:
            file: Output file name
            lower_thresholds: List of lower thresholds for the weights
            upper_thresholds: List of upper thresholds for the weights
        """
        with PdfPages(file) as pdf:
            for observable in self.observables:
                self.plot_single_clustering(
                    pdf,
                    observable,
                    lower_thresholds,
                    upper_thresholds
                )


    def plot_single_clustering(
        self,
        pdf: PdfPages,
        observable: Observable,
        lower_thresholds: list[float],
        upper_thresholds: list[float]
    ):
        """
        Plots the clustering histograms for a single observable

        Args:
            pdf: Multipage PDF object
            observable: Observable to be plotted
            lower_thresholds: List of lower thresholds for the weights
            upper_thresholds: List of upper thresholds for the weights
        """
        bins = observable.bins
        if self.bayesian:
            weights_fake = np.median(self.weights_fake, axis=1)
        else:
            weights_fake = self.weights_fake

        masks, labels = [], []
        for threshold in lower_thresholds:
            masks.append(weights_fake < threshold)
            labels.append(f"$w < {threshold}$")
        for threshold in upper_thresholds:
            masks.append(weights_fake > threshold)
            labels.append(f"$w > {threshold}$")
        true_hist, _ = np.histogram(observable.true_data, bins=bins, density=True)
        hists = [
            np.histogram(
                observable.fake_data[self.fake_mask][mask],
                bins=bins,
                density=True
            )[0]
            for mask in masks
        ]
        lines = [
            Line(
                y = true_hist,
                label = "Truth",
                color = "k",
                fill = True
            ),
            *[
                Line(y=hist, label=label, color=color)
                for hist, label, color in zip(hists, labels, self.colors)
            ]
        ]
        self.hist_plot(pdf, lines, bins, observable, show_ratios=False, show_weights=False)


    def corner_text(
        self,
        ax: mpl.axes.Axes,
        text: str,
        horizontal_pos: str,
        vertical_pos: str
    ):
        ax.text(
            x = 0.95 if horizontal_pos == "right" else 0.05,
            y = 0.95 if vertical_pos == "top" else 0.05,
            s = text,
            horizontalalignment = horizontal_pos,
            verticalalignment = vertical_pos,
            transform = ax.transAxes
        )
        # Dummy line for automatic legend placement
        plt.plot(
            0.8 if horizontal_pos == "right" else 0.2,
            0.8 if vertical_pos == "top" else 0.2,
            transform=ax.transAxes,
            color="none"
        )


    def hist_plot(
        self,
        pdf: PdfPages,
        lines: list[Line],
        bins: np.ndarray,
        observable: Observable,
        show_ratios: bool = True,
        show_weights: bool = True
    ):
        """
        Makes a single histogram plot, used for the observable histograms and clustering
        histograms.

        Args:
            pdf: Multipage PDF object
            lines: List of line objects describing the histograms
            bins: Numpy array with the bin boundaries
            show_ratios: If True, show a panel with ratios
            show_weights: If True, show a panel with marginalized weights
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            n_panels = 1 + int(show_ratios) + int(show_weights)
            fig, axs = plt.subplots(
                n_panels, 1,
                sharex = True,
                figsize = (6, 4.5),
                gridspec_kw = {"height_ratios": (4, 1, 1)[:n_panels], "hspace": 0.00}
            )
            if n_panels == 1:
                axs = [axs]

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
                    color=line.color,
                    fill=line.fill
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
                        if len(line.y_err.shape) == 2:
                            ratio_err = (line.y_err * scale) / (y_ref * ref_scale)
                            ratio_err[:,ratio_isnan] = 0.
                        else:
                            ratio_err = np.sqrt((line.y_err / line.y)**2)
                            ratio_err[ratio_isnan] = 0.
                    else:
                        ratio_err = None
                    ratio[ratio_isnan] = 1.
                    self.hist_line(ax, bins, ratio, ratio_err, label=None, color=line.color)

            axs[0].legend(frameon=False)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale)
            self.corner_text(axs[0], self.title, "right", "top")

            if show_ratios:
                axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
                axs[1].set_yticks([0.8,1,1.2])
                axs[1].set_ylim([0.75,1.25])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

            if show_weights:
                ax_idx = 1+int(show_ratios)
                axs[ax_idx].set_ylabel(r"$w$")
                axs[ax_idx].axhline(y=1, c="black", ls="--", lw=0.7)
                ymin, ymax = axs[ax_idx].get_ylim()
                axs[ax_idx].set_ylim(ymin, min(ymax, 2))

            unit = "" if observable.unit is None else f" [{observable.unit}]"
            axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[-1].set_xscale(observable.xscale)
            axs[-1].set_xlim(bins[0], bins[-1])

            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()


    def hist_line(
        self,
        ax: mpl.axes.Axes,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str,
        fill: bool = False
    ):
        """
        Plot a stepped line for a histogram, optionally with error bars.

        Args:
            ax: Matplotlib Axes
            bins: Numpy array with bin boundaries
            y: Y values for the bins
            y_err: Y errors for the bins
            label: Label of the line
            color: Color of the line
            fill: Filled histogram
        """

        dup_last = lambda a: np.append(a, a[-1])

        if fill:
            ax.fill_between(
                bins,
                dup_last(y),
                label = label,
                facecolor = color,
                step = "post",
                alpha = 0.2
            )
        else:
            ax.step(
                bins,
                dup_last(y),
                label = label,
                color = color,
                linewidth = 1.0,
                where = "post",
            )
        if y_err is not None:
            if len(y_err.shape) == 2:
                y_low = y_err[0]
                y_high = y_err[1]
            else:
                y_low = y - y_err
                y_high = y + y_err

            ax.step(
                bins,
                dup_last(y_high),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            ax.step(
                bins,
                dup_last(y_low),
                color = color,
                alpha = 0.5,
                linewidth = 0.5,
                where = "post"
            )
            ax.fill_between(
                bins,
                dup_last(y_low),
                dup_last(y_high),
                facecolor = color,
                alpha = 0.3,
                step = "post"
            )
