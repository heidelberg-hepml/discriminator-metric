import warnings
from collections import namedtuple
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import lognorm, norm, binned_statistic
from scipy.optimize import curve_fit
from sklearn.calibration import calibration_curve

from .observable import Observable

from .calo_plotting_helper import *

Line = namedtuple(
    "Line",
    ["y", "y_err", "y_ref", "y_orig", "label", "color", "fill", "linestyle"],
    defaults = [None, None, None, None, None, False, None]
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
        labels_w_hist: list[str],
        add_comb: bool,
        log_gen_weights: Optional[np.ndarray] = None,
        showers_fake: Optional[np.ndarray] = None,
        showers_true: Optional[np.ndarray] = None,
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
            labels: Labels of weight histograms
            add_comb: add combined weights hist line
        """
        self.observables = observables
        self.bayesian = len(weights_true.shape) == 2
        self.weights_true, self.weights_fake = self.process_weights(weights_true, weights_fake)
        self.losses = losses
        self.title = title
        self.log_gen_weights = log_gen_weights
        self.labels_w_hist = labels_w_hist
        self.add_comb = add_comb
        self.eps = 1.0e-10
        self.showers_fake = showers_fake
        self.showers_true = showers_true

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]

    def process_weights(self, weights_true, weights_fake):
        w_comb = np.concatenate((weights_true, weights_fake), axis=0)
        self.p_low = np.percentile(w_comb[w_comb!=0], 0.005)
        self.p_high = np.percentile(w_comb[w_comb!=np.inf], 99.995)

        weights_true[weights_true >= self.p_high] = self.p_high
        weights_fake[weights_fake <= self.p_low] = self.p_low

        weights_true[weights_true <= self.p_low] = self.p_low
        weights_fake[weights_fake >= self.p_high] = self.p_high
        return weights_true, weights_fake

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
        ax.set_yscale(yscale)
        if any(label is not None for label in labels):
            ax.legend(loc="best", frameon=False, title=self.title)
        else:
            self.corner_text(ax, self.title, "right", "top")
        plt.savefig(pdf, format="pdf")
        plt.close()


    def plot_roc(self, file: str):
        """
        Plots the ROC curve and computes the AUC. For a Bayesian network, one curve is plotted
        for each sample from the distribution over the trainable weights.

        Args:
            file: Output file name
        """
        scores = -np.concatenate((self.weights_true, self.weights_fake), axis=0)
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
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
        if self.bayesian:
            for i in range(self.weights_fake.shape[1]):
                ax.plot(fpr[:,i], tpr[:,i], alpha=0.3, color=self.colors[0])
        else:
            ax.plot(fpr, tpr, color=self.colors[0])
        ax.plot([0,1], [0,1], color="k", ls="dashed")

        ax.set_xlabel(r"$\epsilon_B$")
        ax.set_ylabel(r"$\epsilon_S$")
        self.corner_text(
            ax,
            f"AUC = ${np.mean(auc):.3f} \\pm {np.std(auc):.3f}$"
                if self.bayesian else f"AUC = {auc:.3f}",
            "left",
            "top"
        )
        self.corner_text(ax, self.title, "right", "bottom")
        plt.savefig(file)
        plt.close()

    def plot_calibration_curve(self, file: str):
        """
        plot calibration curve
        """
        nlt, nlf = np.mean(self.weights_true < 1), np.mean(self.weights_fake < 1)
        print(f"      ${nlt*100:.1f}\\%$ & ${nlf*100:.1f}\\%$ & ${(1-nlt)*100:.1f}\\%$ & ${(1-nlf)*100:.1f}\\%$")
        with PdfPages(file) as pdf:
            scores = np.concatenate((self.weights_fake, self.weights_true[:len(self.weights_fake)]))
            labels = np.concatenate((
                        np.zeros_like(self.weights_fake),
                        np.ones_like(self.weights_true[:len(self.weights_fake)]))
                        )
            cls_output = scores/(1+scores)

            fig, ax = plt.subplots(figsize=(5, 4.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))

            prob_true, prob_pred = calibration_curve(labels, cls_output, n_bins=30)
            ax.plot(prob_true, prob_pred)
            ax.plot([0,1], [0,1], color="k", ls="dashed")
            self.corner_text(ax, self.title, "left", "top")
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("fraction of positives")
            plt.savefig(pdf, format="pdf")
            plt.close()
            
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            p_low, p_high = 1e-2, 1e+2
            wc = np.logspace(np.log10(p_low), np.log10(p_high), 30)
            #wc = np.linspace(p_low, p_high, 30)
            n_true, n_fake = np.zeros_like(wc), np.zeros_like(wc)
            for i, w in enumerate(wc):
                n_true[i] = np.mean(self.weights_true < w)
                n_fake[i] = np.mean(self.weights_fake < w)
            
            ax.plot(wc, n_true/n_fake, label=r"$w < w_c$")
            ax.plot(wc, (1-n_true)/(1-n_fake), label=r"$w > w_c$")
            ax.plot([p_low,p_high], [p_low,p_high], color="k", ls="dashed")
            self.corner_text(ax, self.title, "left", "top")
            ax.set_xlabel(r"$w_c$")
            ax.set_ylabel(r"$N_\text{truth} / N_\text{gen}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(p_low, p_high)
            ax.legend(frameon=False)

            plt.savefig(pdf, format="pdf")
            plt.close()
            
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            wc_middle = (wc[1:] + wc[:-1]) / 2
            true_bins, _ = np.histogram(self.weights_true, bins=wc)
            fake_bins, _ = np.histogram(self.weights_fake, bins=wc)
            true_bins_norm, fake_bins_norm = len(self.weights_true), len(self.weights_fake)
            true_bins_err, fake_bins_err = np.sqrt(true_bins), np.sqrt(fake_bins)
            ratio = (true_bins/true_bins_norm)/(fake_bins/fake_bins_norm)
            ratio_err = ratio * np.sqrt(
                (true_bins_err/true_bins)**2 + (fake_bins_err/fake_bins)**2
            )
            self.hist_line(ax, wc, ratio, ratio_err, label="", color=self.colors[0])
            #ax.plot(wc_middle, true_bins/fake_bins)
            #ax.plot(wc_middle, (1-n_true)/(1-n_fake), label=r"$w > w_c$")
            ax.plot([p_low,p_high], [p_low,p_high], color="k", ls="dashed")
            self.title = self.title
            self.corner_text(ax, self.title, "left", "top")
            self.corner_text(ax, 'Ep. 150', 'right', 'bottom')
            ax.set_xlabel(r"$w_c$")
            ax.set_ylabel(r"$N_\text{truth} / N_\text{gen}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(p_low, p_high)
            #ax.legend(frameon=False)

            plt.savefig(pdf, format="pdf")
            plt.close()

    def plot_weight_hist(self, file: str):
        """
        Plots the weight histograms of the generated, truth and combined test data. Multiple
        histograms with different axis limits and scales are made.

        Args:
            file: Output file name
        """
        with PdfPages(file) as pdf:
            wmin = min(
                np.min(self.weights_true),
                np.min(self.weights_fake)
            )
            wmax = max(np.max(self.weights_true), np.max(self.weights_fake))
            self.plot_single_weight_hist(
                pdf,
                bins=np.linspace(0, 3, 50),
                xscale="linear",
                yscale="linear",
                secax=False,
            )
            self.plot_single_weight_hist(
                pdf,
                bins=np.logspace(np.log10(self.p_low)-1e-5, np.log10(self.p_high)+1e-5, 50),
                xscale="symlog",
                yscale="log",
                secax=False,
            )
            self.plot_single_weight_hist(
                pdf,
                bins=np.logspace(-4, 4, 80),
                xscale="log",
                yscale="log",
                secax=False,
            )


    def plot_single_weight_hist(
        self,
        pdf: PdfPages,
        bins: np.ndarray,
        xscale: str,
        yscale: str,
        secax: bool
    ):
        """
        Plots a single weight histogram.

        Args:
            pdf: Multipage PDF object
            bins: Numpy array with the bin boundaries
            xscale: X axis scale, "linear" or "log"
            yscale: Y axis scale, "linear" or "log"
            secax: secondary axes for linear plot
        """
        weights_combined = np.concatenate((self.weights_true, self.weights_fake), axis=0)
        if self.bayesian:
            true_hists = np.stack([
                np.histogram(
                    self.weights_true[:,i], bins=bins
                )[0] for i in range(self.weights_true.shape[1])
            ], axis=1)
            fake_hists = np.stack([
                np.histogram(
                    self.weights_fake[:,i], bins=bins
                )[0] for i in range(self.weights_fake.shape[1])
            ], axis=1)
            combined_hists = np.stack([
                np.histogram(
                    weights_combined[:,i], bins=bins
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
        else:
            y_true = np.histogram(self.weights_true, bins=bins)[0]
            y_true_err = None
            y_fake = np.histogram(self.weights_fake, bins=bins)[0]
            y_fake_err = None
            y_combined = np.histogram(weights_combined, bins=bins)[0]
            y_combined_err = None

        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
        if self.add_comb:
            self.hist_line(
                ax,
                bins,
                y_combined / np.sum(y_combined),
                y_combined_err / np.sum(y_combined) if y_combined_err is not None else None,
                label = self.labels_w_hist[0],
                color = self.colors[0]
            )
        self.hist_line(
            ax,
            bins,
            y_true / np.sum(y_true),
            y_true_err / np.sum(y_true) if y_true_err is not None else None,
            label = self.labels_w_hist[1],
            color = self.colors[1]
        )
        self.hist_line(
            ax,
            bins,
            y_fake / np.sum(y_fake),
            y_fake_err / np.sum(y_fake) if y_fake_err is not None else None,
            label = self.labels_w_hist[2],
            color = self.colors[2]
        )
        #self.corner_text(ax, self.title, "right", "top")
        ax.set_xlabel("$w(x)$")
        ax.set_ylabel("a.u.")
        if xscale == 'symlog':
            ax.set_xscale(xscale, linthresh=self.p_low)
        else:
            ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(bins[0], bins[-1])

        ax.axvline(x=1.0, linewidth=1.0, color='k',linestyle='--')
        
        #adding Delta
        if secax:
            def wtoD(x):
                return x-1

            def Dtow(x):
                return x+1

            secax = ax.secondary_xaxis('top', functions=(wtoD, Dtow))
            secax.set_xlabel('$\Delta(x)$')
            secax.tick_params()
        
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        l = ax.legend(fancybox=True, framealpha=1.0, frameon=True, title=self.title, handlelength=0.0)
        frame = l.get_frame()
        frame.set_edgecolor('white')
        frame.set_facecolor('white')
        frame.set_alpha(1)
        for i, text in enumerate(l.get_texts()):
            text.set_color(self.colors[i+1])

        plt.savefig(pdf, format="pdf")
        plt.close()


    def plot_bgen_weights(self, file: str):
        """
        Plots 2d histogram of the error on the weights from a Bayesian generator network
        against the weights found by the discriminator.

        Args:
            file: Output file name
        """
        assert self.log_gen_weights is not None

        with PdfPages(file) as pdf:
            #mean_w_bgen = np.mean(np.exp(self.log_gen_weights), axis=1)
            #std_w_bgen = np.std(np.exp(self.log_gen_weights), axis=1)
            #w_bgen = np.exp(self.log_gen_weights)
            w_bgen = self.log_gen_weights
            mean_w_bgen = np.nanmedian(w_bgen, axis=1)

            std_w_bgen = (
                np.quantile(w_bgen, 0.841, axis=1) -
                np.quantile(w_bgen, 0.159, axis=1)
            ) / 2
            std_log_w_bgen = np.nanstd(self.log_gen_weights, axis=1)

            if self.bayesian:
                mean_w_bgen = np.repeat(
                    mean_w_bgen[:,None], self.weights_fake.shape[1], axis=1
                ).flatten()
                std_w_bgen = np.repeat(
                    std_w_bgen[:,None], self.weights_fake.shape[1], axis=1
                ).flatten()
                std_log_w_bgen = np.repeat(
                    std_log_w_bgen[:,None], self.weights_fake.shape[1], axis=1
                ).flatten()
                w_disc = self.weights_fake.flatten()
            else:
                w_disc = self.weights_fake
            ratio = std_w_bgen / mean_w_bgen
            wbinn_bins = np.linspace(0,0.01,30)
            wdisc_bins = np.linspace(0,3,30)
            print(w_disc.shape, ratio.shape, wdisc_bins.shape)
            mswbg, _, _ = binned_statistic(w_disc, ratio, lambda x: np.nanmedian(x), wdisc_bins)
            mswbg_stds = np.stack([
                binned_statistic(w_disc, ratio, lambda x: np.nanquantile(x, 0.159), wdisc_bins)[0],
                binned_statistic(w_disc, ratio, lambda x: np.nanquantile(x, 0.841), wdisc_bins)[0]
            ], axis=0)

            fig, ax = plt.subplots(figsize=(4,3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            _, _, _, img = ax.hist2d(
                ratio, #std_log_w_bgen,
                w_disc,
                bins=(wbinn_bins, wdisc_bins),
                rasterized=True,
                norm = mpl.colors.LogNorm(),
                density=True,
                cmap="jet"
            )
            #cb = plt.colorbar(img)
            #cb.set_label('norm.')
            ax.set_xlabel(r"$\sigma(\log p_\text{model}) / \mu(\log p_\text{model})$")
            ax.set_ylabel(r"$w$")

            plt.savefig(pdf, format="pdf")
            plt.close()

            fig, axs = plt.subplots(
                2, 1,
                sharex = True,
                figsize = (4, 5),
                gridspec_kw = {"height_ratios": (2, 1.5), "hspace": 0.00}
            )
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.083,0.07,1.00,0.98))
            _, _, _, img = axs[0].hist2d(
                w_disc,
                ratio, #std_log_w_bgen,
                bins=(wdisc_bins, wbinn_bins),
                rasterized=True,
                norm = mpl.colors.LogNorm(),
                density=True,
                cmap="jet"
            )
            self.hist_line(
                axs[1],
                wdisc_bins,
                mswbg,
                y_err = None, #mswbg_stds,
                label = "Shitbull",
                color = self.colors[0]
            )
            #cb = plt.colorbar(img)
            #cb.set_label('norm.')
            axs[0].set_ylabel(r"$\sigma(\log p_\text{model}) / \mu(\log p_\text{model})$")
            axs[1].set_ylabel(r"median $\sigma / \mu$")
            axs[1].set_xlabel(r"$w$")

            fig.tight_layout()
            fig.align_ylabels()
            self.corner_text(axs[1], self.title, "right", "top", h_offset=0.15)
            plt.savefig(pdf, format="pdf")
            plt.close()

            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            pull = mean_w_bgen * (w_disc - 1) / std_w_bgen
            low_lim = np.quantile(pull[np.isfinite(pull)], 0.005)
            high_lim = np.quantile(pull[np.isfinite(pull)], 0.995)
            lim = max(high_lim, -low_lim)
            bins = np.linspace(-lim, lim, 50)
            y, _ = np.histogram(pull, bins, density=True)
            self.hist_line(
                ax,
                bins,
                y,
                y_err = None,
                label = "Pull",
                color = self.colors[0]
            )
            x_fit, bins_fit, norm_param = self.fit_norm(bins, y, p0=(y.mean(), y.var()))
            ax.plot(
                bins_fit,
                x_fit,
                label="Fit",
                color=self.colors[1]
            )
            ax.set_xlabel(r"$\mu(p_\text{model}) (w - 1) / \sigma(p_\text{model})$")
            ax.set_ylabel("normalized")
            #ax.set_yscale("log")
            ax.set_xlim(bins[0], bins[-1])
            ax.legend(loc="upper right", frameon=False, title = self.title)
            self.corner_text(
                ax,
                f"$\\mu={norm_param[0]:.3f}$\n$\\sigma={norm_param[1]:.3f}$",
                "left",
                "top",
            )
            #ax.set_ylim(top=2.0e1, bottom=1.0e-4)
            plt.savefig(pdf, format="pdf")
            plt.close()

            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            log_pull = np.log(w_disc) / std_log_w_bgen
            low_lim = np.quantile(log_pull[np.isfinite(log_pull)], 0.005)
            high_lim = np.quantile(log_pull[np.isfinite(log_pull)], 0.995)
            lim = max(high_lim, -low_lim)
            bins = np.linspace(-lim, lim, 50)
            y, _ = np.histogram(log_pull, bins, density=True)
            self.hist_line(
                ax,
                bins,
                y,
                y_err = None,
                label = "Pull",
                color = self.colors[0]
            )
            x_fit, bins_fit, norm_param = self.fit_norm(bins, y, p0=(y.mean(), y.var()))
            ax.plot(
                bins_fit,
                x_fit,
                label="Fit",
                color=self.colors[1]
            )
            ax.set_xlabel(r"$\log w / \sigma(\log p_\text{model})$")
            ax.set_ylabel("normalized")
            #ax.set_yscale("log")
            ax.set_xlim(bins[0], bins[-1])
            #ax.set_ylim(bottom=1.0e-2, top=y.max()+5)
            ax.legend(loc="upper right", frameon=False)
            self.corner_text(
                ax,
                f"$\\mu={norm_param[0]:.3f}$\n$\\sigma={norm_param[1]:.3f}$",
                "left",
                "top",
            )
            plt.savefig(pdf, format="pdf")
            plt.close()

            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.11,0.09,1.00,1.00))
            self.hist_line(
                ax,
                wdisc_bins,
                mswbg,
                y_err = None,
                label = "",
                color = self.colors[1]
            )
            ax.set_xlabel(r"$w$")
            ax.set_ylabel(r"median $\sigma(p_\text{model}) / \mu(p_\text{model})$")
            self.corner_text(ax, self.title, "right", "top")
            plt.savefig(pdf, format="pdf")
            plt.close()

    def fit_norm(self, bins, y, p0, points=1000):
        to_fit = norm.pdf
        param, cov = curve_fit(to_fit, (bins[:-1]+bins[1:])/2, y, p0=p0)
        bins_fit = np.linspace(bins[0], bins[-1], points)
        pred = norm.pdf(bins_fit, loc=param[0], scale=param[1])
        return pred, bins_fit, param

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
                    observable.fake_data,
                    bins = bins,
                    weights = self.weights_fake[:,i],
                    density = True
                )[0] for i in range(self.weights_fake.shape[1])
            ], axis=1)
            rw_mean = np.mean(rw_hists, axis=1)
            rw_std = np.std(rw_hists, axis=1)
            #rw_mean = np.median(rw_hists, axis=1)
            #rw_std = np.stack((
            #    np.quantile(rw_hists, 0.159, axis=1),
            #    np.quantile(rw_hists, 0.841, axis=1)
            #), axis=0)
        else:
            rw_mean = np.histogram(
                observable.fake_data,
                bins=bins,
                weights=self.weights_fake
            )[0]
            rw_std = None
        true_hist, _ = np.histogram(observable.true_data, bins=bins, density=True)
        fake_hist, _ = np.histogram(observable.fake_data, bins=bins, density=True)

        if self.log_gen_weights is None:
            fake_std = None
        else:
            gen_weights = np.exp(self.log_gen_weights)
            mean_w_bgen = np.mean(gen_weights, axis=1, keepdims=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                norm_w_bgen = gen_weights / mean_w_bgen
            norm_w_bgen[~np.isfinite(norm_w_bgen)] = 1.
            hists = np.stack([
                np.histogram(observable.fake_data, bins=bins, weights=norm_w_bgen[:,i])[0]
                for i in range(norm_w_bgen.shape[1])
            ], axis=1)
            fake_median = np.median(hists, axis=1)
            fake_std = np.stack((
                np.quantile(hists, 0.159, axis=1) / fake_median * fake_hist,
                np.quantile(hists, 0.841, axis=1) / fake_median * fake_hist
            ), axis=0)

        lines = [
            Line(
                y = true_hist,
                label = "Geant",
                color = self.colors[0],
            ),
            Line(
                y = fake_hist,
                y_err = fake_std,
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
            #mean_w_bgen = np.mean(np.exp(self.log_gen_weights), axis=1)
            #std_w_bgen = np.std(np.exp(self.log_gen_weights), axis=1)
            #std_log_w_bgen = np.std(self.log_gen_weights, axis=1)
            #weights_fake = std_log_w_bgen

        masks_t, labels_t = [], []
        masks_f, labels_f = [], []
        for threshold in lower_thresholds:
            masks_f.append(weights_fake < threshold)
            labels_f.append(f"$w < {threshold}$")
        for threshold in upper_thresholds:
            masks_t.append(self.weights_true > threshold)
            labels_t.append(f"$w > {threshold}$")
            #labels_t.append(f"$w > {threshold}"+r", E_2 < 10^{-4}$")

        mks = self.showers_true[:, 432:504].sum(1) > 2.e-4
        mks2 = self.showers_fake[:, 432:504].sum(1) > 2.e-4
        mks3 = self.showers_true[:, 432:504].sum(1) > 1.e-6
        mks4 = self.showers_fake[:, 432:504].sum(1) > 1.e-6
        
        #mks_pi_f = n_brightest_voxel(self.showers_fake[:, 432:504], [1]).T.flatten() < 0.05
        #mks_pi_t = n_brightest_voxel(self.showers_true[:, 432:504], [1]).T.flatten() < 0.05
        #masks_t.append((masks_t[0]&mks3))
        #masks_t[0] &= ~mks3
        #labels_t.append(r"$w > 1.6, E_2 > 10^{-4}$")
        #print(observable.tex_label, observable.fake_data[masks_f[0] & mks_pi_f].shape, observable.fake_data[mks_pi_f].shape)
        #print(observable.tex_label, observable.true_data[masks_t[0] & mks_pi_t].shape, observable.true_data[mks_pi_t].shape)
        true_hist, _ = np.histogram(observable.true_data, bins=bins, density=True)
        fake_hist, _ = np.histogram(observable.fake_data, bins=bins, density=True)
        true_count, _ = np.histogram(observable.true_data, bins=bins, density=False)

        hists_f = [
            np.histogram(
                observable.fake_data[mask_f],
                bins=bins,
                density=True
            )[0]
            for mask_f in masks_f
        ]
        hists_t = [
                np.histogram(
                    observable.true_data[mask_t],
                    bins=bins,
                    density=True,
                    )[0]
                for mask_t in masks_t
                ]

        lines = [
            Line(
                y = fake_hist,
                y_ref = true_hist,
                y_err = true_hist/np.sqrt(true_count),
                label = "Gen.",
                color = "k",
            ),
            Line(
                y = true_hist,
                label = "Geant",
                color = "k",
                linestyle = '--',
                #fill = True
            ),
            *[
                Line(y=hist, label=label, color=self.colors[0])
                for hist, label, color in zip(hists_f, labels_f, self.colors)
            ],
            *[
                Line(y=hist, label=label, color=color)
                for hist, label, color in zip(hists_t, labels_t, self.colors[1:])
                ]
        ]
        self.hist_plot(pdf, lines, bins, observable, show_ratios=True, show_weights=False)


    def corner_text(
        self,
        ax: mpl.axes.Axes,
        text: str,
        horizontal_pos: str,
        vertical_pos: str,
        h_offset: float = 0.05,
        v_offset: float = 0.05
    ):
        ax.text(
            x = 1 - h_offset if horizontal_pos == "right" else h_offset,
            y = 1 - v_offset if vertical_pos == "top" else v_offset,
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
                figsize = (4, 4),
                gridspec_kw = {"height_ratios": (4, 1, 1)[:n_panels], "hspace": 0.00}
            )
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.08,0.08,0.99,0.98))
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
                    fill=line.fill,
                    linestyle=line.linestyle,
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

            axs[0].legend(frameon=False, title=self.title)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale)
            #ad hoc for E1
            if axs[0].get_ylim()[0]<1.e-4:
                axs[0].set_ylim(bottom=1.0e-4)
            #self.corner_text(axs[0], self.title, "right", "top")

            if show_ratios:
                axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Geant}}$")
                axs[1].set_yticks([0.8,1,1.2])
                axs[1].set_ylim([0.71,1.29])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

            if show_weights:
                ax_idx = 1+int(show_ratios)
                axs[ax_idx].set_ylabel(r"$w$")
                axs[ax_idx].axhline(y=1, c="black", ls="--", lw=0.7)
                ymin, ymax = axs[ax_idx].get_ylim()
                ymax = min(ymax, 2.3)
                yrange = ymax - ymin
                axs[ax_idx].set_ylim(ymin - 0.05*yrange, ymax + 0.05*yrange)

            unit = "" if observable.unit is None else f" [{observable.unit}]"
            axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[-1].set_xscale(observable.xscale)
            axs[-1].set_xlim(bins[0], bins[-1])

            plt.savefig(pdf, format="pdf")
            plt.close()


    def hist_line(
        self,
        ax: mpl.axes.Axes,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str,
        fill: bool = False,
        linestyle: str = None,
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
            linestyle
        """

        dup_last = lambda a: np.append(a, a[-1])

        alpha_factor = 0.4 if color == 'k' else 1.0

        if fill:
            ax.fill_between(
                bins,
                dup_last(y),
                label = label,
                facecolor = color,
                step = "post",
                alpha = 0.2,
                linestyle = linestyle
            )
        else:
            ax.step(
                bins,
                dup_last(y),
                label = label,
                color = color,
                linewidth = 1.0,
                where = "post",
                alpha = 1.0*alpha_factor,
                linestyle = linestyle
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
                alpha = 0.5*alpha_factor,
                linewidth = 0.5,
                where = "post"
            )
            ax.step(
                bins,
                dup_last(y_low),
                color = color,
                alpha = 0.5*alpha_factor,
                linewidth = 0.5,
                where = "post"
            )
            ax.fill_between(
                bins,
                dup_last(y_low),
                dup_last(y_high),
                facecolor = color,
                alpha = 0.3*alpha_factor,
                step = "post"
            )
    
    def plot_avg_showers(self, file: str, lower_thresholds: list[float],
            upper_thresholds: list[float]):

        masks, labels = [], []
        for threshold in lower_thresholds:
            masks.append(self.weights_fake < threshold)
            labels.append(f"$w < {threshold}$")
        for threshold in upper_thresholds:
            masks.append(self.weights_fake > threshold)
            labels.append(f"$w > {threshold}$")
        
        bin1 = np.arange(0, 96, 10)
        bin2 = np.arange(0, 3, 1)
        bin3 = np.arange(0, 12, 1)
        bin4 = np.arange(0, 6, 1)
        
        mks = self.showers_true[:, 432:504].sum(1) > 2.e-3
        mks2 = self.showers_fake[:, 432:504].sum(1) > 2.e-3
 
        with PdfPages(file) as pdf:
            for i, _ in enumerate(masks):
                fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            
                lay_0 = self.showers_fake[masks[i] & ~mks2][:, :288].reshape(-1, 3, 96).mean(0)
                lay_1 = self.showers_fake[masks[i] & ~mks2][:, 288:432].reshape(-1, 12, 12).mean(0)
                lay_2 = self.showers_fake[masks[i] & ~mks2][:, 432:504].reshape(-1, 12, 6).mean(0)

                img0 = ax[0].imshow(lay_0, norm=LogNorm(1.e-7, 2.e-2), aspect='auto', cmap='viridis', interpolation='none')
                img1 = ax[1].imshow(lay_1, norm=LogNorm(1.e-6, 1.e-1), aspect='auto', cmap='viridis', interpolation='none')
                img2 = ax[2].imshow(lay_2, norm=LogNorm(1.e-7, 5.e-3), aspect='auto', cmap='viridis', interpolation='none')

                fig.colorbar(img0, ax=ax[0])
                fig.colorbar(img1, ax=ax[1])
                fig.colorbar(img2, ax=ax[2])
                ax[0].set_xticks(bin1)
                ax[0].set_yticks(bin2)
                ax[1].set_xticks(bin3)
                ax[1].set_yticks(bin3)
                ax[2].set_xticks(bin4)
                ax[2].set_yticks(bin3)

                plt.suptitle('Average energy deposition, '+self.title+' '+labels[i])
                plt.savefig(pdf, format='pdf')
                plt.close()

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            
            lay_0 = self.showers_fake[~mks2][:, :288].reshape(-1, 3, 96).mean(0)
            lay_1 = self.showers_fake[~mks2][:, 288:432].reshape(-1, 12, 12).mean(0)
            lay_2 = self.showers_fake[~mks2][:, 432:504].reshape(-1, 12, 6).mean(0)

            img0 = ax[0].imshow(lay_0, norm=LogNorm(1.e-7, 2.e-2), aspect='auto', interpolation='none')
            img1 = ax[1].imshow(lay_1, norm=LogNorm(1.e-6, 1.e-1), aspect='auto', interpolation='none')
            img2 = ax[2].imshow(lay_2, norm=LogNorm(1.e-7, 5.e-3), aspect='auto', interpolation='none')
            
            fig.colorbar(img0, ax=ax[0])
            fig.colorbar(img1, ax=ax[1])
            fig.colorbar(img2, ax=ax[2])
            ax[0].set_xticks(bin1)
            ax[0].set_yticks(bin2)
            ax[1].set_xticks(bin3)
            ax[1].set_yticks(bin3)
            ax[2].set_xticks(bin4)
            ax[2].set_yticks(bin3)
            
            plt.suptitle('Average energy deposition, '+self.title+' Model')
            plt.savefig(pdf, format='pdf')
            plt.close()

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            
            lay_0 = self.showers_true[~mks][:, :288].reshape(-1, 3, 96).mean(0)
            lay_1 = self.showers_true[~mks][:, 288:432].reshape(-1, 12, 12).mean(0)
            lay_2 = self.showers_true[~mks][:, 432:504].reshape(-1, 12, 6).mean(0)

            img0 = ax[0].imshow(lay_0, norm=LogNorm(1.e-7, 2.e-2), aspect='auto', interpolation='none')
            img1 = ax[1].imshow(lay_1, norm=LogNorm(1.e-6, 1.e-1), aspect='auto', interpolation='none')
            img2 = ax[2].imshow(lay_2, norm=LogNorm(1.e-7, 5.e-3), aspect='auto', interpolation='none')

            fig.colorbar(img0, ax=ax[0])
            fig.colorbar(img1, ax=ax[1])
            fig.colorbar(img2, ax=ax[2])
            
            ax[0].set_xticks(bin1)
            ax[0].set_yticks(bin2)
            ax[1].set_xticks(bin3)
            ax[1].set_yticks(bin3)
            ax[2].set_xticks(bin4)
            ax[2].set_yticks(bin3)
            
            plt.suptitle('Average energy deposition, '+self.title+' Geant')
            plt.savefig(pdf, format='pdf')
            plt.close()
