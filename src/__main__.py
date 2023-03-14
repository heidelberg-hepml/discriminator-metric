import yaml
import argparse
from importlib import import_module
import torch
import pickle
import numpy as np

from .documenter import Documenter
from .train import DiscriminatorTraining
from .plots import Plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    args = parser.parse_args()

    if args.load_model:
        doc, params = Documenter.from_saved_run(args.paramfile)
    else:
        doc, params = Documenter.from_param_file(args.paramfile)

    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print("Loading data")
    loader = import_module(f"src.loaders.{params['loader_module']}")
    datasets = loader.load(params["loader_params"])

    for data in datasets:
        print()
        print(f"Dataset {data.suffix}")
        print(f"  Train points: {len(data.train_true)} truth, {len(data.train_fake)} generated")
        print(f"  Test points: {len(data.test_true)} truth, {len(data.test_fake)} generated")
        print(f"  Val points: {len(data.val_true)} truth, {len(data.val_fake)} generated")

        print("  Building model")
        training = DiscriminatorTraining(params, device, data)

        if args.load_model:
            print("  Loading model")
            training.load(doc.get_file(f"model_{data.suffix}.pth"))

        if not args.load_model:
            print("  Running training")
            training.train()
            print("  Saving model")
            training.save(doc.get_file(f"model_{data.suffix}.pth"))

        if args.load_model and args.load_weights:
            print("  Loading weights")
            with open(doc.get_file(f"weights_{data.suffix}.pkl"), "rb") as f:
                saved_weights = pickle.load(f)
            weights_true = saved_weights["true"]
            weights_fake = saved_weights["fake"]
            clf_score = saved_weights["classifier_score"]
        else:
            print("  Calculating weights")
            weights_true, weights_fake, clf_score = training.predict()
            print("  Saving weights")
            with open(doc.get_file(f"weights_{data.suffix}.pkl"), "wb") as f:
                pickle.dump({
                    "true": weights_true,
                    "fake": weights_fake,
                    "classifier_score": clf_score,
                }, f)
        if training.bayesian:
            print(f"  Classifier score: {np.mean(clf_score):.7f} +- {np.std(clf_score):.7f}")
        else:
            print(f"  Classifier score: {clf_score:.7f}")

        print("  Creating plots")
        plots = Plots(data.observables, weights_true, weights_fake, data.label)
        print("    Plotting losses")
        plots.plot_losses(doc.add_file(f"losses_{data.suffix}.pdf"), training.losses)
        print("    Plotting ROC")
        plots.plot_roc(doc.add_file(f"roc_{data.suffix}.pdf"))
        print("    Plotting weights")
        plots.plot_weight_hist(doc.add_file(f"weights_{data.suffix}.pdf"))
        if plots.bayesian:
            print("    Plotting pulls")
            plots.plot_weight_pulls(doc.add_file(f"pulls_{data.suffix}.pdf"))
        print("    Plotting observables")
        plots.plot_observables(doc.add_file(f"observables_{data.suffix}.pdf"))
        lower_thresholds = params.get("lower_cluster_thresholds", [])
        upper_thresholds = params.get("upper_cluster_thresholds", [])
        if len(lower_thresholds) + len(upper_thresholds) > 0:
            print("    Plotting clustering")
            plots.plot_clustering(
                doc.add_file(f"clustering_{data.suffix}.pdf"),
                lower_thresholds,
                upper_thresholds
            )


if __name__ == "__main__":
    main()
