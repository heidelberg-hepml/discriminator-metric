import yaml
import argparse
from importlib import import_module
import torch
import pickle
import numpy as np
import os

from .documenter import Documenter
from .train import DiscriminatorTraining
from .plots import Plots

def main():
    """
    Main function of the discriminator metric program. The path of a parameter file is
    expected as a command line argument, or, if the --load_model flag is present, the
    output folder name of an already trained model. If --load_weights is set, the
    discriminator weights for the test data are not computed again and the saved weights
    are used instead.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_name", type=str, default="best")
    parser.add_argument("--load_weights", action="store_true")
    args = parser.parse_args()

    if args.load_model:
        doc, params = Documenter.from_saved_run(args.paramfile)
    else:
        doc, params = Documenter.from_param_file(args.paramfile)

    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dtype = params.get('dtype', 'float32')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)
    print("Using dtype {}".format(dtype))

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
        model_dir = doc.get_file(f"model_{data.suffix}")
        os.makedirs(model_dir, exist_ok=True)
        training = DiscriminatorTraining(params, device, data, model_dir)

        if not args.load_model:
            print("  Running training")
            training.train()

        print(f"  Loading model {args.model_name}")
        training.load(args.model_name)

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
        lab_def = ["Comb", "Truth", "Gen"]
        labels = params.get('w_labels', lab_def)
        add_comb = params.get('add_w_comb', True)
        plots = Plots(
            data.observables,
            weights_true,
            weights_fake,
            training.losses,
            data.label,
            labels,
            add_comb,
            data.test_logw,
        )
        print("    Plotting losses")
        plots.plot_losses(doc.add_file(f"losses_{data.suffix}.pdf"))
        print("    Plotting ROC")
        plots.plot_roc(doc.add_file(f"roc_{data.suffix}.pdf"))
        print("    Plotting weights")
        plots.plot_weight_hist(doc.add_file(f"weights_{data.suffix}.pdf"))
        print("    Plotting calibration")
        plots.plot_calibration_curve(doc.add_file(f"calibration_{data.suffix}.pdf"))
        if data.test_logw is not None:
            print("    Plotting generator errors")
            plots.plot_bgen_weights(doc.add_file(f"gen_errors_{data.suffix}.pdf"))
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
            #plots.plot_clustering_diff(doc.add_file(f"clustering_diff_{data.suffix}.pdf"))


if __name__ == "__main__":
    main()
