import yaml
import argparse
from importlib import import_module
import torch

from .documenter import Documenter
from .train import DiscriminatorTraining
from .plots import Plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.load:
        doc, params = Documenter.from_saved_run(args.paramfile)
    else:
        doc, params = Documenter.from_param_file(args.paramfile)

    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print("Loading data")
    loader = import_module(params["loader_module"], "src")
    datasets = loader.load(params["loader_params"])

    for data in datasets:
        print()
        print(f"Dataset {data.suffix}")
        print(f"  Train points: {len(data.train_true)} truth, {len(data.train_fake)} generated")
        print(f"  Test points: {len(data.test_true)} truth, {len(data.test_fake)} generated")
        print(f"  Val points: {len(data.val_true)} truth, {len(data.val_fake)} generated")

        print("  Building model")
        training = DiscriminatorTraining(params, device, data)

        if args.load:
            print("  Loading model")
            training.load(doc.get_file(f"model_{data.suffix}.pth"))

        if not args.load:
            print("  Running training")
            training.train()
            print("  Saving model")
            training.save(doc.get_file(f"model_{data.suffix}.pth"))

        print("  Calculating weights")
        weights_true, weights_fake = training.predict()

        print("  Creating plots")
        plots = Plots(data.observables, weights_true, weights_fake, data.label)
        print("    Plotting losses")
        plots.plot_losses(doc.add_file(f"losses_{data.suffix}.pdf"), training.losses)
        print("    Plotting ROC")
        plots.plot_roc(doc.add_file(f"roc_{data.suffix}.pdf"))
        print("    Plotting weights")
        plots.plot_weight_hist(doc.add_file(f"weights_{data.suffix}.pdf"))
        print("    Plotting observables")
        plots.plot_observables(doc.add_file(f"observables_{data.suffix}.pdf"))

if __name__ == "__main__":
    main()
