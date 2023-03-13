import yaml
import argparse
from importlib import import_module

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
    loader = import_module(params["loader_module"])
    datasets = loader.load(params["loader_params"])

    for data in datasets:
        print()
        print(f"Dataset {data.suffix}")

        print("  Building model")
        training = DiscriminatorTraining(params, device, data)

        if args.load:
            print("  Loading model")
            training.load()

        if not args.load:
            print("  Running training")
            training.train()
            print("  Saving model")
            training.save()

        print("  Calculating weights")
        weights_true, weights_fake = training.predict()

        print("  Creating plots")
        plots = Plots(data.observables, weights_true, weights_fake, data.label)
        print("    Plotting losses")
        plots.plot_losses(doc.add_file(f"losses_{suffix}.pdf"), training.losses)
        print("    Plotting ROC")
        plots.plot_roc(doc.add_file(f"roc_{suffix}.pdf"))
        print("    Plotting weights")
        plots.plot_weight_hist(doc.add_file(f"weights_{suffix}.pdf"))
        print("    Plotting observables")
        plots.plot_observables(doc.add_file(f"observables_{suffix}.pdf"))

if __name__ == "__main__":
    main()
