import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.special import softmax

from scripts.particlenet_models import ParticleNet

import matplotlib.pyplot as plt

from tqdm import tqdm

import os, sys
from os import listdir, mkdir, remove
from os.path import exists, dirname, realpath

from sklearn.metrics import confusion_matrix, roc_curve, auc

from jetnet.datasets import JetNet

plt.switch_backend("agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device(0)
torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"
    info_format = "%(asctime)s %(message)s"
    debug_format = "%(asctime)s [%(filename)s:%(lineno)d in %(funcName)s] %(message)s"

    def __init__(self, args):
        if args.log_file == "stdout":
            self.FORMATS = {
                logging.DEBUG: self.blue + self.debug_format + self.reset,
                logging.INFO: self.grey + self.info_format + self.reset,
                logging.WARNING: self.yellow + self.debug_format + self.reset,
                logging.ERROR: self.red + self.debug_format + self.reset,
                logging.CRITICAL: self.bold_red + self.debug_format + self.reset,
            }
        else:
            self.FORMATS = {
                logging.DEBUG: self.debug_format,
                logging.INFO: self.info_format,
                logging.WARNING: self.debug_format,
                logging.ERROR: self.debug_format,
                logging.CRITICAL: self.debug_format,
            }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%d/%m %H:%M:%S")
        return formatter.format(record)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def init_logging(args):
    """logging outputs to a file at ``args.log_file``;
    if ``args.log_file`` is stdout then it outputs to stdout"""
    if args.log_file == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        if args.log_file == "":
            args.log_file = args.outs_path + args.name + "_log.txt"
        handler = logging.FileHandler(args.log_file)

    level = getattr(logging, args.log.upper())

    handler.setLevel(level)
    handler.setFormatter(CustomFormatter(args))

    logging.basicConfig(handlers=[handler], level=level, force=True)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    return args


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help='log file name - default is name of file in outs/ ; "stdout" prints to console',
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser.add_argument(
        "--test-jets",
        type=str,
        help="dataset testing on",
    )

    parser.add_argument(
        "--dir-path",
        type=str,
        default="./",
        help="path where dataset and output will be stored",
    )

    parser.add_argument(
        "--jetnet-dir",
        type=str,
        default="/Users/raghav/Documents/CERN/gen-models/MPGAN/datasets/",
        help="path where dataset and output will be stored",
    )

    add_bool_arg(parser, "n", "run on nautilus cluster", default=False)

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="which epoch to start training on (only makes sense if loading a model)",
    )

    parser.add_argument("--num_hits", type=int, default=30, help="num nodes in graph")
    add_bool_arg(parser, "mask", "use masking", default=False)

    parser.add_argument("--num-epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=384, help="batch size")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="pick optimizer",
        choices=["adam", "rmsprop", "adamw"],
    )
    parser.add_argument("--lr", type=float, default=3e-4)

    add_bool_arg(parser, "scheduler", "use one cycle LR scheduler", default=False)
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--cycle-up-num-epochs", type=int, default=8)
    parser.add_argument("--cycle-cooldown-num-epochs", type=int, default=4)
    parser.add_argument("--cycle-max-lr", type=float, default=3e-3)
    parser.add_argument("--cycle-final-lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="name or tag for model; will be appended with other info",
    )
    args = parser.parse_args()

    if args.n:
        args.dir_path = "/graphganvol/hep-generative-metrics/"
        args.jetnet_dir = "/graphganvol/MPGAN/datasets"

    args.node_feat_size = 4 if args.mask else 3

    return args


class JetsClassifierDataset(Dataset):
    def __init__(self, jetnet_dir: str, dir_path: str, test_jets: str, train: bool = True):
        """True jets assigned label 1, test jets assigned label 0"""
        num_train = 100_000
        num_test = 50_000
        truth_jets_pf = JetNet.getData(
            "g",
            data_dir=jetnet_dir,
            particle_features=["etarel", "phirel", "ptrel"],
            jet_features=None,
            split_fraction=[0.7, 0.3, 0],
            split="train" if train else "valid",
        )[0][: num_train if train else num_test]

        test_jets_pf = np.load(f"{dir_path}/distorted_jets/{test_jets}.npy").astype(np.float32)
        test_jets_pf = (
            test_jets_pf[:num_train] if train else test_jets_pf[num_train : num_train + num_test]
        )

        self.X = JetNet.fpnd_norm(np.concatenate((truth_jets_pf, test_jets_pf), axis=0))
        self.Y = np.concatenate(
            (np.ones(len(truth_jets_pf)), np.zeros(len(test_jets_pf))), axis=0
        ).astype(int)

        logging.info("X shape: " + str(self.X.shape))
        logging.info("Y shape: " + str(self.Y.shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def init(args):
    out_dir = args.dir_path + "/classifier_trainings/"
    if not exists(out_dir):
        mkdir(out_dir)

    prev_models = [f[:-4] for f in listdir(out_dir)]  # removing .txt

    if args.name in prev_models:
        if args.name != "test" and not args.load_model and not args.override_load_check:
            raise RuntimeError(
                "A model directory of this name already exists, either change the name or use the --override-load-check flag"
            )

    os.system(f"mkdir -p {out_dir}/{args.name}")

    args_dict = vars(args)

    dirs = ["models", "losses"]

    for d in dirs:
        args_dict[d + "_path"] = f"{out_dir}/{args.name}/{d}/"
        os.system(f'mkdir -p {args_dict[d + "_path"]}')

    args_dict["args_path"] = f"{out_dir}/{args.name}/"
    args_dict["outs_path"] = f"{out_dir}/{args.name}/"

    args = objectview(args_dict)

    init_logging(args)

    if not args.load_model:
        # save args for posterity
        f = open(args.args_path + args.name + "_args.txt", "w+")
        f.write(str(vars(args)))
        f.close()
    elif not args.override_args:
        # load arguments from previous training
        temp = args.start_epoch, args.num_epochs  # don't load these

        f = open(args.args_path + args.name + "_args.txt", "r")
        args_dict = vars(args)
        load_args_dict = eval(f.read())
        for key in load_args_dict:
            args_dict[key] = load_args_dict[key]
        args = objectview(args_dict)
        f.close()

        args.load_model = True
        args.start_epoch, args.num_epochs = temp

    args.device = device
    return args


def main(args):
    args = init(args)

    train_dataset = JetsClassifierDataset(
        args.jetnet_dir, args.dir_path, args.test_jets, train=True
    )
    test_dataset = JetsClassifierDataset(
        args.jetnet_dir, args.dir_path, args.test_jets, train=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    C = ParticleNet(args.num_hits, args.node_feat_size, num_classes=2).to(args.device)

    if args.load_model:
        C = torch.load(args.model_path + args.name + "/C_" + str(args.start_epoch) + ".pt").to(
            device
        )

    if args.optimizer == "adamw":
        C_optimizer = torch.optim.AdamW(C.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        C_optimizer = torch.optim.Adam(C.parameters(), lr=args.lr)
    elif args.optimizer == "rmsprop":
        C_optimizer = torch.optim.RMSprop(C.parameters(), lr=args.lr)

    if args.scheduler:
        steps_per_epoch = len(train_loader)
        cycle_last_epoch = -1 if not args.load_model else (args.start_epoch * steps_per_epoch) - 1
        cycle_total_epochs = (2 * args.cycle_up_num_epochs) + args.cycle_cooldown_num_epochs

        C_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            C_optimizer,
            max_lr=args.cycle_max_lr,
            pct_start=(args.cycle_up_num_epochs / cycle_total_epochs),
            epochs=cycle_total_epochs,
            steps_per_epoch=steps_per_epoch,
            final_div_factor=args.cycle_final_lr / args.lr,
            anneal_strategy="linear",
            last_epoch=cycle_last_epoch,
        )

    loss = torch.nn.CrossEntropyLoss().to(args.device)

    train_losses = []
    test_losses = []
    fprs = []
    tprs = []
    aucs = []

    def plot_losses(epoch, train_losses, test_losses):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(train_losses)
        ax1.set_title("training")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(test_losses)
        ax2.set_title("testing")

        plt.savefig(args.losses_path + "/" + str(epoch) + ".pdf")
        plt.close()

        try:
            remove(args.losses_path + "/" + str(epoch - 1) + ".pdf")
        except:
            logging.info("couldn't remove loss file")

    def save_model(epoch):
        torch.save(C.state_dict(), args.models_path + "/C_" + str(epoch) + ".pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()

        output = C(data)

        # nll_loss takes class labels as target, so one-hot encoding is not needed
        C_loss = loss(output, y)

        C_loss.backward()
        C_optimizer.step()

        return C_loss.item()

    def test(epoch):
        C.eval()
        test_loss = 0
        correct = 0
        y_outs = []
        logging.info("testing")
        with torch.no_grad():
            for batch_ndx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
                logging.debug(f"x[0]: {x[0]}, y: {y}")
                output = C(x.to(device))
                y = y.to(device)
                test_loss += loss(output, y).item()
                pred = output.max(1, keepdim=True)[1]
                logging.debug(f"pred: {pred}, output: {output}")

                y_outs.append(output.cpu().numpy())
                correct += pred.eq(y.view_as(pred)).sum()

                # break

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        y_outs = np.concatenate(y_outs)
        logging.debug(f"y_outs {y_outs}")
        logging.debug(f"y_true {test_dataset[:][1]}")

        fpr, tpr, _ = roc_curve(test_dataset[:][1], softmax(y_outs, axis=1)[:, 1])
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

        with open(f"{args.losses_path}/fprs.txt", "w") as f:
            f.write(str(fprs))

        with open(f"{args.losses_path}/tprs.txt", "w") as f:
            f.write(str(tprs))

        with open(f"{args.losses_path}/aucs.txt", "w") as f:
            f.write(str(aucs))

        f = open(args.outs_path + args.name + ".txt", "a")
        s = f"After {epoch} epochs, on test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
        s += f"AUC: {roc_auc:.3f}\n"
        logging.info(s)
        f.write(s)
        f.close()

    for i in range(args.start_epoch, args.num_epochs):
        logging.info("Epoch %d %s" % ((i + 1), args.name))
        C_loss = 0
        test(i)
        logging.info("training")
        for batch_ndx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            C_loss += train_C(x.to(device), y.to(device))
            if args.scheduler:
                C_scheduler.step()

            # break

        train_losses.append(C_loss / len(train_loader))

        if (i + 1) % 1 == 0:
            save_model(i + 1)
            plot_losses(i + 1, train_losses, test_losses)

    test(args.num_epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)