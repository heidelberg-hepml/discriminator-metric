# This is based on Raghav's script
import os
from jetnet.datasets import JetNet
import numpy as np
import argparse
from scripts.conversion import *
import jetnet


#os.fchdir()

parser = argparse.ArgumentParser(description='create data for lorentzNet Training')
parser.add_argument('--train', type=bool, default=True, help='train or test data')
parser.add_argument('--test_jets', type=str, default='tailcut', help='test jet file')

args = parser.parse_args()

num_train = 100_000
num_test = 50_000

truth_jets_pf = JetNet.getData(
    "g",
    data_dir="data",
    split_fraction=[1, 0, 0],
    jet_features=["pt", "eta", "mass", "num_particles"],
)[0]

truth_jet_data = JetNet.getData(
    "g",
    data_dir="data",
    split_fraction=[1, 0, 0],
    jet_features=["pt", "eta", "mass", "num_particles"],
)[1]



mass = jetnet.utils.jet_features(truth_jets_pf[:, :, :3])["mass"]

cut = (mass < 0.17)

cut_jets = truth_jets_pf[cut]
cut_jet_data = truth_jet_data[cut]

print(len(truth_jets_pf[cut]))


test_jets_pf = cut_jets[:num_train] if args.train else cut_jets[num_train : num_train + num_test]
test_jet_data = cut_jet_data[:num_train] if args.train else cut_jet_data[num_train : num_train + num_test]




truth_jets_pf = JetNet.getData(
            "g",
            data_dir='data',
            jet_features=None,
            split_fraction=[0.7, 0.3, 0],
            split="train" if args.train else "valid",
        )[0][: num_train if args.train else num_test]

truth_jet_data = JetNet.getData(
            "g",
            data_dir='data',
            jet_features=["pt", "eta", "mass", "num_particles"],
            split_fraction=[0.7, 0.3, 0],
            split="train" if args.train else "valid",
        )[1][: num_train if args.train else num_test]



labels = np.concatenate(
            (np.ones(len(truth_jets_pf)), np.zeros(len(test_jets_pf))), axis=0
        ).astype(int)

data = np.concatenate((truth_jets_pf, test_jets_pf), axis=0)
jet_data = np.concatenate((truth_jet_data, test_jet_data), axis=0)

jet_data = pd.DataFrame(jet_data, columns=["pt", "eta", "mass", "num_particles"])

cartesian_data = etaphipt_rel_to_epxpypz(data, jet_data)

print(cartesian_data.shape)
