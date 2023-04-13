import numpy as np
import pandas as pd
from jetnet.datasets import JetNet
from jetnet.utils import *
import h5py
import argparse
from scripts.conversion import *

parser = argparse.ArgumentParser()
parser.add_argument("--test_jets", type=str, default="tailcut")

args = parser.parse_args()

splits = ['train', 'valid']


num_train = 100_000
num_test = 50_000

for split in splits:
    truth_jets_pf, truth_jets_data = JetNet.getData(
        "g",
        data_dir='data',
        jet_features=["pt", "eta", "mass", "num_particles"],
        split_fraction=[0.7, 0.3, 0],
        split="train" if split=='train' else "valid",
    )[: num_train if split=='train' else num_test]


    truth_jets_data = pd.DataFrame(truth_jets_data, columns=["pt", "eta", "mass", "num_particles"])
    truth_jets_cart = etaphipt_rel_to_epxpypz(truth_jets_pf, truth_jets_data)


    test_jets_pf = np.load(f"data/distorted_jets/{args.test_jets}_cartesian.npy").astype(np.float32)
    test_jets_pf = test_jets_pf[:num_train] if split=='train' else test_jets_pf[num_train : num_train + num_test]
    

    test_jets_data = pd.read_hdf(f'data/distorted_jets/distorted_jet_data.h5', 
                            key=f'{args.test_jets}', mode='r')



    _X = np.concatenate((truth_jets_pf, test_jets_pf), axis=0)
    _Y = np.concatenate((np.ones(len(truth_jets_pf)), np.zeros(len(test_jets_pf))), axis=0
    ).astype(int)

    _X_df = pd.DataFrame(_X.reshape(-1,30*4))
    _y_df = pd.DataFrame(_Y, columns=['labels'])


    print(_X_df.shape, _y_df.shape)

    _X_df.to_hdf('data/jetnet30_data.h5', key=f'particle_data_{split}', mode='a',format='table')  
    _y_df.to_hdf('data/jetnet30_data.h5', key=f'labels_{split}', mode='a',format = 'table')

