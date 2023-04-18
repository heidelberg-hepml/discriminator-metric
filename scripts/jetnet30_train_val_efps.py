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

    print('split: ', split)
   
    efp_true = np.load('data/raghav/efps_true.npy')[:num_train] if split=='train' else np.load('data/raghav/efps_true.npy')[num_train : num_train + num_test]



    efp_distort = np.load(f"data/distorted_efps/efp_{args.test_jets}.npy")
    efp_distort = efp_distort[:num_train] if split=='train' else efp_distort[num_train : num_train + num_test]

    _X = np.concatenate((efp_true, efp_distort), axis=0)
    _Y = np.concatenate((np.ones(len(efp_true)), np.zeros(len(efp_distort))), axis=0
    ).astype(int)

    _X_df = pd.DataFrame(_X)
    _y_df = pd.DataFrame(_Y, columns=['labels'])
    print(_X_df.shape, _y_df.shape)
    #print(_jet.shape, _X_df.shape, _y_df.shape)

    _X_df.to_hdf('data/jetnet30_efp_data.h5', key=f'efp_{args.test_jets}_{split}', mode='a',format='table')  
    _y_df.to_hdf('data/jetnet30_efp_data.h5', key=f'labels_{args.test_jets}_{split}', mode='a',format = 'table')
   # _jet.to_hdf('data/jetnet30_data.h5', key=f'jet_data_{split}', mode='a',format = 'table')

