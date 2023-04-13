# This is based on Raghav's script
import os
from jetnet.datasets import JetNet
import numpy as np
import argparse
from scripts.conversion import *
import jetnet
import numpy as np

parser = argparse.ArgumentParser(description='create data for lorentzNet Training')
parser.add_argument('--test_jets', type=str, default='tailcut', help='test jet file')

args = parser.parse_args()

truth_jets_pf, truth_jet_data = JetNet.getData(
    "g",
    data_dir="data",
    split_fraction=[1, 0, 0],
    jet_features=["pt", "eta", "mass", "num_particles"]
    )

mass = jetnet.utils.jet_features(truth_jets_pf[:, :, :3])["mass"]

tailcut_weights = (mass < 0.17).astype(float)
tailcut_weights /= np.sum(tailcut_weights)

tailcut_original = np.load('data/raghav/tailcut.npy')
num_samples = 150_000

np.random.seed(4)

# This awkard looking loop is to match the datapoints in Raghav's file : tailcut.npy
for i in range(5):
  if i==0:
    inds = np.random.choice(np.arange(truth_jets_pf.shape[0]), num_samples)
  else:
    inds = np.random.choice(np.arange(truth_jets_pf.shape[0]), num_samples, 
                        p=tailcut_weights)
  distorted_jets = truth_jets_pf[inds]
  if np.sum(tailcut_original - distorted_jets[:,:,:3])==0:
    print('Matched')
    print(i)
    break

distorted_jet_data = truth_jet_data[inds]
# Assert that Raghav's data matches with the data generated here
assert (distorted_jets[:,:,:3] == tailcut_original).all()
distorted_jet_data = pd.DataFrame(distorted_jet_data, columns=['pt', 'eta', 'mass', 'num_particles'])

# Use Raghav's script to create training and validation data
cartesian_jets = etaphipt_rel_to_epxpypz(distorted_jets, distorted_jet_data)

# Save the data :

np.save(f'data/distorted_jets/{args.test_jets}.npy', distorted_jets)
np.save(f'data/distorted_jets/{args.test_jets}_cartesian.npy', cartesian_jets)
distorted_jet_data.to_hdf(f'data/distorted_jets/distorted_jet_data.h5', 
                          key=f'{args.test_jets}', mode='w')





