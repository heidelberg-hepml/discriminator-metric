# This is based on Raghav's script
import os
from jetnet.datasets import JetNet
import numpy as np
import argparse
from scripts.conversion import *
import jetnet
import numpy as np
from coffea.lookup_tools.dense_lookup import dense_lookup

from typing import OrderedDict

parser = argparse.ArgumentParser(description='create data for lorentzNet Training')

args = parser.parse_args()

truth_jets_pf, truth_jet_data = JetNet.getData(
    "g",
    data_dir="data",
    split_fraction=[1, 0, 0],
    jet_features=["pt", "eta", "mass", "num_particles"]
    )

mass = jetnet.utils.jet_features(truth_jets_pf[:, :, :3])["mass"]

np.random.seed(4)


# Distribution level distortions
bins = np.linspace(0, np.max(mass), 26)
true_mass_hist = np.histogram(mass, bins)[0]

smeared_hist = np.histogram(mass * np.random.normal(1, 0.25, size=mass.shape), bins)[0]
shifted_hist = np.histogram(mass * np.random.normal(1.1, 0.05, size=mass.shape), bins)[0]

smeared_lookup = dense_lookup(smeared_hist / true_mass_hist, bins)
shifted_lookup = dense_lookup(shifted_hist / true_mass_hist, bins)

smeared_weights = smeared_lookup(mass)
smeared_weights /= np.sum(smeared_weights)

shifted_weights = shifted_lookup(mass)
shifted_weights /= np.sum(shifted_weights)

tailcut_weights = (mass < 0.17).astype(float)
tailcut_weights /= np.sum(tailcut_weights)

dists = OrderedDict(
    [
        ("truth", (np.ones(truth_jets_pf.shape[0]) / truth_jets_pf.shape[0], "Truth")),
        ("smeared", (smeared_weights, "Smeared")),
        ("shifted", (shifted_weights, "Shifted")),
        ("tailcut", (tailcut_weights, "Removing tail")),
    ]
)

num_samples = 150_000

np.random.seed(4)

num_samples = 150_000
sample_masses = OrderedDict()

np.random.seed(4)
true_inds = np.random.choice(np.arange(truth_jets_pf.shape[0]), num_samples)

for key, (weights, _) in dists.items():
   #print(key)
    original = np.load(f'data/raghav/{key}.npy')
    inds = np.random.choice(np.arange(truth_jets_pf.shape[0]), num_samples, p=weights)
    #distorted_jet_data = truth_jet_data[inds]
    distorted_jets_pf = truth_jets_pf[inds]
    assert (distorted_jets_pf[:,:,:3] == original).all()

    if np.sum(original - distorted_jets_pf[:,:,:3])==0:
      print(f'{key} Matched')

    cartesian_jets = etaphipt_epxpypz(distorted_jets_pf)
    np.save(f'data/distorted_jets/{key}.npy', distorted_jets_pf)
    np.save(f'data/distorted_jets/{key}_cartesian.npy', cartesian_jets)




