import jetnet
from jetnet.datasets import JetNet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from IPython.display import Markdown, display
import pickle
from typing import OrderedDict
from scripts.conversion import *

plt.rcParams.update({"font.size": 16})

truth_jets_pf, _ = JetNet.getData(
    "g",
    data_dir="data/",
    split_fraction=[1.0, 0, 0],
    jet_features=["pt", "eta", "mass", "num_particles"],
)

np.random.seed(4)

pf_dists = OrderedDict()

pf_dists["all_smeared"] = (
    np.concatenate(
    (    truth_jets_pf[:,:,:3] * np.random.normal(1, 0.25, size=truth_jets_pf[:,:,:3].shape),
        truth_jets_pf[:,:,3:],
    ),
    axis=-1,
    ),
    r"Particle Features Smeared",
)
pf_dists["eta_smeared"] = (
    np.concatenate(
        (
            truth_jets_pf[..., 0:1] * np.random.normal(1, 0.25, size=truth_jets_pf[..., 0:1].shape),
            truth_jets_pf[..., 1:],
        ),
        axis=-1,
    ),
    r"Particle $\eta^{rel}$ Smeared",
)
pf_dists["pt_smeared"] = (
    np.concatenate(
        (
            truth_jets_pf[..., :2],
            truth_jets_pf[..., 2:3] * np.random.normal(1, 0.25, size=truth_jets_pf[..., 2:3].shape),
            truth_jets_pf[..., 3:],
        ),
        axis=-1,
    ),
    r"Particle $p_T^{rel}$ Smeared",
)
pf_dists["pt_shifted"] = (
    np.concatenate(
        (
            truth_jets_pf[..., :2],
            truth_jets_pf[..., 2:3]
            * np.random.normal(0.9, 0.1, size=truth_jets_pf[..., 2:3].shape),
            truth_jets_pf[..., 3:],
        ),
        axis=-1,
    ),
    r"Particle $p_T^{rel}$ Shifted",
)

for key, (data, _) in pf_dists.items():
    original = np.load(f"data/raghav/{key}.npy")

    assert (original - pf_dists[key][0][:,:,:3]).all() == 0.00000000
    print(np.sum(original - pf_dists[key][0][:,:,:3]))
    cartestian_data = etaphipt_epxpypz(data)
    print(f"{key} matched")
    np.save(f"data/distorted_jets/{key}.npy", data)
    np.save(f"data/distorted_jets/{key}_cartesian.npy", cartestian_data)

