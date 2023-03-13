import pandas as pd
import torch
import numpy as np
from types import SimpleNamespace

from .dataset import DiscriminatorData
from .observable import Observable

def load(params: dict) -> list[DiscriminatorData]:
    true_momenta = pandas.read_hdf(params["truth_file"]).to_numpy().reshape(-1, 5, 4)
    fake_momenta = pandas.read_hdf(params["generated_file"]).to_numpy().reshape(-1, 5, 4)
    multiplicity_true = np.sum(true_momenta[:,:,0] != 0., axis=1)
    multiplicity_fake = np.sum(fake_momenta[:,:,0] != 0., axis=1)

    subsets = [
        {"multiplicity": 3, "label": "$Z+1j$", "suffix": "z1j", "dim": },
        {"multiplicity": 4, "label": "$Z+2j$", "suffix": "z2j", "dim": },
        {"multiplicity": 5, "label": "$Z+3j$", "suffix": "z3j", "dim": },
    ]
    datasets = []
    for subset in subsets:
        mult = subsets["multiplicity"]
        subset_true = true_momenta[multiplicity_true == mult][:, :mult]
        subset_fake = fake_momenta[multiplicity_fake == mult][:, :mult]
        train_true, test_true, val_true = split_data(
            subset_true,
            params["train_split"],
            params["test_split"]
        )
        train_fake, test_fake, val_fake = split_data(
            subset_fake,
            params["train_split"],
            params["test_split"]
        )
        preproc_kwargs = {
            "norm": {},
            "append_mass": params.get("append_mass", False),
            "append_delta_r": params.get("append_delta_r", False)
        }
        datasets.append(DiscriminatorData(
            label = subsets["label"],
            suffix = subsets["suffix"],
            dims = subsets["dim"],
            train_true = compute_preprocessing(train_true, **preproc_kwargs),
            train_fake = compute_preprocessing(train_fake, **preproc_kwargs),
            test_true = compute_preprocessing(test_true, **preproc_kwargs),
            test_fake = compute_preprocessing(test_fake, **preproc_kwargs),
            val_true = compute_preprocessing(val_true, **preproc_kwargs),
            val_fake = compute_preprocessing(val_fake, **preproc_kwargs),
            observables = compute_observables(test_true, test_fake),
        ))
    return datasets


def split_data(
    data: np.ndarray,
    train_split: float,
    test_split: float
) -> tuple[np.ndarray, ...]:
    n_train = int(train_split * len(data))
    n_test = int(test_split * len(data))
    train_data = data[:n_train]
    test_data = data[-n_test:]
    val_data = data[n_train:-n_test]
    return train_data, test_data, val_data


def compute_preprocessing(
    data: np.ndarray,
    norm: dict,
    append_mass: bool,
    append_delta_r: bool
) -> torch.util.data.Dataset:
    obs = observables_one_particle(data)
    dphi = lambda phi1, phi2: (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi
    dr = lambda phi1, phi2, eta1, eta2: np.sqrt(dphi(phi1, phi2)**2 + (eta1 - eta2)**2)

    input_obs = [
        obs.pt[:,0], obs.eta[:,0],
        obs.pt[:,1], obs.eta[:,1], dphi(obs.phi[:,0], obs.phi[:,1]),
        obs.pt[:,2], obs.eta[:,2], dphi(obs.phi[:,1], obs.phi[:,2]), obs.m[:,2],
        obs.pt[:,3], obs.eta[:,3], dphi(obs.phi[:,2], obs.phi[:,3]), obs.m[:,3],
        obs.pt[:,4], obs.eta[:,4], dphi(obs.phi[:,3], obs.phi[:,4]), obs.m[:,4],
    ]
    if append_m_mumu:
        p_mumu = data[:,0] + data[:,1]
        mass = np.sqrt(p_mumu[:,0]**2 - p_mumu[:,1]**2 - p_mumu[:,2]**2 - p_mumu[:,3]**2)
        input_obs.append(mass)
    if append_delta_r:
        input_obs.append(dr(obs.phi[:,2], obs.phi[:,3], obs.eta[:,2], obs.eta[:,3]))
        input_obs.append(dr(obs.phi[:,3], obs.phi[:,4], obs.eta[:,3], obs.eta[:,4]))
        input_obs.append(dr(obs.phi[:,2], obs.phi[:,4], obs.eta[:,2], obs.eta[:,4]))

    data_preproc = np.stack(input_obs, axis=0)
    if "means" not in norm:
        norm["means"] = np.mean(data_preproc, axis=0)
    if "stds" not in norm:
        norm["stds"] = np.std(data_preproc, axis=0)
    return torch.util.data.TensorDataset(
        (data_preproc - norm["means"]) / norm["stds"]
    )


def compute_observables(true_data: np.ndarray, fake_data: np.ndarray) -> list[Observable]:
    mult = true_data.shape[1]
    obs_one_true = [observables_one_particle(true_data[:,i]) for i in range(mult)]
    obs_one_fake = [observables_one_particle(fake_data[:,i]) for i in range(mult)]
    pairs = [(i,j) for i in range(mult) for j in range(mult) if i < j]
    obs_two_true = {
        (i,j): observables_two_particles(obs_one_true[i], obs_one_true[j]) for i,j in pairs
    }
    obs_two_fake = {
        (i,j): observables_two_particles(obs_one_fake[i], obs_one_fake[j]) for i,j in pairs
    }
    return [Observable(
        true_data = obs_one_true[i].pt,
        fake_data = obs_one_fake[i].pt,
        tex_label = f"p_{{T,{i}}}",
        bins = np.linspace(np.min(obs_one_true[i].pt), np.quantile(obs_one_true[i].pt, 0.99)),
        yscale = "log",
        unit = "GeV"
    ) for i in range(mult)]


def observables_one_particle(momenta: np.ndarray) -> SimpleNamespace:
    r = SimpleNamespace()
    r.e = momenta[...,0]
    r.px = momenta[...,1]
    r.py = momenta[...,2]
    r.pz = momenta[...,3]
    r.pt = np.sqrt(r.px**2 + r.py**2)
    r.p = np.sqrt(r.px**2 + r.py**2 + r.pz**2)
    r.eta = np.arctanh(r.pz / r.p)
    r.phi = np.arctan2(r.py, r.px)
    r.m = np.sqrt(r.e**2 - r.px**2 - r.py**2 - r.pz**2)
    return r


def observables_two_particles(obs1: SimpleNamespace, obs2: SimpleNamespace) -> SimpleNamespace:
    r = SimpleNamespace()
    r.delta_phi = (obs1.phi - obs2.phi + np.pi) % (2*np.pi) - np.pi
    r.delta_eta = obs1.eta - obs2.eta
    r.delta_r = np.sqrt(r.delta_phi**2 + r.delta_eta**2)
    return r
