import pandas as pd
import numpy as np
from types import SimpleNamespace

from ..dataset import DiscriminatorData
from ..observable import Observable

def load(params: dict) -> list[DiscriminatorData]:
    """
    Loads the training, test and validation data (truth samples and generated samples)
    and computes observables and preprocessed samples for the classifier training.
    This loader is written for Z+{1,2,3}j events, as generated in the paper
    Generative Networks for Precision Enthusiasts.

    Parameters:
        truth_file: Path to an HDF5 file with the truth data
        generated_file: Path to an HDF5 file with the generated data
        train_split: Fraction of events to be used as training data
        test_split: Fraction of events to be used as test data (the remaining events are used
                    for validation)
        include_momenta: Include the full kinematic data in the training
        append_mass: Append M_{mu mu} to the training data
        append_delta_r: Append the jet delta R to the training data

    Args:
        params: Dict with loader parameters (see above)

    Returns:
        List of three DiscriminatorData objects, one for each jet multiplicity
    """
    true_momenta = pd.read_hdf(params["truth_file"]).to_numpy().reshape(-1, 5, 4)
    fake_momenta = pd.read_hdf(params["generated_file"]).to_numpy().reshape(-1, 5, 4)
    true_momenta = true_momenta[np.all(np.isfinite(true_momenta), axis=(1,2))]
    fake_momenta = fake_momenta[np.all(np.isfinite(fake_momenta), axis=(1,2))]
    multiplicity_true = np.sum(true_momenta[:,:,0] != 0., axis=1)
    multiplicity_fake = np.sum(fake_momenta[:,:,0] != 0., axis=1)

    subsets = [
        {"multiplicity": 3, "label": "$Z+1j$", "suffix": "z1j"},
        {"multiplicity": 4, "label": "$Z+2j$", "suffix": "z2j"},
        {"multiplicity": 5, "label": "$Z+3j$", "suffix": "z3j"},
    ]
    datasets = []
    for subset in subsets:
        mult = subset["multiplicity"]
        subset_true = true_momenta[multiplicity_true == mult][:,:mult]
        subset_fake = fake_momenta[multiplicity_fake == mult][:,:mult]
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
            "include_momenta": params.get("include_momenta", True),
            "append_mass": params.get("append_mass", False),
            "append_delta_r": params.get("append_delta_r", False)
        }
        pp_train_all = compute_preprocessing(
            np.concatenate((train_true, train_fake), axis=0), **preproc_kwargs
        )
        datasets.append(DiscriminatorData(
            label = subset["label"],
            suffix = subset["suffix"],
            dim = pp_train_all.shape[1],
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
    include_momenta: bool,
    append_mass: bool,
    append_delta_r: bool
) -> np.ndarray:
    mult = data.shape[1]
    obs = observables_one_particle(data)
    dphi = lambda phi1, phi2: (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi
    dr = lambda phi1, phi2, eta1, eta2: np.sqrt(dphi(phi1, phi2)**2 + (eta1 - eta2)**2)

    input_obs = []
    if include_momenta:
        input_obs.extend([
            obs.pt[:,0], obs.eta[:,0],
            obs.pt[:,1], obs.eta[:,1], dphi(obs.phi[:,0], obs.phi[:,1]),
        ])
        for i in range(2, mult):
            input_obs.extend([
                obs.pt[:,i], obs.eta[:,i], dphi(obs.phi[:,i-1], obs.phi[:,i]), obs.m[:,i]
            ])

    if append_mass:
        p_mumu = data[:,0] + data[:,1]
        mass = np.sqrt(np.maximum(
            p_mumu[:,0]**2 - p_mumu[:,1]**2 - p_mumu[:,2]**2 - p_mumu[:,3]**2,
            0.
        ))
        input_obs.append(mass)

    if append_delta_r:
        drinv = lambda x: np.minimum(1/(x+1e-7), 20)
        if mult > 3:
            input_obs.append(drinv(dr(obs.phi[:,2], obs.phi[:,3], obs.eta[:,2], obs.eta[:,3])))
        if mult > 4:
            input_obs.append(drinv(dr(obs.phi[:,3], obs.phi[:,4], obs.eta[:,3], obs.eta[:,4])))
            input_obs.append(drinv(dr(obs.phi[:,2], obs.phi[:,4], obs.eta[:,2], obs.eta[:,4])))

    data_preproc = np.stack(input_obs, axis=1)
    if "means" not in norm:
        norm["means"] = np.mean(data_preproc, axis=0)
    if "stds" not in norm:
        norm["stds"] = np.std(data_preproc, axis=0)
    return (data_preproc - norm["means"]) / norm["stds"]


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
    observables = []
    for i in range(mult):
        observables.extend([
            Observable(
                true_data = obs_one_true[i].pt,
                fake_data = obs_one_fake[i].pt,
                tex_label = f"p_{{T,{i}}}",
                bins = np.linspace(
                    np.min(obs_one_true[i].pt),
                    np.quantile(obs_one_true[i].pt, 0.99),
                    50
                ),
                yscale = "log",
                unit = "GeV"
            ),
            Observable(
                true_data = obs_one_true[i].eta,
                fake_data = obs_one_fake[i].eta,
                tex_label = f"\\eta_{i}",
                bins = np.linspace(-6, 6, 50),
            )
        ])
        if i >= 2:
            observables.append(Observable(
                true_data = obs_one_true[i].m,
                fake_data = obs_one_fake[i].m,
                tex_label = f"M_{i}",
                bins = np.linspace(
                    np.quantile(obs_one_true[i].m, 0.005),
                    np.quantile(obs_one_true[i].m, 0.995),
                    50
                ),
                unit = "GeV"
            ))
    observables.append(Observable(
        true_data = obs_two_true[(0,1)].m,
        fake_data = obs_two_fake[(0,1)].m,
        tex_label = r"M_{\mu\mu}",
        bins = np.linspace(
            70,
            110,
            50
        ),
        unit = "GeV"
    ))
    for i, j in [(2,3), (2,4), (3,4)]:
        if i > mult-1 or j > mult-1:
            continue
        observables.extend([
            Observable(
                true_data = obs_two_true[(i,j)].delta_r,
                fake_data = obs_two_fake[(i,j)].delta_r,
                tex_label = f"\\Delta R_{{{i},{j}}}",
                bins = np.linspace(0, 12, 50),
            ),
            Observable(
                true_data = obs_two_true[(i,j)].delta_eta,
                fake_data = obs_two_fake[(i,j)].delta_eta,
                tex_label = f"\\Delta \\eta_{{{i},{j}}}",
                bins = np.linspace(-10, 10, 50),
            ),
            Observable(
                true_data = obs_two_true[(i,j)].delta_phi,
                fake_data = obs_two_fake[(i,j)].delta_phi,
                tex_label = f"\\Delta \\phi_{{{i},{j}}}",
                bins = np.linspace(-np.pi, np.pi, 50),
            )
        ])
    return observables


def observables_one_particle(momenta: np.ndarray) -> SimpleNamespace:
    r = SimpleNamespace()
    r.e = momenta[...,0]
    r.px = momenta[...,1]
    r.py = momenta[...,2]
    r.pz = momenta[...,3]
    r.pt = np.sqrt(r.px**2 + r.py**2)
    r.p = np.sqrt(r.px**2 + r.py**2 + r.pz**2)
    eps = 1e-7
    r.eta = np.arctanh(np.clip(r.pz / (r.p + eps), -1. + eps, 1. - eps))
    r.phi = np.arctan2(r.py, r.px)
    r.m = np.sqrt(np.maximum(r.e**2 - r.px**2 - r.py**2 - r.pz**2, 0.))
    return r


def observables_two_particles(obs1: SimpleNamespace, obs2: SimpleNamespace) -> SimpleNamespace:
    r = SimpleNamespace()
    r.delta_phi = (obs1.phi - obs2.phi + np.pi) % (2*np.pi) - np.pi
    r.delta_eta = obs1.eta - obs2.eta
    r.delta_r = np.sqrt(r.delta_phi**2 + r.delta_eta**2)
    r.m = np.sqrt(np.maximum((obs1.e+obs2.e)**2 - (obs1.px+obs2.px)**2 -
                             (obs1.py+obs2.py)**2 - (obs1.pz+obs2.pz)**2, 0))
    return r
