import pandas as pd
import numpy as np
import torch
import h5py
from types import SimpleNamespace

from ..dataset import DiscriminatorData
from ..observable import Observable
from ..calo_plotting_helper import *

from ..HighLevelFeatures import *

def load(params: dict) -> list[DiscriminatorData]:
    """
    dataloader for calo data
    parameters:

    args:

    return:

    """
    datasets = []
    p_type = params.get('p_type', None)
    if p_type == 'pion':
        p_lab = '$\pi^{+}$'
    elif p_type == 'photon':
        p_lab = '$\gamma$'
    elif p_type == 'electron':
        p_lab = '$e^{+}$'
    preproc_kwargs = {
            "add_norm_Einc": params.get("add_norm_Einc", False),
            "add_norm": params.get("add_norm", False),
            "add_log_energy": params.get("add_log_energy", False),
            "add_logit_step": params.get("add_logit_step", False),
			"add_cut": params.get("add_cut", 0.0),
            "add_logp": params.get("add_logp", False),
            }
    datasets_list = [
            {'level': 'low', 'label': p_lab+' Unnorm.', 'suffix': 'unnorm'},
            {'level': 'low-norm', 'label': p_lab+' Norm.', 'suffix': 'norm'},
            {'level': 'high', 'label': p_lab+' High', 'suffix': 'high'}
            ]

    energy = params.get('single_energy', None)

    xml_handler = XMLHandler(particle_name=p_type, filename=params['xml_filename'])
    layer_boundaries = np.unique(xml_handler.GetBinEdges())
    
    for dataset in datasets_list:
        # load hlf classes
        hlf_class_true = HighLevelFeatures(p_type,
                                        filename=params['xml_filename'])
        hlf_class_fake = HighLevelFeatures(p_type,
                                        filename=params['xml_filename'])
 
        if dataset['level'] == 'low':
            geant_sample, logp_geant = create_data(params['geant_file'], hlf_class_true, layer_boundaries, energy=energy, cut=params.get("add_cut", 0.0))
            gen_sample, logp_samp = create_data(params['generated_file'], hlf_class_fake, layer_boundaries, energy=energy, cut=params.get("add_cut", 0.0))
        elif dataset['level'] == 'low-norm':
            geant_sample, logp_geant = create_data(params['geant_file'], hlf_class_true, layer_boundaries, energy=energy, norm=True, cut=params.get("add_cut", 0.0))
            gen_sample, logp_samp = create_data(params['generated_file'], hlf_class_fake, layer_boundaries, energy=energy, norm=True, cut=params.get("add_cut", 0.0)) 
        elif dataset['level'] == 'high':
            geant_sample = create_data_high(params['geant_file'], hlf_class_true, cut=params.get("add_cut", 0.0), single_energy=energy)
            gen_sample = create_data_high(params['generated_file'], hlf_class_fake, cut=params.get("add_cut", 0.0), single_energy=energy)

        train_true, test_true, val_true = split_data(
            geant_sample,
            params["train_split"],
            params["test_split"]
        )
        train_fake, test_fake, val_fake = split_data(
            gen_sample,
            params["train_split"],
            params["test_split"]
        )
       
        obs_np_true, labels_true, bins_true = get_list_observables(params['geant_file'], hlf_class_true, params.get('add_cut'), energy)
        obs_np_fake, labels_fake, _  = get_list_observables(params['generated_file'], hlf_class_fake, params.get('add_cut'), energy)
        train_obs_true, test_obs_true, val_obs_true = split_data(
                obs_np_true,
                params["train_split"],
                params["test_split"]
                )
        train_obs_fake, test_obs_fake, val_obs_fake = split_data(
                obs_np_fake,
                params["train_split"],
                params["test_split"]
                )
        
        #if logp_geant is not None:
        #    train_logp_true, test_logp_true, val_logp_true = split_data(
        #        logp_geant,
        #        params["train_split"],
        #        params["test_split"]
        #    )
        #else: 
        #    test_logp_true = None

        #if logp_samp is not None:
        #    train_logp_fake, test_logp_fake, val_logp_fake = split_data(
        #        logp_samp,
        #        params["train_split"],
        #        params["test_split"]
        #    )
        #else:
        #    test_logp_fake = None

        observables = []
        observables = compute_observables(test_obs_true, test_obs_fake, labels_true, labels_fake, bins_true)
        datasets.append(DiscriminatorData(
                label = dataset['label'],
                suffix = dataset['suffix'],
                dim = geant_sample.shape[-1],
                train_true = train_true,
                train_fake = train_fake,
                test_true = test_true,
                test_fake = test_fake,
                val_true = val_true,
                val_fake = val_fake,
                observables = observables,
                test_logw = test_logp_fake if logp_samp is not None else None
            )
        )
    return datasets

def create_data(data_path, hlf_class, layer_boundaries, energy=None, norm=False, cut=0.0):
    """ takes hdf5_file, extracts Einc and voxel energies, appends label, returns array """
    voxel, E_inc = extract_shower_and_energy(data_path, single_energy=energy)
    voxel[voxel<cut] = 0.0
    
    hlf_class.CalculateFeatures(voxel)
    hlf_class.Einc = E_inc

    if norm:
        E_norm_rep = []
        E_norm = []
        for idx, layer_id in enumerate(hlf_class.GetElayers()):
            E_norm_rep.append(np.repeat(hlf_class.GetElayers()[layer_id].reshape(-1, 1),
                                        hlf_class.num_voxel[idx], axis=1))
            E_norm.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        E_norm_rep = np.concatenate(E_norm_rep, axis=1)
        E_norm = np.concatenate(E_norm, axis=1)
    if norm:
        voxel = voxel / (E_norm_rep+1e-16)
        ret = np.concatenate([np.log10(E_inc), voxel, np.log10(E_norm+1e-8)], axis=1)
    else:
        voxel = voxel / E_inc
        ret = np.concatenate([np.log10(E_inc), voxel], axis=1)
    return torch.tensor(ret), None


def create_data_high(hdf5_file, hlf_class, cut=0.0, single_energy=None):
     
    """
    from CaloChallenge evaluation script
    takes hdf5_file, extracts high-level features, appends label, returns array 
    """

    voxel, E_inc = extract_shower_and_energy(hdf5_file, single_energy=single_energy)
    voxel[voxel<cut] = 0.0
    
    hlf_class.CalculateFeatures(voxel)
    hlf_class.Einc = E_inc

    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    obs_np = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2], axis=1)
    return obs_np
 
def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""
    
    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data)-1
    
    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"
        
        layers.append(data[layer])
        
    return energy, layers

def compute_observables(obs_np_true, obs_np_fake, labels_true, labels_fake, bins) -> list[Observable]:
    observables = []
    for i in range(obs_np_true.shape[1]):
        if 'Energy deposited' in labels_true[i]:
            observables.extend([
                Observable(
                    true_data = obs_np_true[:,i],
                    fake_data = obs_np_fake[:,i],
                    tex_label = f"{labels_true[i]}",
                    bins = bins[i],
                    xscale = 'log',
                    yscale='log'
                    )
                ])
        elif 'Center of Energy' in labels_true[i]:
            observables.extend([
                Observable(
                    true_data = obs_np_true[:,i],
                    fake_data = obs_np_fake[:,i],
                    tex_label = f"{labels_true[i]}",
                    bins = bins[i],
                    xscale = 'linear',
                    yscale='log'
                    )
                ])
        else:
            observables.extend([
                Observable(
                    true_data = obs_np_true[:,i],
                    fake_data = obs_np_fake[:,i],
                    tex_label = f"{labels_true[i]}",
                    bins = bins[i],
                    xscale = 'linear',
                    yscale='log'
                    )
                ])
    return observables

def get_list_observables(hdf5_file, hlf_class, cut=0.0, single_energy=None):
     
    """
    from CaloChallenge evaluation script
    takes hdf5_file, extracts high-level features, appends label, returns array 
    """

    voxel, E_inc = extract_shower_and_energy(hdf5_file, single_energy=single_energy)
    voxel[voxel<cut] = 0.0
    
    hlf_class.CalculateFeatures(voxel)
    hlf_class.Einc = E_inc

    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
        
    if hlf_class.particle == 'electron':
        lim_wec = (0., 30.)
        lim_ec = (-30., 30.)
    else:
        lim_wec = (0., 100.)
        lim_ec = (-100., 100.)
        
    strings = [f'\\text{{Center of Energy in }}\Delta\eta\\text{{ in layer }}',
                f'\\text{{Center of Energy in }}\Delta\phi\\text{{ in layer }}',
                f'\\text{{Width of Center of Energy in }}\Delta\eta\\text{{ in layer }}',
                f'\\text{{Width of Center of Energy in }}\Delta\phi\\text{{ in layer }}']
    
    bins = []
    bins.append(np.logspace(0, 5, 51))
    labels = []
    labels.append(r"E_{tot}/E_{inc}")
    for layer_id in hlf_class.GetElayers():
        labels.append(f'\\text{{Energy deposited in layer }}{layer_id}')
        bins.append(np.logspace(-3, 5, 51))
    for i, j in enumerate(strings):
        for layer_id in hlf_class.layersBinnedInAlpha:
            labels.append(j + f"{layer_id}")
            
            if layer_id in [12, 13] and hlf_class.particle != 'electron':
                lim_ec = (-400., 400.)
                lim_wec = (0., 400.)
            
            bins_ec = np.linspace(*lim_ec, 51)
            bins_wec = np.linspace(*lim_wec, 51)
            if i<= 1:
                bins.append(bins_ec)
            else:
                bins.append(bins_wec)

    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    obs_np = np.concatenate([np.log10(E_inc), E_layer, EC_etas, EC_phis,
                          Width_etas, Width_phis], axis=1)
    return obs_np, labels, bins
 
def extract_shower_and_energy(given_file, single_energy=None) -> tuple[np.ndarray]:
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(given_file))
    given_file = h5py.File(given_file, 'r')
    if single_energy is not None:
        energy_mask = given_file["incident_energies"][:] == single_energy
        energy = given_file["incident_energies"][:][energy_mask].reshape(-1, 1)
        shower = given_file["showers"][:][energy_mask.flatten()]
    else:
        shower = given_file['showers'][:]
        energy = given_file['incident_energies'][:]
    print("Extracting showers from {} file: DONE.\n".format(given_file))
    return shower, energy

  
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

   
