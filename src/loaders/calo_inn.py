import pandas as pd
import numpy as np
import torch
import h5py
from types import SimpleNamespace

from ..dataset import DiscriminatorData
from ..observable import Observable
from ..calo_plotting_helper import *

def load(params: dict) -> list[DiscriminatorData]:
    """
    dataloader for calo data
    parameters:

    args:

    return:

    """
    datasets = []
    preproc_kwargs = {
            "add_log_energy": params.get("add_log_energy", False),
            "add_log_layer_ens": params.get("add_log_layer_ens", False),
            "add_logit_step": params.get("add_logit_step", False),
			"add_cut": params.get("add_cut", 0.0),
            }
    datasets_list = [
            {'level': 'low', 'normalize': True, 'label': 'Norm.', 'suffix': 'norm'},
            {'level': 'low', 'normalize': False, 'label': 'Unnorm.', 'suffix': 'unnorm'},
            {'level': 'high', 'normalize': False, 'label': 'High', 'suffix': 'high'},
            ] 

    for dataset in datasets_list:
        if dataset['level'] == 'low':
            geant_sample = create_data(params['geant_file'], dataset, **preproc_kwargs)
            gen_sample = create_data(params['generated_file'], dataset, **preproc_kwargs)
        elif dataset['level'] == 'high':
            geant_sample = create_data_high(params['geant_file'], dataset, **preproc_kwargs)
            gen_sample = create_data_high(params['generated_file'], dataset, **preproc_kwargs)
        else:
            raise ValueError('Classifier preprocessing running at unknown level.')
        
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
                observables = [],
            )
        )
    return datasets

def create_data(data_path, dataset_list, **kwargs):
    with h5py.File(data_path, "r") as f:
        en_test = f.get('energy')[:] / 1e2
        lay_0 = f.get('layer_0')[:] / 1e5
        lay_1 = f.get('layer_1')[:] / 1e5
        lay_2 = f.get('layer_2')[:] / 1e5
    data = np.concatenate((lay_0.reshape(-1, 288), lay_1.reshape(-1, 144), lay_2.reshape(-1, 72)), axis=1)

    en0_t = np.sum(data[:, :288], axis=1, keepdims=True)
    en1_t = np.sum(data[:, 288:432], axis=1, keepdims=True)
    en2_t = np.sum(data[:, 432:], axis=1, keepdims=True)
	
    if dataset_list['normalize']:
        data[:, :288] /= en0_t + 1e-16
        data[:, 288:432] /= en1_t + 1e-16
        data[:, 432:] /= en2_t + 1e-16
 
    if kwargs['add_log_energy']:
        data = np.concatenate((data, np.log10(en_test*10).reshape(-1, 1)), axis=1)
    data = np.nan_to_num(data, posinf=0, neginf=0)
        
    en0_t = np.log10(en0_t + 1e-8) + 2.
    en1_t = np.log10(en1_t + 1e-8) +2.
    en2_t = np.log10(en2_t + 1e-8) +2.
 
    if kwargs['add_log_layer_ens']:
        data = np.concatenate((data, en0_t, en1_t, en2_t), axis=1)
    if kwargs['add_logit_step']:
        raise ValueError('Not implemented yet')
    return data

def create_data_high(data_path, dataset_list, **kwargs):
    cut = kwargs['add_cut']
    with h5py.File(data_path, "r") as f:
        en_test = f.get('energy')[:] / 1e2
        lay_0 = f.get('layer_0')[:] / 1e5
        lay_1 = f.get('layer_1')[:] / 1e5
        lay_2 = f.get('layer_2')[:] / 1e5

    incident_energy = torch.log10(torch.tensor(en_test)*10.)
    # scale them back to MeV
    layer0 = torch.tensor(lay_0) * 1e5
    layer1 = torch.tensor(lay_1) * 1e5
    layer2 = torch.tensor(lay_2) * 1e5
    layer0 = to_np_thres(layer0.view(layer0.shape[0], -1), cut)
    layer1 = to_np_thres(layer1.view(layer1.shape[0], -1), cut)
    layer2 = to_np_thres(layer2.view(layer2.shape[0], -1), cut)
    # detour to numpy in order to use same functions as plotting script
    full_shower = np.concatenate((layer0, layer1, layer2), 1)
    E_0 = energy_sum(layer0)
    E_1 = energy_sum(layer1)
    E_2 = energy_sum(layer2)
    E_tot = E_0 + E_1 + E_2
    f_0 = E_0 / E_tot
    f_1 = E_1 / E_tot
    f_2 = E_2 / E_tot
    l_d = depth_weighted_energy(full_shower)
    s_d = l_d / (E_tot * 1e3)
    sigma_sd = depth_weighted_energy_normed_std(full_shower)
    E_1b0, E_2b0, E_3b0, E_4b0, E_5b0 = n_brightest_voxel(layer0, [1, 2, 3, 4, 5]).T
    E_1b1, E_2b1, E_3b1, E_4b1, E_5b1 = n_brightest_voxel(layer1, [1, 2, 3, 4, 5]).T
    E_1b2, E_2b2, E_3b2, E_4b2, E_5b2 = n_brightest_voxel(layer2, [1, 2, 3, 4, 5]).T
    ratio_0 = ratio_two_brightest(layer0)
    ratio_1 = ratio_two_brightest(layer1)
    ratio_2 = ratio_two_brightest(layer2)
    sparsity_0 = layer_sparsity(layer0, cut)
    sparsity_1 = layer_sparsity(layer1, cut)
    sparsity_2 = layer_sparsity(layer2, cut)
    phi_0 = center_of_energy(layer0, 0, 'phi')
    phi_1 = center_of_energy(layer1, 1, 'phi')
    phi_2 = center_of_energy(layer2, 2, 'phi')
    eta_0 = center_of_energy(layer0, 0, 'eta')
    eta_1 = center_of_energy(layer1, 1, 'eta')
    eta_2 = center_of_energy(layer2, 2, 'eta')
    sigma_0 = center_of_energy_std(layer0, 0, 'phi')
    sigma_1 = center_of_energy_std(layer1, 1, 'phi')
    sigma_2 = center_of_energy_std(layer2, 2, 'phi')

    # to be log10-processed:
    ret1 = np.vstack([E_0+1e-8, E_1+1e-8, E_2+1e-8, E_tot,
                      f_0+1e-8, f_1+1e-8, f_2+1e-8, l_d+1e-8,
                      sigma_0+1e-8, sigma_1+1e-8, sigma_2+1e-8]).T
    ret1 = np.log10(ret1)
    # without log10 processing:
    ret2 = np.vstack([s_d, sigma_sd,
                      1e1*E_1b0, 1e1*E_2b0, 1e1*E_3b0, 1e1*E_4b0, 1e1*E_5b0,
                      1e1*E_1b1, 1e1*E_2b1, 1e1*E_3b1, 1e1*E_4b1, 1e1*E_5b1,
                      1e1*E_1b2, 1e1*E_2b2, 1e1*E_3b2, 1e1*E_4b2, 1e1*E_5b2,
                      ratio_0, ratio_1, ratio_2, sparsity_0, sparsity_1, sparsity_2,
                      phi_0/1e2, phi_1/1e2, phi_2/1e2, eta_0/1e2, eta_1/1e2, eta_2/1e2]).T
    ret = torch.from_numpy(np.hstack([ret1, ret2]))

    ret = torch.cat((ret, incident_energy), 1)
    return ret.numpy()

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

   
