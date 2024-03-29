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
    p_type = params.get('p_type', None)
    if p_type == 'pions':
        p_lab = '$\pi^{+}$'
    elif p_type == 'gammas':
        p_lab = '$\gamma$'
    elif p_type == 'eplus':
        p_lab = '$e^{+}$'
    preproc_kwargs = {
            "add_log_energy": params.get("add_log_energy", False),
            "add_log_layer_ens": params.get("add_log_layer_ens", False),
            "add_logit_step": params.get("add_logit_step", False),
			"add_cut": params.get("add_cut", 0.0),
            }
    datasets_list = [
            {'level': 'low', 'normalize': False, 'label': p_lab+' Unnorm.', 'suffix': 'unnorm'},
            {'level': 'low', 'normalize': True, 'label': p_lab+' Norm.', 'suffix': 'norm'},
            {'level': 'high', 'normalize': False, 'label': p_lab+' High', 'suffix': 'high'},
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
        
        observables = []
        if dataset['level']=='low':
            observables = compute_observables(test_true, test_fake)
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
            )
        )
    return datasets

def create_data(data_path, dataset_list, **kwargs):
    with h5py.File(data_path, "r") as f:
        en_test = f.get('energy')[:] / 1e2
        lay_0 = f.get('layer_0')[:] / 1e5
        lay_1 = f.get('layer_1')[:] / 1e5
        lay_2 = f.get('layer_2')[:] / 1e5
    
    torch_dtype = torch.get_default_dtype()
    lay_0 = torch.tensor(lay_0).to(torch_dtype)
    lay_1 = torch.tensor(lay_1).to(torch_dtype)
    lay_2 = torch.tensor(lay_2).to(torch_dtype)
    en_test = torch.tensor(en_test).to(torch_dtype)

    data = torch.cat((lay_0.reshape(-1, 288), lay_1.reshape(-1, 144), lay_2.reshape(-1, 72)), axis=1)

    en0_t = torch.sum(data[:, :288], axis=1, keepdims=True)
    en1_t = torch.sum(data[:, 288:432], axis=1, keepdims=True)
    en2_t = torch.sum(data[:, 432:], axis=1, keepdims=True)
	
    if dataset_list['normalize']:
        data[:, :288] /= en0_t + 1e-16
        data[:, 288:432] /= en1_t + 1e-16
        data[:, 432:] /= en2_t + 1e-16
 
    if kwargs['add_log_energy']:
        data = torch.cat((data, torch.log10(en_test*10).reshape(-1, 1)), axis=1)
    #data = np.nan_to_num(data, posinf=0, neginf=0)
        
    en0_t = torch.log10(en0_t + 1e-8) + 2.
    en1_t = torch.log10(en1_t + 1e-8) +2.
    en2_t = torch.log10(en2_t + 1e-8) +2.
 
    if kwargs['add_log_layer_ens']:
        data = torch.cat((data, en0_t, en1_t, en2_t), axis=1)
    if kwargs['add_logit_step']:
        raise ValueError('Not implemented yet')
    return data.numpy()

def create_data_high(data_path, dataset_list, **kwargs):
    cut = kwargs['add_cut']
    with h5py.File(data_path, "r") as f:
        en_test = f.get('energy')[:] / 1e2
        lay_0 = f.get('layer_0')[:] / 1e5
        lay_1 = f.get('layer_1')[:] / 1e5
        lay_2 = f.get('layer_2')[:] / 1e5
    torch_dtype = torch.get_default_dtype()
    
    incident_energy = torch.log10(torch.tensor(en_test).to(torch_dtype)*10.)
    # scale them back to MeV
    layer0 = torch.tensor(lay_0).to(torch_dtype) * 1e5
    layer1 = torch.tensor(lay_1).to(torch_dtype) * 1e5
    layer2 = torch.tensor(lay_2).to(torch_dtype) * 1e5
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

def compute_observables(true_data: np.ndarray, fake_data: np.ndarray) -> list[Observable]:
    observables = []
    phi0_true = center_of_energy(true_data[:,:288], 0, 'phi')
    phi1_true = center_of_energy(true_data[:,288:432], 1, 'phi')
    phi2_true = center_of_energy(true_data[:,432:504], 2, 'phi')

    phi0_fake = center_of_energy(fake_data[:,:288], 0, 'phi')
    phi1_fake = center_of_energy(fake_data[:,288:432], 1, 'phi')
    phi2_fake = center_of_energy(fake_data[:,432:504], 2, 'phi')

    sparsity0_true = layer_sparsity(true_data[:,:288], 0.0)
    sparsity1_true = layer_sparsity(true_data[:,288:432], 0.0)
    sparsity2_true = layer_sparsity(true_data[:,432:504], 0.0)
    
    sparsity0_fake = layer_sparsity(fake_data[:,:288], 0.0)
    sparsity1_fake = layer_sparsity(fake_data[:,288:432], 0.0)
    sparsity2_fake = layer_sparsity(fake_data[:,432:504], 0.0)

    observables.extend([
        Observable(
            true_data = phi0_true,
            fake_data = phi0_fake,
            tex_label = f'phi0',
            bins = np.linspace(-125, 125, 50),
            xscale = 'linear',
            yscale = 'log',
            ),
         Observable(
            true_data = phi1_true,
            fake_data = phi1_fake,
            tex_label = f'phi1',
            bins = np.linspace(-125, 125, 50),
            xscale = 'linear',
            yscale = 'log',
            ),
        Observable(
            true_data = phi2_true,
            fake_data = phi2_fake,
            tex_label = f'phi2',
            bins = np.linspace(-125, 125, 50),
            xscale = 'linear',
            yscale = 'log',
            ),
        Observable(
            true_data = sparsity0_true,
            fake_data = sparsity0_fake,
            tex_label = f'sparsity0',
            bins = np.linspace(0, 1, 20),
            xscale = 'linear',
            yscale = 'linear',
            ),
        Observable(
            true_data = sparsity1_true,
            fake_data = sparsity1_fake,
            tex_label = f'sparsity1',
            bins = np.linspace(0, 1, 20),
            xscale = 'linear',
            yscale = 'linear',
            ),
        Observable(
            true_data = sparsity2_true,
            fake_data = sparsity2_fake,
            tex_label = f'sparsity2',
            bins = np.linspace(0, 1, 20),
            xscale = 'linear',
            yscale = 'linear',
            ),
        ])

    return observables

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

   
