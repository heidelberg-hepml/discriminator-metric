import pandas as pd
import numpy as np
import h5py
from types import SimpleNamespace

from ..dataset import DiscriminatorData
from ..observable import Observable

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
            }
    datasets_list = [
            {'level': 'low', 'normalize': True, 'label': 'Norm.', 'suffix': 'norm'},
            {'level': 'low', 'normalize': False, 'label': 'Unnorm.', 'suffix': 'unnorm'},
            #{'level': 'high', 'normalize': False, 'label': 'High', 'suffix': 'high'},
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
    pass

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

   
