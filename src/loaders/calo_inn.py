import pandas as pd
import numpy as np
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
            {'level': 'high', 'normalize': False, 'label': 'High', 'suffix': 'high'},
            ] 

    for dataset in datasets_list:
        if dataset['level'] == 'low':
            geant_sample = create_data(params['geant_file'], dataset)
            gen_sample = create_data(params['generated_file'], dataset)
        elif dataset['level'] == 'high':
            geant_sample = create_data_high(params['geant_file'], dataset)
            gen_sample = create_data_high(params['generated_file'], dataset)
        else:
            raise ValueError('Classifier preprocessing running at unknown level.')
        
        train_true, test_true, val_true = split_data(
            geant_sample,
            params["train_split"],
            params["test_split"]
        )
        train_fake, test_fake, val_fake = split_data(
            generated_sample,
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

#def create_data(data_path):
 
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

   
