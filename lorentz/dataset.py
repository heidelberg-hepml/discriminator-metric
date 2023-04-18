import torch
import numpy as np
import energyflow
from scipy.sparse import coo_matrix
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx*n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges

def collate_fn(data):
    data = list(zip(*data)) # label p4s nodes atom_mask
    data = [torch.stack(item) for item in data]
    batch_size, n_nodes, _ = data[1].size()
    atom_mask = data[-1]
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edges = get_adj_matrix(n_nodes, batch_size, edge_mask)
    return data + [edge_mask, edges]

def retrieve_dataloaders(batch_size, num_data = -1, test_jet='tailcut',use_one_hot = True, cache_dir = './data', num_workers=4):
    #raw = energyflow.qg_jets.load(num_data=num_data, pad=True, ncol=4, generator='pythia',
                   #         with_bc=False, cache_dir=cache_dir)
    splits = ['train', 'valid']
    data = {type:{'raw':None,'label':None} for type in splits}
    print(f'Loading data for jets distortion type: {test_jet}')
    for s in splits:
        data[s]['raw'] = pd.read_hdf('data/jetnet30_data.h5', f'particle_data_{test_jet}_{s}').values.reshape(-1,30,4)
        data[s]['label'] = pd.read_hdf('data/jetnet30_data.h5', f'labels__{test_jet}_{s}')['labels'].values
 
    for split, value in data.items():

        p4s = torch.from_numpy(value['raw'])  
        mass = torch.from_numpy(energyflow.ms_from_p4s(p4s)).unsqueeze(-1)

        nodes = mass
        nodes = torch.sign(nodes) * torch.log(torch.abs(nodes) + 1)
        
          #  nodes = torch.sign(nodes) * torch.log(torch.abs(nodes) + 1)
        atom_mask = atom_mask = torch.tensor((value['raw']!=[0,0,0,0])[...,0]) 

        value['p4s'] = p4s
        value['nodes'] = nodes
        value['label'] = torch.from_numpy(value['label'])
        value['atom_mask'] = atom_mask.to(torch.bool)

    datasets = {split: TensorDataset(value['label'], value['p4s'],
                                     value['nodes'], value['atom_mask'])
                for split, value in data.items()}

    # distributed training
    train_sampler = DistributedSampler(datasets['train'], shuffle=True)
    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size,
                                     sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),pin_memory=True,
                                     persistent_workers=True,
                                     drop_last=True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                        for split, dataset in datasets.items()}

    return train_sampler, dataloaders
    #return dataloaders

if __name__ == '__main__':
    train_sampler, dataloaders = retrieve_dataloaders(32, 100)
   # dataloaders = retrieve_dataloaders(32, 100)
    for (label, p4s, nodes, atom_mask, edge_mask, edges) in dataloaders['train']:
        print(label.shape, p4s.shape, nodes.shape, atom_mask.shape,
              edge_mask.shape, edges[0].shape, edges[1].shape)
        break
