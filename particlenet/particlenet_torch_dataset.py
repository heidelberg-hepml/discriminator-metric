import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ParticleNetDataset(Dataset):
    def __init__(self, data_file,split, features=7):
        
        self.data = pd.read_hdf(data_file, f'particle_data_{split}').values.reshape(-1,150,features)
        self.label = pd.read_hdf(data_file, f'labels_{split}')['labels'].values


    def __len__(self):
        return len(self.particle_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        particle_data = self.particle_data[idx]
        label = self.label[idx]

        return particle_data, label

