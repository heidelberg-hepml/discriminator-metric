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


def train(model, device, train_loader, loss_fn, optimizer, epoch):
    size = len(train_loader.dataset)
    train_loss = 0
    model.train()
    for batch_idx, (particle_data, label) in enumerate(train_loader):
        particle_data, label = particle_data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(particle_data)
        loss = loss_fn(output, label)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(particle_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return train_loss/size

def test(model, device, test_loader, loss_fn):
    size = len(test_loader.dataset)
    test_loss = 0
    classifier_output = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (particle_data, label) in enumerate(test_loader):
            particle_data, label = particle_data.to(device), label.to(device)
            output = model(particle_data)
            classifier_output.extend(output)
            test_loss += loss_fn(output, label).item()

    test_loss /= size
    print('Test set: Average loss: {:.4f}')

    return test_loss, classifier_output
    