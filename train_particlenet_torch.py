from particlenet.particlenet_models_torch import *
from particlenet.particlenet_train_test import *
from particlenet.particlenet_torch_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='ParticleNet pytorch Training')
parser.add_argument('--data_file', type=str, default='data/converted/jetnet_converted_data.h5', help='data file')
parser.add_argument('--batch_size', type=int, default=384, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
args = parser.parse_args()


# Training settings
batch_size = args.batch_size
epochs = args.epochs

# Load data
train_dataset = ParticleNetDataset(args.data_file, 'train')
val_dataset = ParticleNetDataset(args.data_file, 'val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define model
device = torch.device(args.device)
model = ParticleNet().to(device)



