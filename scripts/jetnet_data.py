from jetnet.datasets import JetNet
from jetnet.utils import *
import wandb
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
from conversion import *

# Download JetNet data
split = ['train', 'valid', 'test']
particle_data = {}
jet_data = {}

for s in split:
    particle_data[s], jet_data[s] = JetNet.getData(jet_type=["g"], data_dir="data/",
                                             jet_features=["pt", "eta", "mass",
                                                           "num_particles"],split=s,
                                                           num_particles=150)
    print(f'{s} particle_data.shape = ', particle_data[s].shape, 
          f'{s} jet_data.shape = ', jet_data[s].shape)


# Merge valid and test data into one validation set
particle_data['val'] = np.vstack((particle_data['valid'], particle_data['test']))
jet_data['val'] = np.vstack((jet_data['valid'], jet_data['test']))

jet_data['train'] = pd.DataFrame(jet_data['train'],columns=['pt','eta','mass','num_particles'])
jet_data['val'] = pd.DataFrame(jet_data['val'],columns=['pt','eta','mass','num_particles'])

print('particle_data_val.shape = ', particle_data['val'].shape, 
      'jet_data_val.shape = ', jet_data['val'].shape)

print('fraction traindata: ', len(jet_data['train'])/(len(jet_data['val'])+len(jet_data['train'])))


# create new training and validation set with labels
new_split = ['train','val']
new_particle_data = {}
new_jet_data = {}


  


for s in new_split:
      new_particle_data[s] = {}
      new_jet_data[s] = {}

# convert data to cartesian coordinates

      particle_data[s] = etaphipt_rel_to_epxpypz(particle_data[s],jet_data[s])
      #print(particle_data[s][...,0:3].shape)
      #print(jet_data[s].values.shape)
      #particle_data[s] = relEtaPhiPt_to_cartesian(particle_data[s][...,0:3],jet_data[s].values)
      # split data

      jet_data_sig, jet_data_back = train_test_split(jet_data[s], test_size=0.5, random_state=42)
      particle_data_sig, particle_data_back = train_test_split(particle_data[s], test_size=0.5, random_state=42)

      jet_data_back_m = jet_data_back['mass']
      jet_data_back_pt = jet_data_back['pt']
      jet_data_back_m_pt = jet_data_back_m/jet_data_back_pt

      # remove jets with mass/pt > 0.175
      cut = jet_data_back_m_pt < 0.175
      jet_data_back = jet_data_back[cut]
      particle_data_back = particle_data_back[cut]


      # create labels
      labels_sig = np.ones(len(jet_data_sig))
      labels_bag = np.zeros(len(jet_data_back))

      labels = np.concatenate((labels_sig, labels_bag))
      jet_data_new = np.concatenate((jet_data_sig, jet_data_back))
      particle_data_new = np.concatenate((particle_data_sig, particle_data_back))

      # shuffle data
      jet_data_new, particle_data_new, labels = shuffle(jet_data_new, particle_data_new, labels, random_state=42)

      new_particle_data[s]['data'] = pd.DataFrame(particle_data_new.reshape(-1,150*4))
      #new_particle_data[s]['labels'] = labels
      
      new_jet_data[s]['data'] = pd.DataFrame(jet_data_new,columns=['pt','eta','mass','num_particles'])
      new_jet_data[s]['labels'] = pd.DataFrame(labels,columns=['labels'])

# store data in hdf5 file    
for s in new_split:
      new_jet_data[s]['data'].to_hdf('data/jetnet_data_1.h5', key=f'jet_data_{s}', mode='a',format='table')  
      new_particle_data[s]['data'].to_hdf('data/jetnet_data_1.h5', key=f'particle_data_{s}', mode='a',format='table')
      new_jet_data[s]['labels'].to_hdf('data/jetnet_data_1.h5', key=f'labels_{s}', mode='a',format = 'table')




