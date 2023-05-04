import numpy as np
import awkward0
from tensorflow import keras
import tensorflow as tf
import os
import argparse
from scripts.particlenet_utils import *
from scripts.particlenet_models import get_particle_net, get_particle_net_lite
import wandb
from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_dir',default='data/converted/train_file.awkd', help='Train file path [default: data/converted/train_file.awkd]')
parser.add_argument('--val_dir',default='data/converted/val_file.awkd', help='Val file path [default: data/converted/val_file.awkd]')
parser.add_argument('--model_type', default='particle_net_lite', help='Model type [default: particle_net_lite]')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train [default: 100]')
parser.add_argument('--exp_name', default='test_particlenet', help='Experiment name [default: particle_net_lite]')
parser.add_argument('--wandb_project', default='discr-metric', help='Wandb project name [default: discr-metric]')
parser.add_argument('--wandb_group', default='test_particlenet', help='Wandb group name [default: test_particlenet]')
parser.add_argument('--wandb_job_type', default='particlenet', help='Wandb job type [default: train]')
args = parser.parse_args()

# Fix gpu memory growth
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print(args)

# Load data

train_dataset = Dataset(args.train_dir, data_format='channel_last')
val_dataset = Dataset(args.val_dir, data_format='channel_last')


# Define model

model_type = args.model_type # choose between 'particle_net' and 'particle_net_lite'
num_classes = train_dataset.y.shape[1]
input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)

# Training parameters
batch_size = 1024 if 'lite' in model_type else 384
epochs = args.epochs

# initiate wandb
wandb.init(project=args.wandb_project, 
          group=args.wandb_group,
          #group='remove_tail_distribution', 
          job_type=args.wandb_job_type, config=args)
wandb.run.name = f'{args.exp_name}'

# optimizer
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=4, min_lr=0.000001)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy',
         #     optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
         #     metrics=['accuracy'])



# Prepare  model saving directory.
save_dir = f'output/{args.exp_name}'

model_name = '%s_model.{epoch}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  
model_path = f'{save_dir}/{model_name}'


# Prepare callbacks for model saving and for learning rate adjustment.


#lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()
callbacks = [epoch_save(model_path), reduce_lr, progress_bar]

train_dataset.shuffle()
history = model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
#           epochs=epochs,
          epochs=args.epochs, # --- train only for 1 epoch here for demonstration ---
          validation_data=(val_dataset.X, val_dataset.y),
          shuffle=True,
          callbacks=callbacks)



# Evaluate to find the best AUC model
loss_train_list = []
loss_val_list = []
auc_val_list = []
auc_train_list = []


best_loss = 10000
best_auc = 0
for epoch in range(args.epochs):
  model.load_weights(f'{save_dir}/{model_type}_model.{epoch}.h5')
  ypred_val = model.predict(val_dataset.X)
  ypred_train = model.predict(train_dataset.X)

  cce = tf.keras.losses.CategoricalCrossentropy()
  loss_val = cce(val_dataset.y, ypred_val).numpy()

  cce = tf.keras.losses.CategoricalCrossentropy()
  loss_train = cce(train_dataset.y, ypred_train).numpy()

  np.save(f'{save_dir}/score_val_{epoch}.npy', ypred_val)
  np.save(f'{save_dir}/score_train_{epoch}.npy', ypred_train)

  loss_train_list.append(loss_train)
  loss_val_list.append(loss_val)

  auc_val = roc_auc_score(val_dataset.y[:,0], ypred_val[:,0])
  auc_train = roc_auc_score(train_dataset.y[:,0], ypred_train[:,0])

  wandb.log({'loss_train': loss_train})
  wandb.log({'loss_val': loss_val})
  wandb.log({'auc_train': auc_train})
  wandb.log({'auc_val': auc_val})


  auc_train_list.append(auc_train)
  auc_val_list.append(auc_val)

  if loss_val < best_loss:
    best_loss_epoch = epoch
    best_loss = loss_val
    np.save(f'{save_dir}/best_model_val_loss.npy', ypred_val)

  if auc_val > best_auc:
    best_auc_epoch = epoch
    best_auc = auc_val
    np.save(f'{save_dir}/best_model_val_auc.npy', ypred_val)


#wandb.log({'best_val_auc': best_auc})
#wandb.log({'best_val_loss': best_loss})



# Save loss history
np.save(f'{save_dir}/loss_train.npy', np.array(loss_train_list))
np.save(f'{save_dir}/loss_val.npy', np.array(loss_val_list))

# Save AUC history
np.save(f'{save_dir}/auc_train.npy', np.array(auc_train_list))
np.save(f'{save_dir}/auc_val.npy', np.array(auc_val_list))


wandb.log({'best_val_auc': best_auc})


wandb.finish()
