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
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])



# Prepare model model saving directory.
save_dir = f'output/{args.exp_name}'

model_name = '%s_model.{epoch}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  
model_path = f'{save_dir}/{model_name}'


# Prepare callbacks for model saving and for learning rate adjustment.


lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()
callbacks = [epoch_save(model_path), lr_scheduler, progress_bar]

train_dataset.shuffle()
history = model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
#           epochs=epochs,
          epochs=args.epochs, # --- train only for 1 epoch here for demonstration ---
          validation_data=(val_dataset.X, val_dataset.y),
          shuffle=True,
          callbacks=callbacks)



# Evaluate to find the best AUC model
best_auc = 0
for epoch in range(args.epochs):
  model.load_weights(f'{save_dir}/{model_type}_model.{epoch}.h5')
  ypred = model.predict(val_dataset.X)
  auc = roc_auc_score(val_dataset.y[:,1], ypred[:,1])
  wandb.log({'val_auc': auc, 'epoch': epoch})
  if auc > best_auc:
    best_auc = auc
    model.save(f'{save_dir}/best_model.h5')
    np.save(f'{save_dir}/best_model_score.npy', ypred)

wandb.log({'best_val_auc': best_auc})


wandb.finish()
