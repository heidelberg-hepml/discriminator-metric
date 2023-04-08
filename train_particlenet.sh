#!/bin/sh

# This script is used to train ParticleNet on the converted data.
# The converted data is stored in data/converted/ and is in the format of .awkd files.
# The .awkd files are created by notebooks/convert_dataset.ipynb 


for trial in 1 
do
    python -m train_particlenet --gpu 1 --train_dir data/converted/train_file.awkd \
        --val_dir data/converted/val_file.awkd --model_type particle_net \
        --epochs 2 --exp_name test_${trial} \
        --wandb_project test \
        --wandb_group test_particlenet \
        --wandb_job_type particlenet 
done