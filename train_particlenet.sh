#!/bin/sh

# This script is used to train ParticleNet on the converted data.
# The converted data is stored in data/converted/ and is in the format of .awkd files.
# The .awkd files are created by notebooks/convert_dataset.ipynb 


for trial in 1 2 3 4 5
do
    echo "Trial $trial"
    python -m train_particlenet --gpu 2 --train_dir data/converted/train_file.awkd \
        --val_dir data/converted/valid_file.awkd --model_type particle_net \
        --epochs 20 --exp_name test_${trial}_cartesian_particlenet \
        --wandb_project discr-metric \
        --wandb_group remove_tail_distribution \
        --wandb_job_type particlenet_jetnet30_rel_cartesian 
done