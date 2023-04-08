#!/bin/sh

# This script is used to train ParticleNet on the converted data.
# The converted data is stored in data/converted/ and is in the format of .awkd files.
# The .awkd files are created by notebooks/convert_dataset.ipynb 


for trial in 1 2 3 4 5 6 7 8 9 10
do
    echo "Trial $trial"
    python -m train_particlenet --gpu 1 --train_dir data/converted/train_file.awkd \
        --val_dir data/converted/val_file.awkd --model_type particle_net \
        --epochs 20 --exp_name trial_${trial}_remove_tail_PN \
        --wandb_project discr-metric \
        --wandb_group remove_tail_distribution \
        --wandb_job_type particlenet 
done