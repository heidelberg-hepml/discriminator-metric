#!/bin/sh

# This script is used to train ParticleNet on the converted data.
# The converted data is stored in data/converted/ and is in the format of .awkd files.
# The .awkd files are created by notebooks/convert_dataset.ipynb 
declare -a test_jets=('shifted' 'eta_smeared' 'smeared' 'pt_shifted' 'pt_smeared' 'all_smeared')

#declare -a test_jets=('shifted' 'eta_smeared')

for test_jet in "${test_jets[@]}"
do
    for trial in 1
    do
        echo "Trial $test_jet $trial"
        python -m train_particlenet --gpu 2 --train_dir data/converted/train_${test_jet}_file.awkd \
            --val_dir data/converted/valid_${test_jet}_file.awkd --model_type particle_net \
            --epochs 20 --exp_name trial_${trial}_pn_${test_jet} \
            --wandb_project discr-metric \
            --wandb_group ${test_jet} \
            --wandb_job_type particlenet_jetnet30_rel_cartesian 
    done
done