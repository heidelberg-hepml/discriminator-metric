#!/bin/sh


# This script is used to train the Lorentz model on the JetNet dataset
declare -a test_jets=('truth')

#declare -a test_jets=('shifted' 'eta_smeared' 'smeared' 'pt_shifted' 'pt_smeared' 'all_smeared')
#declare -a test_jets=('shifted' 'eta_smeared')

for test_jet in "${test_jets[@]}"
do
        for trial in 1 2 3 4 5
        do
                echo "Trial $test_jet $trial"
                python -m torch.distributed.launch --nproc_per_node=2 \
                        train_lorentz.py --batch_size=32 --epochs=40  --warmup_epochs=5 \
                        --seed $((trial*2)) --n_layers=6 --n_hidden=72 --lr=0.001 \
                        --c_weight=0.001 --dropout=0.2 --weight_decay=0.01 \
                        --exp_name=trial_${trial}_ln_${test_jet} --datadir=data --logdir=output \
                        --test_jet=${test_jet} \
                        --wandb_project discr-metric \
                        --wandb_group ${test_jet} \
                        --wandb_job_type lorentz_jetnet30_rel_cartesian

                python -m torch.distributed.launch --nproc_per_node=1 \
                        train_lorentz.py --batch_size=32 --epochs=40  --warmup_epochs=5 \
                        --seed $((trial*2)) --n_layers=6 --n_hidden=72 --lr=0.001 \
                        --c_weight=0.001 --dropout=0.2 --weight_decay=0.01 \
                        --exp_name=trial_${trial}_ln_${test_jet} --datadir=data --logdir=output \
                        --test_mode \
                        --test_jet=${test_jet} \
                        --wandb_project discr-metric \
                        --wandb_group ${test_jet} \
                        --wandb_job_type test_lorentz_jetnet30_rel_cartesian

        done
done