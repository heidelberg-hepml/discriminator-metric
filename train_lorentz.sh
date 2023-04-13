#!/bin/sh


# This script is used to train the Lorentz model on the JetNet dataset

for trial in 2
do
        echo "Trial $trial"
        python -m torch.distributed.launch --nproc_per_node=2 \
                train_lorentz.py --batch_size=32 --epochs=2  --warmup_epochs=5 \
                --seed $((trial*2)) --n_layers=6 --n_hidden=72 --lr=0.001 \
                --c_weight=0.001 --dropout=0.2 --weight_decay=0.01 \
                --exp_name=test_${trial} --datadir=data --logdir=output \
                --wandb_project discr-metric \
                --wandb_group test \
                --wandb_job_type lorentz 

done