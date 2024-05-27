#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
batch_size=(32 64)
max_iter=600

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        for bs in "${batch_size[@]}"; do
            echo "Running train script: $env-$ds batch_size=$bs max_iter=$max_iter"
            python transformer/gpt_transformer/train/train_transFOLD.py --env_name $env --dataset $ds --batch_size $bs --max_train_iters $max_iter
        done
    done
done