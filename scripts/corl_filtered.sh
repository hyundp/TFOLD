#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
seed=(0 1 2)

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        for s in "${seed[@]}"; do
            echo "Running corl_filtered script: $env-$ds"
            python corl/algorithms/td3_bc.py --env_name $env --dataset $ds --seed $s --filtered
        done
    done
done
