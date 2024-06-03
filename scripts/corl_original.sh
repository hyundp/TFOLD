#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        echo "Running corl_original script: $env-$ds"
        python corl/algorithms/td3_bc.py --env_name $env --dataset $ds
    done
done
