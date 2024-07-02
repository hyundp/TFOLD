#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
type=('filtered')

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        echo "Running MLP statistics script: $env-$ds"
        python mlp/statistics_mlp.py --env_name $env --dataset $ds
    done
done