#!/bin/bash

env_name=("hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
percentage=(0.1)

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        for p in "${percentage[@]}"; do
            echo "Running MLP filtering script: $env-$ds"
            python mlp/filtering_mlp.py --env_name $env --dataset $ds --percentage $p
        done
    done
done
