#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
percentage=(0.25 0.5 0.75)

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        for p in "${percentage[@]}"; do
            echo "Running filtering script: $env-$ds"
            python transformer/gpt_transformer/filtering/filtering_transFOLD.py --env_name $env --dataset $ds --percentage $p
        done
    done
done
