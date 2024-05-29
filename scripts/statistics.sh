#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')
type=('filtered' 'augmented')

for env in "${env_name[@]}"; do
    for ds in "${dataset[@]}"; do
        for t in "${type[@]}"; do
            echo "Running statistics script: $t-$env-$ds"
            python transformer/gpt_transformer/statistics/statistics_transFOLD.py --env_name $env --dataset $ds --type $t
        done
    done
done