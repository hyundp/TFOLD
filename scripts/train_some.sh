#!/bin/bash

env_name=("halfcheetah" "hopper" "walker2d")
dataset=('medium' 'medium-replay' 'medium-expert')

python transformer/gpt_transformer/train/train_transFOLD.py --env_name 'halfcheetah' --dataset 'medium' --batch_size 128 --max_train_iters 600
python transformer/gpt_transformer/train/train_transFOLD.py --env_name 'halfcheetah' --dataset 'medium-replay' --batch_size 128 --max_train_iters 600
python transformer/gpt_transformer/train/train_transFOLD.py --env_name 'walker2d' --dataset 'medium' --batch_size 128 --max_train_iters 600
