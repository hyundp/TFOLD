# TFOLD: Transformer Filtering for Offline Reinforcement Learning Dynamics

Official Code for "TFOLD: Transformer Filtering for Offline Reinforcement Learning Dynamics"

## Install Dependencies
To install dependecies, please run the command ```pip install -r requirement.txt.```

## Train TFOLD model
To train TFOLD model, please run the following command
```
python transformer/gpt_transformer/train/train_transFOLD.py --env_name "<env_name>" --dataset "<dataset>" --batch_size <batch_size> --max_train_iters <max_iter>
```

## Filtering dataset
To filtering augmented dataset, please run the following command
```
python transformer/gpt_transformer/filtering/filtering_transFOLD.py --env_name "<env_name>" --dataset "<dataset>" --percentage <filtering_percentage>
```

## Train Offline RL Algorithm
To train offline RL algorithms with filtered dataset, please run the following command
```
python corl/algorithms/td3_bc.py --env_name "<env_name>" --dataset "<dataset>" --seed <seed> --filtered
```
