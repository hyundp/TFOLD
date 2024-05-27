# import
import argparse
import os
import sys

import gym
import numpy as np
import pandas as pd
import pyrootutils
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from transformer.gpt_transformer.src.model import DecisionTransformer
from transformer.gpt_transformer.src.utils import D4RLTrajectoryDataset


def filter(args):
    
    env_name = args.env_name
    dataset = args.dataset
    
    # train parameter
    embed_dim = args.embed_dim
    activation = args.activation
    drop_out = args.drop_out
    k = args.k # content len
    n_blocks = args.n_blocks
    n_heads = args.n_heads # transformer head

    # total updates = max_train_iters x num_updates_per_iter


    if env_name == 'hopper':
        env = gym.make('Hopper-v2')

    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v2')

    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v2')



    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')


    # check dim
    # used when determining the model shape
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    BEST_MODEL_PATH = f"transformer/gpt_transformer/src/best_model/{env_name}-{dataset}_model_best.pt"

    # load augmented data
    AUG_DATA_PATH = f'transformer/gpt_transformer/src/data/augmented/{env_name}-{dataset}-v2.npz'
    FILTERED_DATA_PATH = f'transformer/gpt_transformer/src/data/filtered/{env_name}-{dataset}-v2.npz'

    best_model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=k,
                n_heads=n_heads,
                drop_p=drop_out,
            ).to(DEVICE)
    

    # load checkpoint
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    best_model.eval()


    # load augmented dataset
    aug_dataset_sample = np.load(AUG_DATA_PATH, allow_pickle=True)
    aug_dataset_sample = aug_dataset_sample['data']
    
    
    
    
    # mod
    # calculate mean, std

    aug_states, aug_next_states, aug_rewards = [], [], []
    for traj in aug_dataset_sample:
        aug_states.append(traj['observations'])
        aug_next_states.append(traj['next_observations'])
        aug_rewards.append(traj['rewards'])
        
    # used for input, output normalization
    aug_states = np.concatenate(aug_states, axis=0)
    # print("state shape: ", states.shape)
    aug_state_mean, aug_state_std = np.mean(aug_states, axis=0), np.std(aug_states, axis=0)

    aug_next_states = np.concatenate(aug_next_states, axis=0)
    aug_next_state_mean, aug_next_state_std = np.mean(aug_next_states, axis=0), np.std(aug_next_states, axis=0)

    aug_rewards = np.concatenate(aug_rewards, axis=0)
    aug_reward_mean, aug_reward_std = np.mean(aug_rewards, axis=0), np.std(aug_rewards, axis=0)
    

    Percentage = args.percentage

    def filtering_transformer(augmented_dataset_sample, model, Percentage=Percentage):
        
        filtered_dataset = pd.DataFrame(columns = ['states', 'next_states', 'actions', 'rewards', 'timesteps', 'traj_mask', 'mse'])
        
        states_list, next_states_list, actions_list, rewards_list, timesteps_list, traj_mask_list, mse_list = [], [], [], [], [], [], []
        
        aug_dataset = D4RLTrajectoryDataset(augmented_dataset_sample, k, not_path=True)

        aug_data_loader = DataLoader(aug_dataset,
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)
                                
        for timesteps, states, next_states, actions, rewards, traj_mask in tqdm(aug_data_loader):
            
            states_list.append(np.array(states.reshape(k, state_dim)))
            next_states_list.append(np.array(next_states.reshape(k, state_dim)))
            actions_list.append(np.array(actions.reshape(k, act_dim)))
            rewards_list.append(np.array(rewards.reshape(-1,)))
            timesteps_list.append(np.array(np.squeeze(timesteps, axis=0)))
            traj_mask_list.append(np.array(np.squeeze(traj_mask, axis=0)))

            # normalization
            states = (states - aug_state_mean) / aug_state_std
            next_states = (next_states - aug_next_state_mean) / aug_next_state_std
            rewards = (rewards - aug_reward_mean) / aug_reward_std
            
            timesteps = timesteps.to(DEVICE)	# B x T
            states = states.to(DEVICE)			# B x T x state_dim
            next_states = next_states.to(DEVICE) # B X T X state_dim
            actions = actions.to(DEVICE)		# B x T x act_dim
            rewards = rewards.to(DEVICE).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(DEVICE)	# B x T
        
            pred_next_states = torch.clone(next_states).detach().to(DEVICE)
            pred_rewards = torch.clone(rewards).detach().to(DEVICE)
        
            real_next_state, real_rewards = model.forward(
                                                            rewards=rewards,
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                        )
            pred_next_states = pred_next_states.view(-1, state_dim)[traj_mask.view(-1,) > 0]
            real_next_state = real_next_state.view(-1, state_dim)[traj_mask.view(-1,) > 0]

            pred_rewards = pred_rewards.view(-1, 1)[traj_mask.view(-1,) > 0]
            real_rewards = real_rewards.view(-1, 1)[traj_mask.view(-1,) > 0]

            state_loss = F.mse_loss(pred_next_states, real_next_state, reduction='mean')
            reward_loss = F.mse_loss(pred_rewards, real_rewards, reduction='mean')
            
            total_loss = state_loss.add(reward_loss)
            total_loss = torch.mean(total_loss)
            mse_list.append(total_loss.detach().cpu().item())
            
                                                        
        filtered_dataset['states'] = states_list
        filtered_dataset['next_states'] = next_states_list
        filtered_dataset['actions'] = actions_list
        filtered_dataset['rewards'] = rewards_list
        filtered_dataset['timesteps'] = timesteps_list
        filtered_dataset['traj_mask'] = traj_mask_list
        filtered_dataset['mse'] = mse_list
        
        filtered_dataset.sort_values(by='mse', ascending=True, inplace=True)
        
        print("# of augmented dataset: ", len(filtered_dataset))
        keep_rows = int(len(filtered_dataset) * (1-Percentage))
        
        filtered_dataset = filtered_dataset.head(keep_rows)
        filtered_dataset = filtered_dataset.sample(frac=1).reset_index(drop=True)
        print("# of filtered dataset: ", len(filtered_dataset))
        
        # dataframe to numpy array with dict
        np_filtered_dataset = []
        
        for i in range(len(filtered_dataset)):
            np_filtered_dataset.append({'observations': np.array(filtered_dataset['states'][i]), 
                                        'next_observations': np.array(filtered_dataset['next_states'][i]),
                                        'actions': np.array(filtered_dataset['actions'][i]),
                                        'rewards': np.array(filtered_dataset['rewards'][i]),
                                        'timesteps': np.array(filtered_dataset['timesteps'][i]),
                                        'traj_mask': np.array(filtered_dataset['traj_mask'][i]),
                                        'mse': np.array(filtered_dataset['mse'][i]),
                                        })
            
        
        return np_filtered_dataset


    # Percentage = 0.1 # 0.1 ~ 1
    filtered_dataset = filtering_transformer(aug_dataset_sample, best_model, Percentage=Percentage)


    # save filtered dataset -> .npz
    temp_array = np.array([1,2,])
    np.savez(FILTERED_DATA_PATH, data=filtered_dataset, config=temp_array)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type= str, default = 'halfcheetah')
    parser.add_argument('--dataset', type= str, default = 'medium')
    
    parser.add_argument('--embed_dim', type=int, default = 128)
    parser.add_argument('--activation', type= str, default = 'relu')
    parser.add_argument('--drop_out', type= float, default = 0.1)
    parser.add_argument('--k', type= int, default = 31)
    parser.add_argument('--n_blocks', type= int, default = 3)
    parser.add_argument('--n_heads', type= int, default = 1)
    
    parser.add_argument('--percentage', type= float, default = 0.1)
    

    args = parser.parse_args()

    # wandb.config.update(HYPERPARAMS)
    
    filter(args)



