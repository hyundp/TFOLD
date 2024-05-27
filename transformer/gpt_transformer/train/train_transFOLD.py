# import library
import argparse
import collections
import csv
import os
import pickle
import sys
from datetime import datetime

import d4rl
import gym
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# import wandb


path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from transformer.gpt_transformer.src.model import DecisionTransformer
from transformer.gpt_transformer.src.utils import (D4RLTrajectoryDataset,
                                                   check_batch, make_dir)

# wandb.init(project='train_transFOLD') #wandb 사이트 Home에 들어가보면, 좌측에 "My Projects"에 프로젝트를 생성한다는 의미
# wandb.run.name = 'training_loss' #실행 이름 설정
# wandb.run.save()


# Define training method
def train(args):
    
    env_name = args.env_name
    dataset = args.dataset
    
    # train parameter
    batch_size = args.batch_size
    embed_dim = args.embed_dim
    activation = args.activation
    drop_out = args.drop_out
    k = args.k # content len
    n_blocks = args.n_blocks
    n_heads = args.n_heads # transformer head

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter
    total_updates = args.total_updates
    min_total_log_loss = args.min_total_log_loss

    wt_decay = args.wt_decay             # weight decay
    lr = args.lr                  # learning rate
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # weight of mse loss
    state_weight = args.state_weight
    reward_weight = args.reward_weight

    # evaluation parameter
    # max_eval_ep_len = 1000      # max len of one evaluation episode
    # num_eval_ep = 10            # num of evaluation episodes per iteration
    
    if env_name == 'hopper':
        env = gym.make('Hopper-v2')

    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v2')

    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v2')

        
        
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    TRAIN_DATA_PATH = f'transformer/gpt_transformer/src/data/train/{env_name}-{dataset}-v2.pkl'
    VAL_DATA_PATH = f'transformer/gpt_transformer/src/data/val/{env_name}-{dataset}-v2.pkl'
    # ORIGINAL_DATA_PATH = f'transformer/gpt_transformer/src/data/original/{args.env_name}-{args.dataset}-v2.pkl'
    LOG_PATH = "transformer/gpt_transformer/src/log/"
    make_dir(LOG_PATH)
    BEST_MODEL_PATH = "transformer/gpt_transformer/src/best_model/"
    make_dir(BEST_MODEL_PATH)


    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
        

    # load validate preprocessing(normalization, fit padding) data

    val_traj_dataset = D4RLTrajectoryDataset(TRAIN_DATA_PATH, k, val=True, val_dataset_path=VAL_DATA_PATH)
    batch_size = check_batch(batch_size, len(val_traj_dataset))

    # load train preprocessing(normalization, fit padding) data

    train_traj_dataset = D4RLTrajectoryDataset(TRAIN_DATA_PATH, k)
    train_traj_data_loader = DataLoader(train_traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
                            
    train_data_iter = iter(train_traj_data_loader)

    ## get state stats from dataset
    state_mean, state_std = train_traj_dataset.get_state_stats()

    # define model

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=k,
                n_heads=n_heads,
                drop_p=drop_out,
            ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
                        model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/args.warmup_steps, 1)
        )
    
    

    
    start_time = datetime.now().replace(microsecond=0)

    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    
    prefix = f"{env_name}-{dataset}"

    save_model_name =  f'{prefix}_model.pt'
    save_best_model_name = f'{prefix}_model_best.pt'
    save_model_path = os.path.join(LOG_PATH, save_model_name)
    save_best_model_path = os.path.join(BEST_MODEL_PATH, save_best_model_name)

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(LOG_PATH, log_csv_name)


    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "total_loss", "state_loss", "reward_loss", "val_total_loss", "val_state_loss", "val_reward_loss"])

    csv_writer.writerow(csv_header)


    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(DEVICE))
    print("dataset: " + prefix)
    print("batch_size: " + str(batch_size))
    print("best model save path: " + save_best_model_path)
    print("log csv save path: " + log_csv_path)

    # train
    for i_train_iter in tqdm(range(max_train_iters)):


        log_state_losses, log_reward_losses, log_total_losses = [], [], []
        val_log_state_losses, val_log_reward_losses, val_log_total_losses = [], [], []
        model.train()
        
        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, next_states, actions, rewards, traj_mask = next(train_data_iter)
            except StopIteration:
                train_traj_data_loader = DataLoader(train_traj_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)
                                        
                train_data_iter = iter(train_traj_data_loader)
                timesteps, states, next_states, actions, rewards, traj_mask = next(train_data_iter)

            timesteps = timesteps.to(DEVICE)	# B x T
            states = states.to(DEVICE)			# B x T x state_dim
            next_states = next_states.to(DEVICE) # B X T X state_dim
            actions = actions.to(DEVICE)		# B x T x act_dim
            rewards = rewards.to(DEVICE).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(DEVICE)	# B x T

            next_states_target = torch.clone(next_states).detach().to(DEVICE)
            rewards_target = torch.clone(rewards).detach().to(DEVICE)
        
            next_state_preds, rewards_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            rewards=rewards,
                                                        )

            # only consider non padded elements
            next_state_preds = next_state_preds.view(-1, state_dim)[traj_mask.view(-1,) > 0]
            next_states_target = next_states_target.view(-1, state_dim)[traj_mask.view(-1,) > 0]
            
            rewards_preds = rewards_preds.view(-1, 1)[traj_mask.view(-1,) > 0]
            rewards_target = rewards_target.view(-1, 1)[traj_mask.view(-1,) > 0]

            state_loss = F.mse_loss(next_state_preds, next_states_target, reduction='mean') * state_weight
            reward_loss = F.mse_loss(rewards_preds, rewards_target, reduction='mean') * reward_weight
            
            total_loss = state_loss.add(reward_loss)
            total_loss = torch.mean(total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            
            #save loss
            log_state_losses.append(state_loss.detach().cpu().item())
            log_reward_losses.append(reward_loss.detach().cpu().item())
            log_total_losses.append(total_loss.detach().cpu().item())
            
        
            
        # validation
        model.eval()
        val_traj_data_loader = DataLoader(val_traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
        for val_timesteps, val_states, val_next_states, val_actions, val_rewards, val_traj_mask in val_traj_data_loader:
            
            val_timesteps = val_timesteps.to(DEVICE)	# B x T
            val_states = val_states.to(DEVICE)			# B x T x state_dim
            val_next_states = val_next_states.to(DEVICE) # B X T X state_dim
            val_actions = val_actions.to(DEVICE)		# B x T x act_dim
            val_rewards = val_rewards.to(DEVICE).unsqueeze(dim=-1) # B x T x 1
            val_traj_mask = val_traj_mask.to(DEVICE)	# B x T
                    
            val_next_states_target = torch.clone(val_next_states).detach().to(DEVICE)
            val_rewards_target = torch.clone(val_rewards).detach().to(DEVICE)
            
            val_next_state_preds, val_rewards_preds = model.forward(
                                                            timesteps=val_timesteps,
                                                            states=val_states,
                                                            actions=val_actions,
                                                            rewards=val_rewards,
                                                        )
                                                        
            # only consider non padded elements
            val_next_state_preds = val_next_state_preds.view(-1, state_dim)[val_traj_mask.view(-1,) > 0]
            val_next_states_target = val_next_states_target.view(-1, state_dim)[val_traj_mask.view(-1,) > 0]
            
            val_rewards_preds = val_rewards_preds.view(-1, 1)[val_traj_mask.view(-1,) > 0]
            val_rewards_target = val_rewards_target.view(-1, 1)[val_traj_mask.view(-1,) > 0]

            val_state_loss = F.mse_loss(val_next_state_preds, val_next_states_target, reduction='mean') * state_weight
            val_reward_loss = F.mse_loss(val_rewards_preds, val_rewards_target, reduction='mean') * reward_weight

            # todo: try to use mae
            
            val_total_loss = val_state_loss.add(val_reward_loss)
            val_total_loss = torch.mean(val_total_loss)
            
            # save val loss
            val_log_state_losses.append(val_state_loss.detach().cpu().item())
            val_log_reward_losses.append(val_reward_loss.detach().cpu().item())
            val_log_total_losses.append(val_total_loss.detach().cpu().item())
            
            
        
        mean_total_log_loss = np.mean(log_total_losses)
        mean_state_log_loss = np.mean(log_state_losses)
        mean_reward_log_loss = np.mean(log_reward_losses)
        
        mean_val_total_log_loss = np.mean(val_log_total_losses)
        mean_val_state_log_loss = np.mean(val_log_state_losses)
        mean_val_reward_log_loss = np.mean(val_log_reward_losses)
        
        # wandb logging
        # wandb.log({'mean_total_log_loss':mean_total_log_loss, 'mean_val_total_log_loss':mean_val_total_log_loss})

        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "train total loss: " + format(mean_total_log_loss, ".5f") + '\n' +
                "train state loss: " + format(mean_state_log_loss, ".5f") + '\n' +
                "train reward loss: " +  format(mean_reward_log_loss, ".5f") + '\n' +
                "val total loss: " + format(mean_val_total_log_loss, ".5f") + '\n' +
                "val state loss: " + format(mean_val_state_log_loss, ".5f") + '\n' +
                "val reward loss: " +  format(mean_val_reward_log_loss, ".5f")
                )

        print(log_str)

        log_data = [time_elapsed, total_updates, mean_total_log_loss, mean_state_log_loss, mean_reward_log_loss,
            mean_val_total_log_loss, mean_val_state_log_loss, mean_val_reward_log_loss]

        csv_writer.writerow(log_data)
        
        # save model
        if mean_val_total_log_loss <= min_total_log_loss:
            print("saving min loss model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            min_total_log_loss = mean_val_total_log_loss

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved min loss model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type= str, default = 'halfcheetah')
    parser.add_argument('--dataset', type= str, default = 'medium')
    
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--embed_dim', type=int, default = 128)
    parser.add_argument('--activation', type= str, default = 'relu')
    parser.add_argument('--drop_out', type= float, default = 0.1)
    parser.add_argument('--k', type= int, default = 31)
    parser.add_argument('--n_blocks', type= int, default = 3)
    parser.add_argument('--n_heads', type= int, default = 1)
    
    parser.add_argument('--max_train_iters', type= int, default = 1000)
    parser.add_argument('--num_updates_per_iter', type= int, default = 100)
    parser.add_argument('--total_updates', type= int, default = 0)
    parser.add_argument('--min_total_log_loss', type= float, default = 1e10)
    
    parser.add_argument('--wt_decay', type= float, default = 1e-4)
    parser.add_argument('--lr', type= float, default = 1e-4)
    parser.add_argument('--warmup_steps', type= int, default = 10000)
    
    parser.add_argument('--state_weight', type= int, default = 1)
    parser.add_argument('--reward_weight', type= int, default = 1)

    args = parser.parse_args()
    
    # HYPERPARAMS = {
    #     'batch_size': args.batch_size,
    #     'max_train_iters' : args.max_train_iters
    # }

    # wandb.config.update(HYPERPARAMS)
    
    train(args)