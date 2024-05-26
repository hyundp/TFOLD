
import argparse
import os
import time
from typing import List

import gym
#import gymnasium as gym
import numpy as np
import pyrootutils
import torch
# import wandb
import yaml
from gym.vector import AsyncVectorEnv
from tqdm import tqdm

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


import transformer.gpt_transformer.src.envs as envs


def transfer_flatten_to_sim(state, env):

    sim = env.unwrapped.sim

    idx_time = 0
    idx_qpos = idx_time + 1
    idx_qvel = idx_qpos + sim.model.nq

    qpos = state[idx_qpos:idx_qpos + sim.model.nq]
    qvel = state[idx_qvel:idx_qvel + sim.model.nv]
    
    return qpos, qvel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--batch_num', type=int, default=6)
    parser.add_argument('--batch_id', type=int, default=0)
    parser.add_argument('--test_partial', action='store_true', default=False)
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--type', type=str, default='filtered')
    args = parser.parse_args()
    
    # wandb.init(
    #     project = args.wandb_project,
    #     entity = args.wandb_entity,
    #     config = config,
    #     name = data_name
    # )
    
    env_name = args.env_name
    dataset = args.dataset
    type = args.type
    
    DATA_PATH = 'transformer/gpt_transformer/src/data/'
    
    if env_name == 'hopper':
        env_nm = "HopperExt-v2"
    elif env_name == 'halfcheetah':
        env_nm = "HalfCheetahExt-v2"
    elif env_name == 'walker2d':
        env_nm = "Walker2dExt-v2"


    # get original dataset 
    original = np.load(f"{DATA_PATH}/original/{env_name}-{dataset}-v2.pkl", allow_pickle=True)
    obs = []
    rewards = []
    for epi in original:
        obs.append(epi["observations"])
        rewards.append(epi["rewards"])
    obs = np.concatenate(obs, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    state_mean = obs.mean(axis=0)
    state_std = obs.std(axis=0)+1e-3

    reward_mean = rewards.mean(axis=0)
    reward_std = rewards.std(axis=0)+1e-3


    # get augmented or filtered dataset
    samples = np.load(f'{DATA_PATH}/{type}/{env_name}-{dataset}-v2.npz', allow_pickle=True)

    data = samples["data"]
    config = samples["config"]

    dynamic_mse = []
    reward_mse = []
    real_rewards = []
    gen_rewards = []

    if args.test_partial:
        # np.random.seed(1)
        # np.random.shuffle(data)
        # data = data[:len(data)//5]
        if args.percentage > 0:
            crit = len(data)//5
            reward = [d["rewards"].sum() for d in data]
            idx = np.argsort(reward)[::-1]
            len_ = int(len(data)*args.percentage)
            data = data[idx[:len_]]

            if len(data) > crit:
                np.random.seed(1)
                np.random.shuffle(data)
                data = data[:crit]
        else:
            print("test randomly")
            np.random.seed(1)
            np.random.shuffle(data)
            data = data[:len(data)//10]

            


    boundaries = np.linspace(0, len(data), args.batch_num+1)
    ranges = []
    for i in range(args.batch_num):
        ranges.append(np.arange(boundaries[i], boundaries[i+1], dtype=int))
    epis = ranges[args.batch_id]

    print(f"batch {args.batch_id} has {len(epis)} episodes")
    for epi in tqdm(epis):
        for timestep in range(data[epi]["observations"].shape[0]):
            env = gym.make(env_nm)
            state = data[epi]["observations"][timestep]
            action = data[epi]["actions"][timestep]
            next_state = data[epi]["next_observations"][timestep]
            env.reset(state = state)
            real_obs, real_reward, done, _ = env.step(action)
            real_obs = (real_obs-state_mean)/state_std
            real_reward = (real_reward-reward_mean)/reward_std
            next_state = (next_state-state_mean)/state_std
            rew = data[epi]["rewards"][timestep]
            rew = (rew-reward_mean)/reward_std

            mse = np.square(real_obs-next_state).mean()
            rewardmse = np.square(real_reward-rew).mean()
            
            real_reward = real_reward*reward_std+reward_mean
            reward_mse.append(rewardmse)
            dynamic_mse.append(mse)
            real_rewards.append(real_reward)
            gen_rewards.append(data[epi]["rewards"][timestep])

            env.close()
    
    dynamic_mse = np.array(dynamic_mse)
    reward_mse = np.array(reward_mse)
    real_rewards = np.array(real_rewards)
    gen_rewards = np.array(gen_rewards)
    

    resfolder = f"{DATA_PATH}/statistics/{type}/"
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)

    np.savez(f"{resfolder}/{env_name}-{dataset}-v2.npz", 
             dynamic_mse=dynamic_mse, 
             reward_mse=reward_mse, 
             real_rewards=real_rewards,
             gen_rewards=gen_rewards,
             config = config)

