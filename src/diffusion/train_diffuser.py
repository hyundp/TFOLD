# Train diffusion model on D4RL transitions.
import argparse
import math
import os
import pickle
import random
from datetime import datetime
from math import gcd
from typing import List, Optional, Union

import gin
import gym
import numpy as np
import pyrootutils
import torch
import tqdm
import wandb
from accelerate import PartialState
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from corl.shared.buffer import (DiffusionDataset, DiffusionTrajectoryDataset,
                                DiffusionTrajectoryFractionDataset)
from corl.shared.utils import (compute_mean_std, merge_dictionary,
                               normalize_states)
from src.data.norm import MinMaxNormalizer, normalizer_factory
from src.diffusion.elucidated_diffusion import (Trainer,
                                                define_rewardweighting_sampler,
                                                define_terminal_sampler)
from src.diffusion.utils import (construct_diffusion_model, make_inputs,
                                 split_diffusion_trajectory)
from src.dynamics.reward import RewardModel, TrainConfig


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 1000,
            modalities : List[str] = ["observations", "actions", "rewards"],
            concat_goal : bool = True
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        print(f'Clamping samples: {self.clamp_samples}')
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        self.modalities = modalities
        self.concat_goal = concat_goal
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        generated_samples = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()
            if "rewards" in self.modalities:
                obs, actions, rewards, next_obs = split_diffusion_trajectory(sampled_outputs, self.env)
                
                for b in range(self.sample_batch_size):
                    temp = {
                        "observations": obs[b,:,:],
                        "actions": actions[b,:,:],
                        "next_observations": next_obs[b,:,:],
                        "rewards": rewards[b,:,:].reshape(-1,1)
                    }
                    generated_samples.append(temp)
            else : 
                obs, actions, next_obs =  split_diffusion_trajectory(sampled_outputs, self.env)
            

                for b in range(self.sample_batch_size):
                    temp = {
                        "observations": obs[b,:,:],
                        "actions": actions[b,:,:],
                        "next_observations": next_obs[b,:,:],
                    }
                    generated_samples.append(temp)


        return generated_samples
    
    
    def prepare_sampling_data(
            self,
            batch,
            device,
            reward_scale : float = 1.0,
            guidance_rewardscale : Optional[float] = None,
            fixed_rewardscale : Optional[float] = None,
            cond_state : bool = False,
            reward_interpolation : bool = False,
            context_conditioning : str = "none",
            max_conditioning_return : Optional[float] = None,
            discounted_return : bool = False,
            on_first_state : bool = False,
    ):
        if is_td:
            states, actions, rewards, next_states, returns, time_steps, terminals, rtg, value = batch
        else:
            states, actions, rewards, next_states, returns, time_steps, terminals, rtg = batch
        
        if is_td:
            if on_first_state:
                conditioning_value = value[:,0].cpu().numpy()
            else:
                conditioning_value = value[:,0].cpu().numpy().mean(axis=-1)
                conditioning_value = conditioning_value.reshape(-1,1)

        else:
            if discounted_return:
                if on_first_state:
                    conditioning_value = returns[:,0,:].cpu().numpy()
                else:
                    conditioning_value = returns.squeeze().cpu().numpy().mean(axis=-1)
                    conditioning_value = conditioning_value.reshape(-1,1)
            else:
                conditioning_value = rewards.squeeze().cpu().numpy().mean(axis=-1)
        

        cond = np.copy(conditioning_value)


        if fixed_rewardscale is not None:
            criteria_reward = max_conditioning_return * fixed_rewardscale
            # cond = np.ones_like(cond) * criteria_reward
            cond[conditioning_value < criteria_reward] = criteria_reward
            cond[conditioning_value >= criteria_reward] = max_conditioning_return
        elif guidance_rewardscale is not None: 
            conditioning_value *= guidance_rewardscale
            cond[conditioning_value>0] = cond[conditioning_value>0] * guidance_rewardscale
            reverse_guidancescale = -guidance_rewardscale+2
            cond[conditioning_value<0] = cond[conditioning_value<0] * reverse_guidancescale
        cond = cond.reshape(-1, 1)


        B, T, state_dim = states.shape
        rewards = rewards.reshape(B, T, 1)



        if reward_interpolation:
            cond_reward = rewards.clone()
            B, T, D = cond_reward.shape
            cond_reward[cond_reward>0] = cond_reward[cond_reward>0]*guidance_rewardscale
            cond_reward[cond_reward<0] = cond_reward[cond_reward<0]*(-guidance_rewardscale+2)
            last_reward = torch.zeros((B, 1, D))
            cond_reward = torch.cat([cond_reward, last_reward], dim=1)
            
            
        else:
            cond_reward = None



        data = []
        for mod in self.modalities:
            if mod == 'observations':
                data.append(states)
            elif mod == 'actions':
                data.append(actions)
            elif mod == 'rewards':
                data.append(rewards) 
        last_state = next_states[:,-1, None,:]
        last_action = torch.zeros_like(actions[:,-1, None,:])
        last_reward = torch.zeros_like(rewards[:,-1, None,:])
        last_transition = torch.cat([last_state, last_action, last_reward], dim=-1).to(device)
            
        data = torch.cat(data, dim=-1).to(device)
        data = torch.cat([data, last_transition], dim=1)

        if cond_state:
            cond_state = data.clone()

        else:
            cond_state = None




        if context_conditioning != "none":
            conditions = {}
            q1 = (T+1)//4
            q2 = (T+1)//2
            q3 = 3*(T+1)//4

            if context_conditioning == "past":
                idx_list = [(0, q1+1)]
            if context_conditioning == "future":
                idx_list = [(q3, (T+1)+1)]
            if context_conditioning == "both":
                idx_list = [(0, q1+1), (q3, (T+1)+1)]
            for batch_idx in range(B):
                conditions[batch_idx] = {}
                for idx in idx_list:
                    conditions[batch_idx][idx] = data[batch_idx, idx[0]:idx[1], :].clone()
                    
                
        else:
            conditions = None


        cond = torch.from_numpy(cond).to(device)
        cond *= reward_scale

        return data, cond, terminals, cond_state, cond_reward, conditions


    def get_high_reward_samples(
            self, 
            dataset,
            stochastic: bool = False,
    ):
        all_idx = range(len(dataset))
        states, actions, rewards, returns, time_steps = dataset[all_idx] #outputs error
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        criteria_rewards = rewards.sum(axis = (1,2))
        

        # Option 2: Proportional to Reward
        # Code from https://github.com/siddarthk97/ddom/tree/main
        
        # Discretize bins and assign weight to each bins
        hist, bin_edges = np.histogram(criteria_rewards, bins=20)
        hist = hist / np.sum(hist)
        
        softmin_prob_unnorm = np.exp(bin_edges[1:] / 5.0)
        softmin_prob = softmin_prob_unnorm / np.sum(softmin_prob_unnorm)

        provable_dist = softmin_prob * (hist / (hist + 1e-3))
        provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)

        bin_indices = np.digitize(criteria_rewards, bin_edges[1:])
        hist_prob = hist[np.minimum(bin_indices, 19)]

        weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
        weights = np.clip(weights, a_min=0.0, a_max=5.0)

        # Select samples proportional to weights
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(states))
        data_loader = DataLoader(dataset, batch_size=self.sample_batch_size, sampler=sampler)
        return next(iter(data_loader))[0]
    


    def sample_back_and_forth(
            self,
            data_loader,
            num_samples: int,
            noise_level : float = 0.5, 
            temperature : float = 1.0,
            reward_scale : float = 1.0,
            guidance_rewardscale : Optional[float] = None,
            fixed_rewardscale : Optional[float] = None,
            device : str = "cuda",
            state_conditioning : bool = False,
            reward_interpolation : bool = False,
            context_conditioning : str = "none",
            max_conditioning_return : Optional[float] = None,
            discounted_return : bool = False,
            on_first_state : bool = False,
            is_td : bool = False,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        if num_samples % self.sample_batch_size != 0:
            num_batches += 1
        
        assert context_conditioning in ["past", "future", "both", "none"], "context_conditioning must be one of ['past', 'future', 'both', 'none']"

        generated_samples = []
        loader_iterator = iter(data_loader)
        for i in range(num_batches):
            try:
                batch = next(loader_iterator)
                if is_td:
                    states, actions, rewards, next_states, returns, time_steps, terminals, rtg, value = batch
                else:
                    states, actions, rewards, next_states, returns, time_steps, terminals, rtg = batch
            except StopIteration:
                loader_iterator = iter(data_loader)
                batch = next(loader_iterator)
                if is_td:
                    states, actions, rewards, next_states, returns, time_steps, terminals, rtg, value = batch
                else:
                    states, actions, rewards, next_states, returns, time_steps, terminals, rtg = batch

            
            state_dim = states.shape[-1]
            samples, cond, terminals, cond_state, cond_reward, cond_sar = self.prepare_sampling_data(batch,
                                                                                                    device,
                                                                                                    reward_scale,
                                                                                                    guidance_rewardscale,
                                                                                                    fixed_rewardscale,
                                                                                                    state_conditioning,
                                                                                                    reward_interpolation,
                                                                                                    context_conditioning,
                                                                                                    max_conditioning_return,
                                                                                                    discounted_return,
                                                                                                    on_first_state
                                                                                                    )

            if not state_conditioning:
                cond_state = None
                
            if not reward_interpolation:
                cond_reward = None

            if context_conditioning == "none":
                cond_sar = None
            else:
                print(f"context conditioning with {context_conditioning}")

            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample_back_and_forth(
                samples=samples,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond = cond,
                cond_state = cond_state,
                cond_reward = cond_reward,
                cond_sar = cond_sar,
                noise_level = noise_level,
                temperature = temperature,
                state_dim = state_dim
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            
            if "rewards" in self.modalities:
                gen_state, gen_action, gen_reward, gen_next_obs = split_diffusion_trajectory(
                                                                        samples = sampled_outputs, 
                                                                        env = self.env,
                                                                        modalities = self.modalities,)


                for b in range(gen_state.shape[0]):
                    
                    if self.concat_goal:
                        temp = {
                        "observations": gen_state[b,:,:-2],
                        "actions": gen_action[b,:,:],
                        "next_observations": gen_next_obs[b,:,:-2],
                        "rewards": gen_reward[b,:,:].squeeze(),
                        "returns": returns[b,:,:].squeeze().cpu().numpy(),
                        "terminals": terminals[b,:].cpu().numpy(),
                        "timesteps": time_steps[b,:].cpu().numpy(),
                        "original_observations": states[b,:,:-2].cpu().numpy(),
                        "original_actions": actions[b,:,:].cpu().numpy(),
                        "original_next_observations": next_states[b,:,:-2].cpu().numpy(),
                        "original_rewards": rewards[b,:,:].squeeze().cpu().numpy(),
                        "RTG" : rtg[b,:].squeeze().cpu().numpy(),
                        "original_goal" : states[b,:,-2:].cpu().numpy(),
                        "generated_goal" : gen_state[b,:,-2:],
                        }

                    else:
                        temp = {
                        "observations": gen_state[b,:,:],
                        "actions": gen_action[b,:,:],
                        "next_observations": gen_next_obs[b,:,:],
                        "rewards": gen_reward[b,:,:].squeeze(),
                        "returns": returns[b,:,:].squeeze().cpu().numpy(),
                        "terminals": terminals[b,:].cpu().numpy(),
                        "timesteps": time_steps[b,:].cpu().numpy(),
                        "original_observations": states[b,:,:].cpu().numpy(),
                        "original_actions": actions[b,:,:].cpu().numpy(),
                        "original_next_observations": next_states[b,:,:].cpu().numpy(),
                        "original_rewards": rewards[b,:,:].squeeze().cpu().numpy(),
                        "RTG" : rtg[b,:].squeeze().cpu().numpy(),
                        }
                        
                    generated_samples.append(temp)
            else : 
                obs, actions, next_obs =  split_diffusion_trajectory(
                                                        samples = sampled_outputs, 
                                                        env = self.env,
                                                        modalities = self.modalities,)
            

                for b in range(self.sample_batch_size):
                    temp = {
                        "observations": obs[b,:,:],
                        "actions": actions[b,:,:],
                        "next_observations": next_obs[b,:,:],
                    }
                    generated_samples.append(temp)


        return generated_samples


    def sample_back_and_forth_online(
            self,
            samples,
            cond,
            state_dim : int,
            num_samples: int,
            noise_level : float = 0.5, 
            temperature : float = 1.0,
            device : str = "cuda",
            state_conditioning : bool = False,
            max_conditioning_return : Optional[float] = None,
            discounted_return : bool = False,
    ):
        # assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        if num_samples % self.sample_batch_size != 0:
            num_batches += 1
        
        generated_samples = []
                
       
        if not state_conditioning:
            cond_state = None


        sampled_outputs = self.diffusion.sample_back_and_forth(
            samples=samples,
            num_sample_steps=self.num_sample_steps,
            clamp=self.clamp_samples,
            cond = cond,
            cond_state = cond_state,
            noise_level = noise_level,
            temperature = temperature,
            state_dim = state_dim
        )
        sampled_outputs = sampled_outputs.cpu().numpy()


        if "rewards" in self.modalities:
            gen_obs, gen_action, gen_reward, gen_next_obs = split_diffusion_trajectory(
                                                                    samples = sampled_outputs, 
                                                                    env = self.env,
                                                                    modalities = self.modalities,)
            
            # for b in range(gen_state.shape[0]):
                
            #     temp = {
            #         "observations": gen_state[b,:,:],
            #         "actions": gen_action[b,:,:],
            #         "next_observations": gen_next_obs[b,:,:],
            #         "rewards": gen_reward[b,:,:].squeeze(),
            #         # "returns": returns[b,:,:].squeeze().cpu().numpy(),
            #         # "terminals": terminals[b,:].cpu().numpy(),
            #         # "timesteps": time_steps[b,:].cpu().numpy(),
            #     }
            #     generated_samples.append(temp)
        


        return gen_obs, gen_action, gen_reward, gen_next_obs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--config_path', type=str, default='../../configs')
    parser.add_argument('--config_name', type=str, default='temporalattention_denoiser.yaml')
    parser.add_argument('--wandb_project', type=str, default="synther")
    parser.add_argument('--wandb_entity', type=str, default="gda-for-orl")
    parser.add_argument('--wandb_group', type=str, default="resmlp1")
    parser.add_argument('--datapath', type=str, default=None)
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    parser.add_argument('--back_and_forth', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument("--guidance_rewardscale", type=float, default=None) #1203 실험용
    parser.add_argument("--noise_level", type=float, default=None) #1203 실험용
    parser.add_argument("--fixed_rewardscale", type=float, default=None) #1203 실험용
    parser.add_argument("--temperature", type=float, default=None) #1203 실험용
    parser.add_argument('--fractions', type=float, default=1.0)
    parser.add_argument('--is_adaptive', type=int, default=0) # if adaptive, non-zero value
    parser.add_argument('--adaptive_idx', type=int, default=0) # max = is_adaptive-1
    args = parser.parse_args()
        
    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name)

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    env = gym.make(args.dataset)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    reward_dim = 1

    data_shape = {
        "observations": (cfg.Dataset.seq_len+1, obs_dim),
        "actions": (cfg.Dataset.seq_len+1, action_dim),
        "rewards": (cfg.Dataset.seq_len+1, reward_dim),
    }
    

    if args.datapath is None:
        with open(f'./data/{args.dataset}.pkl','rb') as f:
            dataset = pickle.load(f) 
    else:
        dataset = np.load(args.datapath, allow_pickle=True)

    if "values" in dataset[0].keys():
        is_td = True
    else:
        is_td = False

    if args.fractions < 1.0:
        random.shuffle(dataset)
        dataset = dataset[:int(len(dataset)*args.fractions)]
        args.wandb_group += f"-{int(args.fractions*100)}"
        save_shuffled_data = dataset



    if ("antmaze" in args.dataset) and "infos/goal" in cfg.Dataset.modalities:
        concat_goal = True       
        t, d = data_shape["observations"] 
        data_shape["observations"] = (t, d+2)
    else:
        concat_goal = False
        
    
    data = merge_dictionary(dataset)
    inputs = []
    for k in cfg.Dataset.modalities : 
        if k == "rewards":
            data[k] = data[k].reshape(-1,1)
        inputs.append(data[k])
    inputs = np.concatenate(inputs, axis=-1)
    inputs = torch.from_numpy(inputs).float()
    D = inputs.shape[-1]

    normalizer = normalizer_factory(cfg.Dataset.normalizer_type, inputs, skip_dims=[])


    if "episode" in cfg.Dataset:
        if cfg.Dataset.episode:
            print("load with Diffusion Dataset")
            dataset = DiffusionDataset(
                dataset,
                args.dataset,
                seq_len = cfg.Dataset.seq_len,
                discounted_return = cfg.Dataset.discounted_return,
                gamma = cfg.Dataset.gamma,
            )
        else:
            dataset = DiffusionTrajectoryDataset(
                dataset,
                args.dataset,
                seq_len = cfg.Dataset.seq_len,
                discounted_return = cfg.Dataset.discounted_return,
                gamma = cfg.Dataset.gamma,
                concat_goal = concat_goal
            )
    else:
        dataset = DiffusionTrajectoryDataset(
                            dataset,
                            args.dataset,
                            seq_len = cfg.Dataset.seq_len,
                            discounted_return = cfg.Dataset.discounted_return,
                            gamma = cfg.Dataset.gamma,
                            restore_rewards=cfg.Dataset.restore_rewards,
                            concat_goal = concat_goal
                    )
    
    if args.is_adaptive > 0:
        with open(f'./data/{args.dataset}.pkl','rb') as f:
            dataset = pickle.load(f) 
        adaptive_idx = args.adaptive_idx
        dataset = DiffusionTrajectoryFractionDataset(
                    dataset,
                    args.dataset,
                    seq_len = cfg.Dataset.seq_len,
                    discounted_return = cfg.Dataset.discounted_return,
                    gamma = cfg.Dataset.gamma,
                    restore_rewards=cfg.Dataset.restore_rewards,
                    num_frac= args.is_adaptive,
                    frac_id = adaptive_idx
                    )
        

    now = datetime.now()
    date_ = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H:%M")
    model_nm = args.config_name.split('.')[0]
    
    fname = args.dataset+"/"+model_nm+"/"+date_+"/"+time_
    
    resfolder = os.path.join(args.results_folder, fname)
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)
    
    if args.fractions < 1.0:
        np.save(os.path.join(resfolder, "train_data.npy"), save_shuffled_data)
    
    #1203실험용, override config
    if args.guidance_rewardscale is not None:
        cfg.SimpleDiffusionGenerator.guidance_rewardscale = args.guidance_rewardscale
        cfg.SimpleDiffusionGenerator.fixed_rewardscale = None
    if args.noise_level is not None:
        cfg.SimpleDiffusionGenerator.noise_level = args.noise_level
    if args.fixed_rewardscale is not None:
        cfg.SimpleDiffusionGenerator.fixed_rewardscale = args.fixed_rewardscale
        cfg.SimpleDiffusionGenerator.guidance_rewardscale = None
    if args.temperature is not None:
        cfg.SimpleDiffusionGenerator.temperature = args.temperature

    
    with open(os.path.join(resfolder, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(data_shape=data_shape,
                                          normalizer=normalizer,
                                          denoising_network=cfg.construct_diffusion_model.denoising_network,
                                          edm_config=cfg.ElucidatedDiffusion,
                                          disable_terminal_norm = cfg.construct_diffusion_model.disable_terminal_norm,
                                          cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
                                          )
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        dataset_nm = args.dataset,
        results_folder=resfolder,
        train_batch_size=cfg.Trainer.train_batch_size,
        train_lr=cfg.Trainer.train_lr,
        lr_scheduler=cfg.Trainer.lr_scheduler,
        weight_decay=cfg.Trainer.weight_decay,
        train_num_steps=cfg.Trainer.train_num_steps,
        save_and_sample_every=cfg.Trainer.save_and_sample_every,
        cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
        modalities = cfg.Dataset.modalities,
        reweighted_training = cfg.Trainer.reweighted_training,
        reward_scale = cfg.Dataset.reward_scale,
        discounted_return = cfg.Dataset.discounted_return,
        on_first_state = cfg.SimpleDiffusionGenerator.on_first_state,
        is_td = is_td,
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=resfolder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        distributed_state = PartialState()
        if trainer.accelerator.is_main_process:
            trainer.ema.to(distributed_state.device)
        # Load the last checkpoint.
        #trainer.load(milestone=trainer.train_num_steps)
        trainer.load(ckpt_path=args.ckpt_path)

    if args.load_checkpoint:
        # Generate samples and save them.
        if trainer.accelerator.is_main_process:
            if args.save_samples:
                sample_batch_size = min(len(dataset),cfg.SimpleDiffusionGenerator.sample_batch_size)
                generator = SimpleDiffusionGenerator(
                    env=env,
                    ema_model=trainer.ema.ema_model,
                    modalities=cfg.Dataset.modalities,
                    sample_batch_size=sample_batch_size,
                    concat_goal = concat_goal
                )
                if args.back_and_forth:
                    if cfg.SimpleDiffusionGenerator.weighted_sampling:
                        if "maze" in args.dataset:
                            weighted_sampler=define_rewardweighting_sampler(dataset=dataset, 
                                                                            dataset_nm=args.dataset, 
                                                                            reward_scale=cfg.Dataset.reward_scale, 
                                                                            weight_param=cfg.SimpleDiffusionGenerator.weight_param, 
                                                                            episodewise=cfg.Dataset.episode)
                        else:
                            weighted_sampler=define_rewardweighting_sampler(dataset, 
                                                                            args.dataset, 
                                                                            cfg.Dataset.reward_scale, 
                                                                            weight_param=cfg.SimpleDiffusionGenerator.weight_param,
                                                                            u = cfg.SimpleDiffusionGenerator.u,
                                                                            q = cfg.SimpleDiffusionGenerator.q,
                                                                            term_weight=cfg.SimpleDiffusionGenerator.term_weight,)
                        if cfg.SimpleDiffusionGenerator.with_strategy:
                            weighted_sampler = define_terminal_sampler(
                                                                        dataset,
                                                                        args.dataset
                                                                        )

                        print(f"sampling with weighted sampling {cfg.SimpleDiffusionGenerator.weight_param}")
                        sample_loader = DataLoader(dataset, 
                                                    batch_size=sample_batch_size,
                                                    sampler=weighted_sampler)                
                    else:
                        sample_loader = DataLoader(dataset, shuffle=True, batch_size=sample_batch_size)  


                    num_transitions = cfg.SimpleDiffusionGenerator.save_num_transitions
                    # samllest num_transition which is bigger than cfg.SimpleDiffusionGenerator.save_num_transitions
                    # and divisible by both cfg.Dataset.seq_len and sample_batch_size

                    lcm = cfg.Dataset.seq_len * sample_batch_size //(gcd(cfg.Dataset.seq_len, sample_batch_size))
                    if num_transitions % lcm != 0:
                        num_transitions = (num_transitions // lcm + 1) * lcm
                        
                    # num_transitions = -(-num_transitions // (cfg.Dataset.seq_len * sample_batch_size // \
                    #                   math.gcd(cfg.Dataset.seq_len, sample_batch_size))) * \
                    #                   (cfg.Dataset.seq_len * sample_batch_size // math.gcd(cfg.Dataset.seq_len, sample_batch_size))
                    # #num_transitions = math.ceil(num_transitions / math.lcm(cfg.Dataset.seq_len, sample_batch_size)) * math.lcm(cfg.Dataset.seq_len, sample_batch_size)
                    num_samples = num_transitions // cfg.Dataset.seq_len
                    

                    generated_samples = generator.sample_back_and_forth(
                        data_loader=sample_loader,
                        num_samples=num_samples,
                        noise_level = cfg.SimpleDiffusionGenerator.noise_level,
                        temperature=cfg.SimpleDiffusionGenerator.temperature,
                        device=distributed_state.device,
                        guidance_rewardscale=cfg.SimpleDiffusionGenerator.guidance_rewardscale,
                        fixed_rewardscale=cfg.SimpleDiffusionGenerator.fixed_rewardscale,
                        reward_scale=cfg.Dataset.reward_scale,
                        state_conditioning = cfg.SimpleDiffusionGenerator.goal_conditioning,
                        reward_interpolation = cfg.SimpleDiffusionGenerator.reward_interpolation,
                        context_conditioning = cfg.SimpleDiffusionGenerator.context_conditioning,
                        max_conditioning_return = dataset.trajectory_returns.max(),
                        discounted_return = cfg.Dataset.discounted_return,
                        on_first_state = cfg.SimpleDiffusionGenerator.on_first_state, 
                        is_td = is_td   
                    )


                # make file name - 5M으로 계산해서 
                num_samples = str(int(num_transitions // 1e6))+"M"


                if cfg.SimpleDiffusionGenerator.guidance_rewardscale is not None : 
                    guide = str(cfg.SimpleDiffusionGenerator.guidance_rewardscale)
                    if '.' in guide : 
                        guide = guide.replace('.','_')
                    guide += "x"
                else:
                    guide = int(100*cfg.SimpleDiffusionGenerator.fixed_rewardscale)
                    guide = "fixed"+str(guide)
                if cfg.construct_diffusion_model.denoising_network.force_dropout:
                    guide = "uncond"
                
                model_name = args.config_name.split('_')[0]

                noise_level = int(100*cfg.SimpleDiffusionGenerator.noise_level)

                mods = ""
                for mod in cfg.Dataset.modalities:
                    if mod == "observations":
                        mods += "s"
                    else:
                        mods += mod[0]

                temp = str(cfg.SimpleDiffusionGenerator.temperature)
                if '.' in temp :
                    temp = temp.replace('.','_')

                wp = int(cfg.SimpleDiffusionGenerator.weight_param)

                if cfg.SimpleDiffusionGenerator.reward_interpolation:
                    save_file_name = f"{num_samples}-{guide}-{model_name}-{noise_level}-{mods}-temp{temp}_{wp}_interpolated.npz"

                save_file_name = f"{num_samples}-{guide}-{model_name}-{noise_level}-{mods}-temp{temp}_{wp}.npz"


                gen_sample = np.array(generated_samples)
                np.random.shuffle(gen_sample)
                gen_sample = gen_sample[:(cfg.SimpleDiffusionGenerator.save_num_transitions//cfg.Dataset.seq_len)+1]
                savepath = os.path.join(resfolder, save_file_name)
                np.savez(savepath, 
                        data = gen_sample,
                        config = dict(cfg),
                        original_datapath = [savepath])
            

    else:
        print("diffusion training is done")

            