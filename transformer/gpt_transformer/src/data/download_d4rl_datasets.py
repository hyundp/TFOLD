import collections
import os
import pickle

import d4rl
import gym
import numpy as np


def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")


def make_dataset(original_name, train_name, val_name, env_plus_type, dataset, train_fraction):
	N = dataset['rewards'].shape[0]
	print("N: ", N)
	train_size = int(train_fraction * N)
	original_data_ = collections.defaultdict(list)
	train_data_ = collections.defaultdict(list)
	val_data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	original_paths = []
	train_paths = []
	val_paths = []
	
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)
		if i < train_size:
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
					train_data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in train_data_:
					episode_data[k] = np.array(train_data_[k])
				train_paths.append(episode_data)
				train_data_ = collections.defaultdict(list)
		else:
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
				val_data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				val_episode_data = {}
				for k in val_data_:
					val_episode_data[k] = np.array(val_data_[k])
				val_paths.append(val_episode_data)
				val_data_ = collections.defaultdict(list)
				
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			original_data_[k].append(dataset[k][i])
		if done_bool or final_timestep:
			episode_step = 0
			original_episode_data = {}
			for k in original_data_:
				original_episode_data[k] = np.array(original_data_[k])
			original_paths.append(original_episode_data)
			original_data_ = collections.defaultdict(list)
		episode_step += 1

	# returns = np.array([np.sum(p['rewards']) for p in paths])
	# num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	# print(f'Number of samples collected: {num_samples}')
	# print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	# check expected split
	print("env name: ", env_plus_type)
	# print("train length: ", len(paths)*len(paths[0]['observations']))
	# print("val length: ", len(val_paths)*len(paths[0]['observations']))
	print("-----------------------------------------------------")
	
	with open(original_name, 'wb') as f:
		pickle.dump(original_paths, f)
	with open(train_name, 'wb') as f:
		pickle.dump(train_paths, f)
	with open(val_name, 'wb') as f:
		pickle.dump(val_paths, f)
		

# make directory
DATA_PATH = 'transformer/gpt_transformer/src/data/'
data_type_list = ['original', 'train', 'val', 'filtered', 'augmented', 'statistics']

for type in data_type_list:
	make_dir(os.path.join(DATA_PATH, type))
		
datasets = []
train_fraction = 0.8

for env_name in ['halfcheetah', 'hopper', 'walker2d']:
	for dataset_type in ['medium', 'medium-replay', 'medium-expert']:
		original_name = f'transformer/gpt_transformer/src/data/original/{env_name}-{dataset_type}-v2.pkl'
		train_name = f'transformer/gpt_transformer/src/data/train/{env_name}-{dataset_type}-v2.pkl'
		val_name = f'transformer/gpt_transformer/src/data/val/{env_name}-{dataset_type}-v2.pkl'
		env_plus_type = f'{env_name}-{dataset_type}-v2'
		env = gym.make(env_plus_type)
		dataset = env.get_dataset()
		
		make_dataset(original_name, train_name, val_name, env_plus_type, dataset, train_fraction)
		


