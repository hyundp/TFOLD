import gym
import numpy as np

import collections
import pickle

import d4rl




def make_dataset(name, val_name, env_plus_type, dataset, train_fraction):
	N = dataset['rewards'].shape[0]
	print("N: ", N)
	train_size = int(train_fraction * N)
	data_ = collections.defaultdict(list)
	val_data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	paths = []
	val_paths = []
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)
		if i < train_size:
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
					data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
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
		episode_step += 1

	# returns = np.array([np.sum(p['rewards']) for p in paths])
	# num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	# print(f'Number of samples collected: {num_samples}')
	# print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	# check expected split
	print("env name: ", env_plus_type)
	print("train length: ", len(paths)*len(paths[0]['observations']))
	print("val length: ", len(val_paths)*len(paths[0]['observations']))
	print("total length: ", len(paths)*len(paths[0]['observations']) + len(val_paths)*len(paths[0]['observations']) )
	print("-----------------------------------------------------")
	
	with open(name, 'wb') as f:
		pickle.dump(paths, f)
	with open(val_name, 'wb') as f:
		pickle.dump(val_paths, f)
		
	
datasets = []
train_fraction = 0.8

for env_name in ['halfcheetah', 'hopper', 'walker2d']:
	for dataset_type in ['medium', 'medium-replay', 'expert']:
		name = f'data/train/{env_name}-{dataset_type}-v2.pkl'
		val_name = f'data/val/val_{env_name}-{dataset_type}-v2.pkl'
		env_plus_type = f'{env_name}-{dataset_type}-v2'
		env = gym.make(env_plus_type)
		dataset = env.get_dataset()
		
		make_dataset(name, val_name, env_plus_type, dataset, train_fraction)
		


