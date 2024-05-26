import gym
import numpy as np
import pyrootutils
from gym.wrappers.flatten_observation import FlattenObservation

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from REDQ.redq.algos.core import ReplayBuffer


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer: ReplayBuffer,
        model_terminals: bool = False,
) -> np.ndarray:
    ptr_location = replay_buffer.ptr
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location]
    inputs = [obs, actions, rewards[:, None], next_obs]
    if model_terminals:
        terminals = replay_buffer.done_buf[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None])
    return np.concatenate(inputs, axis=-1)

\


def make_sequences_from_replay_buffer(
        replay_buffer: ReplayBuffer,
):
    
    ptr_location = replay_buffer.ptr
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location].reshape(-1, 1)

    inputs = [obs, actions, rewards]
    inputs = np.concatenate(inputs, axis=-1)

    return inputs
    