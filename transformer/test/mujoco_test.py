import gym
import d4rl


env = gym.make('Walker2d-v3')
env.reset()
env.step(env.action_space.sample())
env.close()
print("mujoco-py check passed")

env = gym.make('walker2d-medium-v2')
env.reset()
env.step(env.action_space.sample())
env.close()
print("d4rl check passed")