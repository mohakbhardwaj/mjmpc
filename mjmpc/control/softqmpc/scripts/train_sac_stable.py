import gym
import numpy as np
import torch
import random

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from mjmpc.envs import GymEnvWrapper
import mj_envs

seed_val=123456
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)
env = gym.make('cartpole-v0')
env = GymEnvWrapper(env)
env.seed(seed_val)
env.action_space.seed(seed_val)

model = SAC(MlpPolicy, env, learning_starts=10000, verbose=1, seed=seed_val)
# model.learn(total_timesteps=200000, log_interval=4)
# model.save("sac_cartpole")

# del model # remove to demonstrate saving and loading

model = SAC.load("sac_cartpole")

# obs = env.reset()
num_episodes = 0.0
ep_rewards = []
curr_ep_rew = 0.0
env.seed(123)

for i_episode in range(10):
    test_episode_seed = 123 + i_episode*12345
    obs = env.reset(test_episode_seed)
    num_episodes += 1
    curr_ep_rew = 0.0
    t=0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        curr_ep_rew += reward
        # env.render()
        if done:
            ep_rewards.append(curr_ep_rew)
            break
print('Average reward: {}'.format(np.average(ep_rewards)))