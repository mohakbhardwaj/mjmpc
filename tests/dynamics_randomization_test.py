#!/usr/bin/env python
import gym
import numpy as np
import yaml

from mjmpc.envs import GymEnvWrapper
import mj_envs

env = gym.make('cartpole-v0')
env = GymEnvWrapper(env)
env.seed(0)

#Load dynamics params file
with open("../examples/configs/classic_control/cartpole_dyn_randomize.yml") as file:
    param_dict = yaml.load(file, Loader=yaml.FullLoader)

print('Call randomize dynamics')
default_params, randomized_params = env.randomize_dynamics(param_dict)

print('Default params: {}'.format(default_params))
print('Randomized params: {}'.format(randomized_params))

print('Call again to make sure default params are not updated')

default_params, randomized_params = env.randomize_dynamics(param_dict)

print('Default params: {}'.format(default_params))
print('Randomized params: {}'.format(randomized_params))