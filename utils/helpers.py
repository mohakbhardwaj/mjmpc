#!/usr/bin/env python
from envs import *
from policies import *
import numpy as np

def get_env_from_str(env_name):
    pass


def get_policy_from_str(name, params):
    if name == 'mpc_policy':
        return MPCPolicy(**params)
    elif name == 'random_policy':
        return RandomPolicy(**params)
    elif name == 'nn_policy':
        return NNPolicy(**params)
    else:
        raise NotImplementedError("Policy type not found")

def render_env(env, state_vec):
    n_steps = state_vec.shape[0]
    env.reset()
    for i in range(n_steps):
        curr_state = state_vec[:, i]
        # curr_action = action_vec[:, i]
        env.set_state(curr_state)
        env.render()
        # _,_,_,- = env.step

def decay_schedule_linear():
    raise NotImplementedError


