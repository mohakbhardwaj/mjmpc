#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import tqdm
import yaml

from envs import GymEnvWrapper
from envs.vec_env import SubprocVecEnv
from utils import logger, timeit, Buffer
from policies import MPCPolicy


parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config_file', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controller_type', type=str, default='mppi', help='controller to run')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config_file) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

#Setup logging
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
exp_name = '{0}_{1}'.format(args.controller_type, exp_params['env_name'])
LOG_DIR = os.path.join(os.path.abspath(args.save_dir), exp_name + "/" + date_time)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.setup(exp_name, os.path.join(LOG_DIR, 'log.txt'), 'debug')
logger.info('Running experiment: {0}. Results dir: {1}'.format(exp_name, LOG_DIR))

#Create the main environment
env_name  = exp_params['env_name']
np.random.seed(exp_params['seed'])
env = gym.make(env_name)
env.seed(exp_params['seed'])

# Create vectorized environments for MPPI simulations
def make_env():
    env = gym.make(env_name)
    rollout_env = GymEnvWrapper(env)
    return rollout_env

sim_env = SubprocVecEnv([make_env for i in range(exp_params['num_cpu'])])  
seed_list = [exp_params['seed']] * exp_params['num_cpu']
sim_env.seed(seed_list)
_ = sim_env.reset()

#Create functions for MPPI
def set_state_fn(state_dict: dict):
    """
    Set state of simulation environments for rollouts
    """
    state_dicts = [deepcopy(state_dict) for j in range(exp_params['num_cpu'])]
    sim_env.set_env_state(state_dicts)


def rollout_fn(u_vec: np.ndarray):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return rewards and observations 
    received at every timestep
    """
    obs_vec, rew_vec, done_vec, _ = sim_env.rollout(np.transpose(u_vec, (2, 0, 1)))
    return obs_vec, rew_vec.T, done_vec #state_vec


def rollout_callback():
    """
    Callback called after MPPI rollouts for plotting etc.
    """
    pass

#Create dictionary of policy params
d_obs = env.observation_space.high.shape[0]
d_action = env.action_space.high.shape[0]
policy_params = exp_params[args.controller_type]

if len(policy_params['init_cov']) == 1: policy_params['init_cov'] = [policy_params['init_cov']] * d_action
policy_params['num_particles'] = exp_params['particles_per_cpu'] * exp_params['num_cpu']
policy_params['num_actions'] = env.action_space.low.shape[0]
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
policy_params['set_state_fn'] = set_state_fn
policy_params['rollout_fn'] = rollout_fn
policy_params['rollout_callback'] = None

#Create policy
policy = MPCPolicy(controller_type=args.controller_type,
                   param_dict=policy_params, batch_size=1)
n_episodes = exp_params['n_episodes']
max_ep_length = exp_params['max_ep_length']
ep_rewards = np.array([0.] * n_episodes)

#Create experience buffer
buff = Buffer(d_obs, d_action, max_length=n_episodes*max_ep_length)

logger.info('Runnning {0} episodes'.format(n_episodes))
timeit.start('start')

for i in tqdm.tqdm(range(n_episodes)):
    curr_obs = env.reset()
    #prev_action = np.zeros(1, env.d_action)
    for t in tqdm.tqdm(range(max_ep_length)):
        curr_state = env.get_env_state()
        #Get action from policy
        action = policy.get_action(curr_state)
        #Perform action on environment
        obs, reward, done, info = env.step(action)
        #Add transition to buffer
        next_state = env.get_env_state()
        buff.add((curr_obs, action, obs, np.zeros_like(
                  action), reward, done, done, curr_state, next_state))
        curr_obs = obs.copy()
        ep_rewards[i] += reward

timeit.stop('start')
logger.info(timeit)
logger.info('Buffer Length = {0}'.format(len(buff)))
logger.info('Average reward = {0}. Closing...'.format(np.average(ep_rewards)))
buff.save(LOG_DIR)

if exp_params['render']:
    buff.render(env, n_times=2)
sim_env.close()
env.close()
