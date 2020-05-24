#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
from itertools import product
import json
import numpy as np
import pickle
import tqdm
import yaml

import mjmpc.envs
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjrl.utils import tensor_utils
from mjmpc.utils import LoggerClass, timeit, helpers
from mjmpc.policies import MPCPolicy

gym.logger.set_level(40)
parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controller', type=str, default='mppi', help='controller to run')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes' )
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

#Create the main environment
env_name  = exp_params['env_name']
env = gym.make(env_name)
env = GymEnvWrapper(env)
env.real_env_step(True)

# Function to create vectorized environments for controller simulations
def make_env():
    gym_env = gym.make(env_name)
    rollout_env = GymEnvWrapper(gym_env)
    rollout_env.real_env_step(False)
    return rollout_env

d_obs = env.observation_space.high.shape[0]
d_action = env.action_space.high.shape[0]

#unpack params and create policy params
controller_name = args.controller
policy_params = exp_params[controller_name]
policy_params['base_action'] = exp_params['base_action']
policy_params['num_actions'] = env.action_space.low.shape[0]
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']
num_cpu = policy_params['num_cpu']
n_episodes = exp_params['n_episodes']
base_seed = exp_params['seed']
ep_length = exp_params['max_ep_length']


#Create vectorized environments for MPC simulations
sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])  

#Create functions for controller
def set_sim_state_fn(state_dict: dict):
    """
    Set state of simulation environments for rollouts
    """
    sim_env.set_env_state(state_dict)

def rollout_fn(u_vec: np.ndarray):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return sequence of costs
    """
    obs_vec, rew_vec, done_vec, _ = sim_env.rollout(u_vec.copy())
    return -1.0*rew_vec #we assume environment returns rewards, but controller needs consts

del policy_params['particles_per_cpu'], policy_params['num_cpu']

#Create logger
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/" + controller_name
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger(controller_name + "_" + exp_params['env_name'], log_dir, 'debug')
ep_rewards = np.array([0.] * n_episodes)
trajectories = []
logger.info(exp_params[controller_name])

#Main data collection loop
timeit.start('start_'+controller_name)
for i in tqdm.tqdm(range(n_episodes)):
    #seeding to enforce consistent episodes
    episode_seed = base_seed + i*12345
    policy_params['seed'] = episode_seed
    env.reset(seed=episode_seed)
    sim_env.reset()

    #create MPC policy and set appropriate functions
    policy = MPCPolicy(controller_type=controller_name,
                        param_dict=policy_params, batch_size=1) #Note only batch_size=1 is supported for now
    policy.controller.set_sim_state_fn = set_sim_state_fn
    policy.controller.rollout_fn = rollout_fn
    
    #Collect data from interactions with environment
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    for _ in tqdm.tqdm(range(ep_length)):   
        curr_state = deepcopy(env.get_env_state())
        action, value = policy.get_action(curr_state, calc_val=False)
        obs, reward, done, info = env.step(action)

        observations.append(obs); actions.append(action)
        rewards.append(reward); dones.append(done)
        infos.append(info); states.append(curr_state)
        ep_rewards[i] += reward
    
    traj = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        env_infos=tensor_utils.stack_tensor_dict_list(infos),
        states=states
    )
    trajectories.append(traj)
sim_env.close() #Free up memory

timeit.stop('start_'+controller_name) #stop timer after trajectory collection
success_metric = env.env.unwrapped.evaluate_success(trajectories)
average_reward = np.average(ep_rewards)
reward_std = np.std(ep_rewards)

#Display logs on screen and save in txt file
logger.info('Avg. reward = {0}, Std. Reward = {1}, Success Metric = {2}'.format(average_reward, reward_std, success_metric))

#Can also dump data to csv once done
logger.record_tabular("Horizon", policy_params['horizon'])
logger.record_tabular("NumParticles", policy_params['num_particles'])
logger.record_tabular("AverageReward", average_reward)
logger.record_tabular("StdReward", reward_std)
logger.record_tabular("SuccessMetric", success_metric)
logger.record_tabular("NumEpisodes", exp_params['n_episodes'])
logger.dump_tabular()


if exp_params['render']:
    _ = input("Press enter to display optimized trajectories (will be played 3 times) : ")
    helpers.render_trajs(env, trajectories, n_times=3)

env.close()