#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import tqdm

from envs import GymEnvWrapper
from envs.vec_env import SubprocVecEnv
from utils import logger, timeit, Buffer
from policies import MPCPolicy


parser = argparse.ArgumentParser(description='Run random policy on environment')
parser.add_argument('--env', help='Environment name', default='SimplePendulum-v0')
parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes')
parser.add_argument('--H', type=int, default=32, help='Planning horizon')
parser.add_argument('--max_ep_length', type=int, default=200, help='Length of episode before reset')
parser.add_argument('--num_cpu', type=int, default=8, help='number of processes for simulation')
parser.add_argument('--particles_per_cpu', type=int, default=3, help='number of samples for MPPI')
parser.add_argument('--init_cov', type=float, default=3.5, help='standard deviation for noise added to controls')
parser.add_argument('--lam', type=float, default=0.01, help='temperature parameter for mppi')
parser.add_argument('--step_size', type=float, default=0.55, help='step size for mean update for mppi')
parser.add_argument('--alpha', type=int, default=0, help='weight for control seq from passive dynamics (0=passive dynamics has zero control,\
																										1=passive dyn is current control distribution')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--n_iters', type=int, default=1, help='number of update steps per iteraion of mpc')
parser.add_argument('--seed', type=int, default=0, help='number of samples per planning iteration')
parser.add_argument('--render', action='store_true', help='render environment')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')

args = parser.parse_args()

#Setup logging
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
exp_name = 'MPPI_{0}'.format(args.env)
LOG_DIR = os.path.join(os.path.abspath(args.save_dir), exp_name + "/" + date_time)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.setup(exp_name, os.path.join(LOG_DIR, 'log.txt'), 'debug')

#Create the main environment
np.random.seed(args.seed)
print('Environment = {0}'.format(args.env))
env = gym.make(args.env)
env.seed(args.seed)


# kwargs = {'batch_size': int(args.num_particles/args.num_cpu)}

# Create vectorized environments for MPPI simulations
def make_env():
    env = gym.make(args.env)
    rollout_env = GymEnvWrapper(env)
    return rollout_env

sim_env = SubprocVecEnv([make_env for i in range(args.num_cpu)])  
seed_list = [args.seed] * args.num_cpu
sim_env.seed(seed_list)
_ = sim_env.reset()


#Create functions for MPPI
def set_state_fn(state_dict: dict):
    """
    Set state of simulation environments for rollouts
    """
    #sim_env.reset()
    # state = np.tile(state, (args.num_particles, state.shape[0]))
    state_dicts = [deepcopy(state_dict) for j in range(args.num_cpu)]
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
policy_params = {'horizon': args.H,
                 'init_cov': args.init_cov,
                 'lam': args.lam,
                 'num_particles':  args.particles_per_cpu * args.num_cpu,
                 'step_size':  args.step_size,
                 'alpha':  args.alpha,
                 'gamma':  args.gamma,
                 'n_iters':  args.n_iters,
                 'num_actions':  env.action_space.low.shape[0],
                 'action_lows':  env.action_space.low,
                 'action_highs':  env.action_space.high,
                 'set_state_fn':  set_state_fn,
                 'rollout_fn':  rollout_fn,
                 'rollout_callback': None,
                 'seed':  args.seed}

#Create policy
policy = MPCPolicy(controller_type='mppi',
                   param_dict=policy_params, batch_size=1)
n_episodes = args.n_episodes
max_ep_length = args.max_ep_length
ep_rewards = np.array([0.] * n_episodes)

#Create experience buffer
d_obs = env.observation_space.high.shape[0]
d_action = env.action_space.high.shape[0]

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
        try:
            obs, reward, done, info = env.step(action)
        except ValueError:
            print(action.shape)
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

if args.render:
    logger.info('Rendering 2 times')
    buff.render(env, n_times=2)
sim_env.close()
env.close()
