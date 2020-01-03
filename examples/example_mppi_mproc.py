#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
import gym
import numpy as np
import tqdm

import envs
from envs.vec_env import SubprocVecEnv
from utils import logger, timeit, Buffer
from policies import MPCPolicy


parser = argparse.ArgumentParser(description='Run random policy on environment')
parser.add_argument('--env', help='Environment name', default='SimplePendulumEnv-v0')
parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes')
parser.add_argument('--H', type=int, default=32, help='Planning horizon')
parser.add_argument('--max_ep_length', type=int, default=200, help='Length of episode before reset')
parser.add_argument('--num_particles', type=int, default=24, help='number of samples for MPPI')
parser.add_argument('--num_cpu', type=int, default=8, help='number of processes for simulation')
parser.add_argument('--init_cov', type=float, default=3.5, help='standard deviation for noise added to controls')
# parser.add_argument('--min_cov', type=float, default=0.1, help='standard deviation for noise added to controls')
# parser.add_argument('--prior_cov', type=float, default=0.01, help='standard deviation for noise added to controls')
parser.add_argument('--lam', type=float, default=0.01, help='temperature parameter for mppi')
parser.add_argument('--step_size', type=float, default=0.55, help='step size for mean update for mppi')
parser.add_argument('--alpha', type=int, default=0, help='weight for control seq from passive dynamics (0=passive dynamics has zero control,\
																										1=passive dyn is current control distribution')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
# parser.add_argument('--beta', type=float, default=0.1, help='step size for growing covariance')
parser.add_argument('--n_iters', type=int, default=1, help='number of update steps per iteraion of mpc')
parser.add_argument('--seed', type=int, default=0, help='number of samples per planning iteration')
parser.add_argument('--render', action='store_true', help='render environment')
args = parser.parse_args()

#Setiup logging
LOG_DIR = os.path.abspath("..")
logger.setup("MPPI", os.path.join(LOG_DIR, 'log.txt'), 'debug')

#Create the main environment
np.random.seed(args.seed)
print('Environment = {0}'.format(args.env))
kwargs = {'batch_size': 1}
env = gym.make(args.env, **kwargs)
env.seed(args.seed)


kwargs = {'batch_size': int(args.num_particles/args.num_cpu)}

def make_env():
	env = gym.make(args.env, **kwargs)
	return env

sim_env = SubprocVecEnv([make_env for i in range(args.num_cpu)])  # Vectorized environments for MPPI simulations
seed_list = [args.seed] * args.num_cpu
sim_env.seed(seed_list)


#Create functions for MPPI
def get_state_fn() -> np.ndarray:
    """
    Get state of main environment to plan from
    """
    state = env.get_state()
    return state


def set_state_fn(state: np.ndarray):
    """
    Set state of simulation environments for rollouts
    """
    #sim_env.reset()
    state = np.tile(state, (args.num_particles, state.shape[0]))
    sim_env.set_state(state)


def rollout_fn(start_state: np.ndarray, u_vec: np.ndarray):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return rewards and observations 
    received at every timestep
    """
    obs_vec, state_vec, rew_vec, done_vec, _ = sim_env.rollout(np.transpose(u_vec, (2, 0, 1)))
    return obs_vec, state_vec, rew_vec.T, done_vec


def rollout_callback():
    """
    Callback called after MPPI rollouts for plotting etc.
    """
    pass


#Create dictionary of policy params
policy_params = {'horizon': args.H,
                 'init_cov': args.init_cov,
                 'min_cov': None,
                 'prior_cov': None,
                 'lam': args.lam,
                 'num_particles':  args.num_particles,
                 'step_size':  args.step_size,
                 'alpha':  args.alpha,
                 'beta':  None,
                 'gamma':  args.gamma,
                 'n_iters':  args.n_iters,
                 'num_actions':  env.d_action,
                 'action_lows':  env.action_space.low,
                 'action_highs':  env.action_space.high,
                 'set_state_fn':  set_state_fn,
                 'get_state_fn':  get_state_fn,
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
buff = Buffer(env.d_obs, env.d_action, env.d_state,
              max_length=n_episodes*max_ep_length, ensemble_size=1)

logger.info('Runnning {0} episodes'.format(n_episodes))
timeit.start('start')

for i in tqdm.tqdm(range(n_episodes)):
    curr_obs = env.reset()
    #prev_action = np.zeros(1, env.d_action)
    for t in tqdm.tqdm(range(max_ep_length)):
        curr_state = env.get_state()
        #Get action from policy
        action = policy.get_action()
        #Perform action on environment
        obs, reward, done, info = env.step(action)
        #Add transition to buffer
        next_state = env.get_state()
        buff.add((curr_obs, action, obs, np.zeros_like(
            action), reward.item(), done.item(), done.item(), curr_state, next_state))
        curr_obs = obs.copy()
        ep_rewards[i] += reward.item()

timeit.stop('start')
logger.info(timeit)
logger.info('Buffer Length = {0}'.format(len(buff)))
logger.info('Average reward = {0}. Closing...'.format(np.average(ep_rewards)))

if args.render:
    logger.info('Rendering 2 times')
    buff.render(env, n_times=2)
sim_env.close()
env.close()
