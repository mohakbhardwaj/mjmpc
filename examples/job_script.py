#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import pickle
import tqdm
import yaml

from envs import GymEnvWrapper
from envs.vec_env import SubprocVecEnv
from mjrl.utils import tensor_utils
from utils import logger, timeit, helpers
from policies import MPCPolicy


parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config_file', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controllers', type=str, default='mppi', nargs='+', help='controller(s) to run')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config_file) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

#Create the main environment
env_name  = exp_params['env_name']
# np.random.seed(exp_params['seed'])
env = gym.make(env_name)
# env.seed(exp_params['seed'])

# Create vectorized environments for MPPI simulations
def make_env():
    gym_env = gym.make(env_name)
    rollout_env = GymEnvWrapper(gym_env)
    return rollout_env

d_obs = env.observation_space.high.shape[0]
d_action = env.action_space.high.shape[0]


def main(controller_name):    
    num_cpu = exp_params[controller_name]['num_cpu']
    sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])  
    seed_list = [exp_params['seed'] + i for i in range(num_cpu)]
    sim_env.seed(seed_list)
    _ = sim_env.reset()

    #Create functions for controller
    def set_state_fn(state_dict: dict):
        """
        Set state of simulation environments for rollouts
        """
        state_dicts = [deepcopy(state_dict) for j in range(num_cpu)]
        sim_env.set_env_state(state_dicts)

    def rollout_fn(u_vec: np.ndarray):
        """
        Given a batch of sequences of actions, rollout 
        in sim envs and return rewards and observations 
        received at every timestep
        """
        obs_vec, rew_vec, done_vec, _ = sim_env.rollout(u_vec.copy())
        return obs_vec, rew_vec, done_vec #state_vec

    #Create policy
    policy_params = exp_params[controller_name]

    if len(policy_params['init_cov']) == 1: policy_params['init_cov'] = [policy_params['init_cov'][0]] * d_action
    policy_params['base_action'] = exp_params['base_action']
    policy_params['num_particles'] = policy_params['particles_per_cpu'] * policy_params['num_cpu']
    policy_params['num_actions'] = env.action_space.low.shape[0]
    policy_params['action_lows'] = env.action_space.low
    policy_params['action_highs'] = env.action_space.high
    policy_params['set_state_fn'] = set_state_fn
    policy_params['rollout_fn'] = rollout_fn
    policy_params['rollout_callback'] = None
    del policy_params['particles_per_cpu'], policy_params['num_cpu']

    n_episodes = exp_params['n_episodes']
    max_ep_length = exp_params['max_ep_length']
    ep_rewards = np.array([0.] * n_episodes)

    #Create experience buffer
    trajectories = []

    logger.info('Runnning {0} episodes'.format(n_episodes))
    timeit.start('start_'+controller_name)
    for i in tqdm.tqdm(range(n_episodes)):
        #seeding
        curr_seed = exp_params['seed']+i*12345
        policy_params['seed'] = curr_seed
        policy = MPCPolicy(controller_type=controller_name,
                            param_dict=policy_params, batch_size=1)
        env.seed(seed=curr_seed) #To enforce consistent episodes
        sim_env.seed([curr_seed + j for j in range(num_cpu)])

        observations = []; actions = []; rewards = []; dones  = []; 
        infos = []; states = []

        obs = env.reset()
        _ = sim_env.reset()
        
        for t in tqdm.tqdm(range(max_ep_length)):   
            curr_state = deepcopy(env.get_env_state())
            action = policy.get_action(curr_state)
            obs, reward, done, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            states.append(curr_state)

            ep_rewards[i] += reward
        
        traj = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=infos,
            states=states
        )
        trajectories.append(traj)
        logger.record_tabular(controller_name+'episodeReward', ep_rewards[i])
        logger.dump_tabular()
        pickle.dump(trajectories, open(LOG_DIR+"/trajectories.p", 'wb'))
            
    timeit.stop('start_'+controller_name)
    logger.info('Timing info (seconds): {0}'.format(timeit))
    logger.info('Average reward = {0}'.format(np.average(ep_rewards)))
    logger.info('Episode rewards = {0}'.format(ep_rewards))

    if exp_params['render']:
        _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
        helpers.render_trajs(env, trajectories, n_times=10)
        

    sim_env.close()
    return np.average(ep_rewards)


if __name__ == '__main__':
    avg_rewards = np.array([0.] * len(args.controllers))
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    for i, controller in enumerate(args.controllers):
        timeit.reset()
        LOG_DIR = os.path.join(os.path.abspath(args.save_dir) + "/" + exp_params['env_name'] + "/" + date_time, controller)
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        logger.setup(controller, os.path.join(LOG_DIR, 'log.txt'), 'debug')
        logger.info('Running experiment: {0}. Results dir: {1}'.format(controller, LOG_DIR))
        avg_reward = main(controller)
        avg_rewards[i] = avg_reward

    env.close()
