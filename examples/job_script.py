#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
from itertools import product
import numpy as np
import pickle
import tqdm
import yaml

from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjrl.utils import tensor_utils
from mjrl.utils.gym_env import GymEnv
from mjmpc.utils import LoggerClass, timeit, helpers
from mjmpc.policies import MPCPolicy


parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controllers', type=str, default='mppi', nargs='+', help='controller(s) to run')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

#Create the main environment
env_name  = exp_params['env_name']
# env = GymEnv(env_name)
env = gym.make(env_name)
env = GymEnvWrapper(env)
env.real_env_step(True)

# Create vectorized environments for MPPI simulations
def make_env():
    gym_env = gym.make(env_name)
    # gym_env = GymEnv(env_name) 
    rollout_env = GymEnvWrapper(gym_env)
    rollout_env.real_env_step(False)
    return rollout_env

d_obs = env.observation_space.high.shape[0]
d_action = env.action_space.high.shape[0]

def gather_trajectories(controller_name, policy_params, n_episodes, ep_length, base_seed, num_cpu, logger):
    sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])  

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
    

    policy_params['set_state_fn'] = set_state_fn
    policy_params['rollout_fn'] = rollout_fn
    policy_params['rollout_callback'] = None
    del policy_params['particles_per_cpu'], policy_params['num_cpu']


    ep_rewards = np.array([0.] * n_episodes)
    trajectories = []
    logger.info('Runnning {0} episodes'.format(n_episodes))
    timeit.start('start_'+controller_name)

    for i in tqdm.tqdm(range(n_episodes)):
        observations = []; actions = []; rewards = []; dones  = []
        infos = []; states = []
        #seeding
        episode_seed = base_seed + i*12345
        policy_params['seed'] = episode_seed
        env.reset(seed=episode_seed) #To enforce consistent episodes
        sim_env.reset()
        
        policy = MPCPolicy(controller_type=controller_name,
                           param_dict=policy_params, batch_size=1)
        
        for _ in tqdm.tqdm(range(ep_length)):   
            curr_state = deepcopy(env.get_env_state())
            action = policy.get_action(curr_state)
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
        # logger.info('Episode reward = {0}'.format(ep_rewards[i]))
        logger.record_tabular(controller_name+'episodeReward', ep_rewards[i])
        logger.dump_tabular()
            
    timeit.stop('start_'+controller_name)
    success_metric = env.env.unwrapped.evaluate_success(trajectories)
    average_reward = np.average(ep_rewards)
    reward_std = np.std(ep_rewards)
    logger.info('Timing info (seconds) {0}'.format(timeit))
    logger.info('Average reward = {0}'.format(average_reward))
    logger.info('Reward std = {0}'.format(reward_std))
    logger.info('Success metric = {0}'.format(success_metric))
    logger.info('Episode rewards = {0}'.format(ep_rewards))

    if exp_params['render']:
        _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
        helpers.render_trajs(env, trajectories, n_times=10)
    sim_env.close()
    return trajectories, average_reward, reward_std, success_metric

def main(controller_name, main_dir):    
    policy_params = exp_params[controller_name]
    policy_params['base_action'] = exp_params['base_action']
    policy_params['num_actions'] = env.action_space.low.shape[0]
    policy_params['action_lows'] = env.action_space.low
    policy_params['action_highs'] = env.action_space.high
    num_cpu = policy_params['num_cpu']

    # For every combination of horizon and number of particles, 
    # we run all combinations of algorithm parameters and 
    # save the results in sub-folder named H_#_N_#
    
    search_param_keys = []
    search_param_vals = []
    horizon_num_particles = []
    
    for k in policy_params:
        if isinstance(policy_params[k], list) and k not in ['filter_coeffs', 'horizon', 'num_cpu', 'particles_per_cpu']:
            search_param_keys.append(k)
            search_param_vals.append(policy_params[k])

    horizon_num_particles.append(policy_params['horizon'])
    num_particles = []
    for i in range(len(policy_params['particles_per_cpu'])):
        num_particles.append(policy_params['num_cpu'] * policy_params['particles_per_cpu'][i])
    horizon_num_particles.append(num_particles)
    # if len(search_param_keys) > 0:
    for horizon_particle_tuple in product(*horizon_num_particles):
        print(horizon_particle_tuple)
        s = "H_" + str(horizon_particle_tuple[0]) + "_N_" + str(horizon_particle_tuple[1])
        SUB_LOG_DIR = os.path.join(main_dir + "/" + s)
        sub_logger = helpers.get_logger(controller_name + "_" + s, SUB_LOG_DIR, 'debug')
       
        best_avg_reward = -np.inf
        best_success_metric = -np.inf
        best_reward_std = -np.inf
        best_trajectories = None
        best_param_dict = None
        count = 0

        policy_params['horizon'] = horizon_particle_tuple[0]
        policy_params['num_particles'] = horizon_particle_tuple[1] #policy_params['particles_per_cpu'] * policy_params['num_cpu']
        
        for search_param_tuple in product(*search_param_vals):
            best_params = False
            for i in range(len(search_param_tuple)):
                policy_params[search_param_keys[i]] = search_param_tuple[i]
            policy_params['particles_per_cpu'] = int(policy_params['num_particles']/policy_params['num_cpu'])
            sub_logger.info('Current params')
            sub_logger.info(policy_params)

            trajectories, avg_reward, reward_std, success_metric = gather_trajectories(controller_name,
                                                                                        deepcopy(policy_params), 
                                                                                        exp_params['n_episodes'], 
                                                                                        exp_params['max_ep_length'], 
                                                                                        exp_params['seed'],
                                                                                        num_cpu,
                                                                                        sub_logger)
            sub_logger.info('Success metric = {0}, Average reward = {1}, Best success metric = {2}, Best average reward = {3}'.format(success_metric, 
                                                                                                                                        avg_reward, 
                                                                                                                                        best_success_metric, 
                                                                                                                                        best_avg_reward))
            # if exp_params['job_mode'] == 'tune':
            if success_metric is not None: 
                if success_metric > best_success_metric:
                    sub_logger.info('Better success metric, updating best params...')
                    best_params = True
                elif np.allclose(success_metric, best_success_metric) and (avg_reward > best_avg_reward):
                    sub_logger.info('Similar success but better reward, updating params...')
                    best_params = True
            else:
                if avg_reward > best_avg_reward:
                    sub_logger.info('Best average reward, updating best params...')
                    best_params = True

            if best_params:
                best_trajectories = trajectories
                best_avg_reward = avg_reward
                best_reward_std = reward_std
                best_success_metric = success_metric
                best_param_dict = deepcopy(policy_params)
            sub_logger.info('Best params so far ...')
            sub_logger.info(best_param_dict)
                
            if success_metric is not None and best_success_metric > 95:
                sub_logger.info('Success metric greater than 95, early stopping')
                break

            sub_logger.info('Dumping trajectories')
            pickle.dump(best_trajectories, open(SUB_LOG_DIR+"/trajectories.p", 'wb'))
            count += 1
        
    return best_avg_reward, best_reward_std, best_success_metric


if __name__ == '__main__':
    avg_rewards = np.array([0.] * len(args.controllers))
    success = np.array([0.] * len(args.controllers))
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    for i, controller in enumerate(args.controllers):
        timeit.reset()
        LOG_DIR = os.path.join(os.path.abspath(args.save_dir) + "/" + exp_params['env_name'] + "/" + date_time, controller)
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        # logger = LoggerClass()
        # logger.setup(controller, os.path.join(LOG_DIR, 'log.txt'), 'debug')
        # logger.info('Running experiment: {0}. Results dir: {1}'.format(controller, LOG_DIR))
        avg_reward, reward_std, success_metric = main(controller, LOG_DIR)
        avg_rewards[i] = avg_reward
        success[i] = success_metric

    env.env.close()
