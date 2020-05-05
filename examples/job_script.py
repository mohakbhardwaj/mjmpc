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

import mj_envs
from mjmpc.envs import GymEnvWrapper
# from mjmpc.envs.gym_env_wrapper_cy import GymEnvWrapperCy
# from mjmpc.envs import make_wrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjrl.utils import tensor_utils
from mjmpc.utils import LoggerClass, timeit, helpers
from mjmpc.policies import MPCPolicy

gym.logger.set_level(40)
parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--controllers', type=str, default='mppi', nargs='+', help='controller(s) to run')
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

# Create vectorized environments for MPPI simulations
def make_env():
    gym_env = gym.make(env_name)
    rollout_env = GymEnvWrapper(gym_env)
    # rollout_env = make_wrapper(gym_env)
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
        return rew_vec
    

    policy_params['set_state_fn'] = set_state_fn
    policy_params['rollout_fn'] = rollout_fn
    del policy_params['particles_per_cpu'], policy_params['num_cpu']


    ep_rewards = np.array([0.] * n_episodes)
    trajectories = []
    logger.info('Runnning {0} episodes. Base seed = {1}'.format(n_episodes, base_seed))
    timeit.start('start_'+controller_name)

    for i in tqdm.tqdm(range(n_episodes)):
        #seeding to enforce consistent episodes
        episode_seed = base_seed + i*12345
        policy_params['seed'] = episode_seed
        env.reset(seed=episode_seed)
        sim_env.reset()
 
        policy = MPCPolicy(controller_type=controller_name,
                           param_dict=policy_params, batch_size=1)
        
        observations = []; actions = []; rewards = []; dones  = []
        infos = []; states = []
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
        logger.record_tabular(controller_name+'episodeReward', ep_rewards[i])
        logger.dump_tabular()
            
    timeit.stop('start_'+controller_name)
    success_metric = env.env.unwrapped.evaluate_success(trajectories)
    average_reward = np.average(ep_rewards)
    reward_std = np.std(ep_rewards)
    logger.info('Timing info (seconds) {0}'.format(timeit))
    # logger.info('Average reward = {0}'.format(average_reward))
    # logger.info('Reward std = {0}'.format(reward_std))
    # logger.info('Success metric = {0}'.format(success_metric))
    # logger.info('Episode rewards = {0}'.format(ep_rewards))

    # if exp_params['render']:
    #     _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    #     helpers.render_trajs(env, trajectories, n_times=10)
    sim_env.close()
    return trajectories, average_reward, reward_std, success_metric

def main(controller_name, main_dir):    
    policy_params = exp_params[controller_name]
    policy_params['base_action'] = exp_params['base_action']
    policy_params['num_actions'] = env.action_space.low.shape[0]
    policy_params['action_lows'] = env.action_space.low
    policy_params['action_highs'] = env.action_space.high
    num_cpu = policy_params['num_cpu']
    
    num_particles = []
    if isinstance(policy_params['horizon'], list):
        horizons = policy_params['horizon']
    else: horizons =  [policy_params['horizon']]    
    if isinstance(policy_params['particles_per_cpu'], list):
        for i in range(len(policy_params['particles_per_cpu'])):
            num_particles.append(policy_params['num_cpu'] * policy_params['particles_per_cpu'][i])
    else:
        num_particles.append(policy_params['num_cpu'] * policy_params['particles_per_cpu'])


    if exp_params['job_mode'] == 'tune':
        ############### Tune mode #################
        # For every combination of horizon and number of particles, 
        # we run all combinations of controller parameters and 
        # save the results in sub-folder named H_#_N_#
        #############################################
        horizon_num_particles = []
        horizon_num_particles.append(horizons)
        horizon_num_particles.append(num_particles)

        # We will seperate out the parameters that need to be tuned 
        # and the ones that are fixed
        tune_param_keys = []
        tune_param_vals = []
        fix_param_keys = []
        fix_param_vals = []
        for k in policy_params:
            if  isinstance(policy_params[k], list) and \
                k not in ['filter_coeffs', 'horizon', 'num_cpu', 'particles_per_cpu', 'tune_keys']:
                    if k in policy_params['tune_keys']:
                        tune_param_keys.append(k)
                        tune_param_vals.append(policy_params[k])
                    else:
                        assert len(policy_params[k]) == len(horizons) * len(num_particles), \
                        "Please provide correct number of fixed parameters"
                        fix_param_keys.append(k)
                        fix_param_vals.append(policy_params[k])


        # for k in policy_params:
        #     if k not in policy_params['tune_keys'] and \
        #        isinstance(policy_params[k], list) and \
        #        k not in ['filter_coeffs', 'horizon', 'num_cpu', 'particles_per_cpu', 'tune_keys']:
        policy_params.pop('tune_keys', None)
        


        for (n,tup) in enumerate(product(*horizon_num_particles)):
            s = "H_" + str(tup[0]) + "_N_" + str(tup[1])
            SUB_LOG_DIR = os.path.join(main_dir + "/" + s)
            sub_logger = helpers.get_logger(controller_name + "_" + s, SUB_LOG_DIR, 'debug')
        
            best_avg_reward = -np.inf
            best_success_metric = -np.inf
            best_reward_std = -np.inf
            best_trajectories = None
            best_param_dict = None

            policy_params['horizon'] = tup[0]
            policy_params['num_particles'] = tup[1]

            for i in range(len(fix_param_keys)):
                policy_params[fix_param_keys[i]] = fix_param_vals[i][n]

            for tune_param_tuple in product(*tune_param_vals):
                best_params = False
                for i in range(len(tune_param_tuple)):
                    policy_params[tune_param_keys[i]] = tune_param_tuple[i]
                
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
                if args.dump_vids:
                    print('Dumping videos')
                    helpers.dump_videos(env=env, trajectories=trajectories, frame_size=(1280, 720), 
                                        folder=SUB_LOG_DIR, filename='vid_traj_', camera_name=None,
                                        device_id=1)
                if exp_params['render']:
                    _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
                    helpers.render_trajs(env, trajectories, n_times=10)


                sub_logger.info('Success metric = {0}, Average reward = {1}, Std reward = {2}, Best success metric = {3}, Best average reward = {4}, Best std reward = {5}'.format(success_metric, 
                                                                                                                                                                                   avg_reward, 
                                                                                                                                                                                   reward_std,
                                                                                                                                                                                   best_success_metric, 
                                                                                                                                                                                   best_avg_reward,
                                                                                                                                                                                   best_reward_std))
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

                ##Saving results###
                sub_logger.info('Dumping results and trajectories')
                best_results_dict = dict(best_avg_reward=best_avg_reward, best_reward_std=best_reward_std, 
                                         best_success_metric=best_success_metric, num_episodes=exp_params['n_episodes'],
                                         seed_val=exp_params['seed'])
                save_param_dict = deepcopy(best_param_dict)
                save_param_dict.pop('base_action', None)
                save_param_dict.pop('num_actions', None)
                save_param_dict.pop('action_lows', None)
                save_param_dict.pop('action_highs', None)
            
                with open(SUB_LOG_DIR+"/best_results.txt", 'w') as f:
                    json.dump(best_results_dict, f)
                with open(SUB_LOG_DIR+"/best_params.txt", 'w') as f:
                    json.dump(save_param_dict, f)
                pickle.dump(best_trajectories, open(SUB_LOG_DIR+"/trajectories.p", 'wb'))
                ######################
    
    elif exp_params['job_mode'] == 'sweep':
        ############### Sweep mode #################
        # For corresponding horizon and number of particles, 
        # we run corresponding controller parameters and 
        # save the results in sub-folder named H_#_N_#. This 
        # mode is used for benchmarking purposes after tuning
        #############################################
        search_param_keys = []
        search_param_vals = []
        for k in policy_params:
            if isinstance(policy_params[k], list) and \
                k not in ['filter_coeffs', 'horizon', 'num_cpu', 'particles_per_cpu', 'tune_keys']:
                search_param_keys.append(k)
                search_param_vals.append(policy_params[k])
        policy_params.pop('tune_keys', None)
                
        assert len(horizons) == len(num_particles), \
                "Please provide correct number of parameters"
        for i in range(len(horizons)):
            s = "H_" + str(horizons[i]) + "_N_" + str(num_particles[i])
            SUB_LOG_DIR = os.path.join(main_dir + "/" + s)
            if not os.path.exists(SUB_LOG_DIR):
                sub_logger = helpers.get_logger(controller_name + "_" + s, SUB_LOG_DIR, 'debug')
            policy_params['horizon'] = horizons[i]
            policy_params['num_particles'] = num_particles[i]
            
            search_param_tuple = [v[i] for v in search_param_vals]
            # search_param_tuple = search_param_vals[i]
            for j in range(len(search_param_tuple)):
                policy_params[search_param_keys[j]] = search_param_tuple[j]
            policy_params['particles_per_cpu'] = int(policy_params['num_particles']/policy_params['num_cpu'])
            sub_logger.info('Running parameters')
            sub_logger.info(policy_params)

            trajectories, avg_reward, reward_std, success_metric = gather_trajectories(controller_name,
                                                                                        deepcopy(policy_params), 
                                                                                        exp_params['n_episodes'], 
                                                                                        exp_params['max_ep_length'], 
                                                                                        exp_params['seed'],
                                                                                        num_cpu,
                                                                                        sub_logger)

            if args.dump_vids:
                print('Dumping videos')
                helpers.dump_videos(env=env, trajectories=trajectories, frame_size=(1280, 720), 
                                    folder=SUB_LOG_DIR, filename='vid_traj_', camera_name=None,
                                    device_id=1)
            if exp_params['render']:
                _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
                helpers.render_trajs(env, trajectories, n_times=10)
            
            sub_logger.info('Success metric = {0}, Average reward = {1}, Std. Reward = {2}'.format(success_metric, 
                                                                                                   avg_reward, 
                                                                                                   reward_std))
            
            
            sub_logger.info('Dumping trajectories')
            pickle.dump(trajectories, open(SUB_LOG_DIR+"/trajectories.p", 'wb'))
                    
    else:
        raise NotImplementedError('Unidentified job mode. Must be either "tune" or "sweep" ')

    env.close() 


if __name__ == '__main__':
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    for i, controller in enumerate(args.controllers):
        timeit.reset()
        LOG_DIR = os.path.join(os.path.abspath(args.save_dir) + "/" + exp_params['env_name'] + "/" + date_time, controller)
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        main(controller, LOG_DIR)

