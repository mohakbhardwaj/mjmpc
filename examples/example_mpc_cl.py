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
import torch
import pickle
import tqdm
import yaml
os.environ['KMP_DUPLICATE_LIB_OK']='True'
try:
    import mj_envs
except ImportError:
    print('mj_envs not found. Will not be able to run its configs')
import mjmpc.envs
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import TorchModelVecEnv
from mjmpc.utils import LoggerClass, timeit, helpers
from mjmpc.policies import MPCPolicy, LinearGaussianPolicy

gym.logger.set_level(40)
parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--dyn_randomize_config', type=str, help='yaml file with dynamics randomization parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
# parser.add_argument('--load_dir', type=str, default='/tmp', help='folder to load model from')
parser.add_argument('--controller', type=str, default='mppi', help='controller to run')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes' )
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)
if args.dyn_randomize_config is not None:
    with open(args.dyn_randomize_config) as file:
        dynamics_rand_params = yaml.load(file, Loader=yaml.FullLoader)    
else:
    dynamics_rand_params=None

torch.manual_seed(0)
np.random.seed(0)

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

#unpack params and create policy params
controller_name = args.controller
policy_params = exp_params[controller_name]
policy_params['d_obs'] = env.d_obs
policy_params['d_state'] = env.d_state
policy_params['d_action'] = env.d_action
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
print(policy_params)
if 'num_cpu' and 'particles_per_cpu' in policy_params:
    policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']

num_cpu = policy_params['num_cpu']
n_episodes = exp_params['n_episodes']
base_seed = exp_params['seed']
ep_length = exp_params['max_ep_length']
actor_params = exp_params['actor_params']

#Create vectorized environments for MPC simulations
# if dynamics_rand_params is not None:
    # default_params, randomized_params = sim_env.randomize_dynamics(dynamics_rand_params, base_seed=exp_params['seed'])
if actor_params['actor_type'] == "linear_gaussian":
    policy = LinearGaussianPolicy(env.d_obs, env.d_action, actor_params['min_log_std'], actor_params['init_log_std'])
policy_params['policy'] = policy
policy_params['baseline'] = None

sim_env = TorchModelVecEnv([make_env for i in range(num_cpu)], policy)
def rollout_fn(mode='mean', noise=None):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return sequence of costs. The controller is 
    agnostic of how the rollouts are generated.
    """
    obs_vec, act_vec, act_infos, rew_vec, done_vec, next_obs_vec, infos = sim_env.rollout(policy_params['num_particles'], policy_params['horizon'], mode, noise)
    #we assume environment returns rewards, but controller needs costs
    return obs_vec, act_vec, act_infos, -1.0*rew_vec, done_vec, next_obs_vec, infos

policy_params.pop('particles_per_cpu', None)
policy_params.pop('num_cpu', None)

#Create logger
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/" + controller_name
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger(controller_name + "_" + exp_params['env_name'], log_dir, 'debug')
ep_rewards = np.array([0.] * n_episodes)
trajectories = []
logger.info(exp_params[controller_name])


#create MPC policy and set appropriate functions
policy = MPCPolicy(controller_type=controller_name,
                    param_dict=policy_params, batch_size=1) #Only batch_size=1 is supported for now
policy.controller.set_sim_state_fn = sim_env.set_env_state
policy.controller.rollout_fn = rollout_fn
policy.controller.get_sim_obs_fn = sim_env.get_obs

#Main data collection loop
timeit.start('start_'+controller_name)
for i in tqdm.tqdm(range(n_episodes)):
    #seeding to enforce consistent episodes
    episode_seed = base_seed + i*12345
    policy_params['seed'] = episode_seed
    env.reset(seed=episode_seed)
    sim_env.reset()
    policy.controller.reset()

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
        print(action, reward)
        ep_rewards[i] += reward
    
    traj = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        env_infos=helpers.stack_tensor_dict_list(infos),
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
logger.record_tabular("AverageReward", average_reward)
logger.record_tabular("StdReward", reward_std)
logger.record_tabular("SuccessMetric", success_metric)
logger.record_tabular("NumEpisodes", exp_params['n_episodes'])
if 'num_particles' in policy_params:
    logger.record_tabular("NumParticles", policy_params['num_particles'])

logger.dump_tabular()

if args.dump_vids:
    print('Dumping videos')
    helpers.dump_videos(env=env, trajectories=trajectories, frame_size=(1280, 720), 
                        folder=log_dir, filename='vid_traj_', camera_name=None,
                        device_id=1)

if exp_params['render']:
    _ = input("Press enter to display optimized trajectories (will be played 3 times) : ")
    helpers.render_trajs(env, trajectories, n_times=3)

env.close()