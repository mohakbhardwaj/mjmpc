import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import os
import sys
import torch
import tqdm
import yaml 

from mjmpc.control.softqmpc.algs.sac import SAC
from mjmpc.envs.vec_env import TorchModelVecEnv
from mjmpc.policies import MPCPolicy
from mjmpc.envs import GymEnvWrapper
from mjmpc.utils import helpers
import mj_envs

parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--load_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes' )
parser.add_argument('--cuda', action='store_true', help='use cuda or not')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

torch.manual_seed(exp_params['seed'])
np.random.seed(exp_params['seed'])

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

#Create logger
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/random_shooting_nn/test/" 
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger("sac" + "_" + exp_params['env_name'], log_dir, 'debug')
logger.info(exp_params)

# Agent
exp_params["cuda"] = args.cuda
agent = SAC(env.observation_space.shape[0], env.action_space, exp_params)
agent.load_model(args.load_dir+"/actor", args.load_dir+"/critic")
model = agent.policy

test_seed = exp_params['test_seed']
policy_params = exp_params['random_shooting_nn']
policy_params['d_obs'] = env.d_obs
policy_params['d_state'] = env.d_state
policy_params['d_action'] = env.d_action
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
# policy_params['model'] = model

# if 'num_cpu' and 'particles_per_cpu' in policy_params:
policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']
num_particles = policy_params['num_particles']
horizon = policy_params['horizon']
num_cpu = policy_params['num_cpu']
policy_params.pop('particles_per_cpu', None)
policy_params.pop('num_cpu', None)

#Create vectorized environments for MPC simulations
sim_env = TorchModelVecEnv([make_env for i in range(num_cpu)], model)  

def rollout_fn(mode='mean', noise=None):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return sequence of costs. The controller is 
    agnostic of how the rollouts are generated.
    """
    obs_vec, act_vec, log_prob_vec, rew_vec, done_vec, infos = sim_env.rollout(num_particles, horizon, mode, noise)
    #we assume environment returns rewards, but controller needs costs
    return obs_vec, act_vec, log_prob_vec, -1.0*rew_vec, done_vec, infos

#Commence testing
#Main data collection loop
num_test_episodes = exp_params['num_test_episodes']
ep_rewards = np.array([0.] * num_test_episodes)
trajectories = []
for i in tqdm.tqdm(range(num_test_episodes)):
    #seeding to enforce consistent episodes
    episode_seed = test_seed + i*12345
    obs = env.reset(seed=episode_seed)
    
    #create MPC policy and set appropriate functions
    policy = MPCPolicy(controller_type='random_shooting_nn',
                        param_dict=policy_params, batch_size=1)
    policy.controller.set_sim_state_fn = sim_env.set_env_state
    # policy.controller.get_sim_state_fn = sim_env.get_env_state
    # policy.controller.sim_step_fn = sim_env.step
    # policy.controller.sim_reset_fn = sim_env.reset
    # policy.controller.get_sim_obs_fn = sim_env.get_obs
    policy.controller.rollout_fn = rollout_fn

    #Collect data from interactions with environment
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    for t in tqdm.tqdm(range(exp_params['max_ep_length'])): 
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
        env_infos=helpers.stack_tensor_dict_list(infos),
        states=states
    )
    trajectories.append(traj)

success_metric = env.env.unwrapped.evaluate_success(trajectories)
average_reward = np.average(ep_rewards)
reward_std = np.std(ep_rewards)

#Display logs on screen and save in txt file
logger.info('Avg. reward = {0}, Std. Reward = {1}, Success Metric = {2}'.format(average_reward, reward_std, success_metric))

#Can also dump data to csv once done
logger.record_tabular("AverageReward", average_reward)
logger.record_tabular("StdReward", reward_std)
logger.record_tabular("SuccessMetric", success_metric)
logger.record_tabular("NumEpisodes", num_test_episodes)
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
