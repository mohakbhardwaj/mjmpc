import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import itertools
import os
import sys
sys.path.insert(0, '../../../../')
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml

from mjmpc.control.softqmpc.algs.sac import SAC, ReplayMemory
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

#Create logger
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/SAC" 
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger("sac" + "_" + exp_params['env_name'], log_dir, 'debug')
logger.info(exp_params)

# Agent
exp_params["cuda"] = args.cuda
agent = SAC(env.observation_space.shape[0], env.action_space, exp_params)
agent.load_model(args.load_dir+"/actor", args.load_dir+"/critic")


#Tensorboard
# writer = SummaryWriter('{}/{}_{}'.format(args.policy, "autotune" if exp_params['automatic_entropy_tuning'] else ""))
# writer = SummaryWriter(log_dir)

trajectories = []
n_episodes = exp_params['num_test_episodes']
ep_rewards = np.array([0.] * n_episodes)
for i in tqdm.tqdm(range(n_episodes)):
    seed = exp_params['test_seed'] + i*12345
    curr_obs = env.reset(seed=seed)
    episode_reward = 0
    done = False
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    for _ in tqdm.tqdm(range(exp_params['max_ep_length'])):   
        curr_state = deepcopy(env.get_env_state())
        action = agent.get_action(curr_obs, evaluate=True)
        next_obs, reward, done, info = env.step(action)
        
        observations.append(curr_obs); actions.append(action)
        rewards.append(reward); dones.append(done)
        infos.append(info); states.append(curr_state)
        ep_rewards[i] += reward
        curr_obs = next_obs
    
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
logger.record_tabular("NumEpisodes", n_episodes)
logger.dump_tabular()

if args.dump_vids:
    print('Dumping videos')
    helpers.dump_videos(env=env, trajectories=trajectories, frame_size=(1280, 720), 
                        folder=log_dir, filename='vid_traj_', camera_name=None,
                        device_id=1)

if exp_params['render']:
    _ = input("Press enter to display optimized trajectories : ")
    helpers.render_trajs(env, trajectories, n_times=1)

env.close()

