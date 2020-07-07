import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import itertools
import os
import sys
sys.path.insert(0, '../../../../')
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from mjmpc.control.softqmpc.algs.sac import SAC, ReplayMemory
from mjmpc.envs import GymEnvWrapper
from mjmpc.utils import helpers
import mj_envs

parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes' )
parser.add_argument('--cuda', action='store_true', help='use cuda or not')

args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(exp_params['env_name'])
torch.manual_seed(exp_params['seed'])
np.random.seed(exp_params['seed'])
# random.seed(exp_params['seed'])

#Create the main environment
env_name  = exp_params['env_name']
env = gym.make(env_name)
env = GymEnvWrapper(env)
env.real_env_step(True)
env.seed(exp_params['seed'])
env.action_space.seed(exp_params['seed'])


#Create logdir, logger and save exp params
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/SAC" 
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger("sac" + "_" + exp_params['env_name'], log_dir, 'debug')
logger.info(exp_params)
with open(log_dir+"/exp_params.yml", "w") as f:
    yaml.dump(exp_params, f)

exp_params["cuda"] = args.cuda
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, exp_params)

#Tensorboard
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(exp_params["replay_size"])

# Training Loop
total_numsteps = 0
updates = 0
train_trajectories = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    episode_seed = exp_params['seed'] + i_episode*12345
    # exp_params['seed'] = episode_seed
    curr_obs = env.reset(seed=episode_seed)
    #Collect data from interactions with environment 
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []

    while not done:
        # curr_state = deepcopy(env.get_env_state())
        if exp_params["start_steps"] > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.get_action(curr_obs)  # Sample action from policy

        if len(memory) > exp_params["batch_size"]:
            # Number of updates per step in environment
            for i in range(exp_params["updates_per_step"]):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, exp_params["batch_size"], updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                updates += 1

        next_obs, reward, done, info = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        observations.append(curr_obs); actions.append(action)
        rewards.append(reward); dones.append(done)
        # states.append(curr_state); 
        infos.append(info)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(curr_obs, action, reward, next_obs, mask) # Append transition to memory

        curr_obs = next_obs
    
    traj = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        # states=states,
        env_infos = helpers.stack_tensor_dict_list(infos)
    )
    train_trajectories.append(traj)
    ep_success = 0.0 #env.env.unwrapped.evaluate_success([traj])

    if total_numsteps > exp_params["num_steps"]:
        break

    writer.add_scalar('train/episode_reward', episode_reward, i_episode)
    writer.add_scalar('train/episode_success', ep_success, i_episode)
    logger.info("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2))) #success: {}

    if i_episode % 10 == 0 and exp_params["eval"] is True:
        avg_reward = 0.
        episodes = 10
        for j in range(episodes):         
            test_episode_seed = exp_params['test_seed'] + j*12345
            curr_obs = env.reset(seed=test_episode_seed)
            episode_reward = 0
            done = False
            for t in range(exp_params['max_ep_length']):
                action = agent.get_action(curr_obs, evaluate=True)

                next_obs, reward, done, info = env.step(action)
                episode_reward += reward.item()

                curr_obs = next_obs
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        logger.info("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

model_dir = log_dir + "/models/"
os.makedirs(model_dir)
agent.save_model(env_name, suffix="", actor_path=model_dir+"actor", critic_path=model_dir+"critic")
env.close()

