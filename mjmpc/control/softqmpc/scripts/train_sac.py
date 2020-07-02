import argparse
from datetime import datetime
import gym
import numpy as np
import itertools
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from mjmpc.control.softqmpc.algs.sac import SAC, ReplayMemory
from mjmpc.envs import GymEnvWrapper
from mjmpc.utils import helpers
import mj_envs

# parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# parser.add_argument('--env-name', default="HalfCheetah-v2",
#                     help='Mujoco Gym environment (default: HalfCheetah-v2)')
# parser.add_argument('--policy', default="Gaussian",
#                     help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
# parser.add_argument('--eval', type=bool, default=True,
#                     help='Evaluates a policy a policy every 10 episode (default: True)')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor for reward (default: 0.99)')
# parser.add_argument('--tau', type=float, default=0.005, metavar='G',
#                     help='target smoothing coefficient(τ) (default: 0.005)')
# parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
#                     help='learning rate (default: 0.0003)')
# parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
#                     help='Temperature parameter α determines the relative importance of the entropy\
#                             term against the reward (default: 0.2)')
# parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
#                     help='Automaically adjust α (default: False)')
# parser.add_argument('--seed', type=int, default=123456, metavar='N',
#                     help='random seed (default: 123456)')
# parser.add_argument('--batch_size', type=int, default=256, metavar='N',
#                     help='batch size (default: 256)')
# parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
#                     help='maximum number of steps (default: 1000000)')
# parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
#                     help='hidden size (default: 256)')
# parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
#                     help='model updates per simulator step (default: 1)')
# parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
#                     help='Steps sampling random actions (default: 10000)')
# parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
#                     help='Value target update per no. of updates per step (default: 1)')
# parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
#                     help='size of replay buffer (default: 10000000)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
# args = parser.parse_args()

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
# env.seed(exp_params['seed'])

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

exp_params["cuda"] = args.cuda
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, exp_params)

#Tensorboard
# writer = SummaryWriter('{}/{}_{}'.format(args.policy, "autotune" if exp_params['automatic_entropy_tuning'] else ""))
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(exp_params["replay_size"])

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    episode_seed = exp_params['seed'] + i_episode*12345
    exp_params['seed'] = episode_seed
    state = env.reset(seed=episode_seed)

    while not done:
        if exp_params["start_steps"] > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.get_action(state)  # Sample action from policy

        if len(memory) > exp_params["batch_size"]:
            # Number of updates per step in environment
            for i in range(exp_params["updates_per_step"]):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, exp_params["batch_size"], updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == exp_params["max_ep_length"] else float(not done)
        #Done = true is max_ep_length reached
        done = done or (episode_steps % exp_params["max_ep_length"] == 0)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > exp_params["num_steps"]:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2)))

    if i_episode % 10 == 0 and exp_params["eval"] is True:
        avg_reward = 0.
        episodes = 10
        for j in range(episodes):
            seed = exp_params['test_seed'] + j*12345
            state = env.reset(seed=seed)
            episode_reward = 0
            done = False
            for t in range(exp_params['max_ep_length']):
                action = agent.get_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward.item()

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

model_dir = log_dir + "/models/"
os.makedirs(model_dir)
agent.save_model(env_name, suffix="", actor_path=model_dir+"actor", critic_path=model_dir+"critic")
env.close()

