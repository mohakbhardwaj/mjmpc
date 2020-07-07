import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import os
import sys
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml 

from mjmpc.control.softqmpc.algs.sac import SAC
from mjmpc.control.softqmpc.algs import SACMPC
from mjmpc.control.softqmpc.models import GaussianPolicy
from mjmpc.envs.vec_env import TorchModelVecEnv
from mjmpc.envs import GymEnvWrapper
from mjmpc.utils import helpers
import mj_envs

parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--load_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes')
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
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/sac_mpc/test/" 
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger("sac" + "_" + exp_params['env_name'], log_dir, 'debug')
logger.info(exp_params)
with open(log_dir+"/exp_params.yml", "w") as f:
    yaml.dump(exp_params, f)

#Create Tensorboard
writer = SummaryWriter(log_dir)

# policy = GaussianPolicy(env.d_obs, env.d_action, 256, env.action_space.high, env.action_space.low)
# Agent
exp_params["cuda"] = args.cuda
agent = SAC(env.observation_space.shape[0], env.action_space, exp_params)
agent.load_model(args.load_dir+"/actor", args.load_dir+"/critic")
policy = agent.policy
critic = agent.critic

# Function to create vectorized environments for controller simulations
def make_env():
    gym_env = gym.make(env_name)
    rollout_env = GymEnvWrapper(gym_env)
    rollout_env.real_env_step(False)
    return rollout_env

sim_env = TorchModelVecEnv([make_env for i in range(num_cpu)], policy)
def rollout_fn(mode='mean', noise=None):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return sequence of costs. The controller is 
    agnostic of how the rollouts are generated.
    """
    obs_vec, act_vec, log_prob_vec, rew_vec, done_vec, next_obs_vec, infos = sim_env.rollout(num_particles, horizon, mode, noise)
    #we assume environment returns rewards, but controller needs costs
    return obs_vec, act_vec, log_prob_vec, -1.0*rew_vec, done_vec, next_obs_vec, infos

#MPC Agent
exp_params['policy'] = policy
exp_params['critic'] = critic
exp_params['rollout_fn'] = rollout_fn
exp_params['set_sim_state_fn'] = sim_env.set_env_state
exp_params['get_sim_obs_fn'] = sim_env.get_sim_obs
mpc_agent = SACMPC(exp_params)

# Training Loop
total_numsteps = 0
trajectories = []

for i_episode in itertools.count(1):
    #seeding to enforce consistent episodes
    episode_reward = 0
    episode_steps = 0
    done = False
    episode_seed = test_seed + i*12345
    exp_params['seed'] = episode_seed
    curr_obs = env.reset(seed=episode_seed)
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    
    while not done:
        curr_state = deepcopy(env.get_env_state())
        action, value = mpc_agent.get_action(curr_state, calc_val=False)
        agent_infos = mpc_agent.get_agent_infos()

        writer.add_scalar('loss/critic_1', agent_infos["critic_1_loss"], updates)
        writer.add_scalar('loss/critic_2', agent_infos["critic_2_loss"], updates)
        writer.add_scalar('loss/policy', agent_infos["policy_loss"], updates)
        writer.add_scalar('loss/entropy_loss', agent_infos["ent_loss"], updates)
        writer.add_scalar('entropy_temperature/alpha', agent_infos["alpha"], updates)

        next_obs, reward, done, info = env.step(action)
        episode_steps += 1
        total_numstesp += 1
        episode_reward += reward

        observations.append(curr_obs); actions.append(action)
        rewards.append(reward); dones.append(done)
        infos.append(info); states.append(curr_state)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == exp_params["max_ep_length"] else float(not done)
        #Done = true if max_ep_length reached
        done = done or (episode_steps % exp_params["max_ep_length"] == 0)

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
    
    if total_numsteps > exp_params["num_steps"]:
        break

    writer.add_scalar('train/episode_reward', episode_reward, i_episode)
    logger.info("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2)))

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
