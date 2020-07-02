import argparse
from copy import deepcopy
from datetime import datetime
import gym
import numpy as np
import os
import sys
import tqdm
import yaml 

from mjmpc.control.softqmpc.algs import SAC
from mjmpc.envs import GymEnvWrapper
from mjmpc.utils import helpers
import mj_envs
from stable_baselines3.sac import MlpPolicy

parser = argparse.ArgumentParser(description='Run MPC algorithm on given environment')
parser.add_argument('--config', type=str, help='yaml file with experiment parameters')
parser.add_argument('--save_dir', type=str, default='/tmp', help='folder to save data in')
parser.add_argument('--dump_vids', action='store_true', help='flag to dump video of episodes')
parser.add_argument('--load_file', type=str, required=True, help='directory with weight file')
args = parser.parse_args()

#Load experiment parameters from config file
with open(args.config) as file:
    exp_params = yaml.load(file, Loader=yaml.FullLoader)

#Create the main environment
env_name  = exp_params['env_name']
env = gym.make(env_name)
env = GymEnvWrapper(env)
env.real_env_step(True)

#Create logger
date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_dir = args.save_dir + "/" + exp_params['env_name'] + "/" + date_time + "/SAC/test/" 
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger = helpers.get_logger("sac" + "_" + exp_params['env_name'], log_dir, 'debug')
logger.info(exp_params)
exp_params['tensorboard_log'] = log_dir

render = exp_params['render']
num_test_episodes = exp_params['num_test_episodes']
test_seed = exp_params['test_seed']
total_timesteps = exp_params['total_timesteps']
exp_params.pop('env_name', None)
exp_params.pop('render', None)
exp_params.pop('num_test_episodes', None)
exp_params.pop('test_seed', None)
exp_params.pop('total_timesteps', None)

#Define model and train
model = SAC(MlpPolicy, env, **exp_params)
model = SAC.load(args.load_file)


#Commence testing
#Main data collection loop
ep_rewards = np.array([0.] * num_test_episodes)
trajectories = []
for i in tqdm.tqdm(range(num_test_episodes)):
    #seeding to enforce consistent episodes
    episode_seed = test_seed + i*12345
    obs = env.reset(seed=episode_seed)
    
    #Collect data from interactions with environment
    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    for t in tqdm.tqdm(range(exp_params['max_ep_length'])): 
        curr_state = deepcopy(env.get_env_state())
        action, value = model.predict(obs)

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

if render:
    _ = input("Press enter to display optimized trajectories (will be played 3 times) : ")
    helpers.render_trajs(env, trajectories, n_times=3)

env.close()
