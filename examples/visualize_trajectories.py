"""
Script for loading pickle file with trajectories and 
animating them.  
"""

import gym
import pickle
import click 
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import helpers

DESC = '''
Helper script to visualize optimized trajectories (list of trajectories in format).\n
USAGE:\n
    $ python visualize_trajectories.py --file path_to_file.pickle --env_name name_of_env --repeat 100\n
'''
@click.command(help=DESC)
@click.option('--env_name', type=str, prompt='Name of environment', help='name of environment to render', required= True)
@click.option('--file', type=str, prompt='Pickle file with trajectories', help='pickle file with trajectories', required= True)
@click.option('--repeat', type=int, prompt='Number of times to render(default=10)', help='number of times to play trajectories', default=10)

def main(file, env_name, repeat):
    env = gym.make(env_name)
    trajectories = pickle.load(open(file, 'rb'))
    helpers.render_trajs(env, trajectories, n_times=repeat)

if __name__ == '__main__':
	main()