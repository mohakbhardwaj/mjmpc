import gym
from gym import spaces
import numpy as np
import random

from envs.assets.maze_layouts import maze_layouts


class ContinualParticleMaze(gym.Env):
    """
    Particle maze for continual learning. Mazes should be set by a runner class
    of the environment; see agent.py for an example.
    """

    def __init__(self, dense=True):
        grid_name='blank1'
        self.dense = dense
        self.dt = 0.1
        self.num_collision_steps = 10

        self.observation_space = spaces.Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1])
        )
        self.action_space = spaces.Box(
            np.array([-1, -1]), np.array([1, 1])
        )

        self.x = np.zeros(2)
        self.goal = np.zeros(2)
        self.reset_grid(grid_name)

    def step(self, action):
        """
        Action is a clipped dx. Must respect collision with walls.
        """

        # If agent is in a wall (shouldn't happen), reset
        ind = self.get_index(self.x)
        if self.grid[ind[0], ind[1]]:
            self.reset_agent()

        rewards, costs = self.get_rew(), 0

        # Action movement and collision detection
        action = np.clip(action, -1, 1)
        ddt = self.dt / self.num_collision_steps

        for _ in range(self.num_collision_steps):
            x_new = self.x + action * ddt
            ind = self.get_index(x_new)

            # If in wall, back up (reduces problems with learning)
            if self.grid[ind[0], ind[1]]:
                costs += 1
                self.x -= action * ddt
                break
            else:
                self.x = x_new

        self.x = np.clip(self.x, -1, 1)

        return self.get_obs(), rewards - costs, False, {}

    def reset_agent(self, mode=None):
        """
        Reset the agent's position (should be used rarely in lifelong
        learning).
        """
        if self.start_ind is not None:
            # Spawn the agent at the start state
            self.x = self.get_coords(self.start_ind)
        else:
            # Spawn the agent not too close to the goal
            self.x = self.get_random_pos(self.grid_free_index)
            while np.sum(np.square(self.x - self.g[0, :])) < 0.5:
                self.x = self.get_random_pos(self.grid_free_index)

    def reset_grid(self, grid_name):
        """
        Changes the current grid layout, i.e. the walls of the maze. The
        agent's position is not reset, unless it would be placed inside of a
        wall by the change, in which case it spawns in the set start position.
        """
        self.grid = maze_layouts[grid_name]
        self.grid = self.grid.replace('\n', '')
        self.grid_size = int(np.sqrt(len(self.grid)))

        GS = [self.grid_size, self.grid_size]

        self.grid_chars = (np.array(list(self.grid)) != 'S').reshape(GS)
        self.start_ind = np.argwhere(self.grid_chars == False)

        # Check if there is a specified start location S
        if self.start_ind.shape[0] > 0:
            self.start_ind = self.start_ind[0]
        else:
            self.start_ind = None

        # Get the goal location
        self.grid_chars = (np.array(list(self.grid)) != 'G').reshape(GS)
        self.goal_ind = np.argwhere(self.grid_chars == False)
        self.goal_ind = self.goal_ind[0]
        self.goal = self.get_coords(self.goal_ind)

        self.grid = self.grid.replace('S', ' ')
        self.grid = self.grid.replace('G', ' ')

        self.grid = (np.array(list(self.grid)) != ' ').reshape(GS)
        self.grid_wall_index = np.argwhere(self.grid == True)
        self.grid_free_index = np.argwhere(self.grid != True)

        # Reset the agent only if it is stuck in the wall
        cur_ind = self.get_index(self.x)
        if self.grid[cur_ind[0], cur_ind[1]]:
            self.reset()

    def reset(self):
        """
        Only called at initialization of environment (use reset_agent and
        reset_goal as needed elsewhere).
        """
        self.reset_agent()
        return self.get_obs()

    def get_obs(self):
        """
        Observation is the coordinates of the agent and the goal.
        """
        return np.concatenate([self.x, self.goal])

    def get_rew(self):
        """
        Reward for the agent, based on the current state. Environment supports
        dense and sparse reward variants.
        """
        if self.dense:
            return -np.linalg.norm(self.x - self.goal)
        else:
            return 1 if np.linalg.norm(self.x - self.goal) < .1 else 0

    def get_coords(self, index):
        """
        Convert indices of grid into coordinates.
        """
        return ((index + 0.5) / self.grid_size) * 2 - 1

    def get_index(self, coords):
        """
        Convert coordinates to indices of grid.
        """
        return np.clip((((coords + 1) * 0.5) * (self.grid_size)) + 0,
                       0, self.grid_size-1).astype(np.int8)

    def get_env_state(self):
        """
        State is the same as observation
        """
        state_dict = {'x': self.x.copy(),
                      'goal': self.goal.copy()}
        return state_dict.copy()
    
    def set_env_state(self, state_dict):
        self.x = state_dict['x'].copy()
        self.goal = state_dict['goal'].copy()
    
    def evaluate_success(self, paths):
        return 0.0

