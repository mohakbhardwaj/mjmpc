#!/usr/bin/env python
"""
Defines abstract class for specifying environments that
inherits from OpenAI gym environments
"""
from abc import ABC, abstractmethod
import gym
import numpy as np


class Env(gym.Env, ABC):
    def __init__(self, batch_size, env_name, d_state, d_action, d_obs):
        self.batch_size = batch_size
        self.env_name = env_name
        self.d_state = d_state
        self.d_action = d_action
        self.d_obs = d_obs
        self.dyn_params = None
        

    @abstractmethod
    def step(self, u):
        pass

    @abstractmethod
    def rollout(self, u_vec):
        pass

    @abstractmethod
    def set_state(self, state_vec: np.ndarray):
        """Set the state of environment given numpy array"""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Return numpy array of the full environment state
        """
        pass

    @abstractmethod
    def get_reward(self, state, action, next_state):
        '''
        return the reward function for the transition
        '''
        pass

    @abstractmethod
    def set_params(self, param_dict):
        """
        Set the dynamics parameters of the environment
        from input dictionary and set self.dyn_params = param_dict
        """
        pass
        
    @property
    def dynamics_params(self):
        return self.dyn_params