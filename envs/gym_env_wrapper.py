"""
Declares a class that takes a gym environment
as input and implements necessary functions for MPC rollouts

Author: Mohak Bhardwaj
Date: January 9, 2020
"""
import numpy as np
from copy import deepcopy

class GymEnvWrapper():
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.gym_env.reset()
        observation, _reward, done, _info = self.gym_env.step(np.zeros(self.gym_env.action_space.low.shape))
        assert not done, ""
        self.d_obs = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        super(GymEnvWrapper, self).__init__()

    def step(self, u):
        return self.gym_env.step(u)
 
    def set_env_state(self, state_dict: dict):
        """Set the state of environment from dictionary"""
        self.gym_env.set_env_state(state_dict)
 
    def get_env_state(self) -> dict:
        """
        Return dictionary of the full environment state
        """
        return self.gym_env.get_env_state()
 
    def get_reward(self, state, action, next_state):
        '''
        return the reward function for the transition
        '''
        pass
 
    def set_params(self, param_dict: dict):
        """
        set the dynamics parameters of the environment from 
        input dictionary
        """
        pass    
    
    def rollout(self, u_vec: np.ndarray):
        """
        Given batch of action sequences, we perform rollouts 
        and return resulting observations, rewards etc.
        :param u_vec: np.ndarray of shape [batch_size, n_steps, d_action]
        :return:
            obs_vec: np.ndarray [batch_size, n_steps, d_obs]
            state_vec: np.ndarray [batch_size, n_steps, d_state]
            rew_vec: np.ndarray [batch_size, n_steps, 1]
            done_vec: np.ndarray [batch_size, n_steps, 1]
            info: dict
        """
        batch_size, n_steps, d_action = u_vec.shape
        obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
        # state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
        rew_vec = np.zeros((batch_size, n_steps))
        done_vec = np.zeros((batch_size, n_steps))
        curr_state = deepcopy(self.get_env_state())

        for b in range(batch_size):
            #Set the state to the current state
            self.set_env_state(curr_state)
            #Rollout for t steps and store results
            for t in range(n_steps):
                u_curr = u_vec[b, t, :]
                obs, rew, done, _ = self.step(u_curr)
                obs_vec[b, t, :] = obs.copy().reshape(self.d_obs,)
                # state_vec[:, t, :] = self.state.copy()
                rew_vec[b, t] = rew
                done_vec[b, t] = done

        return obs_vec, rew_vec, done_vec, {}
    
    def seed(self, seed=None):
        return self.gym_env.seed(seed)
    
    def reset(self):
        return self.gym_env.reset()

    def render(self):
        self.gym_env.render()