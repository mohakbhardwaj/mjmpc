#!/usr/bin/env python
"""Cross Entropy Method for MPC

Author - Mohak Bhardwaj
Date - Jan 12, 2020
TODO:
 - Make it a work for batch of start states 
"""
from .controller import Controller, scale_ctrl, generate_noise, cost_to_go
import copy
import numpy as np
from scipy.signal import savgol_filter
import scipy.stats
import scipy.special



class CEM(Controller):
    def __init__(self,
                 horizon,
                 init_cov,
                 elite_frac,
                 num_particles,
                 step_size,
                 gamma,
                 n_iters,
                 num_actions,
                 action_lows,
                 action_highs,
                 set_state_fn,
                 rollout_fn,
                 terminal_cost_fn=None,
                 rollout_callback=None,
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):



        super(CEM, self).__init__(num_actions,
                                   action_lows, 
                                   action_highs,  
                                   set_state_fn, 
                                   terminal_cost_fn,
                                   batch_size)
        self.horizon = horizon
        self.init_cov = init_cov  # cov for sampling actions
        self.elite_frac = elite_frac
        self.num_particles = num_particles
        self.step_size = step_size  # step size for mean and covariance
        self.gamma = gamma  # discount factor
        self.n_iters = n_iters  # number of iterations of optimization per timestep
        self.rollout_fn = rollout_fn
        self.rollout_callback = rollout_callback
        self.filter_coeffs = filter_coeffs
        self.seed = seed
        self.num_elite = int(self.num_particles * self.elite_frac)

        self.mean_action = np.zeros(shape=(horizon, num_actions))
        self.cov_action = self.init_cov * np.ones(shape=(horizon, num_actions))
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(horizon, 1)
        self._val = 0.0  # optimal value for current state
        self.num_steps = 0


    def _get_next_action(self, state, sk, act_seq):
        next_action = self.mean_action[0]
        return next_action.reshape(self.num_actions, )

    def _sample_actions(self):
        delta = generate_noise(np.sqrt(self.cov_action[:, :, np.newaxis]), self.filter_coeffs,
                                       shape=(self.horizon, self.num_actions, self.num_particles))
        act_seq = self.mean_action[:, :, np.newaxis] + delta
        return act_seq

    def _update_moments(self, sk, act_seq):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """

        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.matmul(delta, w[:, :, None]).squeeze(axis=-1)

        if np.any(np.isnan(self.mean_action)):
            print('warning: nan in mean_action or cov_action...resetting the controller')
            self.reset()

    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        self.mean_action[:-1] = self.mean_action[1:]
        self.mean_action[-1] = np.random.normal(0, self.init_cov, self.num_actions)


    def _calc_val(self, state):
        """
            Calculate (soft) value or free energy of state under current
            control distribution
        """

        return 0.

    def reset(self):
        self.num_steps = 0
        self.mean_action = np.zeros(shape=(self.horizon, self.num_actions))
        self.cov_action = self.init_cov * np.ones(shape=(horizon, num_actions))
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (self.horizon - 1)).reshape(self.horizon, 1)
        self._val = 0.0
