#!/usr/bin/env python
"""
A version of DMD-MPC with Gaussian sampling, 
exponential utility cost function and 
covariance adaptation 
Author - Mohak Bhardwaj
Date - Jan 19, 2020
 
"""
from .controller import Controller, GaussianMPC, cost_to_go
import copy
import numpy as np
from scipy.signal import savgol_filter
import scipy.stats
import scipy.special

class DMDMPC(GaussianMPC):
    def __init__(self,
                 horizon,
                 init_cov,
                 min_cov, 
                 prior_cov,
                 beta,
                 base_action,
                 lam,
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

        super(DMDMPC, self).__init__(num_actions,
                                   action_lows, 
                                   action_highs,
                                   horizon,
                                   np.array(init_cov),
                                   np.zeros(shape=(horizon, num_actions)),
                                   base_action,
                                   num_particles,
                                   gamma,
                                   n_iters,
                                   step_size, 
                                   filter_coeffs, 
                                   set_state_fn, 
                                   rollout_fn,
                                   rollout_callback,
                                   terminal_cost_fn,
                                   batch_size,
                                   seed)
        self.lam = lam
        self.beta = beta
        self.min_cov = min_cov
        self.prior_cov = prior_cov


    def _update_distribution(self, costs, act_seq):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        delta = act_seq - self.mean_action[None, :, :]
        w = self._exp_util(costs)
        weighted_seq = w * act_seq.T
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.sum(weighted_seq.T, axis=0) 

        self.cov_action = (1.0 - self.step_size) * self.cov_action +\
                                self.step_size * np.matmul(delta ** 2, w)
        self.cov_action = np.clip(self.cov_action, self.min_cov, None)

        
        
    def _exp_util(self, costs):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        # #calculate soft-max
        w = np.exp(-(traj_costs - np.min(traj_costs)) / self.lam)
        w /= np.sum(w) + 1e-6  # normalize the weights
        return w
    
    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        super()._shift()
        if self.update_cov and self.beta > 0.0:
            update = self.cov_action < self.prior_cov
            cov_shifted = (1-self.beta) * self.cov_action + self.beta * self.prior_cov
            self.cov_action = update * cov_shifted + (1.0 - update) * self.cov_action



