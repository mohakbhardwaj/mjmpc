#!/usr/bin/env python
"""Model Predictive Path Integral Controller

Author - Mohak Bhardwaj
Date - Dec 20, 2019
TODO:
 - Make it a work for batch of start states 
"""
from .controller import Controller, GaussianMPC, scale_ctrl, generate_noise, cost_to_go
import copy
import numpy as np
from scipy.signal import savgol_filter
import scipy.stats
import scipy.special

class MPPI(GaussianMPC):
    def __init__(self,
                 horizon,
                 init_cov,
                 base_action,
                 lam,
                 num_particles,
                 step_size,
                 alpha,
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

        super(MPPI, self).__init__(num_actions,
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
        self.alpha = alpha  # 0 means control cost is on, 1 means off


    def _update_distribution(self, costs, act_seq):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        delta = act_seq - self.mean_action[None, :, :]
        w = self._exp_util(costs, delta)
        weighted_seq = w * act_seq.T
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.sum(weighted_seq.T, axis=0) 
        
    def _exp_util(self, costs, delta):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.lam * control_costs 
        # #calculate soft-max
        w = np.exp(-(total_costs - np.min(total_costs)) / self.lam)
        w /= np.sum(w) + 1e-6  # normalize the weights
        return w

    def _control_costs(self, delta):
        if self.alpha == 1:
            return np.zeros(delta.shape[0])
        else:
            u_normalized = self.mean_action/self.cov_action
            control_costs = u_normalized[None, :,:] * delta
            control_costs = np.sum(control_costs, axis=-1)
            control_costs = cost_to_go(control_costs, self.gamma_seq)[:,0]
        return control_costs



