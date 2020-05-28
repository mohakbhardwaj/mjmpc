#!/usr/bin/env python
"""Model Predictive Path Integral Controller

Author - Mohak Bhardwaj
Date - Dec 20, 2019
TODO:
 - Make it a work for batch of start states 
"""
from .controller import Controller, GaussianMPC, cost_to_go
import copy
import numpy as np
# import scipy.stats
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
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):

        super(MPPI, self).__init__(num_actions,
                                   action_lows, 
                                   action_highs,
                                   horizon,
                                   init_cov,
                                   np.zeros(shape=(horizon, num_actions)),
                                   base_action,
                                   num_particles,
                                   gamma,
                                   n_iters,
                                   step_size, 
                                   filter_coeffs, 
                                   set_sim_state_fn, 
                                   rollout_fn,
                                   'diagonal',
                                   sample_mode,
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
        # self.mean_action = np.sum(weighted_seq.T, axis=0)
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
        # w1 = np.exp(-(1.0/self.lam) * (total_costs - np.min(total_costs)))
        # w1 /= np.sum(w1) + 1e-6  # normalize the weights
        w = scipy.special.softmax((-1.0/self.lam) * total_costs)
        return w

    def _control_costs(self, delta):
        if self.alpha == 1:
            return np.zeros(delta.shape[0])
        else:
            u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
            control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
            control_costs = np.sum(control_costs, axis=-1)
            control_costs = cost_to_go(control_costs, self.gamma_seq)[:,0]

        return control_costs
    
    def _calc_val(self, cost_seq, act_seq):
        delta = act_seq - self.mean_action[None, :, :]
        
        traj_costs = cost_to_go(cost_seq,self.gamma_seq)[:,0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs.copy() + self.lam * control_costs.copy()
        
		# calculate log-sum-exp
        # c = (-1.0/self.lam) * total_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val1 = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val1 = -self.lam * val1

        val = -self.lam * scipy.special.logsumexp((-1.0/self.lam) * total_costs, b=(1.0/total_costs.shape[0]))
        return val
        





