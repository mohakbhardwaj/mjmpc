#!/usr/bin/env python
"""
A version of DMD-MPC with Gaussian sampling, 
exponential utility cost function and 
covariance adaptation 
Author - Mohak Bhardwaj
Date - Jan 19, 2020
 
"""
from mjmpc.utils.control_utils import cost_to_go
from .olgaussian_mpc import OLGaussianMPC
import copy
import numpy as np
import scipy.special

class DMDMPC(OLGaussianMPC):
    def __init__(self,
                 d_state,
                 d_obs,
                 d_action,
                 horizon,
                 init_cov,
                 beta,
                 base_action,
                 lam,
                 num_particles,
                 step_size,
                 gamma,
                 n_iters,
                 action_lows,
                 action_highs,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
                 rollout_fn=None,
                 update_cov=False,
                 cov_type='diagonal',
                 sample_mode='mean',
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):

        super(DMDMPC, self).__init__(d_state,
                                     d_obs,
                                     d_action,
                                     action_lows, 
                                     action_highs,
                                     horizon,
                                     init_cov,
                                     np.zeros(shape=(horizon, d_action)),
                                     base_action,
                                     num_particles,
                                     gamma,
                                     n_iters,
                                     step_size, 
                                     filter_coeffs, 
                                     set_sim_state_fn,
                                     get_sim_state_fn,
                                     sim_step_fn,
                                     sim_reset_fn,
                                     rollout_fn,
                                     cov_type,
                                     sample_mode,
                                     batch_size,
                                     seed)
        self.lam = lam
        self.beta = beta
        self.update_cov = update_cov

    def _update_distribution(self, trajectories):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        costs = trajectories["costs"].copy()
        actions = trajectories["actions"].copy()
        delta = actions - self.mean_action[None, :, :]
        w = self._exp_util(costs)

        if self.update_cov:
            if self.cov_type == 'diagonal':
                weighted_delta = w * (delta ** 2).T
                cov_update = np.diag(np.mean(np.sum(weighted_delta.T, axis=0), axis=0))
            elif self.cov_type == 'full':
                weighted_delta = np.sqrt(w) * (delta).T
                weighted_delta = weighted_delta.T.reshape((self.horizon * self.num_particles, self.num_actions))
                cov_update = np.dot(weighted_delta.T, weighted_delta)
                cov_update = cov_update/self.horizon
            else:
                raise ValueError('Unidentified covariance type in update_distribution')

            self.cov_action = (1.0 - self.step_size) * self.cov_action +\
                                    self.step_size * cov_update

        weighted_seq = w * actions.T
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.sum(weighted_seq.T, axis=0)


    def _exp_util(self, costs):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        # #calculate soft-max
        # w = np.exp(-(traj_costs - np.min(traj_costs)) / self.lam)
        # w /= np.sum(w) + 1e-6  # normalize the weights
        w = scipy.special.softmax((-1.0/self.lam) * traj_costs)

        return w
    
    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        super()._shift()
        if self.update_cov:
            # self.cov_action += self.beta * np.eye(self.num_actions)
            self.cov_action += self.beta * np.diag(self.init_cov)
        # if self.update_cov:
        #     self.cov_action = np.clip(self.cov_action, self.min_cov, None)
        #     if self.beta > 0.0:
        #         update = self.cov_action < self.prior_cov
        #         cov_shifted = (1-self.beta) * self.cov_action + self.beta * self.prior_cov
        #         self.cov_action = update * cov_shifted + (1.0 - update) * self.cov_action
    
    def _calc_val(self, trajectories):
        costs = trajectories["costs"].copy()
        traj_costs = cost_to_go(costs,self.gamma_seq)[:,0]

		# calculate log-sum-exp
        # c = (-1.0/self.lam) * traj_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val = -self.lam * val

        val = -self.lam * scipy.special.logsumexp((-1.0/self.lam) * traj_costs, b=(1.0/traj_costs.shape[0]))
        return val

