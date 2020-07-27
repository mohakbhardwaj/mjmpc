#!/usr/bin/env python
"""
A version of Model Predictive Path Integral Controller
that uses Q-function estimates

Author - Mohak Bhardwaj
Date - July 24, 2020
TODO:
 - Make it a work for batch of start states 
"""
from mjmpc.utils.control_utils import cost_to_go
from .olgaussian_mpc import OLGaussianMPC
import copy
import numpy as np
# import scipy.stats
import scipy.special
import time

class MPPIQ(OLGaussianMPC):
    def __init__(self,
                 d_state,
                 d_obs,
                 d_action,
                 horizon,
                 init_cov,
                 base_action,
                 beta,
                 num_particles,
                 step_size,
                 alpha,
                 gamma,
                 n_iters,
                 td_lam,
                 action_lows,
                 action_highs,
                 time_based_weights=True,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):

        super(MPPIQ, self).__init__(d_state,
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
                                   rollout_fn,
                                   'diagonal',
                                   sample_mode,
                                   batch_size,
                                   seed)
        self.beta = beta
        self.td_lam = td_lam
        self.alpha = alpha  # 0 means control cost is on, 1 means off
        self.time_based_weights = time_based_weights


    def _update_distribution(self, trajectories):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        costs = trajectories["costs"].copy()
        actions = trajectories["actions"].copy()
        qvals = trajectories["qvals"].copy() if "qvals" in trajectories else None

        delta = actions - self.mean_action[None, :, :]
        w = self._exp_util(costs, qvals, delta)
        weighted_seq = w.T * actions.T
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.sum(weighted_seq.T, axis=0)

    def _exp_util(self, costs, qvals, delta):
        """
            Calculate weights using exponential utility
        """
        # traj_costs = cost_to_go(costs, self.gamma_seq)
        control_costs = self._control_costs(delta)
        total_costs = costs + self.beta * control_costs
        q_hat = self.calculate_returns(total_costs, qvals, self.gamma, self.td_lam)
        if not self.time_based_weights:
            q_hat = q_hat[:,0]
        w = scipy.special.softmax((-1.0/self.beta) * q_hat, axis=0)
        return w

    def calculate_returns(self, costs, qvals, gamma, td_lam=1.0):
        
        # cost_new = costs.copy()
        # cost_new[:,-1] = qvals[:,-1]
        # q_mc = cost_to_go(cost_new, self.gamma_seq)
        # td_errors = costs + gamma * np.hstack([qvals, qvals[:,[-1]]])[:, 1:] - qvals
        if qvals is None:
            qvals = np.zeros(costs.shape)
            qvals[:,-1] = costs[:,-1]
            # weight_seq = np.cumprod([1.0] + [gamma] * (self.horizon - 1)).reshape(1, self.horizon)
            # return cost_to_go(costs, weight_seq)

        td_errors = costs[:, 0:-1] + gamma * qvals[:, 1:] - qvals[:, 0:-1]
        weight_seq = np.cumprod([1.0] + [gamma*td_lam] * (self.horizon - 2)).reshape(1, self.horizon-1)
        q_lam_minus_q = cost_to_go(td_errors, weight_seq)
        # q_lam = q_lam_minus_q  #+ qvals[:, 0:-1]
        #incorporate 0-step estimate too
        q_lam = qvals[:, 0:-1] + td_lam * q_lam_minus_q
        q_lam = np.hstack([q_lam, qvals[:, [-1]]])

        
        # print(qvals)
        # input('....')
        # print(q_lam)
        # input('....')
        # print(q_mc)
        # print(np.allclose(q_mc, q_lam))
        # print(np.allclose(qvals, q_lam))
        # input('....')
        return q_lam

    def _control_costs(self, delta):
        if self.alpha == 1:
            return np.zeros((delta.shape[0], delta.shape[1]))
        else:
            u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
            control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
            control_costs = np.sum(control_costs, axis=-1)
            # control_costs = cost_to_go(control_costs, self.gamma_seq)
        return control_costs
    
    def _calc_val(self, trajectories):
        costs = trajectories["costs"].copy()
        actions = trajectories["actions"].copy()
        qvals = trajectories["qvals"].copy() if "qvals" in trajectories else None

        delta = actions - self.mean_action[None, :, :]
        control_costs = self._control_costs(delta)
        total_costs = costs + self.beta * control_costs        
        q_hat = self.calculate_returns(total_costs, qvals, self.gamma, self.td_lam)
        # traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        # control_costs = self._control_costs(delta)
        # total_costs = traj_costs.copy() + self.beta * control_costs.copy()
        
		# calculate log-sum-exp
        # c = (-1.0/self.beta) * total_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val1 = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val1 = -self.beta * val1
        q_hat = q_hat[:,0]
        val = -self.beta * scipy.special.logsumexp((-1.0/self.beta) * q_hat, b=(1.0/q_hat.shape[0]))
        return val
        





