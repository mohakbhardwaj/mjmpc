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
        self._val = 0.0  # optimal value for current state


    def _update_distribution(self, sk, act_seq):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        delta = act_seq - self.mean_action[:, :, np.newaxis]
        w = self._exp_util(sk, delta, self.lam)
        # print(w.shape, act_seq.shape, self.mean_action.shape, np.matmul(act_seq, w).shape)
        # input('..')
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.matmul(act_seq, w)
        # self.mean_action = (1.0 - self.step_size) * self.mean_action +\
        #                     self.step_size * np.matmul(act_seq, w[:,:,None]).squeeze(-1)

        if np.any(np.isnan(self.mean_action)):
            print('warning: nan in mean_action, resetting the controller')
            self.reset()

        
    def _exp_util(self, sk, delta, lam):
        """
            Calculate weights using exponential utility
        """
        # The cost to go has been used here https://arxiv.org/pdf/1706.09597.pdf and also in the original MPPI paper
        uk = self._control_costs(delta)
        sk = sk + lam * uk
        sk = cost_to_go(sk, self.gamma_seq)

        # #calculate soft-max
        # sk -= np.min(sk, axis=-1)[:, None]  # shift the weights
        # w = np.exp(-sk / lam)
        # w /= np.sum(w, axis=-1)[:, None] + 1e-6  # normalize the weights

        sk = -sk[0, :]/lam
        w = scipy.special.softmax(sk, axis=0)
        return w

    def _control_costs(self, delta):
        if self.alpha == 1:
            control_costs = np.zeros((delta.shape[0], delta.shape[-1]))
        else:
            u_normalized = self.mean_action/self.cov_action
            control_costs = 0.5 * u_normalized[:, :, np.newaxis] * (self.mean_action[:, :, np.newaxis] + 2.0 * delta)
            control_costs = np.sum(control_costs, axis=1)
        return control_costs


    def _calc_val(self, state):
        """
            Calculate (soft) value or free energy of state under current
            control distribution
        """
        sk, delta = self._generate_rollouts(state)
        uk = self._control_costs(delta)
        sk = sk + self.lam * uk
        sk = cost_to_go(sk, self.gamma_seq)
        sk = -sk / self.lam
        sk = np.array(sk[0, :])

        # calculate log-sum-exp
        skmax = np.max(sk)
        sk -= skmax
        sk = np.exp(sk)
        val = skmax + np.log(np.sum(sk)) - np.log(sk.shape[0])
        val = -self.lam * val
        return val

    def _action_prob(self, x, mean, cov):
        """
        Return Gaussian probability density value of x

        """
        return scipy.stats.norm.pdf(x, mean, np.sqrt(cov))

