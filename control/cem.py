#!/usr/bin/env python
"""Cross Entropy Method for MPC

Author - Mohak Bhardwaj
Date - Jan 12, 2020
TODO:
 - Make it a work for batch of start states 
"""
from .controller import GaussianMPC, scale_ctrl, generate_noise, cost_to_go
import copy
import numpy as np




class CEM(GaussianMPC):
    def __init__(self,
                 horizon,
                 init_cov,
                 base_action,
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
                 update_cov=False,
                 min_cov=1.0,
                 prior_cov=1.0,
                 beta=0.0,
                 terminal_cost_fn=None,
                 rollout_callback=None,
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):



        super(CEM, self).__init__(num_actions,
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

        self.elite_frac = elite_frac
        self.update_cov = update_cov
        self.min_cov = min_cov
        self.beta = beta
        self.prior_cov = prior_cov
        self.num_elite = int(self.num_particles * self.elite_frac)

    def _update_distribution(self, costs, act_seq):
        """
           Update moments using elite samples
        """
        Q = cost_to_go(costs, self.gamma_seq)
        elite_ids = np.argsort(Q[:,0], axis=-1)[0:self.num_elite]
        elite_actions = act_seq[elite_ids, :, :]
        
        if self.update_cov:
            elite_deltas = (act_seq - self.mean_action[None, :,:])[elite_ids, :, :]
            elite_deltas = elite_deltas.reshape(self.horizon * self.num_elite, self.num_actions)
            elite_cov = np.var(elite_deltas, axis=0)
            self.cov_action = (1.0 - self.step_size) * self.cov_action +\
                                self.step_size * elite_cov

            self.cov_action = np.clip(self.cov_action, self.min_cov, None)

        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.mean(elite_actions, axis=0)

        if np.any(np.isnan(self.mean_action)):
            print('warning: nan in mean_action or cov_action...resetting the controller')
            self.reset()

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
        


