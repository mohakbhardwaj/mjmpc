#!/usr/bin/env python
"""Cross Entropy Method for MPC

Author - Mohak Bhardwaj
Date - Jan 12, 2020
TODO:
 - Make it a work for batch of start states 
"""
from .control_utils import cost_to_go
from .olgaussian_mpc import OLGaussianMPC
import copy
import numpy as np


class CEM(OLGaussianMPC):
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
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 beta=0.0,
                 cov_type='diagonal',
                 sample_mode='mean',
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0):


        super(CEM, self).__init__(num_actions,
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
                                   cov_type,
                                   sample_mode,
                                   batch_size,
                                   seed)

        self.elite_frac = elite_frac
        self.beta = beta
        self.num_elite = int(self.num_particles * self.elite_frac)

    def _update_distribution(self, costs, act_seq):
        """
           Update moments using elite samples
        """
        Q = cost_to_go(costs, self.gamma_seq)
        elite_ids = np.argsort(Q[:,0], axis=-1)[0:self.num_elite]
        elite_actions = act_seq[elite_ids, :, :]
        
        elite_deltas = (act_seq - self.mean_action[None, :,:])[elite_ids, :, :]
        elite_deltas = elite_deltas.reshape(self.horizon * self.num_elite, self.num_actions)
        if self.cov_type == 'diagonal':
            cov_update = np.diag(np.var(elite_deltas, axis=0))
        elif self.cov_type == 'full':
            cov_update = np.cov(elite_deltas, rowvar=False)

        self.cov_action = (1.0 - self.step_size) * self.cov_action +\
                            self.step_size * cov_update

        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * np.mean(elite_actions, axis=0)


    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        super()._shift()
        self.cov_action += self.beta * np.diag(self.init_cov) #np.eye(self.num_actions)
            # self.cov_action = np.clip(self.cov_action, self.min_cov, None)
            # if self.beta > 0.0:
            #     update = self.cov_action < self.prior_cov
            #     cov_shifted = (1-self.beta) * self.cov_action + self.beta * self.prior_cov
            #     self.cov_action = update * cov_shifted + (1.0 - update) * self.cov_action

    def _calc_val(self, cost_seq, act_seq):
        # self._set_sim_state_fn(copy.deepcopy(state)) #set state of simulation
        # cost_seq, act_seq = self._generate_rollouts()
        
        traj_costs = cost_to_go(cost_seq,self.gamma_seq)[:,0]
        val = np.average(traj_costs)
        return val