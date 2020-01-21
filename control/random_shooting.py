#!/usr/bin/env python
"""
MPC using naive random shooting
"""
from .controller import GaussianMPC, scale_ctrl, cost_to_go, generate_noise
import copy
import numpy as np

class RandomShooting(GaussianMPC):
    def __init__(self,
                 horizon,
                 init_cov,
                 base_action,
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
                 filter_coeffs = [1.0, 0.0, 0.0],
                 batch_size=1,
                 seed=0):

        super(RandomShooting, self).__init__(num_actions,
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


    def _update_distribution(self, costs, act_seq):
        """
           Update mean in direction of best sampled action
           sequence
        """
        Q = cost_to_go(costs, self.gamma_seq)
        best_id = np.argmin(Q, axis = 0)[0]
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * act_seq[best_id]
     


