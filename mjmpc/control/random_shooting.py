#!/usr/bin/env python
"""
MPC using naive random shooting
"""
from mjmpc.utils.control_utils import cost_to_go
from .olgaussian_mpc import OLGaussianMPC
import copy
import numpy as np

class RandomShooting(OLGaussianMPC):
    def __init__(self,
                 d_state,
                 d_obs,
                 d_action,
                 horizon,
                 init_cov,
                 base_action,
                 num_particles,
                 step_size,
                 gamma,
                 n_iters,
                 action_lows,
                 action_highs,
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 filter_coeffs = [1.0, 0.0, 0.0],
                 batch_size=1,
                 seed=0):

        super(RandomShooting, self).__init__(d_state,
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

    def _update_distribution(self, trajectories):
        """
           Update mean in direction of best sampled action
           sequence
        """
        costs = trajectories["costs"].copy()
        actions = trajectories["actions"].copy()
        Q = cost_to_go(costs, self.gamma_seq)
        best_id = np.argmin(Q, axis = 0)[0]
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * actions[best_id]
    

    def _calc_val(self, trajectories):
        costs = trajectories["costs"].copy()
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        val = np.average(traj_costs)
        return val