#!/usr/bin/env python
"""
MPC using naive random shooting
"""
from .controller import Controller, scale_ctrl, cost_to_go, generate_noise
import copy
import numpy as np

class RandomShootingMPC(Controller):
    def __init__(self,
                 horizon,
                 init_cov,
                 num_particles,
                 gamma,
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

        super(RandomShootingMPC, self).__init__(num_actions,
                                                action_lows, 
                                                action_highs,  
                                                set_state_fn, 
                                                terminal_cost_fn,
                                                batch_size)
        self.horizon = horizon
        self.init_cov = init_cov
        self.num_particles = num_particles
        self.gamma = gamma 
        self.rollout_fn = rollout_fn
        self.rollout_callback = rollout_callback
        self.filter_coeffs = filter_coeffs
        self.seed = seed
        self.best_act_seq = np.zeros(shape=(horizon, num_actions))
        self.cov_action = self.init_cov * np.ones(shape=(horizon, num_actions))
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(horizon, 1)
        self.num_steps = 0
        self.n_iters = 1

    def _get_next_action(self, state, sk, delta):
        """
            Get action to execute on the system based
            on current control distribution
        """
        next_action = self.best_act_seq[0]
        # next_action = scale_ctrl(next_action, action_low_limit=self.action_lows, action_up_limit=self.action_highs)
        return next_action.reshape(self.num_actions, )

    def _sample_actions(self):
        delta = generate_noise(np.sqrt(self.cov_action[:, :, np.newaxis]), self.filter_coeffs,
                                       shape=(self.horizon, self.num_actions, self.num_particles))
        act_seq = self.best_act_seq[:, :, np.newaxis] + delta
        return act_seq

    def _update_moments(self, sk, act_seq):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """
        Q = cost_to_go(sk, self.gamma_seq)
        best_id = np.argmin(Q, axis = -1)[0]
        self.best_act_seq = act_seq[:, :, best_id]

    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        self.best_act_seq[:-1] = self.best_act_seq[1:]
        self.best_act_seq[-1] = np.random.normal(0, self.init_cov, self.num_actions)
        
    def _calc_val(self, state):
        """
            Calculate (soft) value or free energy of state under current
            control distribution
        """
        return 0.0

    def reset(self):
        self.num_steps = 0
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (self.horizon - 1)).reshape(self.horizon, 1)


