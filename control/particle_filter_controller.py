#!/usr/bin/env python
"""
Particle Filter MPC

Author - Mohak Bhardwaj

"""
from .controller import Controller, scale_ctrl, generate_noise, cost_to_go
import copy
import numpy as np
import random
import scipy.special

class PFMPC(Controller):
    def __init__(self,
                 horizon,
                 cov_shift,
                 cov_resample,
                 base_action,
                 lam,
                 num_particles,
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
                 filter_coeffs=[1., 0., 0.],
                 seed=0):

        super(PFMPC, self).__init__(num_actions,
                                    action_lows,
                                    action_highs,
                                    horizon,
                                    num_particles,
                                    gamma,
                                    n_iters,
                                    set_state_fn,
                                    rollout_fn,
                                    rollout_callback,
                                    terminal_cost_fn,
                                    batch_size,
                                    seed)
        self.lam = lam
        self.cov_shift = np.diag(np.array([cov_shift] * self.num_actions))
        self.cov_resample = np.diag(np.array([cov_resample] * self.num_actions))
        self.base_action = base_action
        self.filter_coeffs = filter_coeffs
        random.seed(self.seed)
        self.action_samples = generate_noise(self.cov_resample, self.filter_coeffs,
                                             shape=(self.num_particles, self.horizon),
                                             base_seed=self.seed)


    def _update_distribution(self, costs, act_seq):
        """
            Calculate weights for particles and resample

        """
        w = self._exp_util(costs)
        random.seed(self.seed + self.num_steps)
        np.random.seed(self.seed + self.num_steps)
        self.action_samples = self._resampling(self.action_samples, w, low_variance=True)

        
    def _exp_util(self, costs):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        # #calculate soft-max
        w = np.exp(-(traj_costs - np.min(traj_costs)) / self.lam)
        w /= np.sum(w) + 1e-6  # normalize the weights
        # w = scipy.special.softmax((-1.0/self.lam) * traj_costs)
        return w
    
    def _sample_actions(self):
        return self.action_samples

    def _get_next_action(self):
        """
            Return the average of current action samples
            TODO: Add other sampling strategies
        """
        action = np.mean(self.action_samples, axis=0)[0]
        return action

    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the action samples by one timestep and 
            adding noise. 
        """
        #shift forward
        self.action_samples[:, :-1] = self.action_samples[:, 1:]
        #add noise
        delta = generate_noise(self.cov_shift, self.filter_coeffs,
                               shape=(self.num_particles, self.horizon),
                               base_seed=self.seed + self.num_steps)
        self.action_samples = self.action_samples + delta
        #append base action to the end
        if self.base_action == 'random':
            self.action_samples[:, -1] = np.random.normal(
                0, self.cov_resample, self.num_actions)
        elif self.base_action == 'null':
            self.action_samples[:, -1] = np.zeros((self.num_particles, self.num_actions))
        elif self.base_action == 'repeat':
            self.action_samples[:, -1] = self.action_samples[:, -2]
        else:
            raise NotImplementedError(
                "invalid option for base action during shift")

    def reset(self):
        self.num_steps = 0
        self.action_samples = generate_noise(self.cov_resample, self.filter_coeffs,
                                             shape=(self.num_particles, self.horizon),
                                             base_seed=self.seed)


    def _calc_val(self, state):
        raise NotImplementedError("_calc val not implemented yet")


    def _resampling(self, act_seq, weights, low_variance=True):
        if low_variance:
            M = act_seq.shape[0]
            act_seq2 = np.zeros_like(act_seq)
            r = random.uniform(0.0, 1.0/M*1.0)
            c  = 0.0; i = 0
            for m in range(M):
                u = r + m*1.0/M*1.0
                while (c < u and i < M):
                    c += weights[i]
                    i += 1
                act_seq2[m] = act_seq[i-1]
        else:
            act_seq2 = np.array(random.choices(self.action_samples, weights=weights, k=self.num_particles))
        
        return act_seq2
