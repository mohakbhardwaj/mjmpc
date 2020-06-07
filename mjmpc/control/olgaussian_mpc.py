"""
Open-loop Gaussian MPC
"""
from .olcontroller import OLController
from .control_utils import generate_noise
import copy
import numpy as np
# import scipy.stats
import scipy.special

class OLGaussianMPC(OLController):
    def __init__(self,                 
                 num_actions,
                 action_lows,
                 action_highs,
                 horizon,
                 init_cov,
                 init_mean,
                 base_action,
                 num_particles,
                 gamma,
                 n_iters,
                 step_size,
                 filter_coeffs,
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 cov_type='diagonal',
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):


        super(OLGaussianMPC, self).__init__(num_actions,
                                            action_lows, 
                                            action_highs,
                                            horizon,
                                            num_particles,
                                            gamma,  
                                            n_iters,
                                            set_sim_state_fn,
                                            rollout_fn,
                                            sample_mode,
                                            batch_size,
                                            seed)
        self.init_cov = np.array([init_cov] * self.num_actions)
        self.init_mean = init_mean.copy()
        self.mean_action = init_mean
        self.base_action = base_action
        self.cov_type = cov_type
        self.cov_action = np.diag(self.init_cov)
        self.step_size = step_size
        self.filter_coeffs = filter_coeffs

    def _get_next_action(self, mode='mean'):
        if mode == 'mean':
            next_action = self.mean_action[0].copy()
        elif mode == 'sample':
            delta = generate_noise(self.cov_action, self.filter_coeffs,
                                   shape=(1, 1), base_seed=self.seed + self.num_steps)
            next_action = self.mean_action[0].copy() + delta.reshape(self.num_actions).copy()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return next_action
    
    def sample_actions(self):
        delta = generate_noise(self.cov_action, self.filter_coeffs,
                               shape=(self.num_particles, self.horizon), 
                               base_seed = self.seed + self.num_steps)        
        act_seq = self.mean_action[None, :, :] + delta
        return np.array(act_seq)
    
    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        self.mean_action[:-1] = self.mean_action[1:]
        if self.base_action == 'random':
            self.mean_action[-1] = np.random.normal(0, self.init_cov, self.num_actions)
        elif self.base_action == 'null':
            self.mean_action[-1] = np.zeros((self.num_actions, ))
        elif self.base_action == 'repeat':
            self.mean_action[-1] = self.mean_action[-2]
        else:
            raise NotImplementedError("invalid option for base action during shift")

    def reset(self):
        self.num_steps = 0
        self.mean_action = np.zeros(shape=(self.horizon, self.num_actions))
        self.cov_action = np.diag(self.init_cov)

    def _calc_val(self, cost_seq, act_seq):
        raise NotImplementedError("_calc_val not implemented")

