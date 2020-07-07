"""
Open-loop Gaussian MPC
"""
from .controller import Controller
from mjmpc.utils.control_utils import generate_noise, scale_ctrl
import copy
import numpy as np
import scipy.special

class OLGaussianMPC(Controller):
    def __init__(self, 
                 d_state,
                 d_obs,
                 d_action,                
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
        """
        Parameters
        __________
        base_action : str
            Action to append at the end when shifting solution to next timestep
            'random' : appends random action
            'null' : appends zero action
            'repeat' : repeats second to last action
        num_particles : int
            Number of particles sampled at every iteration
        """

        super(OLGaussianMPC, self).__init__(d_state,
                                            d_obs,
                                            d_action,
                                            action_lows, 
                                            action_highs,
                                            horizon,
                                            gamma,  
                                            n_iters,
                                            set_sim_state_fn,
                                            rollout_fn,
                                            sample_mode,
                                            batch_size,
                                            seed)
        self.init_cov = np.array([init_cov] * self.d_action)
        self.init_mean = init_mean.copy()
        self.mean_action = init_mean
        self.base_action = base_action
        self.num_particles = num_particles
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
            next_action = self.mean_action[0].copy() + delta.reshape(self.d_action).copy()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return next_action
    
    def sample_actions(self):
        delta = generate_noise(self.cov_action, self.filter_coeffs,
                               shape=(self.num_particles, self.horizon), 
                               base_seed = self.seed_val + self.num_steps)        
        act_seq = self.mean_action[None, :, :] + delta
        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs)
        return np.array(act_seq)
    
    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step
        """
        self.mean_action[:-1] = self.mean_action[1:]
        if self.base_action == 'random':
            self.mean_action[-1] = np.random.normal(0, self.init_cov, self.d_action)
        elif self.base_action == 'null':
            self.mean_action[-1] = np.zeros((self.d_action, ))
        elif self.base_action == 'repeat':
            self.mean_action[-1] = self.mean_action[-2]
        else:
            raise NotImplementedError("invalid option for base action during shift")

    def reset(self):
        self.num_steps = 0
        self.mean_action = np.zeros(shape=(self.horizon, self.d_action))
        self.cov_action = np.diag(self.init_cov)

    def _calc_val(self, cost_seq, act_seq):
        raise NotImplementedError("_calc_val not implemented")

