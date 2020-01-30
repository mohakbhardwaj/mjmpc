#!/usr/bin/env python
"""
Author - Mohak Bhardwaj
Date: 3 Jan, 2020
"""
from abc import ABC, abstractmethod
import copy
import numpy as np

def scale_ctrl(ctrl, action_low_limit, action_up_limit):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_up_limit - action_low_limit) / 2.0
    act_mid_range = (action_up_limit + action_low_limit) / 2.0
    ctrl = np.clip(ctrl, -1.0, 1.0)
    return act_mid_range[np.newaxis, :, np.newaxis] + ctrl * act_half_range[np.newaxis, :, np.newaxis]

# def generate_noise(std_dev, filter_coeffs, shape, base_seed):
#     """
#         Generate noisy samples using autoregressive process
#     """
#     np.random.seed(base_seed)
#     beta_0, beta_1, beta_2 = filter_coeffs
#     eps = np.random.normal(loc=0.0, scale=1.0, size=shape) * std_dev
#     for i in range(2, eps.shape[1]):
#         eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
#     return eps 

def generate_noise(cov, filter_coeffs, shape, base_seed):
    """
        Generate noisy samples using autoregressive process
    """
    np.random.seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    for i in range(2, eps.shape[1]):
        eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
    return eps 



def cost_to_go(sk, gamma_seq):
    """
        Calculate (discounted) cost to go for given reward sequence
    """
    sk = gamma_seq * sk  # discounted reward sequence
    sk = np.cumsum(sk[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    sk /= gamma_seq  # un-scale it to get true discounted cost to go
    return sk

class Controller(ABC):
    def __init__(self,
                 num_actions,
                 action_lows,
                 action_highs,
                 horizon,
                 num_particles,
                 gamma,
                 n_iters,
                 set_state_fn,
                 rollout_fn,
                 rollout_callback,
                 terminal_cost_fn=None,
                 batch_size=1,
                 seed=0):
                 
        self.num_actions = num_actions
        self.action_lows = action_lows
        self.action_highs = action_highs
        self.horizon = horizon
        self.num_particles = num_particles
        self.gamma = gamma
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(1, horizon)
        self.n_iters = n_iters
        self.set_state_fn = set_state_fn
        self.rollout_fn = rollout_fn
        self.rollout_callback = rollout_callback
        self.terminal_cost_fn = terminal_cost_fn
        self.batch_size = batch_size
        self.seed = seed
        self.num_steps = 0

    @abstractmethod
    def _get_next_action(self):
        """
            Get action to execute on the system based
            on current control distribution
        """        
        pass

    @abstractmethod
    def _sample_actions(self):
        """
        Sample actions from current control distribution
        """
        pass
    
    @abstractmethod
    def _update_distribution(self, costs: np.ndarray, act_seq: np.ndarray):
        """
        Update current control distribution based on the results of rollouts
        params - 
            costs: np.ndarray of step costs during rollouts
            act_seq: action sequence sampled from control distribution
        """
        pass

    @abstractmethod
    def _shift(self):
        """
        Shift the current control distribution
        to hotstart the next timestep
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller
        """
        pass
    
    @abstractmethod
    def _calc_val(self, state):
        """
        Calculate optimal value of a state
        (Must call step function before this)
        """
        pass

    def set_terminal_cost_fn(self, fn):
        self.terminal_cost_fn = fn

    def _generate_rollouts(self):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, states, costs and 
            actions
         """

        act_seq = self._sample_actions() #sample actions using current control distribution
        obs_vec, rew_vec, _ = self.rollout_fn(act_seq)  # rollout function returns the observations, rewards 
        sk = -rew_vec  # rollout_fn returns a REWARD and we need a COST
        if self.terminal_cost_fn is not None:
            term_states = obs_vec[:, -1, :].reshape(obs_vec.shape[0], obs_vec.shape[-1])
            sk[-1, :] = self.terminal_cost_fn(term_states, act_seq[-1].T)

        if self.rollout_callback is not None: self.rollout_callback(obs_vec, act_seq) #state_vec # for example, visualize rollouts

        return obs_vec, sk, act_seq #state_vec

    def step(self, state):
        """
            Optimize for best action at current state
        """
        for _ in range(self.n_iters):
            self.set_state_fn(copy.deepcopy(state)) #set state of simulation
            # generate random trajectories
            obs_vec, sk, act_seq = self._generate_rollouts() #state_vec
            # update moments
            self._update_distribution(sk, act_seq)
            #calculate best action
            curr_action = self._get_next_action()
            
        self.num_steps += 1
        # shift distribution to hotstart next timestep
        self._shift()

        return curr_action


class GaussianMPC(Controller):
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
                 set_state_fn,
                 rollout_fn,
                 rollout_callback,
                 cov_type='diagonal',
                 terminal_cost_fn=None,
                 batch_size=1,
                 seed=0):


        super(GaussianMPC, self).__init__(num_actions,
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
        self.init_cov = np.array([init_cov] * self.num_actions)
        self.mean_action = init_mean
        self.base_action = base_action
        self.cov_type = cov_type
        self.cov_action = np.diag(self.init_cov)
        self.step_size = step_size
        self.filter_coeffs = filter_coeffs

    def _get_next_action(self):
        next_action = self.mean_action[0]
        return next_action.reshape(self.num_actions, )
    
    def _sample_actions(self):
        delta = generate_noise(self.cov_action, self.filter_coeffs,
                               shape=(self.num_particles, self.horizon), 
                               base_seed = self.seed + self.num_steps)        
        act_seq = self.mean_action[None, :, :] + delta
        return act_seq
    
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
        self.cov_action = self.init_cov # * np.ones(shape=(self.horizon, self.num_actions))

    def _calc_val(self, state):
        raise NotImplementedError("_calc val not implemented yet")

