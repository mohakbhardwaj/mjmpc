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

# def generate_noise(std_dev, filter_coeffs, base_act):
#     """
#         Generate noisy samples using autoregressive process
#     """
#     beta_0, beta_1, beta_2 = filter_coeffs
#     eps = np.random.normal(loc=0.0, scale=1.0, size=base_act.shape) * std_dev
#     for i in range(2, eps.shape[0]):
#         eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
#     return base_act + eps 

def generate_noise(cov, filter_coeffs, shape, base_seed):
    """
        Generate correlated noisy samples using autoregressive process
    """
    np.random.seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    for i in range(2, eps.shape[1]):
        eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
    return eps 


def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    cost_seq = gamma_seq * cost_seq  # discounted reward sequence
    cost_seq = np.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq

class Controller(ABC):
    def __init__(self,
                 num_actions,
                 action_lows,
                 action_highs,
                 horizon,
                 num_particles,
                 gamma,
                 n_iters,
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
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
        self._set_sim_state_fn = set_sim_state_fn
        self._rollout_fn = rollout_fn
        self.sample_mode = sample_mode
        self.batch_size = batch_size
        self.seed = seed
        self.num_steps = 0

    @abstractmethod
    def _get_next_action(self, mode='mean'):
        """
            Get action to execute on the system based
            on current control distribution

            :param mode (str): mode for sampling. 
                              - mean returns mean first action of distribution
                              - sample returns a sampled action
        """        
        pass

    @abstractmethod
    def sample_actions(self):
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
    def _calc_val(self, cost_seq, act_seq):
        """
        Calculate value of state given 
        rollouts from a policy
        """
        pass
    
    @property
    def rollout_fn(self):
        return self._rollout_fn
    
    @rollout_fn.setter
    def rollout_fn(self, fn):
        """
        Set the rollout function used to 
        given function pointer
        """
        self._rollout_fn = fn
    
    @property
    def set_sim_state_fn(self):
        return self._set_sim_state_fn
    
    @set_sim_state_fn.setter
    def set_sim_state_fn(self, fn):
        """
        Set function that sets the simulation 
        environment to a particular state
        """
        self._set_sim_state_fn = fn


    def generate_rollouts(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, states, costs and 
            actions
         """
        
        self._set_sim_state_fn(copy.deepcopy(state)) #set state of simulation
        act_seq = self.sample_actions() #sample actions using current control distribution
        cost_seq = self._rollout_fn(act_seq)  # rollout function returns the costs 
        return cost_seq, act_seq

    def step(self, state, calc_val=False):
        """
            Optimize for best action at current state

            :param state (np.ndarray): state to calculate optimal action from
            :param calc_val (bool): If true, calculate the optimal value estimate
                                    of the state while doing online MPC
        """
        if self._rollout_fn is None or self.set_sim_state_fn is None:
            raise Exception("rollout_fn and set_sim_state_fn not set!!")

        for _ in range(self.n_iters):
            # generate random simulated trajectories
            cost_seq, act_seq = self.generate_rollouts(copy.deepcopy(state))
            # update distribution parameters
            self._update_distribution(cost_seq, act_seq)
        
        #calculate best action
        curr_action = self._get_next_action(mode=self.sample_mode)
        #calculate optimal value estimate if required
        value = 0.0
        if calc_val:
            cost_seq, act_seq = self.generate_rollouts(copy.deepcopy(state))
            value = self._calc_val(cost_seq, act_seq)

        self.num_steps += 1
        # shift distribution to hotstart next timestep
        self._shift()

        return curr_action, value

    def get_optimal_value(self, state):
        """
        Calculate optimal value of a state, i.e 
        value under optimal policy. 
        
        :param n_iters(int): number of iterations of optimization
        :param horizon(int): number of actions in rollout
        :param num_particles(int): number of particles in rollout

        """
        self.reset() #reset the control distribution
        _, value = self.step(state, calc_val=True)
        return value


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
                 set_sim_state_fn=None,
                 rollout_fn=None,
                 cov_type='diagonal',
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):


        super(GaussianMPC, self).__init__(num_actions,
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
            # print(self.cov_action)
            # input('..')
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

