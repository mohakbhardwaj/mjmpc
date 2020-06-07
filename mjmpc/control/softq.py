#!/usr/bin/env python
"""
Soft Q-learning based MPC
Author - Mohak Bhardwaj
Date: 7 June, 2020
"""
from .clcontroller import CLController
from .control_utils import generate_noise, cost_to_go
import copy
import numpy as np

class SoftQMPC(CLController):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_lows,
                 action_highs,
                 horizon,
                 max_iters,
                 gamma,
                 init_cov,
                 lam,
                 alpha,
                 th_init=None,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):
        """
        Parameters
        ----------
        state_dim : int
            size of state space
        action_dim : int
            size of action space
        action_lows : np.ndarray 
            lower limits for each action dim
        action_highs : np.ndarray  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        max_iters : int  
            max number of optimization iterations/rollouts
        gamma : float
            discount factor
        init_cov : float
            initial covariance for sampling actions
        lam : float
            temperature
        alpha : float, [0,1]
            learning rate for q learning
        th_init : None, optional
            initial parameters for Q-function
        set_sim_state_fn : function  
            set state of simulator using input
        get_sim_state_fn : function  
            get current state of simulator
        sim_step_fn : function  
            step the simulatio using action
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        batch_size : int
            optimize for a batch of states
        seed : int  
            seed value
        """
        super(SoftQMPC, self).__init__(                 
                                       state_dim,
                                       action_dim,
                                       action_lows,
                                       action_highs,
                                       horizon,
                                       max_iters,
                                       gamma,
                                       set_sim_state_fn,
                                       get_sim_state_fn,
                                       sim_step_fn,
                                       sample_mode,
                                       batch_size,
                                       seed)

        self.init_cov = np.array([init_cov] * self.action_dim)
        self.lam = lam
        self.alpha = alpha
        if th_init is None:
            self.th_init = np.zeros((self.state_dim + self.action_dim))
        else:
            self.th_init = th_init
        self.th = self.th_init.copy()
        self.cov_action = np.diag(self.init_cov)
        self.num_steps = 0

    def _get_next_action(self, mode='mean'):
        """
        Get action to execute on the system based
        on current control distribution
        Parameters
        ----------
        mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        """        
        pass


    def _shift(self):
        """
        Shift the current control distribution
        to hotstart the next timestep
        """
        pass


    def _calc_val(self, cost_seq, act_seq):
        """
        Calculate value of state given 
        rollouts from some policy
        """
        return 0.0
    
    def forward_pass(self, state):
        for t in range(self.horizon):
            pass


    def backward_pass(self, state_seq, cost_seq, act_seq):

        pass

    def converged(self):
        """
        Checks convergence
        """
        return False
    
    def reset(self):
        """
        Reset the controller
        """
        self.th = self.th_init.copy()
        self.cov_action = np.diag(self.init_cov)
        self.num_steps = 0
    # def _update_distribution(self, costs, act_seq):
    #     """
    #        Update moments in the direction of current gradient estimated
    #        using samples
    #     """
    #     delta = act_seq - self.mean_action[None, :, :]
    #     w = self._exp_util(costs, delta)

    #     weighted_seq = w * act_seq.T
    #     # self.mean_action = np.sum(weighted_seq.T, axis=0)
    #     self.mean_action = (1.0 - self.step_size) * self.mean_action +\
    #                         self.step_size * np.sum(weighted_seq.T, axis=0) 

    # def _exp_util(self, costs, delta):
    #     """
    #         Calculate weights using exponential utility
    #     """
    #     traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
    #     control_costs = self._control_costs(delta)
    #     total_costs = traj_costs + self.lam * control_costs
    #     # #calculate soft-max
    #     # w1 = np.exp(-(1.0/self.lam) * (total_costs - np.min(total_costs)))
    #     # w1 /= np.sum(w1) + 1e-6  # normalize the weights
    #     w = scipy.special.softmax((-1.0/self.lam) * total_costs)
    #     return w

    # def _control_costs(self, delta):
    #     if self.alpha == 1:
    #         return np.zeros(delta.shape[0])
    #     else:
    #         u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
    #         control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
    #         control_costs = np.sum(control_costs, axis=-1)
    #         control_costs = cost_to_go(control_costs, self.gamma_seq)[:,0]

    #     return control_costs
