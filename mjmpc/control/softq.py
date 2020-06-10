#!/usr/bin/env python
"""
Soft Q-learning based MPC
Author - Mohak Bhardwaj
Date: 7 June, 2020
"""
from .clcontroller import CLController
from .control_utils import generate_noise, cost_to_go
from copy import deepcopy
import numpy as np
import scipy.special


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
                 lr,
                 tol,
                 num_samples,
                 th_init=None,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
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
        lr : float, [0,1]
            learning rate for sarsa
        tol : float
            tolerance for convergence of sarsa
        num_samples : int
            number of action samples from prior Gaussian
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
        self.lr = lr
        self.tol = tol
        self.num_samples = num_samples
        if th_init is None:
            self.th_init = np.zeros((self.state_dim + self.action_dim))
        else:
            self.th_init = th_init
        self.th = self.th_init.copy()
        self.bias = 0.0
        self.old_th = self.th.copy()
        self.old_bias = self.bias
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
        #sample an action from prior gaussian
        ftrs = self.get_sim_state_fn()
        delta =  generate_noise(self.cov_action,[1.0, 0.0, 0.0],
                                shape=(self.num_samples,1), 
                                base_seed = self.seed + self.num_steps)
        delta = delta.reshape(self.num_samples,self.action_dim)
        #evaluate q value
        ftrs_batch = np.repeat(ftrs[np.newaxis,...], self.num_samples, axis=0)
        inp = np.concatenate((ftrs_batch, delta), axis=-1)
        qvals = inp.dot(self.th) + self.bias
        #update mean of gaussian using I.S
        mean_action = self._update_distribution(qvals, delta)
        return mean_action

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
        ftrs_seq = np.zeros((self.state_dim, self.horizon))
        cost_seq = np.zeros((self.horizon,))
        act_seq = np.zeros((self.action_dim, self.horizon))
        
        self.set_sim_state_fn(deepcopy(state)) #set simulator to initial state 
        for t in range(self.horizon):
            curr_ftrs = self.get_sim_state_fn()
            action = self._get_next_action()
            obs, cost = self.sim_step_fn(action.copy())
            #TODO: Add KL divergence term too

            ftrs_seq[:,t] = curr_ftrs
            cost_seq[t] = cost
            act_seq[:, t] = action
        return ftrs_seq, act_seq, cost_seq 

    def backward_pass(self, ftrs, actions, costs):
        self.old_th = self.th.copy()
        self.old_bias = self.bias
        inps = np.concatenate((ftrs, actions), axis=0)
        n_samples = inps.shape[1]


        # prediction
        q_hat = self.th.dot(inps[:,:-1]) + self.bias
        
        # #Generate targets TD(0)
        # q_next = self.th.dot(inps[:, 1:]) + self.bias
        # q_targets = costs[:-1] + self.gamma * q_next 
        # naive_q_target = cost_to_go(costs, self.gamma_seq).flatten()[:-1]
        #Generate targets TD(N)
        costs[-1] = self.th.dot(inps[:,-1]) + self.bias #lastcost is prediction
        q_targets = cost_to_go(costs, self.gamma_seq).flatten()
        q_targets = q_targets[:-1]
        # print(q_targets, naive_q_target)
        # input('...')
        #Gradient update
        err = q_targets.flatten() - q_hat.flatten()
        dth = err.dot(inps[:,0:-1].T) / (n_samples * 1.0)
        db = np.sum(err) / (n_samples * 1.0)

        self.th = self.th + self.lr * dth
        self.bias = self.bias + self.lr * db


    def converged(self):
        """
        Checks convergence
        """
        # delta = np.average(np.abs(self.old_th - self.th))
        delta = np.linalg.norm(self.old_th - self.th)
        if delta <= self.tol:
            print('Converged')
            return True
        else:
            return False


    def _update_distribution(self, qvals, act_seq):
        """
        Update mean using importance sampling
        """
        w = self._exp_util(qvals, act_seq)
        weighted_seq = w * act_seq.T
        mean_action = np.sum(weighted_seq.T, axis=0)
        return mean_action

    def _exp_util(self, qvals, delta):
        """
            Calculate weights using exponential utility
        """
        # traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        # control_costs = self._control_costs(delta)
        # total_costs = traj_costs + self.lam * control_costs
        # total_costs = qvals #+ self.lam * control_costs

        # #calculate soft-max
        # w1 = np.exp(-(1.0/self.lam) * (total_costs - np.min(total_costs)))
        # w1 /= np.sum(w1) + 1e-6  # normalize the weights
        w = scipy.special.softmax((-1.0/self.lam) * qvals)
        return w



    # def _control_costs(self, delta):
    #     if self.alpha == 1:
    #         return np.zeros(delta.shape[0])
    #     else:
    #         u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
    #         control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
    #         control_costs = np.sum(control_costs, axis=-1)
    #         control_costs = cost_to_go(control_costs, self.gamma_seq)[:,0]

    #     return control_costs


    
    def reset(self):
        """
        Reset the controller
        """
        self.th = self.th_init.copy()
        self.bias = 0.0
        self.old_th = self.th.copy()
        self.old_bias = self.bias
        self.cov_action = np.diag(self.init_cov)
        self.num_steps = 0
