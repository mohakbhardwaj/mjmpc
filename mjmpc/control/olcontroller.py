#!/usr/bin/env python
"""
Base class for open-loop controllers
Author - Mohak Bhardwaj
Date: 3 Jan, 2020
"""
from abc import ABC, abstractmethod
import copy
import numpy as np

class OLController(ABC):
    """
    Base class for open-loop controllers

    """
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
        """
        Parameters
        ----------
        num_actions : int
            size of action space
        action_lows : np.ndarray 
            lower limits for each action dim
        action_highs : np.ndarray  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        num_particles : int
            number of particles/rollouts
        gamma : float
            discount factor
        n_iters : int  
            number of optimization iterations
        set_sim_state_fn : function  
            set state of simulator using input
        rollout_fn : function  
            rollout batch of open loop actions in simulator
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        batch_size : int
            optimize for a batch of states
        seed : int  
            seed value
        """
                 
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
        Parameters
        ----------
        mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        """        
        pass

    @abstractmethod
    def sample_actions(self):
        """
        Sample actions from current control distribution
        """
        pass
    
    @abstractmethod
    def _update_distribution(self, costs, act_seq):
        """
        Update current control distribution based on 
        the results of rollouts
        params - 
            costs : np.ndarray 
                step costs from rollouts
            act_seq : np.ndarray 
                action sequence sampled from control distribution
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

        Parameters
        ----------
        state : 
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action

        Raises
        ------
        ValueError
            If self._rollout_fn or self._set_sim_state_fn are None

        """

        if (self.rollout_fn is None) or (self.set_sim_state_fn is None):
            raise ValueError("rollout_fn and set_sim_state_fn not set!!")

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


