#!/usr/bin/env python
"""
Base class for controllers
Author - Mohak Bhardwaj
Date: 3 Jan, 2020
"""
from abc import ABC, abstractmethod
import copy
from gym.utils import seeding
import numpy as np
from mjmpc.utils import helpers

class Controller(ABC):
    def __init__(self,
                 d_state,
                 d_obs,
                 d_action,
                 action_lows,
                 action_highs,
                 horizon,
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
        d_state : int
            size of state/observation space
        d_action : int
            size of action space
        action_lows : np.ndarray 
            lower limits for each action dim
        action_highs : np.ndarray  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        gamma : float
            discount factor
        n_iters : int  
            number of optimization iterations
        set_sim_state_fn : function  
            set state of simulator using input
        get_sim_state_fn : function
            get state from the simulator
        sim_step_fn : function
            steps the simulator and returns obs, reward, done, info
        sim_reset_fn : function
            resets the simulator
        rollout_fn : function  
            rollout policy (or actions) in simulator and return obs, reward, done, info
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        batch_size : int
            optimize for a batch of states
        seed : int  
            seed value
        """
        self.d_state = d_state
        self.d_obs = d_obs
        self.d_action = d_action
        self.action_lows = action_lows
        self.action_highs = action_highs
        self.horizon = horizon
        self.gamma = gamma
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(1, horizon)
        self.n_iters = n_iters
        self._set_sim_state_fn = set_sim_state_fn
        self._rollout_fn = rollout_fn
        self.sample_mode = sample_mode
        self.batch_size = batch_size
        self.num_steps = 0
        self.seed_val = self.seed(seed)

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

    def sample_actions(self):
        """
        Sample actions from current control distribution
        """
        raise NotImplementedError('sample_actions funtion not implemented')
    
    @abstractmethod
    def _update_distribution(self, trajectories):
        """
        Update current control distribution using 
        rollout trajectories
        
        Parameters

        trajectories : dict
            Rollout trajectories. Contains the following fields
            observations : np.ndarray ()
                observations along rollouts
            actions : np.ndarray 
                actions sampled from control distribution along rollouts
            costs : np.ndarray 
                step costs along rollouts
            dones : np.ndarray
                bool signalling end of episode
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

    def check_convergence(self):
        """
        Checks if controller has converged
        Returns False by default
        """
        return False
        
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

    def generate_rollouts(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        
        self._set_sim_state_fn(copy.deepcopy(state)) #set state of simulation
        act_seq = self.sample_actions() #sample actions using current control distribution
        obs_seq, cost_seq, done_seq, info_seq = self._rollout_fn(act_seq)  # rollout function returns the costs 
        trajectories = dict(
            observations=obs_seq,
            actions=act_seq,
            costs=cost_seq,
            dones=done_seq,
            infos=helpers.stack_tensor_dict_list(info_seq)
        )
        return trajectories

    def optimize(self, state, calc_val=False, hotstart=True):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : 
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action
        
        Returns
        -------
        action : np.ndarray ()

        Raises
        ------
        ValueError
            If self._rollout_fn, self._set_sim_state_fn or 
               self._sim_step_fn are None

        """

        # if (self.rollout_fn is None) or (self.set_sim_state_fn is None):
        #     raise ValueError("rollout_fn and set_sim_state_fn not set!!")

        for _ in range(self.n_iters):
            # generate random simulated trajectories
            trajectory = self.generate_rollouts(copy.deepcopy(state))
            # update distribution parameters
            self._update_distribution(trajectory)
            # check if converged
            if self.check_convergence():
                break
        
        #calculate best action
        curr_action = self._get_next_action(mode=self.sample_mode)
        #calculate optimal value estimate if required
        value = 0.0
        if calc_val:
            trajectories = self.generate_rollouts(copy.deepcopy(state))
            value = self._calc_val(trajectories)

        self.num_steps += 1
        if hotstart:
            # shift distribution to hotstart next timestep
            self._shift()

        return curr_action, value

    def get_optimal_value(self, state):
        """
        Calculate optimal value of a state, i.e 
        value under optimal policy. 

        Parameters
        ----------
        state : dict or np.ndarray
            state to calculate optimal value estimate for
        Returns
        -------
        value : float
            optimal value estimate of the state
        """
        self.reset() #reset the control distribution
        _, value = self.optimize(state, calc_val=True)
        return value
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed




