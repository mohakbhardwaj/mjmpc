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

class Controller(ABC):
    def __init__(self,
                 num_actions,
                 action_lows,
                 action_highs,
                 set_state_fn,
                 terminal_cost_fn=None,
                 batch_size=1):
                #  get_state_fn,
        self.num_actions = num_actions
        self.action_lows = action_lows
        self.action_highs = action_highs
        # self.get_state_fn = get_state_fn
        self.set_state_fn = set_state_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.batch_size = batch_size

    @abstractmethod
    def _sample_actions(self):
        """
        Sample actions from current control distribution
        """
        pass
    
    @abstractmethod
    def _get_next_action(self):
        """
            Get action to execute on the system based
            on current control distribution
        """        
        pass

    @abstractmethod
    def _update_moments(self, costs: np.ndarray, action_deltas: np.ndarray):
        """
        Update moments of current control distribution 
        based on the results of rollouts
        params - 
            costs: np.ndarray of step costs during rollouts
            action_deltas: delta actions sampled by zero centering the control distribution
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

    def step(self, state):
        """
            Optimize for best action at current state
        """
        # state = self.get_state_fn() #get state to plan from (should this be an input?)
        self.set_state_fn(state) #set state of simulation

        for itr in range(self.n_iters):
            # generate random trajectories
            obs_vec, state_vec, sk, delta = self._generate_rollouts(state)
            # update moments and calculate best action
            self._update_moments(sk, delta)
            curr_action = self._get_next_action(state, sk, delta)
            
        # restore state to original
        # self.set_state_fn(state) #(Q: do we need this?)
        self.num_steps += 1
        # shift moments one timestep (dynamic step)
        self._shift()

        return curr_action

    def _generate_rollouts(self, state):
        """
            Samples actions generates random trajectories for each particle
            and returns the resulting observations, states, costs and 
            actions
         """
        delta = self._sample_actions()
        _ctrl = self.mean_action[:, :, np.newaxis] + delta
        act_seq = scale_ctrl(_ctrl, action_low_limit=self.action_lows, action_up_limit=self.action_highs)
        obs_vec, state_vec, rew_vec, _ = self.rollout_fn(copy.deepcopy(state), act_seq)  # rollout function returns the rewards and final states
        sk = -rew_vec  # rollout_fn returns a REWARD and we need a COST
        if self.terminal_cost_fn is not None:
            term_states = obs_vec[:, -1, :].reshape(obs_vec.shape[0], obs_vec.shape[-1])
            sk[-1, :] = self.terminal_cost_fn(term_states, act_seq[-1].T)  # replace terminal cost with something else
            # if utils.is_nan_np(sk):
            #     print('NaN rewards', sk)
            #     print('states', st)
            #     print('action seq', act_seq)
            #     print('mean', self.mean_action)
            #     print('covariance', self.cov_action)
            #     input('..')

        if self.rollout_callback is not None: self.rollout_callback(obs_vec, state_vec, self.mean_action, self.cov_action,
                                                                    act_seq)  # for example, visualize rollouts

        return obs_vec, state_vec, sk, delta
