"""
Base-class for closed loop controllers
"""
from copy import deepcopy
import numpy as np
import torch

from mjmpc.control.controller import Controller
from mjmpc.utils.control_utils import cost_to_go, generate_noise, scale_ctrl
from mjmpc.utils import helpers

class CLMPC(Controller):
    def __init__(self, 
                 d_state,
                 d_obs,
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 gamma,
                 step_size,
                 filter_coeffs,
                 n_iters,
                 num_particles,
                 init_cov,
                 base_action,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 get_sim_obs_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):
        """
        Parameters
        __________

        """

        super(CLMPC, self).__init__(d_state,
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
        torch.manual_seed(seed)
        self.init_cov = np.array([init_cov] * self.d_action)
        self.mean_action = np.zeros(shape=(self.horizon, self.d_action))
        self.num_particles = num_particles
        self.base_action = base_action
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
    
    def _sample_noise(self):
        delta = generate_noise(self.cov_action, self.filter_coeffs,
                               shape=(self.num_particles, self.horizon), 
                               base_seed = self.seed_val + self.num_steps)
        return delta.copy()
        
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
        self._set_sim_state_fn(deepcopy(state)) #set state of simulation
        delta = self._sample_noise() #sample noise sequence
        obs_seq, act_seq, logprob_seq, cost_seq, done_seq, next_obs_seq, info_seq = self._rollout_fn(mode='mean',
                                                                                       noise=delta)
        trajectories = dict(
            observations=obs_seq,
            actions=act_seq,
            log_probs=logprob_seq,
            costs=cost_seq,
            dones=done_seq,
            infos=helpers.stack_tensor_dict_list(info_seq)
        )
        return trajectories

    def _update_distribution(self, trajectories):
        """
        Update current control distribution using 
        rollout trajectories
        
        Parameters
        -----------
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
        costs = trajectories["costs"].copy()
        actions = trajectories["actions"].copy()
        Q = cost_to_go(costs, self.gamma_seq)
        best_id = np.argmin(Q, axis = 0)[0]
        self.mean_action = (1.0 - self.step_size) * self.mean_action +\
                            self.step_size * actions[best_id]
        
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
        """
        Reset the controller
        """
        self.num_steps = 0
        self.mean_action = np.zeros(shape=(self.horizon, self.d_action))
        self.cov_action = np.diag(self.init_cov)

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


















# #!/usr/bin/env python
# """
# Base class for closed-loop controllers
# Author - Mohak Bhardwaj
# Date: 7 June, 2020

# DEPRECATED!!!!!
# """
# from abc import ABC, abstractmethod
# import copy
# from gym.utils import seeding
# import numpy as np

# class CLController(ABC):
#     """
#     Base class for closed-loop controllers

#     """
#     def __init__(self,
#                  state_dim,
#                  action_dim,
#                  action_lows,
#                  action_highs,
#                  horizon,
#                  max_iters,
#                  gamma,
#                  set_sim_state_fn=None,
#                  get_sim_state_fn=None,
#                  sim_step_fn=None,
#                  sample_mode='mean',
#                  batch_size=1,
#                  seed=0):
#         """
#         Parameters
#         ----------
#         state_dim : int
#             size of state space
#         action_dim : int
#             size of action space
#         action_lows : np.ndarray 
#             lower limits for each action dim
#         action_highs : np.ndarray  
#             upper limits for each action dim
#         horizon : int  
#             horizon of rollouts
#         max_iters : int  
#             max number of optimization iterations/rollouts
#         gamma : float
#             discount factor
#         set_sim_state_fn : function  
#             set state of simulator
#         get_sim_state_fn : function  
#             get current state of simulator
#         sim_step_fn : function  
#             step the simulatio using action
#         sample_mode : {'mean', 'sample'}  
#             how to choose action to be executed
#             'mean' plays the first mean action and  
#             'sample' samples from the distribution
#         batch_size : int
#             optimize for a batch of states
#         seed : int  
#             seed value
#         """
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.action_lows = action_lows
#         self.action_highs = action_highs
#         self.horizon = horizon
#         self.max_iters = max_iters
#         self.gamma = gamma
#         self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(1, horizon)
#         self._set_sim_state_fn = set_sim_state_fn
#         self._get_sim_state_fn = get_sim_state_fn
#         self._sim_step_fn = sim_step_fn        
#         self.sample_mode = sample_mode
#         self.batch_size = batch_size
#         self.seed = seed
#         self._seed(seed)
#         self.num_steps = 0

#     @abstractmethod
#     def _get_next_action(self, mode='mean'):
#         """
#         Get action to execute on the system based
#         on current control distribution
#         Parameters
#         ----------
#         mode : {'mean', 'sample'}  
#             how to choose action to be executed
#             'mean' plays the first mean action and  
#             'sample' samples from the distribution
#         """        
#         pass

#     # @abstractmethod
#     # def sample_actions(self):
#     #     """
#     #     Sample actions from current control distribution
#     #     """
#     #     pass
    
#     # @abstractmethod
#     # def _update_distribution(self, costs, act_seq):
#     #     """
#     #     Update current control distribution based on 
#     #     the results of rollouts
#     #     params - 
#     #         costs : np.ndarray 
#     #             step costs from rollouts
#     #         act_seq : np.ndarray 
#     #             action sequence sampled from control distribution
#     #     """
#     #     pass

#     @abstractmethod
#     def _shift(self):
#         """
#         Shift the current control distribution
#         to hotstart the next timestep
#         """
#         pass

#     @abstractmethod
#     def reset(self):
#         """
#         Reset the controller
#         """
#         pass

#     @abstractmethod
#     def _calc_val(self, cost_seq, act_seq):
#         """
#         Calculate value of state given 
#         rollouts from some policy
#         """
#         pass

#     @abstractmethod
#     def forward_pass(self, state):
#         """
#         A single forward pass to generate
#         new rollout/candidate trajectory

#         Parameters
#         ----------
#         state : starting state

#         Returns
#         -------
#         state_seq : list
#             list of states encountered
#         cost_seq : np.ndarray
#             sequence of costs encountered
#         act_seq : np.ndarray
#             sequence of actions taken
#         """
#         pass

#     @abstractmethod
#     def backward_pass(self, state_seq, cost_seq, act_seq):
#         """
#         A single backward pass to update 
#         value function. 

#         Parameters
#         ----------
#         state_seq : list
#             list of states encountered
#         cost_seq : np.ndarray
#             sequence of costs encountered
#         act_seq : np.ndarray
#             sequence of actions taken
#         """
#         pass

#     @abstractmethod
#     def converged(self):
#         """
#         Checks convergence
#         """
#         pass
    
#     @property
#     def sim_step_fn(self):
#         return self._sim_step_fn
    
#     @sim_step_fn.setter
#     def sim_step_fn(self, fn):
#         """
#         Set the rollout function used to 
#         given function pointer
#         """
#         self._sim_step_fn = fn
    
#     @property
#     def set_sim_state_fn(self):
#         return self._set_sim_state_fn
    
#     @set_sim_state_fn.setter
#     def set_sim_state_fn(self, fn):
#         """
#         Set function that sets the simulation 
#         environment to a particular state
#         """
#         self._set_sim_state_fn = fn
    
    
#     @property
#     def get_sim_state_fn(self):
#         return self._get_sim_state_fn
    
#     @get_sim_state_fn.setter
#     def get_sim_state_fn(self, fn):
#         """
#         Set function that gets the state from
#         simulator 
#         """
#         self._get_sim_state_fn = fn


#     def step(self, state, calc_val=False):
#         """
#         Optimize for best action at current state

#         Parameters
#         ----------
#         state : 
#             state to calculate optimal action from
        
#         calc_val : bool
#             If true, calculate the optimal value estimate
#             of the state along with action

#         Raises
#         ------
#         ValueError
#             If get_sim_state_fn, set_sim_state_fn or sim_step_fn are None

#         """

#         if ((self.get_sim_state_fn is None)
#              or (self.set_sim_state_fn is None)
#              or (self.sim_step_fn is None)
#             ):
#             raise ValueError("rollout_fn and set_sim_state_fn not set!!")

#         for itr in range(self.max_iters):
#             # print(itr)
#             #generate new rollout/candidate trajectory
#             state_seq, act_seq, cost_seq, kl_seq = self.forward_pass(copy.deepcopy(state))
#             # update distribution parameters/value functon etc.
#             delta_theta, delta_loss =self.backward_pass(state_seq, act_seq, cost_seq, kl_seq)
#             # check convergence
#             # TODO: Make this proper
#             if self.converged(delta_theta, delta_loss):
#                 break

#         #calculate best action
#         self.set_sim_state_fn(copy.deepcopy(state)) #set simulator to initial state
#         ftrs = self.get_sim_state_fn()
#         curr_action, mean, cov = self._get_next_action(ftrs.copy(), mode=self.sample_mode)
#         #calculate optimal value estimate if required
#         value = 0.0
#         # if calc_val:
#         #     cost_seq, act_seq = self.generate_rollouts(copy.deepcopy(state))
#         #     value = self._calc_val(cost_seq, act_seq)

#         self.num_steps += 1

#         # shift distribution/policy/value to hotstart next timestep
#         self._shift()

#         return curr_action, value




#     # def generate_rollouts(self, state):
#     #     """
#     #         Samples a batch of actions, rolls out trajectories for each particle
#     #         and returns the resulting observations, states, costs and 
#     #         actions
#     #      """
        
#     #     self._set_sim_state_fn(copy.deepcopy(state)) #set state of simulation
#     #     act_seq = self.sample_actions() #sample actions using current control distribution
#     #     cost_seq = self._rollout_fn(act_seq)  # rollout function returns the costs 
#     #     return cost_seq, act_seq

#     # def step(self, state, calc_val=False):
#     #     """
#     #         Optimize for best action at current state

#     #         :param state (np.ndarray): state to calculate optimal action from
#     #         :param calc_val (bool): If true, calculate the optimal value estimate
#     #                                 of the state while doing online MPC
#     #     """
#     #     if self._rollout_fn is None or self.set_sim_state_fn is None:
#     #         raise Exception("rollout_fn and set_sim_state_fn not set!!")

#     #     for _ in range(self.n_iters):
#     #         # generate random simulated trajectories
#     #         cost_seq, act_seq = self.generate_rollouts(copy.deepcopy(state))
#     #         # update distribution parameters
#     #         self._update_distribution(cost_seq, act_seq)
        
#     #     #calculate best action
#     #     curr_action = self._get_next_action(mode=self.sample_mode)
#     #     #calculate optimal value estimate if required
#     #     value = 0.0
#     #     if calc_val:
#     #         cost_seq, act_seq = self.generate_rollouts(copy.deepcopy(state))
#     #         value = self._calc_val(cost_seq, act_seq)

#     #     self.num_steps += 1
#     #     # shift distribution to hotstart next timestep
#     #     self._shift()

#     #     return curr_action, value

#     def get_optimal_value(self, state):
#         """
#         Calculate optimal value of a state, i.e 
#         value under optimal policy. 
        
#         :param num_particles(int): number of particles in rollout

#         """
#         self.reset() #reset the control distribution
#         _, value = self.step(state, calc_val=True)
#         return value



#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
        