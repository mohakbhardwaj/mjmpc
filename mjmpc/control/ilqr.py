"""
Iterative Linear Quadratic Regulator
"""
from copy import deepcopy
import numpy as np
from .controller import Controller
from mjmpc.utils import helpers

class ILQR(Controller):
    def __init__(self, 
                 d_state,
                 d_obs,
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 base_action,
                 gamma,
                 n_iters,
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
        base_action : str
            Action to append at the end when shifting solution to next timestep
            'random' : appends random action
            'null' : appends zero action
            'repeat' : repeats second to last action
        """

        super(ILQR, self).__init__(d_state,
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
                                   
        self._get_sim_state_fn = get_sim_state_fn
        self._sim_step_fn = sim_step_fn
        self._sim_reset_fn = sim_reset_fn
        self._get_sim_obs_fn = get_sim_obs_fn
        self.base_action = base_action
        self.mean_action = np.zeros((self.horizon, self.d_action))
    
    @property
    def sim_step_fn(self):
        return self._sim_step_fn
    
    @sim_step_fn.setter
    def sim_step_fn(self, fn):
        """
        Set function that steps the simulation
        environment given an action
        """
        self._sim_step_fn = fn

    @property
    def sim_reset_fn(self):
        return self._sim_reset_fn
     
    @sim_step_fn.setter
    def sim_reset_fn(self, fn):
        """
        Set function that steps the simulation
        environment given an action
        """
        self._sim_reset_fn = fn

    @property
    def get_sim_state_fn(self):
        return self._get_sim_state_fn
    
    
    @get_sim_state_fn.setter
    def get_sim_state_fn(self, fn):
        """
        Set function that gets the simulation 
        environment to a particular state
        """
        self._get_sim_state_fn = fn

    @property
    def get_sim_obs_fn(self):
        return self._get_sim_obs_fn
    
    @get_sim_obs_fn.setter
    def get_sim_obs_fn(self, fn):
        self._get_sim_obs_fn = fn
        
    def _get_next_action(self, mode='mean'):
        """
        Get action to execute on the system based
        on current control distribution
        Parameters
        ----------
        mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
        """  

        if mode == 'mean':
            next_action = self.mean_action[0].copy()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return next_action


    def generate_rollouts(self, state):
        """
            Rolls out trajectories using current distribution
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to rollout from
                
         """
        
        self._set_sim_state_fn(deepcopy(state)) #set state of simulation
        obs_seq = np.zeros((self.horizon, self.d_obs))
        act_seq = np.zeros((self.horizon, self.d_action))
        state_seq = []
        cost_seq = np.zeros((self.horizon, 1))
        done_seq = np.zeros((self.horizon, 1), dtype=bool)

        curr_state = self._get_sim_state_fn()[0]
        curr_obs = self._get_sim_obs_fn()
        for t in range(self.horizon):
            #use control matrix + sim_step_fn
            curr_act = np.zeros(self.d_action) #TODO: You choose an action
            next_obs, rew, done, _ = self._sim_step_fn(curr_act) #step current action
            next_state = self._get_sim_state_fn()[0]

            #Append data to sequence
            obs_seq[t, :] = curr_obs.copy()
            act_seq[t, :] = curr_act.copy()
            cost_seq[t, :] = -1.0 * rew
            state_seq.append(deepcopy(curr_state))
            done_seq[t, :] = done

            curr_obs = next_obs.copy()
            curr_state = deepcopy(next_state)
                
        trajectories = dict(
            observations=obs_seq,
            actions=act_seq,
            costs=cost_seq,
            dones=done_seq,
            states=state_seq
        )

        return trajectories

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
        self.mean_action = np.zeros((self.horizon, self.d_action))

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