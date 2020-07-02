"""
Random shooting algorithm that utilizes learned NN sampling policy
"""
from copy import deepcopy
import numpy as np
import torch

from mjmpc.control.controller import Controller
from mjmpc.utils.control_utils import cost_to_go, generate_noise, scale_ctrl
from mjmpc.utils import helpers

class RandomShootingNN(Controller):
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

        super(RandomShootingNN, self).__init__(d_state,
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
                               base_seed = self.seed + self.num_steps)
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
        obs_seq, act_seq, logprob_seq, cost_seq, done_seq, info_seq = self._rollout_fn(mode='mean',
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


    # def generate_rollouts(self, start_state):
    #     """
    #         Rolls out trajectories using current distribution
    #         and returns the resulting observations, costs,  
    #         actions

    #         Parameters
    #         ----------
    #         state : dict or np.ndarray
    #             Initial state to rollout from
                
    #      """
    #     self._set_sim_state_fn(deepcopy(start_state)) #set state of simulation
    #     obs_seq = np.zeros((self.num_particles, self.horizon, self.d_obs))
    #     act_seq = np.zeros((self.num_particles, self.horizon, self.d_action))
    #     state_seq = []
    #     cost_seq = np.zeros((self.num_particles, self.horizon))
    #     done_seq = np.zeros((self.num_particles, self.horizon), dtype=bool)

    #     num_chunks = int(self.num_particles / self.num_cpu)
    #     delta = self._sample_noise() #sample noise sequence
    #     with torch.no_grad():
    #         for i in range(num_chunks):
    #             self._set_sim_state_fn(deepcopy(start_state)) #set state of simulation
    #             curr_state = self._get_sim_state_fn()
    #             curr_obs = self._get_sim_obs_fn()
    #             curr_state_seq = [[] for _ in range(num_chunks)]
    #             for t in range(self.horizon):
    #                 #choose an action using mean + covariance
    #                 obs_torch = torch.from_numpy(np.float32(curr_obs))
    #                 mean, log_prob = self.model.get_action(obs_torch, mode='mean')
    #                 curr_act = mean.numpy().copy() + delta[i,t].copy()
    #                 # curr_act = scale_ctrl(curr_act, self.action_lows, self.action_highs)
                    
    #                 next_obs, rew, done, _ = self._sim_step_fn(curr_act) #step current action
    #                 next_state = self._get_sim_state_fn()

    #                 #Append data to sequence
    #                 batch_idxs = range(i*self.num_cpu, (i+1)*self.num_cpu)
    #                 obs_seq[batch_idxs, t, :] = curr_obs.copy()
    #                 act_seq[batch_idxs, t, :] = curr_act.copy()
    #                 cost_seq[batch_idxs, t] = -1.0 * rew
    #                 done_seq[batch_idxs, t] = done
    #                 for l in range(num_chunks): curr_state_seq[l].append(deepcopy(curr_state[l]))

    #                 curr_obs = next_obs.copy()
    #                 curr_state = deepcopy(next_state)
    #             state_seq.append(curr_state_seq)

    #         trajectories = dict(
    #             observations=obs_seq,
    #             actions=act_seq,
    #             costs=cost_seq,
    #             dones=done_seq,
    #             states=state_seq
    #         )

    #     return trajectories






    # def generate_rollouts(self, state):
    #     """
    #         Rolls out trajectories using current distribution
    #         and returns the resulting observations, costs,  
    #         actions

    #         Parameters
    #         ----------
    #         state : dict or np.ndarray
    #             Initial state to rollout from
                
    #      """
    #     self._set_sim_state_fn(deepcopy(state)) #set state of simulation
    #     self.start_state = state.copy()
    #     self.start_obs = self._get_sim_obs_fn()
    #     obs_seq = np.zeros((self.num_particles, self.horizon, self.d_obs))
    #     act_seq = np.zeros((self.num_particles, self.horizon, self.d_action))
    #     state_seq = []
    #     cost_seq = np.zeros((self.num_particles, self.horizon))
    #     done_seq = np.zeros((self.num_particles, self.horizon), dtype=bool)

    #     num_chunks = int(self.num_particles / self.num_cpu)
    #     delta = self._sample_noise() #sample noise sequence
    #     with torch.no_grad():
    #         for i in range(self.num_particles):
    #             self._set_sim_state_fn(deepcopy(state)) #set state of simulation
    #             curr_state = self._get_sim_state_fn()[0]
    #             curr_obs = self._get_sim_obs_fn()
    #             curr_state_seq = []
    #             for t in range(self.horizon):
    #                 #choose an action using mean + covariance
    #                 obs_torch = torch.from_numpy(np.float32(curr_obs))
    #                 mean = self.model.get_action(obs_torch, evaluate=True)
    #                 curr_act = mean.copy() + delta[i,t].copy()
    #                 curr_act = scale_ctrl(curr_act, self.action_lows, self.action_highs)
                    
    #                 next_obs, rew, done, _ = self._sim_step_fn(curr_act) #step current action
    #                 next_state = self._get_sim_state_fn()[0]

    #                 #Append data to sequence
    #                 obs_seq[i, t, :] = curr_obs.copy()
    #                 act_seq[i, t, :] = curr_act.copy()
    #                 cost_seq[i, t] = -1.0 * rew
    #                 curr_state_seq.append(deepcopy(curr_state))
    #                 done_seq[i, t] = done

    #                 curr_obs = next_obs[0].copy()
    #                 curr_state = deepcopy(next_state)
    #             state_seq.append(curr_state_seq)

    #         trajectories = dict(
    #             observations=obs_seq,
    #             actions=act_seq,
    #             costs=cost_seq,
    #             dones=done_seq,
    #             states=state_seq
    #         )

    #     return trajectories