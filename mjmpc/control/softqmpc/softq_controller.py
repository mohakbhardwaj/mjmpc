"""
Iterative Linear Quadratic Regulator
"""
from copy import deepcopy
import numpy as np
from mjmpc.control.controller import Controller
from mjmpc.utils import control_utils, helpers
from .simple_quadratic_model import SimpleQuadraticQFunc
from .simple_quadratic_model_2 import SimpleQuadraticQFunc2
import torch

class SoftQMPC(Controller):
    def __init__(self, 
                 d_state,
                 d_obs,
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 gamma,
                 n_iters,
                 n_rollouts,
                 lam,
                 lr,
                 reg,
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

        super(SoftQMPC, self).__init__(d_state,
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
        self._get_sim_state_fn = get_sim_state_fn
        self._sim_step_fn = sim_step_fn
        self._sim_reset_fn = sim_reset_fn
        self._get_sim_obs_fn = get_sim_obs_fn
        self.n_rollouts = n_rollouts
        self.lam = lam
        self.lr = lr
        self.reg = reg
        self.model = SimpleQuadraticQFunc(self.d_obs, self.d_action)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.)
    
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
        with torch.no_grad():
            obs_torch = torch.from_numpy(np.float32(self.start_obs))
            mean, cov = self.model.get_act_mean_sigma(obs_torch, self.lam)
            mean = mean.numpy(); cov = cov.numpy()
            print(mean, cov)

        if mode == 'mean':
            next_action = mean.copy()#self.mean_action[0].copy()
        elif mode == 'sample':
            delta = control_utils.generate_noise(cov, [1.0, 0., 0.], (1,), self.seed + self.num_steps) #TODO: Different seed when doing multiple iterations
            next_action = mean.copy() + delta.copy()
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
        self.start_state = state.copy()
        self.start_obs = self._get_sim_obs_fn()
        obs_seq = np.zeros((self.n_rollouts, self.horizon, self.d_obs))
        act_seq = np.zeros((self.n_rollouts, self.horizon, self.d_action))
        state_seq = []
        entropy_seq = np.zeros((self.n_rollouts, self.horizon))
        cost_seq = np.zeros((self.n_rollouts, self.horizon))
        done_seq = np.zeros((self.n_rollouts, self.horizon), dtype=bool)

        with torch.no_grad():
            for i in range(self.n_rollouts):
                self._set_sim_state_fn(deepcopy(state)) #set state of simulation
                curr_state = self._get_sim_state_fn()[0]
                curr_obs = self._get_sim_obs_fn()
                curr_state_seq = []
                for t in range(self.horizon):
                    #choose an action using mean + covariance
                    obs_torch = torch.from_numpy(np.float32(curr_obs))
                    mean, cov = self.model.get_act_mean_sigma(obs_torch, self.lam)
                    mean = mean.numpy(); cov = cov.numpy()
                    delta = control_utils.generate_noise(cov, [1.0, 0., 0.], (1,), self.seed + 111*self.num_steps + 123*i + 999*t) #TODO: Different seed when doing multiple iterations
                    curr_act = mean.copy() + delta.copy()
                    
                    next_obs, rew, done, _ = self._sim_step_fn(curr_act) #step current action
                    next_state = self._get_sim_state_fn()[0]

                    #Append data to sequence
                    obs_seq[i, t, :] = curr_obs.copy()
                    act_seq[i, t, :] = curr_act.copy()
                    entropy_seq[i, t] = control_utils.gaussian_entropy(cov)
                    cost_seq[i, t] = -1.0 * rew
                    curr_state_seq.append(deepcopy(curr_state))
                    done_seq[i, t] = done

                    curr_obs = next_obs[0].copy()
                    curr_state = deepcopy(next_state)
                state_seq.append(curr_state_seq)

            trajectories = dict(
                observations=obs_seq,
                actions=act_seq,
                costs=cost_seq,
                entropies=entropy_seq,
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
        
        obs = trajectories["observations"]
        actions = trajectories["actions"]
        costs = trajectories["costs"]
        entropies = trajectories["entropies"]
        # # #Get terminal costs
        obs_input = torch.from_numpy(np.float32(obs[:,-1]))
        act_input = torch.from_numpy(np.float32(actions[:,-1]))
        with torch.no_grad():
            terminal_costs = self.model(obs_input, act_input)
            costs[:,-1] = terminal_costs.squeeze(-1)
        
        # #Get Q-targets
        #Q(s,a) = c(s,a) + \Sum_l=1^{N} \gamma^l(c(s_l, a_l) + H(a_l|s_l))
        # We first calculate normal cost+entropy-to-go then subtract 
        # entropy of first state
        total_costs = costs - self.lam * entropies
        targets = control_utils.cost_to_go(total_costs, self.gamma_seq)
        targets += self.lam * entropies 

        #Training
        #Ignore the final state and action in sequence
        obs_input = np.concatenate(obs[:,:-1], axis=0)
        act_input = np.concatenate(actions[:,:-1], axis=0)
        targets_input = np.concatenate(targets[:,:-1], axis=0)

        # #Randomly shuffle dataset (not needed necessarily)
        # num = obs_input.shape[0]
        # indices = np.random.permutation(range(num)) #random permutation
        # obs_input = obs_input[indices]
        # act_input = act_input[indices]
        # targets_input = targets_input[indices]

        obs_input = torch.from_numpy(np.float32(obs_input))
        act_input = torch.from_numpy(np.float32(act_input))
        targets_input = torch.from_numpy(np.float32(targets_input)).unsqueeze(-1)
        
        #Do a few gradient steps
        for i in range(1):
            self.optimizer.zero_grad()
            print(torch.max(targets_input))
            loss = self.model.loss(obs_input, act_input, targets_input, self.reg)
            loss.backward()
            # print("Loss = {0}".format(loss.item()))
            # self.model.print_gradients()
            self.optimizer.step()
            print(self.model)
            # with torch.no_grad():
                # self.model.grow_cov(0.1, self.lam)

    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step
        """
        # self.mean_action[:-1] = self.mean_action[1:]
        # if self.base_action == 'random':
        #     self.mean_action[-1] = np.random.normal(0, self.init_cov, self.d_action)
        # elif self.base_action == 'null':
        #     self.mean_action[-1] = np.zeros((self.d_action, ))
        # elif self.base_action == 'repeat':
        #     self.mean_action[-1] = self.mean_action[-2]
        # else:
        #     raise NotImplementedError("invalid option for base action during shift")
        # self.model.grow_cov(10)
        pass


    def reset(self):
        """
        Reset the controller
        """
        pass

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