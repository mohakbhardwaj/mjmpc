"""
Random shooting algorithm that utilizes learned NN sampling policy
"""
from copy import deepcopy
import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from mjmpc.control.controller import Controller
from mjmpc.utils.control_utils import cost_to_go, generate_noise, scale_ctrl
from mjmpc.utils import helpers

class Reinforce(Controller):
    def __init__(self, 
                 d_state,
                 d_obs,
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 gamma,
                 n_iters,
                 num_particles,
                 lr,
                 policy,
                 baseline,
                 delta_kl=None,
                 max_linesearch_iters=100,
                 delta_reg = 0.0,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 get_sim_obs_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 filter_coeffs = [1., 0., 0.],
                 seed=0,
                 verbose=False):
        """
        Parameters
        __________

        """
        super(Reinforce, self).__init__(d_state,
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
        # torch.manual_seed(seed)
        self.num_particles = num_particles
        self.lr = lr
        self.policy = policy
        self.baseline = baseline
        self.old_policy = deepcopy(policy)
        self._get_sim_obs_fn = get_sim_obs_fn
        self.delta_kl = delta_kl
        self.delta_reg = delta_reg
        self.max_linesearch_iters = max_linesearch_iters
        self.filter_coeffs = filter_coeffs
        self.verbose = verbose

    @property
    def get_sim_obs_fn(self):
        return self._get_sim_obs_fn
    
    @get_sim_obs_fn.setter
    def get_sim_obs_fn(self, fn):
        self._get_sim_obs_fn = fn

    def _get_next_action(self, mode='mean'):
        with torch.no_grad():
            obs_torch = torch.FloatTensor(self.start_obs)
            action, _ = self.policy.get_action(obs_torch, mode='mean')
            action = action.detach().numpy()
        return action.copy()

    def sample_noise(self):
        """
            Generate correlated noisy samples using autoregressive process
        """
        beta_0, beta_1, beta_2 = self.filter_coeffs
        N = self.d_action
        eps = self.np_random.multivariate_normal(mean=np.zeros((N,)), 
                                                 cov = np.eye(self.d_action), 
                                                 size=(self.num_particles, self.horizon))
        for i in range(2, eps.shape[1]):
            eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
        return eps 

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
        self.start_obs = deepcopy(self._get_sim_obs_fn()[0,:])
        noise = self.sample_noise() #sample noise sequence from N(0,1)
        obs_seq, act_seq, act_info_seq, cost_seq, done_seq, next_obs_seq, info_seq = self._rollout_fn(mode='sample',
                                                                                                      noise=noise)

        trajectories = dict(
            observations=obs_seq,
            actions=act_seq,
            act_info_seq=act_info_seq,
            costs=cost_seq,
            next_observations=next_obs_seq,
            dones=done_seq,
            infos=helpers.stack_tensor_dict_list(info_seq)
        )
        # print(trajectories["observations"][:,-1])
        # print(trajectories["actions"][:,-1])
        # input('....')

        return trajectories

    def _update_distribution(self, trajectories):
        """
        Update policy from rollout trajectories
        using vanill policy gradients
        Parameters
        -----------
        """
        #compute cost to go and advantages
        self.compute_returns(trajectories)
        self.compute_advantages(trajectories)
        observations, actions, returns, advantages = self.process_trajs(trajectories)

        #calculate CPI-surrogate loss
        if self.verbose:
            with torch.no_grad():
                surr_before = self.cpi_surrogate(observations, actions, advantages).item()
        
        #Backtracking line search
        if self.delta_kl is not None:
            curr_param_dict = deepcopy(self.policy.state_dict())
            curr_lr = self.lr
            #calculate loss and backpropagate gradients
            surr = self.cpi_surrogate(observations, actions, advantages)
            self.policy.zero_grad()
            surr.backward()
            #linesearch on gradient direction for max_linesearch_iters
            for ctr in range(self.max_linesearch_iters):
                self.update_policy_parameters(curr_lr)
                with torch.no_grad():
                    obs_input = torch.FloatTensor(observations)
                    new_distrib = self.policy.action_distribution(obs_input)
                    old_distrib = self.old_policy.action_distribution(obs_input)
                # avg KL(\pi_new || \pi_old)
                mean_kl_div = torch.mean(kl_divergence(new_distrib, old_distrib))
                # print(mean_kl_div)
                if mean_kl_div <= self.delta_kl:
                    break
                else:
                    #reset policy parameters and decrease learning rate
                    # print('backtracking') 
                    self.policy.load_state_dict(curr_param_dict)
                    curr_lr *= 0.9
        else:
            #backpropagate loss and update policy params
            surr = self.cpi_surrogate(observations, actions, advantages)
            self.policy.zero_grad()
            surr.backward()
            self.update_policy_parameters(self.lr)

        if self.verbose:
            print("Gradients")
            self.policy.print_gradients()
            print("Updated parameters")
            self.policy.print_parameters()
            with torch.no_grad():
                surr_after = self.cpi_surrogate(observations, actions, advantages).item()
            surr_improvement = surr_before - surr_after
            print("Surrogate improvement = {}".format(surr_improvement))

        #update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        #update baseline using regression
        if self.baseline is not None:
            self.fit_baseline(observations, returns, self.delta_reg)

    def cpi_surrogate(self, observations, actions, advantages):
        observations = torch.FloatTensor(observations)
        actions = torch.FloatTensor(actions)

        new_log_prob = self.policy.log_prob(observations, actions)
        with torch.no_grad():
            old_log_prob = self.old_policy.log_prob(observations, actions)
            advantages = torch.FloatTensor(advantages)

        likelihood_ratio = torch.exp(new_log_prob - old_log_prob)
        surr = torch.mean(likelihood_ratio * advantages.unsqueeze(-1))
        return surr

    def compute_returns(self, trajs):
        if self.baseline is None:
            trajs["returns"] = cost_to_go(trajs["costs"], self.gamma_seq)
        else:
            with torch.no_grad():
                N, H = trajs["costs"].shape
                gamma_seq_a = np.pad(self.gamma_seq, ((0,0),(0,1)), constant_values=self.gamma ** H)  
                next_obs_last = torch.FloatTensor(trajs["next_observations"][:,0])
                obs_last = torch.FloatTensor(trajs["observations"][:,0])
                term_cost = self.baseline(next_obs_last).detach().numpy()#.view(N,1)
                costs_new = np.concatenate((trajs["costs"], term_cost), axis=-1)
                returns = cost_to_go(costs_new, gamma_seq_a)
                trajs["returns"] = returns[:,:-1]                


    def compute_advantages(self, trajs):
        if self.baseline is None:
            baseline_vals = np.average(trajs["returns"], axis=0) #time dependent constant baseline
        else:
            with torch.no_grad():
                obs = torch.FloatTensor(np.concatenate([p for p in trajs["observations"]]))
                baseline_vals = self.baseline(obs)
                baseline_vals = baseline_vals.view(*torch.Size(trajs["returns"].shape))
                baseline_vals = baseline_vals.detach().numpy()
        
                #compare
                # bval_list = np.zeros(trajs["returns"].shape)
                # for i, p in enumerate(trajs["observations"]):
                #     curr_bval = self.baseline(torch.FloatTensor(p))
                #     bval_list[i] = curr_bval.detach().numpy()[:,0]
                # print(np.max(baseline_vals - bval_list))
                # # print(bval_list[0])
                # print(trajs['returns'][0])
                # input('....')
        trajs["advantages"] = trajs["returns"] - baseline_vals 

        # print('naive baseline', np.average(trajs["returns"], axis=0)) #time dependent constant baseline
        # print(baseline_vals)
        # input('...')
    
    def vpg_grad(self, observations, actions, advantages):
        surr = self.cpi_surrogate(observations, actions, advantages)
        grad = torch.autograd.grad(surr, self.policy.parameters())
        return grad        
    
    def update_policy_parameters(self, learning_rate):
        for p in self.policy.parameters():
            p.data.sub_(p.grad.data * learning_rate)

    def fit_baseline(self, observations, returns, delta_reg):
        observations = torch.FloatTensor(observations)
        returns = torch.FloatTensor(returns)
        errs = self.baseline.fit(observations, returns, delta_reg, True)
        print(errs)

    def process_trajs(self, trajs):
        """ 
            Return training data from given
            trajectories

            :return observations (np.ndarray): 
            :return actions (np.ndarray):
            :return rewards (np.ndarray):
            :return states (dict of np.ndarray): [len(trajs)) * len(trajs[0])]
            :return next_observations (np.ndarray)
            :return next_states (list)
            :return stats
        """
        observations = np.concatenate([p for p in trajs["observations"]])
        actions = np.concatenate([p for p in trajs["actions"]])
        returns = np.concatenate([p for p in trajs["returns"]])
        advantages = np.concatenate([p for p in trajs["advantages"]])
        # Advantage whitening
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        return observations, actions, returns, advantages
         
    def _shift(self):
        self.policy.grow_cov()

    def reset(self, seed=None):
        """
        Reset the controller
        """
        if seed is not None:
            self.seed_val = self.seed(seed)
        self.policy.reset()
        self.old_policy.load_state_dict(self.policy.state_dict())
        # self.policy.print_parameters()
        # self.old_policy.print_parameters()

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




