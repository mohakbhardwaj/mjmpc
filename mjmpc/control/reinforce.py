"""
Random shooting algorithm that utilizes learned NN sampling policy
"""
from copy import deepcopy
import numpy as np
import torch

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
                 kl_delta=None,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 get_sim_obs_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
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
        torch.manual_seed(seed)
        self.lr = lr
        self.policy = policy
        self.baseline = baseline
        self.old_policy = deepcopy(policy)
        self._get_sim_obs_fn = get_sim_obs_fn
        self.kl_delta = kl_delta
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
        delta = None #sample noise sequence
        obs_seq, act_seq, act_info_seq, cost_seq, done_seq, next_obs_seq, info_seq = self._rollout_fn(mode='sample',
                                                                                                      noise=delta)

        trajectories = dict(
            observations=obs_seq,
            actions=act_seq,
            act_info_seq=act_info_seq,
            costs=cost_seq,
            next_obs_seq=next_obs_seq,
            dones=done_seq,
            infos=helpers.stack_tensor_dict_list(info_seq)
        )
        return trajectories

    def _update_distribution(self, trajectories):
        """
        Update policy from rollout trajectories
        using vanill policy gradients
        Parameters
        -----------
        """
        #compute cost to go and advantages
        trajectories["returns"] = cost_to_go(trajectories["costs"], self.gamma_seq)
        self.compute_advantages(trajectories)
        observations, actions, advantages = self.process_trajs(trajectories)

        #calculate CPI-surrogate loss
        surr = self.cpi_surrogate(observations, actions, advantages)
        surr_before = surr.item()
        if self.kl_delta is not None:
            raise NotImplementedError("Line search not implemented yet")
        else:
            #backpropagate loss to calculate gradients and update policy params
            self.policy.zero_grad()
            surr.backward()
            self.update_policy_parameters()

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

    def compute_advantages(self, trajs):
        trajs["advantages"] = trajs["returns"] - np.average(trajs["returns"], axis=0) #time dependent constant baseline
    
    def vpg_grad(self, observations, actions, advantages):
        surr = self.cpi_surrogate(observations, actions, advantages)
        grad = torch.autograd.grad(surr, self.policy.parameters())
        return grad
    
    def update_policy_parameters(self):
        for p in self.policy.parameters():
            p.data.sub_(p.grad.data * self.lr)

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
        advantages = np.concatenate([p for p in trajs["advantages"]])
        # Advantage whitening
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        return observations, actions, advantages
         
    def _shift(self):
        pass

    def reset(self):
        """
        Reset the controller
        """
        self.policy.reset()

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




