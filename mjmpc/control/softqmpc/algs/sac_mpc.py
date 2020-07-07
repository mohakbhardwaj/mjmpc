"""
Random shooting algorithm that utilizes learned NN sampling policy
"""
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from mjmpc.control.controller import Controller
from mjmpc.control.softqmpc.algs.sac import ReplayMemoryTraj, soft_update, hard_update
from mjmpc.utils.control_utils import cost_to_go, generate_noise, scale_ctrl
from mjmpc.utils import helpers

# class SACMPC(Controller):
#     def __init__(self, 
#                  d_state,
#                  d_obs,
#                  d_action,                
#                  action_lows,
#                  action_highs,
#                  horizon,
#                  gamma,
#                  step_size,
#                  filter_coeffs,
#                  n_iters,
#                  num_particles,
#                  init_cov,
#                  base_action,
#                  policy,
#                  critic,
#                  lr,
#                  alpha,
#                  tau,
#                  train_batch_size,
#                  updates_per_step,
#                  start_steps,
#                  target_update_interval,
#                  replay_size,
#                  automatic_entropy_tuning,
#                  set_sim_state_fn=None,
#                  rollout_fn=None,
#                  sample_mode='mean',
#                  batch_size=1,
#                  seed=0):
class SACMPC(Controller):
    def __init__(self, param_dict):
        """
        Parameters
        __________

        """

        super(SACMPC, self).__init__(param_dict['d_state'],
                                     param_dict['d_obs'],
                                     param_dict['d_action'],
                                     param_dict['action_lows'], 
                                     param_dict['action_highs'],
                                     param_dict['horizon'],
                                     param_dict['gamma'],  
                                     param_dict['n_iters'],
                                     param_dict['set_sim_state_fn'],
                                     param_dict['rollout_fn'],
                                     param_dict['sample_mode'],
                                     param_dict['batch_size'],
                                     param_dict['seed'])
        self.param_dict = param_dict
        torch.manual_seed(seed)
        self.init_cov = np.array([param_dict['init_cov']] * self.d_action)
        self.mean_action = np.zeros(shape=(self.horizon, self.d_action))
        self.num_particles = param_dict['num_particles']
        self.base_action = param_dict['base_action']
        self.cov_action = np.diag(self.init_cov)
        self.step_size = param_dict['step_size']
        self.filter_coeffs = param_dict['filter_coeffs']
        self.lr = param_dict['lr']
        self.alpha = param_dict['alpha']
        self.tau = param_dict['tau']
        self.automatic_entropy_tuning = param_dict['automatic_entropy_tuning']
        self.train_batch_size = param_dict['train_batch_size']
        self.updates_per_step = param_dict['updates_per_step']
        self.start_steps = param_dict['start_steps']
        self.target_update_interval = param_dict['target_update_interval']
        self.replay_size = param_dict['replay_size']
        self._get_sim_obs_fn = None

        #policy and qfunction objects
        self.policy = policy
        self.critic = critic
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = deepcopy(self.critic)

        self.memory = ReplayMemoryTraj(self.replay_size)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.d_action)).item() #.to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True) #, device=self.device
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

    @property
    def get_sim_obs_fn(self):
        return self._get_sim_obs_fn
    
    @get_sim_obs_fn.setter
    def get_sim_obs_fn(self, fn):
        self._get_sim_obs_fn = fn

    def _get_next_action(self, mode='mean'):
        with torch.no_grad():
            obs_torch = torch.FloatTensor(self.start_obs).unsqueeze(0)
            next_action, _ = self.policy.get_action(obs_torch, mode)
            next_action = next_action.numpy().copy().reshape(self.d_action,) 

        # if mode == 'mean':
        #     next_action = self.mean_action[0].copy()
        # elif mode == 'sample':
        #     delta = generate_noise(self.cov_action, self.filter_coeffs,
        #                            shape=(1, 1), base_seed=self.seed + self.num_steps)
        #     next_action = self.mean_action[0].copy() + delta.reshape(self.d_action).copy()
        # else:
        #     raise ValueError('Unidentified sampling mode in get_next_action')
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
        self.start_obs = (self.get_sim_obs_fn()[0]).copy()
        # delta = self._sample_noise() #sample noise sequence
        delta = None
        obs_seq, act_seq, logprob_seq, cost_seq, done_seq, next_obs_seq, info_seq = self._rollout_fn(mode='sample',
                                                                                                     noise=delta)
        trajectories = dict(
            observations=obs_seq,
            next_observations=next_obs_seq,
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
        # costs = trajectories["costs"].copy()
        # actions = trajectories["actions"].copy()
        # Q = cost_to_go(costs, self.gamma_seq)
        # best_id = np.argmin(Q, axis = 0)[0]
        # self.mean_action = (1.0 - self.step_size) * self.mean_action +\
        #                     self.step_size * actions[best_id]

        # Q = cost_to_go(trajectories['costs'], self.gamma_seq)
        # log_prob_to_go = cost_to_go(trajectories['log_probs'], self.gamma_seq)
        # trajectories["cost_to_go"] = Q
        # trajectories["log_prob_to_go"] = log_prob_to_go
        # for k in trajectories.keys():
        #     if k is not 'infos':
        #         trajectories[k] = np.concatenate(trajectories[k], axis=0)
        self.memory.push(trajectories)
        self.update_parameters(True)

    def update_parameters(self, suppress_stdout=True):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.train_batch_size)

        state_batch = torch.FloatTensor(state_batch)#.to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch)#.to(self.device)
        action_batch = torch.FloatTensor(action_batch)#.to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)#.to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)#.to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.num_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def get_agent_infos(self):
        return None

    # def fit(self, traj_data, suppress_stdout=True):
    #     epoch_losses_qfunc = np.zeros(shape=(self.num_epochs,))
    #     epoch_losses_policy = np.zeros(shape=(self.num_epochs,))
    #     for i in range(self.num_epochs):
    #         batch_losses_qfunc = []
    #         batch_losses_policy = []
    #         for batch_data in self.train_batches(traj_data, self.train_batch_size, 1):
    #             obs_batch = torch.FloatTensor(batch_data['obs_batch'])
    #             next_obs_batch = torch.FloatTensor(batch_data['next_obs_batch'])
    #             action_batch = torch.FloatTensor(batch_data['action_batch'])
    #             reward_batch = -1.0 * torch.FloatTensor(batch_data['cost_batch'])
    #             done_batch = torch.FloatTensor(batch_data['done_batch'])
    #             #Update Q-function
    #             #Generate Q-targets
    #             with torch.no_grad():
    #                 next_action, next_log_pi, _ = self.policy.sample(next_obs_batch)
    #                 # qf1_next_target, qf2_next_target = self.qfunc(next_obs_batch, next_obs_action)
    #                 qf1_next_target, qf2_next_target = self.qfunc_target(next_obs_batch, next_action)
    #                 min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.lam * next_log_pi
    #                 next_q_value = reward_batch + self.gamma * (min_qf_next_target) 

    #             qf1, qf2 = self.qfunc(obs_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
    #             qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    #             qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    #             qf_loss = qf1_loss + qf2_loss

    #             self.qfunc_optim.zero_grad()
    #             qf_loss.backward()
    #             self.qfunc_optim.step()

    #             pi, log_pi, _ = self.policy.sample(obs_batch)

    #             qf1_pi, qf2_pi = self.qfunc(obs_batch, pi)
    #             min_qf_pi = torch.min(qf1_pi, qf2_pi)

    #             policy_loss = ((self.lam * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

    #             self.policy_optim.zero_grad()
    #             policy_loss.backward()
    #             self.policy_optim.step()

    #             batch_losses_qfunc.append(qf_loss.item())
    #             batch_losses_policy.append(policy_loss.item())
            
    #         epoch_losses_qfunc[i] = np.average(batch_losses_qfunc)
    #         epoch_losses_policy[i] = np.average(batch_losses_policy)

    #         if not suppress_stdout:
    #             print('Training epoch = {0}, Avg. Policy Loss = {1}, Avg. Qfunc Loss = {2}'.format(i, 
    #                                                                                         epoch_losses_policy[i],
    #                                                                                         epoch_losses_qfunc[i]))
        
    #     #update target network
    #     if self.num_steps%self.update_target_steps == 0:
    #         self.qfunc_target.load_state_dict(self.qfunc.state_dict())
        
    #     return epoch_losses_policy, epoch_losses_qfunc

    
    # def train_batches(self, trajectories, batch_size, ensemble_size):
    #     """

    #     """
    #     #unpack data
    #     observations = trajectories["observations"]; actions = trajectories["actions"]
    #     costs = trajectories["costs"]; log_probs = trajectories["log_probs"]; 
    #     cost_to_go = trajectories["cost_to_go"]; log_prob_to_go = trajectories["log_prob_to_go"]
    #     dones = trajectories["dones"]; next_observations = trajectories["next_observations"]
    #     num, d_obs = observations.shape
    #     indices = [self.np_random.permutation(range(num)) for _ in range(ensemble_size)]
    #     indices = np.stack(indices).T
    #     for i in range(0, num, batch_size):
    #         j = min(num, i + batch_size)

    #         if (j - i) < batch_size and i != 0:
    #             # drop incomplete last batch
    #             return

    #         batch_size = j - i

    #         batch_indices = indices[i:j]
    #         batch_indices = batch_indices.flatten()

    #         obs_batch      = observations[batch_indices]
    #         action_batch   = actions[batch_indices]
    #         cost_batch     = costs[batch_indices]
    #         log_prob_batch = log_probs[batch_indices]
    #         cost_to_go_batch = cost_to_go[batch_indices]
    #         log_prob_to_go_batch = log_prob_to_go[batch_indices]
    #         done_batch     = dones[batch_indices] 
    #         next_obs_batch = next_observations[batch_indices]

    #         if ensemble_size>1:shape=[ensemble_size, batch_size]
    #         else: shape= [batch_size]          

    #         obs_batch      = obs_batch.reshape(*shape, d_obs)
    #         action_batch   = action_batch.reshape(*shape, 1)
    #         cost_batch     = cost_batch.reshape(*shape, 1)
    #         log_prob_batch = log_prob_batch.reshape(*shape, 1)
    #         cost_to_go_batch = cost_to_go_batch.reshape(*shape, 1)
    #         log_prob_to_go_batch = log_prob_to_go_batch.reshape(*shape, 1)
    #         done_batch     = done_batch.reshape(*shape, 1)
    #         next_obs_batch = next_obs_batch.reshape(*shape, d_obs)
            
    #         yield {'obs_batch': obs_batch, 
    #                 'action_batch': action_batch,
    #                 'next_obs_batch': next_obs_batch,
    #                 'cost_batch': cost_batch,
    #                 'cost_to_go_batch': cost_to_go_batch,
    #                 'log_prob_batch': log_prob_batch,
    #                 'log_prob_to_go_batch': log_prob_to_go_batch,
    #                 'done_batch': done_batch,
    #                 }

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
        pass

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
