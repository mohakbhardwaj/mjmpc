
"""
TODO: Currently works for batch_size = 1 only
"""
#!/usr/bin/env python
import copy
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from .logger import logger
from .timer import timeit


class Buffer(Dataset):
    def __init__(self, d_state, d_action, d_sim_state, max_length=1e6, ensemble_size=1, batch_size=1):
        self._max_length = int(max_length)
        self._ensemble_size = ensemble_size; self._batch_size = batch_size
        self._d_state = d_state
        self._d_action = d_action
        self._d_sim_state = d_sim_state
        self._states = torch.zeros(self._max_length, int(d_state)).float()
        self._actions = torch.zeros(self._max_length, int(d_action)).float()
        self._next_states = torch.zeros(self._max_length, int(d_state)).float()
        self._next_actions = torch.zeros(self._max_length, int(d_action)).float()
        self._sim_states = torch.zeros(self._max_length, int(d_sim_state)).float()
        self._sim_next_states = torch.zeros(self._max_length, int(d_sim_state)).float()
        self._rewards = torch.zeros(self._max_length,1).float()
        self._dones = torch.zeros(self._max_length,1).byte()
        self._successes = torch.zeros(self._max_length,1).byte()
        self._expert_targets = torch.zeros(self._max_length,1).float()
        self._value_targets = torch.zeros(self._max_length,1).float()
        self.normalizer = None
        self._n_elements = 0

        self._num_episodes_done = 0.0
        self._tot_reward = 0.0
        self._avg_reward = 0.0
        self._avg_reward_old = 0.0
        self._sk_reward = 0.0
        self._sk_reward_old = 0.0
        self._min_reward = 0.0
        self._max_reward = 0.0
        self._curr_ep_rew = 0.0
        self._num_saved_already = 0

    @property
    def is_empty(self):
        return self._n_elements == 0

    @property
    def state_mean(self):
        num = len(self)
        if num == 0: return torch.zeros(1,self._d_state)
        return torch.mean(self._states[0:num], dim=0)

    @property
    def state_std(self):
        num = len(self)
        if num < 2: return torch.zeros(1,self._d_state)
        return torch.std(self._states[0:num], dim=0)

    @property
    def action_mean(self):
        num = len(self)
        if num == 0: return torch.zeros(1,self._d_action)
        return torch.mean(self._actions[0:num], dim=0)

    @property
    def action_std(self):
        num = len(self)
        if num < 2: return torch.zeros(1,self._d_action)
        return torch.std(self._actions[0:num], dim=0)

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def add(self, transition):
        state, action, next_state, next_action = torch.from_numpy(transition[0]).float().clone(), torch.from_numpy(transition[1]).float().clone(),\
                                                 torch.from_numpy(transition[2]).float().clone(), torch.from_numpy(transition[3]).float().clone()
        reward = torch.tensor(transition[4]).float().clone()
        done = torch.tensor(transition[5]).byte().clone()
        success = torch.tensor(transition[6]).byte().clone()
        sim_state = torch.tensor(transition[7]).float().clone()
        sim_next_state = torch.tensor(transition[8]).float().clone()
        idx = self._n_elements % self._max_length


        self._states[idx] = state
        self._actions[idx] = action
        self._next_states[idx] = next_state
        self._next_actions[idx] = next_action
        self._rewards[idx] = reward
        self._dones[idx] = done
        self._successes[idx] = success
        self._sim_states[idx] = sim_state
        self._sim_next_states[idx] = sim_next_state
        if len(transition) > 9:
            expert_target = torch.from_numpy(transition[9]).float().clone()
            self._expert_targets[idx] = expert_target

        self._n_elements += 1
        self._curr_ep_rew += reward.item()

        if self.normalizer is not None:
            self.normalizer.update(state, action, next_state - state)

        if self._n_elements > self._max_length:
            logger.warn("buffer full, rewriting over old samples")

        if done:
            self._num_episodes_done += 1
            self.update_reward_stats()
            self._curr_ep_rew = 0.0
        

    def update_reward_stats(self):
        # end_idxs = torch.nonzero(self._dones)[:,0] + 1
        # end_idx = end_idxs[-1]
        # start_idx = 0
        # start_idx = -1 - self._curr_ep_len
        # curr_rew_sum = torch.sum(self._rewards[start_idx:]).item() #total reward for final episode

        self._tot_reward = self._curr_ep_rew
        if self._num_episodes_done <= 1:
            self._avg_reward = self._curr_ep_rew
            self._avg_reward_old = self._curr_ep_rew
            self._sk_reward = 0.0
            self._sk_reward_old = 0.0
            self._min_reward = self._curr_ep_rew
            self._max_reward = self._curr_ep_rew
        else:
            self._avg_reward = self._avg_reward_old +  (self._curr_ep_rew - self._avg_reward_old)/self._num_episodes_done*1.0
            self._sk_reward  = self._sk_reward_old  +  (self._curr_ep_rew - self._avg_reward_old) * (self._curr_ep_rew - self._avg_reward)
            if self._curr_ep_rew <= self._min_reward: self._min_reward = self._curr_ep_rew
            elif self._curr_ep_rew >= self._max_reward: self._max_reward = self._curr_ep_rew
            #setup for next iteration
            self._avg_reward_old = copy.copy(self._avg_reward)
            self._sk_reward_old = copy.copy(self._sk_reward)

    def __len__(self):
        return min(self._n_elements, self._max_length)

    def __getitem__(self, idx):

        if len(self._states) == 0:
            raise ValueError("No experiences in the buffer")

        sample = {'state': self._states[idx],
                  'action': self._actions[idx],
                  'next_state': self._next_states[idx],
                  'next_action': self._next_actions[idx],
                  'reward': self._rewards[idx],
                  'done': self._dones[idx],
                  'success': self._successes[idx],
                  'expert_target': self._expert_targets[idx],
                  'sim_state': self._sim_states[idx],
                  'sim_next_state': self._sim_next_states[idx]} #, 'expert_target': None}

        return sample

    @property
    def max_length(self):
        return self._max_length


    def train_batches(self, batch_size):
        """
        return an iterator of batches
        Args:
            batch_size: number of samples to be returned
        Returns:
            state of size (ensemble_size, batch_size, d_state)
            action of size (ensemble_size, batch_size, d_action)
            next state of size (ensemble_size, batch_size, d_state)
        """
        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(self._ensemble_size)]
        indices = np.stack(indices).T
        #self.generate_value_targets()
        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
            # drop last incomplete last batch
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states          = self._states[batch_indices]
            actions         = self._actions[batch_indices]
            next_states     = self._next_states[batch_indices]
            next_actions    = self._next_actions[batch_indices]
            rewards         = self._rewards[batch_indices]
            dones           = self._dones[batch_indices]
            successes       = self._successes[batch_indices]
            exp_targets     = self._expert_targets[batch_indices]
            sim_states      = self._sim_states[batch_indices]
            sim_next_states = self._sim_next_states[batch_indices]
            # value_targets = self._value_targets[batch_indices]

            states          = states.reshape(self._ensemble_size, batch_size, self._d_state)
            actions         = actions.reshape(self._ensemble_size, batch_size, self._d_action)
            next_states     = next_states.reshape(self._ensemble_size, batch_size, self._d_state)
            next_actions    = next_actions.reshape(self._ensemble_size, batch_size, self._d_action )
            rewards         = rewards.reshape(self._ensemble_size, batch_size, 1)
            dones           = dones.reshape(self._ensemble_size, batch_size, 1)
            successes       = successes.reshape(self._ensemble_size, batch_size, 1)
            exp_targets     = exp_targets.reshape(self._ensemble_size, batch_size, 1)
            sim_states      = sim_states.reshape(self._ensemble_size, batch_size, self._d_sim_state)
            sim_next_states = sim_next_states.reshape(self._ensemble_size, batch_size, self._d_sim_state)
            # value_targets = value_targets.reshape(self._ensemble_size, batch_size, 1)

            yield states.clone(), actions.clone(), next_states.clone(), next_actions.clone(), rewards.clone(), dones.clone(), successes.clone(), exp_targets.clone(), sim_states.clone(), sim_next_states.clone() #value_targets


    def generate_value_targets(self, gamma=1, mode='mc'):
        end_idxs = torch.nonzero(self._dones)[:,0] + 1
        start_idx = 0

        for end_idx in end_idxs:
            rewards = self._rewards[start_idx:end_idx]
            cum_rew = torch.cumsum(rewards.flip(dims=(0,1)), dim=0).flip(dims=(0,1))
            self._value_targets[start_idx:end_idx] = cum_rew
            start_idx = end_idx

    def update_exp_targets(self, exp_targets):
        self._expert_targets = torch.from_numpy(exp_targets).float().clone()


    ###############
    ### Logging ###
    ###############

    def log(self, mode='train'):
        # end_idxs = torch.nonzero(self._dones)[:,0] + 1
        # disc_returns = []
        # start_idx = 0
        # # end_idxs = torch.cat((end_idxs, torch.tensor([-1], dtype=torch.int64)))
        # total_reward = torch.sum(self._rewards).item()
        # if len(end_idxs) > 0:

        #     for end_idx in end_idxs:
        #         rewards = self._rewards[start_idx:end_idx]
        #         # disc_ret = 0.0
        #         # for i,r in enumerate(rewards):
        #             # disc_ret += (gamma ** i) * r.item()
        #         disc_returns.append(torch.sum(rewards).item())
        #         # disc_returns.append(copy.deepcopy(disc_ret))
        #         start_idx = end_idx
        #     logger.record_tabular('ReturnAvgV_'+mode, np.mean(disc_returns))
        #     logger.record_tabular('ReturnStdV_'+mode, np.std(disc_returns, ddof=1))
        #     logger.record_tabular('ReturnMinV_'+mode, np.min(disc_returns))
        #     logger.record_tabular('ReturnMaxV_'+mode, np.max(disc_returns))

        std_rew = 0.0
        if self._num_episodes_done > 1:
            cov_rew = self._sk_reward/(self._num_episodes_done-1.0)
            std_rew = np.sqrt(cov_rew)

        logger.record_tabular('ReturnCurr'+mode, self._tot_reward)
        logger.record_tabular('ReturnAvg_'+mode, self._avg_reward)
        logger.record_tabular('ReturnStd_'+mode, std_rew)
        logger.record_tabular('ReturnMin_'+mode, self._min_reward)
        logger.record_tabular('ReturnMax_'+mode, self._max_reward)



    ######################
    ### Saving/Loading ###
    ######################

    def save(self, folder):
        states_np          = self._states[self._num_saved_already:self._n_elements].numpy()
        actions_np         = self._actions[self._num_saved_already:self._n_elements].numpy()
        next_states_np     = self._next_states[self._num_saved_already:self._n_elements].numpy()
        next_actions_np    = self._next_actions[self._num_saved_already:self._n_elements].numpy()
        rewards_np         = self._rewards[self._num_saved_already:self._n_elements].numpy()
        dones_np           = self._dones[self._num_saved_already:self._n_elements].numpy()
        success_np         = self._successes[self._num_saved_already:self._n_elements].numpy()
        expert_targets_np  = self._expert_targets[self._num_saved_already:self._n_elements].numpy()
        sim_states_np      = self._sim_states[self._num_saved_already:self._n_elements].numpy()
        sim_next_states_np = self._sim_next_states[self._num_saved_already:self._n_elements].numpy()


        if not os.path.exists(folder): os.makedirs(folder)
        np.savetxt(folder+"/buffer_states.csv",states_np)
        np.savetxt(folder+"/buffer_actions.csv", actions_np)
        np.savetxt(folder+"/buffer_next_states.csv", next_states_np)
        np.savetxt(folder+"/buffer_next_actions.csv", next_actions_np)
        np.savetxt(folder+"/buffer_rewards.csv",rewards_np)
        np.savetxt(folder+"/buffer_dones.csv", dones_np)
        np.savetxt(folder+"/buffer_successes.csv", success_np)
        np.savetxt(folder+"/buffer_expert_targets.csv", expert_targets_np)
        np.savetxt(folder+"/buffer_sim_states.csv", sim_states_np)
        np.savetxt(folder+"/buffer_sim_next_states.csv", sim_next_states_np)

        self._num_saved_already = self._n_elements

    def load(self, folder):
        states_np          = np.loadtxt(os.path.join(folder,"buffer_states.csv"))
        actions_np         = np.loadtxt(os.path.join(folder,"buffer_actions.csv"))
        next_states_np     = np.loadtxt(os.path.join(folder,"buffer_next_states.csv"))
        next_actions_np    = np.loadtxt(os.path.join(folder,"buffer_next_actions.csv"))
        rewards_np         = np.loadtxt(os.path.join(folder,"buffer_rewards.csv"))
        dones_np           = np.loadtxt(os.path.join(folder,"buffer_dones.csv"))
        successes_np       = np.loadtxt(os.path.join(folder,"buffer_successes.csv"))
        expert_targets_np  = np.loadtxt(os.path.join(folder,"buffer_expert_targets.csv"))
        sim_states_np      = np.loadtxt(os.path.join(folder,"buffer_sim_states.csv"))
        sim_next_states_np = np.loadtxt(os.path.join(folder,"buffer_sim_next_states.csv"))


        self._states = torch.from_numpy(states_np).float().clone(); self._actions = torch.from_numpy(actions_np).float().clone();
        self._next_states = torch.from_numpy(next_states_np).float().clone(); self._next_actions = torch.from_numpy(next_actions_np).float().clone();
        self._rewards = torch.from_numpy(rewards_np).float().clone(); self._dones = torch.from_numpy(dones_np).byte().clone(); self._successes = torch.from_numpy(successes_np).byte().clone();
        self._expert_targets = torch.from_numpy(expert_targets_np).float().clone(); self._sim_states = torch.from_numpy(sim_states_np).float().clone();
        self._sim_next_states = torch.from_numpy(sim_next_states_np).float().clone(); self._n_elements = self._states.shape[0];

    def load_partial(self, folder):
        print(folder)
        states_np          = np.loadtxt(os.path.join(folder,"buffer_states.csv"))
        actions_np         = np.loadtxt(os.path.join(folder,"buffer_actions.csv"))
        next_states_np     = np.loadtxt(os.path.join(folder,"buffer_next_states.csv"))
        next_actions_np    = np.loadtxt(os.path.join(folder,"buffer_next_actions.csv"))
        rewards_np         = np.loadtxt(os.path.join(folder,"buffer_rewards.csv"))
        dones_np           = np.loadtxt(os.path.join(folder,"buffer_dones.csv"))
        successes_np       = np.loadtxt(os.path.join(folder,"buffer_successes.csv"))
        expert_targets_np  = np.loadtxt(os.path.join(folder,"buffer_expert_targets.csv"))
        sim_states_np      = np.loadtxt(os.path.join(folder,"buffer_sim_states.csv"))
        sim_next_states_np = np.loadtxt(os.path.join(folder,"buffer_sim_next_states.csv"))

        self._n_elements = states_np.shape[0]
        self._states[0:self._n_elements] = torch.from_numpy(states_np).float().clone()
        self._actions[0:self._n_elements] = torch.from_numpy(actions_np).float().clone();
        self._next_states[0:self._n_elements] = torch.from_numpy(next_states_np).float().clone()
        self._next_actions[0:self._n_elements] = torch.from_numpy(next_actions_np).float().clone()
        self._rewards[0:self._n_elements] = torch.from_numpy(rewards_np).float().clone().unsqueeze(1)
        self._dones[0:self._n_elements] = torch.from_numpy(dones_np).byte().clone().unsqueeze(1)
        self._successes[0:self._n_elements] = torch.from_numpy(successes_np).byte().clone().unsqueeze(1)
        self._expert_targets[0:self._n_elements] = torch.from_numpy(expert_targets_np).float().clone().unsqueeze(1)
        self._sim_states[0:self._n_elements] = torch.from_numpy(sim_states_np).float().clone()
        self._sim_next_states[0:self._n_elements] = torch.from_numpy(sim_next_states_np).float().clone()

    #######################
    ###### Rendering ######
    #######################

    def render(self, env, n_times=10):
        """
        Renders the frames on an environment using sim states
        Args:
            env - environment to render frames on
            n_times - number of times to render
        """
        print('Rendering {0} times'.format(n_times))
        timeit.start('render')
        sim_states_np = self._sim_states.numpy()
        actions_np = self._actions.numpy()
        env.reset()
        # env.render()
        for ep in range(n_times):
            for i in range(len(self)):
                env.unwrapped.set_state(sim_states_np[i].reshape(1, self._d_sim_state))
                env.step(actions_np[i].reshape(1, self._d_action))
                env.render()
                # time.sleep(0.1)
        timeit.stop('render')

