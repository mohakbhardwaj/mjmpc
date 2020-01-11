
"""
Buffer for storing transitions and 
rendering
"""
#!/usr/bin/env python
import copy
import numpy as np
import os
from .logger import logger
from .timer import timeit


class Buffer():
    def __init__(self, d_obs, d_action, max_length=1e6):
        self._max_length = int(max_length)
        self._d_obs = d_obs
        self._d_action = d_action

        self._obs = np.zeros((self._max_length, int(d_obs)), dtype=np.float32)
        self._actions = np.zeros((self._max_length, int(d_action)), dtype=np.float32)
        self._next_obs = np.zeros((self._max_length, int(d_obs)), dtype=np.float32)
        self._next_actions = np.zeros((self._max_length, int(d_action)), dtype=np.float32)
        self._states = []
        self._next_states = []
        self._rewards = np.zeros((self._max_length,1), dtype=np.float32)
        self._dones = np.zeros((self._max_length,1), dtype=bool)
        self._successes = np.zeros((self._max_length,1), dtype=bool)

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

    def add(self, transition):
        obs, action, next_obs, next_action, reward, done, success, state, next_state = transition
        idx = self._n_elements % self._max_length
        self._obs[idx] = obs.copy().reshape(self._d_obs,)
        self._actions[idx] = action.copy().reshape(self._d_action,)
        self._next_obs[idx] = next_obs.copy().reshape(self._d_obs,)
        self._next_actions[idx] = next_action.reshape(self._d_action,)
        self._rewards[idx] = reward
        self._dones[idx] = done
        self._successes[idx] = success
        self._states.append(copy.deepcopy(state))
        self._next_states.append(copy.deepcopy(next_state))

        self._n_elements += 1
        self._curr_ep_rew += reward

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

    @property
    def max_length(self):
        return self._max_length

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
        obs_save          = self._obs[self._num_saved_already:self._n_elements]
        actions_save         = self._actions[self._num_saved_already:self._n_elements]
        next_obs_save     = self._next_obs[self._num_saved_already:self._n_elements]
        next_actions_save    = self._next_actions[self._num_saved_already:self._n_elements]
        rewards_save         = self._rewards[self._num_saved_already:self._n_elements]
        dones_save           = self._dones[self._num_saved_already:self._n_elements]
        success_save         = self._successes[self._num_saved_already:self._n_elements]
        # sim_states_save      = self._sim_states[self._num_saved_already:self._n_elements]
        # sim_next_states_save = self._sim_next_states[self._num_saved_already:self._n_elements]


        if not os.path.exists(folder): os.makedirs(folder)
        np.savetxt(folder+"/observations.csv",obs_save)
        np.savetxt(folder+"/actions.csv", actions_save)
        np.savetxt(folder+"/next_observations.csv", next_obs_save)
        np.savetxt(folder+"/next_actions.csv", next_actions_save)
        np.savetxt(folder+"/rewards.csv",rewards_save)
        np.savetxt(folder+"/dones.csv", dones_save)
        np.savetxt(folder+"/successes.csv", success_save)
        # np.savetxt(folder+"/sim_states.csv", sim_states_save)
        # np.savetxt(folder+"/sim_next_states.csv", sim_next_states_save)

        if len(self) < self._max_length:
            self._num_saved_already = self._n_elements

    def load(self, folder):
        self._obs          = np.loadtxt(os.path.join(folder,"observations.csv"))
        self._actions      = np.loadtxt(os.path.join(folder,"actions.csv"))
        self._next_obs     = np.loadtxt(os.path.join(folder,"next_observations.csv"))
        self._next_actions = np.loadtxt(os.path.join(folder,"next_actions.csv"))
        self._rewards      = np.loadtxt(os.path.join(folder,"rewards.csv"))
        self._dones        = np.loadtxt(os.path.join(folder,"dones.csv"))
        self._successes    = np.loadtxt(os.path.join(folder,"successes.csv"))
        # sim_states_np      = np.loadtxt(os.path.join(folder,"sim_states.csv"))
        # sim_next_states_np = np.loadtxt(os.path.join(folder,"sim_next_states.csv"))
        self._n_elements = self._obs.shape[0];

    def load_partial(self, folder):
        self._obs[0:self._n_elements]= np.loadtxt(os.path.join(folder,"observations.csv"))
        self._actions[0:self._n_elements]= np.loadtxt(os.path.join(folder,"actions.csv"))
        self._next_obs[0:self._n_elements]= np.loadtxt(os.path.join(folder,"next_observations.csv"))
        self._next_actions[0:self._n_elements]= np.loadtxt(os.path.join(folder,"next_actions.csv"))
        self._rewards[0:self._n_elements]= np.loadtxt(os.path.join(folder,"rewards.csv"))
        self._dones[0:self._n_elements]= np.loadtxt(os.path.join(folder,"dones.csv"))
        self._successes[0:self._n_elements]= np.loadtxt(os.path.join(folder,"successes.csv"))
        self._n_elements = states_np.shape[0]

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
        env.reset()
        # env.render()
        for ep in range(n_times):
            for i in range(len(self)):
                env.unwrapped.set_env_state(self._states[i])
                env.step(self._actions[i].reshape(self._d_action,))
                env.render()
                # time.sleep(0.1)
        timeit.stop('render')

