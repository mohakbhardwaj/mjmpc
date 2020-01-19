
"""
Buffer for storing transitions and 
rendering
"""
#!/usr/bin/env python
import copy
import json
import numpy as np
import os


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

        if self._n_elements > self._max_length:
            logger.warn("buffer full, rewriting over old samples")


    def __len__(self):
        return min(self._n_elements, self._max_length)

    @property
    def max_length(self):
        return self._max_length


    ######################
    ### Saving/Loading ###
    ######################

    def save(self, folder):
        obs_save          = self._obs[self._num_saved_already:self._n_elements]
        actions_save      = self._actions[self._num_saved_already:self._n_elements]
        next_obs_save     = self._next_obs[self._num_saved_already:self._n_elements]
        next_actions_save = self._next_actions[self._num_saved_already:self._n_elements]
        rewards_save      = self._rewards[self._num_saved_already:self._n_elements]
        dones_save        = self._dones[self._num_saved_already:self._n_elements]
        success_save      = self._successes[self._num_saved_already:self._n_elements]
        states_save      = self._states[self._num_saved_already:self._n_elements]
        next_states_save = self._next_states[self._num_saved_already:self._n_elements]


        if not os.path.exists(folder): os.makedirs(folder)
        np.savetxt(folder+"/observations.csv",obs_save)
        np.savetxt(folder+"/actions.csv", actions_save)
        np.savetxt(folder+"/next_observations.csv", next_obs_save)
        np.savetxt(folder+"/next_actions.csv", next_actions_save)
        np.savetxt(folder+"/rewards.csv",rewards_save)
        np.savetxt(folder+"/dones.csv", dones_save)
        np.savetxt(folder+"/successes.csv", success_save)
        # with open(folder+"/states.json","w") as fout:
            # json.dump(states_save, fout)
        # with open(folder+"/next_states.json","w") as fout:
            # json.dump(next_states_save, fout)
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
        # timeit.start('render')
        env.reset()
        # env.render()
        for ep in range(n_times):
            env.reset()
            for i in range(len(self)):
                env.unwrapped.set_env_state(self._states[i])
                env.step(self._actions[i].reshape(self._d_action,))
                env.render()
        # timeit.stop('render')

