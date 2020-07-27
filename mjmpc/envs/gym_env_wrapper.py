"""
Declares a class that takes a gym environment
as input and implements necessary functions for MPC rollouts

Author: Mohak Bhardwaj
Date: January 9, 2020
"""
from collections import defaultdict
from copy import deepcopy
from gym import spaces
import numpy as np
import time
import torch

class GymEnvWrapper():
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.metadata = self.env.metadata
        observation, _reward, done, _info = self.env.step(np.zeros(self.env.action_space.low.shape))
        assert not done, ""
        if type(observation) is tuple:
            self.d_obs = np.sum([o.size for o in observation])
        elif type(observation) is dict:
            self.d_obs = 0.
            for k in observation.keys():
                self.d_obs += observation[k].size
        else: self.d_obs = observation.size
        state = self.get_env_state()
        if type(state) is tuple:
            self.d_state = np.sum([o.size for o in state])
        elif type(state) is dict:
            self.d_state = 0
            for k in state.keys():
                self.d_state += state[k].size
        else: self.d_state = state.size
        self.d_action = self.env.action_space.low.shape[0]

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps

        self.default_dyn_params = defaultdict(dict)
        self.randomized_dyn_params = defaultdict(dict)
        
        self.seed()
        super(GymEnvWrapper, self).__init__()

    def step(self, u):
        obs, rew, done, info = self.env.step(u)
        return deepcopy(obs), deepcopy(rew), deepcopy(done), deepcopy(info)
 
    def set_env_state(self, state_dict: dict):
        """
        Set the state of environment from dictionary
        """
        try:
            self.env.set_env_state(state_dict)
        except:
            self.env.env.set_env_state(state_dict)
 
    def get_env_state(self) -> dict:
        """
        Return dictionary of the full environment state
        """
        try:
            return self.env.get_env_state()
        except:
            return self.env.env.get_env_state()

    def get_obs(self) -> dict:
        """
        Return dictionary of the full environment state
        """
        try:
            return self.env.get_obs().copy()
        except:
            return self.env.env.get_obs().copy()
 
    def get_reward(self, state, action, next_state):
        '''
        Return the reward function for the transition
        '''
        pass
    
    def rollout(self, u_vec: np.ndarray):
        """
        Given batch of action sequences, we perform rollouts 
        and return resulting observations, rewards etc.
        :param u_vec: np.ndarray of shape [batch_size, horizon, d_action]
        :return:
            obs_vec: np.ndarray [batch_size, horizon, d_obs]
            state_vec: np.ndarray [batch_size, horizon, d_state]
            rew_vec: np.ndarray [batch_size, horizon, 1]
            done_vec: np.ndarray [batch_size, horizon, 1]
            info: dict
        """
        start_t = time.time()
        batch_size, horizon, d_action = u_vec.shape
        if type(self.observation_space) is spaces.Dict:
            obs_vec = []; next_obs_vec = []
        else: 
            obs_vec = np.zeros((batch_size, horizon, self.d_obs))
            next_obs_vec = np.zeros((batch_size, horizon, self.d_obs))
        state_vec = [] #np.zeros((self.batch_size, horizon, self.d_state))
        rew_vec = np.zeros((batch_size, horizon))
        done_vec = np.zeros((batch_size, horizon))
        curr_state = deepcopy(self.get_env_state())

        for b in range(batch_size):
            #Set the state to the current state
            self.set_env_state(curr_state)
            curr_obs = deepcopy(self.get_obs())

            #Rollout for t steps and store results
            for t in range(horizon):
                u_curr = u_vec[b, t, :]
                next_obs, rew, done, _ = self.step(u_curr)
                if type(self.observation_space) is spaces.Dict:
                    obs_vec.append(curr_obs.copy())
                    next_obs_vec.append(next_obs.copy())
                else:
                    obs_vec[b, t, :] = curr_obs.copy().reshape(self.d_obs,)
                    next_obs_vec[b, t, :] = next_obs.copy().reshape(self.d_obs,)
                # state = self.get_env_state()
                # state_vec.append(state.copy())
                rew_vec[b, t] = rew
                done_vec[b, t] = done
                curr_obs = next_obs.copy()

        info = {'total_time' : time.time() - start_t}
        return obs_vec, rew_vec, done_vec, info, next_obs_vec
    
    def rollout_cl(self, policy, batch_size, horizon, mode='mean', noise=None):
        """
        Given a policy object, we perform rollouts 
        and return resulting trajectroy.
        Parameters
        ----------
        policy: 
        batch_size: number of rollouts
        horizon: rollout horizon

        Returns
        -------
        obs_vec: np.ndarray [batch_size, horizon, d_obs]
        state_vec: np.ndarray [batch_size, horizon, d_state]
        rew_vec: np.ndarray [batch_size, horizon, 1]
        done_vec: np.ndarray [batch_size, horizon, 1]
        info: dict
        """
        start_t = time.time()
        total_inf_time = 0.0
        # if (noise is not None) and mode =='sample':
            # raise ValueError('Added noise must be None when using sample mode from policy')
        if type(self.observation_space) is spaces.Dict:
            obs_vec = []; next_obs_vec = []
        else: 
            obs_vec = np.zeros((batch_size, horizon, self.d_obs))
            next_obs_vec = np.zeros((batch_size, horizon, self.d_obs))
        act_vec = np.zeros((batch_size, horizon, self.d_action))
        # log_prob_vec = np.zeros((batch_size, horizon))
        act_infos = []
        rew_vec = np.zeros((batch_size, horizon))
        done_vec = np.zeros((batch_size, horizon))
        curr_state = deepcopy(self.get_env_state())
        with torch.no_grad():
            for b in range(batch_size):
                #Set the state to the current state
                self.set_env_state(curr_state)
                curr_obs = self.get_obs()
                #Rollout for t steps and store results
                for t in range(horizon):
                    #Get action prediction from model
                    before_inf = time.time()
                    obs_torch = torch.FloatTensor(curr_obs).unsqueeze(0)
                    noise_sample = noise[b,t] if noise is not None else None
                    u_curr, act_info = policy.get_action(obs_torch, mode, noise_sample)
                    u_curr = u_curr.numpy().copy().reshape(self.d_action,) 
                    # log_prob = log_prob.numpy().item()
                    inf_time = time.time() - before_inf
                    total_inf_time += inf_time
                    #Add noise if provided
                    # if noise is not None:
                        # u_curr = u_curr + noise[b,t]
                    #step environment
                    next_obs, rew, done, _ = self.step(u_curr)
                    
                    #collect data
                    if type(self.observation_space) is spaces.Dict:
                        obs_vec.append(curr_obs.copy())
                        next_obs_vec.append(next_obs.copy())
                    else:
                        obs_vec[b, t, :] = curr_obs.copy().reshape(self.d_obs,)
                        next_obs_vec[b, t, :] = next_obs.copy().reshape(self.d_obs,)
                    act_vec[b, t, :] = u_curr.copy()
                    # log_prob_vec[b, t] = log_prob
                    act_infos.append(act_info)
                    rew_vec[b, t] = rew
                    done_vec[b, t] = done
                    curr_obs = next_obs.copy()

        info = {'total_time' : time.time() - start_t, 'inference_time': total_inf_time}
        return obs_vec, act_vec, act_infos, rew_vec, done_vec, next_obs_vec, info
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def reset(self, seed=None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except AttributeError:
                self.env._seed(seed)
        return self.env.reset()

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            try:
                self.env.real_step = bool_val
            except:
                raise NotImplementedError
    
    def evaluate_success(self, trajs):
        try:
            succ_metric = self.env.evaluate_success(trajs)
        except:
            try:
                succ_metric = self.env.env.evaluate_success(trajs)
            except:
                raise NotImplementedError('Evaluate success not implemented')
        return succ_metric

    def close(self):
        try:
            self.env.close()
        except:
            print('No close')
            pass
    
    ########################
    # Domain Randomization #
    ########################
    def randomize_dynamics(self, param_dict={}):
        """
            Randomizes dynamics parameters based on provided
            values
        """
        for param_id in param_dict.keys(): 
            #param_id - type of dynamics parameter eg. body_mass, frictionloss etc.
            for name, dist_params in param_dict[param_id].items():
                #name = name of body or joint etc. in mujoco xml
                #dist_params = [noise_scale, bias_scale] for randomization distribution
                noise_scale, bias_scale = dist_params
                if param_id == "body_mass":
                    idx = self.env.env.sim.model.body_name2id(name)
                    field = self.env.env.sim.model.body_mass
                elif param_id == "body_inertia":
                    idx = self.env.env.sim.model.body_name2id(name)
                    field = self.env.env.sim.model.body_inertia
                elif param_id == "dof_damping":
                    idx = self.env.env.sim.model.joint_name2id(name)
                    field = self.env.env.sim.model.dof_damping
                elif param_id == "dof_frictionloss":
                    idx = self.env.env.sim.model.joint_name2id(name)
                    field = self.env.env.sim.model.dof_frictionloss
                elif param_id == "geom_size":
                    idx = self.env.env.sim.model.geom_name_2id(name)
                    field = self.env.env.sim.model.geom_size           
                elif param_id == "geom_friction":
                    idx = self.env.env.sim.model.geom_name_2id(name)
                    field = self.env.env.sim.model.geom_friction
                else:
                    raise ValueError("Unknown dynamics field")
                
                if name not in self.default_dyn_params[param_id].keys():
                    curr_val = deepcopy(field[idx])
                    self.default_dyn_params[param_id][name] = curr_val #update default params only first time
                else:
                    curr_val = deepcopy(self.default_dyn_params[param_id][name])

                biased_mean = (1.0 + bias_scale) * curr_val
                rand_val = self.env.np_random.uniform(biased_mean - biased_mean * noise_scale, 
                                                      biased_mean + biased_mean * noise_scale)
                # rand_val = biased_mean + noise_sample
                #update the relevant field and idx
                field[idx] = rand_val
                self.randomized_dyn_params[param_id][name] = rand_val

        return self.default_dyn_params, self.randomized_dyn_params


    def render(self):
        try:
            self.env.env.mj_render()
        except:
            try:
                self.env.render()
            except:
                print('Rendering not available')
                pass 

    def get_curr_frame(self, frame_size=(640,480), 
                       camera_name=None, device_id=0):
        try:
            curr_frame = self.env.env.sim.render(width=frame_size[0], height=frame_size[1], 
                                                mode="offscreen", camera_name=camera_name, device_id=device_id)
            return curr_frame[::-1,:,:]
        except:
            try:
                curr_frame = self.env.env.sim.render(width=frame_size[0], height=frame_size[1], 
                                                    mode="offscreen", camera_name=camera_name, device_id=device_id)
            except:
                raise NotImplementedError('Getting frame not implemented')    




if __name__ == "__main__":
    import mj_envs
    import gym
    import mjmpc.envs
    from mjmpc.policies import MPCPolicy
    import yaml
    from copy import deepcopy


    env = gym.make('pen-v0')
    env = GymEnvWrapper(env)

    rollout_env = deepcopy(env)

    def rollout_fn(u_vec: np.ndarray):
        """
        Given a batch of sequences of actions, rollout 
        in sim envs and return sequence of costs
        """
        obs_vec, rew_vec, done_vec, _ = rollout_env.rollout(u_vec.copy())
        return -1.0*rew_vec #we assume environment returns rewards, but controller needs consts
    
    #Create functions for controller
    def set_sim_state_fn(state_dict: dict):
        """
        Set state of simulation environments for rollouts
        """
        rollout_env.set_env_state(state_dict)   

    with open("../../examples/configs/hand/pen-v0.yml") as file:
        exp_params = yaml.load(file, Loader=yaml.FullLoader)
    policy_params = {}
    policy_params = exp_params["mppi"]
    policy_params['base_action'] = exp_params['base_action']
    policy_params['num_actions'] = env.action_space.low.shape[0]
    policy_params['action_lows'] = env.action_space.low
    policy_params['action_highs'] = env.action_space.high
    policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']

    del policy_params['particles_per_cpu'], policy_params['num_cpu']
    print(policy_params)
    policy = MPCPolicy(controller_type="mppi",
                        param_dict=policy_params, batch_size=1) #Only batch_size=1 is supported for now
    policy.controller.set_sim_state_fn = set_sim_state_fn
    policy.controller.rollout_fn = rollout_fn

    action, _  = policy.get_action(env.get_env_state())
    print(action)