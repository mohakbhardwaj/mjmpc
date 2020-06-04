"""
Declares a class that takes a gym environment
as input and implements necessary functions for MPC rollouts

Author: Mohak Bhardwaj
Date: January 9, 2020
"""
from gym import spaces
import numpy as np
from copy import deepcopy
import time

class GymEnvWrapper():
    def __init__(self, env):
        self.env = env
        self.env.reset()
        observation, _reward, done, _info = self.env.step(np.zeros(self.env.action_space.low.shape))
        assert not done, ""
        if type(observation) is tuple:
            self.d_obs = np.sum([o.size for o in observation])
        elif type(observation) is dict:
            self.d_obs = 0.
            for k in observation.keys():
                self.d_obs += observation[k].size
        else: self.d_obs = observation.size
        # self.d_obs = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        super(GymEnvWrapper, self).__init__()

    def step(self, u):
        # cdef rew, done
        # cdef double[:] obs_view
        obs_view, rew, done, info = self.env.step(u)
        return obs_view, rew, done, info
        # return self.env.step(u)
 
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
            return self.env.get_obs()
        except:
            return self.env.env.get_obs()
 
    def get_reward(self, state, action, next_state):
        '''
        Return the reward function for the transition
        '''
        pass
    
    def rollout(self, u_vec: np.ndarray):
        """
        Given batch of action sequences, we perform rollouts 
        and return resulting observations, rewards etc.
        :param u_vec: np.ndarray of shape [batch_size, n_steps, d_action]
        :return:
            obs_vec: np.ndarray [batch_size, n_steps, d_obs]
            state_vec: np.ndarray [batch_size, n_steps, d_state]
            rew_vec: np.ndarray [batch_size, n_steps, 1]
            done_vec: np.ndarray [batch_size, n_steps, 1]
            info: dict
        """
        # start_t = time.time()
        batch_size, n_steps, d_action = u_vec.shape
        if type(self.observation_space) is spaces.Dict:
            obs_vec = []
        else: obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
        state_vec = [] #np.zeros((self.batch_size, n_steps, self.d_state))
        rew_vec = np.zeros((batch_size, n_steps))
        done_vec = np.zeros((batch_size, n_steps))
        curr_state = deepcopy(self.get_env_state())

        for b in range(batch_size):
            #Set the state to the current state
            self.set_env_state(curr_state)
            #Rollout for t steps and store results
            for t in range(n_steps):
                u_curr = u_vec[b, t, :]
                obs, rew, done, _ = self.step(u_curr)
                if type(self.observation_space) is spaces.Dict:
                    obs_vec.append(obs.copy())
                else:
                    obs_vec[b, t, :] = obs.copy().reshape(self.d_obs,)
                # state = self.get_env_state()
                # state_vec.append(state.copy())
                rew_vec[b, t] = rew
                done_vec[b, t] = done
        
        # print('Normal rollout time = {0}'.format(time.time()-start_t))

        # self.rollout_test(u_vec)
        return obs_vec, rew_vec, done_vec, {} #state_vec
    
    def rollout_test(self, u_vec: np.ndarray):
        start_t = time.time()
        batch_size, n_steps, d_action = u_vec.shape
        if type(self.observation_space) is spaces.Dict:
            obs_vec = []
        else: obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
        # state_vec = [] #np.zeros((self.batch_size, n_steps, self.d_state))
        # rew_vec = np.zeros((batch_size, n_steps))
        # done_vec = np.zeros((batch_size, n_steps))
        curr_state = deepcopy(self.get_env_state())

        for b in range(batch_size):
            #Set the state to the current state
            self.set_env_state(curr_state)
            #Rollout for t steps and store results
            for t in range(n_steps):
                u_curr = u_vec[b, t, :]
            
                for i in range(self.env.env.model.nu):
                    self.env.env.sim.data.ctrl[i] = u_curr[i]
                for _ in range(self.env.env.frame_skip):
                    self.env.env.sim.step()
                # obs, rew, done, _ = self.step(u_curr)
                # if type(self.observation_space) is spaces.Dict:
                #     obs_vec.append(obs.copy())
                # else:
                #     obs_vec[b, t, :] = obs.copy().reshape(self.d_obs,)
                # state = self.get_env_state()
                # state_vec.append(state.copy())
                # rew_vec[b, t] = rew
                # done_vec[b, t] = done
        print('Test rollout time = {0}'.format(time.time()-start_t))

        return None, None, None, None #obs_vec, rew_vec, done_vec, {} #state_vec 
    

    # cpdef rollout(self, double[:,:,:] u_vec):
    #     cdef size_t batch_size, n_steps, d_action
    #     batch_size = u_vec.shape[0]
    #     n_steps = u_vec.shape[1]
    #     d_action = u_vec.shape[2]
    #     obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
    #     # state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
    #     rew_vec = np.zeros((batch_size, n_steps))
    #     done_vec = np.zeros((batch_size, n_steps))

    #     self.rollout_cy(u_vec, obs_vec, rew_vec, done_vec)
    #     return obs_vec, rew_vec, done_vec, {}


    # cdef rollout_cy(self, double[:,:,:] u_vec,
    #                       double[:,:,:] obs_vec,
    #                       double[:,:] rew_vec,
    #                       double[:,:] done_vec):
    #     """
    #     Given batch of action sequences, we perform rollouts 
    #     and return resulting observations, rewards etc.
    #     :param u_vec: np.ndarray of shape [batch_size, n_steps, d_action]
    #     :return:
    #         obs_vec: np.ndarray [batch_size, n_steps, d_obs]
    #         state_vec: np.ndarray [batch_size, n_steps, d_state]
    #         rew_vec: np.ndarray [batch_size, n_steps, 1]
    #         done_vec: np.ndarray [batch_size, n_steps, 1]
    #         info: dict
    #     """
    #     cdef size_t batch_size, n_steps, d_action
    #     # batch_size, n_steps, d_action = u_vec.shape
    #     batch_size = u_vec.shape[0]
    #     n_steps = u_vec.shape[1]
    #     d_action = u_vec.shape[2]
    #     # obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
    #     # # state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
    #     # rew_vec = np.zeros((batch_size, n_steps))
    #     # done_vec = np.zeros((batch_size, n_steps))
    #     curr_state = deepcopy(self.get_env_state())
    #     cdef double rew, done
    #     cdef double[:] obs_view
    #     for b in range(batch_size):
    #         #Set the state to the current state
    #         self.set_env_state(curr_state)
    #         #Rollout for t steps and store results
    #         for t in range(n_steps):
    #             # u_curr = u_vec[b, t, :]
    #             # obs, rew, done, _ = self.step(u_curr)
                
    #             obs_view, rew, done, _ = self.step(u_vec[b,t,:].copy())
    #             obs_vec[b, t, :] = obs_view.copy() #obs.copy().reshape(self.d_obs,)
    #             # state_vec[:, t, :] = self.state.copy()
    #             rew_vec[b, t] = rew
    #             done_vec[b, t] = done
 
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def reset(self, seed=None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except AttributeError:
                self.env._seed(seed)
        return self.env.reset()

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
    


# def make_wrapper(env):
#     # cdef GymEnvWrapper env_wrap = GymEnvWrapper(env)
#     env_wrap = GymEnvWrapper(env)
#     return env_wrap

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