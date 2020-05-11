"""
Declares a class that takes a gym environment
as input and implements necessary functions for MPC rollouts

Author: Mohak Bhardwaj
Date: January 9, 2020
"""
import numpy as np
from copy import deepcopy

cdef class GymEnvWrapperCy():
    cdef public int d_obs
    cdef public object env, observation_space, action_space 

    def __init__(self, env):
        # self.env = env
        # self.env.reset()
        # observation, _reward, done, _info = self.env.step(np.zeros(self.env.action_space.low.shape))
        # assert not done, ""
        # self.d_obs = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
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


    cdef step(self, double[:] u):
        cdef rew, done
        # cdef double[:] obs_view
        obs_view, rew, done, info = self.env.step(u)
        return obs_view, rew, done, info
 
    def set_env_state(self, state_dict):
        """
        Set the state of environment from dictionary
        """
        try:
            self.env.set_env_state(state_dict)
        except:
            self.env.env.set_env_state(state_dict)
 
    def get_env_state(self):
        """
        Return dictionary of the full environment state
        """
        try:
            return self.env.get_env_state()
        except:
            return self.env.env.get_env_state()
 
    def get_reward(self, state, action, next_state):
        '''
        Return the reward function for the transition
        '''
        pass
    
    
    cpdef rollout(self, double[:,:,:] u_vec):
        cdef size_t batch_size, n_steps, d_action
        batch_size = u_vec.shape[0]
        n_steps = u_vec.shape[1]
        d_action = u_vec.shape[2]
        obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
        # state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
        rew_vec = np.zeros((batch_size, n_steps))
        done_vec = np.zeros((batch_size, n_steps))

        self.rollout_cy(u_vec, rew_vec, done_vec)
        return obs_vec, rew_vec, done_vec, {}


    cdef rollout_cy(self, double[:,:,:] u_vec,
                          double[:,:] rew_vec,
                          double[:,:] done_vec):
                                                    # double[:,:,:] obs_vec,
#
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
        cdef size_t batch_size, n_steps, d_action
        batch_size = u_vec.shape[0]
        n_steps = u_vec.shape[1]
        d_action = u_vec.shape[2]
        # obs_vec = np.zeros((batch_size, n_steps, self.d_obs))
        # # state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
        # rew_vec = np.zeros((batch_size, n_steps))
        # done_vec = np.zeros((batch_size, n_steps))
        cdef double[:] curr_state_view 
        curr_state_view = self.get_env_state().copy() #deepcopy(self.get_env_state())
        cdef double rew, done
        # cdef double[:] obs_view
        for b in range(batch_size):
            #Set the state to the current state
            self.set_env_state(curr_state_view)
            #Rollout for t steps and store results
            for t in range(n_steps):
                # u_curr = u_vec[b, t, :]
                # obs, rew, done, _ = self.step(u_curr)
                obs_view, rew, done, _ = self.step(u_vec[b,t,:].copy())
                if type(self.observation_space) is spaces.Dict:
                    obs_vec.append(obs_view.copy())
                else:
                    obs_vec[b, t, :] = obs_view.copy() #obs.copy().reshape(self.d_obs,)
                # state_vec[:, t, :] = self.state.copy()
                rew_vec[b, t] = rew
                done_vec[b, t] = done
        return obs_vec, rew_vec, done_vec, {}
    
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
            raise NotImplementedError('Getting frame not implemented')


    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
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
    


def make_wrapper(env):
    cdef GymEnvWrapperCy env_wrap = GymEnvWrapperCy(env)
    # env_wrap = GymEnvWrapper(env)
    return env_wrap