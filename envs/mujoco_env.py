#!/usr/bin/env python
"""

General class for mujoco environments based on mujoco_py

Built upon mujoco_env provided by https://github.com/aravindr93/trajopt


"""

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import time as timer

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from .environment import Env

DEFAULT_SIZE = 500

class MujocoEnv(Env):
    """
        Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, env_name, d_state, d_action, d_obs, batch_size=1):
        """
            params - 
                model_path: Path for mujoco xml file for environment model
                frame_skip: Number of times to apply same action to underlying simulation
                batch_size: Number of simulations 
        """

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        # self.sim = MjSim(self.model)
        # self.data = self.sim.data
        # if batch_size > 1:
        self.sim = []
        self.data = []
        # self.sim = [MjSim(self.model)] * batch_size
        for i in range(batch_size):
            sim = MjSim(self.model)
            data = sim.data
            self.sim.append(sim)
            self.data.append(data)
        
        # self.data = [sim.data for sim in self.sim]
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data[0].qpos.ravel().copy()
        self.init_qvel = self.data[0].qvel.ravel().copy()
        # observation, _reward, done, _info = self.step(np.zeros((self.batch_size, self.model.nu)))
        # assert not done
        # self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(d_obs)  # self.obs_dim
        low = -high
        self.observation_space = spaces.Box(low, high)

        # self.seed()
        self.viewer = None
        super(MujocoEnv, self).__init__(batch_size, env_name, d_state, d_action, d_obs)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    # def get_env_state(self):
    #     """
    #     Get full state of the environment beyond qpos and qvel
    #     For example, if targets are defined using sites, this function should also
    #     contain location of the sites (which are not included in qpos).
    #     Must return a dictionary that can be used in the set_env_state function
    #     """
    #     raise NotImplementedError

    # def set_env_state(self, state):
    #     """
    #     Uses the state dictionary to set the state of the world
    #     """
    #     raise NotImplementedError

    # -----------------------------


    def reset(self):
        for i in range(self.batch_size):
            self.sim[i].reset()
            self.sim[i].forward()
        ob = self.reset_model()
        return ob

    def get_state(self) -> np.ndarray:
        """
        Return np array of entire state relevant for 
        planning
        """
        raise NotImplementedError

    def get_reward(self, state, action, next_state) -> np.array:
        '''
        return the reward function for the transition
        '''
        # raise NotImplementedError
        pass

    def set_sim_state(self, qpos, qvel):
        """
        Set the qpos and qvel of the mujoco simulation
        """
        assert qpos.shape == (self.batch_size, self.model.nq) and qvel.shape == (self.batch_size, self.model.nv)
        for i in range(self.batch_size):
            state = self.sim[i].get_state()
            for j in range(self.model.nq):
                state.qpos[j] = qpos[i, j]
            for j in range(self.model.nv):
                state.qvel[j] = qvel[i, j]
            self.sim[i].set_state(state)
            self.sim[i].forward()
    

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def state(self):
        return self.get_state()

    def do_simulation(self, ctrl):
        for i in range(self.batch_size):
            for j in range(self.model.nu):
                self.sim[i].data.ctrl[j] = ctrl[i, j]
            for _ in range(self.frame_skip):
                self.sim[i].step()
                # if self.mujoco_render_frames is True:
                #     self.mj_render()
    
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        # self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim[0])
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim[0], device_id=-1)
            self._viewer_setup()
            # self._viewers[mode] = self.viewer
        return self.viewer

    def _viewer_setup(self):
        """
        Environment specific viewer setup (Implement in every subclass)
        """
        raise(NotImplementedError, "Implement viewer setup")
    
    def _render_callback(self):
        """
        Optional callback called before rendering
        """
        pass

    def rollout(self, u_vec):
        """
        :param u_vec: np.ndarray of shape [batch_size, n_steps, d_action]
        :return:
            obs_vec: np.ndarray [batch_size, n_steps, d_obs]
            state_vec: np.ndarray [batch_size, n_steps, d_state]
            rew_vec: np.ndarray [batch_size, n_steps, 1]
            done_vec: np.ndarray [batch_size, n_steps, 1]
            info: dict
        """
        assert u_vec.shape[0] == self.batch_size, "Number of sequences must be equal to batch size"
        n_steps = u_vec.shape[1]
        obs_vec = np.zeros((self.batch_size, n_steps, self.d_obs))
        state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
        rew_vec = np.zeros((self.batch_size, n_steps))
        done_vec = np.zeros((self.batch_size, n_steps))
        # for i in range(self.batch_size):
        for t in range(n_steps):
            u_curr = u_vec[:, t, :]
            obs, rew, done, _ = self.step(u_curr)
            obs_vec[:, t, :] = obs.copy()
            state_vec[:, t, :] = self.state.copy()
            rew_vec[:, t] = rew
            done_vec[:, t] = done

        return obs_vec, state_vec, rew_vec, done_vec, {}


    # def mj_render(self):
    #     try:
    #         self.viewer.render()
    #     except:
    #         self.mj_viewer_setup()
    #         self.viewer._run_speed = 0.5
    #         #self.viewer._run_speed /= self.frame_skip
    #         self.viewer.render()

    # def _get_viewer(self):
    #     return None

    # def state_vector(self):
    #     state = self.sim.get_state()
    #     return np.concatenate([
    #         state.qpos.flat, state.qvel.flat])

    # -----------------------------

    # def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
    #     self.mujoco_render_frames = True
    #     for ep in range(num_episodes):
    #         o = self.reset()
    #         d = False
    #         t = 0
    #         while t < horizon and d is False:
    #             a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
    #             o, r, d, _ = self.step(a)
    #             t = t+1
    #     self.mujoco_render_frames = False

    # def visualize_policy_offscreen(self, policy, horizon=1000,
    #                                num_episodes=1,
    #                                frame_size=(640,480),
    #                                mode='exploration',
    #                                save_loc='/tmp/',
    #                                filename='newvid',
    #                                camera_name=None):
    #     import skvideo.io
    #     for ep in range(num_episodes):
    #         print("Episode %d: rendering offline " % ep, end='', flush=True)
    #         o = self.reset()
    #         d = False
    #         t = 0
    #         arrs = []
    #         t0 = timer.time()
    #         while t < horizon and d is False:
    #             a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
    #             o, r, d, _ = self.step(a)
    #             t = t+1
    #             curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
    #                                          mode='offscreen', camera_name=camera_name, device_id=0)
    #             arrs.append(curr_frame[::-1,:,:])
    #             print(t, end=', ', flush=True)
    #         file_name = save_loc + filename + str(ep) + ".mp4"
    #         skvideo.io.vwrite( file_name, np.asarray(arrs))
    #         print("saved", file_name)
    #         t1 = timer.time()
    #         print("time taken = %f"% (t1-t0))
