"""
TODO: Implement step and rollout functions using Numba
"""
import copy
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from envs import Env
import torch


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class PendulumEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, batch_size=1):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        statehigh = np.array([np.pi, np.inf])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.state_space = spaces.Box(low=-statehigh, high=statehigh, dtype=np.float32)
        self.seed()
        self.g = 10.
        self.last_u = None
        super(PendulumEnv, self).__init__(batch_size, 'simple_pendulum', 2, 1, 3)
        self.dyn_params = {'mass': 1., 'length': 1.}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        """
        :param u: np.ndarray of size [batch_size X 1]
        :return: obs: np.ndarray [batch_size X d_obs], reward: np.ndarray [batch_size]
                 done: np.ndarray [batch_size, info: dict{}
        """
        assert u.shape[0] == self.batch_size, "Number of controls must be equal to batch size"
        th = self.state[:, 0]
        thdot = self.state[:, 1]
        m = self.dyn_params['mass']; l = self.dyn_params['length'];dt = self.dt
        costs = np.zeros((self.batch_size))
        done = np.zeros((self.batch_size), dtype=bool)
        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u.copy()  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2  # + .001*(u**2)
        newthdot = thdot + (-3 * self.g / (2 * l) * np.sin(th + np.pi) + 3. /
                                             (m * l ** 2) * u[:, 0]) * dt
        newth = th + newthdot * dt
        newth = angle_normalize(newth)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        self.state = np.concatenate([newth[:, np.newaxis].copy(), newthdot[:, np.newaxis]].copy(), axis=1)
        return self._get_obs(), -costs, done, {}

    def rollout(self, u_vec: np.ndarray):
        """
        :param u_vec: np.ndarray of shape [batch_size, n_steps, d_action]
        :return:
            obs_vec: np.ndarray [batch_size, n_steps, d_obs]
            rew_vec: np.ndarray [batch_size, n_steps, 1]
            done_vec: np.ndarray [batch_size, n_steps, 1]
            info: dict
        """
        assert u_vec.shape[0] == self.batch_size, "Number of sequences must be equal to batch size"
        _, n_steps, _ = u_vec.shape
        obs_vec = np.zeros((self.batch_size, n_steps, self.d_obs))
        state_vec = np.zeros((self.batch_size, n_steps, self.d_state))
        rew_vec = np.zeros((self.batch_size, n_steps))
        done_vec = np.zeros((self.batch_size, n_steps))
        for t in range(n_steps):
            u_curr = u_vec[:, t]
            obs, rew, done, _ = self.step(u_curr)
            obs_vec[:, t, :] = obs
            state_vec[:, t, :] = self.state
            rew_vec[:, t] = rew
            done_vec[:, t] = done
            
        return  obs_vec, state_vec, rew_vec, done_vec, {}


    def reset(self) -> np.ndarray:
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high, size=(self.batch_size, self.d_state))
        self.last_u = None
        return self._get_obs()

    def get_state(self) -> np.ndarray:
        """Returns the state vector"""
        return copy.deepcopy(self.state).reshape(self.batch_size, self.d_state)

    def set_state(self, state: np.ndarray):
        """Sets the state of the system from vector"""
        assert state.shape[0] == self.batch_size, "States must be of shape: [batch size, d_state]"
        self.state = copy.deepcopy(state)

    def get_reward(self, state, action, next_state):
        '''
        return the reward function for the transition
        '''
        # th = np.arctan2(state[:, 1], state[:, 0])
        # thdot = state[:, 2]
        # costs = th ** 2 + .1 * thdot ** 2  # + .001*(u**2)
        th = state[:, 0]
        thdot = state[:, 1]
        costs = th ** 2 + .1 * thdot ** 2  # + .001*(u**2)
        return -costs

    # def clipped_action(self, u):
    #     u = np.clip(u, -self.max_torque, self.max_torque)
    #     return u

    def _get_obs(self) -> np.ndarray:
        # theta, thetadot = self.state
        theta = self.state[:, 0]
        thetadot = self.state[:, 1]
        return np.array([np.cos(theta), np.sin(theta), thetadot]).reshape(self.batch_size, self.d_obs)

    def get_obs_vec(self) -> np.ndarray:
        """Return observation vector"""
        return self._get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500*self.batch_size, 500)
            #self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            self.viewer.set_bounds(-2.2, 2.2*self.batch_size, -2.2, 2.2)

            self.rods = []
            self.axle_transforms = []
            self.pole_transforms = []
            self.imgs = []
            self.imgtranss = []
            fname = path.join(path.dirname(__file__), "../assets/clockwise.png")


            for i in range(self.batch_size):
                rod = rendering.make_capsule(1, .2)
                rod.set_color(.8, .3, .3)
                pole_transform = rendering.Transform(translation=(i*self.batch_size,0.0))
                self.pole_transforms.append(pole_transform)
                rod.add_attr(pole_transform)
                self.viewer.add_geom(rod)
                axle = rendering.make_circle(.05)
                axle.set_color(0, 0, 0)
                axle_transform = rendering.Transform(translation=(i*self.batch_size,0.0))
                self.axle_transforms.append(axle_transform)
                axle.add_attr(axle_transform)
                self.viewer.add_geom(axle)

                img = rendering.Image(fname, 1., 1.)
                imgtrans = rendering.Transform(translation=(i*self.batch_size,0.0))
                self.imgtranss.append(imgtrans)
                img.add_attr(imgtrans)
                self.imgs.append(img)

        for i in range(self.batch_size):
            self.viewer.add_onetime(self.imgs[i])
            self.pole_transforms[i].set_rotation(self.state[i,0] + np.pi / 2)
            if self.last_u.all():
                self.imgtranss[i].scale = (-self.last_u[i,0] / 2, np.abs(self.last_u[i,0]) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # def randomize_params(self):
    #     self.m = np.random.uniform(self.mean_m - self.mean_m * self.noise_scale,
    #                                self.mean_m + self.mean_m * self.noise_scale) + self.bias_scale * self.mean_m
    #     self.l = np.random.uniform(self.mean_l - self.mean_l * self.noise_scale,
    #                                self.mean_l + self.mean_l * self.noise_scale) + self.bias_scale * self.mean_l

    def print_params(self):
        print(self.dyn_params)


    def set_params(self, params):
        """Sets dynamics parameters from dictionary"""
        self.dyn_params = copy.deepcopy(params)

    def __repr__(self):
        str = 'Name: %r \n, D_STATE = , %r \n, D_ACTION = %r \n, D_OBS = %r \n \
               Dynamics Parameters: %r'.format(self.env_name, self.d_state, self.d_action,
                                                self.d_obs, self.dyn_params)
        return str




