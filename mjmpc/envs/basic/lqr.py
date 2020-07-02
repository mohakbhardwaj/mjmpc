import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class LQREnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.d_state = A.shape[0]
        self.d_action = B.shape[-1]
        self.viewer = None

        high = np.array([100] * self.d_state)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.d_action,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        cost = self.state.T.dot(self.Q).dot(self.state) \
               + u.T.dot(self.R).dot(u)
        self.state = self.A.dot(self.state) + self.B.dot(u)
        return self._get_obs(), -cost, False, {}

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        high = np.array([100] * self.d_state)
        self.state = self.np_random.uniform(low=-high, high=high).reshape(self.d_state, 1)
        return self.state.copy()

    def _get_obs(self):
        return self.state.copy()

    def render(self, mode='human'):
        pass
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(500,500)
        #     self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
        #     rod = rendering.make_capsule(1, .2)
        #     rod.set_color(.8, .3, .3)
        #     self.pole_transform = rendering.Transform()
        #     rod.add_attr(self.pole_transform)
        #     self.viewer.add_geom(rod)
        #     axle = rendering.make_circle(.05)
        #     axle.set_color(0,0,0)
        #     self.viewer.add_geom(axle)
        #     fname = path.join(path.dirname(__file__), "../assets/clockwise.png")
        #     self.img = rendering.Image(fname, 1., 1.)
        #     self.imgtrans = rendering.Transform()
        #     self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        # self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        # return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass
    
    def get_env_state(self):
        return {'state':self.state.copy()}
    
    def set_env_state(self, state_dict):
        self.state = state_dict['state']
        