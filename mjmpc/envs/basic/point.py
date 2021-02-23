import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class PointEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 30}

    def __init__(self, mass=1., target_dist=5., xlim=2.5):
        self.mass = mass
        self.dt = 0.1
        self.target_dist = target_dist
        self.world_width = 1.5*2*target_dist
        self.max_speed = 2.
        self.lim = np.array([xlim, self.world_width])

        high_state = np.array([self.world_width, self.world_width, 1., 1.],
                              dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_state, high=high_state,
                                            dtype=np.float32)
        self.reward_range = (-1., 1.)

        self.seed()
        self.state = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        posn = self.np_random.uniform(low=-0.1, high=0.1, size=2)
        self.state = np.concatenate([posn, [0., 0.]]).astype(np.float32)
        return np.array(self.state)

    def get_state(self):
        return np.array(self.state)

    def get_env_state(self):
        return self.get_state()

    def get_obs(self):
        return self.get_state()

    def set_env_state(self, state):
        self.state = np.array(state)

    def step(self, a):
        a = np.clip(a, self.action_space.low[0], self.action_space.high[0])
        pos, vel = self.state[:2], self.state[2:]

        done = (np.abs(pos) > self.lim).any() # constraint violation
        if done:
            rew = -1000 #0.
        else:
            rew = self.state[-2:].dot([-self.state[1], self.state[0]])
            rew /= (1. + np.abs(np.linalg.norm(self.state[:2]) - self.target_dist))

            # Normalizing to range [-1, 1]
            rew /= self.max_speed*self.target_dist

        cost = float(done)
        distance = self.dist_to_unsafe()

        info = dict(cost=cost, distance=distance)

        # State update
        pos += vel*self.dt + a*self.dt**2 / (2*self.mass)
        vel += a*self.dt/self.mass

        # Ensure agent is within reasonable range
        #self.state[:2] = np.clip(pos, -self.world_width, self.world_width)
        vel[np.isclose(vel, 0)] = 0.

        # Clip speed, if necessary
        speed = np.linalg.norm(self.state[-2:])
        if speed > self.max_speed:
            self.state[-2:] *= self.max_speed / speed

        return np.array(self.state), rew, done, info

    def dist_to_unsafe(self):
        return max(0, self.signed_dist_to_unsafe())

    def signed_dist_to_unsafe(self):
        return min(self.lim[0] - self.state[0], self.lim[0] + self.state[0],
                   self.lim[1] - self.state[1], self.lim[1] + self.state[1])

    def render(self, mode='human'):
        viewer_size = 500
        center, scale = viewer_size // 2, viewer_size / self.world_width
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(viewer_size, viewer_size)
            self.ring_trans = rendering.Transform((viewer_size/2, viewer_size/2))
            self.ring = rendering.make_circle(self.target_dist*scale, res=100, filled=False)
            self.ring.set_color(0., 0.8, 0.)
            self.ring.add_attr(self.ring_trans)
            self.viewer.add_geom(self.ring)

            self.left_boundary = rendering.Line(start=(center - scale*self.lim[0], 0),
                                                end=(center - scale*self.lim[0],
                                                     viewer_size))
            self.left_boundary.set_color(0.8, 0., 0.)
            self.viewer.add_geom(self.left_boundary)

            self.right_boundary = rendering.Line(start=(center + scale*self.lim[0], 0),
                                                 end=(center + scale*self.lim[0],
                                                      viewer_size))
            self.right_boundary.set_color(0.8, 0., 0.)
            self.viewer.add_geom(self.right_boundary)

            self.agent = rendering.make_circle(scale*0.1, res=100)
            self.agent_trans = rendering.Transform((viewer_size/2, viewer_size/2))
            self.agent.add_attr(self.agent_trans)
            self.viewer.add_geom(self.agent)

        if self.state is None:
            return None

        posn = self.state[:2]
        self.agent_trans.set_translation(center + scale*posn[0], center + scale*posn[1])

        print("{:.2f}, {:.2f}".format(self.state[0], self.state[1]), end='\r')

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def evaluate_success(self, paths):
        return 0.
