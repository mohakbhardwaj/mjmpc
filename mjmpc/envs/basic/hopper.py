import numpy as np
from gym import utils
from .mujoco import MujocoEnv

class HopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, asset_path='hopper.xml'):
        MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        #alive_bonus = 1.0
        #reward = (posafter - posbefore) / self.dt
        #reward += alive_bonus
        #reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .65) and (abs(ang) < .15))
        reward = -1000. if done else 0.
        ob = self.get_obs()
        return ob, reward, done, dict(cost=float(done))

    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self.get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
