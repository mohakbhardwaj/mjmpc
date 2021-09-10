import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, asset_path='half_cheetah.xml'):
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self.get_obs()
        height = self.get_body_com("fthigh")[-1]

        if height <= 0.45 or height >= 0.95:
            reward_fail, done = -1000., True
        else:
            reward_fail, done = 0., False
        return ob, reward_fail, done, dict(height=height)

    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_env_state(self):
        state = self.sim.get_state()
        qpos = state.qpos.flat.copy()
        qvel = state.qvel.flat.copy()
        state = {'qpos': qpos, 'qvel': qvel}
        return state

    def get_state(self):
        return self.get_env_state()

    def set_env_state(self, state_dict):
        qpos = state_dict['qpos'].copy()
        qvel = state_dict['qvel'].copy()
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()


    def evaluate_success(self, paths):
        return 0.
