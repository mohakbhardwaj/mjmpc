import numpy as np
from gym import utils
from envs import mujoco_env
from mujoco_py import MjViewer


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, batch_size):
        model_path = 'xml/half_cheetah.xml'
        super(HalfCheetahEnv, self).__init__(model_path, 5,
                                            'half_cheetah', 18, 6, 17, batch_size=batch_size)
        utils.EzPickle.__init__(self)
        self.dyn_params = None

    def step(self, a):
        """
        :param a: np.ndarray of size [batch_size X 1]
        :return: obs: np.ndarray [batch_size X d_obs], reward: np.ndarray [batch_size]
                 done: np.ndarray [batch_size, info: dict{}
        """
        assert a.shape[0] == self.batch_size, "Number of control vectors must be equal to batch size"

        xposbefore = np.array([data.qpos[0] for data in self.data])
        self.do_simulation(a)
        xposafter = np.array([data.qpos[0] for data in self.data])

        vel_reward = (xposafter-xposbefore)/self.dt
        # vel_reward = vel_x  # make cheetah move in negative x direction
        ctrl_cost = 0.1 * np.square(a).sum(axis=1)
        reward = vel_reward  - ctrl_cost
        done = np.zeros((self.batch_size), dtype=bool)

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        obs_vec = np.zeros((self.batch_size, self.d_obs))
        for i in range(self.batch_size):
            obs_vec[i, :] = np.concatenate([
                self.data[i].qpos.flat[1:],
                self.data[i].qvel.flat,
            ])
        return obs_vec

    def reset_model(self):
        qpos_init = self.init_qpos.copy() + self.np_random.uniform(low=-.1,
                                                                   high=.1, size=self.model.nq)
        qvel_init = self.init_qvel.copy() + self.np_random.randn(self.model.nv) * .1
        qpos_init = np.tile(qpos_init, (self.batch_size, 1))
        qvel_init = np.tile(qvel_init, (self.batch_size, 1))

        self.set_sim_state(qpos_init, qvel_init)
        for sim in self.sim:
            sim.forward()
        return self._get_obs()

    def _viewer_setup(self):
        # self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.viewer.cam.distance = self.model.stat.extent*1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
        self.sim[0].forward()

    def get_state(self) -> np.ndarray:
        state_vec = np.zeros((self.batch_size, self.d_state))
        for i in range(self.batch_size):
            state = self.sim[i].get_state()
            state_vec[i, :] = np.concatenate([
                state.qpos.flat, state.qvel.flat])
        return state_vec

    def set_state(self, state: np.ndarray):
        qpos = state[:, 0:self.model.nq].copy()
        qvel = state[:, self.model.nq: self.model.nq + self.model.nv].copy()
        self.set_sim_state(qpos, qvel)

    def set_params(self):
        raise(NotImplementedError, "implement set params")
    
    def get_params(self):
        raise(NotImplementedError, "implement set params")
