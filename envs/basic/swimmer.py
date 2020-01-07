import numpy as np
from gym import utils, spaces
from envs import mujoco_env
from mujoco_py import MjViewer

"""
Possible issues: 
1. self.dyn_params (in __init__) and set_params function needs to be implemented - while this is not urgent right now, it might be useful
                                                                                   for dynamics randomization later
"""

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    def __init__(self, batch_size):
        model_path = 'xml/swimmer.xml'
        # super(SwimmerEnv, self).__init__(model_path, 5, 'swimmer', 14, 4, 12, batch_size=batch_size)
        # AL: make it fully observable for now
        super(SwimmerEnv, self).__init__(model_path, 5, 'swimmer', 14, 4, 14, batch_size=batch_size)
        utils.EzPickle.__init__(self)
        self.dyn_params = None

        # high = np.concatenate([np.inf * np.ones(2), np.ones(10), np.inf * np.ones(7)])# self.obs_dim
        # low = -high
        # self.observation_space = spaces.Box(low, high)

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
        
        vel_x = (xposafter-xposbefore)/self.dt
        vel_reward = -vel_x  # make swimmer move in negative x direction
        # ctrl_cost = 1e-3 * np.square(a).sum()
        reward = vel_reward #- ctrl_cost
        done = np.zeros((self.batch_size), dtype=bool)

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        obs_vec = np.zeros((self.batch_size, self.d_obs))
        for i in range(self.batch_size):
            obs_vec[i, :] = np.concatenate([
                self.data[i].qpos.flat[:2],
                np.cos(self.data[i].qpos.flat[2:]),
                np.sin(self.data[i].qpos.flat[2:]),
                self.data[i].qvel.flat,
            ])
        return obs_vec

    def get_obs_vec(self):
        return self._get_obs()


    def reset_model(self):
        qpos_init = self.init_qpos.copy()
        qpos_init[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        qpos_init = np.tile(qpos_init, (self.batch_size, 1))
        qvel_init = np.tile(self.init_qvel, (self.batch_size, 1))

        self.set_sim_state(qpos_init, qvel_init)
        for sim in self.sim: sim.forward()
        return self._get_obs()

    def _viewer_setup(self):
        # self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim[0].forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2
    
    def get_state(self) -> np.ndarray:
        state_vec = np.zeros((self.batch_size, self.d_state))
        for i in range(self.batch_size):
            state = self.sim[i].get_state()
            state_vec[i,:] = np.concatenate([
                state.qpos.flat, state.qvel.flat])
        return state_vec

    def get_reward(self, state, action, next_state):
        xposbefore = state[:, 0]
        xposafter = next_state[:, 0]
        vel_x = (xposafter - xposbefore) / self.dt
        return -vel_x

    def set_state(self, state: np.ndarray):
        qpos = state[:, 0:self.model.nq].copy()
        qvel = state[:, self.model.nq: self.model.nq + self.model.nv].copy()
        
        self.set_sim_state(qpos, qvel)
    
    def set_params(self):
        raise(NotImplementedError, "implement set params")
    
