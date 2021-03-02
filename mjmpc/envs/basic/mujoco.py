from gym.envs.mujoco import mujoco_env


class MujocoEnv(mujoco_env.MujocoEnv):

    def get_env_state(self):
        state = self.sim.get_state()
        qpos = state.qpos.flat.copy()
        qvel = state.qvel.flat.copy()
        state = dict(qpos=qpos, qvel=qvel)
        return state

    get_state = get_env_state  # alias

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
