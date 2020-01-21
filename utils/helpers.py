#!/usr/bin/env python
from envs import *
from policies import *
import numpy as np


def set_qpos_qvel(sim, qpos, qvel, nq, nv):
    state = sim.get_state()
    for i in range(nq):
        state.qpos[i] = qpos[i]
    for i in range(nv):
        state.qvel[i] = qvel[i]
    sim.set_state(state)
    sim.forward()

def render_trajs(env, trajectories, n_times=1):
    for _ in range(n_times):
        for traj in trajectories:
            env.reset()
            state = traj['states'][0]
            env.unwrapped.set_env_state(state)
            for action in traj['actions']:
                env.render()
                env.step(action)


