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

def render_env(env, state_vec):
    n_steps = state_vec.shape[0]
    env.reset()
    for i in range(n_steps):
        curr_state = state_vec[:, i]
        # curr_action = action_vec[:, i]
        env.set_state(curr_state)
        env.render()
        # _,_,_,- = env.step


