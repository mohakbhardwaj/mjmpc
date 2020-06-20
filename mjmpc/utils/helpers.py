#!/usr/bin/env python
import numpy as np
import os

from mjmpc.envs import *
from mjmpc.policies import *
from .logger import LoggerClass


def set_qpos_qvel(sim, qpos, qvel, nq, nv):
    state = sim.get_state()
    for i in range(nq):
        state.qpos[i] = qpos[i]
    for i in range(nv):
        state.qvel[i] = qvel[i]
    sim.set_state(state)
    sim.forward()

def render_trajs(env, trajectories, n_times=1):
    try:
        for _ in range(n_times):
            for traj in trajectories:
                env.reset()
                state = traj['states'][0]
                env.set_env_state(state)
                for action in traj['actions']:
                    env.render()
                    env.step(action)
    except KeyboardInterrupt:
        print('Exiting ...')
        

def dump_videos(env,
                trajectories,
                frame_size=(640,480),
                folder='/tmp/',
                filename='newvid',
                camera_name=None,
                device_id=0):

    import skvideo.io
    for ep, traj in enumerate(trajectories):
        arrs = []
        env.reset()
        state = traj['states'][0]
        env.set_env_state(state)
        for action in traj['actions']:
            env.step(action)
            curr_frame = env.get_curr_frame(frame_size=frame_size, camera_name=camera_name, device_id=device_id)
            arrs.append(curr_frame)

    # for ep in range(num_episodes):
    #     print("Episode %d: rendering offline " % ep, end='', flush=True)
    #     o = self.reset()
    #     d = False
    #     t = 0
    #     arrs = []
    #     t0 = timer.time()
    #     while t < horizon and d is False:
    #         a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
    #         o, r, d, _ = self.step(a)
    #         t = t+1
    #         curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
    #                                      mode='offscreen', camera_name=camera_name, device_id=0)
    #         arrs.append(curr_frame[::-1,:,:])
            # print(t, end=', ', flush=True)
        out_file = os.path.join(folder, filename + str(ep) + ".mp4")
        skvideo.io.vwrite(out_file, np.asarray(arrs))
        print("saved", out_file)
            # t1 = timer.time()
            # print("time taken = %f"% (t1-t0))



def get_logger(display_name, log_dir, mode):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = LoggerClass()
    logger.setup(display_name, os.path.join(log_dir, 'log.txt'), 'debug')
    return logger

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def tensor_dict_list_to_array(tensor_dict_list):
    """
    Stack a list of dictionaries into a numpy array 
    :param tensor_dict_list: a list of dictionaries of tensors
    :return numpy array
    """
    ret = []
    for d in tensor_dict_list:
        # curr_vals = []
        # for k in d.keys():
            # curr_vals.append(d[k])
        # curr_vals = np.array(curr_vals)
        curr_vals = np.concatenate([d[k] for k in d.keys()])
        
        ret.append(curr_vals.copy())
    return np.array(ret)

def tensor_dict_to_array(tensor_dict):
    vals = np.concatenate([tensor_dict[k] for k in tensor_dict.keys()])
    return np.array(vals)
