from copy import deepcopy
import torch.multiprocessing as multiprocessing
from collections import OrderedDict

import gym
import numpy as np
import time

from .base_vec_env import VecEnv, CloudpickleWrapper
from .tile_images import tile_images
from .util import flatten_obs


def _worker(remote, parent_remote, env_fn_wrapper, model):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':   
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'get_obs':
                remote.send(env.get_obs())
            elif cmd == 'set_env_state':
                remote.send(env.set_env_state(data))
            elif cmd == 'get_env_state':
                state = env.get_env_state()
                remote.send(state)
            elif cmd == 'rollout':
                obs_vec, act_vec, log_prob_vec, rew_vec, done_vec, next_obs_vec, info = env.rollout_cl(model, **data)
                remote.send((obs_vec, act_vec, log_prob_vec, rew_vec, done_vec, next_obs_vec, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'get_seed':
                remote.send((env.seed))
            else:
                raise NotImplementedError
        except EOFError:
            break


class TorchModelVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments
    with a torch model as policy for rollouts

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow
        sessions or other non thread-safe libraries are used in the parent (see issue #217).
        However, compared to 'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods,
        users must wrap the code in an ``if __name__ == "__main__":``
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param model: ([torch.nn model]) Model used for parallel rollouts
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'fork' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, model, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        # NOTE: this is required for the ``fork`` method to work
        self.model = model
        self.model.share_memory()

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            fork_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if fork_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), self.model)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def update_model_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def step_async(self, actions):
        if np.size(actions.shape) > 1:
            for remote, action in zip(self.remotes, actions):
                remote.send(('step', action))
        else:
            for remote in self.remotes:
                remote.send(('step', actions))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def rollout(self, num_rollouts, horizon, mode='mean', noise=None): 
        """
        Rollout the environments to a horizon given open loop action sequence

        :param 
        """
        self.rollout_async(num_rollouts, horizon, mode, noise)
        return self.rollout_wait()

    def rollout_async(self, num_rollouts, horizon, mode='mean', noise=None):
        assert num_rollouts % len(self.remotes) == 0, "Number of particles must be divisible by number of cpus"
        batch_size = int(num_rollouts/len(self.remotes))
        for i,remote in enumerate(self.remotes):
            if noise is not None:
                noise_vec_i = noise[i*batch_size: (i+1)*batch_size, :, :].copy() 
            else:
                noise_vec_i = None
            data = {'batch_size': batch_size, 'horizon': horizon, 'mode': mode, 'noise': noise_vec_i}
            remote.send(('rollout', data))
        self.waiting = True

    def rollout_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting=False
        obs_vec = [res[0] for res in results]
        act_vec = [res[1] for res in results]
        log_prob_vec = [res[2] for res in results]
        rew_vec = [res[3] for res in results]
        done_vec = [res[4] for res in results]
        next_obs_vec = [res[5] for res in results] 
        info = [res[6] for res in results]
        stacked_obs  = np.concatenate(obs_vec, axis=0)
        stacked_act = np.concatenate(act_vec, axis=0)
        stacked_log_prob = np.concatenate(log_prob_vec, axis=0)
        stacked_rews = np.concatenate(rew_vec, axis=0)
        stacked_done = np.concatenate(done_vec, axis=0)
        stacked_next_obs = np.concatenate(next_obs_vec, axis=0)
        return stacked_obs, stacked_act, stacked_log_prob, stacked_rews, stacked_done, stacked_next_obs, info

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
    
    def get_obs(self):
        for remote in self.remotes:
            remote.send(('get_obs', None))
        observations = [remote.recv() for remote in self.remotes]
        stacked_obs = np.vstack(observations)
        return stacked_obs


    # --------------------------------
    # get and set states
    # --------------------------------

    def set_env_state(self, state_dicts):
        """
        Set the state of all envs given a list of 
        state dicts
        If only one state is provided, we set state of all envs to that
        else each env must be provided one state
        """
        if isinstance(state_dicts, list):
            num_states = len(state_dicts)
            assert num_states == 1 or num_states == len(self.remotes), \
                "num states should equal 1 (same for all envs) or 1 per env"
            if num_states == 1:
                state_dicts = [deepcopy(state_dicts[0]) for j in range(len(self.remotes))]
        else: state_dicts = [deepcopy(state_dicts) for j in range(len(self.remotes))]
        for i,remote in enumerate(self.remotes):
            remote.send(('set_env_state', state_dicts[i]))
        for remote in self.remotes: remote.recv()

    def get_env_state(self):
        for remote in self.remotes:
            remote.send(('get_env_state', None))
        states = [remote.recv() for remote in self.remotes]
        return states

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def seed(self, seed_list):
        assert len(seed_list) == len(self.remotes), "Each environment must be provided a seed"
        for i,remote in enumerate(self.remotes):
            remote.send(('seed', seed_list[i]))
        results = [remote.recv() for remote in self.remotes]