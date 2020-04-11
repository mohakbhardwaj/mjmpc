# #!/usr/bin/env python
# import sys, os
# sys.path.insert(0, os.path.abspath('..'))
# import argparse
# import gym
# import numpy as np
# import tqdm

# import envs
# from envs.vec_env import SubprocVecEnv
# from utils import logger, timeit
# from policies import RandomPolicy

# parser = argparse.ArgumentParser(description='Run random policy on environment')
# parser.add_argument('--env', help='Environment name', default='SimplePendulumEnv-v0')
# parser.add_argument('--n_steps', type=int, default=100, help='Number of timesteps')
# parser.add_argument('--render', action='store_true', help='Render the environment')
# parser.add_argument('--seed', type=int, default=0, help='Seed value for consistent results')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size for environment rollouts')
# args = parser.parse_args()

# np.random.seed(args.seed)
# print('Environment = {0}'.format(args.env))
# kwargs = {'batch_size': args.batch_size}

# def make_env():
#     return gym.make(args.env, **kwargs)

# # env = gym.make(args.env, **kwargs)
# # env.seed(args.seed)
# env = make_env()
# env.seed(args.seed)

# policy = RandomPolicy(env.action_space.low, env.action_space.high, **kwargs)
# total_reward = np.zeros(shape=(args.batch_size, 1))
# n_steps = args.n_steps
# obs = env.reset()

# LOG_DIR = os.path.abspath("..")
# logger.setup("Random Policy", os.path.join(LOG_DIR, 'log.txt'), 'debug')
# logger.info('Runnning {0} steps'.format(n_steps))
# timeit.start('start')

# #Get a batch of random actions
# action_seq = policy.get_action_seq(horizon=n_steps)
# # for t in tqdm.tqdm(range(n_steps)):
# #     action = action_seq[:,t,:]#policy.get_action()
# #     obs, reward, done, info = env.step(action)
# #     total_reward += reward
# #     if args.render:
# #         env.render()
# obs_vec, state_vec, rew_vec, done_vec, _ = env.rollout(action_seq)

# #TODO: Add rendering using state_vec
# if args.render:
#     print('Rendering')
#     for t in tqdm.tqdm(range(state_vec.shape[1])):
#         env.set_state(state_vec[:, t, :])
#         env.render()

# timeit.stop('start')
# logger.info(timeit)
# logger.info('Average reward = {0}. Closing...'.format(np.average(rew_vec)))
# env.close()


import gym
env = gym.make('Humanoid-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

