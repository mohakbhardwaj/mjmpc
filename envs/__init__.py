from .gym_env_wrapper import GymEnvWrapper
from gym.envs.registration import register

register(
  id='SimplePendulum-v0',
  entry_point='envs.basic.pendulum:PendulumEnv',
)

register(
  id='Swimmer-v0',
  entry_point='envs.basic.swimmer:SwimmerEnv',
)

register(
    id='HalfCheetah-v0',
    entry_point='envs.basic.half_cheetah:HalfCheetahEnv',
)

register(
    id='trajopt_reacher-v0',
    entry_point='envs.basic.reacher_env:Reacher7DOFEnv',
    max_episode_steps=75,
)

register(
    id='trajopt_continual_reacher-v0',
    entry_point='envs.basic.reacher_env:ContinualReacher7DOFEnv',
    max_episode_steps=250,
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from envs.basic.reacher_env import Reacher7DOFEnv, ContinualReacher7DOFEnv