from .gym_env_wrapper import GymEnvWrapper
# from mjmpc.envs.gym_env_wrapper_cy import GymEnvWrapperCy
from gym.envs.registration import register

register(
  id='SimplePendulum-v0',
  entry_point='mjmpc.envs.basic.pendulum:PendulumEnv',
  max_episode_steps=200
)

register(
  id='Swimmer-v0',
  entry_point='mjmpc.envs.basic.swimmer:SwimmerEnv',
)

register(
    id='half_cheetah-v0',
    entry_point='mjmpc.envs.basic.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000
)

register(
    id='reacher_7dof-v0',
    entry_point='mjmpc.envs.basic.reacher_env:Reacher7DOFEnv',
    max_episode_steps=75,
)

register(
    id='continual_reacher-v0',
    entry_point='mjmpc.envs.basic.reacher_env:ContinualReacher7DOFEnv',
    max_episode_steps=250,
)

register(
    id='continual_maze-v0',
    entry_point='mjmpc.envs.basic.maze_env:ContinualParticleMaze',
    max_episode_steps=200,
)

register(
    id='humanoid-v0',
    entry_point='mjmpc.envs.basic.humanoid:HumanoidEnv',
    max_episode_steps=1000
)

register(
    id='point-v0',
    entry_point='mjmpc.envs.basic.point:PointEnv',
    max_episode_steps=500
)

register(
    id='hopper-v0',
    entry_point='mjmpc.envs.basic.hopper:HopperEnv',
    max_episode_steps=1000
)

register(
    id='walker-v0',
    entry_point='mjmpc.envs.basic.walker2d:Walker2dEnv',
    max_episode_steps=1000
)


# from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjmpc.envs.basic.reacher_env import Reacher7DOFEnv, ContinualReacher7DOFEnv
from mjmpc.envs.basic.lqr import LQREnv
# import mj_envs
