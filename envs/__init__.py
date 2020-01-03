from .environment import Env
from gym.envs.registration import register

register(
  id='SimplePendulumEnv-v0',
  entry_point='envs.basic.pendulum:PendulumEnv',
)

register(
  id='SimulatedSimplePendulumEnv-v0',
  entry_point='envs.basic.pendulum:SimulatedPendulumEnv'
)

register(
  id='SwimmerEnv-v0',
  entry_point='envs.basic.swimmer:SwimmerEnv',
)

register(
    id='HalfCheetahEnv-v0',
    entry_point='envs.basic.half_cheetah:HalfCheetahEnv',
)
