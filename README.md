# mjmpc
Sampling based Model Predictive Control using MuJoCo simulator
## Installation
```
conda create --name mjmpc python=3.7
conda activate mjmpc
conda env update -f environment.yml
```
Install gym and mujoco_py

## Examples
By default the examples run on SimplePendulumEnv-v0
```
cd examples
python example_random_policy.py
python example_mppi.py
```

Use --render flag to visualize at the end.
