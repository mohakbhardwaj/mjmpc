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


## MPPI Parameters

These parameters are currently manually tuned.

| Env Name             | Episode Length | Horizon | Num Particles | Lambda | Covariance | Step Size | Gamma | Num Iters |
|----------------------|----------------|---------|---------------|--------|------------|-----------|-------|-----------|
| SimplePendulumEnv-v0 | 200            | 32      | 24            | 0.01   | 3.5        | 0.55      | 1.0   | 1         |
| SwimmerEnv-v0        | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| HalfCheetahEnv-v0    | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |


## TODO
1. Documentation for meaning of dynamics parameters for different environments
2. Batch version of controller class
3. Environment render function must have functionality to save frames/videos and work for batch size > 1
4. <span style="color:red">Implement rollout function for mujoco_env in Cython.</span>
5. <span style="color:red"> Grid search for tuning controller hyperparameters.</span>
