# mjmpc
Sampling based Model Predictive Control using MuJoCo simulator
## Installation
[1]
```
git clone git@github.com:mohakbhardwaj/mjmpc.git && cd mjmpc
conda create --name mjmpc python=3.7
conda activate mjmpc
conda env update -f environment.yml
```
[2] Install gym and mujoco_py

[3] Install mj_envs, mjrl

## Examples
```
cd examples
python job_script.py --config_file <config_file> --controller_type  <controller_name>
```
For example, to run MPPI on trajopt_reacher-v0 we can use
```
python job_script.py --config_file configs/sawyer_reacher-v0.yml --controller_type mppi
```
Replace mppi with random_shooting, cem or dmd to run different controllers using parameters provided in the config files.


## MPPI Parameters

These parameters are currently manually tuned.

| Env Name          | Episode Length | Horizon | Num Particles | Lambda | Covariance | Step Size | Gamma | Num Iters |
|-------------------|----------------|---------|---------------|--------|------------|-----------|-------|-----------|
| SimplePendulum-v0 | 200            | 32      | 24            | 0.01   | 3.5        | 0.55      | 1.0   | 1         |
| Swimmer-v0        | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| HalfCheetah-v0    | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| trajopt_reacher-v0| 200            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |


## TODO
1. Batch version of controller class
2. Environment render function must have functionality to save frames/videos and work for batch size > 1
3. <span style="color:red">Implement rollout function for mujoco_env in Cython.</span>
4. <span style="color:red"> Grid search for tuning controller hyperparameters.</span>
