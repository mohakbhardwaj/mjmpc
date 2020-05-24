# mjmpc
A collection of sampling based Model Predictive Control algorithms

## Installation
You first need to download MuJoCO and obtain a licence key from [here](https://www.roboti.us/index.html)

[1] Create conda environment 
```
conda create --name mjmpc python=3.7
conda activate mjmpc
```

[2] Install OpenAI Gym and `mujoco_py`

[2] Clone mjmpc
```
git clone git@github.com:mohakbhardwaj/mjmpc.git && cd mjmpc
conda env update -f setup/environment.yml
```

[3] Install [mjrl](https://github.com/aravindr93/mjrl)

[4] (Optional) Install [mj_envs](https://github.com/vikashplus/mj_envs) (only required if you want to run hand_manipulation_suite, sawyer or classic_control environments)

[5] Install mjmpc
```
cd mjmpc
pip install -e .
```

## Examples
Take a look at the examples directory.
```
cd examples
```

To run a single instance of a controller by loading parameters from a config file run
```
python example_mpc.py --config_file <config_file> --controller_type  <controller_name> --save_dir .
```
For example, to run MPPI for a reaching task with Sawyer robot, run the following
```
python example_mpc.py --config_file configs/reacher_7d0f-v0 --controller_type  mppi --save_dir .
``` 
After running MPPI the results will be stored in ./reacher_7dof-v0/ <timestamp>/mppi/. 
Use the flag `--dump_vids` to dump videos of all the trajectories.

We have provided example config files in `examples/configs` folder. The parameters for individual algorithms are explained below. 


## List of Controllers 
We have implemented the following control algorithms

[1] `random_shooting`

[2] `mppi`

[3] `cem`

[4] `gaussian_dmd`

[5] `pfmpc`

<span style="color:red">TODO: Add details of algorithms + parameters.</span>

## Tuning Controllers and Running Parameter Sweeps 
We have also provided a job_script for tuning and benchmarking various MPC algorithms. 
```
python job_script.py --config_file <config_file> --controller_type  <controller_name>
```
For example, to run MPPI on trajopt_reacher-v0 we can use
```
python job_script.py --config_file configs/sawyer_reacher-v0.yml --controller_type mppi
```
Replace mppi with random_shooting, cem or dmd to run different controllers using parameters provided in the config files. The working of the job script are explained below.


## Using Your Own Environments
As such the control framework is agnostic to the environment definition and only expects as input two functions

[1] `rollout_fn`: Takes as input a batch of actions and returns a vector of costs encountered

[2] `set_sim_state_fn`: sets the state of simulation environments.

However, if you wish to use our GymEnvWrapper, it expects the environment to have a few additional functions implemented

<span style="color:red">TODO: Add details. Currently can be seen from reacher_env.py in envs folder</span>

## Control Parameters

(TO BE UPDATED!!)

These parameters are currently manually tuned.

| Env Name          | Episode Length | Horizon | Num Particles | Lambda | Covariance | Step Size | Gamma | Num Iters |
|-------------------|----------------|---------|---------------|--------|------------|-----------|-------|-----------|
| SimplePendulum-v0 | 200            | 32      | 24            | 0.01   | 3.5        | 0.55      | 1.0   | 1         |
| Swimmer-v0        | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| HalfCheetah-v0    | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| trajopt_reacher-v0| 200            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |


## TODO
1. Batch version of controllers
2. ~~Environment render function must have functionality to save frames/videos and work for batch size > 1~~
3. <span style="color:red">Implement rollout function for mujoco_env in Cython.</span>
4. ~~<span style="color:red"> Grid search for tuning controller hyperparameters.</span>~~
