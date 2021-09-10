# mjmpc
A collection of sampling based Model Predictive Control algorithms. 

If you use this repository as part of your research please cite the following publication::
```
@inproceedings{
bhardwaj2021blending,
title={Blending {\{}MPC{\}} {\&} Value Function Approximation for Efficient Reinforcement Learning},
author={Mohak Bhardwaj and Sanjiban Choudhury and Byron Boots},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=RqCC_00Bg7V}
}
```

## Installation
You first need to download MuJoCO and obtain a licence key from [here](https://www.roboti.us/index.html)

[1] Create conda environment 
```
conda create --name mjmpc python=3.7
conda activate mjmpc
```

[2] Install [mujoco_py](https://github.com/openai/mujoco-py) and [gym](https://gym.openai.com/docs/#installation) 
 
[3] Clone mjmpc
```
git clone git@github.com:mohakbhardwaj/mjmpc.git && cd mjmpc
conda env update -f setup/environment.yml
```

[4] Clone and install [mjrl](https://github.com/aravindr93/mjrl)

[5] (Optional) Clone and install [mj_envs](https://github.com/vikashplus/mj_envs) (only required if you want to run hand_manipulation_suite, sawyer or classic_control environments)

[6] Install mjmpc
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


## Controllers 
Following parameters are common for all controllers
| Parameter         |                                                                       | 
|-------------------|-----------------------------------------------------------------------|
| ``horizon``       | rollout horizon                                                       |
| ``num_particles`` | number of particles to rollout                                        |
| ``n_iters``       | number of iterations of optimization per timestep                     |
| ``gamma``         | discount factor                                                       |
| ``filter_coeffs`` | coefficients for autoregressive filtering (generate correlated noise) |
| ``base_action``   | action to append at the end after shifting distribution for next step | 


Additionally, each controller has it's own specific parameters

### Gaussian Controllers
These controllers use a Gaussian control distribution and have the following common parameters

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``init_cov``  | initial covariance of Gaussian                        |
| ``step_size`` | step size for updating distribution at every timestep |


#### Random Shooting
Samples particles from a Gaussian with fixed covariance and selects next mean to be the rollout with minimum cost. Has no additional parameters.

#### Model Predictive Path Integral Control (MPPI)
Based on [Williams et al.](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf), it samples particles from a Gaussian with fixed covariance and updates the mean using a softmax of rollouts. Has the followig additional parameters:

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``lam``       | temperature for softmax                               |
| ``alpha``     | flag to enable control costs in rollouts (0: enable, 1: disable) |


#### Cross Entropy Method (CEM)
Samples particles from a Gaussian control distribution and updates the mean and covariance using sample estimates from a set of elite samples based on cost. 

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``cov_type``       | 'diag' means covariance is forced to be diagonal, and 'full' allows actions to be correlated to each other.               |
| ``elite_frac``     | fraction of total samples considered elite  |
| ``beta``           | ``beta * I`` is added to covariance to grow it at each timestep |

#### Gaussian DMD-MPC
From [Wagener et al.](https://arxiv.org/pdf/1902.08967.pdf). We use the exponentiated utility function and allow it to adapt the covariance as well. Has the same parameters as MPPI with the addition of ``cov_type`` and ``beta``.

## Non-Gaussian Controllers

#### Particle Filter MPC

Uses a non-parametric distribution represented by particles and updates it using a particle filtering approach, where particles weighted using an exponential of running cost with temperature ``lam``

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``cov_shift``       |  noise added to particles when shifting to next step    |
| ``cov_resample``    | noise for resampling particles  |



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



However, if you wish to use our GymEnvWrapper, it expects the environment to have a few additional functions implemented such as

[1] `set_env_state`: Sets the state of the environment

[2] `get_env_state`: Returns the current state of the environment

Please look at hre Currently can be seen from `envs/reacher_env.py` for an example environment.

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
2. Environment render function must have functionality to save frames/videos and work for batch size > 1
3. <span style="color:red">Implement rollout function for mujoco_env in Cython.</span>
4. ~~<span style="color:red"> Grid search for tuning controller hyperparameters.</span>~~


