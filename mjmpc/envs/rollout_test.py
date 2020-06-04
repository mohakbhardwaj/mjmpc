if __name__ == "__main__":
    import numpy as np
    import mj_envs
    import gym
    import mjmpc.envs
    from mjmpc.policies import MPCPolicy
    import yaml
    from copy import deepcopy
    from mjmpc.envs.gym_env_wrapper_cy import GymEnvWrapperCy
    from mjmpc.envs.gym_env_wrapper import GymEnvWrapper


    env = gym.make('pen-v0')
    env = GymEnvWrapper(env)

    rollout_env = deepcopy(env)

    def rollout_fn(u_vec: np.ndarray):
        """
        Given a batch of sequences of actions, rollout 
        in sim envs and return sequence of costs
        """
        obs_vec, rew_vec, done_vec, _ = rollout_env.rollout(u_vec.copy())
        return -1.0*rew_vec #we assume environment returns rewards, but controller needs consts
    
    #Create functions for controller
    def set_sim_state_fn(state_dict: dict):
        """
        Set state of simulation environments for rollouts
        """
        rollout_env.set_env_state(state_dict)   

    with open("../../examples/configs/hand/pen-v0.yml") as file:
        exp_params = yaml.load(file, Loader=yaml.FullLoader)
    policy_params = {}
    policy_params = exp_params["mppi"]
    policy_params['base_action'] = exp_params['base_action']
    policy_params['num_actions'] = env.action_space.low.shape[0]
    policy_params['action_lows'] = env.action_space.low
    policy_params['action_highs'] = env.action_space.high
    policy_params['num_particles'] = policy_params['num_cpu'] * policy_params['particles_per_cpu']

    del policy_params['particles_per_cpu'], policy_params['num_cpu']
    print(policy_params)
    policy = MPCPolicy(controller_type="mppi",
                        param_dict=policy_params, batch_size=1) #Only batch_size=1 is supported for now
    policy.controller.set_sim_state_fn = set_sim_state_fn
    policy.controller.rollout_fn = rollout_fn

    action, _  = policy.get_action(env.get_env_state())
    print(action)