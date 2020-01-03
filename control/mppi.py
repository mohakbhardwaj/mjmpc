#!/usr/bin/env python
"""Model Predictive Path Integral Controller
TODO: Make it a batch version """
from .controller import Controller, scale_ctrl
import copy
import numpy as np
from scipy.signal import savgol_filter
import scipy.stats
import scipy.special



class MPPI(Controller):
    def __init__(self,
                 horizon,
                 init_cov,
                 min_cov,
                 prior_cov,
                 lam,
                 num_particles,
                 step_size,
                 alpha,
                 beta,
                 gamma,
                 n_iters,
                 num_actions,
                 action_lows,
                 action_highs,
                 set_state_fn,
                 get_state_fn,
                 rollout_fn,
                 terminal_cost_fn=None,
                 update_val=False,
                 rollout_callback=None,
                 batch_size=1,
                 seed=0):

        super(MPPI, self).__init__(num_actions,
                                   action_lows, 
                                   action_highs, 
                                   get_state_fn, 
                                   set_state_fn, 
                                   terminal_cost_fn)
        self.horizon = horizon
        self.init_cov = init_cov  # initial cov for sampling actions
        self.min_cov = min_cov  # clamp cov so it does not fall below minimum
        self.prior_cov = prior_cov  # grow the cov by this amount every timestep
        self.lam = lam
        self.num_particles = num_particles
        self.step_size = step_size  # step size for mean and covariance
        self.alpha = alpha  # weight on control cost (0 means passive distribution has zero control, 1 means passive distribution is same as the active control distribution)
        self.beta = beta  # step size for growing covariance
        self.gamma = gamma  # discount factor
        self.n_iters = n_iters  # number of iterations of optimization per timestep
        self.rollout_fn = rollout_fn
        self.rollout_callback = rollout_callback
        self.update_val = update_val
        self.update_cov = (self.prior_cov is not None and self.beta is not None)
        self.seed = seed
        self.batch_size = batch_size

        self.mean_action = np.zeros(shape=(horizon, num_actions))
        self.cov_action = self.init_cov * np.ones(shape=(horizon, num_actions))
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (horizon - 1)).reshape(horizon, 1)
        self._val = 0.0  # optimal value for current state
        self.num_steps = 0


    def _generate_rollouts(self, state):
        """
            Samples actions generates random trajectories for each particle
         """
        delta = np.random.normal(0.0, np.sqrt(self.cov_action[:, :, np.newaxis]),
                                 size=(self.horizon, self.num_actions, self.num_particles))
        _ctrl = self.mean_action[:, :, np.newaxis] + delta
        act_seq = scale_ctrl(_ctrl, action_low_limit=self.action_lows, action_up_limit=self.action_highs)
        obs_vec, state_vec, rew_vec, _ = self.rollout_fn(copy.deepcopy(state), act_seq)  # rollout function returns the rewards and final states
        sk = -rew_vec  # rollout_fn returns a REWARD and we need a COST
        if self.terminal_cost_fn is not None:
            term_states = obs_vec[:, -1, :].reshape(obs_vec.shape[0], obs_vec.shape[-1])
            sk[-1, :] = self.terminal_cost_fn(term_states, act_seq[-1].T)  # replace terminal cost with something else
            # if utils.is_nan_np(sk):
            #     print('NaN rewards', sk)
            #     print('states', st)
            #     print('action seq', act_seq)
            #     print('mean', self.mean_action)
            #     print('covariance', self.cov_action)
            #     input('..')

        if self.rollout_callback is not None: self.rollout_callback(obs_vec, state_vec, self.mean_action, self.cov_action,
                                                                    act_seq)  # for example, visualize rollouts

        return sk, delta

    def _calculate_next_action(self, state, sk, delta):
        """
            Calculate next action based on sampled trajectories and cost fn
        """
        _w = self._exp_util(sk, delta, self.lam)
        self._update_moments(state, delta, _w)  # take gradient step
        next_action = self.mean_action[0]
        next_action = scale_ctrl(next_action, action_low_limit=self.action_lows, action_up_limit=self.action_highs)
        return next_action.reshape(1, self.num_actions)

    def step(self):
        """
            Optimize for best action at current state
        """
        # save state
        state = self.get_state_fn() #get state to plan from (should this be an input?)
        self.set_state_fn(state) #set state of simulation
        # curr_action = np.zeros((self.num_actions, 1))

        for itr in range(self.n_iters):
            # generate random trajectories
            sk, delta = self._generate_rollouts(state)
            # update moments and calculate best action
            curr_action = self._calculate_next_action(state, sk, delta)
        
        if self.update_val:
            self._val = self._calc_val(state, self.lam)
        
        # restore state to original
        self.set_state_fn(state) #(Q: do we need this?)
        self.num_steps += 1
        # shift moments one timestep (dynamic step)
        self._shift()

        return curr_action

    def _update_moments(self, state, delta, w):
        """
           Update moments in the direction of current gradient estimated
           using samples
        """

        self.mean_action = (1.0 - self.step_size) * self.mean_action + self.step_size * np.matmul(delta, w[:, :,
                                                                                                         None]).squeeze(
            axis=-1)

        # self.mean_action = savgol_filter(self.mean_action, len(self.mean_action) - 1, 3, axis=0)
        if self.update_cov:
            print('Updating covariance')
            self.cov_action = (1.0 - self.step_size) * self.cov_action + self.step_size * np.matmul(delta ** 2, w[:, :,
                                                                                                                None]).squeeze(
                axis=-1)
            self.cov_action = np.clip(self.cov_action, self.min_cov, None)  # clip covariance to avoid collapse

        if np.any(np.isnan(self.mean_action)) or np.any(np.isnan(self.cov_action)):
            print('warning: nan in mean_action or cov_action...resetting the controller')
            self.reset()

    def _shift(self):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        self.mean_action[:-1] = self.mean_action[1:]
        self.mean_action[-1] = np.random.normal(0, self.init_cov, self.num_actions)

        if self.update_cov:
            update = (self.cov_action < self.prior_cov)
            cov_shifted = (1.0 - self.beta) * self.cov_action + self.beta * self.prior_cov
            self.cov_action = update * cov_shifted + (1 - update) * self.cov_action

    def _cost_to_go(self, sk):
        """
            Calculate (discounted) cost to go for given reward sequence
        """
        sk = self.gamma_seq * sk  # discounted reward sequence
        sk = np.cumsum(sk[::-1, :], axis=0)[::-1, :]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        sk /= self.gamma_seq  # un-scale it to get true discounted cost to go
        return sk


    def _exp_util(self, sk, delta, lam):
        """
            Calculate weights using exponential utility
        """
        # This is meant to be cost to go. We could comment it out if we want instantaneous cost
        # The cost to go has been used here https://arxiv.org/pdf/1706.09597.pdf and also in the original MPPI paper
        uk = self._control_costs(delta)
        sk = sk + lam * uk
        sk = self._cost_to_go(sk)

        sk -= np.min(sk, axis=-1)[:, None]  # shift the weights
        w = np.exp(-sk / lam)
        w /= np.sum(w, axis=-1)[:, None] + 1e-6  # normalize the weights
        return w

    def _control_costs(self, delta):
        # _ctrl = self.mean_action[:,:,np.newaxis] + delta
        # act_seq = scale_ctrl(_ctrl, action_low_limit=self.action_lows, action_up_limit=self.action_highs)
        # delta = act_seq - self.mean_action[:,:,np.newaxis]
        if self.alpha == 1:
            control_costs = np.zeros((delta.shape[0], delta.shape[-1]))
        else:
            # delta_normalized = delta / self.init_cov
            u_normalized = self.mean_action[:, :, np.newaxis] / self.init_cov
            control_costs = 0.5 * u_normalized * (self.mean_action[:, :, np.newaxis] + 2.0 * delta)
            control_costs = np.sum(control_costs, axis=1)
        return control_costs


    def _calc_val(self, state, lam):
        """
            Calculate (soft) value or free energy of state under current
            control distribution
        """
        sk, delta = self._generate_rollouts(state)
        uk = self._control_costs(delta)
        sk = sk + lam * uk
        sk = self._cost_to_go(sk)
        sk = -sk / lam
        sk = np.array(sk[0, :])

        # calculate log-sum-exp
        skmax = np.max(sk)
        sk -= skmax
        sk = np.exp(sk)
        val = skmax + np.log(np.sum(sk)) - np.log(sk.shape[0])
        val = -lam * val
        return val

    def _action_prob(self, x, mean, cov):
        """
        Return Gaussian probability density value of x

        """
        return scipy.stats.norm.pdf(x, mean, np.sqrt(cov))

    @property
    def state_val(self):
        return self._val

    def reset(self):
        self.num_steps = 0
        self.mean_action = np.zeros(shape=(self.horizon, self.num_actions))
        self.cov_action = self.init_cov * np.ones(shape=(self.horizon, self.num_actions))
        self.gamma_seq = np.cumprod([1.0] + [self.gamma] * (self.horizon - 1)).reshape(self.horizon, 1)
        self._val = 0.0

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.reset()
