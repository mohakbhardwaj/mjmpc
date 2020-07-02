#!/usr/bin/env python
"""
Soft Q-learning based MPC
Author - Mohak Bhardwaj
Date: 7 June, 2020
"""
from mjmpc.control.controller import Controller
from mjmpc.utils.control_utils import generate_noise, cost_to_go
from copy import deepcopy
import numpy as np
import scipy.special
import scipy.optimize


class SoftQMPC(Controller):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_lows,
                 action_highs,
                 horizon,
                 max_iters,
                 gamma,
                 init_cov,
                 lam,
                 lr,
                 reg,
                 num_grad_steps,
                 tol,
                 num_samples,
                 beta,
                 th_init=None,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):
        """
        Parameters
        ----------
        state_dim : int
            size of state space
        action_dim : int
            size of action space
        action_lows : np.ndarray 
            lower limits for each action dim
        action_highs : np.ndarray  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        max_iters : int  
            max number of optimization iterations/rollouts
        gamma : float
            discount factor
        init_cov : float
            initial covariance for sampling actions
        lam : float
            temperature
        lr : float, [0,1]
            learning rate for sarsa
        reg : float [0,1]
            regularization coeff
        num_grad_steps : int
            number of gradient steps in backward pass
        tol : float
            tolerance for convergence of sarsa
        num_samples : int
            number of action samples from prior Gaussian
        th_init : None, optional
            initial parameters for Q-function
        set_sim_state_fn : function  
            set state of simulator using input
        get_sim_state_fn : function  
            get current state of simulator
        sim_step_fn : function  
            step the simulatio using action
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        batch_size : int
            optimize for a batch of states
        seed : int  
            seed value
        """
        super(SoftQMPC, self).__init__(                 
                                       state_dim,
                                       action_dim,
                                       action_lows,
                                       action_highs,
                                       horizon,
                                       max_iters,
                                       gamma,
                                       set_sim_state_fn,
                                       get_sim_state_fn,
                                       sim_step_fn,
                                       sample_mode,
                                       batch_size,
                                       seed)

        self.init_cov = np.array([init_cov] * self.action_dim)
        self.lam = lam
        self.lr = lr
        self.reg = reg
        self.num_grad_steps = num_grad_steps
        self.tol = tol
        self.num_samples = num_samples
        if th_init is None:
            self.th_init = np.zeros((self.state_dim + self.action_dim))
        else:
            self.th_init = th_init
        self.th = self.th_init.copy()
        self.bias = 0.0
        # self.old_th = self.th.copy()
        # self.old_bias = self.bias
        self.beta = beta
        self.cov_action = np.diag(self.init_cov)
        self.num_steps = 0

    def _get_next_action(self, ftrs, mode='mean'):
        """
        Get action to execute on the system based
        on current control distribution
        Parameters
        ----------
        mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        """        
        #sample actions from prior gaussian
        # ftrs = self.get_sim_state_fn()
        # delta =  generate_noise(self.cov_action,[1.0, 0.0, 0.0],
        #                         shape=(self.num_samples,1), 
        #                         base_seed = self.seed + self.num_steps)
        
        delta = self.np_random.multivariate_normal(mean=np.zeros((self.action_dim,)), cov = self.cov_action, size=(self.num_samples,1))
        delta = delta.reshape(self.num_samples,self.action_dim)

        #evaluate q value
        ftrs_batch = np.repeat(ftrs[np.newaxis,...], self.num_samples, axis=0)
        inp = np.concatenate((ftrs_batch, delta), axis=-1)

        qvals = inp.dot(self.th) # + self.bias
        #update mean + cov of gaussian using I.S
        new_mean, new_cov = self._update_distribution(qvals, delta)
        # print('new_mean={0}, new_cov={1}'.format(new_mean, new_cov))

        if mode == 'mean':
            action = new_mean.copy()
        elif mode == 'sample':
            delta = self.np_random.multivariate_normal(mean=np.zeros((self.action_dim,)), cov = new_cov, size=(1,1))
            action = new_mean.copy() +  delta.reshape(self.action_dim).copy()
            # print('mean={0}, delta={1}, action={2}'.format(new_mean, delta, action))
            # input('...')

        return action.copy(), new_mean.copy(), new_cov.copy()

    def _shift(self):
        """
        Shift the current control distribution
        to hotstart the next timestep
        """
        pass


    def _calc_val(self, cost_seq, act_seq):
        """
        Calculate value of state given 
        rollouts from some policy
        """
        return 0.0
    
    def forward_pass(self, state):
        ftrs_seq = np.zeros((self.state_dim, self.horizon))
        cost_seq = np.zeros((self.horizon,))
        kl_seq = np.zeros((self.horizon,))
        act_seq = np.zeros((self.action_dim, self.horizon))

        
        self.set_sim_state_fn(deepcopy(state)) #set simulator to initial state 
        curr_ftrs = deepcopy(self.get_sim_state_fn())
        for t in range(self.horizon):
            # curr_ftrs = self.get_sim_state_fn()
            # print('got obs', curr_ftrs)
            # input('...')
            action, mean, cov = self._get_next_action(curr_ftrs.copy(), mode='sample')
            curr_ftrs, cost = self.sim_step_fn(action)
            #TODO: Add KL divergence term too
            kl = self.kl_mvn(mean, cov, np.zeros(mean.shape), self.cov_action)
            ftrs_seq[:,t] = curr_ftrs.copy()
            cost_seq[t] = cost
            kl_seq[t] = kl
            act_seq[:, t] = action.copy()
        return ftrs_seq, act_seq, cost_seq, kl_seq

    def backward_pass(self, ftrs, actions, costs, kl_seq):
        th_old = self.th.copy()
        inps = np.concatenate((ftrs, actions), axis=0)
        n_samples = inps.shape[1]

        # #Generate targets TD(N)
        costs[-1] = self.th.dot(inps[:,-1]) #+ self.bias #last cost is prediction
        qc = cost_to_go(costs, self.gamma_seq).flatten()
        qkl = cost_to_go(kl_seq, self.gamma_seq).flatten() - kl_seq.flatten() 
        q_targets = qc + self.lam * qkl
        q_targets = q_targets[:-1]
        err_init = self.th.dot(inps[:,:-1]) - q_targets


        A = inps[:,:-1].T
        AtA = A.T.dot(A)
        Aty = A.T.dot(q_targets)
        AtAinv = np.linalg.inv(AtA + self.reg * np.identity(self.state_dim + self.action_dim))

        self.th = AtAinv.dot(Aty)
        err_final = self.th.dot(inps[:,:-1]) - q_targets

        # print(np.linalg.norm(err_init), np.linalg.norm(err_final))
        # input('...')
        # print('old th', th_old)
        # print('new th', self.th)
        # input('...')
        # A = inps[:,:-1]
        # AtA = A.T.dot(A)
        # z = AtA + self.reg * np.identity(n_samples-1)
        # print(A.shape)
        # Aty = A.T.dot(q_targets)
        # print(A.shape, AtA.shape, z.shape, Aty.shape)
        # res = np.linalg.lstsq(z, Aty)
        # self.th = rse[0].copy()



        # # res = scipy.optimize.lsq_linear(inps[:,:-1].T, q_targets)
        # res = np.linalg.lstsq(inps[:,:-1].T, q_targets, rcond=None)
        # # print(res)
        # # input('...')
        # self.th = res[0].copy()
        # delta_theta = np.linalg.norm(self.th - th_old)
        # delta_loss = np.abs(res[1] - np.inf)

        
        

        # prediction
        # q_hat = self.th.dot(inps[:,:-1]) + self.bias
        
        # #Generate targets TD(0)
        # q_next = self.th.dot(inps[:, 1:]) + self.bias
        # q_targets = costs[:-1] + self.gamma * q_next 
        # naive_q_target = cost_to_go(costs, self.gamma_seq).flatten()[:-1]
        
        #Generate targets TD(N)
        # costs[-1] = self.th.dot(inps[:,-1]) #+ self.bias #last cost is prediction
        # q_targets = cost_to_go(costs, self.gamma_seq).flatten()
        # q_targets = q_targets[:-1]
        
        # loss_init = np.average((q_hat - q_targets) ** 2)
        # th_new = self.th.copy()
        # bias_new = self.bias
        # for n in range(self.num_grad_steps):
            # q_hat = th_new.dot(inps[:,:-1]) # + bias_new
            #Gradient update
            # err = q_targets.flatten() - q_hat.flatten()
            # dth = (-err.dot(inps[:,0:-1].T)  + self.reg * th_new) / (n_samples * 1.0)
            # db = (-np.sum(err) + self.reg * bias_new) / (n_samples * 1.0)

            # th_new = th_new - self.lr * dth
            # bias_new = bias_new - self.lr * db

        # q_hat = th_new.dot(inps[:,:-1]) #  + bias_new
        # loss_f = np.average((q_hat - q_targets) ** 2)

        # delta_theta = np.sqrt(np.sum((th_new - self.th)**2) + (bias_new - self.bias) **2 )
        # delta_theta = np.linalg.norm(th_new - self.th)
        # delta_loss = np.abs(loss_f - loss_init)
        
        # self.th = th_new.copy()
        # self.bias = bias_new.copy()

        return None, None

    def kl_mvn(self, m0, S0, m1, S1):
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        S1_inv = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term   = np.trace(S1_inv.dot(S0))
        det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) 
        quad_term = diff.T.dot(S1_inv).dot(diff) 

        # print(tr_term,det_term,quad_term)
        return .5 * (tr_term + det_term + quad_term - N) 

    def converged(self, delta_theta, delta_loss):
        """
        Checks convergence
        """
        # delta = np.average(np.abs(self.old_th - self.th))
        # delta = np.linalg.norm(self.old_th - self.th)
        if (delta_theta is not None) and (delta_theta <= self.tol):
            print('delta_theta converged')
            return True
        elif (delta_loss is not None) and (delta_loss <= self.tol):
            print('delta_loss converged')
            return True
        else:
            return False


    def _update_distribution(self, qvals, action_samples):
        """
        Update mean using importance sampling
        """
        w = self._exp_util(qvals, action_samples)       

        # print(action_samples.shape, w.shape)
        # input('...')
        # print(cov_action)
        # input('...')
        # print(action_samples.T.dot(action_samples))
        # cov_2 = (w * action_samples).T.dot(action_samples)
        # print(cov_2, cov_2.shape)
        # input('...')        
        
        weighted_samples = w * action_samples.T
        mean_action = np.sum(weighted_samples.T, axis=0)

        delta = action_samples - mean_action
        weighted_delta = np.sqrt(w) * (delta).T
        weighted_delta = weighted_delta.T.reshape(action_samples.shape)
        cov_action = np.dot(weighted_delta.T, weighted_delta)

        cov_action += self.beta * np.diag(self.init_cov)



        return mean_action, cov_action

    def _exp_util(self, qvals, delta):
        """
            Calculate weights using exponential utility
        """
        # traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        # control_costs = self._control_costs(delta)
        # print(control_costs)
        # input('...')
        # total_costs = qvals + self.lam * control_costs
        # total_costs = qvals #+ self.lam * control_costs

        # #calculate soft-max
        # w1 = np.exp(-(1.0/self.lam) * (total_costs - np.min(total_costs)))
        # w1 /= np.sum(w1) + 1e-6  # normalize the weights
        w = scipy.special.softmax((-1.0/self.lam) * qvals)
        return w



    def _control_costs(self, delta):
        u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
        control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
        control_costs = np.sum(control_costs, axis=-1)
        control_costs = cost_to_go(control_costs, self.gamma_seq)[:,0]

        return control_costs
   
    def reset(self):
        """
        Reset the controller
        """
        self.th = self.th_init.copy()
        self.bias = 0.0
        self.old_th = self.th.copy()
        self.old_bias = self.bias
        self.cov_action = np.diag(self.init_cov)
        self.num_steps = 0

    # def forward_pass(self, state):
    #     ftrs_seq = np.zeros((self.num_rollouts, self.state_dim, self.horizon))
    #     cost_seq = np.zeros((self.num_rollouts, self.horizon,))
    #     act_seq = np.zeros((self.num_rollouts, self.action_dim, self.horizon))
        
    #     for i in range(self.num_rollouts):
    #         self.set_sim_state_fn(deepcopy(state)) #set simulator to initial state 
    #         for t in range(self.horizon):
    #             curr_ftrs = self.get_sim_state_fn()
    #             action = self._get_next_action(curr_ftrs.copy())
    #             obs, cost = self.sim_step_fn(action.copy())
    #             #TODO: Add KL divergence term too

    #             ftrs_seq[i,:,t] = curr_ftrs
    #             cost_seq[i,t] = cost
    #             act_seq[i,:, t] = action
    #     return ftrs_seq, act_seq, cost_seq 

    # def backward_pass(self, ftrs, actions, costs):
    #     th_old = self.th.copy()
    #     inps = np.concatenate((ftrs, actions), axis=1)
    #     n_samples = inps.shape[0] * inps.shape[1]
    #     N = self.state_dim + self.action_dim


    #     # #Generate targets TD(N)
    #     costs[:, -1] = inps[:, :,-1].dot(self.th) #+ self.bias #last cost is prediction
    #     q_targets = cost_to_go(costs, self.gamma_seq) #.flatten()
    #     q_targets = q_targets[:, :-1].flatten()

    #     # print(inps[:,:,:-1])

    #     # print(inps[:,:,:-1].reshape(self.num_rollouts * N, self.horizon-1))
    #     # input('...')
    #     # print(inps[:,:,:-1])
    #     # print()
    #     # input(...)
    #     A = inps[:,:,:-1].transpose(1,0,2).reshape(N, self.num_rollouts*(self.horizon-1))
    #     A = A.T
    #     AtA = A.T.dot(A)
    #     Aty = A.T.dot(q_targets)
    #     AtAinv = np.linalg.inv(AtA + self.reg * np.identity(N))

    #     self.th = AtAinv.dot(Aty).copy()