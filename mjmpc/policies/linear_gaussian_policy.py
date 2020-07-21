from copy import deepcopy
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn as nn
from gym.utils import seeding

from mjmpc.utils import EnsembleModel

class LinearGaussianPolicy(nn.Module):
    def __init__(self, d_obs, d_action, min_log_std=-3, init_log_std=0, seed=0, device=torch.device('cpu')):
        super().__init__()
        self.d_obs = d_obs
        self.d_action = d_action
        self.min_log_std = min_log_std
        self.init_log_std = init_log_std
        self.device = device
        self.seed(seed)
        
        # Policy Parameters
        self.linear_mean = nn.Linear(self.d_obs, self.d_action, bias=True)
        torch.nn.init.zeros_(self.linear_mean.bias)
        torch.nn.init.zeros_(self.linear_mean.weight)
        # torch.nn.init.xavier_normal_(self.linear_mean.weight)
        # self.linear_mean.weight.data *= 1e-2
        self.log_std = nn.Parameter(torch.ones(self.d_action) * self.init_log_std)
        self.trainable_params = list(self.parameters())

        # Initial Parameters
        self.init_linear_mean_weight = deepcopy(self.linear_mean.weight.data)
        self.init_linear_mean_bias = deepcopy(self.linear_mean.bias.data)
        self.init_log_std = deepcopy(self.log_std.data)

        # # Old Policy Parameters
        # self.old_linear_mean = nn.Linear(self.d_obs, self.d_action)
        # self.old_log_std = nn.Parameter(torch.ones(self.d_action) * init_log_std, requires_grad=False)
        # for idx, p in enumerate(self.old_linear_mean.parameters()): 
        #     p.data = self.trainable_params[idx].data.clone()
        #     p.requires_grad = False
        # self.old_parameters = list(self.old_linear_mean.parameters()) + [self.old_log_std]

        # Easy access variables
        # -------------------------
        # self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        # self.param_shapes = [p.data.numpy().shape for p in self.parameters()]
        # self.param_sizes = [p.data.numpy().size for p in self.parameters()]
        # self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        # self.obs_var = Variable(torch.randn(self.d_obs), requires_grad=False)
    
    def forward(self, observation):
        pass
    
    def get_action(self, observation, mode='sample', white_noise=None):
        mean = self.linear_mean(observation)
        std = self.log_std.exp()
        normal = Normal(mean, std)
        
        if mode == 'mean':
            action = deepcopy(mean)
            # print(std, np.exp(self.min_log_std))
        elif mode == 'sample':
            std_np = std.data.detach() 
            if white_noise is None:
                white_noise = self.np_random.randn(self.d_action) 
            noise_sample = std_np * white_noise
            action = mean + torch.from_numpy(np.float32(noise_sample))
        
        log_prob = normal.log_prob(action).detach().numpy()
        mean_np = mean.data.detach().numpy().ravel()
        log_std_np = self.log_std.data.detach().numpy().ravel() 
        return action, {'mean': mean_np, 'log_std': log_std_np, 'evaluation': mean_np, 'log_prob': log_prob}
    
    def log_prob(self, observations, actions):
        means_new = self.linear_mean(observations)
        std_new = self.log_std.exp()
        prob_new = Normal(means_new, std_new)
        return prob_new.log_prob(actions)

    def action_distribution(self, observations):
        means_new = self.linear_mean(observations)
        std_new = self.log_std.exp()
        prob_new = Normal(means_new, std_new)
        return prob_new

    def reset(self):
        self.log_std.data.copy_(self.init_log_std)
        self.linear_mean.weight.data.copy_(self.init_linear_mean_weight)
        self.linear_mean.bias.data.copy_(self.init_linear_mean_bias)

    def clamp_cov(self):
        self.log_std.data.copy_(torch.clamp(self.log_std.data, min=self.min_log_std))

    def grow_cov(self, beta):
        self.log_std.data.add_(beta)        

    #################
    ### Utilities ###
    #################
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.parameters()])
        return params.copy()
    
    # def set_param_values(self, new_params, set_new=True, set_old=False):
    #     if set_new:
    #         current_idx = 0
    #         for idx, param in enumerate(self.trainable_params):
    #             vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
    #             vals = vals.reshape(self.param_shapes[idx])
    #             param.data = torch.from_numpy(vals).float()
    #             current_idx += self.param_sizes[idx]
            
    #         # clip std at minimum value
    #         self.trainable_params[0].data = \
    #             torch.clamp(self.trainable_params[0], self.min_log_std).data
            # update log_std_val for sampling
            # self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        # if set_old:
        #     current_idx = 0
        #     for idx, param in enumerate(self.old_params):
        #         vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
        #         vals = vals.reshape(self.param_shapes[idx])
        #         param.data = torch.from_numpy(vals).float()
        #         current_idx += self.param_sizes[idx]
        #     # clip std at minimum value
        #     self.old_params[-1].data = \
        #         torch.clamp(self.old_params[-1], self.min_log_std).data

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def print_gradients(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print("Layer = {}, grad norm = {}".format(name, param.grad.data.norm(2).item()))

    def get_gradient_dict(self):
        grad_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_dict[name] = param.grad.data.norm(2).item()
        return grad_dict
    
    #############
    ## Seeding ##
    #############
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    policy = LinearGaussianPolicy(2, 1)
    policy.print_parameters()
    params = policy.get_param_values()


    print('Initial params',params)
    policy.set_param_values(params*10)
    params = policy.get_param_values()
    print('Params*10', params)
    observation = torch.randn(2)
    print('Observation', observation)
    with torch.no_grad():
        action, act_infos = policy.get_action(observation)
        log_prob = policy.log_prob(observation, action)
    

    print('Got action', action)
    print('Log prob', log_prob)
    print('act_infos', act_infos)
    print('mean log_prob', policy.log_prob(observation, torch.FloatTensor(act_infos['mean'])))