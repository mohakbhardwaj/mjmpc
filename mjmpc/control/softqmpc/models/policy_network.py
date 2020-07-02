# from .ensemble_model import EnsembleModel
# import numpy as np
# import torch


# class GaussianPolicy(EnsembleModel):
#     def __init__(self, d_obs, d_action, n_hidden, n_layers, ensemble_size, non_linearity='leaky_relu', device=torch.device('cpu')):
#         super(NNPolicy, self).__init__(d_in=d_obs,
#                                        d_out=d_action, 
#                                        n_hidden=n_hidden, 
#                                        n_layers=n_layers, 
#                                        ensemble_size=ensemble_size, 
#                                        non_linearity=non_linearity, 
#                                        device=device)
    
#     def _propagate_network(self, observations):
#         op = self.layers(observations)
#         return op

#     def forward(self, obs, lam=1.0):
#         """
#         predict mean action for batch of states and actions for all models.

#         Parameters
#         ----------
#         states (torch tensor): (batch size, d_obs)
#         actions (torch tensor): (batch size, d_action)

#         Returns
#         --------
#         mean action (torch tensor): (batch size, 1)
#         """
#         observations = obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         ensemble_pred = self._propagate_network(observations)
#         mean_action = -lam * torch.logsumexp((-1.0/lam) * ensemble_pred, dim=0)
#         return mean_action.transpose(0, 1)

#     def loss(self, observations, targets, training_noise_stdev=0):
#         """
#         TODO: This is not correct right now.
#         compute loss given obs and targets

#         Parameters:
#         obs (torch tensor): (ensemble_size, batch size, d_obs)
#         targets (torch tensor): (ensemble_size, batch size, 1)
#         training_noise_stdev (float): noise to add to normalized state, action inputs and state delta outputs

#         Returns:
#         loss (torch 0-dim tensor): `.backward()` can be called on it to compute gradients
#         """
#         if not np.allclose(training_noise_stdev, 0):
#             observations  += torch.randn_like(observations)  * training_noise_stdev
#             targets += torch.randn_like(targets) * training_noise_stdev

#         means = self._propagate_network(states, actions)
#         loss = F.mse_loss(means, targets, reduction='mean')
#         return loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_highs=None, action_lows=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_highs is None or action_lows is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_highs - action_lows) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_highs + action_lows) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, state, mode="mean"):
        action, log_prob, mean = self.sample(state)
        if mode == 'mean':
            return mean, log_prob
        elif mode == 'sample':
            return action, log_prob
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    #################
    ### Utilities ###
    #################

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

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def serializable_parameter_dict(self):
        serializable_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                serializable_dict[name] = param.tolist()
        return serializable_dict


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_highs=None, action_lows=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_highs is None or action_lows is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_highs - action_lows) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_highs + action_lows) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def get_action(self, state, mode="mean"):
        action, log_prob, mean = self.sample(state)
        if mode == 'mean':
            return mean, log_prob
        elif mode == 'sample':
            return action, log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


