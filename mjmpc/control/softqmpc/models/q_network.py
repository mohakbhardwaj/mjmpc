import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2




# import numpy as np
# import torch
# import torch.nn.functional as F
# from mpc_rl.models import EnsembleModel

# class QNetwork(EnsembleModel):
#     def __init__(self, d_state, d_action, n_hidden, n_layers, ensemble_size, non_linearity='leaky_relu', device=torch.device('cpu')):
#         """
#         Q function model.
#         predicts value function for a given state and action pair.

#         Args:
#             d_action (int): dimensionality of action
#             d_state (int): dimensionality of state or observation for prediction
#             n_hidden (int): size or width of hidden layers
#             n_layers (int): number of hidden layers (number of non-lineatities). should be >= 2
#             ensemble_size (int): number of models in the ensemble
#             non_linearity (str): 'linear', 'swish' or 'leaky_relu'
#             device (torch.device): device of the model
#         """

#         super(QFunction, self).__init__(d_in=d_action + d_state,
#                                         d_out=1, 
#                                         n_hidden=n_hidden, 
#                                         n_layers=n_layers, 
#                                         ensemble_size=ensemble_size, 
#                                         non_linearity=non_linearity, 
#                                         device=device)

#     def _propagate_network(self, states, actions):
#         inp = torch.cat((states, actions), dim=2)
#         op = self.layers(inp)
#         return op

#     def forward(self, states, actions, lam=1.0):
#         """
#         predict log-sum-exp q function for a batch of states and actions for all models.

#         Parameters
#         ----------
#         states (torch tensor): (batch size, dim_state)
#         actions (torch tensor): (batch size, dim_action)

#         Returns
#         -------
#         value function(torch tensor): (batch size, 1)
#         """
#         states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         #calculate stable log-sum-exp of values
#         # ensemble_pred = self._propagate_network(states, actions)
#         # ensemble_pred *= -(1.0/lam)
#         # pred_max = torch.max(ensemble_pred, dim=0)[0]
#         # ensemble_pred -= pred_max
#         # ensemble_pred = torch.exp(ensemble_pred)
#         # ensemble_pred = torch.sum(ensemble_pred, dim=0)
#         # value1 = pred_max + torch.log(ensemble_pred)
#         # value1 = -lam * value1
#         ensemble_pred = self._propagate_network(states, actions)
#         value = -lam * torch.logsumexp((-1.0/lam) * ensemble_pred, dim=0)
#         return value.transpose(0, 1)

# 		# if (np.argwhere(np.isnan(value.clone().detach().numpy()) > 0).size > 0) or (np.argwhere(np.isinf(value.clone().detach().numpy()) > 0).size > 0):
# 		# 	print('value', value.clone().detach().numpy())
# 		# 	print('states', states.clone().detach().numpy())
# 		# 	print('actions', actions.clone().detach().numpy())
# 		# 	input('...') 


#     def loss(self, states, actions, targets, training_noise_stdev=0):
#         """
#         compute loss given states, actions and value function targets

#         Args:
#         states (torch tensor): (ensemble_size, batch size, dim_state)
#         actions (torch tensor): (ensemble_size, batch size, dim_action)
#         targets (torch tensor): (ensemble_size, batch size, 1)
#         training_noise_stdev (float): noise to add to normalized state, action inputs and state delta outputs

#         Returns:
#         loss (torch 0-dim tensor): `.backward()` can be called on it to compute gradients
#         """
#         if not np.allclose(training_noise_stdev, 0):
#             states  += torch.randn_like(states)  * training_noise_stdev
#             actions += torch.randn_like(actions) * training_noise_stdev
#             targets += torch.randn_like(targets) * training_noise_stdev

#         vals = self._propagate_network(states, actions)      # delta and variance
#         # loss = (vals - targets) ** 2
#         # loss = 0.5 * torch.mean(loss)

#         loss = F.mse_loss(vals, targets, reduction='mean')
#         # if (np.argwhere(np.isnan(loss.clone().detach().numpy()) > 0).size > 0) or (np.argwhere(np.isinf(loss.clone().detach().numpy()) > 0).size > 0):
#         #     print('loss', loss.clone().detach().numpy())
#         #     print('states', states.clone().detach().numpy())
#         #     print('actions', actions.clone().detach().numpy())
#         #     print('targets', targets.clone().detach().numpy())
#         #     input('...')        

#         return loss
    


