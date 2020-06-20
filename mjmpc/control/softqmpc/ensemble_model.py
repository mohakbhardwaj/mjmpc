import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
	return x * torch.sigmoid(x)

def linear(x):
	return x

class EnsembleDenseLayer(nn.Module):
	def __init__(self, n_in, n_out, ensemble_size, non_linearity='leaky_relu'):
		"""
		linear + activation Layer
		there are `ensemble_size` layers
		computation is done using batch matrix multiplication
		hence forward pass through all models in the ensemble can be done in one call

		weights initialized with xavier normal for leaky relu and linear, xavier uniform for swish
		biases are always initialized to zeros

		Args:
		    n_in: size of input vector
		    n_out: size of output vector
		    ensemble_size: number of models in the ensemble
		    non_linearity: 'linear', 'swish' or 'leaky_relu'
		"""

		super().__init__()

		weights = torch.zeros(ensemble_size, n_in, n_out).float()
		biases = torch.zeros(ensemble_size, 1, n_out).float()

		for weight in weights:
			if non_linearity == 'swish':
				nn.init.xavier_uniform_(weight)
			elif non_linearity == 'relu':
				nn.init.kaiming_normal_(weight)
			elif non_linearity == 'leaky_relu':
				nn.init.kaiming_normal_(weight)
			elif non_linearity == 'tanh':
				nn.init.xavier_uniform_(weight)
			elif non_linearity == 'linear':
				nn.init.xavier_normal_(weight)

		self.weights = nn.Parameter(weights)
		self.biases = nn.Parameter(biases)

		if non_linearity == 'swish':
			self.non_linearity = swish
		elif non_linearity == 'relu':
			self.non_linearity = F.relu
		elif non_linearity == 'leaky_relu':
			self.non_linearity = F.leaky_relu
		elif non_linearity == 'tanh':
			self.non_linearity = torch.tanh
		elif non_linearity == 'linear':
			self.non_linearity = linear

	def forward(self, inp):
		op = torch.baddbmm(self.biases, inp, self.weights)
		return self.non_linearity(op)

class EnsembleModel(nn.Module):
	def __init__(self, d_in, d_out, n_hidden, n_layers, ensemble_size, non_linearity='leaky_relu', device=torch.device('cpu')):
		assert n_layers >= 2, "minimum depth of model is 2"

		super().__init__()

		layers = []
		for lyr_idx in range(n_layers + 1):
			if lyr_idx == 0:
				lyr = EnsembleDenseLayer(d_in, n_hidden, ensemble_size, non_linearity=non_linearity)
			elif 0 < lyr_idx < n_layers:
				lyr = EnsembleDenseLayer(n_hidden, n_hidden, ensemble_size, non_linearity=non_linearity)
			elif lyr_idx == n_layers:
				lyr = EnsembleDenseLayer(n_hidden, d_out , ensemble_size, non_linearity='linear')
			
			layers.append(lyr)

		self.layers = nn.Sequential(*layers)

		self.to(device)
		self.d_in = d_in
		self.d_out = d_out
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.ensemble_size = ensemble_size
		self.device = device


    ######################
    ### Saving/Loading ###
    ###################### 
    
	def save(self, save_file_path):
		torch.save(self.state_dict(), save_file_path)

	def load(self, load_file_path):
		self.load_state_dict(torch.load(load_file_path))


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


