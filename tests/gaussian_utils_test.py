import numpy as np
import time
import torch
import torch.autograd as autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from mjmpc.utils.control_utils import gaussian_logprob, gaussian_entropy, gaussian_kl, gaussian_logprobgrad
import datetime

np.random.seed(12345)
ATOL=1e-7
N = 10
num_samples = 1000
x = np.random.randn(N, num_samples) #random data

mean1 = np.ones((N,num_samples))
cov1 = np.eye(N)

#log_prob test
log_prob = gaussian_logprob(mean1, cov1, x)
log_prob_torch = []
for i in range(num_samples):
    torch_dist = MultivariateNormal(torch.FloatTensor(mean1[:,i]), torch.FloatTensor(cov1))
    log_prob_torch.append(torch_dist.log_prob(torch.FloatTensor(x[:,i])).item())

print("log_prob same as torch? : {}".format(np.allclose(log_prob, log_prob_torch, atol=ATOL)))

#using efficient computation
log_prob = gaussian_logprob(mean1, cov1, x, cov_type="diagonal") 
print("log_prob same as torch (using diagonal formula)? : {}".format(np.allclose(log_prob, log_prob_torch, atol=ATOL)))

#entropy test
entropy = gaussian_entropy(cov1)
entropy_torch = []
torch_dist = MultivariateNormal(torch.FloatTensor(mean1[:,0]), torch.FloatTensor(cov1))
entropy_torch = torch_dist.entropy()
print("entropy same as torch? : {}".format(np.abs(entropy - entropy_torch) < ATOL))

entropy = gaussian_entropy(cov1, cov_type="diagonal")
print("entropy same as torch (using diagonal formula)? : {}".format(np.abs(entropy - entropy_torch) < ATOL))

#kl divergence test
mean2 = 2.0*np.ones((N,num_samples))
cov2 = 2.0 * np.eye(N)

kl_div = gaussian_kl(mean1, cov1, mean2, cov2)
kl_torch = []
for i in range(num_samples):
    torch_dist1 = MultivariateNormal(torch.FloatTensor(mean1[:,i]), torch.FloatTensor(cov1))
    torch_dist2 = MultivariateNormal(torch.FloatTensor(mean2[:,i]), torch.FloatTensor(cov2))
    kl_torch.append(kl_divergence(torch_dist1, torch_dist2).item())

print("kl same as torch? : {}. max absolute diff = {}".format(np.allclose(kl_div, kl_torch, atol=ATOL), np.max(np.abs(kl_div-kl_torch))))
kl_div = gaussian_kl(mean1, cov1, mean2, cov2, cov_type="diagonal")
print("kl same as torch (using diagonal formula)? : {}".format(np.allclose(kl_div, kl_torch, atol=ATOL)))

#gradient test
d1 = datetime.timedelta()
log_prob_grad = gaussian_logprobgrad(mean1, cov1, x)
print('numpy grad time', d1.microseconds)




x_new = torch.FloatTensor(x)
x_new.requires_grad_(True)
mean_new = torch.FloatTensor(mean1)
cov_new = torch.FloatTensor(cov1)
mean_new.requires_grad_(True)
cov_new.requires_grad_(False)

log_prob_curr = torch.Tensor(num_samples,1)
for i in range(num_samples):
    torch_dist = MultivariateNormal(torch.FloatTensor(mean_new[:,i]), torch.FloatTensor(cov_new))
    log_prob_curr[i] = torch_dist.log_prob(torch.FloatTensor(x_new[:,i]))
d2 = datetime.timedelta()
log_prob_curr.backward(torch.ones(log_prob_curr.shape))
print('torch grad time', d2.microseconds)

print("gradient same as torch? : {}. max absolute diff = {}".format(np.allclose(log_prob_grad, mean_new.grad.numpy().T, atol=ATOL), np.max(np.abs(log_prob_grad - mean_new.grad.numpy().T))))
log_prob_grad = gaussian_logprobgrad(mean1, cov1, x, cov_type="diagonal")
print("gradient same as torch (using diagonal formula)? : {}".format(np.allclose(log_prob_grad, mean_new.grad.numpy().T, atol=ATOL)))







