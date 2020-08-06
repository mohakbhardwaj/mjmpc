import argparse
import numpy as np
import time
import torch
import torch.autograd as autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from mjmpc.utils.control_utils import gaussian_logprob, gaussian_entropy, gaussian_kl, gaussian_logprobgrad
from mjmpc.utils.timer import timeit

parser = argparse.ArgumentParser(description='Test gradient time taken')
parser.add_argument('--test', type=str, help='numpy or torch')
np.random.seed(12345)
ATOL=1e-7
N = 10
num_samples = 100
x = np.random.randn(N, num_samples) #random data

mean1 = np.ones((N,num_samples))
cov1 = np.eye(N)

#gradient test
timeit.start("numpy_grad")
log_prob_grad = gaussian_logprobgrad(mean1, cov1, x)
timeit.stop("numpy_grad")


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
timeit.start("torch_grad")
log_prob_curr.backward(torch.ones(log_prob_curr.shape))
timeit.stop("torch_grad")

print("gradient same as torch? : {}. max absolute diff = {}".format(np.allclose(log_prob_grad, mean_new.grad.numpy().T, atol=ATOL), np.max(np.abs(log_prob_grad - mean_new.grad.numpy().T))))
log_prob_grad = gaussian_logprobgrad(mean1, cov1, x, cov_type="diagonal")
print("gradient same as torch (using diagonal formula)? : {}".format(np.allclose(log_prob_grad, mean_new.grad.numpy().T, atol=ATOL)))
print(timeit)
