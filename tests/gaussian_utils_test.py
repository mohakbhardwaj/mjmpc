import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from mjmpc.utils.control_utils import gaussian_logprob, gaussian_entropy, gaussian_kl

N = 10
num_samples = 100
x = np.random.randn(N, num_samples) #random data

mean1 = np.ones((N,num_samples))
cov1 = np.eye(N)

#log_prob test
log_prob = gaussian_logprob(mean1, cov1, x)
log_prob_torch = []
for i in range(num_samples):
    torch_dist = MultivariateNormal(torch.FloatTensor(mean1[:,i]), torch.FloatTensor(cov1))
    log_prob_torch.append(torch_dist.log_prob(torch.FloatTensor(x[:,i])).item())

print("log_prob same as torch? : {}".format(np.allclose(log_prob, log_prob_torch)))

#using efficient computation
log_prob = gaussian_logprob(mean1, cov1, x, cov_type="diagonal") 
print("log_prob same as torch (using diagonal formula)? : {}".format(np.allclose(log_prob, log_prob_torch)))

#entropy test
entropy = gaussian_entropy(cov1)
entropy_torch = []
torch_dist = MultivariateNormal(torch.FloatTensor(mean1[:,0]), torch.FloatTensor(cov1))
entropy_torch = torch_dist.entropy()
print("entropy same as torch? : {}".format((entropy - entropy_torch) < 1e-5 ))

entropy = gaussian_entropy(cov1, cov_type="diagonal")
print("entropy same as torch (using diagonal formula)? : {}".format((entropy - entropy_torch) < 1e-5 ))


#kl divergence test
mean2 = 2.0*np.ones((N,num_samples))
cov2 = 2.0 * np.eye(N)

kl_div = gaussian_kl(mean1, cov1, mean2, cov2)
kl_torch = []
for i in range(num_samples):
    torch_dist1 = MultivariateNormal(torch.FloatTensor(mean1[:,i]), torch.FloatTensor(cov1))
    torch_dist2 = MultivariateNormal(torch.FloatTensor(mean2[:,i]), torch.FloatTensor(cov2))
    kl_torch.append(kl_divergence(torch_dist1, torch_dist2).item())

print("kl same as torch? : {}".format(np.allclose(kl_div, kl_torch)))
kl_div = gaussian_kl(mean1, cov1, mean2, cov2, cov_type="diagonal")
print("kl same as torch (using diagonal formula)? : {}".format(np.allclose(kl_div, kl_torch)))



