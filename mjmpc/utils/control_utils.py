import numpy as np

def scale_ctrl(ctrl, action_low_limit, action_up_limit, squash_fn='clip'):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_up_limit - action_low_limit) / 2.0
    act_mid_range = (action_up_limit + action_low_limit) / 2.0
    if squash_fn == 'clip':
        ctrl = np.clip(ctrl, -1.0, 1.0)
    elif squash_fn == 'tanh':
        ctrl = np.tanh(ctrl)
    return act_mid_range[np.newaxis, :] + ctrl * act_half_range[np.newaxis, :]

# def generate_noise(std_dev, filter_coeffs, base_act):
#     """
#         Generate noisy samples using autoregressive process
#     """
#     beta_0, beta_1, beta_2 = filter_coeffs
#     eps = np.random.normal(loc=0.0, scale=1.0, size=base_act.shape) * std_dev
#     for i in range(2, eps.shape[0]):
#         eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
#     return base_act + eps 

def generate_noise(cov, filter_coeffs, shape, base_seed):
    """
        Generate correlated noisy samples using autoregressive process
    """
    np.random.seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    for i in range(2, eps.shape[1]):
        eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
    return eps 


def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    if np.any(gamma_seq == 0):
        return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted reward sequence
    cost_seq = np.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


####################
## Gaussian Utils ##
####################

def gaussian_logprob(mean, cov, x, cov_type="full"):
    """
    Calculate gaussian log prob for given input batch x
    Parameters
    ----------
    mean (np.ndarray): [N x num_samples] batch of means
    cov (np.ndarray): [N x N] covariance matrix
    x  (np.ndarray): [N x num_samples] batch of sample values

    Returns
    --------
    log_prob (np.ndarray): [num_sampls] log probability of each sample
    """
    N = cov.shape[0]
    if cov_type == "diagonal":
        cov_diag = cov.diagonal()
        cov_inv = np.diag(1.0 / cov_diag)
        cov_logdet = np.sum(np.log(cov_diag))
    else:
        cov_logdet = np.log(np.linalg.det(cov))
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    mahalanobis_dist = -0.5 * np.sum((diff @ cov_inv) * diff, axis=1)
    const1 = -0.5 * N * np.log(2.0 * np.pi)    
    const2 = -0.5*cov_logdet
    log_prob = mahalanobis_dist + const1 + const2
    return log_prob

def gaussian_logprobgrad(mean, cov, x, cov_type="full"):
    if cov_type == "diagonal":
        cov_inv = np.diag(1.0/cov.diagonal())
    else:
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    grad = diff @ cov_inv
    return grad

def gaussian_entropy(cov, cov_type="full"):
    """
    Entropy of multivariate gaussian given covariance
    """
    N = cov.shape[0]
    if cov_type == "diagonal":
        cov_logdet =  np.sum(np.log(cov.diagonal())) 
    else:
        cov_logdet = np.log(np.linalg.det(cov))
    term1 = 0.5 * cov_logdet
    term2 = 0.5 * N * (1.0 + np.log(2.0 * np.pi))
    return term1 + term2

def gaussian_kl(mean0, cov0, mean1, cov1, cov_type="full"):
    """
    KL-divergence between Gaussians given mean and covariance
    KL(p||q) = E_{p}[log(p) - log(q)]

    """
    N = cov0.shape[0]
    if cov_type == "diagonal":
        cov1_diag = cov1.diagonal()
        cov1_inv = np.diag(1.0 / cov1_diag)
        cov0_logdet = np.sum(np.log(cov0.diagonal()))
        cov1_logdet = np.sum(np.log(cov1_diag))
    else:
        cov1_inv = np.linalg.inv(cov1)
        cov0_logdet = np.log(np.linalg.det(cov0))
        cov1_logdet = np.log(np.linalg.det(cov1))

    term1 = 0.5 * np.trace(cov1_inv @ cov0)
    diff = (mean1 - mean0).T
    mahalanobis_dist = 0.5 * np.sum((diff @ cov1_inv) * diff, axis=1)
    term3 = 0.5 * (-1.0*N + cov1_logdet - cov0_logdet)
    return term1 + mahalanobis_dist + term3