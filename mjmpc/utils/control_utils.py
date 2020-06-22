import numpy as np

def scale_ctrl(ctrl, action_low_limit, action_up_limit):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_up_limit - action_low_limit) / 2.0
    act_mid_range = (action_up_limit + action_low_limit) / 2.0
    ctrl = np.clip(ctrl, -1.0, 1.0)
    return act_mid_range[np.newaxis, :, np.newaxis] + ctrl * act_half_range[np.newaxis, :, np.newaxis]

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
    cost_seq = gamma_seq * cost_seq  # discounted reward sequence
    cost_seq = np.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq

def gaussian_entropy(cov):
    """
    Calculate entropy of multivariate gaussian given covariance
    """
    N = cov.shape[0]
    det_cov = np.linalg.det(cov)
    term1 = 0.5 * np.log(det_cov)
    term2 = 0.5 * N * (1.0 + np.log(2.0 * np.pi))
    return term1 + term2
