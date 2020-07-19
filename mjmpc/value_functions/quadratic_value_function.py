import torch
import torch.nn as nn


class QuadraticValueFunction(nn.Module):
    def __init__(self, d_obs):
        super(QuadraticValueFunction, self).__init__()
        self.d_obs = d_obs
        self.d_input = int(d_obs + (d_obs * (d_obs+1))/2 + 1) #linear + quadratic + time
        self.linear = nn.Linear(self.d_input, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
    
    def forward(self, observation, horizon):
        feat_mat = self.feature_mat(observation, horizon)
        value = self.linear(feat_mat)
        return value
    
    def feature_mat(self, obs, horizon):
        num_samples = obs.shape[0]
        feat_mat = torch.zeros(num_samples, self.d_input) #inputs

        #linear features
        feat_mat[:,:self.d_obs] = obs
        
        #quadratic features
        k = self.d_obs
        for i in range(self.d_obs):
            for j in range(i, self.d_obs):
                feat_mat[:,k] = obs[:,i]*obs[:,j]  # element-wise product
                k += 1

        tsteps = torch.arange(1, horizon + 1, out=torch.FloatTensor()) / horizon
        num_paths = int(num_samples / horizon)
        tcol = tsteps.repeat(num_paths).float()
        feat_mat[:,-1] = tcol #torch.cat((feat_mat, tcol), dim=-1)
        return feat_mat

    def fit(self, observations, returns, horizon, delta_reg=0., return_errors=False):
        # observations = torch.clamp(observations, -10.0, 10.0) / 10.0
        feat_mat = self.feature_mat(observations, horizon)
        #append 1 to columns for bias
        new_col = torch.ones(observations.shape[0],1)
        feat_mat = torch.cat((feat_mat, new_col), axis=-1)

        if return_errors:
            predictions = self(observations, horizon)
            errors = returns - predictions.flatten()
            error_before = torch.sum(errors**2)/torch.sum(returns**2)

        for _ in range(10):
            coeffs = torch.lstsq(
                feat_mat.T.mv(returns),
                feat_mat.T.mm(feat_mat) + delta_reg * torch.eye(feat_mat.shape[1])
            )[0]
            if not torch.any(torch.isnan(coeffs)):
                break
            print('Got a nan')
            delta_reg *= 10
        self.linear.weight.data.copy_(coeffs[0:-1].T)
        self.linear.bias.data.copy_(coeffs[-1]) 

        if return_errors:
            predictions = self(observations, horizon)
            errors = returns - predictions.flatten()
            error_after = torch.sum(errors**2)/torch.sum(returns**2)
            return error_before, error_after

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)



    
    
