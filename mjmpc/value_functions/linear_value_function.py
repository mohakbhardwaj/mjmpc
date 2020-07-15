import torch
import torch.nn as nn


class LinearValueFunction(nn.Module):
    def __init__(self, d_obs):
        super(LinearValueFunction, self).__init__()
        self.d_obs = d_obs
        self.linear = nn.Linear(d_obs, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
    
    def forward(self, observation):
        value = self.linear(observation)
        return value
    
    def fit(self, observations, returns, delta_reg=0., return_errors=False):
        observations = torch.clamp(observations, -10.0, 10.0) / 10.0
        #append 1 to columns for bias
        new_col = torch.ones(observations.shape[0],1)
        obs_1 = torch.cat((observations, new_col), axis=-1)
        print('reg', delta_reg)
        
        if return_errors:
            predictions = self(observations)
            errors = returns - predictions
            error_before = torch.sum(errors**2)/torch.sum(returns**2)

        for _ in range(10):
            # coeffs = torch.lstsq(
            #     observations.T.mm(observations) + delta_reg * torch.eye(observations.shape[1]),
            #     observations.T.mv(returns).unsqueeze(-1)
            # )[0]
            coeffs = torch.lstsq(
                obs_1.T.mv(returns),
                obs_1.T.mm(obs_1) + delta_reg * torch.eye(obs_1.shape[1])
            )[0]
            if not torch.any(torch.isnan(coeffs)):
                break
            print('Got a nan')
            delta_reg *= 10
        self.linear.weight.data = coeffs[0:-1].T
        self.linear.bias.data = coeffs[-1] 

        if return_errors:
            predictions = self(observations)
            errors = returns - predictions
            error_after = torch.sum(errors**2)/torch.sum(returns**2)
            return error_before, error_after





    
    
