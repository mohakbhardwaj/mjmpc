import torch
import torch.nn as nn


class LinearTimeVaryingVF(nn.Module):
    def __init__(self, d_obs, horizon):
        super(LinearTimeVaryingVF, self).__init__()
        self.d_obs = d_obs
        self.horizon = horizon
		
        weights = torch.zeros(horizon, d_obs).float()
        biases = torch.zeros(horizon).float()
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, observations):
        """
        Parameters
        ----------
        observations: torch.Tensor [num_paths x horizon x d_obs]
        
        Returns
        --------
        values: torch.Tensor [num_paths x horizon]
        
        """
        values = (observations * self.weights.data).sum(-1)
        values += self.biases.data.T
        return values

    def fit(self, observations, returns, delta_reg=0., return_errors=False):

        """
        Parameters
        -----------
        observations: torch.Tensor [num_paths x horizon x d_obs]
        returns: torch.Tensor [num_paths x horizon]

        """
        num_paths = observations.shape[0]
        
        if return_errors:
            predictions = self(observations)
            errors = returns - predictions
            error_before = torch.sum(errors**2)/torch.sum(returns**2)
        
        #make horizon the batch dimension
        obs = observations.permute(1,0,2)
        obs = torch.cat((obs, torch.ones(self.horizon, num_paths, 1)), axis=-1) 
        ret = returns.permute(1,0).unsqueeze(-1)

        #linear solve to get weights for each timestep in horizon
        I = torch.eye(self.d_obs+1).repeat(self.horizon,1,1)
        obs_t = obs.transpose(1,2)
        X, _ = torch.solve(obs_t.bmm(ret), obs_t.bmm(obs) + delta_reg * I)
        X = X.squeeze(-1)

        self.weights.data.copy_(X[:,:-1])
        self.biases.data.copy_(X[:,-1])



        # for h in range(self.horizon):
        #     obs = observations[:,h,:]
        #     bias_col = torch.ones(num_paths)
        #     obs_new = torch.cat()
        
        #     for _ in range(10): 
            
        #         coeffs = torch.lstsq(
        #             feat_mat.T.mv(returns),
        #             feat_mat.T.mm(feat_mat) + delta_reg * torch.eye(feat_mat.shape[1])
        #             )[0]
        #         if not torch.any(torch.isnan(coeffs)):
        #             break
        #         print('Got a nan')
        #         delta_reg *= 10

            # self.weight.data.copy_(coeffs[0:-1].T)
            # self.linear.bias.data.copy_(coeffs[-1]) 

        if return_errors:
            predictions = self(observations)
            errors = returns - predictions
            error_after = torch.sum(errors**2)/torch.sum(returns**2)
            return error_before, error_after


    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)



    
if __name__ == "__main__":
    horizon = 2
    d_obs = 4
    num_paths = 3

    value_fn = LTVValueFunction(d_obs, horizon)

    obs = torch.randn(num_paths, horizon, d_obs)
    ret = torch.ones(num_paths, horizon)

    print(obs)
    values = value_fn(obs)
    print('Values before', values)


    err_before, err_after = value_fn.fit(obs, ret, return_errors=True)
    values = value_fn(obs)
    print('values after', values)
    print('err_before', err_before, 'err_after', err_after)

    # print(obs)
    # print(obs.permute(1,0,2))
    # obs = obs.permute(1,0,2)
    # obs = torch.cat((obs, torch.ones(horizon,num_paths,1)), dim=-1)
    # print(obs)

    # print('multiplying')
    # print(obs.transpose(1,2).bmm(obs))
