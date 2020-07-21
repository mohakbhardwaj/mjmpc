import torch
import torch.nn as nn


class QuadraticTimeVaryingVF(nn.Module):
    def __init__(self, d_obs, horizon):
        super(QuadraticTimeVaryingVF, self).__init__()
        self.d_obs = d_obs
        self.d_input = int(d_obs + (d_obs * (d_obs+1))/2 + 1) #linear + quadratic + time
        self.horizon = horizon
		
        weights = torch.zeros(horizon, self.d_input).float()
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
        features = self.feature_mat(observations)
        values = (features * self.weights.data).sum(-1)
        values += self.biases.data.T
        return values
    
    def feature_mat(self, observations):
        num_paths = observations.shape[0]
        feat_mat = torch.zeros((num_paths, self.horizon, self.d_input))
        
        #linear features
        feat_mat[:,:,0:self.d_obs] = observations
        
        #quadratic features
        k = self.d_obs
        for i in range(self.d_obs):
            for j in range(i, self.d_obs):
                feat_mat[:,:,k] = observations[:,:,i]*observations[:,:,j]  # element-wise product
                k += 1
        return feat_mat

    def fit(self, observations, returns, delta_reg=0., return_errors=False):

        """
        Parameters
        -----------
        observations: torch.Tensor [num_paths x horizon x d_obs]
        returns: torch.Tensor [num_paths x horizon]

        """
        num_paths = observations.shape[0]
        features = self.feature_mat(observations)
        if return_errors:
            predictions = self(observations)
            errors = returns - predictions
            error_before = torch.sum(errors**2)/torch.sum(returns**2)
        
        #make horizon the batch dimension
        feat = features.permute(1,0,2)
        ret = returns.permute(1,0).unsqueeze(-1)

        #append 1 to features to account for bias
        feat = torch.cat((feat, torch.ones(self.horizon, num_paths, 1)), axis=-1) 

        #linear solve to get weights for each timestep in horizon
        I = torch.eye(self.d_input+1).repeat(self.horizon,1,1)
        feat_t = feat.transpose(1,2)
        X, _ = torch.solve(feat_t.bmm(ret), feat_t.bmm(feat) + delta_reg * I)
        X = X.squeeze(-1)

        self.weights.data.copy_(X[:,:-1])
        self.biases.data.copy_(X[:,-1])
        

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

    value_fn = QuadraticTimeVaryingVF(d_obs, horizon)

    obs = torch.randn(num_paths, horizon, d_obs)
    ret = torch.ones(num_paths, horizon)

    print(obs)
    values = value_fn(obs)
    print('Values before', values)


    # err_before, err_after = value_fn.fit(obs, ret, return_errors=True)
    # values = value_fn(obs)
    # print('values after', values)
    # print('err_before', err_before, 'err_after', err_after)

    # print(obs)
    # print(obs.permute(1,0,2))
    # obs = obs.permute(1,0,2)
    # obs = torch.cat((obs, torch.ones(horizon,num_paths,1)), dim=-1)
    # print(obs)

    # print('multiplying')
    # print(obs.transpose(1,2).bmm(obs))
