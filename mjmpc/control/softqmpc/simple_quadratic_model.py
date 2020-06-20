"""
Q(x, u) = [x, u]^T P [x, u] + [x, u]^T R + c

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQuadraticQFunc(nn.Module):
    def __init__(self, d_state, d_act):

        super().__init__()

        self.d_state = d_state
        self.d_act = d_act
        self.d_total = d_state + d_act
        self.d_L = int(self.d_total * (self.d_total + 1) / 2)
        self.d_R = self.d_total
        self.d_out = self.d_L + self.d_R + 1

        L = torch.zeros(self.d_L)
        R = torch.zeros(self.d_R)
        c = torch.zeros(1,1)
        torch.nn.init.normal_(L)
        torch.nn.init.normal_(R)
        self.L = nn.Parameter(L)
        self.R = nn.Parameter(R)
        self.c = nn.Parameter(c)

    def forward(self, states, actions):
        """
        Parameters
        ----------
        states: np.ndarray (batch_size x d_state)
        actions: np.ndarray (batch_size x d_action)
        """

        P = self.get_P_matrix()
        inps = torch.cat((states, actions))
        inps_T = inps.t().unsqueeze(1)
        quad_term = -0.5 * F.bilinear(inps_T, inps_T, P.unsqueeze(0)).squeeze(-1)
        lin_term = F.linear(inps_T, self.R)
        out = quad_term + lin_term + self.c
        return out

    def loss(self, states, actions, targets):
        out = self(states, actions)
        loss = 0.5 * F.mse_loss(out, targets, reduction='mean')
        return loss

    def get_P_matrix(self):
        Lmat = torch.zeros(self.d_total, self.d_total)
        tril_indices = torch.tril_indices(row=self.d_total, col=self.d_total, offset=0)
        Lmat[tril_indices[0], tril_indices[1]] = self.L
        P = torch.mm(Lmat, Lmat.t())
        return P
    
    def get_matrix_repr(self):
        P = self.get_P_matrix()
        return P, self.R, self.c

 


if __name__ == "__main__":
    import torch.optim as optim
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    torch.manual_seed(0)
    d_state = 1
    d_action = 1
    Ptrue = torch.eye(d_state + d_action)
    Rtrue = torch.ones(d_state + d_action)
    ctrue = 0.0
    print('True parameters')
    print("P = ", Ptrue)
    print("R = ", Rtrue)
    print("c = ", ctrue)
    
    def get_targets(states, actions):
        inps = torch.cat((states, actions))
        #Targets 
        targets = -0.5 * F.bilinear(inps.t().unsqueeze(1), inps.t().unsqueeze(1), Ptrue.unsqueeze(0)).squeeze(-1) \
                + F.linear(inps.t().unsqueeze(1), Rtrue) + ctrue
        return targets

    qfunc = SimpleQuadraticQFunc(d_state, d_action)
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(-5,5,0.1)
    # y = np.arange(-5,5,0.1)
    # X,Y = np.meshgrid(x,y)
    # X = torch.from_numpy(np.float32(X))
    # Y = torch.from_numpy(np.float32(Y))
    # print(X.flatten().shape, Y.flatten().shape)
    # ax.plot_surface(X.numpy(), Y.numpy(), get_targets(X.flatten().unsqueeze(0),Y.flatten().unsqueeze(0)))
    # plt.show()

    #Create some data
    num_pts = 2048
    torch.manual_seed(0)
    states = torch.rand(1, num_pts)
    actions = torch.rand(1, num_pts)
    targets = get_targets(states, actions)
    #Fit a quadratic Q function
    Q = SimpleQuadraticQFunc(d_state, d_action)
    optimizer = optim.SGD(Q.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    for i in range(1000): 
        optimizer.zero_grad()
        loss = Q.loss(states, actions, targets)
        loss.backward()
        optimizer.step()
        # print('loss = {0}'.format(loss.item()))
    P, R, c = Q.get_matrix_repr()
    print('Optimized parameters')
    print("P = ", P)
    print("R = ", R)
    print("c = ", c)
