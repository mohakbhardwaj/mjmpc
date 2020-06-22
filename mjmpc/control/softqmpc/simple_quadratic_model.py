"""
A Quadratic Q-Function
Q(s, a) = [s, a]^T P [s, a] + [s, a]^T R + c
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
        self.d_J = self.d_total
        self.d_out = self.d_L + self.d_J + 1

        L = torch.zeros(self.d_L)
        J = torch.zeros(self.d_J)
        # c = torch.zeros(1,1)
        torch.nn.init.normal_(L)
        torch.nn.init.normal_(J)
        self.L = nn.Parameter(L)
        self.J = nn.Parameter(J)
        # self.c = nn.Parameter(c)

    def forward(self, states, actions):
        """
        Parameters
        ----------
        states: Tensor (batch_size x d_state)
        actions: Tensor (batch_size x d_action)

        Returns
        -------
        out: Tensor (batch_size x 1)
            Q estimates
        """
        P = self.P
        inps = torch.cat((states, actions), axis=-1)
        inps = inps.unsqueeze(1)
        
        quad_term = -0.5 * F.bilinear(inps, inps, P.unsqueeze(0)).squeeze(-1)
        # quad_term = 0.5 * F.bilinear(inps, inps, P.unsqueeze(0)).squeeze(-1)
        lin_term = F.linear(inps, self.J)
        out = quad_term + lin_term #+ self.c
        return out

    def loss(self, states, actions, targets):
        """
        Parameters
        ----------
        states: Tensor (batch_size x d_state)
        actions: Tensor (batch_size x d_action)
        targets: Tensor (batch_size x 1)

        Returns
        -------
        loss: Tensor (1 x 1)
            Q-estimates
        """
        out = self(states, actions)
        # print((out - targets) ** 2)
        loss = 0.5 * F.mse_loss(out, targets, reduction='mean')
        return loss

    @property
    def P(self):
        Lmat = torch.zeros(self.d_total, self.d_total)
        tril_indices = torch.tril_indices(row=self.d_total, col=self.d_total, offset=0)
        Lmat[tril_indices[0], tril_indices[1]] = self.L
        P = torch.mm(Lmat, Lmat.t())
        return P
    
    def get_act_mean_sigma(self, state):
        """
        Return conditional mean and covariance of 
        actions given state
        In Natural Parameterization we have,
            J = [Js, Ja], P = [Pss, Pas^T; Pas, Paa]
            p(a | s) = Normal(Ja - Pas*state, Paa)
        which gives Moment Parameterization
            Sigma = Paa^-1
            mu = Sigma * (Ja - Psa * state)
        
        Parameters
        ----------
        state: Tensor (1 x self.d_state)

        Returns
        -------
        Parameters of conditional distribution over actions
        given state
        mu: Tensor (1 x self.d_act)
            Mean
        Sigma: Tensor (self.d_act x self.d_act)
            Covariance
        """
        P = self.P.data
        J = self.J.data
        
        Pas = P[-self.d_act:, 0:self.d_state]
        Paa = P[-self.d_act:, -self.d_act:]
        Paa_inv = torch.cholesky_inverse(Paa)
        Sigma = Paa_inv
        # print(Sigma)

        # Sigma += 0.5 * torch.eye(Sigma.shape[0])
        Ja = J[-self.d_act:]
        Jout = Ja - torch.mv(Pas, state)
        print(Ja)

        # Jout = Ja + torch.mv(Pas, state)
        mu = torch.mv(Sigma, Jout)
        return mu, Sigma

    def grow_cov(self, beta):
        # print(self.L)
        P = self.P
        Sigma = torch.cholesky_inverse(P)
        Sigma += beta * torch.eye(Sigma.shape[0])
        Pnew = torch.cholesky_inverse(Sigma)
        Lmat = torch.cholesky(Pnew)
        tril_indices = torch.tril_indices(row=self.d_total, col=self.d_total, offset=0)
        self.L.data = Lmat[tril_indices[0], tril_indices[1]]
        # print(self.L)
        # input('....')



    def __str__(self):
        P = self.P
        string = "L = {0}\nP = {1}\nJ = {2}".format(
                                                    self.L.data,
                                                    P.data,
                                                    self.J.data)
                                                    # \nc = {3} self.c.data)
        return string

    def print_gradients(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print("Layer = {}, grad norm = {}".format(name, param.grad.data.norm(2).item()))

    def get_gradient_dict(self):
        grad_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_dict[name] = param.grad.data.norm(2).item()
        return grad_dict

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)


if __name__ == "__main__":
    import torch.optim as optim
    #from mpl_toolkits.mplot3d import axes3d
    #import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    test_regression = True
    test_mean_sigma = False

    if test_mean_sigma:
        d_state = 3
        d_action = 3
        Q = SimpleQuadraticQFunc(d_state, d_action)
        state = torch.ones(d_state)

    if test_regression:
        test_case = 0
        torch.manual_seed(0)
        d_state = 1
        d_action = 1
        if test_case == 0: 
            Ptrue = torch.eye(d_state + d_action)
        elif test_case == 1:
            Ptrue = torch.rand(d_state + d_action, d_state + d_action)
        Jtrue = torch.ones(d_state + d_action)
        ctrue = 0.0
        
        def get_targets(states, actions):
            inps = torch.cat((states, actions), axis=-1)
            inps = inps.unsqueeze(1)

            targets = -0.5 * F.bilinear(inps, inps, Ptrue.unsqueeze(0)).squeeze(-1) \
                    + F.linear(inps, Jtrue) + ctrue
            return targets

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
        states = torch.rand(num_pts, d_state)
        actions = torch.rand(num_pts, d_action)
        targets = get_targets(states, actions)

        #Fit a quadratic Q function
        Q = SimpleQuadraticQFunc(d_state, d_action)
        optimizer = optim.SGD(Q.parameters(), lr=1.0, weight_decay=0.00001)
        init_loss = Q.loss(states, actions, targets)
        for i in range(10000): 
            optimizer.zero_grad()
            loss = Q.loss(states, actions, targets)
            loss.backward()
            optimizer.step()
            # print('loss = {0}'.format(loss.item()))
        final_loss = Q.loss(states, actions, targets)
        print("Initial Loss", init_loss)
        print("Final Loss", final_loss)

        print('True parameters')
        print("P = ", Ptrue)
        print("J = ", Jtrue)
        print("c = ", ctrue)

        print('Optimized parameters')
        print(Q)

        print("Eval")
        print(targets)
        print(Q(states, actions))

        print(Q.get_act_mean_sigma(states[0]))