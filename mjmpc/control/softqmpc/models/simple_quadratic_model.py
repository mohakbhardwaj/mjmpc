"""
A Quadratic Q-Function
Q(s, a) = [s, a]^T P [s, a] - [s, a]^T J + c
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
        c = torch.zeros(1,1)
        torch.nn.init.normal_(L)
        torch.nn.init.normal_(J)
        self.L = nn.Parameter(L)
        self.J = nn.Parameter(J)
        self.c = nn.Parameter(c)
        self.eye = torch.eye(self.d_total)

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
        
        quad_term = 0.5 * F.bilinear(inps, inps, P.unsqueeze(0)).squeeze(-1)
        lin_term = F.linear(inps, self.J)
        out = quad_term + lin_term  + self.c
        return out

    def loss(self, states, actions, targets, reg):
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
        loss_term = 0.5 * F.mse_loss(out, targets, reduction='mean')
        # reg_term = reg * torch.norm(self.P - self.eye) 
        loss = loss_term #+ reg_term 
        return loss

    @property
    def P(self):
        Lmat = torch.zeros(self.d_total, self.d_total)
        tril_indices = torch.tril_indices(row=self.d_total, col=self.d_total, offset=0)
        Lmat[tril_indices[0], tril_indices[1]] = self.L
        P = torch.mm(Lmat, Lmat.t())
        return P
    
    def get_act_mean_sigma(self, state, lam):
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
        Sigma = lam * Paa_inv

        Ja = J[-self.d_act:]
        Jout = (-Ja - torch.mv(Pas, state))
        mu = torch.mv(Paa_inv, Jout)
        return mu, Sigma

    def grow_cov(self, beta, lam):
        P = self.P
        Sigma = lam * torch.cholesky_inverse(P)
        Sigma += beta * torch.eye(Sigma.shape[0])
        Pnew = (1./lam) * torch.cholesky_inverse(Sigma)
        Lmat = torch.cholesky(Pnew)
        tril_indices = torch.tril_indices(row=self.d_total, col=self.d_total, offset=0)
        self.L.data = Lmat[tril_indices[0], tril_indices[1]]

    def __str__(self):
        P = self.P
        string = "L = {0}\nP = {1}\nJ = {2}\n c={3}".format(
                                                    self.L.data,
                                                    P.data,
                                                    self.J.data,
                                                    self.c.data)
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


