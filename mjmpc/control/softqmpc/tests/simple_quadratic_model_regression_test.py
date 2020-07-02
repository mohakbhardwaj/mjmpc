from mjmpc.control.softqmpc import SimpleQuadraticQFunc
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys, os
sys.path.insert(0, '../../..')
from utils.control_utils import gaussian_entropy
os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
test_regression = False
test_mean_sigma = False
test_entropy = True

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

        targets = 0.5 * F.bilinear(inps, inps, Ptrue.unsqueeze(0)).squeeze(-1) \
                    + F.linear(inps, Jtrue) + ctrue
        return targets

    #Create some data
    num_pts = 2048
    torch.manual_seed(0)
    states = torch.rand(num_pts, d_state)
    actions = torch.rand(num_pts, d_action)
    targets = get_targets(states, actions)

    #Fit a quadratic Q function
    Q = SimpleQuadraticQFunc(d_state, d_action)
    optimizer = optim.SGD(Q.parameters(), lr=1.0, weight_decay=0.00001)
    reg = 0.00001
    init_loss = Q.loss(states, actions, targets, reg)
    for i in range(10000): 
        optimizer.zero_grad()
        loss = Q.loss(states, actions, targets, reg)
        loss.backward()
        optimizer.step()
        # print('loss = {0}'.format(loss.item()))
    final_loss = Q.loss(states, actions, targets, reg)
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

if test_entropy:
    test_case = 0
    torch.manual_seed(0)
    d_state = 1
    d_action = 1
    lam=1.0

    Ptrue = torch.eye(d_state + d_action, d_state + d_action)
    Jtrue = torch.ones(d_state + d_action)
    Paa = Ptrue[-d_action:, -d_action:]
    Paa_inv = torch.cholesky_inverse(Paa)
    Sigma = lam * Paa_inv    
    ctrue = -gaussian_entropy(Sigma) 
    Q = SimpleQuadraticQFunc(d_state, d_action)

    def get_targets(states, actions):
        inps = torch.cat((states, actions), axis=-1)
        inps = inps.unsqueeze(1)

        targets = 0.5 * F.bilinear(inps, inps, Ptrue.unsqueeze(0)).squeeze(-1) \
                    + F.linear(inps, Jtrue) + ctrue
        
        P = Q.P.data
        Paa = P[-d_action:, -d_action:]
        Paa_inv = torch.cholesky_inverse(Paa)
        Sigma = lam * Paa_inv    
        # targets -= gaussian_entropy(Sigma)   
        # print(gaussian_entropy(Sigma))        
        return targets

    #Create some data
    num_pts = 2048
    torch.manual_seed(0)

    print('True parameters')
    print("P = ", Ptrue)
    print("J = ", Jtrue)
    print("c = ", ctrue)
    optimizer = optim.SGD(Q.parameters(), lr=1.0, weight_decay=0.00001)

    for n in range(10):
        states = torch.rand(num_pts, d_state)
        actions = torch.rand(num_pts, d_action)
        targets = get_targets(states, actions)
        #Fit a quadratic Q function
        reg = 0.00001
        init_loss = Q.loss(states, actions, targets, reg)
        for i in range(100): 
            optimizer.zero_grad()
            loss = Q.loss(states, actions, targets, reg)
            loss.backward()
            if loss.item() < 1e-8:
                break
            optimizer.step()
        final_loss = Q.loss(states, actions, targets, reg)
        print("Initial Loss", init_loss)
        print("Final Loss", final_loss)
        print('Optimized parameters')
        print(Q)

    print("Eval")
    print(targets)
