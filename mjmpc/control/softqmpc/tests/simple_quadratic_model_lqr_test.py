#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from mjmpc.envs import LQREnv


def solve_lqr(A, B, Q, R, num_steps, plot_gain=False):
    P = Q.copy()
    gains = [] #Store all intermediate gains here
    for i in range(num_steps):
        Hd = np.linalg.pinv(R + B.T.dot(P).dot(B))
        J = B.T.dot(P).dot(A)
        K = -Hd.dot(J)
        AbK = A + B.dot(K)
        P = Q + K.T.dot(R).dot(K) + AbK.T.dot(P).dot(AbK)
        gains.append(K.copy())
    Kss = gains[-1] #Final optimized gain

    if plot_gain:
        K_shape = Kss.shape
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Control gain terms with timesteps')
        for i in range(K_shape[0]):
            for j in range(K_shape[1]): 
                ax.plot(np.arange(num_steps), [K[i][j] for K in gains])
        plt.show()
    return Kss, P

dt = 0.001
A = np.array([[1.0]])
B = np.array([[1.0]])
Q = np.array([[1.0]])
R = np.array([[0.1]])

env = LQREnv(A, B, Q, R)
print(env.A, env.B, env.Q, env.R)
num_steps_lqr = 200

Kss, P = solve_lqr(A, B, Q, R, num_steps_lqr)
print(Kss, P)

obs = env.reset(seed=0)
print("Initial state = {0}".format(obs))
states = [obs.copy()]
rewards = []
controls = []
#simulate optimal controller
max_steps = 100
for i in range(max_steps):
    u = Kss.dot(obs)
    print(u)
    obs, rew, done, info = env.step(u)
    states.append(obs.copy())
    rewards.append(rew)
    controls.append(u)
print("Final state = {0}".format(obs))

fig, ax = plt.subplots(2,2)
ax[0,0].plot(range(0, max_steps+1), [s[0] for s in states])
ax[0,1].plot(range(0, max_steps), [r[0] for r in rewards])
ax[1,0].plot(range(0, max_steps), [u[0] for r in controls])

plt.show(block=False)




