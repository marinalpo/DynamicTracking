from utils_dynamics import *
import numpy as np
import matplotlib.pyplot as plt
import torch
np.random.seed(25)


def plot_AR(n, var_noise, L, L2, z1, z2, z3, t_0, distance, d2, d3):
    distance_names = ['JBLD', 'JKL']
    dist = distance_names[distance]
    fig, ax = plt.subplots()
    k1 = np.arange(n + L)
    k23 = np.arange(n + L + L2)
    ax.plot(k1, z1.squeeze(), marker='o', c='k', zorder=2, label='Root sequence')
    ax.plot(k23, z2.squeeze(), marker='o', c='orange', zorder=1, label='Different ARM sequence')
    ax.plot(k23, z3.squeeze(), marker='o', c='g', zorder=1, label='Same ARM sequence')
    ax.scatter(np.arange(len(t_0.squeeze())), t_0.squeeze(), c='r', zorder=3, label='Initial Conditions')
    ax.set_xlabel(dist + ' different model: ' + str(np.around(d2, 2)) + '\n' + dist + ' same model: ' + str(np.around(d3, 2)), color='b')
    ax.set_ylabel('t[k]')
    ax.legend()
    ax.set_title('AR Model of n=' + str(n) + ' and noise variance=' + str(var_noise))
    plt.show()


def AR_sequence(a, t_0, l):
    t = t_0
    n = a.shape[1]
    dim = t_0.shape[0]
    for k in range(l):
        t_k = np.zeros([dim, 1])
        for d in range(dim):
            t_dim = t[d, -n:]
            t_k[d, 0] = np.matmul(a, t_dim)
        t = np.hstack([t, t_k])
    return t


# Parameters
dim = 1  # Number of dimensions: dim=2 --> (x, y)
n = 5  # System memory/order (eq. T0)
L = 0  # Length of the stitched sequence
L2 = 2
var_noise = 1
debug = False
distance = 0  # ['JBLD', 'JKL']


# Generate random coefficients and initial conditions
maxim = 10
a1 = np.random.randint(maxim, size=(1, n)) - maxim/2  # Model 1 coefficients
a2 = np.random.randint(maxim, size=(1, n)) - maxim/2  # Model 2 coefficients
# a2 = 5*np.ones((1, n))
print('a1:', a1)
print('a2:', a2)
maxim = 5
t_0 = np.random.randint(maxim, size=(dim, n)) - maxim/2  # Initial conditions

# Model the system
t1 = AR_sequence(a1, t_0, L)  # shape: (dim, n+L)
t2 = AR_sequence(a2, t1, L2)   # shape: (dim, n+L+L2)
t3 = AR_sequence(a1, t1, L2)   # shape: (dim, n+L+L2)

# Generate random noise
noise = np.random.normal(0, var_noise, (dim, n + L))

# Create measurements
z1 = t1 + noise
z2 = t2 + np.hstack([noise, np.random.normal(0, var_noise, (dim, L2))])
z3 = t3 + np.hstack([noise, np.random.normal(0, var_noise, (dim, L2))])

d1_2 = compare_dyn(np.transpose(z1), np.transpose(z2), var_noise, distance)
d1_3 = compare_dyn(np.transpose(z1), np.transpose(z3), var_noise, distance)

if dim == 1:
    plot_AR(n, var_noise, L, L2, z1, z2, z3, t_0, distance, d1_2, d1_3)

if debug:
    print('AR Model with dim =', dim, ' and memory n =', n, '\n')
    print('Model Coefficients:\n', a1, '\n')
    print('Initial Conditions:\n', t_0, '\n')
    print('Sequence:\n', t.astype(int), '\n')
    print('Hankel Matrix:\n', H.numpy().astype(int), '\n')
    print('Gram Matrix with noise variance of ', var_noise, ':\n', H.numpy().astype(int), '\n')
