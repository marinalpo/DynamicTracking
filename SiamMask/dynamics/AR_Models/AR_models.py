from utils_dynamics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch
np.random.seed(1)


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
dim = 3  # Number of dimensions: dim=2 --> (x, y)
n = 4  # System memory/order (eq. T0)
L = 5  # Length of the stitched sequence
L2 = 3
var_noise = 0.001
debug = False
distance = 0  # ['JBLD', 'JKL']
a1_max = 1
a2_max = 1
t_0_max = 50


# Generate random coefficients and initial condition
a1 = np.random.randint(2*a1_max, size=(1, n)) - a1_max  # Model 1 coefficients
a2 = np.random.randint(2*a2_max, size=(1, n)) - a2_max  # Model 2 coefficients
a2 = 5*np.ones((1, n))



t_0 = np.random.randint(2*t_0_max, size=(dim, n)) - t_0_max # Initial conditions
# t_0 = np.ones((dim, n))  # Initial conditions

savemat('/Users/marinaalonsopoal/Documents/MATLAB/AR_Models/data/a1.mat', {'data': a1})
savemat('/Users/marinaalonsopoal/Documents/MATLAB/AR_Models/data/t_0.mat', {'data': t_0})

# Model the system
t1 = AR_sequence(a1, t_0, L)  # shape: (dim, n+L)
t2 = AR_sequence(a2, t1, L2)   # shape: (dim, n+L+L2)
t3 = AR_sequence(a1, t1, L2)   # shape: (dim, n+L+L2)

print('a1:', a1)
print('t0:', t_0)
print('t1:', t1)

# Generate random noise
n1 = np.random.normal(0, var_noise, (dim, n + L))
n2 = np.random.normal(0, var_noise, (dim, L2))
n3 = np.random.normal(0, var_noise, (dim, L2))

# Create measurements
z1 = t1 + n1
z2 = t2 + np.hstack([n1, n2])
z3 = t3 + np.hstack([n1, n3])

d1_2 = compare_dyn(np.transpose(z1), np.transpose(z2), var_noise, distance)
d1_3 = compare_dyn(np.transpose(z1), np.transpose(z3), var_noise, distance)
print('dhigh:', d1_2)
print('dlow:', d1_3)

if dim == 1:
    plot_AR(n, var_noise, L, L2, z1, z2, z3, t_0, distance, d1_2, d1_3)

if debug:
    print('AR Model with dim =', dim, ' and memory n =', n, '\n')
    print('Model Coefficients:\n', a1, '\n')
    print('Initial Conditions:\n', t_0, '\n')
    print('Sequence:\n', t.astype(int), '\n')
    print('Hankel Matrix:\n', H.numpy().astype(int), '\n')
    print('Gram Matrix with noise variance of ', var_noise, ':\n', H.numpy().astype(int), '\n')
