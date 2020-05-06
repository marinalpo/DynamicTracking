from utils_dynamics import *
import numpy as np
import matplotlib.pyplot as plt
import torch
np.random.seed(2)


def plot_AR(n, L, L2, z1, z2, z3, t_0):
    k1 = np.arange(n + L)
    k23 = np.arange(n + L + L2)
    plt.plot(k1, z1.squeeze(), marker='o', zorder=2, label='z1')
    plt.plot(k23, z2.squeeze(), marker='o', zorder=1, label='z2')
    plt.plot(k23, z3.squeeze(), marker='o', zorder=1, label='z3')
    plt.scatter(np.arange(len(t_0.squeeze())), t_0.squeeze(), c='r', zorder=3, label='t_0')
    plt.xlabel('k')
    plt.ylabel('t[k]')
    plt.legend()
    plt.title('AR Model of n=' + str(n))
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
n = 3  # System memory/order (eq. T0)
L = 2  # Length of the stitched sequence
L2 = 5
var_noise = 0.001
debug = False

# Generate random coefficients and initial conditions
a1 = np.random.randint(5, size=(1, n)) - 2.5  # Model 1 coefficients
a2 = np.random.randint(5, size=(1, n)) - 2.5  # Model 2 coefficients
a2 = 2*np.ones((1, n))
t_0 = np.random.randint(10, size=(dim, n)) - 5  # Initial conditions

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

# print('z1', z1.astype(int))
# print('z2', z2.astype(int))
# print('z3:', z3.astype(int))

# Compute Matrices
# H = Hankel(torch.from_numpy(np.transpose(z)))
# G = Gram(H, var_noise)


d1_2 = compare_dyn(np.transpose(z1), np.transpose(z2), var_noise, 0)
d1_3 = compare_dyn(np.transpose(z1), np.transpose(z3), var_noise, 0)
print('dhigh (z2-taronja)', d1_2)
print('dlow (z3-verd)', d1_3)

if dim == 1:
    plot_AR(n, L, L2, z1, z2, z3, t_0)
    print('t0:', t_0)

if debug:
    print('AR Model with dim =', dim, ' and memory n =', n, '\n')
    print('Model Coefficients:\n', a1, '\n')
    print('Initial Conditions:\n', t_0, '\n')
    print('Sequence:\n', t.astype(int), '\n')
    print('Hankel Matrix:\n', H.numpy().astype(int), '\n')
    print('Gram Matrix with noise variance of ', var_noise, ':\n', H.numpy().astype(int), '\n')
