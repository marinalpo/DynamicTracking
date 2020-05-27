from utils_dynamics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
# np.random.seed(1)

# Parameters
dim = 2  # Number of dimensions: dim=2 --> (x, y)
n = 4  # System memory/order (eq. T0)
L = 3  # Length of the stitched sequence
L2 = 5
distance = 0  # ['JBLD', 'JKL']
trials_per_var = 100
debug = False

vars = np.array([1e2, 1e1, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0])
print('vars:', vars)

# Generate random coefficients and initial conditions
maxim = 10
t_0 = np.random.randint(maxim, size=(dim, n)) - maxim / 2  # Initial conditions
maxim = 50
a1 = np.random.randint(maxim, size=(1, n)) - maxim / 2  # Model 1 coefficients
a2 = np.random.randint(maxim, size=(1, n)) - maxim / 2  # Model 2 coefficients

# Model the system
t1 = AR_sequence(a1, t_0, L)  # shape: (dim, n+L)
t2 = AR_sequence(a2, t1, L2)  # shape: (dim, n+L+L2)
t3 = AR_sequence(a1, t1, L2)  # shape: (dim, n+L+L2)

dh = np.zeros(len(vars))
dl = np.zeros(len(vars))
for v, var in enumerate(vars):
    print('Var:', var)
    dh_s = np.zeros(trials_per_var)
    dl_s = np.zeros(trials_per_var)
    for t in range(trials_per_var):
        # Generate random noise
        noise = np.random.normal(0, var, (dim, n + L))

        # Create measurements
        z1 = t1 + noise
        z2 = t2 + np.hstack([noise, np.random.normal(0, var, (dim, L2))])
        z3 = t3 + np.hstack([noise, np.random.normal(0, var, (dim, L2))])

        dh_s[t] = compare_dyn(np.transpose(z1), np.transpose(z2), var, distance)
        dl_s[t] = compare_dyn(np.transpose(z1), np.transpose(z3), var, distance)
    # dh[v] = stats.mode(dh_s)[0]
    # dl[v] = stats.mode(dl_s)[0]
    dh[v] = np.mean(dh_s)
    dl[v] = np.mean(dl_s)


fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title('Mean distances')
ax1.plot(np.arange(len(vars)), dh, label='dhigh')
ax1.plot(np.arange(len(vars)), dl, linestyle=':', label='dlow')
ax1.legend()
ax2.plot(np.arange(len(vars)), dh-dl, c='k', label='diff')
ax2.set_title('Difference between distances')
ax1.set_xticks(np.arange(len(vars)))
ax1.set_xticklabels(vars)
ax2.set_xticks(np.arange(len(vars)))
ax2.set_xticklabels(vars)
# ax1.set_legend()
plt.show()

if debug:
    print('AR Model with dim =', dim, ' and memory n =', n, '\n')
    print('Model Coefficients:\n', a1, '\n')
    print('Initial Conditions:\n', t_0, '\n')
    print('Sequence:\n', t.astype(int), '\n')
    print('Hankel Matrix:\n', H.numpy().astype(int), '\n')
    print('Gram Matrix with noise variance of ', var_noise, ':\n', H.numpy().astype(int), '\n')
