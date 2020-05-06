from utils_dynamics import *
import numpy as np
import torch

l0 = 4  # Length of root sequence
l1 = 2  # Length of sequence to be stitched
dim = 2
var_noise = 1
debug = True

# Create Data
s0 = np.arange(1, l0 * dim + 1).reshape(l0, dim)  # Root sequence
s1 = np.arange(l0 * dim + 1, l0 * dim + 1 + l1 * dim).reshape(l1, dim)  # Stitch l1 sequences(s)

# Compute Hankel and Gram Matrices
H = Hankel(torch.from_numpy(s0))
G = Gram(H, var_noise)
H_s = Hankel(torch.from_numpy(s0), True, torch.from_numpy(s1))
G_s = Gram(H_s, var_noise)

# Compute JBLD distance
dist = JBLD(G, G_s)
dist2 = JKL(G, G_s)
dist3 = compare_dyn(s0, s1, var_noise, 0)

# Predict Hankel
c_x_pred = predict_Hankel(H)
print('pred:', c_x_pred)

if debug:
    print('Sequence of dim:', dim, 'and s0 len:', l0, 'and s1 len:', l1, '\n')
    print('s0:\n', s0, '\n')
    print('Hankel s0:\n', H.numpy().astype(int), '\n')
    print('Gram s0:\n', np.around(G.numpy(), 2), '\n')
    print('------------------------------------')
    print('s1:\n', s1, '\n')
    print('Stitched Hankel s1:\n', H_s.numpy().astype(int), '\n')
    print('Gram s1:\n', np.around(G_s.numpy(), 2), '\n')
    print('------------------------------------')
    print('JBLD distance:', dist, '\n')
    print('JKL distance:', dist2, '\n')
