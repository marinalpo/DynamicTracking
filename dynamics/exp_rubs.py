from utils_dynamics import *
import numpy as np
import torch

eps = 0.001


s0 = np.array([[2, 3, 3], [4, 5, 3], [4, 2, 3]])
s1 = np.array([[1, 1, 3], [2, 2, 3]])

h0 = Hankel(torch.from_numpy(a), False, 0)
G = Gram(h0, eps)

h1 = Hankel(torch.from_numpy(a), True, torch.from_numpy(a2))
G1 = Gram(h1, eps)

print('Gram 0:', G, '\n')
print('Gram 1:', G1, '\n')

d = compare_dyn(a, a2, 0.001, 0)
print(d)