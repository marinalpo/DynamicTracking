import torch
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(16, edgeitems=8, linewidth=150)
import numpy as np
from utils_dynamics import *
from torch_utils import *
from torch_utils import *
import matplotlib.pyplot as plt
from time import time

t1 = time()




# Load data
c = np.load('/Users/marinaalonsopoal/Desktop/Objects/centr_obj_1.npy')  # (154, 2)
# c = np.load('/data/Marina/centroids/centr_obj_1.npy')

# Parameters
T = c.shape[0]
T0 = 11
R = 5
eps = 1

etas = torch.zeros((T, 2))
mses = torch.zeros((T, 2))

for t in range(T):  #11
    print('Frame:', t)
    if t >= T0 - 1:
        for d in range(2):
            data_root = c[t - T0+1:t+1, d]
            data_root = torch.from_numpy(data_root)
            data_root = data_root.view(1, len(data_root))
            data_root = data_root - torch.mean(data_root)
            [xhat, eta, mse] = fast_hstln_mo(data_root, R)
            etas[t, d] = torch.norm(eta, 'fro')
            mses[t, d] = mse

t2 = time()
total_time = t2 - t1
print('\nTotal time:', np.around(total_time/60, 1), ' min.\n')

frames = np.arange(T)
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(frames, etas[:, 0], c='r')
ax[0, 0].set_title('Frobenius norm of Eta in Centroid x')

ax[0, 1].plot(frames, etas[:, 1], c='g')
ax[0, 1].set_title('Frobenius norm of Eta in Centroid y')

ax[1, 0].plot(frames, mses[:, 0], c='r')
ax[1, 0].set_title('MSE of Eta in Centroid x')

ax[1, 1].plot(frames, mses[:, 1], c='g')
ax[1, 1].set_title('MSE of Eta in Centroid y')
plt.show()
