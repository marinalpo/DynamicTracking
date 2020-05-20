import torch
import numpy as np
from utils_dynamics import *
from torch_utils import *
torch.set_default_dtype(torch.float64)
#
# print('TORCH VERSION:', torch.__version__)

def fast_hstln_mo(U, R):
    """
    Hankel Structured Total Least Squares (HSTLN) - SIMPLIFIED VERSION:
    Fits an order R system to the input vector using HSTLN formula (A+E)x = b+f
    Input:
        - U: Input sequence (DxN) D: dimension, N: sequence length
        - R: Order of the fit
    Output:
        - x_hat:  AR coefficients  (DxN)
        - eta:  corrections (negative estimated noise) (DxN)
    """
    D_u, N_u = U.shape  # Dimension, Length

    # Default values
    tol = 1e-7
    max_iter = 100
    omega = torch.ones((1, N_u))
    x0 = []
    eta = torch.zeros((D_u, N_u))
    w = 100000000

    nc = R + 1
    nr = (N_u - nc + 1) * D_u

    hi = np.arange(1, N_u * D_u + 1).reshape(N_u, D_u)
    hi = Hankel(torch.from_numpy(hi))  # Hankel matrix (hr, hc)

    # TODO: Falten dos lineas de codi
    # Replace missing values with mean of the data
    # mu = ...
    # u = u + ...

    Ab = Hankel(torch.transpose(U, 0, 1))  # Hankel matrix (hr, hc)
    A = Ab[:, :-1]  # [hr, hc - 1]
    b = torch.unsqueeze(Ab[:, -1], 1)  # [hr, 1]

    # Fix rang deficiency
    A[0:R, 0:R] = A[0:R, 0:R] + 1e-5*torch.eye(R)

    x = torch.lstsq(b, A)
    x = x.solution
    x = x[0:A.shape[1], :]

    P1 = torch.cat([torch.zeros((nr, R*D_u)), torch.eye(nr)], dim=1)
    P1 = to_sparse(P1)
    eta = torch.reshape(eta, [D_u*N_u, 1])

    D = torch.diagflat(tile(omega+1e-5, 0, D_u).view(eta.shape))
    D = to_sparse(D)

    Yrow = torch.zeros((1, D_u * (N_u - 1)))
    YP0 = torch.sparse.FloatTensor(nr, D_u * N_u)  # Sparse tensor
    ti = torch.arange(0, nr * nr + 1, nr + 1)

    M = torch.sparse.FloatTensor(nr+D_u*N_u, D_u*N_u+R)  # Sparse tensor
    M = M.to_dense()

    # for i in range(max_iter):
    norm_dparam = torch.zeros(max_iter)
    for i in range(50):
        print('Iteration:', i)

        hi_2 = hi[:, 0:-1] - 1
        E = fill_by_ind(eta, hi_2)

        for r in range(R):
            ind = ti + r * D_u * nr
            vec = x[r] * torch.ones(ind.shape)
            YP0 = fill_sparse(YP0, ind, vec)

        f = eta[-nr:]

        r = b + f - torch.matmul((A+E), x)

        # Fill M

        M[0:nr, 0: D_u * N_u] = (w*(P1 - YP0)).to_dense()
        M[0:nr, D_u*N_u: D_u*N_u + nc - 1] = -w * (A + E)
        M[nr:, 0:D_u*N_u] = D.to_dense()

        # Solve minimization problem
        if i == 0:
            Mm, Mn = M.shape
            I = torch.eye(max(Mm, Mn))
            I = I[0:Mm, 0:Mn]
            I = 1e-2 * I

        M = M + I

        (MQ, MR) = torch.qr(M, True)
        A_2 = MR
        b_2 = torch.matmul(MQ.t(), - torch.cat([w*r, torch.matmul(D.to_dense(), eta)]))
        x_2 = torch.lstsq(b_2, A_2)
        x_2 = x_2.solution
        dparam = x_2[0:A_2.shape[1], :]

        # Update parameters
        deta = dparam[0:N_u * D_u, :]
        dx = dparam[N_u * D_u:, :]


        eta = eta + deta
        x = x + dx

        norm_dparam[i] = torch.norm(dparam)

        # if norm_dparam[i] < tol:
        #     break

    eta = eta.view(D_u, N_u)
    u_hat = U + eta
    return u_hat, eta

# Load data
# c = np.load('/Users/marinaalonsopoal/Desktop/Objects/centr_obj_1.npy')  # (154, 2)

# Parameters
t = 40 - 1  # Equivalent to MATLAB's t=40
T0 = 11
R = 5

# Create input data
u = np.array([4, 5, 6, 7, 10, 9, 10, 11, 12, 13]).reshape(1, 10)  # Toy example
# x = np.around(c[t:t+T0, 0], 1).reshape(1, T0)  # 1D example
# xy = np.around(c[t:t+T0, :], 1).transpose()  # 2D example

# Convert to PyTorch Tensors
u = torch.from_numpy(u)  # shape: (1, 10)
# x = torch.from_numpy(x)  # shape: (1, T0)
# xy = torch.from_numpy(xy)  # shape: (2, T0)



print('Toy example (u) ---------------------')
u_hat, eta = fast_hstln_mo(u, R)
print('\n')

# print('1D example (x) ---------------------')
# x_hat, eta = fast_hstln_mo(x, R)
# print('\n')
#
# print('2D example (xy) ---------------------')
# xy_hat, eta = fast_hstln_mo(xy, R)
# print('\n')

value = u
print('u')
print('value shape:', value.shape)
print('value sum:', torch.sum(value))

value = u_hat
print('u_hat')
print('value shape:', value.shape)
print('value sum:', torch.sum(value))

value = eta
print('eta')
print('value shape:', value.shape)
print('value sum:', torch.sum(value))
