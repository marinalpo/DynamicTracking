import torch
import numpy as np
from scipy.special import comb
from utils.utils_dynamics import *
from mpmath import matrix, qr

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)


def exponent_nk(n, K):
    id = np.diag(np.ones(K))
    exp = id

    for i in range(1, n):
        rene = np.asarray([])
        for j in range(0, K):
            for k in range(exp.shape[0] - int(comb(i + K - j - 1, i)), exp.shape[0]):
                if rene.shape[0] == 0:
                    rene = id[j, :] + exp[k, :]
                    rene = np.expand_dims(rene, axis=0)
                else:
                    rene = np.concatenate([rene, np.expand_dims(id[j, :] + exp[k, :], axis=0)], axis=0)
        exp = rene.copy()
    return exp


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile, dtype=np.float64) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def fill_sparse(M, ind, v):
    # M: Sparse Matrix
    # ind: Array with indexes in flat version
    # v: Array with values to fill the matrix with
    rows, cols = M.shape
    num_idxs = ind.shape[0]
    ind_tuples = torch.zeros((num_idxs, 2)).long()
    for idx in range(num_idxs):
        # Set order='F' (Fortran style, columns first)
        ind_double = np.array(np.unravel_index(int(ind[idx]), (rows, cols), order='F')).reshape(1, 2)
        ind_tuples[idx, :] = torch.from_numpy(ind_double)
    M = torch.sparse.FloatTensor(ind_tuples.t(), v, torch.Size([rows, cols])) + M
    return M


def fill_dense(M, ind, v):
    # M: Sparse Matrix
    # ind: Array with indexes in flat version
    # v: Array with values to fill the matrix with
    rows, cols = M.shape
    num_idxs = ind.shape[0]
    ind_tuples = torch.zeros((num_idxs, 2)).long()
    for idx in range(num_idxs):
        # Set order='F' (Fortran style, columns first)
        ind_double = np.array(np.unravel_index(int(ind[idx]), (rows, cols), order='F')).reshape(1, 2)
        ind_tuples[idx, :] = torch.from_numpy(ind_double)
    M[ind_tuples[:, 0], ind_tuples[:, 1]] = v
    return M


def fill_by_ind(val, ind):
    # val: Array that will define the values to fill the matrix with.
    # ind: Matrix that will define the size and indexing of A.
    # X: Matrix with shape of ind filled with val values in the positions specified by the index.
    X = ind.clone()
    D_u, N_u = X.shape
    maxvalue = int(torch.max(ind))
    val_nonzero = torch.nonzero(val)[:, 0]
    val_copy = torch.zeros_like(val)
    val_copy[val_nonzero] = 1.0
    for i in range(maxvalue + 1):
        if i > 0:
            val2mat = exponent_nk(i, 2)
            valid_idxs = []
            for l in range(val2mat.shape[0]):
                if (val2mat[l, 0] < D_u) and (val2mat[l, 1] < N_u):
                    valid_idxs.append(val2mat[l, :])
            val2mat = torch.from_numpy(np.asarray(valid_idxs)).long()
            X[val2mat[:, 0], val2mat[:, 1]] = val_copy[i] * val[i]
        else:
            X[0, 0] = val[i]
    return X


def fast_hstln_mo(u, R, slow):
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
    D_u, N_u = u.shape  # Dimension, Length

    # Default values
    tol = 1e-7
    max_iter = 50
    omega = torch.ones((1, N_u))
    x0 = []
    eta = torch.zeros((D_u, N_u))
    w = 1e8

    nc = R + 1
    nr = (N_u - nc + 1) * D_u

    hi = np.arange(1, N_u * D_u + 1, dtype=np.float64).reshape(N_u, D_u)
    hi = Hankel(torch.from_numpy(hi))  # Hankel matrix (hr, hc)

    # Replace missing values with mean of the data
    mu = torch.sum(u) / torch.sum(omega)
    u = u + (mu * torch.ones((1, N_u))) * (torch.zeros((D_u, N_u)))

    Ab = Hankel(torch.transpose(u, 0, 1))  # Hankel matrix (hr, hc)
    A = Ab[:, :-1]  # [hr, hc - 1]
    b = torch.unsqueeze(Ab[:, -1], 1)  # [hr, 1]

    # Fix rang deficiency
    A[0:R, 0:R] = A[0:R, 0:R] + 1e-5 * torch.eye(R)

    x = torch.lstsq(b, A)
    x = x.solution
    x = x[0:A.shape[1], :]

    P1 = torch.cat([torch.zeros((nr, R * D_u)), torch.eye(nr)], dim=1)
    eta = torch.reshape(eta, [D_u * N_u, 1])

    D = torch.diagflat(tile(omega + 1e-5, 0, D_u).view(eta.shape))

    YP0 = torch.zeros((nr, D_u * N_u))
    ti = torch.arange(0, nr * nr + 1, nr + 1)

    M = torch.zeros((nr + D_u * N_u, D_u * N_u + R))

    max_iter = 55
    norm_dparam = torch.zeros(max_iter)

    for i in range(max_iter):
        # print('Iter: ', i)

        hi_2 = hi[:, 0:-1] - 1
        E = fill_by_ind(eta, hi_2)

        for j in range(R):
            ind = ti + j * D_u * nr
            vec = x[j] * torch.ones(ind.shape)
            YP0 = fill_dense(YP0, ind, vec)

        f = eta[-nr:]

        r = b + f - torch.matmul((A + E), x)

        # Fill M
        M[0:nr, 0: D_u * N_u] = (w * (P1 - YP0))
        M[0:nr, D_u * N_u: D_u * N_u + nc - 1] = -w * (A + E)
        M[nr:, 0:D_u * N_u] = D

        # Solve minimization problem
        if i == 0:
            Mm, Mn = M.shape
            I = torch.eye(max(Mm, Mn))
            I = I[0:Mm, 0:Mn]
            I = 1e-2 * I

        M = M + I


        if slow:
            # Using mpmath library: More precision but way slower
            (MQ, MR) = qr(matrix(M.tolist()), mode='skinny')
            MQ = np.array(MQ.tolist(), dtype=np.float64)
            MR = np.array(MR.tolist(), dtype=np.float64)
            MQ = torch.from_numpy(MQ)
            MR = torch.from_numpy(MR)
        else:
            # Using torch: Less precision (results differ a lot from MATLAB's) but faster and with possibility to
            # include sparse versions of the matrices
            (MQ, MR) = torch.qr(M, True)

        A_2 = MR
        b_2 = torch.matmul(MQ.t(), - torch.cat([w * r, torch.matmul(D, eta)]))
        x_2 = torch.lstsq(b_2, A_2)
        x_2 = x_2.solution
        dparam = x_2[0:A_2.shape[1], :]

        # Update parameters
        deta = dparam[0:N_u * D_u, :]
        dx = dparam[N_u * D_u:, :]

        eta = eta + deta
        x = x + dx

        norm_dparam[i] = torch.norm(dparam)

        if norm_dparam[i] < tol:
            break

    eta = eta.view(D_u, N_u)
    u_hat = u + eta
    mse = (torch.norm(eta * torch.ones((D_u, N_u)), 'fro') ** 2) / (torch.sum(omega * D_u))
    return u_hat, eta, mse
