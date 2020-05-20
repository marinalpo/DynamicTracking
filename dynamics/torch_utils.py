import torch
import numpy as np
from scipy.special import comb

torch.set_default_dtype(torch.float64)


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
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
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


def fill_by_ind(val, ind):
    # val: Array that will define the values to fill the matrix with.
    # ind: Matrix that will define the size and indexing of A.
    # X: Matrix with shape of ind filled with val values in the positions specified by the index.
    X = ind.clone()
    D_u, N_u = X.shape
    for i in range(D_u * N_u):
        val2mat = exponent_nk(i, 2)
        valid_idxs = []
        for l in range(val2mat.shape[0]):
            if (val2mat[l, 0] < D_u) and (val2mat[l, 1] < N_u):
                valid_idxs.append(val2mat[l, :])
        val2mat = np.asarray(valid_idxs)
        if (val2mat.shape[0] > 0):
            if (val[i] != 0.0):
                X[np.ix_(val2mat[:, 0]), np.ix_(val2mat[:, 1])] = val[i]
            else:
                X[np.ix_(val2mat[:, 0]), np.ix_(val2mat[:, 1])] = 0.0
    return X
