import torch
import numpy as np
from scipy.special import comb
from mpmath import matrix, qr
# torch.set_default_dtype(torch.float64)


def compute_centroid(loc):
    centx = 0.25 * (loc[0] + loc[2] + loc[4] + loc[6])
    centy = 0.25 * (loc[1] + loc[3] + loc[5] + loc[7])
    return np.array([centx, centy])


def JKL(X, Y):
    """ Computes the Jensen-Bregman LogDet (JBLD) distance between two Gram matrices
    Args:
        - X and Y: Normalized Gram matrices
        - det: Boolean indicating if the determinant is going to be computed or not
    Returns:
        - d: JBLD distance value between X and Y
    """
    d = 0.5*torch.trace(torch.matmul(torch.inverse(X), Y) + torch.matmul(torch.inverse(Y), X) - 2*torch.eye(X.shape[0]))
    return torch.sqrt(d)


def Hankel(s0, stitch=False, s1=0, mean=True):
    """ Creates the Hankel Matrix given a sequence
    Args:
        - s0: Root sequence shape(l0, dim) or (l0,)
        - switch: Boolean to indicate if Hankel must be stitched or not
        - s1: Sequence to add if Hankel must be stitched shape(l1, dim) or (l1,)
    Returns:
        - H: Hankel matrix
    """
    # print('data s0:', s0.shape)

    # Retrieve sequence's lengths and dimensions
    l1 = 0
    l0, dim = s0.size()
    # print('sequence:', s0)
    # print('sequence shape:', s0.shape)
    # print('sequence mean:', torch.mean(s0))
    # print('new seq:', s0 - torch.mean(s0) )
    # s0 = s0 - torch.mean(s0)
    if stitch:
        # print('data s1:', s1.shape)
        l1 = s1.size()[0]
        s0 = torch.cat([s0, s1])  # Vertical stack

    if mean:
        s0 = s0 - torch.mean(s0)


    # Compute Hankel dimensions
    if l0 % 2 == 0:  # l is even
        num_rows = int(l0/2) * dim
        num_cols = int(l0/2) + 1 + l1
    else:
        num_rows = int(torch.ceil(torch.tensor(l0 / 2))) * dim
        num_cols = int(torch.ceil(torch.tensor(l0 / 2))) + l1
    H = torch.zeros([num_rows, num_cols])


    # Fill Hankel matrix
    for i in range(int(num_rows/dim)):
        # if dim == 1:
        #     H[dim * i, :] = (s0[i:i + num_cols]).view(1, num_cols)
        # else:
          for d in range(dim):
              H[dim * i + d, :] = s0[i:i + num_cols, d]
    return H


def Gram(H, var):
    """ Generates a normalized Gram matrix given the Hankel matrix and a noise factor
    Args:
        - H: Hankel matrix containing temporal information
        - eps: noise factor (1)
    Returns:
        - Gnorm: normalized Gram matrix
    """
    N = H.shape[0] * np.power(var, 2) * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + torch.matmul(N, N.t())
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def JBLD(X, Y):
    """ Computes the Jensen-Bregman LogDet (JBLD) distance between two Gram matrices
    Args:
        - X and Y: Normalized Gram matrices
    Returns:
        - d: JBLD distance value between X and Y
    """
    X = X.double()
    Y = Y.double()
    d = torch.sqrt(torch.logdet((X + Y)/2) - 0.5*torch.logdet(torch.matmul(X, Y)))
    d = d.float()
    return d


def compare_dyn(s0, s1, eps, mean):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence to be added
        - eps: Noise value
    Returns:
        - dist: scalar distance between sequences
    """
    # Compute Hankel and Gram Matrices
    # mean_root = np.mean(s0)
    # s0 = s0 - mean_root
    # s1 = s1 - mean_root
    H = Hankel(s0, mean=mean)
    G = Gram(H, eps)
    H_s = Hankel(s0, stitch=True, s1=s1, mean=mean)
    G_s = Gram(H_s, eps)
    distance = JBLD(G, G_s)
    return np.nan_to_num(distance.numpy())

def Hankel_new(s0, num_rows, num_cols, mean=True):
    """ Creates the Hankel Matrix given a sequence
    Args:
        - s0: Root sequence shape(l0, dim) or (l0,)
        - switch: Boolean to indicate if Hankel must be stitched or not
        - s1: Sequence to add if Hankel must be stitched shape(l1, dim) or (l1,)
    Returns:
        - H: Hankel matrix
    """
    l0, dim = s0.size()
    if mean:
        s0 = s0 - torch.mean(s0)

    H = torch.zeros([num_rows, num_cols])

    # Fill Hankel matrix
    for i in range(int(num_rows / dim)):
        # if dim == 1:
        #     H[dim * i, :] = (s0[i:i + num_cols]).view(1, num_cols)
        # else:
        for d in range(dim):
            H[dim * i + d, :] = s0[i:i + num_cols, d]
    return H


def compare_dyn_R(s0, s1, eps, R, mean):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence. TORCH shape: (l0, 1)
        - data: Data sequence to be added. TORCH shape: (l1, 1)
        - eps: Noise value
        - R: Order
    Returns:
        - dist: scalar distance between sequences
    """
    # Compute Hankel and Gram Matrices
    nc = R + 1
    nr = s0.shape[0] - R
    H = Hankel_new(s0, nr, nc, mean=mean)
    G = Gram(H.t(), eps)

    # s01 = torch.cat((s0, s1))
    s01 = s1

    nr = s01.shape[0] - R
    H_s = Hankel_new(s01, nr, nc, mean=mean)
    G_s = Gram(H_s.t(), eps)

    distance = JBLD(G, G_s)

    return np.nan_to_num(distance.numpy())


def fast_hstln_mo(u, R, slow, max_iter=50, tol=1e-7):
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
    torch.set_default_dtype(torch.float64)
    D_u, N_u = u.shape  # Dimension, Length

    # print('type u:', u.type())
    # u = u.type(torch.DoubleTensor)
    # print('type u:', u.type())

    # Default values
    omega = torch.ones((1, N_u))
    x0 = []
    eta = torch.zeros((D_u, N_u))
    w = 1e8

    nc = R + 1
    nr = (N_u - nc + 1) * D_u

    hi = np.arange(1, N_u * D_u + 1, dtype=np.float64).reshape(N_u, D_u)
    hi = Hankel_new(torch.from_numpy(hi), nr, nc, mean=False)  # Hankel matrix (hr, hc)

    # Replace missing values with mean of the data
    mu = torch.sum(u) / torch.sum(omega)
    # u = u.type(torch.float)
    u = u + (mu * torch.ones((1, N_u))) * (torch.zeros((D_u, N_u)))

    Ab = Hankel_new(torch.transpose(u, 0, 1), nr, nc, mean=False)  # Hankel matrix (hr, hc)
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
        # print('type MQ:', MQ.type())
        # # print('type w:', w.type())
        # print('type eta:', eta.type())
        # print('type D:', D.type())
        # print('type r:', r.type())
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
    # print('type u_hat:', u_hat.type())
    # print('type eta:', eta.type())
    torch.set_default_dtype(torch.float32)
    return u_hat, eta, mse, x


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


def fast_incremental_hstln_mo(u, eta_max, slow):
    D_u, N_u = u.shape  # Dimension, Length

    nc = np.floor((N_u + 1) * D_u / (D_u + 1))  # Pick as fewest columns as possible
    nr = (N_u - nc + 1) * D_u

    R_max = int(min([nr, nc]))
    R_min = 1

    for R in range(R_min, R_max):

        [u_hat, eta, mse, x] = fast_hstln_mo(u, R, slow, max_iter=10, tol=1e-4)
        av_eta = torch.norm(eta, 'fro') ** 2 / (D_u * N_u)

        if av_eta < (eta_max ** 2) * 1.5:
            [u_hat, eta, mse, x] = fast_hstln_mo(u, R, slow, max_iter=50, tol=1e-7)
            av_eta = torch.norm(eta, 'fro') ** 2 / (D_u * N_u)

        if av_eta < eta_max ** 2:
            break

    return u_hat, eta, x, R


def predict_Hankel(H):
    """ Predicts one trajectory value given the Hankel Matrix
    Args:
        - H: Hankel matrix of the sequence
    Returns:
        - first_term: Predicted value
    """
    # Singular Value Decomposition of the Hankel Matrix
    U, S, V = torch.svd(H)

    # Last column of V
    r = V[:, -1]

    # All elements minus the first of the last column of H
    l = H[-1, 1:]

    pred = - torch.matmul(l, r[:-1]) / r[-1]

    return pred