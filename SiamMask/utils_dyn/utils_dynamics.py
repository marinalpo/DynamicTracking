# import torch
# import numpy as np
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


def predict_Hankel(H):
    """ Predicts one trajectory value given the Hankel Matrix
    Args:
        - H: Hankel matrix of the sequence
    Returns:
        - first_term: Predicted value
    """
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:, -1]
    last_column_of_H = H[-1, :]
    last_column_of_H = last_column_of_H[1:]
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    return first_term


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


def Hankel(s0, stitch=False, s1=0):
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

    d = torch.sqrt(torch.logdet((X + Y)/2) - 0.5*torch.logdet(torch.matmul(X, Y)))
    return d


def compare_dyn(s0, s1, eps):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence to be added
        - eps: Noise value
    Returns:
        - dist: scalar distance between sequences
    """
    # Compute Hankel and Gram Matrices
    mean_root = np.mean(s0)
    s0 = s0 - mean_root
    s1 = s1 - mean_root
    H = Hankel(torch.from_numpy(s0))
    G = Gram(H, eps)
    H_s = Hankel(torch.from_numpy(s0), True, torch.from_numpy(s1))
    G_s = Gram(H_s, eps)
    distance = JBLD(G, G_s)
    return np.nan_to_num(distance.numpy())