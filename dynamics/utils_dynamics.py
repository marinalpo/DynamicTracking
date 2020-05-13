import torch
import numpy as np
device = torch.device('cpu')



def compute_centroid(loc):
    centx = 0.25 * (loc[0] + loc[2] + loc[4] + loc[6])
    centy = 0.25 * (loc[1] + loc[3] + loc[5] + loc[7])
    return np.array([centx, centy])


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
    if len(s0.shape) == 1:  # s0 is 1D with shape (l0,) not (l0, 1)
        dim = 1
        l0 = s0.shape[0]
    else:
        l0, dim = s0.shape

    if stitch:
        # print('data s1:', s1.shape)
        l1 = s1.shape[0]
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
        if dim == 1:
            H[dim * i, :] = (s0[i:i + num_cols]).view(1, num_cols)
        else:
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


def JBLD(X, Y, det=True):
    """ Computes the Jensen-Bregman LogDet (JBLD) distance between two Gram matrices
    Args:
        - X and Y: Normalized Gram matrices
        - det: Boolean indicating if the determinant is going to be computed or not
    Returns:
        - d: JBLD distance value between X and Y
    """
    print('primera part: ', torch.log(torch.det((X + Y)/2)))
    print('primer det:', torch.det((X + Y)/2))
    print('segona part:', 0.5*torch.log(torch.det(torch.matmul(X, Y))))
    print('det:',torch.det(torch.matmul(X, Y)))
    print('prod:', torch.matmul(X, Y))
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    if not det:
        d = (torch.det((X + Y) / 2)) - 0.5 * (torch.det(torch.matmul(X, Y)))
    return torch.sqrt(d)


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



def compare_dynamics2(data_root, data, eps, coord, BS=1):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence including data_root
        - eps: Noise value
        - coord: Coordinate value (0: x, 1: y, 2: mean)
        - BS: Batch Size
    Returns:
        - dist: (BS, 2) JBLD distance between sequences
    """
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps), True)
    if coord == 2:
        dist = torch.mean(dist, dim=1)
    else:
        dist = dist[0][coord].item()  # Print only x information
    return dist


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


def compare_dyn(s0, s1, eps, dist):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence to be added
        - eps: Noise value
        - dist: if 0: JBLD if 1: JKL
    Returns:
        - dist: scalar distance between sequences
    """
    # Compute Hankel and Gram Matrices
    H = Hankel(torch.from_numpy(s0))
    G = Gram(H, eps)
    H_s = Hankel(torch.from_numpy(s0), True, torch.from_numpy(s1))
    G_s = Gram(H_s, eps)

    if dist == 0:
        distance = JBLD(G, G_s)
    elif dist == 1:
        distance = JKL(G, G_s)
    return np.nan_to_num(distance.numpy())


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

