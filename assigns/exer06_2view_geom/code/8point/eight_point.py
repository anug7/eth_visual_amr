import numpy as np


def compute_fmatrix(p1, p2):
    """
    Computes Fundamental matrix from point correspondence
    @param: p1[Nx3]: homogeneous coordinates for points in img1
    @param: p2[Nx3]: homogeneous coordinates for points in img2
    @return: F[3x3]: fundamental matrix
    """
    qmat = []
    for n in range(p1.shape[0]):
        qmat.append(np.kron(p1[n], p2[n]))
    qmat = np.asarray(qmat)
    u, s, v = np.linalg.svd(qmat.T.dot(qmat))
    pmat = v[-1, :].reshape((3, 3))
    u1, s1, v1 = np.linalg.svd(pmat)
    sdiag = np.diag(s1)
    sdiag[2, 2] = 0
    pmat1 = u1.dot(sdiag.dot(v1))
    return pmat1

def compute_dist2epilines(F, p1, p2):
    """
    Computes distance to epipolar lines from points
    to assess quality of F
    @param: p1[3xN] points in img1 in homogeneous coordinates
    @param: p2[3xN] points in img2 in homogeneous coordinates
    @param: F[3xx] fundamental matrix
    """
    l1 = F.T.dot(p2.T).T
    l2 = F.dot(p1.T).T
    cost = 0.0
    for n in range(p1.shape[0]):
        denom1 = l1[n]**2 + l2[n]**2
        op = ((l1[n].dot(p1[n]))**2 + (l2[n].dot(p2[n]))**2) / denom1
        cost += np.sum(op/denom1)
    return np.sqrt(cost / p1.shape[0])


