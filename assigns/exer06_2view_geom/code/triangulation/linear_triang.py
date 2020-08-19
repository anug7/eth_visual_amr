import numpy as np

def get_anti_symmatrix(vec):
    """
    """
    mat = np.asarray([[0, -vec[2], vec[1]],
                      [vec[2], 0, -vec[0]],
                      [-vec[1], vec[0], 0]])
    return mat

def triangulate_linear(p1, p2, M1, M2):
    """
    triangulates point in 3D using image correspondences
    in two images
    @param: p1: [Nx3]
    @param: p2: [Nx3]
    @param: M1: [3x4]
    @param: M2: [3x4]
    """
    points = []
    for i in range(p1.shape[0]):
        pp1 = get_anti_symmatrix(p1[i, :])
        pp2 = get_anti_symmatrix(p2[i, :])
        op1 = pp1.dot(M1)
        op2 = pp2.dot(M2)
        amat = np.vstack((op1, op2))
        u, s, v = np.linalg.svd(amat)
        min_vec = v[-1, :]
        min_vec /= min_vec[3]
        points.append(min_vec.tolist())
    return np.array(points)
