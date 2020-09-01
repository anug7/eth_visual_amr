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
  u, s, v = np.linalg.svd(qmat)
  pmat = v[-1].reshape((3, 3))
  u1, s1, v1 = np.linalg.svd(pmat)
  sdiag = np.diag(s1)
  sdiag[2, 2] = 0
  pmat1 = (u1.dot(sdiag)).dot(v1)
  return pmat1


def compute_algebraic_error(fmat, p1, p2):
  """
  """
  cost = 0.0
  for i in range(p1.shape[0]):
    tmp = (p2[i, :].T.dot(fmat)).dot(p2[i, :])
    cost += tmp
  cost = np.sqrt((cost ** 2) / p1.shape[0])
  return cost


def compute_dist2epilines(fmat, p1, p2):
  """
  Computes distance to epipolar lines from points
  to assess quality of F
  @param: p1[3xN] points in img1 in homogeneous coordinates
  @param: p2[3xN] points in img2 in homogeneous coordinates
  @param: F[3x3] fundamental matrix
  """
  cost = 0.0
  for n in range(p1.shape[0]):
    l1, l2 = fmat.T.dot(p2[n]), fmat.dot(p1[n])
    # Normalization as per: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
    l1 = l1 / (np.sqrt(l1[0]**2 + l1[1]**2))
    l2 = l2 / (np.sqrt(l2[0]**2 + l2[1]**2))
    cost += l1.dot(p1[n]) ** 2 + l2.dot(p2[n]) ** 2
  return np.sqrt(cost / p1.shape[0])


def normalize_2d_points(p1):
  """
  Normalizes 2d points in image plane to make it robust
  to numerical stablities
  spread points around a centroid of the points
  """
  mean = np.mean(p1[:, :2], axis=0)
  mean_dist = np.sum((np.linalg.norm(mean - p1[:, :2], axis=1))**2) / p1.shape[0]
  sj = np.sqrt(2) / np.sqrt(mean_dist)
  tmat = np.asarray([[sj, 0, -sj * mean[0]], [0, sj, -sj * mean[1]], [0, 0, 1]])
  normp1 = tmat.dot(p1.T).T
  return normp1, tmat
