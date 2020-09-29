
import numpy as np
from scipy.linalg import logm, expm


def mat2cross(se_mat):
  """
  Converts mat repo to vector rep
  for transformation
  """
  return np.asarray([-se_mat[1, 2], se_mat[0, 2], -se_mat[0, 1]])


def cross2mat(vec):
  """
  COnvert vector rep to matrix repo for
  transformation
  """
  return np.asarray([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def homo_mat_to_twist(homo_mat):
  """
  Converts 4x4 homogeneous matrix to
  twist coordinates
  """
  se_mat = logm(homo_mat)
  v = se_mat[0:3, 3]
  w = mat2cross(se_mat[0:3, 0:3])

  return np.hstack((w.T, v.T))


def twist_tot_homo_mat(twist):
  """
  Converts twist vector to transformation
  homogeneous matrix
  """
  v = twist[0:3]
  w = twist[3:]
  se_mat = np.zeros((4, 4))
  se_mat[0:3, 0:3] = cross2mat(w)
  se_mat[0:3, 3] = v
  return expm(se_mat)
