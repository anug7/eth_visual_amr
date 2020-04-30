
import math
import numpy as np


_EPS = np.finfo(float).eps * 4.0

def AxisAngleToQuad(vect):
  """
  Convert Axis-Angle rotation to Quaternion
  @param: vect: [ox, oy, oz] with ||vect||=angle of rotation
          and vect / ||vect|| gives unit axis of rotation
  @return: [qx, qy, yz, qw]: quaternion vector
  """

  # theta = ||omega||
  # Axis vector = omega / ||omega||
  theta = np.linalg.norm(vect)
  axis = vect / theta
  # quad = [i, j, k, w]
  # https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/geometric/quaternionTraining001.pdf
  quad = [axis[0] * np.sin(theta/2),  axis[1] * np.sin(theta/2),
          axis[2] * np.sin(theta/2), np.cos(theta/2)]
  
  return np.asarray(quad)


def quad2AxisAngle(quad):
  """
  Refer for AxisAngleToQuad 
  """
  if quad[3] > 1.0:
    quad = np.linalg.norm(quad)
  theta = 2 * np.arccos(quad[3])
  s = np.sqrt(1 - quad[3]**2)
  if s > 0.001:
    ax = quad[0] / s
    ay = quad[1] / s
    az = quad[2] / s
  else:
    ax = quad[0] 
    ay = quad[1]
    az = quad[2]

  return np.asarray([ax, ay, az]) * theta


def quad2RotationMatrix(q):
  """
  Refer: tf transformations
  Converts quad to Homogenous Rotation matrix
  @param: quad: [qx, qy, qz, qw]
  @param: matrix of 4x4 shape
  """
  q = np.array(quaternion[:4], dtype=np.float64, copy=True)
  nq = np.dot(q, q)
  if nq < _EPS:
      return numpy.identity(4)
  q *= math.sqrt(2.0 / nq)
  q = np.outer(q, q)
  return np.array((
      (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
      (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
      (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
      (                0.0,                 0.0,                 0.0, 1.0)
      ), dtype=np.float64)


def rotationMatrix2Quad(matrix):
  """
  Refer: tf transformations
  Converts rotmatrix to Quad.
  @param: matrix: [4x4] rotation matrix
  @return:  [qx, qy, qz, qw] quaternion
  """
  q = np.empty((4, ), dtype=np.float64)
  M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
  t = np.trace(M)
  if t > M[3, 3]:
      q[3] = t
      q[2] = M[1, 0] - M[0, 1]
      q[1] = M[0, 2] - M[2, 0]
      q[0] = M[2, 1] - M[1, 2]
  else:
      i, j, k = 0, 1, 2
      if M[1, 1] > M[0, 0]:
          i, j, k = 1, 2, 0
      if M[2, 2] > M[i, i]:
          i, j, k = 2, 0, 1
      t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
      q[i] = t
      q[j] = M[i, j] + M[j, i]
      q[k] = M[k, i] + M[i, k]
      q[3] = M[k, j] - M[j, k]
  q *= 0.5 / math.sqrt(t * M[3, 3])
  return q
