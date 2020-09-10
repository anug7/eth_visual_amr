import numpy as np
import cv2


import conversion as conv


def getIndividualCorners(ips, n=12):
  """
  """
  points = []
  for i in range(n):
    points.append(np.asarray([ips[i*2], ips[i*2+1], 1]))
  return np.asarray(points)


def getNormalizedPoints(points, kmat):
  """
  """
  # kmat^-1 * [u, v, 1]T
  return (np.linalg.inv(kmat).dot(points.T).T)


def buildQMatrix(p, P, n=12):
  """
  """
  q_mat = []
  for i in range(n):
    b = np.asarray([1.0, 0.0, -1.0 * p[i][0]])
    c = np.asarray([0.0, 1.0, -1.0 * p[i][1]])
    q_mat.append(np.kron(b, P[i]))
    q_mat.append(np.kron(c, P[i]))

  return np.asarray(q_mat)


def computeMMatrixFromQ(q_mat):
  """
  """
  u, s, v = np.linalg.svd(q_mat)
  # each row corresponds to Eigen vector of q_mat.T * q_mat
  # Get eigen values corresponding to lowest eigen values
  # which is last row in v
  min_vec = v[-1, :].T
  if min_vec[-1] < 0:
    min_vec = -1 * min_vec
  min_vec = min_vec.reshape((3, 4))
  return min_vec


def decomposeRTMatrix(augmat):
  """
  """
  r, t = augmat[:3, :3], augmat[:, -1].T
  u, s, v = np.linalg.svd(r)
  # force unit eigen value property for R matrix
  rnew = np.matmul(u, v)
  val = np.linalg.det(rnew)
  if not (np.allclose(1.0, val) or np.allclose(np.eye(3), np.matmul(rnew.T, rnew))):
    print("Rot matrix error.{}").format(val)
  scale = np.linalg.norm(rnew) / np.linalg.norm(r)
  # remove scale from R matrix .ie what we have found is alpha * R
  rnew, t = rnew, t * scale
  return np.hstack((rnew, t.reshape(3, 1)))


def estimatePoseDLT(p, P, kmat):
  """
  """
  p_dash = getNormalizedPoints(p, kmat)
  qmat = buildQMatrix(p_dash, P)
  m = computeMMatrixFromQ(qmat)
  M = decomposeRTMatrix(m)
  return M


def projectPoints_mat(kmat, trans, coords):
  """
  """
  tmp = np.matmul(kmat, trans)
  tmp = np.matmul(tmp, coords.T).T
  tmp = tmp / tmp[:, -1].reshape((12, 1))
  return tmp.astype('int64')[:, :-1]


def drawCorners(oc, rc, img, n=12):
  """
  """
  for i in range(n):
    try:
      cv2.circle(img, (oc[i][0], oc[i][1]), 2, (0, 255, 0), 1)
      cv2.circle(img, (rc[i][0], rc[i][1]), 2, (0, 0, 255), 1)
    except Exception as e:
      import ipdb; ipdb.set_trace()
      pass
  return img


img_path = "../data/images_undistorted/img_{}.jpg"
n = 12
kmat = np.loadtxt("../data/K.txt")
obj_points = np.loadtxt("../data/p_W_corners.txt", delimiter=",")
img_points = np.loadtxt("../data/detected_corners.txt")
P = np.hstack((obj_points, np.ones((n, 1))))
cv2.namedWindow("op", 0)

for idx, indv_img_points in enumerate(img_points):
  img = cv2.imread(img_path.format(str(idx+1).zfill(4)))
  p = getIndividualCorners(indv_img_points)
  M = estimatePoseDLT(p, P, kmat)
  rep_p = projectPoints_mat(kmat, M, P)
  img = drawCorners(p.astype('int64'), rep_p, img)

  cv2.imshow("op", img)
  cv2.waitKey(30)
