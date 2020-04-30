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


def getNormalizedPoints(points, ikmat):
  """
  """
  # kmat^-1 * [u, v, 1]T
  return (ikmat.dot(points.T).T)


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
  min_vec = v[-1, :]
  min_vec = min_vec.reshape((3, 4))
  if min_vec[2, 3] < 0:
    min_vec = -1 * min_vec
  val = np.linalg.det(min_vec[:3, :3])
  if not np.allclose(1.0, val):
    print "Rot matrix error.{}".format(val)
  return min_vec


def decomposeRTMatrix(augmat):
  """
  """
  r, t = augmat[:3, :3], augmat[:, -1].T
  u, s, v = np.linalg.svd(r)
  # force unit eigen value property for R matrix
  rnew = np.matmul(u, v)
  scale = np.linalg.norm(rnew)
  # remove scale from R matrix .ie what we have found is alpha * R
  rnew, t = rnew / scale, t
  return np.hstack((rnew, t.reshape(3, 1)))


def estimatePoseDLT(p, P, inv_kmat):
  """
  """
  p_dash = getNormalizedPoints(p, inv_kmat)
  qmat = buildQMatrix(p_dash, P)
  m = computeMMatrixFromQ(qmat)
  M = decomposeRTMatrix(m)
  return M


def projectPoints(P, M):
  """
  """
  return M.dot(P.T).T


def drawCorners(oc, rc, img, n=12):
  """
  """
  for i in range(n):
    cv2.circle(img, (oc[i][0], oc[i][1]), 2, (0, 255, 0), 1)
    cv2.circle(img, (rc[i][0], rc[i][1]), 2, (0, 0, 255), 1)
  return img


img_path = "../data/images_undistorted/img_{}.jpg"
n = 12
kmat = np.loadtxt("../data/K.txt")
inv_kmat = np.linalg.inv(kmat)
obj_points = np.loadtxt("../data/p_W_corners.txt", delimiter=",") / 100.0
img_points = np.loadtxt("../data/detected_corners.txt")
P = np.hstack((obj_points, np.ones((n, 1))))
cv2.namedWindow("op", 0)

for idx, indv_img_points in enumerate(img_points):
  img = cv2.imread(img_path.format(str(idx+1).zfill(4)))
  p = getIndividualCorners(indv_img_points)
  M = estimatePoseDLT(p, P, inv_kmat)
  rep_p = projectPoints(P, M)
  img = drawCorners(p.astype('int64'), rep_p.astype('int64'), img)

  cv2.imshow("op", img)
  cv2.waitKey(30)
