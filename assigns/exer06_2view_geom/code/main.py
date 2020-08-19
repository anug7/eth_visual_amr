
import numpy as np
import cv2

import eight_point.eight_point as ep
import triangulation.linear_triang as lt

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    print(xyz_points.shape[0])
    idx = 0
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        x, y, z = xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2]
        fid.write(bytearray(struct.pack("fffccc", x, y, z,
                           rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                           rgb_points[i,2].tostring())))
    fid.close()

def get_essential_matrix(fmat, k1, k2):
  """
  Decomposes essential matrix from fundamental matrix
  with calibration matrices
  """
  ik1 = np.linalg.inv(k1)
  ik2 = np.linalg.inv(k2)
  emat = (ik2.T.dot(fmat)).dot(ik1)
  return emat


def decompose_essential_matrix(E):
  """
  Decomposes essential matrix into R & T
  """
  u, s, v = np.linalg.svd(E)
  t = u[:, -1] # last column of u is translation vector
  w = np.asarray([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
  r1 = (u.dot(w)).dot(v)
  if np.linalg.det(r1) < 0:
    r1 *= -1
  r2 = (u.dot(w.T)).dot(v)
  if np.linalg.det(r2) < 0:
    r2 *= -1
  return [r1, r2], t


def disambiguate_relative_pose(rots, t, p1, p2, K1, K2):
  """
  Calculcate correct rotation & translation matrix
  among 4 possible combination
  """
  m1, T = K1.dot(np.eye(3, 4)), t.reshape((3, 1))
  p1m2 = K2.dot(np.hstack((rots[0], T)))
  p2m2 = K2.dot(np.hstack((rots[1], T)))
  p3m2 = K2.dot(np.hstack((rots[0], -T)))
  p4m2 = K2.dot(np.hstack((rots[1], -T)))
  mats = [[rots[0], t], [rots[1], t],
          [rots[0], -1], [rots[1], -t]]

  P1 = lt.triangulate_linear(p1, p2, m1, p1m2)
  P2 = lt.triangulate_linear(p1, p2, m1, p2m2)
  P3 = lt.triangulate_linear(p1, p2, m1, p3m2)
  P4 = lt.triangulate_linear(p1, p2, m1, p4m2)

  P1_counts = np.sum(P1, axis=0)[-2]
  P2_counts = np.sum(P2, axis=0)[-2]
  P3_counts = np.sum(P3, axis=0)[-2]
  P4_counts = np.sum(P4, axis=0)[-2]

  counts = [P1_counts, P2_counts, P3_counts, P4_counts]
  max_count = max(counts)
  idx = counts.index(max_count)

  return mats[idx]

img1 = cv2.imread("../data/0001.jpg")
img2 = cv2.imread("../data/0002.jpg")

K = np.asarray([[1379.74, 0, 760.35],
                   [0, 1382.08, 503.41],
                   [0,       0,      1]])

p1 = np.loadtxt("../data/matches0001.txt")
p2 = np.loadtxt("../data/matches0002.txt")

p1 = p1.T
p2 = p2.T

p1 = np.hstack((p1, np.ones((p1.shape[0], 1))))
p2 = np.hstack((p2, np.ones((p2.shape[0], 1))))

np1, tmat1 = ep.normalize_2d_points(p1)
np2, tmat2 = ep.normalize_2d_points(p2)

nF = ep.compute_fmatrix(np1, np2)
F = (tmat2.T.dot(nF)).dot(tmat1)

E = get_essential_matrix(F, K, K)

rots, trans = decompose_essential_matrix(E)

R, T = disambiguate_relative_pose(rots, trans, p1, p2, K, K)

M1 = K.dot(np.eye(3, 4))
M2 = K.dot(np.hstack((R, T.reshape((3, -1)))))

P = lt.triangulate_linear(p1, p2, M1, M2)

cam1 = np.asarray([[0,   0,   0,  1],
                   [0.05, 0,   0,  1],
                   [0, 0.05,   0,  1],
                   [0,   0, 0.05,  1]])
tmat = np.hstack((R, T.reshape((3, -1))))
tmat = np.vstack((tmat, np.zeros(4)))
tmat[3, 3] = 1
cam2 = tmat.dot(cam1.T).T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs=P[:, 0], ys=P[:, 1], zs=P[:, 2])
ax.scatter(xs=cam1[:, 0], ys=cam1[:, 1], zs=cam1[:, 2], c="Red")
ax.scatter(xs=cam2[:, 0], ys=cam2[:, 1], zs=cam2[:, 2], c="Green")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

"""
for p in p1:
  img1 = cv2.circle(img1, (int(p[0]), int(p[1])), 3, (0, 255, 0))

for p in p2:
  img2 = cv2.circle(img2, (int(p[0]), int(p[1])), 3, (255, 0, 0))

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)

cv2.waitKey(0)
"""
