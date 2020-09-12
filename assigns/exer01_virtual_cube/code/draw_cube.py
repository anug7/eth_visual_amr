
import cv2
import numpy as np


def hat(tup_v):                         
  v1 = tup_v[0]   #* axis-x 
  v2 = tup_v[1]   #* axis-y               
  v3 = tup_v[2]   #* axis-z  
  hatMap = np.asarray([ [0,-v3, v2], [v3, 0, -v1], [-v2, v1, 0] ])   
  return hatMap


def poseVectorToTransformationMatrix(pose):
  """
  """
  r, t = pose[:3], pose[3:].reshape((3, -1))
  theta = np.linalg.norm(r)
  k = r / theta
  kx = hat(k)
  R = np.eye(3) + np.sin(theta) * kx + ((1 - np.cos(theta)) * np.matmul(kx, kx))
  return np.hstack((R, t))


def projectPoints_mat(kmat, trans, coords):
  """
  """
  tmp = kmat.dot(trans)
  trans_coords = coords.dot(tmp.T)
  scales = trans_coords[:, -1].reshape((coords.shape[0], -1))
  norm_coords = trans_coords / scales
  return norm_coords.astype('uint64')[:, :-1]


def projectPoints(kmat, trans, coords):
  """
  """
  points, tmp  = [], np.dot(kmat, trans)
  for coord in coords:
    tmp = np.dot(trans, coord.reshape((4, 1)))
    pnt = np.dot(kmat, tmp)
    pnt = (pnt / pnt[2]).astype("uint64").reshape((3, ))
    points.append(pnt[:2])

  return points


def drawCube(img, coords):
  """
  """
  img = cv2.line(img, tuple(coords[0]), tuple(coords[1]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[1]), tuple(coords[2]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[2]), tuple(coords[3]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[3]), tuple(coords[0]), (0, 0, 255), 2)
  
  img = cv2.line(img, tuple(coords[4]), tuple(coords[5]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[5]), tuple(coords[6]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[6]), tuple(coords[7]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[7]), tuple(coords[4]), (0, 0, 255), 2)
  
  img = cv2.line(img, tuple(coords[0]), tuple(coords[4]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[1]), tuple(coords[5]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[2]), tuple(coords[6]), (0, 0, 255), 2)
  img = cv2.line(img, tuple(coords[3]), tuple(coords[7]), (0, 0, 255), 2)

  return img



def getCheckerCorners(size=4):
  """
  """
  points = []
  for i in range(9):
    for j in range(6):
      points.append([i * size/ 100.0, j * size/ 100., 0, 1])
  return np.asarray(points)


def getCubeCoordsWorld(s=0.04, o=(0, 0)):
  """
  """
  patmat = np.asarray([[o[0],         o[1],  0, 1],
                       [o[0],     s + o[1],  0, 1],
                       [s + o[0], s + o[1],  0, 1],
                       [s + o[0],     o[1],  0, 1],
                       [o[0],         o[1], -s, 1],
                       [o[0],     s + o[1], -s, 1],
                       [s + o[0], s + o[1], -s, 1],
                       [s + o[0],     o[1], -s, 1],
                      ])
  return patmat


def projectPointsWithDist(kmat, aug, dcoef, points):
  """
  """
  trans_coords = points.dot(aug.T)
  scales = trans_coords[:, -1].reshape((points.shape[0], -1))
  norm_coords = trans_coords / scales
  op_coords = []
  for coord in norm_coords:
    r = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
    dist_coord = (1 + dcoef[0] * r**2 +  dcoef[1] * r**4) * coord[:2]
    hc_coord = np.append(dist_coord, 1).reshape((-1, 1))
    cam_coord = kmat.dot(hc_coord)
    cam_coord = cam_coord / cam_coord[2]
    op_coords.append(cam_coord)
  return np.asarray(op_coords).astype('uint64')[:, :-1]


if __name__ == "__main__":
  poses = np.loadtxt("data/poses.txt")
  kmat = np.loadtxt("data/K.txt")
  dist = np.loadtxt("data/D.txt")
  
  cv2.namedWindow("cube", 0)
  
  for idx, pose in enumerate(poses):
  
    aug = poseVectorToTransformationMatrix(pose)
    cube_coords = getCubeCoordsWorld(0.04, (0.08, 0.08))
    #points = projectPoints_mat(kmat, aug, cube_coords)
    
    points_dist = projectPointsWithDist(kmat, aug, dist, cube_coords)
    #corners = getCorners()
    # corners = np.asarray([[0, 0, 0, 1], [0.04, 0, 0, 1]])
    #trans_corners = projectPoints(kmat, aug, corners)
    
    # img = cv2.imread("data/images_undistorted/img_0001.jpg")
    img = cv2.imread("data/images/img_{}.jpg".format(str(idx+1).zfill(4)))
   
    if not isinstance(img, np.ndarray):
      print("Invalid image")
      continue
  
    img = drawCube(img, points_dist)
    #for trans_cor in trans_corners:
    #  img = cv2.circle(img, tuple(trans_cor), 2, (0, 0, 255), 2)
    
    cv2.imshow("cube", img)
    cv2.waitKey(30)
