import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2

sys.path.insert(0, "../../exer01_virtual_cube/code")
sys.path.insert(0, "../../exer02_pnp/code/")
sys.path.insert(0, "../../exer03_keypoint_tracker/code/")

import tracker as tk
import pnp as pnp
import draw_cube as dc

np.random.seed(42)

harris_patch_size = 9
harris_kappa = 0.08
non_maxima_supp_radius = 8
descriptor_radius = 9
match_lambda = 5

no_of_keypoints = 1000

kmat = np.loadtxt("../data/K.txt")
db_keypoints = np.loadtxt("../data/keypoints.txt")
p_w_landmarks = np.loadtxt("../data/p_W_landmarks.txt")

db_keypoints = db_keypoints[:, [1, 0]]
db_keypoints_h = np.hstack((db_keypoints, np.ones((db_keypoints.shape[0], 1))))
p_w_landmarks_h = np.hstack((p_w_landmarks, np.ones((p_w_landmarks.shape[0], 1))))

dimg = cv2.imread("../data/000000.png", 0)
ddesp = tk.describe_keypoints(dimg, db_keypoints[:, [1, 0]], (descriptor_radius * 2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p_w_landmarks[:, 0], p_w_landmarks[:, 1], p_w_landmarks[:, 2], c='b', marker='^')

for iidx in range(0, 9):
  
  qimg = cv2.imread("../data/00000{}.png".format(iidx), 0)
  
  qharris = tk.harris(qimg, harris_patch_size, harris_kappa)
  qkps = tk.select_keypoints(qharris, no_of_keypoints, non_maxima_supp_radius)
  qkps = np.asarray(qkps)[:, [1, 0]]
  
  qkps_h = np.hstack((qkps, np.ones((qkps.shape[0], 1))))
  qdesp = tk.describe_keypoints(qimg, qkps[:, [1, 0]], descriptor_radius * 2)
  
  matches = tk.match_descriptors(qdesp, ddesp, match_lambda)
  
  mat_query, mat_train = matches
  idcs = np.arange(db_keypoints.shape[0]).tolist()
  rem_train = [idx for idx in idcs if idx not in mat_train]
  
  op_img = tk.draw_matching_points(cv2.cvtColor(qimg, cv2.COLOR_GRAY2BGR),
                                   matches, db_keypoints[:, [1, 0]],
                                   qkps[:, [1, 0]])
  cv2.imshow("t", op_img)
  
  def ransace_dlt(dbkps, dbObj, kmat, msample=7):
    """
    Ransac for DLT to remove outliers
    """
    min_count, hist_inliers = 0, []
    aug, op_idcs = None, None
    for _ in range(2000):
      idcs = np.arange(dbkps.shape[0])
      np.random.shuffle(idcs)
      model_idcs = idcs[:msample]
      test_idcs = idcs[msample:]
      samp_ipts = dbkps[model_idcs]
      samp_wpts = dbObj[model_idcs]
      # samp_ipts = np.linalg.inv(kmat).dot(samp_ipts.T).T
      aug = pnp.estimatePoseDLT(samp_ipts,
                                samp_wpts,
                                kmat)
      test_wpts = dbObj[test_idcs]
      reproj_pts = dc.projectPoints_mat(kmat, aug, test_wpts)
      test_ipts = dbkps[test_idcs]
      proj_err = np.sum((reproj_pts[:, :2] - test_ipts[:, :2])**2, axis=1)
      inliers = proj_err <= 250
      _sum = np.sum(inliers)
      if _sum > min_count:
        ipts = test_ipts[proj_err <= 250]
        wpts = test_wpts[proj_err <= 250]
        op_idcs = test_idcs[proj_err <= 250]
        aug = pnp.estimatePoseDLT(ipts, wpts, kmat)
        min_count = _sum
    is_suc = True
    if min_count > 8:
      aug = pnp.estimatePoseDLT(ipts, wpts, kmat)
    else:
      is_suc = False
      print("Not enough points")
    return aug, op_idcs, is_suc
  
  aug, idcs, is_suc = ransace_dlt(qkps_h[mat_query], p_w_landmarks_h[mat_train], kmat)
  if not is_suc:
    continue
  op_img = tk.draw_matching_points(cv2.cvtColor(qimg, cv2.COLOR_GRAY2BGR),
                                   [[mat_query[i] for i in idcs], [mat_train[i] for i in idcs]],
                                   db_keypoints[:, [1, 0]],
                                   qkps[:, [1, 0]])
  
  print("Rot: {}".format(aug[:3, :3]))
  print("Trans: {}".format(aug[:, 3:]))
  t = aug[:, 3:]
  ax.scatter(t[0], t[1], t[2], c='r', marker="o")
  cv2.imshow("op", op_img)
  key = cv2.waitKey(10)
  if key == 27:
    break

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
