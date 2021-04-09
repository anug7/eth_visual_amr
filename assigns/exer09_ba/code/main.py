"""
Experiments on Bundle Adjustment
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import helpers as hp


def align_estimate_to_gt(grnd_trth, est):
  """
  Align estimate to grounth using Non linear
  Least Squares
  """
  x_start = np.eye(4, 4)
  xes = hp.homo_mat_to_twist(x_start)
  # add scale to the vector
  xes = np.hstack((xes, [1])) # add scale to 1

  def eval_func(tvec, x_ip, data):
    """
    """
    scale = tvec[6]
    hmat = hp.twist_to_homo_mat(tvec[:6])
    hmat[:3, :3] *= scale
    hip = np.hstack((x_ip, np.ones((x_ip.shape[0], 1))))
    _out = hmat.dot(hip.T).T[:, :3]
    err = data - _out
    return err.flatten()

  res = optimize.leastsq(eval_func, xes, args=(est, grnd_trth))[0]
  scale, hmat = res[6], hp.twist_to_homo_mat(res[:6])
  hmat[:3, :3] = scale * hmat[:3, :3]

  out = hmat.dot(np.hstack((est, np.ones((est.shape[0], 1)))).T).T[:, :3]
  return out


def project_points(twist, points_3d, kmat):
  """
  Projects 3D points on to image plane
  """
  hmat = hp.twist_to_homo_mat(twist)
  hpoints_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
  intm = hmat.dot(hpoints_3d.T)
  out = kmat.dot(intm).T
  return out

def run_ba(hstates, aobsers, kmat):
  """
  Run BA on image points and actual positions in 3D
  @param: hstates: image coordinates of points
  @param: aobsers: actual 3D point in the world
  @param: kmat: Intrinsics of the camera
  """
  nframes, nlandmarks = int(aobsers[0]), int(aobsers[1])
  offset, pidx = 2, 0
  twists = hstates[: nframes * 6].reshape((-1, 6))
  points_3d = hstates[nframes * 6: ].reshape((-1, 3))
  for i in range(nframes):
    cur_nlandmarks = int(aobsers[offset])
    lpoints = aobsers[ offset + 1: offset + 1 + cur_nlandmarks * 2].reshape((-1,2))
    # swap row, column to x, y
    lpoints[:, [1, 0]] = lpoints[:, [0, 1]]
    lpoints = lpoints.flatten()

    offset += 1 + cur_nlandmarks * 2
    indices = aobsers[offset: offset + cur_nlandmarks].astype('uint')
    cur_3d_points = points_3d[indices]
    pro_points = project_points(twists[i], cur_3d_points, kmat)

    offset += cur_nlandmarks


hidden_states = np.loadtxt("../data/hidden_state.txt")
observations = np.loadtxt("../data/observations.txt")
NUM_FRAMES = 150
K = np.loadtxt("../data/K.txt")
poses = np.loadtxt("../data/poses.txt")
pp_G_C = poses[:, [3, 7, 11]]

hidden_states, observations, pp_G_C = hp.crop_problem(hidden_states,
                                                      observations,
                                                      pp_G_C,
                                                      NUM_FRAMES)
cropped_hidden_states, cropped_observations, _ = hp.crop_problem(hidden_states,
                                                                 observations,
                                                                 pp_G_C, 4)

T_V_C = hidden_states[: NUM_FRAMES * 6].reshape(-1, 6)
p_V_C = np.zeros((NUM_FRAMES, 3))

for i in range(NUM_FRAMES):
  single_T_V_C = hp.twist_to_homo_mat(T_V_C[i])
  p_V_C[i, :] = single_T_V_C[:3, 3]

p_G_C = align_estimate_to_gt(pp_G_C, p_V_C)

plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
plt.plot(p_V_C[:, 2], -p_V_C[:, 0], color='r', label='Original estimate')
plt.plot(p_G_C[:, 2], -p_G_C[:, 0], color='b', label='Aligned estimate')
#

#plt.legend()
#plt.show()
run_ba(cropped_hidden_states, cropped_observations, K)
