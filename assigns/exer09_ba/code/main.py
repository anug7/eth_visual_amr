"""
Experiments on Bundle Adjustment
"""

import numpy as np
import scipy.optimize as optimize
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

import cv2

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
  hmat = hp.twist_to_homo_mat(twist[:6])
  hmat = np.linalg.inv(hmat)[:3, :]
  points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
  intm = hmat.dot(points_3d.T).T
  out = kmat.dot(intm.T).T
  out /= out[:, [2, 2, 2]]
  return out[:, :2]


def error_function(hstates, observations, kmat):
  """
  """
  err, sidx = [], 2
  nframes = int(observations[0])
  poses = hstates[: nframes * 6].reshape((-1, 6))
  p3d_points = hstates[nframes * 6: ].reshape((-1, 3))
  for i in range(nframes):
    cur_obs = int(observations[sidx])
    sidx += 1
    p2d_points = observations[sidx: sidx + 2 * cur_obs].reshape((-1, 2))
    p2d_points[:, [1, 0]] = p2d_points[:, [0, 1]]

    sidx += 2 * cur_obs

    # - 1 for 0 based index
    idcs = observations[sidx: sidx + cur_obs].astype('uint32') - 1

    re_points = project_points(poses[i], p3d_points[idcs], kmat)
    cur_err = (re_points - p2d_points).flatten()

    sidx += cur_obs
    err.extend(cur_err)

  return err


def calc_jacobian(hstates, observations):
  """
  Calculates jacobian
  """
  # Refer: https://www.telesens.co/2016/10/13/bundle-adjustment-part-1-jacobians/
  nframes = int(observations[0])
  cols = hstates.shape[0]
  indcs, sidx, rows = [], 2, 0
  for i in range(nframes):
    cur_nlandmarks = int(observations[sidx])
    rows += cur_nlandmarks
    sidx += cur_nlandmarks * 2 + 1

    #get index of observations wrt landmarks
    cidcs = observations[sidx: sidx + cur_nlandmarks].astype('uint')
    indcs.append(cidcs)
    sidx += cur_nlandmarks
  rows *= 2 # 2 per observation
  jacob = lil_matrix((rows, cols), dtype=np.uint32)
  # jacob = np.zeros((rows, cols), dtype=np.uint16)

  sidx = 0
  for i in range(nframes):
    # - 1 for 0 based inex
    p3d_idcs, p2d_idcs = indcs[i] - 1, np.arange(len(indcs[i]))

    for cidx in range(6):
      jacob[p2d_idcs * 2 + sidx, i*6 + cidx] = 1
      jacob[p2d_idcs * 2 + 1 + sidx, i*6 + cidx] = 1

    for pidx in range(3):
      jacob[p2d_idcs * 2 + sidx, nframes*6 + p3d_idcs*3 + pidx] = 1
      jacob[p2d_idcs * 2 + 1 + sidx, nframes*6 + p3d_idcs*3 + pidx] = 1

    sidx += 2 * len(indcs[i])

  return jacob


def plot_map(_plt, hhidden_states, hobservations, pose_only=False):
  """
  """
  nnos = int(hobservations[0])
  p_W_frames = np.zeros((nnos, 3))
  T_W_frames = hhidden_states[: nnos * 6].reshape((-1, 6))
  p_W_landmarks = hhidden_states[nnos * 6: ].reshape((-1, 3))
  for n in range(nnos):
    T_W_frame = hp.twist_to_homo_mat(T_W_frames[n])
    p_W_frames[n, :] = T_W_frame[:3, 3]
  if not pose_only:
    _plt.scatter(p_W_landmarks[:, 2], -p_W_landmarks[:, 0])
  _plt.scatter(p_W_frames[:, 2], -p_W_frames[:, 0])


def run_ba(hstates, obsers, kmat):
  """
  Run BA on image points and actual positions in 3D
  @param: hstates: image coordinates of points
  @param: aobsers: actual 3D point in the world
  @param: kmat: Intrinsics of the camera
  """
  jacob = calc_jacobian(hstates, obsers)
  x0 = hstates
  # err = error_function(hstates, obsers, kmat)
  res = optimize.least_squares(error_function, x0,
                               x_scale='jac', method="trf",
                               jac_sparsity=jacob, verbose=2,
                               max_nfev=100,
                               args=(obsers, kmat))
  return res.x

if __name__ == "__main__":
  hidden_states = np.loadtxt("../data/hidden_state.txt")
  observations = np.loadtxt("../data/observations.txt")
  NUM_FRAMES = 150
  K = np.loadtxt("../data/K.txt")
  poses = np.loadtxt("../data/poses.txt")
  pp_G_C = poses[:, [3, 7, 11]]

  test_no_frame = 8
  hidden_states, observations, pp_G_C = hp.crop_problem(hidden_states,
                                                        observations,
                                                        pp_G_C,
                                                        NUM_FRAMES)
  cropped_hidden_states, cropped_observations, _ = hp.crop_problem(hidden_states,
                                                                   observations,
                                                                   pp_G_C,
                                                                   test_no_frame)

  T_V_C = hidden_states[: NUM_FRAMES * 6].reshape(-1, 6)
  p_V_C = np.zeros((NUM_FRAMES, 3))

  for i in range(NUM_FRAMES):
    single_T_V_C = hp.twist_to_homo_mat(T_V_C[i])
    p_V_C[i, :] = single_T_V_C[:3, 3]

  p_G_C = align_estimate_to_gt(pp_G_C, p_V_C)

  #plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
  #plt.plot(p_V_C[:, 2], -p_V_C[:, 0], color='r', label='Original estimate')
  #plt.plot(p_G_C[:, 2], -p_G_C[:, 0], color='b', label='Aligned estimate')

  #plt.legend()
  #plt.show()

  #res = run_ba(cropped_hidden_states, cropped_observations, K)
  #plot_map(plt, res, cropped_observations)
  #plt.xlim([0, 20])
  #plt.ylim([-5, 5])
  
  no_of_frames = NUM_FRAMES
  no_of_frames = test_no_frame
  res = run_ba(hidden_states, observations, K)
  
  T_V_C = res[: NUM_FRAMES * 6].reshape(-1, 6)
  p_V_C = np.zeros((NUM_FRAMES, 3))

  for i in range(NUM_FRAMES):
    single_T_V_C = hp.twist_to_homo_mat(T_V_C[i])
    p_V_C[i, :] = single_T_V_C[:3, 3]

  opt_p_G_C = align_estimate_to_gt(pp_G_C, p_V_C)
  plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
  plt.plot(p_G_C[:, 2], -p_G_C[:, 0], color='r', label='Aligned estimate')
  plt.plot(opt_p_G_C[:, 2], -opt_p_G_C[:, 0], color='b', label='Opt estimate')

  plt.legend()
  plt.show()
