"""
Experiments on Bundle Adjustment
"""

import numpy as np
import scipy.optimize as optimize
from scipy.sparse import lil_matrix
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
  hmat = hp.twist_to_homo_mat(twist[:6])[:3, :]
  hpoints_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
  intm = kmat.dot(hmat)
  out = intm.dot(hpoints_3d.T).T
  out = out / out[:, [2, 2, 2]]
  return out

def run_ba(hstates, aobsers, kmat):
  """
  Run BA on image points and actual positions in 3D
  @param: hstates: image coordinates of points
  @param: aobsers: actual 3D point in the world
  @param: kmat: Intrinsics of the camera
  """
  nframes, nlandmarks = int(aobsers[0]), int(aobsers[1])
  offset, pidx, err = 2, 0, []
  total_3d_indices, total_2d_points = [], []
  for i in range(nframes):
    cur_nlandmarks = int(aobsers[offset])
    lpoints = aobsers[ offset + 1: offset + 1 + cur_nlandmarks * 2].reshape((-1,2))
    # swap row, column to x, y
    lpoints[:, [1, 0]] = lpoints[:, [0, 1]]

    offset += 1 + cur_nlandmarks * 2
    indices = aobsers[offset: offset + cur_nlandmarks].astype('uint') - 1 # make indices 0 based instead of 1 in Matlab

    total_3d_indices.append(indices)
    total_2d_points.append(lpoints)

    offset += cur_nlandmarks

  def peval_func(thstates, t3d_indices, t2d_points, kmat):
    """
    """
    no_frames = len(t2d_points)
    err, ttwists = [], np.asarray(thstates[:no_frames * 6]).reshape((-1, 6))
    _points_3d = thstates[no_frames * 6: ].reshape((-1, 3))

    for twist, c3d_indices, c2d_points in zip(ttwists, t3d_indices, t2d_points):
      pro_points = project_points(twist, _points_3d[c3d_indices], kmat)[:, :2]
      cerr = (pro_points - c2d_points).flatten()
      err.extend(cerr)
    return err

  def calc_jacobian(thstates, t3d_indices):
    """
    Calculates jacobian
    """
    # Refer: https://www.telesens.co/2016/10/13/bundle-adjustment-part-1-jacobians/
    nframes = len(t3d_indices)
    ttl_obs = 0
    for i in range(nframes):
      ttl_obs +=  len(t3d_indices[i])
    rows, sidx = ttl_obs * 2, 0
    #jac = np.zeros((rows, thstates.shape[0])).astype('uint')
    jac = lil_matrix((rows, thstates.shape[0]), dtype=int)
    for i in range(nframes):
      cur_idx, lidx = t3d_indices[i], np.arange(len(t3d_indices[i]))
      for cidx in range(6):
        jac[sidx + lidx * 2, i * 6 + cidx] = 1
        jac[sidx + lidx * 2 + 1, i * 6 + cidx] = 1

      for pidx in range(3):
        jac[sidx + lidx * 2, (nframes * 6) + (cur_idx * 3) + pidx] = 1
        jac[sidx + lidx * 2 + 1, (nframes * 6) + (cur_idx * 3) + pidx] = 1
      sidx += len(cur_idx) * 2

    return jac

  jacob = calc_jacobian(hstates, total_3d_indices)
  # peval_func(hstates, total_3d_indices, total_2d_points, kmat)
  res = optimize.least_squares(peval_func, hstates, ftol=1e-4,
                               x_scale='jac', method="trf",
                               jac_sparsity=jacob, verbose=2, max_nfev=20,
                               args=(total_3d_indices, total_2d_points, kmat))
  return res.x

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

#plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
#plt.plot(p_V_C[:, 2], -p_V_C[:, 0], color='r', label='Original estimate')
#plt.plot(p_G_C[:, 2], -p_G_C[:, 0], color='b', label='Aligned estimate')

#plt.legend()
#plt.show()

def plot_map(_plt, hhidden_states, hobservations):
  """
  """
  nnos = int(hobservations[0])
  p_W_frames = np.zeros((nnos, 3))
  T_W_frames = hhidden_states[: nnos * 6].reshape((-1, 6))
  p_W_landmarks = hhidden_states[nnos * 6: ].reshape((-1, 3))
  for n in range(nnos):
    T_W_frame = hp.twist_to_homo_mat(T_W_frames[n])
    p_W_frames[n, :] = T_W_frame[:3, 3]
  _plt.scatter(p_W_landmarks[:, 2], -p_W_landmarks[:, 0])
  _plt.scatter(p_W_frames[:, 2], -p_W_frames[:, 0])

#no_of_frames = 4
#res = run_ba(cropped_hidden_states, cropped_observations, K)
#plot_map(plt, res, cropped_observations)
#plt.xlim([0, 20])
#plt.ylim([-5, 5])

no_of_frames = NUM_FRAMES
res = run_ba(hidden_states, observations, K)
plot_map(plt, hidden_states, observations)
plt.xlim([0, 40])
plt.ylim([-10, 10])

#T_V_C = res[: NUM_FRAMES * 6].reshape(-1, 6)
#p_V_C = np.zeros((NUM_FRAMES, 3))
#
#for i in range(NUM_FRAMES):
#  single_T_V_C = hp.twist_to_homo_mat(T_V_C[i])
#  p_V_C[i, :] = single_T_V_C[:3, 3]
#
#p_G_C = align_estimate_to_gt(pp_G_C, p_V_C)
#plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
#plt.plot(p_V_C[:, 2], -p_V_C[:, 0], color='r', label='Original estimate')
#plt.plot(p_G_C[:, 2], -p_G_C[:, 0], color='b', label='Aligned estimate')

plt.legend()
plt.show()

