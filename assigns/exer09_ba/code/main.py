
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import helpers as hp


def align_estimate_to_gt(gt, est):
  """
  Align estimate to grounth using Non linear
  Least Squares
  """
  x_start = np.eye(4, 4)
  x = hp.homo_mat_to_twist(x_start)
  # add scale to the vector
  x = np.hstack((x, [1])) # add scale to 1

  def eval_func(tvec, ip, data):
    """
    """
    scale = tvec[6]
    hmat = hp.twist_tot_homo_mat(tvec[:6])
    hmat[:3, :3] *= scale
    ip2 = np.hstack((ip, np.ones((ip.shape[0], 1))))
    op = hmat.dot(ip2.T).T[:, :3]
    err = data - op
    return err.flatten()
    
  out = optimize.leastsq(eval_func, x, args=(est, gt))
  return out


hidden_states = np.loadtxt("../data/hidden_state.txt")
observations = np.loadtxt("../data/observations.txt")
num_frames = 150
K = np.loadtxt("../data/K.txt")
poses = np.loadtxt("../data/poses.txt")

pp_G_C = poses[:, [3, 7, 11]]

hidden_states, observations, pp_G_C = hp.crop_problem(hidden_states,
                                                      observations,
                                                      pp_G_C,
                                                      num_frames)
cropped_hidden_states, cropped_observations, _ = hp.crop_problem(hidden_states,
                                                                 observations,
                                                                 pp_G_C, 4)

T_V_C = hidden_states[: num_frames * 6].reshape(-1, 6)
p_V_C = np.zeros((num_frames, 3))

for i in range(num_frames):
  single_T_V_C = hp.twist_tot_homo_mat(T_V_C[i])
  p_V_C[i, :] = single_T_V_C[:3, 3]

res = align_estimate_to_gt(pp_G_C, p_V_C)[0]
scale, hmat = res[6], hp.twist_tot_homo_mat(res[:6])
hmat[:3, :3] = scale * hmat[:3, :3]

p_G_C = hmat.dot(np.hstack((p_V_C, np.ones((p_V_C.shape[0], 1)))).T).T[:, :3]

plt.plot(pp_G_C[:, 2], -pp_G_C[:, 0], color='g', label='Ground Truth')
plt.plot(p_V_C[:, 2], -p_V_C[:, 0],  color='r', label='Original estimate')
plt.plot(p_G_C[:, 2], -p_G_C[:, 0],  color='b', label='Aligned estimate')
#
plt.legend()
plt.show()
