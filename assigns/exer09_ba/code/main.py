
import numpy as np

import helpers as hp


def align_estimate_to_gt(gt, est):
  """
  Align estimate to grounth using Non linear
  Least Squares
  """




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

