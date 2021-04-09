
import numpy as np
from scipy.linalg import logm, expm


def mat2cross(se_mat):
  """
  Converts mat repo to vector rep
  for transformation
  """
  return np.asarray([-se_mat[1, 2], se_mat[0, 2], -se_mat[0, 1]])


def cross2mat(vec):
  """
  COnvert vector rep to matrix repo for
  transformation
  """
  return np.asarray([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def homo_mat_to_twist(homo_mat):
  """
  Converts 4x4 homogeneous matrix to
  twist coordinates
  """
  se_mat = logm(homo_mat)
  v = se_mat[0:3, 3]
  w = mat2cross(se_mat[0:3, 0:3])

  return np.hstack((w.T, v.T))


def twist_to_homo_mat(twist):
  """
  Converts twist vector to transformation
  homogeneous matrix
  """
  v = twist[0:3]
  w = twist[3:]
  se_mat = np.zeros((4, 4))
  se_mat[0:3, 0:3] = cross2mat(w)
  se_mat[0:3, 3] = v
  return expm(se_mat)


def crop_problem(hidden_state, observations, gt, num_frames):
  """
  Extract data from files into meaningful format
  """
  total_frames = int(observations[0])
  assert num_frames < total_frames
  obs_id = 2
  for i in range(num_frames):
    num_observations = int(observations[obs_id])
    if i == num_frames - 1:
      cropped_num_landmarks = max(observations[range(obs_id + 1 +
              num_observations * 2, obs_id + num_observations * 3 + 1)])
      cropped_num_landmarks = int(cropped_num_landmarks)
    obs_id = obs_id + num_observations * 3 + 1
  
  cropped_hidden_state = np.hstack((hidden_state[: 6 * num_frames],
                                    hidden_state[range(6 * total_frames, 6 *
                                                 total_frames + 3 *
                                                 cropped_num_landmarks)]))
  cropped_observations = np.hstack(([num_frames], [cropped_num_landmarks],
                                   observations[range(2, obs_id)]))
  cropped_gt = gt[: num_frames]
  return cropped_hidden_state, cropped_observations, cropped_gt
