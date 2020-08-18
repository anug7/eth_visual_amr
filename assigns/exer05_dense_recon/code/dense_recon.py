import struct
from heapq import heapify, heappush
import numpy as np
from scipy.optimize import fmin

import cv2


def get_sub_pixel_disp(disp_dict, cur_disp):
    """
    """
    mid = disp_dict[cur_disp]
    if cur_disp + 1 in disp_dict:
        rval = disp_dict[cur_disp + 1]
    else:
        rval = mid - 100
    if cur_disp - 1 in disp_dict:
        lval = disp_dict[cur_disp - 1]
    else:
        rval = mid + 100
    coefs = np.polyfit([cur_disp - 1, cur_disp, cur_disp + 1], [lval, mid, rval], 2)
    return -coefs[1] / (2. * coefs[0])


def fit_poly(xs, ys, cur_min):
    """
    """
    coefs = np.polyfit(xs, ys, 2)
    def func(x):
        return ((x**2) * coefs[0]) +  (x * coefs[1]) + coefs[2]
    op = fmin(func, cur_min)
    return op

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


kmat = np.loadtxt("../data/K.txt").reshape((3, -1)) / 2.0
kmat[2, 2] = 1
#kmat = np.loadtxt("../data/K.txt").reshape((3, -1)) 
poses = np.loadtxt("../data/poses.txt")
poses = np.hstack((poses, np.zeros((poses.shape[0], 1))))
poses = np.hstack((poses, np.zeros((poses.shape[0], 1))))
poses = np.hstack((poses, np.zeros((poses.shape[0], 1))))
poses = np.hstack((poses, np.ones((poses.shape[0], 1))))

no_of_imgs = poses.shape[0]
poses = poses.reshape((no_of_imgs, 4, 4))
kinv = np.linalg.inv(kmat)
cam_to_world = np.asarray([[0, -1, 0],
                           [0, 0, -1], 
                           [1, 0,  0]])
cam_to_world = np.linalg.inv(cam_to_world)
# Given by the KITTI dataset:
baseline = 0.54
oppoints, opcolors = [], []
# Carefully tuned by the TAs:
patch_radius = 5
min_disp, max_disp = 2, 25

for idx in range(100):

    sidx = str(idx).zfill(6) + ".png"
    limg = cv2.imread("../data/left/{}".format(sidx), 0)
    rimg = cv2.imread("../data/right/{}".format(sidx), 0)
    cimg = cv2.imread("../data/left/{}".format(sidx))

    limg = cv2.resize(limg, (620, 188))
    rimg = cv2.resize(rimg, (620, 188))
    cimg = cv2.resize(cimg, (620, 188))
    points = [] 
    disparity = np.zeros_like(limg)
    for x in range(patch_radius + max_disp + 1, limg.shape[1] - patch_radius - 1):
        for y in range(patch_radius + 1, limg.shape[0] - patch_radius - 1):
            lblock = limg[y - patch_radius - 1: y + patch_radius,
                          x - patch_radius - 1: x + patch_radius]
            min_val, values, val_idx = 10000000, [], 0
            heapify(values)
            disp_dict = {}
            for d in range(min_disp, max_disp + 1):
                rblock = rimg[y - patch_radius - 1: y + patch_radius,
                              x - patch_radius - 1 - d: x + patch_radius - d]
                val = np.sum((rblock - lblock)**2)
                heappush(values, val)
                disp_dict[d] = val
                if val < min_val:
                    val_idx = d
                    min_val = val
            deno = float(values[0]) if values[0] > 0 else 200.0
            fact = np.asarray(values[1:4]) / deno
            if fact[0] < 1.5 and fact[1] < 1.5 and fact[2] < 1.5:
                disparity[y, x] = 0
            else:
                if val_idx != min_disp  and val_idx != max_disp:
                    disparity[y, x] = val_idx
                    sub_pixel = get_sub_pixel_disp(disp_dict, val_idx)
                    # build LS solution for Lambda
                    pmat = kinv.dot(np.array([[x, -(x + sub_pixel)], [y, -y], [1, -1]]))
                    lbda = np.linalg.pinv(pmat).dot(np.asarray([baseline, 0, 0]).T)
                    point = lbda[0] * kinv.dot(np.asarray([x, y, 1]).T)
                    points.append(point)
                    opcolors.append(cimg[y, x, :])
    print(idx)
    points = np.asarray(points)
    # points = cam_to_world.dot(points.T).T
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    wpoints = poses[idx].dot(points.T).T[:, :3]
    oppoints.extend(wpoints.tolist())

    #op_img = cv2.applyColorMap(disparity * 5, cv2.COLORMAP_JET)
    #cv2.imshow("op", op_img)
    #cv2.waitKey(1)

write_pointcloud("op.ply", np.asarray(oppoints), np.asarray(opcolors))
