import struct
from heapq import heapify, heappush
import numpy as np
import cv2


def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
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

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()



limg = cv2.imread("../data/left/000000.png", 0)
rimg = cv2.imread("../data/right/000000.png", 0)
cimg = cv2.imread("../data/left/000000.png")

# Given by the KITTI dataset:
baseline = 0.54
kmat = np.loadtxt("../data/K.txt").reshape((3, -1))
kinv = np.linalg.inv(kmat)
points, colors = [], []

# Carefully tuned by the TAs:
patch_radius = 5
min_disp, max_disp = 5, 50
disparity = np.zeros_like(limg)
for x in range(patch_radius + max_disp + 1, limg.shape[1] - patch_radius - 1):
    for y in range(patch_radius + 1, limg.shape[0] - patch_radius - 1):
        lblock = limg[y - patch_radius - 1: y + patch_radius,
                      x - patch_radius - 1: x + patch_radius]
        min_val, values, val_idx = 10000000, [], 0
        heapify(values)
        for d in range(min_disp, max_disp + 1):
            rblock = rimg[y - patch_radius - 1: y + patch_radius,
                          x - patch_radius - 1 - d: x + patch_radius - d]
            val = np.sum((rblock - lblock)**2)
            heappush(values, val)
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
                pmat = kinv.dot(np.array([[x, -(x + val_idx)], [y, -y], [1, -1]]))
                lbda = np.linalg.pinv(pmat).dot(np.asarray([baseline, 0, 0]).T)
                point = lbda[0] * kinv.dot(np.asarray([x, y, 1]).T)
                points.append(point)
                colors.append(cimg[y, x, :])

write_pointcloud("op.ply", np.asarray(points), np.asarray(colors))
op_img = cv2.applyColorMap(disparity * 5, cv2.COLORMAP_JET)
cv2.imshow("op", op_img)
cv2.waitKey(0)
