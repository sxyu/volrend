"""
This script extracts 4x4 matrix poses for the headless renderer from the 
nerf_synthetic json files.

Usage: python extract_test_poses.py <nerf_synthetic_root>
"""
from glob import glob
import json
import os.path as osp
import os
import numpy as np
import sys

transforms = glob(osp.join(sys.argv[1], '*', 'transforms_test.json'))
for transform_path in transforms:
    print(transform_path)
    root_dir = osp.dirname(transform_path)
    poses_dir = osp.join(root_dir, 'pose')
    os.makedirs(poses_dir, exist_ok=True)
    with open(transform_path, 'r') as f:
        j = json.load(f)
        for frame in j['frames']:
            basename = osp.basename(frame['file_path'])
            mtx = np.array(frame['transform_matrix'])
            np.savetxt(osp.join(poses_dir, basename + '.txt'), mtx)
        hW = 400
        focal = hW / np.tan(0.5 * j['camera_angle_x'])
        K = np.diag([focal, focal, 1, 1])
        K[:2, 2] = [hW, hW]
        np.savetxt(osp.join(root_dir, 'intrinsics.txt'), K)

