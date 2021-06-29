"""
This script extracts camera pose drawlists from nerf_synthetic json files,
which you can use with ./volrend --draw <dir>/cams.draw.npz
or using "Load Local" in the web version (look in "Layers" to enable if hidden)

Usage: python extract_cams_drawlist.py <nerf_synthetic_root>
"""
from glob import glob
import json
import os.path as osp
import os
import numpy as np
from scipy.spatial.transform import Rotation
import sys

transforms = glob(osp.join(sys.argv[1], '*', 'transforms_train.json'))
for transform_path in transforms:
    root_dir = osp.dirname(transform_path)
    out_path = osp.join(root_dir, osp.basename(root_dir) + "_cams.draw.npz")
    print(transform_path, 'to', out_path)
    poses_dir = osp.join(root_dir, 'pose')
    os.makedirs(poses_dir, exist_ok=True)
    with open(transform_path, 'r') as f:
        j = json.load(f)
        mtx = np.array([frame['transform_matrix'] for frame in j['frames']])
        t = mtx[:, :3, 3]
        R = mtx[:, :3, :3]
        r = Rotation.from_matrix(R).as_rotvec()
        hW = 400
        focal = hW / np.tan(0.5 * j['camera_angle_x'])
        np.savez_compressed(out_path,
                cameras="camerafrustum",
                cameras__t=t,
                cameras__r=r,
                cameras__focal_length=focal,
                cameras_image_width=hW*2,
                cameras_image_height=hW*2,
                cameras_z=-0.25,
                cameras_color=np.array([1.0, 0.5, 0.0]),
                )



