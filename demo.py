# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import os
import sys
import numpy as np
from ocmesher import OcMesher

import vnoise
noise = vnoise.Noise()

def f(XYZ):
    scale = 2
    h = noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4)

    return XYZ[:, 2] - h


cam_poses = [np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 3],
    [0, 0, 0, 1],
])]
Ks = [np.array([
    [2000, 0, 640],
    [0, 2000, 360],
    [0, 0, 1]
])]
Hs = [720]
Ws = [1280]

mesher = OcMesher((cam_poses, Ks, Hs, Ws), pixels_per_cube=16)
meshes, in_view_tags = mesher([f])
os.makedirs("results", exist_ok=True)
meshes[0].export("results/demo.obj")