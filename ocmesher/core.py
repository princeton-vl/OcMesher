# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from pathlib import Path
import sys
import gin
import numpy as np
import trimesh
from tqdm import tqdm

from .utils.interface import AC, POINTER, AsDouble, AsFloat, AsInt, AsBool, c_bool, c_double, c_float, c_int32, load_cdll, register_func
from .utils.timer import Timer

@gin.configurable
class OcMesher:
    def __init__(self,
        cameras,
        bounds,
        pixels_per_cube=8,
        inv_scale=10,
        min_dist=1,
        memory_limit_mb=1000,
        bisection_iters=15,
        enclosed=True,
        simplify_occluded=True,
        visible_relax_iter=2,
        coarse_count=500000,
    ):
        dll = load_cdll(str(Path(__file__).parent.resolve()/"lib"/"core.so"))
        self.float_type = c_double
        self.np_float_type = np.float64
        self.AF = AsDouble
        self.sdf_float_type = c_float
        self.sdf_np_float_type = np.float32
        self.sdf_AF = AsFloat
        self.bounds = bounds
        self.memory_limit_mb = memory_limit_mb

        cam_poses, Ks, Hs, Ws = cameras
        self.n_cameras = len(cam_poses)
        self.cameras = np.zeros(23 * self.n_cameras, dtype=self.np_float_type)
        for i in range(self.n_cameras):
            self.cameras[23 * i: 23 * (i+1)] = np.concatenate([
                np.linalg.inv(cam_poses[i])[:3, :4].reshape(-1),
                Ks[i].reshape(-1), [Hs[i]], [Ws[i]]
            ]).astype(self.np_float_type)
        
        self.inview_pixels_per_cube = self.np_float_type(pixels_per_cube)
        self.inv_scale = self.np_float_type(inv_scale)
        self.min_dist = self.np_float_type(min_dist)

        self.center = np.array([(bounds[0]+bounds[1]) / 2, (bounds[2]+bounds[3]) / 2, (bounds[4]+bounds[5]) / 2], self.np_float_type)
        self.size = self.np_float_type(max(max(bounds[1] - bounds[0], bounds[3] - bounds[2]), bounds[5] - bounds[4]) * 1.1)

        self.bisection_iters = bisection_iters
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        register_func(self, dll, "run_coarse", [
            POINTER(self.float_type), self.float_type,
            c_int32, POINTER(self.float_type),
            self.float_type, self.float_type, self.float_type,
            c_int32, c_int32, c_int32,
        ], c_int32)
        register_func(self, dll, "fine_group", [], c_int32)
        register_func(self, dll, "fine_iteration", [POINTER(self.sdf_float_type)], c_int32)
        register_func(self, dll, "fine_iteration_output", [POINTER(self.float_type)])
        register_func(self, dll, "vis_filter", [c_bool, c_int32], c_int32)
        register_func(self, dll, "final_iteration", [], c_int32)
        register_func(self, dll, "final_iteration_occluded", [], c_int32)
        register_func(self, dll, "final_iteration2", [POINTER(self.float_type)])
        register_func(self, dll, "final_iteration3", [POINTER(self.sdf_float_type)], c_int32)
        register_func(self, dll, "final_iteration3_occluded", [POINTER(self.sdf_float_type)])
        register_func(self, dll, "final_remaining", [POINTER(c_int32)])
        register_func(self, dll, "get_verts_center", [c_int32, POINTER(self.float_type)])
        register_func(self, dll, "update_verts", [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)])
        register_func(self, dll, "get_lr_verts", [c_int32, POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "finalize_verts", [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)])
        register_func(self, dll, "construct_faces", [c_int32, POINTER(self.float_type), POINTER(c_int32)])
        register_func(self, dll, "get_extra_verts_center", [POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "update_extra_verts", [
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type),
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type),
            POINTER(self.float_type), POINTER(self.float_type),
        ])
        register_func(self, dll, "get_lr_extra_verts", [POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "finalize_extra_verts", [
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type),
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type),
        ])
        register_func(self, dll, "get_faces", [POINTER(c_int32)])
        register_func(self, dll, "get_in_view_tag", [c_int32, POINTER(c_bool)])


    def kernel_caller(self, kernels, XYZ_all):
        n_XYZ = len(XYZ_all)
        if n_XYZ == 0: return np.zeros((0, len(kernels)), dtype=self.sdf_np_float_type)
        step = 10000000
        sdfs = []
        for i in range(0, n_XYZ, step):
            XYZ = XYZ_all[i: i+step]
            sdfs_i = []
            if self.enclosed:
                out_bound = np.zeros(len(XYZ), dtype=bool)
                for c in range(3):
                    out_bound |= XYZ[:, c] <= self.bounds[c*2]
                    out_bound |= XYZ[:, c] >= self.bounds[c*2+1]
            for kernel in kernels:
                sdf = kernel(XYZ)
                if self.enclosed: sdf[out_bound] = 1
                sdfs_i.append(sdf)
            sdfs.append(np.stack(sdfs_i, -1).astype(self.sdf_np_float_type))
        return np.concatenate(sdfs, 0)
    
    def __call__(self, kernels):
        n_elements = len(kernels)
        # octree only considering cameras, not sdf
        with Timer("coarse step part1"):
            n_blocks = self.run_coarse(
                self.AF(self.center), self.size,
                self.n_cameras, self.AF(self.cameras),
                self.inview_pixels_per_cube, 
                self.inv_scale, self.min_dist,
                self.coarse_count, self.memory_limit_mb, n_elements
            )
        # start considering sdf
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = self.fine_group()
                if inc == 0: break
                pbar.update(inc)
                n = self.fine_iteration(POINTER(self.sdf_float_type)())
                while n > 0:
                    positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                    self.fine_iteration_output(self.AF(positions))
                    sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1))
                    n = self.fine_iteration(self.sdf_AF(sdf))
        with Timer("filter visible blocks"):
            n_vis_block = self.vis_filter(self.simplify_occluded, self.visible_relax_iter)

        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            nv = np.zeros(1, dtype=np.int32)
            while True:
                n = self.final_iteration(AsInt(nv))
                if n == 0: break
                positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                self.final_iteration2(self.AF(positions))
                sdf = AC(self.kernel_caller(kernels, positions))
                inc = self.final_iteration3(self.sdf_AF(sdf))
                pbar.update(inc)
            n = self.final_iteration_occluded(AsInt(nv))
            if n != 0:
                positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                self.final_iteration2(self.AF(positions))
                sdf = AC(self.kernel_caller(kernels, positions))
                self.final_iteration3_occluded(self.sdf_AF(sdf))
            nv = np.zeros(n_elements, dtype=np.int32)
            self.final_remaining(AsInt(nv))
            del positions, sdf
        
        with Timer("construct mesh"):
            meshes = []
            in_view_tags = []
            for e in range(n_elements):
                k_e = kernels[e:e+1]
                centers = np.zeros((nv[e], 3), dtype=self.np_float_type)
                self.get_verts_center(e, self.AF(centers))
                center_sdf = self.kernel_caller(k_e, centers)
                cubes = AC(np.zeros((nv[e] * 8, 3), dtype=self.np_float_type))
                self.update_verts(e, POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(), self.AF(cubes))
                for _ in tqdm(range(self.bisection_iters)):
                    sdf = self.kernel_caller(k_e, cubes)
                    self.update_verts(e, self.sdf_AF(AC(sdf)), self.sdf_AF(AC(center_sdf)), self.AF(cubes))
                cubes_r = AC(np.zeros((nv[e] * 8, 3), dtype=self.np_float_type))
                self.get_lr_verts(e, self.AF(cubes), self.AF(cubes_r))
                sdf_l = self.kernel_caller(k_e, cubes)
                sdf_r = self.kernel_caller(k_e, cubes_r)
                del cubes, cubes_r, centers, center_sdf
                vertices = np.zeros((nv[e], 3), dtype=self.np_float_type)
                self.finalize_verts(e, self.sdf_AF(sdf_l), self.sdf_AF(sdf_r), self.AF(vertices))
                del sdf_l, sdf_r
                cnts = np.zeros(3, dtype=np.int32)
                self.construct_faces(e, self.AF(vertices), AsInt(cnts))
                nve, nvf, nf = cnts
                edge_vertices_c = AC(np.zeros((nve, 3), dtype=self.np_float_type))
                face_vertices_c = AC(np.zeros((nvf, 3), dtype=self.np_float_type))
                self.get_extra_verts_center(self.AF(edge_vertices_c), self.AF(face_vertices_c))
                ecenter_sdf = self.kernel_caller(k_e, edge_vertices_c)
                fcenter_sdf = self.kernel_caller(k_e, face_vertices_c)
                edge_vertices_lr = AC(np.zeros((nve * 2, 3), dtype=self.np_float_type))
                face_vertices_lr = AC(np.zeros((nvf * 4, 3), dtype=self.np_float_type))
                self.update_extra_verts(
                    POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(),
                    POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(),
                    self.AF(edge_vertices_lr), self.AF(face_vertices_lr),
                )
                for _ in range(self.bisection_iters):
                    e_sdf = self.kernel_caller(k_e, edge_vertices_lr)
                    f_sdf = self.kernel_caller(k_e, face_vertices_lr)
                    self.update_extra_verts(
                        self.sdf_AF(e_sdf), self.sdf_AF(f_sdf),
                        self.sdf_AF(ecenter_sdf), self.sdf_AF(fcenter_sdf),
                        self.AF(edge_vertices_lr), self.AF(face_vertices_lr),
                    )
                del edge_vertices_c, face_vertices_c, ecenter_sdf, fcenter_sdf
                edge_vertices_r = AC(np.zeros((nve * 2, 3), dtype=self.np_float_type))
                face_vertices_r = AC(np.zeros((nvf * 4, 3), dtype=self.np_float_type))
                self.get_lr_extra_verts(self.AF(edge_vertices_lr), self.AF(edge_vertices_r), self.AF(face_vertices_lr), self.AF(face_vertices_r))
                esdf_l = self.kernel_caller(k_e, edge_vertices_lr)
                esdf_r = self.kernel_caller(k_e, edge_vertices_r)
                fsdf_l = self.kernel_caller(k_e, face_vertices_lr)
                fsdf_r = self.kernel_caller(k_e, face_vertices_r)
                del edge_vertices_lr, edge_vertices_r, face_vertices_lr, face_vertices_r
                edge_vertices = np.zeros((nve, 3), dtype=self.np_float_type)
                face_vertices = np.zeros((nvf, 3), dtype=self.np_float_type)
                self.finalize_extra_verts(self.sdf_AF(esdf_l), self.sdf_AF(esdf_r), self.AF(edge_vertices), self.sdf_AF(fsdf_l), self.sdf_AF(fsdf_r), self.AF(face_vertices))
                del esdf_l, esdf_r, fsdf_l, fsdf_r
                faces = AC(np.zeros((nf, 3), dtype=np.int32))
                self.get_faces(AsInt(faces))
                vertices = np.concatenate((vertices, edge_vertices, face_vertices))
                in_view_tag = np.zeros(vertices.shape[0], dtype=bool)
                self.get_in_view_tag(e, AsBool(in_view_tag))
                in_view_tags.append(in_view_tag)
                meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
                print(f"element {e} has vertices #{meshes[-1].vertices.shape[0]} faces #{meshes[-1].faces.shape[0]}")
        return meshes, in_view_tags