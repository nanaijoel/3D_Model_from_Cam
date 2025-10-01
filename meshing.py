# meshing.py
import os
from typing import Optional
import numpy as np

import open3d as o3d

# Basic: save point cloud

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz)
        d, _ = tree.query(points_xyz[idx], k=2)
        return float(np.median(d[:, 1]))
    except Exception:
        bb = points_xyz.max(0) - points_xyz.min(0)
        diag = float(np.linalg.norm(bb))
        return max(1e-3, diag / max(300.0, len(points_xyz) ** (1 / 3)))

def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None):
    """
    Saves a point cloud as PLY (with optional light downsampling/outlier filtering).
    """
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if len(pts) >= int(filter_min_points):
        nn_med = _median_nn_distance(pts)
        voxel = max(1e-4, 0.5 * nn_med)  # adaptive; not too coarse to preserve thin structures
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            # gentle – try to preserve thin structures
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=4, std_ratio=0.8)
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path

# Poisson/Alpha: helper functions

def _diag_from_pcd(pcd):
    bb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(bb.get_extent(), dtype=float)
    return float(np.linalg.norm(ext))

def _estimate_normals_o3d(pcd, k: int = 40):
    diag = _diag_from_pcd(pcd)
    radius = max(1e-3, 0.02 * diag)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(k))
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(int(k))
    except Exception:
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))

def _mesh_poisson(pcd, depth=10, scale=1.1, linear_fit=True):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=float(scale), linear_fit=bool(linear_fit)
    )
    return mesh, np.asarray(dens, float)

def _mesh_alpha(pcd, alpha):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))

def _mesh_cleanup(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh

def _mesh_crop_bbox(mesh, pcd, expand_ratio: float):
    if expand_ratio <= 0:
        return mesh
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.0 + float(expand_ratio), bb.get_center())
    try:
        return mesh.crop(bb)
    except Exception:
        # Fallback: manually clip against the AABB
        mins = bb.get_min_bound()
        maxs = bb.get_max_bound()
        verts = np.asarray(mesh.vertices)
        inside = np.all((verts >= mins) & (verts <= maxs), axis=1)
        mesh.remove_vertices_by_mask(~inside)
        return mesh

def _trim_by_density(mesh, densities: np.ndarray, q: float):
    if densities is None or len(densities) != len(mesh.vertices) or q <= 0:
        return mesh
    thr = float(np.quantile(densities, np.clip(q, 0.0, 0.5)))
    mask = densities < thr
    mesh.remove_vertices_by_mask(mask)
    return mesh

def _mesh_smooth(mesh, iters=8):
    if iters <= 0:
        return mesh
    try:
        return mesh.filter_smooth_taubin(number_of_iterations=int(iters))
    except Exception:
        return mesh.filter_smooth_simple(number_of_iterations=int(iters))

def _mesh_simplify(mesh, target_tris=0):
    if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
        return mesh.simplify_quadric_decimation(int(target_tris))
    return mesh

def _transfer_vertex_colors(mesh, pcd, k=1):
    if not pcd.has_colors():
        return mesh
    tree = o3d.geometry.KDTreeFlann(pcd)
    mverts = np.asarray(mesh.vertices)
    pcols = np.asarray(pcd.colors)
    cols = np.zeros((len(mverts), 3), dtype=np.float64)
    for i, v in enumerate(mverts):
        _, idx, _ = tree.search_knn_vector_3d(v, max(1, int(k)))
        cols[i] = np.mean(pcols[idx], axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    return mesh

# Main function: Poisson/Alpha from PLY

def reconstruct_solid_mesh_from_ply(
    ply_in: str,
    mesh_out: str,
    *,
    method: str = "poisson",      # "poisson" | "alpha"
    depth: int = 12,
    scale: float = 1.1,
    no_linear_fit: bool = False,
    dens_quantile: float = 0.02,
    alpha: float = 0.0,
    bbox_expand: float = 0.02,
    pre_filter: bool = False,
    pre_filter_neighbors: int = 12,
    pre_filter_std: float = 2.0,
    voxel: float = 0.0,
    normals_k: int = 40,
    smooth: int = 10,
    simplify: int = 100000,
    color_transfer: bool = False,
    on_log=None,
    on_progress=None,
) -> str:
    """
    Builds a watertight mesh directly from a PLY point cloud.
    """
    on_log and on_log(f"[mesh] solid: load cloud -> {ply_in}")
    pcd = o3d.io.read_point_cloud(ply_in)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("Loaded point cloud is empty.")

    if pre_filter:
        try:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=max(8, int(pre_filter_neighbors)),
                std_ratio=float(pre_filter_std)
            )
        except Exception:
            pass
    if float(voxel) > 0:
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass

    _estimate_normals_o3d(pcd, k=int(normals_k))

    if method == "poisson":
        mesh, dens = _mesh_poisson(
            pcd, depth=int(depth), scale=float(scale), linear_fit=(not bool(no_linear_fit))
        )
        if float(dens_quantile) > 0:
            mesh = _trim_by_density(mesh, dens, float(dens_quantile))
    elif method == "alpha":
        if float(alpha) <= 0.0:
            nn = _median_nn_distance(np.asarray(pcd.points))
            alpha = 2.5 * float(nn)
        mesh = _mesh_alpha(pcd, float(alpha))
    else:
        raise ValueError(f"Unknown method: {method}")

    mesh = _mesh_crop_bbox(mesh, pcd, expand_ratio=float(bbox_expand))
    mesh = _mesh_cleanup(mesh)
    if int(smooth) > 0:
        mesh = _mesh_smooth(mesh, iters=int(smooth))
    if int(simplify) > 0:
        mesh = _mesh_simplify(mesh, target_tris=int(simplify))
    mesh.compute_vertex_normals()
    if bool(color_transfer):
        mesh = _transfer_vertex_colors(mesh, pcd, k=1)

    os.makedirs(os.path.dirname(mesh_out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(mesh_out, mesh, write_vertex_colors=bool(color_transfer))
    if not ok:
        raise RuntimeError(f"Failed to write mesh: {mesh_out}")
    on_log and on_log(f"[mesh] solid saved -> {mesh_out} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")
    on_progress and on_progress(100, "Watertight Mesh")
    return mesh_out

