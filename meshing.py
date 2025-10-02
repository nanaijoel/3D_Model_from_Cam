# meshing.py
import os
from typing import Optional
import numpy as np
import open3d as o3d


# ------------------------------------------------------------
# Basic: save point cloud (leichtes Downsampling/Outlier-Filter)
# ------------------------------------------------------------

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    """
    Schätzt einen robusten mittleren nächsten Nachbar-Abstand.
    Nutzt scipy.cKDTree, fällt bei fehlender Abhängigkeit auf AABB-Heuristik zurück.
    """
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree  # optional dependency
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
    Speichert eine Punktwolke als PLY (mit sehr sanfter Verdichtung & Outlier-Filter).
    """
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if len(pts) >= int(filter_min_points):
        nn_med = _median_nn_distance(pts)
        voxel = max(1e-4, 0.5 * nn_med)  # adaptiv; nicht zu grob, um dünne Strukturen zu erhalten
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            # sehr sanft, um dünne Strukturen zu bewahren
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path


# ------------------------------------------------------------
# Helpers: Normals, Meshing und Nachbearbeitung
# ------------------------------------------------------------

def _diag_from_pcd(pcd: o3d.geometry.PointCloud) -> float:
    bb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(bb.get_extent(), dtype=float)
    return float(np.linalg.norm(ext))


def _estimate_normals_o3d(pcd: o3d.geometry.PointCloud, k: int = 40):
    """
    Normalschätzung mit hybrider Suche (Radius+max_nn), anschließend konsistente Orientierung.
    """
    diag = _diag_from_pcd(pcd)
    radius = max(1e-3, 0.02 * diag)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(k))
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(int(k))
    except Exception:
        # Fallback: gegen „virtuelle Kamera“ aus der Mitte orientieren
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))


def _load_camera_centers_npz(npz_path: str) -> Optional[np.ndarray]:
    """
    Versucht Kamerazentren aus einer NPZ-Datei zu laden.
    Erwartete Varianten:
      - R_list (Nx3x3), t_list (Nx3) => C = -R^T t
      - Rs (Nx3x3), ts (Nx3)
      - C_list (Nx3) direkt
      - poses (N, 2) mit (R, t)
    Gibt None zurück, wenn nichts Passendes gefunden wird.
    """
    try:
        if (npz_path is None) or (not os.path.isfile(npz_path)):
            return None
        data = np.load(npz_path, allow_pickle=True)
        if "C_list" in data:
            C = np.asarray(data["C_list"], dtype=float)
            return C if C.size else None

        def _Rt_to_C(R_arr, t_arr):
            centers = []
            for R, t in zip(R_arr, t_arr):
                R = np.asarray(R, float).reshape(3, 3)
                t = np.asarray(t, float).reshape(3)
                C = -(R.T @ t.reshape(3, 1)).reshape(3)
                centers.append(C)
            return np.asarray(centers, float)

        if ("R_list" in data) and ("t_list" in data):
            return _Rt_to_C(data["R_list"], data["t_list"])
        if ("Rs" in data) and ("ts" in data):
            return _Rt_to_C(data["Rs"], data["ts"])
        if "poses" in data:
            poses = data["poses"]
            centers = []
            for item in poses:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    R, t = item
                    R = np.asarray(R, float).reshape(3, 3)
                    t = np.asarray(t, float).reshape(3)
                    C = -(R.T @ t.reshape(3, 1)).reshape(3)
                    centers.append(C)
            if centers:
                return np.asarray(centers, float)
    except Exception:
        return None
    return None


def _orient_normals_from_poses(pcd: o3d.geometry.PointCloud, poses_npz: Optional[str]):
    """
    Orientiert Normalen in Richtung einer robusten „Kameraposition“ (Median aller Zentren).
    Fällt auf 'towards_center' zurück, wenn keine Posen gefunden werden.
    """
    centers = _load_camera_centers_npz(poses_npz) if poses_npz else None
    if centers is None or len(centers) == 0:
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))
        return
    med = np.median(centers, axis=0).astype(float)
    pcd.orient_normals_towards_camera_location(med)


def _mesh_poisson(pcd: o3d.geometry.PointCloud, depth=10, scale=1.1, linear_fit=True):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=float(scale), linear_fit=bool(linear_fit)
    )
    return mesh, np.asarray(dens, float)


def _mesh_alpha(pcd: o3d.geometry.PointCloud, alpha: float):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))


def _mesh_cleanup(mesh: o3d.geometry.TriangleMesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_crop_bbox(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand_ratio: float):
    if expand_ratio <= 0:
        return mesh
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.0 + float(expand_ratio), bb.get_center())
    try:
        return mesh.crop(bb)
    except Exception:
        # Fallback: manuelles Clipping gegen AABB
        mins = bb.get_min_bound()
        maxs = bb.get_max_bound()
        verts = np.asarray(mesh.vertices)
        inside = np.all((verts >= mins) & (verts <= maxs), axis=1)
        mesh.remove_vertices_by_mask(~inside)
        return mesh


def _trim_by_density(mesh: o3d.geometry.TriangleMesh, densities: np.ndarray, q: float):
    """
    Entfernt sehr dünn besetzte Bereiche anhand Poisson-Dichte-Quantil.
    """
    if densities is None or len(densities) != len(mesh.vertices) or q <= 0:
        return mesh
    thr = float(np.quantile(densities, np.clip(q, 0.0, 0.5)))
    mask = densities < thr
    mesh.remove_vertices_by_mask(mask)
    return mesh


def _mesh_smooth(mesh: o3d.geometry.TriangleMesh, iters=8):
    if iters <= 0:
        return mesh
    try:
        return mesh.filter_smooth_taubin(number_of_iterations=int(iters))
    except Exception:
        return mesh.filter_smooth_simple(number_of_iterations=int(iters))


def _mesh_simplify(mesh: o3d.geometry.TriangleMesh, target_tris=0):
    if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
        return mesh.simplify_quadric_decimation(int(target_tris))
    return mesh


def _transfer_vertex_colors(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, k=1):
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


def _prefilter_pcd(
    pcd: o3d.geometry.PointCloud,
    nn_med: float,
    do_radius: bool,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """
    Vorsichtige Vorfilterung: Statistischer Outlier-Filter + optional Radius-Filter.
    """
    try:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=max(8, int(nb_neighbors)),
            std_ratio=float(std_ratio)
        )
    except Exception:
        pass
    if do_radius and (nn_med > 0):
        try:
            rad = 0.75 * float(nn_med)
            pcd, _ = pcd.remove_radius_outlier(nb_points=3, radius=max(1e-6, rad))
        except Exception:
            pass
    return pcd


def _remove_small_components(
    mesh: o3d.geometry.TriangleMesh,
    min_tris_ratio: float = 0.002,
    min_area_ratio: float = 0.001
) -> o3d.geometry.TriangleMesh:
    """
    Entfernt sehr kleine, zusammenhängende Dreiecks-Komponenten (Tri- und Flächenanteil).
    """
    try:
        cl, n_tris, areas = mesh.cluster_connected_triangles()
        n_tris = np.asarray(n_tris)
        areas = np.asarray(areas, float)
        total_tris = max(1, len(mesh.triangles))
        keep = np.ones(len(n_tris), dtype=bool)

        if min_tris_ratio > 0:
            keep &= (n_tris >= int(np.ceil(min_tris_ratio * total_tris)))

        if (areas.size > 0) and (min_area_ratio > 0):
            total_area = float(np.sum(areas))
            if total_area > 0:
                keep &= (areas >= (min_area_ratio * total_area))

        tri_labels = np.asarray(cl)
        tri_mask = ~keep[tri_labels]
        mesh.remove_triangles_by_mask(tri_mask)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


# ------------------------------------------------------------
# Main: Poisson/Alpha aus PLY erzeugen (watertight)
# ------------------------------------------------------------

def reconstruct_solid_mesh_from_ply(
        ply_in: str,
        mesh_out: str,
        *,
        method: str = "poisson",
        depth: int = 11,
        scale: float = 1.10,
        no_linear_fit: bool = False,
        dens_quantile: float = 0.0,
        alpha: float = 0.0,
        bbox_expand: float = 0.02,
        pre_filter: bool = False,
        pre_filter_neighbors: int = 24,
        pre_filter_std: float = 1.2,
        pre_filter_radius: bool = False,
        voxel: float = 0.0,
        normals_k: int = 40,
        smooth: int = 6,
        simplify: int = 0,
        color_transfer: bool = False,
        poses_npz: Optional[str] = None,
        remove_small_comp: bool = False,
        min_comp_tris_ratio: float = 0.002,
        min_comp_area_ratio: float = 0.001,
        on_log=None,
        on_progress=None,
    ) -> str:
    """
    Erzeugt ein wasserfestes Mesh direkt aus einer PLY-Punktwolke.
    """
    on_log and on_log(f"[mesh] solid: load cloud -> {ply_in}")
    pcd = o3d.io.read_point_cloud(ply_in)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("Loaded point cloud is empty.")

    pts_np = np.asarray(pcd.points)
    nn_med = _median_nn_distance(pts_np)

    # Vorfilterung
    if pre_filter:
        pcd = _prefilter_pcd(
            pcd,
            nn_med=nn_med,
            do_radius=bool(pre_filter_radius),
            nb_neighbors=int(pre_filter_neighbors),
            std_ratio=float(pre_filter_std)
        )

    # Normale + Orientierung (zuerst konsistent, dann zur Kamera)
    _estimate_normals_o3d(pcd, k=int(normals_k))
    if poses_npz:
        _orient_normals_from_poses(pcd, poses_npz)

    # Meshing
    if method == "poisson":
        mesh, dens = _mesh_poisson(
            pcd,
            depth=int(depth),
            scale=float(scale),
            linear_fit=(not bool(no_linear_fit))
        )
        if float(dens_quantile) > 0:
            mesh = _trim_by_density(mesh, dens, float(dens_quantile))
    elif method == "alpha":
        if float(alpha) <= 0.0:
            # Alpha automatisch aus NN-Abstand schätzen
            nn = _median_nn_distance(np.asarray(pcd.points))
            alpha = 2.5 * float(nn)
        mesh = _mesh_alpha(pcd, float(alpha))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bounding-Box beschneiden, Cleanup und Fragment-Entfernung
    mesh = _mesh_crop_bbox(mesh, pcd, expand_ratio=float(bbox_expand))
    mesh = _mesh_cleanup(mesh)
    if bool(remove_small_comp):
        mesh = _remove_small_components(
            mesh,
            min_tris_ratio=float(min_comp_tris_ratio),
            min_area_ratio=float(min_comp_area_ratio)
        )

    # Glätten / Vereinfachen
    if int(smooth) > 0:
        mesh = _mesh_smooth(mesh, iters=int(smooth))
    if int(simplify) > 0:
        mesh = _mesh_simplify(mesh, target_tris=int(simplify))

    mesh.compute_vertex_normals()
    if bool(color_transfer):
        mesh = _transfer_vertex_colors(mesh, pcd, k=1)

    # Schreiben
    os.makedirs(os.path.dirname(mesh_out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(mesh_out, mesh, write_vertex_colors=bool(color_transfer))
    if not ok:
        raise RuntimeError(f"Failed to write mesh: {mesh_out}")

    on_log and on_log(f"[mesh] solid saved -> {mesh_out} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")
    on_progress and on_progress(100, "Watertight Mesh")
    return mesh_out
