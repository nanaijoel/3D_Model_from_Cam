# meshing.py
import os
import numpy as np
import open3d as o3d
import cv2 as cv


# -------------------------------
# Utilities
# -------------------------------
def _median_nn_distance(points_xyz: np.ndarray) -> float:
    """Robuste Schätzung eines mittleren Next-Neighbor-Abstands (mit Fallback)."""
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree  # optional
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz)
        d, _ = tree.query(points_xyz[idx], k=2)
        return float(np.median(d[:, 1]))
    except Exception:
        bb = points_xyz.max(0) - points_xyz.min(0)
        diag = float(np.linalg.norm(bb))
        return max(1e-3, diag / max(300.0, len(points_xyz) ** (1 / 3)))


# 1) Sparse Point Cloud speichern
# -------------------------------
def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None) -> str:
    """
    Speichert eine (leicht gefilterte) Punktwolke als PLY.
    """
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if len(pts) >= int(filter_min_points):
        nn_med = _median_nn_distance(pts)
        voxel = max(1e-4, 0.5 * nn_med)  # sanft, um feine Strukturen zu erhalten
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path