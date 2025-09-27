# meshing.py
import open3d as o3d, numpy as np, os

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz)
        d, _ = tree.query(points_xyz[idx], k=2)
        return float(np.median(d[:,1]))
    except Exception:
        bb = points_xyz.max(0) - points_xyz.min(0)
        diag = float(np.linalg.norm(bb))
        return max(1e-3, diag / max(300.0, len(points_xyz) ** (1/3)))

def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None):
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))

    if len(points_xyz) >= filter_min_points:
        nn_med = _median_nn_distance(points_xyz)
        voxel = max(1e-4, 0.5 * nn_med)            # adaptiv; nicht zu grob, um dünne Bereiche zu erhalten
        pcd = pcd.voxel_down_sample(voxel_size=voxel)

        # weniger aggressiv in dünnen Regionen
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.0)

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path
