import open3d as o3d, numpy as np, os

def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None):
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    if len(points_xyz) >= filter_min_points:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path
