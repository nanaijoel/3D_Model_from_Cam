# meshing.py
import os
import numpy as np
from typing import Optional

# ---------- Minimal: Point-Cloud speichern (ohne Filter/Downsample) ----------

def _write_ply_ascii_xyz(points_xyz: np.ndarray, out_path: str) -> None:
    """Schreibt eine reine XYZ-ASCII-PLY ohne jegliche Nachbearbeitung."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")

def save_point_cloud(points_xyz, out_path: str,
                     on_log=None, on_progress=None) -> str:
    """
    Speichert die Punktwolke exakt so, wie sie vom SfM kommt.
    - Kein Downsample
    - Kein Outlier-Filter
    - Kein Automatik-Kram
    """
    on_log and on_log(f"[mesh] save point cloud (no-filter) -> {out_path}")

    # Dict-Varianten robust abholen
    if isinstance(points_xyz, dict):
        for key in ("points_xyz", "points", "X", "xyz", "pts"):
            if key in points_xyz:
                points_xyz = points_xyz[key]
                break
        else:
            raise TypeError(f"save_point_cloud: unsupported dict keys {list(points_xyz.keys())}")

    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        raise RuntimeError("No points to save.")

    # Schreibe immer ASCII-PLY (maximale Kompatibilität, garantiert gleiche Punktzahl)
    _write_ply_ascii_xyz(pts, out_path)

    on_progress and on_progress(100, "Save PLY")
    return out_path


# ---------- (Optional) spätere Meshing-Funktionen bleiben, aber werden hier nicht benutzt ----------

def _estimate_normals_o3d(pcd, k: int = 40):
    import open3d as o3d
    # Radius so wählen, dass es „vernünftig“ ist – wird nur benutzt, falls du später meshen willst
    bb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(bb.get_extent(), dtype=float)
    diag = float(np.linalg.norm(ext))
    radius = max(1e-3, 0.02 * diag)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(k)))
    try:
        pcd.orient_normals_consistent_tangent_plane(int(k))
    except Exception:
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))


def reconstruct_solid_mesh_from_ply(*args, **kwargs):
    """
    Platzhalter, falls der GUI-Flow das importiert.
    Hier aktuell NICHT benutzt, weil wir dein Run erstmal ohne Meshing stabilisieren wollten.
    """
    raise NotImplementedError("Meshing ist aktuell deaktiviert (wir speichern nur sparse.ply).")
