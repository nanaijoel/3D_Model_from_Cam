# fusion.py
# Silhouette- und (optional) Depth-konsistente Fusion mehrerer points_ref_*.ply
# Ergebnis: fused_points.ply (+ optional fused_mesh.ply)

from __future__ import annotations
import os, glob, json
from typing import List, Tuple
import numpy as np
import cv2 as cv

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


# ---------- Utils ----------

def _log(msg, on_log=None):
    print(msg)
    if on_log:
        try: on_log(msg)
        except: pass

def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_pcd(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    """Liest PLY. Gibt (Nx3 Punkte, Nx3 RGB[0..255] oder None) zurück."""
    if HAS_O3D:
        p = o3d.io.read_point_cloud(path)
        if p is None:
            raise FileNotFoundError(path)
        pts = np.asarray(p.points, dtype=np.float32)
        cols = None
        if p.has_colors():
            cols = (np.asarray(p.colors) * 255.0).astype(np.uint8)[:, ::-1]  # RGB->BGR
        return pts, cols

    # ASCII-Fallback (x y z [r g b])
    with open(path, "r") as f:
        header = True; n = 0; props = []
        while header:
            line = f.readline()
            if not line: break
            line = line.strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                header = False
                break
        data = []
        for _ in range(n):
            parts = f.readline().strip().split()
            data.append(parts)
    arr = np.array(data, dtype=float)
    pts = arr[:, :3].astype(np.float32)
    cols = None
    if arr.shape[1] >= 6:
        cols = arr[:, 3:6].astype(np.uint8)  # angenommen BGR
    return pts, cols

def _write_points_with_colors(path, pts: np.ndarray, cols: np.ndarray | None):
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float32)))
        if cols is not None and cols.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1].astype(np.float32) / 255.0)  # BGR->RGB
        o3d.io.write_point_cloud(path, pcd)
        return
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uchar blue\nproperty uchar green\nproperty uchar red\n")
        f.write("end_header\n")
        if cols is None:
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(pts, cols):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _load_masks_for_index(masks_dir: str, frame_idx: int):
    pat = os.path.join(masks_dir, f"frame_{frame_idx:04d}_*_mask.png")
    cands = sorted(glob.glob(pat))
    if not cands:
        # evtl. alternative Namensschemata
        cands = sorted(glob.glob(os.path.join(masks_dir, f"frame_{frame_idx:04d}_mask.png")))
    if not cands:
        return None
    m = cv.imread(cands[0], cv.IMREAD_GRAYSCALE)
    return m


def _load_depth_for_index(depth_dir: str, frame_idx: int):
    npy = os.path.join(depth_dir, f"depth_{frame_idx:04d}.npy")
    if os.path.isfile(npy):
        try: return np.load(npy).astype(np.float32)
        except Exception: return None
    return None


def _project(K, X_cam):
    x = X_cam[:, 0]; y = X_cam[:, 1]; z = np.maximum(X_cam[:, 2], 1e-9)
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v, z


# ---------- Kern: Fusion ----------

def fuse_selected_pointclouds(paths, K,
                              on_log=None, on_progress=None):
    """
    - Nimmt points_ref_*.ply (aus sparse-paint).
    - Liest ausgewählte Refs aus mesh/selected_refs.json (falls vorhanden),
      sonst alle vorhandenen points_ref_*.ply.
    - Prüft Silhouetten-Konsistenz in allen gewählten Masken.
    - Optional: vergleicht reprojizierte z mit gespeicherter depth_{idx}.npy (Toleranz).
    - Speichert fused_points.ply (+ optional fused_mesh.ply).
    """

    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir   = os.path.join(root, "mesh")
    frames_dir = os.path.join(root, "raw_frames")
    masks_dir  = os.path.join(root, "features", "masks")
    poses_npz  = os.path.join(root, "poses", "camera_poses.npz")
    depth_dir  = os.path.join(mesh_dir, "depth")

    _mkdir(mesh_dir)

    # --- ENV / Config ---
    mode = os.getenv("FUSION_MODE", "majority").strip().lower()   # any | majority | all
    min_in_mask_env = os.getenv("FUSION_MIN_IN_MASK", "").strip()
    min_in_mask = int(float(min_in_mask_env)) if min_in_mask_env != "" else -1
    use_depth   = os.getenv("FUSION_USE_DEPTH", "true").lower() == "true"
    depth_tol   = float(os.getenv("FUSION_DEPTH_TOL", "0.03"))   # relative/absolut? Wir nehmen absolut in Z
    poisson_depth = int(float(os.getenv("FUSION_POISSON_DEPTH", "10")))
    export_mesh   = os.getenv("FUSION_EXPORT_MESH", "true").lower() == "true"
    chunk_size    = int(float(os.getenv("FUSION_CHUNK", "400000")))

    # --- Posen laden ---
    npz = np.load(poses_npz)
    R_all = npz["R"]               # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1, 3, 1)

    # --- Welche Refs? ---
    sel_file = os.path.join(mesh_dir, "selected_refs.json")
    if os.path.isfile(sel_file):
        with open(sel_file, "r") as f:
            refs = json.load(f)
        refs = sorted(set(int(i) for i in refs))
    else:
        # Fallback: nimm alle vorhandenen points_ref_*.ply
        refs = []
        for p in glob.glob(os.path.join(mesh_dir, "points_ref_*.ply")):
            try:
                ridx = int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
                refs.append(ridx)
            except Exception:
                pass
        refs = sorted(set(refs))

    if not refs:
        _log("[fusion] Keine Referenz-PLYs gefunden.", on_log)
        return None

    _log(f"[fusion] use refs: {refs}", on_log)

    # --- Refs: Masken + optional Depth vorbereiten ---
    ref_masks: dict[int, np.ndarray] = {}
    ref_depths: dict[int, np.ndarray] = {}

    H = W = None
    for ridx in refs:
        m = _load_masks_for_index(masks_dir, ridx)
        if m is None:
            _log(f"[fusion] WARN: keine Maske für ref={ridx} gefunden → skip diesen Ref", on_log)
            continue
        ref_masks[ridx] = m
        if H is None: H, W = m.shape[:2]
        if use_depth:
            d = _load_depth_for_index(depth_dir, ridx)
            ref_depths[ridx] = d  # darf auch None sein

    refs = [r for r in refs if r in ref_masks]
    if not refs:
        _log("[fusion] Keine gültigen Masken gefunden.", on_log); return None

    # --- Punkte laden (alle points_ref_*.ply aus 'refs') ---
    all_pts = []
    all_cols = []
    for ridx in refs:
        ply_path = os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply")
        if not os.path.isfile(ply_path):
            _log(f"[fusion] WARN: {ply_path} fehlt, skip", on_log)
            continue
        P, C = _read_pcd(ply_path)
        if P.size == 0:
            _log(f"[fusion] WARN: {ply_path} leer, skip", on_log)
            continue
        all_pts.append(P.astype(np.float32))
        all_cols.append(C.astype(np.uint8) if C is not None else None)

    if not all_pts:
        _log("[fusion] Keine Punkte geladen.", on_log); return None

    P = np.concatenate(all_pts, axis=0)
    C = None
    if any(c is None for c in all_cols):
        C = None
    else:
        C = np.concatenate(all_cols, axis=0)

    _log(f"[fusion] total input points: {P.shape[0]}", on_log)

    # --- Sichtbarkeitszählung über Refs ---
    counts = np.zeros(P.shape[0], np.int32)

    # Vorab Kamerazentren
    cam_C = {r: (-R_all[r].T @ t_all[r]).reshape(3) for r in refs}

    # Chunked (Speicher freundlich)
    for start in range(0, P.shape[0], max(1, chunk_size)):
        end = min(P.shape[0], start + chunk_size)
        Pw = P[start:end]  # (M,3)
        keep_any = np.zeros(Pw.shape[0], dtype=bool)

        for ridx in refs:
            R = R_all[ridx]; t = t_all[ridx]; Cw = cam_C[ridx]
            M = ref_masks[ridx]
            H, W = M.shape[:2]
            # world -> cam
            X_cam = (R @ (Pw.T - Cw.reshape(3,1))).T  # (M,3)
            u, v, z = _project(K, X_cam)
            ui = np.floor(u).astype(np.int32)
            vi = np.floor(v).astype(np.int32)

            in_img = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not np.any(in_img):
                continue

            ok = in_img.copy()
            ok[in_img] &= (M[vi[in_img], ui[in_img]] > 0)

            if use_depth:
                D = ref_depths.get(ridx)
                if D is not None:
                    # nearest lookup
                    di = D[vi[in_img], ui[in_img]]
                    # valide depth?
                    valid_d = np.isfinite(di) & (di > 1e-6)
                    mask_depth = np.zeros_like(ok)
                    sub_ok = ok[in_img]
                    sub_ok &= valid_d
                    # z <= di + tol
                    sub_ok &= (z[in_img] <= di + float(depth_tol))
                    ok[in_img] = sub_ok

            counts[start:end] += ok.astype(np.int32)

        on_progress and on_progress(int(100.0 * end / max(1, P.shape[0])), "Fusion (silhouette)")

    L = len(refs)
    if mode == "all":
        thr = L
    elif mode == "any":
        thr = 1
    else:
        # majority
        thr = (L + 1) // 2
        if min_in_mask > 0:
            thr = max(1, min(L, min_in_mask))

    keep = counts >= thr
    P_out = P[keep]
    C_out = C[keep] if C is not None else None
    _log(f"[fusion] kept {P_out.shape[0]} / {P.shape[0]} (mode={mode}, thr={thr})", on_log)

    fused_points = os.path.join(mesh_dir, "fused_points.ply")
    _write_points_with_colors(fused_points, P_out, C_out)
    _log(f"[fusion] saved -> {fused_points}", on_log)

    if export_mesh and HAS_O3D and P_out.shape[0] >= 1500:
        try:
            pcd = o3d.io.read_point_cloud(fused_points)
            pcd.estimate_normals()
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
            dens = np.asarray(dens)
            keep_idx = np.where(dens > np.quantile(dens, 0.02))[0]
            mesh = mesh.select_by_index(keep_idx)
            fused_mesh = os.path.join(mesh_dir, "fused_mesh.ply")
            o3d.io.write_triangle_mesh(fused_mesh, mesh)
            _log(f"[fusion] mesh saved -> {fused_mesh}", on_log)
        except Exception as e:
            _log(f"[fusion] meshing failed: {e}", on_log)

    return fused_points
