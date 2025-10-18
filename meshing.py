from __future__ import annotations
import os, glob
from typing import List, Tuple
import numpy as np
import cv2 as cv

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

GLOBAL_INTRINSICS_K = None


def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _log(msg, on_log=None):
    print(msg)
    if on_log:
        try: on_log(msg)
        except Exception: pass

def _load_color(path: str) -> np.ndarray:
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(path)
    if im.ndim == 2: im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    return im

def _sorted_frames(frames_dir: str) -> list[str]:
    lst = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    lst += sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    return lst

def _mask_for_frame(masks_dir: str, frame_path: str) -> np.ndarray | None:
    base = os.path.basename(frame_path)
    stem, _ = os.path.splitext(base)
    cands = [
        os.path.join(masks_dir, f"{stem}.png"),
        os.path.join(masks_dir, f"{stem}.jpg"),
        os.path.join(masks_dir, f"{stem}_mask.png"),
        os.path.join(masks_dir, f"{stem}_mask.jpg"),
    ]
    for c in cands:
        if os.path.isfile(c):
            m = cv.imread(c, cv.IMREAD_GRAYSCALE)
            if m is not None: return m
    return None

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    if points_xyz is None or len(points_xyz) < 3:
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
        return max(1e-3, diag / max(300.0, len(points_xyz)**(1/3)))

def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None) -> str:
    _log(f"[mesh] save point cloud -> {out_path}", on_log)
    _mkdir(os.path.dirname(out_path) or ".")
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)

    if not HAS_O3D:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pts: f.write(f"{p[0]} {p[1]} {p[2]}\n")
        if on_progress: on_progress(100, "Save PLY")
        return out_path

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # feinere Default-Voxelwahl
    ds_enable = str(os.getenv("MESH_DOWNSAMPLE", "1")).lower() in ("1","true","yes","on")
    voxel_env = os.getenv("MESH_VOXEL_SIZE", "")
    try:
        voxel_env = float(voxel_env) if voxel_env not in ("", None, "0", "0.0") else None
    except Exception:
        voxel_env = None

    out_nb  = int(float(os.getenv("MESH_OUTLIER_NB", "20")))
    out_std = float(os.getenv("MESH_OUTLIER_STD", "1.6"))

    if len(pts) >= int(filter_min_points):
        if ds_enable:
            voxel = voxel_env
            if voxel is None:
                voxel = max(1e-4, 0.25 * _median_nn_distance(pts))
            try:   pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
            except Exception: pass
        try:
            if out_nb > 0 and out_std > 0:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(out_nb), std_ratio=float(out_std))
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    if on_progress: on_progress(100, "Save PLY")
    return out_path


def _read_ply_xyz(ply_path: str) -> np.ndarray | None:
    if not os.path.isfile(ply_path): return None
    if HAS_O3D:
        p = o3d.io.read_point_cloud(ply_path)
        if p is None or np.asarray(p.points).size == 0:
            return None
        return np.asarray(p.points, dtype=np.float32)
    # ASCII xyz
    pts: list[tuple[float, float, float]] = []
    with open(ply_path, "r", encoding="utf-8") as f:
        header = True
        n_verts = 0
        for line in f:
            if header:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                if line.strip() == "end_header":
                    header = False
                    break
        for _ in range(n_verts):
            parts = f.readline().strip().split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                pts.append((x, y, z))
    if not pts:
        return None
    return np.array(pts, dtype=np.float32)


# ------------------------ Posen & Projektion ------------------------

def _load_poses_npz(poses_npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(poses_npz_path)
    R_all = npz["R"]                  # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1, 3, 1)
    idx = npz.get("frame_idx", np.arange(len(R_all)))
    return R_all, t_all, idx

def _project(K: np.ndarray, X_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = X_cam[..., 0]; y = X_cam[..., 1]; z = np.maximum(X_cam[..., 2], 1e-9)
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v

def _backproject_pixels_to_world(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                                 u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    Xc = np.stack([x, y, z], axis=-1)            # Nx3 camera coords
    C = (-R.T @ t).reshape(3)                    # camera center in world
    Pw = (C[None, :] + (R.T @ Xc.T).T)           # Nx3 world coords
    return Pw


# -------------------- Sichtbare Sparse + Gap-Fill --------------------

def _score_frames_by_visible_sparse(K: np.ndarray,
                                    R_all: np.ndarray, t_all: np.ndarray,
                                    sparse_pts: np.ndarray,
                                    masks: list[np.ndarray | None],
                                    on_log=None) -> np.ndarray:
    N = min(len(masks), len(R_all))
    if sparse_pts is None or sparse_pts.size == 0:
        _log("[visible-sparse] WARNING: sparse.ply leer → Scoring=0", on_log)
        return np.zeros(N, dtype=np.int32)

    scores = np.zeros(N, dtype=np.int32)
    H = W = None
    for i in range(N):
        m = masks[i]
        if m is None:
            scores[i] = 0
            continue
        C = (-R_all[i].T @ t_all[i]).reshape(3)
        X_cam = (R_all[i] @ (sparse_pts.T - C.reshape(3, 1))).T  # Nx3
        Z = X_cam[:, 2]
        if H is None:
            H, W = m.shape[:2]
        u, v = _project(K, X_cam)
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 1e-6)
        if not np.any(in_img):
            scores[i] = 0
            continue
        ui = np.clip(np.floor(u[in_img]).astype(np.int32), 0, W - 1)
        vi = np.clip(np.floor(v[in_img]).astype(np.int32), 0, H - 1)
        scores[i] = int(np.count_nonzero(m[vi, ui] > 0))
    return scores

def _choose_ref_indices(N: int, scores: np.ndarray,
                        strategy: str = "step", step: int = 2,
                        topk: int = 0, min_gap: int = 3) -> list[int]:
    strategy = (strategy or "step").strip().lower()
    if strategy == "auto":
        strategy = "bestk" if topk and topk > 0 else "step"
    if strategy == "step":
        step = max(1, int(step))
        return list(range(0, N, step))
    # bestk:
    idx_sorted = np.argsort(-scores)
    chosen: list[int] = []
    for i in idx_sorted:
        if topk and len(chosen) >= int(topk):
            break
        if all(abs(int(i) - int(j)) >= int(min_gap) for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    return chosen


def _densify_small_gaps_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray,
                              mask_img: np.ndarray, H: int, W: int,
                              max_gap: int = 4, iters: int = 3,
                              depth_std: float = 0.03, depth_rel: float = 0.06,
                              stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    valid = (z > 1e-6)
    if not np.any(valid):
        return np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    ui = np.clip(np.floor(u).astype(np.int32), 0, W - 1)
    vi = np.clip(np.floor(v).astype(np.int32), 0, H - 1)

    D = np.zeros((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.uint8)
    D[vi[valid], ui[valid]] = z[valid].astype(np.float32)
    V[vi[valid], ui[valid]] = 1

    roi = (mask_img > 0).astype(np.uint8)
    if roi.shape != V.shape:
        roi = cv.resize(roi, (W, H), interpolation=cv.INTER_NEAREST)

    inv = (1 - V) * roi
    dist = cv.distanceTransform(inv, distanceType=cv.DIST_L2, maskSize=3)
    fill_ok = (dist <= float(max_gap)).astype(np.uint8)

    kernel = np.ones((3, 3), np.float32)
    for _ in range(int(max(1, iters))):
        Vf = V.astype(np.float32)
        sum_w = cv.filter2D(Vf, -1, kernel, borderType=cv.BORDER_REPLICATE)
        sum_z = cv.filter2D(D * Vf, -1, kernel, borderType=cv.BORDER_REPLICATE)
        sum_z2 = cv.filter2D((D * Vf) * (D * Vf), -1, kernel, borderType=cv.BORDER_REPLICATE)

        mean = np.divide(sum_z, np.maximum(sum_w, 1e-6))
        var = np.maximum(sum_z2 / np.maximum(sum_w, 1e-6) - mean * mean, 0.0)
        std = np.sqrt(var)

        cand = (V == 0) & (fill_ok == 1) & (roi == 1) & (sum_w >= 2.0)
        if not np.any(cand):
            break

        ok_rel = (std <= depth_rel * np.maximum(mean, 1e-6))
        ok_abs = (std <= depth_std)
        accept = cand & (ok_rel | ok_abs)

        D[accept] = mean[accept].astype(np.float32)
        V[accept] = 1

    seeds = np.zeros_like(V, dtype=np.uint8)
    seeds[vi[valid], ui[valid]] = 1
    new_mask = (V == 1) & (seeds == 0)

    if stride and stride > 1:
        grid = np.zeros_like(new_mask, dtype=np.uint8)
        grid[::stride, ::stride] = 1
        new_mask = new_mask & (grid == 1)

    ny, nx = np.nonzero(new_mask)
    if ny.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    uu = nx.astype(np.float32)
    vv = ny.astype(np.float32)
    zz = D[ny, nx].astype(np.float32)
    return uu, vv, zz


def run_visible_sparse_with_fill(mesh_dir: str, frames_dir: str, poses_npz: str, masks_dir: str | None,
                                 on_log=None, on_progress=None) -> str:

    _mkdir(mesh_dir)

    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    R_all, t_all, _ = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}.")
    images = [_load_color(f) for f in frame_files]
    if masks_dir and os.path.isdir(masks_dir):
        masks = [_mask_for_frame(masks_dir, f) for f in frame_files]
    else:
        masks = [np.ones(images[0].shape[:2], np.uint8) * 255 for _ in frame_files]

    V = min(len(R_all), len(images))
    images = images[:V]; masks = masks[:V]

    sparse_path = os.path.join(mesh_dir, "sparse.ply")
    P = _read_ply_xyz(sparse_path)
    if P is None or P.size == 0:
        raise RuntimeError("sparse.ply ist leer oder fehlt.")

    REF_STRAT = (os.getenv("MVS_REF_STRATEGY", "step") or "step").strip().lower()
    REF_TOPK  = int(float(os.getenv("MVS_REF_TOPK", "60")))
    REF_MIN_G = int(float(os.getenv("MVS_REF_MIN_GAP", "3")))
    REF_STEP  = int(float(os.getenv("MVS_REF_STEP", "2")))

    scores = _score_frames_by_visible_sparse(K, R_all[:V], t_all[:V], P, masks[:V], on_log)
    refs = _choose_ref_indices(V, scores, REF_STRAT, REF_STEP, REF_TOPK, REF_MIN_G)
    if not refs: refs = [0]
    _log(f"[visible-sparse+fill] refs: {refs}", on_log)

    H, W = images[0].shape[:2]
    keep_idx = np.zeros(P.shape[0], dtype=bool)
    new_points_world: list[np.ndarray] = []

    FILL_ENABLE = (os.getenv("MVS_GAP_FILL_ENABLE", "true").strip().lower() in ("1","true","yes","on"))
    MAX_GAP  = int(float(os.getenv("MVS_FILL_MAX_GAP", "4")))
    N_ITERS  = int(float(os.getenv("MVS_FILL_ITERS", "3")))
    D_STD    = float(os.getenv("MVS_FILL_DEPTH_STD", "0.03"))
    D_REL    = float(os.getenv("MVS_FILL_DEPTH_REL", "0.06"))
    STRIDE   = int(float(os.getenv("MVS_FILL_STRIDE", "1")))

    for k, ridx in enumerate(refs):
        if on_progress:
            on_progress(int(100.0 * k / max(1, len(refs))), f"visible-sparse ref={ridx}")

        mask = masks[ridx] if masks[ridx] is not None else np.ones((H, W), np.uint8) * 255

        C = (-R_all[ridx].T @ t_all[ridx]).reshape(3)
        X_cam = (R_all[ridx] @ (P.T - C.reshape(3, 1))).T  # Nx3
        Z = X_cam[:, 2]
        u, v = _project(K, X_cam)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 1e-6)
        if not np.any(in_img):
            continue

        ui = np.clip(np.floor(u[in_img]).astype(np.int32), 0, W - 1)
        vi = np.clip(np.floor(v[in_img]).astype(np.int32), 0, H - 1)
        ok = (mask[vi, ui] > 0)
        idx = np.nonzero(in_img)[0][ok]
        keep_idx[idx] = True

        if FILL_ENABLE:
            u_valid = u[in_img][ok].astype(np.float32)
            v_valid = v[in_img][ok].astype(np.float32)
            z_valid = Z[in_img][ok].astype(np.float32)

            uf, vf, zf = _densify_small_gaps_depth(
                u_valid, v_valid, z_valid,
                mask_img=mask, H=H, W=W,
                max_gap=MAX_GAP, iters=N_ITERS,
                depth_std=D_STD, depth_rel=D_REL, stride=STRIDE
            )
            if uf.size:
                Pw = _backproject_pixels_to_world(K, R_all[ridx], t_all[ridx], uf, vf, zf)
                new_points_world.append(Pw.astype(np.float32))

    fused_core = P[keep_idx].astype(np.float32)
    fused = fused_core if not len(new_points_world) else np.vstack([fused_core] + new_points_world).astype(np.float32)

    # --- Cap & Downsample nach Füllen (gegen 20M+ Punkte) ---
    MAX_FILL_POINTS = int(float(os.getenv("MVS_FILL_MAX_POINTS", "4000000")))
    VOX_AFTER = float(os.getenv("MVS_VOXEL_AFTER_FILL", "0.0015"))
    if fused.shape[0] > MAX_FILL_POINTS and HAS_O3D:
        _log(f"[fill] too many points ({fused.shape[0]}) -> voxel downsample to ~{MAX_FILL_POINTS}", on_log)
        pcd_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused.astype(np.float64)))
        vox = VOX_AFTER
        for _ in range(6):
            pcd_ds = pcd_tmp.voxel_down_sample(voxel_size=vox)
            if np.asarray(pcd_ds.points).shape[0] <= MAX_FILL_POINTS:
                fused = np.asarray(pcd_ds.points, dtype=np.float32)
                break
            vox *= 1.3
        else:
            idx = np.random.choice(fused.shape[0], size=MAX_FILL_POINTS, replace=False)
            fused = fused[idx]
        _log(f"[fill] after cap: {fused.shape[0]} pts (voxel≈{vox:.5f})", on_log)

    out_path = os.path.join(mesh_dir, "fused_points.ply")
    save_point_cloud(fused, out_path, on_log=on_log, on_progress=on_progress)
    _log(f"[visible-sparse+fill] fused_points: {fused.shape[0]} (orig kept {fused_core.shape[0]} / {P.shape[0]}, filled {sum(len(x) for x in new_points_world) if new_points_world else 0})", on_log)
    return out_path


# ------------------------------ Carving --------------------------------

def _dilate_mask(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return m
    k = (2*px+1, 2*px+1)
    return cv.dilate(m, cv.getStructuringElement(cv.MORPH_ELLIPSE, k))

def carve_points_like_texturing(mesh_dir: str, frames_dir: str, poses_npz: str, masks_dir: str | None,
                                use_all_masks: bool = False, n_views: int = 24,
                                tau: float = 0.50, mask_dilate_px: int = 2,
                                on_log=None, on_progress=None) -> str:

    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    fused_path = os.path.join(mesh_dir, "fused_points.ply")
    P = _read_ply_xyz(fused_path)
    if P is None or P.size == 0:
        return fused_path  # nichts zu carven

    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        return fused_path
    R_all, t_all, _ = _load_poses_npz(poses_npz)

    V = min(len(frame_files), len(R_all))
    sel = list(range(V)) if use_all_masks else list(range(0, V, max(1, V // max(1, n_views))))
    sel = sorted(set(np.clip(sel, 0, V - 1)))
    _log(f"[carve] using {len(sel)} views (requested {n_views}, available_masks {V}) tau={tau:.2f}, dil={mask_dilate_px}px -> {sel}", on_log)

    masks = []
    for i in sel:
        m = _mask_for_frame(masks_dir, frame_files[i]) if (masks_dir and os.path.isdir(masks_dir)) else None
        if m is None:
            m = np.ones((1080, 1920), np.uint8) * 255
        if mask_dilate_px > 0:
            m = _dilate_mask(m, mask_dilate_px)
        masks.append(m)

    keep = np.ones(P.shape[0], dtype=bool)
    for idx_k, i in enumerate(sel):
        if on_progress:
            on_progress(int(100.0 * idx_k / max(1, len(sel))), f"carve view={i}")
        C = (-R_all[i].T @ t_all[i]).reshape(3)
        X_cam = (R_all[i] @ (P.T - C.reshape(3, 1))).T
        Z = X_cam[:, 2]
        u, v = _project(K, X_cam)
        m = masks[idx_k]; H, W = m.shape[:2]
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 1e-6)
        ui = np.clip(np.floor(u[inside]).astype(np.int32), 0, W - 1)
        vi = np.clip(np.floor(v[inside]).astype(np.int32), 0, H - 1)
        ok = np.zeros(P.shape[0], dtype=np.uint8)
        ok[np.where(inside)[0]] = (m[vi, ui] > 0).astype(np.uint8)
        # Mehrheitsentscheidung über alle Views: tau=0.5 → Hälfte muss „sichtbar“ sein
        keep &= (ok > 0) | (np.random.rand(ok.shape[0]) < (1.0 - tau))  # weiche Schranke
    out = os.path.join(mesh_dir, "fused_points.ply")
    save_point_cloud(P[keep], out, on_log=on_log, on_progress=on_progress)
    _log(f"[carve] kept {np.count_nonzero(keep)}/{P.shape[0]} points", on_log)
    return out

# --- in meshing.py (zusätzlich einfügen) ---
def carve_sparse_by_masks(mesh_dir: str, frames_dir: str, poses_npz: str, masks_dir: str | None,
                          n_views: int = 32, tau: float = 0.70, mask_dilate_px: int = 2,
                          on_log=None, on_progress=None) -> str:
    """
    Pre-Carve direkt auf sparse.ply mit Mehrheitsvoting über Masken.
    Behalte Punkte, die in >= tau Anteil der geprüften Views IN der Maske liegen,
    sofern sie überhaupt in genügend Views ins Bild projizieren.
    """
    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None: raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    sparse_path = os.path.join(mesh_dir, "sparse.ply")
    P = _read_ply_xyz(sparse_path)
    if P is None or P.size == 0:
        return sparse_path

    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0: return sparse_path
    R_all, t_all, _ = _load_poses_npz(poses_npz)

    V = min(len(frame_files), len(R_all))
    sel = list(range(0, V, max(1, V // max(1, n_views))))
    sel = sorted(set(np.clip(sel, 0, V - 1)))
    _log(f"[pre-carve] using {len(sel)} views (tau={tau:.2f}, dil={mask_dilate_px}px)", on_log)

    # Masken laden (oder Full-On)
    masks = []
    for i in sel:
        m = _mask_for_frame(masks_dir, frame_files[i]) if (masks_dir and os.path.isdir(masks_dir)) else None
        if m is None:
            m = np.ones((1080, 1920), np.uint8) * 255
        if mask_dilate_px > 0:
            m = _dilate_mask(m, mask_dilate_px)
        masks.append(m)

    votes_in = np.zeros(P.shape[0], dtype=np.int32)
    votes_vis = np.zeros(P.shape[0], dtype=np.int32)

    for idx_k, i in enumerate(sel):
        if on_progress: on_progress(int(100.0 * idx_k / max(1, len(sel))), f"pre-carve view={i}")
        C = (-R_all[i].T @ t_all[i]).reshape(3)
        X_cam = (R_all[i] @ (P.T - C.reshape(3, 1))).T
        Z = X_cam[:, 2]
        u, v = _project(K, X_cam)
        m = masks[idx_k]; H, W = m.shape[:2]

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 1e-6)
        votes_vis += inside.astype(np.int32)

        if np.any(inside):
            ui = np.clip(np.floor(u[inside]).astype(np.int32), 0, W - 1)
            vi = np.clip(np.floor(v[inside]).astype(np.int32), 0, H - 1)
            ok = (m[vi, ui] > 0)
            idx_in = np.where(inside)[0][ok]
            votes_in[idx_in] += 1

    # Konsensus: mindestens ein paar valide Projektionen (robust), und
    # Anteil innerhalb Maske >= tau
    MIN_VIS = max(3, int(round(0.25 * len(sel))))  # z.B. mindestens 25% der geprüften Views sichtbar
    keep = (votes_vis >= MIN_VIS) & (votes_in >= np.ceil(tau * np.maximum(votes_vis, 1)))

    out_sparse = os.path.join(mesh_dir, "sparse.ply")
    save_point_cloud(P[keep], out_sparse, on_log=on_log, on_progress=on_progress)
    _log(f"[pre-carve] kept {np.count_nonzero(keep)}/{P.shape[0]} points (min_vis={MIN_VIS})", on_log)
    return out_sparse



## meshing.py — Ersatz für close_surface_after_fill
def close_surface_after_fill(mesh_dir: str,
                             target_points: int = 800_000,
                             poisson_depth: int = -1,
                             poisson_scale: float = 1.1,
                             max_poisson_points: int = 450_000,
                             on_log=None, on_progress=None) -> str:
    """
    Robuster 'zweiter Gang':
      1) adaptiv ausdünnen (<= max_poisson_points)
      2) Poisson (auto depth 7..10, wenn poisson_depth<0)
      3) Dichte-Trim + Hole-Fill
      4) Speichere geschlossenes Mesh: closed_mesh.ply
      5) Zurück sampeln -> fused_points.ply (für Punkt-Texturing)
    """
    if not HAS_O3D:
        _log("[surface-close] skipped (open3d not available)", on_log)
        return os.path.join(mesh_dir, "fused_points.ply")

    src = os.path.join(mesh_dir, "fused_points.ply")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    pcd = o3d.io.read_point_cloud(src)
    n_in = np.asarray(pcd.points).shape[0]
    if n_in == 0:
        _log("[surface-close] empty point cloud", on_log)
        return src

    _log(f"[surface-close] input points={n_in}", on_log)

    # 1) adaptiv ausdünnen
    def median_nn_dist(pcd_in, k=16):
        p = np.asarray(pcd_in.points)
        if p.shape[0] < k+1:
            return 0.01
        kdt = o3d.geometry.KDTreeFlann(pcd_in)
        dists = []
        step = max(1, p.shape[0] // 50_000)
        for i in range(0, p.shape[0], step):
            _, idx, _ = kdt.search_knn_vector_3d(pcd_in.points[i], k)
            nn = p[idx[1:], :] - p[i]
            dd = np.linalg.norm(nn, axis=1)
            dists.append(np.median(dd))
        return float(np.median(dists)) if dists else 0.01

    if n_in > max_poisson_points:
        dmed = median_nn_dist(pcd)
        vox = max(1e-4, 0.6 * dmed)
        pcd = pcd.voxel_down_sample(voxel_size=vox)
        n_ds = np.asarray(pcd.points).shape[0]
        _log(f"[surface-close] pre-voxel: voxel={vox:.5f} -> {n_ds} pts", on_log)

    # Normals
    try:
        rad = max(1e-3, 4.0 * median_nn_dist(pcd))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=32))
        pcd.orient_normals_consistent_tangent_plane(40)
    except Exception:
        pcd.estimate_normals()

    # 2) Poisson (auto depth)
    n_poisson = np.asarray(pcd.points).shape[0]
    if poisson_depth <= 0:
        if n_poisson < 120_000: depth = 8
        elif n_poisson < 300_000: depth = 9
        else: depth = 10
    else:
        depth = int(poisson_depth)

    _log(f"[surface-close] poisson: n={n_poisson}, depth={depth}, scale={poisson_scale}", on_log)

    try:
        if on_progress: on_progress(5, "Poisson reconstruction")
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(depth), scale=float(poisson_scale), linear_fit=True
        )

        dens = np.asarray(dens)
        keep = dens > np.percentile(dens, 5.0)
        mesh = mesh.select_by_index(np.where(keep)[0])

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        try:
            mesh = mesh.fill_holes()
        except Exception:
            pass

    except Exception as e:
        _log(f"[surface-close] Poisson failed ({e}) → fallback BPA", on_log)
        r = 1.5 * median_nn_dist(pcd)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([r, 2.0 * r])
            )
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
            try:
                mesh = mesh.fill_holes()
            except Exception:
                pass
        except Exception as ee:
            _log(f"[surface-close] BPA also failed ({ee})", on_log)
            return src

    # 4) geschlossenes Mesh persistieren
    closed_mesh_path = os.path.join(mesh_dir, "closed_mesh.ply")
    o3d.io.write_triangle_mesh(closed_mesh_path, mesh)
    _log(f"[surface-close] wrote: {closed_mesh_path}", on_log)

    # 5) zurück sampeln (Punktwolke für dein Punkt-Texturing)
    n_target = int(target_points)
    try:
        pcd_surf = mesh.sample_points_poisson_disk(number_of_points=n_target)
    except Exception:
        pcd_surf = mesh.sample_points_uniformly(number_of_points=n_target)
    P_surf = np.asarray(pcd_surf.points, dtype=np.float32)

    out = os.path.join(mesh_dir, "fused_points.ply")
    save_point_cloud(P_surf, out, on_log=on_log, on_progress=on_progress)
    _log(f"[surface-close] wrote: {out} ({P_surf.shape[0]} pts)", on_log)
    return out


# ------------------------- Öffentliche API --------------------------------

def reconstruct_mvs_depth_and_mesh(paths, K,
                                   on_log=None, on_progress=None) -> str:
    """
    Sichtbare Sparse + Gap-Fill -> fused_points.ply (und dort speichern).
    """
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()
    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir   = os.path.join(root, "mesh")
    frames_dir = os.path.join(root, "raw_frames")
    poses_npz  = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir  = os.path.join(root, "features", "masks")
    return run_visible_sparse_with_fill(mesh_dir, frames_dir, poses_npz, masks_dir, on_log, on_progress)

def run_mvs_and_carve(paths, K, on_log=None, on_progress=None) -> str:
    """
    Wrapper: Dense (sichtbare Sparse + Fill) + optional Carving + optional Surface-Close.
    - Dense/Filling:  reconstruct_mvs_depth_and_mesh(...)
    - Optional Carving: carve_points_like_texturing(...)
    - Optional Surface-Close: close_surface_after_fill(...)  -> schließt Oberflächenlöcher
    Rückgabe: Pfad zu mesh/fused_points.ply (ggf. nach Surface-Close ersetzt).
    """
    # 1) Dense + Fill
    fused_ply = reconstruct_mvs_depth_and_mesh(paths, K, on_log, on_progress)

    # 2) Optional: Carving-Pass (Masken-basiert)
    if str(os.getenv("CARVE_ENABLE", "true")).lower() in ("1", "true", "yes", "on"):
        root = paths.root if hasattr(paths, "root") else paths["root"]
        mesh_dir   = os.path.join(root, "mesh")
        frames_dir = os.path.join(root, "raw_frames")
        poses_npz  = os.path.join(root, "poses", "camera_poses.npz")
        masks_dir  = os.path.join(root, "features", "masks")

        use_all = str(os.getenv("CARVE_USE_ALL_MASKS", "false")).lower() in ("1", "true", "yes", "on")
        nviews  = int(float(os.getenv("CARVE_VIEWS", "24")))
        tau     = float(os.getenv("CARVE_TAU", "0.50"))
        dil     = int(float(os.getenv("CARVE_MASK_DILATE_PX", "2")))

        fused_ply = carve_points_like_texturing(
            mesh_dir, frames_dir, poses_npz, masks_dir,
            use_all_masks=use_all, n_views=nviews, tau=tau, mask_dilate_px=dil,
            on_log=on_log, on_progress=on_progress
        )

    # 3) Optional: „zweiter Gang“ – Surface schließen (Poisson + zurück sampeln)
    if str(os.getenv("SURFACE_CLOSE_ENABLE", "false")).lower() in ("1", "true", "yes", "on"):
        root = paths.root if hasattr(paths, "root") else paths["root"]
        mesh_dir = os.path.join(root, "mesh")
        try:
            target_points = int(float(os.getenv("SURFACE_CLOSE_TARGET_POINTS", "1000000")))
            poisson_depth = int(float(os.getenv("SURFACE_CLOSE_POISSON_DEPTH", "9")))
            poisson_scale = float(os.getenv("SURFACE_CLOSE_POISSON_SCALE", "1.1"))

            # erzeugt neue, geschlossene Punktwolke unter mesh/fused_points.ply
            fused_ply = close_surface_after_fill(
                mesh_dir,
                target_points=target_points,
                poisson_depth=poisson_depth,
                poisson_scale=poisson_scale,
                on_log=on_log, on_progress=on_progress
            )

            # sicherstellen, dass das Texturing die geschlossene Wolke nimmt
            os.environ["TEXTURE_IN_PLY"] = "fused_points.ply"

        except Exception as e:
            if on_log:
                on_log(f"[surface-close] failed -> keep original fused_points.ply ({e})")

    return fused_ply
