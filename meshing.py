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

# Wird vom pipeline_runner gesetzt
GLOBAL_INTRINSICS_K = None


# --------- kleine Utilities ---------

def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _log(msg, on_log=None):
    print(msg)
    if on_log:
        try:
            on_log(msg)
        except Exception:
            pass

def _load_color(path: str) -> np.ndarray:
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    return im

def _sorted_frames(frames_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

def _mask_for_frame(masks_dir: str, frame_path: str) -> np.ndarray | None:
    base = os.path.basename(frame_path)
    stem, _ = os.path.splitext(base)
    cands = [
        os.path.join(masks_dir, f"{stem}_mask.png"),
        os.path.join(masks_dir, f"{stem}_mask.jpg"),
    ]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "frame":
        cands.append(os.path.join(masks_dir, f"frame_{parts[1]}_mask.png"))
        cands.append(os.path.join(masks_dir, f"frame_{parts[1]}_mask.jpg"))
    for c in cands:
        if os.path.isfile(c):
            m = cv.imread(c, cv.IMREAD_GRAYSCALE)
            if m is not None:
                return m
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


# --------- PLY I/O ---------

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
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        if on_progress:
            on_progress(100, "Save PLY")
        return out_path

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if len(pts) >= int(filter_min_points):
        voxel = max(1e-4, 0.5 * _median_nn_distance(pts))
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception:
            pass
    o3d.io.write_point_cloud(out_path, pcd)
    if on_progress:
        on_progress(100, "Save PLY")
    return out_path

def _read_sparse_points(ply_path: str) -> np.ndarray | None:
    if not os.path.isfile(ply_path):
        return None
    if HAS_O3D:
        p = o3d.io.read_point_cloud(ply_path)
        if p is None or np.asarray(p.points).size == 0:
            return None
        return np.asarray(p.points, dtype=np.float32)

    # PLY-ASCII Fallback (xyz)
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


# --------- Posen / Projektion ---------

def _load_poses_npz(poses_npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(poses_npz_path)
    R_all = npz["R"]                  # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1, 3, 1)
    idx = npz.get("frame_idx", np.arange(len(R_all)))
    return R_all, t_all, idx

def _project(K: np.ndarray, X_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = X_cam[..., 0]
    y = X_cam[..., 1]
    z = np.maximum(X_cam[..., 2], 1e-9)
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v

def _backproject_pixels_to_world(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                                 u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    Xc = np.stack([x, y, z], axis=-1)  # Nx3 camera coords
    C = (-R.T @ t).reshape(3)          # camera center in world
    Pw = (C[None, :] + (R.T @ Xc.T).T) # Nx3 world coords
    return Pw


# --------- Sichtbarkeits-Scoring & Ref-View Auswahl ---------

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
                        strategy: str = "step", step: int = 3,
                        topk: int = 0, min_gap: int = 2) -> list[int]:
    strategy = (strategy or "step").strip().lower()
    if strategy == "auto":
        strategy = "bestk" if topk and topk > 0 else "step"
    if strategy == "step":
        step = max(1, int(step))
        return list(range(0, N, step))
    # bestk
    idx_sorted = np.argsort(-scores)  # desc
    chosen: list[int] = []
    for i in idx_sorted:
        if topk and len(chosen) >= int(topk):
            break
        if all(abs(int(i) - int(j)) >= int(min_gap) for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    return chosen


# --------- Lokale Tiefen-Füllung (kleine Lücken) ---------

def _densify_small_gaps_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray,
                              mask_img: np.ndarray, H: int, W: int,
                              max_gap: int = 4, iters: int = 3,
                              depth_std: float = 0.02, depth_rel: float = 0.05,
                              stride: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baut aus den projizierten Sparse-Punkten (u,v,z) eine dünne Depth-Map,
    füllt kleine Lücken (<= max_gap) nur dort, wo die Nachbar-Tiefen konsistent sind
    (Gate über absolute UND/ODER relative Std), und gibt neue (u,v,z) Pixel zurück,
    die vorher noch nicht belegt waren. Optional Subsampling über 'stride'.
    """
    valid = (z > 1e-6)
    if not np.any(valid):
        return np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    ui = np.clip(u.astype(np.int32), 0, W - 1)
    vi2 = np.clip(v.astype(np.int32), 0, H - 1)

    D = np.zeros((H, W), dtype=np.float32)      # depth
    V = np.zeros((H, W), dtype=np.uint8)        # valid flag
    D[vi2[valid], ui[valid]] = z[valid].astype(np.float32)
    V[vi2[valid], ui[valid]] = 1

    # ROI = Objektmaske
    roi = (mask_img > 0).astype(np.uint8)
    if roi.shape != V.shape:
        roi = cv.resize(roi, (W, H), interpolation=cv.INTER_NEAREST)

    # Entfernung zum nächsten Seed, nur Pixel mit dist<=max_gap zulassen
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

        # Kandidaten: noch nicht belegt, innerhalb Maske, nahe an Seeds, mind. 2 Nachbarn
        cand = (V == 0) & (fill_ok == 1) & (roi == 1) & (sum_w >= 2.0)
        if not np.any(cand):
            break

        ok_rel = (std <= depth_rel * np.maximum(mean, 1e-6))
        ok_abs = (std <= depth_std)
        accept = cand & (ok_rel | ok_abs)

        D[accept] = mean[accept].astype(np.float32)
        V[accept] = 1

    # Neue Pixel = jetzt gültig, aber nicht ursprüngliche Seeds
    seeds = np.zeros_like(V, dtype=np.uint8)
    seeds[vi2[valid], ui[valid]] = 1
    new_mask = (V == 1) & (seeds == 0)

    # optional: Subsampling, um Punktzahl zu begrenzen
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


# --------- (optional) farbiges PLY speichern ---------

def _save_colored_ply_any(P: np.ndarray, C: np.ndarray | None, out_path: str) -> str:
    _mkdir(os.path.dirname(out_path) or ".")
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        if C is not None and C.shape[0] == P.shape[0]:
            pc.colors = o3d.utility.Vector3dVector((C[:, ::-1] / 255.0).astype(np.float64))  # BGR->RGB
        o3d.io.write_point_cloud(out_path, pc)
        return out_path
    except Exception:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {P.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if C is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            if C is None:
                for p in P:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
            else:
                Cu8 = np.clip(C, 0, 255).astype(np.uint8)
                for p, c in zip(P, Cu8):
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
        return out_path


# --------- Hauptpfad: sichtbare Sparse + Lückenfüllung ---------

def run_visible_sparse_with_fill(mesh_dir: str, frames_dir: str, poses_npz: str, masks_dir: str | None,
                                 on_log=None, on_progress=None) -> str:
    """
    1) Lädt sparse.ply (Weltkoordinaten).
    2) Wählt Referenz-Views (step/bestk).
    3) Für jede Ref-View:
       - Projiziert Sparse → behält nur sichtbare & maskierte Punkte
       - (optional) füllt kleine Lücken im Bildraum mit konsistenter Tiefe
       - back-projiziert die gefüllten Pixel als zusätzliche Weltpunkte
    4) Speichert Union als fused_points.ply
    """
    _mkdir(mesh_dir)

    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    # Ressourcen laden
    R_all, t_all, _ = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}.")
    images = [_load_color(f) for f in frame_files]
    if masks_dir and os.path.isdir(masks_dir):
        masks = [_mask_for_frame(masks_dir, f) for f in frame_files]
    else:
        masks = [np.ones(images[0].shape[:2], np.uint8) * 255 for _ in frame_files]

    Vposes = len(R_all)
    Nimgs = len(images)
    V = min(Vposes, Nimgs)
    images = images[:V]
    masks = masks[:V]

    # Sparse laden
    sparse_path = os.path.join(mesh_dir, "sparse.ply")
    P = _read_sparse_points(sparse_path)
    if P is None or P.size == 0:
        raise RuntimeError("sparse.ply ist leer oder fehlt.")

    # Referenz-Views
    REF_STRAT = (os.getenv("MVS_REF_STRATEGY", "step") or "step").strip().lower()
    REF_TOPK  = int(float(os.getenv("MVS_REF_TOPK", "0")))
    REF_MIN_G = int(float(os.getenv("MVS_REF_MIN_GAP", "2")))
    REF_STEP  = int(float(os.getenv("MVS_REF_STEP", "3")))

    scores = _score_frames_by_visible_sparse(K, R_all[:V], t_all[:V], P, masks[:V], on_log)
    refs = _choose_ref_indices(V, scores, REF_STRAT, REF_STEP, REF_TOPK, REF_MIN_G)
    if not refs:
        refs = [0]
    _log(f"[visible-sparse+fill] refs: {refs}", on_log)

    H, W = images[0].shape[:2]
    keep_idx = np.zeros(P.shape[0], dtype=bool)
    new_points_world: list[np.ndarray] = []

    # Fill-Parameter (ENV)
    FILL_ENABLE = (os.getenv("MVS_GAP_FILL_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on"))
    MAX_GAP  = int(float(os.getenv("MVS_FILL_MAX_GAP", "4")))
    N_ITERS  = int(float(os.getenv("MVS_FILL_ITERS", "3")))
    D_STD    = float(os.getenv("MVS_FILL_DEPTH_STD", "0.02"))
    D_REL    = float(os.getenv("MVS_FILL_DEPTH_REL", "0.05"))
    STRIDE   = int(float(os.getenv("MVS_FILL_STRIDE", "2")))
    EXPORT_PER_VIEW = (os.getenv("MVS_EXPORT_MESH", "false").strip().lower() in ("1","true","yes","on"))

    for k, ridx in enumerate(refs):
        if on_progress:
            on_progress(int(100.0 * k / max(1, len(refs))), f"visible-sparse ref={ridx}")

        mask = masks[ridx] if masks[ridx] is not None else np.ones((H, W), np.uint8) * 255

        # Welt → Kamera
        C = (-R_all[ridx].T @ t_all[ridx]).reshape(3)
        X_cam = (R_all[ridx] @ (P.T - C.reshape(3, 1))).T  # Nx3
        Z = X_cam[:, 2]
        u, v = _project(K, X_cam)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 1e-6)
        if not np.any(in_img):
            continue

        ui = np.clip(np.floor(u[in_img]).astype(np.int32), 0, W - 1)
        vi2 = np.clip(np.floor(v[in_img]).astype(np.int32), 0, H - 1)
        ok = (mask[vi2, ui] > 0)
        idx = np.nonzero(in_img)[0][ok]
        keep_idx[idx] = True

        # Lücken füllen (optional)
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

        # Optionales Debug: pro-View sichtbare Punkte farbig abspeichern
        if EXPORT_PER_VIEW and HAS_O3D:
            Pv = P[idx].astype(np.float32)
            im = images[ridx]
            cols = im[vi2[ok], ui[ok]].astype(np.uint8)  # BGR
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pv))
            pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1] / 255.0)  # → RGB
            o3d.io.write_point_cloud(os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply"), pcd)

    # Union: sichtbare Sparse + neu gefüllte Punkte
    fused_core = P[keep_idx].astype(np.float32)
    if len(new_points_world):
        fused = np.vstack([fused_core] + new_points_world).astype(np.float32)
    else:
        fused = fused_core

    out_path = os.path.join(mesh_dir, "fused_points.ply")
    save_point_cloud(fused, out_path, on_log=on_log, on_progress=on_progress)
    _log(f"[visible-sparse+fill] fused_points: {fused.shape[0]} (orig kept {fused_core.shape[0]} / {P.shape[0]}, filled {sum(len(x) for x in new_points_world) if new_points_world else 0})", on_log)
    return out_path


# --------- Pipeline-API ---------

def reconstruct_mvs_depth_and_mesh(paths, K,
                                   scale=1.0, max_views=0, n_planes=0,
                                   depth_expand=0.0, patch=0,
                                   cost_thr=0.0, min_valid_frac=0.0,
                                   poisson_depth=0,
                                   on_log=None, on_progress=None):
    """
    Wird vom pipeline_runner aufgerufen.
    """
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()
    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir   = os.path.join(root, "mesh")
    frames_dir = os.path.join(root, "raw_frames")
    poses_npz  = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir  = os.path.join(root, "features", "masks")
    return run_visible_sparse_with_fill(mesh_dir, frames_dir, poses_npz, masks_dir, on_log, on_progress)

def reconstruct_mvs_depth_and_mesh_all(paths, K,
                                       scale=1.0, max_views=0, n_planes=0,
                                       depth_expand=0.0, patch=0,
                                       cost_thr=0.0, min_valid_frac=0.0,
                                       poisson_depth=0,
                                       on_log=None, on_progress=None):
    # aktuell gleich wie 'single'
    return reconstruct_mvs_depth_and_mesh(paths, K, scale, max_views, n_planes,
                                          depth_expand, patch, cost_thr, min_valid_frac,
                                          poisson_depth, on_log, on_progress)





# ===== Carving im Texturing-Stil =============================================

def _evenly_pick_from_index_list(idxs: list[int], n_views: int) -> list[int]:
    """Wählt n gleichmäßig verteilte Einträge aus einer Liste vorhandener Indizes."""
    if not idxs:
        return []
    n_views = max(1, min(int(n_views), len(idxs)))
    if n_views == len(idxs):
        return list(idxs)
    pos = np.linspace(0, len(idxs)-1, num=n_views, dtype=int)
    pos = np.unique(np.clip(pos, 0, len(idxs)-1))
    return [idxs[p] for p in pos.tolist()]

def carve_points_like_texturing(mesh_dir: str, frames_dir: str, poses_npz: str, masks_dir: str | None,
                                use_all_masks: bool = False, n_views: int = 4,
                                on_log=None, on_progress=None) -> str:
    """
    - Wählt Views gleichmäßig verteilt wie beim Texturing (aber mit fixer Anzahl n_views).
    - Nutzt NUR Frames, zu denen es eine Maske gibt (→ nie mehr Views als Masken).
    - Maske i ↔ Pose i ↔ Frame i bleiben gekoppelt.
    - Behalten wird ein Punkt, wenn er in JEDEM gewählten View entweder
      (unsichtbar) ODER (sichtbar & in Maske) liegt.
    - Überschreibt mesh/fused_points.ply.
    """
    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    # 3D Punkte laden (fused > sparse als Fallback)
    fused_path = os.path.join(mesh_dir, "fused_points.ply")
    P = _read_sparse_points(fused_path)
    if P is None or P.size == 0:
        fused_path = os.path.join(mesh_dir, "sparse.ply")
        P = _read_sparse_points(fused_path)
    if P is None or P.size == 0:
        raise RuntimeError("Keine Punktwolke zum Carven gefunden.")

    # Frames / Posen
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}.")
    R_all, t_all, _ = _load_poses_npz(poses_npz)

    V = min(len(frame_files), len(R_all))
    frame_files = frame_files[:V]
    R_all = R_all[:V]
    t_all = t_all[:V]

    # Bildgröße
    H, W = _load_color(frame_files[0]).shape[:2]

    # Masken-Frames sammeln
    mask_indices: list[int] = []
    masks: dict[int, np.ndarray] = {}
    if masks_dir and os.path.isdir(masks_dir):
        for i, f in enumerate(frame_files):
            m = _mask_for_frame(masks_dir, f)
            if m is not None:
                if m.shape[:2] != (H, W):
                    m = cv.resize(m, (W, H), interpolation=cv.INTER_NEAREST)
                masks[i] = m
                mask_indices.append(i)
    if not mask_indices:
        _log("[carve] no masks found -> skip", on_log)
        return os.path.join(mesh_dir, "fused_points.ply")

    # View-Auswahl
    if use_all_masks:
        refs = list(mask_indices)
    else:
        n_req = max(1, int(n_views))
        refs = _evenly_pick_from_index_list(mask_indices, n_req)

    _log(f"[carve] using {len(refs)} views (requested {n_views}, available_masks {len(mask_indices)}) -> {refs}", on_log)

    # Carving
    keep = np.ones(P.shape[0], dtype=bool)
    for k, ridx in enumerate(refs):
        if on_progress:
            on_progress(int(100.0 * k / max(1, len(refs))), f"carve v={ridx}")
        m = masks.get(ridx, None)
        if m is None:
            # sollte nicht passieren, da refs aus mask_indices kommt
            continue

        C = (-R_all[ridx].T @ t_all[ridx]).reshape(3)
        Xc = (R_all[ridx] @ (P.T - C.reshape(3, 1))).T
        z = Xc[:, 2]
        u = K[0,0]*Xc[:,0]/np.maximum(z,1e-9) + K[0,2]
        v = K[1,1]*Xc[:,1]/np.maximum(z,1e-9) + K[1,2]
        inside_img = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        ui = np.clip(np.floor(u).astype(np.int32), 0, W - 1)
        vi2 = np.clip(np.floor(v).astype(np.int32), 0, H - 1)

        ok_mask = np.zeros_like(keep, dtype=bool)
        idx = np.where(inside_img)[0]
        if idx.size:
            ok_mask[idx] = (m[vi2[idx], ui[idx]] > 0)

        # Punkt bleibt, wenn er (unsichtbar) ODER (sichtbar & in Maske) ist.
        keep &= (~inside_img) | (ok_mask)

    out_path = os.path.join(mesh_dir, "fused_points.ply")
    save_point_cloud(P[keep], out_path, on_log=on_log, on_progress=on_progress)
    _log(f"[carve] kept {int(np.count_nonzero(keep))}/{P.shape[0]} points", on_log)
    return out_path
