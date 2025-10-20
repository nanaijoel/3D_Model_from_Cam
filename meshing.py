# meshing.py
from __future__ import annotations
import os, glob
from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

GLOBAL_INTRINSICS_K = None  # set by pipeline_runner


def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _log(msg, on_log=None):
    print(msg)
    if on_log:
        try: on_log(msg)
        except: pass

def _load_color(path):
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(path)
    if im.ndim == 2: im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    return im

def _sorted_frames(frames_dir):
    return sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

def _mask_for_frame(masks_dir, frame_path):
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
            if m is not None: return m
    return None

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    if points_xyz is None or len(points_xyz) < 3: return 0.01
    try:
        from scipy.spatial import cKDTree
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz); d, _ = tree.query(points_xyz[idx], k=2)
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
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in pts: f.write(f"{p[0]} {p[1]} {p[2]}\n")
        on_progress and on_progress(100, "Save PLY")
        return out_path
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if len(pts) >= int(filter_min_points):
        voxel = max(1e-4, 0.5 * _median_nn_distance(pts))
        try: pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception: pass
        try: pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception: pass
    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path

def _read_sparse_points(ply_path):
    if not os.path.isfile(ply_path): return None
    if HAS_O3D:
        p = o3d.io.read_point_cloud(ply_path)
        if p is None or np.asarray(p.points).size == 0: return None
        return np.asarray(p.points, dtype=np.float32)
    pts = []
    with open(ply_path, "r") as f:
        header = True; n_verts = 0
        for line in f:
            if header:
                if line.startswith("element vertex"): n_verts = int(line.split()[-1])
                if line.strip() == "end_header": header = False; break
        for _ in range(n_verts):
            parts = f.readline().strip().split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3]); pts.append((x, y, z))
    if not pts: return None
    return np.array(pts, dtype=np.float32)

def _load_poses_npz(poses_npz_path):
    npz = np.load(poses_npz_path)
    R_all = npz["R"]                         # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1,3,1)         # (N,3,1)
    idx = npz.get("frame_idx", np.arange(len(R_all)))
    return R_all, t_all, idx


# Camera / projection / ROI

def _project(K, X_cam):
    x = X_cam[...,0]; y = X_cam[...,1]; z = np.maximum(X_cam[...,2], 1e-9)
    u = K[0,0]*x/z + K[0,2]; v = K[1,1]*y/z + K[1,2]
    return u, v

def _estimate_roi_from_mask(mask, pad=6):
    ys, xs = np.where(mask > 0)
    if ys.size == 0: return (0, mask.shape[0], 0, mask.shape[1])
    y0 = max(0, ys.min()-pad); y1 = min(mask.shape[0], ys.max()+1+pad)
    x0 = max(0, xs.min()-pad); x1 = min(mask.shape[1], xs.max()+1+pad)
    return y0, y1, x0, x1


# Boundary-Volume-Fill

def _boundary_volume_fill_from_seeds(
    K, R_ref, t_ref,
    mask_ref: np.ndarray,
    seed_depth: np.ndarray,   # from _sparse_to_depth_seeds
    seed_mask:  np.ndarray,   # from _sparse_to_depth_seeds (if mask pixel = 1 --> Seed)
    sample_step_px: int = 3,  # "hole" pixel grid
    samples_per_pix: int = 8, # create points per "hole" pixel
    px_sigma: float = 1.0,    # (dx,dy) jitter in px
    z_sigma_rel: float = 0.01,# z-jitter rel. to z*
    focus_bottom: bool = True,
    focus_frac: float = 0.40, # focus lower part
    z_bias_rel: float = 0.02,
    dy_bias_px: float = 0.8
):

    H, W = mask_ref.shape[:2]
    seed_valid = (seed_mask > 0)
    holes = (mask_ref > 0) & (~seed_valid)

    if focus_bottom:
        y_cut = int((1.0 - float(np.clip(focus_frac, 0.0, 1.0))) * H)
        m = np.zeros_like(holes, dtype=bool); m[y_cut:] = True
        holes &= m

    ys, xs = np.where(holes)
    if ys.size == 0:
        return None, None

    step = max(1, int(sample_step_px))
    ys = ys[::step]; xs = xs[::step]
    if ys.size == 0:
        return None, None

    try:
        from scipy import ndimage as ndi
        inv = ~seed_valid
        _, (iy, ix) = ndi.distance_transform_edt(inv, return_distances=True, return_indices=True)
        z_map = seed_depth.copy()
        z_map[~seed_valid] = np.nan
        z_near = z_map[iy[ys, xs], ix[ys, xs]]
    except Exception:
        dv = seed_depth[seed_valid]
        z_med = float(np.median(dv)) if dv.size > 0 else 0.5
        z_near = np.full(ys.shape, z_med, dtype=np.float32)

    bad = ~np.isfinite(z_near)
    if np.any(bad):
        dv = seed_depth[seed_valid]
        z_med = float(np.median(dv)) if dv.size > 0 else 0.5
        z_near[bad] = z_med

    S = max(1, int(samples_per_pix))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    Ccam = (-R_ref.T @ t_ref).reshape(3)

    seed_hash = int((float(abs(t_ref[0]))*1e6) % (2**32 - 1))
    rng = np.random.default_rng(seed_hash if seed_hash > 0 else 42)

    all_world = []
    pix_idx = []

    for y, x, z0 in zip(ys, xs, z_near.astype(np.float32)):
        if not np.isfinite(z0) or z0 <= 1e-6:
            continue
        dx = rng.normal(0.0, px_sigma, size=S).astype(np.float32)
        dy = rng.normal(dy_bias_px, px_sigma, size=S).astype(np.float32)  # bias nach unten
        dz = rng.normal(z_bias_rel * max(z0, 1e-3), z_sigma_rel * max(z0, 1e-3), size=S).astype(np.float32)

        z = np.maximum(z0 + dz, 1e-6)
        u = (x + dx).astype(np.float32)
        v = (y + dy).astype(np.float32)

        Xc = np.stack([
            (u - cx) / fx * z,
            (v - cy) / fy * z,
            z
        ], axis=1)  # (S,3) Kamera
        Xw = (R_ref.T @ Xc.T + Ccam.reshape(3,1)).T  # (S,3) Welt

        all_world.append(Xw)
        pix_idx.extend([(int(y), int(x))] * S)

    if not all_world:
        return None, None

    Xw_fill = np.concatenate(all_world, axis=0).astype(np.float32)
    pix_idx = np.array(pix_idx, dtype=np.int32)
    return Xw_fill, pix_idx


# Seeds from sparse

def _sparse_to_depth_seeds(K, R_ref, t_ref, sparse_pts, mask_ref, seed_radius=2,
                           sample_max: int | None = None):
    """Project sparse into the reference frame - Depth seed map + Seed mask."""
    H, W = mask_ref.shape[:2]
    seed_depth = np.full((H, W), np.nan, np.float32)
    seed_mask  = np.zeros((H, W), np.uint8)
    if sparse_pts is None or sparse_pts.size == 0:
        return seed_depth, seed_mask

    pts = sparse_pts
    if (sample_max is not None) and (pts.shape[0] > sample_max):
        sel = np.random.choice(pts.shape[0], sample_max, replace=False)
        pts = pts[sel]

    C = (-R_ref.T @ t_ref).reshape(3)
    X_cam = (R_ref @ (pts.T - C.reshape(3,1))).T  # Nx3
    Z = X_cam[:,2]
    u, v = _project(K, X_cam)
    H, W = mask_ref.shape[:2]
    in_img = (u>=0)&(u<W)&(v>=0)&(v<H)&(Z>1e-6)
    if not np.any(in_img): return seed_depth, seed_mask

    ui = np.floor(u[in_img]).astype(np.int32); vi = np.floor(v[in_img]).astype(np.int32)
    ui = np.clip(ui, 0, W-1); vi = np.clip(vi, 0, H-1)

    keep = (mask_ref[vi, ui] > 0)
    ui = ui[keep]; vi = vi[keep]; z = Z[in_img][keep]
    if ui.size == 0: return seed_depth, seed_mask

    r = int(max(0, seed_radius))
    if r == 0:
        seed_depth[vi, ui] = z.astype(np.float32)
        seed_mask[vi, ui]  = 1
        return seed_depth, seed_mask

    for x,y,zz in zip(ui, vi, z.astype(np.float32)):
        x0 = max(0, x-r); x1 = min(W, x+r+1)
        y0 = max(0, y-r); y1 = min(H, y+r+1)
        sub = seed_depth[y0:y1, x0:x1]
        sub_mask = seed_mask[y0:y1, x0:x1]
        write_here = (sub_mask == 0)
        sub[write_here] = zz
        sub_mask[write_here] = 1
        seed_depth[y0:y1, x0:x1] = sub
        seed_mask[y0:y1, x0:x1]  = sub_mask
    return seed_depth, seed_mask


# Edge-aware GPU depth fill

def _edge_aware_fill_gpu(seed_depth, seed_mask, ref_img, roi, mask_ref,
                         iters=150, beta=4.0, device_str="cuda"):
    import torch
    device = torch.device(device_str)
    y0,y1,x0,x1 = roi
    Hroi, Wroi = y1-y0, x1-x0

    I = ref_img[y0:y1, x0:x1]
    if I.ndim == 3:
        Ig = (0.2989*I[:,:,2] + 0.5870*I[:,:,1] + 0.1140*I[:,:,0]).astype(np.float32)
    else:
        Ig = I.astype(np.float32)
    I_t  = torch.from_numpy(Ig/255.0).to(device=device).view(1,1,Hroi,Wroi)

    M_ref = (mask_ref[y0:y1, x0:x1] > 0).astype(np.uint8)
    M_t   = torch.from_numpy(M_ref).to(device=device, dtype=torch.float32).view(1,1,Hroi,Wroi)

    S_depth = seed_depth[y0:y1, x0:x1]
    S_mask  = seed_mask[y0:y1, x0:x1]
    S_t = torch.from_numpy(np.nan_to_num(S_depth, nan=0.0)).to(device=device, dtype=torch.float32).view(1,1,Hroi,Wroi)
    A_t = torch.from_numpy((S_mask>0).astype(np.uint8)).to(device=device, dtype=torch.float32).view(1,1,Hroi,Wroi)

    with torch.no_grad():
        k = torch.ones((1,1,3,3), device=device, dtype=torch.float32)
        mean_seed = torch.nn.functional.conv2d(S_t, k, padding=1) / torch.clamp(torch.nn.functional.conv2d(A_t, k, padding=1), min=1.0)
        D = torch.where(A_t>0, S_t, mean_seed)
        v = S_t[A_t>0]
        if v.numel() >= 10:
            lo = torch.quantile(v, 0.02); hi = torch.quantile(v, 0.98)
            D = torch.clamp(D, float(lo.item()), float(hi.item()))

    def shift(t, dy, dx):
        return torch.nn.functional.pad(t[:,:,max(0,-dy):Hroi-max(0,dy), max(0,-dx):Wroi-max(0,dx)],
                                       (max(0,dx),max(0,-dx),max(0,dy),max(0,-dy)), mode="replicate")

    with torch.no_grad():
        I_r = shift(I_t, 0, 1);  I_l = shift(I_t, 0,-1)
        I_d = shift(I_t, 1, 0);  I_u = shift(I_t,-1, 0)
        w_r = torch.exp(-beta * torch.abs(I_t - I_r))
        w_l = torch.exp(-beta * torch.abs(I_t - I_l))
        w_d = torch.exp(-beta * torch.abs(I_t - I_d))
        w_u = torch.exp(-beta * torch.abs(I_t - I_u))
        Mm = (M_t>0).float()
        w_r *= Mm; w_l *= Mm; w_d *= Mm; w_u *= Mm

    lam = 1.0
    for _ in range(int(iters)):
        Dr = shift(D, 0, 1); Dl = shift(D, 0,-1)
        Dd = shift(D, 1, 0); Du = shift(D,-1, 0)
        num = w_r*Dr + w_l*Dl + w_d*Dd + w_u*Du
        den = torch.clamp(w_r + w_l + w_d + w_u, min=1e-6)
        Dnew = num/den
        D    = torch.where(A_t>0, S_t, (1-lam)*D + lam*Dnew)

    return D.squeeze().detach().cpu().numpy()


# Frame scoring and selection

def _score_frames_by_seeds(K, R_all, t_all, sparse_pts, masks, seed_radius: int,
                           sample_max: int, on_log=None):
    N = len(masks)
    if sparse_pts is None or sparse_pts.size == 0:
        _log("[sparse-paint] WARNING: sparse.ply leer → Scoring=0", on_log)
        return np.zeros(N, dtype=np.int32)

    pts = sparse_pts
    if (sample_max is not None) and (pts.shape[0] > sample_max):
        sel = np.random.choice(pts.shape[0], sample_max, replace=False)
        pts = pts[sel]

    scores = np.zeros(N, dtype=np.int32)
    for i in range(N):
        m = masks[i]
        if m is None: scores[i] = 0; continue
        C = (-R_all[i].T @ t_all[i]).reshape(3)
        X_cam = (R_all[i] @ (pts.T - C.reshape(3,1))).T
        Z = X_cam[:,2]
        u, v = _project(K, X_cam)
        H, W = m.shape[:2]
        in_img = (u>=0)&(u<W)&(v>=0)&(v<H)&(Z>1e-6)
        if not np.any(in_img): scores[i] = 0; continue
        ui = np.clip(np.floor(u[in_img]).astype(np.int32), 0, W-1)
        vi = np.clip(np.floor(v[in_img]).astype(np.int32), 0, H-1)
        scores[i] = int(np.count_nonzero(m[vi, ui] > 0))
    return scores

def _choose_ref_indices(N, scores, strategy: str, step: int, topk: int, min_gap: int):
    if strategy == "auto":
        strategy = "bestk" if topk and topk > 0 else "step"

    if strategy == "step":
        step = max(1, int(step))
        return list(range(0, N, step))

    # bestk
    idx_sorted = np.argsort(-scores)  # desc
    chosen = []
    for i in idx_sorted:
        if topk and len(chosen) >= int(topk): break
        if all(abs(int(i)-int(j)) >= int(min_gap) for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    return chosen


def _save_colored_ply_any(P, C, out_path):
    _mkdir(os.path.dirname(out_path) or ".")
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        if C is not None and C.shape[0] == P.shape[0]:
            pc.colors = o3d.utility.Vector3dVector((C[:, ::-1] / 255.0).astype(np.float64))  # BGR->RGB
        o3d.io.write_point_cloud(out_path, pc)
        return out_path
    except Exception:
        with open(out_path,"w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {P.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if C is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            if C is None:
                for p in P: f.write(f"{p[0]} {p[1]} {p[2]}\n")
            else:
                for p,c in zip(P, C.astype(np.uint8)):
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
        return out_path

def _max_consecutive_true_per_row(B: np.ndarray) -> np.ndarray:
    if B.size == 0:
        return np.zeros((B.shape[0],), dtype=np.int32)
    N, V = B.shape
    run = np.zeros((N, V), dtype=np.int32)
    run[:, 0] = B[:, 0].astype(np.int32)
    for j in range(1, V):
        run[:, j] = (run[:, j-1] + 1) * B[:, j].astype(np.int32)
    return run.max(axis=1)

def _carve_points_with_masks(P, K, R_all, t_all, masks, depth_dir=None,
                             mode="all", use_depth=False, depth_tol=0.03, chunk=400000, on_log=None):
    V = len(masks); N = P.shape[0]
    if V == 0 or N == 0:
        return np.ones(N, np.bool_)

    consec_N = int(float(os.getenv("CARVE_CONSEC_N", "3")))
    min_views_keep = int(float(os.getenv("CARVE_MIN_VIEWS", "0")))  # 0 = aus

    remove_mat = np.zeros((N, V), dtype=bool)
    keep_mat   = np.zeros((N, V), dtype=bool)  # Sight inside mask

    for vi in range(V):
        m = masks[vi]
        if m is None:
            continue
        H,W = m.shape[:2]
        R = R_all[vi]; t = t_all[vi]
        D = None
        if use_depth and depth_dir:
            p = os.path.join(depth_dir, f"depth_{vi:04d}.npy")
            if os.path.isfile(p):
                try: D = np.load(p)
                except: D = None

        Cc = (-R.T @ t).reshape(3)
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            Q = P[s:e]
            Xc = (R @ (Q.T - Cc.reshape(3,1))).T
            z  = Xc[:,2]
            u  = K[0,0]*Xc[:,0]/np.maximum(z,1e-9) + K[0,2]
            v  = K[1,1]*Xc[:,1]/np.maximum(z,1e-9) + K[1,2]
            ui = np.floor(u).astype(np.int32)
            vi2= np.floor(v).astype(np.int32)
            inside = (z>1e-6)&(ui>=0)&(ui<W)&(vi2>=0)&(vi2<H)

            rem = ~inside
            if np.any(inside):
                ui = np.clip(ui,0,W-1); vi2 = np.clip(vi2,0,H-1)
                m_inside = (m[vi2, ui] > 0)
                if D is not None:
                    d = D[vi2, ui]
                    m_inside &= (z <= (d + depth_tol*(1.0 + d)))
                rem_inside = ~m_inside

                tmp = remove_mat[s:e, vi]
                tmp2= keep_mat[s:e, vi]
                tmp |= rem_inside
                tmp2 |= m_inside
                remove_mat[s:e, vi] = tmp
                keep_mat[s:e, vi]   = tmp2
            else:
                remove_mat[s:e, vi] |= True

    if mode == "consecutive":
        max_run = _max_consecutive_true_per_row(remove_mat)
        keep = (max_run < max(1, consec_N))
        if min_views_keep > 0:
            keep &= (keep_mat.sum(axis=1) >= min_views_keep)
        return keep

    counts = keep_mat.sum(axis=1).astype(np.int32)
    if mode == "all":
        return counts == V
    if mode == "majority":
        return counts >= int(np.ceil(V/2.0))
    if mode == "any":
        return counts >= 1
    return counts >= int(np.ceil(V/2.0))

def _carve_and_save_from_arrays(mesh_dir, frames_dir, poses_npz, masks_dir,
                                P_list, C_list, on_log=None):

    _log("[carve] in-memory carve start", on_log)
    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    P = np.concatenate(P_list, axis=0) if len(P_list) else np.zeros((0,3), np.float64)
    C = (np.concatenate(C_list, axis=0) if (len(C_list) and all(c is not None for c in C_list))
         else None)

    _log(f"[carve] stacked (in-memory): {P.shape[0]} pts", on_log)

    # optional: save raw (pre-carve) merged for debug
    if os.getenv("MVS_SAVE_RAW_BEFORE_CARVE", "false").lower() in ("1","true","yes","on"):
        raw_path = os.path.join(mesh_dir, "fused_points_raw.ply")
        _save_colored_ply_any(P, C, raw_path)
        _log(f"[carve] wrote pre-carve raw -> {raw_path}", on_log)

    # Load poses & masks
    R_all, t_all, _ = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    masks_all   = [_mask_for_frame(masks_dir, f) for f in frame_files]

    use_all   = os.getenv("CARVE_USE_ALL_MASKS", "true").lower() in ("1","true","yes","on")
    mode      = (os.getenv("CARVE_MODE", "consecutive") or "consecutive").strip().lower()
    use_depth = os.getenv("CARVE_USE_DEPTH", "false").lower() in ("1","true","yes","on")
    depth_tol = float(os.getenv("CARVE_DEPTH_TOL", "0.03"))
    chunk     = int(float(os.getenv("CARVE_CHUNK", "400000")))
    # defaults for „consecutive“
    os.environ.setdefault("CARVE_CONSEC_N",   os.getenv("CARVE_CONSEC_N", "10"))
    os.environ.setdefault("CARVE_MIN_VIEWS",  os.getenv("CARVE_MIN_VIEWS", "0"))

    if use_all:
        view_ids = list(range(min(len(frame_files), len(R_all))))
    else:
        view_ids = list(range(min(len(frame_files), len(R_all))))

    masks = [masks_all[i] if 0 <= i < len(masks_all) else None for i in view_ids]
    R_sel = np.array([R_all[i] for i in view_ids])
    t_sel = np.array([t_all[i] for i in view_ids])
    depth_dir = os.path.join(mesh_dir, "depth")

    keep = _carve_points_with_masks(
        P, K, R_sel, t_sel, masks,
        depth_dir=depth_dir, mode=mode,
        use_depth=use_depth, depth_tol=depth_tol,
        chunk=chunk, on_log=on_log
    )

    Pk = P[keep]
    Ck = (C[keep] if C is not None else None)
    _log(f"[carve] after carving: {Pk.shape[0]} pts", on_log)

    # optional downsample & save fused
    out_path = os.path.join(mesh_dir, "fused_points.ply")
    try:
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pk))
            if Ck is not None:
                pcd.colors = o3d.utility.Vector3dVector((Ck[:, ::-1] / 255.0))
            if Pk.shape[0] > 8000:
                vox = max(1e-4, 0.6 * _median_nn_distance(Pk))
                pcd = pcd.voxel_down_sample(voxel_size=float(vox))
            o3d.io.write_point_cloud(out_path, pcd)
        else:
            _save_colored_ply_any(Pk, Ck, out_path)
    finally:
        _log(f"[carve] saved -> {out_path}", on_log)

    return out_path


def run_sparse_paint_gpu(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                         on_log=None, on_progress=None):
    _mkdir(mesh_dir); depth_dir = os.path.join(mesh_dir, "depth"); _mkdir(depth_dir)

    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None: raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    DEVICE = os.getenv("MVS_DEVICE", "cuda").lower()
    if DEVICE != "cuda":
        raise RuntimeError("GPU-only Variante. Setze MVS_DEVICE=cuda (CUDA muss verfügbar sein).")

    REF_STRAT  = os.getenv("MVS_REF_STRATEGY", "auto").strip().lower()
    REF_TOPK   = int(float(os.getenv("MVS_REF_TOPK", "0")))
    REF_MIN_G  = int(float(os.getenv("MVS_REF_MIN_GAP", "2")))
    REF_STEP   = int(float(os.getenv("MVS_REF_STEP", "3")))
    MASK_PAD   = int(float(os.getenv("MVS_MASK_PAD", "6")))
    SCALE      = float(os.getenv("MVS_SCALE", "1.0"))
    SEED_R     = int(float(os.getenv("MVS_SEED_RADIUS", "2")))
    SEED_MIN   = int(float(os.getenv("MVS_SEED_MIN", "200")))
    BETA       = float(os.getenv("MVS_BETA", "4.0"))
    FILL_ITERS = int(float(os.getenv("MVS_FILL_ITERS", "150")))
    SAMPLE_MAX = int(float(os.getenv("MVS_SEED_SAMPLE", "200000")))
    EXPORT_MESH= os.getenv("MVS_EXPORT_MESH", "true").lower() == "true"
    DEBUG_SAVE_PER_REF = os.getenv("MVS_DEBUG_SAVE_PER_REF", "false").lower() in ("1","true","yes","on")

    R_all, t_all, frame_idx = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0: raise RuntimeError(f"No frames found in {frames_dir}.")
    images = [_load_color(f) for f in frame_files]

    if masks_dir and os.path.isdir(masks_dir):
        masks = [_mask_for_frame(masks_dir, f) for f in frame_files]
    else:
        masks = [np.ones(images[0].shape[:2], np.uint8)*255 for _ in frame_files]

    # optional scale
    if abs(SCALE - 1.0) > 1e-6:
        ims, msks = [], []
        for im, m in zip(images, masks):
            ims.append(cv.resize(im, dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_LINEAR))
            msks.append(cv.resize(m,  dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_NEAREST) if m is not None else None)
        images, masks = ims, msks
        K = K.copy()
        K[0,0]*=SCALE; K[1,1]*=SCALE; K[0,2]*=SCALE; K[1,2]*=SCALE
        globals()["GLOBAL_INTRINSICS_K"] = K

    sparse_path = os.path.join(mesh_dir, "sparse.ply")
    sparse_pts  = _read_sparse_points(sparse_path)

    # Frame scoring and selection
    scores = _score_frames_by_seeds(K, R_all, t_all, sparse_pts, masks, SEED_R, SAMPLE_MAX, on_log)
    refs = _choose_ref_indices(len(images), scores, REF_STRAT, REF_STEP, REF_TOPK, REF_MIN_G)
    _log(f"[sparse-paint] selection strategy={REF_STRAT}  step={REF_STEP}  topk={REF_TOPK}  min_gap={REF_MIN_G}", on_log)
    _log(f"[sparse-paint] selected refs: {refs[:10]}{'...' if len(refs)>10 else ''} (total {len(refs)})", on_log)

    all_pts = []
    all_cols = []

    for k, ridx in enumerate(refs):
        on_progress and on_progress(int(100.0 * k / max(1,len(refs))), f"sparse-paint ref={ridx}")
        ref_img = images[ridx]
        ref_msk = masks[ridx] if masks[ridx] is not None else np.ones(ref_img.shape[:2], np.uint8)*255

        # ROI
        y0,y1,x0,x1 = _estimate_roi_from_mask(ref_msk, pad=MASK_PAD)
        ROI = (y0,y1,x0,x1)

        # Seeds
        seed_depth, seed_mask = _sparse_to_depth_seeds(K, R_all[ridx], t_all[ridx], sparse_pts, ref_msk,
                                                       seed_radius=SEED_R, sample_max=None)
        n_seeds = int(np.count_nonzero(seed_mask))
        _log(f"[sparse-paint] ref={ridx} seeds={n_seeds} (score={scores[ridx]})", on_log)
        if n_seeds < SEED_MIN:
            _log(f"[sparse-paint] skip ref={ridx} (zu wenige Seeds: {n_seeds} < {SEED_MIN})", on_log)
            continue

        # Edge-aware Fill (GPU)
        depth_roi = _edge_aware_fill_gpu(seed_depth, seed_mask, ref_img, ROI, ref_msk,
                                         iters=FILL_ITERS, beta=BETA, device_str="cuda")

        # Depth back to full image
        depth_full = np.zeros(ref_msk.shape, np.float32)
        depth_full[y0:y1, x0:x1] = depth_roi

        # Preview
        d = depth_full.copy(); d[ref_msk==0] = np.nan
        dv = d[np.isfinite(d)]
        if dv.size>10:
            lo,hi = np.percentile(dv,2), np.percentile(dv,98); hi = max(hi, lo+1e-6)
            vis = (np.clip((d-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        else:
            vis = np.zeros_like(ref_msk)
        cv.imwrite(os.path.join(depth_dir, f"depth_{ridx:04d}.png"), vis)
        np.save(os.path.join(depth_dir, f"depth_{ridx:04d}.npy"), depth_full)

        # Boundary-volume fill
        BV_ENABLE = os.getenv("MVS_BOUNDARY_FILL_ENABLE", "true").lower() in ("1","true","yes","on")
        if BV_ENABLE:
            bv_step   = int(float(os.getenv("MVS_BVF_STEP_PX", "3")))
            bv_spp    = int(float(os.getenv("MVS_BVF_SAMPLES", "10")))
            bv_px_sig = float(os.getenv("MVS_BVF_PX_SIGMA", "1.2"))
            bv_z_rel  = float(os.getenv("MVS_BVF_Z_SIGMA_REL", "0.015"))
            bv_focus  = os.getenv("MVS_BVF_FOCUS_BOTTOM", "true").lower() in ("1","true","yes","on")
            bv_frac   = float(os.getenv("MVS_BVF_FOCUS_FRAC", "0.40"))
            bv_zbias  = float(os.getenv("MVS_BVF_Z_BIAS_REL", "0.02"))
            bv_dybias = float(os.getenv("MVS_BVF_DY_BIAS_PX", "0.8"))

            Xw_fill, pix_idx = _boundary_volume_fill_from_seeds(
                K, R_all[ridx], t_all[ridx],
                ref_msk, seed_depth, seed_mask,
                sample_step_px=bv_step, samples_per_pix=bv_spp,
                px_sigma=bv_px_sig, z_sigma_rel=bv_z_rel,
                focus_bottom=bv_focus, focus_frac=bv_frac,
                z_bias_rel=bv_zbias, dy_bias_px=bv_dybias
            )
            if Xw_fill is not None and Xw_fill.size > 0:
                cols_fill = ref_img[pix_idx[:, 0], pix_idx[:, 1]]
                all_pts.append(Xw_fill.astype(np.float32))
                all_cols.append(cols_fill.astype(np.uint8))
                _log(f"[bvf] ref={ridx} fill_points={Xw_fill.shape[0]} step={bv_step} spp={bv_spp} "
                     f"px_sigma={bv_px_sig} z_rel={bv_z_rel} z_bias={bv_zbias} dy_bias={bv_dybias} "
                     f"focus_bottom={bv_focus} frac={bv_frac}", on_log)

        # Back-Projection
        ys, xs = np.where(ref_msk>0)
        if ys.size == 0: continue
        z = depth_full[ys, xs].astype(np.float32)
        keep = np.isfinite(z) & (z>1e-6)
        ys = ys[keep]; xs = xs[keep]; z = z[keep]
        if ys.size == 0: continue

        x_cam = np.stack([(xs - K[0,2]) * z / K[0,0],
                          (ys - K[1,2]) * z / K[1,1],
                          z], axis=1)
        Xw = (R_all[ridx].T @ x_cam.T + (-R_all[ridx].T @ t_all[ridx]).reshape(3,1)).T
        cols = ref_img[ys, xs]

        if DEBUG_SAVE_PER_REF and HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xw))
            pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1] / 255.0)  # BGR->RGB
            o3d.io.write_point_cloud(os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply"), pcd)

        all_pts.append(Xw.astype(np.float32)); all_cols.append(cols.astype(np.uint8))

    _log("[ui] Done Sparse-Paint.", on_log)

    if os.getenv("CARVE_ENABLE", "false").lower() in ("1","true","yes","on"):
        out = _carve_and_save_from_arrays(mesh_dir, frames_dir, poses_npz, masks_dir,
                                          all_pts, all_cols, on_log=on_log)
        _log(f"[ui] In-memory Merge&Carve -> {out}", on_log)


# ENV wrapper

def _set_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                       min_valid_frac, poisson_depth, mode: str):
    os.environ.setdefault("MVS_DEVICE",        os.getenv("MVS_DEVICE","cuda"))
    os.environ.setdefault("MVS_SCALE",         os.getenv("MVS_SCALE", str(scale)))
    os.environ.setdefault("MVS_EXPORT_MESH",   os.getenv("MVS_EXPORT_MESH","true"))
    os.environ.setdefault("MVS_POISSON_DEPTH", os.getenv("MVS_POISSON_DEPTH", str(poisson_depth)))

    os.environ.setdefault("MVS_REF_STRATEGY",  os.getenv("MVS_REF_STRATEGY","auto"))
    os.environ.setdefault("MVS_REF_STEP",      os.getenv("MVS_REF_STEP","3"))
    os.environ.setdefault("MVS_REF_TOPK",      os.getenv("MVS_REF_TOPK","0"))
    os.environ.setdefault("MVS_REF_MIN_GAP",   os.getenv("MVS_REF_MIN_GAP","2"))
    os.environ.setdefault("MVS_SEED_SAMPLE",   os.getenv("MVS_SEED_SAMPLE","200000"))
    os.environ.setdefault("MVS_SEED_RADIUS",   os.getenv("MVS_SEED_RADIUS","2"))
    os.environ.setdefault("MVS_SEED_MIN",      os.getenv("MVS_SEED_MIN","200"))
    os.environ.setdefault("MVS_FILL_ITERS",    os.getenv("MVS_FILL_ITERS","150"))
    os.environ.setdefault("MVS_BETA",          os.getenv("MVS_BETA","4.0"))
    os.environ.setdefault("MVS_MASK_PAD",      os.getenv("MVS_MASK_PAD","6"))
    _log(f"[sparse-paint] mode={mode}")

def reconstruct_mvs_depth_and_mesh(paths, K,
                                   scale=1.0, max_views=8, n_planes=128,
                                   depth_expand=0.08, patch=7,
                                   cost_thr=0.55, min_valid_frac=0.01,
                                   poisson_depth=10,
                                   on_log=None, on_progress=None):
    _set_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                       min_valid_frac, poisson_depth, mode="single")
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()
    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir     = os.path.join(root, "mesh")
    frames_dir   = os.path.join(root, "raw_frames")
    features_dir = os.path.join(root, "features")
    poses_npz    = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir    = os.path.join(root, "features", "masks")
    run_sparse_paint_gpu(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                         on_log=on_log, on_progress=on_progress)

def reconstruct_mvs_depth_and_mesh_all(paths, K,
                                       scale=1.0, max_views=8, n_planes=128,
                                       depth_expand=0.08, patch=7,
                                       cost_thr=0.55, min_valid_frac=0.01,
                                       poisson_depth=10,
                                       on_log=None, on_progress=None):
    _set_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                       min_valid_frac, poisson_depth, mode="all")
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()
    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir     = os.path.join(root, "mesh")
    frames_dir   = os.path.join(root, "raw_frames")
    features_dir = os.path.join(root, "features")
    poses_npz    = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir    = os.path.join(root, "features", "masks")
    run_sparse_paint_gpu(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                         on_log=on_log, on_progress=on_progress)


def _read_points_ref_with_colors(ply_path):
    if not os.path.isfile(ply_path):
        return None, None
    try:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(ply_path)
        if pc is None or np.asarray(pc.points).size == 0:
            return None, None
        P = np.asarray(pc.points, dtype=np.float64)
        C = None
        if len(pc.colors) > 0:
            C = (np.asarray(pc.colors)[:, ::-1] * 255.0).astype(np.float64)  # RGB->BGR uint8
        return P, C
    except Exception:
        # ASCII-Fallback
        with open(ply_path, "r") as f:
            header = True; n=0; has_color=False
            while header:
                line = f.readline()
                if not line: break
                if line.startswith("element vertex"): n = int(line.split()[-1])
                if line.startswith("property uchar red"): has_color=True
                if line.strip() == "end_header":
                    header = False; break
            pts=[]; cols=[]
            for _ in range(n):
                parts = f.readline().strip().split()
                if len(parts) < 3: continue
                x,y,z = map(float, parts[:3]); pts.append((x,y,z))
                if has_color and len(parts) >= 6:
                    r,g,b = map(float, parts[3:6]); cols.append((b,g,r))  # BGR
        P = np.asarray(pts, np.float64)
        C = np.asarray(cols, np.float64) if cols else None
        return P, C

def merge_all_points_ref_and_carve(mesh_dir, frames_dir, masks_dir, poses_npz, on_log=None):
    _log("[carve] merge & carve start", on_log)
    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt.")

    use_all   = os.getenv("CARVE_USE_ALL_MASKS", "true").lower() in ("1", "true", "yes", "on")
    mode      = (os.getenv("CARVE_MODE", "consecutive") or "consecutive").strip().lower()
    use_depth = os.getenv("CARVE_USE_DEPTH", "false").lower() in ("1", "true", "yes", "on")
    depth_tol = float(os.getenv("CARVE_DEPTH_TOL", "0.03"))
    chunk     = int(float(os.getenv("CARVE_CHUNK", "400000")))
    os.environ.setdefault("CARVE_CONSEC_N",   os.getenv("CARVE_CONSEC_N", "10"))
    os.environ.setdefault("CARVE_MIN_VIEWS",  os.getenv("CARVE_MIN_VIEWS", "0"))

    plys = sorted(glob.glob(os.path.join(mesh_dir, "points_ref_*.ply")))
    if not plys:
        raise RuntimeError("[carve] points_ref_*.ply not found.")

    P_list, C_list = [], []
    for p in plys:
        P, C = _read_points_ref_with_colors(p)
        if P is None or P.size == 0:
            continue
        P_list.append(P.astype(np.float64))
        C_list.append(C if (C is not None and C.shape[0] == P.shape[0]) else None)
        _log(f"[carve] load {os.path.basename(p)} : {P.shape[0]} pts", on_log)

    if not P_list:
        raise RuntimeError("[carve] no points loaded.")

    P = np.concatenate(P_list, axis=0)
    C = np.concatenate(C_list, axis=0) if all(c is not None for c in C_list) else None
    _log(f"[carve] stacked: {P.shape[0]} pts", on_log)

    R_all, t_all, _ = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    masks_all   = [_mask_for_frame(masks_dir, f) for f in frame_files]

    if use_all:
        view_ids = list(range(min(len(frame_files), len(R_all))))
    else:
        def _extract_view_id(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "points" and parts[1] == "ref":
                try:
                    return int(parts[2])
                except ValueError:
                    return None
            return None

        view_ids = sorted({vid for vid in (_extract_view_id(p) for p in plys) if vid is not None})
        _log(f"[carve] using view_ids={view_ids}", on_log)

    masks = [masks_all[i] if 0 <= i < len(masks_all) else None for i in view_ids]
    R_sel = np.array([R_all[i] for i in view_ids])
    t_sel = np.array([t_all[i] for i in view_ids])
    depth_dir = os.path.join(mesh_dir, "depth")

    keep = _carve_points_with_masks(
        P, K, R_sel, t_sel, masks,
        depth_dir=depth_dir, mode=mode,
        use_depth=use_depth, depth_tol=depth_tol,
        chunk=chunk, on_log=on_log
    )

    Pk = P[keep]
    Ck = (C[keep] if C is not None else None)
    _log(f"[carve] after carving: {Pk.shape[0]} pts", on_log)

    out_path = os.path.join(mesh_dir, "fused_points.ply")
    try:
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pk))
            if Ck is not None:
                pcd.colors = o3d.utility.Vector3dVector((Ck[:, ::-1] / 255.0))
            if Pk.shape[0] > 8000:
                vox = max(1e-4, 0.6 * _median_nn_distance(Pk))
                pcd = pcd.voxel_down_sample(voxel_size=float(vox))
            o3d.io.write_point_cloud(out_path, pcd)
        else:
            _save_colored_ply_any(Pk, Ck, out_path)
    finally:
        _log(f"[carve] saved -> {out_path}", on_log)

    return out_path
