# meshing.py — GPU-only Sparse-Paint (sparse-guided depth completion + back-projection)
#
# Pipeline pro Referenzframe:
#   1) sparse.ply -> in Frame projizieren -> Depth-Seeds (nur Maske, nur Vorderseite).
#   2) Edge-aware Depth-Completion (Torch, GPU), Seeds bleiben hart fixiert.
#   3) Back-Projection aller maskierten Pixel mit finaler Depth -> farbige 3D Punkte.
#   4) Fusion: optionales Voxel-Downsampling + Poisson (Open3D).
#
# Frame-Selektion:
#   - MVS_REF_STRATEGY=bestk|step|auto
#     * bestk: nimm die MVS_REF_TOPK Frames mit meisten Seeds, mit Mindestabstand MVS_REF_MIN_GAP
#     * step : nimm jeden MVS_REF_STEP-ten Frame
#     * auto : bestk falls TOPK>0, sonst step
#
# Wichtige ENV:
#   MVS_DEVICE=cuda (erforderlich, GPU-only)
#   MVS_REF_STRATEGY=auto|bestk|step  (auto)
#   MVS_REF_TOPK=40                   (0 = deaktiviert)
#   MVS_REF_MIN_GAP=2                 (Mindestabstand bei bestk)
#   MVS_REF_STEP=3                    (bei step)
#   MVS_SEED_SAMPLE=200000            (max #Sparse-Punkte für Selektion, zufällig)
#   MVS_SEED_RADIUS=2, MVS_SEED_MIN=200
#   MVS_FILL_ITERS=150, MVS_BETA=4.0, MVS_MASK_PAD=6
#   MVS_EXPORT_MESH=true, MVS_POISSON_DEPTH=10
#   MVS_SCALE=1.0
#
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

GLOBAL_INTRINSICS_K = None  # vom Runner gesetzt

# --------------------- Utils / I/O ---------------------

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

# --------------------- Kamera/Projektion/ROI ---------------------

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

# --------------------- Seeds aus sparse in Frame ---------------------

def _sparse_to_depth_seeds(K, R_ref, t_ref, sparse_pts, mask_ref, seed_radius=2,
                           sample_max: int | None = None):
    """Projiziere sparse in den Ref-Frame → Depth-Seed-Map + Seed-Maske."""
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

# --------------------- Edge-aware GPU-Füllung ---------------------

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

# --------------------- Frame-Scoring & Auswahl ---------------------

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

    # bestk (greedy mit Mindestabstand)
    idx_sorted = np.argsort(-scores)  # desc
    chosen = []
    for i in idx_sorted:
        if topk and len(chosen) >= int(topk): break
        if all(abs(int(i)-int(j)) >= int(min_gap) for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    return chosen

# --------------------- Hauptlauf: Sparse-Paint ---------------------

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

    R_all, t_all, frame_idx = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0: raise RuntimeError(f"Keine Frames in {frames_dir} gefunden.")
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

    # Frame-Scoring & Auswahl
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

        # Depth zurück ins volle Bild
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

        # Back-Projection (nur Maske)
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

        # pro-Ref PLY (optional)
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xw))
            pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1] / 255.0)  # BGR->RGB
            o3d.io.write_point_cloud(os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply"), pcd)

        all_pts.append(Xw.astype(np.float32)); all_cols.append(cols.astype(np.uint8))

    # Fusion
    if all_pts:
        P = np.concatenate(all_pts, axis=0)
        C = np.concatenate(all_cols, axis=0) if all_cols else None
        dense_pts_path = os.path.join(mesh_dir, "dense_points.ply")
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            if C is not None and C.size == P.shape[0]*3:
                pcd.colors = o3d.utility.Vector3dVector(C[:, ::-1] / 255.0)
            if P.shape[0] > 5000:
                try: pcd = pcd.voxel_down_sample(voxel_size=float(_median_nn_distance(P)*0.6))
                except Exception: pass
            o3d.io.write_point_cloud(dense_pts_path, pcd)
        else:
            save_point_cloud(P, dense_pts_path)
        _log(f"[sparse-paint] saved dense points -> {dense_pts_path}", on_log)

        if EXPORT_MESH and HAS_O3D and P.shape[0] >= 1500:
            try:
                pcd = o3d.io.read_point_cloud(dense_pts_path)
                pcd.estimate_normals()
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=int(os.getenv("MVS_POISSON_DEPTH", "10"))
                )
                densities = np.asarray(densities)
                keep = densities > np.quantile(densities, 0.02)
                mesh = mesh.select_by_index(np.where(keep)[0])
                mesh_path = os.path.join(mesh_dir, "dense_mesh.ply")
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                _log(f"[sparse-paint] saved mesh -> {mesh_path}", on_log)
            except Exception as e:
                _log(f"[warn] meshing failed: {e}", on_log)

    _log("[ui] Done Sparse-Paint.", on_log)

# --------------------- Wrapper (API-kompatibel) ---------------------

def _set_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                       min_valid_frac, poisson_depth, mode: str):
    os.environ.setdefault("MVS_DEVICE",        os.getenv("MVS_DEVICE","cuda"))
    os.environ.setdefault("MVS_SCALE",         os.getenv("MVS_SCALE", str(scale)))
    os.environ.setdefault("MVS_EXPORT_MESH",   os.getenv("MVS_EXPORT_MESH","true"))
    os.environ.setdefault("MVS_POISSON_DEPTH", os.getenv("MVS_POISSON_DEPTH", str(poisson_depth)))
    # sinnvolle Defaults für Sparse-Paint:
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
    masks_dir    = os.path.join(features_dir, "masks")
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
    masks_dir    = os.path.join(features_dir, "masks")
    run_sparse_paint_gpu(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                         on_log=on_log, on_progress=on_progress)
