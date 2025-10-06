# meshing.py
import os
import glob
import math
import numpy as np
import cv2 as cv

# Open3D ist optional (für PLY I/O und Meshing)
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _median_nn_distance(points_xyz: np.ndarray) -> float:
    """Robuste Schätzung eines mittleren Next-Neighbor-Abstands (mit Fallback)."""
    if points_xyz is None or len(points_xyz) < 3:
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

def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _log(msg, on_log=None):
    print(msg)
    if on_log:
        try: on_log(msg)
        except: pass

def _load_color(path):
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    return im

def _to_gray_u8(im):
    if im.ndim == 3:
        g = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        g = im.copy()
    if g.dtype != np.uint8:
        g = cv.convertScaleAbs(g)
    return g

def _sorted_frames(frames_dir):
    # akzeptiert frame_????_*.png
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return files

def _mask_for_frame(masks_dir, frame_path):
    """
    Robust gegen unterschiedliche Benennungen:
      frame_0000_src_000000.png -> frame_0000_src_000000_mask.png
      frame_0000_src_000000.png -> frame_0000_mask.png (Fallback)
    """
    base = os.path.basename(frame_path)
    stem, _ = os.path.splitext(base)
    cands = [
        os.path.join(masks_dir, f"{stem}_mask.png"),
        os.path.join(masks_dir, f"{stem}_mask.jpg"),
    ]
    # Fallback: frame_0000_mask.png
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

def _load_poses_npz(poses_npz_path):
    npz = np.load(poses_npz_path)
    # Erwartete Keys: 'R', 't', 'frame_idx' (und evtl. 'C')
    R_all = npz["R"]         # (N,3,3)
    t_all = npz["t"]         # (N,3)
    frame_idx = npz.get("frame_idx", np.arange(len(R_all)))
    # Sicherstellen shape
    t_all = t_all.reshape(-1, 3, 1)
    return R_all, t_all, frame_idx

def _project(K, X_cam):
    x = X_cam[..., 0]
    y = X_cam[..., 1]
    z = X_cam[..., 2]
    z = np.maximum(z, 1e-9)
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v

# --------------------------------------------------------------------------
# 1) Sparse Point Cloud speichern (EXAKT wie gefordert)
# --------------------------------------------------------------------------
def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None) -> str:
    """
    Speichert eine (leicht gefilterte) Punktwolke als PLY.
    """
    _log(f"[mesh] save point cloud -> {out_path}", on_log)
    _mkdir(os.path.dirname(out_path) or ".")

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if not HAS_O3D:
        # Minimaler ASCII-PLY-Writer als Fallback
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        on_progress and on_progress(100, "Save PLY")
        return out_path

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

# --------------------------------------------------------------------------
# 2) Depth-Bounds aus sparse.ply ∩ Maske
# --------------------------------------------------------------------------
def _read_sparse_points(ply_path):
    if not os.path.isfile(ply_path):
        return None
    if HAS_O3D:
        p = o3d.io.read_point_cloud(ply_path)
        if p is None or np.asarray(p.points).size == 0:
            return None
        return np.asarray(p.points, dtype=np.float32)
    # einfacher ASCII-Reader
    pts = []
    with open(ply_path, "r") as f:
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

def _estimate_scene_depth_bounds(K, R_ref, t_ref, sparse_pts, mask_ref,
                                 percentiles=(5, 95), default=(0.25, 2.0)):
    """
    Schätzt Near/Far aus sparse.ply (NUR innerhalb der Referenzmaske).
    Skala ist die SfM-Skala (nicht metrisch).
    """
    if sparse_pts is None or sparse_pts.size == 0:
        return default

    # Kamera-Zentrum der Referenz
    C = (-R_ref.T @ t_ref).reshape(3)
    X_cam = (R_ref @ (sparse_pts.T - C.reshape(3, 1))).T
    Z = X_cam[:, 2]

    # Punkte in die Ref projizieren und mit Maske schneiden
    u, v = _project(K, X_cam)
    h, w = mask_ref.shape[:2]
    uv_valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (Z > 0.01)
    if not np.any(uv_valid):
        return default

    u_f = u[uv_valid]
    v_f = v[uv_valid]
    u_i = np.floor(u_f).astype(np.int64)
    v_i = np.floor(v_f).astype(np.int64)
    u_i = np.clip(u_i, 0, w - 1)
    v_i = np.clip(v_i, 0, h - 1)
    m_ok = mask_ref[v_i, u_i] > 0
    Z = Z[uv_valid][m_ok]

    # robuste Mindestanzahl abhängig von gewähltem Perzentilbereich
    if Z.size == 0:
        return default
    p_low = min(percentiles[0], 100 - percentiles[1]) / 100.0  # z.B. 0.05
    k_tail = int(os.getenv("MVS_PCTL_KTAIL", "2"))             # mind. k Punkte je Rand
    min_pts = int(math.ceil(k_tail / max(p_low, 1e-6)))        # z.B. 2 / 0.05 = 40

    if Z.size < min_pts:
        # robuste Notlösung: Median ± 2.5*MAD
        if Z.size >= max(10, 2 * k_tail):
            med = float(np.median(Z))
            mad = float(np.median(np.abs(Z - med))) * 1.4826
            z0 = med - 2.5 * mad
            z1 = med + 2.5 * mad
            z0 = max(0.05, z0); z1 = max(z0 + 0.05, z1)
            return (z0, z1)
        return default

    z0 = float(np.percentile(Z, percentiles[0]))
    z1 = float(np.percentile(Z, percentiles[1]))
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        return default
    return (max(0.05, z0), max(z0 + 0.05, z1))

def _make_depth_values(near, far, n_planes, sampling="inverse"):
    if sampling == "inverse":
        inv_near, inv_far = 1.0 / near, 1.0 / far
        inv = np.linspace(inv_far, inv_near, n_planes, dtype=np.float32)
        z = 1.0 / inv
    else:
        z = np.linspace(near, far, n_planes, dtype=np.float32)
    return z

# --------------------------------------------------------------------------
# 3) Plane-Sweep CPU (Fallback)
# --------------------------------------------------------------------------
def _plane_sweep_depth_cpu(K, images, masks, R_list, t_list, ref_idx, src_ids, depth_values,
                           patch=3, on_log=None):
    ref_img = images[ref_idx]
    ref_msk = masks[ref_idx]
    H, W = ref_img.shape[:2]
    g_ref = _to_gray_u8(ref_img)

    ys, xs = np.where(ref_msk > 0)
    pad = int(os.getenv("MVS_MASK_PAD", "6"))
    y0 = max(0, int(ys.min()) - pad); y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad); x1 = min(W, int(xs.max()) + 1 + pad)
    Hroi = y1 - y0; Wroi = x1 - x0

    # Crop + K shift
    Kc = K.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0

    g_ref_roi = g_ref[y0:y1, x0:x1]
    ref_patch = cv.blur(g_ref_roi, (patch, patch))

    # Vorbereite Pixel in Kamerakoords
    yy, xx = np.meshgrid(np.arange(Hroi), np.arange(Wroi), indexing="ij")
    pix = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).astype(np.float32)
    Kinv = np.linalg.inv(Kc).astype(np.float32)
    dirs = (pix @ Kinv.T).reshape(Hroi, Wroi, 3)  # Strahlen

    Rr = R_list[ref_idx]; tr = t_list[ref_idx].reshape(3,1)
    Cr = (-Rr.T @ tr).reshape(3)

    best_cost = np.full((Hroi, Wroi), 1e9, np.float32)
    best_depth = np.zeros((Hroi, Wroi), np.float32)

    # Quelle vorbereiten
    src_imgs = [images[i] for i in src_ids]
    src_msks = [masks[i] for i in src_ids]
    R_src = [R_list[i] for i in src_ids]
    t_src = [t_list[i].reshape(3,1) for i in src_ids]
    C_src = [(-R.T @ t).reshape(3) for R, t in zip(R_src, t_src)]

    DCHUNK = int(os.getenv("MVS_DEPTH_CHUNK", "16"))
    D = depth_values.shape[0]
    n_chunks = math.ceil(D / DCHUNK)
    _log(f"[mvs] cpu ROI={Hroi}x{Wroi}, D={D} (chunks={n_chunks}) src={len(src_ids)}", on_log)

    for ci in range(n_chunks):
        d0 = ci * DCHUNK
        d1 = min(D, (ci + 1) * DCHUNK)
        dv = depth_values[d0:d1]  # (Dc,)

        acc_cost = np.zeros((d1 - d0, Hroi, Wroi), np.float32)
        acc_w = np.zeros_like(acc_cost)

        for s, (im_s, ms_s, Rs, Cs) in enumerate(zip(src_imgs, src_msks, R_src, C_src)):
            g_s = _to_gray_u8(im_s)
            m_s = (ms_s > 0).astype(np.uint8)

            # 3D Punkte in Welt (für alle Tiefen)
            X_ref = dirs[None, ...] * dv[:, None, None, None]  # Dc,H,W,3
            Xw = (Rr.T @ X_ref.reshape(-1,3).T + Cr.reshape(3,1)).T.reshape(d1-d0, Hroi, Wroi, 3)
            # in Quelle
            Xs = (Rs @ (Xw.reshape(-1,3).T - Cs.reshape(3,1))).T.reshape(d1-d0, Hroi, Wroi, 3)
            u = Kc[0,0]*Xs[...,0]/np.maximum(Xs[...,2],1e-6) + Kc[0,2]
            v = Kc[1,1]*Xs[...,1]/np.maximum(Xs[...,2],1e-6) + Kc[1,2]

            # bilineares Sampling per remap (Schleife über Dc)
            for k in range(d1-d0):
                mapx = u[k].astype(np.float32)
                mapy = v[k].astype(np.float32)
                Iw = cv.remap(g_s, mapx, mapy, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
                Mw = cv.remap(m_s, mapx, mapy, interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)

                Iw = cv.blur(Iw, (patch, patch))
                cost = np.abs(ref_patch.astype(np.float32) - Iw.astype(np.float32))
                valid = ((Mw > 0) & (ref_msk[y0:y1, x0:x1] > 0)).astype(np.float32)
                cost = np.where(valid > 0, cost, 1e6)
                acc_cost[k] += cost
                acc_w[k] += valid

        mean_cost = acc_cost / np.maximum(acc_w, 1e-3)
        imin = np.argmin(mean_cost, axis=0)  # Hroi,Wroi
        zbest = dv[imin]
        cmin = mean_cost[imin, np.arange(Hroi)[:,None], np.arange(Wroi)]

        upd = cmin < best_cost
        best_cost[upd] = cmin[upd]
        best_depth[upd] = zbest[upd]

    # in volles Bild
    depth_full = np.zeros((H, W), np.float32)
    valid_full = np.zeros((H, W), np.uint8)
    depth_full[y0:y1, x0:x1] = best_depth
    valid_full[y0:y1, x0:x1] = (ref_msk[y0:y1, x0:x1] > 0).astype(np.uint8)
    return depth_full, valid_full, (y0, y1, x0, x1)

# --------------------------------------------------------------------------
# 4) Plane-Sweep GPU (PyTorch/CUDA)
# --------------------------------------------------------------------------
def _plane_sweep_depth_torch(K, images, masks, R_list, t_list, ref_idx, src_ids, depth_values, on_log=None):
    import torch
    import torch.nn.functional as F

    dev_flag = os.getenv("MVS_DEVICE", "cpu").lower()
    use_cuda = (dev_flag == "cuda") and torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("CUDA nicht aktiv (MVS_DEVICE!=cuda oder keine GPU).")

    DCHUNK  = int(os.getenv("MVS_DEPTH_CHUNK", "16"))
    SBATCH  = int(os.getenv("MVS_SRC_BATCH",   "2"))
    PATCH   = int(os.getenv("MVS_PATCH",       "3"))
    DTYPE   = os.getenv("MVS_TORCH_DTYPE", "float16")
    use_amp = (DTYPE == "float16")
    torch_dtype = torch.float16 if DTYPE == "float16" else torch.float32
    device = torch.device("cuda")

    ref_img = images[ref_idx]
    ref_msk = masks[ref_idx]
    H, W = ref_img.shape[:2]

    ys, xs = np.where(ref_msk > 0)
    pad = int(os.getenv("MVS_MASK_PAD", "6"))
    y0 = max(0, int(ys.min()) - pad); y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad); x1 = min(W, int(xs.max()) + 1 + pad)
    Hroi = y1 - y0; Wroi = x1 - x0

    with torch.no_grad():
        ref = ref_img[y0:y1, x0:x1]
        ref_t = torch.from_numpy(ref).to(device=device, dtype=torch_dtype)
        if ref_t.ndim == 3 and ref_t.shape[2] == 3:
            ref_t = ref_t.permute(2,0,1)
            ref_t = 0.2989*ref_t[0:1] + 0.5870*ref_t[1:2] + 0.1140*ref_t[2:3]
        else:
            ref_t = ref_t[None, ...]
        ref_t = ref_t[None, ...] / 255.0  # 1,1,Hroi,Wroi

        ref_mask_t = torch.from_numpy((ref_msk[y0:y1, x0:x1] > 0).astype(np.uint8)).to(device)
        ref_mask_t = ref_mask_t[None, None, ...]

        # Intrinsics an ROI anpassen
        Kc = K.copy()
        Kc[0, 2] -= x0; Kc[1, 2] -= y0
        Kinv = np.linalg.inv(Kc).astype(np.float32)
        K_t = torch.from_numpy(Kc).to(device=device, dtype=torch.float32)
        Kinv_t = torch.from_numpy(Kinv).to(device=device, dtype=torch.float32)

        Rr = torch.from_numpy(R_list[ref_idx].astype(np.float32)).to(device)
        tr = torch.from_numpy(t_list[ref_idx].astype(np.float32)).to(device).view(3, 1)
        Cr = (-Rr.transpose(0,1) @ tr).view(3)

        R_src = [torch.from_numpy(R_list[i].astype(np.float32)).to(device) for i in src_ids]
        t_src = [torch.from_numpy(t_list[i].astype(np.float32)).to(device).view(3,1) for i in src_ids]
        C_src = [(-R.transpose(0,1) @ t).view(3) for R, t in zip(R_src, t_src)]

        ys_t, xs_t = torch.meshgrid(
            torch.arange(Hroi, device=device, dtype=torch.float32),
            torch.arange(Wroi, device=device, dtype=torch.float32),
            indexing="ij"
        )
        ones = torch.ones_like(xs_t)
        pix = torch.stack([xs_t, ys_t, ones], dim=-1)  # Hroi,Wroi,3
        dirs = (pix @ Kinv_t.T)  # Hroi,Wroi,3

        ksz = PATCH
        box = torch.ones((1,1,ksz,ksz), device=device, dtype=torch.float32) / (ksz*ksz)

        best_cost  = torch.full((Hroi, Wroi), 1e9, device=device, dtype=torch.float32)
        best_depth = torch.zeros((Hroi, Wroi), device=device, dtype=torch.float32)

        def project_norm(X_cam):
            X = X_cam[..., 0];
            Y = X_cam[..., 1];
            Z = X_cam[..., 2].clamp_min(1e-6)
            u = (K_t[0, 0] * X / Z + K_t[0, 2])  # Pixelcoords in der ROI
            v = (K_t[1, 1] * Y / Z + K_t[1, 2])
            # align_corners=True  ->  [-1,1] entspricht Pixelzentren [0, W-1] / [0, H-1]
            gn_u = (u / (Wroi - 1.0)) * 2.0 - 1.0
            gn_v = (v / (Hroi - 1.0)) * 2.0 - 1.0
            grid = torch.stack([gn_u, gn_v], dim=-1)
            # kleine Sicherheitsmarge gegen Randüberschreitung:
            eps = 1e-4
            return torch.clamp(grid, -1.0 + eps, 1.0 - eps)

        def to_gray_gpu(im_np):
            t = torch.from_numpy(im_np[y0:y1, x0:x1]).to(device=device, dtype=torch_dtype)
            if t.ndim == 3 and t.shape[2] == 3:
                t = t.permute(2,0,1)
                t = 0.2989*t[0:1] + 0.5870*t[1:2] + 0.1140*t[2:3]
            else:
                t = t[None, ...]
            return (t[None, ...] / 255.0)

        def to_mask_gpu(m_np):
            if m_np is None:
                return torch.ones((1,1,Hroi,Wroi), device=device, dtype=torch.float32)
            m = (m_np[y0:y1, x0:x1] > 0).astype(np.uint8)
            return torch.from_numpy(m)[None,None,...].to(device=device, dtype=torch.float32)

        src_imgs  = [images[i] for i in src_ids]
        src_masks = [masks[i]  for i in src_ids]

        D = depth_values.shape[0]
        n_chunks = math.ceil(D / DCHUNK)
        _log(f"[mvs-gpu] ROI={Hroi}x{Wroi}, D={D} (chunks={n_chunks}, chunk={DCHUNK}), src={len(src_ids)} (batch={SBATCH})", on_log)

        import torch as _torch
        _torch.set_grad_enabled(False)
        for ci in range(n_chunks):
            d0 = ci * DCHUNK
            d1 = min(D, (ci + 1) * DCHUNK)
            dv = _torch.from_numpy(depth_values[d0:d1].astype(np.float32)).to(device)  # (Dc,)
            Dc = dv.shape[0]

            X_ref = dirs[None, ...] * dv.view(Dc, 1, 1, 1)
            Xw = _torch.einsum('ij,dhwj->dhwi', Rr.transpose(0,1), X_ref) + Cr.view(1,1,1,3)

            acc_cost = _torch.zeros((Dc, Hroi, Wroi), device=device, dtype=_torch.float32)
            acc_w    = _torch.zeros_like(acc_cost)

            for sb in range(0, len(src_ids), SBATCH):
                s_ids = src_ids[sb: sb+SBATCH]
                I_src = _torch.cat([to_gray_gpu(src_imgs[k])  for k in range(len(s_ids))], dim=0).to(dtype=torch_dtype)
                M_src = _torch.cat([to_mask_gpu(src_masks[k+sb]) for k in range(len(s_ids))], dim=0).to(dtype=_torch.float32)

                for j, sid in enumerate(s_ids):
                    Rs = R_src[sb + j]; Cs = C_src[sb + j]
                    Xs = _torch.einsum('ij,dhwj->dhwi', Rs, (Xw - Cs.view(1,1,1,3)))
                    grid = project_norm(Xs).view(Dc, Hroi, Wroi, 2)

                    with _torch.cuda.amp.autocast(enabled=use_amp):
                        Is_warp = _torch.nn.functional.grid_sample(
                            I_src[j:j+1].expand(Dc,-1,-1,-1), grid,
                            mode='bilinear', padding_mode='zeros', align_corners=True)
                        Ms_warp = _torch.nn.functional.grid_sample(
                            M_src[j:j+1].expand(Dc,-1,-1,-1), grid,
                            mode='nearest', padding_mode='zeros', align_corners=True)

                        ref_blur = _torch.nn.functional.conv2d(ref_t.to(_torch.float32), box, padding=ksz//2)
                        src_blur = _torch.nn.functional.conv2d(Is_warp.to(_torch.float32), box, padding=ksz//2)
                        cost = _torch.abs(ref_blur - src_blur).squeeze(1)  # Dc,H,W
                        valid = (Ms_warp.squeeze(1) > 0.5) & (ref_mask_t.squeeze(1) > 0)
                        cost = _torch.where(valid, cost, _torch.full_like(cost, 1e6))

                    acc_cost += cost
                    acc_w    += valid.to(_torch.float32)

                del I_src, M_src, cost, valid, Is_warp, Ms_warp, ref_blur, src_blur, grid, Xs
                _torch.cuda.empty_cache()

            mean_cost = acc_cost / acc_w.clamp_min(1e-3)
            cmin, imin = _torch.min(mean_cost, dim=0)
            zbest_chunk = dv[imin]

            upd = cmin < best_cost
            best_cost  = _torch.where(upd, cmin, best_cost)
            best_depth = _torch.where(upd, zbest_chunk, best_depth)

            del X_ref, Xw, acc_cost, acc_w, mean_cost, cmin, imin, zbest_chunk, dv
            _torch.cuda.empty_cache()

        depth_full = _torch.zeros((H, W), device=device, dtype=_torch.float32)
        valid_full = _torch.zeros((H, W), device=device, dtype=_torch.uint8)
        depth_full[y0:y1, x0:x1] = best_depth
        valid_full[y0:y1, x0:x1] = (ref_mask_t.squeeze(0).squeeze(0) > 0).to(_torch.uint8)

        depth_np = depth_full.detach().cpu().numpy()
        valid_np = valid_full.detach().cpu().numpy()

        del depth_full, valid_full, best_cost, best_depth, ref_t, ref_mask_t, dirs, pix, xs_t, ys_t
        import torch as __t; __t.cuda.empty_cache()

    return depth_np, valid_np, (y0, y1, x0, x1)

# --------------------------------------------------------------------------
# 5) Back-Projection & Export
# --------------------------------------------------------------------------
def _depth_to_points(depth, valid, K, R, t, color_img, mask_ref=None):
    H, W = depth.shape
    ys, xs = np.where(valid > 0)
    if ys.size == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    z = depth[ys, xs].astype(np.float32)
    if mask_ref is not None:
        keep = (mask_ref[ys, xs] > 0) & (z > 1e-6)
        ys = ys[keep]; xs = xs[keep]; z = z[keep]
        if ys.size == 0:
            return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    x = (xs - K[0,2]) * z / K[0,0]
    y = (ys - K[1,2]) * z / K[1,1]
    X_cam = np.stack([x,y,z], axis=1)  # (N,3)
    Xw = (R.T @ X_cam.T + (-R.T @ t).reshape(3,1)).T

    cols = color_img[ys, xs]
    return Xw.astype(np.float32), cols.astype(np.uint8)

def _save_depth_preview(path_png, depth, valid):
    d = depth.copy()
    d[valid == 0] = 0
    if np.count_nonzero(valid) > 0:
        v = d[valid > 0]
        vmin, vmax = np.percentile(v, 2), np.percentile(v, 98)
        vmax = max(vmax, vmin + 1e-6)
        d = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    d = (d * 255).astype(np.uint8)
    cv.imwrite(path_png, d)

# --------------------------------------------------------------------------
# 6) MVS Orchestrierung
# --------------------------------------------------------------------------
def run_plane_sweep_mvs(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                        on_log=None, on_progress=None):
    """
    Orchestriert pro-Frame Plane-Sweep:
      - lädt Bilder & Masken,
      - bestimmt Depth-Bounds aus sparse.ply∩Maske,
      - CPU/GPU Plane-Sweep,
      - speichert pro-Ref depth & points, kumuliert dense_points,
      - optional Mesh per Poisson.
    """
    _mkdir(mesh_dir)
    depth_dir = os.path.join(mesh_dir, "depth")
    _mkdir(depth_dir)

    # Intrinsics kommen aus Pipeline (K wird via pipeline_runner übergeben und hier NICHT geladen);
    # daher erwarten wir, dass pipeline_runner diese Funktion mit einem global gesetzten K aufruft.
    K = globals().get("GLOBAL_INTRINSICS_K", None)
    if K is None:
        raise RuntimeError("GLOBAL_INTRINSICS_K ist nicht gesetzt. Setze vor dem Aufruf GLOBAL_INTRINSICS_K = K.")

    # Settings (ENV)
    REF_STEP   = int(os.getenv("MVS_REF_STEP",   "1"))
    MAX_VIEWS  = int(os.getenv("MVS_MAX_VIEWS",  "12"))
    N_PLANES   = int(os.getenv("MVS_N_PLANES",   "128"))
    SAMPLING   = os.getenv("MVS_DEPTH_SAMPLING", "inverse").lower()
    EXPAND     = float(os.getenv("MVS_DEPTH_EXPAND", "0.08"))  # erweitert near/far
    DEVICE     = os.getenv("MVS_DEVICE", "cpu").lower()
    EXPORT_MESH= os.getenv("MVS_EXPORT_MESH", "true").lower() == "true"

    # Daten einlesen
    R_all, t_all, frame_idx = _load_poses_npz(poses_npz)
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"Keine Frames in {frames_dir} gefunden.")
    # Bilder & Masken
    images = [_load_color(f) for f in frame_files]
    if masks_dir and os.path.isdir(masks_dir):
        masks = [_mask_for_frame(masks_dir, f) for f in frame_files]
    else:
        masks = [np.ones(images[0].shape[:2], np.uint8)*255 for _ in frame_files]
    # evtl. Scale
    scale = float(os.getenv("MVS_SCALE", "1.0"))
    if abs(scale - 1.0) > 1e-6:
        ims = []
        msks = []
        for im, m in zip(images, masks):
            ims.append(cv.resize(im, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_AREA))
            msks.append(cv.resize(m,  dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST))
        images, masks = ims, msks
        K = K.copy()
        K[0,0] *= scale; K[1,1] *= scale
        K[0,2] *= scale; K[1,2] *= scale
        globals()["GLOBAL_INTRINSICS_K"] = K  # update global

    sparse_path = os.path.join(mesh_dir, "sparse.ply")
    sparse_pts = _read_sparse_points(sparse_path)

    use_gpu = (DEVICE == "cuda")
    all_pts = []
    all_cols = []

    _log("MVS (plane-sweep) reconstruction", on_log)
    N = len(images)
    total_refs = (N + REF_STEP - 1) // REF_STEP
    ref_counter = 0

    for ridx in range(0, N, REF_STEP):
        ref_counter += 1
        _log(f"[mvs] frame {ridx}/{N-1}", on_log)

        ref_img = images[ridx]
        ref_msk = masks[ridx]
        # Depth-Bounds
        near, far = _estimate_scene_depth_bounds(
            K, R_all[ridx], t_all[ridx], sparse_pts, ref_msk,
            percentiles=(5, 95),
            default=tuple(map(float, os.getenv("MVS_DEPTH_DEFAULT","0.25,2.0").split(",")))
        )
        dz = (far - near) * float(os.getenv("MVS_DEPTH_EXPAND", "0.08"))
        near = max(0.05, near - dz)
        far  = far + dz

        _log(f"[mvs] ref={ridx} depth-range: {near:.3f}–{far:.3f} units ( {N_PLANES} planes )", on_log)
        depth_values = _make_depth_values(near, far, N_PLANES, sampling=SAMPLING)

        # Quellen wählen (Nachbarn um ref_idx)
        src_ids = []
        r = 1
        while len(src_ids) < MAX_VIEWS and (ridx - r >= 0 or ridx + r < N):
            if ridx - r >= 0: src_ids.append(ridx - r)
            if len(src_ids) >= MAX_VIEWS: break
            if ridx + r < N: src_ids.append(ridx + r)
            r += 1
        src_ids = [s for s in src_ids if s != ridx]

        # Depth berechnen
        try:
            if use_gpu:
                depth, valid, roi = _plane_sweep_depth_torch(K, images, masks, R_all, t_all,
                                                             ridx, src_ids, depth_values, on_log)
            else:
                depth, valid, roi = _plane_sweep_depth_cpu(K, images, masks, R_all, t_all,
                                                           ridx, src_ids, depth_values,
                                                           patch=int(os.getenv("MVS_PATCH","3")),
                                                           on_log=on_log)
        except Exception as e:
            _log(f"[error] MVS on ref={ridx} failed: {e}", on_log)
            continue

        # speichern depth
        depth_dir = os.path.join(mesh_dir, "depth")
        _mkdir(depth_dir)
        depth_np_path = os.path.join(depth_dir, f"depth_{ridx:04d}.npy")
        np.save(depth_np_path, depth.astype(np.float32))
        depth_png_path = os.path.join(depth_dir, f"depth_{ridx:04d}.png")
        _save_depth_preview(depth_png_path, depth, valid)
        _log(f"[mvs] saved depth -> {depth_np_path}", on_log)

        # Punkte exportieren (nur maskierter valider Bereich)
        pts, cols = _depth_to_points(depth, valid, K, R_all[ridx], t_all[ridx], ref_img, mask_ref=ref_msk)
        if pts.size > 0 and HAS_O3D:
            ref_ply = os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply")
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            if cols is not None and cols.size == pts.shape[0]*3:
                pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1] / 255.0)  # BGR->RGB
            o3d.io.write_point_cloud(ref_ply, pcd)
            _log(f"[mvs] saved points -> {ref_ply}", on_log)

        if pts.size > 0:
            all_pts.append(pts)
            all_cols.append(cols)

        # Fortschritt
        if on_progress:
            pct = int(100.0 * ref_counter / max(1,total_refs))
            on_progress(pct, f"mvs {ref_counter}/{total_refs}")

    # alles sammeln
    if all_pts:
        P = np.concatenate(all_pts, axis=0)
        C = np.concatenate(all_cols, axis=0) if all_cols else None
        dense_pts_path = os.path.join(mesh_dir, "dense_points.ply")
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            if C is not None and C.size == P.shape[0]*3:
                pcd.colors = o3d.utility.Vector3dVector(C[:, ::-1] / 255.0)
            o3d.io.write_point_cloud(dense_pts_path, pcd)
        else:
            save_point_cloud(P, dense_pts_path)
        _log(f"[mvs] saved dense points -> {dense_pts_path}", on_log)

        # optional Mesh
        if EXPORT_MESH and HAS_O3D and P.shape[0] >= 1000:
            try:
                pcd = o3d.io.read_point_cloud(dense_pts_path)
                pcd.estimate_normals()
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=int(os.getenv("MVS_POISSON_DEPTH", "10"))
                )
                # pruning via density
                densities = np.asarray(densities)
                keep = densities > np.quantile(densities, 0.02)
                mesh = mesh.select_by_index(np.where(keep)[0])
                mesh_path = os.path.join(mesh_dir, "dense_mesh.ply")
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                _log(f"[mvs] saved mesh -> {mesh_path}", on_log)
            except Exception as e:
                _log(f"[warn] meshing failed: {e}", on_log)

    _log("[ui] Done MVS.", on_log)

# --------------------------------------------------------------------------
# 7) Kompatible Wrapper für pipeline_runner.py
# --------------------------------------------------------------------------
def _set_mvs_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                           min_valid_frac, poisson_depth, mode: str):
    # Nur setzen, wenn nicht schon extern gesetzt:
    os.environ.setdefault("MVS_SCALE",         str(scale))
    os.environ.setdefault("MVS_MAX_VIEWS",     str(max_views))
    os.environ.setdefault("MVS_N_PLANES",      str(n_planes))
    os.environ.setdefault("MVS_DEPTH_EXPAND",  str(depth_expand))
    os.environ.setdefault("MVS_PATCH",         str(patch))
    os.environ.setdefault("MVS_COST_THR",      str(cost_thr))
    os.environ.setdefault("MVS_MIN_VALID_FRAC",str(min_valid_frac))
    os.environ.setdefault("MVS_POISSON_DEPTH", str(poisson_depth))
    os.environ.setdefault("MVS_EXPORT_MESH",   "true")
    # sampling/step:
    if mode == "all":
        os.environ.setdefault("MVS_REF_STEP", "1")       # jeden Frame
    else:
        # "single" ≈ stark ausgedünnt: z.B. jeder 12. Frame
        os.environ.setdefault("MVS_REF_STEP", os.getenv("MVS_REF_STEP_SINGLE", "12"))

def reconstruct_mvs_depth_and_mesh(paths, K,
                                   scale=0.55, max_views=26, n_planes=144,
                                   depth_expand=0.08, patch=7,
                                   cost_thr=0.55, min_valid_frac=0.01,
                                   poisson_depth=10,
                                   on_log=None, on_progress=None):
    """Wrapper: 'single' Modus (ausgedünnte Refs)."""
    _set_mvs_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                           min_valid_frac, poisson_depth, mode="single")
    # globale Intrinsics setzen (wie von pipeline_runner vorgesehen)
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()

    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir     = os.path.join(root, "mesh")
    frames_dir   = os.path.join(root, "raw_frames")
    features_dir = os.path.join(root, "features")
    poses_npz    = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir    = os.path.join(features_dir, "masks")

    run_plane_sweep_mvs(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                        on_log=on_log, on_progress=on_progress)

def reconstruct_mvs_depth_and_mesh_all(paths, K,
                                       scale=0.55, max_views=26, n_planes=144,
                                       depth_expand=0.08, patch=7,
                                       cost_thr=0.55, min_valid_frac=0.01,
                                       poisson_depth=10,
                                       on_log=None, on_progress=None):
    """Wrapper: 'all' Modus (jeden Frame)."""
    _set_mvs_env_from_args(scale, max_views, n_planes, depth_expand, patch, cost_thr,
                           min_valid_frac, poisson_depth, mode="all")
    globals()["GLOBAL_INTRINSICS_K"] = K.copy()

    root = paths.root if hasattr(paths, "root") else paths["root"]
    mesh_dir     = os.path.join(root, "mesh")
    frames_dir   = os.path.join(root, "raw_frames")
    features_dir = os.path.join(root, "features")
    poses_npz    = os.path.join(root, "poses", "camera_poses.npz")
    masks_dir    = os.path.join(features_dir, "masks")

    run_plane_sweep_mvs(mesh_dir, frames_dir, features_dir, poses_npz, masks_dir,
                        on_log=on_log, on_progress=on_progress)
