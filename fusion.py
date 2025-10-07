# fusion.py — Silhouette-Fill + Source-Depth-Gate + Multi-View-Konsens + Front-Weighted Merge
# - wählt gleichmäßig verteilte Keyframes (bucketed best-by-sparse)
# - lädt points_ref_XXXX.ply (+ Farben) nur für diese Keyframes
# - FILL: fehlende Pixel innerhalb der Maske via depth.npy (+ optional Inpaint) befüllen und back-projizieren
# - 1) Source-Depth-Gate (entfernt Rays aus Ursprungsansicht)
# - 2) Multi-View-Silhouette (+optional Depth-Konsistenz) über alle gewählten Refs
# - 3) Softes Sparse-Pruning via KDTree-Distanz (nur als Prior)
# - 4) Voxel-Merge mit Frontalsicht-Priorität

import os, glob, math
import numpy as np
import cv2 as cv

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# ---------------- I/O helpers ----------------

def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _sorted_frames(frames_dir):
    # Deine Frames sind 1:1 indexiert (frame_0000_src_*.png ... frame_0119_*.png)
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
            if m is not None:
                return m
    return None

def _load_poses_npz(poses_npz_path):
    npz = np.load(poses_npz_path)
    R_all = npz["R"].astype(np.float64)                  # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1,3,1).astype(np.float64)  # (N,3,1)
    idx = npz.get("frame_idx", np.arange(len(R_all)))
    return R_all, t_all, idx

def _read_ascii_ply_points_colors(path):
    if not os.path.isfile(path): return None, None
    with open(path, "r") as f:
        header = True; n=0; has_color=False
        while header:
            line = f.readline()
            if not line: break
            if line.startswith("element vertex"): n = int(line.split()[-1])
            if line.startswith("property uchar red"): has_color=True
            if line.strip() == "end_header":
                header = False
                break
        pts=[]; cols=[]
        for _ in range(n):
            parts = f.readline().strip().split()
            if len(parts) < 3: continue
            x,y,z = map(float, parts[:3]); pts.append((x,y,z))
            if has_color and len(parts) >= 6:
                r,g,b = map(float, parts[3:6])  # ASCII i.d.R. RGB
                cols.append((b,g,r))           # -> BGR
    P = np.asarray(pts, np.float64)
    C = np.asarray(cols, np.float64) if cols else None
    return P, C

def _read_points_ref_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        if pc is None or np.asarray(pc.points).size == 0:
            return None, None
        P = np.asarray(pc.points, dtype=np.float64)
        C = None
        if len(pc.colors) > 0:
            C = (np.asarray(pc.colors)[:, ::-1] * 255.0).astype(np.float64)  # RGB->BGR
        return P, C
    return _read_ascii_ply_points_colors(path)

def _save_colored_ply(P, C, out_path):
    _mkdir(os.path.dirname(out_path) or ".")
    if HAS_O3D:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        if C is not None and C.shape[0] == P.shape[0]:
            pc.colors = o3d.utility.Vector3dVector((C[:, ::-1] / 255.0).astype(np.float64))  # BGR->RGB
        o3d.io.write_point_cloud(out_path, pc)
        return out_path
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

# ---------------- Kamera/Mathe ----------------

def _project(K, X_cam):
    x = X_cam[...,0]; y = X_cam[...,1]; z = np.maximum(X_cam[...,2], 1e-9)
    u = K[0,0]*x/z + K[0,2];  v = K[1,1]*y/z + K[1,2]
    return u, v

def _cam_center(R,t): return (-R.T @ t).reshape(3)
def _forward_dir(R):  return (R.T @ np.array([0.0,0.0,1.0])).reshape(3)

# ---------------- Auswahl (bucketed best-by-sparse) ----------------

def _score_frames_by_sparse(K, R_all, t_all, sparse_pts, masks):
    N = len(masks)
    if sparse_pts is None or sparse_pts.size == 0:
        return np.zeros(N, dtype=np.int32)
    scores = np.zeros(N, np.int32)
    for i in range(N):
        m = masks[i]
        if m is None: continue
        H,W = m.shape[:2]
        Cc = _cam_center(R_all[i], t_all[i])
        X_cam = (R_all[i] @ (sparse_pts.T - Cc.reshape(3,1))).T
        Z = X_cam[:,2]
        u,v = _project(K, X_cam)
        in_img = np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<W)&(v>=0)&(v<H)&(Z>1e-6)
        if not np.any(in_img): continue
        ui = np.clip(np.floor(u[in_img]).astype(np.int32),0,W-1)
        vi = np.clip(np.floor(v[in_img]).astype(np.int32),0,H-1)
        scores[i] = int(np.count_nonzero(m[vi,ui] > 0))
    return scores

def _choose_bucketed_best(candidates, scores, total_N, num_refs):
    chosen = []
    if not candidates: return chosen
    candidates = sorted(set(int(c) for c in candidates))
    num_refs  = max(1, int(num_refs))
    bucket_w  = total_N / float(num_refs)
    for b in range(num_refs):
        a = int(round(b*bucket_w)); z = int(round((b+1)*bucket_w))
        bucket = [c for c in candidates if a <= c < z]
        if bucket:
            best = max(bucket, key=lambda i: int(scores[i]))
            chosen.append(best)
        else:
            center = int(round((a+z)/2.0))
            best = min(candidates, key=lambda i: abs(i-center))
            if best not in chosen: chosen.append(best)
    return sorted(set(chosen))

# ---------------- Depth/Masks/Frames laden ----------------

def _load_depth(depth_dir, idx):
    p = os.path.join(depth_dir, f"depth_{idx:04d}.npy")
    if os.path.isfile(p):
        try: return np.load(p)
        except Exception: return None
    return None

def _load_frame_image(frame_files, idx):
    # Deine file-liste ist 0..N-1 sortiert, der idx passt 1:1
    if 0 <= idx < len(frame_files):
        im = cv.imread(frame_files[idx], cv.IMREAD_COLOR)
        return im
    return None

# ---------------- Filter 1: Source-Depth-Gate ----------------

def _source_depth_gate(P, src_ids, K, R_all, t_all, masks, depth_maps, tol_rel=0.03, chunk=300000):
    N = P.shape[0]
    ok_all = np.zeros(N, dtype=np.bool_)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, sid in enumerate(src_ids): groups[int(sid)].append(i)
    for sid, idxs in groups.items():
        m = masks.get(sid, None); D = depth_maps.get(sid, None)
        R = R_all[sid]; t = t_all[sid]
        if m is None or D is None:
            ok_all[idxs] = True
            continue
        H,W = m.shape[:2]; Cc = _cam_center(R,t)
        idxs = np.asarray(idxs, np.int64)
        for s in range(0, idxs.size, chunk):
            ids = idxs[s:s+chunk]; Q = P[ids]
            Xc = (R @ (Q.T - Cc.reshape(3,1))).T
            z = Xc[:,2]; u,v = _project(K, Xc)
            finite = np.isfinite(u)&np.isfinite(v)
            inside = np.zeros_like(finite, dtype=bool)
            if np.any(finite):
                ui = np.clip(np.floor(u[finite]).astype(np.int32),0,W-1)
                vi = np.clip(np.floor(v[finite]).astype(np.int32),0,H-1)
                d  = D[vi,ui]
                ii = np.where(finite)[0]
                inside[ii] = (z[ii] > 1e-6) & (m[vi,ui] > 0) & (z[ii] <= (d + tol_rel*(1.0 + d)))
            ok_all[ids] = inside
    return ok_all

# ---------------- Filter 2: Multi-View Silhouette (+optional Depth) ----------------

def _multi_view_consensus(P, K, R_list, t_list, masks_list, depth_list,
                          mode="majority", tol_rel=0.03, chunk=400000):
    V = len(R_list); N = P.shape[0]
    counts = np.zeros(N, np.int32)
    for v in range(V):
        R = R_list[v]; t = t_list[v]; m = masks_list[v]; D = depth_list[v]
        if m is None: continue
        H,W = m.shape[:2]; Cc = _cam_center(R,t)
        for s in range(0, N, chunk):
            e = min(N, s+chunk); Q = P[s:e]
            Xc = (R @ (Q.T - Cc.reshape(3,1))).T
            z = Xc[:,2]; u,v = _project(K, Xc)
            finite = np.isfinite(u)&np.isfinite(v)
            if not np.any(finite): continue
            ui = np.clip(np.floor(u[finite]).astype(np.int32),0,W-1)
            vi = np.clip(np.floor(v[finite]).astype(np.int32),0,H-1)
            ii = np.where(finite)[0]
            inside = np.zeros(e-s, dtype=bool)
            inside[ii] = (z[ii] > 1e-6) & (m[vi,ui] > 0)
            if D is not None:
                d = np.zeros(e-s, dtype=np.float32); d[ii] = D[vi,ui]
                inside &= (z <= (d + tol_rel*(1.0 + d)))
            counts[s:e] += inside.astype(np.int32)
    if mode == "all":       thr = V
    elif mode == "any":     thr = 1
    else:                   thr = int(math.ceil(V/2.0))  # majority
    return counts, (counts >= thr)

# ---------------- Soft Sparse Prior (KDTree Distanz) ----------------

def _dist_to_sparse(P, sparse_pts):
    if sparse_pts is None or sparse_pts.shape[0] == 0:
        return np.full(P.shape[0], np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(sparse_pts)
        d, _ = tree.query(P, k=1)
        return d.astype(np.float64)
    except Exception:
        bb = sparse_pts.max(0)-sparse_pts.min(0)
        diag = float(np.linalg.norm(bb))+1e-9
        return np.full(P.shape[0], diag, dtype=np.float64)

# ---------------- Silhouette-Fill (pro View) ----------------

def _coverage_map_from_points(P_ref, R, t, K, shape_hw):
    H, W = shape_hw
    cov = np.zeros((H, W), np.uint8)
    if P_ref is None or P_ref.size == 0:
        return cov
    Cc = _cam_center(R, t)
    Xc = (R @ (P_ref.T - Cc.reshape(3,1))).T
    z = Xc[:,2]; u,v = _project(K, Xc)
    ok = np.isfinite(u)&np.isfinite(v)&(z>1e-6)
    if not np.any(ok): return cov
    ui = np.clip(np.floor(u[ok]).astype(np.int32),0,W-1)
    vi = np.clip(np.floor(v[ok]).astype(np.int32),0,H-1)
    cov[vi, ui] = 255
    return cov

def _robust_inpaint_depth(D, mask_fg, radius_px=3):
    # mask_fg: 0/255 (Silhouette)
    # Inpaint nur WO D invalid ist & innerhalb Silhouette
    valid = np.isfinite(D) & (D > 0)
    if not np.any(valid):
        return None
    dvals = D[valid]
    lo = np.quantile(dvals, 0.01); hi = np.quantile(dvals, 0.99)
    if hi <= lo: hi = lo + 1.0
    D8 = np.clip((D - lo) / (hi - lo), 0, 1)
    D8 = (D8 * 255.0).astype(np.uint8)
    hole = ((~valid) & (mask_fg > 0)).astype(np.uint8)
    if np.count_nonzero(hole) == 0:
        return D.copy()
    D8_inp = cv.inpaint(D8, hole, radius_px, cv.INPAINT_TELEA)
    D_f = D8_inp.astype(np.float32) / 255.0 * (hi - lo) + lo
    # wo D ursprünglich gültig war, belassen
    D_f[valid] = D[valid]
    return D_f

def _backproject_pixels(u, v, z, K, R, t):
    # world->cam: x_cam = R (X - Cc),  Cc = -R^T t
    # invert: X = Cc + R^T * (z * K^{-1} [u v 1]^T)
    Cc = _cam_center(R, t)
    invK = np.linalg.inv(K)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=0)    # (3,N)
    rays = invK @ uv1                                   # (3,N)
    Xc = (rays * z)                                     # (3,N)
    Xw = (R.T @ Xc).T + Cc.reshape(1,3)                 # (N,3)
    return Xw

def _silhouette_fill_for_ref(ridx, K, R, t, mask_ref, depth_ref, frame_img,
                             P_ref_existing, stride=2, inpaint=True, inpaint_radius=3,
                             max_pts=None):
    H, W = mask_ref.shape[:2]
    # Coverage aus bereits vorhandenen Punkten dieses Views
    cov = _coverage_map_from_points(P_ref_existing, R, t, K, (H, W))
    # Löcher = Silhouette ohne Coverage
    holes = (mask_ref > 0) & (cov == 0)

    if not np.any(holes):
        return None, None

    # Tiefe: bevorzugt depth_ref, optional inpaint
    D = None
    if depth_ref is not None:
        D = depth_ref.astype(np.float32)
        if inpaint:
            D = _robust_inpaint_depth(D, mask_ref, radius_px=inpaint_radius)
    if D is None:
        # ohne Depth können wir keine z vergeben -> skip
        return None, None

    # Sample Rasterpunkte im Lochgebiet
    ys, xs = np.where(holes)
    if ys.size == 0:
        return None, None

    if stride > 1:
        ys = ys[::stride]; xs = xs[::stride]
    if max_pts is not None and ys.size > max_pts:
        idx = np.linspace(0, ys.size-1, max_pts).astype(np.int64)
        ys = ys[idx]; xs = xs[idx]

    z = D[ys, xs]
    valid_z = np.isfinite(z) & (z > 0)
    if not np.any(valid_z):
        return None, None
    ys = ys[valid_z]; xs = xs[valid_z]; z = z[valid_z]

    # Back-Projection
    Xw = _backproject_pixels(xs.astype(np.float64), ys.astype(np.float64), z.astype(np.float64), K, R, t)

    # Farben aus dem Frame
    C = None
    if frame_img is not None and frame_img.shape[0] == H and frame_img.shape[1] == W:
        C = frame_img[ys, xs, :].astype(np.float64)  # BGR

    return Xw, C

# ---------------- Merge (Frontalsicht gewinnt) ----------------

def _voxel_merge_weighted(P, C, src_ids, R_all, t_all, voxel=None):
    if P.size == 0: return P, C
    if voxel is None or voxel <= 0:
        try:
            from scipy.spatial import cKDTree
            idx = np.random.choice(P.shape[0], size=min(6000,P.shape[0]), replace=False)
            Q = P[idx]; tree = cKDTree(Q); d,_ = tree.query(Q, k=2)
            med = np.median(d[:,1]) if d.ndim>1 else np.median(d)
        except Exception:
            bb = P.max(0)-P.min(0); med = float(np.linalg.norm(bb))/400.0
        voxel = max(1e-4, 0.6*float(med))

    src_set = set(int(s) for s in src_ids.tolist())
    Cc = {sid: _cam_center(R_all[sid], t_all[sid]) for sid in src_set}
    Fd = {sid: _forward_dir(R_all[sid])            for sid in src_set}

    W = np.zeros(P.shape[0], np.float64)
    for i in range(P.shape[0]):
        sid = int(src_ids[i])
        v = P[i]-Cc[sid]; nv = np.linalg.norm(v)+1e-12; v = v/nv
        f = Fd[sid]; nf = np.linalg.norm(f)+1e-12
        W[i] = float(np.clip(np.dot(v,f)/nf, -1.0, 1.0))

    q = np.floor(P/voxel).astype(np.int64)
    key = q[:,0]*73856093 ^ q[:,1]*19349663 ^ q[:,2]*83492791
    order = np.argsort(key)

    out_P=[]; out_C=[]
    last=None; best_i=-1; best_w=-1e9
    for idx in order:
        k = key[idx]
        if last is None:
            last=k; best_i=idx; best_w=W[idx]; continue
        if k!=last:
            out_P.append(P[best_i])
            if C is not None: out_C.append(C[best_i])
            last=k; best_i=idx; best_w=W[idx]
        else:
            if W[idx] > best_w:
                best_i=idx; best_w=W[idx]
    if best_i>=0:
        out_P.append(P[best_i]);
        if C is not None: out_C.append(C[best_i])

    Pm = np.asarray(out_P, np.float64)
    Cm = np.asarray(out_C, np.float64) if C is not None else None
    return Pm, Cm

# ---------------- Haupt-Fuse ----------------

def fuse_selected_pointclouds(paths, K, on_log=None, on_progress=None):
    log = (lambda s: (on_log(s) if on_log else None))
    root = paths.root if hasattr(paths,"root") else paths["root"]
    mesh_dir   = os.path.join(root, "mesh")
    frames_dir = os.path.join(root, "raw_frames")
    masks_dir  = os.path.join(root, "features", "masks")
    poses_npz  = os.path.join(root, "poses", "camera_poses.npz")
    depth_dir  = os.path.join(mesh_dir, "depth")
    _mkdir(mesh_dir)

    # --- Konfig
    num_refs    = int(float(os.getenv("FUSION_NUM_REFS", "6")))
    mode        = os.getenv("FUSION_MODE", "majority").strip().lower()  # any|majority|all
    min_in_mask = int(float(os.getenv("FUSION_MIN_IN_MASK", "2")))
    use_depth   = str(os.getenv("FUSION_USE_DEPTH","true")).lower() in ("1","true","yes","on")
    tol_rel     = float(os.getenv("FUSION_DEPTH_TOL","0.03"))
    export_mesh = str(os.getenv("FUSION_EXPORT_MESH","false")).lower() in ("1","true","yes","on")
    chunk       = int(float(os.getenv("FUSION_CHUNK","400000")))
    voxel_cfg   = float(os.getenv("FUSION_VOXEL","0.0"))
    tau_rel     = float(os.getenv("FUSION_SDF_TAU_REL","0.010"))
    tau_abs     = float(os.getenv("FUSION_SDF_TAU","0.0"))

    # Fill
    fill_enable   = str(os.getenv("FUSION_FILL_ENABLE","true")).lower() in ("1","true","yes","on")
    fill_stride   = int(float(os.getenv("FUSION_FILL_STRIDE","2")))
    fill_inpaint  = str(os.getenv("FUSION_FILL_INPAINT","true")).lower() in ("1","true","yes","on")
    fill_inprad   = int(float(os.getenv("FUSION_FILL_INPAINT_RADIUS","3")))
    fill_max_view = int(float(os.getenv("FUSION_FILL_MAX_PER_VIEW","150000")))

    log and log("[fusion] start")

    # --- Lade Frames/Masks/Posen
    frame_files = _sorted_frames(frames_dir)
    if not frame_files: raise RuntimeError(f"Keine Frames in {frames_dir}")
    masks_all = [_mask_for_frame(masks_dir, f) for f in frame_files]
    R_all, t_all, _ = _load_poses_npz(poses_npz)
    N = min(len(frame_files), len(R_all))

    # --- sparse.ply (für Scoring & Distanz)
    sparse_ply = os.path.join(mesh_dir, "sparse.ply")
    if HAS_O3D and os.path.isfile(sparse_ply):
        sp=o3d.io.read_point_cloud(sparse_ply)
        sparse_pts = np.asarray(sp.points, np.float64) if sp is not None else None
    else:
        sparse_pts,_ = _read_ascii_ply_points_colors(sparse_ply)

    # --- Kandidaten (points_ref_XXXX.ply + Maske vorhanden?)
    candidates = []
    for i in range(N):
        if masks_all[i] is None: continue
        if os.path.isfile(os.path.join(mesh_dir, f"points_ref_{i:04d}.ply")):
            candidates.append(i)
    if not candidates:
        raise RuntimeError("[fusion] Keine points_ref_*.ply gefunden.")

    # --- Auswahl (bucketed best-by-sparse)
    scores = _score_frames_by_sparse(K, R_all[:N], t_all[:N], sparse_pts, masks_all[:N])
    refs = _choose_bucketed_best(candidates, scores, total_N=N, num_refs=num_refs)
    if not refs: refs = candidates[:min(num_refs,len(candidates))]
    log and log(f"[fusion] chosen refs (bucketed): {refs}")

    # --- Punkte + Farben laden
    P_list=[]; C_list=[]; SRC=[]
    per_view_points = {}  # für Coverage beim Fill
    for ridx in refs:
        ply = os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply")
        P_ref, C_ref = _read_points_ref_ply(ply)
        if P_ref is None or P_ref.size==0:
            per_view_points[ridx] = (None, None)
            continue
        P_list.append(P_ref)
        SRC.append(np.full(P_ref.shape[0], ridx, np.int32))
        C_list.append(C_ref if C_ref is not None and C_ref.shape[0]==P_ref.shape[0] else None)
        per_view_points[ridx] = (P_ref, C_ref)
        log and log(f"[fusion] load {os.path.basename(ply)} : {P_ref.shape[0]} pts")

    if not P_list: raise RuntimeError("[fusion] Keine Punkte zu fusionieren.")
    P = np.concatenate(P_list, axis=0)
    src_ids = np.concatenate(SRC, axis=0)
    C = np.concatenate(C_list, axis=0) if all(c is not None for c in C_list) else None
    log and log(f"[fusion] stacked: {P.shape[0]} pts")

    # --- View-Assets (nur gewählte Refs)
    masks_ref = {i: masks_all[i] for i in refs}
    depth_ref = {i: (_load_depth(depth_dir, i) if use_depth else None) for i in refs}

    # --- (NEU) Silhouette-Fill pro ausgewählter Referenz
    if fill_enable:
        addedP=[]; addedC=[]; addedS=[]
        for ridx in refs:
            m = masks_ref.get(ridx, None)
            if m is None: continue
            D = depth_ref.get(ridx, None)
            img = _load_frame_image(frame_files, ridx)
            R = R_all[ridx]; t = t_all[ridx]
            Pref, _Cref = per_view_points.get(ridx, (None,None))
            Xadd, Cadd = _silhouette_fill_for_ref(
                ridx, K, R, t, m, D, img, Pref,
                stride=fill_stride, inpaint=fill_inpaint, inpaint_radius=fill_inprad,
                max_pts=fill_max_view
            )
            if Xadd is not None and Xadd.size>0:
                addedP.append(Xadd)
                addedS.append(np.full(Xadd.shape[0], ridx, np.int32))
                if Cadd is not None: addedC.append(Cadd)
                else: addedC.append(None)
                log and log(f"[fusion] fill ref={ridx} -> +{Xadd.shape[0]} pts")
        if addedP:
            Padd = np.concatenate(addedP, axis=0)
            Sadd = np.concatenate(addedS, axis=0)
            if all(c is not None for c in addedC):
                Cadd = np.concatenate(addedC, axis=0)
            else:
                Cadd = None
            P = np.concatenate([P, Padd], axis=0)
            src_ids = np.concatenate([src_ids, Sadd], axis=0)
            if C is not None and Cadd is not None:
                C = np.concatenate([C, Cadd], axis=0)
        log and log(f"[fusion] after fill: {P.shape[0]} pts")

    # 1) Source-Depth-Gate
    keep_src = _source_depth_gate(P, src_ids, K, R_all, t_all, masks_ref, depth_ref, tol_rel=tol_rel, chunk=chunk)
    P = P[keep_src]; src_ids = src_ids[keep_src];
    if C is not None: C = C[keep_src]
    log and log(f"[fusion] after source-depth-gate: {P.shape[0]} pts")

    # 2) Multi-View-Konsens
    R_sel = [R_all[i] for i in refs]; t_sel = [t_all[i] for i in refs]
    M_sel = [masks_ref[i] for i in refs]; D_sel = [depth_ref[i] for i in refs]
    counts, keep_mv = _multi_view_consensus(P, K, R_sel, t_sel, M_sel, D_sel if use_depth else [None]*len(refs),
                                            mode=mode, tol_rel=tol_rel, chunk=chunk)
    keep_mv &= (counts >= max(1, int(min_in_mask)))

    # 3) Soft Sparse Prior (KDTree)
    dist = _dist_to_sparse(P, sparse_pts)
    if sparse_pts is not None and sparse_pts.shape[0] > 0:
        bb = sparse_pts.max(0)-sparse_pts.min(0)
        diag = float(np.linalg.norm(bb))+1e-9
    else:
        diag = 1.0
    tau = max(tau_abs, tau_rel*diag) if (tau_abs>0.0 or tau_rel>0.0) else np.inf
    keep_sdf = (dist <= tau)

    keep = keep_mv | keep_sdf
    P = P[keep]; src_ids = src_ids[keep];
    if C is not None: C = C[keep]
    log and log(f"[fusion] after consensus+soft-sparse: {P.shape[0]} pts")

    # 4) Voxel-Merge (Front gewinnt)
    voxel = None if voxel_cfg <= 0 else float(voxel_cfg)
    Pm, Cm = _voxel_merge_weighted(P, C, src_ids, R_all, t_all, voxel=voxel)
    log and log(f"[fusion] after voxel-merge: {Pm.shape[0]} pts")

    # 5) Save
    out_points = os.path.join(mesh_dir, "fused_points.ply")
    _save_colored_ply(Pm, Cm, out_points)
    log and log(f"[fusion] saved -> {out_points}")

    if export_mesh and HAS_O3D and Pm.shape[0] >= 2000:
        try:
            pcd = o3d.io.read_point_cloud(out_points)
            pcd.estimate_normals()
            depth = int(float(os.getenv("FUSION_POISSON_DEPTH","10")))
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            dens = np.asarray(dens)
            keep_idx = np.where(dens > np.quantile(dens, 0.02))[0]
            mesh = mesh.select_by_index(keep_idx)
            out_mesh = os.path.join(mesh_dir, "fused_mesh.ply")
            o3d.io.write_triangle_mesh(out_mesh, mesh)
            log and log(f"[fusion] saved -> {out_mesh}")
        except Exception as e:
            log and log(f"[fusion] WARN: poisson failed: {e}")

    return out_points
