# fusion.py — Silhouette-Carving + Depth-Gate + front-view-weighted Merge
# - wählt gleichmäßig verteilte Keyframes (bucketed best-by-sparse)
# - lädt points_ref_XXXX.ply (+ Farben) nur für diese Keyframes
# - schneidet Ray-Artefakte: (a) strenges Source-Depth-Gate, (b) Multi-View-Silhouette (AND/Mehrheit)
# - voxel-merging mit Frontalsicht-Priorität (Detail aus Vorderansicht “gewinnt”)

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
    R_all = npz["R"].astype(np.float64)             # (N,3,3) world->cam
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
            x,y,z = map(float, parts[:3])
            pts.append((x,y,z))
            if has_color and len(parts) >= 6:
                r,g,b = map(float, parts[3:6])       # ASCII i.d.R. RGB
                cols.append((b,g,r))                 # -> BGR für cv-Style
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
            # Open3D liefert float RGB [0..1] -> BGR uint8
            C = (np.asarray(pc.colors)[:, ::-1] * 255.0).astype(np.float64)
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


# ---------------- Scoring & Auswahl ----------------

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
        in_img = (u>=0)&(u<W)&(v>=0)&(v<H)&(Z>1e-6)
        if not np.any(in_img): continue
        ui = np.clip(np.floor(u[in_img]).astype(np.int32),0,W-1)
        vi = np.clip(np.floor(v[in_img]).astype(np.int32),0,H-1)
        scores[i] = int(np.count_nonzero(m[vi,ui] > 0))
    return scores

def _choose_bucketed_best_from_candidates(scores_all, candidates, total_N, num_refs):
    """
    Teile [0..total_N) in num_refs Buckets und wähle je Bucket
    den Kandidaten mit maximalem Score (falls keiner im Bucket: nimm
    den Kandidaten mit minimaler Distanz zum Bucketzentrum).
    """
    chosen = []
    if not candidates:
        return chosen
    candidates = sorted(set(int(c) for c in candidates))
    num_refs  = max(1, int(num_refs))
    bucket_w  = total_N / float(num_refs)
    for b in range(num_refs):
        a = int(round(b*bucket_w))
        z = int(round((b+1)*bucket_w))
        bucket = [c for c in candidates if a <= c < z]
        if bucket:
            best = max(bucket, key=lambda i: int(scores_all[i]))
            chosen.append(best)
        else:
            # fallback: nearest candidate to bucket center
            center = int(round((a+z)/2.0))
            best = min(candidates, key=lambda i: abs(i-center))
            if best not in chosen:
                chosen.append(best)
    return sorted(set(chosen))


# ---------------- Silhouette/Depth-Filter ----------------

def _load_depth(depth_dir, idx):
    p = os.path.join(depth_dir, f"depth_{idx:04d}.npy")
    if os.path.isfile(p):
        try: return np.load(p)
        except Exception: return None
    return None

def _source_depth_gate(P, src_ids, K, R_all, t_all, masks, depth_maps, tol_rel=0.03, chunk=400000):
    """Trimmt Ray-Schweife streng im Source-View: z <= D(u,v)+(tol_rel*(1+D))."""
    N = P.shape[0]
    ok_all = np.zeros(N, dtype=np.bool_)
    # Gruppiere Punkte nach Source-ID für effizientes Projizieren
    from collections import defaultdict
    groups = defaultdict(list)
    for i, sid in enumerate(src_ids): groups[int(sid)].append(i)
    for sid, idxs in groups.items():
        m = masks.get(sid, None); D = depth_maps.get(sid, None)
        R = R_all[sid]; t = t_all[sid]
        if m is None or D is None:  # ohne Maske/Depth können wir nicht prüfen -> verwerfen wir NICHT
            ok_all[idxs] = True
            continue
        H,W = m.shape[:2]
        Cc = _cam_center(R,t)
        idxs = np.asarray(idxs, np.int64)
        for s in range(0, idxs.size, chunk):
            ids = idxs[s:s+chunk]
            Q = P[ids]
            Xc = (R @ (Q.T - Cc.reshape(3,1))).T
            z = Xc[:,2]
            u,v = _project(K, Xc)
            ui = np.floor(u).astype(np.int32)
            vi = np.floor(v).astype(np.int32)
            inside = (z>1e-6)&(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
            if not np.any(inside):
                continue
            ui = np.clip(ui,0,W-1); vi = np.clip(vi,0,H-1)
            inside &= (m[vi,ui] > 0)
            d = D[vi,ui]
            inside &= (z <= (d + tol_rel*(1.0 + d)))
            ok_all[ids] = inside
    return ok_all

def _multi_view_silhouette(P, K, R_list, t_list, masks_list, depth_list, mode="all", tol_rel=0.03, chunk=400000):
    """Behalte Punkte, die in mehreren Sichten in der Silhouette liegen; optional Tiefen-Gate."""
    V = len(R_list); N = P.shape[0]
    counts = np.zeros(N, np.int32)
    for v in range(V):
        R = R_list[v]; t = t_list[v]
        m = masks_list[v]; D = depth_list[v]
        if m is None: continue
        H,W = m.shape[:2]
        Cc = _cam_center(R,t)
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            Q = P[s:e]
            Xc = (R @ (Q.T - Cc.reshape(3,1))).T
            z = Xc[:,2]
            u,v = _project(K, Xc)
            ui = np.floor(u).astype(np.int32)
            vi = np.floor(v).astype(np.int32)
            inside = (z>1e-6)&(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
            if not np.any(inside):
                continue
            ui = np.clip(ui,0,W-1); vi = np.clip(vi,0,H-1)
            inside &= (m[vi,ui] > 0)
            if D is not None:
                d = D[vi,ui]
                inside &= (z <= (d + tol_rel*(1.0 + d)))
            counts[s:e] += inside.astype(np.int32)
    if mode == "all":
        return counts == V
    if mode == "majority":
        return counts >= int(math.ceil(V/2.0))
    return counts >= 1  # any


# ---------------- Merge (Frontalsicht gewinnt) ----------------

def _voxel_merge_weighted(P, C, src_ids, R_all, t_all, voxel=None):
    if P.size == 0: return P, C
    # Voxelgröße schätzen (Median-NN)
    if voxel is None or voxel <= 0:
        try:
            from scipy.spatial import cKDTree
            idx = np.random.choice(P.shape[0], size=min(6000,P.shape[0]), replace=False)
            Q = P[idx]; tree = cKDTree(Q); d,_ = tree.query(Q, k=2)
            med = np.median(d[:,1]) if d.ndim>1 else np.median(d)
        except Exception:
            bb = P.max(0)-P.min(0); med = float(np.linalg.norm(bb))/400.0
        voxel = max(1e-4, 0.6*float(med))

    # Kamerazentren & Forward
    src_set = set(int(s) for s in src_ids.tolist())
    Cc = {sid: _cam_center(R_all[sid], t_all[sid]) for sid in src_set}
    Fd = {sid: _forward_dir(R_all[sid])            for sid in src_set}

    # Frontalsicht-Gewichte
    W = np.zeros(P.shape[0], np.float64)
    for i in range(P.shape[0]):
        sid = int(src_ids[i])
        v = P[i]-Cc[sid]; nv = np.linalg.norm(v)+1e-12; v = v/nv
        f = Fd[sid]; nf = np.linalg.norm(f)+1e-12
        W[i] = float(np.clip(np.dot(v,f)/nf, -1.0, 1.0))

    # Voxel-Hash + max-by-weight
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
    num_refs   = int(float(os.getenv("FUSION_NUM_REFS", "6")))
    mode       = os.getenv("FUSION_MODE", "all").strip().lower()         # any|majority|all
    use_depth  = str(os.getenv("FUSION_USE_DEPTH","true")).lower() in ("1","true","yes","on")
    tol_rel    = float(os.getenv("FUSION_DEPTH_TOL","0.03"))
    export_mesh= str(os.getenv("FUSION_EXPORT_MESH","false")).lower() in ("1","true","yes","on")  # default aus!
    chunk      = int(float(os.getenv("FUSION_CHUNK","400000")))

    # --- Lade Frames/Masks/Posen
    frame_files = _sorted_frames(frames_dir)
    if not frame_files: raise RuntimeError(f"Keine Frames in {frames_dir}")
    masks_all = [_mask_for_frame(masks_dir, f) for f in frame_files]
    R_all, t_all, _ = _load_poses_npz(poses_npz)
    N = min(len(frame_files), len(R_all))

    # --- sparse.ply für Scores
    sparse_ply = os.path.join(mesh_dir, "sparse.ply")
    if HAS_O3D:
        sp=o3d.io.read_point_cloud(sparse_ply)
        sparse_pts = np.asarray(sp.points, np.float64) if sp is not None else None
    else:
        sparse_pts,_ = _read_ascii_ply_points_colors(sparse_ply)

    # --- Kandidaten = Frames, für die points_ref_XXXX.ply existiert (+Maske)
    candidates = []
    for i in range(N):
        if masks_all[i] is None: continue
        if os.path.isfile(os.path.join(mesh_dir, f"points_ref_{i:04d}.ply")):
            candidates.append(i)
    if not candidates:
        raise RuntimeError("[fusion] Keine points_ref_*.ply gefunden.")

    # --- Scores und bucketed Auswahl
    scores = _score_frames_by_sparse(K, R_all[:N], t_all[:N], sparse_pts, masks_all[:N])
    refs = _choose_bucketed_best_from_candidates(scores, candidates, total_N=N, num_refs=num_refs)
    if not refs:
        # Fallback: gleichmäßig aus candidates
        step = max(1, len(candidates)//max(1,num_refs))
        refs = candidates[::step][:num_refs]
    log and log(f"[fusion] chosen refs (bucketed): {refs}")

    # --- Punkte + Farben laden
    P_list=[]; C_list=[]; SRC=[]
    for ridx in refs:
        ply = os.path.join(mesh_dir, f"points_ref_{ridx:04d}.ply")
        P,C = _read_points_ref_ply(ply)
        if P is None or P.size==0: continue
        P_list.append(P)
        SRC.append(np.full(P.shape[0], ridx, np.int32))
        C_list.append(C if C is not None and C.shape[0]==P.shape[0] else None)
        log and log(f"[fusion] load {os.path.basename(ply)} : {P.shape[0]} pts")
    if not P_list: raise RuntimeError("[fusion] Keine Punkte zu fusionieren.")

    P = np.concatenate(P_list, axis=0)
    src_ids = np.concatenate(SRC, axis=0)
    C = np.concatenate(C_list, axis=0) if all(c is not None for c in C_list) else None

    # --- View-Assets für die ausgewählten Refs
    masks_ref = {i: masks_all[i] for i in refs}
    depth_ref = {i: (_load_depth(depth_dir, i) if use_depth else None) for i in refs}

    # 1) Source-Depth-Gate (straffes Trimmen pro Herkunftssicht)
    keep_src = _source_depth_gate(P, src_ids, K, R_all, t_all, masks_ref, depth_ref, tol_rel=tol_rel, chunk=chunk)
    P = P[keep_src]; src_ids = src_ids[keep_src];
    if C is not None: C = C[keep_src]
    log and log(f"[fusion] after source-depth-gate: {P.shape[0]} pts")

    # 2) Multi-View-Silhouette (alle gewählten Refs, ‘all’ als Default)
    R_sel = [R_all[i] for i in refs]; t_sel = [t_all[i] for i in refs]
    M_sel = [masks_ref[i] for i in refs]; D_sel = [depth_ref[i] for i in refs]
    keep_mv = _multi_view_silhouette(P, K, R_sel, t_sel, M_sel, D_sel, mode=mode, tol_rel=tol_rel, chunk=chunk)
    P = P[keep_mv]; src_ids = src_ids[keep_mv];
    if C is not None: C = C[keep_mv]
    log and log(f"[fusion] after multi-view silhouette: {P.shape[0]} pts")

    # 3) Merge mit Frontalsicht-Priorität
    Pm, Cm = _voxel_merge_weighted(P, C, src_ids, R_all, t_all, voxel=None)
    log and log(f"[fusion] after voxel-merge: {Pm.shape[0]} pts")

    # 4) Save
    out_points = os.path.join(mesh_dir, "fused_points.ply")
    _save_colored_ply(Pm, Cm, out_points)
    log and log(f"[fusion] saved -> {out_points}")

    # (optional) Mesh – per Default AUS, erst aktivieren, wenn Oberfläche stabil ist
    if export_mesh and HAS_O3D and Pm.shape[0] >= 2000:
        try:
            pcd = o3d.io.read_point_cloud(out_points)
            pcd.estimate_normals()
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=int(float(os.getenv("FUSION_POISSON_DEPTH","10")))
            )
            dens = np.asarray(dens)
            keep = dens > np.quantile(dens, 0.02)
            mesh = mesh.select_by_index(np.where(keep)[0])
            out_mesh = os.path.join(mesh_dir, "fused_mesh.ply")
            o3d.io.write_triangle_mesh(out_mesh, mesh)
            log and log(f"[fusion] saved -> {out_mesh}")
        except Exception as e:
            log and log(f"[fusion] WARN: poisson failed: {e}")

    return out_points
