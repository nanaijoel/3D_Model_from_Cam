# texturing.py
import os
import glob
import traceback
import numpy as np
import cv2 as cv

# ----------- Utils -----------

def _log(msg, on_log=None):
    if on_log: on_log(msg)
    else: print(msg, flush=True)

def _safe_imread(path):
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None: raise IOError(f"Failed to load image: {path}")
    if im.ndim == 2: im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    if im.shape[2] == 4: im = im[:, :, :3]
    return im

def _list_raw_frames(raw_dir, pattern="*.png"):
    paths = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    return paths

def _load_intrinsics(npy_path):
    k = np.load(npy_path)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics.npy has wrong shape {k.shape}")
    return k

def _read_poses_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "poses" in d:
        poses = np.asarray(d["poses"])
        idx = np.asarray(d["indices"]) if "indices" in d else np.arange(len(poses))
        return idx.tolist(), poses
    if "Tcw_list" in d:
        poses = np.asarray(d["Tcw_list"])
        idx = np.arange(len(poses))
        return idx.tolist(), poses
    # unser SfM-Export: R,t
    if "R" in d and "t" in d:
        R = d["R"]; t = d["t"].reshape(-1, 3, 1)
        T = np.repeat(np.eye(4)[None, :, :], R.shape[0], axis=0)
        T[:, :3, :3] = R
        T[:, :3, 3:4] = t
        idx = d.get("frame_idx", np.arange(R.shape[0]))
        return idx.tolist(), T
    raise ValueError("Unsupported pose npz format")

# ----------- Open3D optional I/O (robust) -----------

def _o3d_safe():
    try:
        import open3d as o3d
        return o3d
    except Exception:
        return None

def _read_ply_ascii(in_path):
    with open(in_path, "r") as f:
        header = []
        while True:
            line = f.readline()
            if not line: raise IOError("Invalid PLY (no end_header)")
            header.append(line.strip())
            if line.strip() == "end_header": break
        n = 0
        for h in header:
            if h.startswith("element vertex"):
                n = int(h.split()[-1]); break
        if n <= 0: raise IOError("PLY missing 'element vertex'")
        xyz, rgb = [], []
        for _ in range(n):
            parts = f.readline().strip().split()
            x, y, z = map(float, parts[:3])
            if len(parts) >= 6:
                r, g, b = map(float, parts[3:6])
                if max(r,g,b) > 1.5: rgb.append([r/255.0, g/255.0, b/255.0])
                else:                rgb.append([r, g, b])
            else:
                rgb.append([0,0,0])
            xyz.append([x, y, z])
    return np.asarray(xyz, np.float64), np.asarray(rgb, np.float64)

def _read_ply(in_path, on_log=None):
    o3d = _o3d_safe()
    if o3d is not None:
        try:
            pcd = o3d.io.read_point_cloud(in_path)
            if pcd is None or pcd.is_empty(): raise IOError("open3d read empty")
            xyz = np.asarray(pcd.points, dtype=np.float64)
            rgb = np.asarray(pcd.colors, dtype=np.float64) if len(pcd.colors) else np.zeros_like(xyz)
            return xyz, rgb
        except Exception as e:
            _log(f"[texture] open3d read failed → ASCII fallback ({e})", on_log)
    return _read_ply_ascii(in_path)

def _write_ply(out_path, xyz, rgb, on_log=None):
    o3d = _o3d_safe()
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(rgb, np.float64), 0, 1))
            ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
            if not ok: raise IOError("open3d write returned False")
            return
        except Exception as e:
            _log(f"[texture] open3d write failed → ASCII fallback ({e})", on_log)
    # ASCII fallback
    xyz = np.asarray(xyz, np.float64); rgb = np.asarray(rgb, np.float64)
    rgb255 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property double x\nproperty double y\nproperty double z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb255):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# ----------- Projektion & Sampling -----------

def _project_points(K, Tcw, pts3d):
    R = Tcw[:3, :3]; t = Tcw[:3, 3:4]
    P = (R @ pts3d.T) + t
    z = P[2, :]
    valid = z > 1e-6
    u = (K[0, 0] * (P[0, :] / z)) + K[0, 2]
    v = (K[1, 1] * (P[1, :] / z)) + K[1, 2]
    return np.stack([u, v], axis=1), valid

def _bilinear_sample(img, uv):
    h, w = img.shape[:2]
    u = uv[:, 0]; v = uv[:, 1]
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1; v1 = v0 + 1
    u0c = np.clip(u0, 0, w - 1); u1c = np.clip(u1, 0, w - 1)
    v0c = np.clip(v0, 0, h - 1); v1c = np.clip(v1, 0, h - 1)

    Ia = img[v0c, u0c, :].astype(np.float64)
    Ib = img[v0c, u1c, :].astype(np.float64)
    Ic = img[v1c, u0c, :].astype(np.float64)
    Id = img[v1c, u1c, :].astype(np.float64)

    wa = (u1 - u) * (v1 - v)
    wb = (u - u0) * (v1 - v)
    wc = (u1 - u) * (v - v0)
    wd = (u - u0) * (v - v0)

    out = (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None])
    return out / 255.0

def _inside_mask(mask, uv, margin=1):
    h, w = mask.shape[:2]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    ok = (u >= margin) & (v >= margin) & (u < (w - margin)) & (v < (h - margin))
    res = np.zeros_like(ok)
    idx = np.where(ok)[0]
    res[idx] = mask[v[idx], u[idx]] > 127
    return res

def _weight_by_angle_and_dist(Tcw, pts3d):
    cam_z = Tcw[:3, 2]
    cam_pos = -Tcw[:3, :3].T @ Tcw[:3, 3]
    vec = pts3d - cam_pos[None, :]
    dist = np.linalg.norm(vec, axis=1) + 1e-6
    vec_n = vec / dist[:, None]
    cosang = np.abs(np.dot(vec_n, cam_z))
    return (cosang ** 3) / (dist ** 1.0)

# ----------- View-Auswahl -----------

def _parse_views_env(on_log=None):
    mode = os.getenv("TEXTURE_MODE", "auto").strip().lower()
    divisor = max(1, int(float(os.getenv("TEXTURE_DIVISOR", "12"))))
    manual_csv = os.getenv("TEXTURE_MANUAL_VIEWS", "").strip()
    manual = []
    if manual_csv:
        try:
            manual = [int(x) for x in manual_csv.replace(" ", "").split(",") if x != ""]
        except Exception:
            _log(f"[texture] WARN: cannot parse TEXTURE_MANUAL_VIEWS='{manual_csv}'", on_log)
    return mode, divisor, manual

def _select_views(n_total, mode, divisor, manual):
    if n_total <= 0: return []
    if mode == "manual" and manual:
        return sorted({i for i in manual if 0 <= i < n_total})
    step = max(1, int(round(n_total / max(1, divisor))))
    idx = list(range(0, n_total, step))
    if (n_total - 1) not in idx: idx.append(n_total - 1)
    return sorted(idx)

# ----------- Kern-Texturing -----------

def texture_points_from_views(in_ply, out_ply, intrinsics_npy, poses_npz,
                              raw_frames_dir, view_indices, use_masks=True,
                              mask_dir=None, mask_dilate=0, weight_power=2.0,
                              drop_untextured=False, on_log=None):
    _log(f"[texture] input ply: {os.path.basename(in_ply)}", on_log)

    pts3d, _ = _read_ply(in_ply, on_log=on_log)
    if pts3d.size == 0:
        raise RuntimeError("No points in input PLY.")

    K = _load_intrinsics(intrinsics_npy)
    _, Tcw_all = _read_poses_npz(poses_npz)
    frames = _list_raw_frames(raw_frames_dir)
    if len(frames) == 0:
        raise RuntimeError("No frames found for texturing.")

    used = [i for i in view_indices if 0 <= i < len(frames)]
    if not used: raise RuntimeError("No valid views for texturing.")

    masks = {}
    if use_masks and mask_dir and os.path.isdir(mask_dir):
        for vi in used:
            # mask-Dateinamen: 000000.png, 000001.png, …
            mp = os.path.join(mask_dir, f"{vi:06d}.png")
            if os.path.isfile(mp):
                m = cv.imread(mp, cv.IMREAD_GRAYSCALE)
                if m is not None and mask_dilate > 0:
                    m = cv.dilate(m, np.ones((mask_dilate, mask_dilate), np.uint8))
                masks[vi] = m

    acc = np.zeros((pts3d.shape[0], 3), dtype=np.float64)
    wsum = np.zeros((pts3d.shape[0],), dtype=np.float64)

    for vi in used:
        img = _safe_imread(frames[vi])
        h, w = img.shape[:2]
        if vi >= len(Tcw_all): continue
        Tcw = Tcw_all[vi]

        uv, z_ok = _project_points(K, Tcw, pts3d)
        in_img = ((uv[:, 0] >= 0) & (uv[:, 1] >= 0) &
                  (uv[:, 0] < w - 1) & (uv[:, 1] < h - 1) & z_ok)

        if use_masks and vi in masks:
            in_mask = _inside_mask(masks[vi], uv, margin=1)
            in_img = in_img & in_mask

        idx = np.where(in_img)[0]
        if idx.size == 0: continue

        samp = _bilinear_sample(img, uv[idx, :])
        ww = _weight_by_angle_and_dist(Tcw, pts3d[idx, :]) ** float(weight_power)
        acc[idx, :] += samp * ww[:, None]
        wsum[idx] += ww

    colored = wsum > 1e-9
    rgb = np.zeros_like(acc)
    rgb[colored, :] = (acc[colored, :] / wsum[colored, None])

    if not colored.all() and drop_untextured:
        keep_idx = np.where(colored)[0]
        _log(f"[texture] dropping {pts3d.shape[0]-keep_idx.size} untextured points", on_log)
        pts3d = pts3d[keep_idx, :]
        rgb = rgb[keep_idx, :]

    _write_ply(out_ply, pts3d, rgb, on_log=on_log)
    _log(f"[texture] wrote: {out_ply}", on_log)

# ----------- Public Entry -----------

def run_texturing(mesh_dir, on_log=None):
    if os.getenv("TEXTURE_ENABLE", "true").lower() not in ("1","true","yes","on"):
        _log("[texture] disabled.", on_log)
        return None

    in_ply = os.getenv("TEXTURE_IN_PLY", "fused_points.ply")
    out_ply = os.getenv("TEXTURE_OUT_PLY", "fused_textured_points.ply")
    use_masks = os.getenv("TEXTURE_USE_MASKS", "true").lower() in ("1","true","yes","on")
    mask_dilate = int(float(os.getenv("TEXTURE_MASK_DILATE", "2")))
    weight_power = float(os.getenv("TEXTURE_WEIGHT_POWER", "2.0"))
    drop_untextured = os.getenv("TEXTURE_DROP_UNTEXTURED", "false").lower() in ("1","true","yes","on")

    intr = os.path.join(mesh_dir, "intrinsics.npy")
    in_ply_path = os.path.join(mesh_dir, in_ply)
    out_ply_path = os.path.join(mesh_dir, out_ply)

    root = os.path.abspath(os.path.join(mesh_dir, os.pardir))
    poses_npz = os.path.join(root, "poses", "camera_poses.npz")
    raw_dir = os.path.join(root, "raw_frames")
    mask_dir = os.path.join(root, "features", "masks") if use_masks else None

    frames = _list_raw_frames(raw_dir)
    n_total = len(frames)

    mode, divisor, manual = _parse_views_env(on_log)
    views = _select_views(n_total, mode, divisor, manual)
    _log(f"[texture] auto-views (div={divisor}) -> {views}" if mode!="manual" else f"[texture] manual-views -> {views}", on_log)

    try:
        texture_points_from_views(
            in_ply=in_ply_path,
            out_ply=out_ply_path,
            intrinsics_npy=intr,
            poses_npz=poses_npz,
            raw_frames_dir=raw_dir,
            view_indices=views,
            use_masks=use_masks,
            mask_dir=mask_dir,
            mask_dilate=mask_dilate,
            weight_power=weight_power,
            drop_untextured=drop_untextured,
            on_log=on_log
        )
        return out_ply_path
    except Exception as e:
        _log(f"[texture] failed (robust path): {e}\n{traceback.format_exc()}", on_log)
        raise
