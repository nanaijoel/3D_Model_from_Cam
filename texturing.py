#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import cv2 as cv
import os

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


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

def _read_points_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        if pc is None or np.asarray(pc.points).size == 0:
            raise RuntimeError(f"Leere Punktwolke: {path}")
        P = np.asarray(pc.points, dtype=np.float64)
        C = None
        if len(pc.colors) > 0:
            # open3d saves RGB [0..1] -> convert to BGR [0..255]
            C = (np.asarray(pc.colors)[:, ::-1] * 255.0).astype(np.float64)
        return P, C

    # ASCII-Fallback
    with open(path, "r") as f:
        header = True
        n = 0
        has_col = False
        while header:
            line = f.readline()
            if not line:
                break
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("property uchar red"):
                has_col = True
            if line.strip() == "end_header":
                header = False
                break
        pts = []
        cols = []
        for _ in range(n):
            parts = f.readline().strip().split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            pts.append((x, y, z))
            if has_col and len(parts) >= 6:
                r, g, b = map(float, parts[3:6])
                cols.append((b, g, r))  # BGR
    P = np.asarray(pts, dtype=np.float64)
    C = np.asarray(cols, dtype=np.float64) if cols else None
    return P, C

def _save_points_ply(path: str, P: np.ndarray, C: np.ndarray | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if HAS_O3D:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        if C is not None and C.shape[0] == P.shape[0]:
            pc.colors = o3d.utility.Vector3dVector((C[:, ::-1] / 255.0).astype(np.float64))  # BGR->RGB
        o3d.io.write_point_cloud(path, pc)
        return

    with open(path, "w") as f:
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


def _load_poses_npz(path_npz: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(path_npz)
    R_all = npz["R"]                  # (N,3,3) world->cam
    t_all = npz["t"].reshape(-1, 3, 1)
    idx = npz.get("frame_idx", np.arange(len(R_all)))
    return R_all, t_all, idx

def _intrinsics_from_image(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    f = 0.92 * float(max(w, h))
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1]], dtype=np.float64)
    return K

def _project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, Pw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # World to camera
    C = (-R.T @ t).reshape(3)                 # Kamerazentrum in Welt
    Xc = (R @ (Pw.T - C.reshape(3, 1))).T     # Nx3 in Kamera
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    return u, v, z


def compute_auto_views(n_frames: int, divisor: int) -> list[int]:
    divisor = max(1, int(divisor))
    if n_frames <= 1:
        return [0]
    idx = np.linspace(0, n_frames - 1, num=divisor + 1, dtype=int)
    idx = np.unique(np.clip(idx, 0, n_frames - 1))
    return idx.tolist()


def texture_points_from_views(
    project_root: str,
    view_ids: list[int],
    in_ply: str = "fused_points.ply",
    out_ply: str = "fused_textured_points.ply",
    weight_power: float = 2.0
) -> str:
    """
    Projects colors from selected view frames onto the 3D point cloud.
    - Colors: always from raw_frames
    - Masks: from features/masks (optional, with dilation)
    """
    frames_dir = os.path.join(project_root, "raw_frames")
    masks_dir = os.path.join(project_root, "features", "masks")
    mesh_dir = os.path.join(project_root, "mesh")
    poses_npz = os.path.join(project_root, "poses", "camera_poses.npz")

    use_masks = str(os.getenv("TEXTURE_USE_MASKS", "true")).lower() in ("1","true","yes","y","on")
    mask_dilate = int(float(os.getenv("TEXTURE_MASK_DILATE", "0")))

    # Load 3D points
    P, C_init = _read_points_ply(os.path.join(mesh_dir, in_ply))
    N = P.shape[0]
    col_sum = np.zeros((N, 3), dtype=np.float64)
    w_sum = np.zeros((N,), dtype=np.float64)

    # Frames and corresponding poses
    frame_files = _sorted_frames(frames_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"Keine Frames in {frames_dir} gefunden.")
    R_all, t_all, _ = _load_poses_npz(poses_npz)

    # Intrinsics of first frame
    im0 = _load_color(frame_files[0])
    K = _intrinsics_from_image(im0)
    H, W = im0.shape[:2]

    # Normalize view_ids to valid range
    vmax = min(len(frame_files), len(R_all))
    view_ids = [int(v) for v in view_ids if 0 <= int(v) < vmax]
    if len(view_ids) == 0:
        view_ids = [0]

    # Prebuild dilate kernel if needed
    kernel = None
    if mask_dilate > 0:
        k = max(1, int(mask_dilate))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*k+1, 2*k+1))

    for vi in view_ids:
        im = _load_color(frame_files[vi])

        # Mask handling
        mask = None
        if use_masks:
            mask = _mask_for_frame(masks_dir, frame_files[vi])
            if mask is None:
                mask = np.ones(im.shape[:2], np.uint8) * 255
            else:
                if kernel is not None:
                    mask = cv.dilate((mask > 0).astype(np.uint8)*255, kernel)

        if mask is None:
            mask = np.ones(im.shape[:2], np.uint8) * 255

        # project points to this view
        u, v, z = _project_points(K, R_all[vi], t_all[vi], P)

        # Valid projections in image + camera + mask
        ui = np.floor(u).astype(np.int32)
        vi2 = np.floor(v).astype(np.int32)
        inside = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi2 >= 0) & (vi2 < H)
        if not np.any(inside):
            continue
        ui = ui[inside]; vi2 = vi2[inside]; z_valid = z[inside]
        pidx = np.nonzero(inside)[0]

        # Mask filter
        vis_mask = (mask[vi2, ui] > 0)
        if not np.any(vis_mask):
            continue
        ui = ui[vis_mask]; vi2 = vi2[vis_mask]
        z_valid = z_valid[vis_mask]; pidx = pidx[vis_mask]

        # Z-Buffer per pixel: nearest point wins
        lin = vi2 * W + ui
        order = np.argsort(z_valid)  # near first
        lin_sorted = lin[order]
        _, first_idx = np.unique(lin_sorted, return_index=True)
        sel = order[first_idx]

        # Sample colors
        cols = im[vi2[sel], ui[sel]].astype(np.float64)

        # Weights (Distance falloff 1/z^power)
        w = 1.0 / np.maximum(z_valid[sel], 1e-6) ** weight_power

        # Accumulate
        np.add.at(col_sum, pidx[sel], (cols.T * w).T)
        np.add.at(w_sum, pidx[sel], w)

    # Final colors
    C = np.zeros((N, 3), dtype=np.float64)
    have = w_sum > 1e-9
    C[have] = (col_sum[have] / w_sum[have][:, None])

    # Fallback: keep old color or mid gray
    if C_init is not None and C_init.shape[0] == N:
        C[~have] = C_init[~have]
    else:
        C[~have] = 127.0

    out_path = os.path.join(mesh_dir, out_ply)
    _save_points_ply(out_path, P, C)
    return out_path


def texture_project_auto(
    project_root: str,
    divisor: int = 6,
    in_ply: str = "fused_points.ply",
    out_ply: str = "fused_textured_points.ply",
    weight_power: float = 2.0
) -> tuple[str, list[int]]:
    frames_dir = os.path.join(project_root, "raw_frames")
    frame_files = _sorted_frames(frames_dir)
    n_frames = len(frame_files)
    if n_frames <= 0:
        raise RuntimeError(f"Keine Frames in {frames_dir} gefunden.")
    views = compute_auto_views(n_frames, divisor)
    out_path = texture_points_from_views(
        project_root=project_root,
        view_ids=views,
        in_ply=in_ply,
        out_ply=out_ply,
        weight_power=weight_power
    )
    return out_path, views
