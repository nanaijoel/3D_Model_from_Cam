# texturing.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustes Texturing mit sauberem Frame↔Pose-Mapping, kompatibel zu pipeline_runner.run_texturing(mesh_dir, on_log).

Projektstruktur:
- projects/NAME/raw_frames/{000123.png|frame_0123.png|*.jpg}
- projects/NAME/features/masks/{000123.png|frame_0123_mask.png|*.jpg}
- projects/NAME/poses/camera_poses.npz               (R, t, frame_idx)
- projects/NAME/mesh/{fused_points.ply, intrinsics.npy}
- projects/NAME/mesh/fused_textured_points.ply       (output)
"""

import os
import re
import glob
import numpy as np
import cv2 as cv
from typing import List, Tuple, Dict, Optional

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


# ----------------------------- I/O Utilities ------------------------------

def _log(msg, on_log=None):
    if on_log:
        on_log(msg)
    else:
        print(msg, flush=True)

def _imread_color(path: str) -> np.ndarray:
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = im[:, :, :3]
    return im

def _frame_id_from_name(path: str) -> int:
    """Extrahiert die letzte zusammenhängende Zahl aus dem Dateinamen (sonst -1)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.findall(r"(\d+)", stem)
    return int(m[-1]) if m else -1

def _list_frames_with_ids(frames_dir: str) -> List[Tuple[str, int]]:
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg")))
    return [(p, _frame_id_from_name(p)) for p in files]

def _mask_for_frame(masks_dir: str, frame_path: str, frame_id: int, dilate_px: int = 0) -> Optional[np.ndarray]:
    """Findet passende Maske zu einem Frame – mehrere Namensschemata werden probiert."""
    if not (masks_dir and os.path.isdir(masks_dir)):
        return None
    base = os.path.basename(frame_path)
    stem, _ = os.path.splitext(base)
    cands = [
        os.path.join(masks_dir, f"{stem}_mask.png"),
        os.path.join(masks_dir, f"{stem}_mask.jpg"),
        os.path.join(masks_dir, f"{stem}.png"),
        os.path.join(masks_dir, f"{stem}.jpg"),
    ]
    if frame_id >= 0:
        cands += [
            os.path.join(masks_dir, f"{frame_id:06d}.png"),
            os.path.join(masks_dir, f"{frame_id:06d}.jpg"),
            os.path.join(masks_dir, f"frame_{frame_id:04d}_mask.png"),
            os.path.join(masks_dir, f"frame_{frame_id:04d}_mask.jpg"),
        ]
    for c in cands:
        if os.path.isfile(c):
            m = cv.imread(c, cv.IMREAD_GRAYSCALE)
            if m is not None and dilate_px > 0:
                m = cv.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
            return m
    return None

def _read_points_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        if pc is None or np.asarray(pc.points).size == 0:
            raise RuntimeError(f"Empty point cloud: {path}")
        P = np.asarray(pc.points, dtype=np.float64)
        C = None
        if len(pc.colors) > 0:
            # open3d colors sind RGB [0..1]; intern BGR [0..255]
            C = (np.asarray(pc.colors)[:, ::-1] * 255.0).astype(np.float64)
        return P, C

    # ASCII fallback
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
        pts, cols = [], []
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

def _save_points_ply(path: str, P: np.ndarray, C: Optional[np.ndarray] = None) -> None:
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


# --------------------------- Poses & Intrinsics ----------------------------

def _read_poses_rt_idx(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    if "R" in d and "t" in d:
        R = d["R"]                 # (N,3,3) world->cam
        t = d["t"].reshape(-1, 3, 1)
        idx = d.get("frame_idx", np.arange(len(R)))
        return np.asarray(R), np.asarray(t), np.asarray(idx)
    if "poses" in d and "indices" in d:
        T = np.asarray(d["poses"])     # 4x4; wir nehmen als w->c
        idx = np.asarray(d["indices"])
        R = T[:, :3, :3]
        t = T[:, :3, 3:4]
        return R, t, idx
    raise ValueError("camera_poses.npz missing (R,t) or (poses,indices)")

def _load_intrinsics(mesh_dir: str, fallback_image: np.ndarray) -> np.ndarray:
    npy = os.path.join(mesh_dir, "intrinsics.npy")
    if os.path.isfile(npy):
        K = np.load(npy)
        if K.shape == (3, 3):
            return K.astype(np.float64)
    # robuste Heuristik
    h, w = fallback_image.shape[:2]
    f = 0.92 * float(max(w, h))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


# ------------------------- Projection & Sampling ---------------------------

def _project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, Pw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # world->camera
    C = (-R.T @ t).reshape(3)                 # camera center in world
    Xc = (R @ (Pw.T - C.reshape(3, 1))).T     # N x 3 in camera
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    return u, v, z

def _inside_mask(mask: np.ndarray, u: np.ndarray, v: np.ndarray, w: int, h: int) -> np.ndarray:
    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    res = np.zeros_like(ok)
    idx = np.where(ok)[0]
    res[idx] = (mask[vi[idx], ui[idx]] > 0)
    return res

def _weight_by_depth(z: np.ndarray, power: float) -> np.ndarray:
    return 1.0 / np.maximum(z, 1e-6) ** power

def _in_image_ratio(u, v, z, w, h) -> float:
    ok = (z > 1e-6) & (u >= 0) & (u < (w - 1)) & (v >= 0) & (v < (h - 1))
    if ok.size == 0:
        return 0.0
    return float(np.mean(ok))


# ------------------------------- Public Core --------------------------------

def compute_auto_views(n_frames: int, divisor: int) -> List[int]:
    """Gleichmäßig verteilte View-Indizes (divisor+1 Stück, inkl. erster/letzter)."""
    divisor = max(1, int(divisor))
    if n_frames <= 1:
        return [0]
    idx = np.linspace(0, n_frames - 1, num=divisor + 1, dtype=int)
    idx = np.unique(np.clip(idx, 0, n_frames - 1))
    return idx.tolist()

def texture_points_from_views(
    project_root: str,
    view_ids: List[int],
    in_ply: str = "fused_points.ply",
    out_ply: str = "fused_textured_points.ply",
    use_masks: bool = True,
    mask_dilate: int = 0,
    weight_power: float = 2.0,
    on_log=None
) -> str:
    """
    Robuste Farbprojektion:
    - Frame-Dateien werden über numerische ID auf Posen (frame_idx) gemappt,
      mit automatischem Fallback auf Reihenfolge (per-View), falls die ID-Mapping-Hypothese unplausibel ist.
    - Z-Buffer pro Pixel (nächster Punkt gewinnt)
    - Gewichtet per 1/z^power
    """
    frames_dir = os.path.join(project_root, "raw_frames")
    masks_dir  = os.path.join(project_root, "features", "masks")
    mesh_dir   = os.path.join(project_root, "mesh")
    poses_npz  = os.path.join(project_root, "poses", "camera_poses.npz")

    # 3D Punkte laden
    P, C_init = _read_points_ply(os.path.join(mesh_dir, in_ply))
    N = P.shape[0]
    col_sum = np.zeros((N, 3), dtype=np.float64)
    w_sum   = np.zeros((N,), dtype=np.float64)

    # Frames & Posen
    frames = _list_frames_with_ids(frames_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No frames in {frames_dir}.")
    R_all, t_all, idx = _read_poses_rt_idx(poses_npz)

    # Intrinsics
    im0 = _imread_color(frames[0][0])
    K = _load_intrinsics(mesh_dir, im0)

    # Maps
    pose_by_id   = {int(fi): (R_all[k], t_all[k]) for k, fi in enumerate(idx)}
    pose_by_rank = {k: (R_all[k], t_all[k]) for k in range(len(R_all))}

    # Nur valide view ids
    vmax = len(frames)
    view_ids = [int(v) for v in view_ids if 0 <= int(v) < vmax]
    if not view_ids:
        view_ids = [0]

    # Plausibilitätswarnung bei massiv fehlenden Posen (ID-Mapping)
    missing = [vi for vi in view_ids if frames[vi][1] not in pose_by_id]
    if missing and (len(missing) / max(1, len(view_ids))) > 0.3:
        _log(f"[texture] WARN: pose/frame mismatch likely – falling back to order for many views", on_log)

    for vi in view_ids:
        fpath, fid = frames[vi]
        im = _imread_color(fpath)
        h, w = im.shape[:2]
        mask = _mask_for_frame(masks_dir, fpath, fid, dilate_px=mask_dilate) if use_masks else None
        if mask is None:
            mask = np.ones((h, w), np.uint8) * 255

        # 1) Versuch: Mapping per frame-id
        R, t = pose_by_id.get(fid, (None, None))
        choose_rank = False
        if R is None:
            choose_rank = True
        else:
            u_try, v_try, z_try = _project_points(K, R, t, P)
            if _in_image_ratio(u_try, v_try, z_try, w, h) < 0.15:
                choose_rank = True

        # Fallback: Mapping per Reihenfolge (vi)
        if choose_rank:
            Rt = pose_by_rank.get(vi, (None, None))
            if Rt[0] is None:
                continue
            R, t = Rt
            u_try, v_try, z_try = _project_points(K, R, t, P)

        # endgültige Projektion
        u, v, z = u_try, v_try, z_try

        inside = (z > 1e-6) & (u >= 0) & (u < (w - 1)) & (v >= 0) & (v < (h - 1))
        if not np.any(inside):
            continue

        in_mask = np.zeros_like(inside)
        idx_in  = np.where(inside)[0]
        in_mask[idx_in] = _inside_mask(mask, u[idx_in], v[idx_in], w, h)
        vis = inside & in_mask
        if not np.any(vis):
            continue

        # Z-Buffer pro Pixel (Nearest wins)
        ui = np.floor(u[vis]).astype(np.int32)
        vi_pix = np.floor(v[vis]).astype(np.int32)
        z_vis  = z[vis]
        pidx   = np.nonzero(vis)[0]

        lin = vi_pix * w + ui
        order = np.argsort(z_vis)  # nahe zuerst
        lin_sorted = lin[order]
        _, first_idx = np.unique(lin_sorted, return_index=True)
        sel = order[first_idx]  # Indizes innerhalb pidx

        cols = im[vi_pix[sel], ui[sel]].astype(np.float64)
        ww   = _weight_by_depth(z_vis[sel], weight_power)

        np.add.at(col_sum, pidx[sel], (cols.T * ww).T)
        np.add.at(w_sum,   pidx[sel], ww)

    # Finalfarben
    C = np.zeros((N, 3), dtype=np.float64)
    have = (w_sum > 1e-9)
    C[have] = (col_sum[have] / w_sum[have][:, None])

    # Fallback: alte Farben beibehalten, sonst neutrales Grau
    if C_init is not None and C_init.shape[0] == N:
        C[~have] = C_init[~have]
    else:
        C[~have] = 127.0

    out_path = os.path.join(mesh_dir, out_ply)
    _save_points_ply(out_path, P, C)
    _log(f"[texture] wrote: {out_path}", on_log)
    return out_path


def texture_project_auto(
    project_root: str,
    divisor: int = 12,
    in_ply: str = "fused_points.ply",
    out_ply: str = "fused_textured_points.ply",
    use_masks: bool = True,
    mask_dilate: int = 0,
    weight_power: float = 2.0,
    on_log=None
) -> Tuple[str, List[int]]:
    frames_dir = os.path.join(project_root, "raw_frames")
    n_frames = len(_list_frames_with_ids(frames_dir))
    views = compute_auto_views(n_frames, divisor)
    out_path = texture_points_from_views(
        project_root=project_root,
        view_ids=views,
        in_ply=in_ply,
        out_ply=out_ply,
        use_masks=use_masks,
        mask_dilate=mask_dilate,
        weight_power=weight_power,
        on_log=on_log
    )
    return out_path, views


# ---------------------------- Runner-kompatibel ------------------------------

def _parse_bool_env(s: Optional[str], default: bool = False) -> bool:
    if s is None: return default
    return str(s).strip().lower() in ("1","true","yes","y","on")

def run_texturing(mesh_dir: str, on_log=None) -> Optional[str]:
    """
    Wrapper für pipeline_runner:
    - liest ENV: TEXTURE_ENABLE, TEXTURE_MODE, TEXTURE_DIVISOR, TEXTURE_MANUAL_VIEWS,
                 TEXTURE_IN_PLY, TEXTURE_OUT_PLY, TEXTURE_USE_MASKS, TEXTURE_MASK_DILATE,
                 TEXTURE_WEIGHT_POWER
    - ruft texture_points_from_views(...) mit Auto- oder Manual-Views.
    """
    if not _parse_bool_env(os.getenv("TEXTURE_ENABLE", "true"), True):
        _log("[texture] disabled.", on_log)
        return None

    in_ply  = os.getenv("TEXTURE_IN_PLY", "fused_points.ply")
    out_ply = os.getenv("TEXTURE_OUT_PLY", "fused_textured_points.ply")
    use_masks = _parse_bool_env(os.getenv("TEXTURE_USE_MASKS", "true"), True)
    mask_dilate = int(float(os.getenv("TEXTURE_MASK_DILATE", "2")))
    weight_power = float(os.getenv("TEXTURE_WEIGHT_POWER", "2.0"))
    mode = (os.getenv("TEXTURE_MODE", "auto") or "auto").strip().lower()
    divisor = max(1, int(float(os.getenv("TEXTURE_DIVISOR", "12"))))
    manual_csv = (os.getenv("TEXTURE_MANUAL_VIEWS", "") or "").strip()

    project_root = os.path.abspath(os.path.join(mesh_dir, os.pardir))

    # Views bestimmen
    frames_dir = os.path.join(project_root, "raw_frames")
    n_frames = len(_list_frames_with_ids(frames_dir))
    if mode == "manual" and manual_csv:
        try:
            view_ids = sorted({int(x) for x in manual_csv.replace(" ", "").split(",") if x != ""})
        except Exception:
            _log(f"[texture] WARN: cannot parse TEXTURE_MANUAL_VIEWS='{manual_csv}'", on_log)
            view_ids = compute_auto_views(n_frames, divisor)
        _log(f"[texture] manual-views -> {view_ids}", on_log)
    else:
        view_ids = compute_auto_views(n_frames, divisor)
        _log(f"[texture] auto-views (div={divisor}) -> {view_ids}", on_log)

    # Texturierung
    out_path = texture_points_from_views(
        project_root=project_root,
        view_ids=view_ids,
        in_ply=in_ply,
        out_ply=out_ply,
        use_masks=use_masks,
        mask_dilate=mask_dilate,
        weight_power=weight_power,
        on_log=on_log
    )
    return out_path
