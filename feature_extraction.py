import os
from typing import Callable, List, Tuple, Optional

import cv2 as cv
import numpy as np
import torch

# helpers

def _rootsift(des):
    if des is None or len(des) == 0:
        return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)

def _kp_to_np(kps):
    out = np.zeros((len(kps), 7), np.float32)
    for i, k in enumerate(kps):
        out[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
    return out

# SuperPoint glue

def _to_tensor(img_gray: np.ndarray) -> torch.Tensor:
    # uint8 [H,W] -> float [1,1,H,W] in [0,1]
    t = torch.from_numpy(img_gray).float() / 255.0
    return t[None, None, :, :]

def _sp_to_cv_kp(sp_kp: torch.Tensor, scores: torch.Tensor) -> List[cv.KeyPoint]:
    kps = []
    xy = sp_kp.detach().cpu().numpy()
    sc = scores.detach().cpu().numpy()
    for (x, y), s in zip(xy, sc):
        # cv.KeyPoint(x, y, size, angle, response, octave, class_id)
        kps.append(cv.KeyPoint(float(x), float(y), 3.0, -1.0, float(s), 0, -1))
    return kps

# API: extract

def extract_features(
    images: List[str],
    out_dir: str,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None
) -> Tuple[list, list, list, dict]:
    """
    Returns:
      keypoints: List[List[cv.KeyPoint]]
      descriptors: List[np.ndarray] (float32 for learned, float32 RootSIFT for SIFT)
      shapes: List[(H,W)]
      meta: dict with backend + optional tensors for LightGlue (keypoint scores, image_size)
    Backend is chosen via env FEATURE_BACKEND = 'superpoint' | 'sift'
    """
    def log(m):  on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(int(p), s)

    backend = os.getenv("FEATURE_BACKEND", "sift").lower()
    use_mask = os.getenv("FEATURE_USE_MASK", "0").lower() in ("1", "true", "yes", "on")
    max_kp = int(os.getenv("FEATURE_MAX_KP", "4096"))
    debug_every = int(os.getenv("FEATURE_DEBUG_EVERY", "0"))
    device_env = os.getenv("FEATURE_DEVICE", "").lower().strip()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    keypoints, descriptors, shapes = [], [], []
    meta = {"backend": backend, "sp_scores": [], "sp_sizes": []}

    cuda_available = torch.cuda.is_available()
    if device_env in ("cuda", "cpu"):
        device = device_env
    else:
        device = "cuda" if cuda_available else "cpu"

    # Optional, simple foreground mask
    def _make_statue_mask(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        H, S, V = cv.split(hsv)
        white_like = (S.astype(np.float32) <= 35) & (V.astype(np.float32) >= 200)
        lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        very_bright = L >= 235
        fg = ~(white_like | very_bright)
        fg = fg.astype(np.uint8) * 255
        k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, k)
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, k, iterations=2)
        return fg

    # choose extractor
    sp = None
    if backend == "superpoint":
        try:
            from lightglue import SuperPoint  # stable SP impl that matches LightGlue expectations
            sp = SuperPoint(max_num_keypoints=max_kp).to(device).eval()
            log(f"[features] SuperPoint (lightglue) on {device} "
                f"(cuda_available={cuda_available}), max_kp={max_kp}")
        except Exception as e:
            log(f"[features] WARN: SuperPoint (lightglue) failed: {e}. Falling back to SIFT.")
            backend = "sift"

    if backend == "sift":
        sift = cv.SIFT_create(
            nfeatures=max_kp,
            nOctaveLayers=3,
            contrastThreshold=0.02,
            edgeThreshold=14,
            sigma=1.2
        )
        log(f"[features] SIFT (RootSIFT), max_kp={max_kp}")

    N = len(images)
    for i, path in enumerate(images):
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)
        if img_bgr is None:
            log(f"[features] WARN: cannot read {path}")
            keypoints.append([]); descriptors.append(None); shapes.append((0, 0))
            continue

        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        mask = _make_statue_mask(img_bgr) if use_mask else None

        if backend == "superpoint" and sp is not None:
            with torch.no_grad():
                tin = _to_tensor(gray).to(device)
                out = sp({"image": tin})
                # Possible key name differences between versions:
                # 'scores' (lightglue>=0.0) vs 'keypoint_scores' (some variants)
                kp_t = out["keypoints"][0]                           # [N,2] (x,y)
                sc_t = out.get("scores", None)
                if sc_t is None:
                    sc_t = out.get("keypoint_scores", None)
                if sc_t is None:
                    raise KeyError("SuperPoint output has neither 'scores' nor 'keypoint_scores'.")
                sc_t = sc_t[0]                                       # [N]

                desc_t = out["descriptors"][0].transpose(0, 1)       # [N,C]
                # image_size is [B,2] with [H, W]
                img_size_t = out.get("image_size", None)
                if img_size_t is not None:
                    H, W = int(img_size_t[0, 0].item()), int(img_size_t[0, 1].item())
                else:
                    H, W = gray.shape[:2]

                if kp_t.numel() == 0:
                    kps, des = [], np.empty((0, 256), np.float32)
                    sc_np = np.empty((0,), np.float32)
                else:
                    kps = _sp_to_cv_kp(kp_t, sc_t)
                    des = desc_t.detach().cpu().numpy().astype(np.float32)
                    sc_np = sc_t.detach().cpu().numpy().astype(np.float32)

                keypoints.append(kps)
                descriptors.append(des)
                shapes.append(gray.shape)

                meta["sp_scores"].append(sc_np)
                meta["sp_sizes"].append((H, W))
        else:
            kps, des = sift.detectAndCompute(gray, mask)
            des = _rootsift(des)
            keypoints.append(kps)
            descriptors.append(des if des is not None else np.empty((0, 128), np.float32))
            shapes.append(gray.shape)

        if debug_every and (i % debug_every == 0):
            dbg = cv.drawKeypoints(img_bgr, keypoints[-1], None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if mask is not None:
                dbg = cv.addWeighted(dbg, 0.85, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.15, 0)
            cv.imwrite(os.path.join(out_dir, f"debug_{i:04d}.jpg"), dbg)

        if N:
            prog(30 + (i + 1) / max(1, N) * 10, f"Feature Extraction ({backend})")

        # persist per-frame
        np.savez(
            os.path.join(out_dir, f"features_{i:04d}.npz"),
            kps=_kp_to_np(keypoints[-1]),
            des=descriptors[-1],
            shape=np.array(shapes[-1], dtype=np.int32)
        )

    total_kp = sum(len(k) for k in keypoints)
    log(f"[features] done: {total_kp} keypoints | backend={backend} | "
        f"torch={torch.__version__} cuda_avail={cuda_available}")

    # meta consistency for classic backend
    if backend != "superpoint":
        meta["sp_scores"] = None
        meta["sp_sizes"] = None

    return keypoints, descriptors, shapes, meta
