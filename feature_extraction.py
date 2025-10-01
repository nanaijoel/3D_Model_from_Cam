# feature_extraction.py
# Masken werden NICHT intern erzeugt.
# Vorhandene Masken werden aus out_dir/masks/<basename>_mask.png geladen
# und bei allen Backends angewendet (SIFT via detectAndCompute, learned via Post-Filter).

import os
from typing import Callable, List, Tuple, Optional

import cv2 as cv
import numpy as np
import torch


# ------------------------ kleine Helfer ------------------------

def _rootsift(des: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Wandelt SIFT-Deskriptoren in RootSIFT."""
    if des is None or len(des) == 0:
        return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)


def _kp_to_np(kps: List[cv.KeyPoint]) -> np.ndarray:
    """KeyPoints → flaches Array (zum Speichern)"""
    out = np.zeros((len(kps), 7), np.float32)
    for i, k in enumerate(kps):
        out[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
    return out


def _to_tensor(img_gray: np.ndarray) -> torch.Tensor:
    """Graubild → LightGlue-Input-Format"""
    t = torch.from_numpy(img_gray).float() / 255.0
    return t[None, None, :, :]


def _sp_to_cv_kp(sp_kp: torch.Tensor, scores: torch.Tensor) -> List[cv.KeyPoint]:
    """LightGlue-Keypoints + Scores → OpenCV-KeyPoints"""
    kps: List[cv.KeyPoint] = []
    xy = sp_kp.detach().cpu().numpy()
    sc = scores.detach().cpu().numpy()
    for (x, y), s in zip(xy, sc):
        kps.append(cv.KeyPoint(float(x), float(y), 3.0, -1.0, float(s), 0, -1))
    return kps


def _ensure_desc_is_NC(des_np: np.ndarray, n_kp: int) -> np.ndarray:
    """
    Bringt Deskriptor-Array robust auf [N, C].
    Viele LG-Backends liefern [C, N]; hier transponieren wir falls nötig.
    """
    if des_np.ndim != 2:
        return des_np
    # Falls Achse 0 wie 128/256 aussieht und Achse 1 == N: transponieren
    if des_np.shape[0] in (64, 96, 128, 256) and des_np.shape[1] == n_kp:
        return des_np.T
    # Falls Achse 0 == N: alles gut
    return des_np


def _filter_kp_des_scores(
    kps: List[cv.KeyPoint],
    des: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    dilate: int = 3
) -> Tuple[List[cv.KeyPoint], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filtert Keypoints, Deskriptoren UND Scores anhand einer 0/255-Maske.
    Für learned Backends (SuperPoint/DISK/ALIKED).
    """
    if mask is None or len(kps) == 0:
        return kps, des, scores

    if dilate > 0:
        mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate)))

    h, w = mask.shape[:2]
    keep_idx = []
    for i, kp in enumerate(kps):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep_idx.append(i)

    if not keep_idx:
        # nichts behalten
        empty_des = None
        if des is not None and des.ndim == 2:
            empty_des = np.empty((0, des.shape[1]), des.dtype)
        empty_sc = np.empty((0,), np.float32) if scores is not None else None
        return [], empty_des, empty_sc

    keep_idx = np.asarray(keep_idx, dtype=np.int32)
    kps_f = [kps[i] for i in keep_idx]
    des_f = None if des is None else des[keep_idx]
    sc_f = None if scores is None else scores[keep_idx]
    return kps_f, des_f, sc_f


# ------------------------ Hauptfunktion ------------------------

def extract_features(
    images: List[str],
    out_dir: str,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None
) -> Tuple[list, list, list, dict]:
    """
    Returns:
      keypoints:   List[List[cv.KeyPoint]]
      descriptors: List[np.ndarray] (float32 for learned, float32 RootSIFT for SIFT; Shape [N,C])
      shapes:      List[(H,W)]
      meta:        dict with backend + (sp_scores, sp_sizes) für LightGlue
    """

    def log(m: str):
        if on_log:
            on_log(m)

    def prog(p: float, s: str):
        if on_progress:
            on_progress(int(p), s)

    backend_cfg = os.getenv("FEATURE_BACKEND", "sift").lower()
    use_mask = os.getenv("FEATURE_USE_MASK", "0").lower() in ("1", "true", "yes", "on")
    max_kp = int(os.getenv("FEATURE_MAX_KP", "4096"))
    debug_every = int(os.getenv("FEATURE_DEBUG_EVERY", "0"))
    device_env = os.getenv("FEATURE_DEVICE", "").lower().strip()

    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    keypoints: List[List[cv.KeyPoint]] = []
    descriptors: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    meta = {"backend": backend_cfg, "sp_scores": [], "sp_sizes": []}

    cuda_available = torch.cuda.is_available()
    device = device_env if device_env in ("cuda", "cpu") else ("cuda" if cuda_available else "cpu")

    # --- Backends vorbereiten ---
    sp = None  # LightGlue Feature-Extractor (SuperPoint/DISK/ALIKED)
    sift = None
    lg_name = None

    backend = backend_cfg
    if backend == "superpoint":
        try:
            from lightglue import SuperPoint
            sp = SuperPoint(max_num_keypoints=max_kp).to(device).eval()
            lg_name = "superpoint"
            log(f"[features] SuperPoint (lightglue) on {device} (cuda_available={cuda_available}), max_kp={max_kp}")
        except Exception as e:
            log(f"[features] WARN: SuperPoint init failed: {e}. Falling back to SIFT.")
            backend = "sift"

    elif backend == "disk":
        try:
            from lightglue import DISK
            sp = DISK(max_num_keypoints=max_kp).to(device).eval()
            lg_name = "disk"
            backend = "superpoint"  # gemeinsamer Pfad unten
            log(f"[features] DISK (lightglue) on {device}, max_kp={max_kp}")
        except Exception as e:
            log(f"[features] WARN: DISK init failed: {e}. Falling back to SIFT.")
            backend = "sift"

    elif backend in ("aliked", "alike", "aliked-lightglue"):
        try:
            from lightglue import ALIKED
            sp = ALIKED(max_num_keypoints=max_kp).to(device).eval()
            lg_name = "aliked"
            backend = "superpoint"  # gemeinsamer Pfad unten
            log(f"[features] ALIKED (lightglue) on {device}, max_kp={max_kp}")
        except Exception as e:
            log(f"[features] WARN: ALIKED init failed: {e}. Falling back to SIFT.")
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

    meta["lg_feature_name"] = lg_name

    # --- Hauptschleife ---
    N = len(images)
    for i, path in enumerate(images):
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)
        if img_bgr is None:
            log(f"[features] WARN: cannot read {path}")
            keypoints.append([])
            descriptors.append(np.empty((0, 128), np.float32))
            shapes.append((0, 0))
            continue

        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        # Maske laden (falls aktiviert und vorhanden)
        mask = None
        if use_mask:
            mpath = os.path.join(mask_dir, os.path.splitext(os.path.basename(path))[0] + "_mask.png")
            if os.path.isfile(mpath):
                mask = cv.imread(mpath, cv.IMREAD_GRAYSCALE)

        if backend == "superpoint" and sp is not None:
            # ---- Learned Features (SP/DISK/ALIKED via LightGlue) ----
            with torch.no_grad():
                tin = _to_tensor(gray).to(device)
                out = sp({"image": tin})

                kp_t = out["keypoints"][0]             # [N, 2]
                sc_t = out.get("scores", None)
                if sc_t is None:
                    sc_t = out.get("keypoint_scores", None)
                if sc_t is None:
                    raise KeyError("LightGlue features: 'scores' or 'keypoint_scores' missing.")
                sc_t = sc_t[0]                          # [N]

                desc_t = out["descriptors"][0]         # [C, N] ODER [N, C]
                img_size_t = out.get("image_size", None)
                if img_size_t is not None:
                    H, W = int(img_size_t[0, 0].item()), int(img_size_t[0, 1].item())
                else:
                    H, W = gray.shape[:2]

                if kp_t.numel() == 0:
                    kps: List[cv.KeyPoint] = []
                    des_np = np.empty((0, 256), np.float32)
                    sc_np = np.empty((0,), np.float32)
                else:
                    # Tensor → numpy
                    kps = _sp_to_cv_kp(kp_t, sc_t)
                    des_np = desc_t.detach().cpu().numpy().astype(np.float32)
                    sc_np = sc_t.detach().cpu().numpy().astype(np.float32)

                    # ---- WICHTIG: Deskriptoren auf [N, C] bringen ----
                    des_np = _ensure_desc_is_NC(des_np, n_kp=len(kps))

                    # ---- Maskenfilter anwenden (KP, Des, Scores gemeinsam) ----
                    kps, des_np, sc_np = _filter_kp_des_scores(kps, des_np, sc_np, mask, dilate=10)

                keypoints.append(kps)
                descriptors.append(des_np)
                shapes.append(gray.shape)
                meta["sp_scores"].append(sc_np if len(kps) else np.empty((0,), np.float32))
                meta["sp_sizes"].append((H, W))

        else:
            # ---- SIFT (RootSIFT) ----
            kps, des = sift.detectAndCompute(gray, mask)
            des = _rootsift(des)
            keypoints.append(kps)
            descriptors.append(des if des is not None else np.empty((0, 128), np.float32))
            shapes.append(gray.shape)

        # Debug-Overlay
        if debug_every and (i % debug_every == 0):
            dbg = cv.drawKeypoints(img_bgr, keypoints[-1], None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if mask is not None:
                dbg = cv.addWeighted(dbg, 0.85, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.15, 0)
            cv.imwrite(os.path.join(out_dir, f"debug_{i:04d}.jpg"), dbg)

        if N:
            prog(30 + (i + 1) / max(1, N) * 10, f"Feature Extraction ({lg_name or backend})")

        # Persistenter Dump (für spätere Schritte)
        np.savez(
            os.path.join(out_dir, f"features_{i:04d}.npz"),
            kps=_kp_to_np(keypoints[-1]),
            des=descriptors[-1],
            shape=np.array(shapes[-1], dtype=np.int32)
        )

    total_kp = sum(len(k) for k in keypoints)
    log(f"[features] done: {total_kp} keypoints | backend={lg_name or backend} | "
        f"torch={torch.__version__} cuda_avail={cuda_available}")

    # Für SIFT braucht Matching diese Extras nicht
    if lg_name is None:
        meta["sp_scores"] = None
        meta["sp_sizes"] = None

    return keypoints, descriptors, shapes, meta
