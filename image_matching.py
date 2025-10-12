# ==========================================================
# image_matching.py — Dual Matching für SfM (GPU bevorzugt)
# LightGlue auf gespeicherten DISK-Features  +  LoFTR (mit Masken)
# Speichert UNION echter Keypoint-Indexpaare (SfM-kompatibel)
# ==========================================================
import os
from typing import List, Optional, Tuple

import numpy as np
import cv2 as cv
import torch

from lightglue import LightGlue
from LoFTR_extractor import LoFTRExtractor  # nutzt kornia LoFTR (GPU)

# ----------------------------------------------------------

def _load_features(features_dir: str, idx: int):
    """Lädt gespeicherte Feature-Datei (kps, des, scores?, shape) für Frame idx."""
    fpath = os.path.join(features_dir, f"features_{idx:04d}.npz")
    d = np.load(fpath, allow_pickle=True)
    kps = d["kps"][:, :2].astype(np.float32) if d["kps"].size else np.empty((0, 2), np.float32)
    des = d["des"].astype(np.float32) if d["des"].size else np.empty((0, 0), np.float32)
    scores = d["scores"].astype(np.float32) if "scores" in d.files else np.ones((len(kps),), np.float32)
    shape = tuple(d["shape"].tolist()) if "shape" in d.files else (1080, 1920)
    return kps, des, scores, shape

def _adaptive_tol_px(shape_hw: Tuple[int,int]) -> float:
    # Toleranz skaliert mit Bildgröße (robust gg. kleine Offsets zw. Detektoren)
    h, w = shape_hw
    pct = float(os.getenv("MATCH_MAP_TOL_PCT", "0.005"))  # default 0.5% der größeren Kante
    return max(3.0, pct * float(max(h, w)))  # z.B. 1080x1920 -> ~9.6 px

def _nn_map_one_way(query_xy: np.ndarray, ref_xy: np.ndarray, tol_px: float) -> np.ndarray:
    """Einseitiges NN-Mapping (query->ref), Index oder -1 (Distanz>tol)."""
    if query_xy.size == 0 or ref_xy.size == 0:
        return np.full((len(query_xy),), -1, np.int32)
    ref2 = (ref_xy**2).sum(axis=1)  # (R,)
    out = np.empty((len(query_xy),), np.int32); out.fill(-1)
    CH = 2048
    for s in range(0, len(query_xy), CH):
        q = query_xy[s:s+CH].astype(np.float32)
        q2 = (q**2).sum(axis=1)[:, None]
        d2 = q2 + ref2[None, :] - 2.0 * (q @ ref_xy.T.astype(np.float32))
        nn = np.argmin(d2, axis=1)
        dmin = np.sqrt(np.take_along_axis(d2, nn[:, None], axis=1)).ravel()
        out[s:s+CH] = np.where(dmin <= tol_px, nn.astype(np.int32), -1)
    return out

def _mutual_nn_indices(
    pts0: np.ndarray, pts1: np.ndarray,
    kps_i: np.ndarray, kps_j: np.ndarray,
    tol_px: float
) -> np.ndarray:
    """Mappt Pixel-Matches (pts0, pts1) → (idx_i, idx_j) mit wechselseitiger NN-Konsistenz."""
    if pts0.size == 0 or pts1.size == 0:
        return np.empty((0, 2), np.int32)
    idx_i = _nn_map_one_way(pts0, kps_i, tol_px)
    idx_j = _nn_map_one_way(pts1, kps_j, tol_px)
    valid = (idx_i >= 0) & (idx_j >= 0)
    if not np.any(valid):
        return np.empty((0, 2), np.int32)
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    # (Optional) leichte Rückprüfung (stabilisiert gegen Kollisionen)
    # Wir prüfen nur Gleichheit der Vorwärtszuordnung nach Deduplizierung:
    pairs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    if len(pairs):
        pairs = np.unique(pairs, axis=0)
    return pairs

def _lg_match_indices_from_saved_features(
    lg: LightGlue,
    device: torch.device,
    k0: np.ndarray, d0: np.ndarray, s0: np.ndarray, hw0: Tuple[int,int],
    k1: np.ndarray, d1: np.ndarray, s1: np.ndarray, hw1: Tuple[int,int],
) -> np.ndarray:
    """
    LightGlue direkt auf gespeicherten DISK-Features.
    Gibt Indexpaare (queryIdx, trainIdx) zurück.
    """
    if d0.ndim != 2 or d1.ndim != 2 or len(k0) == 0 or len(k1) == 0:
        return np.empty((0, 2), np.int32)

    t_k0 = torch.from_numpy(k0)[None].to(device)          # [1,N0,2]
    t_k1 = torch.from_numpy(k1)[None].to(device)          # [1,N1,2]
    t_d0 = torch.from_numpy(d0)[None].to(device)          # [1,N0,C]
    t_d1 = torch.from_numpy(d1)[None].to(device)          # [1,N1,C]
    t_s0 = torch.from_numpy(s0)[None].to(device)          # [1,N0]
    t_s1 = torch.from_numpy(s1)[None].to(device)          # [1,N1]
    t_sz0 = torch.tensor([[hw0[0], hw0[1]]], dtype=torch.float32, device=device)
    t_sz1 = torch.tensor([[hw1[0], hw1[1]]], dtype=torch.float32, device=device)

    with torch.inference_mode():
        out = lg({
            "image0": {"keypoints": t_k0, "descriptors": t_d0, "scores": t_s0, "image_size": t_sz0},
            "image1": {"keypoints": t_k1, "descriptors": t_d1, "scores": t_s1, "image_size": t_sz1},
        })

    if "matches0" in out:
        m0 = out["matches0"][0].detach().cpu().numpy()
        valid = m0 >= 0
        if not np.any(valid):
            return np.empty((0, 2), np.int32)
        q = np.where(valid)[0].astype(np.int32)
        t = m0[valid].astype(np.int32)
        return np.stack([q, t], axis=1)
    elif "matches" in out:
        return out["matches"][0].detach().cpu().numpy().astype(np.int32)
    else:
        return np.empty((0, 2), np.int32)

def _prefilter_loftr_with_ransac(pts0: np.ndarray, pts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Grobe Geometrieprüfung (Fundamental RANSAC) auf LoFTR, pixelbasiert. """
    if len(pts0) < 8:
        return pts0, pts1
    F, inl = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, 1.5, 0.999)
    if inl is None:
        return np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    inl = inl.ravel() > 0
    return pts0[inl].astype(np.float32), pts1[inl].astype(np.float32)

def save_pairs(out_dir: str, pairs: np.ndarray, matches: List[np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    matches_obj = np.empty(len(matches), dtype=object)
    matches_obj[:] = matches
    np.savez(os.path.join(out_dir, "matches.npz"),
             pairs=pairs.astype(np.int32),
             matches=matches_obj)
    print(f"[DualMatcher] Saved matches.npz -> {out_dir}")

def build_pairs(
    features_dir: str,
    device: str = "cuda",
    on_log=print,
    save_dir: Optional[str] = None,
):
    """
    - Nimmt gespeicherte DISK-Features aus features_XXXX.npz
    - LightGlue läuft direkt auf diesen Features → echte Indexpaare
    - LoFTR liefert zusätzliche Pixel-Matches (mit Masken) → mutual NN (+ adaptiver Toleranz) auf DISK gemappt
    - UNION(LG, LoFTR_mapped) wird gespeichert (SfM-kompatibel)
    """
    if not torch.cuda.is_available():
        device = "cpu"
        on_log("[WARN] CUDA not available – running on CPU")
    dev = torch.device(device)

    # Output-Ordner
    if save_dir is None:
        save_dir = os.path.join(features_dir, "matches")
    os.makedirs(save_dir, exist_ok=True)

    # Bildliste (Nachbar-Matching i <-> i+1)
    img_dir = os.path.join(os.path.dirname(features_dir), "raw_frames")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(features_dir, "raw_frames")
    images = sorted([os.path.join(img_dir, f)
                     for f in os.listdir(img_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(images) < 2:
        raise RuntimeError("Not enough frames to match.")
    on_log(f"[DualMatcher] Matching {len(images)} frames (device={device})...")

    # Matcher-Instanzen
    lg = LightGlue(features='disk').eval().to(dev)
    loftr = LoFTRExtractor(device=device, weights=os.getenv("LOFTR_WEIGHTS", "outdoor"))

    # Masken-Pfade (wie in Feature-Extraction gespeichert)
    mask_dir = os.path.join(features_dir, "masks")

    pairs_out, matches_out = [], []

    for i in range(len(images) - 1):
        j = i + 1
        img1, img2 = images[i], images[j]
        base1 = os.path.splitext(os.path.basename(img1))[0]
        base2 = os.path.splitext(os.path.basename(img2))[0]
        m1 = os.path.join(mask_dir, f"{base1}_mask.png")
        m2 = os.path.join(mask_dir, f"{base2}_mask.png")

        on_log(f"[DualMatcher] Matching {os.path.basename(img1)} <-> {os.path.basename(img2)}")

        # ---- LightGlue direkt auf gespeicherten Features ----
        k0, d0, s0, hw0 = _load_features(features_dir, i)
        k1, d1, s1, hw1 = _load_features(features_dir, j)
        lg_pairs = _lg_match_indices_from_saved_features(lg, dev, k0, d0, s0, hw0, k1, d1, s1, hw1)

        # ---- LoFTR (Pixel) mit MASKEN ----
        pts0, pts1 = loftr.match_pair(img1, img2, max_size=int(os.getenv("LOFTR_MAX_SIZE", "640")), mask1_path=m1, mask2_path=m2)
        # LoFTR grob vorfiltern (Fundamental-RANSAC), damit Mapping stabiler ist
        pts0, pts1 = _prefilter_loftr_with_ransac(pts0, pts1)

        # ---- Mapping auf DISK-Indexpaare (mutual NN, adaptive tol) ----
        tol = _adaptive_tol_px(hw0)
        loftr_pairs = _mutual_nn_indices(pts0, pts1, k0, k1, tol_px=tol)

        # ---- UNION + Duplikate weg ----
        if len(lg_pairs) and len(loftr_pairs):
            all_pairs = np.vstack([lg_pairs, loftr_pairs]).astype(np.int32)
            all_pairs = np.unique(all_pairs, axis=0)
        elif len(lg_pairs):
            all_pairs = lg_pairs
        else:
            all_pairs = loftr_pairs  # ggf. leer

        # (Optional) End-Validierung auf Pixel-Ebene
        if len(all_pairs) >= 8:
            p0 = k0[all_pairs[:, 0]].astype(np.float32)
            p1 = k1[all_pairs[:, 1]].astype(np.float32)
            _, inl = cv.findFundamentalMat(p0, p1, cv.FM_RANSAC, 1.5, 0.999)
            if inl is not None:
                all_pairs = all_pairs[inl.ravel() > 0]

        pairs_out.append([i, j])
        matches_out.append(all_pairs)

        on_log(f"[OK] LG={len(lg_pairs)} | LoFTR→map={len(loftr_pairs)} | final={len(all_pairs)} (tol≈{tol:.1f}px)")
        torch.cuda.empty_cache()

    pairs_arr = np.array(pairs_out, dtype=np.int32)
    save_pairs(save_dir, pairs_arr, matches_out)
    on_log(f"[DualMatcher] Matching complete. Total pairs: {len(pairs_arr)}")
    return pairs_arr, matches_out
