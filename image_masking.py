# image_masking.py
import os
from typing import List, Tuple, Optional, Dict

import cv2 as cv
import numpy as np

Mask = np.ndarray  # uint8, 0/255

# --- Optionale Backends:
# A) rembg (U^2-Net) -> pip install rembg
# B) SAM 2           -> pip install git+https://github.com/facebookresearch/sam2.git
#                      und ein Checkpoint, Pfad via env SAM2_CKPT (z.B. checkpoints/sam2_hiera_tiny.pt)
# Hinweis: Wenn Backend fehlt, liefern wir eine "volle" Maske (255), damit die Pipeline nicht bricht.

# -------- Utils --------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _postprocess_mask(m: Mask, open_k: int = 3, close_k: int = 5) -> Mask:
    m = (m > 0).astype(np.uint8) * 255
    if open_k:
        m = cv.morphologyEx(m, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k:
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k)))
    return m

# -------- rembg (U^2-Net) --------
def _rembg_mask(img_bgr: np.ndarray, alpha_thr: float = 0.5) -> Mask:
    try:
        from rembg import remove
        from PIL import Image
    except Exception as e:
        print(f"[mask][rembg] not available: {e}")
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil_in = Image.fromarray(img_rgb)
    try:
        pil_out = remove(pil_in)  # RGBA Image mit Alpha-Matte
    except Exception as e:
        print(f"[mask][rembg] remove failed: {e}")
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    out_np = np.array(pil_out)
    if out_np.ndim == 3 and out_np.shape[2] == 4:
        alpha = out_np[:, :, 3].astype(np.float32) / 255.0
    else:
        # Falls kein Alpha geliefert wird, alles FG
        alpha = np.ones(img_bgr.shape[:2], np.float32)

    m = (alpha >= float(alpha_thr)).astype(np.uint8) * 255
    return _postprocess_mask(m, open_k=3, close_k=5)

# -------- SAM 2 --------
_SAM2_CACHE = {"predictor": None, "device": "cpu"}

def _get_sam2_predictor() -> Optional[object]:
    if _SAM2_CACHE["predictor"] is not None:
        return _SAM2_CACHE["predictor"]
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt_path = os.environ.get("SAM2_CKPT", "").strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Set env SAM2_CKPT to a valid checkpoint path (e.g. sam2_hiera_tiny.pt).")
        model = build_sam2(ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)
        _SAM2_CACHE["predictor"] = predictor
        _SAM2_CACHE["device"] = device
        return predictor
    except Exception as e:
        print(f"[mask][sam2] init failed: {e}")
        return None

def _sam2_mask(img_bgr: np.ndarray, keep: str = "largest") -> Mask:
    predictor = _get_sam2_predictor()
    if predictor is None:
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    try:
        predictor.set_image(img_rgb)
        # Ohne Prompt: mehrere Kandidatenmasken; wir wählen genau EINE
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=None, multimask_output=True
        )
    except Exception as e:
        print(f"[mask][sam2] predict failed: {e}")
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    if masks is None or len(masks) == 0:
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    masks_np = np.asarray(masks).astype(np.uint8)  # [K,H,W]
    if keep == "best_score" and scores is not None and len(scores) == len(masks_np):
        idx = int(np.argmax(np.asarray(scores)))
    else:
        areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
    m = (masks_np[idx] > 0).astype(np.uint8) * 255
    return _postprocess_mask(m, open_k=2, close_k=4)

# -------- Public API --------
def build_mask(img_bgr: np.ndarray, method: str = "rembg", params: Optional[Dict] = None) -> Mask:
    """
    method: 'rembg' | 'sam2'
    Genau EINE finale Binärmaske (0/255) pro Bild.
    """
    params = params or {}
    if method == "rembg":
        return _rembg_mask(img_bgr, alpha_thr=float(params.get("alpha_thr", 0.5)))
    if method == "sam2":
        return _sam2_mask(img_bgr, keep=str(params.get("keep", "largest")))
    # Fallback: keine Maske
    h, w = img_bgr.shape[:2]
    return np.full((h, w), 255, np.uint8)

def preprocess_images(
    images: List[str],
    out_mask_dir: str,
    overwrite_images: bool = False,
    method: str = "rembg",
    params: Optional[Dict] = None,
    save_debug: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Erzeugt pro Bild genau EINE Maske:
      projects/<name>/features/masks/<basename>_mask.png
    Optional: schreibt FG-boosted Bild zurück (overwrite_images=True).
    """
    _ensure_dir(out_mask_dir)
    mask_paths: List[str] = []

    for p in images:
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None:
            mask_paths.append("")
            continue

        m = build_mask(img, method=method, params=params)

        # optional: FG leicht boosten – NICHT notwendig für FE-Filtering
        if overwrite_images:
            fg = cv.bitwise_and(img, img, mask=m)
            bg = cv.bitwise_and(img, img, mask=cv.bitwise_not(m))
            alpha = float((params or {}).get("fg_alpha", 1.15))
            beta = float((params or {}).get("fg_beta", 6))
            fg = cv.convertScaleAbs(fg, alpha=alpha, beta=beta)
            merged = cv.add(fg, bg)
            cv.imwrite(p, merged)  # gleicher Ort & Name

        # Maske speichern (genau eine)
        base = os.path.splitext(os.path.basename(p))[0]
        mpath = os.path.join(out_mask_dir, f"{base}_mask.png")
        cv.imwrite(mpath, m)
        mask_paths.append(mpath)

        # Debug-Overlay optional
        if save_debug:
            dbg = img.copy()
            cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(dbg, cnts, -1, (0, 255, 0), 2)
            overlay = cv.addWeighted(dbg, 0.85, cv.cvtColor(m, cv.COLOR_GRAY2BGR), 0.15, 0)
            dbgp = os.path.join(out_mask_dir, f"{base}_dbg.jpg")
            cv.imwrite(dbgp, overlay)

    return images, mask_paths
