# image_masking.py (generisch, ohne projektspezifische Heuristiken)
import os
from typing import List, Tuple, Optional, Dict
import cv2 as cv
import numpy as np

Mask = np.ndarray  # uint8, 0/255

# ----------------- Helpers -----------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _largest_cc(mask: np.ndarray) -> np.ndarray:
    num, labels = cv.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:  # nur Hintergrund
        return mask
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0  # Hintergrund ignorieren
    keep = int(np.argmax(counts))
    return ((labels == keep).astype(np.uint8) * 255)

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    inv = cv.bitwise_not(mask)
    flood = inv.copy()
    ffmask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(flood, ffmask, (0, 0), 255)
    flood_inv = cv.bitwise_not(flood)
    # Löcher = flood_inv & ~inv == flood_inv & mask
    return cv.bitwise_or(mask, cv.bitwise_and(flood_inv, mask))

def _postprocess_mask(m: Mask, open_k: int = 3, close_k: int = 5,
                      keep_largest: bool = True, fill_holes: bool = True) -> Mask:
    m = (m > 0).astype(np.uint8) * 255
    if open_k:
        m = cv.morphologyEx(m, cv.MORPH_OPEN,
                            cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k:
        m = cv.morphologyEx(m, cv.MORPH_CLOSE,
                            cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k)))
    if fill_holes:
        m = _fill_holes(m)
    if keep_largest:
        m = _largest_cc(m)
    return m

# ----------------- rembg backend -----------------

_REMBG_SESS = None

def _get_rembg_session(model_name: str = "isnet-general"):
    """Lädt/mergt eine rembg-Session. Fallback auf 'u2net'."""
    global _REMBG_SESS
    if _REMBG_SESS is not None:
        return _REMBG_SESS
    try:
        from rembg import new_session
        try:
            _REMBG_SESS = new_session(model_name)
        except Exception:
            _REMBG_SESS = new_session("u2net")
    except Exception:
        _REMBG_SESS = None
    return _REMBG_SESS

def _rembg_mask(
    img_bgr: np.ndarray,
    alpha_thr: float = 0.5,
    model_name: str = "isnet-general",
    alpha_matting: bool = True,
    fg_thr: int = 210,              # 0..255
    bg_thr: int = 20,               # 0..255
    erode: int = 8,                 # 0..∞ (px)
    base_size: int = 1000,          # für alpha_matting
    only_mask: bool = False,        # wenn True: rembg liefert direkt die Binärmaske
    post_process_mask_flag: bool = False  # rembg-eigener Postprozess (abhängig von Version)
) -> Mask:
    """
    Verwendet rembg.remove mit erweiterten Parametern. Gibt Binärmaske (0/255) zurück.
    """
    try:
        from rembg import remove
        from PIL import Image
    except Exception as e:
        print(f"[mask][rembg] not available: {e}")
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    session = _get_rembg_session(model_name)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil_in = Image.fromarray(img_rgb)

    kwargs = dict(
        session=session,
        alpha_matting=bool(alpha_matting),
        alpha_matting_foreground_threshold=int(fg_thr),
        alpha_matting_background_threshold=int(bg_thr),
        alpha_matting_erode_size=int(erode),
        alpha_matting_base_size=int(base_size),
        only_mask=bool(only_mask),
        post_process_mask=bool(post_process_mask_flag),
        bgcolor=None
    )

    try:
        pil_out = remove(pil_in, **kwargs)
    except TypeError:
        # ältere rembg-Versionen kennen evtl. nicht alle Keys → minimaler Fallback
        kwargs_fallback = dict(
            session=session,
            alpha_matting=bool(alpha_matting),
            alpha_matting_foreground_threshold=int(fg_thr),
            alpha_matting_background_threshold=int(bg_thr),
            alpha_matting_erode_size=int(erode),
            bgcolor=None
        )
        pil_out = remove(pil_in, **kwargs_fallback)
    except Exception as e:
        print(f"[mask][rembg] remove failed: {e}")
        return np.full(img_bgr.shape[:2], 255, np.uint8)

    out_np = np.array(pil_out)

    # Fall A: only_mask=True → out_np ist bereits 1-Kanal-Maske (0/255 oder 0/1)
    if only_mask and out_np.ndim == 2:
        m = out_np
        if m.dtype != np.uint8:
            m = (m > 0).astype(np.uint8) * 255
        else:
            # falls 0/1
            if m.max() <= 1:
                m = (m > 0).astype(np.uint8) * 255
        return _postprocess_mask(m, open_k=3, close_k=5)

    # Fall B: RGBA → Alpha in Binärmaske
    if out_np.ndim == 3 and out_np.shape[2] == 4:
        alpha = out_np[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img_bgr.shape[:2], np.float32)

    m = (alpha >= float(alpha_thr)).astype(np.uint8) * 255
    return _postprocess_mask(m, open_k=3, close_k=5)

# ----------------- SAM 2 (optional) -----------------

_SAM2_CACHE = {"predictor": None}

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
            raise FileNotFoundError("Set env SAM2_CKPT to a valid checkpoint path (e.g. checkpoints/sam2_hiera_tiny.pt).")
        model = build_sam2(ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)
        _SAM2_CACHE["predictor"] = predictor
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
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=None, multimask_output=True)
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

# ----------------- Public API -----------------

def build_mask(
    img_bgr: np.ndarray,
    method: str = "rembg",
    params: Optional[Dict] = None
) -> Mask:
    """
    method: 'rembg' | 'sam2'
    params (rembg):
      - model_name: 'isnet-general' | 'u2net' | 'u2net_human_seg' ...
      - alpha_thr: 0..1 (Binär-Schwelle auf Alpha)
      - alpha_matting: bool
      - fg_thr, bg_thr: 0..255
      - erode: int px
      - base_size: int
      - only_mask: bool
      - post_process_mask: bool
    params (sam2):
      - keep: 'largest' | 'best_score'
    """
    params = params or {}
    if method == "sam2":
        return _sam2_mask(img_bgr, keep=str(params.get("keep", "largest")))
    # default: rembg mit erweiterten Optionen
    return _rembg_mask(
        img_bgr,
        alpha_thr=float(params.get("alpha_thr", 0.5)),
        model_name=str(params.get("model_name", "isnet-general")),
        alpha_matting=bool(params.get("alpha_matting", True)),
        fg_thr=int(params.get("fg_thr", 210)),
        bg_thr=int(params.get("bg_thr", 20)),
        erode=int(params.get("erode", 8)),
        base_size=int(params.get("base_size", 1000)),
        only_mask=bool(params.get("only_mask", False)),
        post_process_mask_flag=bool(params.get("post_process_mask", False)),
    )

def preprocess_images(
    images: List[str],
    out_mask_dir: str,
    overwrite_images: bool = False,
    method: str = "rembg",
    params: Optional[Dict] = None,
    save_debug: bool = True
) -> Tuple[List[str], List[str]]:
    """
    schreibt <basename>_mask.png in out_mask_dir
    """
    _ensure_dir(out_mask_dir)
    mask_paths: List[str] = []

    for p in images:
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None:
            mask_paths.append("")
            continue

        m = build_mask(img, method=method, params=params)

        if overwrite_images:
            fg = cv.bitwise_and(img, img, mask=m)
            bg = cv.bitwise_and(img, img, mask=cv.bitwise_not(m))
            alpha = float((params or {}).get("fg_alpha", 1.12))
            beta = float((params or {}).get("fg_beta", 6))
            fg = cv.convertScaleAbs(fg, alpha=alpha, beta=beta)
            merged = cv.add(fg, bg)
            cv.imwrite(p, merged)

        base = os.path.splitext(os.path.basename(p))[0]
        mpath = os.path.join(out_mask_dir, f"{base}_mask.png")
        cv.imwrite(mpath, m)
        mask_paths.append(mpath)

        if save_debug:
            dbg = img.copy()
            cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(dbg, cnts, -1, (0, 255, 0), 2)
            overlay = cv.addWeighted(dbg, 0.85, cv.cvtColor(m, cv.COLOR_GRAY2BGR), 0.15, 0)
            cv.imwrite(os.path.join(out_mask_dir, f"{base}_dbg.jpg"), overlay)

    return images, mask_paths
