# lowlight_enhancer.py
from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Iterable, Callable, Optional

import numpy as np
import cv2 as cv
import torch
import os


# --- Zero-DCE++ Model direkt aus Datei laden ---------------------------------
def _load_zerodce_model_module():
    root = Path(__file__).resolve().parent
    candidates = [
        root / "Zero-DCE++" / "model.py",                                 # falls direkt
        root / "Zero-DCE_extension" / "Zero-DCE++" / "model.py",          # so liegt's bei dir
        root / "projects" / "damnbro" / "Zero-DCE++" / "model.py",        # fallback
    ]
    for mp in candidates:
        if mp.is_file():
            spec = importlib.util.spec_from_file_location("zerodce_model", str(mp))
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod, mp
    raise FileNotFoundError(
        "Zero-DCE++ model.py nicht gefunden in:\n" + "\n".join(str(p) for p in candidates)
    )


_ZERODCE_MOD, _ZERODCE_PATH = _load_zerodce_model_module()
enhance_net_nopool = _ZERODCE_MOD.enhance_net_nopool  # type: ignore[attr-defined]
# -----------------------------------------------------------------------------


def _to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return module.to(device)


def load_zerodcepp(weights_path: str | Path,
                   device: torch.device | None = None,
                   scale_factor: int | float | None = None):
    """
    Lädt Zero-DCE++ Netz + Gewichte. Gibt (net, device) zurück.
    - scale_factor: falls das Modell ihn verlangt (manche Varianten).
      Wenn None, wird aus LOWLIGHT_SCALE (Default '1') gelesen.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scale_factor is None:
        try:
            scale_factor = int(float(os.getenv("LOWLIGHT_SCALE", "1")))
        except Exception:
            scale_factor = 1

    # Robust gegen verschiedene Zero-DCE++-Varianten
    try:
        net = enhance_net_nopool(scale_factor=scale_factor)  # typische Signatur
    except TypeError:
        # Manche Forks brauchen keinen Parameter
        net = enhance_net_nopool()

    ckpt = torch.load(str(weights_path), map_location=device)
    state = ckpt.get("state_dict", ckpt)  # beides unterstützen
    net.load_state_dict(state, strict=False)
    net.eval()
    return _to_device(net, device), device


@torch.no_grad()
def enhance_zerodcepp(img_bgr: np.ndarray, net, device: torch.device) -> np.ndarray:
    """
    Verbessert ein einzelnes BGR-Bild (np.uint8) mit Zero-DCE++ und gibt BGR zurück.
    """
    h, w = img_bgr.shape[:2]
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    y = net(x)
    if isinstance(y, tuple):
        y = y[0]
    y = y.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    out_bgr = cv.cvtColor((y * 255.0 + 0.5).astype(np.uint8), cv.COLOR_RGB2BGR)
    if out_bgr.shape[:2] != (h, w):
        out_bgr = cv.resize(out_bgr, (w, h), interpolation=cv.INTER_LINEAR)
    return out_bgr


# --- Batch-Processing (in place) ---------------------------------------------
def enhance_list_inplace(
    img_paths: Iterable[str | Path],
    weights_path: str | Path,
    device: torch.device | None = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Liest die gegebenen Bildpfade, verbessert sie mit Zero-DCE++ und
    überschreibt sie *in place* (gleicher Name, gleicher Ort).
    """
    log = on_log or (lambda *_: None)
    img_paths = [str(p) for p in img_paths]
    if not img_paths:
        return

    net, dev = load_zerodcepp(weights_path, device=device)
    log(f"[lowlight] Zero-DCE++ loaded ({dev}), processing {len(img_paths)} frames")

    for p in img_paths:
        try:
            img = cv.imread(p, cv.IMREAD_COLOR)
            if img is None:
                log(f"[lowlight] skip (cannot read): {p}")
                continue
            out = enhance_zerodcepp(img, net, dev)
            if not cv.imwrite(p, out):  # IN PLACE
                log(f"[lowlight] failed to write: {p}")
        except Exception as e:
            log(f"[lowlight] error: {p}: {e}")

    log("[lowlight] done (in place).")


def enhance_project_raw_frames_inplace(
    project_root: str | Path,
    weights_path: str | Path,
    pattern: str = "*.png",
    device: torch.device | None = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Sucht {project_root}/raw_frames, verbessert alle Bilder in place
    und gibt die Liste der bearbeiteten Pfade zurück.
    """
    log = on_log or (lambda *_: None)
    raw_dir = Path(project_root) / "raw_frames"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_frames not found: {raw_dir}")

    img_paths = sorted(str(p) for p in raw_dir.glob(pattern))
    enhance_list_inplace(img_paths, weights_path=weights_path, device=device, on_log=log)
    return img_paths
# -----------------------------------------------------------------------------
