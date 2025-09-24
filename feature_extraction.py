import os, cv2 as cv, numpy as np
from typing import Callable, List, Tuple

# -------------------- helpers --------------------

def _rootsift(des):
    if des is None or len(des) == 0:
        return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    des = np.sqrt(des)
    return des

def _kp_to_np(kps):
    out = np.zeros((len(kps), 7), np.float32)
    for i, k in enumerate(kps):
        out[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
    return out

# -------------------- preprocessing --------------------

def _homomorphic_brightfix(img_gray: np.ndarray) -> np.ndarray:
    """Homomorphic/Retinex-ähnlich: Beleuchtung glätten, Highlights komprimieren, sanft schärfen."""
    f = img_gray.astype(np.float32) / 255.0
    h, w = f.shape
    sigma = 0.03 * max(h, w)  # ~3% der langen Kante
    base = cv.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)

    eps = 1e-6
    r = np.log(f + eps) - np.log(base + eps)  # high-pass in log-domain
    # Clipping gegen Ausreißer
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    r = np.clip(r, lo, hi)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)

    # Highlights-Kompression (Gamma>1 dunkelt Highlights ab)
    r = np.power(r, 1.6)

    g = (r * 255.0).astype(np.uint8)
    # sanfte Schärfung
    blur = cv.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv.addWeighted(g, 1.4, blur, -0.4, 0)

    # leichte lokale Kontrastanhebung
    clahe = cv.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    out = clahe.apply(sharp)
    return out

# -------------------- adaptive foreground mask --------------------

def _make_statue_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Ziel: weiße/nahe-weiße, schwach gesättigte Flächen (Papier, helle Fliesen) unterdrücken
          und die Statue priorisieren. Zusätzlich Gradient-Gate, damit echte Struktur durchkommt.
    Rückgabe: uint8 Maske {0,255}.
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr

    # HSV: Weiß ~ hohe V, sehr niedrige S
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)
    white_like = (S.astype(np.float32) <= 35) & (V.astype(np.float32) >= 200)  # Toleranz anpassen falls nötig

    # Lab: sehr helle Pixel (L hoch) ebenfalls als Hintergrund
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    L = lab[:,:,0].astype(np.float32)
    very_bright = L >= 235

    bg = (white_like | very_bright).astype(np.uint8)  # 0/1

    # Gradient-Gate: wo echtes Relief ist, wollen wir nicht zu aggressiv maskieren
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    # Schwelle relativ zur Szene
    t = np.percentile(mag, 70)
    texture = (mag >= max(t, 10.0)).astype(np.uint8)  # 0/1

    # Hintergrund nur dort, wo wenig Textur: verhindert, dass wir die Statue wegradieren
    bg_refined = np.where(texture==1, 0, bg).astype(np.uint8)

    # Invertieren = Vordergrund
    fg = (1 - bg_refined).astype(np.uint8)

    # Zentralitäts-Prior: wir akzeptieren eher die Mitte (robust gegen Ränder)
    Y, X = np.ogrid[:h, :w]
    cx, cy = w/2.0, h/2.0
    rx, ry = 0.48*w, 0.48*h
    ell = (((X - cx)/rx)**2 + ((Y - cy)/ry)**2) <= 1.0
    central = np.zeros((h,w), np.uint8); central[ell] = 1

    fg = fg * central

    # Morphologie
    k = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, k)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, k, iterations=2)

    # größte Komponente behalten (stabilisiert gegen Ausreißer)
    num_labels, labels = cv.connectedComponents(fg)
    if num_labels > 1:
        counts = np.bincount(labels.flatten())
        counts[0] = 0  # Hintergrund ignorieren
        idx = np.argmax(counts)
        fg = (labels == idx).astype(np.uint8)

    return (fg*255).astype(np.uint8)

# -------------------- main extractor --------------------

def sift_extract(images: List[str], out_dir: str,
                 on_log: Callable[[str], None] = None,
                 on_progress: Callable[[int,str], None] = None
                 ) -> Tuple[list, list, list]:
    """
    Bright-aware + foreground-masked SIFT:
      - Homomorphic Beleuchtungskorrektur
      - Adaptive Maske (weiße/schwach gesättigte Flächen unterdrückt, Statue bevorzugt)
      - SIFT nur innerhalb der Maske
      - RootSIFT
    """
    def log(m):  on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    # SIFT sensibel, aber stabil
    sift = cv.SIFT_create(
        nfeatures=14000,
        nOctaveLayers=3,
        contrastThreshold=0.02,   # niedriger -> mehr auf flachen/hellen Oberflächen
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints, descriptors, shapes = [], [], []
    N = len(images)

    for i, path in enumerate(images):
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)
        if img_bgr is None:
            log(f"[sift-masked] WARN: {path} unlesbar")
            keypoints.append([]); descriptors.append(None); shapes.append((0,0))
            continue

        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        proc = _homomorphic_brightfix(img_gray)

        # adaptive Vordergrundmaske
        mask = _make_statue_mask(img_bgr)

        # Detektion nur in Maske
        kps, des = sift.detectAndCompute(proc, mask)
        des = _rootsift(des)

        keypoints.append(kps)
        descriptors.append(des)
        shapes.append(proc.shape)

        # speichern pro Frame
        np.savez(os.path.join(out_dir, f"features_{i:04d}.npz"),
                 kps=_kp_to_np(kps), des=des, shape=np.array(proc.shape))

        # Debug: jede 30. Maske speichern
        if i % 30 == 0:
            dbg = cv.cvtColor(proc, cv.COLOR_GRAY2BGR)
            dbg[mask==0] = (0,0,0)
            cv.imwrite(os.path.join(out_dir, "masks", f"mask_{i:04d}.png"), mask)
            cv.imwrite(os.path.join(out_dir, "masks", f"proc_{i:04d}.png"), dbg)

        if N > 0:
            prog(int(30 + (i+1)/max(1, N)*10), "Feature Extraction (FG-masked)")

    total_kp = sum(len(k) for k in keypoints)
    log(f"[sift-masked] done: {total_kp} Keypoints (masked)")
    return keypoints, descriptors, shapes
