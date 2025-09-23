import os, cv2 as cv, numpy as np
from typing import Callable, List, Tuple

def _preprocess_gray(img_gray: np.ndarray) -> np.ndarray:
    # Kontrast anheben
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def _center_mask(h, w, ratio=0.85):
    # optional: Ellipsen-ROI (ignoriert Rand/Hintergrund)
    mh, mw = int(h*ratio/2), int(w*ratio/2)
    mask = np.zeros((h,w), np.uint8)
    cv.ellipse(mask, (w//2, h//2), (mw, mh), 0, 0, 360, 255, -1)
    return mask

def _rootsift(des):
    if des is None or len(des)==0: return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    des = np.sqrt(des)
    return des

def _kp_to_np(kps):
    # speicherbare Form: (x,y,size,angle,response,octave,class_id)
    out = np.zeros((len(kps), 7), np.float32)
    for i,k in enumerate(kps):
        out[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
    return out

def sift_extract(images: List[str], out_dir: str,
                 on_log: Callable[[str], None] = None,
                 on_progress: Callable[[int,str], None] = None
                 ) -> Tuple[list, list, list]:
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    os.makedirs(out_dir, exist_ok=True)
    sift = cv.SIFT_create(nfeatures=8000)  # mehr KPs
    keypoints, descriptors, shapes = [], [], []

    for i, path in enumerate(images):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            log(f"[sift] WARN: {path} unlesbar");
            continue

        img = _preprocess_gray(img)
        mask = _center_mask(*img.shape)  # falls zu streng: ratio=0.9 setzen/entfernen
        kps, des = sift.detectAndCompute(img, mask)
        des = _rootsift(des)

        keypoints.append(kps); descriptors.append(des); shapes.append(img.shape)
        # speichern
        np.savez(os.path.join(out_dir, f"features_{i:04d}.npz"),
                 kps=_kp_to_np(kps), des=des, shape=np.array(img.shape))
        prog(int(30 + (i+1)/len(images)*10), "Feature Extraction (RootSIFT+CLAHE)")

    log(f"[sift] done: {sum(len(k) for k in keypoints)} Keypoints (RootSIFT)")
    return keypoints, descriptors, shapes
