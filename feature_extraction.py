import os, cv2 as cv, numpy as np
from typing import Callable, List, Tuple

def sift_extract(images: List[str], out_dir: str,
                 on_log: Callable[[str], None] = None,
                 on_progress: Callable[[int,str], None] = None
                 ) -> Tuple[list, list, list]:
    """Extrahiert SIFT-Features. Gibt (keypoints, descriptors, shapes) zurück."""
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    sift = cv.SIFT_create(nfeatures=6000)
    keypoints, descriptors, shapes = [], [], []
    for i, path in enumerate(images):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            log(f"[sift] WARN: {path} unlesbar"); continue
        kps, des = sift.detectAndCompute(img, None)
        keypoints.append(kps); descriptors.append(des); shapes.append(img.shape)
        prog(int(30 + (i+1)/len(images)*10), "Feature Extraction (SIFT)")
    log(f"[sift] done: {sum(len(k) for k in keypoints)} Keypoints")
    return keypoints, descriptors, shapes
