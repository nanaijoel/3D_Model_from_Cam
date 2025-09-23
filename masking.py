import os, cv2 as cv, numpy as np
from typing import Callable, List

def generate_object_masks(images: List[str], out_dir: str,
                          on_log: Callable[[str], None] = None,
                          on_progress: Callable[[int, str], None] = None) -> List[str]:
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(int(p), s)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, path in enumerate(images):
        bgr = cv.imread(path)
        if bgr is None:
            log(f"[mask] WARN: {path} unlesbar"); continue
        h, w = bgr.shape[:2]

        # Statue dunkel, Hintergrund hell -> LAB-L-Kanal, Otsu, invertieren
        L = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)[:, :, 0]
        blur = cv.GaussianBlur(L, (5, 5), 0)
        _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        mask = 255 - th

        # Morphologie + größte Komponente nahe Bildzentrum
        kernel = np.ones((7, 7), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(mask)
        if cnts:
            c = max(cnts, key=cv.contourArea)
            cv.drawContours(keep, [c], -1, 255, -1)
        else:
            keep = mask

        # Elliptische Zentral-ROI, um Randartefakte zu cutten
        roi = np.zeros_like(keep)
        cv.ellipse(roi, (w//2, h//2), (int(w*0.40), int(h*0.45)), 0, 0, 360, 255, -1)
        keep = cv.bitwise_and(keep, roi)

        out = os.path.join(out_dir, f"mask_{i:04d}.png")
        cv.imwrite(out, keep)
        paths.append(out)

        prog(28 + (i+1)/max(1,len(images))*2, "Masken erzeugen")

    log(f"[mask] saved {len(paths)}")
    return paths
