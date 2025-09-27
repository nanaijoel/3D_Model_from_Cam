# image_matching.py
import os, numpy as np, cv2 as cv
from typing import Callable, List, Tuple, Dict, Optional

# --- Matching-Parameter ------------------------------------------------
RATIO = 0.85
F_THRESH_NEAR = 0.5          # RANSAC-Threshold (px) für Nachbarn (step <= max_span)
F_THRESH_WIDE = 1.5         # strenger für Loop-Closure/weite Paare
MIN_INLIERS_NEAR = 40
MIN_INLIERS_WIDE = 120       # deutlich höher für robuste Loop-Closures

def _knn(desc_a, desc_b, k=2):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return []
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    return bf.knnMatch(desc_a, desc_b, k=k)

def _ratio_filter(knn_list, ratio=RATIO):
    out = []
    for p in knn_list:
        if len(p) == 2 and p[0].distance < ratio * p[1].distance:
            out.append(p[0])
    return out

def _mutual_ratio(desc_a, desc_b, ratio=RATIO):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return []
    ab = _ratio_filter(_knn(desc_a, desc_b, 2), ratio)
    ba = _ratio_filter(_knn(desc_b, desc_a, 2), ratio)
    back = {m.queryIdx: m.trainIdx for m in ba}
    return [m for m in ab if back.get(m.trainIdx, -1) == m.queryIdx]

def _geom_ransac(kps_a, kps_b, matches, thr_px):
    if not matches:
        return []
    pts_a = np.float32([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in matches])
    F, mask = cv.findFundamentalMat(pts_a, pts_b, cv.FM_RANSAC,
                                    ransacReprojThreshold=float(thr_px),
                                    confidence=0.999)
    if F is None or mask is None:
        return []
    mask = mask.ravel().astype(bool)
    return [mm for mm, ok in zip(matches, mask) if ok]

def build_pairs(descriptors: List[np.ndarray],
                on_log: Optional[Callable[[str], None]] = None,
                on_progress: Optional[Callable[[int, str], None]] = None,
                save_dir: Optional[str] = None,
                keypoints: Optional[List[List[cv.KeyPoint]]] = None,
                # Graph-Parameter:
                max_span: int = 6,
                long_spans: Tuple[int, ...] = (),   # keine generischen Long-Spans
                add_loop_closures: bool = True      # genau 1–2 robuste Loops
                ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[cv.DMatch]]]:

    def log(m):  on_log and on_log(m)
    def prog(v, s): on_progress and on_progress(int(v), s)

    N = len(descriptors)
    if N < 2:
        return [], {}

    # 1) Pair list (lokaler Ketten-Graph + wenige robuste Loop-Kandidaten)
    pairs: List[Tuple[int, int]] = []
    for i in range(N - 1):
        pairs.append((i, i + 1))
    for d in range(2, max_span + 1):
        for i in range(0, N - d):
            pairs.append((i, i + d))

    if add_loop_closures and N >= 20:
        # Wenige, feste, N-robuste Loop-Kandidaten
        loop = {(0, N - 1)}
        if N >= 60:
            loop.add((N // 4, 3 * N // 4))
        pairs.extend([(a, b) for (a, b) in loop if 0 <= a < b < N])

    pairs = sorted(set(pairs))
    log(f"[match] building pairs: N={N}, pairs={len(pairs)}")
    prog(35, "Image Matching – build pairs")

    # 2) Matching
    matches: Dict[Tuple[int, int], List[cv.DMatch]] = {}

    for j, (a, b) in enumerate(pairs):
        des_a = descriptors[a]; des_b = descriptors[b]
        m = _mutual_ratio(des_a, des_b, ratio=RATIO)

        step = b - a
        is_near = (step <= max_span)
        thr = F_THRESH_NEAR if is_near else F_THRESH_WIDE
        min_inl = MIN_INLIERS_NEAR if is_near else MIN_INLIERS_WIDE

        if keypoints is not None and len(m) >= 8:
            m = _geom_ransac(keypoints[a], keypoints[b], m, thr_px=thr)

        if len(m) < min_inl:
            m = []

        matches[(a, b)] = m

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, f"matches_{a:04d}_{b:04d}.npz"),
                     a=np.int32([mm.queryIdx for mm in m]),
                     b=np.int32([mm.trainIdx for mm in m]))
        log(f"[match] ({a:04d},{b:04d}) → inliers={len(m)} (step={step})")
        prog(40 + int((j + 1) / max(1, len(pairs)) * 10), "Image Matching")

    return pairs, matches
