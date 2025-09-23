import os, numpy as np, cv2 as cv
from typing import Callable, List, Tuple, Dict, Optional

RATIO = 0.85
F_THRESH_NEAR = 2.5
F_THRESH_WIDE = 3.0
MIN_KEEP_AFTER_RANSAC = 40  # Fallback-Schwelle

def knn_ratio(desc_a, desc_b, ratio=RATIO):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    if desc_a is None or desc_b is None or len(desc_a)==0 or len(desc_b)==0:
        return []
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    out=[]
    for pair in knn:
        if len(pair)==2:
            m,n = pair
            if m.distance < ratio*n.distance:
                out.append(m)
    return out

def symmetric_filter(m_ab, m_ba):
    # mutual consistency
    back = {m.trainIdx: m.queryIdx for m in m_ba}
    return [m for m in m_ab if back.get(m.trainIdx, -1) == m.queryIdx]

def fundamental_ransac(kps_a, kps_b, matches, thresh=F_THRESH_NEAR, prob=0.999):
    if len(matches) < 8:
        return [], None
    pts1 = np.float32([kps_a[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps_b[m.trainIdx].pt for m in matches])
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, thresh, prob)
    if F is None or mask is None:
        return [], None
    mask = mask.ravel().astype(bool)
    return [m for m,keep in zip(matches, mask) if keep], F

def build_pairs(descriptors: List[np.ndarray],
                on_log: Callable[[str], None] = None,
                on_progress: Callable[[int,str], None] = None,
                save_dir: Optional[str] = None,
                keypoints: Optional[List[List[cv.KeyPoint]]] = None
                ) -> Tuple[List[Tuple[int,int]], Dict[Tuple[int,int], List[cv.DMatch]]]:
    """Nachbar-Paare + weite Basen; robustes Matching + Speichern + Fallbacks."""
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    N = len(descriptors)
    pairs: List[Tuple[int,int]] = []

    # Nachbarn
    for i in range(N - 1):
        pairs.append((i, i + 1))
    # Weite Basen (ein paar Schritte)
    for step in [3, 5, 8, 13, 20, 30]:
        for i in range(0, N - step, step):
            pairs.append((i, i + step))

    if save_dir: os.makedirs(save_dir, exist_ok=True)
    matches: Dict[Tuple[int,int], List[cv.DMatch]] = {}

    for j,(a,b) in enumerate(pairs):
        # 1) Grund-Matching (Ratio)
        m_ab = knn_ratio(descriptors[a], descriptors[b], ratio=RATIO)

        # 2) Mutual nur für weite Paare (bei Nachbarn oft kontraproduktiv)
        wide = (b - a) > 1
        if wide:
            m_ba = knn_ratio(descriptors[b], descriptors[a], ratio=RATIO)
            m = symmetric_filter(m_ab, m_ba)
        else:
            m = m_ab

        # 3) F-RANSAC
        if keypoints is not None:
            thr = F_THRESH_WIDE if wide else F_THRESH_NEAR
            m_ransac, F = fundamental_ransac(keypoints[a], keypoints[b], m, thresh=thr, prob=0.999)
            # Fallback: wenn fast alles wegradiert wurde, nimm die pre-RANSAC Matches (begrenzte Menge)
            if len(m_ransac) < MIN_KEEP_AFTER_RANSAC and len(m) >= MIN_KEEP_AFTER_RANSAC:
                log(f"[match] ({a},{b}) RANSAC kept {len(m_ransac)}/{len(m)} < {MIN_KEEP_AFTER_RANSAC} → fallback (no F)")
                kept = sorted(m, key=lambda mm: mm.distance)[:max(MIN_KEEP_AFTER_RANSAC, len(m_ransac))]
                m = kept
            else:
                m = m_ransac

        matches[(a,b)] = m

        # speichern (Indices)
        if save_dir:
            np.savez(os.path.join(save_dir, f"matches_{a:04d}_{b:04d}.npz"),
                     a=np.int32([mm.queryIdx for mm in m]),
                     b=np.int32([mm.trainIdx for mm in m]))

        log(f"[match] ({a:04d},{b:04d}) → {len(m)}")
        prog(int(40 + (j+1)/len(pairs)*10), "Image Matching")

    return pairs, matches
