import numpy as np, cv2 as cv
from typing import Callable, List, Tuple

def knn_ratio_match(desc_a, desc_b, ratio=0.78):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    if desc_a is None or desc_b is None or len(desc_a)==0 or len(desc_b)==0:
        return []
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair)==2:
            m,n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

def build_pairs(descriptors: List[np.ndarray],
                on_log: Callable[[str], None] = None,
                on_progress: Callable[[int,str], None] = None):
    """Erzeugt Nachbar-Paare (i,i+1) + einige Sprungpaare (i,i+3) für robustere Initialisierung."""
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    N = len(descriptors)
    pairs = []
    # Nachbarn
    for i in range(N-1):
        pairs.append((i, i+1))
    # Sprungpaare (alle 3 Frames)
    for i in range(0, N-3, 3):
        pairs.append((0, i+3))
    log(f"[pairs] {len(pairs)} Paare")
    matches = {}
    for j,(a,b) in enumerate(pairs):
        m = knn_ratio_match(descriptors[a], descriptors[b])
        matches[(a,b)] = m
        prog(int(40 + (j+1)/len(pairs)*10), "Image Matching")
    return pairs, matches
