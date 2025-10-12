# loftr_only_matching.py
import os
import cv2 as cv
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from LoFTR_extractor import LoFTRExtractor

def _mask_for(img_path: str, mask_dir: Optional[str]) -> Optional[str]:
    if not mask_dir: return None
    base = os.path.splitext(os.path.basename(img_path))[0]
    mp = os.path.join(mask_dir, f"{base}_mask.png")
    return mp if os.path.isfile(mp) else None

def _find_or_add_kp(kpt_map: Dict[Tuple[int,int], int],
                    kplist: List[cv.KeyPoint],
                    xy: Tuple[float, float],
                    quant: float = 0.5) -> int:
    """
    Finde (x,y) in der globalen Keypoint-Tabelle des Bildes wieder – robust gegen Subpixel-Noise –,
    sonst lege einen neuen Keypoint an und gib dessen Index zurück.
    """
    qx, qy = int(round(xy[0] / quant)), int(round(xy[1] / quant))
    key = (qx, qy)
    if key in kpt_map:
        return kpt_map[key]
    idx = len(kplist)
    kplist.append(cv.KeyPoint(float(xy[0]), float(xy[1]), 3.0, -1.0, 1.0, 0, -1))
    kpt_map[key] = idx
    return idx

def build_pairs_loftr(
    images: List[str],
    features_dir: str,
    device: str = "cuda",
    max_size: int = 640,
    on_log: Optional[Callable[[str], None]] = print,
    save_dir: Optional[str] = None,
    use_masks: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], List[List[cv.KeyPoint]]]:
    """
    Liefert:
      pairs_np:      (M,2) int32 mit Frame-Indizes (i,j)
      matches_list:  Liste von (K,2)-int32 Arrays mit [queryIdx, trainIdx]
      keypoints:     Liste pro Bild: OpenCV-KeyPoints (Indices passen zu matches_list)
    """
    log = (lambda m: on_log(m) if on_log else None)
    os.makedirs(save_dir or "", exist_ok=True)
    mask_dir = os.path.join(features_dir, "masks") if use_masks else None

    # 1) Globale Keypoint-Container pro Bild
    all_kps: List[List[cv.KeyPoint]] = [[] for _ in images]
    #    …und schnelles Quantisierungs-Indexing (pro Bild)
    all_maps: List[Dict[Tuple[int,int], int]] = [dict() for _ in images]

    # 2) LoFTR initialisieren
    loftr = LoFTRExtractor(device=device, weights=os.getenv("LOFTR_WEIGHTS", "outdoor"))  # :contentReference[oaicite:3]{index=3}

    # 3) Wähle zu matchende Paare (hier: Nachbarn + kleiner Span)
    span = int(float(os.getenv("LOFTR_MATCH_SPAN", "4")))
    pairs: List[Tuple[int,int]] = []
    N = len(images)
    for i in range(N):
        for d in range(1, span + 1):
            j = i + d
            if j < N:
                pairs.append((i, j))

    # 4) Für jedes Paar: LoFTR-Matches → Keypoint-Indizes → Match-Indextabelle
    matches_list: List[np.ndarray] = []
    for (i, j) in pairs:
        m1 = _mask_for(images[i], mask_dir)
        m2 = _mask_for(images[j], mask_dir)
        pts_i, pts_j = loftr.match_pair(images[i], images[j],
                                        max_size=max_size,
                                        mask1_path=m1, mask2_path=m2)  # :contentReference[oaicite:4]{index=4}
        if len(pts_i) == 0:
            matches_list.append(np.empty((0, 2), np.int32))
            continue

        # robustes Duplikat-Filtering innerhalb des Paars (≈1px)
        A = np.hstack([pts_i, pts_j])
        _, keep = np.unique(A.round().astype(np.int32), axis=0, return_index=True)
        pts_i = pts_i[keep]; pts_j = pts_j[keep]

        # optional: RANSAC auf F (LoFTR ist gut, aber SfM freut sich über saubere Inlier)
        if len(pts_i) >= 8:
            _, inl = cv.findFundamentalMat(pts_i, pts_j, cv.FM_RANSAC, 1.5, 0.999)
            if inl is not None:
                mask = inl.ravel().astype(bool)
                pts_i = pts_i[mask]; pts_j = pts_j[mask]

        # mappe Pixel → Keypoint-Indizes (pro Bild konsistent)
        qidx = []
        tidx = []
        for a, b in zip(pts_i, pts_j):
            qi = _find_or_add_kp(all_maps[i], all_kps[i], (float(a[0]), float(a[1])))
            tj = _find_or_add_kp(all_maps[j], all_kps[j], (float(b[0]), float(b[1])))
            qidx.append(qi); tidx.append(tj)

        matches_list.append(np.stack([np.array(qidx, np.int32),
                                      np.array(tidx, np.int32)], axis=1))

        log(f"[loftr-only] pair ({i},{j}): kept={len(qidx)}")

        # optional Debugbild
        if save_dir:
            try:
                im1 = cv.imread(images[i], cv.IMREAD_COLOR)
                im2 = cv.imread(images[j], cv.IMREAD_COLOR)
                h1, w1 = im1.shape[:2]
                vis = np.zeros((max(h1, im2.shape[0]), w1 + im2.shape[1], 3), np.uint8)
                vis[:h1, :w1] = im1; vis[:im2.shape[0], w1:] = im2
                for a, b in zip(pts_i, pts_j):
                    p1 = (int(round(a[0])), int(round(a[1])))
                    p2 = (int(round(b[0] + w1)), int(round(b[1])))
                    cv.line(vis, p1, p2, (0, 255, 0), 1, cv.LINE_AA)
                cv.imwrite(os.path.join(save_dir, f"loftr_pair_{i:04d}_{j:04d}.jpg"), vis)
            except Exception:
                pass

    pairs_np = np.array(pairs, dtype=np.int32)
    return pairs_np, matches_list, all_kps
