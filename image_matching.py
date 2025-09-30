import os
from typing import Callable, Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np
import torch

Pair = Tuple[int, int]


def _ratio_test(matches, ratio: float):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def build_pairs(
    descriptors: List[np.ndarray],
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    save_dir: Optional[str] = None,
    keypoints: Optional[List[List[cv.KeyPoint]]] = None,
    meta: Optional[dict] = None
) -> Tuple[List[Pair], Dict[Pair, List[cv.DMatch]]]:
    """
    MATCH_BACKEND == 'lightglue'  -> uses LightGlue (expects SuperPoint/DISK-like descriptors)
    otherwise: classic BF + ratio test.
    """
    def log(m):  on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(int(p), s)

    backend = os.getenv("MATCH_BACKEND", "classic").lower()
    ratio = float(os.getenv("MATCH_RATIO", "0.82"))

    N = len(descriptors)
    idxs = list(range(N))

    # simple neighborhood pairs
    pairs: List[Pair] = []
    for i in idxs:
        for d in range(1, 5):
            j = i + d
            if j < N:
                pairs.append((i, j))
    if N > 8:
        for i in range(0, N, max(1, N // 8)):
            j = N - 1
            if i < j:
                pairs.append((i, j))

    matches: Dict[Pair, List[cv.DMatch]] = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # LightGlue
    if backend == "lightglue":
        from lightglue import LightGlue

        # device selection (same convention as features)
        device_env = os.getenv("FEATURE_DEVICE", "").lower().strip()
        device = device_env if device_env in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
        matcher = LightGlue(features="superpoint").to(device).eval()
        log(f"[match] LightGlue on {device}")

        assert keypoints is not None and meta is not None, \
            "[match] LightGlue requires keypoints and meta (scores, image_size) from SuperPoint."

        sp_scores = meta.get("sp_scores", None)
        sp_sizes  = meta.get("sp_sizes", None)

        def to_torch_feat(i: int):
            kps = keypoints[i]
            des = descriptors[i]
            if des is None or len(kps) == 0 or des.size == 0:
                return None

            # [N,2] (x,y)
            kp_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32, order="C")
            kp_xy_t = torch.from_numpy(kp_xy)[None, ...].to(device)               # [1,N,2]

            # descriptors: LightGlue expects [B,C,N]
            desc = des.astype(np.float32, copy=False)                             # [N,C]
            desc_t = torch.from_numpy(desc).T.contiguous()[None, ...].to(device)  # [1,C,N]

            # scores (optional but helpful)
            if sp_scores is not None and len(sp_scores) > i and sp_scores[i] is not None:
                sc = torch.from_numpy(sp_scores[i].astype(np.float32, copy=False))[None, ...].to(device)  # [1,N]
            else:
                sc = torch.ones((1, kp_xy.shape[0]), dtype=torch.float32, device=device)

            # image size [B,2] = [H,W]
            if sp_sizes is not None and len(sp_sizes) > i and sp_sizes[i] is not None:
                H, W = sp_sizes[i]
            else:
                # fallback: rough estimate from keypoints
                H = int(max(1, max(int(k.pt[1]) for k in kps)))
                W = int(max(1, max(int(k.pt[0]) for k in kps)))
            size_t = torch.tensor([[int(H), int(W)]], dtype=torch.int32, device=device)  # [1,2]

            return {
                "keypoints": kp_xy_t,     # [1,N,2]
                "descriptors": desc_t,    # [1,C,N]
                "scores": sc,             # [1,N]
                "image_size": size_t      # [1,2]
            }

        torch.set_grad_enabled(False)
        with torch.no_grad():
            for t, (i, j) in enumerate(pairs):
                Fi = to_torch_feat(i)
                Fj = to_torch_feat(j)
                if Fi is None or Fj is None:
                    matches[(i, j)] = []
                else:
                    out = matcher({"image0": Fi, "image1": Fj})
                    m0 = out["matches0"][0].detach().cpu().numpy()  # [N0], indices in image1 or -1
                    valid = np.where(m0 >= 0)[0]
                    if valid.size == 0:
                        matches[(i, j)] = []
                    else:
                        dj = m0[valid]
                        ms = [cv.DMatch(_queryIdx=int(qi), _trainIdx=int(int(tj)), _imgIdx=0, _distance=0.0)
                              for qi, tj in zip(valid.tolist(), dj.tolist())]
                        matches[(i, j)] = ms

                if save_dir:
                    arr = np.array([[m.queryIdx, m.trainIdx] for m in matches[(i, j)]], dtype=np.int32)
                    np.savez_compressed(os.path.join(save_dir, f"match_{i:04d}_{j:04d}.npz"),
                                        idx_i=i, idx_j=j, matches=arr)
                if N:
                    prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (lightglue)")
        return pairs, matches

    # Classic (BF + ratio)
    log("[match] classic BF matcher")
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    for t, (i, j) in enumerate(pairs):
        di = descriptors[i]; dj = descriptors[j]
        if di is None or dj is None or len(di) == 0 or len(dj) == 0:
            matches[(i, j)] = []
            if N:
                prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (classic)")
            continue

        knn = bf.knnMatch(di, dj, k=2)
        good = _ratio_test(knn, ratio)
        matches[(i, j)] = good

        if save_dir:
            np.savez_compressed(os.path.join(save_dir, f"match_{i:04d}_{j:04d}.npz"),
                                idx_i=i, idx_j=j,
                                matches=np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int32))
        if N:
            prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (classic)")

    return pairs, matches
