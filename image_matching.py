import os
import traceback
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


def _sanitize_np(a: np.ndarray, want_float: bool = True, make_contig: bool = True) -> np.ndarray:
    if want_float and a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not want_float and a.dtype != np.int32:
        a = a.astype(np.int32, copy=False)
    if make_contig and not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    # NaN/Inf filtern
    if want_float and (not np.isfinite(a).all()):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _build_pairs(N: int) -> List[Pair]:
    pairs: List[Pair] = []
    for i in range(N):
        for d in range(1, 5):
            j = i + d
            if j < N:
                pairs.append((i, j))
    if N > 8:
        for i in range(0, N, max(1, N // 8)):
            j = N - 1
            if i < j:
                pairs.append((i, j))
    return pairs


def build_pairs(
    descriptors: List[np.ndarray],
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    save_dir: Optional[str] = None,
    keypoints: Optional[List[List[cv.KeyPoint]]] = None,
    meta: Optional[dict] = None
) -> Tuple[List[Pair], Dict[Pair, List[cv.DMatch]]]:
    def log(m):  on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(int(p), s)

    backend = os.getenv("MATCH_BACKEND", "classic").lower()
    ratio = float(os.getenv("MATCH_RATIO", "0.82"))
    N = len(descriptors)
    pairs = _build_pairs(N)
    matches: Dict[Pair, List[cv.DMatch]] = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --------- LightGlue backend ----------
    if backend == "lightglue":
        try:
            from lightglue import LightGlue
        except Exception as e:
            log(f"[match] ERROR: LightGlue import failed: {e}")
            return pairs, {p: [] for p in pairs}

        # device wählen wie in features
        device_env = os.getenv("FEATURE_DEVICE", "").lower().strip()
        device = device_env if device_env in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

        assert keypoints is not None and meta is not None, \
            "[match] LightGlue requires keypoints and meta (scores, image_size) from features."

        lg_feat = meta.get("lg_feature_name", "superpoint")
        if lg_feat not in ("superpoint", "disk", "aliked"):
            lg_feat = "superpoint"

        # optional limitieren (zu viele KPs killen LG-Performance/Memory)
        max_kp_match = int(os.getenv("MATCH_MAX_KP", os.getenv("FEATURE_MAX_KP", "4096")))

        matcher = LightGlue(features=lg_feat).to(device).eval()
        log(f"[match] LightGlue on {device} (features='{lg_feat}', max_kp_match={max_kp_match})")

        sp_scores = meta.get("sp_scores", None)
        sp_sizes  = meta.get("sp_sizes", None)

        def to_feat(i: int):
            kps = keypoints[i]
            des = descriptors[i]
            if des is None or len(kps) == 0 or (isinstance(des, np.ndarray) and des.shape[0] == 0):
                return None, 0

            # --- Keypoints → [N,2] ---
            kp_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
            kp_xy = _sanitize_np(kp_xy, want_float=True)

            # --- Descriptors: [N,C] erwartet (haben wir aus features so gespeichert) ---
            des = _sanitize_np(des, want_float=True)
            if des.ndim != 2 or des.shape[0] != kp_xy.shape[0]:
                # robust kürzen auf gemeinsame Länge
                M = min(des.shape[0], kp_xy.shape[0])
                if M <= 0:
                    return None, 0
                des = des[:M]
                kp_xy = kp_xy[:M]

            # --- Scores: [N] ---
            if sp_scores is not None and len(sp_scores) > i and sp_scores[i] is not None and sp_scores[i].shape[0] > 0:
                sc_arr = sp_scores[i].astype(np.float32, copy=False)
                if sc_arr.shape[0] != kp_xy.shape[0]:
                    M = min(sc_arr.shape[0], kp_xy.shape[0])
                    sc_arr = sc_arr[:M]; kp_xy = kp_xy[:M]; des = des[:M]
            else:
                sc_arr = np.ones((kp_xy.shape[0],), np.float32)

            # --- optionale Limitierung nach Score (Top-K) ---
            if max_kp_match > 0 and kp_xy.shape[0] > max_kp_match:
                idx = np.argsort(-sc_arr)[:max_kp_match]  # top nach Score
                kp_xy = kp_xy[idx]
                des   = des[idx]
                sc_arr = sc_arr[idx]

            # --- torch Tensors in gewünschtem Layout ---
            kp_t = torch.from_numpy(kp_xy)[None, ...].to(device)                 # [1,N,2]
            desc_t = torch.from_numpy(des.T).contiguous()[None, ...].to(device)  # [1,C,N]
            sc_t = torch.from_numpy(sc_arr)[None, ...].to(device)                # [1,N]

            if sp_sizes is not None and len(sp_sizes) > i and sp_sizes[i] is not None:
                H, W = sp_sizes[i]
            else:
                # Fallback (nicht kritisch)
                H = int(max(1, np.max(kp_xy[:, 1])))
                W = int(max(1, np.max(kp_xy[:, 0])))
            size_t = torch.tensor([[int(H), int(W)]], dtype=torch.int32, device=device)  # [1,2]

            feat = {"keypoints": kp_t, "descriptors": desc_t, "scores": sc_t, "image_size": size_t}
            return feat, kp_xy.shape[0]

        torch.set_grad_enabled(False)
        with torch.no_grad():
            for t, (i, j) in enumerate(pairs):
                try:
                    Fi, ni = to_feat(i)
                    Fj, nj = to_feat(j)
                    if Fi is None or Fj is None or ni == 0 or nj == 0:
                        matches[(i, j)] = []
                    else:
                        # ---- Achtung: API je nach Version ----
                        # Einige Versionen erwarten matcher({"image0": Fi, "image1": Fj}),
                        # andere matcher.match(Fi, Fj). Wir probieren beides.
                        out = None
                        try:
                            out = matcher({"image0": Fi, "image1": Fj})
                        except Exception:
                            out = matcher.match(Fi, Fj)

                        m0 = out["matches0"][0].detach().cpu().numpy()
                        valid = np.where(m0 >= 0)[0]
                        if valid.size == 0:
                            matches[(i, j)] = []
                        else:
                            dj = m0[valid]
                            ms = [cv.DMatch(_queryIdx=int(qi), _trainIdx=int(int(tj)), _imgIdx=0, _distance=0.0)
                                  for qi, tj in zip(valid.tolist(), dj.tolist())]
                            matches[(i, j)] = ms

                except Exception:
                    # ausführliches Logging + Fallback auf classic für dieses Paar
                    err = traceback.format_exc()
                    log(f"[match] ERROR pair ({i},{j}):\n{err}")

                    # --- Fallback Classic ---
                    di = descriptors[i]; dj = descriptors[j]
                    if di is None or dj is None or len(di) == 0 or len(dj) == 0:
                        matches[(i, j)] = []
                    else:
                        di = _sanitize_np(di, want_float=True)
                        dj = _sanitize_np(dj, want_float=True)
                        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
                        try:
                            knn = bf.knnMatch(di, dj, k=2)
                            good = _ratio_test(knn, ratio)
                            matches[(i, j)] = good
                        except Exception as e2:
                            log(f"[match] classic fallback failed for ({i},{j}): {e2}")
                            matches[(i, j)] = []

                if save_dir:
                    arr = np.array([[m.queryIdx, m.trainIdx] for m in matches[(i, j)]], dtype=np.int32)
                    np.savez_compressed(os.path.join(save_dir, f"match_{i:04d}_{j:04d}.npz"),
                                        idx_i=i, idx_j=j, matches=arr)
                prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (lightglue)")

        return pairs, matches

    # --------- Classic (BF + ratio) ----------
    log("[match] classic BF matcher")
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    for t, (i, j) in enumerate(pairs):
        di = descriptors[i]; dj = descriptors[j]
        if di is None or dj is None or len(di) == 0 or len(dj) == 0:
            matches[(i, j)] = []
            prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (classic)")
            continue

        di = _sanitize_np(di, want_float=True)
        dj = _sanitize_np(dj, want_float=True)

        knn = bf.knnMatch(di, dj, k=2)
        good = _ratio_test(knn, ratio)
        matches[(i, j)] = good

        if save_dir:
            np.savez_compressed(os.path.join(save_dir, f"match_{i:04d}_{j:04d}.npz"),
                                idx_i=i, idx_j=j,
                                matches=np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int32))
        prog(40 + (t + 1) / max(1, len(pairs)) * 15, "Image Matching (classic)")

    return pairs, matches
