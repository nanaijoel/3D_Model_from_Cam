import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2 as cv
import torch
from sklearn.cluster import MiniBatchKMeans  # neu: BoW-Retrieval

PAIR = Tuple[int, int]


def _log(msg: str, fn=None):
    if fn:
        fn(msg)


def _safe_load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    out = {
        "kps": d["kps"],                            # (N,7)
        "des": d["des"].astype(np.float32),         # (N,D) oder (0,D)
        "shape": tuple(d["shape"].tolist())         # (H,W)
    }

    if "scores" in d.files:
        out["scores"] = d["scores"].astype(np.float32)
    return out


def _kps_to_xy(kps_np: np.ndarray) -> np.ndarray:
    if kps_np.size == 0:
        return np.empty((0, 2), np.float32)
    return kps_np[:, :2].astype(np.float32)


def _infer_features_name_from_dim(dim: int) -> Optional[str]:
    if dim == 256: return "superpoint"
    if dim == 128: return "disk"
    if dim == 64:  return "aliked"
    return None


def _expected_dim_from_name(name: str) -> int:
    return {"superpoint": 256, "disk": 128, "aliked": 128}[name]


def _call_lightglue_try_both_apis(
    lg,
    k0: np.ndarray, d0: np.ndarray, s0: np.ndarray, hw0: Tuple[int, int],
    k1: np.ndarray, d1: np.ndarray, s1: np.ndarray, hw1: Tuple[int, int],
    device: str,
) -> np.ndarray:

    H0, W0 = hw0
    H1, W1 = hw1
    t_k0 = torch.from_numpy(k0)[None].to(device)       # [1,N0,2]
    t_k1 = torch.from_numpy(k1)[None].to(device)       # [1,N1,2]
    t_d0 = torch.from_numpy(d0)[None].to(device)       # [1,N0,C]
    t_d1 = torch.from_numpy(d1)[None].to(device)       # [1,N1,C]
    t_s0 = torch.from_numpy(s0)[None].to(device)       # [1,N0]
    t_s1 = torch.from_numpy(s1)[None].to(device)       # [1,N1]
    t_sz0 = torch.tensor([[H0, W0]], dtype=torch.float32, device=device)
    t_sz1 = torch.tensor([[H1, W1]], dtype=torch.float32, device=device)

    with torch.no_grad():
        # 1) Suffixed-API
        try:
            out = lg({
                "keypoints0": t_k0, "keypoints1": t_k1,
                "descriptors0": t_d0, "descriptors1": t_d1,
                "scores0": t_s0, "scores1": t_s1,
                "image_size0": t_sz0, "image_size1": t_sz1,
            })

            if "matches" in out:  # [B,M,2]
                m = out["matches"][0].detach().cpu().numpy().astype(np.int32)
                return m
            if "matches0" in out:  # [1,N0] mit -1
                mvec = out["matches0"][0].detach().cpu().numpy()
                valid0 = np.where(mvec >= 0)[0].astype(np.int32)
                if valid0.size == 0:
                    return np.empty((0, 2), np.int32)
                valid1 = mvec[valid0].astype(np.int32)
                return np.stack([valid0, valid1], axis=1).astype(np.int32)
        except Exception:
            pass

        # 2) image0/image1-API
        try:
            out = lg({
                "image0": {
                    "keypoints": t_k0,
                    "descriptors": t_d0,
                    "scores": t_s0,
                    "image_size": t_sz0,
                },
                "image1": {
                    "keypoints": t_k1,
                    "descriptors": t_d1,
                    "scores": t_s1,
                    "image_size": t_sz1,
                },
            })
            if "matches" in out:  # [B,M,2]
                m = out["matches"][0].detach().cpu().numpy().astype(np.int32)
                return m
            if "matches0" in out:
                mvec = out["matches0"][0].detach().cpu().numpy()
                valid0 = np.where(mvec >= 0)[0].astype(np.int32)
                if valid0.size == 0:
                    return np.empty((0, 2), np.int32)
                valid1 = mvec[valid0].astype(np.int32)
                return np.stack([valid0, valid1], axis=1).astype(np.int32)
        except Exception:
            pass

    return np.empty((0, 2), np.int32)


def build_pairs(
    features_dir: str,
    ratio: float = 0.82,
    device: Optional[str] = None,
    backend: str = "lightglue",
    on_log=None,
    save_dir: Optional[str] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    _log(f"[match] scan features in {features_dir}", on_log)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # find feature data files
    fnames = sorted([f for f in os.listdir(features_dir) if f.startswith("features_") and f.endswith(".npz")])
    paths = [os.path.join(features_dir, f) for f in fnames]
    N = len(paths)
    if N < 2:
        _log("[match] not enough frames", on_log)
        return np.empty((0, 2), np.int32), []

    # Define descriptor-dim
    first_dim = None
    for p in paths:
        f = _safe_load_npz(p)
        if f["des"].ndim == 2 and f["des"].shape[0] > 0:
            first_dim = int(f["des"].shape[1])
            break
    if first_dim is None:
        _log("[match] all frames are empty -> no pairs", on_log)
        return np.empty((0, 2), np.int32), []

    env_feat = os.getenv("MATCH_FEATURES", "").strip().lower()
    if env_feat in ("superpoint", "disk", "aliked"):
        features_name = env_feat
    else:
        features_name = _infer_features_name_from_dim(first_dim) or "superpoint"
    exp_dim = _expected_dim_from_name(features_name)

    # Preprocess matcher
    lg = None
    classic = None
    if backend.lower() == "lightglue":
        try:
            from lightglue import LightGlue
            lg = LightGlue(
                features=features_name,
                depth_confidence=float(os.getenv("MATCH_DEPTH", "0.95")),
                width_confidence=float(os.getenv("MATCH_WIDTH", "0.99")),
                filter_threshold=float(os.getenv("MATCH_FILTER", "0.10")),
            ).to(device).eval()
            _log(f"[match] LightGlue on {device} (features='{features_name}')", on_log)
        except Exception as e:
            _log(f"[match] WARN: LightGlue init failed ({e})", on_log)

    if lg is None and exp_dim == 128:
        classic = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        _log(f"[match] Classic BF (SNN, ratio={ratio}) for 128-D", on_log)
    elif lg is None:
        raise RuntimeError("LightGlue not available and descriptors are not 128-D; cannot match.")

    # ---------------- Retrieval-Graph (BoW + Cosine) ----------------
    retr_enable   = (os.getenv("RETR_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on"))
    retr_K        = int(float(os.getenv("RETR_K", "4096")))
    retr_sample   = int(float(os.getenv("RETR_SAMPLE_PER_IMG", "3000")))
    retr_topk     = int(float(os.getenv("RETR_TOPK", "12")))
    retr_min_sim  = float(os.getenv("RETR_MIN_SIM", "0.0"))

    pair_list_retr: List[PAIR] = []
    if retr_enable:
        _log(f"[match][retr] build BoW (K={retr_K}, sample/img={retr_sample})", on_log)

        # 1) subsample of descriptors across images
        desc_sub = []
        for p in paths:
            f = _safe_load_npz(p)
            D = f["des"]
            if D.ndim == 2 and len(D) > 0:
                take = min(len(D), retr_sample)
                if take < len(D):
                    idx = np.random.choice(len(D), take, replace=False)
                    D = D[idx]
                desc_sub.append(D.astype(np.float32))

        if len(desc_sub) >= 1:
            X = np.vstack(desc_sub).astype(np.float32)

            # 2) vocabulary
            kmeans = MiniBatchKMeans(n_clusters=retr_K, batch_size=10000, n_init=1, verbose=False)
            kmeans.fit(X)

            # 3) BoW per image (TF-IDF + L2)
            B = []
            for p in paths:
                f = _safe_load_npz(p)
                D = f["des"]
                if D.ndim != 2 or len(D) == 0:
                    B.append(np.zeros((retr_K,), np.float32))
                    continue
                c = kmeans.predict(D.astype(np.float32))
                h, _ = np.histogram(c, bins=np.arange(retr_K + 1))
                B.append(h.astype(np.float32))
            B = np.stack(B, axis=0)  # [N,K]

            df = (B > 0).sum(axis=0).clip(1)
            idf = np.log((len(B) + 1) / df).astype(np.float32)
            B *= idf
            B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)

            # 4) cosine similarity → Top-K similar per image
            S = B @ B.T
            np.fill_diagonal(S, -1.0)

            for i in range(N):
                js = np.argsort(-S[i])[:retr_topk]
                for j in js:
                    if S[i, j] >= retr_min_sim:
                        a, b = (i, j) if i < j else (j, i)
                        pair_list_retr.append((a, b))

            pair_list_retr = sorted(list(set(pair_list_retr)))
            _log(f"[match][retr] candidate pairs: {len(pair_list_retr)}", on_log)
        else:
            _log("[match][retr] skipped (no descriptors)", on_log)

    # --------------- Neighbor pairs (konfigurierbar) ---------------
    span = int(float(os.getenv("MATCH_NEIGHBOR_SPAN", "5")))
    pair_list_ngb: List[PAIR] = []
    for i in range(N):
        for j in range(i + 1, min(N, i + 1 + span)):
            pair_list_ngb.append((i, j))

    # --------------- Union aus Retrieval ∪ Nachbarn ----------------
    if retr_enable and pair_list_retr:
        pair_list = sorted(list(set(pair_list_retr).union(pair_list_ngb)))
        _log(f"[match] using {len(pair_list)} pairs (retrieval ∪ neighbors)", on_log)
    else:
        pair_list = pair_list_ngb
        _log(f"[match] using {len(pair_list)} neighbor pairs (span={span})", on_log)

    # ------------------------- Matching ----------------------------
    pairs_out: List[List[int]] = []
    matches_out: List[np.ndarray] = []

    for (i, j) in pair_list:
        Fi = _safe_load_npz(paths[i])
        Fj = _safe_load_npz(paths[j])

        k0 = _kps_to_xy(Fi["kps"])
        k1 = _kps_to_xy(Fj["kps"])
        d0 = Fi["des"]
        d1 = Fj["des"]

        if d0.ndim != 2 or d1.ndim != 2 or d0.shape[0] == 0 or d1.shape[0] == 0:
            _log(f"[match] skip pair ({i},{j}): empty descriptors", on_log)
            pairs_out.append([i, j])
            matches_out.append(np.empty((0, 2), np.int32))
            continue
        if d0.shape[1] != exp_dim or d1.shape[1] != exp_dim:
            _log(f"[match] skip pair ({i},{j}): descriptor dim mismatch "
                 f"({d0.shape[1]} vs {d1.shape[1]}, expected {exp_dim})", on_log)
            pairs_out.append([i, j])
            matches_out.append(np.empty((0, 2), np.int32))
            continue

        # Scores
        s0 = Fi.get("scores", np.ones((len(k0),), np.float32))
        s1 = Fj.get("scores", np.ones((len(k1),), np.float32))
        H0, W0 = Fi["shape"]
        H1, W1 = Fj["shape"]

        # If lightglue, use corresponding variables, else classic BF
        if lg is not None:
            m = _call_lightglue_try_both_apis(
                lg, k0, d0, s0, (H0, W0),
                k1, d1, s1, (H1, W1),
                device
            )
        else:
            knn = classic.knnMatch(d0, d1, k=2)
            good = []
            for a, b in knn:
                if a.distance < ratio * b.distance:
                    good.append([a.queryIdx, a.trainIdx])
            m = np.array(good, dtype=np.int32) if good else np.empty((0, 2), np.int32)

        pairs_out.append([i, j])
        matches_out.append(m)

    pairs_arr = np.array(pairs_out, dtype=np.int32)

    if save_dir:
        save_pairs(save_dir, pairs_arr, matches_out)

    return pairs_arr, matches_out


def save_pairs(out_dir: str, pairs: np.ndarray, matches: List[np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # variable-length arrays -> dtype=object!
    matches_obj = np.empty(len(matches), dtype=object)
    matches_obj[:] = matches
    np.savez(os.path.join(out_dir, "matches.npz"),
             pairs=pairs.astype(np.int32),
             matches=matches_obj)
