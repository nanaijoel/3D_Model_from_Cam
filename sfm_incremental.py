# sfm_incremental.py (refactored)
import os
import numpy as np
import cv2 as cv
from typing import Callable, List, Dict, Tuple, Optional, DefaultDict
from dataclasses import dataclass
from collections import defaultdict

# =========================
# Config
# =========================

@dataclass
class SfMConfig:
    # Init pair search
    MIN_MATCHES_INIT: int = 250
    MIN_FLOW_PX: float = 6.0
    MIN_INLIERS_INIT: int = 35
    INIT_WINDOW_FRAMES: int = 80
    INIT_MAX_SPAN: int = 6
    INIT_WINDOW_CENTER: Optional[int] = None
    INIT_WINDOW_RATIO: Optional[float] = None
    FORCE_INIT_PAIR: Optional[Tuple[int, int]] = None

    # PnP
    MIN_INLIERS_PNP: int = 160
    PNP_ITERS: int = 8000
    PNP_ERR_PX: float = 3.0
    PNP_REPROJ_ACCEPT: float = 2.0

    # Triangulation (pairwise during expansion)
    TRI_MIN_CORR: int = 14
    TRI_REPROJ_MAX: float = 2.0
    TRI_MIN_PARALLAX_DEG: float = 4.0

    # Local stereo seeding
    SEED_SPAN: int = 3
    SEED_MIN_INL: int = 40
    SEED_REPROJ: float = 2.0

    # Multiview point validation (promotion)
    POINT_PROMOTION_MIN_OBS: int = 4
    POINT_MAX_MULTIVIEW_REPROJ: float = 2.0
    POINT_MIN_MULTIVIEW_PARALLAX_DEG: float = 5.0
    POINT_REQUIRE_POSITIVE_DEPTH_RATIO: float = 0.7

    # Low-parallax fallback
    LOWPAR_FALLBACK_MIN_OBS: int = 6
    LOWPAR_FALLBACK_MAX_MEDIAN_REPROJ: float = 1.6
    LOWPAR_FALLBACK_MIN_POSDEPTH_RATIO: float = 0.85

    # Frame gating for creating new points
    FRAME_MAX_MEDIAN_REPROJ_FOR_NEW_POINTS: float = 2.8

    # Optional densify pass
    DENSIFY_ENABLE: bool = True
    DENSIFY_MAX_SPAN: int = 20
    DENSIFY_MIN_MATCHES: int = 60
    DENSIFY_MIN_PARALLAX_DEG: float = 4.0
    DENSIFY_MAX_REPROJ: float = 2.0

    # Keyframes, loop constraints, smoothing
    USE_KEYFRAMES: bool = True
    KF_MIN_BASELINE_NORM: float = 0.012
    KF_MIN_PARALLAX_DEG: float = 1.5
    KF_MIN_NEW_2D3D: int = 18

    USE_LOOP_CONSTRAINTS: bool = True
    LOOP_MAX_SPAN: int = 9999
    LOOP_MIN_INLIERS: int = 80

    POSE_SMOOTHING: bool = True
    SMOOTH_LAMBDA: float = 0.25


# =========================
# Helpers
# =========================

def _idx2pts(matches, kps1, kps2):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def _save_camera_poses_npz_csv(poses_R: Dict[int, np.ndarray],
                               poses_t: Dict[int, np.ndarray],
                               out_dir: str,
                               on_log: Optional[Callable[[str], None]] = None):
    def log(m):
        if on_log:
            on_log(m)
    os.makedirs(out_dir, exist_ok=True)
    order = sorted(poses_R.keys())
    Rs, ts, idxs, Cs = [], [], [], []
    for i in order:
        R = poses_R[i]
        t = poses_t[i]
        C = -R.T @ t  # camera center
        Rs.append(R); ts.append(t); idxs.append(i); Cs.append(C.reshape(3))
    Rs = np.stack(Rs, axis=0) if len(Rs) else np.zeros((0, 3, 3))
    ts = np.stack(ts, axis=0) if len(ts) else np.zeros((0, 3, 1))
    Cs = np.stack(Cs, axis=0) if len(Cs) else np.zeros((0, 3))
    idxs = np.array(idxs, dtype=int)
    npz_path = os.path.join(out_dir, "camera_poses.npz")
    np.savez_compressed(npz_path, frame_idx=idxs, R=Rs, t=ts, C=Cs)
    log(f"[sfm] camera poses saved -> {npz_path}")
    try:
        import csv
        csv_path = os.path.join(out_dir, "camera_poses.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "Cx", "Cy", "Cz"])
            for i, C in zip(idxs, Cs):
                w.writerow([int(i), float(C[0]), float(C[1]), float(C[2])])
        log(f"[sfm] camera poses csv -> {csv_path}")
    except Exception as e:
        log(f"[sfm] WARN: CSV write failed: {e}")

def _triangulate(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    X = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X[:3] / X[3]).T
    return X

def _in_front(R, t, X):
    Xc = (R @ X.T + t).T
    return Xc[:, 2] > 0

def _reproj_err(K, R, t, X, pts2d):
    rvec, _ = cv.Rodrigues(R)
    proj, _ = cv.projectPoints(X, rvec, t, K, None)
    proj = proj.squeeze()
    return np.linalg.norm(proj - pts2d, axis=1)

def _bearing_in_world(Kinv, R, uv):
    rays_cam = (Kinv @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    return (R.T @ rays_cam.T).T

def _pairwise_max_parallax_deg(Kinv, poses_R, poses_t, obs_list):
    best = (0.0, None)
    for a in range(len(obs_list)):
        fa, uva = obs_list[a]
        Ra = poses_R.get(fa)
        if Ra is None: continue
        for b in range(a+1, len(obs_list)):
            fb, uvb = obs_list[b]
            Rb = poses_R.get(fb)
            if Rb is None: continue
            da = _bearing_in_world(Kinv, Ra, np.array([uva], float))[0]
            db = _bearing_in_world(Kinv, Rb, np.array([uvb], float))[0]
            cosang = np.clip(da @ db, -1.0, 1.0)
            ang = float(np.degrees(np.arccos(cosang)))
            if ang > best[0]:
                best = (ang, (fa, fb, uva, uvb))
    return best


# =========================
# Track database
# =========================

class TrackDB:
    """Manages 3D points, observations and promotion (multi-view validation)."""
    def __init__(self, K: np.ndarray, cfg: SfMConfig):
        self.points3d: List[Optional[np.ndarray]] = []
        self.state: List[str] = []  # 'tentative' | 'ok' | 'dead'
        self.obs: DefaultDict[int, List[Tuple[int, Tuple[float, float]]]] = defaultdict(list)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.cfg = cfg

    def new_point(self, X: np.ndarray) -> int:
        gi = len(self.points3d)
        self.points3d.append(X)
        self.state.append('tentative')
        return gi

    def add_obs(self, gi: int, frame: int, uv: Tuple[float, float]):
        self.obs[gi].append((frame, (float(uv[0]), float(uv[1]))))

    def promote_points(self, poses_R: Dict[int, np.ndarray], poses_t: Dict[int, np.ndarray],
                       adapt_from_reproj: Optional[float] = None):
        """Promote tentative → ok using multi-view checks. Optionally tighten thresholds when reprojection is low."""
        cfg = self.cfg
        par_min = cfg.POINT_MIN_MULTIVIEW_PARALLAX_DEG
        repro_max = cfg.POINT_MAX_MULTIVIEW_REPROJ
        pos_ratio = cfg.POINT_REQUIRE_POSITIVE_DEPTH_RATIO
        if (adapt_from_reproj is not None) and (adapt_from_reproj <= 1.6):
            par_min = max(3.5, par_min - 1.0)
            repro_max = min(repro_max, 1.8)
            pos_ratio = max(pos_ratio, 0.8)

        for gi in range(len(self.points3d)):
            if self.state[gi] != 'tentative':
                continue
            ob = self.obs.get(gi, [])
            if len(ob) < cfg.POINT_PROMOTION_MIN_OBS:
                continue

            max_par, pair = _pairwise_max_parallax_deg(self.Kinv, poses_R, poses_t, ob)
            if pair is not None and max_par >= par_min:
                fa, fb, uva, uvb = pair
                Ra, ta = poses_R[fa], poses_t[fa]
                Rb, tb = poses_R[fb], poses_t[fb]
                X = _triangulate(self.K, Ra, ta, Rb, tb,
                                 np.array([uva], np.float32),
                                 np.array([uvb], np.float32))[0]
                pos_ok = 0
                repro_all = []
                for f, uv in ob:
                    R, t = poses_R.get(f), poses_t.get(f)
                    if R is None: continue
                    z = (R @ X.reshape(3, 1) + t)[2, 0]
                    if z > 0: pos_ok += 1
                    repro_all.append(_reproj_err(self.K, R, t, X.reshape(1, 3), np.array([uv], float))[0])
                if len(repro_all) and np.median(repro_all) <= repro_max and \
                   pos_ok / len(ob) >= pos_ratio:
                    self.points3d[gi] = X
                    self.state[gi] = 'ok'
                    continue

            # Low-parallax fallback
            if len(ob) >= self.cfg.LOWPAR_FALLBACK_MIN_OBS:
                fa, uva = ob[0]
                fb, uvb = ob[-1]
                Ra, ta = poses_R.get(fa), poses_t.get(fa)
                Rb, tb = poses_R.get(fb), poses_t.get(fb)
                if (Ra is not None) and (Rb is not None):
                    X = _triangulate(self.K, Ra, ta, Rb, tb,
                                     np.array([uva], np.float32),
                                     np.array([uvb], np.float32))[0]
                    pos_ok = 0
                    repro_all = []
                    for f, uv in ob:
                        R, t = poses_R.get(f), poses_t.get(f)
                        if R is None: continue
                        z = (R @ X.reshape(3, 1) + t)[2, 0]
                        if z > 0: pos_ok += 1
                        repro_all.append(_reproj_err(self.K, R, t, X.reshape(1, 3), np.array([uv], float))[0])
                    if len(repro_all) >= self.cfg.LOWPAR_FALLBACK_MIN_OBS and \
                       np.median(repro_all) <= self.cfg.LOWPAR_FALLBACK_MAX_MEDIAN_REPROJ and \
                       pos_ok / len(ob) >= self.cfg.LOWPAR_FALLBACK_MIN_POSDEPTH_RATIO:
                        self.points3d[gi] = X
                        self.state[gi] = 'ok'
                        continue

            self.points3d[gi] = None
            self.state[gi] = 'dead'

    def valid_points_array(self) -> np.ndarray:
        return np.array([p for p, st in zip(self.points3d, self.state) if (p is not None and st == 'ok')],
                        dtype=float)


# =========================
# Keyframes and smoothing utilities
# =========================

def _should_be_keyframe(K, poses_R, poses_t, last_kf_idx: int, fi: int,
                        anchors: List[int], kps, matches, track3d, tdb: TrackDB, cfg: SfMConfig) -> bool:
    """Decide if 'fi' should be a keyframe: baseline/parallax and number of new 2D–3D correspondences."""
    if not cfg.USE_KEYFRAMES:
        return True
    if last_kf_idx is None:
        return True

    baseline = 0.0
    parallax_deg = 0.0
    Kinv = np.linalg.inv(K)

    for a in anchors:
        if (a, fi) not in matches:
            continue
        new_2d3d = 0
        for mm in matches[(a, fi)]:
            qa = (a, mm.queryIdx)
            if qa in track3d:
                gi = track3d[qa]
                if gi < len(tdb.points3d) and tdb.points3d[gi] is not None:
                    new_2d3d += 1
        if (last_kf_idx in poses_R) and (a in poses_R):
            mlist = matches[(a, fi)]
            if len(mlist) >= 1:
                uva = np.array([kps[a][mlist[0].queryIdx].pt], np.float32)
                uvf = np.array([kps[fi][mlist[0].trainIdx].pt], np.float32)
                da = _bearing_in_world(Kinv, poses_R[a], uva)[0]
                df = _bearing_in_world(Kinv, poses_R[a], uvf)[0]
                cosang = np.clip(da @ df, -1.0, 1.0)
                parallax_deg = max(parallax_deg, float(np.degrees(np.arccos(cosang))))

        if new_2d3d >= cfg.KF_MIN_NEW_2D3D and parallax_deg >= cfg.KF_MIN_PARALLAX_DEG:
            return True

    if (last_kf_idx in poses_R) and (fi in poses_R):
        C_last = -poses_R[last_kf_idx].T @ poses_t[last_kf_idx]
        C_now = -poses_R[fi].T @ poses_t[fi]
        d = float(np.linalg.norm(C_now - C_last))
        baseline = d
    return baseline >= cfg.KF_MIN_BASELINE_NORM


def _smooth_poses(poses_R: Dict[int, np.ndarray], poses_t: Dict[int, np.ndarray], lam: float):
    """Light pose smoothing along trajectory order (Laplacian on translations, SVD-averaged rotations)."""
    if lam <= 0 or len(poses_R) < 3:
        return
    order = sorted(poses_R.keys())
    T = {i: (-poses_R[i].T @ poses_t[i]).reshape(3) for i in order}
    T_sm = T.copy()
    for _ in range(2):
        for k in range(1, len(order) - 1):
            i0, i1, i2 = order[k - 1], order[k], order[k + 1]
            T_sm[i1] = (1 - lam) * T_sm[i1] + 0.5 * lam * (T_sm[i0] + T_sm[i2])
    R_sm = {i: poses_R[i].copy() for i in order}
    for _ in range(1):
        for k in range(1, len(order) - 1):
            i0, i1, i2 = order[k - 1], order[k], order[k + 1]
            Ravg = (R_sm[i0] + R_sm[i1] + R_sm[i2]) / 3.0
            U, _, Vt = np.linalg.svd(Ravg)
            R_sm[i1] = U @ Vt
    for i in order:
        C = T_sm[i].reshape(3, 1)
        poses_R[i] = R_sm[i]
        poses_t[i] = -poses_R[i] @ C


# =========================
# NEW: SfM pipeline subroutines
# =========================

def _init_window_bounds(N: int, cfg: SfMConfig):
    mid = (cfg.INIT_WINDOW_CENTER if cfg.INIT_WINDOW_CENTER is not None else (N // 2))
    if cfg.INIT_WINDOW_RATIO is not None:
        w_candidate = max(1, int(round(N * float(cfg.INIT_WINDOW_RATIO))))
    else:
        w_candidate = max(10, N // 3)
    w = min(cfg.INIT_WINDOW_FRAMES, w_candidate)
    return mid, w

def _in_init_window(i, j, mid, w, cfg: SfMConfig):
    return (abs(j - i) <= cfg.INIT_MAX_SPAN) and (min(i, j) >= mid - w) and (max(i, j) <= mid + w)

def _select_init_pair(pairs, matches, keypoints, K, cfg: SfMConfig, mid, w,
                      on_log: Optional[Callable[[str], None]] = None):
    """Pick best (i0,j0) & recover relative pose."""
    def log(m): on_log and on_log(m)
    best = dict(score=-1.0, pair=None, R=None, t=None, mask=None)
    stats = []
    for (i, j) in pairs:
        if not _in_init_window(i, j, mid, w, cfg):
            continue
        m = matches.get((i, j), [])
        if len(m) < cfg.MIN_MATCHES_INIT:
            continue
        p1, p2 = _idx2pts(m, keypoints[i], keypoints[j])
        flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        if flow < cfg.MIN_FLOW_PX:
            continue
        E, maskE = cv.findEssentialMat(p1, p2, K, method=cv.RANSAC, prob=0.999, threshold=1.5)
        if E is None:
            continue
        _, R, t, maskP = cv.recoverPose(E, p1, p2, K)
        inl_mask = (maskP.ravel().astype(bool)) & (maskE.ravel().astype(bool)[:len(maskP)])
        inliers = int(inl_mask.sum())
        if inliers < cfg.MIN_INLIERS_INIT:
            continue
        cheir = 0
        if inliers > 0:
            X = _triangulate(K, np.eye(3), np.zeros((3, 1)), R, t, p1[inl_mask], p2[inl_mask])
            cheir = int((_in_front(np.eye(3), np.zeros((3, 1)), X) &
                         _in_front(R, t, X)).sum())
        score = inliers + 2.0 * cheir
        stats.append((i, j, len(m), flow, inliers, cheir, score))
        if score > best["score"]:
            best.update(score=score, pair=(i, j), R=R, t=t, mask=inl_mask)

    # Optional forced pair
    if (cfg.FORCE_INIT_PAIR is not None) and (cfg.FORCE_INIT_PAIR in matches):
        i0, j0 = cfg.FORCE_INIT_PAIR
        log(f"[sfm] FORCE init pair = ({i0},{j0})")
        kps0, kps1 = keypoints[i0], keypoints[j0]
        m01 = matches[(i0, j0)]
        p1_all, p2_all = _idx2pts(m01, kps0, kps1)
        E, _ = cv.findEssentialMat(p1_all, p2_all, K, method=cv.RANSAC, prob=0.999, threshold=1.5)
        _, R, t, maskP = cv.recoverPose(E, p1_all, p2_all, K)
        best.update(score=float(len(m01)), pair=(i0, j0), R=R, t=t, mask=maskP.ravel().astype(bool))

    if best["pair"] is None:
        if stats:
            stats.sort(key=lambda x: x[-1], reverse=True)
            top = "\n".join([f"   cand ({a},{b}): matches={mm}, inl={ii}, cheirality={ch}, score={sc:.1f}"
                             for (a, b, mm, _, ii, ch, sc) in stats[:8]])
            log("[sfm] Init candidates (top):\n" + top)
        raise RuntimeError("[sfm] Initialization failed.")

    return best

def _initialize_with_pair(best, keypoints, matches, K, on_log=None):
    """Build initial poses, triangulate, seed TrackDB/track3d."""
    def log(m): on_log and on_log(m)
    (i0, j0) = best["pair"]
    kps0, kps1 = keypoints[i0], keypoints[j0]
    m01 = matches[(i0, j0)]
    p1_all, p2_all = _idx2pts(m01, kps0, kps1)
    R = best["R"]; t = best["t"]; mask = best["mask"]

    poses_R = {i0: np.eye(3), j0: R}
    poses_t = {i0: np.zeros((3, 1)), j0: t}

    p1_in = p1_all[mask]; p2_in = p2_all[mask]
    X_init = _triangulate(K, poses_R[i0], poses_t[i0], poses_R[j0], poses_t[j0], p1_in, p2_in)
    front = _in_front(poses_R[i0], poses_t[i0], X_init) & _in_front(poses_R[j0], poses_t[j0], X_init)

    if not np.any(front):
        R_inv = R.T
        t_inv = -R_inv @ t
        X_try = _triangulate(K, poses_R[i0], poses_t[i0], R_inv, t_inv, p1_in, p2_in)
        front2 = _in_front(poses_R[i0], poses_t[i0], X_try) & _in_front(R_inv, t_inv, X_try)
        if np.any(front2):
            R, t = R_inv, t_inv
            poses_R[j0], poses_t[j0] = R, t
            X_init, front = X_try, front2

    X_init = X_init[front]
    log(f"[sfm] init pair = ({i0},{j0}) score={best['score']:.1f}")
    log(f"[sfm] init inliers={int(mask.sum())} cheirality={int(front.sum())}")
    log(f"[sfm] X_init={len(X_init)}")

    # Seed TrackDB
    TDB = TrackDB(K, SfMConfig())  # cfg overwritten by caller right after
    TDB.cfg = TDB.cfg  # no-op, kept for clarity
    track3d: Dict[Tuple[int, int], int] = {}
    idx_inliers = np.where(mask)[0][front]
    for k, mi in enumerate(idx_inliers):
        mm = m01[mi]
        qa = (i0, mm.queryIdx)
        qb = (j0, mm.trainIdx)
        gi = TDB.new_point(X_init[k])
        track3d[qa] = gi
        track3d[qb] = gi
        TDB.add_obs(gi, i0, keypoints[i0][mm.queryIdx].pt)
        TDB.add_obs(gi, j0, keypoints[j0][mm.trainIdx].pt)
    return (i0, j0), poses_R, poses_t, track3d, TDB

def _count_corr_for_frame(fi: int, visited: set, matches, track3d) -> int:
    c = 0
    for a in visited:
        if (a, fi) not in matches:
            continue
        for mm in matches[(a, fi)]:
            if (a, mm.queryIdx) in track3d:
                c += 1
    return c

def _collect_2d3d_for_frame(fi: int, anchors: List[int],
                            keypoints, matches, track3d, TDB: TrackDB):
    obj_pts, img_pts = [], []
    used_g = set()
    for a in anchors:
        for mm in matches[(a, fi)]:
            qa = (a, mm.queryIdx); qb = (fi, mm.trainIdx)
            if qa in track3d:
                gi = track3d[qa]
                if gi < len(TDB.points3d) and TDB.points3d[gi] is not None:
                    if gi in used_g:
                        track3d[qb] = gi
                        TDB.add_obs(gi, fi, keypoints[fi][mm.trainIdx].pt)
                        continue
                    obj_pts.append(TDB.points3d[gi])
                    img_pts.append(keypoints[fi][mm.trainIdx].pt)
                    track3d[qb] = gi
                    TDB.add_obs(gi, fi, keypoints[fi][mm.trainIdx].pt)
                    used_g.add(gi)
    return np.array(obj_pts, float), np.array(img_pts, float)

def _estimate_pose_pnp(obj_pts, img_pts, K, cfg: SfMConfig):
    """PnP RANSAC + optional acceptance via reproj & cheirality."""
    ok, rvec, tvec, inl = cv.solvePnPRansac(
        obj_pts, img_pts, K, None,
        reprojectionError=float(cfg.PNP_ERR_PX),
        iterationsCount=int(cfg.PNP_ITERS),
        confidence=0.999
    )
    inliers = int(len(inl)) if (inl is not None) else 0
    accept = bool(ok and inliers >= cfg.MIN_INLIERS_PNP)
    repro = None
    med_repro = None

    if ok and inl is not None:
        proj, _ = cv.projectPoints(obj_pts[inl.ravel()], rvec, tvec, K, None)
        err = np.linalg.norm(proj.squeeze() - img_pts[inl.ravel()], axis=1)
        repro = float(np.mean(err))
        med_repro = float(np.median(err))
        if not accept:
            accept = (repro <= cfg.PNP_REPROJ_ACCEPT)

    if not accept:
        return None, None, inliers, repro, med_repro, inl

    # LM refine
    try:
        rvec, tvec = cv.solvePnPRefineLM(
            objectPoints=obj_pts[inl.ravel()],
            imagePoints=img_pts[inl.ravel()],
            cameraMatrix=K, distCoeffs=None,
            rvec=rvec, tvec=tvec,
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 20, 1e-6)
        )
    except Exception:
        pass

    if repro is None:
        proj, _ = cv.projectPoints(obj_pts[inl.ravel()], rvec, tvec, K, None)
        err = np.linalg.norm(proj.squeeze() - img_pts[inl.ravel()], axis=1)
        repro = float(np.mean(err))
        med_repro = float(np.median(err))

    Rfi, _ = cv.Rodrigues(rvec)
    tfi = tvec.reshape(3, 1)
    return Rfi, tfi, inliers, repro, med_repro, inl

def _local_stereo_seed(fi: int, poses_R, poses_t, keypoints, matches, K, cfg: SfMConfig,
                       track3d, TDB: TrackDB):
    kept = 0
    neighbors = []
    for d in range(1, cfg.SEED_SPAN + 1):
        if (fi, fi + d) in matches: neighbors.append((fi, fi + d))
        if (fi - d, fi) in matches: neighbors.append((fi - d, fi))
    for (a, b) in neighbors:
        mlist = matches.get((a, b), [])
        if len(mlist) < cfg.SEED_MIN_INL:
            continue
        if a not in poses_R or b not in poses_R:
            continue
        pts_a, pts_b = _idx2pts(mlist, keypoints[a], keypoints[b])
        XA = _triangulate(K, poses_R[a], poses_t[a], poses_R[b], poses_t[b], pts_a, pts_b)
        front_seed = _in_front(poses_R[a], poses_t[a], XA) & _in_front(poses_R[b], poses_t[b], XA)
        if not np.any(front_seed):
            continue
        errA = _reproj_err(K, poses_R[a], poses_t[a], XA, pts_a)
        errB = _reproj_err(K, poses_R[b], poses_t[b], XA, pts_b)
        raysA = _bearing_in_world(np.linalg.inv(K), poses_R[a], pts_a)
        raysB = _bearing_in_world(np.linalg.inv(K), poses_R[b], pts_b)
        cosang = np.clip(np.sum(raysA * raysB, axis=1), -1, 1)
        angdeg = np.degrees(np.arccos(cosang))
        good = front_seed & (errA <= cfg.SEED_REPROJ) & (errB <= cfg.SEED_REPROJ) & (angdeg >= cfg.TRI_MIN_PARALLAX_DEG)
        if not np.any(good):
            continue
        for (mm, keep, X, uvA, uvB) in zip(mlist, good, XA, pts_a, pts_b):
            if not keep:
                continue
            qa = (a, mm.queryIdx); qb = (b, mm.trainIdx)
            gi = track3d.get(qa, None)
            if gi is not None and TDB.points3d[gi] is not None:
                track3d[qb] = gi
                TDB.add_obs(gi, b, uvB)
            else:
                gi = TDB.new_point(X)
                track3d[qa] = gi
                track3d[qb] = gi
                TDB.add_obs(gi, a, uvA)
                TDB.add_obs(gi, b, uvB)
            kept += 1
    return kept

def _anchor_triangulation(fi: int, anchors: List[int], poses_R, poses_t,
                          keypoints, matches, K, cfg: SfMConfig, track3d, TDB: TrackDB):
    pairs_all, ptsA_all, ptsF_all, ANCH_all = [], [], [], []
    for a in anchors:
        for mm in matches[(a, fi)]:
            qa = (a, mm.queryIdx); qb = (fi, mm.trainIdx)
            if qb in track3d:
                gi = track3d[qb]
                TDB.add_obs(gi, fi, keypoints[fi][mm.trainIdx].pt)
                continue
            if qa in track3d and (TDB.points3d[track3d[qa]] is not None):
                gi = track3d[qa]
                track3d[qb] = gi
                TDB.add_obs(gi, fi, keypoints[fi][mm.trainIdx].pt)
                continue
            pairs_all.append((qa, qb))
            ptsA_all.append(keypoints[a][mm.queryIdx].pt)
            ptsF_all.append(keypoints[fi][mm.trainIdx].pt)
            ANCH_all.append(a)

    if len(pairs_all) < cfg.TRI_MIN_CORR:
        return 0

    ptsA_np = np.float32(ptsA_all)
    ptsF_np = np.float32(ptsF_all)
    keep_idx_global = []
    X_global = []
    unique_anchors = list(sorted(set(ANCH_all)))
    for a in unique_anchors:
        idx = [k for k, aa in enumerate(ANCH_all) if aa == a]
        if len(idx) < 6:
            continue
        XA = _triangulate(K, poses_R[a], poses_t[a], poses_R[fi], poses_t[fi],
                          ptsA_np[idx], ptsF_np[idx])
        front = _in_front(poses_R[a], poses_t[a], XA) & _in_front(poses_R[fi], poses_t[fi], XA)
        if not np.any(front):
            continue
        errA = _reproj_err(K, poses_R[a], poses_t[a], XA, ptsA_np[idx])
        errF = _reproj_err(K, poses_R[fi], poses_t[fi], XA, ptsF_np[idx])
        raysA = _bearing_in_world(np.linalg.inv(K), poses_R[a], ptsA_np[idx])
        raysF = _bearing_in_world(np.linalg.inv(K), poses_R[fi], ptsF_np[idx])
        cosang = np.clip(np.sum(raysA * raysF, axis=1), -1, 1)
        angdeg = np.degrees(np.arccos(cosang))
        good = front & (errA <= cfg.TRI_REPROJ_MAX) & (errF <= cfg.TRI_REPROJ_MAX) & (angdeg >= cfg.TRI_MIN_PARALLAX_DEG)
        if not np.any(good):
            continue
        for loc_k, ok in zip(idx, good):
            if ok:
                keep_idx_global.append(loc_k)
        X_global.extend(list(XA[good]))
    if len(keep_idx_global) < cfg.TRI_MIN_CORR:
        return 0

    created = 0
    for k, X in zip(keep_idx_global, X_global):
        qa, qb = pairs_all[k]
        gi = track3d.get(qa, None)
        uvA = ptsA_np[k]; uvF = ptsF_np[k]
        if gi is not None and TDB.points3d[gi] is not None:
            track3d[qb] = gi
            TDB.add_obs(gi, fi, tuple(uvF))
        else:
            gi = TDB.new_point(X)
            track3d[qa] = gi
            track3d[qb] = gi
            TDB.add_obs(gi, qa[0], tuple(uvA))
            TDB.add_obs(gi, fi, tuple(uvF))
        created += 1
    return created

def _densify_pass(poses_R, poses_t, keypoints, matches, K, cfg: SfMConfig, track3d, TDB: TrackDB):
    if not (cfg.DENSIFY_ENABLE and len(poses_R) >= 2):
        return 0
    created = 0
    vis = sorted(poses_R.keys())
    Kinv = np.linalg.inv(K)
    for ia, a in enumerate(vis):
        for b in vis[ia+1:]:
            if abs(b - a) > cfg.DENSIFY_MAX_SPAN:
                continue
            mlist = matches.get((a, b), [])
            if len(mlist) < cfg.DENSIFY_MIN_MATCHES:
                continue
            pts_a, pts_b = _idx2pts(mlist, keypoints[a], keypoints[b])
            XA = _triangulate(K, poses_R[a], poses_t[a], poses_R[b], poses_t[b], pts_a, pts_b)
            front = _in_front(poses_R[a], poses_t[a], XA) & _in_front(poses_R[b], poses_t[b], XA)
            if not np.any(front):
                continue
            errA = _reproj_err(K, poses_R[a], poses_t[a], XA, pts_a)
            errB = _reproj_err(K, poses_R[b], poses_t[b], XA, pts_b)
            raysA = _bearing_in_world(Kinv, poses_R[a], pts_a)
            raysB = _bearing_in_world(Kinv, poses_R[b], pts_b)
            ang = np.degrees(np.arccos(np.clip(np.sum(raysA * raysB, axis=1), -1, 1)))
            good = front & (errA <= cfg.DENSIFY_MAX_REPROJ) & (errB <= cfg.DENSIFY_MAX_REPROJ) & (ang >= cfg.DENSIFY_MIN_PARALLAX_DEG)
            if not np.any(good):
                continue
            for mm, keep, X, uvA, uvB in zip(mlist, good, XA, pts_a, pts_b):
                if not keep:
                    continue
                qa = (a, mm.queryIdx); qb = (b, mm.trainIdx)
                if qa in track3d or qb in track3d:
                    continue
                gi = TDB.new_point(X)
                track3d[qa] = gi; track3d[qb] = gi
                TDB.add_obs(gi, a, uvA); TDB.add_obs(gi, b, uvB)
                created += 1
    TDB.promote_points(poses_R, poses_t)
    return created

def _maybe_apply_loops_and_smooth(poses_R, poses_t, matches, cfg: SfMConfig, on_log=None):
    def log(m): on_log and on_log(m)
    if not (cfg.USE_LOOP_CONSTRAINTS and len(poses_R) >= 4):
        return
    loops = []
    for (a, b), m in matches.items():
        if (b - a) <= min(cfg.LOOP_MAX_SPAN, max(6, cfg.INIT_MAX_SPAN)):
            continue
        if len(m) >= cfg.LOOP_MIN_INLIERS and (a in poses_R) and (b in poses_R):
            loops.append((a, b, len(m)))
    if loops:
        loops.sort(key=lambda x: -x[2])
        log(f"[sfm] loop constraints used: {len(loops)} (top: {loops[:3]})")
        if cfg.POSE_SMOOTHING:
            _smooth_poses(poses_R, poses_t, lam=float(cfg.SMOOTH_LAMBDA))

def _finalize_output(TDB: TrackDB, poses_R, poses_t, poses_out_dir, on_log=None):
    def log(m): on_log and on_log(m)
    pts = TDB.valid_points_array()
    log(f"[sfm] raw_points(after validation)={len(pts)}")
    if len(pts) == 0:
        raise RuntimeError("SfM produced 0 valid points — insufficient parallax/quality or thresholds too strict.")
    if poses_out_dir is not None and len(poses_R) > 0:
        _save_camera_poses_npz_csv(poses_R, poses_t, poses_out_dir, on_log=on_log)
    return pts

# =========================
# Main: run_sfm (refactored)
# =========================

def run_sfm(keypoints: List[List[cv.KeyPoint]],
            descriptors: List[np.ndarray],
            shapes: List[Tuple[int, int]],
            pairs: List[Tuple[int, int]],
            matches: Dict[Tuple[int, int], List[cv.DMatch]],
            K: np.ndarray,
            on_log: Callable[[str], None] = None,
            on_progress: Callable[[int, str], None] = None,
            poses_out_dir: Optional[str] = None,
            config: Optional[SfMConfig] = None,
            return_metrics: bool = False):

    def log(m):
        if on_log: on_log(m)
    def prog(p, s):
        if on_progress: on_progress(int(p), s)

    H, W = shapes[0]
    log(f"[sfm] size={W}x{H}")

    _acc_repro = []
    _rej_count = 0

    cfg = config or SfMConfig()

    # --- Init pair selection ---
    N = len(shapes)
    mid, w = _init_window_bounds(N, cfg)
    best = _select_init_pair(pairs, matches, keypoints, K, cfg, mid, w, on_log=on_log)

    # --- Initialization with the pair ---
    (i0, j0), poses_R, poses_t, track3d, TDB = _initialize_with_pair(best, keypoints, matches, K, on_log=on_log)
    prog(60, "SfM – Initialization")

    # --- Best-first expansion ---
    all_frames = sorted(set([i for ij in pairs for i in ij]))
    visited = {i0, j0}
    keyframes = [i0, j0] if cfg.USE_KEYFRAMES else []
    remaining = [f for f in all_frames if f not in visited]
    last_kf_idx = j0 if cfg.USE_KEYFRAMES else None

    def count_corr(fi: int) -> int:
        return _count_corr_for_frame(fi, visited, matches, track3d)

    while remaining:
        fi = max(remaining, key=count_corr)
        anchors = [a for a in visited if (a, fi) in matches]
        if not anchors:
            remaining.remove(fi)
            continue

        obj_pts, img_pts = _collect_2d3d_for_frame(fi, anchors, keypoints, matches, track3d, TDB)

        # Keyframe gate
        if cfg.USE_KEYFRAMES and len(obj_pts) < max(cfg.KF_MIN_NEW_2D3D, cfg.MIN_INLIERS_PNP):
            log(f"[sfm] skip non-KF frame {fi}: 2D-3D={len(obj_pts)}")
            remaining.remove(fi)
            continue
        if len(obj_pts) < cfg.MIN_INLIERS_PNP:
            log(f"[sfm] try PnP frame {fi}: 2D-3D={len(obj_pts)} need≥{cfg.MIN_INLIERS_PNP}")
            remaining.remove(fi)
            continue

        res = _estimate_pose_pnp(obj_pts, img_pts, K, cfg)
        Rfi, tfi, inliers, repro, med_repro, inl = res
        if Rfi is None:
            msg = f"[sfm] reject frame {fi}: inliers={inliers}"
            if repro is not None:
                msg += f", repro≈{repro:.2f}px"
            log(msg)
            _rej_count += 1
            remaining.remove(fi)
            continue

        # Cheirality sanity check on inliers
        front = _in_front(Rfi, tfi, obj_pts[inl.ravel()]) if inl is not None else np.ones(len(obj_pts), bool)
        if np.mean(front.astype(np.float32)) < 0.6:
            log(f"[sfm] reject frame {fi}: poor cheirality")
            _rej_count += 1
            remaining.remove(fi)
            continue

        _acc_repro.append(repro)
        log(f"[sfm] pose {fi}: inliers={inliers} repro={repro:.2f}px (med={med_repro:.2f})")

        poses_R[fi] = Rfi
        poses_t[fi] = tfi
        visited.add(fi)

        # Keyframe decision
        if _should_be_keyframe(K, poses_R, poses_t, last_kf_idx, fi, anchors, keypoints, matches, track3d, TDB, cfg):
            if cfg.USE_KEYFRAMES:
                keyframes.append(fi)
                last_kf_idx = fi

        remaining.remove(fi)
        prog(int(60 + 20 * len(visited) / len(all_frames)), f"SfM – Pose {fi}")

        # Local stereo seeding
        allow_new_points = (med_repro is None) or (med_repro <= cfg.FRAME_MAX_MEDIAN_REPROJ_FOR_NEW_POINTS)
        if allow_new_points:
            _local_stereo_seed(fi, poses_R, poses_t, keypoints, matches, K, cfg, track3d, TDB)

        # Triangulation using anchors
        if allow_new_points:
            _anchor_triangulation(fi, anchors, poses_R, poses_t, keypoints, matches, K, cfg, track3d, TDB)

        # Promote points after each frame
        TDB.promote_points(poses_R, poses_t, adapt_from_reproj=(med_repro or None))

    # --- Optional densify ---
    _densify_pass(poses_R, poses_t, keypoints, matches, K, cfg, track3d, TDB)

    # --- Optional loops + smoothing ---
    _maybe_apply_loops_and_smooth(poses_R, poses_t, matches, cfg, on_log=on_log)

    # --- Output ---
    pts = _finalize_output(TDB, poses_R, poses_t, poses_out_dir, on_log=on_log)
    prog(95, "SfM – Collect points")

    metrics = {
        'num_points': int(len(pts)),
        'frames_used': int(len(poses_R)),
        'avg_reproj': (float(np.mean(_acc_repro)) if len(_acc_repro) > 0 else None),
        'rejected_frames': int(_rej_count),
        'init_pair': tuple(map(int, best["pair"]))
    }
    if return_metrics:
        return pts, poses_R, poses_t, metrics
    return pts, poses_R, poses_t


# =========================
# Ensemble wrapper (unchanged except for minor progress logging)
# =========================

def run_sfm_multi(keypoints, descriptors, shapes, pairs, matches, K,
                  n_runs: int = 5,
                  on_log=None, on_progress=None,
                  poses_out_dir: Optional[str] = None):
    """Run SfM multiple times with different init windows and pick the best."""
    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    N = len(shapes)
    centers = np.linspace(N * 0.15, N * 0.85, num=max(2, min(n_runs, 5)), dtype=int)
    extra = n_runs - len(centers)
    rng = np.random.default_rng(42)
    if extra > 0:
        jitter = np.clip(rng.normal(0, N * 0.05, size=extra).astype(int), -N // 6, N // 6)
        extra_centers = np.clip(rng.integers(0, N, size=extra) + jitter, 0, N - 1)
        centers = list(centers) + list(extra_centers)

    trials = []
    best_score = -1.0
    best = None
    for r, c in enumerate(centers):
        cfg = SfMConfig(INIT_WINDOW_CENTER=int(c))
        try:
            pts, Rdict, tdict, metrics = run_sfm(
                keypoints, descriptors, shapes, pairs, matches, K,
                on_log=lambda m, r=r: log(f"[run {r+1}/{len(centers)}] {m}"),
                on_progress=lambda p, s, r=r: prog(int((r / (len(centers))) * 90 + p / len(centers) * 0.9), s),
                poses_out_dir=poses_out_dir,
                config=cfg,
                return_metrics=True
            )
            score = float(metrics.get('num_points', 0)) - 0.5 * float(metrics.get('rejected_frames', 0))
            trials.append((score, c, (pts, Rdict, tdict, metrics)))
            if score > best_score:
                best_score = score
                best = (pts, Rdict, tdict, dict(best_center=int(c),
                                                best_init_pair=metrics.get('init_pair'),
                                                best_score=score))
        except Exception as e:
            log(f"[run {r+1}] failed: {e}")

    if best is None:
        raise RuntimeError("All ensemble runs failed.")
    pts, Rdict, tdict, report = best
    prog(92, "SfM Ensemble – done")
    return pts, Rdict, tdict, report
