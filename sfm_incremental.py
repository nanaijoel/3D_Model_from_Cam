# sfm_incremental.py
import os
import numpy as np
import cv2 as cv
from typing import Callable, List, Dict, Tuple, Optional

# --------------------------- helpers ---------------------------------
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
        C = -R.T @ t  # camera center in world
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

# --------------------------- core geometry ----------------------------
def _triangulate(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    X = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X[:3] / X[3]).T
    return X

def _in_front(R, t, X):
    Xc = (R @ X.T + t).T
    return Xc[:, 2] > 0

def _cheirality_count(K, R, t, pts1, pts2):
    X = _triangulate(K, np.eye(3), np.zeros((3, 1)), R, t, pts1, pts2)
    front = _in_front(np.eye(3), np.zeros((3, 1)), X) & _in_front(R, t, X)
    return int(front.sum())

def _reproj_err(K, R, t, X, pts2d):
    rvec, _ = cv.Rodrigues(R)
    proj, _ = cv.projectPoints(X, rvec, t, K, None)
    proj = proj.squeeze()
    return np.linalg.norm(proj - pts2d, axis=1)

# --------------------------- main ------------------------------------
def run_sfm(keypoints: List[List[cv.KeyPoint]],
            descriptors: List[np.ndarray],
            shapes: List[Tuple[int, int]],
            pairs: List[Tuple[int, int]],
            matches: Dict[Tuple[int, int], List[cv.DMatch]],
            K: np.ndarray,
            on_log: Callable[[str], None] = None,
            on_progress: Callable[[int, str], None] = None,
            poses_out_dir: Optional[str] = None):

    def log(m):
        if on_log: on_log(m)
    def prog(p, s):
        if on_progress: on_progress(int(p), s)

    H, W = shapes[0]
    log(f"[sfm] size={W}x{H}")

    # --------- Parameter (robust ggü. N) ------------------------------
    # Init
    MIN_MATCHES_INIT = 50
    MIN_FLOW_PX = 6.0
    MIN_INLIERS_INIT = 35
    INIT_WINDOW_FRAMES = 80         # Fenster um Sequenzmitte für Init-Paare
    INIT_MAX_SPAN = 6               # nur lokale Nachbarschaft als Basis

    # PnP
    MIN_INLIERS_PNP = 28
    PNP_ITERS = 8000
    PNP_ERR_PX = 3.0
    PNP_REPROJ_ACCEPT = 2.0

    # Triangulation
    TRI_MIN_CORR = 14               # globale Schwelle (kombiniert über alle Anker)
    TRI_REPROJ_MAX = 2.5
    TRI_MIN_PARALLAX_DEG = 2.0

    # Local stereo seeding
    SEED_SPAN = 3                   # fi±1..±3
    SEED_MIN_INL = 40               # pro Nachbarpaar
    SEED_REPROJ = 2.0

    # --------- Init-Paar lokal um die Mitte ---------------------------
    N = len(shapes)
    mid = N // 4
    w = min(INIT_WINDOW_FRAMES, max(10, N // 3))
    def in_init_window(i, j):
        return (abs(j - i) <= INIT_MAX_SPAN) and (min(i, j) >= mid - w) and (max(i, j) <= mid + w)

    best = dict(score=-1.0, pair=None, R=None, t=None, mask=None)
    stats = []

    for (i, j) in pairs:
        if not in_init_window(i, j):
            continue
        m = matches.get((i, j), [])
        if len(m) < MIN_MATCHES_INIT:
            continue
        p1, p2 = _idx2pts(m, keypoints[i], keypoints[j])
        flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        if flow < MIN_FLOW_PX:
            continue
        E, maskE = cv.findEssentialMat(p1, p2, K, method=cv.RANSAC, prob=0.999, threshold=1.5)
        if E is None:
            continue
        _, R, t, maskP = cv.recoverPose(E, p1, p2, K)
        inl_mask = (maskP.ravel().astype(bool)) & (maskE.ravel().astype(bool)[:len(maskP)])
        inliers = int(inl_mask.sum())
        if inliers < MIN_INLIERS_INIT:
            continue
        ch = _cheirality_count(K, R, t, p1[inl_mask], p2[inl_mask]) if inliers > 0 else 0
        score = inliers + 2.0 * ch
        stats.append((i, j, len(m), flow, inliers, ch, score))
        if score > best["score"]:
            best.update(score=score, pair=(i, j), R=R, t=t, mask=inl_mask)

    if best["pair"] is None:
        for (i, j) in pairs:
            if not in_init_window(i, j):
                continue
            m = matches.get((i, j), [])
            if len(m) < 40:
                continue
            p1, p2 = _idx2pts(m, keypoints[i], keypoints[j])
            E, maskE = cv.findEssentialMat(p1, p2, K, method=cv.RANSAC, prob=0.999, threshold=2.0)
            if E is None:
                continue
            _, R, t, maskP = cv.recoverPose(E, p1, p2, K)
            inl_mask = (maskP.ravel().astype(bool)) & (maskE.ravel().astype(bool)[:len(maskP)])
            inliers = int(inl_mask.sum())
            ch = _cheirality_count(K, R, t, p1[inl_mask], p2[inl_mask]) if inliers > 0 else 0
            score = inliers + 2.0 * ch
            stats.append((i, j, len(m), 0.0, inliers, ch, score))
            if ch > 0 and score > best["score"]:
                best.update(score=score, pair=(i, j), R=R, t=t, mask=inl_mask)

    if best["pair"] is None:
        if stats:
            stats.sort(key=lambda x: x[-1], reverse=True)
            top = "\n".join([f"   cand ({a},{b}): matches={mm}, inl={ii}, cheiral={ch}, score={sc:.1f}"
                             for (a, b, mm, _, ii, ch, sc) in stats[:8]])
            log("[sfm] Init-Kandidaten TOP:\n" + top)
        raise RuntimeError("[sfm] Initialisierung fehlgeschlagen.")

    i0, j0 = best["pair"]
    log(f"[sfm] init pair = ({i0},{j0}) score={best['score']:.1f}")
    kps0, kps1 = keypoints[i0], keypoints[j0]
    m01 = matches[(i0, j0)]
    p1_all, p2_all = _idx2pts(m01, kps0, kps1)
    R = best["R"]; t = best["t"]; mask = best["mask"]

    # --------- init triangulation ------------------------------------
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
    log(f"[sfm] init inliers={int(mask.sum())} cheiral={int(front.sum())}")
    log(f"[sfm] X_init={len(X_init)}")
    prog(60, "SfM – Initialisierung")

    # Track-Map
    track3d: Dict[Tuple[int, int], int] = {}
    points3d: List[np.ndarray] = []

    idx_inliers = np.where(mask)[0][front]
    for k, mi in enumerate(idx_inliers):
        mm = m01[mi]
        qa = (i0, mm.queryIdx)
        qb = (j0, mm.trainIdx)
        gi = len(points3d)
        track3d[qa] = gi
        track3d[qb] = gi
        points3d.append(X_init[k])

    # --------- best-first expansion -----------------------------------
    all_frames = sorted(set([i for ij in pairs for i in ij]))
    visited = {i0, j0}

    def count_corr(fi: int) -> int:
        c = 0
        for a in visited:
            if (a, fi) not in matches:
                continue
            for mm in matches[(a, fi)]:
                if (a, mm.queryIdx) in track3d:
                    c += 1
        return c

    remaining = [f for f in all_frames if f not in visited]

    while remaining:
        fi = max(remaining, key=count_corr)
        anchors = [a for a in visited if (a, fi) in matches]
        if not anchors:
            remaining.remove(fi)
            continue

        # ---------- 2D-3D farming ------------------------------------
        obj_pts, img_pts = [], []
        used_g = set()
        for a in anchors:
            for mm in matches[(a, fi)]:
                qa = (a, mm.queryIdx); qb = (fi, mm.trainIdx)
                if qa in track3d:
                    gi = track3d[qa]
                    if gi < len(points3d) and points3d[gi] is not None:
                        if gi in used_g:
                            track3d[qb] = gi
                            continue
                        obj_pts.append(points3d[gi])
                        img_pts.append(keypoints[fi][mm.trainIdx].pt)
                        track3d[qb] = gi
                        used_g.add(gi)

        if len(obj_pts) < MIN_INLIERS_PNP:
            log(f"[sfm] try PnP frame {fi}: 2D-3D={len(obj_pts)} need≥{MIN_INLIERS_PNP}")
            remaining.remove(fi)
            continue

        obj_pts = np.array(obj_pts, dtype=float)
        img_pts = np.array(img_pts, dtype=float)

        ok, rvec, tvec, inl = cv.solvePnPRansac(
            obj_pts, img_pts, K, None,
            reprojectionError=float(PNP_ERR_PX),
            iterationsCount=int(PNP_ITERS),
            confidence=0.999
        )
        inliers = int(len(inl)) if (inl is not None) else 0
        accept = bool(ok and inliers >= MIN_INLIERS_PNP)
        repro = None

        if ok and inl is not None:
            proj, _ = cv.projectPoints(obj_pts[inl.ravel()], rvec, tvec, K, None)
            repro = float(np.mean(np.linalg.norm(proj.squeeze() - img_pts[inl.ravel()], axis=1)))
            if not accept:
                accept = (repro <= PNP_REPROJ_ACCEPT)

        if accept:
            Rfi, _ = cv.Rodrigues(rvec); tfi = tvec.reshape(3, 1)
            front = _in_front(Rfi, tfi, obj_pts[inl.ravel()]) if inl is not None else np.ones(len(obj_pts), bool)
            if np.mean(front.astype(np.float32)) < 0.6:
                accept = False

        if not accept:
            msg = f"[sfm] reject frame {fi}: inliers={inliers}"
            if repro is not None:
                msg += f", repro≈{repro:.2f}px"
            log(msg)
            remaining.remove(fi)
            continue

        # refine
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

        Rfi, _ = cv.Rodrigues(rvec)
        tfi = tvec.reshape(3, 1)
        if repro is None:
            proj, _ = cv.projectPoints(obj_pts[inl.ravel()], rvec, tvec, K, None)
            repro = float(np.mean(np.linalg.norm(proj.squeeze() - img_pts[inl.ravel()], axis=1)))
        log(f"[sfm] pose {fi}: inliers={inliers} repro={repro:.2f}px")

        poses_R[fi] = Rfi
        poses_t[fi] = tfi
        visited.add(fi)
        remaining.remove(fi)
        prog(int(60 + 30 * len(visited) / len(all_frames)), f"SfM – Pose {fi}")

        # ---------- Local stereo seeding (frische Punkte vorne) --------
        neighbors = []
        for d in range(1, SEED_SPAN + 1):
            if (fi, fi + d) in matches: neighbors.append((fi, fi + d))
            if (fi - d, fi) in matches: neighbors.append((fi - d, fi))
        for (a, b) in neighbors:
            mlist = matches.get((a, b), [])
            if len(mlist) < SEED_MIN_INL:
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
            good = front_seed & (errA <= SEED_REPROJ) & (errB <= SEED_REPROJ)
            if not np.any(good):
                continue
            for (mm, keep, X) in zip(mlist, good, XA):
                if not keep:
                    continue
                qa = (a, mm.queryIdx); qb = (b, mm.trainIdx)
                gi = track3d.get(qa, None)
                if gi is not None and gi < len(points3d) and points3d[gi] is not None:
                    track3d[qb] = gi
                else:
                    gi = len(points3d)
                    track3d[qa] = gi
                    track3d[qb] = gi
                    points3d.append(X)

        # ---------- Triangulation (kombiniert über alle Anker) ---------
        pairs_all, ptsA_all, ptsF_all, ANCH_all = [], [], [], []
        for a in anchors:
            for mm in matches[(a, fi)]:
                qa = (a, mm.queryIdx); qb = (fi, mm.trainIdx)
                if qb in track3d:
                    continue
                if qa in track3d and (track3d[qa] < len(points3d)) and (points3d[track3d[qa]] is not None):
                    track3d[qb] = track3d[qa]
                    continue
                pairs_all.append((qa, qb))
                ptsA_all.append(keypoints[a][mm.queryIdx].pt)
                ptsF_all.append(keypoints[fi][mm.trainIdx].pt)
                ANCH_all.append(a)

        if len(pairs_all) >= TRI_MIN_CORR:
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
                repro_mask = (errA <= TRI_REPROJ_MAX) & (errF <= TRI_REPROJ_MAX)
                Kinv = np.linalg.inv(K)
                rayA = (Kinv @ np.hstack([ptsA_np[idx], np.ones((len(idx), 1))]).T).T
                rayF = (Kinv @ np.hstack([ptsF_np[idx], np.ones((len(idx), 1))]).T).T
                rayA /= np.linalg.norm(rayA, axis=1, keepdims=True)
                rayF /= np.linalg.norm(rayF, axis=1, keepdims=True)
                cosang = np.sum(rayA * rayF, axis=1).clip(-1, 1)
                angdeg = np.degrees(np.arccos(cosang))
                para_mask = angdeg > TRI_MIN_PARALLAX_DEG
                good = front & repro_mask & para_mask
                if not np.any(good):
                    continue
                for loc_k, ok in zip(idx, good):
                    if ok:
                        keep_idx_global.append(loc_k)
                X_global.extend(list(XA[good]))
            if len(keep_idx_global) >= TRI_MIN_CORR:
                for k, X in zip(keep_idx_global, X_global):
                    qa, qb = pairs_all[k]
                    gi = track3d.get(qa, None)
                    if gi is not None and gi < len(points3d) and points3d[gi] is not None:
                        track3d[qb] = gi
                    else:
                        gi = len(points3d)
                        track3d[qa] = gi
                        track3d[qb] = gi
                        points3d.append(X)

    # --------- output --------------------------------------------------
    pts = np.array([p for p in points3d if p is not None], dtype=float)
    log(f"[sfm] raw_points={len(pts)}")
    if len(pts) == 0:
        raise RuntimeError("SfM erzeugte 0 Punkte. Mehr Parallaxe/Frames und gute Textur nötig.")
    prog(95, "SfM – Punkte sammeln")

    if poses_out_dir is not None and len(poses_R) > 0:
        _save_camera_poses_npz_csv(poses_R, poses_t, poses_out_dir, on_log=on_log)

    return pts, poses_R, poses_t
