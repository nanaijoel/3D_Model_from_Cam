import os
import numpy as np
import cv2 as cv
from typing import Callable, List, Dict, Tuple, Optional

def _idx2pts(matches, kps1, kps2):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def _save_camera_poses_npz_csv(poses_R: Dict[int, np.ndarray],
                               poses_t: Dict[int, np.ndarray],
                               out_dir: str,
                               on_log: Optional[Callable[[str], None]] = None):
    """Speichert Kamera-Posen als .npz (inkl. Zentren C) + .csv in out_dir."""
    def log(m):
        if on_log: on_log(m)
    os.makedirs(out_dir, exist_ok=True)
    order = sorted(poses_R.keys())
    Rs, ts, idxs, Cs = [], [], [], []
    for i in order:
        R = poses_R[i]; t = poses_t[i]
        C = -R.T @ t  # Kamera-Zentrum in Weltkoordinaten
        Rs.append(R); ts.append(t); idxs.append(i); Cs.append(C.reshape(3))
    Rs = np.stack(Rs, axis=0) if len(Rs) else np.zeros((0,3,3))
    ts = np.stack(ts, axis=0) if len(ts) else np.zeros((0,3,1))
    Cs = np.stack(Cs, axis=0) if len(Cs) else np.zeros((0,3))
    idxs = np.array(idxs, dtype=int)

    npz_path = os.path.join(out_dir, "camera_poses.npz")
    np.savez_compressed(npz_path, frame_idx=idxs, R=Rs, t=ts, C=Cs)
    log(f"[sfm] camera poses saved -> {npz_path}")

    # CSV zusätzlich (für schnellen Blick)
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

def run_sfm(keypoints: List[List[cv.KeyPoint]],
            descriptors: List[np.ndarray],
            shapes: List[Tuple[int,int]],
            pairs: List[Tuple[int,int]],
            matches: Dict[Tuple[int,int], List[cv.DMatch]],
            K: np.ndarray,
            on_log: Callable[[str], None] = None,
            on_progress: Callable[[int,str], None] = None,
            poses_out_dir: Optional[str] = None):
    """Sehr kompakte inkrementelle SfM-Variante (SIFT/CPU)."""
    def log(m):
        if on_log: on_log(m)
    def prog(p, s):
        if on_progress: on_progress(int(p), s)

    H, W = shapes[0]
    log(f"[sfm] size={W}x{H}")

    # ---------- 1) Initialpaar automatisch wählen ----------
    MIN_MATCHES_INIT = 60
    MIN_FLOW_PX = 8.0
    MIN_INLIERS_INIT = 40

    def cheirality_count(R, t, pts1, pts2):
        def triangulate(R1, t1, R2, t2, p1, p2):
            P1 = K @ np.hstack((R1, t1));
            P2 = K @ np.hstack((R2, t2))
            X = cv.triangulatePoints(P1, P2, p1.T, p2.T)
            X = (X[:3] / X[3]).T
            return X
        def in_front(R, t, X):
            Xc = (R @ X.T + t).T
            return Xc[:, 2] > 0
        X = triangulate(np.eye(3), np.zeros((3, 1)), R, t, pts1, pts2)
        front = in_front(np.eye(3), np.zeros((3, 1)), X) & in_front(R, t, X)
        return int(front.sum())

    best = dict(score=-1.0, pair=None, R=None, t=None, mask=None)
    stats = []

    for (i, j) in pairs:
        m = matches.get((i, j), [])
        if len(m) < MIN_MATCHES_INIT: continue
        p1, p2 = _idx2pts(m, keypoints[i], keypoints[j])
        flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        if flow < MIN_FLOW_PX: continue

        E, maskE = cv.findEssentialMat(p1, p2, K, method=cv.RANSAC, prob=0.999, threshold=1.5)
        if E is None: continue
        _, R, t, maskP = cv.recoverPose(E, p1, p2, K)
        inl_mask = (maskP.ravel().astype(bool)) & (maskE.ravel().astype(bool)[:len(maskP)])
        inliers = int(inl_mask.sum())
        if inliers < MIN_INLIERS_INIT: continue

        ch = cheirality_count(R, t, p1[inl_mask], p2[inl_mask]) if inliers > 0 else 0
        score = inliers + 2.0 * ch
        stats.append((i, j, len(m), flow, inliers, ch, score))
        if score > best["score"]:
            best.update(score=score, pair=(i, j), R=R, t=t, mask=inl_mask)

    if best["pair"] is None:
        for (i, j) in pairs:
            m = matches.get((i, j), [])
            if len(m) < 40: continue
            p1, p2 = _idx2pts(m, keypoints[i], keypoints[j])
            flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
            E, maskE = cv.findEssentialMat(p1, p2, K, method=cv.RANSAC, prob=0.999, threshold=2.0)
            if E is None: continue
            _, R, t, maskP = cv.recoverPose(E, p1, p2, K)
            inl_mask = (maskP.ravel().astype(bool)) & (maskE.ravel().astype(bool)[:len(maskP)])
            inliers = int(inl_mask.sum())
            ch = cheirality_count(R, t, p1[inl_mask], p2[inl_mask]) if inliers > 0 else 0
            score = inliers + 2.0 * ch
            stats.append((i, j, len(m), flow, inliers, ch, score))
            if ch > 0 and score > best["score"]:
                best.update(score=score, pair=(i, j), R=R, t=t, mask=inl_mask)

    if best["pair"] is None:
        if stats:
            stats.sort(key=lambda x: x[-1], reverse=True)
            top = "\n".join([f"   cand ({a},{b}): matches={mm}, flow={fl:.1f}px, inl={ii}, cheiral={ch}, score={sc:.1f}"
                             for (a, b, mm, fl, ii, ch, sc) in stats[:5]])
            log("[sfm] Init-Kandidaten TOP5:\n" + top)
        raise RuntimeError("[sfm] Initialisierung fehlgeschlagen (keine Basis).")

    i0, j0 = best["pair"]
    log(f"[sfm] init pair = ({i0},{j0}) score={best['score']:.1f}")
    kps0, kps1 = keypoints[i0], keypoints[j0]
    m01 = matches[(i0, j0)]
    p1, p2 = _idx2pts(m01, kps0, kps1)
    R = best["R"]; t = best["t"]; mask = best["mask"]

    # ---------- 2) Triangulation Init ----------
    poses_R = {i0: np.eye(3), j0: R}
    poses_t = {i0: np.zeros((3, 1)), j0: t}

    def triangulate(R1, t1, R2, t2, pts1, pts2):
        P1 = K @ np.hstack((R1, t1))
        P2 = K @ np.hstack((R2, t2))
        X = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X = (X[:3] / X[3]).T
        return X

    def in_front(R, t, X):
        Xc = (R @ X.T + t).T
        return Xc[:, 2] > 0

    p1_in = p1[mask]; p2_in = p2[mask]
    X_init = triangulate(poses_R[i0], poses_t[i0], poses_R[j0], poses_t[j0], p1_in, p2_in)
    front = in_front(poses_R[i0], poses_t[i0], X_init) & in_front(poses_R[j0], poses_t[j0], X_init)

    if not np.any(front):
        R_inv = R.T
        t_inv = -R_inv @ t
        X_try = triangulate(poses_R[i0], poses_t[i0], R_inv, t_inv, p1_in, p2_in)
        front2 = in_front(poses_R[i0], poses_t[i0], X_try) & in_front(R_inv, t_inv, X_try)
        if np.any(front2):
            R, t = R_inv, t_inv
            poses_R[j0], poses_t[j0] = R, t
            X_init, front = X_try, front2

    X_init = X_init[front]
    log(f"[sfm] init inliers={int(mask.sum())} cheiral={int(front.sum())}")
    log(f"[sfm] X_init={len(X_init)}")
    prog(60, "SfM – Initialisierung")

    # Track-Map
    track3d: Dict[Tuple[int,int], int] = {}
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

    # ---------- 3) Inkrementell ----------
    order = sorted(set([i for ij in pairs for i in ij]))
    visited = {i0, j0}
    min_inliers_pnp = 40  # war 80
    pnp_reproj_accept = 1.6  # Reproj.-Fehler, um knappe Fälle zuzulassen
    pnp_iters = 6000  # mehr RANSAC-Iterationen
    pnp_err = 3.0  # war 2.5

    for step, fi in enumerate(order, start=1):
        if fi in visited:
            continue
        anchors = [a for a in visited if (a, fi) in matches]
        if not anchors:
            continue

        obj_pts, img_pts = [], []
        for a in anchors:
            mlist = matches[(a, fi)]
            for mm in mlist:
                qa = (a, mm.queryIdx);
                qb = (fi, mm.trainIdx)
                if qa in track3d:
                    gi = track3d[qa]
                    if gi < len(points3d) and points3d[gi] is not None:
                        obj_pts.append(points3d[gi])
                        img_pts.append(keypoints[fi][mm.trainIdx].pt)
                        track3d[qb] = gi

        if len(obj_pts) < min_inliers_pnp:
            log(f"[sfm] try PnP frame {fi}: 2D-3D={len(obj_pts)} need≥{min_inliers_pnp}")
            continue

        obj_pts = np.array(obj_pts, dtype=float)
        img_pts = np.array(img_pts, dtype=float)

        ok, rvec, tvec, inl = cv.solvePnPRansac(
            obj_pts, img_pts, K, None,
            reprojectionError=float(pnp_err),
            iterationsCount=int(pnp_iters),
            confidence=0.999
        )
        inliers = int(len(inl)) if (inl is not None) else 0
        accept = bool(ok and inliers >= min_inliers_pnp)
        repro = None

        # knappe Fälle zulassen: genug Inlier + sehr kleiner Reproj.-Fehler
        if not accept and ok and inliers >= 25 and inl is not None:
            proj, _ = cv.projectPoints(obj_pts[inl.ravel()], rvec, tvec, K, None)
            repro = float(np.mean(np.linalg.norm(proj.squeeze() - img_pts[inl.ravel()], axis=1)))
            accept = (repro <= pnp_reproj_accept)

        if not accept:
            if repro is None:
                log(f"[sfm] reject frame {fi}: inliers={inliers} (<{min_inliers_pnp})")
            else:
                log(f"[sfm] reject frame {fi}: inliers={inliers}, repro≈{repro:.2f}px")
            continue

        # optionales Refinement
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

        prog(int(60 + 30 * len(visited) / len(order)), f"SfM – Pose {fi}")

        # neue Punkte triangulieren (kleine Lockerung hilft)
        for a in anchors:
            mlist = matches[(a, fi)]
            pts_a, pts_f, pairs_af = [], [], []
            for mm in mlist:
                qa = (a, mm.queryIdx);
                qb = (fi, mm.trainIdx)
                if qb in track3d:
                    continue
                pts_a.append(keypoints[a][mm.queryIdx].pt)
                pts_f.append(keypoints[fi][mm.trainIdx].pt)
                pairs_af.append((qa, qb))

            if len(pts_a) < 25:  # war 40
                continue

            Xaf = triangulate(poses_R[a], poses_t[a], poses_R[fi], poses_t[fi],
                              np.float32(pts_a), np.float32(pts_f))
            front_af = in_front(poses_R[a], poses_t[a], Xaf) & in_front(poses_R[fi], poses_t[fi], Xaf)
            Xaf = Xaf[front_af]

            k = 0
            for qa, qb in pairs_af:
                gi = track3d.get(qa, None)
                if gi is not None and gi < len(points3d) and points3d[gi] is not None:
                    track3d[qb] = gi
                    continue
                gi = len(points3d)
                track3d[qa] = gi
                track3d[qb] = gi
                if k < len(Xaf):
                    points3d.append(Xaf[k]);
                    k += 1
                else:
                    points3d.append(None)

    # ---------- 4) Rückgabe + Pose-Datei schreiben ----------
    pts = np.array([p for p in points3d if p is not None], dtype=float)
    log(f"[sfm] raw_points={len(pts)}")
    if len(pts) == 0:
        raise RuntimeError("SfM erzeugte 0 Punkte. Mehr Parallaxe/Frames und gute Textur nötig.")
    prog(95, "SfM – Punkte sammeln")

    # Speichern der Kameraposen (falls Verzeichnis angegeben)
    if poses_out_dir is not None and len(poses_R) > 0:
        _save_camera_poses_npz_csv(poses_R, poses_t, poses_out_dir, on_log=on_log)

    return pts, poses_R, poses_t