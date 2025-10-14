import os
from typing import Callable, List, Tuple, Optional
import inspect
import cv2 as cv
import numpy as np
import torch


# ---------------- LightGlue-Extractor (SP/DISK/ALIKED) ----------------
def _make_lg_extractor(kind: str, device: str, max_kp: int, log=print):
    det_thr = float(os.getenv("FEATURE_LG_DET_THR", "0.001"))
    nms_r   = int(float(os.getenv("FEATURE_LG_NMS", "3")))
    resize  = int(float(os.getenv("FEATURE_LG_RESIZE", "1024")))
    rm_b    = int(float(os.getenv("FEATURE_LG_REMOVE_BORDERS", "2")))
    force_n = os.getenv("FEATURE_LG_FORCE_NUM", "0").lower() in ("1","true","yes")

    conf = dict(
        max_num_keypoints=max_kp,
        detection_threshold=det_thr,
        nms_radius=nms_r,
        resize=resize,
        remove_borders=rm_b,
        force_num_keypoints=force_n,
    )

    from lightglue import SuperPoint, DISK, ALIKED
    cls = {"superpoint": SuperPoint, "disk": DISK, "aliked": ALIKED}[kind]
    sig = inspect.signature(cls.__init__)
    conf = {k: v for k, v in conf.items() if k in sig.parameters}
    extractor = cls(**conf).to(device).eval()
    log(f"[features] {kind.upper()} on {device} (thr={det_thr}, nms={nms_r}, resize={resize}, force={force_n})")
    return extractor



def _rootsift(des: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if des is None or des.size == 0: return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)

def _kp_to_np(kps: List[cv.KeyPoint]) -> np.ndarray:
    out = np.zeros((len(kps), 7), np.float32)
    for i, k in enumerate(kps):
        out[i] = (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
    return out

def _to_tensor(gray: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(gray).float() / 255.0
    return t[None, None, :, :]

def _sp_to_cv_kp(sp_kp: torch.Tensor, scores: torch.Tensor) -> List[cv.KeyPoint]:
    kps: List[cv.KeyPoint] = []
    xy = sp_kp.detach().cpu().numpy()
    sc = scores.detach().cpu().numpy()
    for (x, y), s in zip(xy, sc):
        kps.append(cv.KeyPoint(float(x), float(y), 3.0, -1.0, float(s), 0, -1))
    return kps

def _ensure_desc_is_NC(des_np: np.ndarray, n_kp: int) -> np.ndarray:
    if des_np.ndim != 2: return des_np
    if des_np.shape[0] in (64, 96, 128, 256) and des_np.shape[1] == n_kp:
        return des_np.T
    return des_np

def _filter_kp_des_scores(kps, des, scores, mask, dilate=3):
    if mask is None or len(kps) == 0:
        return kps, des, scores
    if dilate > 0:
        mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate)))
    h, w = mask.shape[:2]
    keep = []
    for i, kp in enumerate(kps):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep.append(i)
    if not keep:
        empty_des = None if des is None else np.empty((0, des.shape[1]), des.dtype)
        empty_sc = None if scores is None else np.empty((0,), np.float32)
        return [], empty_des, empty_sc
    keep = np.asarray(keep, np.int32)
    kps_f = [kps[i] for i in keep]
    des_f = None if des is None else des[keep]
    sc_f  = None if scores is None else scores[keep]
    return kps_f, des_f, sc_f

def _boost_gray(gray0: np.ndarray) -> np.ndarray:
    use_clahe   = os.getenv("FEATURE_PREPROC_CLAHE", "1").lower() in ("1","true","yes")
    use_unsharp = os.getenv("FEATURE_PREPROC_UNSHARP", "1").lower() in ("1","true","yes")
    use_noise   = os.getenv("FEATURE_PREPROC_NOISE", "0").lower() in ("1","true","yes")

    gray = gray0.copy()
    if use_clahe:
        clip = float(os.getenv("FEATURE_PREPROC_CLAHE_CLIP", "3.0"))
        tiles = int(float(os.getenv("FEATURE_PREPROC_CLAHE_TILES", "8")))
        clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        gray  = clahe.apply(gray)
    if use_unsharp:
        sigma = float(os.getenv("FEATURE_PREPROC_UNSHARP_SIGMA", "1.0"))
        amount= float(os.getenv("FEATURE_PREPROC_UNSHARP_AMOUNT", "1.5"))
        blur  = cv.GaussianBlur(gray, (0,0), sigma)
        gray  = cv.addWeighted(gray, amount, blur, -(amount-1.0), 0)
    if use_noise:
        amp = float(os.getenv("FEATURE_PREPROC_NOISE_STD", "1.5"))
        noise = (np.random.randn(*gray.shape).astype(np.float32) * amp)
        gray = cv.add(gray.astype(np.float32), noise, dtype=cv.CV_8U)
    return gray

def _boost_gray_strong(gray0: np.ndarray) -> np.ndarray:
    """stärkerer Pass nur für Restmaske: aufhellen + stärkeres CLAHE + unsharp"""
    gray = gray0.copy()
    # Gamma < 1 → heller
    gamma = float(os.getenv("FEATURE_FILL_GAMMA", "0.75"))
    x = (gray.astype(np.float32) / 255.0) ** gamma
    gray = (x * 255.0).clip(0,255).astype(np.uint8)

    # Stärkeres CLAHE
    clahe = cv.createCLAHE(
        clipLimit=float(os.getenv("FEATURE_FILL_CLAHE_CLIP", "6.0")),
        tileGridSize=(int(float(os.getenv("FEATURE_FILL_CLAHE_TILES","8"))),) * 2
    )
    gray = clahe.apply(gray)

    # Unsharp stärker
    sigma  = float(os.getenv("FEATURE_FILL_UNSHARP_SIGMA", "1.2"))
    amount = float(os.getenv("FEATURE_FILL_UNSHARP_AMOUNT", "1.8"))
    blur = cv.GaussianBlur(gray, (0,0), sigma)
    gray = cv.addWeighted(gray, amount, blur, -(amount-1.0), 0)

    if os.getenv("FEATURE_FILL_NOISE","0").lower() in ("1","true","yes"):
        amp = float(os.getenv("FEATURE_FILL_NOISE_STD", "2.0"))
        noise = (np.random.randn(*gray.shape).astype(np.float32) * amp)
        gray = cv.add(gray.astype(np.float32), noise, dtype=cv.CV_8U)
    return gray

def _density_mask_from_kps(shape_hw, kps_xy: np.ndarray, kp_radius=4, blur=21, thr=0.05):
    h, w = shape_hw
    dens = np.zeros((h, w), np.float32)
    for (x, y) in kps_xy:
        cv.circle(dens, (int(x), int(y)), int(kp_radius), 1.0, -1)
    dens = cv.GaussianBlur(dens, (blur, blur), 0)
    norm = cv.normalize(dens, None, 0.0, 1.0, cv.NORM_MINMAX)
    # dichte = norm >= thr
    return norm, (norm >= float(thr))

def _merge_by_radius(kpsA: List[cv.KeyPoint], desA: np.ndarray,
                     kpsB: List[cv.KeyPoint], desB: np.ndarray, r=2):
    if not kpsA:
        return kpsB, desB
    if not kpsB:
        return kpsA, desA
    xyA = np.array([[k.pt[0], k.pt[1]] for k in kpsA], np.float32)
    respA = np.array([k.response for k in kpsA], np.float32)
    xyB = np.array([[k.pt[0], k.pt[1]] for k in kpsB], np.float32)
    respB = np.array([k.response for k in kpsB], np.float32)
    all_xy = np.vstack([xyA, xyB])
    all_resp = np.hstack([respA, respB])
    all_kps = kpsA + kpsB
    all_des = np.vstack([desA, desB]) if desA.size and desB.size else (desA if desB.size==0 else desB)

    grid = {}
    rr = max(1, int(r))
    for idx, (x, y) in enumerate(all_xy):
        gx, gy = int(round(x/rr)), int(round(y/rr))
        key = (gx, gy)
        if key not in grid or all_resp[idx] > all_resp[grid[key]]:
            grid[key] = idx
    keep_idx = sorted(grid.values())
    kps = [all_kps[i] for i in keep_idx]
    des = all_des[keep_idx] if all_des is not None and all_des.ndim==2 and all_des.shape[0]==len(all_kps) else all_des
    return kps, des



def extract_features(
    images: List[str],
    out_dir: str,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None
) -> Tuple[list, list, list, dict]:
    def log(m: str):
        if on_log: on_log(m)
    def prog(p: float, s: str):
        if on_progress: on_progress(int(p), s)

    backend_cfg = os.getenv("FEATURE_BACKEND", "disk").lower()
    use_mask    = os.getenv("FEATURE_USE_MASK", "1").lower() in ("1","true","yes","on")
    max_kp      = int(float(os.getenv("FEATURE_MAX_KP", "25000")))
    debug_every = int(float(os.getenv("FEATURE_DEBUG_EVERY", "0")))
    device_env  = os.getenv("FEATURE_DEVICE", "cuda").lower().strip()
    mask_dilate = int(float(os.getenv("FEATURE_MASK_DILATE", "6")))

    # Iterative Fill
    fill_enable   = os.getenv("FEATURE_FILL_ENABLE","1").lower() in ("1","true","yes")
    fill_thr      = float(os.getenv("FEATURE_FILL_THR","0.05"))   # Dichteschwelle [0..1]
    fill_kpr      = int(float(os.getenv("FEATURE_FILL_KP_RADIUS","4")))
    fill_morph_er = int(float(os.getenv("FEATURE_FILL_ERODE","2")))
    fill_morph_di = int(float(os.getenv("FEATURE_FILL_DILATE","4")))
    fill_merge_r  = int(float(os.getenv("FEATURE_FILL_MERGE_R","2")))

    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, "masks"); os.makedirs(mask_dir, exist_ok=True)

    keypoints: List[List[cv.KeyPoint]] = []
    descriptors: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    meta = {"backend": backend_cfg, "lg_feature_name": None, "sp_scores": []}

    device = device_env if device_env in ("cuda","cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Backends
    sp = None; sift = None; lg_name = None; backend = backend_cfg
    if backend_cfg in ("superpoint","disk","aliked","alike","aliked-lightglue"):
        kind = "aliked" if backend_cfg in ("aliked","alike","aliked-lightglue") else backend_cfg
        try:
            sp = _make_lg_extractor(kind, device, max_kp, log); lg_name = kind; backend="superpoint"
        except Exception as e:
            log(f"[features] WARN: {kind} init failed: {e}. Falling back to SIFT."); backend = "sift"
    if backend == "sift":
        sift = cv.SIFT_create(
            nfeatures=max_kp, nOctaveLayers=4,
            contrastThreshold=float(os.getenv("FEATURE_SIFT_CONTRAST","0.004")),
            edgeThreshold=int(float(os.getenv("FEATURE_SIFT_EDGE","12"))),
            sigma=float(os.getenv("FEATURE_SIFT_SIGMA","1.2"))
        )
        log(f"[features] SIFT (RootSIFT), max_kp={max_kp}")
    meta["lg_feature_name"] = lg_name


    N = len(images)
    for i, path in enumerate(images):
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)
        if img_bgr is None:
            log(f"[features] WARN: cannot read {path}")
            keypoints.append([]); descriptors.append(np.empty((0,128),np.float32)); shapes.append((0,0))
            continue
        gray0 = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        gray1 = _boost_gray(gray0)


        base = os.path.splitext(os.path.basename(path))[0]
        mask = None
        if use_mask:
            mpath = os.path.join(mask_dir, base + "_mask.png")
            if os.path.isfile(mpath):
                mask = cv.imread(mpath, cv.IMREAD_GRAYSCALE)


        if sp is not None:
            with torch.no_grad():
                tin = _to_tensor(gray1).to(device)
                out = sp({"image": tin})
            kp_t = out["keypoints"][0]
            sc_t = out.get("scores", out.get("keypoint_scores"))[0]
            desc_t = out["descriptors"][0]
            if kp_t.numel()==0:
                kps1, des1, sc1 = [], np.empty((0,256),np.float32), np.empty((0,),np.float32)
            else:
                kps1 = _sp_to_cv_kp(kp_t, sc_t)
                des1 = _ensure_desc_is_NC(desc_t.detach().cpu().numpy().astype(np.float32), len(kps1))
                sc1  = sc_t.detach().cpu().numpy().astype(np.float32)
                kps1, des1, sc1 = _filter_kp_des_scores(kps1, des1, sc1, mask, dilate=mask_dilate if mask is not None else 0)
        else:
            kps1, des1 = sift.detectAndCompute(gray1, mask)
            des1 = _rootsift(des1); sc1 = np.array([kp.response for kp in (kps1 or [])], np.float32)

        shapes.append(gray1.shape)


        if debug_every and (i % debug_every == 0):
            dbg1 = cv.drawKeypoints(img_bgr, kps1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if mask is not None:
                dbg1 = cv.addWeighted(dbg1, 0.85, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.15, 0)
            cv.imwrite(os.path.join(out_dir, f"debug_{i:04d}.jpg"), dbg1)


        rest_mask = None
        if fill_enable and (mask is not None):

            kxy = np.array([[kp.pt[0], kp.pt[1]] for kp in (kps1 or [])], np.float32) if kps1 else np.empty((0,2),np.float32)
            norm, dense = _density_mask_from_kps(gray1.shape, kxy, kp_radius=fill_kpr, thr=fill_thr)

            rest_mask = ((mask > 127) & (~dense)).astype(np.uint8) * 255

            if fill_morph_er > 0:
                rest_mask = cv.erode(rest_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE,(fill_morph_er,fill_morph_er)))
            if fill_morph_di > 0:
                rest_mask = cv.dilate(rest_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE,(fill_morph_di,fill_morph_di)))


            cv.imwrite(os.path.join(out_dir, f"restmask_{i:04d}.png"), rest_mask)


        kps2, des2 = [], np.empty((0, des1.shape[1] if des1 is not None and des1.ndim==2 else 128), np.float32)
        if fill_enable and (rest_mask is not None) and rest_mask.any():
            gray2 = _boost_gray_strong(gray0)
            if sp is not None:
                with torch.no_grad():
                    tin2 = _to_tensor(gray2).to(device)
                    out2 = sp({"image": tin2})
                kp2_t = out2["keypoints"][0]
                sc2_t = out2.get("scores", out2.get("keypoint_scores"))[0]
                des2_t= out2["descriptors"][0]
                if kp2_t.numel()>0:
                    kps2 = _sp_to_cv_kp(kp2_t, sc2_t)
                    des2 = _ensure_desc_is_NC(des2_t.detach().cpu().numpy().astype(np.float32), len(kps2))
                    sc2  = sc2_t.detach().cpu().numpy().astype(np.float32)
                    kps2, des2, sc2 = _filter_kp_des_scores(kps2, des2, sc2, rest_mask, dilate=0)
                else:
                    sc2 = np.empty((0,), np.float32)
            else:
                kps2, des2 = sift.detectAndCompute(gray2, rest_mask)
                des2 = _rootsift(des2); sc2 = np.array([kp.response for kp in (kps2 or [])], np.float32)


            if debug_every and (i % debug_every == 0):
                dbg2 = img_bgr.copy()
                overlay = cv.applyColorMap(rest_mask, cv.COLORMAP_HOT)
                dbg2 = cv.addWeighted(dbg2, 0.7, overlay, 0.3, 0)
                dbg2 = cv.drawKeypoints(dbg2, kps2, None, color=(0,255,255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imwrite(os.path.join(out_dir, f"debug_fill_{i:04d}.jpg"), dbg2)


        kps_merged, des_merged = _merge_by_radius(kps1 or [], des1 if des1 is not None else np.empty((0,128),np.float32),
                                                  kps2 or [], des2 if des2 is not None else np.empty((0,128),np.float32),
                                                  r=fill_merge_r)
        keypoints.append(kps_merged)
        descriptors.append(des_merged if des_merged is not None else np.empty((0,128),np.float32))


        if debug_every and (i % debug_every == 0):
            dbgM = cv.drawKeypoints(img_bgr, kps_merged, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(out_dir, f"debug_merged_{i:04d}.jpg"), dbgM)


        des_last = descriptors[-1]
        des_last = des_last.astype(np.float32).reshape(-1, des_last.shape[-1]) if des_last is not None else np.empty((0,128),np.float32)
        np.savez(
            os.path.join(out_dir, f"features_{i:04d}.npz"),
            kps=_kp_to_np(keypoints[-1]),
            des=des_last,
            shape=np.array(shapes[-1], dtype=np.int32)
        )

        if N:
            prog(30 + (i + 1) / max(1, N) * 10, "Feature Extraction + Fill")

    total_kp = sum(len(k) for k in keypoints)
    log(f"[features] done: {total_kp} keypoints (with fill={fill_enable}) | backend={lg_name or backend} | torch={torch.__version__}")
    meta["sp_scores"] = None
    return keypoints, descriptors, shapes, meta
