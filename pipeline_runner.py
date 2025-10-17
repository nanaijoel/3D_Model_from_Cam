# pipeline_runner.py
import os
import json
from typing import Callable, Tuple, Any, Dict, Optional, List

import numpy as np
import cv2 as cv

from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from lowlight_enhancer import (
    enhance_project_raw_frames_inplace,
    enhance_project_raw_frames_to_dir
)
from image_masking import preprocess_images
from feature_extraction import extract_features
from image_matching import build_pairs
from sfm_incremental import run_sfm, SfMConfig

# Korrekte Herkunft:
from meshing import save_point_cloud, run_mvs_and_carve
from texturing import run_texturing


# -------------------------- Config laden & spiegeln ---------------------------

def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    else:
        raise RuntimeError(f"Unknown config extension: {ext}")

def _maybe_find_config(base_dir: str) -> Optional[str]:
    for name in ("config.yaml", "config.yml", "config.json"):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p): return p
    for name in ("config.yaml", "config.yml", "config.json"):
        p = os.path.join(os.getcwd(), name)
        if os.path.isfile(p): return p
    return None

def _parse_bool(s: Any, default: bool) -> bool:
    if s is None: return default
    return str(s).strip().lower() in ("1","true","yes","y","on")

def _apply_env_from_config(cfg: Dict[str, Any]) -> None:
    def setenv(k, v):
        if v is None: return
        os.environ[k] = str(v)

    def pick_bool(x, default=False):
        if isinstance(x, dict): return bool(x.get("enable", default))
        if isinstance(x, (int, float, bool)): return bool(x)
        if isinstance(x, str): return str(x).strip().lower() in ("1","true","yes","y","on")
        return default

    def pick_num(x, default=0):
        if isinstance(x, dict):
            for k in ("iters","n","value","val","amount","k","grid","target",
                      "radius","r","thr","threshold","std","sigma","resize",
                      "topk","gap","step"):
                if k in x and isinstance(x[k], (int, float)):
                    return x[k]
            return default
        if isinstance(x, (int, float)): return x
        if isinstance(x, str):
            try: return float(x)
            except Exception: return default
        return default

    ll = cfg.get("lowlight", {}) or {}
    setenv("LOWLIGHT_ENABLE", pick_bool(ll.get("enable", False)))
    setenv("LOWLIGHT_WEIGHTS", ll.get("weights"))
    setenv("LOWLIGHT_PATTERN", ll.get("pattern", "*.png"))
    setenv("LOWLIGHT_SCALE", ll.get("scale_factor", 1))
    setenv("LOWLIGHT_OUTPUT_MODE", (ll.get("output_mode", "separate") or "separate"))
    setenv("LOWLIGHT_OUT_DIR", ll.get("out_dir", "processed_frames"))

    fe = cfg.get("features", {}) or {}
    setenv("FEATURE_BACKEND", fe.get("backend"))
    setenv("FEATURE_DEVICE", fe.get("device"))
    setenv("FEATURE_USE_MASK", fe.get("use_mask"))
    setenv("FEATURE_MAX_KP", pick_num(fe.get("max_kp", 25000)))
    setenv("FEATURE_DEBUG_EVERY", pick_num(fe.get("debug_every", 0)))

    fill = fe.get("fill", {}) or {}
    setenv("FEATURE_FILL_ENABLE", fill.get("enable", True))
    setenv("FEATURE_FILL_THR", fill.get("thr", 0.05))
    setenv("FEATURE_FILL_KP_RADIUS", fill.get("kp_radius", 4))
    setenv("FEATURE_FILL_ERODE", fill.get("erode", 2))
    setenv("FEATURE_FILL_DILATE", fill.get("dilate", 6))
    setenv("FEATURE_FILL_MERGE_R", fill.get("merge_r", 3))
    setenv("FEATURE_FILL_GAMMA", fill.get("gamma", 0.5))
    setenv("FEATURE_FILL_CLAHE_CLIP", fill.get("clahe_clip", 8.0))
    setenv("FEATURE_FILL_CLAHE_TILES", fill.get("clahe_tiles", 24))
    setenv("FEATURE_FILL_UNSHARP_SIGMA", fill.get("unsharp_sigma", 1.0))
    setenv("FEATURE_FILL_UNSHARP_AMOUNT", fill.get("unsharp_amount", 1.5))
    setenv("FEATURE_FILL_NOISE", fill.get("noise", False))
    setenv("FEATURE_FILL_NOISE_STD", fill.get("noise_std", 1.4))

    pp = fe.get("preproc", {}) or {}
    setenv("FEATURE_PREPROC_CLAHE", pick_bool(pp.get("clahe", True)))
    setenv("FEATURE_PREPROC_CLAHE_CLIP", pick_num(pp.get("clahe_clip", 4.0)))
    setenv("FEATURE_PREPROC_CLAHE_TILES", int(pick_num(pp.get("clahe_tiles", 16))))
    setenv("FEATURE_PREPROC_UNSHARP", pick_bool(pp.get("unsharp", True)))
    setenv("FEATURE_PREPROC_UNSHARP_SIGMA", pick_num(pp.get("unsharp_sigma", 1.0)))
    setenv("FEATURE_PREPROC_UNSHARP_AMOUNT", pick_num(pp.get("unsharp_amount", 1.3)))
    setenv("FEATURE_PREPROC_NOISE", pick_bool(pp.get("noise", False)))
    setenv("FEATURE_PREPROC_NOISE_STD", pick_num(pp.get("noise_std", 1.5)))
    setenv("FEATURE_MASK_DILATE", int(pick_num(pp.get("mask_dilate", 5))))

    lg = fe.get("lg", {}) or {}
    setenv("FEATURE_LG_DET_THR", pick_num(lg.get("detection_threshold", 0.0000045)))
    setenv("FEATURE_LG_NMS", int(pick_num(lg.get("nms", 3))))
    setenv("FEATURE_LG_RESIZE", int(pick_num(lg.get("resize", 1024))))
    setenv("FEATURE_LG_REMOVE_BORDERS", int(pick_num(lg.get("remove_borders", 2))))
    setenv("FEATURE_LG_FORCE_NUM", pick_bool(lg.get("force_num", True)))

    mask_cfg = cfg.get("masking", {}) or {}
    setenv("MASK_ENABLE", pick_bool(mask_cfg.get("enable", True)))
    setenv("MASK_METHOD", mask_cfg.get("method", "rembg"))
    setenv("MASK_OVERWRITE_IMAGES", pick_bool(mask_cfg.get("overwrite_images", False)))

    ma = cfg.get("matching", {}) or {}
    setenv("MATCH_BACKEND", ma.get("backend", "lightglue"))
    setenv("MATCH_RATIO", pick_num(ma.get("ratio", 0.82)))
    setenv("MATCH_DEPTH", ma.get("depth_confidence", -1))
    setenv("MATCH_WIDTH", ma.get("width_confidence", -1))
    setenv("MATCH_FILTER", ma.get("filter_threshold", 0.07))
    if "features" in ma:
        setenv("MATCH_FEATURES", str(ma["features"]).lower())
    setenv("MATCH_RETR_ENABLE", pick_bool(ma.get("retr_enable", True)))
    setenv("MATCH_RETR_K", pick_num(ma.get("retr_k", 4096)))
    setenv("MATCH_RETR_SAMPLE_PER_IMG", pick_num(ma.get("retr_sample_per_img", 3000)))
    setenv("MATCH_RETR_TOPK", pick_num(ma.get("retr_topk", 32)))
    setenv("MATCH_RETR_MIN_SIM", pick_num(ma.get("retr_min_sim", 0.07)))
    setenv("MATCH_NEIGHBOR_SPAN", pick_num(ma.get("neighbor_span", 6)))

    sfm = cfg.get("sfm", {}) or {}
    setenv("SFM_INIT_RATIO", pick_num(sfm.get("init_ratio", 0.5)))
    setenv("SFM_INIT_MAX_SPAN", pick_num(sfm.get("init_max_span", 8)))
    setenv("SFM_DENSIFY", pick_bool(sfm.get("densify", True)))
    setenv("SFM_USE_KEYFRAMES", pick_bool(sfm.get("use_keyframes", True)))
    setenv("SFM_USE_LOOP_CONSTRAINTS", pick_bool(sfm.get("use_loops", True)))
    setenv("SFM_POSE_SMOOTHING", pick_bool(sfm.get("pose_smoothing", True)))
    setenv("SFM_SMOOTH_LAMBDA", pick_num(sfm.get("smooth_lambda", 0.01)))

    mvs = cfg.get("mvs", {}) or {}
    setenv("MVS_ENABLE", pick_bool(mvs.get("enable", True)))
    setenv("MVS_MODE", (mvs.get("mode", "single") or "single"))
    mp = mvs.get("params", {}) or {}
    setenv("MVS_DEVICE", mp.get("device", fe.get("device", "cuda")))
    setenv("MVS_SCALE", mp.get("scale", 1.0))
    setenv("MVS_REF_STRATEGY", mp.get("ref_strategy", "step"))
    setenv("MVS_REF_TOPK", mp.get("ref_topk", 60))
    setenv("MVS_REF_MIN_GAP", mp.get("ref_min_gap", 3))
    setenv("MVS_REF_STEP", mp.get("ref_step", 2))
    setenv("MVS_SEED_RADIUS", mp.get("seed_radius", 2))
    setenv("MVS_SEED_MIN", mp.get("seed_min", 200))
    setenv("MVS_SEED_SAMPLE", mp.get("seed_sample", 200000))
    setenv("MVS_BETA", mp.get("beta", 4.0))
    setenv("MVS_FILL_ITERS", mp.get("fill_iters", 150))
    setenv("MVS_MASK_PAD", mp.get("mask_pad", 6))
    setenv("MVS_GAP_FILL_ENABLE", mp.get("gap_fill_enable", True))
    setenv("MVS_FILL_MAX_GAP", mp.get("fill_max_gap", 4))
    setenv("MVS_FILL_DEPTH_STD", mp.get("fill_depth_std", 0.03))
    setenv("MVS_FILL_DEPTH_REL", mp.get("fill_depth_rel", 0.06))
    setenv("MVS_FILL_STRIDE", mp.get("fill_stride", 1))
    # Cap/Downsample nach Fill (gegen Punkt-Explosion)
    setenv("MVS_FILL_MAX_POINTS", mp.get("fill_max_points", 3000000))
    setenv("MVS_VOXEL_AFTER_FILL", mp.get("voxel_after_fill", 0.0015))

    cv_cfg = cfg.get("carve", {}) or {}
    setenv("CARVE_ENABLE", cv_cfg.get("enable", True))
    setenv("CARVE_USE_ALL_MASKS", cv_cfg.get("use_all_masks", False))
    setenv("CARVE_VIEWS", cv_cfg.get("views", 24))
    setenv("CARVE_TAU", cv_cfg.get("tau", 0.50))
    setenv("CARVE_MASK_DILATE_PX", cv_cfg.get("mask_dilate", 2))

    tx = cfg.get("texturing", {}) or {}
    setenv("TEXTURE_ENABLE", pick_bool(tx.get("enable", True)))
    setenv("TEXTURE_MODE", (tx.get("mode", "auto") or "auto"))
    setenv("TEXTURE_DIVISOR", int(pick_num(tx.get("divisor", 12))))
    setenv("TEXTURE_WEIGHT_POWER", pick_num(tx.get("weight_power", 2.0)))
    setenv("TEXTURE_IN_PLY", tx.get("in_ply", "fused_points.ply"))
    setenv("TEXTURE_OUT_PLY", tx.get("out_ply", "fused_textured_points.ply"))
    setenv("TEXTURE_USE_MASKS", tx.get("use_masks", True))
    setenv("TEXTURE_MASK_DILATE", int(pick_num(tx.get("mask_dilate", 2))))
    setenv("TEXTURE_BACKFILL", tx.get("backfill", "nearest"))
    setenv("TEXTURE_DROP_UNTEXTURED", tx.get("drop_untextured", False))

    mesh_cfg = cfg.get("mesh", {}) or {}
    setenv("MESH_DOWNSAMPLE", mesh_cfg.get("downsample", 1))
    setenv("MESH_VOXEL_SIZE", mesh_cfg.get("voxel_size", 0.0))
    setenv("MESH_OUTLIER_NB", mesh_cfg.get("outlier_nb", 20))
    setenv("MESH_OUTLIER_STD", mesh_cfg.get("outlier_std", 1.6))


def _dump_resolved_config(project_root):
    out = os.path.join(project_root, "resolved_config.yaml")
    try:
        import yaml
        cfg = {}
        for k, v in os.environ.items():
            if k.startswith(("FEATURE_", "MATCH_", "SFM_", "MVS_", "CARVE_", "TEXTURE_", "MASK_", "MESH_","LOWLIGHT_")):
                cfg[k] = v
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
    except Exception:
        with open(out, "w") as f:
            for k, v in sorted(os.environ.items()):
                if k.startswith(("FEATURE_", "MATCH_", "SFM_", "MVS_", "CARVE_", "TEXTURE_", "MASK_", "MESH_","LOWLIGHT_")):
                    f.write(f"{k}: {v}\n")
    return out


# ------------------------------ Intrinsics ------------------------------------

def _estimate_f_from_pair(p1, p2, W, H):
    # robuste F-Schätzung → heuristische f-Näherung (stabiler als blinde Raterei)
    F, _ = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 1.0, 0.999, 5000)
    if F is None or F.shape != (3, 3):
        return None
    return 0.92 * float(max(W, H))  # konservative Heuristik

def _clamp_f(f_est, W, H):
    f_min = 0.70 * float(max(W, H))
    f_max = 1.80 * float(max(W, H))
    f_heur = 0.92 * float(max(W, H))
    if f_est is not None and np.isfinite(f_est) and (f_min <= float(f_est) <= f_max):
        return float(f_est), "(auto)"
    return f_heur, "(heuristic)"


# --------------------------------- Runner ------------------------------------

class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str], None], on_progress: Callable[[int, str], None]):
        self.base_dir = base_dir
        self.on_log = on_log
        self.on_progress = on_progress

    def run(self, video_path: str, project_name: str, target_frames: int) -> Tuple[str, object]:
        log, prog = self.on_log, self.on_progress
        paths = make_project_paths(self.base_dir, project_name)
        log(f"[pipeline] Project: {paths.root}")

        cfg_path = _maybe_find_config(self.base_dir)
        cfg = {}
        if cfg_path:
            cfg = _load_yaml_or_json(cfg_path)
            _apply_env_from_config(cfg)
            log(f"[pipeline] config loaded: {cfg_path}")

        # 1) Frames
        imgs = extract_and_save_frames(video_path, target_frames, paths.raw_frames, log, prog)
        if not imgs:
            raise RuntimeError("No frames were extracted.")
        log(f"[frames] saved: {len(imgs)}")

        # 1b) Lowlight → bevorzugt separates Verzeichnis (Texturing nutzt weiterhin raw_frames)
        processed_imgs = None
        try:
            if _parse_bool(os.getenv("LOWLIGHT_ENABLE", "false"), False):
                weights = os.getenv("LOWLIGHT_WEIGHTS", "") or ""
                pattern = os.getenv("LOWLIGHT_PATTERN", "*.png")
                mode = (os.getenv("LOWLIGHT_OUTPUT_MODE", "separate") or "separate").lower()
                out_dir = os.getenv("LOWLIGHT_OUT_DIR", "processed_frames")
                if not weights:
                    log("[lowlight] skipped (no LOWLIGHT_WEIGHTS given)")
                else:
                    if mode == "inplace":
                        log(f"[lowlight] start -> {paths.root}/raw_frames (pattern={pattern})")
                        enhance_project_raw_frames_inplace(paths.root, weights, pattern, None, log)
                        log("[lowlight] done (in place).")
                    else:
                        log(f"[lowlight] start(separate) -> {paths.root}/{out_dir} (from raw_frames, pattern={pattern})")
                        processed_imgs = enhance_project_raw_frames_to_dir(paths.root, weights, pattern, out_dir, None, log)
                        log("[lowlight] done (separate dir).")
        except Exception as e:
            log(f"[lowlight] failed: {e}")

        imgs_for_features = processed_imgs if (processed_imgs and len(processed_imgs) == len(imgs)) else imgs

        # 2) Masking
        prog(20, "Image preprocessing / masking")
        if _parse_bool(os.getenv("MASK_ENABLE", "false"), False):
            log("[mask] preprocessing enabled (SOURCE=raw_frames)")
            method = os.getenv("MASK_METHOD", "rembg")
            mask_dir = os.path.join(paths.features, "masks")
            params = (cfg.get("masking", {}) or {}).get("params", {}) or {}
            log(f"[mask] method={method}, overwrite=False")
            log(f"[mask] params={params}")
            preprocess_images(imgs, out_mask_dir=mask_dir, overwrite_images=False, method=method, params=params, save_debug=True)

        # 3) Features
        kps, descs, shapes, _ = extract_features(imgs_for_features, paths.features, log, prog)
        if not kps or not shapes:
            raise RuntimeError("Feature extraction returned no keypoints.")

        # 4) Matching
        ratio = float(os.getenv("MATCH_RATIO", "0.82"))
        device = os.getenv("FEATURE_DEVICE", "cuda")
        pairs_np, matches_list = build_pairs(paths.features, ratio=ratio, device=device,
                                             backend=os.getenv("MATCH_BACKEND", "lightglue"),
                                             on_log=log, save_dir=paths.matches)

        # 5) Intrinsics
        H0, W0 = shapes[0]  # (H,W)
        W, H = int(W0), int(H0)
        cx, cy = W * 0.5, H * 0.5

        def _as_sfm_matches(pairs_arr: np.ndarray, matches_any) -> Dict[tuple[int, int], list]:
            mdict: Dict[tuple[int, int], list] = {}
            for (i, j), m in zip(pairs_arr.tolist(), matches_any):
                if isinstance(m, list):
                    m = np.array(m, dtype=np.int32) if len(m) else np.empty((0, 2), np.int32)
                if m is None or m.size == 0:
                    mdict[(i, j)] = []
                    continue
                mdict[(i, j)] = [
                    cv.DMatch(_queryIdx=int(q), _trainIdx=int(t), _imgIdx=0, _distance=0.0)
                    for q, t in m.astype(np.int32)
                ]
            return mdict

        matches_for_sfm = _as_sfm_matches(pairs_np, matches_list)

        # bestes Paar für f-Schätzung
        best = None
        for (i, j), m in matches_for_sfm.items():
            if best is None or len(m) > best[0]:
                best = (len(m), i, j, m)

        f_est = None
        if best is not None and best[0] >= 16:
            _, bi, bj, bm = best
            p1 = np.float32([kps[bi][mm.queryIdx].pt for mm in bm])
            p2 = np.float32([kps[bj][mm.trainIdx].pt for mm in bm])
            f_est = _estimate_f_from_pair(p1, p2, W, H)

        f, f_tag = _clamp_f(f_est, W, H)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
        log(f"[pipeline] intrinsics: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f} {f_tag}")

        poses_dir = os.path.join(paths.root, "poses")
        os.makedirs(poses_dir, exist_ok=True)
        snap = _dump_resolved_config(paths.root)
        log(f"[pipeline] saved resolved config -> {snap}")

        # 6) SfM
        cfg_sfm = SfMConfig(
            INIT_WINDOW_RATIO=float(os.getenv("SFM_INIT_RATIO", "0.5")),
            INIT_MAX_SPAN=int(float(os.getenv("SFM_INIT_MAX_SPAN", "8"))),
            DENSIFY_ENABLE=_parse_bool(os.getenv("SFM_DENSIFY", "true"), True),
            USE_KEYFRAMES=_parse_bool(os.getenv("SFM_USE_KEYFRAMES", "true"), True),
            USE_LOOP_CONSTRAINTS=_parse_bool(os.getenv("SFM_USE_LOOP_CONSTRAINTS", "true"), True),
            POSE_SMOOTHING=_parse_bool(os.getenv("SFM_POSE_SMOOTHING", "true"), True),
            SMOOTH_LAMBDA=float(os.getenv("SFM_SMOOTH_LAMBDA", "0.01")),
        )

        res = run_sfm(kps, descs, shapes, pairs_np, matches_for_sfm, K, log, prog,
                      poses_out_dir=poses_dir, config=cfg_sfm)

        # 7) Sparse speichern (keine harte Drosselung hier)
        if not (isinstance(res, tuple) and len(res) >= 1):
            raise RuntimeError("SfM did not return a points array.")
        points3d = np.asarray(res[0], dtype=np.float64).reshape(-1, 3)

        mesh_dir = os.path.join(paths.root, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        sparse_ply = os.path.join(mesh_dir, "sparse.ply")
        save_point_cloud(points3d, sparse_ply, filter_min_points=10**9, on_log=log, on_progress=prog)
        log(f"[sfm] raw_points(after validation)={points3d.shape[0]:d}")
        log(f"[ui] Done: {sparse_ply}")

        # 8) Dense + Carving (gekapselt in meshing.run_mvs_and_carve)
        fused_ply = run_mvs_and_carve(paths, K, on_log=log, on_progress=prog)

        # 9) Intrinsics für Texturing persistieren
        intr_npy = os.path.join(mesh_dir, "intrinsics.npy")
        try:
            np.save(intr_npy, K)
            log(f"[texture] saved intrinsics -> {intr_npy}")
        except Exception as e:
            log(f"[texture] warn: could not save intrinsics.npy ({e})")

        # 10) Texturing – Manual-Flow unterstützen
        os.environ["TEXTURE_IN_PLY"] = os.getenv("TEXTURE_IN_PLY", "fused_points.ply")
        os.environ["TEXTURE_OUT_PLY"] = os.getenv("TEXTURE_OUT_PLY", "fused_textured_points.ply")

        # Manual-Modus: wenn (noch) keine Auswahl gesetzt → GUI soll Textur-Views sammeln,
        # Pipeline liefert erstmal die nicht-texturierte fused_ply zurück.
        if (os.getenv("TEXTURE_MODE", "auto").lower() == "manual"
            and not os.getenv("TEXTURE_MANUAL_VIEWS", "").strip()):
            log("[texture] manual mode selected – waiting for GUI to set TEXTURE_MANUAL_VIEWS …")
            return fused_ply, paths

        try:
            out_tex = run_texturing(mesh_dir, on_log=log)
            if out_tex:
                return out_tex, paths
        except Exception as e:
            log(f"[texture] failed: {e}")

        # Fallback: gib zumindest die gefusete Punktwolke zurück
        return fused_ply, paths
