# pipeline_runner.py

import os, json
import numpy as np
import cv2 as cv
from typing import Callable, Tuple, Any, Dict

from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from feature_extraction import extract_features
from image_matching import build_pairs
from sfm_incremental import run_sfm, SfMConfig
from meshing import save_point_cloud, reconstruct_solid_mesh_from_ply
from image_masking import preprocess_images


# ---------------- config utils ----------------

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

def _maybe_find_config(base_dir: str) -> str | None:
    for name in ("config.yaml", "config.yml", "config.json"):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    for name in ("config.yaml", "config.yml", "config.json"):
        p = os.path.join(os.getcwd(), name)
        if os.path.isfile(p):
            return p
    return None

def _parse_bool(s: str, default: bool) -> bool:
    if s is None: return default
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")

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
            for k in ("iters","n","value","val","amount","k","grid","target","radius","r","thr","threshold","std","sigma","resize"):
                if k in x and isinstance(x[k], (int, float)):
                    return x[k]
            return default
        if isinstance(x, (int, float)): return x
        if isinstance(x, str):
            try: return float(x)
            except Exception: return default
        return default

    fe = cfg.get("features", {})
    setenv("FEATURE_BACKEND", fe.get("backend"))
    setenv("FEATURE_DEVICE", fe.get("device"))
    setenv("FEATURE_USE_MASK", fe.get("use_mask"))  # "true"/"false" aus config ok
    setenv("FEATURE_MAX_KP", pick_num(fe.get("max_kp", 4096)))
    setenv("FEATURE_DEBUG_EVERY", pick_num(fe.get("debug_every", 0)))

    # optional gray preproc
    pp = fe.get("preproc", {})
    setenv("FEATURE_PREPROC_CLAHE", pick_bool(pp.get("clahe", True)))
    setenv("FEATURE_PREPROC_CLAHE_CLIP", pick_num(pp.get("clahe_clip", 3.0)))
    setenv("FEATURE_PREPROC_CLAHE_TILES", int(pick_num(pp.get("clahe_tiles", 8))))
    setenv("FEATURE_PREPROC_UNSHARP", pick_bool(pp.get("unsharp", True)))
    setenv("FEATURE_PREPROC_UNSHARP_SIGMA", pick_num(pp.get("unsharp_sigma", 1.0)))
    setenv("FEATURE_PREPROC_UNSHARP_AMOUNT", pick_num(pp.get("unsharp_amount", 1.5)))
    setenv("FEATURE_PREPROC_NOISE", pick_bool(pp.get("noise", True)))
    setenv("FEATURE_PREPROC_NOISE_STD", pick_num(pp.get("noise_std", 1.5)))
    setenv("FEATURE_MASK_DILATE", int(pick_num(pp.get("mask_dilate", 5))))

    # LightGlue extractor params
    lg = fe.get("lg", {})
    setenv("FEATURE_LG_DET_THR", pick_num(lg.get("detection_threshold", 0.001)))
    setenv("FEATURE_LG_NMS", int(pick_num(lg.get("nms", 3))))
    setenv("FEATURE_LG_RESIZE", int(pick_num(lg.get("resize", 1024))))
    setenv("FEATURE_LG_REMOVE_BORDERS", int(pick_num(lg.get("remove_borders", 2))))
    setenv("FEATURE_LG_FORCE_NUM", pick_bool(lg.get("force_num", False)))

    # SIFT tuning (fallback/explicit)
    sf = fe.get("sift", {})
    setenv("FEATURE_SIFT_CONTRAST", pick_num(sf.get("contrast", 0.004)))
    setenv("FEATURE_SIFT_EDGE", int(pick_num(sf.get("edge", 12))))
    setenv("FEATURE_SIFT_SIGMA", pick_num(sf.get("sigma", 1.2)))

    # Matching ENV
    ma = cfg.get("matching", {})
    setenv("MATCH_BACKEND", ma.get("backend"))  # "lightglue" | "classic"
    setenv("MATCH_RATIO", pick_num(ma.get("ratio", 0.82)))
    setenv("MATCH_DEPTH", ma.get("depth_confidence"))  # z.B. 0.90 | -1 (disable)
    setenv("MATCH_WIDTH", ma.get("width_confidence"))  # z.B. 0.97 | -1
    setenv("MATCH_FILTER", ma.get("filter_threshold"))  # z.B. 0.05..0.3

    if "features" in ma:
        setenv("MATCH_FEATURES", str(ma["features"]).lower())  # "superpoint"|"disk"|"aliked"|"sift"

    # SfM ENV (unchanged)
    sfm = cfg.get("sfm", {})
    setenv("SFM_INIT_RATIO", pick_num(sfm.get("init_ratio", 0.5)))
    setenv("SFM_INIT_MAX_SPAN", pick_num(sfm.get("init_max_span", 6)))
    setenv("SFM_DENSIFY", pick_bool(sfm.get("densify", True)))
    setenv("SFM_USE_KEYFRAMES", pick_bool(sfm.get("use_keyframes", True)))
    setenv("SFM_USE_LOOP_CONSTRAINTS", pick_bool(sfm.get("use_loops", True)))
    setenv("SFM_POSE_SMOOTHING", pick_bool(sfm.get("pose_smoothing", True)))
    setenv("SFM_SMOOTH_LAMBDA", pick_num(sfm.get("smooth_lambda", 0.25)))

def _dump_resolved_config(project_root: str) -> str:
    keys = [k for k in os.environ.keys() if k.startswith(("FEATURE_", "MATCH_", "SFM_"))]
    kv = {k: os.environ[k] for k in sorted(keys)}
    out_path = os.path.join(project_root, "resolved_config.yaml")
    try:
        import yaml
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(kv, f, sort_keys=True, allow_unicode=True)
    except Exception:
        with open(out_path.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(kv, f, indent=2, ensure_ascii=False)
    return out_path


# ---------------- pipeline runner (sparse.ply only) ----------------
class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str], None], on_progress: Callable[[int, str], None]):
        self.base_dir = base_dir
        self.on_log = on_log
        self.on_progress = on_progress

    def run(self, video_path: str, project_name: str, target_frames: int) -> Tuple[str, object]:
        log, prog = self.on_log, self.on_progress
        paths = make_project_paths(self.base_dir, project_name)
        log(f"[pipeline] Project: {paths.root}")

        # config
        cfg_path = _maybe_find_config(self.base_dir)
        if cfg_path:
            cfg = _load_yaml_or_json(cfg_path)
            _apply_env_from_config(cfg)
            log(f"[pipeline] config loaded: {cfg_path}")
        else:
            cfg = {}

        # 1) frames
        imgs = extract_and_save_frames(video_path, target_frames, paths.raw_frames, log, prog)
        if not imgs:
            raise RuntimeError("No frames were extracted.")
        log(f"[frames] saved: {len(imgs)}")

        prog(20, "Image preprocessing /  masking.")
        # optional Preprocessing/Masking
        mask_cfg = cfg.get("masking", {})
        if bool(mask_cfg.get("enable", False)):
            log("[mask] preprocessing enabled")
            method = str(mask_cfg.get("method", "auto"))
            overwrite = bool(mask_cfg.get("overwrite_images", False))  # gleiche Orte & Namen
            params = dict(mask_cfg.get("params", {}))
            mask_dir = os.path.join(paths.features, "masks")
            preprocess_images(
                imgs,
                out_mask_dir=mask_dir,
                overwrite_images=overwrite,
                method=method,
                params=params,
                save_debug=True
            )

        # 2) features
        kps, descs, shapes, meta = extract_features(imgs, paths.features, log, prog)
        if not kps or not shapes:
            raise RuntimeError("Feature extraction returned no keypoints.")

        # 3) matching – build_pairs liest direkt aus paths.features
        ratio = float(os.getenv("MATCH_RATIO", "0.82"))
        device = os.getenv("FEATURE_DEVICE", "cuda")
        pairs_np, matches_list = build_pairs(
            paths.features,
            ratio=ratio,
            device=device,
            backend=os.getenv("MATCH_BACKEND", "lightglue"),
            on_log=log,
            save_dir=paths.matches
        )

        # ---- Convert to SfM format: dict[(i,j)] -> List[cv.DMatch]
        def _as_sfm_matches(pairs_arr: np.ndarray, matches_any) -> Dict[Tuple[int, int], list]:
            mdict: Dict[Tuple[int, int], list] = {}
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

        # 4) intrinsics (simpel)
        h, w = shapes[0]
        focal = 0.92 * float(max(w, h))
        pp = (w / 2.0, h / 2.0)
        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], dtype=float)
        log(f"[pipeline] intrinsics: f={focal:.1f}, cx={pp[0]:.1f}, cy={pp[1]:.1f}")

        poses_dir = os.path.join(paths.root, "poses")
        os.makedirs(poses_dir, exist_ok=True)
        snap = _dump_resolved_config(paths.root)
        log(f"[pipeline] saved resolved config -> {snap}")

        # 5) SfM
        cfg_sfm = SfMConfig(
            INIT_WINDOW_RATIO=float(os.getenv("SFM_INIT_RATIO", "0.5")),
            INIT_MAX_SPAN=int(float(os.getenv("SFM_INIT_MAX_SPAN", "6"))),
            DENSIFY_ENABLE=_parse_bool(os.getenv("SFM_DENSIFY", "true"), True),
            USE_KEYFRAMES=_parse_bool(os.getenv("SFM_USE_KEYFRAMES", "true"), True),
            USE_LOOP_CONSTRAINTS=_parse_bool(os.getenv("SFM_USE_LOOP_CONSTRAINTS", "true"), True),
            POSE_SMOOTHING=_parse_bool(os.getenv("SFM_POSE_SMOOTHING", "true"), True),
            SMOOTH_LAMBDA=float(os.getenv("SFM_SMOOTH_LAMBDA", "0.01")),
        )

        res = run_sfm(
            kps, descs, shapes,
            pairs_np, matches_for_sfm,
            K, log, prog,
            poses_out_dir=poses_dir, config=cfg_sfm
        )

        # 6) save sparse
        if not (isinstance(res, tuple) and len(res) >= 1):
            raise RuntimeError("SfM did not return a points array.")
        points3d = np.asarray(res[0], dtype=np.float64).reshape(-1, 3)
        sparse_ply = os.path.join(paths.root, "mesh/sparse.ply")
        save_point_cloud(points3d, sparse_ply, on_log=log, on_progress=prog)
        log(f"[sfm] raw_points(after validation)={points3d.shape[0]:d}")
        log(f"[ui] Done: {sparse_ply}")

        # 7) meshing (Poisson)
        mesh_dir = os.path.join(paths.root, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        solid_mesh = os.path.join(mesh_dir, "solid_mesh_poisson.ply")
        log("[mesh] reconstruct solid mesh (poisson)")
        try:
            reconstruct_solid_mesh_from_ply(
                sparse_ply, solid_mesh,
                method="poisson",
                depth=10, scale=1.1, no_linear_fit=False,
                dens_quantile=0.015,
                bbox_expand=0.03,
                pre_filter=False, voxel=0.0,
                normals_k=40, smooth=16, simplify=0,
                color_transfer=False,
                on_log=log, on_progress=prog
            )
            log(f"[ui] Solid mesh: {solid_mesh}")
        except Exception as e:
            log(f"[mesh] solid reconstruction failed: {e}")

        return sparse_ply, paths