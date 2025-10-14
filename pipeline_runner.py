import os, json
import numpy as np
import cv2 as cv
from typing import Callable, Tuple, Any, Dict, Optional, List

from image_masking import preprocess_images
from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from lowlight_enhancer import (
    enhance_project_raw_frames_inplace,
    enhance_project_raw_frames_to_dir
)
from feature_extraction import extract_features
from image_matching import build_pairs
from sfm_incremental import run_sfm, SfMConfig
from meshing import (
    save_point_cloud,
    reconstruct_mvs_depth_and_mesh,
    reconstruct_mvs_depth_and_mesh_all)
from texturing import texture_points_from_views


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
        if os.path.isfile(p): return p
    for name in ("config.yaml", "config.yml", "config.json"):
        p = os.path.join(os.getcwd(), name)
        if os.path.isfile(p): return p
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
            for k in ("iters","n","value","val","amount","k","grid","target","radius","r","thr","threshold","std","sigma","resize","topk","gap"):
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
    # NEU: Output-Steuerung
    setenv("LOWLIGHT_OUTPUT_MODE", (ll.get("output_mode", "separate") or "separate"))
    setenv("LOWLIGHT_OUT_DIR", ll.get("out_dir", "processed_frames"))

    fe = cfg.get("features", {})
    setenv("FEATURE_BACKEND", fe.get("backend"))
    setenv("FEATURE_DEVICE", fe.get("device"))
    setenv("FEATURE_USE_MASK", fe.get("use_mask"))
    setenv("FEATURE_MAX_KP", pick_num(fe.get("max_kp", 4096)))
    setenv("FEATURE_DEBUG_EVERY", pick_num(fe.get("debug_every", 0)))

    fill = fe.get("fill", {}) or {}
    setenv("FEATURE_FILL_ENABLE", fill.get("enable", True))
    setenv("FEATURE_FILL_THR", fill.get("thr", 0.05))
    setenv("FEATURE_FILL_KP_RADIUS", fill.get("kp_radius", 4))
    setenv("FEATURE_FILL_ERODE", fill.get("erode", 2))
    setenv("FEATURE_FILL_DILATE", fill.get("dilate", 4))
    setenv("FEATURE_FILL_MERGE_R", fill.get("merge_r", 2))
    setenv("FEATURE_FILL_GAMMA", fill.get("gamma", 0.75))
    setenv("FEATURE_FILL_CLAHE_CLIP", fill.get("clahe_clip", 6.0))
    setenv("FEATURE_FILL_CLAHE_TILES", fill.get("clahe_tiles", 8))
    setenv("FEATURE_FILL_UNSHARP_SIGMA", fill.get("unsharp_sigma", 1.2))
    setenv("FEATURE_FILL_UNSHARP_AMOUNT", fill.get("unsharp_amount", 1.8))
    setenv("FEATURE_FILL_NOISE", fill.get("noise", False))
    setenv("FEATURE_FILL_NOISE_STD", fill.get("noise_std", 2.0))

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

    lg = fe.get("lg", {})
    setenv("FEATURE_LG_DET_THR", pick_num(lg.get("detection_threshold", 0.001)))
    setenv("FEATURE_LG_NMS", int(pick_num(lg.get("nms", 3))))
    setenv("FEATURE_LG_RESIZE", int(pick_num(lg.get("resize", 1024))))
    setenv("FEATURE_LG_REMOVE_BORDERS", int(pick_num(lg.get("remove_borders", 2))))
    setenv("FEATURE_LG_FORCE_NUM", pick_bool(lg.get("force_num", False)))

    sf = fe.get("sift", {})
    setenv("FEATURE_SIFT_CONTRAST", pick_num(sf.get("contrast", 0.004)))
    setenv("FEATURE_SIFT_EDGE", int(pick_num(sf.get("edge", 12))))
    setenv("FEATURE_SIFT_SIGMA", pick_num(sf.get("sigma", 1.2)))

    mask_cfg = cfg.get("masking", {})
    setenv("MASK_ENABLE", pick_bool(mask_cfg.get("enable", False)))
    setenv("MASK_METHOD", mask_cfg.get("method"))
    setenv("MASK_OVERWRITE_IMAGES", pick_bool(mask_cfg.get("overwrite_images", False)))

    ma = cfg.get("matching", {})
    setenv("MATCH_BACKEND", ma.get("backend"))
    setenv("MATCH_RATIO", pick_num(ma.get("ratio", 0.82)))
    setenv("MATCH_DEPTH", ma.get("depth_confidence"))
    setenv("MATCH_WIDTH", ma.get("width_confidence"))
    setenv("MATCH_FILTER", ma.get("filter_threshold"))
    if "features" in ma:
        setenv("MATCH_FEATURES", str(ma["features"]).lower())

    sfm = cfg.get("sfm", {})
    setenv("SFM_INIT_RATIO", pick_num(sfm.get("init_ratio", 0.5)))
    setenv("SFM_INIT_MAX_SPAN", pick_num(sfm.get("init_max_span", 6)))
    setenv("SFM_DENSIFY", pick_bool(sfm.get("densify", True)))
    setenv("SFM_USE_KEYFRAMES", pick_bool(sfm.get("use_keyframes", True)))
    setenv("SFM_USE_LOOP_CONSTRAINTS", pick_bool(sfm.get("use_loops", True)))
    setenv("SFM_POSE_SMOOTHING", pick_bool(sfm.get("pose_smoothing", True)))
    setenv("SFM_SMOOTH_LAMBDA", pick_num(sfm.get("smooth_lambda", 0.25)))

    mvs = cfg.get("mvs", {})
    setenv("MVS_ENABLE", pick_bool(mvs.get("enable", True)))
    setenv("MVS_MODE", (mvs.get("mode", "all") or "all"))
    mp = mvs.get("params", {}) or {}
    setenv("MVS_DEVICE", mp.get("device", fe.get("device", "cuda")))
    setenv("MVS_SCALE", mp.get("scale"))
    setenv("MVS_REF_STRATEGY", mp.get("ref_strategy"))
    setenv("MVS_REF_TOPK", mp.get("ref_topk"))
    setenv("MVS_REF_MIN_GAP", mp.get("ref_min_gap"))
    setenv("MVS_REF_STEP", mp.get("ref_step"))
    setenv("MVS_REF_BUCKETS", mp.get("ref_buckets", mp.get("ref_topk", 6)))
    setenv("MVS_SEED_RADIUS", mp.get("seed_radius"))
    setenv("MVS_SEED_MIN", mp.get("seed_min"))
    setenv("MVS_SEED_SAMPLE", mp.get("seed_sample"))
    setenv("MVS_BETA", mp.get("beta"))
    setenv("MVS_FILL_ITERS", mp.get("fill_iters"))
    setenv("MVS_MASK_PAD", mp.get("mask_pad"))
    setenv("MVS_EXPORT_MESH", mp.get("export_mesh"))
    setenv("MVS_POISSON_DEPTH", mp.get("poisson_depth"))

    cv_cfg = cfg.get("carve", {}) or {}
    setenv("CARVE_ENABLE", cv_cfg.get("enable"))
    setenv("CARVE_USE_ALL_MASKS", cv_cfg.get("use_all_masks"))
    setenv("CARVE_MODE", cv_cfg.get("mode"))
    setenv("CARVE_USE_DEPTH", cv_cfg.get("use_depth"))
    setenv("CARVE_DEPTH_TOL", cv_cfg.get("depth_tol"))
    setenv("CARVE_CHUNK", cv_cfg.get("chunk"))
    setenv("CARVE_VIEWS", cv_cfg.get("views"))

    tx = cfg.get("texturing", {}) or {}
    setenv("TEXTURE_ENABLE", pick_bool(tx.get("enable", True)))
    setenv("TEXTURE_DIVISOR", int(pick_num(tx.get("divisor", 6))))
    setenv("TEXTURE_WEIGHT_POWER", pick_num(tx.get("weight_power", 2.0)))
    setenv("TEXTURE_IN_PLY", tx.get("in_ply", "fused_points.ply"))
    setenv("TEXTURE_OUT_PLY", tx.get("out_ply", "fused_textured_points.ply"))
    # NEU:
    setenv("TEXTURE_USE_MASKS", tx.get("use_masks", True))
    setenv("TEXTURE_MASK_DILATE", int(pick_num(tx.get("mask_dilate", 5))))

def _auto_views(n_frames: int, divisor: int) -> list[int]:
    import numpy as _np
    divisor = max(1, int(divisor))
    if n_frames <= 1:
        return [0]
    idx = _np.linspace(0, n_frames - 1, num=divisor + 1, dtype=int)
    idx = _np.unique(_np.clip(idx, 0, n_frames - 1))
    return idx.tolist()

def _dump_resolved_config(project_root: str) -> str:
    keys = [k for k in os.environ.keys() if k.startswith(("FEATURE_", "MATCH_", "SFM_", "MVS_", "MASK_", "LOWLIGHT_"))]
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

class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str], None], on_progress: Callable[[int, str], None]):
        self.base_dir = base_dir
        self.on_log = on_log
        self.on_progress = on_progress

    def run(
    self,
    video_path: str,
    project_name: str,
    target_frames: int,
    ask_manual_texture_cb: Optional[Callable[[str, List[str]], List[str]]] = None
    ) -> Tuple[str, object]:

        log, prog = self.on_log, self.on_progress
        paths = make_project_paths(self.base_dir, project_name)
        log(f"[pipeline] Project: {paths.root}")

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

        # 1b) Lowlight:
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
                        enhance_project_raw_frames_inplace(
                            project_root=paths.root,
                            weights_path=weights,
                            pattern=pattern,
                            device=None,
                            on_log=log
                        )
                        log("[lowlight] finished (in place)")
                    else:
                        log(f"[lowlight] start(separate) -> {paths.root}/{out_dir} (from raw_frames, pattern={pattern})")
                        processed_imgs = enhance_project_raw_frames_to_dir(
                            project_root=paths.root,
                            weights_path=weights,
                            pattern=pattern,
                            out_dir_name=out_dir,
                            device=None,
                            on_log=log
                        )
        except Exception as e:
            log(f"[lowlight] failed: {e}")

        imgs_for_features = processed_imgs if (processed_imgs and len(processed_imgs) == len(imgs)) else imgs

        # 2) Masking
        prog(20, "Image preprocessing / masking")
        if _parse_bool(os.getenv("MASK_ENABLE", "false"), False):
            log("[mask] preprocessing enabled (SOURCE=raw_frames)")
            method = os.getenv("MASK_METHOD", "auto")

            overwrite = False
            mask_dir = os.path.join(paths.features, "masks")
            params = (cfg.get("masking", {}) or {}).get("params", {}) or {}
            log(f"[mask] method={method}, overwrite={overwrite}")
            log(f"[mask] params={params}")
            preprocess_images(
                imgs,  # <-- RAW frames!
                out_mask_dir=mask_dir,
                overwrite_images=overwrite,
                method=method,
                params=params,
                save_debug=True
            )

        kps, descs, shapes, meta = extract_features(imgs_for_features, paths.features, log, prog)
        if not kps or not shapes:
            raise RuntimeError("Feature extraction returned no keypoints.")

        # 4) matching
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

        # 5) intrinsics (pinhole guess)
        h, w = shapes[0]
        focal = 0.92 * float(max(w, h))
        pp = (w / 2.0, h / 2.0)
        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], dtype=float)
        log(f"[pipeline] intrinsics: f={focal:.1f}, cx={pp[0]:.1f}, cy={pp[1]:.1f}")

        poses_dir = os.path.join(paths.root, "poses")
        os.makedirs(poses_dir, exist_ok=True)
        snap = _dump_resolved_config(paths.root)
        log(f"[pipeline] saved resolved config -> {snap}")

        # 6) SfM
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

        # 7) save sparse cloud
        if not (isinstance(res, tuple) and len(res) >= 1):
            raise RuntimeError("SfM did not return a points array.")
        points3d = np.asarray(res[0], dtype=np.float64).reshape(-1, 3)

        mesh_dir = os.path.join(paths.root, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        sparse_ply = os.path.join(mesh_dir, "sparse.ply")
        save_point_cloud(points3d, sparse_ply, filter_min_points=10**9, on_log=log, on_progress=prog)
        log(f"[sfm] raw_points(after validation)={points3d.shape[0]:d}")
        log(f"[ui] Done: {sparse_ply}")

        # 8) Sparse-Paint (MVS)
        if _parse_bool(os.getenv("MVS_ENABLE", "true"), True):
            log("Sparse-Paint (sparse-guided depth completion)")
            try:
                mode = os.getenv("MVS_MODE", "all").strip().lower()
                common_kwargs = dict(
                    paths=paths, K=K,
                    scale=float(os.getenv("MVS_SCALE","1.0")),
                    max_views=0, n_planes=0, depth_expand=0.0, patch=0,
                    cost_thr=0.0, min_valid_frac=0.0,
                    poisson_depth=int(os.getenv("MVS_POISSON_DEPTH","10")),
                    on_log=log, on_progress=prog
                )
                if mode == "all":
                    reconstruct_mvs_depth_and_mesh_all(**common_kwargs)
                else:
                    reconstruct_mvs_depth_and_mesh(**common_kwargs)
            except Exception as e:
                log("[error]\nSparse-Paint failed: " + str(e))

        # 9) Carving
        try:
            if _parse_bool(os.getenv("CARVE_ENABLE", "false"), False):
                from meshing import carve_points_like_texturing
                import meshing as _meshing
                _meshing.GLOBAL_INTRINSICS_K = K

                frames_dir = os.path.join(paths.root, "raw_frames")
                masks_dir = os.path.join(paths.features, "masks")
                poses_npz = os.path.join(paths.root, "poses", "camera_poses.npz")

                use_all = _parse_bool(os.getenv("CARVE_USE_ALL_MASKS", "false"), False)
                n_req = os.getenv("CARVE_VIEWS", "").strip()
                if n_req.isdigit():
                    n_req = int(n_req)
                else:
                    n_req = int(float(os.getenv("TEXTURE_DIVISOR", "6"))) + 1

                log(f"[carve] start (use_all={use_all}, views={n_req})")
                carve_points_like_texturing(
                    mesh_dir=os.path.join(paths.root, "mesh"),
                    frames_dir=frames_dir,
                    poses_npz=poses_npz,
                    masks_dir=masks_dir,
                    use_all_masks=use_all,
                    n_views=n_req,
                    on_log=log, on_progress=prog
                )
        except Exception as e:
            log(f"[carve] failed: {e}")

        # 10) Texturing
        try:
            if _parse_bool(os.getenv("TEXTURE_ENABLE", "true"), True):
                mesh_dir = os.path.join(paths.root, "mesh")
                in_ply = os.getenv("TEXTURE_IN_PLY", "fused_points.ply")
                fused_ply = os.path.join(mesh_dir, in_ply)
                if os.path.isfile(fused_ply):
                    mode = (os.getenv("TEXTURE_MODE", "auto") or "auto").strip().lower()
                    out_ply = os.getenv("TEXTURE_OUT_PLY", "fused_textured_points.ply")
                    wpow = float(os.getenv("TEXTURE_WEIGHT_POWER", "4.0"))


                    import glob as _glob
                    frames_dir = os.path.join(paths.root, "raw_frames")
                    frame_files = sorted(_glob.glob(os.path.join(frames_dir, "frame_*.png")))
                    frame_basenames = [os.path.basename(p) for p in frame_files]

                    def _names_to_indices(names: List[str], universe: List[str]) -> List[int]:
                        if not names:
                            return []
                        s = {n.strip() for n in names if n and n.strip()}
                        idxs = [i for i, base in enumerate(universe) if base in s]
                        return sorted(set(i for i in idxs if 0 <= i < len(universe)))

                    views: List[int] = []

                    if mode == "manual":

                        if ask_manual_texture_cb is not None:
                            selected_names = ask_manual_texture_cb(paths.root, frame_basenames)
                        else:

                            spec = (os.getenv("TEXTURE_VIEWS", "") or "").replace(";", ",")
                            selected_names = [t for t in spec.split(",") if t.strip()]
                        views = _names_to_indices(selected_names, frame_basenames)
                        if not views:
                            mode = "auto"
                            log("[texture] manual selected but no frames chosen -> fallback to auto")

                    if mode != "manual":

                        divisor = int(float(os.getenv("TEXTURE_DIVISOR", "6")))
                        n = len(frame_files) if frame_files else len(imgs)
                        views = _auto_views(n, divisor)
                        log(f"[texture] auto-views (div={divisor}) -> {views}")
                    else:
                        log(f"[texture] manual-views -> {views}")

                    out_path = texture_points_from_views(
                        project_root=paths.root,
                        view_ids=views,
                        in_ply=in_ply,
                        out_ply=out_ply,
                        weight_power=wpow
                    )
                    log(f"[texture] saved: {out_path}")
                else:
                    log(f"[texture] skipped (no {fused_ply})")
        except Exception as e:
            log(f"[texture] failed: {e}")


        prog(100, "finished")
        return sparse_ply, paths
