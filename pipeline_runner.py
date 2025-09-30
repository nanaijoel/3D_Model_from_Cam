import os, json
import numpy as np
from typing import Callable, Tuple, Any, Dict

from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from feature_extraction import extract_features
from image_matching import build_pairs
from sfm_incremental import run_sfm, SfMConfig
from meshing import save_point_cloud, reconstruct_solid_mesh_from_ply, voxel_close_mesh


# config utils

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
            for k in ("iters","n","value","val","amount","k","grid","target"):
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
    setenv("FEATURE_USE_MASK", pick_bool(fe.get("use_mask", False)))
    setenv("FEATURE_MAX_KP", pick_num(fe.get("max_kp", 4096)))
    setenv("FEATURE_DEBUG_EVERY", pick_num(fe.get("debug_every", 0)))

    ma = cfg.get("matching", {})
    setenv("MATCH_BACKEND", ma.get("backend"))
    setenv("MATCH_RATIO", pick_num(ma.get("ratio", 0.8)))

    sf = cfg.get("sfm", {})
    setenv("SFM_INIT_RATIO", pick_num(sf.get("init_ratio", 0.5)))
    setenv("SFM_INIT_MAX_SPAN", pick_num(sf.get("init_max_span", 6)))
    setenv("SFM_DENSIFY", pick_bool(sf.get("densify", True)))
    setenv("SFM_USE_KEYFRAMES", pick_bool(sf.get("use_keyframes", True)))
    setenv("SFM_USE_LOOP_CONSTRAINTS", pick_bool(sf.get("use_loops", True)))
    setenv("SFM_POSE_SMOOTHING", pick_bool(sf.get("pose_smoothing", True)))
    setenv("SFM_SMOOTH_LAMBDA", pick_num(sf.get("smooth_lambda", 0.25)))

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

# pipeline runner (sparse.ply only)

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

        # 1) frames
        imgs = extract_and_save_frames(video_path, target_frames, paths.raw_frames, log, prog)
        if not imgs:
            raise RuntimeError("No frames were extracted.")
        log(f"[frames] saved: {len(imgs)}")

        # 2) features
        kps, descs, shapes, meta = extract_features(imgs, paths.features, log, prog)
        if not kps or not shapes:
            raise RuntimeError("Feature extraction returned no keypoints.")

        # 3) matching
        pairs, matches = build_pairs(descs, log, prog, save_dir=paths.matches, keypoints=kps, meta=meta)

        # 4) intrinsics (simple)
        h, w = shapes[0]
        focal = 0.9 * float(max(w, h))
        pp = (w / 2.0, h / 2.0)
        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], dtype=float)
        log(f"[pipeline] intrinsics: f≈{focal:.1f}, cx={pp[0]:.1f}, cy={pp[1]:.1f}")

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
            SMOOTH_LAMBDA=float(os.getenv("SFM_SMOOTH_LAMBDA", "0.25")),
        )
        res = run_sfm(kps, descs, shapes, pairs, matches, K, log, prog,
                      poses_out_dir=poses_dir, config=cfg_sfm)

        # 6) save sparse (no reduction) -> projects/<name>/sparse.ply
        if not (isinstance(res, tuple) and len(res) >= 1):
            raise RuntimeError("SfM did not return a points array.")
        points3d = np.asarray(res[0], dtype=np.float64).reshape(-1, 3)
        sparse_ply = os.path.join(paths.root, "mesh/sparse.ply")
        save_point_cloud(points3d, sparse_ply, on_log=log, on_progress=prog)
        log(f"[sfm] raw_points(after validation)={points3d.shape[0]:d}")
        log(f"[ui] Done: {sparse_ply}")

        # 7) meshing (simple, no config)
        mesh_dir = os.path.join(paths.root, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)

        # 7a) Poisson mesh
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

        # 7b) voxel-closed from the Poisson mesh
        voxel_mesh = os.path.join(mesh_dir, "voxel_closed_mesh.ply")
        log("[mesh] voxel-close mesh (make fully closed volume)")
        try:
            voxel_close_mesh(
                solid_mesh, voxel_mesh,
                grid=180, smooth=16, simplify=0,
                on_log=log, on_progress=prog
            )
            log(f"[ui] Voxel-closed mesh: {voxel_mesh}")
        except Exception as e:
            log(f"[mesh] voxel_close failed: {e}")
        return sparse_ply, paths
