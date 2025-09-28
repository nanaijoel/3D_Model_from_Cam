# pipeline_runner.py
import os
import numpy as np
from typing import Callable, Tuple

from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from feature_extraction import sift_extract
from image_matching import build_pairs
from sfm_incremental import run_sfm, run_sfm_multi, SfMConfig
from meshing import save_point_cloud, reconstruct_solid_mesh_from_ply, voxel_close_mesh
from camera_pose_plot import plot_camera_poses


def _parse_bool(s: str, default: bool) -> bool:
    if s is None:
        return default
    s = s.strip().lower()
    return s in ("1", "true", "yes", "y", "on")


class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str], None], on_progress: Callable[[int, str], None]):
        self.base_dir = base_dir
        self.on_log = on_log
        self.on_progress = on_progress

    def run(self, video_path: str, project_name: str, target_frames: int) -> Tuple[str, object]:

        log, prog = self.on_log, self.on_progress
        paths = make_project_paths(self.base_dir, project_name)
        log(f"[pipeline] Projekt: {paths.root}")

        # 1) Frames
        imgs = extract_and_save_frames(video_path, target_frames, paths.raw_frames, log, prog)
        if len(imgs) == 0:
            raise RuntimeError("Keine Frames extrahiert.")

        # 2) Features
        kps, descs, shapes = sift_extract(imgs, paths.features, log, prog)
        if len(kps) == 0 or len(shapes) == 0:
            raise RuntimeError("Feature-Extraktion ergab keine Keypoints.")

        # 3) Matching
        pairs, matches = build_pairs(descs, log, prog, save_dir=paths.matches, keypoints=kps)

        # 4) Intrinsics
        h, w = shapes[0]
        focal = 0.9 * max(w, h)
        pp = (w / 2.0, h / 2.0)
        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], dtype=float)
        log(f"[pipeline] intrinsics: f≈{focal:.1f}, cx={pp[0]:.1f}, cy={pp[1]:.1f}")

        # Pose-Ordner
        poses_dir = os.path.join(paths.root, "poses")
        os.makedirs(poses_dir, exist_ok=True)

        # 5) S f M
        ensemble_runs = int(os.getenv("SFM_ENSEMBLE_RUNS", "1"))
        init_ratio = float(os.getenv("SFM_INIT_RATIO", "0.5"))
        init_max_span = int(os.getenv("SFM_INIT_MAX_SPAN", "6"))
        init_frames_cap = int(float(os.getenv("SFM_INIT_FRAMES_CAP", "1e9")))
        front_friendly = _parse_bool(os.getenv("SFM_FRONT_FRIENDLY", "true"), True)
        densify_enable = _parse_bool(os.getenv("SFM_DENSIFY", "true"), True)

        cfg = SfMConfig(
            INIT_WINDOW_RATIO=init_ratio,
            INIT_WINDOW_FRAMES=init_frames_cap,
            INIT_MAX_SPAN=init_max_span,
            DENSIFY_ENABLE=densify_enable,
        )

        if front_friendly:
            cfg.POINT_PROMOTION_MIN_OBS = int(os.getenv("SFM_POINT_PROMOTION_MIN_OBS", "5"))
            cfg.POINT_MIN_MULTIVIEW_PARALLAX_DEG = float(os.getenv("SFM_POINT_MIN_PARALLAX_DEG", "3.0"))
            cfg.POINT_MAX_MULTIVIEW_REPROJ = float(os.getenv("SFM_POINT_MAX_MULTIVIEW_REPROJ", "1.6"))
            cfg.POINT_REQUIRE_POSITIVE_DEPTH_RATIO = float(os.getenv("SFM_POINT_POSDEPTH_RATIO", "0.8"))
            cfg.TRI_MIN_PARALLAX_DEG = float(os.getenv("SFM_TRI_MIN_PARALLAX_DEG", "3.0"))
            cfg.SEED_SPAN = int(os.getenv("SFM_SEED_SPAN", "4"))
            cfg.SEED_MIN_INL = int(os.getenv("SFM_SEED_MIN_INL", "35"))
            cfg.FRAME_MAX_MEDIAN_REPROJ_FOR_NEW_POINTS = float(os.getenv("SFM_FRAME_MAX_MEDIAN_REPROJ", "2.8"))

        log(f"[pipeline] SFM cfg: INIT_WINDOW_RATIO={cfg.INIT_WINDOW_RATIO}, "
            f"INIT_MAX_SPAN={cfg.INIT_MAX_SPAN}, DENSIFY={cfg.DENSIFY_ENABLE}, "
            f"FRONT_FRIENDLY={front_friendly}, ENSEMBLE_RUNS={ensemble_runs}")

        # -> SfM ausführen
        if ensemble_runs > 1:
            pts, poses_R, poses_t, report = run_sfm_multi(
                kps, descs, shapes, pairs, matches, K,
                n_runs=ensemble_runs,
                on_log=log, on_progress=prog,
                poses_out_dir=poses_dir
            )
            log(f"[pipeline] ensemble: best_center={report.get('best_center')}, "
                f"best_init_pair={report.get('best_init_pair')}, best_score={report.get('best_score'):.1f}")
        else:
            pts, poses_R, poses_t = run_sfm(
                kps, descs, shapes, pairs, matches, K,
                on_log=log, on_progress=prog,
                poses_out_dir=poses_dir,
                config=cfg
            )

        # 5b) Kamera-Plot (optional)
        npz_path = os.path.join(poses_dir, "camera_poses.npz")
        if os.path.isfile(npz_path):
            sfm_dir = getattr(paths, "sfm", os.path.join(paths.root, "sfm"))
            os.makedirs(sfm_dir, exist_ok=True)
            out_png = os.path.join(sfm_dir, "camera_poses_plot.png")
            try:
                out_file = plot_camera_poses(npz_path, out_png_path=out_png, show=False)
                log(f"[pipeline] camera pose plot -> {out_file}")
            except Exception as e:
                log(f"[pipeline] WARN: camera pose plot failed: {e}")

        # 6) Mesh / Save (sparse)
        ply_path = os.path.join(paths.mesh, "reconstruction_sparse.ply")
        save_point_cloud(pts, ply_path, on_log=log, on_progress=prog)
        log(f"[pipeline] saved sparse cloud -> {ply_path}")

        # 7) Watertight Mesh (Poisson/Alpha, optional via ENV)
        solid_enable = _parse_bool(os.getenv("MESH_SOLID_ENABLE", "true"), True)
        mesh_out = None
        if solid_enable:
            mesh_name = os.getenv("MESH_OUT_NAME", "mesh_solid_poisson.ply")
            mesh_out = os.path.join(paths.mesh, mesh_name)

            method = os.getenv("MESH_METHOD", "poisson")  # "poisson" | "alpha"
            depth = int(os.getenv("MESH_DEPTH", "10"))
            scale = float(os.getenv("MESH_SCALE", "1.1"))
            no_linear_fit = _parse_bool(os.getenv("MESH_NO_LINEAR_FIT", "false"), False)
            dens_q = float(os.getenv("MESH_DENS_Q", "0.02"))
            alpha = float(os.getenv("MESH_ALPHA", "1.0"))
            bbox_expand = float(os.getenv("MESH_BBOX_EXPAND", "0.03"))
            pre_filter = _parse_bool(os.getenv("MESH_PRE_FILTER", "false"), False)
            pre_filter_neighbors = int(os.getenv("MESH_PRE_FILTER_NEIGHBORS", "12"))
            pre_filter_std = float(os.getenv("MESH_PRE_FILTER_STD", "2.0"))
            voxel = float(os.getenv("MESH_VOXEL", "0.0"))
            normals_k = int(os.getenv("MESH_NORMALS_K", "40"))
            smooth = int(os.getenv("MESH_SMOOTH", "12"))
            simplify = int(os.getenv("MESH_SIMPLIFY", "0"))
            color_transfer = _parse_bool(os.getenv("MESH_COLOR_TRANSFER", "false"), False)

            prog(96, "Watertight Mesh – start")
            reconstruct_solid_mesh_from_ply(
                ply_in=ply_path, mesh_out=mesh_out,
                method=method, depth=depth, scale=scale, no_linear_fit=no_linear_fit,
                dens_quantile=dens_q, alpha=alpha, bbox_expand=bbox_expand,
                pre_filter=pre_filter, pre_filter_neighbors=pre_filter_neighbors, pre_filter_std=pre_filter_std,
                voxel=voxel, normals_k=normals_k, smooth=smooth, simplify=simplify,
                color_transfer=color_transfer,
                on_log=log, on_progress=prog
            )
            log(f"[pipeline] solid mesh -> {mesh_out}")

        # 8) NEU: Voxel-Closed Mesh zusätzlich erzeugen (optional)
        voxel_enable = _parse_bool(os.getenv("MESH_VOXELCLOSE_ENABLE", "true"), True)
        if voxel_enable and mesh_out is not None:
            voxel_name = os.getenv("MESH_VOXEL_OUT_NAME", "mesh_voxel_closed.ply")
            voxel_out = os.path.join(paths.mesh, voxel_name)
            voxel_grid = int(os.getenv("MESH_VOXEL_GRID", "180"))       # Auflösung (~marching cubes)
            voxel_smooth = int(os.getenv("MESH_VOXEL_SMOOTH", "4"))     # leicht glätten
            voxel_simplify = int(os.getenv("MESH_VOXEL_SIMPLIFY", "0")) # optional decimation

            try:
                prog(97, "Voxel-Closed Mesh – start")
                voxel_close_mesh(
                    mesh_in=mesh_out, mesh_out=voxel_out,
                    grid=voxel_grid, smooth=voxel_smooth, simplify=voxel_simplify,
                    on_log=log, on_progress=prog
                )
                log(f"[pipeline] voxel-closed mesh -> {voxel_out}")
            except Exception as e:
                log(f"[pipeline] WARN: voxel-close failed: {e}")

        return ply_path, paths