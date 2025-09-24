import os, cv2 as cv, numpy as np
from typing import Callable
from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from feature_extraction import sift_extract
from image_matching import build_pairs
from sfm_incremental import run_sfm
from depth_mvs import run_depth_mvs
from meshing import save_point_cloud

# Plotter
from camera_pose_plot import plot_camera_poses

class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str], None], on_progress: Callable[[int, str], None]):
        self.base_dir = base_dir
        self.on_log = on_log
        self.on_progress = on_progress

    def run(self, video_path: str, project_name: str, target_frames: int):
        log, prog = self.on_log, self.on_progress
        paths = make_project_paths(self.base_dir, project_name)
        log(f"[pipeline] Projekt: {paths.root}")

        # 1) Frames
        imgs = extract_and_save_frames(video_path, target_frames, paths.raw_frames, log, prog)

        # 2) Features
        kps, descs, shapes = sift_extract(imgs, paths.features, log, prog)

        # 3) Matching
        pairs, matches = build_pairs(descs, log, prog, save_dir=paths.matches, keypoints=kps)

        # 4) Intrinsics (grob)
        h, w = shapes[0]
        focal = 0.9 * max(w, h); pp = (w/2.0, h/2.0)
        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], float)

        # --- NEU: Posen-Verzeichnis definieren ---
        poses_dir = os.path.join(paths.root, "poses")
        os.makedirs(poses_dir, exist_ok=True)

        # 5) SfM inkl. Pose-Speicherung
        pts, poses_R, poses_t = run_sfm(
            kps, descs, shapes, pairs, matches, K,
            log, prog,
            poses_out_dir=poses_dir
        )

        # 5b) 3D-Plot der Kameraposen erzeugen (PNG)
        npz_path = os.path.join(poses_dir, "camera_poses.npz")
        if os.path.isfile(npz_path):
            # bevorzugt im sfm-Ordner speichern (übersichtlicher)
            sfm_dir = getattr(paths, "sfm", os.path.join(paths.root, "sfm"))
            os.makedirs(sfm_dir, exist_ok=True)
            out_png = os.path.join(sfm_dir, "camera_poses_plot.png")
            try:
                out_file = plot_camera_poses(npz_path, out_png_path=out_png, show=False)
                log(f"[pipeline] camera pose plot -> {out_file}")
            except Exception as e:
                log(f"[pipeline] WARN: camera pose plot failed: {e}")

        # 6) Depth (stub)
        run_depth_mvs(paths.raw_frames, pts, paths.depth, log, prog)

        # 7) Save / Mesh
        ply_path = os.path.join(paths.mesh, "reconstruction_sparse.ply")
        save_point_cloud(pts, ply_path, on_log=log, on_progress=prog)
        return ply_path, paths
