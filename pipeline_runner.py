import os, cv2 as cv, numpy as np
from typing import Callable
from io_paths import make_project_paths
from frame_extractor import extract_and_save_frames
from feature_extraction import sift_extract
from image_matching import build_pairs
from sfm_incremental import run_sfm
from depth_mvs import run_depth_mvs
from meshing import save_point_cloud

class PipelineRunner:
    def __init__(self, base_dir: str, on_log: Callable[[str],None], on_progress: Callable[[int,str],None]):
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

        # 3) Matching (Paare & Matches)
        pairs, matches = build_pairs(descs, log, prog, save_dir=paths.matches, keypoints=kps)

        # 4) Intrinsics (grob)
        h,w = shapes[0]
        focal = 0.9 * max(w,h); pp=(w/2.0,h/2.0)
        K = np.array([[focal,0,pp[0]],[0,focal,pp[1]],[0,0,1]], float)

        # 5) SfM
        pts = run_sfm(kps, descs, shapes, pairs, matches, K, log, prog)

        # 6) Depth (stub)
        run_depth_mvs(paths.raw_frames, pts, paths.depth, log, prog)

        # 7) Save / Mesh
        ply_path = os.path.join(paths.mesh, "reconstruction_sparse.ply")
        save_point_cloud(pts, ply_path, on_log=log, on_progress=prog)
        return ply_path, paths
