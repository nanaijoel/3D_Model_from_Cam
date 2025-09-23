from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class ProjectPaths:
    root: str
    raw_frames: str
    features: str
    matches: str
    sfm: str
    depth: str
    mesh: str
    logs: str

def make_project_paths(base_dir: str, project_name: str) -> ProjectPaths:
    root = os.path.join(base_dir, "projects", project_name)
    paths = {
        "root": root,
        "raw_frames": os.path.join(root, "raw_frames"),
        "features": os.path.join(root, "features"),
        "matches": os.path.join(root, "matches"),
        "sfm": os.path.join(root, "sfm"),
        "depth": os.path.join(root, "depth"),
        "mesh": os.path.join(root, "mesh"),
        "logs": os.path.join(root, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return ProjectPaths(**paths)
