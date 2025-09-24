# camera_pose_plot.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def plot_camera_poses(npz_path: str, out_png_path: str | None = None, show: bool = False, title: str = "Camera Trajectory"):
    data = np.load(npz_path)
    idx = data["frame_idx"]
    C = data["C"]  # shape (N,3)

    # sort by frame index to get a consistent trajectory
    order = np.argsort(idx)
    C = C[order]; idx = idx[order]

    xs, ys, zs = C[:,0], C[:,1], C[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=12)
    ax.plot(xs, ys, zs, linewidth=1)

    # origin
    ax.scatter([0], [0], [0], s=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if out_png_path is None:
        out_png_path = os.path.join(os.path.dirname(npz_path), "camera_poses_plot.png")
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to camera_poses.npz")
    ap.add_argument("--out", default=None, help="Output PNG (optional)")
    ap.add_argument("--show", action="store_true", help="Show plot interactively")
    args = ap.parse_args()
    out = plot_camera_poses(args.npz, args.out, args.show)
    print(out)
