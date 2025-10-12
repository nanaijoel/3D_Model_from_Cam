
import sys, os, glob, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def find_pose_file(arg: str | None):
    # 1) explizit: Pfad zur Datei oder Projektordner
    if arg:
        p = os.path.expanduser(arg)
        if os.path.isdir(p):
            # Projektordner -> in poses nachschauen
            cand = sorted(
                glob.glob(os.path.join(p, "poses", "camera_poses.*")),
                key=os.path.getmtime,
                reverse=True,
            )
            if cand:
                return cand[0]
        if os.path.isfile(p):
            return p

    # 2) gleiche Ebene wie Script
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("camera_poses.csv", "camera_poses.npz"):
        candidate = os.path.join(here, name)
        if os.path.isfile(candidate):
            return candidate

    # 3) automatisch in projects/*/poses/
    cands = []
    for ext in ("csv", "npz"):
        cands += glob.glob(os.path.join(here, "projects", "default_project", "poses", f"camera_poses.{ext}"))
    if not cands:
        return None
    return sorted(cands, key=os.path.getmtime, reverse=True)[0]

def load_from_csv(path):
    # Versuche Header-basiert tx,ty,tz zu finden, sonst die letzten 3 Spalten
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = [h.strip().lower() for h in header.split(",")]
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as e:
        raise RuntimeError(f"CSV lesen fehlgeschlagen: {e}")

    if data.ndim == 1:
        data = data[None, :]

    # Suche nach tx,ty,tz
    idx_map = {name: i for i, name in enumerate(cols)}
    if all(k in idx_map for k in ("tx", "ty", "tz")):
        tx, ty, tz = (data[:, idx_map["tx"]],
                      data[:, idx_map["ty"]],
                      data[:, idx_map["tz"]])
    else:
        # fallback: letzte 3 Spalten
        tx, ty, tz = data[:, -3], data[:, -2], data[:, -1]
    return np.vstack([tx, ty, tz]).T

def load_from_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)

    # Häufige Schlüsselvarianten
    cand_R = [k for k in ("poses_R", "R", "rots", "rotations") if k in keys]
    cand_t = [k for k in ("poses_t", "t", "trans", "translations") if k in keys]
    if cand_R and cand_t:
        R = z[cand_R[0]]
        t = z[cand_t[0]]
        # t kann (N,3) sein; manchmal (3,N)
        t = np.asarray(t)
        if t.shape[0] == 3 and t.ndim == 2 and t.shape[1] != 3:
            t = t.T
        return t

    # Manchmal ist Pose als 4x4 pro Frame abgelegt
    for k in keys:
        arr = z[k]
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            T = arr  # (N,4,4)
            xyz = T[:, :3, 3]
            return xyz

    raise RuntimeError(f"Unbekanntes NPZ-Format. Keys: {sorted(keys)}")

def load_positions(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        print(f"[plot_debug] Using CSV: {path}")
        return load_from_csv(path)
    elif ext == ".npz":
        print(f"[plot_debug] Using NPZ: {path}")
        return load_from_npz(path)
    else:
        raise RuntimeError(f"Nicht unterstützte Datei-Endung: {ext}")

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    pose_file = find_pose_file(arg)
    if not pose_file:
        raise FileNotFoundError(
            "Keine Poses-Datei gefunden.\n"
            "Gib entweder an:\n"
            "  - den direkten Pfad zu camera_poses.csv/.npz\n"
            "  - oder den Projektordner (z.B. projects/BUDDHA_AWESOME)\n"
            "Beispiel:\n"
            "  python plot_debug.py projects/BUDDHA_AWESOME\n"
            "  python plot_debug.py projects/BUDDHA_AWESOME/poses/camera_poses.csv"
        )

    P = load_positions(pose_file)  # (N,3)
    if P.size == 0:
        raise RuntimeError("Keine Positionen geladen (leer).")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:,0], P[:,1], P[:,2], marker='o', linewidth=1)
    ax.scatter(P[0,0], P[0,1], P[0,2], s=60)      # Start
    ax.scatter(P[-1,0], P[-1,1], P[-1,2], s=60)   # Ende
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(os.path.basename(pose_file))
    ax.view_init(elev=25, azim=35)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()