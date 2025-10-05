# meshing.py
import os
import numpy as np
import open3d as o3d
import cv2 as cv


# -------------------------------
# Utilities
# -------------------------------
def _median_nn_distance(points_xyz: np.ndarray) -> float:
    """Robuste Schätzung eines mittleren Next-Neighbor-Abstands (mit Fallback)."""
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree  # optional
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz)
        d, _ = tree.query(points_xyz[idx], k=2)
        return float(np.median(d[:, 1]))
    except Exception:
        bb = points_xyz.max(0) - points_xyz.min(0)
        diag = float(np.linalg.norm(bb))
        return max(1e-3, diag / max(300.0, len(points_xyz) ** (1 / 3)))


def _prefilter_mask(m: np.ndarray) -> np.ndarray:
    """Kleine Artefakte entfernen & Kanten beruhigen (sehr billig, bringt viel)."""
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    # median gegen Salz/Pfeffer, dann kleine Löcher/Spitzen entfernen
    m = cv.medianBlur(m, 3)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    m = cv.morphologyEx(m, cv.MORPH_OPEN, se)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, se)
    return m


def _denoise_sparse(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Sehr vorsichtige Ent-Rauschung für sparse.ply, damit Snapping stabil ist."""
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.2)
    except Exception:
        pass
    try:
        nn = _median_nn_distance(np.asarray(pcd.points))
        if nn > 0:
            pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=0.75 * float(nn))
    except Exception:
        pass
    return pcd


def _refine_hull_with_sparse(
    mesh: o3d.geometry.TriangleMesh,
    sparse_pcd: o3d.geometry.PointCloud,
    snap_radius: float,
    snap_weight: float = 0.5,
    on_log=None
) -> o3d.geometry.TriangleMesh:
    """
    Zieht Hull-Vertices leicht in Richtung nächster valider Sparse-Stütze.
    Robust: säubert Sparse, nutzt bevorzugt SciPy-KDTree, sonst Open3D-Radius-Suche.
    - snap_radius: nur bewegen, wenn Stütze innerhalb dieses Radius liegt
    - snap_weight in [0..1]: 0.5 = Mittelwert zwischen Vertex & Stütze
    """
    if (sparse_pcd is None) or (len(sparse_pcd.points) == 0):
        return mesh

    try:
        # --- Sparse säubern (NaNs/Inf/duplikate) + vorsichtig denoisen
        sparse_pcd = _denoise_sparse(sparse_pcd)

        # sanitize numerics
        pts = np.asarray(sparse_pcd.points, dtype=np.float64)
        if pts.size == 0:
            return mesh
        finite = np.all(np.isfinite(pts), axis=1)
        if not np.all(finite):
            pts = pts[finite]
        if pts.size == 0:
            return mesh
        # remove duplicates (auf numpy-Ebene)
        pts = np.unique(np.round(pts, 9), axis=0)

        V = np.asarray(mesh.vertices, dtype=np.float64)
        if V.size == 0:
            return mesh

        r = float(max(1e-9, snap_radius))
        w = float(np.clip(snap_weight, 0.0, 1.0))
        moved = 0

        # --- 1) Versuch: SciPy KD-Tree (schnell & stabil)
        used_scipy = False
        try:
            from scipy.spatial import cKDTree  # optional
            tree = cKDTree(pts)
            d, idx = tree.query(V, k=1, distance_upper_bound=r)
            # cKDTree liefert idx == len(pts) wenn nichts gefunden
            valid = np.isfinite(d) & (d <= r) & (idx < len(pts))
            if np.any(valid):
                V[valid] = V[valid] * (1.0 - w) + pts[idx[valid]] * w
                moved = int(np.count_nonzero(valid))
            used_scipy = True
        except Exception:
            used_scipy = False

        # --- 2) Fallback: Open3D Radius-Suche
        if not used_scipy:
            # KDTreeFlann auf bereinigter PCL (neu zusammenbauen)
            pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            try:
                pcl.remove_duplicated_points()
            except Exception:
                pass
            tree = o3d.geometry.KDTreeFlann(pcl)
            for i, v in enumerate(V):
                try:
                    k, idx, _ = tree.search_radius_vector_3d(v.astype(np.float64), r)
                except Exception:
                    # Manche Open3D-Builds mögen kein float32 – wir geben double.
                    try:
                        k, idx, _ = tree.search_radius_vector_3d(
                            np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64), r
                        )
                    except Exception:
                        continue
                if k <= 0:
                    continue
                neigh = pts[np.asarray(idx, dtype=int)]
                # wähle den nächsten Nachbarn
                j = int(np.argmin(np.sum((neigh - v)**2, axis=1)))
                p = neigh[j]
                V[i] = v * (1.0 - w) + p * w
                moved += 1

        mesh.vertices = o3d.utility.Vector3dVector(V)
        on_log and on_log(f"[vh] snapped {moved} vertices towards sparse cloud (r={r:.6f}, w={w:.2f})")
    except Exception as e:
        on_log and on_log(f"[vh] snap-to-sparse failed (robust version): {e}")
    return mesh



# -------------------------------
# 1) Sparse Point Cloud speichern
# -------------------------------
def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None) -> str:
    """
    Speichert eine (leicht gefilterte) Punktwolke als PLY.
    """
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if len(pts) >= int(filter_min_points):
        nn_med = _median_nn_distance(pts)
        voxel = max(1e-4, 0.5 * nn_med)  # sanft, um feine Strukturen zu erhalten
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path


# -----------------------------------------
# 2) Visual Hull aus Masken + Posen (PLY)
#    + optionales Snapping zur sparse.ply
# -----------------------------------------
def reconstruct_visual_hull_from_masks(
    masks_dir: str,
    poses_npz: str,
    K: np.ndarray,
    image_hw: tuple[int, int],
    mesh_out: str,
    *,
    voxel: float = 0.0,             # 0 ➜ auto (0.5% der Diagonale)
    min_views: float = 0.85,        # float in (0,1] = Anteil der genutzten Masken; int = absolute Zahl
    bbox_expand: float = 0.05,      # Sicherheitsrand um sparse.ply
    smooth_iters: int = 8,
    simplify_tris: int = 150_000,
    max_views: int = 40,
    mask_clean: bool = True,        # Masken vor dem Voting säubern
    snap_to_sparse: bool = True,    # Hull per KD leicht an sparse.ply ausrichten
    snap_radius_mul: float = 1.5,   # Radius = snap_radius_mul * voxel
    snap_weight: float = 0.5,       # 0..1, Anteil Richtung Sparse
    on_log=None, on_progress=None,
) -> str:
    """
    Erzeugt ein Visual-Hull-Mesh aus Silhouettenmasken + Kamera-Posen.
    Erwartet eine vorhandene sparse.ply im selben Ordner wie 'mesh_out' (BBox + optionales Snapping).
    """
    def log(m):  on_log and on_log(m)
    def prog(p, s):  on_progress and on_progress(int(p), s)

    log("[vh] build visual hull")

    # --- 0) Bounds aus sparse.ply ableiten
    project_mesh_dir = os.path.dirname(mesh_out)
    sparse_ply = os.path.join(project_mesh_dir, "sparse.ply")
    pcd = o3d.io.read_point_cloud(sparse_ply)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("No sparse.ply available for bounding box.")
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.0 + float(bbox_expand), bb.get_center())
    mins = bb.get_min_bound(); maxs = bb.get_max_bound()
    diag = float(np.linalg.norm(maxs - mins))

    # --- 1) Voxelauflösung bestimmen + Caps
    if not (voxel and voxel > 0):
        voxel = 0.005 * diag  # 0.5% der Diagonale (auto)

    def axis_count(span, vox):
        return max(8, int(np.ceil(span / max(vox, 1e-9))))

    nx = axis_count(maxs[0]-mins[0], voxel)
    ny = axis_count(maxs[1]-mins[1], voxel)
    nz = axis_count(maxs[2]-mins[2], voxel)

    max_axis  = 220           # Cap pro Achse
    max_total = 8_000_000     # Cap Gesamtanzahl Voxel

    scale_axis = max(nx / max_axis, ny / max_axis, nz / max_axis, 1.0)
    if scale_axis > 1.0:
        voxel *= scale_axis
        nx = axis_count(maxs[0]-mins[0], voxel)
        ny = axis_count(maxs[1]-mins[1], voxel)
        nz = axis_count(maxs[2]-mins[2], voxel)

    total = nx * ny * nz
    if total > max_total:
        factor = (total / max_total) ** (1/3)
        voxel *= factor
        nx = axis_count(maxs[0]-mins[0], voxel)
        ny = axis_count(maxs[1]-mins[1], voxel)
        nz = axis_count(maxs[2]-mins[2], voxel)

    xs = np.linspace(mins[0], maxs[0], nx, dtype=np.float32)
    ys = np.linspace(mins[1], maxs[1], ny, dtype=np.float32)
    zs = np.linspace(mins[2], maxs[2], nz, dtype=np.float32)
    dx = float(xs[1]-xs[0]) if nx > 1 else float(voxel)
    dy = float(ys[1]-ys[0]) if ny > 1 else float(voxel)
    dz = float(zs[1]-zs[0]) if nz > 1 else float(voxel)

    H, W = map(int, image_hw)
    K = np.asarray(K, float)

    log(f"[vh] bbox diag={diag:.3f}  voxel={float(voxel):.5f}  grid=({nx},{ny},{nz})  total≈{nx*ny*nz:,}")

    # --- 2) Posen laden (R,t: world->cam)
    data = np.load(poses_npz, allow_pickle=True)
    frame_idx = data.get("frame_idx")
    Rs = data.get("R")
    ts = data.get("t")
    if frame_idx is None or Rs is None or ts is None:
        raise RuntimeError("poses npz must contain frame_idx, R, t")

    pose_map = {int(i): (R.astype(np.float32), t.astype(np.float32)) for i, R, t in zip(frame_idx, Rs, ts)}

    # --- 3) Masken sammeln (nur Frames mit Pose) und ggf. subsamplen
    all_names = sorted([f for f in os.listdir(masks_dir) if f.endswith("_mask.png")])
    mask_items = []
    for name in all_names:
        try:
            fi = int(name.split("_")[1])  # frame_XXXX_...
        except Exception:
            continue
        if fi not in pose_map:
            continue
        path = os.path.join(masks_dir, name)
        m = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            m = cv.resize(m, (W, H), interpolation=cv.INTER_NEAREST)
        if mask_clean:
            m = _prefilter_mask(m)
        mask_items.append((fi, (m > 0)))

    if not mask_items:
        raise RuntimeError("No masks found that match available poses.")

    # gleichmäßig auf max_views reduzieren
    if len(mask_items) > max_views:
        step = len(mask_items) / float(max_views)
        picked = [mask_items[int(round(i*step))] for i in range(max_views)]
        seen, picked2 = set(), []
        for fi, m in picked:
            if fi in seen:
                continue
            seen.add(fi)
            picked2.append((fi, m))
        mask_items = sorted(picked2, key=lambda x: x[0])

    nviews = len(mask_items)
    if nviews < 3:
        raise RuntimeError(f"Too few masks for visual hull (got {nviews}).")

    # adaptiver Mindest-Vote
    if isinstance(min_views, float) and (0 < min_views <= 1.0):
        need_votes = int(np.ceil(min_views * nviews))
    else:
        need_votes = int(min_views)
    need_votes = int(np.clip(need_votes, 2, nviews))
    log(f"[vh] nviews={nviews}  min_votes={need_votes}")

    # zugehörige Projektionsmatrizen exakt passend zu mask_items
    Pmats = []
    for fi, _ in mask_items:
        R, t = pose_map[fi]
        P = K @ np.hstack([R, t])  # 3x4
        Pmats.append(P.astype(np.float32))

    # --- 4) Voting (slabweise) ins 3D-Occupancy-Grid
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')  # (ny, nz)

    def proj(P, X):
        Xh = np.hstack([X, np.ones((X.shape[0], 1), np.float32)])
        u = (P @ Xh.T).T
        u = u[:, :2] / np.maximum(u[:, 2:3], 1e-9)
        return u

    for ix, x in enumerate(xs):
        prog(40 + 40*(ix+1)/len(xs), "Visual hull")
        slab = np.stack(
            [np.full_like(Y, x, dtype=np.float32), Y.astype(np.float32), Z.astype(np.float32)],
            axis=-1
        ).reshape(-1, 3)

        votes = np.zeros((slab.shape[0],), dtype=np.int32)
        for (fi, m), P in zip(mask_items, Pmats):
            uv = proj(P, slab)
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not np.any(valid):
                continue
            iu = np.clip(np.round(uv[valid, 0]).astype(np.int32), 0, W - 1)
            iv = np.clip(np.round(uv[valid, 1]).astype(np.int32), 0, H - 1)
            votes[valid] += m[iv, iu].astype(np.int32)  # bool -> 0/1

        keep = (votes >= need_votes)
        if np.any(keep):
            occ[ix].reshape(-1)[:] = keep.astype(np.uint8)

    if occ.sum() == 0:
        raise RuntimeError("Visual hull produced empty occupancy.")

    # --- 5) Marching Cubes → Mesh
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(
            occ.astype(np.uint8), level=0.5, spacing=(dx, dy, dz)
        )
        verts = verts + np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
    except Exception as e:
        log(f"[vh] marching_cubes unavailable ({e}); fallback to alpha-shape")
        # Fallback: Alpha-Shape auf Voxelzentren
        pts = []
        for ix in range(nx):
            yy, zz = np.where(occ[ix] > 0)
            if yy.size == 0:
                continue
            px = xs[ix]
            pts.append(np.stack([np.full_like(yy, px, dtype=np.float32), ys[yy], zs[zz]], axis=1))
        pts = np.vstack(pts)
        pcd_hull = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd_hull.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd_hull, alpha=1.8 * float(voxel)
        )

    # --- 6) (NEU) Snap zu sparse.ply (Detail-Rückgewinnung, bleibt wasserdicht)
    if snap_to_sparse:
        try:
            pcd_sparse = o3d.io.read_point_cloud(sparse_ply)
            if (pcd_sparse is not None) and (len(pcd_sparse.points) > 0):
                snap_r = float(max(1e-9, snap_radius_mul)) * float(voxel if voxel > 0 else dz)
                mesh = _refine_hull_with_sparse(
                    mesh, pcd_sparse, snap_radius=snap_r, snap_weight=float(snap_weight), on_log=on_log
                )
        except Exception as e:
            log(f"[vh] snap skipped: {e}")

    # --- 7) Cleanup / Glättung / Vereinfachung
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if smooth_iters > 0:
        try:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
        except Exception:
            pass
    if simplify_tris > 0 and len(mesh.triangles) > simplify_tris:
        mesh = mesh.simplify_quadric_decimation(int(simplify_tris))
    mesh.compute_vertex_normals()

    # --- 8) Schreiben
    os.makedirs(os.path.dirname(mesh_out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(mesh_out, mesh)
    if not ok:
        raise RuntimeError(f"Failed to write hull mesh: {mesh_out}")

    log(f"[vh] mesh -> {mesh_out} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")
    prog(85, "Visual hull mesh")
    return mesh_out
