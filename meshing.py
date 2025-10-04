# meshing.py
import os
from typing import Optional
import numpy as np
import open3d as o3d
import cv2 as cv



# ------------------------------------------------------------
# Basic: save point cloud (leichtes Downsampling/Outlier-Filter)
# ------------------------------------------------------------

def _median_nn_distance(points_xyz: np.ndarray) -> float:
    """
    Schätzt einen robusten mittleren nächsten Nachbar-Abstand.
    Nutzt scipy.cKDTree, fällt bei fehlender Abhängigkeit auf AABB-Heuristik zurück.
    """
    if len(points_xyz) < 3:
        return 0.01
    try:
        from scipy.spatial import cKDTree  # optional dependency
        idx = np.random.choice(len(points_xyz), size=min(4000, len(points_xyz)), replace=False)
        tree = cKDTree(points_xyz)
        d, _ = tree.query(points_xyz[idx], k=2)
        return float(np.median(d[:, 1]))
    except Exception:
        bb = points_xyz.max(0) - points_xyz.min(0)
        diag = float(np.linalg.norm(bb))
        return max(1e-3, diag / max(300.0, len(points_xyz) ** (1 / 3)))


def save_point_cloud(points_xyz: np.ndarray, out_path: str, filter_min_points: int = 1000,
                     on_log=None, on_progress=None):
    """
    Speichert eine Punktwolke als PLY (mit sehr sanfter Verdichtung & Outlier-Filter).
    """
    on_log and on_log(f"[mesh] save point cloud -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if len(pts) >= int(filter_min_points):
        nn_med = _median_nn_distance(pts)
        voxel = max(1e-4, 0.5 * nn_med)  # adaptiv; nicht zu grob, um dünne Strukturen zu erhalten
        try:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
        except Exception:
            pass
        try:
            # sehr sanft, um dünne Strukturen zu bewahren
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
        except Exception:
            pass

    o3d.io.write_point_cloud(out_path, pcd)
    on_progress and on_progress(100, "Save PLY")
    return out_path


# ------------------------------------------------------------
# Helpers: Normals, Meshing und Nachbearbeitung
# ------------------------------------------------------------

def _diag_from_pcd(pcd: o3d.geometry.PointCloud) -> float:
    bb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(bb.get_extent(), dtype=float)
    return float(np.linalg.norm(ext))


def _estimate_normals_o3d(pcd: o3d.geometry.PointCloud, k: int = 40):
    """
    Normalschätzung mit hybrider Suche (Radius+max_nn), anschließend konsistente Orientierung.
    """
    diag = _diag_from_pcd(pcd)
    radius = max(1e-3, 0.02 * diag)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(k))
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(int(k))
    except Exception:
        # Fallback: gegen „virtuelle Kamera“ aus der Mitte orientieren
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))


def _load_camera_centers_npz(npz_path: str) -> Optional[np.ndarray]:
    """
    Versucht Kamerazentren aus einer NPZ-Datei zu laden.
    Erwartete Varianten:
      - R_list (Nx3x3), t_list (Nx3) => C = -R^T t
      - Rs (Nx3x3), ts (Nx3)
      - C_list (Nx3) direkt
      - poses (N, 2) mit (R, t)
    Gibt None zurück, wenn nichts Passendes gefunden wird.
    """
    try:
        if (npz_path is None) or (not os.path.isfile(npz_path)):
            return None
        data = np.load(npz_path, allow_pickle=True)
        if "C_list" in data:
            C = np.asarray(data["C_list"], dtype=float)
            return C if C.size else None

        def _Rt_to_C(R_arr, t_arr):
            centers = []
            for R, t in zip(R_arr, t_arr):
                R = np.asarray(R, float).reshape(3, 3)
                t = np.asarray(t, float).reshape(3)
                C = -(R.T @ t.reshape(3, 1)).reshape(3)
                centers.append(C)
            return np.asarray(centers, float)

        if ("R_list" in data) and ("t_list" in data):
            return _Rt_to_C(data["R_list"], data["t_list"])
        if ("Rs" in data) and ("ts" in data):
            return _Rt_to_C(data["Rs"], data["ts"])
        if "poses" in data:
            poses = data["poses"]
            centers = []
            for item in poses:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    R, t = item
                    R = np.asarray(R, float).reshape(3, 3)
                    t = np.asarray(t, float).reshape(3)
                    C = -(R.T @ t.reshape(3, 1)).reshape(3)
                    centers.append(C)
            if centers:
                return np.asarray(centers, float)
    except Exception:
        return None
    return None


def _orient_normals_from_poses(pcd: o3d.geometry.PointCloud, poses_npz: Optional[str]):
    """
    Orientiert Normalen in Richtung einer robusten „Kameraposition“ (Median aller Zentren).
    Fällt auf 'towards_center' zurück, wenn keine Posen gefunden werden.
    """
    centers = _load_camera_centers_npz(poses_npz) if poses_npz else None
    if centers is None or len(centers) == 0:
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(np.asarray(center, dtype=float))
        return
    med = np.median(centers, axis=0).astype(float)
    pcd.orient_normals_towards_camera_location(med)


def _mesh_poisson(pcd: o3d.geometry.PointCloud, depth=10, scale=1.1, linear_fit=True):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=float(scale), linear_fit=bool(linear_fit)
    )
    return mesh, np.asarray(dens, float)


def _mesh_alpha(pcd: o3d.geometry.PointCloud, alpha: float):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))


def _mesh_cleanup(mesh: o3d.geometry.TriangleMesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_crop_bbox(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand_ratio: float):
    if expand_ratio <= 0:
        return mesh
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.0 + float(expand_ratio), bb.get_center())
    try:
        return mesh.crop(bb)
    except Exception:
        # Fallback: manuelles Clipping gegen AABB
        mins = bb.get_min_bound()
        maxs = bb.get_max_bound()
        verts = np.asarray(mesh.vertices)
        inside = np.all((verts >= mins) & (verts <= maxs), axis=1)
        mesh.remove_vertices_by_mask(~inside)
        return mesh


def _trim_by_density(mesh: o3d.geometry.TriangleMesh, densities: np.ndarray, q: float):
    """
    Entfernt sehr dünn besetzte Bereiche anhand Poisson-Dichte-Quantil.
    """
    if densities is None or len(densities) != len(mesh.vertices) or q <= 0:
        return mesh
    thr = float(np.quantile(densities, np.clip(q, 0.0, 0.5)))
    mask = densities < thr
    mesh.remove_vertices_by_mask(mask)
    return mesh


def _mesh_smooth(mesh: o3d.geometry.TriangleMesh, iters=8):
    if iters <= 0:
        return mesh
    try:
        return mesh.filter_smooth_taubin(number_of_iterations=int(iters))
    except Exception:
        return mesh.filter_smooth_simple(number_of_iterations=int(iters))


def _mesh_simplify(mesh: o3d.geometry.TriangleMesh, target_tris=0):
    if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
        return mesh.simplify_quadric_decimation(int(target_tris))
    return mesh


def _transfer_vertex_colors(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, k=1):
    if not pcd.has_colors():
        return mesh
    tree = o3d.geometry.KDTreeFlann(pcd)
    mverts = np.asarray(mesh.vertices)
    pcols = np.asarray(pcd.colors)
    cols = np.zeros((len(mverts), 3), dtype=np.float64)
    for i, v in enumerate(mverts):
        _, idx, _ = tree.search_knn_vector_3d(v, max(1, int(k)))
        cols[i] = np.mean(pcols[idx], axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    return mesh


def _prefilter_pcd(
    pcd: o3d.geometry.PointCloud,
    nn_med: float,
    do_radius: bool,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """
    Vorsichtige Vorfilterung: Statistischer Outlier-Filter + optional Radius-Filter.
    """
    try:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=max(8, int(nb_neighbors)),
            std_ratio=float(std_ratio)
        )
    except Exception:
        pass
    if do_radius and (nn_med > 0):
        try:
            rad = 0.75 * float(nn_med)
            pcd, _ = pcd.remove_radius_outlier(nb_points=3, radius=max(1e-6, rad))
        except Exception:
            pass
    return pcd


def _remove_small_components(
    mesh: o3d.geometry.TriangleMesh,
    min_tris_ratio: float = 0.002,
    min_area_ratio: float = 0.001
) -> o3d.geometry.TriangleMesh:
    """
    Entfernt sehr kleine, zusammenhängende Dreiecks-Komponenten (Tri- und Flächenanteil).
    """
    try:
        cl, n_tris, areas = mesh.cluster_connected_triangles()
        n_tris = np.asarray(n_tris)
        areas = np.asarray(areas, float)
        total_tris = max(1, len(mesh.triangles))
        keep = np.ones(len(n_tris), dtype=bool)

        if min_tris_ratio > 0:
            keep &= (n_tris >= int(np.ceil(min_tris_ratio * total_tris)))

        if (areas.size > 0) and (min_area_ratio > 0):
            total_area = float(np.sum(areas))
            if total_area > 0:
                keep &= (areas >= (min_area_ratio * total_area))

        tri_labels = np.asarray(cl)
        tri_mask = ~keep[tri_labels]
        mesh.remove_triangles_by_mask(tri_mask)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


# ------------------------------------------------------------
# Main: Poisson/Alpha aus PLY erzeugen (watertight)
# ------------------------------------------------------------

def reconstruct_solid_mesh_from_ply(
        ply_in: str,
        mesh_out: str,
        *,
        method: str = "poisson",
        depth: int = 11,
        scale: float = 1.10,
        no_linear_fit: bool = False,
        dens_quantile: float = 0.0,
        alpha: float = 0.0,
        bbox_expand: float = 0.02,
        pre_filter: bool = False,
        pre_filter_neighbors: int = 24,
        pre_filter_std: float = 1.2,
        pre_filter_radius: bool = False,
        voxel: float = 0.0,
        normals_k: int = 40,
        smooth: int = 6,
        simplify: int = 0,
        color_transfer: bool = False,
        poses_npz: Optional[str] = None,
        remove_small_comp: bool = False,
        min_comp_tris_ratio: float = 0.002,
        min_comp_area_ratio: float = 0.001,
        on_log=None,
        on_progress=None,
    ) -> str:
    """
    Erzeugt ein wasserfestes Mesh direkt aus einer PLY-Punktwolke.
    """
    on_log and on_log(f"[mesh] solid: load cloud -> {ply_in}")
    pcd = o3d.io.read_point_cloud(ply_in)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("Loaded point cloud is empty.")

    pts_np = np.asarray(pcd.points)
    nn_med = _median_nn_distance(pts_np)

    # Vorfilterung
    if pre_filter:
        pcd = _prefilter_pcd(
            pcd,
            nn_med=nn_med,
            do_radius=bool(pre_filter_radius),
            nb_neighbors=int(pre_filter_neighbors),
            std_ratio=float(pre_filter_std)
        )

    # Normale + Orientierung (zuerst konsistent, dann zur Kamera)
    _estimate_normals_o3d(pcd, k=int(normals_k))
    if poses_npz:
        _orient_normals_from_poses(pcd, poses_npz)

    # Meshing
    if method == "poisson":
        mesh, dens = _mesh_poisson(
            pcd,
            depth=int(depth),
            scale=float(scale),
            linear_fit=(not bool(no_linear_fit))
        )
        if float(dens_quantile) > 0:
            mesh = _trim_by_density(mesh, dens, float(dens_quantile))
    elif method == "alpha":
        if float(alpha) <= 0.0:
            # Alpha automatisch aus NN-Abstand schätzen
            nn = _median_nn_distance(np.asarray(pcd.points))
            alpha = 2.5 * float(nn)
        mesh = _mesh_alpha(pcd, float(alpha))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bounding-Box beschneiden, Cleanup und Fragment-Entfernung
    mesh = _mesh_crop_bbox(mesh, pcd, expand_ratio=float(bbox_expand))
    mesh = _mesh_cleanup(mesh)
    if bool(remove_small_comp):
        mesh = _remove_small_components(
            mesh,
            min_tris_ratio=float(min_comp_tris_ratio),
            min_area_ratio=float(min_comp_area_ratio)
        )

    # Glätten / Vereinfachen
    if int(smooth) > 0:
        mesh = _mesh_smooth(mesh, iters=int(smooth))
    if int(simplify) > 0:
        mesh = _mesh_simplify(mesh, target_tris=int(simplify))

    mesh.compute_vertex_normals()
    if bool(color_transfer):
        mesh = _transfer_vertex_colors(mesh, pcd, k=1)

    # Schreiben
    os.makedirs(os.path.dirname(mesh_out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(mesh_out, mesh, write_vertex_colors=bool(color_transfer))
    if not ok:
        raise RuntimeError(f"Failed to write mesh: {mesh_out}")

    on_log and on_log(f"[mesh] solid saved -> {mesh_out} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")
    on_progress and on_progress(100, "Watertight Mesh")
    return mesh_out


# --- Visual Hull aus Masken + Posen -----------------------------------------

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
    max_views: int = 40,            # wir sampeln höchstens so viele Masken gleichmäßig
    on_log=None, on_progress=None,
) -> str:
    """
    Visual-Hull aus Masken + Posen. Erwartet Masken in:
      <project>/features/masks/frame_XXXX*_mask.png
    'image_hw' = (H,W) des Bild-/Maskenrasters (OHNE Rotation).
    """
    def log(m):
        on_log and on_log(m)
    def prog(p, s):
        on_progress and on_progress(int(p), s)

    log("[vh] build visual hull")

    # --- 0) Bounds aus sparse.ply ableiten (robust, kompakt)
    project_mesh_dir = os.path.dirname(mesh_out)
    sparse_ply = os.path.join(project_mesh_dir, "sparse.ply")
    pcd = o3d.io.read_point_cloud(sparse_ply)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("No sparse.ply available for bounding box.")
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.0 + float(bbox_expand), bb.get_center())
    mins = bb.get_min_bound(); maxs = bb.get_max_bound()
    diag = float(np.linalg.norm(maxs - mins))

    # --- 1) Voxelauflösung bestimmen + Caps (RAM-sicher)
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
        mask_items.append((fi, (m > 0)))

    if not mask_items:
        raise RuntimeError("No masks found that match available poses.")

    # gleichmäßig auf max_views runter sampeln
    if len(mask_items) > max_views:
        step = len(mask_items) / float(max_views)
        picked = [mask_items[int(round(i*step))] for i in range(max_views)]
        # uniq & sort
        seen, picked2 = set(), []
        for fi, m in picked:
            if fi in seen:
                continue
            seen.add(fi); picked2.append((fi, m))
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
        slab = np.stack([
            np.full_like(Y, x, dtype=np.float32),
            Y.astype(np.float32),
            Z.astype(np.float32)
        ], axis=-1).reshape(-1, 3)

        votes = np.zeros((slab.shape[0],), dtype=np.int32)
        for (fi, m), P in zip(mask_items, Pmats):
            uv = proj(P, slab)
            valid = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
            if not np.any(valid):
                continue
            iu = np.clip(np.round(uv[valid,0]).astype(np.int32), 0, W-1)
            iv = np.clip(np.round(uv[valid,1]).astype(np.int32), 0, H-1)
            votes[valid] += m[iv, iu].astype(np.int32)  # bool -> 0/1

        keep = (votes >= need_votes)
        if np.any(keep):
            occ[ix].reshape(-1)[:] = keep.astype(np.uint8)

    if occ.sum() == 0:
        raise RuntimeError("Visual hull produced empty occupancy.")

    # --- 5) Marching Cubes über occ (robust & schnell)
    mesh = None
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(
            occ.astype(np.uint8), level=0.5, spacing=(dx, dy, dz)
        )
        # marching_cubes liefert Koords relativ zu (0,0,0) des Grids -> in Welt verschieben
        verts = verts + np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.remove_duplicated_vertices(); mesh.remove_degenerate_triangles()
    except Exception as e:
        log(f"[vh] marching_cubes unavailable ({e}); fallback to alpha-shape")
        # Fallback: Alpha-Shape auf Voxelzentren (langsamer, aber funktioniert)
        pts = []
        for ix in range(nx):
            yy, zz = np.where(occ[ix] > 0)
            if yy.size == 0:
                continue
            px = xs[ix]
            pts.append(np.stack([np.full_like(yy, px, dtype=np.float32), ys[yy], zs[zz]], axis=1))
        pts = np.vstack(pts)
        pcd_hull = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        # einfache Normalen (für spätere Glättung nicht zwingend nötig)
        pcd_hull.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_hull, alpha=1.8*float(voxel))

    # Cleanup, Glätten, Vereinfachen
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
    if simplify_tris > 0 and len(mesh.triangles) > simplify_tris:
        mesh = mesh.simplify_quadric_decimation(int(simplify_tris))

    mesh.compute_vertex_normals()

    # schreiben
    os.makedirs(os.path.dirname(mesh_out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(mesh_out, mesh)
    if not ok:
        raise RuntimeError(f"Failed to write hull mesh: {mesh_out}")

    log(f"[vh] mesh -> {mesh_out} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")
    prog(85, "Visual hull mesh")
    return mesh_out


# ---- cloud filter by silhouette votes ----
def filter_cloud_by_masks(
    ply_in: str, masks_dir: str, poses_npz: str, K: np.ndarray,
    image_hw: tuple[int,int], out_path: str, min_views: int = 3,
    on_log=None, on_progress=None
) -> str:
    import cv2 as cv, numpy as np, os
    import open3d as o3d
    on_log and on_log(f"[mesh] filter cloud by masks (min_views={min_views})")

    # load cloud
    pcd = o3d.io.read_point_cloud(ply_in)
    if (pcd is None) or (len(pcd.points) == 0):
        raise RuntimeError("Input cloud is empty.")
    P = np.asarray(pcd.points, np.float32)

    # intrinsics & image size
    H, W = image_hw
    K = np.asarray(K, np.float32)

    # load poses
    data = np.load(poses_npz, allow_pickle=True)
    Rs, ts, frame_idx = data["R"], data["t"], data["frame_idx"]
    # collect masks in dict
    mdict = {}
    for fi in frame_idx:
        fi = int(fi)
        cand = [f for f in os.listdir(masks_dir)
                if f.startswith(f"frame_{fi:04d}_") and f.endswith("_mask.png")]
        if not cand:
            continue
        m = cv.imread(os.path.join(masks_dir, sorted(cand)[0]), cv.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            m = cv.resize(m, (W, H), interpolation=cv.INTER_NEAREST)
        mdict[fi] = (m > 0)

    if len(mdict) < max(3, min_views):
        on_log and on_log("[mesh] warn: too few masks; returning original cloud.")
        o3d.io.write_point_cloud(out_path, pcd)
        return out_path

    # projection helpers
    Ps = [K @ np.hstack([R, t]) for R, t in zip(Rs, ts)]
    votes = np.zeros(len(P), np.int32)

    # process in chunks to keep RAM low
    B = 200000
    for s in range(0, len(P), B):
        on_progress and on_progress(70 + int(10*s/len(P)), "Mask filter")
        X = P[s:s+B]
        Xh = np.concatenate([X, np.ones((len(X),1), np.float32)], axis=1).T  # 4xB
        for Pmat, fi in zip(Ps, frame_idx):
            if int(fi) not in mdict:
                continue
            uvw = (Pmat @ Xh).T   # Bx3
            wpos = uvw[:,2] > 1e-6
            u = uvw[wpos,0] / uvw[wpos,2]
            v = uvw[wpos,1] / uvw[wpos,2]
            valid = (u>=0)&(u<W)&(v>=0)&(v<H)
            if not np.any(valid):
                continue
            mu = np.clip(np.round(u[valid]).astype(np.int32), 0, W-1)
            mv = np.clip(np.round(v[valid]).astype(np.int32), 0, H-1)
            m = mdict[int(fi)]
            idx = np.where(wpos)[0][valid]
            votes[s + idx] += m[mv, mu].astype(np.int32)

    keep = votes >= int(min_views)  # m ist bool→0/1
    pcd_f = pcd.select_by_index(np.where(keep)[0])
    on_log and on_log(f"[mesh] mask-filter: kept {len(pcd_f.points)}/{len(pcd.points)} points")
    o3d.io.write_point_cloud(out_path, pcd_f)
    return out_path
