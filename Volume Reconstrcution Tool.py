import os
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend for cluster runs
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

# from IPython.display import display
# import ipywidgets as widgets

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from skimage.measure import marching_cubes
from itertools import combinations


# ============================================================
# User parameters
# ============================================================

NPY_PATH = "sim_4000.npy"

# ------------------------------------------------------------
# Timestep control
# ------------------------------------------------------------
# Save/process every 4th timestep:
#   0, 4, 8, 12, ...
#
# Change to 1 if you want every timestep.
TIMESTEP_STRIDE = 4

# Create a browser-playable Plotly HTML animation from the
# selected timesteps.
SAVE_TIMESTEP_ANIMATION = True

# Maximum number of points displayed in each animation frame.
# This controls HTML size significantly.
ANIMATION_PC_MAX = 20000

# Output filename for timestep animation.
TIMESTEP_ANIMATION_FILENAME = None
# None -> automatically use:
# <base_name>_every_4_timesteps.html


# ------------------------------------------------------------
# Clustering
# ------------------------------------------------------------

DBSCAN_EPS = 3.0
DBSCAN_MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 50


# ------------------------------------------------------------
# Normals estimation
# ------------------------------------------------------------

K_NORMALS = 10
ORIENT_K = 4


# ------------------------------------------------------------
# MLS SDF parameters
# ------------------------------------------------------------

MLS_K = 12
MLS_SIGMA_MODE = "kth"   # 'kth' | 'median' | float
MLS_SIGMA_SCALE = 1.0

# Narrow band: MLS is only evaluated at grid points within this
# distance of a cell point; farther points are marked "outside".
# MUST be larger than the radius of your thickest vessel
# (same units as the point coordinates).
SDF_BAND = 20.0


# ------------------------------------------------------------
# Grid for cluster boxes
# ------------------------------------------------------------

VOXEL_SIZE = 0.5
PADDING_FRAC = 0.20


# ------------------------------------------------------------
# Plotting control
# ------------------------------------------------------------

DS_VOL = 2
PC_MAX = 20000
PLOT_EACH_CLUSTER = False
PLOT_ALL_CLUSTERS_TOGETHER = True
TARGET_RANK = None


# ------------------------------------------------------------
# HTML saving
# ------------------------------------------------------------

SAVE_IMAGES = True
IMAGE_FORMAT = "html"

# If a Plotly figure already contains frames, this additionally
# reduces the frames stored in that figure before writing HTML.
#
# For the timestep animation generated below, the selected
# timesteps are already spaced by TIMESTEP_STRIDE.
HTML_FRAME_STRIDE = 1


# ------------------------------------------------------------
# Colab interactive editing
# ------------------------------------------------------------

ENABLE_COLAB_WIDGETS = False
AUTO_OPEN_EDITOR = False


# ============================================================
# Paths / filenames
# ============================================================

input_path = Path(NPY_PATH)
base_name = input_path.stem

output_dir = (
    input_path.parent
    if str(input_path.parent) != ""
    else Path(".")
)

summary_csv_path = output_dir / f"{base_name}_summary.csv"

if TIMESTEP_ANIMATION_FILENAME is None:
    timestep_animation_path = (
        output_dir /
        f"{base_name}_every_{TIMESTEP_STRIDE}_timesteps.html"
    )
else:
    timestep_animation_path = output_dir / TIMESTEP_ANIMATION_FILENAME


# ============================================================
# Timestep loading utilities
# ============================================================

def load_all_timesteps(path):
    """
    Load the complete timestep sequence from an NPY file.

    Expected common formats:

        object array:
            data[timestep] -> (N, 3)

        regular ndarray:
            data.shape == (T, N, 3)

    Returns
    -------
    list of np.ndarray
        One (N,3) point cloud per timestep.
    """

    data = np.load(path, allow_pickle=True)

    print(f"[load] raw array type : {type(data)}")
    print(f"[load] raw array shape: {getattr(data, 'shape', None)}")
    print(f"[load] raw array dtype: {getattr(data, 'dtype', None)}")

    # --------------------------------------------------------
    # Object array
    # --------------------------------------------------------
    if isinstance(data, np.ndarray) and data.dtype == object:
        timesteps = []

        for i in range(len(data)):
            pts = np.asarray(data[i])

            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"Timestep {i}: expected (N,3), got {pts.shape}"
                )

            timesteps.append(pts)

        return timesteps

    # --------------------------------------------------------
    # Regular 3D ndarray: (T, N, 3)
    # --------------------------------------------------------
    if isinstance(data, np.ndarray):
        if data.ndim == 3 and data.shape[-1] == 3:
            return [
                np.asarray(data[i])
                for i in range(data.shape[0])
            ]

        # ----------------------------------------------------
        # Single point cloud: (N,3)
        # ----------------------------------------------------
        if data.ndim == 2 and data.shape[1] == 3:
            return [np.asarray(data)]

    # --------------------------------------------------------
    # Python list
    # --------------------------------------------------------
    if isinstance(data, list):
        timesteps = []

        for i, item in enumerate(data):
            pts = np.asarray(item)

            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"Timestep {i}: expected (N,3), got {pts.shape}"
                )

            timesteps.append(pts)

        return timesteps

    raise TypeError(
        "Unsupported NPY structure. Expected a list/object array "
        "of (N,3) point clouds or an ndarray of shape (T,N,3)."
    )


def select_timesteps(timesteps, stride=4):
    """
    Select every `stride` timestep.

    Example:
        stride=4
        -> timestep indices 0,4,8,12,...

    The final timestep is also included if it is not already
    represented by the regular stride.
    """

    if stride < 1:
        raise ValueError("TIMESTEP_STRIDE must be >= 1.")

    n = len(timesteps)

    if n == 0:
        return [], []

    indices = list(range(0, n, stride))

    # Include the final timestep so the animation reaches
    # the final physical state.
    if indices[-1] != n - 1:
        indices.append(n - 1)

    selected = [timesteps[i] for i in indices]

    return indices, selected


def safe_load_last_pointcloud(path):
    """
    Backward-compatible helper.

    Loads only the final timestep.
    """
    timesteps = load_all_timesteps(path)

    if len(timesteps) == 0:
        raise ValueError("No timesteps found.")

    return timesteps[-1]


# ============================================================
# Timestep animation
# ============================================================

def _downsample_points(points, max_points):
    """
    Randomly downsample a point cloud for visualization only.
    """
    if points.shape[0] <= max_points:
        return points

    idx = np.random.choice(
        points.shape[0],
        max_points,
        replace=False
    )

    return points[idx]


def make_timestep_animation(
    timesteps,
    timestep_indices,
    output_path,
    pc_max=20000
):
    """
    Create a lightweight browser-playable Plotly HTML animation
    from selected point-cloud timesteps.

    Only the selected timesteps are placed into the HTML.

    Example:
        timestep_indices = [0,4,8,12,...]
    """

    if len(timesteps) == 0:
        print("[animation] no timesteps supplied.")
        return

    print("\n[animation] creating timestep animation...")
    print(f"[animation] number of frames: {len(timesteps)}")
    print(f"[animation] timestep indices: {timestep_indices}")

    # --------------------------------------------------------
    # Determine common axis ranges from all selected frames.
    # This prevents the camera from jumping during animation.
    # --------------------------------------------------------

    all_mins = []
    all_maxs = []

    for pts in timesteps:
        if pts.size == 0:
            continue

        all_mins.append(np.min(pts, axis=0))
        all_maxs.append(np.max(pts, axis=0))

    if not all_mins:
        print("[animation] no valid points.")
        return

    global_min = np.min(np.vstack(all_mins), axis=0)
    global_max = np.max(np.vstack(all_maxs), axis=0)

    center = 0.5 * (global_min + global_max)
    span = global_max - global_min
    max_span = float(np.max(span))

    if max_span <= 0:
        max_span = 1.0

    half = 0.55 * max_span

    x_range = [center[0] - half, center[0] + half]
    y_range = [center[1] - half, center[1] + half]
    z_range = [center[2] - half, center[2] + half]

    # --------------------------------------------------------
    # Prepare first frame
    # --------------------------------------------------------

    first_pts = _downsample_points(
        timesteps[0],
        pc_max
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=first_pts[:, 0],
            y=first_pts[:, 1],
            z=first_pts[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.65,
                color="gray"
            ),
            name="Point cloud"
        )
    )

    # --------------------------------------------------------
    # Build only selected frames.
    #
    # Since `timesteps` already contains only:
    # 0,4,8,12,...
    #
    # the HTML contains no intermediate frames.
    # --------------------------------------------------------

    frames = []

    for pts, timestep_idx in zip(
        timesteps,
        timestep_indices
    ):

        pts_plot = _downsample_points(
            pts,
            pc_max
        )

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=pts_plot[:, 0],
                    y=pts_plot[:, 1],
                    z=pts_plot[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        opacity=0.65,
                        color="gray"
                    ),
                    name="Point cloud"
                )
            ],
            name=str(timestep_idx),
            layout=go.Layout(
                title=f"Timestep {timestep_idx}"
            )
        )

        frames.append(frame)

    fig.frames = frames

    # --------------------------------------------------------
    # Slider
    # --------------------------------------------------------

    slider_steps = []

    for timestep_idx in timestep_indices:
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(timestep_idx)],
                    dict(
                        mode="immediate",
                        frame=dict(
                            duration=100,
                            redraw=True
                        ),
                        transition=dict(
                            duration=0
                        )
                    )
                ],
                label=str(timestep_idx)
            )
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=f"Timestep {timestep_indices[0]}",
        scene=dict(
            xaxis=dict(
                title="x",
                range=x_range
            ),
            yaxis=dict(
                title="y",
                range=y_range
            ),
            zaxis=dict(
                title="z",
                range=z_range
            ),
            aspectmode="data"
        ),
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),
        showlegend=True,

        # Play button
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.10,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=150,
                                    redraw=True
                                ),
                                transition=dict(
                                    duration=0
                                ),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(
                                    duration=0,
                                    redraw=False
                                ),
                                transition=dict(
                                    duration=0
                                ),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],

        sliders=[
            dict(
                active=0,
                x=0.05,
                y=0.02,
                len=0.90,
                currentvalue=dict(
                    prefix="Timestep: "
                ),
                steps=slider_steps
            )
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(output_path)
    output_path = output_path.with_suffix(".html")

    print(
        f"[animation] writing HTML with "
        f"{len(fig.frames)} frames..."
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn"
    )

    file_size_mb = (
        output_path.stat().st_size /
        (1024 * 1024)
    )

    print(
        f"[save] timestep animation written: "
        f"{output_path}"
    )

    print(
        f"[save] HTML size: "
        f"{file_size_mb:.2f} MB"
    )


# ============================================================
# Utilities
# ============================================================

def estimate_normals_oriented(points, k=20, orient_k=8):
    """
    Estimate normals via PCA on k-NN and propagate orientation
    via a graph walk.
    """

    tree = cKDTree(points)
    N = points.shape[0]

    normals = np.zeros_like(
        points,
        dtype=np.float64
    )

    for i, p in enumerate(points):

        _, idx = tree.query(
            p,
            k=min(k, N)
        )

        nbrs = points[idx]

        pca = PCA(n_components=3)
        pca.fit(nbrs - nbrs.mean(axis=0))

        n = pca.components_[-1]

        normals[i] = n / (
            np.linalg.norm(n) + 1e-12
        )

    knn_idx = tree.query(
        points,
        k=min(orient_k + 1, N)
    )[1][:, 1:]

    visited = np.zeros(
        N,
        dtype=bool
    )

    stack = [0]
    visited[0] = True

    while stack:

        i = stack.pop()
        ni = normals[i]

        for j in knn_idx[i]:

            if not visited[j]:

                if np.dot(
                    ni,
                    normals[j]
                ) < 0:
                    normals[j] *= -1.0

                visited[j] = True
                stack.append(j)

    return normals


def build_cubic_grid_for_cluster(
    points,
    voxel_size=0.5,
    padding_frac=0.2
):
    """
    Build a cubic grid that contains the cluster
    and padded by padding_frac.
    """

    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)

    span = pts_max - pts_min

    max_span = np.max(span)

    L = max_span * (
        1.0 + padding_frac
    )

    center = 0.5 * (
        pts_min + pts_max
    )

    half_L = 0.5 * L

    x_min = center[0] - half_L
    x_max = center[0] + half_L

    y_min = center[1] - half_L
    y_max = center[1] + half_L

    z_min = center[2] - half_L
    z_max = center[2] + half_L

    xs = np.arange(
        x_min,
        x_max + 0.5 * voxel_size,
        voxel_size
    )

    ys = np.arange(
        y_min,
        y_max + 0.5 * voxel_size,
        voxel_size
    )

    zs = np.arange(
        z_min,
        z_max + 0.5 * voxel_size,
        voxel_size
    )

    return xs, ys, zs


def mls_signed_dist_slice(
    slice_pts,
    points,
    normals,
    tree,
    k=12,
    sigma_mode="kth",
    sigma_scale=1.0
):

    dists, idx = tree.query(
        slice_pts,
        k=min(k, len(points)),
        workers=-1
    )

    P = points[idx]
    N = normals[idx]

    V = (
        slice_pts[:, None, :] - P
    )

    di = np.sum(
        V * N,
        axis=2
    )

    if isinstance(
        sigma_mode,
        (int, float)
    ):

        sigma = float(sigma_mode)

        w = np.exp(
            -(dists ** 2) /
            (2.0 * sigma * sigma)
        )

        denom = (
            w.sum(axis=1) + 1e-12
        )

        sdf = (
            w * di
        ).sum(axis=1) / denom

    else:

        if sigma_mode == "kth":

            sigma = (
                dists[:, -1] *
                sigma_scale +
                1e-12
            )

        elif sigma_mode == "median":

            sigma = (
                np.median(
                    dists,
                    axis=1
                ) *
                sigma_scale +
                1e-12
            )

        else:
            raise ValueError(
                "bad sigma_mode"
            )

        s2 = (
            2.0 *
            (sigma ** 2)
        )[:, None]

        w = np.exp(
            -(dists ** 2) /
            s2
        )

        denom = (
            w.sum(axis=1) +
            1e-12
        )

        sdf = (
            w * di
        ).sum(axis=1) / denom

    return sdf


def sdf_mls(
    points,
    normals,
    xs,
    ys,
    zs,
    k=12,
    sigma_mode="kth",
    sigma_scale=1.0,
    band=None
):
    """
    Evaluate MLS SDF on a rectilinear grid, but only within `band`
    of the point cloud (narrow band). Grid points farther than
    `band` from every cell point are marked as outside without
    running the expensive MLS evaluation.
    """

    if band is None:
        band = SDF_BAND

    tree = cKDTree(points)

    nx_, ny_, nz_ = len(xs), len(ys), len(zs)

    phi = np.full((nz_, ny_, nx_), np.nan, dtype=np.float32)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    XY = np.column_stack([X.ravel(), Y.ravel()])

    # z-slices entirely above/below the cluster (plus band) are skipped
    z_lo = points[:, 2].min() - band
    z_hi = points[:, 2].max() + band

    outer_samples = []
    n_evaluated = 0

    for iz, z in enumerate(zs):

        if z_lo <= z <= z_hi:

            slice_pts = np.column_stack([XY, np.full(X.size, z)])

            # cheap check: is any cell point within `band`?
            d1, _ = tree.query(
                slice_pts,
                k=1,
                distance_upper_bound=band,
                workers=-1
            )

            near = np.isfinite(d1)

            if near.any():

                sdf = mls_signed_dist_slice(
                    slice_pts[near],
                    points,
                    normals,
                    tree,
                    k=k,
                    sigma_mode=sigma_mode,
                    sigma_scale=sigma_scale
                )

                tmp = np.full(X.size, np.nan, dtype=np.float32)
                tmp[near] = sdf
                phi[iz] = tmp.reshape(ny_, nx_)

                n_evaluated += int(near.sum())

                # values near the band edge tell us which sign
                # the MLS field uses for "outside"
                outer = d1[near] > 0.5 * band

                if outer.any():
                    outer_samples.append(sdf[outer])

        if iz % 20 == 0 or iz == nz_ - 1:
            print(
                f"      sdf slice {iz + 1}/{nz_}",
                flush=True
            )

    # Fill skipped points with the same sign MLS uses for "outside",
    # so the corner-based sign check in the main loop still works.
    outside_sign = 1.0

    if outer_samples and np.median(np.concatenate(outer_samples)) < 0:
        outside_sign = -1.0

    phi[np.isnan(phi)] = outside_sign * band

    print(
        f"      MLS evaluated at {n_evaluated} of "
        f"{nx_ * ny_ * nz_} grid points "
        f"({100.0 * n_evaluated / (nx_ * ny_ * nz_):.1f}%)",
        flush=True
    )

    return phi


# ============================================================
# Geometry metrics
# ============================================================

def voxel_volume_and_surface_area(
    vof,
    dx,
    dy,
    dz
):

    occ = (
        vof > 0
    ).astype(np.int32)

    n_vox = np.count_nonzero(occ)

    V = (
        n_vox *
        dx *
        dy *
        dz
    )

    Nx = (
        np.sum(occ[:, :, 0]) +
        np.sum(occ[:, :, -1]) +
        np.sum(
            np.abs(
                np.diff(
                    occ,
                    axis=2
                )
            )
        )
    )

    Ny = (
        np.sum(occ[:, 0, :]) +
        np.sum(occ[:, -1, :]) +
        np.sum(
            np.abs(
                np.diff(
                    occ,
                    axis=1
                )
            )
        )
    )

    Nz = (
        np.sum(occ[0, :, :]) +
        np.sum(occ[-1, :, :]) +
        np.sum(
            np.abs(
                np.diff(
                    occ,
                    axis=0
                )
            )
        )
    )

    n_faces = (
        Nx + Ny + Nz
    )

    A_voxel = (
        Nx * dy * dz +
        Ny * dx * dz +
        Nz * dx * dy
    )

    return (
        V,
        A_voxel,
        n_vox,
        Nx,
        Ny,
        Nz,
        n_faces
    )


def triangle_areas(
    verts,
    faces
):

    tri = verts[faces]

    v1 = (
        tri[:, 1] -
        tri[:, 0]
    )

    v2 = (
        tri[:, 2] -
        tri[:, 0]
    )

    cross = np.cross(
        v1,
        v2
    )

    areas = (
        0.5 *
        np.linalg.norm(
            cross,
            axis=1
        )
    )

    return np.sum(areas)


def mesh_surface_area_from_vof(
    vof,
    dx,
    dy,
    dz,
    level=0.5
):

    if np.count_nonzero(vof) == 0:
        return np.nan, 0, 0

    try:

        verts, faces, normals, values = \
            marching_cubes(
                vof.astype(np.float32),
                level=level,
                spacing=(
                    dz,
                    dy,
                    dx
                )
            )

        A_mesh = triangle_areas(
            verts,
            faces
        )

        return (
            A_mesh,
            len(verts),
            len(faces)
        )

    except Exception as e:

        print(
            f"  [warning] marching_cubes "
            f"on vof failed: {e}"
        )

        return np.nan, 0, 0


def mesh_surface_area_from_phi(
    phi,
    dx,
    dy,
    dz,
    level=0.0
):

    phi_min = np.nanmin(phi)
    phi_max = np.nanmax(phi)

    if not (
        phi_min <= level <= phi_max
    ):

        print(
            f"  [warning] level={level} "
            f"is outside phi range "
            f"[{phi_min:.6f}, "
            f"{phi_max:.6f}]"
        )

        return np.nan, 0, 0

    try:

        verts, faces, normals, values = \
            marching_cubes(
                phi.astype(np.float32),
                level=level,
                spacing=(
                    dz,
                    dy,
                    dx
                )
            )

        A_mesh = triangle_areas(
            verts,
            faces
        )

        return (
            A_mesh,
            len(verts),
            len(faces)
        )

    except Exception as e:

        print(
            f"  [warning] marching_cubes "
            f"on phi failed: {e}"
        )

        return np.nan, 0, 0


def sphericity_from_volume_area(
    V,
    A
):

    if (
        V <= 0 or
        A <= 0 or
        np.isnan(A)
    ):
        return np.nan

    return (
        (np.pi ** (1.0 / 3.0)) *
        ((6.0 * V) ** (2.0 / 3.0)) /
        A
    )


# ============================================================
# Figure saving
# ============================================================

def save_plotly_figure_as_html(
    fig,
    output_path,
    frame_stride=HTML_FRAME_STRIDE
):
    """
    Save Plotly figure as HTML.

    If animation frames already exist, only every
    frame_stride-th frame is retained.

    This is an additional safety mechanism.
    """

    html_path = str(
        Path(output_path).with_suffix(".html")
    )

    if getattr(fig, "frames", None):

        original_nframes = len(
            fig.frames
        )

        if frame_stride > 1:

            fig.frames = (
                fig.frames[::frame_stride]
            )

        saved_nframes = len(
            fig.frames
        )

        print(
            f"[html] animation frames: "
            f"{original_nframes} -> "
            f"{saved_nframes}"
        )

    fig.write_html(
        html_path,
        include_plotlyjs="cdn"
    )

    print(
        f"[save] interactive HTML written: "
        f"{html_path}"
    )


# ============================================================
# Plotting for reconstruction
# ============================================================

def plot_volume_world(
    vof,
    xs,
    ys,
    zs,
    points,
    title,
    ds=2,
    pc_max=20000,
    save_path=None
):

    vof_ds = vof[
        ::ds,
        ::ds,
        ::ds
    ]

    xs_ds = xs[::ds]
    ys_ds = ys[::ds]
    zs_ds = zs[::ds]

    Xw, Yw, Zw = np.meshgrid(
        xs_ds,
        ys_ds,
        zs_ds,
        indexing="xy"
    )

    Xw = np.transpose(
        Xw,
        (2, 0, 1)
    )

    Yw = np.transpose(
        Yw,
        (2, 0, 1)
    )

    Zw = np.transpose(
        Zw,
        (2, 0, 1)
    )

    pc_plot = _downsample_points(
        points,
        pc_max
    )

    fig = go.Figure()

    fig.add_trace(
        go.Volume(
            x=Xw.ravel(),
            y=Yw.ravel(),
            z=Zw.ravel(),
            value=vof_ds.ravel().astype(float),
            isomin=0.5,
            isomax=1.0,
            opacity=1.0,
            surface_count=1,
            opacityscale=[
                [0.0, 0.0],
                [0.49, 0.0],
                [0.5, 1.0],
                [1.0, 1.0]
            ],
            colorscale=[
                [0, "green"],
                [1, "green"]
            ],
            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            ),
            name="VOF=1 (solid)"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=pc_plot[:, 0],
            y=pc_plot[:, 1],
            z=pc_plot[:, 2],
            mode="markers",
            marker=dict(
                size=2.0,
                opacity=0.55,
                color="gray"
            ),
            name="Point cloud"
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),
        showlegend=True
    )

    # fig.show()   # disabled: no display on cluster

    if (
        SAVE_IMAGES and
        save_path is not None
    ):

        save_plotly_figure_as_html(
            fig,
            save_path
        )


def plot_all_reconstructed_clusters(
    cluster_results,
    ds=2,
    pc_max=20000,
    save_path=None
):

    fig = go.Figure()

    for res in cluster_results:

        rank = res["rank"]
        lab = res["label"]
        size = res["size"]

        vof = res["vof"]
        xs = res["xs"]
        ys = res["ys"]
        zs = res["zs"]

        pts = res["points"]

        vof_ds = vof[
            ::ds,
            ::ds,
            ::ds
        ]

        xs_ds = xs[::ds]
        ys_ds = ys[::ds]
        zs_ds = zs[::ds]

        Xw, Yw, Zw = np.meshgrid(
            xs_ds,
            ys_ds,
            zs_ds,
            indexing="xy"
        )

        Xw = np.transpose(
            Xw,
            (2, 0, 1)
        )

        Yw = np.transpose(
            Yw,
            (2, 0, 1)
        )

        Zw = np.transpose(
            Zw,
            (2, 0, 1)
        )

        pts_plot = _downsample_points(
            pts,
            pc_max
        )

        cluster_name = (
            f"Cluster {rank} "
            f"(label={lab}, N={size})"
        )

        fig.add_trace(
            go.Volume(
                x=Xw.ravel(),
                y=Yw.ravel(),
                z=Zw.ravel(),
                value=vof_ds.ravel().astype(float),
                isomin=0.5,
                isomax=1.0,
                opacity=0.12,
                surface_count=1,
                opacityscale=[
                    [0.0, 0.0],
                    [0.49, 0.0],
                    [0.5, 1.0],
                    [1.0, 1.0]
                ],
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                name=cluster_name,
                showscale=False
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=pts_plot[:, 0],
                y=pts_plot[:, 1],
                z=pts_plot[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    opacity=0.45
                ),
                name=(
                    f"{cluster_name} points"
                )
            )
        )

    fig.update_layout(
        title="All reconstructed retained clusters",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),
        showlegend=True
    )

    # fig.show()   # disabled: no display on cluster

    if (
        SAVE_IMAGES and
        save_path is not None
    ):

        save_plotly_figure_as_html(
            fig,
            save_path
        )


# ============================================================
# Skeleton pruning utilities
# ============================================================

def _node_positions(
    G,
    nodes
):

    ns = list(nodes)

    pts = np.array(
        [
            G.nodes[n]["pos"]
            for n in ns
        ],
        dtype=float
    )

    return pts, ns


def _reclassify_nodes(G):

    end_nodes = [
        n for n in G.nodes
        if G.degree[n] == 1
    ]

    branch_nodes = [
        n for n in G.nodes
        if G.degree[n] > 2
    ]

    return (
        end_nodes,
        branch_nodes
    )


def prune_end_nodes_adjacent_to_branch(
    G,
    comp_nodes,
    comp_end,
    comp_branch,
    verbose=False
):

    if (
        not comp_end or
        not comp_branch
    ):

        end_nodes, branch_nodes = \
            _reclassify_nodes(G)

        comp_nodes_alive = [
            n for n in comp_nodes
            if G.has_node(n)
        ]

        cluster_empty = (
            len(comp_nodes_alive) == 0
        )

        comp_branch_new = [
            n for n in comp_nodes_alive
            if G.degree[n] > 2
        ]

        comp_end_new = [
            n for n in comp_nodes_alive
            if G.degree[n] == 1
        ]

        return (
            comp_nodes_alive,
            comp_branch_new,
            comp_end_new,
            end_nodes,
            branch_nodes,
            cluster_empty
        )

    branch_set = set(
        comp_branch
    )

    end_adj_to_branch = [
        e for e in comp_end
        if any(
            nb in branch_set
            for nb in G.neighbors(e)
        )
    ]

    if end_adj_to_branch:

        if verbose:
            print(
                f"[prune] removing "
                f"{len(end_adj_to_branch)} "
                f"end node(s) adjacent "
                f"to branch in cluster"
            )

        G.remove_nodes_from(
            end_adj_to_branch
        )

    end_nodes, branch_nodes = \
        _reclassify_nodes(G)

    comp_nodes_alive = [
        n for n in comp_nodes
        if G.has_node(n)
    ]

    cluster_empty = (
        len(comp_nodes_alive) == 0
    )

    if cluster_empty:

        return (
            comp_nodes_alive,
            [],
            [],
            end_nodes,
            branch_nodes,
            True
        )

    comp_branch_new = [
        n for n in comp_nodes_alive
        if G.degree[n] > 2
    ]

    comp_end_new = [
        n for n in comp_nodes_alive
        if G.degree[n] == 1
    ]

    return (
        comp_nodes_alive,
        comp_branch_new,
        comp_end_new,
        end_nodes,
        branch_nodes,
        False
    )


def prune_sparse_branch_clusters(
    G,
    distance_threshold=12.0,
    verbose=True
):

    end_nodes, branch_nodes = \
        _reclassify_nodes(G)

    candidates = sorted(
        set(end_nodes) |
        set(branch_nodes)
    )

    if verbose:
        print(
            f"[prune] candidates: "
            f"{len(candidates)} "
            f"(branches={len(branch_nodes)}, "
            f"ends={len(end_nodes)})"
        )

    if not candidates:

        loops = list(
            nx.cycle_basis(G)
        )

        return (
            G,
            end_nodes,
            branch_nodes,
            loops
        )

    cand_pts, cand_nodes = \
        _node_positions(
            G,
            candidates
        )

    kdt = cKDTree(cand_pts)

    pairs = list(
        kdt.query_pairs(
            distance_threshold
        )
    )

    H = nx.Graph()

    H.add_nodes_from(
        range(len(cand_nodes))
    )

    H.add_edges_from(
        pairs
    )

    index_to_node = list(
        cand_nodes
    )

    node_to_index = {
        n: i
        for i, n in enumerate(
            index_to_node
        )
    }

    normals_added = 0

    for c in cand_nodes:

        for nb in G.neighbors(c):

            if (
                nb not in node_to_index and
                G.degree[nb] == 2
            ):

                node_to_index[nb] = \
                    len(index_to_node)

                index_to_node.append(nb)

                H.add_node(
                    node_to_index[nb]
                )

                normals_added += 1

            if G.degree[nb] == 2:

                H.add_edge(
                    node_to_index[c],
                    node_to_index[nb]
                )

    if verbose:
        print(
            f"[prune] added "
            f"{normals_added} normal "
            f"neighbor nodes to candidate graph"
        )

    clusters = list(
        nx.connected_components(H)
    )

    if verbose:
        print(
            f"[prune] {len(clusters)} "
            f"clusters at radius "
            f"{distance_threshold}"
        )

    for comp in clusters:

        idxs = sorted(
            list(comp)
        )

        comp_nodes = [
            index_to_node[i]
            for i in idxs
        ]

        comp_nodes_alive = [
            n for n in comp_nodes
            if G.has_node(n)
        ]

        if not comp_nodes_alive:
            continue

        comp_branch = [
            n for n in comp_nodes_alive
            if G.degree[n] > 2
        ]

        comp_end = [
            n for n in comp_nodes_alive
            if G.degree[n] == 1
        ]

        (
            comp_nodes_alive,
            comp_branch,
            comp_end,
            end_nodes,
            branch_nodes,
            cluster_empty
        ) = prune_end_nodes_adjacent_to_branch(
            G,
            comp_nodes,
            comp_end,
            comp_branch,
            verbose
        )

        if cluster_empty:
            continue

        while True:

            if len(comp_branch) > 1:

                b_pts, b_nodes = \
                    _node_positions(
                        G,
                        comp_branch
                    )

                b_centroid = \
                    b_pts.mean(axis=0)

                dists = np.linalg.norm(
                    b_pts - b_centroid,
                    axis=1
                )

                rm_node = b_nodes[
                    int(
                        np.argmin(dists)
                    )
                ]

                if verbose:
                    print(
                        f"[prune] cluster "
                        f"size={len(comp_nodes_alive)} "
                        f"-> remove branch "
                        f"{rm_node}"
                    )

                if G.has_node(rm_node):
                    G.remove_node(
                        rm_node
                    )

            end_nodes, branch_nodes = \
                _reclassify_nodes(G)

            comp_nodes_alive = [
                n for n in comp_nodes
                if G.has_node(n)
            ]

            if not comp_nodes_alive:
                break

            comp_branch = [
                n for n in comp_nodes_alive
                if G.degree[n] > 2
            ]

            comp_end = [
                n for n in comp_nodes_alive
                if G.degree[n] == 1
            ]

            if len(comp_end) > 1:

                branch_set = set(
                    comp_branch
                )

                free_ends = [
                    e for e in comp_end
                    if len(
                        set(G.neighbors(e))
                        & branch_set
                    ) == 0
                ]

                if not free_ends:

                    if verbose:
                        print(
                            "[prune] no free ends "
                            "available; exiting "
                            "this cluster without "
                            "end-connection"
                        )

                    isolates = list(
                        nx.isolates(G)
                    )

                    if isolates:
                        G.remove_nodes_from(
                            isolates
                        )

                    break

                target_end = free_ends[0]

                other_ends = [
                    e for e in comp_end
                    if e != target_end
                ]

                mate_end = None

                if other_ends:

                    t_pt = np.array(
                        G.nodes[
                            target_end
                        ]["pos"],
                        dtype=float
                    )

                    other_free = [
                        e for e in other_ends
                        if len(
                            set(
                                G.neighbors(e)
                            ) & branch_set
                        ) == 0
                    ]

                    other_branch_adj = [
                        e for e in other_ends
                        if len(
                            set(
                                G.neighbors(e)
                            ) & branch_set
                        ) > 0
                    ]

                    if other_free:

                        o_pts, o_nodes = \
                            _node_positions(
                                G,
                                other_free
                            )

                        nn = int(
                            np.argmin(
                                np.linalg.norm(
                                    o_pts - t_pt,
                                    axis=1
                                )
                            )
                        )

                        mate_end = o_nodes[nn]

                    elif other_branch_adj:

                        o_pts, o_nodes = \
                            _node_positions(
                                G,
                                other_branch_adj
                            )

                        nn = int(
                            np.argmin(
                                np.linalg.norm(
                                    o_pts - t_pt,
                                    axis=1
                                )
                            )
                        )

                        mate_end = o_nodes[nn]

                    if (
                        mate_end is not None and
                        not G.has_edge(
                            target_end,
                            mate_end
                        )
                    ):

                        w = float(
                            np.linalg.norm(
                                t_pt -
                                np.array(
                                    G.nodes[
                                        mate_end
                                    ]["pos"],
                                    dtype=float
                                )
                            )
                        )

                        G.add_edge(
                            target_end,
                            mate_end,
                            weight=w
                        )

                    to_remove = [
                        e for e in comp_end
                        if e not in (
                            target_end,
                            mate_end
                        )
                    ]

                    if to_remove:

                        if verbose:
                            print(
                                f"[prune] removing "
                                f"{len(to_remove)} "
                                f"spare end nodes "
                                f"in cluster"
                            )

                        G.remove_nodes_from(
                            to_remove
                        )

            isolates = list(
                nx.isolates(G)
            )

            if isolates:
                G.remove_nodes_from(
                    isolates
                )

            end_nodes, branch_nodes = \
                _reclassify_nodes(G)

            comp_nodes_alive = [
                n for n in comp_nodes
                if G.has_node(n)
            ]

            if not comp_nodes_alive:
                break

            comp_branch = [
                n for n in comp_nodes_alive
                if G.degree[n] > 2
            ]

            comp_end = [
                n for n in comp_nodes_alive
                if G.degree[n] == 1
            ]

            (
                comp_nodes_alive,
                comp_branch,
                comp_end,
                end_nodes,
                branch_nodes,
                cluster_empty
            ) = prune_end_nodes_adjacent_to_branch(
                G,
                comp_nodes,
                comp_end,
                comp_branch,
                verbose
            )

            if cluster_empty:
                break

            if len(comp_branch) > 1:

                if verbose:
                    print(
                        f"[prune] cluster loop: "
                        f"{len(comp_branch)} "
                        f"branches remain -> "
                        f"repeat B-E"
                    )

                continue

            break

    G.remove_nodes_from(
        list(nx.isolates(G))
    )

    end_nodes, branch_nodes = \
        _reclassify_nodes(G)

    if end_nodes and branch_nodes:

        branch_set = set(
            branch_nodes
        )

        ends_to_remove = [
            e for e in end_nodes
            if any(
                nb in branch_set
                for nb in G.neighbors(e)
            )
        ]

        if ends_to_remove:

            if verbose:
                print(
                    f"[prune] global: removing "
                    f"{len(ends_to_remove)} "
                    f"end node(s) adjacent "
                    f"to branches"
                )

            G.remove_nodes_from(
                ends_to_remove
            )

            G.remove_nodes_from(
                list(nx.isolates(G))
            )

    end_nodes, branch_nodes = \
        _reclassify_nodes(G)

    loops = list(
        nx.cycle_basis(G)
    )

    if verbose:
        print(
            f"[prune] final |V|="
            f"{G.number_of_nodes()}, "
            f"|E|={G.number_of_edges()}, "
            f"ends={len(end_nodes)}, "
            f"branches={len(branch_nodes)}, "
            f"loops={len(loops)}"
        )

    return (
        G,
        end_nodes,
        branch_nodes,
        loops
    )


# ============================================================
# Skeleton plotting
# ============================================================

def make_graph_3d_figure(
    G,
    end_nodes=None,
    branch_nodes=None,
    selected_nodes=None,
    title="Final Merged Skeleton Graph"
):

    fig = go.Figure()

    node_ids = list(
        G.nodes()
    )

    selected_nodes = set(
        [] if selected_nodes is None
        else selected_nodes
    )

    if len(node_ids) == 0:

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"
            ),
            margin=dict(
                l=0,
                r=0,
                t=40,
                b=0
            ),
            legend=dict(
                x=0.01,
                y=0.99
            )
        )

        return fig

    positions = np.array(
        [
            G.nodes[i]["pos"]
            for i in node_ids
        ],
        dtype=float
    )

    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]

    for u, v in G.edges:

        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]

        highlight = (
            u in selected_nodes and
            v in selected_nodes
        )

        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(
                    color=(
                        "orange"
                        if highlight
                        else "blue"
                    ),
                    width=(
                        5
                        if highlight
                        else 2
                    )
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    marker_colors = []
    marker_sizes = []
    node_text = []

    end_set = set(
        [] if end_nodes is None
        else end_nodes
    )

    branch_set = set(
        [] if branch_nodes is None
        else branch_nodes
    )

    for n in node_ids:

        if n in selected_nodes:

            marker_colors.append(
                "gold"
            )

            marker_sizes.append(8)

        elif n in end_set:

            marker_colors.append(
                "red"
            )

            marker_sizes.append(6)

        elif n in branch_set:

            marker_colors.append(
                "green"
            )

            marker_sizes.append(6)

        else:

            marker_colors.append(
                "lightgray"
            )

            marker_sizes.append(4)

        ptype = G.nodes[n].get(
            "ptype",
            "unknown"
        )

        node_text.append(
            f"Index: {n}<br>"
            f"Type: {ptype}"
        )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                opacity=0.95
            ),
            text=node_text,
            customdata=np.array(
                node_ids,
                dtype=object
            ),
            hovertemplate=(
                "%{text}<extra></extra>"
            ),
            name="All Nodes"
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        ),
        legend=dict(
            x=0.01,
            y=0.99
        )
    )

    return fig


def plot_graph_3d(
    G,
    end_nodes=None,
    branch_nodes=None,
    save_path=None,
    selected_nodes=None
):

    if G.number_of_nodes() == 0:
        print(
            "[plot_graph_3d] Graph is empty."
        )

    fig = make_graph_3d_figure(
        G,
        end_nodes=end_nodes,
        branch_nodes=branch_nodes,
        selected_nodes=selected_nodes
    )

    # fig.show()   # disabled: no display on cluster

    if save_path is not None:

        save_plotly_figure_as_html(
            fig,
            save_path
        )


# ============================================================
# Skeleton graph construction
# ============================================================

def build_vessel_graph_from_coords(
    coords,
    radius=12.0
):

    coords = coords.astype(
        np.float64
    )

    tree = cKDTree(coords)

    pairs = tree.query_pairs(
        radius
    )

    print(
        f"Found {len(pairs)} pairs "
        f"within radius {radius}"
    )

    print(
        "Example pairs (up to 5):",
        list(pairs)[:5]
    )

    G = nx.Graph()

    for i, pt in enumerate(coords):

        G.add_node(
            i,
            pos=tuple(pt)
        )

    for i, j in pairs:

        dist = np.linalg.norm(
            coords[i] -
            coords[j]
        )

        G.add_edge(
            i,
            j,
            weight=dist
        )

    triangles_checked = set()

    for node in list(G.nodes):

        neighbors = list(
            G.neighbors(node)
        )

        for u, v in combinations(
            neighbors,
            2
        ):

            if G.has_edge(u, v):

                triangle = tuple(
                    sorted(
                        [node, u, v]
                    )
                )

                if triangle in \
                        triangles_checked:
                    continue

                triangles_checked.add(
                    triangle
                )

                edge_weights = []

                if G.has_edge(
                    node,
                    u
                ):

                    edge_weights.append(
                        (
                            (node, u),
                            G[node][u]["weight"]
                        )
                    )

                if G.has_edge(
                    node,
                    v
                ):

                    edge_weights.append(
                        (
                            (node, v),
                            G[node][v]["weight"]
                        )
                    )

                if G.has_edge(
                    u,
                    v
                ):

                    edge_weights.append(
                        (
                            (u, v),
                            G[u][v]["weight"]
                        )
                    )

                if len(edge_weights) == 3:

                    longest_edge = max(
                        edge_weights,
                        key=lambda x: x[1]
                    )[0]

                    G.remove_edge(
                        *longest_edge
                    )

    all_cycles = list(
        nx.cycle_basis(G)
    )

    quads_checked = set()

    for cycle in all_cycles:

        if len(cycle) == 4:

            cycle_nodes = tuple(
                sorted(cycle)
            )

            if cycle_nodes in \
                    quads_checked:
                continue

            quads_checked.add(
                cycle_nodes
            )

            edge_weights = []

            for u, v in combinations(
                cycle,
                2
            ):

                if G.has_edge(u, v):

                    edge_weights.append(
                        (
                            (u, v),
                            G[u][v]["weight"]
                        )
                    )

            if len(edge_weights) >= 4:

                longest_edge = max(
                    edge_weights,
                    key=lambda x: x[1]
                )[0]

                G.remove_edge(
                    *longest_edge
                )

    end_nodes = [
        n for n in G.nodes
        if G.degree[n] == 1
    ]

    branch_nodes = [
        n for n in G.nodes
        if G.degree[n] > 2
    ]

    loops = list(
        nx.cycle_basis(G)
    )

    return (
        G,
        end_nodes,
        branch_nodes,
        loops
    )


def simple_skeleton(volume):

    distance = ndi.distance_transform_edt(
        volume
    )

    local_max = (
        ndi.maximum_filter(
            distance,
            size=3
        ) == distance
    )

    skeleton = (
        local_max *
        volume
    )

    return skeleton


def average_nearby_points(
    coords,
    distance_threshold=3.0
):

    clustering = DBSCAN(
        eps=distance_threshold,
        min_samples=1
    ).fit(coords)

    labels = clustering.labels_

    unique_labels = np.unique(
        labels
    )

    averaged_coords = []

    for label in unique_labels:

        cluster_points = \
            coords[labels == label]

        averaged_coords.append(
            np.mean(
                cluster_points,
                axis=0
            )
        )

    return np.array(
        averaged_coords
    )


def build_graph_from_vof(
    vof,
    xs,
    ys,
    zs,
    dbscan_eps=2.0,
    connect_radius=12.0,
    prune_radius=12.0,
    verbose=True,
    save_path=None
):

    volume = np.transpose(
        vof,
        (2, 1, 0)
    ).copy()

    skeleton = simple_skeleton(
        volume
    )

    idx = np.argwhere(
        skeleton > 0
    )

    if idx.shape[0] == 0:

        if verbose:
            print(
                "[build_graph_from_vof] "
                "WARNING: skeleton empty "
                "for this cluster."
            )

        G_empty = nx.Graph()

        return (
            G_empty,
            [],
            [],
            []
        )

    xw = xs[idx[:, 0]]
    yw = ys[idx[:, 1]]
    zw = zs[idx[:, 2]]

    coords_world = np.column_stack(
        (
            xw,
            yw,
            zw
        )
    )

    coords_world = \
        average_nearby_points(
            coords_world,
            distance_threshold=dbscan_eps
        )

    if verbose:
        print(
            "[build_graph_from_vof] "
            "skeleton points "
            "(after averaging): "
            f"{coords_world.shape[0]}"
        )

    (
        G,
        end_nodes,
        branch_nodes,
        loops
    ) = build_vessel_graph_from_coords(
        coords_world,
        radius=connect_radius
    )

    (
        G,
        end_nodes,
        branch_nodes,
        loops
    ) = prune_sparse_branch_clusters(
        G,
        distance_threshold=prune_radius,
        verbose=verbose
    )

    if coords_world.shape[0] > 1:

        dists = pdist(
            coords_world
        )

        print(
            "[cluster] min/median/max "
            f"node spacing: "
            f"{np.min(dists):.2f}, "
            f"{np.median(dists):.2f}, "
            f"{np.max(dists):.2f}"
        )

    if save_path is not None:

        plot_graph_3d(
            G,
            end_nodes=end_nodes,
            branch_nodes=branch_nodes,
            save_path=save_path
        )

    return (
        G,
        end_nodes,
        branch_nodes,
        loops
    )


def merge_graphs(
    graph_list,
    verbose=True
):

    G_all = nx.Graph()

    next_id = 0

    for gi, G in enumerate(
        graph_list
    ):

        if G.number_of_nodes() == 0:

            if verbose:
                print(
                    f"[merge_graphs] "
                    f"cluster {gi}: empty "
                    f"graph, skipping."
                )

            continue

        current_offset = next_id

        mapping = {
            old: new
            for old, new in zip(
                G.nodes(),
                range(
                    current_offset,
                    current_offset +
                    G.number_of_nodes()
                )
            )
        }

        G_shifted = nx.relabel_nodes(
            G,
            mapping,
            copy=True
        )

        G_all.add_nodes_from(
            G_shifted.nodes(
                data=True
            )
        )

        G_all.add_edges_from(
            G_shifted.edges(
                data=True
            )
        )

        if verbose:
            print(
                f"[merge_graphs] "
                f"added cluster {gi}: "
                f"{G.number_of_nodes()} "
                f"nodes, "
                f"{G.number_of_edges()} "
                f"edges "
                f"(offset={current_offset})"
            )

        next_id += \
            G.number_of_nodes()

    end_nodes_all = [
        n for n in G_all.nodes
        if G_all.degree[n] == 1
    ]

    branch_nodes_all = [
        n for n in G_all.nodes
        if G_all.degree[n] > 2
    ]

    loops_all = list(
        nx.cycle_basis(G_all)
    )

    if verbose:
        print(
            f"[merge_graphs] GLOBAL: "
            f"|V|={G_all.number_of_nodes()}, "
            f"|E|={G_all.number_of_edges()}, "
            f"ends={len(end_nodes_all)}, "
            f"branches={len(branch_nodes_all)}, "
            f"loops={len(loops_all)}"
        )

    return (
        G_all,
        end_nodes_all,
        branch_nodes_all,
        loops_all
    )


# ============================================================
# Manual graph editing
# ============================================================

def _as_node_list(node_ids):

    try:

        if isinstance(
            node_ids,
            (str, bytes)
        ):
            return [node_ids]

        return list(node_ids)

    except TypeError:

        return [node_ids]


def _update_point_types(G):

    for n in G.nodes:

        deg = G.degree[n]

        if deg == 0:
            ptype = "isolated"

        elif deg == 1:
            ptype = "endpoint"

        elif deg >= 3:
            ptype = "branch"

        else:
            ptype = "internal"

        G.nodes[n]["ptype"] = \
            ptype


def delete_nodes_by_id(
    G,
    node_ids
):

    nodes = _as_node_list(
        node_ids
    )

    missing = [
        n for n in nodes
        if n not in G
    ]

    if missing:

        raise KeyError(
            f"Nodes not found in graph: "
            f"{missing}"
        )

    G.remove_nodes_from(
        nodes
    )

    _update_point_types(G)

    return G


def add_edges_by_node_ids(
    G,
    edge_pairs
):

    for u, v in edge_pairs:

        if u not in G or v not in G:

            raise KeyError(
                f"One or both nodes "
                f"({u}, {v}) "
                f"not found in graph."
            )

        if u == v:
            continue

        if (
            "pos" in G.nodes[u] and
            "pos" in G.nodes[v]
        ):

            pu = np.array(
                G.nodes[u]["pos"],
                dtype=float
            )

            pv = np.array(
                G.nodes[v]["pos"],
                dtype=float
            )

            w = float(
                np.linalg.norm(
                    pu - pv
                )
            )

            G.add_edge(
                u,
                v,
                weight=w
            )

        else:

            G.add_edge(
                u,
                v
            )

    _update_point_types(G)

    return G


def remove_edges_by_node_ids(
    G,
    edge_pairs
):

    for u, v in edge_pairs:

        if not G.has_edge(u, v):

            raise ValueError(
                f"Edge ({u}, {v}) "
                f"does not exist in graph."
            )

        G.remove_edge(
            u,
            v
        )

    _update_point_types(G)

    return G


def delete_nodes_and_connected(
    G,
    node_ids
):

    seeds = _as_node_list(
        node_ids
    )

    missing = [
        n for n in seeds
        if n not in G
    ]

    if missing:

        raise KeyError(
            f"Nodes not found in graph: "
            f"{missing}"
        )

    to_remove = set()

    for s in seeds:

        to_remove.update(
            nx.node_connected_component(
                G,
                s
            )
        )

    G.remove_nodes_from(
        to_remove
    )

    _update_point_types(G)

    return G


def analyze_graph_structure(G):

    end_nodes = []
    branch_nodes = []
    internal_nodes = []
    isolated_nodes = []

    for n in G.nodes:

        d = G.degree[n]

        if d == 0:
            isolated_nodes.append(n)

        elif d == 1:
            end_nodes.append(n)

        elif d == 2:
            internal_nodes.append(n)

        elif d >= 3:
            branch_nodes.append(n)

    loops = list(
        nx.cycle_basis(G)
    )

    return (
        end_nodes,
        branch_nodes,
        internal_nodes,
        isolated_nodes,
        loops
    )


def parse_int_list(text):

    text = text.strip()

    if not text:
        return []

    return [
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def parse_edge_list(text):

    text = text.strip()

    if not text:
        return []

    edges = []

    chunks = [
        c.strip()
        for c in text.split(",")
        if c.strip()
    ]

    for chunk in chunks:

        if "-" not in chunk:

            raise ValueError(
                f"Bad edge format: "
                f"'{chunk}'. "
                f"Use u-v, e.g. 12-18"
            )

        a, b = chunk.split(
            "-",
            1
        )

        edges.append(
            (
                int(a.strip()),
                int(b.strip())
            )
        )

    return edges


# ------------------------------------------------------------
# Colab interactive editor: disabled for batch (cluster) runs.
# It needs a live notebook with ipywidgets to click nodes.
# ------------------------------------------------------------

# def _try_enable_colab_widgets():

#     if not ENABLE_COLAB_WIDGETS:
#         return

#     try:

#         from google.colab import output \
#             as colab_output

#         colab_output.enable_custom_widget_manager()

#         print(
#             "[colab] custom widget "
#             "manager enabled."
#         )

#     except Exception as e:

#         print(
#             "[colab] widget manager "
#             "could not be enabled "
#             f"automatically: {e}"
#         )


# # ============================================================
# # Colab skeleton editor
# # ============================================================

# class ColabSkeletonEditor:

#     def __init__(
#         self,
#         G,
#         output_dir,
#         base_name
#     ):

#         self.G = G.copy()

#         self.output_dir = Path(
#             output_dir
#         )

#         self.base_name = \
#             base_name

#         self.review_round = 0

#         self.selected_nodes = []

#         self.accepted = False

#         self.final_graph = None

#         _update_point_types(
#             self.G
#         )

#         self.status_html = \
#             widgets.HTML()

#         self.help_html = widgets.HTML(
#             value=(
#                 "<b>How to use:</b> "
#                 "Click nodes to select/"
#                 "deselect them. "
#                 "Use the buttons below "
#                 "for delete/add/remove "
#                 "operations. "
#                 "For edge operations, "
#                 "select exactly 2 nodes."
#             )
#         )

#         self.btn_delete_nodes = \
#             widgets.Button(
#                 description="Delete selected nodes",
#                 button_style="danger"
#             )

#         self.btn_delete_components = \
#             widgets.Button(
#                 description="Delete selected component(s)",
#                 button_style="warning"
#             )

#         self.btn_add_edge = \
#             widgets.Button(
#                 description="Add edge (2 selected)",
#                 button_style="success"
#             )

#         self.btn_remove_edge = \
#             widgets.Button(
#                 description="Remove edge (2 selected)"
#             )

#         self.btn_clear = \
#             widgets.Button(
#                 description="Clear selection"
#             )

#         self.btn_redraw = \
#             widgets.Button(
#                 description="Redraw / save review HTML"
#             )

#         self.btn_accept = \
#             widgets.Button(
#                 description="Accept graph",
#                 button_style="success"
#             )

#         self.btn_delete_nodes.on_click(
#             self._on_delete_nodes
#         )

#         self.btn_delete_components.on_click(
#             self._on_delete_components
#         )

#         self.btn_add_edge.on_click(
#             self._on_add_edge
#         )

#         self.btn_remove_edge.on_click(
#             self._on_remove_edge
#         )

#         self.btn_clear.on_click(
#             self._on_clear
#         )

#         self.btn_redraw.on_click(
#             self._on_redraw
#         )

#         self.btn_accept.on_click(
#             self._on_accept
#         )

#         self.controls = widgets.HBox(
#             [
#                 self.btn_delete_nodes,
#                 self.btn_delete_components,
#                 self.btn_add_edge,
#                 self.btn_remove_edge,
#                 self.btn_clear,
#                 self.btn_redraw,
#                 self.btn_accept
#             ]
#         )

#         self.fig_widget = None
#         self.container = None

#         self._rebuild_widget()

#     def _stats_html(self):

#         (
#             end_nodes,
#             branch_nodes,
#             internal_nodes,
#             isolated_nodes,
#             loops
#         ) = analyze_graph_structure(
#             self.G
#         )

#         selected_text = (
#             ", ".join(
#                 str(n)
#                 for n in self.selected_nodes
#             )
#             if self.selected_nodes
#             else "None"
#         )

#         return (
#             f"<pre>"
#             f"[review] round "
#             f"{self.review_round}\n"
#             f"nodes      : "
#             f"{self.G.number_of_nodes()}\n"
#             f"edges      : "
#             f"{self.G.number_of_edges()}\n"
#             f"endpoints  : "
#             f"{len(end_nodes)}\n"
#             f"branches   : "
#             f"{len(branch_nodes)}\n"
#             f"internals  : "
#             f"{len(internal_nodes)}\n"
#             f"isolated   : "
#             f"{len(isolated_nodes)}\n"
#             f"loops      : "
#             f"{len(loops)}\n"
#             f"selected   : "
#             f"{selected_text}"
#             f"</pre>"
#         )

#     def _review_html_path(self):

#         return (
#             self.output_dir /
#             f"{self.base_name}"
#             f"_final_merged_skeleton_review_"
#             f"{self.review_round:02d}.html"
#         )

#     def _save_current_review_html(self):

#         (
#             end_nodes,
#             branch_nodes,
#             _,
#             _,
#             _
#         ) = analyze_graph_structure(
#             self.G
#         )

#         html_path = \
#             self._review_html_path()

#         plot_graph_3d(
#             self.G,
#             end_nodes=end_nodes,
#             branch_nodes=branch_nodes,
#             save_path=html_path,
#             selected_nodes=self.selected_nodes
#         )

#         return html_path

#     def _rebuild_widget(self):

#         self.review_round += 1

#         self.selected_nodes = [
#             n for n in self.selected_nodes
#             if n in self.G.nodes
#         ]

#         (
#             end_nodes,
#             branch_nodes,
#             _,
#             _,
#             _
#         ) = analyze_graph_structure(
#             self.G
#         )

#         fig = make_graph_3d_figure(
#             self.G,
#             end_nodes=end_nodes,
#             branch_nodes=branch_nodes,
#             selected_nodes=self.selected_nodes,
#             title=(
#                 "Final Merged Skeleton Graph "
#                 "(click nodes to edit)"
#             )
#         )

#         self.fig_widget = \
#             go.FigureWidget(fig)

#         if len(self.fig_widget.data) > 0:

#             node_trace = \
#                 self.fig_widget.data[-1]

#             node_trace.on_click(
#                 self._on_node_click
#             )

#         self.status_html.value = \
#             self._stats_html()

#         if self.container is None:

#             self.container = widgets.VBox(
#                 [
#                     self.help_html,
#                     self.status_html,
#                     self.controls,
#                     self.fig_widget
#                 ]
#             )

#         else:

#             self.container.children = (
#                 self.help_html,
#                 self.status_html,
#                 self.controls,
#                 self.fig_widget
#             )

#     def _refresh(
#         self,
#         save_review_html=False
#     ):

#         self._rebuild_widget()

#         if save_review_html:

#             html_path = \
#                 self._save_current_review_html()

#             self.status_html.value += (
#                 f"<pre>[save] review HTML "
#                 f"saved to: "
#                 f"{html_path}</pre>"
#             )

#     def _set_message(
#         self,
#         msg
#     ):

#         self.status_html.value = (
#             self._stats_html() +
#             f"<pre>{msg}</pre>"
#         )

#     def _on_node_click(
#         self,
#         trace,
#         points,
#         state
#     ):

#         if not points.point_inds:
#             return

#         idx = points.point_inds[0]

#         node_id = \
#             trace.customdata[idx]

#         if node_id in \
#                 self.selected_nodes:

#             self.selected_nodes.remove(
#                 node_id
#             )

#         else:

#             self.selected_nodes.append(
#                 node_id
#             )

#         self._refresh(
#             save_review_html=False
#         )

#     def _on_delete_nodes(self, _):

#         if not self.selected_nodes:

#             self._set_message(
#                 "[edit] no nodes selected."
#             )

#             return

#         try:

#             delete_nodes_by_id(
#                 self.G,
#                 self.selected_nodes
#             )

#             removed = list(
#                 self.selected_nodes
#             )

#             self.selected_nodes = []

#             self._refresh(
#                 save_review_html=True
#             )

#             self._set_message(
#                 f"[edit] deleted nodes: "
#                 f"{removed}"
#             )

#         except Exception as e:

#             self._set_message(
#                 f"[error] delete selected "
#                 f"nodes failed: {e}"
#             )

#     def _on_delete_components(self, _):

#         if not self.selected_nodes:

#             self._set_message(
#                 "[edit] no nodes selected."
#             )

#             return

#         try:

#             seeds = list(
#                 self.selected_nodes
#             )

#             delete_nodes_and_connected(
#                 self.G,
#                 seeds
#             )

#             self.selected_nodes = []

#             self._refresh(
#                 save_review_html=True
#             )

#             self._set_message(
#                 "[edit] deleted connected "
#                 "component(s) from seeds: "
#                 f"{seeds}"
#             )

#         except Exception as e:

#             self._set_message(
#                 f"[error] delete component(s) "
#                 f"failed: {e}"
#             )

#     def _require_two_selected_nodes(self):

#         if len(
#             self.selected_nodes
#         ) != 2:

#             raise ValueError(
#                 "Please select exactly "
#                 "2 nodes."
#             )

#     def _on_add_edge(self, _):

#         try:

#             self._require_two_selected_nodes()

#             u, v = \
#                 self.selected_nodes

#             add_edges_by_node_ids(
#                 self.G,
#                 [(u, v)]
#             )

#             self._refresh(
#                 save_review_html=True
#             )

#             self._set_message(
#                 f"[edit] added edge: "
#                 f"({u}, {v})"
#             )

#         except Exception as e:

#             self._set_message(
#                 f"[error] add edge failed: "
#                 f"{e}"
#             )

#     def _on_remove_edge(self, _):

#         try:

#             self._require_two_selected_nodes()

#             u, v = \
#                 self.selected_nodes

#             remove_edges_by_node_ids(
#                 self.G,
#                 [(u, v)]
#             )

#             self._refresh(
#                 save_review_html=True
#             )

#             self._set_message(
#                 f"[edit] removed edge: "
#                 f"({u}, {v})"
#             )

#         except Exception as e:

#             self._set_message(
#                 f"[error] remove edge failed: "
#                 f"{e}"
#             )

#     def _on_clear(self, _):

#         self.selected_nodes = []

#         self._refresh(
#             save_review_html=False
#         )

#         self._set_message(
#             "[edit] selection cleared."
#         )

#     def _on_redraw(self, _):

#         self._refresh(
#             save_review_html=True
#         )

#         self._set_message(
#             "[edit] graph redrawn and "
#             "review HTML saved."
#         )

#     def _on_accept(self, _):

#         self.accepted = True

#         self.final_graph = \
#             self.G.copy()

#         (
#             end_nodes,
#             branch_nodes,
#             _,
#             _,
#             _
#         ) = analyze_graph_structure(
#             self.final_graph
#         )

#         final_html = (
#             self.output_dir /
#             f"{self.base_name}"
#             f"_final_merged_skeleton_final.html"
#         )

#         plot_graph_3d(
#             self.final_graph,
#             end_nodes=end_nodes,
#             branch_nodes=branch_nodes,
#             save_path=final_html,
#             selected_nodes=[]
#         )

#         self._set_message(
#             "[accept] final accepted "
#             "skeleton HTML written: "
#             f"{final_html}"
#         )

#     def display(self):

#         display(
#             self.container
#         )

#     def get_graph(self):

#         if (
#             self.accepted and
#             self.final_graph is not None
#         ):

#             return self.final_graph.copy()

#         raise RuntimeError(
#             "Graph has not been accepted yet. "
#             "Click 'Accept graph' first."
#         )


# def review_and_edit_skeleton_colab(
#     G,
#     output_dir,
#     base_name
# ):

#     _try_enable_colab_widgets()

#     editor = ColabSkeletonEditor(
#         G,
#         output_dir,
#         base_name
#     )

#     editor.display()

#     print(
#         "\n[info] Interactive editor launched."
#     )

#     print(
#         "[info] Click nodes to select them, "
#         "use the buttons to edit, then "
#         "click 'Accept graph'."
#     )

#     print(
#         "[info] After you accept the graph, "
#         "run the FOLLOW-UP CELL:"
#     )

#     print(
#         "       G_all = editor.get_graph()"
#     )

#     return editor


# ============================================================
# Global VOF merge
# ============================================================

def merge_cluster_results_to_global_vof(
    cluster_results,
    voxel_size=None,
    verbose=True
):

    if len(cluster_results) == 0:

        raise ValueError(
            "cluster_results is empty."
        )

    if voxel_size is None:

        dx0 = (
            cluster_results[0]["xs"][1] -
            cluster_results[0]["xs"][0]
        )

        dy0 = (
            cluster_results[0]["ys"][1] -
            cluster_results[0]["ys"][0]
        )

        dz0 = (
            cluster_results[0]["zs"][1] -
            cluster_results[0]["zs"][0]
        )

        if not (
            np.isclose(dx0, dy0) and
            np.isclose(dx0, dz0)
        ):

            raise ValueError(
                f"First cluster spacing "
                f"not isotropic: "
                f"dx={dx0}, "
                f"dy={dy0}, "
                f"dz={dz0}"
            )

        voxel_size = float(
            dx0
        )

    for i, res in enumerate(
        cluster_results
    ):

        xs = res["xs"]
        ys = res["ys"]
        zs = res["zs"]

        dx = (
            xs[1] - xs[0]
            if len(xs) > 1
            else voxel_size
        )

        dy = (
            ys[1] - ys[0]
            if len(ys) > 1
            else voxel_size
        )

        dz = (
            zs[1] - zs[0]
            if len(zs) > 1
            else voxel_size
        )

        if not (
            np.isclose(
                dx,
                voxel_size
            ) and
            np.isclose(
                dy,
                voxel_size
            ) and
            np.isclose(
                dz,
                voxel_size
            )
        ):

            raise ValueError(
                f"Cluster {i} spacing "
                f"mismatch: "
                f"dx={dx}, dy={dy}, dz={dz}, "
                f"expected {voxel_size}"
            )

    x_min = min(
        res["xs"][0]
        for res in cluster_results
    )

    x_max = max(
        res["xs"][-1]
        for res in cluster_results
    )

    y_min = min(
        res["ys"][0]
        for res in cluster_results
    )

    y_max = max(
        res["ys"][-1]
        for res in cluster_results
    )

    z_min = min(
        res["zs"][0]
        for res in cluster_results
    )

    z_max = max(
        res["zs"][-1]
        for res in cluster_results
    )

    global_xs = np.arange(
        x_min,
        x_max +
        0.5 * voxel_size,
        voxel_size
    )

    global_ys = np.arange(
        y_min,
        y_max +
        0.5 * voxel_size,
        voxel_size
    )

    global_zs = np.arange(
        z_min,
        z_max +
        0.5 * voxel_size,
        voxel_size
    )

    Nx = len(global_xs)
    Ny = len(global_ys)
    Nz = len(global_zs)

    if verbose:

        print(
            "[global merge]"
        )

        print(
            f"  voxel_size = "
            f"{voxel_size}"
        )

        print(
            f"  global grid shape "
            f"(Nz, Ny, Nx) = "
            f"({Nz}, {Ny}, {Nx})"
        )

        print(
            f"  x range = "
            f"[{global_xs[0]:.3f}, "
            f"{global_xs[-1]:.3f}]"
        )

        print(
            f"  y range = "
            f"[{global_ys[0]:.3f}, "
            f"{global_ys[-1]:.3f}]"
        )

        print(
            f"  z range = "
            f"[{global_zs[0]:.3f}, "
            f"{global_zs[-1]:.3f}]"
        )

    global_vof_nzyx = np.zeros(
        (Nz, Ny, Nx),
        dtype=np.uint8
    )

    for i, res in enumerate(
        cluster_results
    ):

        local_vof = \
            res["vof"].astype(
                np.uint8
            )

        xs = res["xs"]
        ys = res["ys"]
        zs = res["zs"]

        nx_local = len(xs)
        ny_local = len(ys)
        nz_local = len(zs)

        ix0 = int(
            np.round(
                (
                    xs[0] -
                    global_xs[0]
                ) /
                voxel_size
            )
        )

        iy0 = int(
            np.round(
                (
                    ys[0] -
                    global_ys[0]
                ) /
                voxel_size
            )
        )

        iz0 = int(
            np.round(
                (
                    zs[0] -
                    global_zs[0]
                ) /
                voxel_size
            )
        )

        ix1 = ix0 + nx_local
        iy1 = iy0 + ny_local
        iz1 = iz0 + nz_local

        if verbose:

            print(
                f"  cluster {i}: "
                f"local shape "
                f"(nz,ny,nx)="
                f"({nz_local},"
                f"{ny_local},"
                f"{nx_local}) "
                f"-> global slices "
                f"z[{iz0}:{iz1}] "
                f"y[{iy0}:{iy1}] "
                f"x[{ix0}:{ix1}]"
            )

        global_vof_nzyx[
            iz0:iz1,
            iy0:iy1,
            ix0:ix1
        ] = np.maximum(
            global_vof_nzyx[
                iz0:iz1,
                iy0:iy1,
                ix0:ix1
            ],
            local_vof
        )

    global_volume_nxyz = \
        np.transpose(
            global_vof_nzyx,
            (2, 1, 0)
        ).copy()

    if verbose:

        print(
            f"  merged occupied "
            f"voxels = "
            f"{np.count_nonzero(global_vof_nzyx)}"
        )

        print(
            f"  global_volume_nxyz "
            f"shape = "
            f"{global_volume_nxyz.shape}"
        )

    return (
        global_vof_nzyx,
        global_xs,
        global_ys,
        global_zs,
        global_volume_nxyz
    )


# ============================================================
# Distribution analysis
# ============================================================

LENGTH_RANGE_UM = (
    0.0,
    200.0
)

TORTUOSITY_RANGE = (
    1.0,
    3.0
)

DIAMETER_RANGE_UM = (
    0.0,
    30.0
)

LENGTH_BIN_WIDTH_UM = 5.0
TORT_BIN_WIDTH = 0.05
DIAMETER_BIN_WIDTH_UM = 1.0


def _fixed_edges(
    value_range,
    bin_width
):

    vmin, vmax = value_range

    return np.arange(
        vmin,
        vmax + bin_width,
        bin_width
    )


LENGTH_EDGES = _fixed_edges(
    LENGTH_RANGE_UM,
    LENGTH_BIN_WIDTH_UM
)

TORT_EDGES = _fixed_edges(
    TORTUOSITY_RANGE,
    TORT_BIN_WIDTH
)

DIAMETER_EDGES = _fixed_edges(
    DIAMETER_RANGE_UM,
    DIAMETER_BIN_WIDTH_UM
)


def extract_vessel_segments(G):
    """
    Split the skeleton graph into vessel segments.

    A segment always starts and ends at a TRUE terminal node:
        - end point    (degree == 1)
        - branch point (degree >= 3)
    and runs through any number of internal (degree == 2) nodes.

    The walk is started ONLY from terminal nodes, so a segment can
    never begin at an arbitrary internal point in the middle of a
    vessel. Pure closed loops that contain no terminal node at all
    are handled in a second pass and reported as closed loops
    (length computed, tortuosity = NaN).
    """

    visited_edges = set()
    segments = []

    def build_segment(path, closed):
        coords = np.array(
            [G.nodes[n]["pos"] for n in path],
            dtype=float
        )

        if len(coords) < 2:
            length = np.nan
            tort = np.nan
        else:
            length = np.linalg.norm(
                coords[1:] - coords[:-1],
                axis=1
            ).sum()

            if closed:
                tort = np.nan
            else:
                chord = np.linalg.norm(coords[-1] - coords[0])
                tort = length / chord if chord > 0 else np.nan

        segments.append(
            {
                "nodes": path,
                "coords": coords,
                "length": length,
                "tortuosity": tort,
                "is_closed_loop": closed
            }
        )

    def is_terminal(n):
        d = G.degree[n]
        return d == 1 or d >= 3

    def walk_to_next_terminal(u, v):
        path = [u, v]
        prev, curr = u, v
        while not is_terminal(curr):
            nbrs = [n for n in G.neighbors(curr) if n != prev]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            path.append(nxt)
            prev, curr = curr, nxt
        return path

    # --------------------------------------------------------
    # Pass 1: segments between terminal nodes
    # (end point / branch point -> end point / branch point)
    # --------------------------------------------------------
    for u in [n for n in G.nodes if is_terminal(n)]:
        for v in G.neighbors(u):
            if frozenset((u, v)) in visited_edges:
                continue
            path = walk_to_next_terminal(u, v)
            for i in range(len(path) - 1):
                visited_edges.add(frozenset((path[i], path[i + 1])))
            build_segment(path, closed=False)

    # --------------------------------------------------------
    # Pass 2: pure closed loops with no terminal node
    # --------------------------------------------------------
    handled = set()
    for u, v in G.edges:
        e = frozenset((u, v))
        if e in visited_edges or e in handled:
            continue
        path = [u, v]
        handled.add(e)
        prev, curr = u, v
        while curr != u:
            nbrs = [n for n in G.neighbors(curr) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            handled.add(frozenset((curr, nxt)))
            path.append(nxt)
            prev, curr = curr, nxt
        build_segment(path, closed=True)

    return segments


def segments_to_dataframe(
    segments
):

    return pd.DataFrame(
        {
            "start_node": [s["nodes"][0] for s in segments],
            "end_node": [s["nodes"][-1] for s in segments],
            "start_x": [s["coords"][0, 0] for s in segments],
            "start_y": [s["coords"][0, 1] for s in segments],
            "start_z": [s["coords"][0, 2] for s in segments],
            "end_x": [s["coords"][-1, 0] for s in segments],
            "end_y": [s["coords"][-1, 1] for s in segments],
            "end_z": [s["coords"][-1, 2] for s in segments],
            "length": [s["length"] for s in segments],
            "tortuosity": [s["tortuosity"] for s in segments],
            "is_closed_loop": [s["is_closed_loop"] for s in segments],
            "n_points_in_path": [len(s["nodes"]) for s in segments],
        }
    )


def plot_count_and_pdf(
    data,
    edges,
    xlabel,
    title,
    xlim,
    color,
    save_path=None
):

    data = np.asarray(
        data,
        dtype=float
    )

    data = data[
        np.isfinite(data)
    ]

    if data.size == 0:

        return {
            "mean": np.nan,
            "std": np.nan
        }

    mean_val = float(
        np.mean(data)
    )

    std_val = float(
        np.std(
            data,
            ddof=1
        )
    )

    fig, (
        ax1,
        ax2
    ) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        sharex=True
    )

    ax1.hist(
        data,
        bins=edges,
        density=False,
        color=color,
        alpha=0.9
    )

    ax1.set_ylabel(
        "Count"
    )

    ax1.set_title(
        title
    )

    ax1.set_xlim(
        *xlim
    )

    ax1.grid(
        alpha=0.3
    )

    ax2.hist(
        data,
        bins=edges,
        density=True,
        color=color,
        alpha=0.6
    )

    ax2.set_xlabel(
        xlabel
    )

    ax2.set_ylabel(
        "Probability density"
    )

    ax2.set_xlim(
        *xlim
    )

    ax2.grid(
        alpha=0.3
    )

    txt = (
        f"μ = {mean_val:.2f}\n"
        f"σ = {std_val:.2f}"
    )

    ax2.text(
        0.98,
        0.98,
        txt,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.2
        )
    )

    plt.tight_layout()

    if save_path is not None:

        fig.savefig(
            str(save_path),
            dpi=200
        )

        print(
            f"[save] figure written: "
            f"{save_path}"
        )

    plt.close(fig)

    return {
        "mean": mean_val,
        "std": std_val
    }


def plot_distributions(
    df,
    diameter_um=None,
    save_prefix=None
):

    def _png(tag):

        if save_prefix is None:
            return None

        return f"{save_prefix}_{tag}.png"


    stats = {}

    stats["length"] = \
        plot_count_and_pdf(
            df["length"],
            LENGTH_EDGES,
            xlabel="Length (µm)",
            title="Segment Length Distribution",
            xlim=LENGTH_RANGE_UM,
            color="tab:blue",
            save_path=_png("length")
        )

    tort = df[
        "tortuosity"
    ].dropna()

    tort = tort[
        (
            tort >=
            TORTUOSITY_RANGE[0]
        ) &
        (
            tort <=
            TORTUOSITY_RANGE[1]
        )
    ]

    if not tort.empty:

        stats["tortuosity"] = \
            plot_count_and_pdf(
                tort,
                TORT_EDGES,
                xlabel="Tortuosity",
                title="Segment Tortuosity Distribution",
                xlim=TORTUOSITY_RANGE,
                color="tab:green",
                save_path=_png("tortuosity")
            )

    else:

        stats["tortuosity"] = {
            "mean": np.nan,
            "std": np.nan
        }

    if diameter_um is not None:

        d = np.asarray(
            diameter_um,
            dtype=float
        )

        d = d[
            np.isfinite(d)
        ]

        d = d[
            (
                d >=
                DIAMETER_RANGE_UM[0]
            ) &
            (
                d <=
                DIAMETER_RANGE_UM[1]
            )
        ]

        if d.size > 0:

            stats["diameter"] = \
                plot_count_and_pdf(
                    d,
                    DIAMETER_EDGES,
                    xlabel="Diameter (µm)",
                    title="Diameter Distribution",
                    xlim=DIAMETER_RANGE_UM,
                    color="tab:red",
                    save_path=_png("diameter")
                )

        else:

            stats["diameter"] = {
                "mean": np.nan,
                "std": np.nan
            }

    else:

        stats["diameter"] = {
            "mean": np.nan,
            "std": np.nan
        }

    return stats


def analyze_vessel_graph(
    G,
    diameter_um=None,
    save_prefix=None
):

    segments = \
        extract_vessel_segments(G)

    df = segments_to_dataframe(
        segments
    )

    print(
        f"Segments: {len(df)}"
    )

    print(
        f"Raw mean length: "
        f"{df['length'].mean():.2f} µm"
    )

    print(
        f"Raw mean tortuosity: "
        f"{df['tortuosity'].mean():.2f}"
    )

    stats = plot_distributions(
        df,
        diameter_um=diameter_um,
        save_prefix=save_prefix
    )

    print(
        "\n=== PDF Statistics ==="
    )

    if not np.isnan(
        stats["length"]["mean"]
    ):

        print(
            f"Length  -> μ = "
            f"{stats['length']['mean']:.2f} "
            f"µm, σ = "
            f"{stats['length']['std']:.2f} µm"
        )

    if not np.isnan(
        stats["tortuosity"]["mean"]
    ):

        print(
            f"Tort    -> μ = "
            f"{stats['tortuosity']['mean']:.3f}, "
            f"σ = "
            f"{stats['tortuosity']['std']:.3f}"
        )

    if not np.isnan(
        stats["diameter"]["mean"]
    ):

        print(
            f"Diameter-> μ = "
            f"{stats['diameter']['mean']:.2f} "
            f"µm, σ = "
            f"{stats['diameter']['std']:.2f} µm"
        )

    return (
        df,
        stats
    )


def simple_skeleton_with_radius(
    volume
):

    dist = ndi.distance_transform_edt(
        volume
    )

    local_max = (
        ndi.maximum_filter(
            dist,
            size=3
        ) == dist
    )

    skeleton = (
        local_max &
        (volume > 0)
    )

    radii = dist[
        skeleton
    ]

    return (
        skeleton,
        radii
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

print(
    "[env] CPUs available:",
    len(os.sched_getaffinity(0)),
    flush=True
)

print(
    "[load] reading:",
    NPY_PATH
)

# ------------------------------------------------------------
# Load ALL timesteps
# ------------------------------------------------------------

all_timesteps = \
    load_all_timesteps(
        NPY_PATH
    )

print(
    f"[load] total timesteps: "
    f"{len(all_timesteps)}"
)

for i, pts in enumerate(
    all_timesteps[:5]
):

    print(
        f"  timestep {i}: "
        f"{pts.shape[0]} points"
    )

if len(all_timesteps) > 5:

    print(
        "  ..."
    )

    print(
        f"  last timestep: "
        f"{len(all_timesteps)-1}, "
        f"{all_timesteps[-1].shape[0]} points"
    )


# ------------------------------------------------------------
# Select every 4th timestep
# ------------------------------------------------------------

selected_timestep_indices, \
selected_timesteps = \
    select_timesteps(
        all_timesteps,
        stride=TIMESTEP_STRIDE
    )

print(
    "\n[timestep selection]"
)

print(
    f"  requested stride = "
    f"{TIMESTEP_STRIDE}"
)

print(
    f"  original timesteps = "
    f"{len(all_timesteps)}"
)

print(
    f"  selected timesteps = "
    f"{len(selected_timesteps)}"
)

print(
    f"  selected indices = "
    f"{selected_timestep_indices}"
)


# ------------------------------------------------------------
# Create lightweight timestep animation
# ------------------------------------------------------------

if SAVE_TIMESTEP_ANIMATION:

    make_timestep_animation(
        selected_timesteps,
        selected_timestep_indices,
        timestep_animation_path,
        pc_max=ANIMATION_PC_MAX
    )


# ------------------------------------------------------------
# IMPORTANT:
# The reconstruction/cluster analysis below is performed
# on the FINAL timestep, preserving your original workflow.
# ------------------------------------------------------------

points_all = all_timesteps[-1]

print(
    "\n[load] processing final timestep:"
)

print(
    f"  timestep index = "
    f"{len(all_timesteps)-1}"
)

print(
    f"  total points = "
    f"{points_all.shape[0]}"
)


# ============================================================
# Clustering
# ============================================================

print(
    "[cluster] DBSCAN clustering..."
)

db = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES
)

labels = db.fit_predict(
    points_all
)

unique_labels = [
    lab
    for lab in np.unique(labels)
    if lab != -1
]

print(
    f"[cluster] found "
    f"{len(unique_labels)} clusters "
    f"(excluding noise label -1)."
)

cluster_sizes_all = []

for lab in unique_labels:

    size = np.sum(
        labels == lab
    )

    cluster_sizes_all.append(
        (
            lab,
            size
        )
    )

cluster_sizes_all.sort(
    key=lambda x: x[1],
    reverse=True
)

print(
    "[cluster] cluster sizes "
    "before filtering "
    "(label, size):"
)

for lab, size in \
        cluster_sizes_all:

    print(
        f"    label={lab}, "
        f"size={size}"
    )

cluster_sizes = [
    (lab, size)
    for lab, size
    in cluster_sizes_all
    if size >= MIN_CLUSTER_SIZE
]

print(
    f"\n[filter] keeping only "
    f"clusters with at least "
    f"{MIN_CLUSTER_SIZE} particles."
)

print(
    f"[filter] retained "
    f"{len(cluster_sizes)} clusters "
    f"out of "
    f"{len(cluster_sizes_all)} "
    f"total clusters."
)

if len(cluster_sizes) == 0:

    print(
        "[filter] no clusters "
        "passed the size threshold."
    )

    cluster_results = []

else:

    print(
        "[filter] retained cluster "
        "sizes (label, size):"
    )

    for lab, size in \
            cluster_sizes:

        print(
            f"    label={lab}, "
            f"size={size}"
        )


# ============================================================
# Connectivity
# ============================================================

if len(cluster_sizes) > 0:

    largest_cluster_size = \
        cluster_sizes[0][1]

    total_retained_particles = \
        sum(
            size
            for _, size
            in cluster_sizes
        )

    connectivity = (
        largest_cluster_size /
        total_retained_particles
    )

    print(
        "\n[connectivity]"
    )

    print(
        f"    largest cluster size "
        f"= {largest_cluster_size}"
    )

    print(
        f"    total retained particles "
        f"= {total_retained_particles}"
    )

    print(
        f"    connectivity "
        f"= {connectivity:.6f}"
    )

else:

    largest_cluster_size = 0
    total_retained_particles = 0
    connectivity = np.nan


# ============================================================
# Cluster reconstruction
# ============================================================

cluster_results = []

for rank, (
    lab,
    size
) in enumerate(
    cluster_sizes,
    start=1
):

    print(
        f"\n===== Retained cluster "
        f"rank {rank} "
        f"(label={lab}, "
        f"size={size}) ====="
    )

    cluster_pts = \
        points_all[
            labels == lab
        ]

    if cluster_pts.shape[0] < 5:

        print(
            "  [skip] too few points "
            "for reliable MLS; skipping."
        )

        continue

    print(
        "  [grid] building cubic grid..."
    )

    xs, ys, zs = \
        build_cubic_grid_for_cluster(
            cluster_pts,
            voxel_size=VOXEL_SIZE,
            padding_frac=PADDING_FRAC
        )

    print(
        f"      grid shape: "
        f"nx={len(xs)}, "
        f"ny={len(ys)}, "
        f"nz={len(zs)}"
    )

    print(
        "  [normals] estimating + orienting..."
    )

    normals = \
        estimate_normals_oriented(
            cluster_pts,
            k=K_NORMALS,
            orient_k=ORIENT_K
        )

    print(
        "      normals done."
    )

    print(
        "  [sdf] MLS signed distance..."
    )

    phi = sdf_mls(
        cluster_pts,
        normals,
        xs,
        ys,
        zs,
        k=MLS_K,
        sigma_mode=MLS_SIGMA_MODE,
        sigma_scale=MLS_SIGMA_SCALE
    )

    print(
        "      sdf done. "
        "phi shape:",
        phi.shape
    )

    corner_vals = np.array(
        [
            phi[i, j, k]
            for i in (0, -1)
            for j in (0, -1)
            for k in (0, -1)
        ]
    )

    if np.median(
        corner_vals
    ) < 0:

        print(
            "  [sdf] sign appears flipped. "
            "Flipping phi."
        )

        phi = -phi

    vof = (
        phi < 0
    ).astype(
        np.uint8
    )

    dx = (
        xs[1] - xs[0]
        if len(xs) > 1
        else VOXEL_SIZE
    )

    dy = (
        ys[1] - ys[0]
        if len(ys) > 1
        else VOXEL_SIZE
    )

    dz = (
        zs[1] - zs[0]
        if len(zs) > 1
        else VOXEL_SIZE
    )

    (
        V,
        A_voxel,
        n_vox,
        Nx,
        Ny,
        Nz,
        n_faces
    ) = voxel_volume_and_surface_area(
        vof,
        dx,
        dy,
        dz
    )

    psi_voxel = \
        sphericity_from_volume_area(
            V,
            A_voxel
        )

    (
        A_mesh_vof,
        n_verts_vof,
        n_tris_vof
    ) = mesh_surface_area_from_vof(
        vof,
        dx,
        dy,
        dz,
        level=0.5
    )

    psi_mesh_vof = \
        sphericity_from_volume_area(
            V,
            A_mesh_vof
        )

    (
        A_mesh_phi,
        n_verts_phi,
        n_tris_phi
    ) = mesh_surface_area_from_phi(
        phi,
        dx,
        dy,
        dz,
        level=0.0
    )

    psi_mesh_phi = \
        sphericity_from_volume_area(
            V,
            A_mesh_phi
        )

    print(
        "  [shape metrics]"
    )

    print(
        f"      occupied voxels "
        f"= {n_vox}"
    )

    print(
        f"      exposed x-faces "
        f"(Nx) = {Nx}"
    )

    print(
        f"      exposed y-faces "
        f"(Ny) = {Ny}"
    )

    print(
        f"      exposed z-faces "
        f"(Nz) = {Nz}"
    )

    print(
        f"      total exposed faces "
        f"= {n_faces}"
    )

    print(
        f"      voxel volume "
        f"= {V:.6f}"
    )

    print(
        f"      voxel-face area "
        f"= {A_voxel:.6f}"
    )

    print(
        f"      MC area from binary vof "
        f"= {A_mesh_vof:.6f}"
    )

    print(
        f"      MC area from continuous phi "
        f"= {A_mesh_phi:.6f}"
    )

    print(
        f"      sphericity from voxel area "
        f"= {psi_voxel:.6f}"
    )

    print(
        f"      sphericity from MC(vof) "
        f"= {psi_mesh_vof:.6f}"
    )

    print(
        f"      sphericity from MC(phi) "
        f"= {psi_mesh_phi:.6f}"
    )

    print(
        f"      MC(vof) verts / tris "
        f"= {n_verts_vof} / "
        f"{n_tris_vof}"
    )

    print(
        f"      MC(phi) verts / tris "
        f"= {n_verts_phi} / "
        f"{n_tris_phi}"
    )

    cluster_results.append(
        {
            "label": lab,
            "rank": rank,
            "size": size,
            "points": cluster_pts,
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "phi": phi,
            "vof": vof,
            "n_vox": n_vox,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "n_faces": n_faces,
            "volume_vox": V,
            "area_voxel": A_voxel,
            "area_mesh_vof": A_mesh_vof,
            "area_mesh_phi": A_mesh_phi,
            "mesh_vof_n_verts": n_verts_vof,
            "mesh_vof_n_tris": n_tris_vof,
            "mesh_phi_n_verts": n_verts_phi,
            "mesh_phi_n_tris": n_tris_phi,
            "sphericity_voxel": psi_voxel,
            "sphericity_mesh_vof": psi_mesh_vof,
            "sphericity_mesh_phi": psi_mesh_phi
        }
    )

    if PLOT_EACH_CLUSTER:

        title = (
            f"Cluster {rank} "
            f"(label={lab}, N={size})"
            f" | Psi_vox="
            f"{psi_voxel:.3f}"
            f" | Psi_MCvof="
            f"{psi_mesh_vof:.3f}"
            f" | Psi_MCphi="
            f"{psi_mesh_phi:.3f}"
        )

        cluster_img_path = (
            output_dir /
            f"{base_name}"
            f"_cluster_rank_"
            f"{rank:03d}."
            f"{IMAGE_FORMAT}"
        )

        plot_volume_world(
            vof,
            xs,
            ys,
            zs,
            cluster_pts,
            title=title,
            ds=DS_VOL,
            pc_max=PC_MAX,
            save_path=cluster_img_path
        )


print(
    "\n[done] Processed retained "
    f"clusters: {len(cluster_results)}"
)

print(
    "You can now access, e.g. "
    "cluster_results[i]"
    "['area_mesh_phi']."
)


# ============================================================
# Summary
# ============================================================

if len(cluster_results) > 0:

    print(
        "\n[summary] retained "
        "cluster shape metrics"
    )

    print(
        " rank | label | particles | "
        "voxels | volume | A_voxel | "
        "A_MCvof | A_MCphi | "
        "Psi_vox | Psi_MCvof | Psi_MCphi "
    )

    print(
        "--------------------------------------------------"
    )

    for res in cluster_results:

        print(
            f" {res['rank']:4d} |"
            f" {res['label']:5d} |"
            f" {res['size']:9d} |"
            f" {res['n_vox']:6d} |"
            f" {res['volume_vox']:7.3f} |"
            f" {res['area_voxel']:8.3f} |"
            f" {res['area_mesh_vof']:8.3f} |"
            f" {res['area_mesh_phi']:8.3f} |"
            f" {res['sphericity_voxel']:8.6f} |"
            f" {res['sphericity_mesh_vof']:10.6f} |"
            f" {res['sphericity_mesh_phi']:10.6f}"
        )


num_clusters_after_filtering = \
    len(cluster_results)

if num_clusters_after_filtering > 0:

    avg_sphericity_mcphi = \
        np.nanmean(
            [
                res[
                    "sphericity_mesh_phi"
                ]
                for res in cluster_results
            ]
        )

else:

    avg_sphericity_mcphi = np.nan


summary_row = {
    "input_file": input_path.name,
    "connectivity": connectivity,
    "average_sphericity_MCphi":
        avg_sphericity_mcphi,
    "num_clusters_after_filtering":
        num_clusters_after_filtering,
    "num_total_timesteps":
        len(all_timesteps),
    "timestep_stride":
        TIMESTEP_STRIDE,
    "num_animation_timesteps":
        len(selected_timesteps)
}


with open(
    summary_csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "input_file",
            "connectivity",
            "average_sphericity_MCphi",
            "num_clusters_after_filtering",
            "num_total_timesteps",
            "timestep_stride",
            "num_animation_timesteps"
        ]
    )

    writer.writeheader()

    writer.writerow(
        summary_row
    )


print(
    f"\n[save] summary CSV written: "
    f"{summary_csv_path}"
)


# ============================================================
# Plot all reconstructed clusters
# ============================================================

if (
    PLOT_ALL_CLUSTERS_TOGETHER and
    len(cluster_results) > 0
):

    print(
        "\n[visualization] Plotting "
        "ALL retained reconstructed "
        "clusters together..."
    )

    all_clusters_img_path = (
        output_dir /
        f"{base_name}"
        f"_all_clusters."
        f"{IMAGE_FORMAT}"
    )

    plot_all_reconstructed_clusters(
        cluster_results,
        ds=DS_VOL,
        pc_max=PC_MAX,
        save_path=all_clusters_img_path
    )


# ============================================================
# Target cluster
# ============================================================

if TARGET_RANK is not None:

    print(
        f"\n[visualization] Plotting "
        f"VOF only for retained "
        f"cluster with "
        f"rank={TARGET_RANK}..."
    )

    found = False

    for res in cluster_results:

        rank = res["rank"]

        if rank != TARGET_RANK:
            continue

        lab = res["label"]
        size = res["size"]

        vof = res["vof"]

        xs = res["xs"]
        ys = res["ys"]
        zs = res["zs"]

        pts = res["points"]

        psi_v = \
            res["sphericity_voxel"]

        psi_b = \
            res["sphericity_mesh_vof"]

        psi_p = \
            res["sphericity_mesh_phi"]

        title = (
            f"Cluster {rank} "
            f"(label={lab}, N={size}) "
            f"— VOF Reconstruction "
            f"| Psi_vox={psi_v:.3f} "
            f"| Psi_MCvof={psi_b:.3f} "
            f"| Psi_MCphi={psi_p:.3f}"
        )

        target_img_path = (
            output_dir /
            f"{base_name}"
            f"_target_rank_"
            f"{rank:03d}."
            f"{IMAGE_FORMAT}"
        )

        plot_volume_world(
            vof,
            xs,
            ys,
            zs,
            pts,
            title=title,
            ds=DS_VOL,
            pc_max=PC_MAX,
            save_path=target_img_path
        )

        found = True

        break

    if not found:

        print(
            f"[warning] No retained "
            f"cluster with rank="
            f"{TARGET_RANK} found "
            f"in cluster_results."
        )


# ============================================================
# Per-cluster skeleton graphs (batch mode, no interactive editor)
#
# Each cluster's skeleton graph is built from its own local VOF
# and saved as an HTML file. The unedited graphs are used
# directly for quantification.
# ============================================================

per_cluster_graphs = []

for k, data in enumerate(
    cluster_results
):

    vof_k = data["vof"]

    xs_k = data["xs"]
    ys_k = data["ys"]
    zs_k = data["zs"]

    rank_k = data.get("rank", k)

    print(
        f"\n=== Skeleton for cluster rank {rank_k} "
        f"(label={data.get('label', k)}) ===",
        flush=True
    )

    skeleton_html_path = (
        output_dir /
        f"{base_name}_cluster_rank_{rank_k:03d}_skeleton.html"
    )

    (
        G_k,
        end_k,
        branch_k,
        loops_k
    ) = build_graph_from_vof(
        vof_k,
        xs_k,
        ys_k,
        zs_k,
        dbscan_eps=1.0,
        connect_radius=10.0,
        prune_radius=10.0,
        verbose=True,
        save_path=skeleton_html_path
    )

    per_cluster_graphs.append(
        G_k
    )


# ============================================================
# Per-cluster quantification
#
# Each cluster is quantified using ONLY its own local
# vof/xs/ys/zs and its own skeleton graph. Only the cheap,
# already-computed outputs (graphs, segment tables, diameter
# arrays) are combined at the end.
# ============================================================

per_cluster_metrics = []
per_cluster_segment_dfs = []
all_diameter_um_parts = []

for k, (data, G_k) in enumerate(
    zip(cluster_results, per_cluster_graphs)
):

    rank_k = data.get("rank", k)
    label_k = data.get("label", k)

    print(
        f"\n===== Quantifying cluster rank "
        f"{rank_k} (label={label_k}) =====",
        flush=True
    )

    (
        end_k,
        branch_k,
        internal_k,
        isolated_k,
        loops_k
    ) = analyze_graph_structure(G_k)

    print(f"  nodes      : {G_k.number_of_nodes()}")
    print(f"  edges      : {G_k.number_of_edges()}")
    print(f"  endpoints  : {len(end_k)}")
    print(f"  branches   : {len(branch_k)}")
    print(f"  internals  : {len(internal_k)}")
    print(f"  isolated   : {len(isolated_k)}")
    print(f"  loops      : {len(loops_k)}")

    # Diameter from THIS cluster's own local vof.
    # The distance transform returns radii in VOXELS, so they
    # are converted to physical units with the voxel size.
    skeleton_mask_k, skeleton_radii_k = \
        simple_skeleton_with_radius(data["vof"])

    voxel_size_k = float(data["xs"][1] - data["xs"][0])

    diameter_um_k = 2.0 * skeleton_radii_k * voxel_size_k

    cluster_prefix = (
        output_dir /
        f"{base_name}_cluster_rank_{rank_k:03d}"
    )

    df_k, stats_k = analyze_vessel_graph(
        G_k,
        diameter_um=diameter_um_k,
        save_prefix=cluster_prefix
    )

    total_length_um_k = (
        float(np.nansum(df_k["length"]))
        if "length" in df_k.columns
        else np.nan
    )

    average_diameter_um_k = (
        float(stats_k["diameter"]["mean"])
        if "diameter" in stats_k
        else np.nan
    )

    average_tortuosity_k = (
        float(stats_k["tortuosity"]["mean"])
        if "tortuosity" in stats_k
        else np.nan
    )

    per_cluster_metrics.append(
        {
            "rank": rank_k,
            "label": label_k,
            "nodes": G_k.number_of_nodes(),
            "edges": G_k.number_of_edges(),
            "endpoints": len(end_k),
            "branches": len(branch_k),
            "internals": len(internal_k),
            "isolated": len(isolated_k),
            "loops": len(loops_k),
            "segments": len(df_k),
            "total_length_um": total_length_um_k,
            "average_diameter_um": average_diameter_um_k,
            "average_tortuosity": average_tortuosity_k,
        }
    )

    per_cluster_segment_dfs.append(df_k)
    all_diameter_um_parts.append(diameter_um_k)

    print(f"  total_length_um     = {total_length_um_k:.6f}")
    print(f"  average_diameter_um = {average_diameter_um_k:.6f}")
    print(f"  average_tortuosity  = {average_tortuosity_k:.6f}")


# ============================================================
# Compile per-cluster results
# ============================================================

per_cluster_summary_df = pd.DataFrame(per_cluster_metrics)

print("\n[compiled] per-cluster summary")
print(per_cluster_summary_df)

per_cluster_summary_csv_path = (
    output_dir / f"{base_name}_per_cluster_summary.csv"
)

per_cluster_summary_df.to_csv(
    per_cluster_summary_csv_path,
    index=False
)

print(
    f"\n[save] per-cluster summary CSV written: "
    f"{per_cluster_summary_csv_path}"
)


# ============================================================
# Combined (all-clusters) topology + distributions
# ============================================================

if len(per_cluster_graphs) > 0:

    G_all, end_all, branch_all, loops_all = merge_graphs(
        per_cluster_graphs,
        verbose=True
    )

    (
        end_all,
        branch_all,
        internal_all,
        isolated_all,
        loops_all
    ) = analyze_graph_structure(G_all)

    print("\n[compiled] combined topology across all clusters")
    print(f"  nodes      : {G_all.number_of_nodes()}")
    print(f"  edges      : {G_all.number_of_edges()}")
    print(f"  endpoints  : {len(end_all)}")
    print(f"  branches   : {len(branch_all)}")
    print(f"  internals  : {len(internal_all)}")
    print(f"  isolated   : {len(isolated_all)}")
    print(f"  loops      : {len(loops_all)}")

    # Combined skeleton of all clusters as one HTML file.
    plot_graph_3d(
        G_all,
        end_nodes=end_all,
        branch_nodes=branch_all,
        save_path=(
            output_dir /
            f"{base_name}_all_clusters_skeleton.html"
        )
    )

    combined_segments_df = pd.concat(
        per_cluster_segment_dfs,
        ignore_index=True
    )

    combined_diameter_um = np.concatenate(
        all_diameter_um_parts
    )

    combined_stats = plot_distributions(
        combined_segments_df,
        diameter_um=combined_diameter_um,
        save_prefix=output_dir / f"{base_name}_all_clusters"
    )

    total_length_um = float(
        np.nansum(combined_segments_df["length"])
    )

    average_diameter_um = (
        float(combined_stats["diameter"]["mean"])
        if "diameter" in combined_stats
        else np.nan
    )

else:

    print("\n[compiled] no clusters to quantify.")

    total_length_um = np.nan
    average_diameter_um = np.nan

print("\n[compiled] combined distribution statistics")
print(f"  total_length_um     = {total_length_um:.6f}")
print(f"  average_diameter_um = {average_diameter_um:.6f}")


# ============================================================
# Update the run-level summary CSV with compiled results
# ============================================================

summary_df = pd.read_csv(
    summary_csv_path
)

summary_df["total_length_um"] = \
    total_length_um

summary_df["average_diameter_um"] = \
    average_diameter_um

summary_df.to_csv(
    summary_csv_path,
    index=False
)

print(
    f"[save] updated summary CSV "
    f"written: {summary_csv_path}"
)

print("\n[done] pipeline finished.", flush=True)
