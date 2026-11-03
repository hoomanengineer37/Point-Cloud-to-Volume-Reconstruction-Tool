# Vascular-Network-Reconstruction-Toolkit
Python package for vascular network reconstruction from point-cloud data. Includes vessel segmentation, skeleton extraction, and graph generation for length/diameter analysis, and 3D geometry creation for CFD. Built for reproducible workflows in tissue engineering and hemodynamics research

# Point-Cloud to Volume Reconstruction Tool

A Python pipeline that turns a raw 3D **point cloud** (for example, particle positions from a simulation) into a **watertight volume reconstruction**. It then measures that volume: surface area, sphericity, vessel length, diameter, tortuosity and network topology.

The reconstruction uses two methods:

- **PCA** (Principal Component Analysis) estimates a surface normal at every point from its local neighbourhood.
- A **level-set** formulation builds a signed distance field φ(x) from the oriented points. The reconstructed surface is the zero level set φ = 0, and the solid volume is the region where φ < 0.

# Validation
Validation: point cloud, volume reconstruction and skeleton

Validation of the solver on sim_4000.npy (final timestep, 4,000 points). (a) Input point cloud. (b) Volume reconstruction: the zero level set φ = 0 of the signed distance field, enclosing the solid region φ < 0. (c) Centre-line skeleton extracted from the reconstructed volume. The reconstructed volume closely follows the geometry of the input point cloud, and the skeleton reproduces its network topology (branches, end points and loops). Together these validate the reconstruction and the network analysis.

<img width="972" height="451" alt="image" src="https://github.com/user-attachments/assets/714247e7-bc1a-45a8-9e36-e35c92fa13dc" />


*Left: the input point cloud (final timestep of `sim_4000.npy`, 4,000 points). Middle: the reconstructed surface, which is the zero level set of the signed distance field. Right: the network skeleton*

---

## What the pipeline does

```
 point cloud (N×3)  ──►  DBSCAN clustering  ──►  PCA normals  ──►  orientation propagation
                                                                          │
                                                                          ▼
   metrics & plots  ◄──  marching cubes  ◄──  VOF (φ < 0)  ◄──  MLS signed distance field φ(x)
          │
          ▼
 skeletonisation  ──►  graph (nodes / edges)  ──►  pruning  ──►  segment statistics (length, diameter, tortuosity)
```

1. **Load and select timesteps.** Reads a time series of point clouds, keeps every *n*-th timestep and writes an interactive 3D animation of how the cloud evolves.
2. **Cluster.** DBSCAN splits the final point cloud into connected objects and drops noise and small fragments. A *connectivity* index is the fraction of retained points that belong to the largest cluster.
3. **Estimate normals with PCA.** For each point, PCA is fitted to its *k* nearest neighbours. The component with the smallest variance is the local surface normal.
4. **Orient the normals consistently.** PCA leaves the sign of each normal ambiguous. A graph traversal over the *k*-NN graph flips normals so that neighbouring normals agree.
5. **Build the level-set field.** A moving-least-squares (MLS) estimate of the signed distance is evaluated on a cubic voxel grid. It is a Gaussian-weighted average of the distances to the tangent planes of nearby points.
6. **Make the volume.** The solid is the volume fraction field `VOF = (φ < 0)`. The surface is extracted with marching cubes at φ = 0.
7. **Measure the geometry.** Volume, surface area (estimated three ways: voxel faces, marching cubes on the VOF, marching cubes on φ) and sphericity Ψ = π^(1/3) (6V)^(2/3) / A.
8. **Skeletonise and analyse the network.** The volume is reduced to a centre-line skeleton (distance-transform ridges), turned into a NetworkX graph and pruned. It is then split into vessel segments, which give length, local diameter, tortuosity, and the counts of branch points, end points and loops.

---

## Methods

### PCA normal estimation
For a point **p**ᵢ with neighbours {**p**ⱼ}, PCA is applied to the centred neighbourhood. The eigenvector with the **smallest** eigenvalue points out of the locally best-fitting plane, so it is the normal **n**ᵢ. Orientation is then propagated through a depth-first walk on the *k*-NN graph, flipping **n**ⱼ whenever **n**ᵢ · **n**ⱼ < 0.

### Level-set (signed distance) reconstruction
At each grid point **x**, the signed distance is the weighted average of the distances to the tangent planes of the *k* nearest points:

```
φ(x) = Σ wᵢ (x − pᵢ)·nᵢ / Σ wᵢ ,      wᵢ = exp(−|x − pᵢ|² / 2σ²)
```

σ adapts to the local point spacing: it is set from the *k*-th neighbour distance (`"kth"`), the median neighbour distance (`"median"`), or a fixed value you give. The surface is the zero level set φ = 0.

### Speed optimisations
- **Narrow band.** The MLS field is only evaluated within `SDF_BAND` of the point cloud. Grid points farther away are labelled "outside" after a cheap nearest-neighbour check. On `sim_4000.npy` this means 41 % of the 256³ grid is evaluated.
- **Slice-wise vectorisation.** φ is computed one z-slice at a time with vectorised NumPy and parallel `cKDTree` queries (`workers=-1`). This keeps memory bounded and avoids Python loops over voxels.
- **Robust sign check.** The sign used for "outside" is detected from samples near the edge of the band and from the grid corners, so φ is flipped automatically if needed.

---

## Code structure

The tool is one script organised in clear functional layers. Each layer is a set of small, single-purpose functions with docstrings, and a linear main pipeline at the end calls them in order.

| Layer | Key functions | Role |
|---|---|---|
| **Configuration** | top-of-file constants | Every tunable parameter (clustering, normals, MLS, grid, plotting) in one place, with comments. |
| **I/O and timesteps** | `load_all_timesteps`, `select_timesteps` | Reads several `.npy` layouts (object array, `(T,N,3)`, `(N,3)`, list) and checks shapes, with clear error messages. |
| **Visualisation** | `make_timestep_animation`, `plot_volume_world`, `plot_all_reconstructed_clusters`, `plot_graph_3d` | Interactive Plotly HTML output with fixed axes and down-sampling to keep files small. |
| **Surface reconstruction** | `estimate_normals_oriented`, `build_cubic_grid_for_cluster`, `mls_signed_dist_slice`, `sdf_mls` | PCA normals, orientation propagation, narrow-band MLS signed distance field. |
| **Geometry metrics** | `voxel_volume_and_surface_area`, `mesh_surface_area_from_vof`, `mesh_surface_area_from_phi`, `sphericity_from_volume_area` | Volume, surface area estimated three independent ways, sphericity. |
| **Skeleton graph** | `simple_skeleton`, `average_nearby_points`, `build_vessel_graph_from_coords`, `build_graph_from_vof`, `merge_graphs` | Distance-transform skeleton → node merging → radius graph with redundant triangle edges removed. |
| **Graph pruning and editing** | `prune_sparse_branch_clusters`, `prune_end_nodes_adjacent_to_branch`, `delete_nodes_by_id`, `add_edges_by_node_ids`, … | Removes spurious branches and short spurs; tools for manual corrections. |
| **Network statistics** | `extract_vessel_segments`, `analyze_vessel_graph`, `plot_distributions`, `simple_skeleton_with_radius` | Splits the graph into terminal-to-terminal segments; length, diameter, tortuosity histograms and PDFs. |
| **Main pipeline** | bottom of file | Load → cluster → reconstruct each cluster → measure → skeletonise → quantify → write CSV, HTML and PNG output. |

Design choices worth noting:
- **Each cluster is handled on its own.** Every cluster gets its own grid, field and graph, so memory scales with the object size rather than the whole domain. The cheap results are merged at the end.
- **Built for batch runs.** It uses the headless Matplotlib backend (`Agg`), avoids interactive calls and flushes its logs, so it runs as-is in a Slurm job on an HPC cluster. The optional notebook editor is kept separate and switched off by default.
- **Results are cross-checked.** Surface area is computed from voxel faces, a binary-VOF mesh and a continuous-φ mesh, so you can judge how much the discretisation affects the answer.
- **Segments are defined carefully.** A vessel segment always runs between true terminal nodes (end points or branch points), and closed loops are handled separately. This keeps the length and tortuosity statistics free of artefacts.

---

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/hoomanengineer37/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `networkx`, `pandas`, `matplotlib`, `plotly`.

---

## Running the code

### Input file: `sim_4000.npy`
`sim_4000.npy` is the **example input file** that comes with the repository. It holds the time series of point clouds produced by a particle-based simulation:

- **400 timesteps**
- **4,000 points per timestep**, each with 3D coordinates `(x, y, z)` in µm
- stored as a pickled list of `(N, 3)` arrays (loaded with `allow_pickle=True`)

The reconstruction runs on the **final timestep**. All selected timesteps are used for the animation.

### Run

Put `sim_4000.npy` in the same folder as the script and run:

```bash
python Volume_Reconstruction_Tool.py
```

For `sim_4000.npy`, the full pipeline takes about one minute on a typical workstation.

To use your own data, change the input path at the top of the script:

```python
NPY_PATH = "sim_4000.npy"     # path to your point-cloud file
```

The other main parameters are in the same block:

| Parameter | Default | Meaning |
|---|---|---|
| `TIMESTEP_STRIDE` | 4 | Use every *n*-th timestep in the animation |
| `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES` | 3.0, 3 | Clustering neighbourhood |
| `MIN_CLUSTER_SIZE` | 50 | Drop clusters with fewer points |
| `K_NORMALS`, `ORIENT_K` | 10, 4 | Neighbours used for PCA normals and for orientation |
| `MLS_K`, `MLS_SIGMA_MODE` | 12, `"kth"` | MLS neighbours and kernel width |
| `SDF_BAND` | 20.0 | Narrow-band width; must be larger than the thickest vessel radius |
| `VOXEL_SIZE`, `PADDING_FRAC` | 0.5, 0.20 | Grid resolution and padding |

### Running on an HPC cluster (Slurm)

```bash
#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
module load python/3.11
source ~/venv/bin/activate
python Volume_Reconstruction_Tool.py
```

---

## Output

All files are written next to the input and named after it (`sim_4000_*`):

| File | Contents |
|---|---|
| `sim_4000_every_4_timesteps.html` | Interactive 3D animation of the point cloud over time |
| `sim_4000_all_clusters.html` | Interactive 3D view of the reconstructed volumes with the points overlaid |
| `sim_4000_cluster_rank_XXX_skeleton.html` | Skeleton graph of each cluster (end points and branch points marked) |
| `sim_4000_all_clusters_skeleton.html` | Combined skeleton graph |
| `sim_4000_*_length.png`, `*_diameter.png`, `*_tortuosity.png` | Histograms and PDFs of segment length, diameter and tortuosity |
| `sim_4000_summary.csv` | Run-level summary: connectivity, mean sphericity, total length, mean diameter |
| `sim_4000_per_cluster_summary.csv` | Per-cluster topology and morphology |

### Example results for `sim_4000.npy`

| Quantity | Value |
|---|---|
| Clusters retained | 1 (connectivity = 1.0) |
| Grid | 256 × 256 × 256 voxels (0.5 µm) |
| Reconstructed volume | 3.45 × 10⁴ µm³ |
| Surface area (marching cubes on φ) | 1.77 × 10⁴ µm² |
| Sphericity Ψ | 0.29 |
| Total vessel length | 736 µm |
| Mean vessel diameter | 7.2 ± 0.9 µm |
| Mean tortuosity | 1.36 |
| Network | 8 branch points, 8 end points, 8 loops |

![Diameter distribution](docs/diameter_distribution.png)

---

## Author

**Mohammad Hashemi**, PhD, Mechanical Engineering (University of Toronto)
Computational scientist in CFD, multiphase and particle-based modelling, and scientific computing on HPC.
