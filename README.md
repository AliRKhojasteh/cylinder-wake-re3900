<p align="center">
  <img src="docs/hero_banner.svg" alt="CylinderWake3900" width="100%">
</p>

<h1 align="center">CylinderWake3900</h1>

<p align="center">
  <b>ML-ready Lagrangian & Eulerian dataset of turbulent cylinder wake at Re = 3900</b>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.dib.2021.107725"><img src="https://img.shields.io/badge/Paper-Data%20in%20Brief-blue" alt="Paper"></a>
  <a href="https://doi.org/10.15454/GLNRHK"><img src="https://img.shields.io/badge/Data-INRAE%20Repository-green" alt="Data"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" alt="License"></a>
  <a href="https://pypi.org/project/cylinderwake3900/"><img src="https://img.shields.io/pypi/v/cylinderwake3900" alt="PyPI"></a>
  <a href="https://github.com/Ali-Rahimi-Khojasteh/cylinder-wake-re3900/stargazers"><img src="https://img.shields.io/github/stars/Ali-Rahimi-Khojasteh/cylinder-wake-re3900?style=social" alt="Stars"></a>
</p>

---

Direct Numerical Simulation (DNS) of incompressible turbulent flow past a smooth circular cylinder at Reynolds number 3900, computed with [Incompact3d](https://github.com/xcompact3d/Incompact3d). This repository provides **ML-ready data loaders, benchmark tasks, and example notebooks** for the dataset published in:

> **A. R. Khojasteh, S. Laizet, D. Heitz, Y. Yang** (2021). *Lagrangian and Eulerian dataset of the wake downstream of a smooth cylinder at a Reynolds number equal to 3900.* Data in Brief, 40, 107725. [doi:10.1016/j.dib.2021.107725](https://doi.org/10.1016/j.dib.2021.107725)

## What Makes This Dataset Unique

Most fluid ML benchmarks provide only Eulerian (grid-based) data. **CylinderWake3900 offers paired Lagrangian particle trajectories AND Eulerian fields**, enabling research in:

- **Sparse reconstruction / data assimilation** — reconstruct full 3D fields from sparse particle observations
- **Lagrangian-to-Eulerian conversion** — learn the mapping between representations
- **Particle tracking** — predict trajectories from velocity fields
- **Temporal forecasting** — predict future flow states
- **Super-resolution** — reconstruct high-res fields from coarse inputs

## Quick Start

```bash
pip install cylinderwake3900
```

```python
from cylinderwake import CylinderWake3900

# Eulerian velocity + pressure fields (auto-downloads on first use)
ds = CylinderWake3900("eulerian", "near", split="train")
sample = ds[0]
print(sample["velocity"].shape)   # (3, Nx, Ny, Nz)
print(sample["pressure"].shape)   # (1, Nx, Ny, Nz)

# Lagrangian particle trajectories (~100k particles)
ds_lag = CylinderWake3900("lagrangian", "near")
sample = ds_lag[0]
print(sample["positions"].shape)  # (N_particles, 3)

# PyTorch DataLoader — ready for training
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
```

### For AI Agents

```python
from cylinderwake.api import get_dataset_card_json
print(get_dataset_card_json())  # Full structured metadata as JSON
```

### CLI

```bash
cylinderwake-download    # Download raw data from INRAE
cylinderwake-convert     # Convert to HDF5
```

## Dataset Contents

| Sub-domain | Dimensions | Grid | Snapshots | dt | Snapshot Size |
|:----------:|:----------:|:----:|:---------:|:--:|:------------:|
| **1** (near-wake) | 10D × 8D × 6D | 769 × 777 × 256 | 100 | 0.0075 D/U∞ | 4.8 GB |
| **2** (far-wake) | 4D × 2D × 2D | 308 × 328 × 87 | 1000 | 0.00075 D/U∞ | 256 MB |

| Component | Sub-domain 1 | Sub-domain 2 | HDF5 Shape |
|-----------|:----------:|:----------:|:----------:|
| **3D Velocity** (ux, uy, uz) | 100 snapshots | 1000 snapshots | `(3, Nx, Ny, Nz)` |
| **3D Pressure** | — | 1000 snapshots | `(1, Nx, Ny, Nz)` |
| **2D Snapshots** (mid-plane) | ✓ | ✓ | `(2, Nx, Ny)` |
| **Lagrangian** (~100k particles each) | ✓ | ✓ | `(N_particles, 3)` |
| **Grid** | ✓ | ✓ | `(Nx,), (Ny,), (Nz,)` |

**Total raw data size**: ~288 GB (24 zip files on [INRAE](https://doi.org/10.15454/GLNRHK))

## Benchmark Tasks

We define four canonical ML tasks with standardized metrics. See [`benchmarks/tasks.md`](benchmarks/tasks.md) for formal definitions.

### Task 1: Temporal Forecasting
Predict the next Eulerian snapshot from previous ones.
- **Metric**: Relative L2 error
- **Baseline**: [Simple CNN notebook](notebooks/03_baseline_forecasting.ipynb)

### Task 2: Super-Resolution
Reconstruct full-resolution fields from coarsened input (4x, 8x, 16x).
- **Metric**: Relative L2 error, spectral error

### Task 3: Sparse Reconstruction *(unique to this dataset)*
Reconstruct the full Eulerian field from sparse Lagrangian observations at varying density (100%, 10%, 1%, 0.1%).
- **Metric**: Relative L2 error
- **Baseline**: [Sparse reconstruction notebook](notebooks/04_sparse_reconstruction.ipynb)

### Task 4: Lagrangian Prediction
Predict particle trajectories from the Eulerian velocity field.
- **Metric**: Mean trajectory error

### Leaderboard

| Method | Forecasting | Super-Res (8x) | Sparse (1%) | Lagrangian |
|--------|:----------:|:--------------:|:-----------:|:----------:|
| Simple CNN | *TBD* | — | — | — |
| **Your method** | — | — | — | — |

**Submit your results** via [Pull Request](https://github.com/Ali-Rahimi-Khojasteh/cylinder-wake-re3900/pulls)!

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Explore Eulerian](notebooks/01_explore_eulerian.ipynb) | Load, visualize, and inspect Eulerian fields |
| 02 | [Explore Lagrangian](notebooks/02_explore_lagrangian.ipynb) | Particle trajectories, PDFs, 3D scatter |
| 03 | [Baseline Forecasting](notebooks/03_baseline_forecasting.ipynb) | Simple CNN for temporal prediction |
| 04 | [Sparse Reconstruction](notebooks/04_sparse_reconstruction.ipynb) | Lagrangian→Eulerian with varying sparsity |

## Installation

```bash
# Minimal (NumPy + HDF5 only)
pip install cylinderwake3900

# With PyTorch
pip install cylinderwake3900[torch]

# With visualization (matplotlib + plotly)
pip install cylinderwake3900[viz]

# Everything
pip install cylinderwake3900[all]

# Development
git clone https://github.com/Ali-Rahimi-Khojasteh/cylinder-wake-re3900.git
cd cylinder-wake-re3900
pip install -e ".[dev,all]"
```

## Data Source

The raw data is hosted on **Recherche Data Gouv** (French national research data repository):

- **DOI**: [10.15454/GLNRHK](https://doi.org/10.15454/GLNRHK)
- **Format**: `.txt` files in `.zip` archives
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

This repository provides Python tools to automatically download, convert to HDF5, and load the data for ML workflows.

## Simulation Details

| Parameter | Value |
|-----------|-------|
| Reynolds number | 3900 (based on cylinder diameter D and free-stream velocity U∞) |
| Geometry | Smooth circular cylinder (immersed boundary method) |
| Solver | [Incompact3d](https://github.com/xcompact3d/Incompact3d) |
| Simulation type | DNS (Direct Numerical Simulation) |
| Full domain | 20D × 20D × 6D, 1537 × 1025 × 256 grid points |
| Spatial scheme | 6th-order compact finite differences |
| Temporal scheme | 3rd-order Adams-Bashforth |
| DNS time step | 0.00075 D/U∞ (6667 steps per vortex shedding) |
| Finest grid spacing | Δy_min = 0.00563D (near cylinder center) |
| Particle transport | 4th-order Runge-Kutta + trilinear interpolation |
| Particles | ~200,000 synthetic tracers (~100k per sub-domain) |
| Boundary conditions | Inflow/outflow (x), free-slip (y), periodic (z) |

## Citation

If you use this dataset, please cite:

```bibtex
@article{khojasteh2021lagrangian,
  title     = {Lagrangian and {Eulerian} dataset of the wake downstream of a
               smooth cylinder at a {Reynolds} number equal to 3900},
  author    = {Khojasteh, Ali Rahimi and Laizet, Sylvain and
               Heitz, Dominique and Yang, Yang},
  journal   = {Data in Brief},
  volume    = {40},
  pages     = {107725},
  year      = {2021},
  publisher = {Elsevier},
  doi       = {10.1016/j.dib.2021.107725},
}
```

## Contributing

We welcome contributions! In particular:

- **Benchmark results**: Run your model on our tasks and submit results via PR
- **New tasks**: Propose additional benchmark challenges
- **Data loaders**: Add support for TensorFlow, JAX, or other frameworks
- **Visualizations**: Improve or add interactive visualizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

- **Data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT](LICENSE)

## Related Resources

- [Incompact3d](https://github.com/xcompact3d/Incompact3d) — The DNS solver used to generate this data
- [PDEBench](https://github.com/pdebench/PDEBench) — Scientific ML benchmark suite
- [AirfRANS](https://github.com/Extrality/airfrans_lib) — Airfoil RANS dataset
- [CFDBench](https://github.com/luo-yining/CFDBench) — CFD benchmark for ML
