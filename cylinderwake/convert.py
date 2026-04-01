"""
Convert raw .txt/.zip data from the INRAE repository into HDF5 format.

The original data uses:
  - Eulerian 3D fields: flat .txt files with values stored as 1D vectors,
    to be reshaped with three nested loops in (x, y, z) order (Fortran order).
    This is the standard Incompact3d output format.
  - Lagrangian data: text files with columns for particle positions/velocities.
  - Grid: separate .txt file with coordinates.

This module converts everything into well-structured HDF5 with metadata,
making the data immediately usable for ML frameworks.

Data naming convention (from Fig. 3 of the paper):
  Eulerian:  U_sub_domain_1/ux001.txt, ux002.txt, ... (similarly uy, uz, pp)
  Lagrangian: Lagrangian_tracks_sub_domain_1/tracks_001.txt, ...
"""

from __future__ import annotations

import os
import re
import glob
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import h5py
import numpy as np

from .download import get_data_dir


# ── Configuration ───────────────────────────────────────────────────
# From Table 2 of Khojasteh et al. (2021), Data in Brief 40, 107725

GRID_DIMS = {
    # Sub-domain 1: (4-14)D x (6-14)D x 6D = 10D x 8D x 6D
    # 100 Eulerian snapshots, saved every 10 DNS steps (dt_eff = 0.0075 D/U_inf)
    # Snapshot size: 4.8 GB each
    1: {"nx": 769, "ny": 777, "nz": 256,
        "x_range": "(4-14)D", "y_range": "(6-14)D", "z_range": "6D",
        "dt": 0.0075, "n_snapshots": 100},

    # Sub-domain 2: (4-8)D x (9-11)D x (2-4)D = 4D x 2D x 2D
    # 1000 Eulerian snapshots, saved every DNS step (dt = 0.00075 D/U_inf)
    # Snapshot size: 256 MB each
    2: {"nx": 308, "ny": 328, "nz": 87,
        "x_range": "(4-8)D", "y_range": "(9-11)D", "z_range": "(2-4)D",
        "dt": 0.00075, "n_snapshots": 1000},
}

# Map subdomain names used in the Dataset API to numeric IDs
SUBDOMAIN_NAME_MAP = {"near": 1, "far": 2}


# ── Grid coordinates ───────────────────────────────────────────────
#
# Incompact3d uses a Cartesian grid with:
#   - Uniform spacing in x (streamwise) and z (spanwise/periodic)
#   - Non-uniform (stretched) spacing in y (vertical), refined near
#     the cylinder centre at y = 10D  (Δy_min ≈ 0.00563D)
#
# The stretching follows Laizet & Lamballais (2009), JCP 228(16):5989,
# with parameter beta ≈ 2.0 and istret = 1 (symmetric about centre).
#
# The exact 1D coordinate arrays for each sub-domain are bundled with
# this package as  cylinderwake/grid_coordinates.npz  (17 KB), extracted
# from the original simulation grid files provided by the authors.
#
# Coordinate frame:
#   Sub-domain 1 ("near"): x ∈ [4D, 14D], y ∈ [0, 8D] (local), z ∈ [0, 6D]
#   Sub-domain 2 ("far"):  x ∈ [4D, 8D],  y ∈ [3D, 5D] (SD1-local), z ∈ [2D, 4D]
#   y is in a local frame where y = 0 corresponds to y_global = 6D.

_GRID_NPZ = Path(__file__).parent / "grid_coordinates.npz"

_SUBDOMAIN_GRID_KEYS = {
    1: {"x": "x_near", "y": "y_near", "z": "z_near"},
    2: {"x": "x_far",  "y": "y_far",  "z": "z_far"},
}


def load_grid(sd_num: int) -> Dict[str, np.ndarray]:
    """
    Load the exact DNS grid coordinates for a sub-domain.

    The coordinates are the original Incompact3d grid points extracted
    from the simulation.  y is non-uniform (stretched), x and z are
    uniform.

    Parameters
    ----------
    sd_num : int
        Sub-domain number (1 or 2).

    Returns
    -------
    dict with keys "x", "y", "z" — each a 1D float64 array in
    units of cylinder diameter D.
    """
    if not _GRID_NPZ.exists():
        raise FileNotFoundError(
            f"Grid coordinate file not found at {_GRID_NPZ}. "
            f"Reinstall the cylinderwake package."
        )
    data = np.load(_GRID_NPZ)
    keys = _SUBDOMAIN_GRID_KEYS[sd_num]
    return {
        "x": data[keys["x"]].astype(np.float64),
        "y": data[keys["y"]].astype(np.float64),
        "z": data[keys["z"]].astype(np.float64),
    }


# ── Eulerian snapshot parsing ───────────────────────────────────────

def parse_eulerian_snapshot(
    filepath: Path,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """
    Parse a single Eulerian field snapshot from a .txt file.

    The Incompact3d output stores 3D fields as 1D vectors.
    They must be reshaped using three nested loops in (x, y, z) order,
    which corresponds to Fortran column-major ordering.

    Parameters
    ----------
    filepath : Path
        Path to the .txt file containing one scalar field.
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    np.ndarray of shape (nx, ny, nz)
    """
    data = np.loadtxt(filepath, dtype=np.float64)
    # Incompact3d uses Fortran ordering (column-major)
    field = data.reshape((nx, ny, nz), order="F")
    return field.astype(np.float32)


# ── Lagrangian snapshot parsing ─────────────────────────────────────

def parse_lagrangian_snapshot(filepath: Path) -> Dict[str, np.ndarray]:
    """
    Parse a single Lagrangian particle snapshot from a .txt file.

    Expected format: each row is one particle with columns:
      [x, y, z, u, v, w]   (position + velocity)
    or
      [particle_id, x, y, z, u, v, w]

    UPDATE based on your actual file format.
    """
    data = np.loadtxt(filepath, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]

    if n_cols >= 7:
        # Format: id, x, y, z, u, v, w
        return {
            "positions": data[:, 1:4].astype(np.float32),
            "velocities": data[:, 4:7].astype(np.float32),
        }
    elif n_cols == 6:
        # Format: x, y, z, u, v, w
        return {
            "positions": data[:, 0:3].astype(np.float32),
            "velocities": data[:, 3:6].astype(np.float32),
        }
    elif n_cols == 3:
        # Format: x, y, z (positions only)
        return {
            "positions": data[:, 0:3].astype(np.float32),
            "velocities": np.zeros_like(data[:, 0:3], dtype=np.float32),
        }
    else:
        raise ValueError(
            f"Unexpected number of columns ({n_cols}) in {filepath}. "
            f"Expected 3, 6, or 7. Update parse_lagrangian_snapshot()."
        )


# ── Main conversion ─────────────────────────────────────────────────

def convert_raw_to_hdf5(
    root: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Convert all raw .txt data to HDF5.

    Creates:
      - hdf5/eulerian_near.h5
      - hdf5/eulerian_far.h5
      - hdf5/lagrangian_near.h5
      - hdf5/lagrangian_far.h5

    Parameters
    ----------
    root : Path, optional
        Data root directory.
    force : bool
        If True, overwrite existing HDF5 files.
    """
    data_dir = get_data_dir(root)
    raw_dir = data_dir / "raw"
    hdf5_dir = data_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_dir}. "
            f"Run `cylinderwake-download` first."
        )

    # Convert Eulerian data for each sub-domain
    for subdomain in ["near", "far"]:
        sd_num = SUBDOMAIN_NAME_MAP[subdomain]
        print(f"  📐 Loading grid coordinates for sub-domain {sd_num} ({subdomain})")
        grid = load_grid(sd_num)
        _convert_eulerian(raw_dir, hdf5_dir, subdomain, grid, force)
        _convert_lagrangian(raw_dir, hdf5_dir, subdomain, force)

    print(f"\n✅ HDF5 conversion complete! Files in: {hdf5_dir}")


def _convert_eulerian(
    raw_dir: Path,
    hdf5_dir: Path,
    subdomain: str,
    grid: Optional[Dict],
    force: bool,
) -> None:
    """Convert Eulerian raw data for one sub-domain."""
    out_path = hdf5_dir / f"eulerian_{subdomain}.h5"

    if out_path.exists() and not force:
        print(f"  ✓ {out_path.name} already exists (use force=True to overwrite)")
        return

    sd_num = SUBDOMAIN_NAME_MAP[subdomain]
    dims = GRID_DIMS[sd_num]
    nx, ny, nz = dims["nx"], dims["ny"], dims["nz"]
    dt = dims["dt"]

    # Incompact3d output convention:
    #   U = streamwise (ux), V = vertical (uy), W = spanwise (uz), P = pressure
    # Files are extracted from zip into directories like:
    #   raw/U_sub_domain_1 (1_25)/ux001.txt, ux002.txt, ...
    # We search recursively for ux*.txt, uy*.txt, uz*.txt, pp*.txt

    # Find all velocity component files across all chunk directories
    ux_files = sorted(raw_dir.rglob(f"*sub_domain_{sd_num}*/ux*.txt"))
    uy_files = sorted(raw_dir.rglob(f"*sub_domain_{sd_num}*/uy*.txt"))
    uz_files = sorted(raw_dir.rglob(f"*sub_domain_{sd_num}*/uz*.txt"))

    # Alternative: files might be named u001.txt inside U_sub_domain_X directories
    if not ux_files:
        ux_files = sorted(raw_dir.rglob(f"U_sub_domain_{sd_num}*/u*.txt"))
        uy_files = sorted(raw_dir.rglob(f"V_sub_domain_{sd_num}*/v*.txt"))
        uz_files = sorted(raw_dir.rglob(f"W_sub_domain_{sd_num}*/w*.txt"))

    # Pressure (only available for sub-domain 2)
    p_files = sorted(raw_dir.rglob(f"P_sub_domain_{sd_num}*/p*.txt"))
    if not p_files:
        p_files = sorted(raw_dir.rglob(f"*sub_domain_{sd_num}*/pp*.txt"))

    n_snapshots = min(len(ux_files), len(uy_files), len(uz_files))

    if n_snapshots == 0:
        print(f"  ⚠ No Eulerian snapshots found for sub-domain {sd_num} ({subdomain})")
        print(f"     Searched in: {raw_dir}")
        print(f"     Expected patterns: ux*.txt, uy*.txt, uz*.txt in *sub_domain_{sd_num}*/")
        return

    print(f"  📊 Converting {n_snapshots} Eulerian snapshots (sub-domain {sd_num}, '{subdomain}')")
    print(f"     Grid: {nx} × {ny} × {nz}, dt = {dt} D/U∞")

    with h5py.File(out_path, "w") as f:
        # Metadata
        f.attrs["reynolds_number"] = 3900
        f.attrs["solver"] = "Incompact3d"
        f.attrs["simulation_type"] = "DNS"
        f.attrs["subdomain"] = subdomain
        f.attrs["subdomain_number"] = sd_num
        f.attrs["nx"] = nx
        f.attrs["ny"] = ny
        f.attrs["nz"] = nz
        f.attrs["dt"] = dt
        f.attrs["n_snapshots"] = n_snapshots
        f.attrs["x_range"] = dims["x_range"]
        f.attrs["y_range"] = dims["y_range"]
        f.attrs["z_range"] = dims["z_range"]

        # Grid
        if grid:
            g = f.create_group("grid")
            for key, val in grid.items():
                g.create_dataset(key, data=val)

        # Snapshots
        for i in range(n_snapshots):
            grp = f.create_group(f"snapshot_{i:06d}")

            ux = parse_eulerian_snapshot(ux_files[i], nx, ny, nz)
            uy = parse_eulerian_snapshot(uy_files[i], nx, ny, nz)
            uz = parse_eulerian_snapshot(uz_files[i], nx, ny, nz)

            grp.create_dataset("ux", data=ux, compression="gzip", compression_opts=4)
            grp.create_dataset("uy", data=uy, compression="gzip", compression_opts=4)
            grp.create_dataset("uz", data=uz, compression="gzip", compression_opts=4)

            if i < len(p_files):
                p = parse_eulerian_snapshot(p_files[i], nx, ny, nz)
                grp.create_dataset("pressure", data=p, compression="gzip", compression_opts=4)

            # Physical time: snapshot_index * dt
            grp.attrs["time"] = float(i) * dt

            if (i + 1) % 50 == 0 or i == n_snapshots - 1:
                print(f"     [{i+1}/{n_snapshots}]")

    print(f"  ✓ Saved {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")


def _convert_lagrangian(
    raw_dir: Path,
    hdf5_dir: Path,
    subdomain: str,
    force: bool,
) -> None:
    """Convert Lagrangian raw data for one sub-domain."""
    out_path = hdf5_dir / f"lagrangian_{subdomain}.h5"

    if out_path.exists() and not force:
        print(f"  ✓ {out_path.name} already exists")
        return

    sd_num = SUBDOMAIN_NAME_MAP[subdomain]
    dt = GRID_DIMS[sd_num]["dt"]

    # Find Lagrangian files — search for tracks/trajectory files
    lag_files = sorted(raw_dir.rglob(f"Lagrangian*sub_domain_{sd_num}*/*.txt"))
    if not lag_files:
        lag_files = sorted(raw_dir.rglob(f"*lagrangian*{sd_num}*/*.txt"))
    if not lag_files:
        lag_files = sorted(raw_dir.rglob(f"*track*{sd_num}*/*.txt"))

    if not lag_files:
        print(f"  ⚠ No Lagrangian files found for sub-domain {sd_num} ({subdomain})")
        return

    print(f"  🔵 Converting {len(lag_files)} Lagrangian snapshots (sub-domain {sd_num})")

    with h5py.File(out_path, "w") as f:
        f.attrs["reynolds_number"] = 3900
        f.attrs["solver"] = "Incompact3d"
        f.attrs["subdomain"] = subdomain
        f.attrs["subdomain_number"] = sd_num
        f.attrs["dt"] = dt
        f.attrs["n_snapshots"] = len(lag_files)

        for i, lag_file in enumerate(lag_files):
            data = parse_lagrangian_snapshot(lag_file)
            grp = f.create_group(f"snapshot_{i:06d}")
            grp.create_dataset("positions", data=data["positions"], compression="gzip")
            grp.create_dataset("velocities", data=data["velocities"], compression="gzip")
            grp.attrs["time"] = float(i) * dt
            grp.attrs["n_particles"] = data["positions"].shape[0]

            if (i + 1) % 50 == 0 or i == len(lag_files) - 1:
                print(f"     [{i+1}/{len(lag_files)}]")

    print(f"  ✓ Saved {out_path.name}")


# ── CLI ─────────────────────────────────────────────────────────────

def cli_convert():
    """CLI: cylinderwake-convert"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw CylinderWake3900 data to HDF5"
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert_raw_to_hdf5(
        root=Path(args.root) if args.root else None,
        force=args.force,
    )


if __name__ == "__main__":
    cli_convert()
