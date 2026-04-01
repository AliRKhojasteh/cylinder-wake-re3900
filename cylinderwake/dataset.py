"""
PyTorch Dataset classes for the CylinderWake3900 dataset.

Supports both Eulerian (velocity + pressure fields) and Lagrangian (particle trajectories)
data, with automatic download and HDF5 caching.

Designed for:
  - Physics-Informed Neural Networks (PINNs)
  - Neural Operators (FNO, DeepONet)
  - Temporal forecasting / super-resolution
  - Data assimilation from sparse Lagrangian observations
  - Particle tracking algorithm benchmarking

Compatible with: PyTorch, TensorFlow (via .numpy()), JAX, NumPy workflows, and AI agents.
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Tuple, Union

import h5py
import numpy as np

# ── Optional PyTorch support ────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _TorchDataset:
        """Fallback base when PyTorch is not installed."""
        pass

from .download import download_dataset, get_data_dir
from .convert import convert_raw_to_hdf5


# ── Constants ───────────────────────────────────────────────────────
REYNOLDS_NUMBER = 3900
SOLVER = "Incompact3d"
SIMULATION_TYPE = "DNS"

SUBDOMAINS = {
    "near": {
        "description": "Sub-domain 1: large near-wake region",
        "dimensions": "10D x 8D x 6D",
        "x_range": "(4–14)D downstream",
        "grid": "769 x 777 x 256",
        "n_snapshots": 100,
        "dt": "0.0075 D/U∞ (every 10 DNS steps)",
    },
    "far": {
        "description": "Sub-domain 2: small high-temporal-resolution region",
        "dimensions": "4D x 2D x 2D",
        "x_range": "(4–8)D downstream",
        "grid": "308 x 328 x 87",
        "n_snapshots": 1000,
        "dt": "0.00075 D/U∞ (every DNS step)",
    },
}

FIELD_TYPES = ["eulerian", "lagrangian"]


# ── Helper ──────────────────────────────────────────────────────────
def _to_tensor(arr: np.ndarray) -> Any:
    """Convert numpy array to torch.Tensor if PyTorch is available."""
    if HAS_TORCH:
        return torch.from_numpy(arr)
    return arr


# ── Unified entry point ─────────────────────────────────────────────
class CylinderWake3900:
    """
    Unified entry point for the CylinderWake3900 dataset.

    Parameters
    ----------
    field : {"eulerian", "lagrangian"}
        Which data representation to load.
    subdomain : {"near", "far"}
        Spatial sub-domain.
    root : str or Path, optional
        Root directory for data storage. Defaults to ~/.cylinderwake3900/
    download : bool
        If True, download raw data from INRAE and convert to HDF5 if needed.
    split : {"train", "val", "test", None}
        Dataset split. Default None returns all snapshots.
        Canonical splits: train=70%, val=15%, test=15% (temporal ordering).
    transform : callable, optional
        Optional transform applied to each sample dict.
    normalize : bool
        If True, normalize velocity/pressure fields to zero mean, unit variance.

    Returns
    -------
    EulerianDataset or LagrangianDataset
        A PyTorch-compatible Dataset object.

    Examples
    --------
    >>> ds = CylinderWake3900("eulerian", "near", download=True)
    >>> sample = ds[0]
    >>> print(sample["velocity"].shape)  # (3, Nx, Ny, Nz)

    >>> ds = CylinderWake3900("lagrangian", "near", download=True)
    >>> sample = ds[0]
    >>> print(sample["positions"].shape)  # (N_particles, 3)
    """

    def __new__(
        cls,
        field: Literal["eulerian", "lagrangian"] = "eulerian",
        subdomain: Literal["near", "far"] = "near",
        root: Optional[Union[str, Path]] = None,
        download: bool = True,
        split: Optional[Literal["train", "val", "test"]] = None,
        transform: Optional[Any] = None,
        normalize: bool = False,
    ):
        if field == "eulerian":
            return EulerianDataset(
                subdomain=subdomain,
                root=root,
                download=download,
                split=split,
                transform=transform,
                normalize=normalize,
            )
        elif field == "lagrangian":
            return LagrangianDataset(
                subdomain=subdomain,
                root=root,
                download=download,
                split=split,
                transform=transform,
                normalize=normalize,
            )
        else:
            raise ValueError(f"field must be 'eulerian' or 'lagrangian', got '{field}'")


# ── Base Dataset ────────────────────────────────────────────────────
class _BaseDataset(_TorchDataset):
    """Shared logic for Eulerian and Lagrangian datasets."""

    def __init__(
        self,
        subdomain: str,
        root: Optional[Union[str, Path]],
        download: bool,
        split: Optional[str],
        transform: Optional[Any],
        normalize: bool,
    ):
        self.subdomain = subdomain
        self.root = Path(root) if root else get_data_dir()
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self._stats: Optional[Dict[str, Any]] = None

        if download:
            download_dataset(root=self.root)
            convert_raw_to_hdf5(root=self.root)

        self._hdf5_path = self._get_hdf5_path()
        if not self._hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found at {self._hdf5_path}. "
                f"Set download=True or run: cylinderwake-download"
            )

        # Determine indices for this split
        with h5py.File(self._hdf5_path, "r") as f:
            n_total = f.attrs.get("n_snapshots", len(f.keys()))

        self._indices = self._get_split_indices(n_total, split)

    def _get_hdf5_path(self) -> Path:
        raise NotImplementedError

    @staticmethod
    def _get_split_indices(
        n_total: int, split: Optional[str]
    ) -> np.ndarray:
        """Canonical temporal splits: 70/15/15."""
        all_idx = np.arange(n_total)
        if split is None:
            return all_idx
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        if split == "train":
            return all_idx[:n_train]
        elif split == "val":
            return all_idx[n_train : n_train + n_val]
        elif split == "test":
            return all_idx[n_train + n_val :]
        else:
            raise ValueError(f"split must be train/val/test/None, got '{split}'")

    def __len__(self) -> int:
        return len(self._indices)

    def metadata(self) -> Dict[str, Any]:
        """Return dataset metadata as a dict (useful for AI agents)."""
        with h5py.File(self._hdf5_path, "r") as f:
            meta = dict(f.attrs)
        meta.update({
            "reynolds_number": REYNOLDS_NUMBER,
            "solver": SOLVER,
            "simulation_type": SIMULATION_TYPE,
            "subdomain": self.subdomain,
            "split": self.split,
            "n_samples": len(self),
            "hdf5_path": str(self._hdf5_path),
        })
        return meta

    def to_json(self) -> str:
        """Serialize metadata to JSON (for AI agent consumption)."""
        meta = self.metadata()
        # Convert numpy types to native Python
        for k, v in meta.items():
            if isinstance(v, (np.integer,)):
                meta[k] = int(v)
            elif isinstance(v, (np.floating,)):
                meta[k] = float(v)
            elif isinstance(v, np.ndarray):
                meta[k] = v.tolist()
        return json.dumps(meta, indent=2)


# ── Eulerian Dataset ────────────────────────────────────────────────
class EulerianDataset(_BaseDataset):
    """
    Eulerian velocity + pressure field snapshots.

    Each sample is a dictionary:
        {
            "velocity": Tensor of shape (3, Nx, Ny, Nz),   # [ux, uy, uz]
            "pressure": Tensor of shape (1, Nx, Ny, Nz),
            "time":     float,
            "index":    int,
            "grid":     {
                "x": Tensor (Nx,),
                "y": Tensor (Ny,),
                "z": Tensor (Nz,),
            },
        }
    """

    def _get_hdf5_path(self) -> Path:
        return self.root / "hdf5" / f"eulerian_{self.subdomain}.h5"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]

        with h5py.File(self._hdf5_path, "r") as f:
            snap = f[f"snapshot_{real_idx:06d}"]
            velocity = np.stack([
                snap["ux"][()],
                snap["uy"][()],
                snap["uz"][()],
            ], axis=0).astype(np.float32)

            pressure = snap["pressure"][()][np.newaxis].astype(np.float32)
            time_val = float(snap.attrs.get("time", real_idx))

            grid = {}
            if "grid" in f:
                grid = {
                    "x": _to_tensor(f["grid"]["x"][()].astype(np.float32)),
                    "y": _to_tensor(f["grid"]["y"][()].astype(np.float32)),
                    "z": _to_tensor(f["grid"]["z"][()].astype(np.float32)),
                }

        if self.normalize:
            velocity, pressure = self._normalize_fields(velocity, pressure)

        sample = {
            "velocity": _to_tensor(velocity),
            "pressure": _to_tensor(pressure),
            "time": time_val,
            "index": int(real_idx),
            "grid": grid,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _normalize_fields(
        self, velocity: np.ndarray, pressure: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score normalization using dataset-wide statistics."""
        if self._stats is None:
            stats_path = self._hdf5_path.parent / f"stats_{self.subdomain}.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    self._stats = json.load(f)
            else:
                # Compute on first call (slow, but only once)
                self._stats = self._compute_stats()
                with open(stats_path, "w") as f:
                    json.dump(self._stats, f)

        vel_mean = np.array(self._stats["velocity_mean"]).reshape(3, 1, 1, 1)
        vel_std = np.array(self._stats["velocity_std"]).reshape(3, 1, 1, 1)
        p_mean = self._stats["pressure_mean"]
        p_std = self._stats["pressure_std"]

        velocity = (velocity - vel_mean) / (vel_std + 1e-8)
        pressure = (pressure - p_mean) / (p_std + 1e-8)
        return velocity, pressure

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute mean/std over all snapshots."""
        vel_sum = np.zeros(3)
        vel_sq_sum = np.zeros(3)
        p_sum = 0.0
        p_sq_sum = 0.0
        n = 0

        with h5py.File(self._hdf5_path, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("snapshot_"):
                    continue
                snap = f[key]
                for i, comp in enumerate(["ux", "uy", "uz"]):
                    data = snap[comp][()]
                    vel_sum[i] += data.sum()
                    vel_sq_sum[i] += (data ** 2).sum()
                p = snap["pressure"][()]
                p_sum += p.sum()
                p_sq_sum += (p ** 2).sum()
                n += data.size

        vel_mean = vel_sum / n
        vel_std = np.sqrt(vel_sq_sum / n - vel_mean ** 2)
        p_mean = p_sum / n
        p_std = np.sqrt(p_sq_sum / n - p_mean ** 2)

        return {
            "velocity_mean": vel_mean.tolist(),
            "velocity_std": vel_std.tolist(),
            "pressure_mean": float(p_mean),
            "pressure_std": float(p_std),
        }

    def get_sequence(
        self, start: int, length: int
    ) -> Dict[str, Any]:
        """
        Load a contiguous temporal sequence (useful for forecasting tasks).

        Returns dict with tensors of shape (length, C, Nx, Ny, Nz).
        """
        samples = [self[start + i] for i in range(length)]
        velocity = np.stack([s["velocity"] if isinstance(s["velocity"], np.ndarray)
                            else s["velocity"].numpy() for s in samples])
        pressure = np.stack([s["pressure"] if isinstance(s["pressure"], np.ndarray)
                            else s["pressure"].numpy() for s in samples])
        times = [s["time"] for s in samples]

        return {
            "velocity": _to_tensor(velocity),
            "pressure": _to_tensor(pressure),
            "times": times,
        }


# ── Lagrangian Dataset ──────────────────────────────────────────────
class LagrangianDataset(_BaseDataset):
    """
    Lagrangian particle trajectory snapshots.

    Each sample is a dictionary:
        {
            "positions":  Tensor of shape (N_particles, 3),  # [x, y, z]
            "velocities": Tensor of shape (N_particles, 3),  # [u, v, w]
            "time":       float,
            "index":      int,
        }
    """

    def _get_hdf5_path(self) -> Path:
        return self.root / "hdf5" / f"lagrangian_{self.subdomain}.h5"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]

        with h5py.File(self._hdf5_path, "r") as f:
            snap = f[f"snapshot_{real_idx:06d}"]
            positions = snap["positions"][()].astype(np.float32)
            velocities = snap["velocities"][()].astype(np.float32)
            time_val = float(snap.attrs.get("time", real_idx))

        sample = {
            "positions": _to_tensor(positions),
            "velocities": _to_tensor(velocities),
            "time": time_val,
            "index": int(real_idx),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_full_trajectories(self) -> Dict[str, Any]:
        """
        Load all time steps and reconstruct full particle trajectories.

        Returns
        -------
        dict with:
            "positions":  (N_timesteps, N_particles, 3)
            "velocities": (N_timesteps, N_particles, 3)
            "times":      (N_timesteps,)
        """
        all_pos = []
        all_vel = []
        times = []

        with h5py.File(self._hdf5_path, "r") as f:
            for key in sorted(k for k in f.keys() if k.startswith("snapshot_")):
                snap = f[key]
                all_pos.append(snap["positions"][()].astype(np.float32))
                all_vel.append(snap["velocities"][()].astype(np.float32))
                times.append(float(snap.attrs.get("time", 0)))

        return {
            "positions": _to_tensor(np.stack(all_pos)),
            "velocities": _to_tensor(np.stack(all_vel)),
            "times": np.array(times),
        }
