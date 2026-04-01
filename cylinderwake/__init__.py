"""
CylinderWake3900 — ML-ready dataset of the wake downstream of a smooth cylinder at Re = 3900.

Provides Eulerian velocity/pressure fields and Lagrangian particle trajectories
from a Direct Numerical Simulation (DNS) using Incompact3d.

Quick start:
    >>> from cylinderwake import CylinderWake3900
    >>> ds = CylinderWake3900(field="eulerian", subdomain="near")
    >>> sample = ds[0]  # {'velocity': tensor, 'pressure': tensor, 'time': float}

Paper:  https://doi.org/10.1016/j.dib.2021.107725
Data:   https://doi.org/10.15454/GLNRHK
GitHub: https://github.com/AliRKhojasteh/cylinder-wake-re3900
"""

__version__ = "1.0.0"

from .dataset import CylinderWake3900, EulerianDataset, LagrangianDataset
from .download import download_dataset, get_data_dir
from .convert import convert_raw_to_hdf5
from .visualize import plot_velocity_field, plot_trajectories, plot_vorticity

__all__ = [
    "CylinderWake3900",
    "EulerianDataset",
    "LagrangianDataset",
    "download_dataset",
    "get_data_dir",
    "convert_raw_to_hdf5",
    "plot_velocity_field",
    "plot_trajectories",
    "plot_vorticity",
]
