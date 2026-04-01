"""
Visualization utilities for the CylinderWake3900 dataset.

Provides publication-quality and quick-inspection plots for:
  - 2D/3D velocity fields (contours, quiver, streamlines)
  - Vorticity fields
  - Lagrangian particle trajectories
  - Animated sequences

Dependencies: matplotlib, numpy. Optional: plotly (for 3D interactive).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install: pip install matplotlib")


# ── Velocity field ──────────────────────────────────────────────────

def plot_velocity_field(
    velocity: Any,
    grid: Optional[Dict[str, Any]] = None,
    component: str = "magnitude",
    slice_axis: str = "z",
    slice_index: Optional[int] = None,
    ax: Optional[Any] = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Any:
    """
    Plot a 2D slice of the velocity field.

    Parameters
    ----------
    velocity : array-like of shape (3, Nx, Ny, Nz) or (Nx, Ny, Nz)
        Velocity field. If 3 components, select via `component`.
    grid : dict, optional
        {"x": array, "y": array, "z": array} for axis labels.
    component : {"ux", "uy", "uz", "magnitude"}
        Which component to plot.
    slice_axis : {"x", "y", "z"}
        Axis along which to slice.
    slice_index : int, optional
        Index along slice_axis. Default: middle.
    """
    _check_matplotlib()

    vel = np.asarray(velocity)

    # Extract component
    if vel.ndim == 4:
        if component == "magnitude":
            field = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        elif component in ("ux", "u"):
            field = vel[0]
        elif component in ("uy", "v"):
            field = vel[1]
        elif component in ("uz", "w"):
            field = vel[2]
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        field = vel

    # Slice
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map[slice_axis]

    if slice_index is None:
        slice_index = field.shape[ax_idx] // 2

    slices = [slice(None)] * 3
    slices[ax_idx] = slice_index
    field_2d = field[tuple(slices)]

    # Determine axis labels
    remaining = [k for k in ["x", "y", "z"] if k != slice_axis]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if grid and remaining[0] in grid and remaining[1] in grid:
        x = np.asarray(grid[remaining[0]])
        y = np.asarray(grid[remaining[1]])
        im = ax.pcolormesh(x, y, field_2d.T, cmap=cmap, shading="auto",
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel(f"{remaining[0]} / D")
        ax.set_ylabel(f"{remaining[1]} / D")
    else:
        im = ax.imshow(field_2d.T, origin="lower", cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax)

    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)

    if title is None:
        title = f"Velocity ({component}), {slice_axis}={slice_index}"
    ax.set_title(title)
    ax.set_aspect("equal")

    return fig


# ── Vorticity ───────────────────────────────────────────────────────

def compute_vorticity(
    velocity: Any,
    grid: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute vorticity ω = ∇ × u from a 3D velocity field.

    Parameters
    ----------
    velocity : array of shape (3, Nx, Ny, Nz)
    grid : dict with "x", "y", "z" arrays for physical spacing.

    Returns
    -------
    vorticity : array of shape (3, Nx, Ny, Nz)
    """
    vel = np.asarray(velocity, dtype=np.float64)
    ux, uy, uz = vel[0], vel[1], vel[2]

    if grid:
        dx = np.gradient(np.asarray(grid["x"]))
        dy = np.gradient(np.asarray(grid["y"]))
        dz = np.gradient(np.asarray(grid["z"]))
        # Use actual grid spacing
        dwdy = np.gradient(uz, dy, axis=1)
        dvdz = np.gradient(uy, dz, axis=2)
        dudz = np.gradient(ux, dz, axis=2)
        dwdx = np.gradient(uz, dx, axis=0)
        dvdx = np.gradient(uy, dx, axis=0)
        dudy = np.gradient(ux, dy, axis=1)
    else:
        dwdy = np.gradient(uz, axis=1)
        dvdz = np.gradient(uy, axis=2)
        dudz = np.gradient(ux, axis=2)
        dwdx = np.gradient(uz, axis=0)
        dvdx = np.gradient(uy, axis=0)
        dudy = np.gradient(ux, axis=1)

    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy

    return np.stack([omega_x, omega_y, omega_z], axis=0).astype(np.float32)


def plot_vorticity(
    velocity: Any,
    grid: Optional[Dict[str, Any]] = None,
    slice_axis: str = "z",
    slice_index: Optional[int] = None,
    component: str = "z",
    ax: Optional[Any] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> Any:
    """Plot a 2D slice of the vorticity field."""
    _check_matplotlib()

    omega = compute_vorticity(velocity, grid)
    comp_map = {"x": 0, "y": 1, "z": 2}
    omega_comp = omega[comp_map[component]]

    return plot_velocity_field(
        omega_comp, grid=grid,
        slice_axis=slice_axis, slice_index=slice_index,
        ax=ax, cmap=cmap, figsize=figsize,
        title=f"Vorticity ω_{component}, {slice_axis}-slice",
        **kwargs,
    )


# ── Lagrangian trajectories ────────────────────────────────────────

def plot_trajectories(
    positions: Any,
    velocities: Optional[Any] = None,
    max_particles: int = 500,
    color_by: str = "velocity",
    ax: Optional[Any] = None,
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.3,
    s: float = 1.0,
    cmap: str = "viridis",
    title: Optional[str] = None,
    projection: str = "2d",
) -> Any:
    """
    Plot Lagrangian particle positions (scatter) or trajectories.

    Parameters
    ----------
    positions : array of shape (N_particles, 3) or (N_times, N_particles, 3)
    velocities : array, optional, same shape as positions.
    max_particles : int
        Subsample if too many particles.
    color_by : {"velocity", "x", "y", "z", "time", None}
    projection : {"2d", "3d"}
    """
    _check_matplotlib()

    pos = np.asarray(positions)

    # Handle single snapshot vs trajectory
    if pos.ndim == 2:
        # Single snapshot: (N, 3)
        pos = pos[:max_particles]
        vel = np.asarray(velocities)[:max_particles] if velocities is not None else None

        if color_by == "velocity" and vel is not None:
            c = np.linalg.norm(vel, axis=1)
        elif color_by in ("x", "y", "z"):
            c = pos[:, {"x": 0, "y": 1, "z": 2}[color_by]]
        else:
            c = None

        if projection == "3d":
            fig = plt.figure(figsize=figsize)
            ax3d = fig.add_subplot(111, projection="3d")
            sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c,
                             s=s, alpha=alpha, cmap=cmap)
            ax3d.set_xlabel("x / D")
            ax3d.set_ylabel("y / D")
            ax3d.set_zlabel("z / D")
            if c is not None:
                plt.colorbar(sc, shrink=0.6, label=color_by)
            ax3d.set_title(title or "Lagrangian particles")
            return fig
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=c, s=s,
                           alpha=alpha, cmap=cmap)
            ax.set_xlabel("x / D")
            ax.set_ylabel("y / D")
            if c is not None:
                plt.colorbar(sc, ax=ax, label=color_by)
            ax.set_title(title or "Lagrangian particles (x-y plane)")
            ax.set_aspect("equal")
            return ax.figure

    elif pos.ndim == 3:
        # Trajectory: (T, N, 3)
        n_t, n_p, _ = pos.shape
        idx = np.random.choice(n_p, min(max_particles, n_p), replace=False)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for i in idx:
            traj = pos[:, i, :]
            ax.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=0.5)

        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
        ax.set_title(title or f"Lagrangian trajectories ({len(idx)} particles)")
        ax.set_aspect("equal")
        return ax.figure


# ── Interactive 3D (Plotly) ─────────────────────────────────────────

def plot_3d_interactive(
    velocity: Any,
    grid: Optional[Dict[str, Any]] = None,
    slice_axis: str = "z",
    slice_index: Optional[int] = None,
    component: str = "magnitude",
) -> Any:
    """
    Create an interactive 3D velocity field visualization using Plotly.

    Returns a plotly Figure object — call .show() or save to HTML.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install: pip install plotly")

    vel = np.asarray(velocity)
    if component == "magnitude":
        field = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    else:
        comp_map = {"ux": 0, "uy": 1, "uz": 2}
        field = vel[comp_map[component]]

    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map[slice_axis]
    if slice_index is None:
        slice_index = field.shape[ax_idx] // 2

    slices = [slice(None)] * 3
    slices[ax_idx] = slice_index
    field_2d = field[tuple(slices)]

    remaining = [k for k in ["x", "y", "z"] if k != slice_axis]

    fig = go.Figure(data=go.Heatmap(
        z=field_2d.T,
        colorscale="RdBu_r",
        colorbar=dict(title=f"Velocity ({component})"),
    ))

    fig.update_layout(
        title=f"Velocity field ({component}), {slice_axis}={slice_index}",
        xaxis_title=f"{remaining[0]} / D",
        yaxis_title=f"{remaining[1]} / D",
        width=800,
        height=500,
    )

    return fig
