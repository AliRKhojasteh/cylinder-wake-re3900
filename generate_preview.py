#!/usr/bin/env python3
"""
Generate a preview figure for the CylinderWake3900 repository.

Shows:
  (a) Computational domain with both sub-domains and cylinder location
  (b) Grid spacing Δy vs y — highlighting the non-uniform stretching
  (c) SD1 grid (x–y) detail with every Nth line
  (d) SD2 grid (x–y) detail with every Nth line

Requires only numpy + matplotlib (no data download needed).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from pathlib import Path

# ── Load bundled grid coordinates ─────────────────────────────────────
GRID_NPZ = Path(__file__).parent / "cylinderwake" / "grid_coordinates.npz"
d = np.load(GRID_NPZ)

x1, y1, z1 = d["x_near"], d["y_near"], d["z_near"]
x2, y2, z2 = d["x_far"],  d["y_far"],  d["z_far"]

# Global frame: y_global = y_local + 6D  (cylinder center at y_global = 10D)
Y_OFFSET = 6.0
y1g = y1 + Y_OFFSET
y2g = y2 + Y_OFFSET

# Full DNS domain
LX, LY, LZ = 20.0, 20.0, 6.0
CYL_X, CYL_Y = 5.0, 10.0  # cylinder center in global frame

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="white")

# Use gridspec for layout
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                      left=0.06, right=0.97, top=0.93, bottom=0.07)

# ── (a) Full domain overview ──────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0:2])

# Full domain
full_rect = Rectangle((0, 0), LX, LY, linewidth=1.5, edgecolor="#334155",
                       facecolor="#f8fafc", zorder=0)
ax0.add_patch(full_rect)

# SD1 (near-wake)
sd1_rect = Rectangle((x1[0], y1g[0]), x1[-1]-x1[0], y1g[-1]-y1g[0],
                      linewidth=2, edgecolor="#3b82f6", facecolor="#dbeafe",
                      alpha=0.6, label="Sub-domain 1 (near)", zorder=1)
ax0.add_patch(sd1_rect)

# SD2 (far-wake)
sd2_rect = Rectangle((x2[0], y2g[0]), x2[-1]-x2[0], y2g[-1]-y2g[0],
                      linewidth=2, edgecolor="#ef4444", facecolor="#fee2e2",
                      alpha=0.7, label="Sub-domain 2 (far)", zorder=2)
ax0.add_patch(sd2_rect)

# Cylinder
cyl = Circle((CYL_X, CYL_Y), 0.5, facecolor="#1e293b", edgecolor="#0f172a",
             linewidth=1.5, zorder=3)
ax0.add_patch(cyl)
ax0.annotate("Cylinder\n(D = 1)", (CYL_X, CYL_Y), fontsize=8,
             ha="center", va="center", color="white", fontweight="bold")

# Flow arrow
ax0.annotate("", xy=(3.5, 10), xytext=(0.5, 10),
             arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=2))
ax0.text(2.0, 10.5, r"$U_\infty$", fontsize=12, ha="center", color="#64748b")

# Annotations
ax0.text(9, 13.5, f"SD1: {len(x1)}×{len(y1)}×{len(z1)}\n10D × 8D × 6D\n100 snapshots",
         fontsize=8, color="#1d4ed8", ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#3b82f6", alpha=0.9))
ax0.text(6, 8.3, f"SD2: {len(x2)}×{len(y2)}×{len(z2)}\n4D × 2D × 2D\n1000 snapshots",
         fontsize=8, color="#dc2626", ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ef4444", alpha=0.9))

ax0.set_xlim(-0.5, LX + 0.5)
ax0.set_ylim(-0.5, LY + 0.5)
ax0.set_xlabel("x / D", fontsize=11)
ax0.set_ylabel("y / D", fontsize=11)
ax0.set_title("(a)  Computational Domain  (20D × 20D × 6D)", fontsize=12, fontweight="bold")
ax0.set_aspect("equal")
ax0.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax0.grid(True, alpha=0.15)

# ── (b) Grid spacing Δy ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])

dy1 = np.diff(y1)
y1_mid = 0.5 * (y1[:-1] + y1[1:])

ax1.plot(y1_mid + Y_OFFSET, dy1, "-", color="#3b82f6", linewidth=1.5,
         label=f"SD1 ({len(y1)} pts)")
ax1.axhline(dy1.min(), color="#3b82f6", ls="--", alpha=0.4, linewidth=0.8)
ax1.text(14.1, dy1.min() + 0.0003, f"Δy_min = {dy1.min():.5f}D",
         fontsize=8, color="#3b82f6")

# SD2 y-spacing
dy2 = np.diff(y2)
y2_mid = 0.5 * (y2[:-1] + y2[1:])
ax1.plot(y2_mid + Y_OFFSET, dy2, "-", color="#ef4444", linewidth=1.5,
         label=f"SD2 ({len(y2)} pts)")

# Mark cylinder center
ax1.axvline(CYL_Y, color="#1e293b", ls=":", alpha=0.5, linewidth=1)
ax1.text(CYL_Y + 0.1, dy1.max() * 0.9, "cyl. centre", fontsize=8,
         color="#1e293b", rotation=90, va="top")

ax1.set_xlabel("y / D  (global frame)", fontsize=11)
ax1.set_ylabel("Δy / D", fontsize=11)
ax1.set_title("(b)  Grid Stretching in y", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# ── (c) SD1 grid detail ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0:2])

# Plot every Nth grid line to avoid visual clutter
nx_skip = max(1, len(x1) // 60)
ny_skip = max(1, len(y1) // 60)

# Vertical lines (x = const)
for i in range(0, len(x1), nx_skip):
    ax2.axvline(x1[i], color="#3b82f6", alpha=0.25, linewidth=0.3)
# Horizontal lines (y = const)
for j in range(0, len(y1), ny_skip):
    ax2.axhline(y1g[j], color="#3b82f6", alpha=0.25, linewidth=0.3)

# Overlay SD2 grid with denser lines
nx2_skip = max(1, len(x2) // 40)
ny2_skip = max(1, len(y2) // 40)
for i in range(0, len(x2), nx2_skip):
    ax2.axvline(x2[i], ymin=0, ymax=1, color="#ef4444", alpha=0.3, linewidth=0.4)
for j in range(0, len(y2), ny2_skip):
    ax2.axhline(y2g[j], xmin=0, xmax=1, color="#ef4444", alpha=0.3, linewidth=0.4)

# Cylinder
cyl2 = Circle((CYL_X, CYL_Y), 0.5, facecolor="#1e293b", edgecolor="#0f172a",
              linewidth=1, zorder=5)
ax2.add_patch(cyl2)

# SD2 boundary
sd2_rect2 = Rectangle((x2[0], y2g[0]), x2[-1]-x2[0], y2g[-1]-y2g[0],
                       linewidth=1.5, edgecolor="#ef4444", facecolor="none",
                       linestyle="--", zorder=4)
ax2.add_patch(sd2_rect2)

ax2.set_xlim(x1[0] - 0.2, x1[-1] + 0.2)
ax2.set_ylim(y1g[0] - 0.2, y1g[-1] + 0.2)
ax2.set_xlabel("x / D", fontsize=11)
ax2.set_ylabel("y / D  (global frame)", fontsize=11)
ax2.set_title("(c)  SD1 Grid (blue) + SD2 Boundary (red dashed)", fontsize=12, fontweight="bold")
ax2.set_aspect("equal")
ax2.grid(False)

# ── (d) SD2 zoomed grid ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])

# Show actual grid points as a scatter
# Subsample to avoid too many points
xsub = x2[::3]
ysub = y2[::3]
XX, YY = np.meshgrid(xsub, ysub + Y_OFFSET)
ax3.scatter(XX.ravel(), YY.ravel(), s=0.3, color="#ef4444", alpha=0.6)

# Show grid lines
for i in range(0, len(x2), 4):
    ax3.axvline(x2[i], color="#ef4444", alpha=0.2, linewidth=0.4)
for j in range(0, len(y2), 4):
    ax3.axhline(y2g[j], color="#ef4444", alpha=0.2, linewidth=0.4)

ax3.set_xlim(x2[0] - 0.05, x2[-1] + 0.05)
ax3.set_ylim(y2g[0] - 0.05, y2g[-1] + 0.05)
ax3.set_xlabel("x / D", fontsize=11)
ax3.set_ylabel("y / D  (global frame)", fontsize=11)
ax3.set_title(f"(d)  SD2 Grid Detail ({len(x2)}×{len(y2)}×{len(z2)})", fontsize=12, fontweight="bold")
ax3.set_aspect("equal")
ax3.grid(False)

# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle("CylinderWake3900  —  DNS Grid & Sub-domain Layout  (Re = 3900)",
             fontsize=14, fontweight="bold", y=0.98)

out_path = Path(__file__).parent / "docs" / "grid_preview.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

# Also save to repo root for README
root_path = Path(__file__).parent / "grid_preview.png"
fig.savefig(root_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {root_path}")

plt.close()
