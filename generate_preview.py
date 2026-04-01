#!/usr/bin/env python3
"""
Generate a preview figure for the CylinderWake3900 repository.

Shows:
  (a) Computational domain with both sub-domains and cylinder location
  (b) Grid spacing Δy vs y — highlighting the non-uniform stretching
  (c) SD1 grid (x–y) detail with SD2 boundary

Requires only numpy + matplotlib (no data download needed).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

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
fig = plt.figure(figsize=(14, 9), facecolor="white")

gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                      left=0.07, right=0.96, top=0.92, bottom=0.08)

# ══════════════════════════════════════════════════════════════════════
# (a) Full domain overview
# ══════════════════════════════════════════════════════════════════════
ax0 = fig.add_subplot(gs[0, 0])

# Full domain box
full_rect = Rectangle((0, 0), LX, LY, linewidth=1.2, edgecolor="#475569",
                       facecolor="#f1f5f9", zorder=0)
ax0.add_patch(full_rect)

# SD1
sd1_rect = Rectangle((x1[0], y1g[0]), x1[-1] - x1[0], y1g[-1] - y1g[0],
                      linewidth=1.8, edgecolor="#2563eb", facecolor="#bfdbfe",
                      alpha=0.5, zorder=1)
ax0.add_patch(sd1_rect)

# SD2
sd2_rect = Rectangle((x2[0], y2g[0]), x2[-1] - x2[0], y2g[-1] - y2g[0],
                      linewidth=1.8, edgecolor="#dc2626", facecolor="#fecaca",
                      alpha=0.6, zorder=2)
ax0.add_patch(sd2_rect)

# Cylinder
cyl = Circle((CYL_X, CYL_Y), 0.5, facecolor="#1e293b", edgecolor="black",
             linewidth=1.2, zorder=3)
ax0.add_patch(cyl)

# Flow arrow — placed well above the geometry
ax0.annotate("", xy=(3.8, 18), xytext=(1.0, 18),
             arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.8))
ax0.text(2.4, 18.7, r"$U_\infty$", fontsize=11, ha="center", color="#475569")

# Labels outside the boxes using annotation lines
ax0.annotate(
    f"SD1  ({len(x1)}×{len(y1)}×{len(z1)})\n10D × 8D × 6D, 100 snaps",
    xy=(x1[-1], y1g[-1]), xytext=(16.5, 17.5),
    fontsize=8, color="#1e40af", ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2563eb", lw=0.8),
    arrowprops=dict(arrowstyle="-", color="#2563eb", lw=0.8, ls="--"),
)

ax0.annotate(
    f"SD2  ({len(x2)}×{len(y2)}×{len(z2)})\n4D × 2D × 2D, 1000 snaps",
    xy=(x2[-1], y2g[0]), xytext=(16.5, 5.5),
    fontsize=8, color="#b91c1c", ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", lw=0.8),
    arrowprops=dict(arrowstyle="-", color="#dc2626", lw=0.8, ls="--"),
)

# Domain size labels — horizontal (bottom)
ax0.annotate("", xy=(0, -1.8), xytext=(LX, -1.8),
             arrowprops=dict(arrowstyle="<->", color="#475569", lw=0.8))
ax0.text(LX / 2, -2.8, "20D", fontsize=10, ha="center", color="#475569")

# Domain size labels — vertical (left)
ax0.annotate("", xy=(-1.8, 0), xytext=(-1.8, LY),
             arrowprops=dict(arrowstyle="<->", color="#475569", lw=0.8))
ax0.text(-2.8, LY / 2, "20D", fontsize=10, ha="center", va="center",
         color="#475569", rotation=90)

ax0.set_xlim(-4, 22)
ax0.set_ylim(-4, 21)
ax0.set_xlabel("x / D")
ax0.set_ylabel("y / D")
ax0.set_title("(a)  Computational domain", fontweight="bold")
ax0.set_aspect("equal")
ax0.grid(False)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════
# (b) Grid spacing Δy vs y
# ══════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 1])

dy1 = np.diff(y1)
y1_mid = 0.5 * (y1[:-1] + y1[1:]) + Y_OFFSET

ax1.plot(y1_mid, dy1, "-", color="#2563eb", linewidth=1.5,
         label=f"SD1  ({len(y1)} pts)")

# Δy_min annotation — horizontal dashed line + text in the right margin
imin = np.argmin(dy1)
ax1.plot(y1_mid[imin], dy1[imin], "o", color="#2563eb", ms=5, zorder=5)
ax1.axhline(dy1.min(), color="#2563eb", ls="--", alpha=0.35, lw=0.7)
ax1.text(0.97, 0.03, f"Δy$_{{min}}$ = {dy1.min():.5f} D",
         transform=ax1.transAxes, fontsize=9, color="#2563eb",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2563eb",
                   lw=0.6, alpha=0.9))

# SD2 y-spacing
dy2 = np.diff(y2)
y2_mid = 0.5 * (y2[:-1] + y2[1:]) + Y_OFFSET
ax1.plot(y2_mid, dy2, "-", color="#dc2626", linewidth=1.5,
         label=f"SD2  ({len(y2)} pts)")

# Cylinder centre
ax1.axvline(CYL_Y, color="#94a3b8", ls=":", lw=0.8, zorder=0)
ax1.text(CYL_Y + 0.15, dy1.max() * 0.95, "cylinder\ncentre",
         fontsize=7, color="#64748b", va="top")

ax1.set_xlabel("y / D  (global frame)")
ax1.set_ylabel("Δy / D")
ax1.set_title("(b)  Grid stretching in y", fontweight="bold")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.grid(True, alpha=0.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════
# (c) SD1 grid (x–y plane) — spans full bottom row
# ══════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, :])

# Vertical lines (x = const)
nx_skip = max(1, len(x1) // 50)
for i in range(0, len(x1), nx_skip):
    ax2.plot([x1[i], x1[i]], [y1g[0], y1g[-1]],
             color="#2563eb", alpha=0.2, linewidth=0.25)

# Horizontal lines (y = const) — these show the stretching
ny_skip = max(1, len(y1) // 50)
for j in range(0, len(y1), ny_skip):
    ax2.plot([x1[0], x1[-1]], [y1g[j], y1g[j]],
             color="#2563eb", alpha=0.2, linewidth=0.25)

# SD2 boundary
sd2_b = Rectangle((x2[0], y2g[0]), x2[-1] - x2[0], y2g[-1] - y2g[0],
                   linewidth=1.5, edgecolor="#dc2626", facecolor="none",
                   linestyle="--", zorder=4)
ax2.add_patch(sd2_b)

# Cylinder
cyl2 = Circle((CYL_X, CYL_Y), 0.5, facecolor="#1e293b", edgecolor="black",
              linewidth=0.8, zorder=5)
ax2.add_patch(cyl2)

ax2.set_xlim(x1[0] - 0.15, x1[-1] + 0.15)
ax2.set_ylim(y1g[0] - 0.15, y1g[-1] + 0.15)
ax2.set_xlabel("x / D")
ax2.set_ylabel("y / D  (global)")
ax2.set_title("(c)  SD1 mesh  (blue) with SD2 boundary (red dashed)",
              fontweight="bold")
ax2.set_aspect("equal")
ax2.grid(False)


# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle("CylinderWake3900  —  DNS Grid & Sub-domain Layout  (Re = 3900)",
             fontsize=14, fontweight="bold", y=0.97)

# Save
out_path = Path(__file__).parent / "grid_preview.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

out_docs = Path(__file__).parent / "docs" / "grid_preview.png"
fig.savefig(out_docs, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_docs}")

plt.close()
