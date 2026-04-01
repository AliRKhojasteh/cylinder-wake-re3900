# Benchmark Tasks

This document formally defines the benchmark tasks for the CylinderWake3900 dataset. All tasks use the canonical train/val/test split (70/15/15, temporal ordering).

## Task 1: Temporal Forecasting

**Goal**: Predict the Eulerian velocity field at time t+1 given the field(s) at time t (or t, t-1, ..., t-k).

**Setup**:
- **Input**: Velocity field `u(t)` of shape `(3, Nx, Ny, Nz)`, optionally with history `u(t-1), ..., u(t-k)`
- **Output**: Velocity field `u(t+1)` of shape `(3, Nx, Ny, Nz)`
- **Variants**: Single-step (k=0), multi-step history (k=1,2,4), autoregressive rollout (10, 50, 100 steps)
- **Sub-domain**: Near-wake (primary), Far-wake (secondary)

**Metrics**:
- **Primary**: Relative L2 error = `||u_pred - u_true||_2 / ||u_true||_2`
- **Secondary**: SSIM (Structural Similarity), spectral error (energy spectrum comparison)

**Baseline**: Simple CNN with residual connection. See `notebooks/03_baseline_forecasting.ipynb`.

---

## Task 2: Super-Resolution

**Goal**: Reconstruct the full-resolution velocity field from a spatially coarsened version.

**Setup**:
- **Input**: Downsampled velocity field (coarsening factor: 4x, 8x, or 16x in each spatial direction)
- **Output**: Full-resolution velocity field `(3, Nx, Ny, Nz)`
- **Downsampling**: Uniform subsampling (not averaging) to simulate sparse sensor measurements
- **Sub-domain**: Near-wake

**Metrics**:
- **Primary**: Relative L2 error
- **Secondary**: Spectral error (1D energy spectrum along streamwise direction), point-wise max error

**Notes**: Report results for each coarsening factor separately.

---

## Task 3: Sparse Reconstruction (Lagrangian → Eulerian)

**Goal**: Reconstruct the full Eulerian velocity field from sparse Lagrangian particle observations.

This task is **unique to CylinderWake3900** — no other public ML fluid benchmark offers paired Lagrangian–Eulerian data.

**Setup**:
- **Input**: Sparse particle positions and velocities `{(x_i, y_i, z_i, u_i, v_i, w_i)}_{i=1}^{N_obs}`
- **Output**: Full Eulerian velocity field `u(x, y, z)` of shape `(3, Nx, Ny, Nz)`
- **Sparsity levels**: 100%, 10%, 1%, 0.1% of total available particles
- **Subsampling**: Random uniform (fixed seed for reproducibility: `np.random.seed(42)`)

**Metrics**:
- **Primary**: Relative L2 error (volume-averaged)
- **Secondary**: Relative error at specific cross-sections (x/D = 1.06, 1.54, 2.02)

**Challenge levels**:
| Level | Particle fraction | Approx. N_obs | Difficulty |
|-------|:-:|:-:|:-:|
| Easy | 100% | ~100,000 | Low |
| Medium | 10% | ~10,000 | Medium |
| Hard | 1% | ~1,000 | High |
| Extreme | 0.1% | ~100 | Very high |

**Notes**: This task is relevant to experimental PTV/PIV data assimilation. Methods should be evaluated on how gracefully accuracy degrades with decreasing particle density.

---

## Task 4: Lagrangian Prediction

**Goal**: Predict future particle trajectories given the Eulerian velocity field and initial particle positions.

**Setup**:
- **Input**: Eulerian velocity field `u(x, y, z)` at time t, plus initial particle positions at time t
- **Output**: Particle positions at times t+1, t+2, ..., t+T
- **Prediction horizons**: T = 1, 5, 10, 50 steps
- **Reference**: Ground-truth trajectories computed with RK4 + trilinear interpolation

**Metrics**:
- **Primary**: Mean trajectory error = `(1/N) Σ_i ||x_pred_i - x_true_i||_2`
- **Secondary**: Fraction of particles with error < threshold (0.01D, 0.05D, 0.1D)

---

## Reporting Results

When submitting results to the leaderboard, please include:

1. **Method name** and brief description
2. **Results table** for each task you evaluated
3. **Number of parameters** in your model
4. **Training compute** (GPU type, hours)
5. **Code link** (ideally reproducible)
6. **Paper link** (if available)

Submit via Pull Request to update the leaderboard in `README.md`.

## Canonical Data Splits

All tasks use temporal ordering (no shuffling across time):

```
Train: snapshots 0 to int(0.70 * N) - 1
Val:   snapshots int(0.70 * N) to int(0.85 * N) - 1
Test:  snapshots int(0.85 * N) to N - 1
```

This ensures that test performance reflects extrapolation to unseen future dynamics, which is the realistic deployment scenario.
