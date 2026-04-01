# Contributing to CylinderWake3900

We welcome contributions from the fluid mechanics, ML, and data assimilation communities!

## Ways to Contribute

### 1. Submit Benchmark Results

Run your model on one or more of our [benchmark tasks](benchmarks/tasks.md) and submit results:

1. Fork this repository
2. Add your results to the leaderboard table in `README.md`
3. Include: method name, metrics, parameter count, training compute, code link
4. Open a Pull Request with a brief description of your approach

### 2. Add New Benchmark Tasks

Have an idea for a new task? Open an Issue describing:
- The task definition (input/output/metric)
- Why it's scientifically interesting
- How it leverages this dataset's unique properties

### 3. Improve Data Loaders

- Add TensorFlow / JAX dataset classes
- Improve download robustness
- Add support for new data formats (e.g., Zarr, NetCDF)

### 4. Add Visualizations

- Interactive web visualizations (Plotly, D3.js)
- Animations of vortex shedding
- Comparison plots between methods

### 5. Bug Reports

Found a bug? Open an Issue with:
- Python version and OS
- Steps to reproduce
- Error message / traceback

## Development Setup

```bash
git clone https://github.com/AliRKhojasteh/cylinder-wake-re3900.git
cd cylinder-wake-re3900
pip install -e ".[dev,all]"
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged
- Docstrings follow NumPy style

## Citation

If your contribution leads to a publication, please cite the original dataset paper:

```bibtex
@article{khojasteh2021lagrangian,
  title   = {Lagrangian and {Eulerian} dataset of the wake downstream of a smooth cylinder at a {Reynolds} number equal to 3900},
  author  = {Khojasteh, Ali Rahimi and Laizet, Sylvain and Heitz, Dominique and Yang, Yang},
  journal = {Data in Brief},
  volume  = {40},
  pages   = {107725},
  year    = {2021},
  doi     = {10.1016/j.dib.2021.107725},
}
```
