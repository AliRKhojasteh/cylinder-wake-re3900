"""
Automated download of the CylinderWake3900 raw data from the INRAE repository.

Data source: https://doi.org/10.15454/GLNRHK
Hosted on: Recherche Data Gouv (French national research data repository)

This module handles:
  - Downloading individual zip archives per sub-domain and field type
  - Extracting .txt files from zip archives
  - Verifying file integrity via checksums
  - Providing a CLI entry point: `cylinderwake-download`
"""

from __future__ import annotations

import os
import sys
import zipfile
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from urllib.request import urlretrieve
from urllib.error import URLError

# ── Configuration ───────────────────────────────────────────────────

DATASET_DOI = "10.15454/GLNRHK"
BASE_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId="

# ── Complete file registry from INRAE repository (24 files) ─────────
# Verified from: https://doi.org/10.15454/GLNRHK (Version 6.1)
# Last updated: 2026-04-01
#
# Naming convention in the repository:
#   U = streamwise velocity (ux), V = vertical velocity (uy), W = spanwise velocity (uz)
#   Sub-domain 1: 10D x 8D x 6D, grid 769 x 777 x 256 (large, every 10 DNS steps)
#   Sub-domain 2:  4D x 2D x 2D, grid 308 x 328 x  87 (small, every DNS step)

FILE_REGISTRY: Dict[str, Dict] = {
    # ═══════════════════════════════════════════════════════════════
    # Sub-domain 1 — Eulerian 3D velocity (U, V, W)
    # ═══════════════════════════════════════════════════════════════
    "U_sub_domain_1 (1_25).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/KIQ6X7",
        "description": "Streamwise velocity (ux), Sub-domain 1, snapshots 1–25",
        "md5": "3f2c579feed50a354eea0aad76fbd4cd",
        "size_mb": 11219.5,
        "subdomain": 1, "component": "U",
    },
    "U_sub_domain_1 (26_50).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/Q1YTTU",
        "description": "Streamwise velocity (ux), Sub-domain 1, snapshots 26–50",
        "md5": "9beca493ed2f5f960e890e9563dd4dd0",
        "size_mb": 11217.3,
        "subdomain": 1, "component": "U",
    },
    "U_sub_domain_1 (51_75).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/8EUA4C",
        "description": "Streamwise velocity (ux), Sub-domain 1, snapshots 51–75",
        "md5": "80a62f1cfa79aff2a2a2ef0f51479b21",
        "size_mb": 11214.3,
        "subdomain": 1, "component": "U",
    },
    "U_sub_domain_1 (76_100).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/C2FUKQ",
        "description": "Streamwise velocity (ux), Sub-domain 1, snapshots 76–100",
        "md5": "86e48fc985b5e4d020dc8c8c1d431279",
        "size_mb": 11211.2,
        "subdomain": 1, "component": "U",
    },
    "V_sub_domain_1 (1_25).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/M6JOU8",
        "description": "Vertical velocity (uy), Sub-domain 1, snapshots 1–25",
        "md5": "d14bdbfff8152c8db3a8bc8f23df8243",
        "size_mb": 13625.6,
        "subdomain": 1, "component": "V",
    },
    "V_sub_domain_1 (26_50).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/XQYJU9",
        "description": "Vertical velocity (uy), Sub-domain 1, snapshots 26–50",
        "md5": "071a21053447db9bae3d93c45b8f75e4",
        "size_mb": 13632.1,
        "subdomain": 1, "component": "V",
    },
    "V_sub_domain_1 (51_75).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/9ZFT0T",
        "description": "Vertical velocity (uy), Sub-domain 1, snapshots 51–75",
        "md5": "26934345bfc2080a35262b95990470ab",
        "size_mb": 13631.5,
        "subdomain": 1, "component": "V",
    },
    "V_sub_domain_1 (76_100).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/KZIH0G",
        "description": "Vertical velocity (uy), Sub-domain 1, snapshots 76–100",
        "md5": "d16354a730848896e51d5d72899d8b37",
        "size_mb": 13623.4,
        "subdomain": 1, "component": "V",
    },
    "W_sub_domain_1 (1_25).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/NAESYL",
        "description": "Spanwise velocity (uz), Sub-domain 1, snapshots 1–25",
        "md5": "d27bf037ca46ee3d3fd7e9437bb20350",
        "size_mb": 13997.8,
        "subdomain": 1, "component": "W",
    },
    "W_sub_domain_1 (26_50).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/FRNZLU",
        "description": "Spanwise velocity (uz), Sub-domain 1, snapshots 26–50",
        "md5": "5ff2b26db4bee21fc9aa809994584829",
        "size_mb": 13985.1,
        "subdomain": 1, "component": "W",
    },
    "W_sub_domain_1 (51_75).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/FNAO84",
        "description": "Spanwise velocity (uz), Sub-domain 1, snapshots 51–75",
        "md5": "452d31708ddcb0948eac70825bb1ed79",
        "size_mb": 13969.7,
        "subdomain": 1, "component": "W",
    },
    "W_sub_domain_1 (76_100).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/4PXN4F",
        "description": "Spanwise velocity (uz), Sub-domain 1, snapshots 76–100",
        "md5": "9fb6708c007682c1e1eff95724d3b155",
        "size_mb": 13961.3,
        "subdomain": 1, "component": "W",
    },
    # ═══════════════════════════════════════════════════════════════
    # Sub-domain 2 — Eulerian 3D velocity (U, V, W) + Pressure
    # ═══════════════════════════════════════════════════════════════
    "U_sub_domain_2 (1_499).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/9QNEXE",
        "description": "Streamwise velocity (ux), Sub-domain 2, snapshots 1–499",
        "md5": "a735b3cb59783f221e288825307bc1ed",
        "size_mb": 13977.9,
        "subdomain": 2, "component": "U",
    },
    "U_sub_domain_2 (500_1000).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/TXRYLD",
        "description": "Streamwise velocity (ux), Sub-domain 2, snapshots 500–1000",
        "md5": "81ff84ff7f59028ed5bf6e0a26835167",
        "size_mb": 14055.4,
        "subdomain": 2, "component": "U",
    },
    "V_sub_domain_2 (1_499).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/IVBPFB",
        "description": "Vertical velocity (uy), Sub-domain 2, snapshots 1–499",
        "md5": "cf69d8879a40a03530c15d42ce2fccdc",
        "size_mb": 15048.7,
        "subdomain": 2, "component": "V",
    },
    "V_sub_domain_2 (500_1000).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/FTTGED",
        "description": "Vertical velocity (uy), Sub-domain 2, snapshots 500–1000",
        "md5": "47dcfe3ae27eb4d1db68fbc4c7e630fb",
        "size_mb": 15105.2,
        "subdomain": 2, "component": "V",
    },
    "W_sub_domain_2 (1_499).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/6QEK39",
        "description": "Spanwise velocity (uz), Sub-domain 2, snapshots 1–499",
        "md5": "9bf61dd1d595c4e32633ad029e3ed745",
        "size_mb": 15430.4,
        "subdomain": 2, "component": "W",
    },
    "W_sub_domain_2 (500_1000).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/BF9T4X",
        "description": "Spanwise velocity (uz), Sub-domain 2, snapshots 500–1000",
        "md5": "ccd0aef68f9078debd9d8bc07d248605",
        "size_mb": 15456.0,
        "subdomain": 2, "component": "W",
    },
    "P_sub_domain_2 (1_499).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/5ANAP4",
        "description": "Pressure field, Sub-domain 2, snapshots 1–499",
        "md5": "c129554d1606c90a4003ed6f0cde24b4",
        "size_mb": 14171.7,
        "subdomain": 2, "component": "P",
    },
    "P_sub_domain_2 (500_1000).zip": {
        "persistent_id": "doi:10.15454/GLNRHK/CUENCZ",
        "description": "Pressure field, Sub-domain 2, snapshots 500–1000",
        "md5": "0a7fba32fff9bbcd9b56dab84cad2b55",
        "size_mb": 14323.1,
        "subdomain": 2, "component": "P",
    },
    # ═══════════════════════════════════════════════════════════════
    # 2D velocity snapshots (mid-plane)
    # ═══════════════════════════════════════════════════════════════
    "2D_velocity_snapshot_sub_domain_1.zip": {
        "persistent_id": "doi:10.15454/GLNRHK/NT0JFM",
        "description": "2D velocity snapshots (mid-plane), Sub-domain 1",
        "md5": "b228b0b598b0140ef03fabbfc3da23f7",
        "size_mb": 9322.3,
        "subdomain": 1, "component": "2D",
    },
    "2D_velocity_snapshot_sub_domain_2.zip": {
        "persistent_id": "doi:10.15454/GLNRHK/VFDW5Z",
        "description": "2D velocity snapshots (mid-plane), Sub-domain 2",
        "md5": "76d0243019b6fc40d482c483e5440c43",
        "size_mb": 4116.0,
        "subdomain": 2, "component": "2D",
    },
    # ═══════════════════════════════════════════════════════════════
    # Lagrangian particle trajectories
    # ═══════════════════════════════════════════════════════════════
    "Lagrangian_tracks_sub_domain_1.zip": {
        "persistent_id": "doi:10.15454/GLNRHK/RTIP6A",
        "description": "Lagrangian particle trajectories (~100k), Sub-domain 1",
        "md5": "dd4c6088ab570502e91f5522a2cd2f10",
        "size_mb": 3760.0,
        "subdomain": 1, "component": "Lagrangian",
    },
    "Lagrangian_tracks_sub_domain_2.zip": {
        "persistent_id": "doi:10.15454/GLNRHK/0GELBZ",
        "description": "Lagrangian particle trajectories (~100k), Sub-domain 2",
        "md5": "c0c336a318ca5fb7b769215914217fcc",
        "size_mb": 10318.1,
        "subdomain": 2, "component": "Lagrangian",
    },
}

# ── Convenience groupings for selective downloads ────────────────────

SUBDOMAIN_1_FILES = [k for k, v in FILE_REGISTRY.items() if v["subdomain"] == 1]
SUBDOMAIN_2_FILES = [k for k, v in FILE_REGISTRY.items() if v["subdomain"] == 2]
LAGRANGIAN_FILES = [k for k, v in FILE_REGISTRY.items() if v["component"] == "Lagrangian"]
TWOD_FILES = [k for k, v in FILE_REGISTRY.items() if v["component"] == "2D"]

# Total dataset size
TOTAL_SIZE_GB = sum(v["size_mb"] for v in FILE_REGISTRY.values()) / 1000
# Approximately 288 GB total


# ── Paths ───────────────────────────────────────────────────────────

def get_data_dir(root: Optional[Path] = None) -> Path:
    """
    Default data directory: ~/.cylinderwake3900/

    Override with:
      - root parameter
      - CYLINDERWAKE_DATA_DIR environment variable
    """
    if root is not None:
        return Path(root)
    env_dir = os.environ.get("CYLINDERWAKE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cylinderwake3900"


# ── Download logic ──────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int):
    """CLI progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        bar = "█" * (pct // 3) + "░" * (33 - pct // 3)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()


def download_file(
    persistent_id: str,
    dest: Path,
    description: str = "",
    expected_md5: Optional[str] = None,
) -> Path:
    """Download a single file from INRAE Dataverse."""
    if dest.exists():
        if expected_md5:
            actual = hashlib.md5(dest.read_bytes()).hexdigest()
            if actual == expected_md5:
                print(f"  ✓ {dest.name} (cached, checksum OK)")
                return dest
        else:
            print(f"  ✓ {dest.name} (cached)")
            return dest

    url = f"{BASE_URL}{persistent_id}"
    print(f"  ↓ Downloading {dest.name}: {description}")

    try:
        urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()  # newline after progress bar
    except URLError as e:
        raise RuntimeError(
            f"Failed to download {dest.name} from {url}\n"
            f"Error: {e}\n"
            f"You can manually download from: https://doi.org/{DATASET_DOI}"
        ) from e

    if expected_md5:
        actual = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: "
                f"expected {expected_md5}, got {actual}"
            )

    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive."""
    print(f"  📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def download_dataset(
    root: Optional[Path] = None,
    files: Optional[List[str]] = None,
) -> Path:
    """
    Download the full (or partial) dataset from INRAE.

    Parameters
    ----------
    root : Path, optional
        Target directory. Default: ~/.cylinderwake3900/
    files : list of str, optional
        Specific files to download. Default: all files.

    Returns
    -------
    Path
        The root data directory.
    """
    data_dir = get_data_dir(root)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    targets = files if files else list(FILE_REGISTRY.keys())

    print(f"╔══ CylinderWake3900 Dataset Download ══╗")
    print(f"║  Source: https://doi.org/{DATASET_DOI}")
    print(f"║  Target: {data_dir}")
    print(f"║  Files:  {len(targets)}")
    print(f"╚════════════════════════════════════════╝\n")

    for fname in targets:
        if fname not in FILE_REGISTRY:
            print(f"  ⚠ Unknown file: {fname}, skipping")
            continue

        info = FILE_REGISTRY[fname]
        dest = raw_dir / fname

        download_file(
            persistent_id=info["persistent_id"],
            dest=dest,
            description=info["description"],
            expected_md5=info.get("md5"),
        )

        # Auto-extract zip files
        if dest.suffix == ".zip" and dest.exists():
            extract_dir = raw_dir / dest.stem
            if not extract_dir.exists():
                extract_zip(dest, extract_dir)

    print(f"\n✅ Download complete! Raw data in: {raw_dir}")
    print(f"   Next step: convert to HDF5 with `cylinderwake-convert`")
    return data_dir


# ── CLI entry points ────────────────────────────────────────────────

def cli_download():
    """CLI: cylinderwake-download"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download CylinderWake3900 dataset from INRAE"
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Target directory (default: ~/.cylinderwake3900/)"
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Specific files to download (default: all)"
    )
    args = parser.parse_args()
    download_dataset(root=Path(args.root) if args.root else None, files=args.files)


if __name__ == "__main__":
    cli_download()
