"""Small helpers shared by all scripts."""
import os
from pathlib import Path
import numpy as np
import config


def setup_nnunet_env(base: Path = None):
    """Point nnU-Net at its three folders (call before running nnU-Net)."""
    base = Path(base or config.NNUNET_BASE).resolve()
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        (base / name).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(base / name)


def find_nii(folder, stem):
    """Return folder/stem.nii.gz or folder/stem.nii (Kaggle datasets often unzip to .nii)."""
    for ext in (".nii.gz", ".nii"):
        p = Path(folder) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def strip_nii(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def window_normalize(img, lo=None, hi=None):
    """Clip to the vessel window and scale to [0, 1]."""
    if lo is None:
        lo, hi = config.WINDOW_HU
    img = np.nan_to_num(img.astype(np.float32))
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo)).astype(np.float32)
