"""
Step 6 - Quick visual checks.

  # CoW model: CT | ground-truth vessels | prediction (like assets/example_prediction.png)
  python src/06_visualize.py cow --series_id <uid> --z_start 205 --z_end 215

  # Final crops: slice grid with aneurysm mask in red
  python src/06_visualize.py crop --group aneurysm --case 0
"""
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from utils import find_nii, window_normalize


def show_cow(a):
    ct = window_normalize(nib.as_closest_canonical(nib.load(find_nii(config.COW_SEG_DIR, a.series_id))).get_fdata())
    gt_p = find_nii(config.COW_SEG_DIR, f"{a.series_id}_cowseg")
    gt = nib.as_closest_canonical(nib.load(gt_p)).get_fdata() if gt_p else None
    pr = nib.as_closest_canonical(nib.load(find_nii(config.pred_dir(a.group), a.series_id))).get_fdata()

    zs = list(range(a.z_start, min(a.z_end, ct.shape[2] - 1) + 1))
    fig, ax = plt.subplots(len(zs), 3, figsize=(10, 2.2 * len(zs)), squeeze=False)
    for i, z in enumerate(zs):
        ax[i, 0].imshow(ct[:, :, z].T, cmap="gray", origin="lower"); ax[i, 0].set_title(f"CT (z={z})")
        if gt is not None:
            ax[i, 1].imshow((gt[:, :, z] > 0).T, cmap="Greens", origin="lower")
        ax[i, 1].set_title("Ground truth")
        ax[i, 2].imshow((pr[:, :, z] > 0).T, cmap="Reds", origin="lower"); ax[i, 2].set_title("Prediction")
        for j in range(3):
            ax[i, j].axis("off")
    plt.tight_layout(); plt.savefig(a.out, dpi=120); print("saved", a.out)


def show_crop(a):
    d = config.crops_dir(a.group)
    img = nib.load(d / "images" / f"case_{a.case:04d}_0000.nii.gz").get_fdata()
    msk = nib.load(d / "labels" / f"case_{a.case:04d}.nii.gz").get_fdata()
    n, cols = img.shape[2], 8
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for i, a_ in enumerate(ax.ravel()):
        a_.axis("off")
        if i < n:
            a_.imshow(img[:, :, i].T, cmap="gray", origin="lower")
            if msk[:, :, i].any():
                a_.imshow(np.ma.masked_where(msk[:, :, i].T == 0, msk[:, :, i].T), cmap="Reds", alpha=0.5, origin="lower")
            a_.set_title(f"z={i}", fontsize=7)
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print("saved", a.out)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="mode", required=True)
c = sub.add_parser("cow")
c.add_argument("--series_id", required=True); c.add_argument("--group", default="aneurysm")
c.add_argument("--z_start", type=int, default=0); c.add_argument("--z_end", type=int, default=10)
c.add_argument("--out", default="cow_check.png")
k = sub.add_parser("crop")
k.add_argument("--group", default="aneurysm"); k.add_argument("--case", type=int, default=0)
k.add_argument("--out", default="crop_check.png")
args = ap.parse_args()
show_cow(args) if args.mode == "cow" else show_crop(args)
