"""
Step 7 - Dice of the CoW predictions against the ground-truth vessel masks (COW_SEG_DIR).

    python src/07_evaluate_cow_dice.py --group aneurysm
Only works for series that have a <uid>_cowseg.nii in COW_SEG_DIR.
"""
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

import config
from utils import find_nii, strip_nii

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="aneurysm")
args = ap.parse_args()


def dice(p, g):
    p, g = p > 0, g > 0
    d = p.sum() + g.sum()
    return 1.0 if d == 0 else 2.0 * np.logical_and(p, g).sum() / d


rows = []
for f in sorted(config.pred_dir(args.group).glob("*.nii*")):
    uid = strip_nii(f.name)
    gt_p = find_nii(config.COW_SEG_DIR, f"{uid}_cowseg")
    if gt_p is None:
        continue
    gt = nib.as_closest_canonical(nib.load(gt_p)).get_fdata()
    pr = nib.as_closest_canonical(nib.load(f)).get_fdata()
    rows.append({"series_id": uid, "dice": round(float(dice(pr, gt)), 4)})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
if len(df):
    print(f"\nmean Dice: {df['dice'].mean():.4f} over {len(df)} scans")
    df.to_csv(config.pred_dir(args.group).parent / "cow_dice.csv", index=False)
