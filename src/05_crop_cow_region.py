"""
Step 5 - Crop every scan around its predicted Circle of Willis.

    python src/05_crop_cow_region.py --group aneurysm
    python src/05_crop_cow_region.py --group no_aneurysm

Crop rule (per scan):
  1. Find runs of consecutive axial slices that contain predicted vessel voxels.
  2. If the scan is taller than 250 slices and there are several runs, keep the LAST run
     (scan goes chest -> head, so the CoW is the top-most run).   rule = later_range_gt250
     Otherwise keep the LONGEST run.                              rule = largest_range
  3. z-range = that run +/- PAD_Z slices.
  4. x/y = a 96x96 window centred on the vessel bounding box of that run.
  5. No prediction at all -> fixed anatomical box (centre, 45 % height).  rule = anatomical_fallback

Writes (config.crops_dir(group)):
  images/case_XXXX_0000.nii.gz   (HU-windowed to [0,1])   labels/case_XXXX.nii.gz   crop_metadata.csv
`coverage_pct` = % of aneurysm voxels that fall inside the crop (aneurysm group only).
"""
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import config
from utils import find_nii, strip_nii, window_normalize


def contiguous_runs(pred):
    zs = [z for z in range(pred.shape[2]) if pred[:, :, z].any()]
    runs = []
    if zs:
        start = prev = zs[0]
        for z in zs[1:]:
            if z != prev + 1:
                runs.append((start, prev))
                start = z
            prev = z
        runs.append((start, prev))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["aneurysm", "no_aneurysm"], required=True)
    args = ap.parse_args()

    res = config.resampled_dir(args.group)
    img_dir, lbl_dir = res / "images_tr", res / "labels_tr"
    pred_dir = config.pred_dir(args.group)
    out = config.crops_dir(args.group)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # stable case ids = position in the sorted image list
    uids = sorted(strip_nii(f.name).replace("_scan", "") for f in img_dir.iterdir()
                  if f.name.endswith((".nii", ".nii.gz")))
    case_id = {u: i for i, u in enumerate(uids)}

    PAD, S = config.PAD_Z, config.CROP_XY
    rows = []
    for uid in tqdm(uids, desc="cropping"):
        img_p = find_nii(img_dir, f"{uid}_scan")
        lbl_p = find_nii(lbl_dir, f"{uid}_mask")
        pred_p = find_nii(pred_dir, uid)
        if not (img_p and lbl_p and pred_p):
            print(f"skip {uid[-12:]}: missing image/label/prediction")
            continue

        img_nii = nib.load(img_p)
        image = img_nii.get_fdata()
        label = (nib.load(lbl_p).get_fdata() > 0).astype(np.uint8)
        pred = (nib.load(pred_p).get_fdata() > 0).astype(np.uint8)
        if not (pred.shape == image.shape == label.shape):
            print(f"skip {uid[-12:]}: shape mismatch {pred.shape} {image.shape} {label.shape}")
            continue

        runs = contiguous_runs(pred)
        if not runs:
            rule = "anatomical_fallback"
            cx, cy = image.shape[0] // 2, image.shape[1] // 2
            cz = int(image.shape[2] * 0.45)
            z_lo, z_hi = max(0, cz - 60 - PAD), min(image.shape[2] - 1, cz + 60 + PAD)
        else:
            if image.shape[2] > config.MAX_Z_SPAN and len(runs) > 1:
                z_a, z_b = runs[-1]; rule = "later_range_gt250"
            else:
                z_a, z_b = max(runs, key=lambda r: r[1] - r[0]); rule = "largest_range"
            xy = np.argwhere(pred[:, :, z_a:z_b + 1] > 0)[:, :2]
            cx, cy = (xy[:, 0].min() + xy[:, 0].max()) // 2, (xy[:, 1].min() + xy[:, 1].max()) // 2
            z_lo, z_hi = max(0, z_a - PAD), min(image.shape[2] - 1, z_b + PAD)

        # 96x96 window, shifted (not shrunk) at the borders
        x0 = int(np.clip(cx - S // 2, 0, max(0, image.shape[0] - S))); x1 = min(image.shape[0], x0 + S)
        y0 = int(np.clip(cy - S // 2, 0, max(0, image.shape[1] - S))); y1 = min(image.shape[1], y0 + S)

        img_crop = window_normalize(image[x0:x1, y0:y1, z_lo:z_hi + 1])
        lbl_crop = label[x0:x1, y0:y1, z_lo:z_hi + 1]

        affine = img_nii.affine.copy()
        affine[:3, 3] += img_nii.affine[:3, :3] @ np.array([x0, y0, z_lo])

        cid = case_id[uid]
        nib.save(nib.Nifti1Image(img_crop, affine), out / "images" / f"case_{cid:04d}_0000.nii.gz")
        nib.save(nib.Nifti1Image(lbl_crop.astype(np.uint8), affine), out / "labels" / f"case_{cid:04d}.nii.gz")

        total, inside = int(label.sum()), int(lbl_crop.sum())
        rows.append({
            "case": cid, "series_id": uid, "rule": rule,
            "orig_shape": str(image.shape), "crop_shape": str(img_crop.shape),
            "prediction_ranges": str(runs), "selected_z": f"{z_lo}-{z_hi}",
            "x_crop": f"{x0}-{x1 - 1}", "y_crop": f"{y0}-{y1 - 1}",
            "total_aneurysm_voxels": total, "inside_aneurysm_voxels": inside,
            "coverage_pct": round(100 * inside / total, 2) if total > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "crop_metadata.csv", index=False)

    print(f"\ncrops saved: {len(df)}  ->  {out}")
    print(df["rule"].value_counts().to_string())
    if args.group == "aneurysm" and len(df):
        full = int((df["coverage_pct"] == 100).sum())
        print(f"aneurysm fully inside crop: {full}/{len(df)}  | mean coverage {df['coverage_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
