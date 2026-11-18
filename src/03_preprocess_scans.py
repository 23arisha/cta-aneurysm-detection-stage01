"""
Step 3 - Turn raw RSNA DICOM CTA series into 1 mm isotropic NIfTI volumes + aneurysm sphere masks.

    python src/03_preprocess_scans.py --group aneurysm
    python src/03_preprocess_scans.py --group no_aneurysm

Case selection (same as the original notebooks):
  * CTA only, balanced pool: 830 "positive" (artery_sum >= 1) and 830 "negative" (artery_sum == 0)
  * aneurysm group    = first N_PER_GROUP positives after seeded shuffle
  * no_aneurysm group = first N_PER_GROUP negatives after seeded shuffle

Output (config.resampled_dir(group)):
  images_tr/<uid>_scan.nii.gz   labels_tr/<uid>_mask.nii.gz   json_files/<uid>_meta.json   tracking_df.csv
"""
import argparse
import ast
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

import config

ARTERY_COLS = [
    "Left Infraclinoid Internal Carotid Artery", "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery", "Right Middle Cerebral Artery",
    "Anterior Communicating Artery", "Left Anterior Cerebral Artery", "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery", "Right Posterior Communicating Artery",
    "Basilar Tip", "Other Posterior Circulation",
]


# --------------------------------------------------------------------------- case selection
def select_cases(group: str) -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_CSV)
    loc = pd.read_csv(config.LOCALIZERS_CSV)
    df["artery_sum"] = df[ARTERY_COLS].sum(axis=1)
    cta = df[df["Modality"] == "CTA"].copy()

    pool = 830
    g = {k: cta[cta["artery_sum"] == k] for k in range(6)}
    pos_fixed = pd.concat([g[2], g[3], g[4], g[5]])
    pos = pd.concat([pos_fixed, g[1].sample(n=pool - len(pos_fixed), random_state=config.SEED)])
    neg = g[0].sample(n=pool, random_state=config.SEED)
    final = pd.concat([pos, neg]).sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    loc["coordinates"] = loc["coordinates"].apply(ast.literal_eval)
    loc = loc.groupby("SeriesInstanceUID").agg(
        {"SOPInstanceUID": list, "coordinates": list, "location": list}).reset_index()
    merged = final.merge(loc, on="SeriesInstanceUID", how="left")

    want = 1 if group == "aneurysm" else 0
    return merged[merged["Aneurysm Present"] == want].head(config.N_PER_GROUP)


# --------------------------------------------------------------------------- DICOM loading
def load_series(series_path):
    """Read a series as a (Z,Y,X) float32 volume using RAW pixel values (no slope/intercept).
    Returns volume, sorted file list, SOP uids (same order) and (z, y, x) spacing in mm."""
    files = glob(str(Path(series_path) / "*.dcm"))
    if not files:
        raise ValueError("no DICOM files")

    heads = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        try:
            z = float(ds.ImagePositionPatient[2])
        except Exception:
            z = float(ds.InstanceNumber)
        heads.append((f, (ds.Rows, ds.Columns), z, ds.SOPInstanceUID, ds))

    sizes = [h[1] for h in heads]
    target = max(set(sizes), key=sizes.count)                 # most common in-plane size
    heads = sorted([h for h in heads if h[1] == target], key=lambda h: h[2])

    zs = [h[2] for h in heads]
    ds0 = heads[0][4]
    z_sp = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(ds0.SliceThickness)
    xy_sp = float(ds0.PixelSpacing[0])

    vol = np.stack([pydicom.dcmread(h[0]).pixel_array.astype(np.float32) for h in heads], axis=0)
    return vol, [h[0] for h in heads], [h[3] for h in heads], (z_sp, xy_sp, xy_sp), zs


def slice_order(zs):
    if len(zs) < 3:
        return "unknown"
    return "inferior_to_superior" if zs[-1] - zs[0] > 0 else "superior_to_inferior"


# --------------------------------------------------------------------------- resample + mask
def resample(volume, spacing_zyx, new_spacing=config.TARGET_SPACING):
    sz, sy, sx = spacing_zyx
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing((sx, sy, sz))
    old_sp, old_size = img.GetSpacing(), img.GetSize()
    new_size = [int(round(old_size[i] * old_sp[i] / new_spacing[i])) for i in range(3)]

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    out = r.Execute(img)
    return out, sitk.GetArrayFromImage(out).astype(np.float32), old_sp   # old_sp is (x, y, z)


def sphere_mask(shape_zyx, coords, radius):
    D, H, W = shape_zyx
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:D, :H, :W]
    for c in coords:
        z, y, x = (int(round(c[k])) for k in ("z", "y", "x"))
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            mask[(zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = 1
    return mask


# --------------------------------------------------------------------------- one case
def process_case(row, dirs):
    uid = row["SeriesInstanceUID"]
    vol, files, sops, spacing_zyx, zs = load_series(config.SERIES_DIR / uid)
    vol = np.clip(vol, *config.RAW_HU_CLIP)
    img, arr, old_sp = resample(vol, spacing_zyx)

    # aneurysm localizers: (x, y in pixels) + slice index -> mm -> new voxel grid
    coords = []
    if isinstance(row["SOPInstanceUID"], (list, tuple)):
        sop_to_idx = {s: i for i, s in enumerate(sops)}
        for sop, c in zip(row["SOPInstanceUID"], row["coordinates"]):
            if sop in sop_to_idx:
                coords.append({"x": c["x"] * old_sp[0] / config.TARGET_SPACING[0],
                               "y": c["y"] * old_sp[1] / config.TARGET_SPACING[1],
                               "z": sop_to_idx[sop] * old_sp[2] / config.TARGET_SPACING[2]})

    mask = sphere_mask(arr.shape, coords, config.ANEURYSM_RADIUS)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)

    img_p = dirs["images"] / f"{uid}_scan.nii.gz"
    msk_p = dirs["labels"] / f"{uid}_mask.nii.gz"
    sitk.WriteImage(img, str(img_p))
    sitk.WriteImage(mask_img, str(msk_p))

    meta = {
        "series_uid": uid,
        "spacing_before": list(spacing_zyx),
        "spacing_after": list(config.TARGET_SPACING),
        "size_after": list(img.GetSize()),
        "coords_resampled": coords,
        "artery_labels": {c: int(row[c]) for c in ARTERY_COLS if c in row},
        "aneurysm_present": int(row["Aneurysm Present"]),
        "slice_order": slice_order(zs),
    }
    meta_p = dirs["json"] / f"{uid}_meta.json"
    meta_p.write_text(json.dumps(meta))

    return {"series_uid": uid, "scan_path": str(img_p), "mask_path": str(msk_p),
            "json_path": str(meta_p), "resampled_shape": list(arr.shape),
            "aneurysm_present": int(row["Aneurysm Present"]), "slice_order": meta["slice_order"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["aneurysm", "no_aneurysm"], required=True)
    args = ap.parse_args()

    root = config.resampled_dir(args.group)
    dirs = {k: root / v for k, v in {"images": "images_tr", "labels": "labels_tr", "json": "json_files"}.items()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cases = select_cases(args.group)
    print(f"{args.group}: {len(cases)} cases")

    records, failed = [], []
    for _, row in tqdm(cases.iterrows(), total=len(cases)):
        try:
            records.append(process_case(row, dirs))
        except Exception as e:
            failed.append(f"{row['SeriesInstanceUID']}: {e}")

    pd.DataFrame(records).to_csv(root / "tracking_df.csv", index=False)
    (root / "failed.txt").write_text("\n".join(failed))
    print(f"done: {len(records)} ok, {len(failed)} failed (see failed.txt)")


if __name__ == "__main__":
    main()
