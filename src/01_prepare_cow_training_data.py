"""
Step 1 - Build the nnU-Net dataset for the binary Circle-of-Willis (CoW) model.

Input : COW_SEG_DIR  ->  <uid>.nii (CT) + <uid>_cowseg.nii (13 vessel classes)
Output: nnUNet_raw/Dataset001_CoW_Binary/{imagesTr,labelsTr,dataset.json}

All 13 vessel classes are collapsed into one label (1 = CoW region). Only CTA series are used.
Raw Hounsfield units are kept so nnU-Net's own CT normalisation works.
"""
import json
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

import config
from utils import setup_nnunet_env, strip_nii

setup_nnunet_env()
import os
from pathlib import Path

out = Path(os.environ["nnUNet_raw"]) / config.DATASET_NAME
(out / "imagesTr").mkdir(parents=True, exist_ok=True)
(out / "labelsTr").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(config.TRAIN_CSV)
modality = dict(zip(df["SeriesInstanceUID"].astype(str), df["Modality"]))

uids = sorted(strip_nii(f.name) for f in config.COW_SEG_DIR.glob("*.nii*")
              if "_cowseg" not in f.name)
print(f"{len(uids)} candidate scans")

ok, skipped_non_cta, failed = 0, 0, []
for uid in tqdm(uids):
    try:
        if "CTA" not in str(modality.get(uid, "")).upper().replace(" ", ""):
            skipped_non_cta += 1
            continue

        img_p = next(config.COW_SEG_DIR.glob(f"{uid}.nii*"))
        lbl_p = next(config.COW_SEG_DIR.glob(f"{uid}_cowseg.nii*"))

        img_nib = nib.as_closest_canonical(nib.load(img_p))
        lbl_nib = nib.as_closest_canonical(nib.load(lbl_p))
        img = np.nan_to_num(img_nib.get_fdata().astype(np.float32))
        lbl = lbl_nib.get_fdata().astype(np.uint8)

        if img.shape != lbl.shape:                     # pad/crop label to image shape
            fixed = np.zeros(img.shape, dtype=np.uint8)
            s = tuple(slice(0, min(a, b)) for a, b in zip(img.shape, lbl.shape))
            fixed[s] = lbl[s]
            lbl = fixed

        lbl = (lbl > 0).astype(np.uint8)               # 13 classes -> 1
        nib.save(nib.Nifti1Image(img, img_nib.affine), out / "imagesTr" / f"{uid}_0000.nii.gz")
        nib.save(nib.Nifti1Image(lbl, img_nib.affine), out / "labelsTr" / f"{uid}.nii.gz")
        ok += 1
    except Exception as e:
        print(f"  error {uid[:40]}: {e}")
        failed.append(uid)

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "CoW_region": 1},
    "numTraining": ok,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}
(out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
print(f"done: {ok} cases | non-CTA skipped: {skipped_non_cta} | failed: {len(failed)}")
