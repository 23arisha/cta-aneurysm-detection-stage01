"""
Step 4 - Run the trained CoW nnU-Net on the resampled scans. NO cropping here: we only save the
predicted vessel masks next to the original scans.

    python src/04_predict_cow.py --group aneurysm
    python src/04_predict_cow.py --group no_aneurysm

Reads : <resampled>/images_tr/<uid>_scan.nii(.gz)
Writes: <group>/infer_input/<uid>_0000.nii.gz   (staged copies, raw HU)
        <group>/predictions/<uid>.nii.gz        (binary CoW mask from nnU-Net)
Set the nnU-Net results folder (containing Dataset001_CoW_Binary) via --nnunet_base.
"""
import argparse
import subprocess
import numpy as np
import nibabel as nib
from tqdm import tqdm

import config
from utils import setup_nnunet_env, strip_nii

ap = argparse.ArgumentParser()
ap.add_argument("--group", choices=["aneurysm", "no_aneurysm"], required=True)
ap.add_argument("--nnunet_base", default=None, help="folder containing nnUNet_results (default: config.NNUNET_BASE)")
ap.add_argument("--continue_prediction", action="store_true", help="skip cases already predicted")
args = ap.parse_args()

setup_nnunet_env(args.nnunet_base)

img_dir = config.resampled_dir(args.group) / "images_tr"
infer_in = config.infer_input_dir(args.group); infer_in.mkdir(parents=True, exist_ok=True)
infer_out = config.pred_dir(args.group);       infer_out.mkdir(parents=True, exist_ok=True)

files = sorted(f for f in img_dir.iterdir() if f.name.endswith((".nii", ".nii.gz")))
for f in tqdm(files, desc="staging"):
    uid = strip_nii(f.name).replace("_scan", "")
    dst = infer_in / f"{uid}_0000.nii.gz"
    if not dst.exists():
        nii = nib.load(f)
        nib.save(nib.Nifti1Image(np.nan_to_num(nii.get_fdata().astype(np.float32)), nii.affine), dst)

cmd = ["nnUNetv2_predict", "-i", str(infer_in), "-o", str(infer_out),
       "-d", str(config.DATASET_ID), "-p", config.PLANS, "-c", config.CONFIG,
       "-f", config.FOLD, "-chk", config.CHECKPOINT]
if args.continue_prediction:
    cmd.append("--continue_prediction")
print(" ".join(cmd))
subprocess.run(cmd, check=True)
print("prediction done ->", infer_out)
