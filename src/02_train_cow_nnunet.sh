#!/usr/bin/env bash
# Step 2 - plan, preprocess and train the CoW nnU-Net (ResEnc-M, 3d_fullres, fold 0).
# Run from the repo root:  bash src/02_train_cow_nnunet.sh
set -e
BASE="${NNUNET_BASE:-$(pwd)/outputs/nnunet}"
export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM -np 4

# add --c to resume from the last checkpoint
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz
