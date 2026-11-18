"""
Central config for Stage 1. Edit the PATHS block only.
Kaggle equivalents are shown in comments.
"""
from pathlib import Path

# ----------------------------- PATHS (edit me) -----------------------------
# RSNA competition data
RSNA_DIR        = Path("data/rsna-intracranial-aneurysm-detection")   # /kaggle/input/competitions/rsna-intracranial-aneurysm-detection
# CoW training data (CT + 13-class vessel masks, already resampled to 1 mm)
COW_SEG_DIR     = Path("data/ica-segs/resampled_segmentations_1mm")   # /kaggle/input/datasets/arishafarid/ica-segs/...
# Everything we generate
OUT_DIR         = Path("outputs")                                     # /kaggle/working
NNUNET_BASE     = OUT_DIR / "nnunet"                                  # holds nnUNet_raw / _preprocessed / _results

TRAIN_CSV       = RSNA_DIR / "train.csv"
LOCALIZERS_CSV  = RSNA_DIR / "train_localizers.csv"
SERIES_DIR      = RSNA_DIR / "series"

# ----------------------------- CONSTANTS -----------------------------------
TARGET_SPACING  = (1.0, 1.0, 1.0)     # mm, isotropic
RAW_HU_CLIP     = (-1024, 3000)      # applied before resampling
WINDOW_HU       = (-100, 400)        # vessel window used for the crops
ANEURYSM_RADIUS = 4                  # voxels (1 mm) -> sphere mask around each RSNA localizer

N_PER_GROUP     = 400                # aneurysm cases / no-aneurysm cases used
SEED            = 42

# nnU-Net
DATASET_ID      = 1
DATASET_NAME    = "Dataset001_CoW_Binary"
PLANS           = "nnUNetResEncUNetMPlans"
CONFIG          = "3d_fullres"
FOLD            = "0"
CHECKPOINT      = "checkpoint_best.pth"

# Cropping
CROP_XY         = 96                 # in-plane crop size (voxels)
PAD_Z           = 15                 # slices of padding above/below the vessel range
MAX_Z_SPAN      = 250                # scans taller than this may include neck/chest

# Output sub-folders per group ("aneurysm" or "no_aneurysm")
def group_dir(group: str) -> Path:
    return OUT_DIR / group

def resampled_dir(group):   return group_dir(group) / "resampled"      # images_tr/ labels_tr/ json_files/
def infer_input_dir(group): return group_dir(group) / "infer_input"
def pred_dir(group):        return group_dir(group) / "predictions"
def crops_dir(group):       return group_dir(group) / "crops"          # images/ labels/ crop_metadata.csv
