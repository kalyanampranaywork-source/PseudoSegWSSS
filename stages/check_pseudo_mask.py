from pathlib import Path

import numpy as np
from PIL import Image


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

dataset_root = Path("datasets/LUAD-HistoSeg")

image_name = "1031280-8572-14480-[1 0 0 0].png"
# Example:
# image_name = "TCGA-XX-XXXX+[1 0 1 0].png"


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

mask1_path = dataset_root / "train_PM" / "PM_b4_5" / image_name
mask2_path = dataset_root / "train_PM" / "PM_b5_2" / image_name
mask3_path = dataset_root / "train_PM" / "PM_bn7"  / image_name


# ------------------------------------------------------------
# Load masks
# ------------------------------------------------------------

mask1 = np.array(Image.open(mask1_path))
mask2 = np.array(Image.open(mask2_path))
mask3 = np.array(Image.open(mask3_path))


# ------------------------------------------------------------
# Compare masks
# ------------------------------------------------------------

same_12 = np.mean(mask1 == mask2)
same_13 = np.mean(mask1 == mask3)
same_23 = np.mean(mask2 == mask3)

print("=" * 60)
print("Pixel-wise Equality")
print("=" * 60)
print(f"b4_5 vs b5_2 : {same_12:.6f}")
print(f"b4_5 vs bn7  : {same_13:.6f}")
print(f"b5_2 vs bn7  : {same_23:.6f}")
print("=" * 60)


# ------------------------------------------------------------
# Difference statistics
# ------------------------------------------------------------

print()
print("=" * 60)
print("Different Pixels")
print("=" * 60)
print(f"b4_5 vs b5_2 : {np.sum(mask1 != mask2)}")
print(f"b4_5 vs bn7  : {np.sum(mask1 != mask3)}")
print(f"b5_2 vs bn7  : {np.sum(mask2 != mask3)}")
print("=" * 60)