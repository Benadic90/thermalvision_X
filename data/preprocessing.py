"""
data/preprocessing.py
Thermal/Infrared Dataset Preprocessing Script
- Resize images to 224x224
- Normalize pixel values (0 to 1)
- Remove corrupted/missing/wrong-size files
- Save processed images to datasets/processed/
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ──────────────────────────────────────────
# SETTINGS (change these if needed)
# ──────────────────────────────────────────
INPUT_FOLDER  = "datasets/raws/roboflow_thermal"           # where your downloaded images are
OUTPUT_FOLDER = "datasets/processed"     # where to save processed images
TARGET_SIZE   = (224, 224)               # resize all images to this size
# ──────────────────────────────────────────


def resize_image(img):
    """Resize image to TARGET_SIZE using cv2."""
    return cv2.resize(img, TARGET_SIZE)


def normalize_image(img):
    """Normalize pixel values from 0-255 to 0.0-1.0."""
    return img / 255.0


def is_corrupted(filepath):
    """Check if an image file is corrupted or unreadable."""
    img = cv2.imread(str(filepath))
    if img is None:
        return True   # file is corrupted or not a valid image
    return False


def check_dimensions(img):
    """Check if image has valid dimensions (not 0 or wrong shape)."""
    if img is None:
        return False
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return False
    return True


def preprocess_all():
    input_path  = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image formats
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    # Counters for the report
    total       = 0
    processed   = 0
    corrupted   = 0
    wrong_dim   = 0

    print("=" * 50)
    print("  THERMAL DATASET PREPROCESSING")
    print("=" * 50)
    print(f"Input  : {INPUT_FOLDER}")
    print(f"Output : {OUTPUT_FOLDER}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("-" * 50)

    # Loop through all image files in input folder (including subfolders)
    for filepath in input_path.rglob("*"):
        if filepath.suffix.lower() not in extensions:
            continue   # skip non-image files

        total += 1
        filename = filepath.name

        # ── STEP 1: Check for corrupted files ──
        if is_corrupted(filepath):
            print(f"[CORRUPTED] Skipping: {filename}")
            corrupted += 1
            continue

        # ── Read the image ──
        img = cv2.imread(str(filepath))

        # ── STEP 2: Check dimensions ──
        if not check_dimensions(img):
            print(f"[BAD SIZE ] Skipping: {filename}  →  shape: {img.shape}")
            wrong_dim += 1
            continue

        # ── STEP 3: Resize ──
        img_resized = resize_image(img)

        # ── STEP 4: Normalize ──
        img_normalized = normalize_image(img_resized)

        # Convert back to uint8 for saving (0-255 range)
        img_to_save = (img_normalized * 255).astype(np.uint8)

        # ── Save processed image ──
        save_path = output_path / filename
        cv2.imwrite(str(save_path), img_to_save)
        processed += 1
        print(f"[OK]        Processed: {filename}")

    # ── Final Report ──
    print("\n" + "=" * 50)
    print("  PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total images found   : {total}")
    print(f"Successfully processed: {processed}")
    print(f"Corrupted (skipped)  : {corrupted}")
    print(f"Wrong dimensions     : {wrong_dim}")
    print(f"Saved to             : {OUTPUT_FOLDER}")
    print("=" * 50)


if __name__ == "__main__":
    preprocess_all()