"""
data/datasets.py
Thermal Dataset Loader using PyTorch
- Loads images from datasets/processed/
- Returns image tensors ready for model training
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────
PROCESSED_FOLDER = "datasets/processed"   # folder with preprocessed images
IMAGE_SIZE       = (224, 224)             # must match preprocessing.py
BATCH_SIZE       = 8                      # how many images per batch
# ──────────────────────────────────────────


class ThermalDataset(Dataset):
    """
    Custom PyTorch Dataset for Thermal/Infrared Images.
    Loads all images from the processed folder.
    """

    def __init__(self, folder=PROCESSED_FOLDER):
        """
        Constructor - runs when you create ThermalDataset()
        Collects all image file paths from the folder.
        """
        self.folder     = Path(folder)
        self.extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

        # Collect all valid image paths
        self.image_paths = []
        for filepath in self.folder.rglob("*"):
            if filepath.suffix.lower() in self.extensions:
                self.image_paths.append(filepath)

        print(f"[Dataset] Found {len(self.image_paths)} images in '{folder}'")


    def __len__(self):
        """
        Returns total number of images in the dataset.
        PyTorch needs this to know dataset size.
        """
        return len(self.image_paths)


    def __getitem__(self, index):
        """
        Returns one image at a given index as a PyTorch tensor.
        PyTorch calls this automatically during training.

        Steps:
        1. Read image from disk
        2. Resize to IMAGE_SIZE
        3. Normalize to 0.0 - 1.0
        4. Convert to PyTorch tensor (C, H, W) format
        """

        # ── Step 1: Read image ──
        filepath = self.image_paths[index]
        img = cv2.imread(str(filepath))

        if img is None:
            # If image fails to load, return a blank tensor
            print(f"[Warning] Could not load: {filepath.name}")
            return torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

        # ── Step 2: Resize ──
        img = cv2.resize(img, IMAGE_SIZE)

        # ── Step 3: Normalize (0-255 → 0.0-1.0) ──
        img = img / 255.0

        # ── Step 4: Convert to tensor ──
        # OpenCV loads as (H, W, C) → PyTorch needs (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.tensor(img, dtype=torch.float32)

        return tensor


def get_dataloader(folder=PROCESSED_FOLDER, batch_size=BATCH_SIZE, shuffle=True):
    """
    Creates and returns a DataLoader from the ThermalDataset.
    DataLoader batches images together for training.
    """
    dataset    = ThermalDataset(folder=folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# ──────────────────────────────────────────
# TEST - Run this file directly to test
# ──────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("  THERMAL DATASET LOADER TEST")
    print("=" * 50)

    # Create dataset
    dataset = ThermalDataset()

    # Show basic info
    print(f"Total images     : {len(dataset)}")

    # Load one image and show its shape
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape     : {sample.shape}")   # should be (3, 224, 224)
        print(f"Sample min value : {sample.min():.4f}")  # should be ~0.0
        print(f"Sample max value : {sample.max():.4f}")  # should be ~1.0

    print()

    # Create dataloader and load one batch
    print("Testing DataLoader (batch loading)...")
    loader = get_dataloader(batch_size=BATCH_SIZE)

    for batch in loader:
        print(f"Batch shape      : {batch.shape}")  # (8, 3, 224, 224)
        print(f"Batch size       : {batch.shape[0]} images")
        print(f"Image dimensions : {batch.shape[2]}x{batch.shape[3]}")
        break   # only test first batch

    print()
    print("=" * 50)
    print("  DATASET LOADER WORKING CORRECTLY!")
    print("=" * 50)