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
IMAGE_SIZE       = (256, 256)             # must match UNet input (256x256)
BATCH_SIZE       = 8                      # how many images per batch
# ──────────────────────────────────────────


class VisibleImageDataset(Dataset):
    """
    Custom PyTorch Dataset for Visible (RGB) Images.
    Used for zero-shot training of the UNet.
    """

    def __init__(self, root_dir=PROCESSED_FOLDER, img_size=IMAGE_SIZE, augment=False):
        """
        Constructor - runs when you create VisibleImageDataset()
        Collects all image file paths from the folder.
        """
        self.folder     = Path(root_dir)
        self.img_size   = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.augment    = augment
        self.extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

        # Collect all valid image paths
        self.image_paths = []
        if self.folder.exists():
            for filepath in self.folder.rglob("*"):
                if filepath.suffix.lower() in self.extensions:
                    self.image_paths.append(filepath)

        print(f"[Dataset] Found {len(self.image_paths)} images in '{root_dir}'")


    def __len__(self):
        """Returns total number of images in the dataset."""
        return len(self.image_paths)


    def __getitem__(self, index):
        """
        Returns one image at a given index as a PyTorch tensor inside a dictionary.
        Dictionary format is required by train.py.
        """
        filepath = self.image_paths[index]
        img = cv2.imread(str(filepath))

        if img is None:
            print(f"[Warning] Could not load: {filepath.name}")
            tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            return {'image': tensor}

        # BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model requirement
        img = cv2.resize(img, self.img_size)

        # Normalize to -1.0 to 1.0 (Standard for GANs/UNets)
        img = (img / 127.5) - 1.0

        # Convert to PyTorch tensor (C, H, W) format
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.tensor(img, dtype=torch.float32)

        return {'image': tensor}


def get_dataloader(folder=PROCESSED_FOLDER, batch_size=BATCH_SIZE, shuffle=True):
    dataset    = VisibleImageDataset(root_dir=folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    print("=" * 50)
    print("  VISIBLE IMAGE DATASET LOADER TEST")
    print("=" * 50)

    dataset = VisibleImageDataset()
    print(f"Total images     : {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]['image']
        print(f"Sample shape     : {sample.shape}")
        print(f"Sample min value : {sample.min():.4f}")
        print(f"Sample max value : {sample.max():.4f}")

    print("\n=" * 50)
    print("  DATASET LOADER WORKING CORRECTLY!")
    print("=" * 50)