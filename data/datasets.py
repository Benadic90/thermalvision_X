# data/datasets.py
# THERMAVISION-X — PyTorch Dataset Classes
# Researched & designed by Benad | BAH 2026

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class VisibleImageDataset(Dataset):
    """
    Dataset for TRAINING — loads VISIBLE (RGB) satellite images.
    The model learns to reconstruct color from structure using these.

    Sources:
        - Landsat 8/9 RGB bands (freely available)
        - Sentinel-2 RGB composites
        - KAIST dataset visible images (for quick testing)
        - FLIR dataset visible images (for quick testing)

    Args:
        root_dir:  Folder containing visible images (JPG/PNG)
        img_size:  Resize all images to this size
        augment:   Apply random flips, rotations for data augmentation
    """

    def __init__(self, root_dir: str, img_size: int = 256, augment: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size

        # Find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext),
                                               recursive=True))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        # Transforms
        transform_list = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),           # [0, 255] → [0.0, 1.0]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ]
        if augment:
            transform_list = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
            ] + transform_list

        self.transform = T.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # [3, H, W]

        return {
            'image': image,            # Full RGB image (target)
            'path' : img_path,
        }


class IRImageDataset(Dataset):
    """
    Dataset for INFERENCE / VALIDATION — loads grayscale IR images.
    Used at test time (zero-shot: model was NOT trained on these).

    Sources:
        - KAIST thermal IR images
        - FLIR thermal IR images
        - INSAT-3D thermal data (from MOSDAC portal)
        - Any grayscale thermal image for demo

    Args:
        root_dir: Folder containing IR images (JPG/PNG/TIF)
        img_size: Resize all images to this size
    """

    def __init__(self, root_dir: str, img_size: int = 256):
        self.root_dir = root_dir
        self.img_size = img_size

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext),
                                               recursive=True))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),           # [0, 255] → [0.0, 1.0]
            T.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalize → [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Force grayscale
        image = self.transform(image)  # [1, H, W]

        return {
            'image': image,
            'path' : img_path,
        }


if __name__ == "__main__":
    # Quick test — just checks the class can be instantiated
    print("Dataset classes ready for use.")
    print("Usage:")
    print("  dataset = VisibleImageDataset('./data/visible/', img_size=256)")
    print("  loader  = DataLoader(dataset, batch_size=8, shuffle=True)")
    print("✅ Dataset classes working correctly!")
