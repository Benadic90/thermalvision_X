# train.py
# THERMAVISION-X — Main Training Script
# Self-supervised training on VISIBLE images only (zero-shot learning)
# Researched & designed by Benad | BAH 2026

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import ThermaVisionUNet
from models.frequency import FrequencyDecouplingModule
from data.datasets import VisibleImageDataset
from training.loss import ThermaVisionLoss


def train(args):
    # ── Device Setup ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"  THERMAVISION-X Training")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*50}\n")

    # ── Model Setup ─────────────────────────────────────────────────────────
    model    = ThermaVisionUNet(in_channels=1, out_channels=3).to(device)
    freq_mod = FrequencyDecouplingModule(radius_ratio=args.radius_ratio).to(device)
    criterion = ThermaVisionLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Dataset Setup ───────────────────────────────────────────────────────
    dataset = VisibleImageDataset(
        root_dir=args.data_dir,
        img_size=args.img_size,
        augment=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    print(f"Dataset: {len(dataset)} visible images found")
    print(f"Starting training for {args.epochs} epochs...\n")

    # ── Training Loop ───────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'freq': 0, 'planck': 0}
        start = time.time()

        with tqdm(loader, desc=f"Epoch {epoch:3d}/{args.epochs}") as pbar:
            for batch in pbar:
                images = batch['image'].to(device)  # [B, 3, H, W] visible RGB

                # Convert to grayscale → simulate IR-like input
                gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]

                # Frequency decoupling → extract structure features
                freq_out = freq_mod(gray)
                structure = freq_out['masked_input']  # High-freq only [B, 1, H, W]

                # Forward pass → predict colorization from structure
                pred = model(structure)  # [B, 3, H, W]

                # Target: original RGB image (normalized to [0,1])
                target = (images + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]

                # Compute loss
                losses = criterion(pred, target)
                loss   = losses['total']

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update progress
                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k] += losses[k].item()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        elapsed = time.time() - start

        # Average losses
        n = len(loader)
        avg_loss = epoch_losses['total'] / n
        print(f"  → Avg loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'loss':        best_loss,
                'args':        vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✅ Best model saved (loss={best_loss:.4f})")

        # Regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    print(f"\n{'='*50}")
    print(f"  Training complete! Best loss: {best_loss:.4f}")
    print(f"  Model saved to: {args.save_dir}/best_model.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='THERMAVISION-X Training')
    parser.add_argument('--data_dir',     type=str,   default='./assets/samples/visible',
                        help='Path to visible training images')
    parser.add_argument('--save_dir',     type=str,   default='./checkpoints',
                        help='Where to save model checkpoints')
    parser.add_argument('--epochs',       type=int,   default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',   type=int,   default=8,
                        help='Batch size (reduce if CUDA OOM)')
    parser.add_argument('--lr',           type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--img_size',     type=int,   default=256,
                        help='Image size for training')
    parser.add_argument('--radius_ratio', type=float, default=0.15,
                        help='FFT mask radius ratio (zero-shot tuning)')

    args = parser.parse_args()
    train(args)
