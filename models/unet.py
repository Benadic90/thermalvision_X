# models/unet.py
# THERMAVISION-X — Core UNet Generator
# Architecture: Encoder-Decoder with Skip Connections
# Input:  [B, 1, H, W]  — grayscale IR structure features (from FFT module)
# Output: [B, 3, H, W]  — HSV colorized image
# Params: ~5M (fits comfortably on RTX 2050 4GB)
# Researched & designed by Benad | BAH 2026

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic building block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    Used in both encoder and decoder.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder step: ConvBlock + MaxPool (downsampling).
    Returns both the skip connection feature and the pooled output.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)      # Save for skip connection
        pooled = self.pool(skip) # Downsample
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    Decoder step: Upsample + Concatenate skip + ConvBlock.
    This is where the U-shape "comes back up".
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 for concat with skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatch (padding if needed)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
        return self.conv(x)


class ThermaVisionUNet(nn.Module):
    """
    THERMAVISION-X Core Generator Network.

    Architecture: Standard UNet with 4 encoder levels and 4 decoder levels.
    - Encoder: Progressively reduces spatial dimensions, increases channels
    - Bottleneck: Deepest representation
    - Decoder: Progressively restores spatial dimensions using skip connections
    - Output: HSV color image (3 channels)

    Total parameters: ~5M (lightweight, fits RTX 2050 4GB VRAM)

    Input:  [B, 1, H, W]  — high-frequency structure features from FFT module
    Output: [B, 3, H, W]  — HSV colorized image (H=Hue, S=Saturation, V=Value)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 base_features: int = 32):
        super().__init__()

        # ── ENCODER (squeeze down) ──────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, base_features)       # 1   → 32
        self.enc2 = EncoderBlock(base_features, base_features * 2) # 32  → 64
        self.enc3 = EncoderBlock(base_features * 2, base_features * 4) # 64 → 128
        self.enc4 = EncoderBlock(base_features * 4, base_features * 8) # 128 → 256

        # ── BOTTLENECK (deepest understanding) ─────────────────────────────
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)  # 256 → 512

        # ── DECODER (expand back up with skip connections) ──────────────────
        self.dec4 = DecoderBlock(base_features * 16, base_features * 8)  # 512 → 256
        self.dec3 = DecoderBlock(base_features * 8, base_features * 4)   # 256 → 128
        self.dec2 = DecoderBlock(base_features * 4, base_features * 2)   # 128 → 64
        self.dec1 = DecoderBlock(base_features * 2, base_features)       # 64  → 32

        # ── OUTPUT HEAD ─────────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Conv2d(base_features, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] for HSV channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: High-frequency structure features [B, 1, H, W]
        Returns:
            HSV colorized image [B, 3, H, W] with values in [0, 1]
        """
        # Encoder pass (save skips)
        skip1, x = self.enc1(x)   # [B, 32, H, W]    → [B, 32, H/2, W/2]
        skip2, x = self.enc2(x)   # [B, 64, H/2, W/2] → [B, 64, H/4, W/4]
        skip3, x = self.enc3(x)   # [B, 128, H/4, W/4]→ [B, 128, H/8, W/8]
        skip4, x = self.enc4(x)   # [B, 256, H/8, W/8]→ [B, 256, H/16, W/16]

        # Bottleneck
        x = self.bottleneck(x)    # [B, 512, H/16, W/16]

        # Decoder pass (use skips)
        x = self.dec4(x, skip4)   # [B, 256, H/8, W/8]
        x = self.dec3(x, skip3)   # [B, 128, H/4, W/4]
        x = self.dec2(x, skip2)   # [B, 64, H/2, W/2]
        x = self.dec1(x, skip1)   # [B, 32, H, W]

        # Output
        return self.output_head(x) # [B, 3, H, W] — HSV in [0, 1]


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test — run this file directly to verify the model works
    model = ThermaVisionUNet(in_channels=1, out_channels=3)
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 IR images at 256x256
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Output min/max: {output.min():.3f} / {output.max():.3f}")
    print("UNet model working correctly!")
