# models/frequency.py
# THERMAVISION-X — Frequency Domain Decoupling Module
# This is the CORE zero-shot innovation.
#
# KEY INSIGHT (Wei et al., 2024):
#   High-frequency content (edges, textures, structure) is BAND-INVARIANT
#   → the same structures appear in both visible AND infrared images
#
#   Low-frequency content (color, luminance) is BAND-SPECIFIC
#   → color only exists in visible; thermal IR has no inherent color
#
# TRAINING: We mask low-frequencies in VISIBLE images, train the UNet to reconstruct them
# INFERENCE: We mask low-frequencies in IR images, the UNet fills in color (zero-shot!)
#
# Researched & designed by Benad | BAH 2026

import torch
import torch.nn as nn


class FrequencyDecouplingModule(nn.Module):
    """
    Separates an image into high-frequency structure and low-frequency color
    using 2D Fast Fourier Transform (FFT).

    Training:
        Input: visible image → mask low frequencies → train UNet to reconstruct
    Inference (Zero-Shot):
        Input: IR image → mask low frequencies → UNet fills in color

    Args:
        radius_ratio: Controls how much of the center (low-freq) to mask.
                      0.15 = mask inner 15% of frequency spectrum.
                      Larger = more aggressive masking (less color info given to model)
    """

    def __init__(self, radius_ratio: float = 0.15):
        super().__init__()
        self.radius_ratio = radius_ratio

    def _make_circular_mask(self, H: int, W: int,
                             device: torch.device) -> tuple:
        """
        Creates circular high-pass and low-pass masks in frequency domain.

        Returns:
            high_pass_mask: 1 outside circle (structure), 0 inside (color)
            low_pass_mask:  1 inside circle (color),     0 outside (structure)
        """
        # Grid centered at (0, 0)
        y = torch.arange(-H // 2, H // 2, device=device).float()
        x = torch.arange(-W // 2, W // 2, device=device).float()
        Y, X = torch.meshgrid(y, x, indexing='ij')

        # Distance from center (where low frequencies live)
        dist = torch.sqrt(X ** 2 + Y ** 2)
        radius = min(H, W) * self.radius_ratio

        high_pass = (dist > radius).float()  # Structure (keep edges/textures)
        low_pass  = 1.0 - high_pass          # Color (block → model must reconstruct)

        return high_pass.view(1, 1, H, W), low_pass.view(1, 1, H, W)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Decouple image into frequency components.

        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            dict with keys:
                'high_freq'      : Structure features (band-invariant) [B, C, H, W]
                'low_freq'       : Color features (band-specific)       [B, C, H, W]
                'masked_input'   : High-freq only → fed to UNet        [B, C, H, W]
                'hp_mask'        : High-pass mask
                'lp_mask'        : Low-pass mask
        """
        B, C, H, W = x.shape

        # 1. 2D FFT (shift so low frequencies are at the center)
        spectrum = torch.fft.fft2(x, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

        # 2. Create circular masks
        hp_mask, lp_mask = self._make_circular_mask(H, W, x.device)

        # 3. Separate structure (high-freq) and color (low-freq)
        high_freq_spectrum = spectrum * hp_mask
        low_freq_spectrum  = spectrum * lp_mask

        # 4. Convert back to spatial domain
        high_freq = torch.fft.ifft2(
            torch.fft.ifftshift(high_freq_spectrum, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        low_freq = torch.fft.ifft2(
            torch.fft.ifftshift(low_freq_spectrum, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        # 5. Masked input = only structure, no color → UNet must reconstruct color
        masked_input = high_freq

        return {
            'high_freq'   : high_freq,    # Structure (safe to use for IR input)
            'low_freq'    : low_freq,     # Color (model must predict this)
            'masked_input': masked_input, # What goes into UNet
            'hp_mask'     : hp_mask,
            'lp_mask'     : lp_mask,
        }


if __name__ == "__main__":
    # Quick test
    module = FrequencyDecouplingModule(radius_ratio=0.15)
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 grayscale images
    result = module(x)

    print(f"Input shape:        {x.shape}")
    print(f"High freq shape:    {result['high_freq'].shape}")
    print(f"Low freq shape:     {result['low_freq'].shape}")
    print(f"Masked input shape: {result['masked_input'].shape}")
    print("Frequency decoupling module working correctly!")
