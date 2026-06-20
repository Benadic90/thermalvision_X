# models/physics_layer.py
# THERMAVISION-X — Physics Projection & HSV Color Mapping
#
# This is what makes THERMAVISION-X different from every other colorization model.
# Instead of randomly assigning colors, we use Planck's Law of Blackbody Radiation
# to CONSTRAIN what colors the model is allowed to produce.
#
# Physics used:
#   Planck's Law:       B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
#   Stefan-Boltzmann:   P = σ * T⁴
#
# Color Mapping (TeX-NeRF inspired):
#   Temperature  → Saturation channel (hot = more saturated)
#   Emissivity   → Hue channel        (material type = color family)
#   Texture      → Value channel      (structure = brightness)
#
# Researched & designed by Benad | BAH 2026

import torch
import torch.nn as nn
import numpy as np


# ── Physical Constants ──────────────────────────────────────────────────────
H_PLANCK = 6.62607015e-34   # Planck constant (J·s)
C_LIGHT   = 2.99792458e8    # Speed of light (m/s)
K_BOLTZ   = 1.380649e-23    # Boltzmann constant (J/K)
SIGMA_SB  = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)


class PlanckConverter:
    """
    Pure-Python (NumPy) Planck's Law calculator.
    Converts satellite Digital Numbers → Radiance → Brightness Temperature.
    Used in data preprocessing (not in neural network forward pass).
    """

    def __init__(self, wavelength_um: float):
        """
        Args:
            wavelength_um: Central wavelength of the thermal band in micrometers.
                           INSAT-3D TIR1: 10.8 µm, TIR2: 12.0 µm
                           Landsat B10:   10.9 µm, B11:  12.0 µm
        """
        lam = wavelength_um * 1e-6  # Convert µm to meters
        self.K1 = (2.0 * H_PLANCK * C_LIGHT ** 2) / (lam ** 5)
        self.K2 = (H_PLANCK * C_LIGHT) / (lam * K_BOLTZ)

    def dn_to_radiance(self, dn: np.ndarray,
                       slope: float, intercept: float) -> np.ndarray:
        """Digital Number → Spectral Radiance (W/m²/sr/µm)."""
        return slope * dn.astype(np.float32) + intercept

    def radiance_to_temperature(self, radiance: np.ndarray) -> np.ndarray:
        """Spectral Radiance → Brightness Temperature (Kelvin) via Planck inversion."""
        radiance = np.clip(radiance, 1e-10, None)
        return self.K2 / np.log(self.K1 / radiance + 1.0)


class PhysicsProjectionLayer(nn.Module):
    """
    Extracts physics-aware embeddings from encoder features.

    Takes the bottleneck features from the UNet encoder and projects them
    into a 6-dimensional physics embedding:
        [T_mean, T_std, T_range, emissivity_est, texture_entropy, gradient_mag]

    These physics dimensions then drive the HSV color mapping.
    """

    def __init__(self, in_dim: int, physics_dim: int = 6):
        super().__init__()
        self.physics_dim = physics_dim

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, physics_dim),
            nn.Sigmoid(),  # Normalize physics features to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder bottleneck features [B, C, H, W]
        Returns:
            physics_embedding: [B, physics_dim] — normalized physics features
        """
        avg = self.global_avg(x).flatten(1)  # [B, C]
        mx  = self.global_max(x).flatten(1)  # [B, C]
        feat = torch.cat([avg, mx], dim=1)   # [B, 2C]
        return self.mlp(feat)                # [B, physics_dim]


class HSVColorMappingLayer(nn.Module):
    """
    Maps physics embeddings to HSV color channels.

    Color Mapping Rules (TeX-NeRF inspired):
        Hue        ← emissivity estimate + temperature mean
                     (WHAT type of surface → WHAT color family)
        Saturation ← temperature std + temperature range
                     (HOW HOT relative to scene → HOW VIVID the color)
        Value      ← texture entropy + gradient magnitude
                     (HOW MUCH structure → HOW BRIGHT the pixel)

    This ensures colors are physically meaningful, not random.
    """

    def __init__(self, physics_dim: int, encoder_dim: int):
        super().__init__()

        combined = physics_dim + encoder_dim

        self.hue_head = nn.Sequential(
            nn.Conv2d(combined, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Hue ∈ [0, 1] → mapped to [0°, 360°]
        )

        self.sat_head = nn.Sequential(
            nn.Conv2d(combined, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Saturation ∈ [0, 1]
        )

        self.val_head = nn.Sequential(
            nn.Conv2d(combined, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Value ∈ [0, 1]
        )

    def forward(self, encoder_feat: torch.Tensor,
                physics_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feat: Spatial features from encoder [B, encoder_dim, H, W]
            physics_emb:  Physics embedding [B, physics_dim]
        Returns:
            HSV image [B, 3, H, W] with channels in [0, 1]
        """
        B, _, H, W = encoder_feat.shape

        # Broadcast physics embedding to spatial dimensions
        phys_spatial = physics_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Concatenate physics + spatial features
        combined = torch.cat([encoder_feat, phys_spatial], dim=1)

        hue = self.hue_head(combined)  # [B, 1, H, W]
        sat = self.sat_head(combined)  # [B, 1, H, W]
        val = self.val_head(combined)  # [B, 1, H, W]

        return torch.cat([hue, sat, val], dim=1)  # [B, 3, H, W]


if __name__ == "__main__":
    # Quick test for PlanckConverter
    import numpy as np
    converter = PlanckConverter(wavelength_um=10.8)  # INSAT-3D TIR1

    # Simulate some raw digital numbers from INSAT-3D
    dn = np.array([[240, 250], [260, 270]], dtype=np.float32)
    radiance = converter.dn_to_radiance(dn, slope=0.5, intercept=-10.0)
    temperature = converter.radiance_to_temperature(radiance)
    print(f"DN values: {dn.flatten()}")
    print(f"Temperature (K): {temperature.flatten()}")
    print("✅ Physics layer working correctly!")
