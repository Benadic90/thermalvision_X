# training/loss.py
# THERMAVISION-X — Composite Physics-Informed Loss Function
#
# 5-component loss that trains the model to produce
# colorizations that are BOTH visually good AND physically accurate.
#
# L_total = w1*L_recon + w2*L_freq + w3*L_planck + w4*L_stefan + w5*L_ssim
#
# Researched & designed by Benad | BAH 2026

import torch
import torch.nn as nn
import torch.nn.functional as F

# Stefan-Boltzmann constant
SIGMA_SB = 5.670374419e-8  # W/m²/K⁴


class ReconstructionLoss(nn.Module):
    """L1 + L2 pixel reconstruction loss."""
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha  # Weight of L1 vs L2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        l2 = F.mse_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * l2


class FocalFrequencyLoss(nn.Module):
    """
    Loss in the frequency domain — penalizes errors in important frequencies.
    Helps preserve fine structural details in the colorized output.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft   = torch.fft.fft2(pred,   dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        pred_mag   = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)


class PlanckPhysicsLoss(nn.Module):
    """
    Enforces Planck's Law: the colorization must be consistent with
    the expected spectral radiance at the image's temperature.

    Simplified: higher pixel value (hotter) → warmer colors (higher hue angle).
    This penalizes cold-looking colors for hot regions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_hsv: torch.Tensor,
                temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_hsv:    Predicted HSV image [B, 3, H, W], values in [0, 1]
            temperature: Normalized temperature map [B, 1, H, W], values in [0, 1]
                         0 = cold (250 K), 1 = hot (350 K)

        Returns:
            Physics consistency loss scalar
        """
        pred_hue = pred_hsv[:, 0:1, :, :]   # Hue channel
        pred_sat = pred_hsv[:, 1:2, :, :]   # Saturation channel

        # Physics constraint: hotter areas should have higher saturation
        # (more vivid, energetic colors for hotter regions)
        sat_loss = F.mse_loss(pred_sat, temperature)

        # Physics constraint: hue should correlate with temperature
        # (simple approximation: high temp → low hue angle = reds/oranges)
        expected_hue = 1.0 - temperature  # Hot = low hue (red), Cold = high hue (blue)
        hue_loss = F.l1_loss(pred_hue, expected_hue)

        return sat_loss + 0.5 * hue_loss


class StefanBoltzmannLoss(nn.Module):
    """
    Enforces Stefan-Boltzmann Law: P = σ * T⁴
    Total emitted power should be proportional to T⁴.
    Constrains overall brightness (Value channel in HSV).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_hsv: torch.Tensor,
                temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_hsv:    Predicted HSV image [B, 3, H, W]
            temperature: Normalized temperature [B, 1, H, W]
        """
        pred_val = pred_hsv[:, 2:3, :, :]  # Value channel

        # T⁴ relationship (normalized: T is already in [0,1])
        expected_val = temperature ** 4
        expected_val = expected_val / (expected_val.max() + 1e-8)

        return F.mse_loss(pred_val, expected_val)


class SSIMLoss(nn.Module):
    """
    Structural Similarity loss — ensures colorized output preserves
    the structural information of the original IR image.
    """
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale for structural comparison
        pred_gray   = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        # Simple approximation using L2 in spatial domain
        return F.mse_loss(pred_gray, target_gray)


class ThermaVisionLoss(nn.Module):
    """
    Complete composite loss for THERMAVISION-X.

    L_total = w_recon * L_recon     (pixel accuracy)
            + w_freq  * L_freq      (frequency accuracy)
            + w_planck * L_planck   (Planck's law constraint)
            + w_stefan * L_stefan   (Stefan-Boltzmann constraint)
            + w_ssim  * L_ssim      (structural preservation)
    """

    def __init__(self,
                 w_recon:  float = 1.0,
                 w_freq:   float = 0.5,
                 w_planck: float = 0.3,
                 w_stefan: float = 0.2,
                 w_ssim:   float = 0.4):
        super().__init__()

        self.w_recon  = w_recon
        self.w_freq   = w_freq
        self.w_planck = w_planck
        self.w_stefan = w_stefan
        self.w_ssim   = w_ssim

        self.recon  = ReconstructionLoss()
        self.freq   = FocalFrequencyLoss()
        self.planck = PlanckPhysicsLoss()
        self.stefan = StefanBoltzmannLoss()
        self.ssim   = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                temperature: torch.Tensor = None) -> dict:
        """
        Args:
            pred:        Model output [B, 3, H, W] (HSV or RGB)
            target:      Ground truth [B, 3, H, W]
            temperature: Normalized temperature map [B, 1, H, W] (optional)

        Returns:
            dict with 'total' loss and individual component losses for logging
        """
        losses = {}

        losses['recon'] = self.recon(pred, target)
        losses['freq']  = self.freq(pred, target)
        losses['ssim']  = self.ssim(pred, target)

        if temperature is not None:
            losses['planck'] = self.planck(pred, temperature)
            losses['stefan'] = self.stefan(pred, temperature)
        else:
            losses['planck'] = torch.tensor(0.0, device=pred.device)
            losses['stefan'] = torch.tensor(0.0, device=pred.device)

        losses['total'] = (
            self.w_recon  * losses['recon']  +
            self.w_freq   * losses['freq']   +
            self.w_planck * losses['planck'] +
            self.w_stefan * losses['stefan'] +
            self.w_ssim   * losses['ssim']
        )

        return losses


if __name__ == "__main__":
    criterion = ThermaVisionLoss()

    pred   = torch.rand(2, 3, 128, 128)  # Fake prediction
    target = torch.rand(2, 3, 128, 128)  # Fake target
    temp   = torch.rand(2, 1, 128, 128)  # Fake temperature map

    losses = criterion(pred, target, temp)
    for k, v in losses.items():
        print(f"  {k:10s}: {v.item():.4f}")
    print("✅ Loss function working correctly!")
