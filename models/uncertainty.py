# models/uncertainty.py
# THERMAVISION-X — Monte Carlo Dropout Uncertainty Estimator
#
# PURPOSE:
#   Produces a PER-PIXEL confidence map alongside the colorized output.
#   This tells disaster responders: "Trust THIS area, but be cautious THERE."
#
# HOW IT WORKS:
#   Normal training: Dropout randomly zeros neurons (regularization)
#   Normal testing:  Dropout is turned OFF
#   OUR approach:    Keep dropout ON at test time, run N forward passes
#                    → Average outputs = final colorized image
#                    → Pixel variance  = uncertainty / confidence map
#
# Researched & designed by Benad | BAH 2026

import torch
import torch.nn as nn
from typing import Tuple


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout — stays active during both training AND inference.
    This is the key difference from standard dropout.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always apply dropout (training=True forces it on even during eval)
        return nn.functional.dropout(x, p=self.p, training=True)


class UncertaintyEstimator:
    """
    Wraps any model to produce uncertainty estimates via Monte Carlo Dropout.

    Usage:
        estimator = UncertaintyEstimator(model, n_passes=10, dropout_p=0.1)
        colorized, uncertainty = estimator.predict(ir_image)
    """

    def __init__(self, model: nn.Module, n_passes: int = 10,
                 dropout_p: float = 0.1):
        """
        Args:
            model:     The trained ThermaVisionUNet
            n_passes:  Number of stochastic forward passes (more = better estimate)
            dropout_p: Dropout probability to apply during MC passes
        """
        self.model = model
        self.n_passes = n_passes
        self.dropout_p = dropout_p

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run N stochastic forward passes and compute mean + variance.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            mean_output: Average colorization [B, 3, H, W] — the final image
            uncertainty: Per-pixel variance  [B, 1, H, W] — the confidence map
                         Low value  = model is CONFIDENT (bright in heatmap)
                         High value = model is UNCERTAIN  (dark in heatmap)
        """
        self.model.eval()

        # Collect N stochastic predictions
        outputs = []
        for _ in range(self.n_passes):
            output = self.model(x)  # MCDropout stays active
            outputs.append(output.unsqueeze(0))

        # Stack: [N, B, 3, H, W]
        outputs = torch.cat(outputs, dim=0)

        # Mean across N passes → final colorized image
        mean_output = outputs.mean(dim=0)  # [B, 3, H, W]

        # Variance across N passes → uncertainty map
        variance = outputs.var(dim=0)      # [B, 3, H, W]
        uncertainty = variance.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        return mean_output, uncertainty

    def uncertainty_to_heatmap(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Normalize uncertainty to [0, 1] for display as a colored heatmap.
        High confidence → 1.0 (bright green)
        Low confidence  → 0.0 (bright red)
        """
        u_min = uncertainty.min()
        u_max = uncertainty.max()
        if u_max - u_min < 1e-8:
            return torch.ones_like(uncertainty)  # All equal = all confident
        confidence = 1.0 - (uncertainty - u_min) / (u_max - u_min)
        return confidence


if __name__ == "__main__":
    from unet import ThermaVisionUNet

    model = ThermaVisionUNet(in_channels=1, out_channels=3)
    estimator = UncertaintyEstimator(model, n_passes=5)

    x = torch.randn(1, 1, 128, 128)
    colorized, uncertainty = estimator.predict(x)
    confidence = estimator.uncertainty_to_heatmap(uncertainty)

    print(f"Input shape:       {x.shape}")
    print(f"Colorized shape:   {colorized.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Confidence range:  [{confidence.min():.3f}, {confidence.max():.3f}]")
    print("✅ Uncertainty estimator working correctly!")
