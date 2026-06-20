# tests/test_frequency.py
# THERMAVISION-X — Unit Tests for FFT Frequency Decoupling
# Run with: pytest tests/
# Researched & designed by Benad | BAH 2026

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.frequency import FrequencyDecouplingModule
from models.unet import ThermaVisionUNet
from training.loss import ThermaVisionLoss


class TestFrequencyDecoupling:
    """Tests for the core zero-shot innovation module."""

    def setup_method(self):
        self.module = FrequencyDecouplingModule(radius_ratio=0.15)

    def test_output_shapes(self):
        x = torch.randn(2, 1, 256, 256)
        result = self.module(x)
        assert result['high_freq'].shape == x.shape
        assert result['low_freq'].shape  == x.shape
        assert result['masked_input'].shape == x.shape

    def test_frequency_decomposition_sums_to_original(self):
        """high_freq + low_freq should reconstruct the original image."""
        x = torch.randn(1, 1, 128, 128)
        result = self.module(x)
        reconstructed = result['high_freq'] + result['low_freq']
        assert torch.allclose(x, reconstructed, atol=1e-4), \
            "Frequency decomposition does not reconstruct original image!"

    def test_different_radius_ratios(self):
        """Module should work for any radius ratio in (0, 1)."""
        x = torch.randn(1, 1, 128, 128)
        for r in [0.05, 0.15, 0.30, 0.50]:
            mod = FrequencyDecouplingModule(radius_ratio=r)
            result = mod(x)
            assert result['masked_input'].shape == x.shape

    def test_batch_independence(self):
        """Results should be the same for a single image regardless of batch."""
        x_single = torch.randn(1, 1, 128, 128)
        x_batch  = x_single.expand(4, -1, -1, -1)
        mod = FrequencyDecouplingModule(radius_ratio=0.15)
        r_single = mod(x_single)
        r_batch  = mod(x_batch)
        assert torch.allclose(r_single['high_freq'],
                               r_batch['high_freq'][0:1], atol=1e-4)


class TestUNet:
    """Tests for the UNet backbone."""

    def setup_method(self):
        self.model = ThermaVisionUNet(in_channels=1, out_channels=3)

    def test_output_shape(self):
        x = torch.randn(2, 1, 256, 256)
        out = self.model(x)
        assert out.shape == (2, 3, 256, 256), f"Expected (2,3,256,256) got {out.shape}"

    def test_output_range(self):
        """Output should be in [0, 1] due to Sigmoid activation."""
        x = torch.randn(4, 1, 128, 128)
        out = self.model(x)
        assert out.min() >= 0.0, "Output below 0!"
        assert out.max() <= 1.0, "Output above 1!"

    def test_different_input_sizes(self):
        """Should handle different image sizes."""
        for size in [128, 256]:
            x = torch.randn(1, 1, size, size)
            out = self.model(x)
            assert out.shape == (1, 3, size, size)

    def test_param_count(self):
        """Model should stay under 10M parameters."""
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert params < 10_000_000, f"Too many params: {params:,}"
        print(f"\nModel has {params:,} parameters ({'✅ under 10M target'})")


class TestLoss:
    """Tests for the physics-informed loss function."""

    def setup_method(self):
        self.criterion = ThermaVisionLoss()

    def test_loss_returns_dict(self):
        pred   = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        losses = self.criterion(pred, target)
        assert 'total' in losses
        assert 'recon' in losses
        assert 'planck' in losses

    def test_loss_is_positive(self):
        pred   = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        losses = self.criterion(pred, target)
        assert losses['total'].item() >= 0.0

    def test_physics_loss_with_temperature(self):
        pred   = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        temp   = torch.rand(2, 1, 64, 64)
        losses = self.criterion(pred, target, temperature=temp)
        assert losses['planck'].item() >= 0.0
        assert losses['stefan'].item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
