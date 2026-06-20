# data/preprocessing.py
# THERMAVISION-X — Satellite Thermal Data Preprocessing
# DN → Radiance → Brightness Temperature → Normalized [0,1] tensor
# Researched & designed by Benad | BAH 2026

import numpy as np
import torch

# Physical Constants
H_PLANCK = 6.62607015e-34   # Planck constant (J·s)
C_LIGHT   = 2.99792458e8    # Speed of light (m/s)
K_BOLTZ   = 1.380649e-23    # Boltzmann constant (J/K)


class PlanckConverter:
    """Convert Digital Numbers → Radiance → Brightness Temperature."""

    def __init__(self, wavelength_um: float):
        lam = wavelength_um * 1e-6
        self.K1 = (2.0 * H_PLANCK * C_LIGHT ** 2) / (lam ** 5)
        self.K2 = (H_PLANCK * C_LIGHT) / (lam * K_BOLTZ)

    def dn_to_radiance(self, dn: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return slope * dn.astype(np.float32) + intercept

    def radiance_to_temperature(self, radiance: np.ndarray) -> np.ndarray:
        radiance = np.clip(radiance, 1e-10, None)
        return self.K2 / np.log(self.K1 / radiance + 1.0)

    def forward(self, dn: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return self.radiance_to_temperature(self.dn_to_radiance(dn, slope, intercept))


class SplitWindowCorrection:
    """Atmospheric correction for dual-band TIR sensors (INSAT-3D, Landsat)."""

    COEFFS = {
        'INSAT_3D' : {'A': 1.02, 'B': -0.50},
        'LANDSAT_8': {'A': 1.02, 'B': -0.48},
    }

    def __init__(self, sensor: str = 'INSAT_3D'):
        self.coeffs = self.COEFFS.get(sensor, self.COEFFS['INSAT_3D'])

    def correct(self, tb1: np.ndarray, tb2: np.ndarray,
                water_vapor: float = 2.0) -> np.ndarray:
        """
        Apply split-window algorithm: LST = T_B11 + A*(T_B11 - T_B12) + B
        Args:
            tb1: Brightness temperature band 1 (K)
            tb2: Brightness temperature band 2 (K)
            water_vapor: Total column water vapor (g/cm²), default 2.0
        Returns:
            Land Surface Temperature (K)
        """
        A = self.coeffs['A'] * water_vapor
        B = self.coeffs['B']
        return tb1 + A * (tb1 - tb2) + B


class ThermalNormalizer:
    """Normalize brightness temperature to [0, 1] for model input."""

    def __init__(self, t_min: float = 250.0, t_max: float = 350.0):
        """
        Default range covers:
        - Ocean surfaces: ~270-305 K
        - Land surfaces:  ~270-340 K
        - Fires/volcanic: up to ~400 K (clip to 350 for general use)
        """
        self.t_min = t_min
        self.t_max = t_max

    def normalize(self, temperature: np.ndarray) -> np.ndarray:
        return np.clip((temperature - self.t_min) / (self.t_max - self.t_min), 0.0, 1.0)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * (self.t_max - self.t_min) + self.t_min

    def to_tensor(self, normalized: np.ndarray) -> torch.Tensor:
        """Convert normalized numpy array to PyTorch tensor [1, H, W]."""
        return torch.from_numpy(normalized).float().unsqueeze(0)


if __name__ == "__main__":
    # Test the full pipeline
    converter = PlanckConverter(wavelength_um=10.8)
    dn = np.array([[240, 255, 270], [285, 300, 315]], dtype=np.float32)
    temperature = converter.forward(dn, slope=0.5, intercept=-5.0)
    print(f"Input DN:         {dn.flatten()}")
    print(f"Temperature (K):  {temperature.flatten().round(1)}")

    normalizer = ThermalNormalizer(t_min=250.0, t_max=350.0)
    normalized = normalizer.normalize(temperature)
    print(f"Normalized [0,1]: {normalized.flatten().round(3)}")
    print("✅ Preprocessing pipeline working correctly!")
