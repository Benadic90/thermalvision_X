# THERMAVISION-X: Physics-Guided Zero-Shot Infrared Colorization

## Comprehensive Technical Architecture Document

**Version**: 1.0  
**Hackathon**: ISRO Bharatiya Antariksh Hackathon 2026  
**Track**: AI/ML for Earth Observation  
**Researched by**: Benad  
**Date**: June 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Physics-Guided Zero-Shot Colorization Architecture](#3-physics-guided-zero-shot-colorization-architecture)
4. [Training Strategy (Self-Supervised)](#4-training-strategy-self-supervised)
5. [Inference Pipeline (Zero-Shot)](#5-inference-pipeline-zero-shot)
6. [Datasets for Development](#6-datasets-for-development)
7. [Implementation Roadmap (30 Hours)](#7-implementation-roadmap-30-hours)
8. [Technology Stack](#8-technology-stack)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Risk Mitigation](#10-risk-mitigation)
11. [GitHub Repository Structure](#11-github-repository-structure)
12. [References](#12-references)

---

## 1. Executive Summary

### 1.1 Problem Statement

Indian Space Research Organisation (ISRO) captures vast quantities of thermal infrared (IR) imagery through missions like INSAT-3D/3DR, Oceansat-3, and the upcoming TRISHNA mission. These images are monochromatic and lack the intuitive interpretability of color imagery, limiting their utility for:

- Disaster monitoring and response coordination
- Agricultural assessment (FASAL program)
- Urban heat island analysis for smart cities
- Fishery advisory and Potential Fishing Zone (PFZ) identification
- Climate and environmental monitoring

Traditional IR colorization approaches require paired RGB-IR training data that is expensive to acquire and fundamentally unavailable for satellite IR bands. Existing deep learning methods also suffer from slow inference (diffusion models), large model sizes unsuitable for edge deployment, and lack of physics-grounded color mapping that leads to unrealistic outputs.

### 1.2 Our Solution: THERMAVISION-X

THERMAVISION-X is a **physics-guided zero-shot infrared colorization system** that:

- **Requires ZERO IR training data**: Trains exclusively on visible (RGB) images, generalizes to any IR spectral band at inference via frequency-domain feature decoupling
- **Incorporates thermal physics**: Integrates Planck's law, Stefan-Boltzmann equation, and emissivity-texture-color mapping to produce physically meaningful colorizations
- **Runs on edge hardware**: Lightweight Mamba-based architecture (~5M parameters) optimized with TensorRT for 15+ FPS on Jetson Nano-class hardware
- **Provides uncertainty quantification**: Per-pixel confidence maps for mission-critical decision support
- **Supports all ISRO thermal missions**: Works across INSAT-3D/3DR, Oceansat-3, TRISHNA, and future missions without retraining

### 1.3 Key Innovations

| Innovation | Description | Impact |
|---|---|---|
| Physics-Guided HSV Mapping | Temperature→Saturation, Emissivity→Hue, Texture→Value via TeX-NeRF principles | Thermally meaningful colors |
| Frequency Domain Zero-Shot Decoupling | 2D DFT high-pass feature extraction with masked reconstruction | No IR training data needed |
| Lightweight Mamba UNet | ~5M params with full-scale skip connections and cosine contrastive loss | 15+ FPS edge inference |
| Physics-Informed Loss Function | Planck + Stefan-Boltzmann constraints as differentiable loss terms | Physical consistency |
| Monte Carlo Uncertainty | Dropout-based per-pixel confidence estimation | Trustworthy outputs |

### 1.4 Target Performance Specifications

| Metric | Target | Method |
|---|---|---|
| Inference Speed | >= 15 FPS | TensorRT FP16 on Jetson Nano |
| Model Size | < 25 MB | INT8 quantization + pruning |
| PSNR (vs. pseudo-reference) | >= 28 dB | Physics-guided colorization |
| SSIM | >= 0.85 | Full-scale skip connections |
| NIQE (no-reference quality) | <= 4.0 | Self-supervised frequency training |
| Thermal Consistency | >= 0.92 | Grayscale SSIM preservation |

---

## 2. System Architecture Overview

### 2.1 High-Level System Diagram

```
+============================================================================+
|                        THERMAVISION-X SYSTEM OVERVIEW                       |
+============================================================================+

  RAW SATELLITE DATA INPUTS
  =========================
  +------------------+  +------------------+  +------------------+
  |   INSAT-3D/3DR   |  |    Oceansat-3    |  |     TRISHNA      |
  |  TIR: 10.3-12.5  |  |   LWIR: 1080m  |  |  4-band TIR: 57m |
  |   Resolution: 4km  |  |   Resolution     |  |   Resolution     |
  +--------+---------+  +--------+---------+  +--------+---------+
           |                     |                     |
           +---------------------+---------------------+
                                 |
                                 v
  +============================================================+
  |            INPUT PREPROCESSING MODULE (Section 3A)          |
  |  - DN -> Radiance -> Brightness Temperature Conversion      |
  |  - Split-Window Atmospheric Correction                      |
  |  - Normalization & Georeferencing Preservation              |
  +============================================================+
                                 |
                                 v
  +============================================================+
  |         FREQUENCY DOMAIN DECOUPLING MODULE (Section 3B)     |
  |  - 2D DFT/FFT Transformation                                |
  |  - High-Pass Filter (Structure Preservation)                |
  |  - Low-Frequency Masking                                    |
  |  - Frequency-Domain Feature Extraction                      |
  +============================================================+
                                 |
                                 v
  +============================================================+
  |      PHYSICS-INFORMED COLORIZATION GENERATOR (Section 3C)   |
  |                                                             |
  |   +------------------+    +-------------------------+       |
  |   |  Mamba UNet      |    |   Physics Projection    |       |
  |   |  Backbone (~5M)  |<---|   Layer (Planck Law)    |       |
  |   |                  |    |                         |       |
  |   |  - Mamba Blocks  |    |  T -> Saturation        |       |
  |   |  - Skip Connects |    |  e -> Hue               |       |
  |   |  - Full-Scale    |    |  X -> Value             |       |
  |   +--------+---------+    +-------------------------+       |
  |            |                                                |
  |            v                                                |
  |   +------------------+                                      |
  |   |  HSV Color Space |                                      |
  |   |  Mapping Layer   |                                      |
  |   +--------+---------+                                      |
  +============================================================+
                                 |
                                 v
  +============================================================+
  |       UNCERTAINTY QUANTIFICATION MODULE (Section 3D)        |
  |  - Monte Carlo Dropout (T forward passes)                   |
  |  - Per-Pixel Variance -> Confidence Heatmap                 |
  |  - Threshold-Based Reliability Masking                      |
  +============================================================+
                                 |
                                 v
  +============================================================+
  |        OUTPUT POSTPROCESSING MODULE (Section 3F)            |
  |  - HSV -> RGB Conversion                                    |
  |  - Quality Assessment (NIQE, Thermal Consistency)           |
  |  - Metadata Preservation (Georeferencing)                   |
  +============================================================+
                                 |
           +---------------------+---------------------+
           |                     |                     |
           v                     v                     v
  +------------------+  +------------------+  +------------------+
  |  Colorized Output |  | Uncertainty Map  |  | Quality Report   |
  |  (RGB GeoTIFF)   |  | (Confidence)     |  | (Metrics JSON)   |
  +------------------+  +------------------+  +------------------+

  EDGE DEPLOYMENT OPTIMIZER (Section 3E)
  =====================================
  +------------------+    +------------------+    +------------------+
  |   ONNX Export    | -> | TensorRT FP16    | -> | INT8 Quantize    |
  |   (IR -> ONNX)   |    | Optimization     |    | Calibration      |
  +------------------+    +------------------+    +------------------+

  DEMO / VISUALIZATION
  ====================
  +------------------+    +------------------+    +------------------+
  |  Streamlit UI    |    | Side-by-Side     |    | Layer Overlay    |
  |  (Interactive)   |    | Comparison       |    | on Map           |
  +------------------+    +------------------+    +------------------+
```

### 2.2 Component Interaction Flow

```
Data Flow Architecture
======================

Phase 1: Data Ingestion (Input Preprocessing)
---------------------------------------------
Raw DN Values -> Radiance (sensor calibration)
Radiance -> Brightness Temperature (Planck inversion)
Brightness Temp -> Atmospheric Correction (Split-Window)
Corrected BT -> Normalize [0, 1] -> Feature Tensor [B, 1, H, W]

Phase 2: Frequency Analysis (Frequency Decoupling)
--------------------------------------------------
Input Tensor -> 2D FFT -> Frequency Spectrum
Frequency Spectrum -> High-Pass Mask -> Structure Features (High Freq)
Frequency Spectrum -> Low-Pass Mask -> Content Features (Low Freq)
Structure Features -> Inverse FFT -> Spatial Domain Feature Maps

Phase 3: Physics-Guided Generation (Colorization Generator)
-----------------------------------------------------------
Structure Features -> Mamba UNet Encoder -> Latent Representations
Latent Representations -> Physics Projection:
    Planck(T, lambda) -> Spectral Radiance Curve
    Radiance Curve -> Temperature Features (statistical moments)
    Temperature Features -> Physics Embedding [B, P, H/8, W/8]
Physics Embedding -> HSV Mapping Layer:
    T_mean -> Saturation channel
    epsilon (emissivity estimate) -> Hue channel  
    Texture gradient magnitude -> Value channel
HSV [B, 3, H, W] -> Decoder (Full-Scale Skip Connections) -> Refined HSV

Phase 4: Uncertainty Estimation
-------------------------------
Refined HSV + MC Dropout (T=10 forward passes) -> Output Distribution
Output Distribution -> Pixel-wise Variance -> Uncertainty Heatmap
Uncertainty -> Threshold -> Confidence Mask

Phase 5: Postprocessing & Output
--------------------------------
Refined HSV -> HSV2RGB -> Colorized Output [B, 3, H, W]
Colorized Output -> Grayscale Conversion -> Thermal SSIM Check
Quality Metrics: NIQE, BRISQUE, PIQE, Thermal Consistency
Metadata: Preserve georeferencing from original satellite data

Phase 6: Edge Deployment (Offline)
----------------------------------
PyTorch Model -> ONNX Export (torch.onnx.export)
ONNX -> TensorRT Engine (trtexec --fp16)
TensorRT -> INT8 Calibration (calibration dataset from visible images)
Benchmark: Jetson Nano @ 15+ FPS target
```

### 2.3 End-to-End Data Pipeline

```
[Satellite IR Image] 
      |
      v
[DN to Radiance] ----(Calibration Coefficients)----> [Radiance]
      |
      v
[Planck Inversion] --(Central Wavelength)---------> [Brightness Temp]
      |
      v
[Split-Window Atmos. Correction] --(Band 1, Band 2)-> [LST]
      |
      v
[Normalize] --------(Per-dataset Min/Max)----------> [Normalized Input]
      |
      v
[2D FFT] ------------------------------------------> [Frequency Domain]
      |
      +--> [High-Pass Filter] --> [Structure Features]
      |
      +--> [Low-Pass Mask] -----> [Content Baseline]
      |
      v
[Mamba UNet + Physics Layer] ----------------------> [HSV Colorized]
      |
      v
[MC Dropout x T] ----------------------------------> [Uncertainty Map]
      |
      v
[HSV to RGB] --------------------------------------> [Final Output]
      |
      +--> [Quality Assessment: NIQE, SSIM, TC]
      +--> [Georeferenced GeoTIFF Export]
      +--> [Confidence Heatmap Export]
```

---

## 3. Physics-Guided Zero-Shot Colorization Architecture

### 3.0 Architecture Philosophy

THERMAVISION-X is built on a **frequency-domain zero-shot colorization** paradigm (inspired by Wei et al. 2024). The core insight is:

> **High-frequency image structure (edges, textures, gradients) is largely spectral-band invariant, while low-frequency content (color, luminance) is band-dependent.**

By decoupling these in the frequency domain and training a generator to reconstruct masked frequency components using only visible (RGB) images, the model learns a **band-agnostic structure-to-color mapping** that transfers zero-shot to any IR band.

The physics guidance ensures that the learned mapping produces **thermally meaningful colors** rather than arbitrary hallucinations, by embedding Planck's law, Stefan-Boltzmann conservation, and the TeX-NeRF color mapping into both the architecture and loss function.

### 3A. Input Preprocessing Module

#### 3A.1 Satellite Thermal Data Ingestion

**Supported ISRO Satellite Formats:**

| Satellite | Sensor | Bands | Format | Resolution |
|---|---|---|---|---|
| INSAT-3D/3DR | VHRS / Imager | 2 TIR (10.3-11.3, 11.5-12.5 um) | HDF5 | 4 km |
| Oceansat-3 | Ocean Color Monitor + | 2 LWIR | HDF5 | 1080 m |
| TRISHNA (2026) | TIR Imager | 4 TIR bands | HDF5 | 57 m |
| Landsat 8/9 | TIRS | 2 TIR (10.6, 11.5 um) | GeoTIFF | 100 m |
| Sentinel-3 | SLSTR | 2 TIR (10.85, 12 um) | NetCDF | 1 km |

**Ingestion Pipeline:**

```python
class SatelliteDataIngestion:
    """
    Unified ingestion pipeline for ISRO and international satellite thermal data.
    """
    
    SUPPORTED_SATELLITES = {
        'INSAT_3D': {
            'format': 'HDF5',
            'bands': {'TIR1': (10.3, 11.3), 'TIR2': (11.5, 12.5)},
            'calibration': 'linear',
            'resolution': 4000,  # meters
        },
        'TRISHNA': {
            'format': 'HDF5',
            'bands': {'TIR1': (8.0, 9.0), 'TIR2': (10.0, 11.0),
                      'TIR3': (11.0, 12.0), 'TIR4': (12.0, 13.0)},
            'calibration': 'nonlinear',
            'resolution': 57,  # meters
        },
        'LANDSAT_8': {
            'format': 'GeoTIFF',
            'bands': {'B10': (10.6, 11.2), 'B11': (11.5, 12.5)},
            'calibration': 'linear',
            'resolution': 100,  # meters
        }
    }
    
    def load_hdf5(self, filepath: str, band: str) -> np.ndarray:
        """Load single thermal band from HDF5 (INSAT format)."""
        with h5py.File(filepath, 'r') as f:
            dn_data = f[f'IMG_{band}'][()]  # Digital Numbers
            cal_coeffs = {
                'slope': f[f'IMG_{band}'].attrs['CALIBRATION_SLOPE'],
                'intercept': f[f'IMG_{band}'].attrs['CALIBRATION_INTERCEPT']
            }
            geo_transform = f['Geolocation'].attrs['GEO_TRANSFORM']
            projection = f['Geolocation'].attrs['PROJECTION']
        return dn_data, cal_coeffs, geo_transform, projection
```

#### 3A.2 DN to Radiance to Brightness Temperature Conversion

**Step 1: DN to Spectral Radiance (Linear Calibration)**

```
L_lambda = Slope * DN + Intercept

Where:
  L_lambda = Spectral radiance (W m^-2 sr^-1 um^-1)
  DN       = Digital Number (raw sensor value)
  Slope, Intercept = Sensor-specific calibration coefficients
```

**Step 2: Spectral Radiance to Brightness Temperature (Planck Inversion)**

Planck's Law for spectral radiance:

```
           2 * h * c^2
B_lambda(T) = -------------------
             lambda^5 * (exp(hc/(lambda*k_B*T)) - 1)

Where:
  h      = 6.62607015 x 10^-34 J s (Planck constant)
  c      = 2.99792458 x 10^8 m/s (Speed of light)
  k_B    = 1.380649 x 10^-23 J/K (Boltzmann constant)
  lambda = Wavelength in meters
  T      = Temperature in Kelvin
```

Inversion (solving for T):

```
                    h * c
T_B = -------------------------------
      lambda * k_B * ln((2hc^2)/(L_lambda * lambda^5) + 1)

Simplified (using sensor-specific K1, K2 coefficients):

             K2
T_B = ------------------
      ln(K1/L_lambda + 1)
```

**Implementation:**

```python
class PlanckConverter:
    """Digital Number -> Radiance -> Brightness Temperature conversion."""
    
    # Physical constants
    H = 6.62607015e-34    # Planck constant (J s)
    C = 2.99792458e8      # Speed of light (m/s)
    K_B = 1.380649e-23    # Boltzmann constant (J/K)
    
    def __init__(self, wavelength_um: float):
        """
        Args:
            wavelength_um: Central wavelength in micrometers
        """
        self.wavelength_m = wavelength_um * 1e-6
        # Pre-compute K1, K2 for this band
        self.K1 = (2 * self.H * self.C**2) / (self.wavelength_m**5)
        self.K2 = (self.H * self.C) / (self.wavelength_m * self.K_B)
    
    def dn_to_radiance(self, dn: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Convert Digital Numbers to spectral radiance."""
        return slope * dn.astype(np.float32) + intercept
    
    def radiance_to_temperature(self, radiance: np.ndarray) -> np.ndarray:
        """Convert spectral radiance to brightness temperature via Planck inversion."""
        # Avoid log of zero
        radiance = np.clip(radiance, 1e-10, None)
        temperature = self.K2 / np.log(self.K1 / radiance + 1.0)
        return temperature
    
    def forward(self, dn: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Complete DN -> Temperature pipeline."""
        radiance = self.dn_to_radiance(dn, slope, intercept)
        temperature = self.radiance_to_temperature(radiance)
        return temperature
```

#### 3A.3 Split-Window Atmospheric Correction

The split-window algorithm corrects for atmospheric effects using two adjacent TIR bands:

```
LST = T_B11 + A * (T_B11 - T_B12) + B

Where:
  T_B11  = Brightness temperature of band 11 (um)
  T_B12  = Brightness temperature of band 12 (um)
  A, B   = Algorithm coefficients (sensor-specific)
```

The coefficients A and B depend on:
- Near-surface air temperature
- Atmospheric water vapor content
- Surface emissivity in both bands

**Implementation (ISRO INSAT-3D specific):**

```python
class SplitWindowCorrection:
    """
    Split-window atmospheric correction for dual-TIR sensors.
    Applicable to: INSAT-3D/3DR, Landsat TIRS, ASTER.
    """
    
    def __init__(self, sensor_type: str = 'INSAT_3D'):
        self.sensor = sensor_type
        self.coeffs = self._get_coefficients()
    
    def _get_coefficients(self):
        """Get sensor-specific SWA coefficients."""
        coeffs = {
            'INSAT_3D': {
                'A_water_vapor': 1.02,
                'B_offset': -0.50,
                'epsilon_ratio': 0.97,
            },
            'LANDSAT_8': {
                'A_water_vapor': 1.02,
                'B_offset': -0.48,
                'epsilon_ratio': 0.97,
            }
        }
        return coeffs.get(self.sensor, coeffs['INSAT_3D'])
    
    def correct(self, tb_band1: np.ndarray, tb_band2: np.ndarray,
                water_vapor: float = 2.0) -> np.ndarray:
        """
        Apply split-window atmospheric correction.
        
        Args:
            tb_band1: Brightness temperature of first TIR band (K)
            tb_band2: Brightness temperature of second TIR band (K)
            water_vapor: Total column water vapor (g/cm^2), default 2.0
        
        Returns:
            lst: Land Surface Temperature (K)
        """
        c = self.coeffs
        A = c['A_water_vapor'] * water_vapor
        B = c['B_offset']
        
        lst = tb_band1 + A * (tb_band1 - tb_band2) + B
        return lst
```

#### 3A.4 Normalization

```python
class ThermalNormalizer:
    """Normalize brightness temperature to [0, 1] range."""
    
    def __init__(self, t_min: float = 250.0, t_max: float = 350.0):
        """
        Typical Earth surface temperature range:
        - Ocean surfaces: ~270-305 K
        - Land surfaces: ~270-340 K  
        - Volcanic/fires: up to ~400 K
        Default covers vast majority of Earth observation scenarios.
        """
        self.t_min = t_min
        self.t_max = t_max
    
    def normalize(self, temperature: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        return np.clip((temperature - self.t_min) / (self.t_max - self.t_min), 0.0, 1.0)
    
    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Convert back to Kelvin."""
        return normalized * (self.t_max - self.t_min) + self.t_min
```

#### 3A.5 Complete Preprocessing Pipeline

```python
class PreprocessingPipeline:
    """Complete end-to-end preprocessing for satellite thermal data."""
    
    def __init__(self, satellite: str, band: str):
        config = SatelliteDataIngestion.SUPPORTED_SATELLITES[satellite]
        wavelength = np.mean(config['bands'][band])  # Central wavelength
        
        self.ingestion = SatelliteDataIngestion()
        self.planck = PlanckConverter(wavelength)
        self.atm_correction = SplitWindowCorrection(satellite)
        self.normalizer = ThermalNormalizer()
    
    def process(self, filepath: str, band: str, band_pair: str = None) -> dict:
        """
        Full preprocessing pipeline.
        
        Returns:
            dict with keys:
                - 'normalized_input': Tensor [1, H, W] ready for model
                - 'temperature': Original brightness temperature [H, W]
                - 'lst': Land surface temperature (if dual-band) [H, W]
                - 'metadata': Georeferencing and sensor info
        """
        # 1. Ingest
        dn, cal, geo_transform, projection = self.ingestion.load(filepath, band)
        
        # 2. DN -> Radiance -> Temperature
        radiance = self.planck.dn_to_radiance(dn, cal['slope'], cal['intercept'])
        tb = self.planck.radiance_to_temperature(radiance)
        
        # 3. Atmospheric correction (if dual-band available)
        lst = tb
        if band_pair:
            dn2, cal2, _, _ = self.ingestion.load(filepath, band_pair)
            radiance2 = self.planck.dn_to_radiance(dn2, cal2['slope'], cal2['intercept'])
            tb2 = self.planck.radiance_to_temperature(radiance2)
            lst = self.atm_correction.correct(tb, tb2)
        
        # 4. Normalize
        normalized = self.normalizer.normalize(lst)
        
        # 5. To tensor
        input_tensor = torch.from_numpy(normalized).float().unsqueeze(0)  # [1, H, W]
        
        return {
            'normalized_input': input_tensor,
            'temperature': tb,
            'lst': lst,
            'metadata': {
                'geo_transform': geo_transform,
                'projection': projection,
                'satellite': self.satellite,
                'band': band,
            }
        }
```

### 3B. Frequency Domain Decoupling Module

#### 3B.1 Architecture Rationale

The frequency domain decoupling module is the **core enabler of zero-shot cross-spectral colorization**. Based on the insight from Wei et al. (2024) that high-frequency structures are band-invariant while low-frequency content is band-specific:

- **High-frequency components** (edges, textures, fine details) contain structural information that transfers across spectral bands
- **Low-frequency components** (smooth regions, color/luminance) are band-specific and need to be hallucinated by the generator

By training the model to reconstruct masked low-frequency content from high-frequency structure using only visible (RGB) images, the model learns a **generalizable structure-to-coloration mapping** that applies zero-shot to any IR band.

#### 3B.2 2D DFT/FFT Transformation

```python
class FrequencyDecouplingModule(nn.Module):
    """
    Frequency domain feature decoupling for zero-shot cross-spectral colorization.
    
    Splits input into structure-preserving high-frequency features and
    content-bearing low-frequency components.
    """
    
    def __init__(self, radius_ratio: float = 0.15):
        """
        Args:
            radius_ratio: Ratio of image dimensions to use as filter cutoff.
                         0.15 means frequencies within 15% of center are masked.
        """
        super().__init__()
        self.radius_ratio = radius_ratio
    
    def _create_circular_mask(self, h: int, w: int, radius: float, 
                               device: torch.device) -> torch.Tensor:
        """
        Create circular high-pass/low-pass mask in frequency domain.
        
        Returns:
            mask: Tensor [H, W] where 1 = pass, 0 = block
        """
        # Create coordinate grid centered at (0, 0)
        y = torch.arange(-h//2, h//2, device=device).float()
        x = torch.arange(-w//2, w//2, device=device).float()
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Distance from center
        distance = torch.sqrt(X**2 + Y**2)
        
        # Circular mask
        high_pass_mask = (distance > radius).float()
        low_pass_mask = 1.0 - high_pass_mask
        
        return high_pass_mask, low_pass_mask
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Decouple input into frequency components.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            dict with keys:
                - 'spectrum': Full 2D FFT magnitude [B, C, H, W]
                - 'phase': Full 2D FFT phase [B, C, H, W]
                - 'high_freq': High-frequency components (structure) [B, C, H, W]
                - 'low_freq': Low-frequency components (content) [B, C, H, W]
                - 'masked_spectrum': Spectrum with low-freq masked [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 2D FFT (shift to center low frequencies)
        spectrum = torch.fft.fft2(x, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        
        # Get magnitude and phase
        magnitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        
        # 2. Create circular masks
        radius = min(H, W) * self.radius_ratio
        hp_mask, lp_mask = self._create_circular_mask(H, W, radius, x.device)
        hp_mask = hp_mask.view(1, 1, H, W)
        lp_mask = lp_mask.view(1, 1, H, W)
        
        # 3. Apply masks to get frequency components
        high_freq_spectrum = spectrum * hp_mask
        low_freq_spectrum = spectrum * lp_mask
        
        # 4. Convert back to spatial domain
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_freq_spectrum, dim=(-2, -1)), 
                                      dim=(-2, -1)).real
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(low_freq_spectrum, dim=(-2, -1)), 
                                     dim=(-2, -1)).real
        
        # 5. Create masked spectrum (for reconstruction training)
        # Mask out low frequencies - model must reconstruct these
        masked_spectrum = spectrum * hp_mask  # Only keep high frequencies
        
        return {
            'spectrum': magnitude,
            'phase': phase,
            'high_freq': high_freq,        # Structure features
            'low_freq': low_freq,          # Content features
            'masked_spectrum': masked_spectrum,  # For training target
            'hp_mask': hp_mask,
            'lp_mask': lp_mask,
        }
```

#### 3B.3 Masked Frequency Reconstruction (Training Objective)

During training on visible images:

```
1. Take visible image I_vis
2. Apply FFT -> get spectrum S_vis
3. Apply high-pass mask -> get S_masked (low-freq removed)
4. Feed S_masked (spatial domain) through Generator G
5. Generator output G(S_masked) should reconstruct I_vis
6. Loss = ||G(S_masked) - I_vis|| in both spatial and frequency domain
```

This trains G to reconstruct missing low-frequency content from high-frequency structure. Since structure is band-invariant, G generalizes to IR at test time.

### 3C. Physics-Informed Colorization Generator

#### 3C.1 Architecture Overview

```
+====================================================================+
|              PHYSICS-INFORMED COLORIZATION GENERATOR               |
|                    Target: ~5M Parameters                          |
+====================================================================+

Input: Structure Features [B, 1, H, W] (from Frequency Decoupling)
       |
       v
+-------------------------------------------------------+
|              MAMBA UNet Encoder                       |
|                                                       |
|  Level 0: [B, 1,  H,   W  ]                           |
|     |                                                |
|     v                                                |
|  Stem: Conv3x3 -> [B, 32, H,   W  ]                   |
|     |                                                |
|     v                                                |
|  Mamba Block 0: SS2D + Conv -> [B, 32, H,   W  ]     |
|     |                                                |
|     v                                                |
|  Downsample 0: ConvStride2 -> [B, 64, H/2, W/2]       |
|     |                                                |
|     v                                                |
|  Mamba Block 1: SS2D + Conv -> [B, 64, H/2, W/2]     |
|     |                                                |
|     v                                                |
|  Downsample 1: ConvStride2 -> [B, 128, H/4, W/4]     |
|     |                                                |
|     v                                                |
|  Mamba Block 2: SS2D + Conv -> [B, 128, H/4, W/4]    |
|     |                                                |
|     v                                                |
|  Downsample 2: ConvStride2 -> [B, 256, H/8, W/8]     |
|     |                                                |
|     v                                                |
|  Mamba Block 3: SS2D + Conv -> [B, 256, H/8, W/8]    |
+-------------------------------------------------------+
       |
       v
+-------------------------------------------------------+
|           PHYSICS PROJECTION LAYER                    |
|                                                       |
|  Input: Encoder features [B, 256, H/8, W/8]           |
|                                                       |
|  1. Global Average Pooling -> Feature vector [B, 256] |
|  2. Planck Feature Extractor:                         |
|     - FC(256 -> 128) + ReLU                           |
|     - FC(128 -> 64) + ReLU                            |
|     - FC(64 -> P) where P = physics dims              |
|                                                       |
|  Physics Dimensions (P=6):                            |
|    - T_mean: Mean temperature                         |
|    - T_std: Temperature std dev                       |
|    - T_range: Temperature range                       |
|    - epsilon_est: Estimated emissivity                |
|    - texture_entropy: Spatial texture complexity      |
|    - gradient_magnitude: Edge strength                |
+-------------------------------------------------------+
       |
       +--------------------------------------------------+
       |                                                  |
       v                                                  v
+-------------------------------------------------------+
|              HSV COLOR MAPPING LAYER                  |
|                                                       |
|  Physics Embedding [B, P, 1, 1] -> Broadcast          |
|       |                                               |
|       v                                               |
|  Hue Channel Generator:                               |
|    Input: epsilon_est + T_mean + Encoder feat         |
|    -> Conv1x1 -> Hue [B, 1, H/8, W/8]                 |
|    Range: [0, 1] (mapped to [0, 360] later)            |
|                                                       |
|  Saturation Channel Generator:                        |
|    Input: T_std + T_range + Encoder feat              |
|    -> Conv1x1 -> Saturation [B, 1, H/8, W/8]          |
|    Range: [0, 1]                                       |
|    Physical meaning: Higher temp variance -> more     |
|    saturated colors (greater thermal contrast)        |
|                                                       |
|  Value Channel Generator:                             |
|    Input: texture_entropy + gradient + Encoder feat   |
|    -> Conv1x1 -> Value [B, 1, H/8, W/8]               |
|    Range: [0, 1]                                       |
|    Physical meaning: Texture/structure -> brightness  |
+-------------------------------------------------------+
       |
       v
+-------------------------------------------------------+
|              MAMBA UNet Decoder                       |
|                                                       |
|  HSV [B, 3, H/8, W/8] (concat from mapping layer)     |
|     |                                                |
|     v                                                |
|  Upsample 2: ConvTranspose2 -> [B, 128, H/4, W/4]    |
|     |                                                |
|     +--> Skip Connect from Encoder L2 (Full-Scale)    |
|     |                                                |
|     v                                                |
|  Mamba Block 5: SS2D + Conv -> [B, 128, H/4, W/4]    |
|     |                                                |
|     v                                                |
|  Upsample 1: ConvTranspose2 -> [B, 64, H/2, W/2]     |
|     |                                                |
|     +--> Skip Connect from Encoder L1 (Full-Scale)    |
|     |                                                |
|     v                                                |
|  Mamba Block 6: SS2D + Conv -> [B, 64, H/2, W/2]     |
|     |                                                |
|     v                                                |
|  Upsample 0: ConvTranspose2 -> [B, 32, H, W]         |
|     |                                                |
|     +--> Skip Connect from Encoder L0 (Full-Scale)    |
|     |                                                |
|     v                                                |
|  Mamba Block 7: SS2D + Conv -> [B, 32, H, W]         |
|     |                                                |
|     v                                                |
|  Output Head: Conv3x3 -> HSV [B, 3, H, W]             |
+-------------------------------------------------------+
       |
       v
  Output: HSV Colorized Image [B, 3, H, W]
```

#### 3C.2 Lightweight Mamba UNet Backbone (~5M params)

```python
class MambaBlock(nn.Module):
    """
    Selective Scan Mamba Block with 2D spatial scanning (SS2D).
    Uses Mamba2 (SSD) for efficient long-range dependency modeling.
    """
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # Mamba2 (SSD) layer - selective state space model
        self.mixer = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=max(dim // 8, 8),
        )
        
        # Local convolution for fine-grained features
        self.conv_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        
        # Drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Local convolution branch
        conv_out = self.conv_local(x)
        
        # Mamba branch (process as flattened sequence)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        x_norm = self.norm(x_flat)
        mamba_out = self.mixer(x_norm)  # [B, HW, C]
        mamba_out = mamba_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Combine with residual
        output = x + self.drop_path(mamba_out + conv_out)
        return output


class FullScaleSkipConnection(nn.Module):
    """
    Full-scale skip connections as in CCLGAN (2025).
    Aggregates features from all encoder scales at each decoder level.
    """
    
    def __init__(self, encoder_dims: list, decoder_dim: int):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        
        # Projection layers for each encoder level
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ed, decoder_dim, kernel_size=1),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
                if i > 0 else nn.Identity(),
            )
            for i, ed in enumerate(encoder_dims)
        ])
        
        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_dim * len(encoder_dims), decoder_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
    
    def forward(self, encoder_features: list, decoder_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: List of [B, C_i, H_i, W_i] from each encoder level
            decoder_feature: [B, decoder_dim, H_d, W_d]
        Returns:
            Fused features [B, decoder_dim, H_d, W_d]
        """
        # Project and upsample all encoder features to decoder resolution
        proj_features = [proj(feat) for proj, feat in zip(self.projs, encoder_features)]
        
        # Concatenate all projected features with decoder feature
        all_features = proj_features + [decoder_feature]
        fused = torch.cat(all_features, dim=1)
        
        # Fuse
        output = self.fusion(fused)
        return output


class PhysicsProjectionLayer(nn.Module):
    """
    Projects encoder features into physics-aware embeddings.
    Extracts: temperature statistics, emissivity estimates, texture features.
    """
    
    def __init__(self, in_dim: int, physics_dim: int = 6):
        super().__init__()
        self.physics_dim = physics_dim
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        
        # Planck feature extractor
        self.planck_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, physics_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] encoder features
        Returns:
            physics_embedding: [B, P, 1, 1] physics-aware embedding
        """
        # Global statistics
        avg_feat = self.global_pool(x).flatten(1)  # [B, C]
        max_feat = self.global_max(x).flatten(1)   # [B, C]
        global_feat = torch.cat([avg_feat, max_feat], dim=1)  # [B, 2C]
        
        # Physics projection
        physics_vec = self.planck_mlp(global_feat)  # [B, P]
        physics_embedding = physics_vec.view(-1, self.physics_dim, 1, 1)
        
        return physics_embedding


class HSVColorMappingLayer(nn.Module):
    """
    Maps physics embeddings to HSV color channels.
    Follows TeX-NeRF principles: T->Saturation, e->Hue, X->Value.
    """
    
    def __init__(self, physics_dim: int, encoder_dim: int):
        super().__init__()
        
        # Hue channel (driven by emissivity and temperature)
        self.hue_gen = nn.Sequential(
            nn.Conv2d(physics_dim + encoder_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Hue in [0, 1]
        )
        
        # Saturation channel (driven by temperature variance)
        self.sat_gen = nn.Sequential(
            nn.Conv2d(physics_dim + encoder_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Saturation in [0, 1]
        )
        
        # Value channel (driven by texture/structure)
        self.val_gen = nn.Sequential(
            nn.Conv2d(physics_dim + encoder_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Value in [0, 1]
        )
    
    def forward(self, physics_emb: torch.Tensor, 
                encoder_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_emb: [B, P, 1, 1] - broadcasted to feature resolution
            encoder_feat: [B, C, H, W]
        Returns:
            hsv: [B, 3, H, W] color in HSV space
        """
        B, C, H, W = encoder_feat.shape
        
        # Broadcast physics embedding to spatial resolution
        physics_broadcast = physics_emb.expand(B, -1, H, W)
        
        # Concatenate physics with encoder features
        combined = torch.cat([physics_broadcast, encoder_feat], dim=1)
        
        # Generate HSV channels
        hue = self.hue_gen(combined)      # [B, 1, H, W]
        saturation = self.sat_gen(combined)  # [B, 1, H, W]
        value = self.val_gen(combined)    # [B, 1, H, W]
        
        hsv = torch.cat([hue, saturation, value], dim=1)  # [B, 3, H, W]
        return hsv


class ThermaVisionGenerator(nn.Module):
    """
    Complete THERMAVISION-X Physics-Guided Colorization Generator.
    Target: ~5M parameters.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 base_dim: int = 32, num_levels: int = 4):
        super().__init__()
        
        # --- Encoder ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )
        
        encoder_dims = []
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        dim = base_dim
        for i in range(num_levels):
            self.encoder_blocks.append(MambaBlock(dim))
            encoder_dims.append(dim)
            
            if i < num_levels - 1:
                self.downsample.append(
                    nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
                )
                dim *= 2
        
        # --- Physics Projection ---
        self.physics_proj = PhysicsProjectionLayer(in_dim=dim, physics_dim=6)
        
        # --- HSV Color Mapping ---
        self.hsv_mapping = HSVColorMappingLayer(physics_dim=6, encoder_dim=dim)
        
        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(num_levels - 1, 0, -1):
            # Full-scale skip connections
            self.skip_connections.append(
                FullScaleSkipConnection(encoder_dims[:i+1], dim // 2)
            )
            
            self.upsample.append(
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1)
            )
            self.decoder_blocks.append(MambaBlock(dim // 2))
            dim //= 2
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: [B, 1, H, W] preprocessed thermal input
        
        Returns:
            dict with:
                - 'hsv': [B, 3, H, W] colorized output in HSV
                - 'rgb': [B, 3, H, W] colorized output in RGB
                - 'physics_embedding': [B, P, 1, 1] physics features
        """
        # Encoder
        x = self.stem(x)
        encoder_features = []
        
        for i, (enc_block, down) in enumerate(zip(self.encoder_blocks, 
                                                     self.downsample + [None])):
            x = enc_block(x)
            encoder_features.append(x)
            if down is not None:
                x = down(x)
        
        # Physics projection
        physics_emb = self.physics_proj(x)
        
        # HSV color mapping
        hsv = self.hsv_mapping(physics_emb, x)
        
        # Decoder with full-scale skip connections
        for i, (up, dec_block, skip) in enumerate(zip(
            self.upsample, self.decoder_blocks, self.skip_connections)):
            x = up(hsv if i == 0 else x)
            x = skip(encoder_features[:len(encoder_features)-i], x)
            x = dec_block(x)
        
        # Output
        output = self.output_head(x)  # [B, 3, H, W] in HSV
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb(output)
        
        return {
            'hsv': output,
            'rgb': rgb,
            'physics_embedding': physics_emb,
        }


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV [B, 3, H, W] to RGB [B, 3, H, W].
    H: [0, 1], S: [0, 1], V: [0, 1]
    """
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    
    h = h * 6.0  # Scale to [0, 6]
    i = torch.floor(h).long()
    f = h - i.float()
    
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    # Handle each hue sector
    i = i % 6
    
    # Create RGB channels
    zeros = torch.zeros_like(v)
    
    # Use gather for clean implementation
    conditions = [
        (i == 0, torch.cat([v, t, p], dim=1)),
        (i == 1, torch.cat([q, v, p], dim=1)),
        (i == 2, torch.cat([p, v, t], dim=1)),
        (i == 3, torch.cat([p, q, v], dim=1)),
        (i == 4, torch.cat([t, p, v], dim=1)),
        (i == 5, torch.cat([v, p, q], dim=1)),
    ]
    
    rgb = torch.zeros_like(hsv)
    for condition, value in conditions:
        mask = condition.float()
        rgb = rgb + mask * value
    
    return rgb
```

#### 3C.3 Parameter Budget

| Component | Parameters | Notes |
|---|---|---|
| Stem (Conv) | ~0.3K | 1 -> 32 channels |
| Encoder Mamba Blocks (x4) | ~2.5M | SS2D + local conv |
| Downsample Layers (x3) | ~0.2M | Strided convolutions |
| Physics Projection Layer | ~0.1M | MLP for physics features |
| HSV Color Mapping Layer | ~0.2M | 3x Conv1x1 generators |
| Decoder Mamba Blocks (x3) | ~1.2M | With skip connections |
| Upsample Layers (x3) | ~0.5M | Transposed convolutions |
| Full-Scale Skip Connections | ~0.15M | Feature fusion |
| Output Head | ~9K | 32 -> 3 channels |
| **Total** | **~5.0M** | **~20 MB in FP32** |

#### 3C.4 Cosine Contrastive Loss for Feature Alignment

Inspired by CCLGAN (2025), cosine contrastive loss aligns features from different scales:

```python
class CosineContrastiveLoss(nn.Module):
    """
    Cosine contrastive loss for cross-scale feature alignment.
    Pulls positive feature pairs together, pushes negatives apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors from different scales
                      Each: [B, C_i, H_i, W_i]
        
        Returns:
            loss: Scalar contrastive loss
        """
        total_loss = 0.0
        num_pairs = 0
        
        # Sample pairs of features from different scales
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_i = F.adaptive_avg_pool2d(features[i], 1).flatten(1)  # [B, C_i]
                feat_j = F.adaptive_avg_pool2d(features[j], 1).flatten(1)  # [B, C_j]
                
                # Project to common dimension
                dim = min(feat_i.shape[1], feat_j.shape[1])
                feat_i = F.normalize(feat_i[:, :dim], dim=1)
                feat_j = F.normalize(feat_j[:, :dim], dim=1)
                
                # Cosine similarity
                similarity = torch.sum(feat_i * feat_j, dim=1) / self.temperature
                
                # Positive pairs: same batch index
                labels = torch.arange(feat_i.shape[0], device=feat_i.device)
                
                # Cross-entropy style loss
                logits = similarity / self.temperature
                loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
```

### 3D. Uncertainty Quantification Module

#### 3D.1 Monte Carlo Dropout for Confidence Estimation

At inference time, we apply Monte Carlo Dropout (Gal & Ghahramani, 2016) to estimate prediction uncertainty:

```python
class UncertaintyQuantifier:
    """
    Monte Carlo Dropout-based uncertainty estimation.
    Runs T stochastic forward passes and computes per-pixel variance.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10,
                 dropout_rate: float = 0.1):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    @torch.no_grad()
    def estimate(self, x: torch.Tensor) -> dict:
        """
        Run MC Dropout to estimate uncertainty.
        
        Args:
            x: [B, 1, H, W] input thermal image
        
        Returns:
            dict with:
                - 'mean_output': [B, 3, H, W] mean colorized output
                - 'uncertainty': [B, 1, H, W] per-pixel uncertainty
                - 'confidence': [B, 1, H, W] confidence map (1 - normalized uncertainty)
                - 'samples': [T, B, 3, H, W] all samples
        """
        self.model.train()  # Keep dropout active!
        
        # Collect samples
        samples = []
        for _ in range(self.num_samples):
            output = self.model(x)['rgb']
            samples.append(output)
        
        samples = torch.stack(samples, dim=0)  # [T, B, 3, H, W]
        
        # Compute statistics
        mean_output = samples.mean(dim=0)  # [B, 3, H, W]
        variance = samples.var(dim=0)  # [B, 3, H, W]
        
        # Per-pixel uncertainty (average across channels)
        uncertainty = variance.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize to [0, 1] for confidence
        uncertainty_norm = (uncertainty - uncertainty.min()) / \
                           (uncertainty.max() - uncertainty.min() + 1e-8)
        confidence = 1.0 - uncertainty_norm
        
        return {
            'mean_output': mean_output,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'samples': samples,
        }
    
    def apply_confidence_threshold(self, output: torch.Tensor, 
                                   confidence: torch.Tensor,
                                   threshold: float = 0.5) -> torch.Tensor:
        """
        Mask output where confidence is below threshold.
        Low-confidence regions are flagged for human review.
        """
        mask = (confidence >= threshold).float()
        flagged_output = output * mask  # Zero out low-confidence regions
        return flagged_output, mask
```

#### 3D.2 Uncertainty-Guided Decision Support

```
Confidence Level    | Action
--------------------|-------------------------------------------
> 0.8 (High)        | Auto-approve colorization
0.5 - 0.8 (Medium)  | Include in output with warning annotation
< 0.5 (Low)         | Flag for expert review, show grayscale fallback
```

### 3E. Edge Deployment Optimizer

#### 3E.1 ONNX Export

```python
class ONNXExporter:
    """Export PyTorch model to ONNX format for cross-platform deployment."""
    
    def export(self, model: nn.Module, save_path: str,
               input_shape: tuple = (1, 1, 256, 256)):
        """
        Export model to ONNX.
        
        Args:
            model: ThermaVisionGenerator instance
            save_path: Path to save .onnx file
            input_shape: Example input shape [B, C, H, W]
        """
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['thermal_input'],
            output_names=['hsv_output', 'rgb_output'],
            dynamic_axes={
                'thermal_input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'rgb_output': {0: 'batch_size', 2: 'height', 3: 'width'},
            },
            # Use DFT as ATen op for FFT operations
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        
        print(f"ONNX model exported to {save_path}")
        print(f"Model size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
```

#### 3E.2 TensorRT Optimization

```bash
#!/bin/bash
# TensorRT FP16 Optimization Script

MODEL_PATH="thermavision_x.onnx"
ENGINE_PATH="thermavision_x_fp16.engine"
INPUT_SHAPE="1x1x256x256"  # Batch x Channel x Height x Width

trtexec \
    --onnx=${MODEL_PATH} \
    --saveEngine=${ENGINE_PATH} \
    --fp16 \
    --workspace=4096 \
    --minShapes=thermal_input:1x1x128x128 \
    --optShapes=thermal_input:1x1x256x256 \
    --maxShapes=thermal_input:1x1x512x512 \
    --verbose \
    --dumpProfile \
    --exportProfile=thermavision_x_profile.json

echo "TensorRT engine saved to ${ENGINE_PATH}"
```

#### 3E.3 INT8 Quantization with Calibration

```python
class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator using visible images (no IR data needed).
    Calibrates on synthetic IR-like inputs derived from grayscale visible images.
    """
    
    def __init__(self, calibration_dir: str, batch_size: int = 8,
                 input_shape: tuple = (1, 256, 256)):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        
        # Load calibration images (visible -> synthetic IR)
        self.images = self._load_calibration_data(calibration_dir)
        self.current_index = 0
        
        # Allocate device memory
        self.device_input = cuda.mem_alloc(
            batch_size * np.prod(input_shape) * np.dtype(np.float32).itemsize
        )
    
    def _load_calibration_data(self, calib_dir: str) -> np.ndarray:
        """Load and preprocess calibration images (visible -> synthetic IR)."""
        image_paths = glob(os.path.join(calib_dir, "*.jpg")) + \
                      glob(os.path.join(calib_dir, "*.png"))
        
        calib_data = []
        for path in image_paths[:1000]:  # Max 1000 calibration images
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.input_shape[1:])
            img = img.astype(np.float32) / 255.0
            calib_data.append(img)
        
        return np.stack(calib_data)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None
        
        batch = self.images[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        cuda.memcpy_htod(self.device_input, batch.ravel())
        return [self.device_input]
    
    def read_calibration_cache(self):
        if os.path.exists("calibration.cache"):
            with open("calibration.cache", "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)


# Build INT8 TensorRT engine
def build_int8_engine(onnx_path: str, calibrator: INT8Calibrator) -> trt.ICudaEngine:
    """Build optimized INT8 TensorRT engine."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # Build config
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    # Profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("thermal_input", 
                      min=(1, 1, 128, 128),
                      opt=(1, 1, 256, 256),
                      max=(1, 1, 512, 512))
    config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_engine(network, config)
    return engine
```

#### 3E.4 Benchmark Target

| Metric | Target | Platform | Notes |
|---|---|---|---|
| FP32 Inference | 5-8 FPS | Jetson Nano 4GB | Baseline PyTorch |
| FP16 TensorRT | 12-15 FPS | Jetson Nano 4GB | TensorRT optimized |
| INT8 TensorRT | 18-22 FPS | Jetson Nano 4GB | Quantized + optimized |
| INT8 TensorRT | 40-60 FPS | Jetson Orin NX | Higher-end edge platform |
| Model Size (FP32) | ~20 MB | All | ~5M parameters |
| Model Size (FP16) | ~10 MB | All | Half precision |
| Model Size (INT8) | ~5 MB | All | Quantized |

### 3F. Output Postprocessing Module

#### 3F.1 Color Space Conversion & Quality Assessment

```python
class OutputPostprocessor:
    """
    Postprocessing pipeline: HSV->RGB conversion, quality assessment,
    metadata preservation.
    """
    
    def __init__(self):
        self.quality_metrics = QualityMetrics()
    
    def process(self, model_output: dict, input_metadata: dict) -> dict:
        """
        Full postprocessing pipeline.
        
        Args:
            model_output: Dict from ThermaVisionGenerator.forward()
            input_metadata: Georeferencing and sensor metadata
        
        Returns:
            dict with final outputs and quality metrics
        """
        hsv = model_output['hsv']  # [B, 3, H, W]
        rgb = model_output['rgb']  # [B, 3, H, W]
        
        # Quality assessment
        metrics = self.quality_metrics.compute(hsv, rgb)
        
        # Preserve metadata
        output = {
            'colorized_rgb': rgb,
            'colorized_hsv': hsv,
            'quality_metrics': metrics,
            'metadata': input_metadata,
        }
        
        return output


class QualityMetrics:
    """
    No-reference quality assessment for IR colorization.
    Since ground truth color is unavailable, we use no-reference metrics
    and thermal consistency checks.
    """
    
    def __init__(self):
        pass
    
    def compute(self, hsv: torch.Tensor, rgb: torch.Tensor) -> dict:
        """
        Compute all quality metrics.
        
        Returns:
            dict with metric values
        """
        metrics = {}
        
        # 1. No-reference perceptual quality
        rgb_np = rgb.detach().cpu().numpy()
        
        # NIQE (Natural Image Quality Evaluator)
        # Lower is better (typical range: 2-10)
        try:
            from pyiqa import create_metric
            niqe_metric = create_metric('niqe')
            metrics['NIQE'] = float(niqe_metric(rgb).mean())
        except:
            metrics['NIQE'] = 'N/A (install pyiqa)'
        
        # 2. Thermal Consistency
        # Convert RGB back to grayscale, compare with input thermal structure
        rgb_gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        # This should be computed against input thermal in pipeline
        
        # 3. Color distribution statistics
        metrics['mean_hue'] = float(hsv[:, 0].mean())
        metrics['mean_saturation'] = float(hsv[:, 1].mean())
        metrics['mean_value'] = float(hsv[:, 2].mean())
        metrics['saturation_std'] = float(hsv[:, 1].std())
        
        # 4. Sharpness (gradient magnitude)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=rgb.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=rgb.device).view(1, 1, 3, 3)
        
        gray = rgb.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(gx**2 + gy**2)
        metrics['sharpness'] = float(gradient_magnitude.mean())
        
        return metrics
```

#### 3F.2 GeoTIFF Export with Metadata Preservation

```python
class GeoTIFFExporter:
    """Export colorized output as GeoTIFF with preserved georeferencing."""
    
    def export(self, rgb: np.ndarray, metadata: dict, output_path: str):
        """
        Export colorized image as GeoTIFF.
        
        Args:
            rgb: [H, W, 3] uint8 RGB image
            metadata: Dict with 'geo_transform' and 'projection'
            output_path: Path to save GeoTIFF
        """
        H, W, C = rgb.shape
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, W, H, C, gdal.GDT_Byte)
        
        # Set georeferencing
        out_ds.SetGeoTransform(metadata['geo_transform'])
        out_ds.SetProjection(metadata['projection'])
        
        # Write bands
        for i in range(C):
            out_band = out_ds.GetRasterBand(i + 1)
            out_band.WriteArray(rgb[:, :, i])
            out_band.SetNoDataValue(0)
        
        out_ds = None  # Close and flush
        print(f"GeoTIFF exported to {output_path}")
```

---

## 4. Training Strategy (Self-Supervised)

### 4.1 Zero-Shot Training Paradigm

**Core Principle**: The model is trained **exclusively on visible (RGB) images** and generalizes zero-shot to any IR band at inference.

```
Training (Only Visible Images):
================================
Visible Image I_vis [B, 3, H, W]
  |
  v
Grayscale Conversion I_gray [B, 1, H, W]
  |
  v
Frequency Decoupling:
  - 2D FFT -> Spectrum
  - High-Pass Mask -> Structure Features (keep)
  - Low-Pass Mask -> Content Features (mask out)
  |
  v
Input to Generator: Masked Spectrum (spatial domain) [B, 1, H, W]
  |
  v
Generator G -> Predicted Colorized Output G(I) [B, 3, H, W]
  |
  v
Loss = L_total(G(I), I_vis)  # Compare with original visible image

Inference (Any IR Band - Zero-Shot):
=====================================
IR Image I_ir [B, 1, H, W] (from any satellite, any band)
  |
  v
Same Frequency Decoupling -> Structure Features
  |
  v
Same Generator G (weights frozen from visible training)
  |
  v
Colorized Output G(I_ir) [B, 3, H, W]
  |
  +--> Works because high-frequency structure is band-invariant!
```

### 4.2 Training Data Pipeline

```python
class ZeroShotTrainingDataset(Dataset):
    """
    Self-supervised training dataset using ONLY visible (RGB) images.
    Creates synthetic masked frequency inputs for reconstruction training.
    """
    
    def __init__(self, image_dir: str, image_size: int = 256,
                 mask_ratio: float = 0.15):
        self.image_paths = glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + \
                          glob(os.path.join(image_dir, "**/*.png"), recursive=True)
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.freq_decouple = FrequencyDecouplingModule(radius_ratio=mask_ratio)
        
        # Data augmentation
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    def __getitem__(self, idx: int) -> dict:
        # Load visible image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        img_tensor = TF.to_tensor(img)  # [3, H, W]
        
        # Convert to grayscale (simulating IR input)
        gray = img_tensor.mean(dim=0, keepdim=True)  # [1, H, W]
        
        # Frequency decoupling
        freq_out = self.freq_decouple(gray.unsqueeze(0))
        
        # Input: high-frequency structure only (masked)
        structure_input = freq_out['high_freq'].squeeze(0)  # [1, H, W]
        
        # Target: original visible image
        target = img_tensor  # [3, H, W]
        
        return {
            'input': structure_input,      # Structure features [1, H, W]
            'target': target,              # Original RGB [3, H, W]
            'grayscale': gray,             # Grayscale reference [1, H, W]
        }
```

### 4.3 Multi-Component Loss Function

The total loss combines five components to ensure both visual quality and physical consistency:

```
L_total = lambda_1 * L_recon + lambda_2 * L_focal_freq + lambda_3 * L_planck 
        + lambda_4 * L_stefan + lambda_5 * L_cosine
```

#### 4.3.1 L_recon: Spatial and Frequency Domain Reconstruction Loss

```python
class ReconstructionLoss(nn.Module):
    """
    Combined spatial (L1) and frequency domain (L2) reconstruction loss.
    """
    
    def __init__(self, spatial_weight: float = 1.0, freq_weight: float = 0.5):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted RGB [B, 3, H, W]
            target: Target RGB [B, 3, H, W]
        """
        # Spatial L1 loss
        spatial_loss = self.l1_loss(pred, target)
        
        # Frequency domain L2 loss (on each channel)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        total = self.spatial_weight * spatial_loss + self.freq_weight * freq_loss
        return total, {'spatial': spatial_loss, 'frequency': freq_loss}
```

#### 4.3.2 L_focal_freq: Focal Frequency Loss

```python
class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss from "Real-ESRGAN" (Wang et al., 2021).
    Adaptively focuses on "hard" frequencies that are difficult to reconstruct.
    
    This is particularly important for IR colorization because:
    - High frequencies (edges) are well-preserved from structure
    - Low frequencies (smooth color regions) need more attention
    """
    
    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # Focusing parameter
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
        """
        # FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Amplitude
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # Amplitude distance
        amp_distance = torch.abs(pred_amp - target_amp)
        
        # Weighted distance (focus on hard frequencies)
        weighted_distance = amp_distance ** self.alpha
        
        # Mean over batch and spatial dimensions
        loss = weighted_distance.mean()
        
        return loss * self.loss_weight
```

#### 4.3.3 L_planck: Planck's Law Temperature Consistency

```python
class PlanckConsistencyLoss(nn.Module):
    """
    Physics-informed loss enforcing Planck's law temperature consistency.
    
    For a physically plausible colorization:
    - Regions with similar brightness temperature should have similar
      color distributions (hue/saturation statistics)
    - Temperature gradients should correlate with color gradients
    """
    
    def __init__(self, loss_weight: float = 0.1, lambda_center: float = 11.0):
        """
        Args:
            lambda_center: Central wavelength in um (default: 11 um, typical TIR)
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.lambda_m = lambda_center * 1e-6
        
        # Pre-compute Planck constants for this wavelength
        h = 6.62607015e-34
        c = 2.99792458e8
        k_B = 1.380649e-23
        self.C1 = 2 * h * c**2  # First radiation constant
        self.C2 = h * c / k_B   # hc/k_B (second radiation constant / lambda)
    
    def planck_radiance(self, T: torch.Tensor) -> torch.Tensor:
        """
        Compute Planck spectral radiance for given temperature.
        B_lambda(T) = (2hc^2/lambda^5) / (exp(hc/lambda*k_B*T) - 1)
        
        Args:
            T: Temperature [B, 1, H, W] in range [0, 1] (normalized)
        
        Returns:
            Radiance [B, 1, H, W]
        """
        # Denormalize temperature (assuming normalized to [250, 350] K)
        T_kelvin = T * 100 + 250  # [B, 1, H, W] in Kelvin
        
        lambda_5 = self.lambda_m**5
        exponent = self.C2 / (self.lambda_m * T_kelvin)
        
        radiance = self.C1 / (lambda_5 * (torch.exp(exponent) - 1.0))
        return radiance
    
    def forward(self, pred_hsv: torch.Tensor, 
                input_thermal: torch.Tensor) -> torch.Tensor:
        """
        Enforce that colorization respects Planck's law.
        
        Args:
            pred_hsv: Predicted HSV [B, 3, H, W]
            input_thermal: Input thermal (normalized) [B, 1, H, W]
        
        Returns:
            Planck consistency loss
        """
        # Compute Planck radiance from input temperature
        radiance = self.planck_radiance(input_thermal)
        
        # Radiance should correlate with saturation (more radiance = more color)
        saturation = pred_hsv[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Compute correlation between radiance gradient and saturation gradient
        rad_grad_x = torch.abs(radiance[:, :, :, 1:] - radiance[:, :, :, :-1])
        sat_grad_x = torch.abs(saturation[:, :, :, 1:] - saturation[:, :, :, :-1])
        
        rad_grad_y = torch.abs(radiance[:, :, 1:, :] - radiance[:, :, :-1, :])
        sat_grad_y = torch.abs(saturation[:, :, 1:, :] - saturation[:, :, :-1, :])
        
        # Correlation loss: radiance gradients should align with saturation gradients
        grad_correlation_x = F.mse_loss(
            rad_grad_x / (rad_grad_x.mean() + 1e-8),
            sat_grad_x / (sat_grad_x.mean() + 1e-8)
        )
        grad_correlation_y = F.mse_loss(
            rad_grad_y / (rad_grad_y.mean() + 1e-8),
            sat_grad_y / (sat_grad_y.mean() + 1e-8)
        )
        
        loss = (grad_correlation_x + grad_correlation_y) / 2.0
        return loss * self.loss_weight
```

#### 4.3.4 L_stefan: Stefan-Boltzmann Energy Conservation

```python
class StefanBoltzmannLoss(nn.Module):
    """
    Physics-informed loss enforcing Stefan-Boltzmann energy conservation.
    
    P = epsilon * sigma * A * T^4
    
    The total "energy" represented in the colorized image should be 
    proportional to the fourth power of temperature, preserving the
    physical relationship between temperature and radiated power.
    """
    
    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
    
    def forward(self, pred_hsv: torch.Tensor,
                input_thermal: torch.Tensor) -> torch.Tensor:
        """
        Enforce Stefan-Boltzmann energy conservation.
        
        Args:
            pred_hsv: Predicted HSV [B, 3, H, W]
            input_thermal: Input thermal (normalized) [B, 1, H, W]
        
        Returns:
            Stefan-Boltzmann consistency loss
        """
        B = pred_hsv.shape[0]
        
        # Denormalize temperature to Kelvin [250, 350]
        T = input_thermal * 100 + 250  # [B, 1, H, W]
        
        # Compute Stefan-Boltzmann power per pixel: P ~ T^4
        power = T**4  # [B, 1, H, W]
        
        # Compute total "color energy" = saturation * value (intensity)
        color_energy = pred_hsv[:, 1:2, :, :] * pred_hsv[:, 2:3, :, :]  # S * V
        
        # Global energy conservation: mean color energy should correlate with mean power
        mean_power = power.view(B, -1).mean(dim=1)  # [B]
        mean_color_energy = color_energy.view(B, -1).mean(dim=1)  # [B]
        
        # Normalize both to [0, 1] within batch
        power_norm = (mean_power - mean_power.min()) / (mean_power.max() - mean_power.min() + 1e-8)
        energy_norm = (mean_color_energy - mean_color_energy.min()) / \
                      (mean_color_energy.max() - mean_color_energy.min() + 1e-8)
        
        # MSE loss between normalized distributions
        loss = F.mse_loss(energy_norm, power_norm)
        
        return loss * self.loss_weight
```

#### 4.3.5 L_cosine: Cosine Contrastive Feature Alignment

```python
class CosineContrastiveAlignmentLoss(nn.Module):
    """
    Cosine contrastive loss for cross-scale and cross-domain feature alignment.
    Ensures features from different encoder levels are semantically consistent.
    """
    
    def __init__(self, loss_weight: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors from different scales [B, C_i, H_i, W_i]
        
        Returns:
            Cosine contrastive loss
        """
        ccl = CosineContrastiveLoss(temperature=self.temperature)
        loss = ccl(features)
        return loss * self.loss_weight
```

### 4.4 Combined Loss Function

```python
class ThermaVisionLoss(nn.Module):
    """
    Combined loss function for THERMAVISION-X training.
    All components are differentiable and computed on GPU.
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        default_config = {
            'lambda_recon': 1.0,      # Spatial + freq reconstruction
            'lambda_focal_freq': 0.5, # Focal frequency loss
            'lambda_planck': 0.1,     # Planck's law consistency
            'lambda_stefan': 0.05,    # Stefan-Boltzmann conservation
            'lambda_cosine': 0.1,     # Cosine contrastive alignment
        }
        self.config = config or default_config
        
        # Initialize loss components
        self.recon_loss = ReconstructionLoss()
        self.focal_freq_loss = FocalFrequencyLoss()
        self.planck_loss = PlanckConsistencyLoss()
        self.stefan_loss = StefanBoltzmannLoss()
        self.cosine_loss = CosineContrastiveAlignmentLoss()
    
    def forward(self, predictions: dict, batch: dict, 
                encoder_features: list = None) -> dict:
        """
        Compute total loss.
        
        Args:
            predictions: Dict with 'hsv', 'rgb' from generator
            batch: Dict with 'input', 'target', 'grayscale'
            encoder_features: List of encoder feature tensors for cosine loss
        
        Returns:
            dict with 'total_loss' and individual loss components
        """
        pred_rgb = predictions['rgb']
        pred_hsv = predictions['hsv']
        target_rgb = batch['target']
        input_thermal = batch['grayscale']
        
        # 1. Reconstruction loss
        recon, recon_components = self.recon_loss(pred_rgb, target_rgb)
        
        # 2. Focal frequency loss
        focal_freq = self.focal_freq_loss(pred_rgb, target_rgb)
        
        # 3. Planck consistency loss
        planck = self.planck_loss(pred_hsv, input_thermal)
        
        # 4. Stefan-Boltzmann loss
        stefan = self.stefan_loss(pred_hsv, input_thermal)
        
        # 5. Cosine contrastive loss
        cosine = 0.0
        if encoder_features:
            cosine = self.cosine_loss(encoder_features)
        
        # Total
        total = (
            self.config['lambda_recon'] * recon +
            self.config['lambda_focal_freq'] * focal_freq +
            self.config['lambda_planck'] * planck +
            self.config['lambda_stefan'] * stefan +
            self.config['lambda_cosine'] * cosine
        )
        
        return {
            'total_loss': total,
            'recon_loss': recon,
            'focal_freq_loss': focal_freq,
            'planck_loss': planck,
            'stefan_loss': stefan,
            'cosine_loss': cosine,
            'recon_spatial': recon_components['spatial'],
            'recon_frequency': recon_components['frequency'],
        }
```

### 4.5 Training Hyperparameters

```yaml
# THERMAVISION-X Training Configuration

# Model
model:
  base_dim: 32
  num_levels: 4
  physics_dim: 6
  dropout_rate: 0.1

# Training
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 2.0e-4
  weight_decay: 1.0e-4
  optimizer: AdamW
  scheduler: cosine_annealing
  warmup_epochs: 5
  
  # Loss weights
  lambda_recon: 1.0
  lambda_focal_freq: 0.5
  lambda_planck: 0.1
  lambda_stefan: 0.05
  lambda_cosine: 0.1

# Data
data:
  image_size: 256
  mask_ratio: 0.15  # Frequency mask radius ratio
  num_workers: 8
  pin_memory: true
  
  # Augmentation
  random_crop: true
  random_flip: true
  color_jitter: 0.2
  
# Hardware
hardware:
  device: cuda
  mixed_precision: true  # AMP
  num_gpus: 1

# Validation
validation:
  frequency: 5  # Validate every N epochs
  metrics: [NIQE, thermal_consistency, sharpness]

# Checkpointing
checkpoint:
  save_frequency: 10
  keep_last_n: 5
  save_best: true
```

### 4.6 Training Command

```bash
# Launch training with PyTorch Lightning
python training/train_zero_shot.py \
    --data_dir /path/to/visible/images \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --lr 2e-4 \
    --epochs 100 \
    --image_size 256 \
    --mask_ratio 0.15 \
    --lambda_recon 1.0 \
    --lambda_focal_freq 0.5 \
    --lambda_planck 0.1 \
    --lambda_stefan 0.05 \
    --lambda_cosine 0.1 \
    --gpus 1 \
    --precision 16
```

---

## 5. Inference Pipeline (Zero-Shot)

### 5.1 Complete Inference Pipeline

```python
class ThermaVisionInference:
    """
    End-to-end zero-shot inference pipeline for THERMAVISION-X.
    Works on any IR band from any satellite without retraining.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 model: nn.Module = None,
                 device: str = 'cuda',
                 use_uncertainty: bool = True,
                 num_mc_samples: int = 10,
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: Path to checkpoint file
            model: Pre-initialized model (alternative to model_path)
            device: Device to run inference on
            use_uncertainty: Enable MC dropout uncertainty estimation
            num_mc_samples: Number of MC dropout forward passes
            confidence_threshold: Minimum confidence for auto-approval
        """
        self.device = device
        self.use_uncertainty = use_uncertainty
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        if model_path:
            self.model = ThermaVisionGenerator().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = model.to(device)
        
        self.model.eval()
        
        # Initialize uncertainty quantifier
        if use_uncertainty:
            self.uncertainty = UncertaintyQuantifier(
                self.model, num_samples=num_mc_samples
            )
        
        # Initialize preprocessing
        self.preprocessing = PreprocessingPipeline()
        self.postprocessing = OutputPostprocessor()
    
    @torch.no_grad()
    def colorize(self, thermal_input: torch.Tensor) -> dict:
        """
        Zero-shot colorization of thermal IR image.
        
        Args:
            thermal_input: [B, 1, H, W] preprocessed thermal image
        
        Returns:
            dict with colorized output, uncertainty, and metrics
        """
        thermal_input = thermal_input.to(self.device)
        
        # Option 1: Standard inference (fast, no uncertainty)
        if not self.use_uncertainty:
            output = self.model(thermal_input)
            return {
                'rgb': output['rgb'],
                'hsv': output['hsv'],
                'uncertainty': None,
                'confidence': None,
            }
        
        # Option 2: Uncertainty-aware inference (slower, with confidence)
        result = self.uncertainty.estimate(thermal_input)
        
        # Apply confidence threshold
        flagged_rgb, confidence_mask = self.uncertainty.apply_confidence_threshold(
            result['mean_output'],
            result['confidence'],
            threshold=self.confidence_threshold
        )
        
        return {
            'rgb': result['mean_output'],
            'hsv': None,  # Can be obtained from single forward pass
            'uncertainty': result['uncertainty'],
            'confidence': result['confidence'],
            'flagged_rgb': flagged_rgb,
            'confidence_mask': confidence_mask,
        }
    
    def process_satellite_file(self, filepath: str, satellite: str, 
                                band: str) -> dict:
        """
        Complete pipeline from satellite file to colorized output.
        
        Args:
            filepath: Path to satellite data file
            satellite: Satellite name (e.g., 'INSAT_3D')
            band: Band name (e.g., 'TIR1')
        
        Returns:
            dict with colorized output and metadata
        """
        # 1. Preprocessing
        preprocessed = self.preprocessing.process(filepath, satellite, band)
        input_tensor = preprocessed['normalized_input'].to(self.device)
        
        # 2. Colorization (zero-shot)
        colorized = self.colorize(input_tensor)
        
        # 3. Postprocessing
        output = self.postprocessing.process(colorized, preprocessed['metadata'])
        
        return output
```

### 5.2 Inference Pipeline Diagram

```
+================================================================+
|                    ZERO-SHOT INFERENCE PIPELINE                 |
+================================================================+

Input: Satellite Thermal IR File (HDF5/GeoTIFF)
  |
  v
+-----------------------------------------+
| [PREPROCESSING]                         |
| - DN -> Radiance -> Temperature         |
| - Atmospheric Correction                |
| - Normalize [0, 1]                      |
+-----------------------------------------+
  |
  v  [B, 1, H, W]
+-----------------------------------------+
| [FREQUENCY DECOUPLING]                  |
| - 2D FFT                                |
| - High-Pass Filter -> Structure Feats   |
| - Low-Freq Masked                       |
+-----------------------------------------+
  |
  v  [B, 1, H, W] (structure features)
+-----------------------------------------+
| [MAMBA UNet ENCODER]                    |
| - Mamba Blocks x 4                      |
| - Downsample x 3                        |
| - Latent Features [B, 256, H/8, W/8]    |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [PHYSICS PROJECTION]                    |
| - Global Pooling                        |
| - Planck MLP -> Physics Embedding       |
| [B, 6, 1, 1]                            |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [HSV COLOR MAPPING]                     |
| T -> Saturation                         |
| epsilon -> Hue                          |
| Texture -> Value                        |
| [B, 3, H/8, W/8]                        |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [MAMBA UNet DECODER]                    |
| - Full-Scale Skip Connections           |
| - Upsample x 3                          |
| - Mamba Blocks x 3                      |
| - Output: HSV [B, 3, H, W]              |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [HSV -> RGB CONVERSION]                 |
+-----------------------------------------+
  |
  +-----> Standard Output
  |
  v
+-----------------------------------------+
| [MC DROPOUT UNCERTAINTY] (Optional)     |
| - T=10 Forward Passes                   |
| - Per-Pixel Variance                    |
| - Confidence Heatmap                    |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [QUALITY ASSESSMENT]                    |
| - NIQE, BRISQUE, PIQE                   |
| - Thermal Consistency Check             |
+-----------------------------------------+
  |
  v
+-----------------------------------------+
| [OUTPUT]                                |
| - Colorized RGB Image                   |
| - Uncertainty Heatmap (optional)        |
| - Quality Report JSON                   |
| - GeoTIFF with Metadata                 |
+-----------------------------------------+
```

---

## 6. Datasets for Development

### 6.1 Public Benchmark Datasets

| Dataset | Type | Size | Bands | Use Case |
|---|---|---|---|---|
| **KAIST Multispectral** | Ground | 95K pairs | RGB + LWIR | Zero-shot validation |
| **FLIR ADAS** | Ground | 14K pairs | RGB + LWIR | Domain transfer test |
| **LLVIP** | Ground | 30.9K pairs | RGB + LWIR | Night scene validation |
| **FLIR Thermal Faces** | Ground | 14K images | Thermal only | Generalization test |
| **M3FD** | Ground | 4.2K pairs | RGB + IR | Multi-scene evaluation |

### 6.2 ISRO Satellite Data (Free Access via MOSDAC)

| Satellite/Sensor | Bands | Resolution | Access |
|---|---|---|---|
| **INSAT-3D Imager** | 1 TIR | 8 km | MOSDAC (free) |
| **INSAT-3DR Imager** | 2 TIR | 4 km | MOSDAC (free) |
| **INSAT-3DS** | 2 TIR | 4 km | MOSDAC (free) |
| **Kalpana-1 VHRR** | 1 TIR, 1 WV | 8 km | MOSDAC (free) |

### 6.3 International Validation Data

| Satellite/Sensor | Bands | Resolution | Access |
|---|---|---|---|
| **Landsat-8/9 TIRS** | 2 TIR | 100 m | USGS EarthExplorer (free) |
| **Sentinel-3 SLSTR** | 2 TIR + Fire | 1 km | Copernicus Open Hub (free) |
| **Terra/Aqua MODIS** | Multiple TIR | 1 km | LAADS DAAC (free) |
| **NOAA-20/VIIRS** | 3 TIR + DNB | 375-750 m | NOAA (free) |
| **ECOSTRESS** | 1 TIR (multispectral) | 38 m | NASA (free) |
| **TRISHNA (2026)** | 4 TIR | 57 m | ISRO-CNES (upcoming) |

### 6.4 Data Preparation Strategy

```python
# Dataset download and preparation pipeline

# 1. Visible Images for Training (Zero-Shot)
#    - COCO 2017 (118K images)
#    - ImageNet subset (100K images)  
#    - Places365 (1.8M images - subset of 100K)
#    Total: ~300K visible images for self-supervised training

# 2. Validation on Ground-Based IR-RGB Pairs
#    - KAIST Multispectral (for quantitative metrics)
#    - FLIR ADAS (for qualitative assessment)

# 3. Satellite IR Data for Real-World Testing
#    - INSAT-3D/3DR from MOSDAC
#    - Landsat-8 TIRS for cross-sensor validation

# 4. Synthetic IR Data (for ablation studies)
#    - Convert visible images to pseudo-thermal using:
#      a) Grayscale conversion
#      b) Color temperature mapping (cool->warm LUT)
#      c) Planck radiance simulation at known wavelength
```

---

## 7. Implementation Roadmap (30 Hours)

### 7.1 Hour-by-Hour Breakdown

```
+====================================================================+
|            THERMAVISION-X: 30-HOUR HACKATHON ROADMAP              |
+====================================================================+

Phase 1: Foundation (Hours 1-3)
=================================
Hour 1  | Environment setup
        | - Clone repository, install dependencies
        | - PyTorch, mamba_ssm, CUDA setup
        | - Verify GPU availability
        |
Hour 2  | Data preparation
        | - Download COCO/ImageNet subset (visible images)
        | - Download INSAT-3D sample from MOSDAC
        | - Set up data loaders and preprocessing pipeline
        | - Verify DN->Radiance->Temperature conversion
        |
Hour 3  | Project skeleton
        | - Implement repository structure
        | - Create base configuration files
        | - Set up logging and experiment tracking
        | - Write unit tests for preprocessing

[MILESTONE 1: Training-ready environment]

Phase 2: Core Architecture (Hours 4-8)
======================================
Hour 4  | Mamba block implementation
        | - Implement SS2D Mamba block
        | - Test forward/backward pass
        | - Verify gradient flow
        |
Hour 5  | Encoder + Decoder UNet
        | - Implement encoder pathway with downsampling
        | - Implement decoder pathway with upsampling
        | - Add skip connections (basic)
        |
Hour 6  | Full-scale skip connections
        | - Implement FullScaleSkipConnection module
        | - Integrate into decoder pathway
        | - Test multi-scale feature fusion
        |
Hour 7  | Physics projection layer
        | - Implement Planck feature extractor
        | - Add HSV color mapping layer
        | - Integrate physics into forward pass
        |
Hour 8  | Complete generator + test
        | - Wire all components together
        | - Test forward pass end-to-end
        | - Count parameters (target: ~5M)
        | - Verify output shapes

[MILESTONE 2: Complete generator architecture]

Phase 3: Frequency + Physics Loss (Hours 9-12)
===============================================
Hour 9  | Frequency decoupling module
        | - Implement 2D FFT transformation
        | - Implement circular high-pass/low-pass masks
        | - Test frequency domain operations
        |
Hour 10 | Physics-informed losses
        | - Implement Planck consistency loss
        | - Implement Stefan-Boltzmann loss
        | - Test gradient computation
        |
Hour 11 | Focal frequency + reconstruction loss
        | - Implement focal frequency loss
        | - Implement combined spatial + freq reconstruction
        | - Integrate cosine contrastive loss
        |
Hour 12 | Complete loss function + training loop
        | - Combine all loss components with weights
        | - Write training loop skeleton
        | - Test single training step
        | - Verify all gradients flow correctly

[MILESTONE 3: Full loss function, single training step works]

Phase 4: Training Pipeline (Hours 13-16)
=========================================
Hour 13 | Training infrastructure
        | - Implement PyTorch Lightning module
        | - Set up data module with augmentation
        | - Configure optimizer (AdamW) + scheduler
        |
Hour 14 | Training launch + monitoring
        | - Start training on visible images
        | - Monitor loss components
        | - Check for NaN/instability
        |
Hour 15 | Checkpointing + validation
        | - Implement checkpoint saving/loading
        | - Add validation loop with metrics
        | - Monitor NIQE and quality metrics
        |
Hour 16 | Training optimization
        | - Profile training speed
        | - Enable AMP mixed precision
        | - Optimize data loading (num_workers, prefetch)
        | - Train for remaining time

[MILESTONE 4: Model training, checkpoints saved]

Phase 5: Inference + Uncertainty (Hours 17-20)
===============================================
Hour 17 | Inference pipeline
        | - Implement colorize() function
        | - Test on visible image (should reconstruct)
        | - Test on IR image (zero-shot generalization)
        |
Hour 18 | Uncertainty quantification
        | - Implement MC Dropout
        | - Generate uncertainty heatmaps
        | - Add confidence thresholding
        |
Hour 19 | Quality assessment
        | - Implement NIQE/BRISQUE metrics
        | - Compute thermal consistency score
        | - Generate quality report
        |
Hour 20 | Batch processing + output export
        | - Implement batch colorization script
        | - Add GeoTIFF export with metadata
        | - Generate sample outputs for demo

[MILESTONE 5: Working inference with uncertainty]

Phase 6: Edge Optimization (Hours 21-24)
=========================================
Hour 21 | ONNX export
        | - Export trained model to ONNX
        | - Verify ONNX inference matches PyTorch
        | - Check operator compatibility
        |
Hour 22 | TensorRT FP16 optimization
        | - Build TensorRT FP16 engine
        | - Benchmark inference speed
        | - Verify output quality vs. FP32
        |
Hour 23 | INT8 quantization
        | - Implement INT8 calibrator
        | - Build INT8 TensorRT engine
        | - Benchmark speed and accuracy
        |
Hour 24 | Edge deployment package
        | - Create Docker container for edge
        | - Write deployment scripts
        | - Document deployment process

[MILESTONE 6: Edge-optimized model ready]

Phase 7: Demo + UI (Hours 25-28)
=================================
Hour 25 | Streamlit UI framework
        | - Create main app layout
        | - Add file upload component
        | - Implement side-by-side comparison view
        |
Hour 26 | Visualization components
        | - Add IR input display
        | - Add colorized output display
        | - Add uncertainty heatmap overlay
        |
Hour 27 | Interactive features
        | - Add satellite selector (INSAT, Landsat, etc.)
        | - Add band selector
        | - Add confidence threshold slider
        | - Add quality metrics display
        |
Hour 28 | Demo polish + sample data
        | - Prepare demo video/gif
        | - Create sample outputs showcase
        | - Test on real satellite data
        | - Prepare presentation slides

[MILESTONE 7: Polished demo ready]

Phase 8: Finalization (Hours 29-30)
====================================
Hour 29 | Testing + bug fixes
        | - End-to-end testing on all supported satellites
        | - Performance benchmarking
        | - Fix any remaining bugs
        |
Hour 30 | Documentation + submission
        | - Complete README.md
        | - Write technical documentation
        | - Prepare submission package
        | - Submit to hackathon portal

[MILESTONE 8: Complete submission]

+================================================================+
|                    COMPETITIVE ADVANTAGES                       |
+================================================================+
- Hour 8:  Architecture complete (Day 1 demo possible)
- Hour 12: Training working (Tier 1 minimum deliverable)
- Hour 20: Full inference pipeline (Tier 2 deliverable)
- Hour 24: Edge optimization (Technical excellence)
- Hour 28: Polished demo (Presentation impact)
- Hour 30: Complete submission (Tier 3 deliverable)
```

### 7.2 Tiered Deliverable Strategy

| Tier | Hours | Minimum Deliverable | Stretch Goal |
|---|---|---|---|
| Tier 1 | 1-12 | Working training + basic colorization | Physics loss integrated |
| Tier 2 | 13-20 | Full inference + uncertainty | Quality metrics + batch processing |
| Tier 3 | 21-30 | Edge deployment + polished demo | Real-time demo with ISRO data |

---

## 8. Technology Stack

### 8.1 Core Framework

| Category | Technology | Version | Purpose |
|---|---|---|---|
| Deep Learning | PyTorch | 2.1+ | Core framework |
| Training Mgmt | PyTorch Lightning | 2.1+ | Training loop, checkpointing |
| Mamba Backbone | mamba_ssm (state-spaces) | 1.2+ | Selective state space models |
| CUDA | cuDNN | 8.9+ | GPU acceleration |

### 8.2 Scientific Computing

| Technology | Purpose |
|---|---|
| NumPy | Array operations, numerical computing |
| SciPy | FFT, signal processing |
| OpenCV (cv2) | Image I/O, preprocessing, visualization |
| Pillow (PIL) | Image loading, basic transforms |
| scikit-image | Image quality metrics |
| rasterio | Satellite GeoTIFF I/O |
| h5py | HDF5 satellite data reading |
| GDAL | Geospatial data processing |

### 8.3 Edge Deployment

| Technology | Purpose |
|---|---|
| ONNX | Model export format |
| TensorRT | GPU inference optimization |
| pycuda | CUDA Python bindings |
| onnxruntime | ONNX inference (fallback) |

### 8.4 Quality Assessment

| Technology | Purpose |
|---|---|
| pyiqa | NIQE, BRISQUE, PIQE, LPIPS |
| lpips | Learned perceptual similarity |

### 8.5 Demo & Visualization

| Technology | Purpose |
|---|---|
| Streamlit | Interactive demo UI |
| Gradio | Alternative demo UI |
| Matplotlib | Static plots, visualization |
| Plotly | Interactive plots |
| Folium | Map visualization for geospatial |

### 8.6 Development Tools

| Technology | Purpose |
|---|---|
| Docker | Containerization |
| Git | Version control |
| Weights & Biases | Experiment tracking |
| pytest | Unit testing |
| Black | Code formatting |

### 8.7 requirements.txt

```txt
# Core
torch>=2.1.0
torchvision>=0.16.0
pytorch-lightning>=2.1.0

# Mamba
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0

# Scientific
numpy>=1.24.0
scipy>=1.11.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-image>=0.21.0

# Geospatial
rasterio>=1.3.0
h5py>=3.9.0
pyproj>=3.6.0

# Quality Assessment
pyiqa>=0.1.0
lpips>=0.1.4

# Edge Deployment
onnx>=1.15.0
onnxruntime-gpu>=1.16.0
tensorrt>=8.6.0
pycuda>=2022.2.2

# Demo
streamlit>=1.28.0
gradio>=4.0.0
matplotlib>=3.8.0
plotly>=5.18.0
folium>=0.15.0

# Development
wandb>=0.16.0
pytest>=7.4.0
black>=23.0.0
python-dotenv>=1.0.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.0
omegaconf>=2.3.0
```

---

## 9. Evaluation Metrics

### 9.1 Image Quality Metrics

| Metric | Range | Target | Description |
|---|---|---|---|
| **PSNR** | Higher better | >= 28 dB | Peak signal-to-noise ratio (vs. pseudo-reference) |
| **SSIM** | [0, 1], higher better | >= 0.85 | Structural similarity |
| **LPIPS** | [0, 1], lower better | <= 0.15 | Learned perceptual similarity |
| **FID** | Lower better | <= 50 | Frechet Inception Distance (realism) |

### 9.2 No-Reference Quality Metrics

| Metric | Range | Target | Description |
|---|---|---|---|
| **NIQE** | Lower better (typ. 2-10) | <= 4.0 | Natural Image Quality Evaluator |
| **BRISQUE** | Lower better | <= 30 | Blind/Referenceless Image Spatial Quality Evaluator |
| **PIQE** | Lower better | <= 30 | Perception-based Image Quality Evaluator |

### 9.3 Thermal Consistency Metrics

| Metric | Range | Target | Description |
|---|---|---|---|
| **TC-SSIM** | [0, 1], higher better | >= 0.92 | SSIM between input IR grayscale and colorized grayscale |
| **TC-PSNR** | Higher better | >= 35 dB | PSNR between input IR grayscale and colorized grayscale |
| **Gradient Correlation** | [0, 1], higher better | >= 0.85 | Correlation of gradient maps |

### 9.4 Efficiency Metrics

| Metric | Target | Description |
|---|---|---|
| **Inference FPS** | >= 15 | Frames per second on Jetson Nano |
| **Model Size (FP32)** | <= 20 MB | Disk size of model weights |
| **Model Size (INT8)** | <= 5 MB | Quantized model size |
| **GPU Memory** | <= 2 GB | VRAM usage during inference |
| **Power Consumption** | <= 10W | On edge hardware |

### 9.5 Evaluation Script

```python
class ThermaVisionEvaluator:
    """
    Comprehensive evaluation suite for THERMAVISION-X.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.quality_metrics = self._init_quality_metrics()
    
    def _init_quality_metrics(self):
        """Initialize all metric calculators."""
        metrics = {}
        
        # Full-reference metrics (for validation on paired data)
        try:
            import lpips
            metrics['lpips'] = lpips.LPIPS(net='alex').to(self.device)
        except:
            pass
        
        # No-reference metrics
        try:
            from pyiqa import create_metric
            metrics['niqe'] = create_metric('niqe')
            metrics['brisque'] = create_metric('brisque')
        except:
            pass
        
        return metrics
    
    def evaluate(self, 
                 colorized: torch.Tensor,
                 input_ir: torch.Tensor = None,
                 reference: torch.Tensor = None) -> dict:
        """
        Evaluate colorized output.
        
        Args:
            colorized: [B, 3, H, W] colorized output
            input_ir: [B, 1, H, W] input IR image (for thermal consistency)
            reference: [B, 3, H, W] reference RGB (for full-reference metrics)
        
        Returns:
            dict with all metric values
        """
        results = {}
        
        # 1. No-reference quality
        if 'niqe' in self.quality_metrics:
            results['NIQE'] = float(self.quality_metrics['niqe'](colorized).mean())
        if 'brisque' in self.quality_metrics:
            results['BRISQUE'] = float(self.quality_metrics['brisque'](colorized).mean())
        
        # 2. Full-reference quality (if reference available)
        if reference is not None:
            results['PSNR'] = self._compute_psnr(colorized, reference)
            results['SSIM'] = self._compute_ssim(colorized, reference)
            if 'lpips' in self.quality_metrics:
                results['LPIPS'] = float(
                    self.quality_metrics['lpips'](colorized, reference).mean()
                )
        
        # 3. Thermal consistency
        if input_ir is not None:
            tc_ssim, tc_psnr = self._compute_thermal_consistency(
                colorized, input_ir
            )
            results['TC-SSIM'] = tc_ssim
            results['TC-PSNR'] = tc_psnr
        
        return results
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
        return float(psnr)
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        from skimage.metrics import structural_similarity
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        ssim_val = structural_similarity(
            pred_np[0].transpose(1, 2, 0),
            target_np[0].transpose(1, 2, 0),
            channel_axis=2,
            data_range=1.0
        )
        return float(ssim_val)
    
    def _compute_thermal_consistency(self, 
                                      colorized: torch.Tensor,
                                      input_ir: torch.Tensor) -> tuple:
        """Compute thermal consistency metrics."""
        # Convert colorized to grayscale
        gray = 0.299 * colorized[:, 0:1] + 0.587 * colorized[:, 1:2] + \
               0.114 * colorized[:, 2:3]
        
        # Resize to match input IR if needed
        if gray.shape != input_ir.shape:
            gray = F.interpolate(gray, size=input_ir.shape[2:], 
                                  mode='bilinear', align_corners=False)
        
        # SSIM between grayscale and IR input
        tc_ssim = self._compute_ssim(gray, input_ir)
        tc_psnr = self._compute_psnr(gray, input_ir)
        
        return tc_ssim, tc_psnr
```

---

## 10. Risk Mitigation

### 10.1 Risk Register

| # | Risk | Probability | Impact | Mitigation Strategy |
|---|---|---|---|---|
| R1 | GAN training instability | Medium | High | Use Mamba backbone (more stable than GANs); No adversarial loss; Pre-train on visible images |
| R2 | Slow inference on edge | Low | High | Target 5M params (not 40M); Skip DDNM diffusion; Direct generator inference |
| R3 | No IR training data | N/A (by design) | N/A | Core zero-shot design: train on visible only; frequency decoupling enables cross-spectral transfer |
| R4 | 30-hour time constraint | High | High | Tiered implementation; Tier 1 (12h) = working demo; Tier 2 (20h) = full pipeline; Tier 3 (30h) = optimization |
| R5 | Mamba CUDA compatibility | Low | High | Fallback to efficient ConvNext blocks if mamba_ssm fails; Pure PyTorch implementation |
| R6 | Quality below expectations | Medium | Medium | Multiple loss components ensure robustness; No-reference metrics for feedback; Physics constraints ground outputs |
| R7 | Dataset download failures | Medium | Low | Use subset of COCO (readily available); Generate synthetic data as fallback |
| R8 | TensorRT export issues | Medium | Medium | ONNX Runtime as fallback; PyTorch Mobile as backup |

### 10.2 Contingency Plans

**If Mamba CUDA compilation fails:**
```python
# Fallback architecture using ConvNeXt blocks
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block as Mamba fallback. Efficient, modern CNN."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x
```

**If training is too slow:**
- Reduce batch size to 8 (from 16)
- Reduce image size to 128x128 for initial training
- Use gradient accumulation (effective batch size = 16)
- Enable full AMP (automatic mixed precision)

**If quality is insufficient:**
- Increase base_dim to 48 (from 32) for ~8M params
- Add attention mechanism at bottleneck
- Increase lambda_focal_freq to 1.0
- Fine-tune with perceptual loss (VGG features)

---

## 11. GitHub Repository Structure

```
thermavision-x/
|
|-- README.md                          # Project overview, quick start guide
|-- requirements.txt                   # Python dependencies
|-- setup.py                          # Package installation
|-- .gitignore                        # Git ignore rules
|-- docker/
|   |-- Dockerfile                    # Container for deployment
|   |-- docker-compose.yml            # Compose configuration
|   |-- entrypoint.sh                 # Container entry point
|
|-- data/                             # Dataset loaders
|   |-- __init__.py
|   |-- datasets.py                   # ZeroShotTrainingDataset
|   |-- preprocessing.py              # PreprocessingPipeline
|   |-- satellite_io.py               # Satellite data ingestion
|   |-- download_data.sh              # Data download script
|   |-- configs/
|   |   |-- insat3d.yaml             # INSAT-3D config
|   |   |-- trishna.yaml             # TRISHNA config
|   |   |-- landsat8.yaml            # Landsat-8 config
|
|-- models/                           # Architecture definitions
|   |-- __init__.py
|   |-- thermavision_generator.py    # ThermaVisionGenerator (main model)
|   |-- mamba_unet.py                # Mamba UNet backbone
|   |-- mamba_block.py               # MambaBlock + ConvNeXt fallback
|   |-- frequency_decouple.py        # FrequencyDecouplingModule
|   |-- physics_projection.py        # PhysicsProjectionLayer
|   |-- hsv_mapping.py               # HSVColorMappingLayer
|   |-- skip_connections.py          # FullScaleSkipConnection
|   |-- uncertainty.py               # UncertaintyQuantifier
|   |-- export_onnx.py               # ONNX export utility
|
|-- physics/                          # Physics-informed components
|   |-- __init__.py
|   |-- planck.py                    # PlanckConverter + PlanckConsistencyLoss
|   |-- stefan_boltzmann.py          # StefanBoltzmannLoss
|   |-- split_window.py              # SplitWindowCorrection
|   |-- normalizer.py                # ThermalNormalizer
|   |-- constants.py                 # Physical constants
|
|-- training/                         # Training scripts
|   |-- __init__.py
|   |-- train_zero_shot.py           # Main training script
|   |-- lightning_module.py          # PyTorch Lightning module
|   |-- losses.py                    # ThermaVisionLoss + components
|   |-- config.yaml                  # Default training configuration
|   |-- callbacks.py                 # Custom callbacks
|
|-- inference/                        # Inference pipeline
|   |-- __init__.py
|   |-- colorize.py                  # ThermaVisionInference class
|   |-- batch_process.py             # Batch processing script
|   |-- single_image.py              # Single image colorization CLI
|   |-- satellite_pipeline.py        # End-to-end satellite processing
|
|-- edge/                             # Edge deployment
|   |-- __init__.py
|   |-- export_onnx.py               # ONNX export
|   |-- tensorrt_optimize.py         # TensorRT engine builder
|   |-- int8_calibrator.py           # INT8 calibration
|   |-- benchmark.py                 # Edge benchmarking
|   |-- deploy.py                    # Edge deployment script
|   |-- scripts/
|   |   |-- build_tensorrt.sh       # TensorRT build script
|   |   |-- benchmark_jetson.sh     # Jetson benchmarking
|
|-- evaluation/                       # Evaluation suite
|   |-- __init__.py
|   |-- evaluator.py                 # ThermaVisionEvaluator
|   |-- metrics.py                   # Individual metric implementations
|   |-- benchmark.py                 # Benchmark runner
|   |-- generate_report.py           # Report generation
|
|-- utils/                            # Utilities
|   |-- __init__.py
|   |-- visualization.py             # Plotting and visualization
|   |-- geotiff.py                   # GeoTIFF export
|   |-- logger.py                    # Logging utilities
|   |-- config.py                    # Configuration management
|   |-- helpers.py                   # General helper functions
|
|-- demo/                             # Streamlit/Gradio demo
|   |-- app.py                       # Main Streamlit application
|   |-- gradio_app.py                # Gradio alternative
|   |-- components/
|   |   |-- sidebar.py              # Sidebar controls
|   |   |-- image_viewer.py         # Image display components
|   |   |-- metrics_display.py      # Metrics dashboard
|   |   |-- map_viewer.py           # Folium map component
|   |-- assets/
|   |   |-- logo.png                # Project logo
|   |   |-- demo_samples/           # Sample images for demo
|   |-- style.css                    # Custom CSS styling
|
|-- docs/                             # Documentation
|   |-- ARCHITECTURE.md              # This document
|   |-- API.md                       # API reference
|   |-- DEPLOYMENT.md                # Deployment guide
|   |-- FAQ.md                       # Frequently asked questions
|   |-- images/                      # Documentation images
|   |   |-- architecture_diagram.png
|   |   |-- sample_outputs/
|
|-- checkpoints/                      # Model weights (gitignored)
|   |-- .gitkeep
|   |-- README.md                    # Instructions to download weights
|
|-- tests/                            # Unit tests
|   |-- test_preprocessing.py
    |-- test_frequency_decouple.py
|   |-- test_generator.py
|   |-- test_losses.py
|   |-- test_inference.py
|
|-- scripts/                          # Helper scripts
|   |-- setup_env.sh                 # Environment setup
|   |-- download_checkpoints.sh      # Download pretrained weights
|   |-- run_demo.sh                  # Launch demo
|   |-- run_training.sh             # Launch training
|   |-- run_evaluation.sh           # Run evaluation suite
```

### 11.1 Quick Start

```bash
# Clone repository
git clone https://github.com/your-team/thermavision-x.git
cd thermavision-x

# Setup environment
bash scripts/setup_env.sh

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
bash scripts/download_checkpoints.sh

# Run single image colorization
python inference/single_image.py \
    --input path/to/thermal_image.tif \
    --output colorized_output.png \
    --checkpoint checkpoints/thermavision_x_best.pth

# Launch demo
streamlit run demo/app.py

# Run edge benchmark
python edge/benchmark.py \
    --engine checkpoints/thermavision_x_int8.engine \
    --device jetson
```

---

## 12. References

### IR Colorization Research

1. **Wei et al. (2024)** - "Zero-Shot Cross-Spectral Image Translation via Frequency Domain Feature Decoupling and Masked Image Modeling." *arXiv preprint*.

2. **Wang et al. (2023)** - "Zero-Shot Image Restoration via Denoising Diffusion Null-Space Model." *ICLR 2023 Oral (DDNM)*.

3. **CCLGAN (2025)** - "Cosine Contrastive Learning GAN with Mamba and Full-Scale Skip Connections for Unsupervised Infrared Image Colorization." *IEEE TIP*.

4. **Jiang et al. (2021)** - "Focal Frequency Loss for Image Reconstruction and Synthesis." *ICCV 2021*.

5. **Guo et al. (2024)** - "ColorMamba: Leveraging State Space Models for Infrared Image Colorization." *arXiv preprint*.

6. **Chen et al. (2024)** - "MambaIR: A Simple Baseline for Image Restoration with State Space Models." *arXiv preprint*.

### Physics-AI Integration

7. **Planck, M. (1901)** - "On the Law of Distribution of Energy in the Normal Spectrum." *Annalen der Physik*.

8. **Stefan, J. (1879)** - "Uber die Beziehung zwischen der Warmestrahlung und der Temperatur." *Sitzungsberichte der mathematisch-naturwissenschaftlichen Classe der kaiserlichen Akademie der Wissenschaften*.

9. **Boltzmann, L. (1884)** - "Ableitung des Stefan'schen Gesetzes, betreffend die Abhangigkeit der Warmestrahlung von der Temperatur aus der electromagnetischen Lichttheorie." *Annalen der Physik und Chemie*.

10. **TeX-NeRF (2023)** - "NeRF-based Temperature-Emissivity-Texture Mapping for Thermal Infrared Images."

11. **Raissi et al. (2019)** - "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations." *Journal of Computational Physics*.

### Mamba / State Space Models

12. **Gu & Dao (2023)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv preprint*.

13. **Dao & Gu (2024)** - "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality."

### ISRO Missions

14. **INSAT-3D/3DR Mission Documentation** - ISRO Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC).

15. **TRISHNA Mission (2026)** - ISRO-CNES Joint Thermal Infrared Imaging Mission for High-Resolution Land Surface Temperature.

16. **Oceansat-3 Mission** - ISRO Ocean Observation Satellite with Thermal Bands.

### Quality Assessment

17. **Mittal et al. (2013)** - "Making a Completely Blind Image Quality Analyzer." *IEEE Signal Processing Letters (NIQE)*.

18. **Mittal et al. (2012)** - "No-Reference Image Quality Assessment in the Spatial Domain." *IEEE TIP (BRISQE)*.

19. **Venkatanath et al. (2015)** - "Blind/Referenceless Image Spatial Quality Evaluator." *IEEE Access (PIQE)*.

### Edge AI

20. **TensorRT Documentation** - NVIDIA TensorRT Optimization Guide.

21. **Jetson Nano Benchmarks** - NVIDIA Edge AI Inference Performance.

22. **GAN Slimming (2020)** - "Compressing Generative Adversarial Networks for Efficient Edge Inference."

---

*Document Version: 1.0*  
*Compiled by: Benad | June 2026*  
*For ISRO Bharatiya Antariksh Hackathon 2026*
