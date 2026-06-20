# Physics-Informed Neural Networks for Thermal Image Colorization
## Comprehensive Research Findings for THERMAVISION-X
### ISRO's Bharatiya Antariksh Hackathon 2026
**Researched by**: Benad | June 2026

---

## Table of Contents

1. [Physics-Informed Neural Networks (PINNs) for Computer Vision](#1-physics-informed-neural-networks-pinns-for-computer-vision)
2. [Thermal Physics Laws for IR Imagery](#2-thermal-physics-laws-for-ir-imagery)
3. [Satellite Remote Sensing Physics](#3-satellite-remote-sensing-physics)
4. [Implementing Physics Constraints in Deep Learning](#4-implementing-physics-constraints-in-deep-learning)
5. [Synthesis: Physics-Guided Zero-Shot Thermal Colorization Architecture](#5-synthesis-physics-guided-zero-shot-thermal-colorization-architecture)
6. [References](#6-references)

---

## 1. Physics-Informed Neural Networks (PINNs) for Computer Vision

### 1.1 How PINNs Work: Foundational Principles

Physics-Informed Neural Networks (PINNs) embed physical laws, typically expressed as Partial Differential Equations (PDEs), directly into the training process of neural networks. Unlike conventional deep learning that relies purely on data-driven optimization, PINNs enforce consistency between predictions and expected physical behavior by incorporating governing equations as soft constraints in the loss function.

**Core Architecture:**

The standard PINN formulation uses a neural network $u_\theta(x)$ to approximate the solution of a PDE. The total loss function is:

$$\mathcal{L}(\theta) = \lambda_{data} \mathcal{L}_{data} + \lambda_{PDE} \mathcal{L}_{PDE} + \lambda_{BC} \mathcal{L}_{BC} + \lambda_{IC} \mathcal{L}_{IC}$$

Where:
- $\mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} |u_\theta(x_i) - u_i^{true}|^2$  (data fidelity term)
- $\mathcal{L}_{PDE} = \frac{1}{N_{col}} \sum_{j=1}^{N_{col}} |\mathcal{F}(u_\theta(x_j), x_j)|^2$  (PDE residual term)
- $\mathcal{L}_{BC}$ = boundary condition residual
- $\mathcal{L}_{IC}$ = initial condition residual
- $\lambda$ terms are weighting hyperparameters

The key innovation is **automatic differentiation**: neural network derivatives are computed via backpropagation (autograd) to evaluate PDE residuals without numerical discretization.

**PINN Advantages for Computer Vision:**
1. **Reduced data requirements**: Physical constraints provide additional training signal
2. **Improved generalization**: Solutions must satisfy physical laws
3. **Better handling of ill-posed problems**: Physics provides regularization
4. **Interpretability**: Network outputs have physical meaning

### 1.2 PINNs for Image Processing: State of the Art

#### 1.2.1 Physics-Informed Image Denoising

A recent breakthrough paper (2025) demonstrates "Physics-Informed Neural Network for Denoising Images Using Nonlinear PDE" achieving **42.24 dB PSNR** (compared to ~34 dB for attention-based methods). The approach incorporates nonlinear diffusion PDEs as regularization:

$$\mathcal{L}_{PDE} = \|\nabla \cdot (D(|
abla u|) \nabla u) - u_t\|^2$$

Where $D(|
abla u|)$ is a diffusion coefficient that preserves edges while smoothing noise.

#### 1.2.2 Physics-Informed Computer Vision Review (ACM 2024)

The comprehensive review "Physics-Informed Computer Vision: A Review and Perspectives" (ACM Computing Surveys, 2024) identifies four strategies for incorporating physics into CV:

1. **Pre-processing**: Physics-guided super-resolution and image synthesis
2. **Model Design**: Hard-coding physical equations into network architecture (e.g., temporally coherent GANs for fluid dynamics)
3. **Loss Function**: Soft penalty constraints enforcing physical relationships
4. **Post-processing**: Physics-based refinement of network outputs

#### 1.2.3 3D Thermal Tomography with PINNs (2024)

A landmark study (PMC, 2024) used 3D CNN with PINN constraints for internal temperature field reconstruction from surface temperature measurements. Key findings:
- PINN-enhanced models showed **better accuracy** in predicting temperature fields
- Heat equation constraints ($\nabla^2 T = \frac{1}{\alpha} \frac{\partial T}{\partial t}$) incorporated directly into loss
- **50ms inference time** vs 24 seconds for Finite Difference Method
- Critical insight: PINNs are especially effective when **data is sparse or noisy**

### 1.3 PINNs for Image Colorization and Thermal-to-RGB Translation

#### 1.3.1 TIC-CGAN: Thermal Infrared Colorization via Conditional GAN (2018)

**Paper**: Kuang et al., "Thermal Infrared Colorization via Conditional Generative Adversarial Network" (CVPRW 2018 / Infrared Physics & Technology 2020)

This is the seminal work in thermal-to-RGB colorization. Key contributions:
- **Coarse-to-fine generator** preserving spatial details
- **Composite loss function**: $\mathcal{L} = \mathcal{L}_{content} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{perc}\mathcal{L}_{perceptual} + \lambda_{TV}\mathcal{L}_{TV}$
- Trained on KAIST multispectral pedestrian dataset (95,000+ image pairs)
- Unlike grayscale colorization, thermal-to-RGB requires estimating **both luminance and chrominance**
- No direct relation between thermal signature and perceived color

**Architecture**: Encoder-decoder with skip connections + 70x70 PatchGAN discriminator

**Key Insight for THERMAVISION-X**: The colorization problem is ill-posed because the same thermal intensity can map to multiple colors. Physics constraints can reduce this ambiguity by linking temperature to physically-plausible color ranges.

#### 1.3.2 TeX-NeRF: Neural Radiance Fields from Pseudo-TeX Vision (2024)

**Paper**: Zhong & Xu, "TeX-NeRF: Neural Radiance Fields from Pseudo-TeX Vision" (arXiv 2024)

This is the **most relevant work** for our project. TeX-NeRF performs 3D reconstruction using **only infrared images** by introducing a novel physics-based colorization approach:

**Core Innovation - Pseudo-TeX Vision:**

The thermal signal equation captures three physical quantities:

$$S_{\alpha\nu} = e_\alpha \cdot B_\nu(T_\alpha) + X_{\alpha\nu}$$

Where:
- $S_{\alpha\nu}$ = infrared radiant heat signal
- $e_\alpha$ = object emissivity (material property)
- $B_\nu(T_\alpha)$ = Planck's blackbody radiance at temperature $T_\alpha$
- $X_{\alpha\nu}$ = texture/reflection component

**Mapping to HSV Color Space:**
- **Hue (H)** $\leftarrow$ Material class / emissivity ($e$)
- **Saturation (S)** $\leftarrow$ Temperature distribution ($T$)
- **Value (V)** $\leftarrow$ Texture details ($X$)

This mapping gives each HSV channel a **physical meaning** — a critical insight for our physics-guided approach.

**Rendering Equation:**

$$C_{HSV}(r) = \sum_{i=1}^{N} T_i(1 - \exp(-\sigma_i \delta_i))(H_i, S_i, V_i)$$

**Results**: PSNR of 24.97-32.72 on different scenes, with temperature estimation MAE of 0.98-1.90 degree C.

#### 1.3.3 Thermal-NeRF: Neural Radiance Fields from Thermal Infrared (2023)

**Paper**: "Thermal-NeRF: Neural Radiance Fields from an Infrared Camera"

- Introduces **structural thermal constraints** on thermal distributions
- Uses **heat diffusion equation** as a regularization term
- Integrates pose refinement with thermal consistency
- Achieves sharper thermal images than standard NeRF

### 1.4 Key Papers Summary Table

| Paper | Year | Approach | Key Innovation | Relevance |
|-------|------|----------|----------------|-----------|
| TIC-CGAN | 2018 | Conditional GAN | Coarse-to-fine generator for thermal colorization | High - baseline architecture |
| TeX-NeRF | 2024 | NeRF + Physics | Pseudo-TeX vision maps T/e/X to HSV | **Very High** - direct model for our approach |
| Thermal-NeRF | 2023 | NeRF + Thermal | Structural thermal constraints | High - heat diffusion regularization |
| PICV Review | 2024 | Survey | Taxonomy of physics-informed CV | High - framework guidance |
| PINN Denoising | 2025 | ResU2Net + PDE | Nonlinear PDE as image prior | Medium - PDE loss implementation |
| 3D Thermal Tomography | 2024 | 3D CNN + PINN | Heat equation for internal temperature | High - PINN implementation pattern |

---

## 2. Thermal Physics Laws for IR Imagery

### 2.1 Planck's Law of Black Body Radiation

#### 2.1.1 Mathematical Formulation

Planck's law describes the spectral radiance of a black body at thermodynamic temperature $T$:

$$B_\lambda(T) = \frac{2hc^2}{\lambda^5} \frac{1}{\exp\left(\frac{hc}{\lambda k_B T}\right) - 1}$$

Where:
- $B_\lambda(T)$ = spectral radiance ($W \cdot m^{-2} \cdot sr^{-1} \cdot m^{-1}$)
- $h = 6.62607015 \times 10^{-34}$ J$\cdot$s (Planck's constant)
- $c = 2.998 \times 10^8$ m/s (speed of light)
- $k_B = 1.380649 \times 10^{-23}$ J/K (Boltzmann constant)
- $\lambda$ = wavelength (m)
- $T$ = absolute temperature (K)

**Alternative form in terms of wavenumber:**

$$B_\nu(T) = \frac{2h\nu^3}{c^2} \frac{1}{\exp\left(\frac{h\nu}{k_B T}\right) - 1}$$

**First and second radiation constants** (commonly used in remote sensing):

$$M_{\lambda}^{bb} = \frac{c_1}{\lambda^5} \frac{1}{\exp\left(\frac{c_2}{\lambda T}\right) - 1}$$

Where $c_1 = 3.7418 \times 10^8$ $W \cdot \mu m^4 / m^2$ and $c_2 = 1.4388 \times 10^4$ $\mu m \cdot K$

#### 2.1.2 Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn

# Physical constants
H = 6.62607015e-34       # Planck's constant [J*s]
C = 2.998e8              # Speed of light [m/s]
K_B = 1.380649e-23       # Boltzmann constant [J/K]
C1 = 3.7418e8            # First radiation constant [W*um^4/m^2]
C2 = 1.4388e4            # Second radiation constant [um*K]
SIGMA = 5.670374419e-8   # Stefan-Boltzmann constant [W/m^2/K^4]
WIEN_B = 2.89776829e6    # Wien's displacement constant [nm*K]

def planck_law_wavelength(wavelength, temperature):
    """
    Compute spectral radiance using Planck's law.
    
    Parameters:
    -----------
    wavelength : float or np.ndarray
        Wavelength in METERS
    temperature : float or np.ndarray
        Temperature in KELVIN
    
    Returns:
    --------
    Spectral radiance in W/(m^2 * sr * m)
    """
    wavelength = np.asarray(wavelength)
    temperature = np.asarray(temperature)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        radiance = (2 * H * C**2) / (wavelength**5) / \
                   (np.exp((H * C) / (wavelength * K_B * temperature)) - 1)
    
    return radiance

def planck_law_remote_sensing(wavelength_um, temperature_k):
    """
    Planck's law in remote sensing units (radiance in W/(m^2 * sr * um)).
    Wavelength in micrometers.
    """
    return (C1 / (wavelength_um**5)) / \
           (np.exp(C2 / (wavelength_um * temperature_k)) - 1)

# PyTorch version for differentiable computation in neural network loss
class PlanckLoss(nn.Module):
    """Differentiable Planck's law for physics-informed neural networks."""
    
    def __init__(self):
        super().__init__()
        self.h = torch.tensor(6.62607015e-34)
        self.c = torch.tensor(2.998e8)
        self.kb = torch.tensor(1.380649e-23)
        self.c1 = torch.tensor(3.7418e8)
        self.c2 = torch.tensor(1.4388e4)
    
    def forward(self, wavelength_um, temperature):
        """
        Compute spectral radiance (differentiable w.r.t temperature).
        
        Parameters:
        -----------
        wavelength_um : torch.Tensor
            Wavelength in micrometers
        temperature : torch.Tensor  
            Temperature in Kelvin (can be NN output)
        
        Returns:
        --------
        torch.Tensor : Spectral radiance
        """
        return self.c1 / (wavelength_um**5) / \
               (torch.exp(self.c2 / (wavelength_um * temperature)) - 1)
    
    def brightness_temperature(self, radiance, wavelength_um):
        """
        Convert spectral radiance to brightness temperature.
        Inverse Planck's law.
        """
        return self.c2 / (wavelength_um * torch.log(
            self.c1 / (wavelength_um**5 * radiance) + 1
        ))
```

#### 2.1.3 Neural Network Constraint Formulation

**Temperature-Radiance Consistency Loss:**

For a neural network predicting temperature $T_{pred}$ from a thermal image, we can enforce that the predicted radiance matches the observed radiance:

$$\mathcal{L}_{Planck} = \frac{1}{N} \sum_{i=1}^{N} \left| B_\lambda(T_{pred}^{(i)}) - L_{obs}^{(i)} \right|^2$$

Where $L_{obs}$ is the observed spectral radiance from the satellite sensor.

**Implementation Strategy:**
1. Convert thermal image DN values to spectral radiance (using sensor calibration)
2. Use the neural network to predict temperature and emissivity
3. Compute predicted radiance using Planck's law with predicted temperature
4. Add $\mathcal{L}_{Planck}$ to enforce radiance-temperature consistency

### 2.2 Stefan-Boltzmann Law

#### 2.2.1 Mathematical Formulation

The total power radiated per unit area of a black body is proportional to the fourth power of its absolute temperature:

$$P = \sigma A T^4$$

Where:
- $P$ = total radiated power (W)
- $\sigma = 5.670374419 \times 10^{-8}$ $W/(m^2 \cdot K^4)$ (Stefan-Boltzmann constant)
- $A$ = surface area ($m^2$)
- $T$ = absolute temperature (K)

For non-black bodies:

$$P = \epsilon \sigma A T^4$$

Where $\epsilon$ is the emissivity ($0 \leq \epsilon \leq 1$).

#### 2.2.2 Neural Network Constraint: Energy Conservation

The Stefan-Boltzmann law provides a **global energy constraint**:

$$\mathcal{L}_{SB} = \left| \int_\Omega \epsilon(x) \sigma T_{pred}(x)^4 \, dx - P_{total}^{obs} \right|^2$$

Where $P_{total}^{obs}$ is the total observed radiative power.

**Use in Colorization:**
- Hotter regions should emit significantly more radiation (T^4 relationship)
- This can guide the network to produce physically-plausible color intensity ranges
- Warmer objects should appear brighter in the rendered image

```python
class StefanBoltzmannLoss(nn.Module):
    """Stefan-Boltzmann energy conservation constraint."""
    
    def __init__(self, sigma=5.670374419e-8):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, temperature, emissivity, observed_radiance_sum):
        """
        Enforce energy conservation across the image.
        
        Parameters:
        -----------
        temperature : torch.Tensor [B, 1, H, W]
            Predicted temperature map in Kelvin
        emissivity : torch.Tensor [B, 1, H, W]
            Predicted emissivity map [0, 1]
        observed_radiance_sum : torch.Tensor [B]
            Total observed radiance per image
        
        Returns:
        --------
        torch.Tensor : Energy conservation loss
        """
        # Predicted total radiated power (per unit area)
        predicted_power = emissivity * self.sigma * temperature**4
        predicted_sum = predicted_power.sum(dim=(1, 2, 3))
        
        return torch.mean((predicted_sum - observed_radiance_sum)**2)
```

### 2.3 Wien's Displacement Law

#### 2.3.1 Mathematical Formulation

The peak emission wavelength of a black body is inversely proportional to its temperature:

$$\lambda_{max} = \frac{b}{T}$$

Where:
- $\lambda_{max}$ = wavelength of maximum emission (m)
- $b = 2.89776829 \times 10^{-3}$ $m \cdot K$ (Wien's displacement constant)
- $T$ = absolute temperature (K)

**Examples:**
- Human body (310 K): $\lambda_{max} \approx 9.35$ $\mu m$ (LWIR)
- Room temperature (300 K): $\lambda_{max} \approx 9.66$ $\mu m$ (LWIR)
- Sun (5778 K): $\lambda_{max} \approx 502$ nm (visible green)
- Hot metal (1500 K): $\lambda_{max} \approx 1.93$ $\mu m$ (SWIR)

#### 2.3.2 Temperature-to-Color Mapping via Wien's Law

**Critical Insight for Colorization**: Wien's law provides a **physics-based mapping from temperature to color**:

```python
def temperature_to_peak_wavelength(temperature_k):
    """Convert temperature to peak emission wavelength in nanometers."""
    WIEN_B_NM = 2.89776829e6  # nm*K
    return WIEN_B_NM / temperature_k

def wavelength_to_rgb(wavelength_nm):
    """
    Approximate RGB color for a given wavelength in the visible spectrum.
    Uses a simplified conversion from spectral wavelength to RGB.
    """
    if wavelength_nm < 380 or wavelength_nm > 780:
        return (0, 0, 0)
    
    # Simplified RGB response curves
    if 380 <= wavelength_nm < 440:
        R = -(wavelength_nm - 440) / (440 - 380)
        G = 0
        B = 1
    elif 440 <= wavelength_nm < 490:
        R = 0
        G = (wavelength_nm - 440) / (490 - 440)
        B = 1
    elif 490 <= wavelength_nm < 510:
        R = 0
        G = 1
        B = -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm < 580:
        R = (wavelength_nm - 510) / (580 - 510)
        G = 1
        B = 0
    elif 580 <= wavelength_nm < 645:
        R = 1
        G = -(wavelength_nm - 645) / (645 - 580)
        B = 0
    elif 645 <= wavelength_nm <= 780:
        R = 1
        G = 0
        B = 0
    
    return (R, G, B)

def temperature_to_approximate_color(temperature_k):
    """
    Map temperature to an approximate color using Wien's displacement law.
    Note: Most terrestrial temperatures peak in IR, not visible.
    For colorization, we use a shifted/scaled mapping.
    """
    peak_wavelength = temperature_to_peak_wavelength(temperature_k)
    
    # For temperatures in typical Earth range (250K - 400K),
    # peak is in LWIR. We map to visible spectrum for display.
    # Using a logarithmic shift to bring into visible range
    if temperature_k < 500:
        # Map typical Earth temperatures to visible range
        # 250K -> deep blue, 400K -> red
        t_norm = (temperature_k - 250) / 150  # [0, 1]
        r = min(1.0, max(0.0, t_norm * 3))
        g = min(1.0, max(0.0, 1.0 - abs(t_norm - 0.5) * 3))
        b = min(1.0, max(0.0, (1 - t_norm) * 3))
        return (r, g, b)
    else:
        return wavelength_to_rgb(peak_wavelength)
```

#### 2.3.3 Neural Network Constraint: Spectral Peak Consistency

For a network predicting temperature from radiance measurements, Wien's law provides a constraint on the spectral shape:

$$\mathcal{L}_{Wien} = \sum_k \left| \lambda_{max}^{(k)} - \frac{b}{T^{(k)}} \right|^2$$

Where $\lambda_{max}^{(k)}$ is the observed peak wavelength for pixel $k$ and $T^{(k)}$ is the predicted temperature.

### 2.4 Kirchhoff's Law of Thermal Radiation

#### 2.4.1 Mathematical Formulation

For a body in thermal equilibrium at a given wavelength:

$$\epsilon_\lambda = \alpha_\lambda$$

Where $\epsilon_\lambda$ is spectral emissivity and $\alpha_\lambda$ is spectral absorptivity.

For opaque objects: $\epsilon_\lambda + \rho_\lambda = 1$ (where $\rho_\lambda$ is reflectivity)

#### 2.4.2 Material Emissivity Database

| Material | Emissivity ($\epsilon$) | Temperature Range (K) |
|----------|------------------------|----------------------|
| Black body (ideal) | 1.00 | All |
| Water | 0.95-0.99 | 273-373 |
| Vegetation (green) | 0.95-0.98 | 273-323 |
| Soil (dry) | 0.92-0.95 | 273-373 |
| Concrete | 0.88-0.94 | 273-373 |
| Sand | 0.84-0.90 | 273-373 |
| Snow (fresh) | 0.98-0.99 | 253-273 |
| Ice | 0.96-0.98 | 253-273 |
| Asphalt | 0.88-0.93 | 273-373 |
| Metal (polished) | 0.02-0.20 | 273-373 |
| Brick | 0.93-0.96 | 273-373 |

### 2.5 Temperature-to-Color Physics for Colorization

#### 2.5.1 Standard Thermal Colormaps

Thermal imaging systems use several standard colormaps:

1. **Iron/Bow**: Black -> Blue -> Purple -> Red -> Orange -> Yellow -> White
2. **Rainbow**: Violet -> Blue -> Green -> Yellow -> Orange -> Red
3. **Arctic**: White -> Blue -> Purple (for cold temperatures)
4. **Hot**: Black -> Red -> Orange -> Yellow -> White
5. **Jet**: Blue -> Cyan -> Yellow -> Red

#### 2.5.2 Physics-Guided Colormap Generation

For a physics-informed approach, we can derive colormaps from actual blackbody radiation curves:

```python
import numpy as np

def generate_blackbody_colormap(temperatures_k, output_resolution=256):
    """
    Generate a physics-based colormap using blackbody radiation curves.
    Maps temperature to RGB values by integrating Planck's law over
    human eye response curves.
    """
    # CIE 1931 color matching functions (simplified)
    # Approximate response curves for RGB cones
    wavelengths = np.linspace(380, 780, 401)  # nm
    
    # Simplified cone response functions
    def s_response(wl):
        """Short wavelength (blue) response"""
        return np.exp(-((wl - 445) / 40)**2)
    
    def m_response(wl):
        """Medium wavelength (green) response"""
        return np.exp(-((wl - 535) / 40)**2)
    
    def l_response(wl):
        """Long wavelength (red) response"""
        return np.exp(-((wl - 575) / 40)**2)
    
    s = s_response(wavelengths)
    m = m_response(wavelengths)
    l = l_response(wavelengths)
    
    colormap = np.zeros((len(temperatures_k), 3))
    
    for i, T in enumerate(temperatures_k):
        # Compute blackbody spectrum for this temperature
        wl_m = wavelengths * 1e-9
        radiance = planck_law_wavelength(wl_m, T)
        
        # Integrate over wavelength weighted by cone responses
        colormap[i, 0] = np.trapz(radiance * l, wavelengths)  # R
        colormap[i, 1] = np.trapz(radiance * m, wavelengths)  # G
        colormap[i, 2] = np.trapz(radiance * s, wavelengths)  # B
    
    # Normalize to [0, 1]
    colormap = colormap / colormap.max(axis=0)
    
    return colormap

def physics_guided_temperature_to_color(temperature_k, t_min=250, t_max=400):
    """
    Map temperature to color using a physics-based approach.
    For temperatures where peak emission is in IR (T < ~500K),
    we use a shifted/scaled mapping that captures the relative
    temperature differences.
    """
    # Normalize temperature to [0, 1]
    t_norm = np.clip((temperature_k - t_min) / (t_max - t_min), 0, 1)
    
    # Use iron/bow colormap (commonly used in thermal imaging)
    # This captures: cold (blue) -> moderate (purple/red) -> hot (yellow/white)
    colors = np.array([
        [0, 0, 0],           # 0.0: Black
        [0, 0, 0.5],         # 0.1: Dark blue
        [0, 0, 1],           # 0.2: Blue
        [0.5, 0, 1],         # 0.3: Purple-blue
        [1, 0, 1],           # 0.4: Magenta
        [1, 0, 0.5],         # 0.5: Pink-red
        [1, 0, 0],           # 0.6: Red
        [1, 0.5, 0],         # 0.7: Orange
        [1, 1, 0],           # 0.8: Yellow
        [1, 1, 0.5],         # 0.9: Light yellow
        [1, 1, 1],           # 1.0: White
    ])
    
    # Interpolate
    indices = t_norm * (len(colors) - 1)
    idx_low = int(np.floor(indices))
    idx_high = min(idx_low + 1, len(colors) - 1)
    frac = indices - idx_low
    
    color = colors[idx_low] * (1 - frac) + colors[idx_high] * frac
    return np.clip(color, 0, 1)
```

---

## 3. Satellite Remote Sensing Physics

### 3.1 Thermal Infrared Sensors Overview

#### 3.1.1 Landsat TIRS (Thermal Infrared Sensor)

**Landsat 8/9 TIRS Specifications:**

| Parameter | Band 10 | Band 11 |
|-----------|---------|---------|
| Wavelength Range | 10.60-11.19 $\mu m$ | 11.50-12.51 $\mu m$ |
| Spatial Resolution | 100m (resampled to 30m) | 100m (resampled to 30m) |
| NEAT (Noise Equivalent dT) | <0.4 K | <0.4 K |
| Quantization | 16-bit | 16-bit |
| Saturation Radiance | ~70 $W/(m^2 \cdot sr \cdot \mu m)$ | ~35 $W/(m^2 \cdot sr \cdot \mu m)$ |

**Landsat TIRS-2 (on Landsat 9):** Improved stray light correction

#### 3.1.2 MODIS Thermal Bands

**MODIS TIR Bands for LST Retrieval:**

| Band | Wavelength ($\mu m$) | Primary Use | Spatial Resolution |
|------|---------------------|-------------|-------------------|
| 20 | 3.660-3.840 | Day/night LST, fires | 1 km |
| 22 | 3.929-3.989 | Day/night LST | 1 km |
| 23 | 4.020-4.080 | Day/night LST | 1 km |
| 29 | 8.400-8.700 | Cloud properties | 1 km |
| 31 | 10.780-11.280 | Split window LST | 1 km |
| 32 | 11.770-12.270 | Split window LST | 1 km |
| 33 | 13.185-13.485 | Cloud/cirrus | 1 km |

#### 3.1.3 ASTER TIR

**ASTER Thermal Bands:**
- 5 TIR bands: 8.125-8.475, 8.625-8.825, 8.925-9.275, 10.25-10.95, 10.95-11.65 $\mu m$
- 90m spatial resolution
- Temperature-Emissivity Separation (TES) algorithm

### 3.2 Spectral Radiance to Brightness Temperature Conversion

#### 3.2.1 Landsat TIRS Conversion

The conversion from Top-of-Atmosphere (TOA) spectral radiance to brightness temperature:

$$T = \frac{K_2}{\ln\left(\frac{K_1}{L_\lambda} + 1\right)}$$

Where:
- $T$ = Top of atmosphere brightness temperature (K)
- $L_\lambda$ = TOA spectral radiance ($W/(m^2 \cdot sr \cdot \mu m)$)
- $K_1$, $K_2$ = Band-specific thermal conversion constants from metadata

**For Landsat 8 TIRS:**
- Band 10: $K_1 = 774.89$, $K_2 = 1321.08$
- Band 11: $K_1 = 480.89$, $K_2 = 1201.14$

**For Landsat 9 TIRS-2:**
- Band 10: $K_1 = 774.89$, $K_2 = 1321.08$
- Band 11: $K_1 = 480.89$, $K_2 = 1201.14$

```python
def landsat_radiance_to_brightness_temp(L_lambda, K1, K2):
    """
    Convert Landsat TIRS spectral radiance to brightness temperature.
    
    Parameters:
    -----------
    L_lambda : float or np.ndarray
        TOA spectral radiance in W/(m^2 * sr * um)
    K1 : float
        Band-specific thermal conversion constant
    K2 : float
        Band-specific thermal conversion constant
    
    Returns:
    --------
    float or np.ndarray : Brightness temperature in Kelvin
    """
    return K2 / np.log(K1 / L_lambda + 1)

def landsat_dn_to_radiance(dn, rad_mult, rad_add):
    """
    Convert Landsat digital numbers to spectral radiance.
    
    Parameters:
    -----------
    dn : int or np.ndarray
        Digital number from Landsat image
    rad_mult : float
        Radiance multiplication factor (from metadata)
    rad_add : float
        Radiance addition factor (from metadata)
    
    Returns:
    --------
    float or np.ndarray : Spectral radiance
    """
    return dn * rad_mult + rad_add
```

#### 3.2.2 MODIS Brightness Temperature Conversion

MODIS uses a lookup-table approach based on the inverse Planck function:

$$T_b = \frac{c_2}{\lambda_{eff} \cdot \ln\left(\frac{c_1}{\lambda_{eff}^5 \cdot L_\lambda} + 1\right)}$$

Where $\lambda_{eff}$ is the effective band wavelength.

### 3.3 Atmospheric Correction Methods

#### 3.3.1 The Radiative Transfer Equation

The TOA radiance measured by a satellite sensor is:

$$L_\lambda = \tau_\lambda \cdot [\epsilon_\lambda \cdot B_\lambda(T_s) + (1 - \epsilon_\lambda) \cdot L_\lambda^{down}] + L_\lambda^{up}$$

Where:
- $L_\lambda$ = TOA spectral radiance
- $\tau_\lambda$ = atmospheric transmittance
- $\epsilon_\lambda$ = surface emissivity
- $B_\lambda(T_s)$ = Planck radiance at surface temperature $T_s$
- $L_\lambda^{down}$ = downwelling atmospheric radiance
- $L_\lambda^{up}$ = upwelling atmospheric radiance

#### 3.3.2 Atmospheric Correction Methods

**1. Radiative Transfer Equation (RTE) Method:**
- Uses atmospheric profile data (temperature, humidity)
- Simulates atmospheric parameters with MODTRAN/6S
- Most accurate but requires atmospheric data

**2. Single-Channel (SC) Algorithm:**

$$T_s = \gamma \cdot \left[\frac{1}{\epsilon}\left(\psi_1 \cdot L_\lambda + \psi_2\right) + \psi_3\right] + \delta$$

Where $\psi_1$, $\psi_2$, $\psi_3$ are atmospheric functions and $\gamma$, $\delta$ are computed from Planck's law derivatives.

**3. Split-Window (SW) Algorithm:**

The generalized split-window algorithm for MODIS:

$$T_s = A_0 + A_1 \cdot T_{31} + A_2 \cdot (T_{31} - T_{32}) + A_3 \cdot (T_{31} - T_{32})^2$$

Where:
- $T_{31}$, $T_{32}$ = brightness temperatures in bands 31 and 32
- $A_0$, $A_1$, $A_2$, $A_3$ = coefficients dependent on view angle and water vapor

**For Landsat 8/9 (Mao et al. SW algorithm):**

$$T_s = A_0 + \frac{A_1 + A_2 \cdot \frac{1 - \epsilon}{\epsilon} + A_3 \cdot \frac{\Delta \epsilon}{\epsilon^2}}{2} \cdot (T_{10} + T_{11}) + \frac{A_4 + A_5 \cdot \frac{1 - \epsilon}{\epsilon} + A_6 \cdot \frac{\Delta \epsilon}{\epsilon^2}}{2} \cdot (T_{10} - T_{11})$$

Where:
- $T_{10}$, $T_{11}$ = brightness temperatures in bands 10 and 11
- $\epsilon = (\epsilon_{10} + \epsilon_{11}) / 2$ (mean emissivity)
- $\Delta \epsilon = \epsilon_{10} - \epsilon_{11}$ (emissivity difference)
- $A_0$-$A_6$ = atmospheric coefficients

### 3.4 Land Surface Temperature (LST) Retrieval Algorithms

#### 3.4.1 Complete LST Retrieval Pipeline

```python
def lst_retrieval_pipeline(dn_b10, dn_b11, metadata, atmospheric_profile=None):
    """
    Complete LST retrieval pipeline for Landsat 8/9 TIRS.
    
    Parameters:
    -----------
    dn_b10, dn_b11 : np.ndarray
        Digital numbers for bands 10 and 11
    metadata : dict
        Landsat metadata with calibration constants
    atmospheric_profile : dict, optional
        Atmospheric water vapor content for SW correction
    
    Returns:
    --------
    np.ndarray : Land Surface Temperature in Kelvin
    """
    # Step 1: Convert DN to TOA radiance
    L_b10 = dn_b10 * metadata['RADIANCE_MULT_BAND_10'] + metadata['RADIANCE_ADD_BAND_10']
    L_b11 = dn_b11 * metadata['RADIANCE_MULT_BAND_11'] + metadata['RADIANCE_ADD_BAND_11']
    
    # Step 2: Convert radiance to brightness temperature
    K1_b10, K2_b10 = metadata['K1_CONSTANT_BAND_10'], metadata['K2_CONSTANT_BAND_10']
    K1_b11, K2_b11 = metadata['K1_CONSTANT_BAND_11'], metadata['K2_CONSTANT_BAND_11']
    
    BT_b10 = K2_b10 / np.log(K1_b10 / L_b10 + 1)
    BT_b11 = K2_b11 / np.log(K1_b11 / L_b11 + 1)
    
    # Step 3: Estimate emissivity (from land cover or NDVI)
    epsilon_b10, epsilon_b11 = estimate_emissivity(dn_b10, metadata)
    
    # Step 4: Apply split-window atmospheric correction
    if atmospheric_profile is not None:
        w = atmospheric_profile['water_vapor']
        lst = split_window_correction(
            BT_b10, BT_b11, 
            epsilon_b10, epsilon_b11,
            w, metadata['SUN_ELEVATION']
        )
    else:
        # Use single-channel method with estimated atmospheric parameters
        lst = single_channel_correction(BT_b10, epsilon_b10, L_b10)
    
    return lst

def estimate_emissivity(ndvi, land_cover_type=None):
    """
    Estimate land surface emissivity using NDVI threshold method.
    
    NDVI < 0.2: Soil (emissivity ~0.97)
    NDVI > 0.5: Vegetation (emissivity ~0.99)
    0.2 <= NDVI <= 0.5: Mixed pixel (linear interpolation)
    """
    epsilon_soil = 0.970
    epsilon_veg = 0.990
    
    if land_cover_type == 'water':
        return 0.995
    elif land_cover_type == 'urban':
        return 0.960
    
    # NDVI-based estimation
    emissivity = np.where(
        ndvi < 0.2, epsilon_soil,
        np.where(ndvi > 0.5, epsilon_veg,
                 epsilon_veg * (ndvi - 0.2) / 0.3 + 
                 epsilon_soil * (0.5 - ndvi) / 0.3)
    )
    return emissivity

def split_window_correction(T10, T11, e10, e11, wv, sun_elev):
    """
    Split-window atmospheric correction for Landsat TIRS.
    Based on Rozenstein et al. (2014) algorithm.
    """
    e_mean = (e10 + e11) / 2
    de = e10 - e11
    
    # Atmospheric coefficients (simplified - actual values from lookup tables)
    A1 = -0.功68
    A2 = 1.功10
    A3 = 0.功2138
    
    lst = A1 + A2 * (T10 + T11) / 2 + A3 * (T10 - T11) / 2
    
    # Emissivity correction
    lst += 54 * (1 - e_mean) - 38 * de
    
    return lst
```

#### 3.4.2 Neural Network-Based LST Retrieval

Recent approaches use neural networks to learn the LST retrieval mapping:

```python
class LSTNeuralNetwork(nn.Module):
    """
    Neural network for Land Surface Temperature retrieval from satellite data.
    Physics-informed version that incorporates Planck's law constraints.
    """
    
    def __init__(self, input_channels=2, hidden_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
        )
        
        # Temperature prediction head
        self.temp_head = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        
        # Emissivity prediction head
        self.emissivity_head = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),  # Emissivity in [0, 1]
        )
    
    def forward(self, x):
        features = self.encoder(x)
        temperature = self.temp_head(features)
        emissivity = self.emissivity_head(features)
        return temperature, emissivity
```

### 3.5 Integration into Colorization Pipeline

**Pipeline Architecture for THERMAVISION-X:**

```
Raw Thermal Image (DN) 
    |
    v
[DN to Radiance Conversion] ---> Spectral Radiance L_lambda
    |
    v
[Brightness Temperature] ---> T_brightness
    |
    v
[Atmospheric Correction] ---> T_surface (with physics constraints)
    |
    v
[Emissivity Estimation] ---> epsilon(x,y) (material map)
    |
    v
[Physics-Informed Colorizer] ---> Physics-guided RGB
    |                              |
    |                              v
    |                    [Planck constraint loss]
    |                              |
    |                    [Stefan-Boltzmann loss]
    |                              |
    |                    [Temperature smoothness loss]
    |                              |
    v                              v
              Final Colorized Image
```

---

## 4. Implementing Physics Constraints in Deep Learning

### 4.1 Physics-Based Regularization Terms

#### 4.1.1 Complete Physics-Informed Loss Function

The total loss function for THERMAVISION-X combines data-driven and physics-based terms:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{Planck} + \lambda_3 \mathcal{L}_{SB} + \lambda_4 \mathcal{L}_{smooth} + \lambda_5 \mathcal{L}_{grad} + \lambda_6 \mathcal{L}_{temporal}$$

Where:
- $\mathcal{L}_{recon}$ = Reconstruction loss (perceptual + pixel-wise)
- $\mathcal{L}_{Planck}$ = Planck's law radiance-temperature consistency
- $\mathcal{L}_{SB}$ = Stefan-Boltzmann energy conservation
- $\mathcal{L}_{smooth}$ = Temperature spatial smoothness
- $\mathcal{L}_{grad}$ = Thermal gradient preservation
- $\mathcal{L}_{temporal}$ = Temporal consistency (for video)

#### 4.1.2 Planck's Law Constraint (Differentiable)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class PhysicsInformedThermalLoss(nn.Module):
    """
    Complete physics-informed loss function for thermal image colorization.
    Implements Planck's law, Stefan-Boltzmann, and smoothness constraints.
    """
    
    def __init__(self, lambda_planck=1.0, lambda_sb=0.1, 
                 lambda_smooth=0.5, lambda_grad=0.3,
                 wavelength_um=11.0, device='cuda'):
        super().__init__()
        self.lambda_planck = lambda_planck
        self.lambda_sb = lambda_sb
        self.lambda_smooth = lambda_smooth
        self.lambda_grad = lambda_grad
        self.wavelength_um = wavelength_um
        self.device = device
        
        # Physical constants as torch tensors
        self.c1 = torch.tensor(3.7418e8, device=device)   # W*um^4/m^2
        self.c2 = torch.tensor(1.4388e4, device=device)   # um*K
        self.sigma = torch.tensor(5.670374419e-8, device=device)
        
        # Perceptual loss components
        self.perceptual_loss = PerceptualLoss().to(device)
    
    def planck_radiance(self, temperature, wavelength_um=None):
        """Differentiable Planck's law computation."""
        if wavelength_um is None:
            wavelength_um = self.wavelength_um
        wavelength_um = torch.tensor(wavelength_um, device=self.device)
        
        # Ensure temperature is positive (physical constraint)
        temp_safe = torch.clamp(temperature, min=200.0, max=500.0)
        
        radiance = self.c1 / (wavelength_um**5) / \
                   (torch.exp(self.c2 / (wavelength_um * temp_safe)) - 1)
        return radiance
    
    def planck_constraint(self, pred_temperature, observed_radiance):
        """
        Enforce that predicted temperature is consistent with Planck's law.
        
        Parameters:
        -----------
        pred_temperature : torch.Tensor [B, 1, H, W]
            Predicted temperature map in Kelvin
        observed_radiance : torch.Tensor [B, 1, H, W]
            Observed spectral radiance from sensor
        
        Returns:
        --------
        torch.Tensor : Planck consistency loss
        """
        pred_radiance = self.planck_radiance(pred_temperature)
        return F.mse_loss(pred_radiance, observed_radiance)
    
    def stefan_boltzmann_constraint(self, temperature, emissivity):
        """
        Enforce energy conservation via Stefan-Boltzmann law.
        
        Parameters:
        -----------
        temperature : torch.Tensor [B, 1, H, W]
            Predicted temperature map
        emissivity : torch.Tensor [B, 1, H, W]
            Predicted emissivity map [0, 1]
        
        Returns:
        --------
        torch.Tensor : Energy conservation loss
        """
        # Local radiated power
        local_power = emissivity * self.sigma * temperature**4
        
        # Enforce smooth variation of power across similar materials
        power_mean = local_power.mean(dim=(2, 3), keepdim=True)
        power_variance = ((local_power - power_mean)**2).mean()
        
        # Penalize unphysically high variance in homogeneous regions
        return power_variance
    
    def temperature_smoothness(self, temperature):
        """
        Enforce spatial smoothness of temperature field.
        Based on heat diffusion: regions close in space should have similar T.
        """
        # Laplacian of temperature field
        laplacian_t = self.compute_laplacian(temperature)
        return torch.mean(laplacian_t**2)
    
    def thermal_gradient_preservation(self, temperature, input_thermal):
        """
        Preserve thermal gradients from input in the predicted temperature.
        Stronger gradients in input should correspond to stronger in output.
        """
        # Compute gradients
        grad_input = self.spatial_gradient(input_thermal)
        grad_temp = self.spatial_gradient(temperature)
        
        # Normalize
        grad_input_norm = F.normalize(grad_input, dim=1)
        grad_temp_norm = F.normalize(grad_temp, dim=1)
        
        # Cosine similarity between gradient directions
        cos_sim = F.cosine_similarity(grad_input_norm, grad_temp_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def compute_laplacian(self, x):
        """Compute Laplacian using finite difference approximation."""
        # 2D Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return F.conv2d(x, laplacian_kernel, padding=1)
    
    def spatial_gradient(self, x):
        """Compute spatial gradient using Sobel operators."""
        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        
        return torch.cat([grad_x, grad_y], dim=1)
    
    def forward(self, pred_rgb, pred_temperature, pred_emissivity,
                target_rgb, observed_radiance, input_thermal):
        """
        Compute total physics-informed loss.
        
        Parameters:
        -----------
        pred_rgb : torch.Tensor [B, 3, H, W]
            Predicted colorized RGB image
        pred_temperature : torch.Tensor [B, 1, H, W]
            Predicted temperature map
        pred_emissivity : torch.Tensor [B, 1, H, W]
            Predicted emissivity map
        target_rgb : torch.Tensor [B, 3, H, W]
            Target RGB image (if available)
        observed_radiance : torch.Tensor [B, 1, H, W]
            Observed spectral radiance
        input_thermal : torch.Tensor [B, 1, H, W]
            Input thermal image
        
        Returns:
        --------
        dict : Dictionary of loss components
        """
        losses = {}
        
        # Reconstruction loss (data-driven)
        if target_rgb is not None:
            losses['recon'] = self.perceptual_loss(pred_rgb, target_rgb)
        else:
            losses['recon'] = torch.tensor(0.0, device=self.device)
        
        # Planck's law constraint
        losses['planck'] = self.planck_constraint(
            pred_temperature, observed_radiance
        )
        
        # Stefan-Boltzmann constraint
        losses['sb'] = self.stefan_boltzmann_constraint(
            pred_temperature, pred_emissivity
        )
        
        # Temperature smoothness
        losses['smooth'] = self.temperature_smoothness(pred_temperature)
        
        # Gradient preservation
        losses['grad'] = self.thermal_gradient_preservation(
            pred_temperature, input_thermal
        )
        
        # Total loss
        total_loss = (losses['recon'] + 
                      self.lambda_planck * losses['planck'] +
                      self.lambda_sb * losses['sb'] +
                      self.lambda_smooth * losses['smooth'] +
                      self.lambda_grad * losses['grad'])
        
        losses['total'] = total_loss
        return losses
```

### 4.2 Temperature Consistency Constraints

#### 4.2.1 Material-Based Temperature Prior

Different materials have characteristic temperature ranges based on their thermal properties:

```python
class MaterialTemperaturePrior(nn.Module):
    """
    Enforce temperature ranges based on material classes.
    Uses semantic segmentation to identify materials and apply
    physics-based temperature priors.
    """
    
    # Typical Earth surface temperature ranges (K)
    MATERIAL_TEMP_RANGES = {
        'water': (270, 320),
        'vegetation': (270, 330),
        'soil': (270, 340),
        'urban': (270, 350),
        'snow': (250, 280),
        'ice': (250, 275),
        'desert': (270, 360),
        'cloud': (220, 290),
    }
    
    def __init__(self, num_material_classes=8):
        super().__init__()
        self.num_classes = num_material_classes
    
    def forward(self, temperature, material_probs):
        """
        Apply material-based temperature constraints.
        
        Parameters:
        -----------
        temperature : torch.Tensor [B, 1, H, W]
            Predicted temperature
        material_probs : torch.Tensor [B, C, H, W]
            Soft material class probabilities
        
        Returns:
        --------
        torch.Tensor : Temperature prior loss
        """
        loss = 0.0
        
        for i, (material, (t_min, t_max)) in enumerate(
            self.MATERIAL_TEMP_RANGES.items()
        ):
            if i >= self.num_classes:
                break
            
            # Probability of this material
            p_material = material_probs[:, i:i+1]
            
            # Penalize temperatures outside material range (weighted by probability)
            below_min = torch.clamp(t_min - temperature, min=0)
            above_max = torch.clamp(temperature - t_max, min=0)
            
            loss += (p_material * (below_min**2 + above_max**2)).mean()
        
        return loss
```

### 4.3 Spectral Response Function Integration

```python
class SpectralResponseLoss(nn.Module):
    """
    Integrate satellite sensor spectral response functions into the loss.
    Ensures predicted radiance matches the sensor's spectral characteristics.
    """
    
    def __init__(self, sensor='landsat8_tirs', band=10):
        super().__init__()
        self.sensor = sensor
        self.band = band
        
        # Load spectral response function
        self.srf = self.load_spectral_response(sensor, band)
    
    def load_spectral_response(self, sensor, band):
        """Load normalized spectral response function for given sensor/band."""
        # Simplified SRF for Landsat 8 TIRS Band 10
        if sensor == 'landsat8_tirs' and band == 10:
            wavelengths = torch.linspace(10.0, 11.5, 150)
            # Approximate Gaussian response
            srf = torch.exp(-((wavelengths - 10.9)**2) / (2 * 0.3**2))
            srf = srf / srf.sum()  # Normalize
            return wavelengths, srf
        
        # Default: delta function at center wavelength
        return None, None
    
    def compute_band_radiance(self, temperature):
        """
        Compute sensor-band-specific radiance by integrating
        Planck's law weighted by the spectral response function.
        """
        wavelengths, srf = self.srf
        
        # Planck radiance at each wavelength
        radiance = self.planck_over_spectrum(temperature, wavelengths)
        
        # Weight by spectral response function
        band_radiance = (radiance * srf).sum()
        
        return band_radiance
```

### 4.4 Heat Equation as Neural Network Constraint

The heat diffusion equation can be used to regularize temperature predictions:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

Where $\alpha$ is the thermal diffusivity ($m^2/s$).

```python
class HeatEquationConstraint(nn.Module):
    """
    Enforce heat diffusion equation as a soft constraint on predicted
    temperature fields. Assumes temporal sequences of thermal images.
    """
    
    def __init__(self, alpha=1e-6):
        super().__init__()
        self.alpha = alpha  # Thermal diffusivity
    
    def forward(self, temperature_sequence, dt):
        """
        Compute heat equation residual.
        
        Parameters:
        -----------
        temperature_sequence : torch.Tensor [B, T, 1, H, W]
            Sequence of temperature maps over time
        dt : float
            Time step between frames (seconds)
        
        Returns:
        --------
        torch.Tensor : Heat equation residual loss
        """
        # Time derivative: dT/dt
        dT_dt = (temperature_sequence[:, 1:] - temperature_sequence[:, :-1]) / dt
        
        # Spatial Laplacian: nabla^2 T
        T_mid = (temperature_sequence[:, 1:] + temperature_sequence[:, :-1]) / 2
        laplacian_T = self.compute_laplacian_3d(T_mid)
        
        # Residual: dT/dt - alpha * nabla^2 T
        residual = dT_dt - self.alpha * laplacian_T
        
        return torch.mean(residual**2)
    
    def compute_laplacian_3d(self, x):
        """Compute 3D Laplacian (spatial + temporal)."""
        # 3D Laplacian kernel
        kernel = torch.tensor([
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        
        return F.conv3d(x, kernel, padding=1)
```

### 4.5 Adaptive Loss Weighting

Adaptive weighting balances the contribution of different loss terms:

```python
class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive weighting scheme that dynamically adjusts loss weights
    based on relative magnitudes during training.
    Prioritizes higher-loss terms, reduces emphasis on less significant terms.
    """
    
    def __init__(self, num_terms, initial_weights=None):
        super().__init__()
        if initial_weights is None:
            initial_weights = torch.ones(num_terms)
        self.weights = nn.Parameter(initial_weights)
        self.num_terms = num_terms
    
    def forward(self, loss_terms):
        """
        Compute weighted sum of loss terms with adaptive weights.
        
        Parameters:
        -----------
        loss_terms : list of torch.Tensor
            Individual loss term values
        
        Returns:
        --------
        torch.Tensor : Weighted total loss
        """
        # Compute relative magnitudes
        magnitudes = torch.stack([l.item() for l in loss_terms])
        
        # Normalize to get relative importance
        relative = magnitudes / (magnitudes.sum() + 1e-8)
        
        # Update weights: higher magnitude -> higher weight
        with torch.no_grad():
            self.weights.data = self.weights.data * 0.9 + relative * 0.1
        
        # Compute weighted loss
        total = sum(w * l for w, l in zip(self.weights, loss_terms))
        
        return total
```

### 4.6 Differentiable Physics Layer (Key Implementation)

```python
class DifferentiablePhysicsLayer(nn.Module):
    """
    A differentiable layer that enforces physical constraints
    through the network's forward pass. Uses implicit differentiation
    to ensure outputs satisfy physical laws.
    
    This is the KEY component for THERMAVISION-X.
    """
    
    def __init__(self, physics_loss_fn, max_iters=10, tol=1e-4):
        super().__init__()
        self.physics_loss = physics_loss_fn
        self.max_iters = max_iters
        self.tol = tol
    
    def forward(self, temperature_pred, observed_radiance):
        """
        Project network output onto the physics manifold.
        Ensures temperature prediction is consistent with Planck's law.
        """
        T = temperature_pred.clone()
        
        # Fixed-point iteration to satisfy physics constraint
        for _ in range(self.max_iters):
            T_prev = T.clone()
            
            # Compute physics residual and gradient
            T.requires_grad_(True)
            radiance_pred = planck_law_torch(T)
            residual = radiance_pred - observed_radiance
            
            # Newton step: T_new = T - residual / (dR/dT)
            dR_dT = grad(residual.sum(), T, create_graph=True)[0]
            T = T - residual / (dR_dT + 1e-8)
            
            # Check convergence
            if torch.max(torch.abs(T - T_prev)) < self.tol:
                break
        
        return T

def planck_law_torch(temperature, wavelength_um=11.0):
    """Differentiable Planck's law."""
    c1 = 3.7418e8
    c2 = 1.4388e4
    return c1 / (wavelength_um**5) / (torch.exp(c2 / (wavelength_um * temperature)) - 1)
```

---

## 5. Synthesis: Physics-Guided Zero-Shot Thermal Colorization Architecture

### 5.1 Proposed THERMAVISION-X Architecture

Based on our research, we propose the following architecture for the hackathon:

```
THERMAVISION-X SYSTEM ARCHITECTURE
====================================

Input: Thermal Infrared Image (grayscale, from satellite or camera)
       Optional: Metadata (calibration constants, atmospheric parameters)

MODULE 1: PHYSICS PREPROCESSING
-------------------------------
[Input Thermal Image]
    |
    v
[DN to Spectral Radiance]  <-- Landsat/MODIS calibration constants
    |
    v
[Brightness Temperature]   <-- Inverse Planck's law (K1, K2 constants)
    |
    v
[Atmospheric Correction]   <-- Split-window or single-channel algorithm
    |                          (produces: surface temperature T_s(x,y))
    v
[Emissivity Estimation]    <-- NDVI-based or classification-based
    |                          (produces: emissivity map e(x,y))
    v
Output: Physics Feature Stack [T_s, e, radiance, uncertainty]

MODULE 2: PHYSICS-INFORMED NEURAL COLORIZER
-------------------------------------------
[Physics Feature Stack]
    |
    v
[Feature Encoder]          <-- UNet-style encoder with ResBlocks
    |                          Extracts multi-scale features
    v
[Physics Attention]        <-- Cross-attention between physics features
    |                          and learned color embeddings
    v
[Color Decoder]            <-- UNet-style decoder with skip connections
    |                          Produces: HSV color channels
    v
[Physics Projection Layer] <-- DifferentiablePhysicsLayer
    |                          Enforces Planck + Stefan-Boltzmann constraints
    v
[HSV to RGB Conversion]    <-- Standard color space conversion
    v
Output: Colorized RGB Image

MODULE 3: PHYSICS LOSS ENGINE
-----------------------------
During training, the following losses are computed:

1. L_reconstruction = PerceptualLoss(pred_rgb, target_rgb)
   --> Data-driven color fidelity

2. L_planck = MSE(planck(T_pred), L_observed)
   --> Temperature-radiance consistency

3. L_stefan_boltzmann = Var(e * sigma * T^4) across materials
   --> Energy conservation

4. L_smooth = Mean(Laplacian(T_pred)^2)
   --> Spatial temperature smoothness (heat diffusion)

5. L_gradient = 1 - CosineSimilarity(grad(T_pred), grad(T_input))
   --> Thermal gradient preservation

6. L_material = MaterialTemperaturePrior(T_pred, material_seg)
   --> Physics-based temperature ranges per material

Total Loss = w1*L_recon + w2*L_planck + w3*L_SB + w4*L_smooth + w5*L_grad + w6*L_material
```

### 5.2 Zero-Shot Adaptation Strategy

For zero-shot operation (no paired training data):

1. **Physics Self-Supervision**: Use only physics-based losses (no L_reconstruction)
2. **Cycle Consistency**: Colorize thermal -> Deradiometricize RGB -> Match thermal
3. **Adversarial Color Prior**: GAN with physics constraints for plausible colors
4. **Material-Aware Colormap**: Use TeX-NeRF's T->S, e->H, X->V mapping

```python
class ZeroShotThermaVisionX(nn.Module):
    """
    Zero-shot physics-guided thermal image colorization.
    No paired thermal-RGB training data required.
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder: Extract features from thermal input
        self.encoder = UNetEncoder(in_channels=1, base_channels=64)
        
        # Physics feature extractor
        self.physics_extractor = PhysicsFeatureExtractor()
        
        # Pseudo-TeX color mapper (inspired by TeX-NeRF)
        self.tex_mapper = PseudoTeXMapper()
        
        # Color refinement network
        self.color_refiner = ColorRefinementUNet()
        
        # Physics projection layer
        self.physics_layer = DifferentiablePhysicsLayer(
            physics_loss_fn=PhysicsInformedThermalLoss()
        )
    
    def forward(self, thermal_image, metadata=None):
        # Extract multi-scale features
        features = self.encoder(thermal_image)
        
        # Extract physics features
        physics_features = self.physics_extractor(
            thermal_image, metadata
        )
        
        # Map to pseudo-TeX color space (HSV with physical meaning)
        hsv_physical = self.tex_mapper(features, physics_features)
        
        # Apply physics constraints
        hsv_constrained = self.physics_layer(
            hsv_physical, physics_features['radiance']
        )
        
        # Refine and convert to RGB
        rgb_output = self.color_refiner(hsv_constrained)
        
        return rgb_output, physics_features
```

### 5.3 Implementation Roadmap

**Phase 1: Physics Foundation**
- Implement Planck's law, Stefan-Boltzmann, Wien's law in PyTorch (differentiable)
- Build spectral radiance conversion pipeline for Landsat/MODIS
- Implement LST retrieval algorithms

**Phase 2: Core Architecture**
- Build UNet-based colorization network
- Implement physics-informed loss functions
- Integrate Pseudo-TeX color mapping

**Phase 3: Zero-Shot Adaptation**
- Remove paired training data dependency
- Implement cycle consistency losses
- Add physics self-supervision

**Phase 4: Evaluation**
- Test on Landsat 8/9 thermal imagery
- Compare against baseline colorization methods
- Validate temperature accuracy

### 5.4 Key Advantages of Physics-Guided Approach

1. **Reduced Training Data**: Physics constraints provide strong regularization
2. **Better Generalization**: Solutions must satisfy physical laws
3. **Interpretable Outputs**: Each output channel has physical meaning
4. **Temperature Accuracy**: Planck's law constraint ensures correct thermal information
5. **Zero-Shot Capability**: Physics self-supervision enables training without paired data
6. **Satellite Compatibility**: Direct integration with Landsat/MODIS processing pipelines

---

## 6. References

### PINNs and Computer Vision
1. Raissi et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.
2. Karniadakis et al. (2021). "Physics-informed machine learning." *Nature Reviews Physics*, 3, 422-440.
3. Kuang et al. (2018/2020). "Thermal Infrared Colorization via Conditional Generative Adversarial Network." *Infrared Physics & Technology*.
4. Zhong & Xu (2024). "TeX-NeRF: Neural Radiance Fields from Pseudo-TeX Vision." arXiv:2410.04873.
5. "Physics-Informed Computer Vision: A Review and Perspectives." *ACM Computing Surveys*, 2024.
6. "Three-Dimensional Thermal Tomography with Physics-Informed Neural Networks." *PMC11679295*, 2024.
7. "Physics-Informed Neural Network for Denoising Images Using Nonlinear PDE." *Electronics*, 2025.
8. "Physics-Informed Neural Networks for Advanced Thermal Management." *Batteries*, 2025.

### Thermal Physics
9. Planck, M. (1901). "On the Law of Distribution of Energy in the Normal Spectrum." *Annalen der Physik*, 4, 553.
10. Stefan, J. (1879). "Uber die Beziehung zwischen der Warmestrahlung und der Temperatur." *Sitzungsberichte der mathematisch-naturwissenschaftlichen Classe der kaiserlichen Akademie der Wissenschaften*, 79, 391-428.
11. Wien, W. (1893). "Eine neue Beziehung der Strahlung schwarzer Korper zum zweiten Hauptsatz der Warmetheorie." *Sitzungsberichte der Akademie der Wissenschaften*, 55-62.

### Satellite Remote Sensing
12. Wan, Z. & Dozier, J. (1996). "A generalized split-window algorithm for retrieving land-surface temperature from space." *IEEE Trans. Geosci. Remote Sensing*, 34(4), 892-905.
13. MODIS Land-Surface Temperature Algorithm Theoretical Basis Document (LST ATBD), Version 3.3.
14. Jimenez-Munoz et al. (2014). "Land Surface Temperature Retrieval Methods from Landsat-8 Thermal Infrared Sensor Data." *IEEE GRSM*, 10.1109/MGRS.2014.2322292.
15. Rozenstein et al. (2014). "Derivation of Land Surface Temperature for Landsat-8 TIRS Using a Split Window Algorithm." *Sensors*, 14, 5768-5780.
16. "Improvements in land surface temperature and emissivity retrieval from Landsat-9 thermal infrared data." *Remote Sensing of Environment*, 2024.
17. USGS Landsat Level-1 Data Product Guide. https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

### Deep Learning for Thermal Image Processing
18. "Spectral Reconstruction from Thermal Infrared Multispectral Image Using CNN and Transformer Joint Network." *Remote Sensing*, 2024.
19. "Compensation for Vanadium Oxide Temperature with Stereo Vision on Long-Wave Infrared Light Measurement." *PMC9655447*.
20. "Interference Factors and Compensation Methods when Using Infrared Thermography for Temperature Measurement: A Review." arXiv:2502.17525.
21. "Temperature calibration of surface emissivities with an improved thermal image enhancement network." arXiv:2506.16803.
22. Berg et al. (2018). "Generating Visible Spectrum Images From Thermal Infrared." *CVPR Workshop*.

### Physics Constraints in Deep Learning
23. "Improved Physics-informed neural networks loss function regularization with a variance-based term." arXiv:2412.13993.
24. "Solving spatial-temporal PDEs with arbitrary boundary conditions using physics-constrained convolutional recurrent neural networks." *Neurocomputing*, 2025.
25. "Implementing physics-informed neural networks with deep learning for differential equations." *Frontiers in AI*, 2026.
26. "Physics Informed Neural Networks to Solve the 2-D Heat Equation." HAL, 2024.

---

*Document prepared for ISRO's Bharatiya Antariksh Hackathon 2026 - Team THERMAVISION-X*
*This research compiles findings from 26+ peer-reviewed papers and technical documents*
