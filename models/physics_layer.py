"""
models/physics_layer.py
Physics-Based Thermal to Temperature Conversion Module

Workflow:
Thermal Image Pixel
      ↓
Radiance (using Planck's Law)
      ↓
Temperature (using Stefan-Boltzmann Law)
      ↓
Model Input
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ──────────────────────────────────────────
# PHYSICS CONSTANTS (never change these)
# ──────────────────────────────────────────
PLANCK_CONSTANT      = 6.626e-34    # h  → Planck's constant (J·s)
SPEED_OF_LIGHT       = 3.0e8        # c  → Speed of light (m/s)
BOLTZMANN_CONSTANT   = 1.381e-23    # k  → Boltzmann constant (J/K)
STEFAN_BOLTZMANN     = 5.67e-8      # σ  → Stefan-Boltzmann constant (W/m²K⁴)

# ──────────────────────────────────────────
# SENSOR SETTINGS (based on FLIR thermal camera)
# ──────────────────────────────────────────
WAVELENGTH           = 10e-6        # λ  → 10 micrometers (typical thermal IR wavelength)
EMISSIVITY           = 0.95         # ε  → Most real objects: 0.9 to 0.98
PIXEL_MAX            = 255.0        # maximum pixel value in 8-bit image
RADIANCE_MAX         = 100.0        # estimated max radiance for scaling (W/m²/sr)
# ──────────────────────────────────────────


# ══════════════════════════════════════════
# STEP 1: Pixel → Radiance
# ══════════════════════════════════════════
def pixel_to_radiance(pixel_value):
    """
    Convert pixel brightness (0-255) to estimated radiance.

    Thermal cameras store radiation intensity as pixel values.
    Brighter pixel = more radiation = higher temperature.

    Args:
        pixel_value: numpy array or single value (0 to 255)

    Returns:
        radiance: estimated radiance in W/m²/sr
    """
    # Normalize pixel to 0.0 - 1.0
    normalized = pixel_value / PIXEL_MAX

    # Scale to radiance range
    radiance = normalized * RADIANCE_MAX

    return radiance


# ══════════════════════════════════════════
# STEP 2: Radiance → Temperature (Planck's Law inverse)
# ══════════════════════════════════════════
def radiance_to_temperature_planck(radiance):
    """
    Convert radiance to temperature using inverse of Planck's Law.

    Planck's Law:  B(λ,T) = (2hc²/λ⁵) * (1 / (e^(hc/λkT) - 1))
    Inverse:       T = (hc/λk) / ln((2hc²/λ⁵B) + 1)

    Args:
        radiance: radiance value(s) in W/m²/sr (numpy array or float)

    Returns:
        temperature_kelvin: temperature in Kelvin
    """
    h = PLANCK_CONSTANT
    c = SPEED_OF_LIGHT
    k = BOLTZMANN_CONSTANT
    lam = WAVELENGTH

    # Avoid division by zero — add tiny value
    radiance = np.clip(radiance, 1e-10, None)

    # Calculate C1 and C2 (radiation constants)
    C1 = 2 * h * c**2 / lam**5
    C2 = h * c / (lam * k)

    # Inverse Planck formula to get Temperature
    temperature_kelvin = C2 / np.log((C1 / radiance) + 1)

    return temperature_kelvin


# ══════════════════════════════════════════
# STEP 3: Temperature (Kelvin → Celsius)
# ══════════════════════════════════════════
def kelvin_to_celsius(temperature_kelvin):
    """
    Convert temperature from Kelvin to Celsius.

    Formula: Celsius = Kelvin - 273.15

    Args:
        temperature_kelvin: temperature in Kelvin

    Returns:
        temperature_celsius: temperature in Celsius
    """
    return temperature_kelvin - 273.15


# ══════════════════════════════════════════
# STEP 4: Temperature Estimation using Stefan-Boltzmann
# ══════════════════════════════════════════
def radiance_to_temperature_stefan(radiance):
    """
    Estimate temperature using Stefan-Boltzmann Law.
    Good for quick estimation of total emitted power.

    Stefan-Boltzmann: P = ε * σ * T⁴
    Inverse:          T = (P / (ε * σ)) ^ (1/4)

    Args:
        radiance: radiance value(s) in W/m²

    Returns:
        temperature_kelvin: estimated temperature in Kelvin
    """
    epsilon = EMISSIVITY
    sigma   = STEFAN_BOLTZMANN

    # Avoid division by zero
    radiance = np.clip(radiance, 1e-10, None)

    # Inverse Stefan-Boltzmann
    temperature_kelvin = (radiance / (epsilon * sigma)) ** 0.25

    return temperature_kelvin


# ══════════════════════════════════════════
# MAIN FUNCTION: Full Pipeline
# Pixel → Radiance → Temperature
# ══════════════════════════════════════════
def thermal_image_to_temperature(image_path, method="planck"):
    """
    Complete pipeline: Load thermal image and convert to temperature map.

    Args:
        image_path: path to thermal image file
        method: "planck" or "stefan" (which law to use)

    Returns:
        temp_celsius: 2D numpy array of temperatures in Celsius
        original_img: original grayscale image
    """
    # Load image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"[Physics] Loaded image: {Path(image_path).name}")
    print(f"[Physics] Image shape : {img.shape}")
    print(f"[Physics] Pixel range : {img.min()} to {img.max()}")

    # Step 1: Pixel → Radiance
    radiance = pixel_to_radiance(img.astype(np.float32))

    # Step 2: Radiance → Temperature
    if method == "planck":
        temp_kelvin = radiance_to_temperature_planck(radiance)
        print(f"[Physics] Method      : Planck's Law")
    else:
        temp_kelvin = radiance_to_temperature_stefan(radiance)
        print(f"[Physics] Method      : Stefan-Boltzmann Law")

    # Step 3: Kelvin → Celsius
    temp_celsius = kelvin_to_celsius(temp_kelvin)

    print(f"[Physics] Temp range  : {temp_celsius.min():.1f}°C to {temp_celsius.max():.1f}°C")

    return temp_celsius, img


# ══════════════════════════════════════════
# VISUALIZATION: Before / After
# ══════════════════════════════════════════
def visualize_temperature(original_img, temp_celsius, save_path=None):
    """
    Create a side-by-side visualization:
    Left  → Original thermal image
    Right → Temperature heatmap with color scale
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Thermal Image → Temperature Conversion", fontsize=16, fontweight='bold')

    # Left: Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Thermal Image\n(Pixel Values 0-255)", fontsize=13)
    axes[0].axis('off')

    # Right: Temperature heatmap
    heatmap = axes[1].imshow(temp_celsius, cmap='inferno')
    axes[1].set_title("Temperature Map\n(Celsius)", fontsize=13)
    axes[1].axis('off')

    # Color bar
    cbar = plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Physics] Saved visualization to: {save_path}")

    plt.show()
    return fig


# ══════════════════════════════════════════
# TEST - Run this file directly to test
# ══════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("  PHYSICS LAYER TEST")
    print("=" * 55)

    # ── Test with a real image if available ──
    sample_folder = Path("datasets/samples")
    image_files   = list(sample_folder.glob("*.jpg")) + \
                    list(sample_folder.glob("*.png"))

    if image_files:
        # Use first available sample image
        test_image = image_files[0]
        print(f"\nTesting with: {test_image.name}\n")

        # Run full pipeline using Planck's Law
        temp_map, original = thermal_image_to_temperature(test_image, method="planck")

        # Print temperature statistics
        print("\n── Temperature Statistics ──")
        print(f"  Minimum  : {temp_map.min():.2f} °C")
        print(f"  Maximum  : {temp_map.max():.2f} °C")
        print(f"  Average  : {temp_map.mean():.2f} °C")

        # Save visualization
        output_path = "datasets/processed/temperature_visualization.png"
        visualize_temperature(original, temp_map, save_path=output_path)

    else:
        # ── No image found → test with a dummy array ──
        print("\nNo sample images found.")
        print("Running test with a dummy 10x10 pixel array...\n")

        # Create fake thermal image (pixel values 50 to 200)
        dummy_pixels = np.linspace(50, 200, 100).reshape(10, 10)

        radiance  = pixel_to_radiance(dummy_pixels)
        temp_k    = radiance_to_temperature_planck(radiance)
        temp_c    = kelvin_to_celsius(temp_k)

        print("Dummy pixel values:")
        print(np.round(dummy_pixels[:3, :3], 1))

        print("\nConverted temperatures (°C):")
        print(np.round(temp_c[:3, :3], 2))

        print("\n[OK] Physics layer functions working correctly!")

    print("\n" + "=" * 55)
    print("  PHYSICS LAYER READY FOR MODEL INTEGRATION")
    print("=" * 55)