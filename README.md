<div align="center">
  <h1>🛰️ THERMAVISION-X</h1>
  <p><strong>Physics-Guided Zero-Shot Infrared Colorization for Space & Satellite Intelligence</strong></p>
  
  [![Hackathon](https://img.shields.io/badge/Hackathon-BAH_2026-ff69b4?style=for-the-badge&logo=rocket)](https://hack2skill.com/event/bah2026)
  [![ISRO Challenge](https://img.shields.io/badge/Challenge-%2310_IR_Colorization-blue?style=for-the-badge&logo=satellite)](https://hack2skill.com/event/bah2026)
  [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
</div>

<br/>

> **Tagline**: *Making Invisible Heat Visible — Accurately, Instantly, Everywhere.*  
> **Team**: Benadic (ML Lead), Madhav (Data & Physics), Adity (Full-Stack), Nisha (Domain Expert)

---

## 🌟 The Vision

ISRO satellites (INSAT-3D, Oceansat-3, TRISHNA) capture massive amounts of thermal infrared data, but it is entirely in grayscale. Human brains struggle to interpret grayscale accurately, slowing down disaster response and medical analysis.

**THERMAVISION-X** solves this using an ultra-lightweight (~5M params) AI that mathematically separates structure from color, allowing it to colorize **any** unseen thermal image (Zero-Shot) while ensuring the colors obey the laws of physics (Planck's Law).

---

## 🔥 5 Core Innovations

1. 🔬 **Zero-Shot Learning via FFT**  
   We decouple images into high-frequency (structure) and low-frequency (color) domains using Fast Fourier Transforms. The network trains *only* on visible RGB images, yet colorizes thermal IR without ever seeing it during training.
   
2. ⚛️ **Physics-Guided Loss (Planck's Law)**  
   The AI doesn't guess colors. We encode Planck's Law and the Stefan-Boltzmann law directly into the loss function so that hotter pixels are strictly mapped to more vivid, warmer colors.

3. 🧠 **Ultra-Lightweight UNet Backbone**  
   With only ~5M parameters, our model doesn't need cloud GPUs. It runs seamlessly on edge devices (like Jetson Nano or local laptops equipped with RTX 2050).

4. 📊 **Uncertainty Confidence Maps**  
   Using Monte Carlo Dropout, our model outputs a pixel-level confidence map alongside the colorized image—telling disaster responders exactly which regions of the image the AI is confident about.

5. 🛸 **Built for ISRO**  
   Natively supports ISRO's HDF5 data formats (INSAT, TRISHNA) and standard GeoTIFFs (Landsat), making it a true Earth Observation tool.

---

## 💻 Project Structure

```text
thermavision_x/
├── models/                 # Core AI architecture
│   ├── unet.py             # 5M parameter backbone
│   ├── frequency.py        # FFT decoupling magic
│   └── physics_layer.py    # Thermodynamics constraints
├── data/                   # Data pipelines (HDF5, GeoTIFF, JPG)
├── training/               # Physics-informed loss & loop
├── inference/              # Zero-shot CLI script
├── demo/                   # Interactive Streamlit Web App
└── docs/                   # Complete deep-research documentation
```

---

## 🚀 Quick Start (Hackathon Day)

Get the project running on your local machine in under 5 minutes.

### 1. Install Dependencies
```bash
git clone https://github.com/Benadic90/thermalvision_X.git
cd thermalvision_X
pip install -r requirements.txt
```

### 2. Launch the Interactive Web App
We've built a gorgeous Streamlit dashboard for testing the model.
```bash
streamlit run demo/app.py
```

### 3. Run Zero-Shot Inference via CLI
Got a raw grayscale IR image? Pass it through the model directly from the command line:
```bash
python inference/colorize.py --input assets/samples/sample_ir.jpg --output output_colorized.png
```

---

## 📚 Deep Research Documents

We didn't just write code; we conducted deep, comprehensive research into State-of-the-Art (SOTA) thermal imaging. Check out our detailed guides in the `docs/` folder:

- 📖 [TEAM_GUIDE.md](docs/TEAM_GUIDE.md) - The master summary of our project architecture.
- 📋 [HACKATHON_ACTION_PLAN.md](docs/HACKATHON_ACTION_PLAN.md) - Our 30-hour execution strategy.
- 🏗️ [THERMAVISION_X_ARCHITECTURE.md](docs/THERMAVISION_X_ARCHITECTURE.md) - The complete technical breakdown.
- 🎤 [THERMAVISION_X_PITCH.md](docs/THERMAVISION_X_PITCH.md) - Our competitive strategy and use cases.

---

<div align="center">
  <i>Researched and built with passion for the Bharatiya Antariksh Hackathon 2026 🇮🇳</i>
</div>
