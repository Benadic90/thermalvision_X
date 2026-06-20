# THERMAVISION-X
### Physics-Guided Zero-Shot Infrared Colorization for Space & Satellite Intelligence

**Hackathon**: Bharatiya Antariksh Hackathon 2026 | ISRO  
**Challenge**: #10 — Infrared image colorization and enhancement  
**Researched & Compiled by**: Benad  
**Tagline**: *Making Invisible Heat Visible — Accurately, Instantly, Everywhere*

---

## What Makes This 10/10 and Out-of-the-Box

| Dimension | What Others Do | What We Do (Different) |
|---|---|---|
| **Training Data** | Need paired IR-RGB datasets (rare!) | **Train on visible images ONLY** — zero-shot to IR |
| **Physics** | Pure data-driven (colors are "hallucinated") | **Planck's law + Stefan-Boltzmann** as AI constraints |
| **Architecture** | CNN/GAN (quadratic complexity, slow) | **Mamba State Space Model** (linear complexity, fast) |
| **Uncertainty** | No confidence measure | **Per-pixel uncertainty maps** for mission decisions |
| **Deployment** | Heavy cloud-only models | **15-30 FPS on Jetson Nano edge devices** |
| **Target** | Generic colorization | **Built for ISRO missions** (INSAT, Oceansat, TRISHNA) |

---

## The Big Idea in 3 Lines

1. **Train** our AI only on visible satellite images (which are abundant) using frequency-domain feature decoupling
2. **At inference**, feed it ANY infrared satellite image — it colorizes without ever seeing IR during training (zero-shot!)
3. **Physics constraints** ensure the colors are physically meaningful — not artistic guesswork — while **uncertainty quantification** tells operators exactly how much to trust each pixel

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-team/thermavision-x.git
cd thermavision-x

# Setup environment
pip install -r requirements.txt

# Download pretrained model (zero-shot, trained on visible images only)
wget https://huggingface.co/thermavision-x/lite-mamba-unet/releases/download/v1.0/thermavisionx_zero_shot.pth

# Colorize a single IR image
python inference/colorize.py --input ir_image.tif --output colorized.png --uncertainty

# Run demo UI
streamlit run demo/app.py
```

---

## Architecture Overview

```
Raw IR Satellite Image
       |
       v
[Preprocessing] DN -> Radiance -> Brightness Temperature -> LST
       |
       v
[Frequency Decoupling] 2D FFT -> High-Pass Filter -> Structure + Color separated
       |
       v
[Physics Feature Extractor] Planck(T) -> HSV mapping -> Physics-aware features
       |
       v
[Mamba UNet Generator] (~5M params, linear complexity)
       |
       v
[Uncertainty Quantifier] MC Dropout -> Confidence heatmap
       |
       v
[Postprocessing] HSV->RGB, Quality assessment, Georeference preservation
       |
       v
Colorized Output + Uncertainty Map + Quality Score
```

---

## 30-Hour Hackathon Plan

| Hours | Milestone | Deliverable |
|---|---|---|
| 0-3 | Environment + Data | Working setup, dataset loaders |
| 4-8 | Base Architecture | Mamba UNet generator running |
| 9-12 | Frequency Decoupling | Zero-shot pipeline working |
| 13-16 | Physics Loss | Planck + Stefan constraints integrated |
| 17-20 | Uncertainty + Training | Full training pipeline on visible images |
| 21-24 | Edge Optimization | ONNX export + TensorRT, 15+ FPS demo |
| 25-28 | Demo + UI | Streamlit app with visualization |
| 29-30 | Polish + Submit | Final testing, documentation |

---

## Key Technologies

- **Mamba/State Space Models** — Linear complexity image generation
- **Frequency Domain Decoupling** — Structure/color separation via FFT
- **Physics-Informed Neural Networks** — Planck's law as loss constraint
- **Zero-Shot Transfer** — No IR training data needed
- **Monte Carlo Uncertainty** — Confidence estimation per pixel
- **TensorRT Edge Deployment** — Real-time on Jetson Nano

---

## Datasets Used

| Dataset | Type | Use |
|---|---|---|
| KAIST Multispectral | Public IR-RGB | Validation/Testing |
| FLIR Thermal | Public IR-RGB | Validation/Testing |
| LLVIP | Public IR-RGB (night) | Validation/Testing |
| MOSDAC INSAT-3D | ISRO TIR | Real satellite data |
| Landsat 8/9 TIRS | Public TIR | Cross-validation |

---

## Target Performance

| Metric | Target |
|---|---|
| PSNR | > 28 dB |
| SSIM | > 0.85 |
| FID | < 35 |
| NIQE | < 4.5 |
| Inference Speed | 15-30 FPS (Jetson Nano) |
| Model Size | < 20 MB (INT8) |
| Zero-Shot Transfer | Works on NIR, SWIR, MWIR, LWIR |

---

## Project Documents

| Document | Description | Path |
|---|---|---|
| **Architecture** | Complete technical design with code | `THERMAVISION_X_ARCHITECTURE.md` |
| **Pitch Strategy** | Full pitch guide, FAQ, deck outline | `THERMAVISION_X_PITCH.md` |
| **IR Colorization Research** | SOTA methods, papers, code links | `research/research_ir_colorization.md` |
| **Physics-AI Research** | PINNs, thermal laws, PyTorch code | `research/research_physics_ai.md` |
| **Edge AI Research** | Mamba benchmarks, deployment configs | `research/research_edge_ai.md` |
| **ISRO Space Research** | Missions, use cases, data access | `research/research_isro_space.md` |

---

## Team Structure

| Role | Responsibility |
|---|---|
| **ML Engineer** | Mamba architecture, training pipeline, frequency decoupling |
| **Physics Expert** | Physics loss design, satellite data preprocessing, LST retrieval |
| **Edge Engineer** | ONNX/TensorRT, model compression, edge benchmarks |
| **Frontend/DevOps** | Streamlit demo, Docker, pitch deck, documentation |

---

## Why This Wins

1. **Novelty**: First to combine 6 innovations (zero-shot + physics + Mamba + frequency decoupling + uncertainty + edge deployment)
2. **Real Impact**: Directly addresses India's disaster management, agriculture, and urban planning needs
3. **ISRO Alignment**: Purpose-built for INSAT-3D, Oceansat-3, and upcoming TRISHNA mission
4. **Feasibility**: Tiered 30-hour plan with clear fallbacks at each stage
5. **Technical Depth**: State-of-the-art architecture with published research backing every decision

---

## References

Key papers this project builds upon:
- Wei et al. (2024) — Cross-modality zero-shot IR colorization
- Liu et al. (2025) — CCLGAN with Mamba + cosine contrastive loss
- Wang et al. (ICLR 2023 Oral) — DDNM zero-shot diffusion
- TeX-NeRF (2024) — Physics-guided thermal-to-HSV mapping
- STFNet (2024) — Self-supervised transformer for IR-visible fusion

---

**Built with passion for ISRO. For India's space future.**

*Researched & compiled by Benad | Bharatiya Antariksh Hackathon 2026*
