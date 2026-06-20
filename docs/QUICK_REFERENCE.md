# THERMAVISION-X — Quick Reference Card
### 1-Page Summary for Team Members | Compiled by Benad

---

## THE PROBLEM (30-Second Pitch)
> ISRO captures terabytes of thermal satellite data daily — all in grayscale. Farmers can't read radiance maps. Disaster responders can't interpret thermal flood images. The upcoming TRISHNA mission (2026) will generate 4-band thermal data at 57m resolution with NO systems to make it usable.

---

## OUR SOLUTION (3 Lines)
1. **Train ONLY on visible images** (abundant) — learn color reconstruction via frequency-domain masked modeling
2. **Colorize ANY IR image at inference** — zero-shot, no IR training data needed
3. **Physics-guided** — Planck's law + Stefan-Boltzmann constrain outputs to be physically meaningful

---

## 6 LAYERS OF OUT-OF-THE-BOX

| # | Innovation | Why It's Different |
|---|---|---|
| 1 | **Zero-Shot Learning** | Train on visible → colorize any IR band. No paired data needed! |
| 2 | **Physics-Informed AI** | Planck's law B_λ(T) = 2hc²/λ⁵(exp(hc/λkT)-1) as neural network constraint |
| 3 | **Frequency Decoupling** | 2D FFT separates structure (high-freq) from color (low-freq) |
| 4 | **Mamba Architecture** | Linear O(n) complexity vs Transformer's O(n²). Global receptive field, fast! |
| 5 | **Uncertainty Quantification** | MC Dropout tells operators "95% confident here, 30% there" |
| 6 | **ISRO-First Design** | Built for INSAT-3D, Oceansat-3, TRISHNA missions specifically |

---

## KEY TECHNICAL DECISIONS

| Decision | Choice | Why |
|---|---|---|
| Backbone | Lightweight Mamba UNet (~5M params) | Linear complexity, global receptive field |
| Color Space | HSV with physics mapping | T→Saturation, ε→Hue, Texture→Value (TeX-NeRF) |
| Loss Function | 5-term composite | L_recon + L_focal_freq + L_planck + L_stefan + L_cosine |
| Training Data | Visible images ONLY | Zero-shot to IR at inference |
| Edge Target | Jetson Nano 4GB | 15-30 FPS with TensorRT INT8 |
| Uncertainty | Monte Carlo Dropout | Per-pixel confidence maps |

---

## 30-HOUR TIMELINE

```
H0-3    Setup + Data         PyTorch env, dataset loaders
H4-8    Mamba UNet           Base generator architecture
H9-12   Frequency Decouple   2D FFT pipeline, zero-shot working
H13-16  Physics Loss         Planck + Stefan constraints
H17-20  Training             Train on visible images
H21-24  Edge Deploy          ONNX + TensorRT, benchmark FPS
H25-28  Demo + UI            Streamlit app with viz
H29-30  Polish + Submit      Final testing, docs
```

**Tier 1 (12h)**: Working zero-shot colorization  
**Tier 2 (24h)**: + Physics + Uncertainty  
**Tier 3 (30h)**: + Edge + Polished Demo  

---

## USE CASES FOR ISRO

1. **Disaster Response**: Colorize INSAT-3D TIR flood maps for NDRF rescue teams
2. **Agriculture**: Make FASAL thermal stress maps farmer-friendly
3. **Urban Heat**: Visualize city heat islands for municipal planning
4. **Fisheries**: Colorize Oceansat-3 SST for PFZ fishing advisories
5. **Nighttime**: Colorize MIR (3.8um) nighttime imagery for aviation safety

---

## COMPETITIVE EDGE

```
                    Paired Data  Physics  Speed      Uncertainty  Edge
CycleGAN            No           No       Slow       No           No
CUT/FastCUT         Unpaired     No       Medium     No           No
DCLGAN              Unpaired     No       Slow       No           No
CCLGAN              Unpaired     No       Medium     No           No
DDNM (Diffusion)    Zero-shot    No       VERY slow  No           No
Palette             Paired       No       Slow       No           No
------------------------------------------------------------------------
THERMAVISION-X      VISIBLE ONLY YES      15-30 FPS  YES          YES
                    (ZERO-SHOT)
```

---

## CRISIS STATISTICS (USE IN PITCH)

- 1,507 Indians died in disasters (2024)
- 49.8 million hectares flood-prone
- 733 heat wave deaths
- 8+ million exposed in Gujarat floods
- TRISHNA mission launching 2026 — 4-band TIR at 57m, NO existing colorization systems

---

## QUOTE FOR JUDGES

> "Every existing solution trains on IR-RGB pairs. We said: what if we NEVER show the AI any infrared during training? Instead, we teach it physics — and it generalizes to any IR band it's never seen. That's zero-shot. That's THERMAVISION-X."

---

## ESSENTIAL LINKS

| Resource | URL |
|---|---|
| Hackathon | https://hack2skill.com/event/bah2026 |
| CCLGAN Code | https://github.com/LTTdouble/CCLGAN |
| DDNM Code | https://github.com/wyhuai/DDNM |
| MambaIR | https://github.com/csguoh/MambaIR |
| ISRO MOSDAC | https://mosdac.gov.in |
| KAIST Dataset | https://soonminhwang.github.io/rgbt-ped-detection |
| FLIR Dataset | https://www.flir.com/oem/adas/thermal-dataset |

---

## TEAM ROLES (4 Members)

| Role | Pre-Hackathon Prep |
|---|---|
| ML Engineer | Review Mamba code, setup PyTorch environment |
| Physics Expert | Study Planck's law implementation, satellite data formats |
| Edge Engineer | Setup Jetson/Raspberry Pi, install TensorRT |
| Frontend/DevOps | Prepare Streamlit template, design pitch deck layout |

---

**Print this page. Keep it visible during the hackathon.**

*THERMAVISION-X — Physics-Guided Zero-Shot Infrared Colorization*  
*Bharatiya Antariksh Hackathon 2026 | Researched by Benad*
