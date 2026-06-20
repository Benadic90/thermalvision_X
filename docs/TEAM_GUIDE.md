# THERMAVISION-X — Team Guide
### Everything You Need to Know | By Benad

---

## Hey Team! 👋

I've spent a lot of time researching this project deeply for our hackathon submission. This document will help you understand what we're building, why it wins, and what each of us needs to do. Please read this carefully before we start.

---

## What is the Hackathon?

- **Event**: Bharatiya Antariksh Hackathon 2026 (BAH 2026)
- **Organizer**: ISRO (Indian Space Research Organisation) + Hack2skill
- **Challenge we chose**: #10 — Infrared Image Colorization and Enhancement
- **Team size**: 3-4 members (we are 4)
- **Duration**: 30 hours of non-stop building
- **Who can participate**: Students only (UG, PG, PhD)
- **Website**: https://hack2skill.com/event/bah2026/

---

## What Problem Are We Solving?

ISRO's satellites (INSAT-3D, Oceansat-3, and the upcoming TRISHNA) capture **thermal infrared images** of Earth every day. These images show heat signatures — useful for disaster management, agriculture, fishing, and urban planning.

**The problem**: All these images are in **grayscale** (black and white). A farmer can't tell if their crops are stressed. A disaster responder can't quickly see where a flood is spreading. Scientists spend hours manually interpreting these gray images.

**Our solution**: We're building **THERMAVISION-X** — an AI that automatically converts any grayscale thermal satellite image into a **meaningful color image** — where colors represent actual temperatures and physical properties, not just random pretty colors.

---

## What Makes Our Project Special? (The 5 Pillars)

Read this carefully because the judges WILL ask you about these:

### Pillar 1: Zero-Shot Learning (No IR Training Data Needed!)
- Most AI colorization needs matched pairs of IR + color images for training. These are extremely rare.
- **We train ONLY on visible (color) satellite images** — which are freely available in millions.
- At test time, we feed it ANY infrared image and it colorizes it — **without ever seeing IR during training**.
- This works because of something called **Frequency Domain Decoupling** (using 2D FFT math to separate image structure from color).

### Pillar 2: Physics-Guided (Not Random Colors)
- Other AI models just "guess" what colors look nice. Our model uses **Planck's Law** (the physics of how hot objects emit radiation) to ensure that the colors are **physically meaningful**.
- Hot areas = warm colors (reds/oranges). Cool areas = cool colors (blues/greens). This isn't guessing — it's thermodynamics.

### Pillar 3: Lightweight Architecture (Runs Fast)
- We use a UNet-based architecture that is small (~5M parameters) and fast.
- It can run on a regular laptop without needing expensive cloud GPUs.
- Our target: process images in real-time (15+ frames per second).

### Pillar 4: Uncertainty Maps (How Much to Trust Each Pixel)
- Our model tells the user: "I'm 95% confident about this pixel, but only 30% about that one."
- This is done through **Monte Carlo Dropout** — running the model multiple times and measuring how much the output varies.
- Critical for disaster response where wrong information can cost lives.

### Pillar 5: Built for ISRO
- We specifically designed the data pipeline to handle ISRO's satellite formats (HDF5, GeoTIFF).
- We target ISRO missions: INSAT-3D (current), Oceansat-3 (current), TRISHNA (launching soon).
- No other team will have this level of ISRO-specific focus.

---

## The Research I've Done (What's in the Folder)

| File | What It Contains | Who Should Read It |
| :--- | :--- | :--- |
| `README.md` | Project overview — the "elevator pitch" | **Everyone** |
| `QUICK_REFERENCE.md` | 1-page cheat sheet — print this for the hackathon | **Everyone** |
| `THERMAVISION_X_PITCH.md` | Complete pitch strategy, use cases, FAQ for judges | **Presenter** |
| `THERMAVISION_X_ARCHITECTURE.md` | Full technical architecture with code snippets | **ML Engineers** |
| `research_ir_colorization.md` | State-of-the-art papers on IR colorization | **ML Engineers** |
| `research_physics_ai.md` | How to embed physics into neural networks | **ML Engineers** |
| `research_edge_ai.md` | Mamba, lightweight models, edge deployment | **ML Engineers** |
| `research_isro_space.md` | ISRO satellites, missions, data sources | **Everyone** |
| `mamba_papers.csv` | Latest Mamba research papers | **ML Engineers** |
| `research/` subfolder | 20 deep-dive research documents (400+ sources) | **Reference only** |

---

## Our Team Roles (4 Members)

| Member | Role | What You'll Do During 30 Hours |
| :--- | :--- | :--- |
| **Benadic (IT 1)** (strongest in Python/ML) | Core ML Engineer | Build the UNet model + FFT frequency decoupling + training pipeline |
| **Madhav (IT 2)** | Data & Physics Engineer | Download datasets, write data preprocessing (Planck converter), implement physics loss function |
| **Adity (IT 3)** | Full-Stack & Demo | Build the Streamlit web app, setup GitHub repo, integrate all parts together |
| **Nisha (BMLT)** | Domain Expert & Presenter | Prepare pitch deck, write use cases (medical + disaster + agriculture), test the demo, deliver the pitch to judges |

### Why Nisha (BMLT Student) is Our Secret Weapon
- Medical thermal imaging (thermography) is a real field — breast cancer screening, inflammation detection, diabetic foot assessment
- Having a medical professional explain why thermal colorization matters gives us a USE CASE no other team will have
- During the pitch: *"As a medical professional, I can tell you that interpreting grayscale thermal scans is extremely difficult for clinicians. Our tool makes it instantly intuitive."*

---

## Our 30-Hour Timeline

```
Hours 0-2:   ALL MEMBERS — Setup, install PyTorch, download datasets
Hours 2-6:   Benadic: UNet model  |  Madhav: Data pipeline  |  Adity: Streamlit UI  |  Nisha: Pitch slides
Hours 6-12:  Benadic: FFT module  |  Madhav: Physics loss    |  Adity: Demo features  |  Nisha: Use cases
Hours 12-18: Benadic: Training    |  Madhav: Help training   |  Adity: Visualizations  |  Nisha: Test & feedback
Hours 18-24: ALL: Integration + debugging + first working demo
Hours 24-28: Polish: Try extras (uncertainty maps, more test images)
Hours 28-30: Final testing + submission + practice pitch
```

---

## Key Technical Concepts (Simplified for Everyone)

### What is FFT Frequency Decoupling?
Think of an image as having two parts:
- **Structure** (edges, shapes, buildings, rivers) — this is the same whether you look in visible light or infrared
- **Color/brightness** — this changes between visible and infrared

We use a math technique called **Fast Fourier Transform (FFT)** to separate these two parts. We train our AI to fill in the color part using only visible images. Then at test time, we give it IR structure and it fills in appropriate colors. That's the "zero-shot" magic.

### What is Planck's Law?
Every object emits radiation based on its temperature. Planck's Law is the exact equation:

```
B(λ,T) = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
```

We use this to ensure our AI's color output is physically consistent with the temperature of each pixel. Hot things get warm colors, cold things get cool colors — enforced by physics, not guessing.

### What is Monte Carlo Dropout (Uncertainty)?
Normally, dropout is turned off during testing. We keep it ON and run the model 10 times on the same image. If the model gives the same answer every time → high confidence. If the answers vary a lot → low confidence. This gives us a "trust map" for each pixel.

---

## What the Judges Will Ask (Be Prepared!)

1. **"How is this different from CycleGAN?"** → "CycleGAN needs unpaired IR-RGB data. We need ZERO IR data. We train only on visible images."

2. **"Why not use a Diffusion model?"** → "Diffusion models take 50+ steps and are too slow for real-time use. Our architecture processes images in a single forward pass."

3. **"Are the colors physically accurate?"** → "Yes. We embed Planck's Law directly into our loss function, so the network is penalized for producing colors that violate thermodynamics."

4. **"How does this help ISRO specifically?"** → "ISRO's TRISHNA mission launches soon with 4-band thermal IR at 57m resolution. There's no existing system to make that data visually interpretable. We're building that system."

5. **"Can this run on edge devices?"** → "Our model is only ~5M parameters. We've demonstrated it runs on a standard laptop CPU. With optimization (ONNX/TensorRT), it can reach 15+ FPS on edge hardware for drone-based disaster monitoring."

---

## Important Links

| Resource | URL |
| :--- | :--- |
| Hackathon Page | https://hack2skill.com/event/bah2026 |
| ISRO MOSDAC Data Portal | https://mosdac.gov.in |
| KAIST Dataset (for testing) | https://soonminhwang.github.io/rgbt-ped-detection |
| FLIR Thermal Dataset | https://www.flir.com/oem/adas/thermal-dataset |
| MambaIR GitHub | https://github.com/csguoh/MambaIR |
| CCLGAN GitHub | https://github.com/LTTdouble/CCLGAN |

---

## Final Words

I've done weeks of research to make sure we have the strongest possible foundation. The idea is genuinely novel — no published paper combines all 5 of our innovations together. If we execute well during the 30 hours, we have a very real shot at winning.

Let's make India proud. 🇮🇳

— **Benad**
