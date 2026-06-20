# Deep Research: Out-of-the-Box Infrared Image Colorization Techniques
## For THERMAVISION-X -- Physics-Guided Zero-Shot Infrared Colorization System
### ISRO Bharatiya Antariksh Hackathon 2026
**Researched by**: Benad | June 2026

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Zero-Shot Cross-Spectral IR Colorization](#2-zero-shot-cross-spectral-ir-colorization)
3. [Frequency-Domain Approaches](#3-frequency-domain-approaches)
4. [Self-Supervised & Contrastive Learning for IR Colorization](#4-self-supervised--contrastive-learning-for-ir-colorization)
5. [Quality Assessment for Colorized IR Images](#5-quality-assessment-for-colorized-ir-images)
6. [Emerging Trends & Novel Ideas for Hackathon](#6-emerging-trends--novel-ideas-for-hackathon)
7. [Recommended Architecture for THERMAVISION-X](#7-recommended-architecture-for-thermavision-x)
8. [References & Code Repositories](#8-references--code-repositories)

---

## 1. Executive Summary

This research document compiles state-of-the-art techniques for **zero-shot infrared image colorization** -- the task of converting grayscale thermal/near-infrared images to realistic RGB color images **without requiring paired IR-RGB training data**. The findings are organized around four key pillars:

- **Zero-shot cross-spectral learning**: Methods that train only on visible images but generalize to infrared at inference
- **Frequency-domain decoupling**: Separating high-frequency structure from low-frequency color via FFT/DCT
- **Contrastive self-supervision**: Using PatchNCE and cosine contrastive losses for cross-domain alignment
- **No-reference quality assessment**: Evaluating colorization quality without ground truth

**Key Insight**: The most promising approach for THERMAVISION-X combines **frequency-domain feature decoupling** (preserve IR high-freq structure, reconstruct low-freq color) with **zero-shot diffusion models** (DDNM) and **physics-informed constraints** (Planck's law for thermal radiation).

---

## 2. Zero-Shot Cross-Spectral IR Colorization

### 2.1 Chiheng Wei et al. -- "Infrared colorization with cross-modality zero-shot learning" (Neurocomputing 2024)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Chiheng Wei, Huawei Chen, Lianfa Bai, Jing Han, Xiaoyu Chen |
| **Venue** | Neurocomputing, 2024 (Available online April 2024) |
| **Paper** | https://doi.org/10.1016/j.neucom.2024.127449 |
| **Key Innovation** | First cross-modality zero-shot IR colorization using frequency-domain feature decoupling + masked image modeling |

#### Technical Approach

**Core Idea**: Visible and infrared images share similar **high-frequency features** (edges, contours, textures) but differ significantly in **low-frequency information** (brightness, color). By decoupling these in the frequency domain, we can:
1. Preserve the IR image's high-frequency structure
2. Reconstruct only the low-frequency color information using a network trained **only on visible images**

**Frequency Domain Feature Decoupling Pipeline**:

```
Step 1: Convert visible image to frequency domain (2D DFT/DCT)
Step 2: Apply high-pass filter mask to separate:
        - High-frequency component (structure/edges)
        - Low-frequency component (color/brightness)
Step 3: Train network to reconstruct low-freq from high-freq (MIM)
Step 4: At inference, apply same high-pass mask to IR image
Step 5: Use trained network to synthesize low-frequency color
Step 6: Combine preserved IR high-freq + synthesized low-freq
Step 7: Inverse DFT to obtain colorized IR
```

**Why This Works for Zero-Shot**:
- Visible and IR images show **increasing structural similarity (SSIM)** as more low-frequency information is removed via high-pass filtering
- The network learns a **reconstruction capability** (not modality-specific mapping)
- At inference, the IR image's high-frequency content is preserved, while color (low-freq) is hallucinated from the learned reconstruction prior

**Key Technical Details**:
- Uses **2D DFT** for frequency transformation
- High-pass filter mask radius is a tunable hyperparameter
- Network is trained with **frequency domain reconstruction loss** (similar to Focal Frequency Loss)
- Incorporates **Masked Image Modeling (MIM)** in frequency domain: randomly mask low-frequency bands during training, reconstruct them
- Supports **multiple IR spectral bands** (near-infrared, long-wave infrared) without retraining

**Experimental Results**:
- Tested on multiple IR datasets across different spectral bands
- Demonstrates excellent cross-modality adaptability
- No infrared data needed during training -- true zero-shot to new IR modalities

**Pros for Hackathon**:
- No IR training data required (perfect for ISRO scenarios where IR data is scarce)
- Lightweight -- no GAN training instability
- Frequency-domain operations are fast (FFT on GPU)
- Naturally preserves thermal structure from IR

**Cons for Hackathon**:
- Reconstructed colors may not be physically accurate to thermal properties
- Requires careful tuning of high-pass mask radius
- Limited diversity in generated colors

**How to Adapt for THERMAVISION-X**:
- Use their frequency decoupling as the **core preprocessing step**
- Add **physics-informed loss** (Planck's law) to ensure thermal radiation consistency
- Combine with DDNM (Section 2.2) for better color diversity

---

### 2.2 DDNM -- "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model" (ICLR 2023 Oral)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Yinhuai Wang, Jiwen Yu, Jian Zhang |
| **Venue** | ICLR 2023 (Oral Presentation) |
| **Paper** | https://arxiv.org/abs/2212.00490 |
| **Code** | https://github.com/wyhuai/DDNM |
| **Key Innovation** | Zero-shot image restoration via Range-Null Space Decomposition (RND) on diffusion models -- no training needed |

#### Technical Approach

**Core Idea**: For any linear inverse problem y = Ax (where A is the degradation operator), the solution space can be decomposed into:
- **Range-space**: Information preserved by A (can be computed exactly)
- **Null-space**: Information lost by A (needs to be generated)

DDNM uses a **pre-trained diffusion model** to generate only the null-space content while fixing the range-space to ensure data consistency.

**Range-Null Space Decomposition for Colorization**:

For colorization, the degradation operator A converts RGB to grayscale:

```
y = A*x where A = [1/3, 1/3, 1/3]  (grayscale averaging)
```

The pseudo-inverse A+ reconstructs a "raw" color image from grayscale:
```
x_r = A+ * y  (replicates grayscale to all 3 channels)
```

RND rectifies any image z to be consistent with y:
```
x_rectified = A+ * y + (I - A+ * A) * z
```

This ensures: A * x_rectified = y (perfect grayscale consistency)

**DDNM Reverse Diffusion Process**:

```
At each timestep t in reverse diffusion:
1. Denoise x_t to get clean estimate x_{0|t}
2. Apply RND: x_{0|t} = A+ * y + (I - A+ * A) * x_{0|t}
3. Continue diffusion with the rectified estimate
```

The key insight: only the null-space content (color information) is refined by the diffusion model, while range-space content (grayscale) is fixed to the input.

**DDNM+ Enhancements**:
- **Denoising support**: Handles noisy IR inputs via scaled noise covariance
- **Time-travel trick**: For hard tasks (large masks, extreme SR), jumps forward and back in diffusion time for better coherence
  - Parameters: travel_length (l), travel_repeat (r), sampling interval (s)
  - Significantly improves FID on challenging colorization cases

**Experimental Results** (Zero-shot colorization on ImageNet):
| Method | PSNR | SSIM | FID |
|--------|------|------|-----|
| DGP (GAN prior) | 23.18 | 0.870 | 64.34 |
| ILVR | N/A | 0.870 | 43.66 |
| DDRM | 26.72 | 0.860 | 65.33 |
| **DDNM** | **26.85** | **0.872** | **35.21** |
| **DDNM+** | **27.12** | **0.879** | **29.83** |

**Pros for Hackathon**:
- **Absolutely zero training** -- uses pre-trained diffusion models off-the-shelf
- Perfect data consistency (grayscale exactly matches input IR)
- Highly diverse, realistic color outputs
- Can handle arbitrary image sizes (Mask-Shift Restoration)
- Natural extension to other IR tasks (super-resolution, deblurring)

**Cons for Hackathon**:
- **Slow inference**: requires 100-1000 diffusion steps per image
- Needs pre-trained diffusion model (256x256 or 512x512)
- May produce inconsistent colors across frames (for video)
- Color choices are arbitrary (no physics guidance)

**How to Adapt for THERMAVISION-X**:
- Use DDNM as the **colorization backbone** for highest quality results
- Replace grayscale operator A with **thermal-to-RGB physics operator** (Planck's law)
- Add **text prompts** (e.g., "satellite thermal image of urban area") to guide colorization direction
- Use time-travel trick (DDNM+) for hard cases (low-contrast IR)
- Implement fast sampling (DDIM) for speedup

---

### 2.3 Palette: Image-to-Image Diffusion Models (Saharia et al., 2022)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Chitwan Saharia et al. (Google Research) |
| **Venue** | ACM ToG / arXiv 2022 |
| **Paper** | https://3dvar.com/Saharia2021Palette.pdf |
| **Unofficial Code** | https://github.com/janspiry/palette-image-to-image-diffusion-models |
| **Key Innovation** | General-purpose diffusion model for multiple image-to-image tasks including colorization |

**Technical Approach**: Palette is a **conditional diffusion model** trained on image-to-image translation tasks. For colorization:
- Input: grayscale image (concatenated as condition)
- Output: colorized RGB image
- Architecture: U-Net with global self-attention
- Trained on 256x256 images with 1M steps

**Results on ImageNet Colorization**:
- FID: 15.78
- Human fooling rate: 47.8%

**Pros for Hackathon**:
- Single model handles multiple tasks
- High-quality, diverse outputs
- Good code implementation available

**Cons for Hackathon**:
- Requires training (not zero-shot)
- Not specifically designed for IR
- Slow inference due to iterative denoising

---

### 2.4 PID: Physics-Informed Diffusion Model for Infrared Image Generation (2024)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Not specified in search results |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/html/2407.09299v1 |
| **Key Innovation** | Integrates Planck's law of blackbody radiation into diffusion models for physically accurate IR generation |

**Technical Approach**:
- Uses **TeV (Temperature-eV) decomposition** based on Planck's radiation law:
  ```
  B_lambda(T) = (2*pi*h*c^2/lambda^5) * 1/(e^(h*c/(lambda*k*T)) - 1)
  ```
- Physics loss ensures generated infrared images follow actual thermal radiation distribution
- Outperforms standard LDM on FLIR dataset (PSNR: 17.26 vs 17.13, FID: 84.68 vs 90.57)

**How to Adapt for THERMAVISION-X**:
- **Critical**: Use Planck's law to guide colorization -- ensure that pixel temperatures map to physically plausible colors
- Add Infrared Radiation Intensity Alignment (IRA) Loss to thermal regularization
- Combine with ThesIS (Section 2.5) for thermal consistency

---

### 2.5 ThesIS: Thermal-Physics Guided Infrared Image Super-Resolution (AAAI 2025)

| Attribute | Detail |
|-----------|--------|
| **Venue** | AAAI 2025 |
| **Paper** | https://ojs.aaai.org/index.php/AAAI/article/view/38381/42343 |
| **Key Innovation** | Infrared Radiation Intensity Alignment Loss + Dynamic High-frequency Amplification |

**Technical Components**:
1. **Infrared Radiation Intensity Alignment (IRA) Loss**: Restores accurate thermal radiation distribution using grayscale histogram similarity metrics (COR, BHAD)
2. **Dynamic Frequency Filter Block (DFFB)**: Extracts and enhances high-frequency components in feature maps

**Results**: Achieves SOTA on InfraredSR-Synthetic dataset with PSNR 23.02 (vs 22.56 without IRA loss)

---

## 3. Frequency-Domain Approaches

### 3.1 Focal Frequency Loss (Jiang et al., ICCV 2021)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy |
| **Venue** | ICCV 2021 |
| **Paper** | https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf |
| **Project Page** | https://www.mmlab-ntu.com/project/ffl/index.html |
| **Key Innovation** | Adaptive frequency-domain loss that focuses on "hard" frequencies |

#### Technical Approach

**Motivation**: Neural networks inherently bias toward "easy" frequencies (low-frequency content), losing high-frequency details (edges, textures). FFL forces the network to focus on hard-to-synthesize frequencies.

**Formulation**:

1. Transform real and fake images to frequency domain via 2D DFT:
   ```
   F_r = DFT(image_real), F_f = DFT(image_fake)
   ```

2. Compute frequency distance:
   ```
   d(F_r, F_f) = (1/MN) * sum |F_r(u,v) - F_f(u,v)|^2
   ```

3. Dynamic spectrum weighting (focus on hard frequencies):
   ```
   w(u,v) = |F_r(u,v) - F_f(u,v)|^alpha   (alpha = 1 by default)
   ```

4. Focal Frequency Loss:
   ```
   FFL = (1/MN) * sum w(u,v) * |F_r(u,v) - F_f(u,v)|^2
   ```

**Results** (image reconstruction):
| Model | Metric | w/o FFL | w/ FFL |
|-------|--------|---------|--------|
| Vanilla AE (CelebA) | PSNR | 20.04 | 21.70 |
| Vanilla AE (CelebA) | SSIM | 0.568 | 0.642 |
| pix2pix | FID | 80.28 | 74.36 |
| StyleGAN2 | FID | 5.696 | 4.972 |

**Implementation**:
```python
import torch.fft as fft

def focal_frequency_loss(real, fake, alpha=1.0):
    # 2D DFT
    real_freq = fft.fft2(real)
    fake_freq = fft.fft2(fake)
    
    # Frequency distance
    freq_diff = torch.abs(real_freq - fake_freq)
    
    # Dynamic weighting
    weight = freq_diff ** alpha
    weight = weight / (weight.max() + 1e-8)  # normalize
    
    # FFL
    loss = (weight * freq_diff ** 2).mean()
    return loss
```

**Pros for Hackathon**:
- Only ~5% computational overhead
- Plug-and-play: can be added to any existing loss function
- Significantly improves high-frequency detail preservation
- Perfect complement to spatial losses (L1, perceptual)

**Cons for Hackathon**:
- Requires paired ground truth for reference spectrum
- For unpaired settings, can use "fake" reference from cycle consistency

---

### 3.2 Frequency Domain Loss Functions (DCT/FFT-based)

**DCT-based Loss** (from exposure correction work):
```
L_DCT = (1/MN) * |DCT(I1) - DCT(I2)|
```
- Applied at multiple scales (full, half, quarter resolution)
- FFT variant: L_FFT = (1/MN) * |FFT(I1) - FFT(I2)|
- Reduces noise, blur, and color artifacts

**Best Practice for THERMAVISION-X**:
- Combine **FFL** (focal, adaptive) with **multi-scale FFT loss** (coarse frequency matching)
- Use FFL for generator training, FFT loss for discriminator

---

### 3.3 Masked Image Modeling in Frequency Domain

**Key Insight from Chiheng Wei et al.**: Instead of masking spatial patches (as in MAE), mask **frequency bands** to force the network to learn frequency-domain reconstruction.

**Implementation Strategy**:
```python
def frequency_domain_masking(image, mask_ratio=0.5):
    """Mask low-frequency components in DFT domain"""
    freq = fft.fft2(image)
    
    # Create low-frequency mask (center of spectrum)
    h, w = freq.shape[-2:]
    center_h, center_w = h // 2, w // 2
    radius = int(min(h, w) * mask_ratio / 2)
    
    mask = torch.ones_like(freq)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    dist = ((y - center_h)**2 + (x - center_w)**2).sqrt()
    mask[dist < radius] = 0  # mask low frequencies
    
    masked_freq = freq * mask
    return masked_freq, mask
```

**Advantages over Spatial MIM**:
- More structured information removal (entire frequency bands vs random patches)
- Better aligns with the physics of image formation
- Easier for network to learn systematic reconstruction

---

## 4. Self-Supervised & Contrastive Learning for IR Colorization

### 4.1 CUT: Contrastive Learning for Unpaired Image-to-Image Translation (ECCV 2020)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu |
| **Venue** | ECCV 2020 |
| **Paper** | https://arxiv.org/abs/2007.15651 |
| **Code** | https://github.com/taesungp/contrastive-unpaired-translation |
| **Key Innovation** | PatchNCE loss: maximizes mutual information between corresponding input-output patches |

#### Technical Approach

**PatchNCE Loss** (core of CUT):
```python
def PatchNCELoss(f_q, f_k, tau=0.07):
    """
    f_q: query features from generated image (BxCxS)
    f_k: key features from input image (BxCxS)
    tau: temperature parameter
    """
    B, C, S = f_q.shape
    
    # Positive: corresponding patch
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]  # BxSx1
    
    # Negatives: all other patches
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)  # BxSxS
    
    # Remove self from negatives
    identity = torch.eye(S)[None, :, :]
    l_neg.masked_fill_(identity, -float('inf'))
    
    # Cross-entropy loss (InfoNCE)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long)
    return F.cross_entropy(predictions, targets)
```

**Full CUT Objective**:
```
L = L_GAN + lambda_X * L_PatchNCE(X) + lambda_Y * L_PatchNCE(Y)
```
- lambda_X = 1, lambda_Y = 1 (with identity loss)
- lambda_X = 10, lambda_Y = 0 (FastCUT variant -- faster but less accurate)

**Key Properties**:
- **Single-sided**: Only one generator (unlike CycleGAN's two)
- **No cycle consistency**: Uses mutual information maximization instead
- **Patch-level**: Operates on features from multiple encoder layers
- **Self-supervised**: No paired data needed

**Pros for Hackathon**:
- Fast training (single generator)
- Strong baselines with proven results
- Excellent code implementation
- PatchNCE is domain-agnostic -- works for IR-to-RGB

**Cons for Hackathon**:
- Mode collapse on some tasks (e.g., Photo->Label)
- Uses single encoder for both domains -- limited for large domain gaps (IR vs RGB)
- Identity loss can overly constrain color diversity

---

### 4.2 DCLGAN: Dual Contrastive Learning GAN (CVPRW 2021)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Junlin Han, Mehrdad Shoeiby, Lars Petersson, Mohammad Ali Armin |
| **Venue** | NTIRE, CVPRW 2021 (Oral) |
| **Paper** | https://arxiv.org/abs/2104.07689 |
| **Code** | https://github.com/JunlinHan/DCLGAN |
| **Key Innovation** | Dual encoder architecture + dual PatchNCE for better cross-domain learning; SimDCL variant avoids mode collapse |

#### Technical Approach

**Architecture**: 
- Two generators (G: X->Y, F: Y->X) -- similar to CycleGAN structure
- Two discriminators (D_X, D_Y)
- **Two separate encoders** G_enc and F_enc for source/target domains
- **Two separate MLP projection heads** H_X and H_Y

**Loss Function**:
```
L = lambda_GAN * L_GAN 
  + lambda_NCE * L_PatchNCE_X(G, H_X, H_Y, X)
  + lambda_NCE * L_PatchNCE_Y(F, H_X, H_Y, Y)
  + lambda_id * L_identity
```
- lambda_GAN = 1, lambda_NCE = 2, lambda_id = 1

**SimDCL Variant** (adds similarity loss to prevent mode collapse):
```
L_sim = similarity loss between real and fake within same domain
```
- lambda_SIM = 10
- Effectively prevents mode collapse in tasks like Photo->Label

**Results** (FID scores, lower is better):
| Method | Horse->Zebra | Cat->Dog | CityScapes |
|--------|-------------|----------|------------|
| CycleGAN | 154.3 | 107.7 | 65.7 |
| CUT | 170.5 | 26.8 | 74.7 |
| **DCLGAN** | **139.6** | **23.2** | **51.1** |
| SimDCL | 152.5 | 22.8 | 61.4 |

**Pros for Hackathon**:
- Separate encoders better handle large domain gap (IR vs RGB)
- More robust than CUT (dual learning stabilizes training)
- SimDCL variant avoids mode collapse entirely
- Geometry changes supported (unlike CycleGAN)

**Cons for Hackathon**:
- Slower than CUT (two generators + two PatchNCE losses)
- More hyperparameters to tune
- Still requires sufficient unpaired data from both domains

---

### 4.3 CCLGAN: Cosine Contrastive Learning GAN (Neurocomputing 2025)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Tingting Liu, Yujue Cai, Guiping Chen, Hongguang Wei, Junqi Bai, Yuan Liu |
| **Venue** | Neurocomputing, July 2025 |
| **Paper** | https://www.sciencedirect.com/science/article/abs/pii/S0925231225013852 |
| **Code** | https://github.com/LTTdouble/CCLGAN |
| **Key Innovation** | Cosine contrastive loss + VSM-UNet with Mamba module for unsupervised IR colorization |

#### Technical Approach

**Three Key Innovations**:

**1. VSM-UNet Generator**:
- Improved UNet with **full-scale skip connections** (connects all encoder layers to all decoder layers)
- **Deep supervision** at multiple decoder levels
- **Mamba module** (Visual State Space Model) for long-range dependency modeling
  - 3D neural attention mechanism (spatial + channel dimensions)
  - Linear computational complexity O(n) vs O(n^2) for transformers
  - Parameter-free attention for feature selection and fusion

**2. Cosine Contrastive Loss**:
Traditional PatchNCE uses Euclidean distance in feature space. CCLGAN replaces this with **cosine similarity** in angular space:

```python
def cosine_contrastive_loss(z_pos, z_neg, z_query, margin=0.5):
    """
    z_pos: positive sample features
    z_neg: negative sample features  
    z_query: query features
    margin: cosine margin
    """
    # Cosine similarity
    sim_pos = F.cosine_similarity(z_query, z_pos, dim=-1)
    sim_neg = F.cosine_similarity(z_query, z_neg, dim=-1)
    
    # Maximize decision margin in cosine space
    loss = -torch.log(
        torch.exp(sim_pos) / 
        (torch.exp(sim_pos) + torch.exp(sim_neg + margin))
    )
    return loss.mean()
```

Key differences from standard PatchNCE:
- **Cosine distance** instead of dot product (angle-based similarity)
- **L-Softmax** for angular margin maximization
- Minimizes intra-class variance, maximizes inter-class variance
- Better feature discrimination for cross-domain tasks

**3. Full-Scale Feature Fusion**:
- Every encoder layer connects to every decoder layer
- Hierarchical feature maps learned via deep supervision
- Multi-scale information integration

**Results** (compared to CUT, DCLGAN, etc.):
- Outperforms existing methods on KAIST and FLIR datasets
- Less error in reconstructed structures
- Better detail preservation

**Pros for Hackathon**:
- State-of-the-art for unsupervised IR colorization
- Mamba provides efficient long-range modeling
- Cosine loss is more discriminative for cross-domain tasks
- Full-scale skip connections preserve fine details

**Cons for Hackathon**:
- Complex architecture (Mamba + full-scale UNet)
- Requires training on unpaired IR-RGB data
- May be computationally expensive

**How to Adapt for THERMAVISION-X**:
- Use VSM-UNet as the generator backbone
- Replace standard PatchNCE with cosine contrastive loss
- Combine with frequency-domain decoupling (Section 2.1)
- Add physics-informed thermal constraints

---

### 4.4 MCL: Multi-feature Contrastive Learning (CIM 2022)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Yi Han, Wei Li |
| **Venue** | Complex & Intelligent Systems, 2022 |
| **Paper** | https://link.springer.com/article/10.1007/s40747-022-00924-1 |
| **Key Innovation** | Contrastive loss on discriminator features (MCL loss) + PatchNCE on generator |

**Technical Approach**:
- Uses **discriminator output layer features** to construct additional contrastive loss
- 30x30 feature matrix from PatchGAN discriminator -> each row is a feature vector
- MCL loss enhances discriminator generalization
- Adds little computational overhead (no extra parameters)

**FastMCL Variant**: Removes one PatchNCE term for faster training, achieves near-CUT performance.

---

### 4.5 MabCUT: Multi-Attention Bidirectional CUT (2024)

| Attribute | Detail |
|-----------|--------|
| **Venue** | Published 2024 |
| **Paper** | https://pmc.ncbi.nlm.nih.gov/articles/PMC11020845/ |
| **Key Innovation** | Bidirectional CUT with attention-based feature selection |

**Technical Approach**:
- Two generators (A: S->T, B: T->S) for bidirectional mapping
- **Attention matrix** selects relevant features for PatchNCE from multiple encoder layers
- Multi-layer feature extraction via embedding blocks (encoder + 2-layer MLP)
- Identity loss prevents distortion

---

### 4.6 CS2Fusion: Contrastive Self-Supervised IVIF (InfoFusion 2024)

| Attribute | Detail |
|-----------|--------|
| **Authors** | Xue Wang, Zheng Guan, Wenhua Qian, Jinde Cao, Shu Liang, Jin Yan |
| **Venue** | Information Fusion, 2024 |
| **Code** | https://github.com/wang-x-1997/CS2Fusion |
| **Key Innovation** | Self-correlation and saliency operation (SSO) for contrastive pair construction in IR-visible fusion |

**Key Idea**: Despite semantic differences between IR and visible, the **self-correlation and saliency feature distributions** are similar within each modality. SSO constructs positive/negative pairs based on this observation.

---

## 5. Quality Assessment for Colorized IR Images

### 5.1 Traditional No-Reference IQA Methods

#### NIQE (Natural Image Quality Evaluator, 2013)
```
NIQE = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
```
- Uses Natural Scene Statistics (NSS) from pristine images
- No training/human labels needed
- Computes Mahalanobis distance from natural image distribution

#### BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator, 2012)
- Locally normalized luminance coefficients (MSCN)
- Fits generalized Gaussian model
- SVR regression to predict quality score

#### PIQE (Perceptual Image Quality Evaluator, 2015)
```
PIQE = (1/N) * sum(d_i)  # average distortion per block
```
- Block-based: divides image into patches
- Computes local variance, block sharpness, saturation
- No reference needed

**Limitations for AI-Generated Images**:
- Designed for natural photographs, not synthetic images
- Misinterprets stylistic choices as distortions
- May penalize legitimate colorization artifacts

---

### 5.2 Deep Learning-Based No-Reference IQA

#### NIMA (Neural Image Assessment, Google, 2018)
- CNN backbone (MobileNet/VGG) for aesthetic quality prediction
- Predicts distribution of human ratings
- ~3M parameters (lightweight variant)
- SROCC: 0.698 on TID2013

#### TReS (2022)
- Transformer-based IQA
- ~90M parameters
- SROCC: 0.863 on TID2013

#### DEIQT (Data-Efficient IQA with Transformer, AAAI 2023)
- ViT + Decoder + Attention Panel
- Simulates human expert variability
- ~88M parameters
- SROCC: 0.892 on TID2013

#### ExIQA (2024)
- CLIP + Attribute Prompting
- Identifies distortion types and strengths
- Explainable predictions
- ~86M parameters
- SROCC: 0.912 on TID2013

#### CoDI-IQA (2025)
- Cross-domain Distortion Identification
- ResNet-50 + Swin Transformer encoders
- Disentangles content from distortion
- ~47M parameters
- SROCC: 0.901 on TID2013

#### LIQE (2023)
- CLIP vision-language multitask
- Template-based embedding for quality prediction
- Unified prediction framework

---

### 5.3 Recommended Approach for THERMAVISION-X

Since we have **no ground truth** for colorized IR images, use a **multi-metric ensemble**:

```python
def evaluate_colorization_quality(colorized_ir, original_ir=None):
    """
    No-reference quality assessment for colorized IR images
    """
    metrics = {}
    
    # 1. Naturalness (how natural the colors look)
    metrics['NIQE'] = compute_niqe(colorized_ir)  # lower is better
    metrics['BRISQUE'] = compute_brisque(colorized_ir)  # lower is better
    
    # 2. Sharpness / Detail preservation
    metrics['PIQE'] = compute_piqe(colorized_ir)  # lower is better
    metrics['grad_score'] = gradient_magnitude(colorized_ir)
    
    # 3. Colorfulness
    metrics['colorfulness'] = compute_colorfulness(colorized_ir)
    
    # 4. Thermal consistency (if original IR available)
    if original_ir is not None:
        # Convert colorized to grayscale, compare structure
        gray = rgb2gray(colorized_ir)
        metrics['SSIM_IR'] = ssim(gray, original_ir)
        # Frequency domain consistency
        metrics['LFD'] = log_frequency_distance(gray, original_ir)
    
    # 5. Overall score (weighted ensemble)
    overall = (
        -0.3 * metrics['NIQE'] + 
        -0.2 * metrics['BRISQUE'] + 
        -0.2 * metrics['PIQE'] + 
        0.15 * metrics['colorfulness'] +
        0.15 * metrics.get('SSIM_IR', 0.5)
    )
    metrics['overall'] = overall
    
    return metrics
```

**Additional Quality Metrics for Generated Images**:
- **FID (Frechet Inception Distance)**: Distribution distance between generated and real RGB images (requires RGB dataset)
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual distance
- **LFD (Log Frequency Distance)**: Frequency domain gap

---

## 6. Emerging Trends & Novel Ideas for Hackathon

### 6.1 Mamba/State Space Models for IR Processing

**Why Mamba for IR Colorization**:
- **Linear complexity** O(n) vs transformers' O(n^2) -- critical for high-res satellite IR images
- **Long-range dependencies** -- can model global temperature patterns
- **3D attention** -- spatial + channel attention for feature selection
- Used in CCLGAN's VSM-UNet generator

**Mamba Block Architecture**:
```
Input -> Linear(2x) -> Split
                        |-> x_proj -> 1D Conv -> SiLU -> SSM -> Output
                        |-> z_proj -> SiLU (gating) ---------> *
```

### 6.2 CLIP-Guided Infrared Colorization (CMMF-Net, 2025)

**Key Idea**: Use **text descriptions** of the IR scene (e.g., "urban road at night with vehicles") to guide colorization via CLIP.

**Architecture**:
- ViT Image_Encoder for IR features
- CLIP Text_Encoder for text features
- Cross-Modal Interaction (CI) module: text features as Query, image as Key/Value
- LK_U-Net decoder for final colorization

**Results on KAIST**:
- SSIM: 0.58 (best among compared methods)
- PSNR: 16.22 dB

**Adaptation for THERMAVISION-X**:
- Auto-generate text descriptions from IR via image captioning
- Use CLIP to align generated colors with scene semantics

### 6.3 Physics-Guided Colorization (Recommended Novel Approach)

**Novel Idea**: Combine multiple physics constraints:

1. **Planck's Law**: Temperature -> color mapping
   ```
   T_pixel -> B_lambda(T) -> RGB values
   ```
   
2. **Atmospheric Transfer Equation**: Account for atmospheric effects in satellite imagery
   
3. **Material Emissivity**: Different materials (water, vegetation, concrete) have different emissivity -> different apparent temperatures -> different colors

**Implementation**:
```python
class PhysicsGuidedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Planck's constants
        self.h = 6.626e-34
        self.c = 3e8
        self.k = 1.381e-23
        
    def planck_radiation(self, T, wavelength):
        """Compute blackbody radiation at temperature T"""
        return (2*np.pi*self.h*self.c**2 / wavelength**5) / \
               (torch.exp(self.h*self.c/(wavelength*self.k*T)) - 1)
    
    def forward(self, generated_rgb, ir_input):
        # Convert RGB back to temperature estimate
        T_estimated = self.rgb_to_temperature(generated_rgb)
        
        # Temperature should match IR intensity
        loss_temp = F.mse_loss(T_estimated, ir_input)
        
        # Colors should follow Planck's law
        expected_rgb = self.planck_to_rgb(T_estimated)
        loss_planck = F.mse_loss(generated_rgb, expected_rgb)
        
        return loss_temp + 0.5 * loss_planck
```

---

## 7. Recommended Architecture for THERMAVISION-X

### Proposed: "THERMAVISION-X" Architecture

Based on our research, we recommend a **hybrid approach** combining the best of multiple methods:

```
                        INPUT: IR Image
                            |
                    [Frequency Decoupling]
                    (Chiheng Wei et al.)
                            |
              +-------------+-------------+
              |                           |
    [High-Freq Component]      [Low-Freq Component]
    (structure preserved)       (to be colorized)
              |                           |
              |               [DDNM Zero-Shot Colorization]
              |               + Physics-Guided Constraints
              |               + CLIP Semantic Guidance
              |                           |
              |               [Colorized Low-Freq]
              |                           |
              +-------------+-------------+
                            |
                   [Frequency Recomposition]
                            |
               [Refinement Network (CCLGAN-style)]
               - Cosine Contrastive Loss
               - Mamba-based Generator
               - Focal Frequency Loss
                            |
                    OUTPUT: Colorized RGB
                            |
               [No-Reference Quality Assessment]
               - NIQE, BRISQUE, PIQE ensemble
               - Thermal consistency check
```

### Key Components

| Component | Method | Purpose |
|-----------|--------|---------|
| Frequency Decoupling | 2D DFT + High-pass mask (Chiheng Wei) | Preserve IR structure |
| Zero-Shot Colorization | DDNM+ with custom operator (Wang et al.) | Generate realistic color |
| Physics Guidance | Planck's Law + IRA Loss (ThesIS, PID) | Thermal consistency |
| Semantic Guidance | CLIP cross-attention (CMMF-Net) | Realistic scene colors |
| Refinement | VSM-UNet with Cosine Loss (CCLGAN) | Detail enhancement |
| Training Loss | Focal Frequency Loss (Jiang et al.) | Frequency-domain quality |
| Quality Assessment | NIQE + BRISQUE + PIQE ensemble | No-reference evaluation |

### Implementation Priority (for hackathon timeline)

**Phase 1 (MVP)**:
1. Frequency-domain feature decoupling (simple FFT operations)
2. DDNM zero-shot colorization (use existing code)
3. Basic no-reference IQA

**Phase 2 (Enhancement)**:
4. Add physics-informed loss (Planck's law)
5. CLIP semantic guidance
6. Focal frequency loss

**Phase 3 (Polish)**:
7. CCLGAN-style refinement network
8. Mamba module integration
9. Full quality assessment pipeline

---

## 8. References & Code Repositories

### Key Papers

| # | Paper | Authors | Year | Venue | Code |
|---|-------|---------|------|-------|------|
| 1 | Infrared colorization with cross-modality zero-shot learning | Chiheng Wei et al. | 2024 | Neurocomputing | N/A |
| 2 | Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model | Yinhuai Wang et al. | 2023 | ICLR (Oral) | https://github.com/wyhuai/DDNM |
| 3 | Focal Frequency Loss for Image Reconstruction and Synthesis | Liming Jiang et al. | 2021 | ICCV | https://github.com/liming-jiang/FFL |
| 4 | Contrastive Learning for Unpaired Image-to-Image Translation | Taesung Park et al. | 2020 | ECCV | https://github.com/taesungp/contrastive-unpaired-translation |
| 5 | Dual Contrastive Learning for Unsupervised Image-to-Image Translation | Junlin Han et al. | 2021 | CVPRW | https://github.com/JunlinHan/DCLGAN |
| 6 | Adversarial network for unsupervised IR colorization based on full-scale feature fusion and cosine contrastive learning (CCLGAN) | Tingting Liu et al. | 2025 | Neurocomputing | https://github.com/LTTdouble/CCLGAN |
| 7 | Palette: Image-to-Image Diffusion Models | Chitwan Saharia et al. | 2022 | ACM ToG | https://github.com/janspiry/palette-image-to-image-diffusion-models |
| 8 | CMMF-Net: CLIP-guided multi-modal feature fusion for TIR image colorization | Qian Jiang et al. | 2025 | Intelligence and Robotics | N/A |
| 9 | PID: Physics-Informed Diffusion Model for Infrared Image Generation |  | 2024 | arXiv | N/A |
| 10 | ThesIS: Thermal-Physics Guided Infrared Image Super-Resolution |  | 2025 | AAAI | N/A |
| 11 | MTSIC: Multi-stage Transformer-based GAN for Spectral IR Image Colorization | Tingting Liu et al. | 2025 | arXiv | N/A |
| 12 | Multi-feature Contrastive Learning for Unpaired Image-to-Image Translation | Yi Han, Wei Li | 2022 | CIM | N/A |
| 13 | Multi-attention Bidirectional Contrastive Learning (MabCUT) |  | 2024 |  | N/A |
| 14 | CS2Fusion: Contrastive Self-Supervised IR and Visible Image Fusion | Xue Wang et al. | 2024 | InfoFusion | https://github.com/wang-x-1997/CS2Fusion |
| 15 | SwinFuSR: RGB-guided Thermal Image Super-Resolution |  | 2024 | CVPRW | N/A |
| 16 | What Is A Mamba Model | IBM | 2025 | Blog | N/A |

### IQA References

| # | Method | Year | Key Feature |
|---|--------|------|-------------|
| 1 | NIQE | 2013 | NSS-based, no training needed |
| 2 | BRISQUE | 2012 | MSCN statistics + SVR |
| 3 | PIQE | 2015 | Block-based distortion |
| 4 | NIMA | 2018 | CNN-based aesthetic prediction |
| 5 | TReS | 2022 | Transformer-based |
| 6 | DEIQT | 2023 | ViT + attention panel |
| 7 | LIQE | 2023 | CLIP-based |
| 8 | ExIQA | 2024 | Explainable, CLIP + attributes |
| 9 | CoDI-IQA | 2025 | Cross-domain distortion ID |

### Datasets for IR Colorization

| Dataset | Description | Link |
|---------|-------------|------|
| KAIST Multispectral | ~95,000 paired IR-visible images | https://soonminhwang.github.io/rgbt-ped-detection/data/ |
| FLIR | Urban scenes, day/night | https://www.flir.com/oem/adas/adas-dataset-form/ |
| IRVI | Infrared-to-Visible Video | https://github.com/BIT-DA/I2V-GAN |
| InfraredSR-Synthetic | IR super-resolution benchmark | AAAI 2025 ThesIS paper |
| LLVIP | Low-light visible-infrared paired | https://bupt-ai-cz.github.io/LLVIP/ |
| TNO | Multi-band military IR | Standard fusion benchmark |

---

## Appendix: Key Equations

### A.1 Range-Null Space Decomposition
For linear inverse problem y = Ax:
```
x = A+ y + (I - A+ A) z
    [range-space]   [null-space]
```
Where A+ is pseudo-inverse, z is any vector.

### A.2 Focal Frequency Loss
```
FFL = (1/MN) * sum_{u,v} w(u,v) * |F_r(u,v) - F_f(u,v)|^2
w(u,v) = |F_r(u,v) - F_f(u,v)|^alpha
```

### A.3 PatchNCE Loss (InfoNCE)
```
L = -log[ exp(sim(v, v+)/tau) / (exp(sim(v, v+)/tau) + sum exp(sim(v, v_n-)/tau)) ]
```

### A.4 Cosine Contrastive Loss (CCLGAN)
```
L = -log[ exp(cos(v, v+)) / (exp(cos(v, v+)) + exp(cos(v, v-) + margin)) ]
```

### A.5 Planck's Radiation Law
```
B_lambda(T) = (2*pi*h*c^2/lambda^5) * 1/(e^(h*c/(lambda*k*T)) - 1)
```

---

*Research compiled for THERMAVISION-X -- ISRO Bharatiya Antariksh Hackathon 2026*
*Last updated: January 2025*
