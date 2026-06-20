# demo/app.py
# THERMAVISION-X — Interactive Streamlit Demo
# Run with: streamlit run demo/app.py
# Researched & designed by Benad | BAH 2026

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import ThermaVisionUNet
from models.frequency import FrequencyDecouplingModule
from models.uncertainty import UncertaintyEstimator
import torchvision.transforms as T

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="THERMAVISION-X | ISRO BAH 2026",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p  { opacity: 0.8; margin: 0.5rem 0 0; }
    .metric-card {
        background: #1e2130; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #303450;
    }
    .innovation-badge {
        display: inline-block; background: #302b63; color: white;
        padding: 0.3rem 0.8rem; border-radius: 20px;
        margin: 0.2rem; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛰️ THERMAVISION-X</h1>
    <p>Physics-Guided Zero-Shot Infrared Colorization for ISRO Satellite Imagery</p>
    <p><small>Bharatiya Antariksh Hackathon 2026 | Challenge #10</small></p>
</div>
""", unsafe_allow_html=True)

# Innovation badges
st.markdown("""
<div style='text-align:center; margin-bottom:1.5rem;'>
    <span class='innovation-badge'>🔬 Zero-Shot Learning</span>
    <span class='innovation-badge'>⚛️ Planck's Law Physics</span>
    <span class='innovation-badge'>🧠 UNet Architecture</span>
    <span class='innovation-badge'>📊 Uncertainty Maps</span>
    <span class='innovation-badge'>🛸 ISRO-First Design</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Settings ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint", value="./checkpoints/best_model.pth"
)
img_size = st.sidebar.slider("Processing Size (px)", 128, 512, 256, 64)
n_passes = st.sidebar.slider("MC Dropout Passes (Uncertainty)", 5, 20, 10)
radius_ratio = st.sidebar.slider("FFT Mask Radius", 0.05, 0.40, 0.15, 0.01)
show_uncertainty = st.sidebar.checkbox("Show Uncertainty Map", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How It Works:**
1. Upload a grayscale thermal/IR image
2. FFT frequency decoupling extracts structure
3. UNet predicts color from structure only
4. Physics constraints (Planck's Law) ensure accuracy
5. MC Dropout generates pixel-level confidence map
""")


# ── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str, img_size: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThermaVisionUNet(in_channels=1, out_channels=3)

    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get('model_state', ckpt)
            model.load_state_dict(state)
            st.sidebar.success("✅ Model loaded from checkpoint")
        except Exception as e:
            st.sidebar.warning(f"Could not load checkpoint: {e}. Using random weights for demo.")
    else:
        st.sidebar.warning("⚠️ No checkpoint found. Using untrained model (demo only).")

    model.eval()
    return model, device


def hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV numpy array [H, W, 3] to RGB."""
    from PIL import Image as PILImage
    import colorsys
    h, w, _ = hsv.shape
    rgb = np.zeros_like(hsv)
    for i in range(h):
        for j in range(w):
            r, g, b = colorsys.hsv_to_rgb(hsv[i,j,0], hsv[i,j,1], hsv[i,j,2])
            rgb[i,j] = [r, g, b]
    return rgb


# ── Main: Upload & Process ────────────────────────────────────────────────
st.markdown("## 📤 Upload Infrared Image")
uploaded = st.file_uploader(
    "Upload a grayscale thermal/infrared image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Supports KAIST, FLIR, INSAT thermal images"
)

if uploaded:
    model, device = load_model(checkpoint_path, img_size)
    freq_mod = FrequencyDecouplingModule(radius_ratio=radius_ratio).to(device)

    # Load image
    img = Image.open(uploaded).convert('L')
    orig_size = img.size

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📷 Input (Grayscale IR)")
        st.image(img, use_container_width=True, caption="Original thermal IR image")

    # Preprocess
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with st.spinner("🔄 Running zero-shot colorization..."):
        with torch.no_grad():
            freq_out  = freq_mod(tensor)
            structure = freq_out['masked_input']

        if show_uncertainty:
            estimator = UncertaintyEstimator(model, n_passes=n_passes)
            pred_hsv, uncert = estimator.predict(structure)
            confidence = estimator.uncertainty_to_heatmap(uncert)
        else:
            with torch.no_grad():
                pred_hsv = model(structure)
            confidence = None

        # Convert to displayable RGB
        pred_np = pred_hsv[0].permute(1, 2, 0).cpu().numpy()
        pred_np = np.clip(pred_np, 0, 1)

        # Simple HSV→RGB using numpy
        from matplotlib.colors import hsv_to_rgb as mpl_hsv2rgb
        pred_rgb = mpl_hsv2rgb(pred_np)
        pred_img = Image.fromarray((pred_rgb * 255).astype(np.uint8)).resize(orig_size)

    with col2:
        st.markdown("#### 🎨 Colorized Output")
        st.image(pred_img, use_container_width=True,
                 caption="Physics-guided colorization (HSV→RGB)")

    with col3:
        if show_uncertainty and confidence is not None:
            st.markdown("#### 📊 Confidence Map")
            conf_np = confidence[0, 0].cpu().numpy()
            conf_img = Image.fromarray((conf_np * 255).astype(np.uint8)).resize(orig_size)
            st.image(conf_img, use_container_width=True,
                     caption="Bright = High confidence | Dark = Low confidence")
        else:
            st.markdown("#### ℹ️ Info")
            st.info("Enable 'Show Uncertainty Map' in sidebar to see the confidence heatmap")

    # Metrics & Download
    st.markdown("---")
    st.markdown("## 📈 Results")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Image Size", f"{orig_size[0]}×{orig_size[1]} px")
    with col_b:
        st.metric("Model Params", "~5M parameters")
    with col_c:
        device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_str)
    with col_d:
        st.metric("Architecture", "UNet + FFT")

    # Download button
    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Colorized Image",
        data=buf.getvalue(),
        file_name="thermavision_colorized.png",
        mime="image/png"
    )

else:
    # Demo placeholder when no image uploaded
    st.info("👆 Upload a grayscale infrared/thermal image to see the magic happen!")
    st.markdown("""
    **Try with:**
    - Any grayscale thermal image (KAIST, FLIR, INSAT datasets)
    - Standard grayscale JPG/PNG images (for demo purposes)
    - Any satellite thermal band exported as grayscale
    """)
