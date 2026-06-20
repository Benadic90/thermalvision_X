# inference/colorize.py
# THERMAVISION-X — Zero-Shot Colorization Script
# Feed any grayscale IR image → get a physics-accurate colorized image back
# Researched & designed by Benad | BAH 2026

import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.unet import ThermaVisionUNet
from models.frequency import FrequencyDecouplingModule
from models.uncertainty import UncertaintyEstimator


def load_model(checkpoint_path: str, device: torch.device) -> ThermaVisionUNet:
    """Load trained model from checkpoint."""
    model = ThermaVisionUNet(in_channels=1, out_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV tensor [B, 3, H, W] to RGB tensor [B, 3, H, W]."""
    h = hsv[:, 0:1, :, :] * 360.0
    s = hsv[:, 1:2, :, :]
    v = hsv[:, 2:3, :, :]

    hi = (h / 60.0).long() % 6
    f  = (h / 60.0) - (h / 60.0).long().float()
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)

    r = torch.where(hi == 0, v, torch.where(hi == 1, q, torch.where(hi == 2, p,
        torch.where(hi == 3, p, torch.where(hi == 4, t, v)))))
    g = torch.where(hi == 0, t, torch.where(hi == 1, v, torch.where(hi == 2, v,
        torch.where(hi == 3, q, torch.where(hi == 4, p, p)))))
    b = torch.where(hi == 0, p, torch.where(hi == 1, p, torch.where(hi == 2, t,
        torch.where(hi == 3, v, torch.where(hi == 4, v, q)))))

    return torch.cat([r, g, b], dim=1).clamp(0, 1)


def colorize(input_path: str, output_path: str, checkpoint_path: str,
             img_size: int = 256, uncertainty: bool = True,
             n_passes: int = 10):
    """
    Zero-shot colorize a single IR image.

    Args:
        input_path:      Path to input grayscale IR image
        output_path:     Path to save colorized output
        checkpoint_path: Path to trained model checkpoint
        img_size:        Processing size
        uncertainty:     Whether to also save uncertainty heatmap
        n_passes:        MC Dropout passes for uncertainty
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model from: {checkpoint_path}")

    # Load model and modules
    model    = load_model(checkpoint_path, device)
    freq_mod = FrequencyDecouplingModule(radius_ratio=0.15).to(device)

    # Load and preprocess input
    print(f"Processing: {input_path}")
    img = Image.open(input_path).convert('L')
    orig_size = img.size  # (W, H)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]

    # Frequency decoupling
    with torch.no_grad():
        freq_out  = freq_mod(tensor)
        structure = freq_out['masked_input']

    # Colorize with uncertainty
    if uncertainty:
        estimator = UncertaintyEstimator(model, n_passes=n_passes)
        pred_hsv, uncert = estimator.predict(structure)
        confidence = estimator.uncertainty_to_heatmap(uncert)
    else:
        with torch.no_grad():
            pred_hsv = model(structure)
        confidence = None

    # Convert HSV → RGB
    pred_rgb = hsv_to_rgb(pred_hsv)

    # Resize back to original size and save
    pred_np  = (pred_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pred_img = Image.fromarray(pred_np).resize(orig_size, Image.BILINEAR)
    pred_img.save(output_path)
    print(f"✅ Colorized image saved: {output_path}")

    # Save uncertainty heatmap
    if confidence is not None:
        conf_np  = (confidence[0, 0].cpu().numpy() * 255).astype(np.uint8)
        conf_img = Image.fromarray(conf_np, mode='L').resize(orig_size, Image.BILINEAR)
        conf_path = output_path.replace('.', '_confidence.')
        conf_img.save(conf_path)
        print(f"✅ Confidence map saved: {conf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='THERMAVISION-X Zero-Shot Colorization')
    parser.add_argument('--input',      type=str, required=True,
                        help='Path to input grayscale IR image')
    parser.add_argument('--output',     type=str, default='output_colorized.png',
                        help='Path for colorized output image')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--img_size',   type=int, default=256,
                        help='Processing image size')
    parser.add_argument('--uncertainty',action='store_true', default=True,
                        help='Generate uncertainty/confidence map')
    parser.add_argument('--n_passes',   type=int, default=10,
                        help='Monte Carlo passes for uncertainty')

    args = parser.parse_args()
    colorize(args.input, args.output, args.checkpoint,
             args.img_size, args.uncertainty, args.n_passes)
