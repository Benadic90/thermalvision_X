# deployment/export.py
# THERMAVISION-X — Model Export for Edge Deployment
# Exports trained PyTorch model to ONNX format for CPU/Raspberry Pi deployment
# Researched & designed by Benad | BAH 2026

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import ThermaVisionUNet


def export_to_onnx(checkpoint_path: str, output_path: str,
                   img_size: int = 256):
    """
    Export trained model to ONNX format for edge deployment.
    ONNX models can run on:
        - Raspberry Pi (via ONNX Runtime)
        - Any CPU without PyTorch installed
        - Future: TensorRT on Jetson Nano
    """
    device = torch.device('cpu')  # Export from CPU
    model = ThermaVisionUNet(in_channels=1, out_channels=3)

    # Load weights
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️  No checkpoint found — exporting untrained model (demo)")

    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, 1, img_size, img_size)

    # Export to ONNX
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=11,
        input_names=['ir_input'],
        output_names=['hsv_output'],
        dynamic_axes={
            'ir_input':   {0: 'batch', 2: 'height', 3: 'width'},
            'hsv_output': {0: 'batch', 2: 'height', 3: 'width'},
        },
        verbose=False,
    )

    # File size report
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ ONNX model saved: {output_path}")
    print(f"   File size: {size_mb:.1f} MB")

    if size_mb < 50:
        print("   ✅ WITHIN 50 MB target for edge deployment")
    else:
        print("   ⚠️  Exceeds 50 MB — consider quantization")

    return output_path


def benchmark_onnx(onnx_path: str, img_size: int = 256, n_runs: int = 20):
    """Run speed benchmark on the ONNX model (simulates CPU/Raspberry Pi speed)."""
    import onnxruntime as ort
    import numpy as np
    import time

    session = ort.InferenceSession(onnx_path,
                                   providers=['CPUExecutionProvider'])
    dummy = np.random.randn(1, 1, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {'ir_input': dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {'ir_input': dummy})
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    fps    = 1000.0 / avg_ms

    print(f"\n📊 CPU Inference Benchmark ({n_runs} runs):")
    print(f"   Average latency: {avg_ms:.1f} ms")
    print(f"   Throughput:      {fps:.1f} FPS")
    print(f"   Image size:      {img_size}×{img_size}")


if __name__ == "__main__":
    os.makedirs('./exports', exist_ok=True)

    onnx_path = export_to_onnx(
        checkpoint_path='./checkpoints/best_model.pth',
        output_path='./exports/thermavision_x.onnx',
        img_size=256,
    )
    benchmark_onnx(onnx_path)
