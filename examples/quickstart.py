"""Minimal end-to-end example: JPEG file -> normalized NCHW float tensor on the GPU.

    uv run python examples/quickstart.py path/to/image.jpg
"""

import sys

import torch

import nvjpeg_bridge as njb

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

dec = njb.Decoder()  # picks jetson -> cuda -> torchvision -> pil
print(dec)

# 1) Simplest and stream-safe: a regular, contiguous torch tensor.
rgb = dec.decode_to_torch(path, layout="CHW")  # (3, H, W) uint8 on cuda
print("decoded by:", dec.last_backend, tuple(rgb.shape), rgb.device)

x = rgb.unsqueeze(0).float().div_(255.0)  # (1, 3, H, W) float32 in [0, 1]
print("model input:", tuple(x.shape), x.dtype)

# 2) Zero-copy: keep the decoder's own buffer and view it as a tensor.
img = dec.decode(path)  # GpuImage (or a tensor if a Python fallback handled it)
view = njb.to_torch(img)  # shares memory via DLPack; keeps `img` alive by itself
print("zero-copy view:", tuple(view.shape), "strides", view.stride())
