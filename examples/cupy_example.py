"""Zero-copy hand-off to CuPy (no PyTorch needed for the native backends)."""

import sys

import cupy as cp

import nvjpeg_bridge as njb

dec = njb.Decoder(backend=("jetson", "cuda"))
img = dec.decode(sys.argv[1] if len(sys.argv) > 1 else "test.jpg")
arr = cp.from_dlpack(img)  # or cp.asarray(img) via __cuda_array_interface__
print(arr.shape, arr.dtype, float(arr.astype(cp.float32).mean()))
