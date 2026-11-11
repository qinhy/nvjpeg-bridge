"""nvjpeg_bridge: decode JPEG images straight into CUDA memory.

Backends (picked automatically, best first):
  jetson      NVJPG hardware decoder + VIC colour conversion (Jetson, JetPack 5, 6, 7)
  cuda        CUDA nvJPEG (any NVIDIA GPU, incl. Jetson's iGPU)
  torchvision torchvision.io.decode_jpeg(device="cuda")
  pil         Pillow on the CPU + upload

Quick start::

    import nvjpeg_bridge as njb
    dec = njb.Decoder()                       # best available backend
    t = dec.decode_to_torch("photo.jpg")      # torch.uint8 (H, W, 3) on cuda
    img = dec.decode(jpeg_bytes)              # zero-copy GpuImage
    t2 = njb.to_torch(img)                    # shares memory via DLPack
"""

from ._backends import AUTO_ORDER, BackendUnavailable
from ._decoder import Decoder, JpegDecodeError, decode, decode_to_torch, to_numpy, to_torch
from ._env import available_backends, report
from ._header import JpegHeader, read_header
from ._platform import JetsonInfo, jetson_info
from ._version import __version__

__all__ = [
    "AUTO_ORDER",
    "BackendUnavailable",
    "Decoder",
    "JpegDecodeError",
    "JetsonInfo",
    "JpegHeader",
    "available_backends",
    "decode",
    "decode_to_torch",
    "jetson_info",
    "read_header",
    "report",
    "to_numpy",
    "to_torch",
    "__version__",
]
