"""Backend adapters. Every backend exposes the same two operations:

decode(data, layout)               -> object with __cuda_array_interface__ / __dlpack__
decode_to_torch(data, layout, out) -> torch.Tensor (uint8, contiguous, on the GPU)

``errors`` lists the exception types that mean "this backend could not decode
this image; another backend may"; the Decoder falls back on exactly those.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import io
import os
import sys
import warnings


class BackendUnavailable(RuntimeError):
    """A backend cannot be used on this machine (not built, no GPU, missing package)."""


AUTO_ORDER = ("jetson", "cuda", "torchvision", "pil")


def _torch():
    try:
        import torch
    except ImportError as e:
        raise BackendUnavailable("PyTorch is not installed") from e
    return torch


def shape_for(width: int, height: int, layout: str):
    return (height, width, 3) if layout == "HWC" else (3, height, width)


def check_out(out, shape, device_index, torch):
    problems = []
    if not isinstance(out, torch.Tensor):
        problems.append("must be a torch.Tensor")
    else:
        if out.dtype != torch.uint8:
            problems.append("dtype must be torch.uint8 (got %s)" % out.dtype)
        if tuple(out.shape) != tuple(shape):
            problems.append("shape must be %s (got %s)" % (tuple(shape), tuple(out.shape)))
        if not out.is_cuda or (out.device.index or 0) != device_index:
            problems.append("must be on cuda:%d (got %s)" % (device_index, out.device))
        if not out.is_contiguous():
            problems.append("must be contiguous")
    if problems:
        raise TypeError("invalid `out` tensor: " + "; ".join(problems))


def native_module_built(name: str) -> bool:
    return importlib.util.find_spec("nvjpeg_bridge._" + name) is not None


class Backend:
    name = "?"
    errors: tuple = ()
    device = 0

    def decode(self, data, layout):  # pragma: no cover - interface
        raise NotImplementedError

    def decode_to_torch(self, data, layout, out=None):  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------ native ---


_dll_handles = []  # keep os.add_dll_directory() registrations alive


def windows_cuda_dll_dirs():
    """Directories that may hold cudart64_NN.dll / nvjpeg64_NN.dll on Windows."""
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    roots += [os.environ[k] for k in sorted((k for k in os.environ if k.upper().startswith("CUDA_PATH_V")),
                                            reverse=True)]
    dirs = []
    for r in roots:
        if r:
            dirs += [os.path.join(r, "bin"), os.path.join(r, "bin", "x64"), os.path.join(r, "bin", "x86_64")]
    for p in sys.path:  # CUDA pip wheels: nvidia/<pkg>/bin (CUDA 12), nvidia/cu13/bin/x86_64 (CUDA 13)
        dirs += glob.glob(os.path.join(p, "nvidia", "*", "bin"))
        dirs += glob.glob(os.path.join(p, "nvidia", "cu*", "bin", "x*"))
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _prepare_windows_dll_search():
    # Since Python 3.8, PATH is not searched for an extension's DLL dependencies.
    if sys.platform != "win32" or _dll_handles or not hasattr(os, "add_dll_directory"):
        return
    for d in windows_cuda_dll_dirs():
        try:
            _dll_handles.append(os.add_dll_directory(d))
        except OSError:
            pass


def load_native(name):
    """Import a native backend module; always use this instead of a plain import, because on
    Windows the CUDA DLL directories must be registered first."""
    if not native_module_built(name):
        if name == "jetson":
            from ._platform import jetson_info

            if jetson_info() is None:
                raise BackendUnavailable("not a Jetson device")
        raise BackendUnavailable("native '%s' backend was not built for this install "
                                 "(reinstall with `uv pip install -v .` to see the build plan)" % name)
    _prepare_windows_dll_search()
    try:
        return importlib.import_module("nvjpeg_bridge._" + name)
    except ImportError as e:  # e.g. "DLL load failed" on Windows
        hint = ""
        if sys.platform == "win32":
            hint = (" -- the CUDA DLLs (cudart64_NN.dll, nvjpeg64_NN.dll) were not found; make sure "
                    "CUDA_PATH points to the toolkit used for the build. Searched: %s"
                    % (windows_cuda_dll_dirs() or "nothing"))
        raise BackendUnavailable("native '%s' backend failed to load: %s%s" % (name, e, hint)) from e


_import_native = load_native  # backwards-compatible alias


class JetsonBackend(Backend):
    """NVJPG hardware decode + VIC colour conversion, zero-copy into CUDA."""

    name = "jetson"

    def __init__(self, device=0, pool_size=4, sync_after_map=False, **_):
        if device != 0:
            raise BackendUnavailable("Jetson has a single integrated GPU (device 0)")
        mod = load_native("jetson")
        self.errors = (mod.UnsupportedJpegError, mod.DecodeError)
        try:
            self._dec = mod.JetsonDecoder(int(pool_size), bool(sync_after_map))
        except Exception as e:  # no GPU, driver problem, ...
            raise BackendUnavailable("Jetson backend could not start: %s" % e) from e
        self.device = 0

    def decode(self, data, layout):
        return self._dec.decode(data, layout)

    def decode_to_torch(self, data, layout, out=None):
        torch = _torch()
        img = self._dec.decode(data, layout)
        view = torch.from_dlpack(img)  # zero-copy, strided RGB view of the RGBA surface
        if out is None:
            out = torch.empty_like(view, memory_format=torch.contiguous_format)
        else:
            check_out(out, tuple(view.shape), 0, torch)
        out.copy_(view)  # one device copy, enqueued on torch's current stream
        stream = torch.cuda.current_stream(out.device).cuda_stream
        del view
        # Surface returns to the pool; it is reused only after `stream` passes the copy.
        img.release(stream)
        return out


class CudaBackend(Backend):
    """CUDA nvJPEG: any NVIDIA GPU, including Jetson's iGPU."""

    name = "cuda"

    def __init__(self, device=0, **_):
        mod = load_native("cuda")
        self.errors = (mod.UnsupportedJpegError, mod.DecodeError)
        try:
            self._dec = mod.CudaDecoder(int(device))
        except Exception as e:
            raise BackendUnavailable("CUDA nvJPEG backend could not start: %s" % e) from e
        self.device = int(device)

    def decode(self, data, layout):
        return self._dec.decode(data, layout)

    def decode_to_torch(self, data, layout, out=None):
        torch = _torch()
        w, h, _, _ = self._dec.info(data)
        shape = shape_for(w, h, layout)
        dev = torch.device("cuda", self.device)
        if out is None:
            out = torch.empty(shape, dtype=torch.uint8, device=dev)
        else:
            check_out(out, shape, self.device, torch)
        stream = torch.cuda.current_stream(dev).cuda_stream
        # Decodes straight into the tensor's memory: no intermediate buffer.
        self._dec.decode_into(data, out.data_ptr(), w, h, layout, stream)
        return out


# ------------------------------------------------------- pure-Python fallbacks ---


class TorchvisionBackend(Backend):
    """torchvision.io.decode_jpeg on the GPU (needs a torchvision built with nvJPEG)."""

    name = "torchvision"
    errors = (RuntimeError, ValueError)

    def __init__(self, device=0, **_):
        torch = _torch()
        if not torch.cuda.is_available():
            raise BackendUnavailable("torch.cuda is not available")
        try:
            from torchvision.io import ImageReadMode, decode_jpeg
        except ImportError as e:
            raise BackendUnavailable("torchvision is not installed") from e
        self._torch = torch
        self._decode_jpeg = decode_jpeg
        self._mode = ImageReadMode.RGB
        self._dev = torch.device("cuda", int(device))
        self.device = int(device)

    def decode(self, data, layout):
        raw = self._torch.frombuffer(bytearray(data), dtype=self._torch.uint8)
        with warnings.catch_warnings():
            # torchvision >= 0.29 deprecates its image decoders (in favour of TorchCodec) and
            # warns on every call; this is only a fallback, so keep the user's logs clean.
            warnings.simplefilter("ignore", DeprecationWarning)
            chw = self._decode_jpeg(raw, mode=self._mode, device=self._dev)
        return chw.contiguous() if layout == "CHW" else chw.permute(1, 2, 0).contiguous()

    def decode_to_torch(self, data, layout, out=None):
        t = self.decode(data, layout)
        if out is None:
            return t
        check_out(out, tuple(t.shape), self.device, self._torch)
        out.copy_(t)
        return out


class PilBackend(Backend):
    """CPU decode with Pillow, then upload. Slowest, but decodes almost anything."""

    name = "pil"
    errors = (OSError, ValueError, RuntimeError)

    def __init__(self, device=0, **_):
        torch = _torch()
        if not torch.cuda.is_available():
            raise BackendUnavailable("torch.cuda is not available")
        try:
            import numpy as np
            from PIL import Image
        except ImportError as e:
            raise BackendUnavailable("Pillow and numpy are required for the 'pil' backend") from e
        self._torch, self._np, self._Image = torch, np, Image
        self._dev = torch.device("cuda", int(device))
        self.device = int(device)

    def decode(self, data, layout):
        with self._Image.open(io.BytesIO(bytes(data))) as im:
            arr = self._np.array(im.convert("RGB"))  # writable copy; EXIF orientation is NOT applied
        t = self._torch.from_numpy(arr).to(self._dev)
        return t if layout == "HWC" else t.permute(2, 0, 1).contiguous()

    def decode_to_torch(self, data, layout, out=None):
        t = self.decode(data, layout)
        if out is None:
            return t
        check_out(out, tuple(t.shape), self.device, self._torch)
        out.copy_(t)
        return out


REGISTRY = {
    "jetson": JetsonBackend,
    "cuda": CudaBackend,
    "torchvision": TorchvisionBackend,
    "pil": PilBackend,
}
