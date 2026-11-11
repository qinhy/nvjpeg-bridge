"""High-level, backend-agnostic JPEG -> GPU decoder."""

from __future__ import annotations

import os
import threading
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ._backends import AUTO_ORDER, REGISTRY, Backend, BackendUnavailable


class JpegDecodeError(RuntimeError):
    """Every backend that was tried failed to decode the image."""

    def __init__(self, attempts: List[Tuple[str, BaseException]]):
        self.attempts = attempts
        if attempts:
            detail = "; ".join("%s: %s" % (name, err) for name, err in attempts)
        else:
            detail = "no backend available"
        super().__init__("JPEG decode failed (%s)" % detail)


Source = Union[bytes, bytearray, memoryview, str, "os.PathLike[str]", object]


def _norm_layout(layout: str) -> str:
    lay = str(layout).upper()
    if lay not in ("HWC", "CHW"):
        raise ValueError("layout must be 'HWC' or 'CHW', got %r" % (layout,))
    return lay


def as_buffer(src):
    """Turn a path or any bytes-like object into a contiguous byte buffer (no copy if possible)."""
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as f:
            return f.read()
    if isinstance(src, (bytes, bytearray)):
        return src
    if hasattr(src, "is_cuda") and hasattr(src, "numpy"):  # torch tensor holding encoded bytes
        if src.is_cuda:
            raise TypeError("encoded JPEG bytes must be in host memory, not a CUDA tensor")
        src = src.contiguous().numpy()
    try:
        mv = memoryview(src)
    except TypeError:
        raise TypeError("expected a path or a bytes-like object, got %s" % type(src).__name__) from None
    if not mv.c_contiguous:
        return mv.tobytes()
    return mv if (mv.ndim == 1 and mv.format == "B") else mv.cast("B")


class Decoder:
    """Decode JPEG images straight into GPU memory.

    Parameters
    ----------
    backend:
        ``"auto"`` (default) picks the best available in the order
        jetson -> cuda -> torchvision -> pil. A name (``"cuda"``) or a sequence
        of names sets the preferred order explicitly.
    device:
        CUDA device index (Jetson: always 0).
    layout:
        Default output layout: ``"HWC"`` (H, W, 3) or ``"CHW"`` (3, H, W). RGB, uint8.
    fallback:
        If True, an image the preferred backend cannot handle (progressive JPEG on
        NVJPG, CMYK on nvJPEG, ...) is retried on the next available backend.
    pool_size:
        Jetson only: number of idle RGBA surfaces kept for reuse.
    sync_after_map:
        Jetson only: add a device sync after each EGL->CUDA mapping (debug aid if
        you ever observe stale frames).
    """

    def __init__(
        self,
        backend: Union[str, Sequence[str]] = "auto",
        device: int = 0,
        layout: str = "HWC",
        fallback: bool = True,
        pool_size: int = 4,
        sync_after_map: bool = False,
    ):
        self.device = int(device)
        self.layout = _norm_layout(layout)
        self.fallback = bool(fallback)
        self._opts = {"device": self.device, "pool_size": pool_size, "sync_after_map": sync_after_map}

        if isinstance(backend, str):
            backend = backend.lower()
            if backend == "auto":
                order = list(AUTO_ORDER)
            else:
                order = [backend] + [b for b in AUTO_ORDER if b != backend]
        else:
            order = [str(b).lower() for b in backend]
        unknown = [b for b in order if b not in REGISTRY]
        if unknown:
            raise ValueError("unknown backend(s): %s (choose from %s)" % (unknown, list(REGISTRY)))
        explicit = not (isinstance(backend, str) and backend == "auto")

        self._order = order
        self._instances = {}
        self._init_errors = {}
        self._lock = threading.Lock()
        self.last_backend: Optional[str] = None

        self._primary = None
        for name in order:
            if self._get(name) is not None:
                self._primary = name
                break
            if explicit and name == order[0] and not self.fallback:
                raise BackendUnavailable("backend %r unavailable: %s" % (name, self._init_errors[name]))
        if self._primary is None:
            raise BackendUnavailable(
                "no usable backend: "
                + "; ".join("%s: %s" % (n, self._init_errors.get(n, "?")) for n in order)
            )

    # ----------------------------------------------------------- internals ---

    def _get(self, name: str) -> Optional[Backend]:
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            try:
                be = REGISTRY[name](**self._opts)
            except BackendUnavailable as e:
                self._init_errors[name] = e
                be = None
            self._instances[name] = be
            return be

    def _candidates(self):
        yield self._primary
        if self.fallback:
            for name in self._order:
                if name != self._primary:
                    yield name

    def _run(self, op: str, src, layout, **kwargs):
        data = as_buffer(src)
        lay = self.layout if layout is None else _norm_layout(layout)
        attempts = []
        for name in self._candidates():
            be = self._get(name)
            if be is None:
                continue
            try:
                result = getattr(be, op)(data, lay, **kwargs)
            except be.errors as e:
                attempts.append((name, e))
                continue
            self.last_backend = name
            return result
        raise JpegDecodeError(attempts)

    # -------------------------------------------------------------- public ---

    @property
    def backend(self) -> str:
        """Name of the preferred backend actually in use."""
        return self._primary

    def decode(self, src: Source, layout: Optional[str] = None):
        """Decode one JPEG. Returns a GPU image supporting ``__cuda_array_interface__``
        and DLPack (native backends return a zero-copy ``GpuImage``; the
        torchvision/pil fallbacks return a ``torch.Tensor``)."""
        return self._run("decode", src, layout)

    def decode_to_torch(self, src: Source, layout: Optional[str] = None, out=None):
        """Decode one JPEG into a contiguous ``torch.uint8`` CUDA tensor.

        Stream-safe: work is ordered on ``torch.cuda.current_stream()``. Pass ``out``
        to decode into a preallocated tensor of the right shape.
        """
        return self._run("decode_to_torch", src, layout, out=out)

    def decode_batch(self, srcs: Iterable[Source], layout: Optional[str] = None, to_torch: bool = False) -> list:
        """Decode several JPEGs (sequentially; sizes may differ)."""
        op = self.decode_to_torch if to_torch else self.decode
        return [op(s, layout) for s in srcs]

    def close(self) -> None:
        """Drop native decoder objects. Images already returned stay valid."""
        with self._lock:
            self._instances.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "<nvjpeg_bridge.Decoder backend=%s device=%d layout=%s fallback=%s>" % (
            self._primary, self.device, self.layout, self.fallback)


# ------------------------------------------------------------- utilities ---


def to_torch(img):
    """Zero-copy wrap of a decoded image as a torch tensor (shares memory)."""
    import torch

    if isinstance(img, torch.Tensor):
        return img
    return torch.from_dlpack(img)


def to_numpy(img):
    """Copy a decoded image to host memory as a writable numpy array."""
    import numpy as np

    if hasattr(img, "_copy_span_to_host"):
        raw = img._copy_span_to_host()
        view = np.ndarray(shape=img.shape, dtype=np.uint8, buffer=raw, strides=img.strides)
        return np.array(view, copy=True)
    if hasattr(img, "cpu"):  # torch tensor
        return img.detach().cpu().numpy()
    raise TypeError("cannot convert %s to numpy" % type(img).__name__)


_default_lock = threading.Lock()
_default: Optional[Decoder] = None


def _default_decoder() -> Decoder:
    global _default
    with _default_lock:
        if _default is None:
            _default = Decoder()
        return _default


def decode(src: Source, layout: str = "HWC"):
    """Convenience: decode with a shared, lazily created ``Decoder()``."""
    return _default_decoder().decode(src, layout)


def decode_to_torch(src: Source, layout: str = "HWC", out=None):
    """Convenience: ``Decoder().decode_to_torch`` with a shared decoder."""
    return _default_decoder().decode_to_torch(src, layout, out=out)
