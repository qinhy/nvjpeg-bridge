"""Environment report: `python -m nvjpeg_bridge` or `nvjpeg-bridge-info`."""

from __future__ import annotations

import os
import platform
import sys

from ._backends import (AUTO_ORDER, REGISTRY, BackendUnavailable, load_native, native_module_built,
                        windows_cuda_dll_dirs)
from ._platform import jetson_info
from ._version import __version__


def available_backends(device: int = 0) -> dict:
    """Try to start every backend. Returns {name: (usable: bool, detail: str)}."""
    out = {}
    for name in AUTO_ORDER:
        try:
            REGISTRY[name](device=device)
            out[name] = (True, "ok")
        except BackendUnavailable as e:
            out[name] = (False, str(e))
        except Exception as e:  # unexpected, but keep reporting the rest
            out[name] = (False, "%s: %s" % (type(e).__name__, e))
    return out


def report(device: int = 0) -> str:
    lines = ["nvjpeg-bridge %s" % __version__,
             "python      %s (%s)" % (sys.version.split()[0], platform.machine())]
    info = jetson_info()
    lines.append("jetson      %s" % (info.describe() if info else "no"))
    if sys.platform == "win32":
        dirs = windows_cuda_dll_dirs()
        lines.append("cuda dlls   %s" % ("; ".join(dirs) if dirs else "no CUDA bin directory found (set CUDA_PATH)"))
    try:
        import torch

        cuda = torch.version.cuda if torch.cuda.is_available() else "unavailable"
        lines.append("torch       %s (cuda %s)" % (torch.__version__, cuda))
        if torch.cuda.is_available():
            lines.append("gpu         %s" % torch.cuda.get_device_name(device))
    except ImportError:
        lines.append("torch       not installed")
    for name in ("jetson", "cuda"):
        built = native_module_built(name)
        extra = ""
        if built:
            try:
                # Same loader as the backends: on Windows it registers the CUDA DLL directories first.
                mod = load_native(name)
                extra = " (cudart %s" % getattr(mod, "cuda_runtime_version", "?")
                if hasattr(mod, "nvjpeg_version"):
                    extra += ", nvjpeg %s" % ".".join(map(str, mod.nvjpeg_version))
                extra += ")"
            except BackendUnavailable as e:
                extra = " (import failed: %s)" % e
        lines.append("native %-6s %s%s" % (name, "built" if built else "not built", extra))
    lines.append("")
    lines.append("backend availability (auto order):")
    for name, (ok, detail) in available_backends(device).items():
        lines.append("  %-12s %s  %s" % (name, "YES" if ok else "no ", "" if ok else detail))
    return "\n".join(lines)


def main(argv=None):
    device = int(os.environ.get("NVJPEG_BRIDGE_DEVICE", "0"))
    print(report(device))
    return 0
