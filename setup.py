"""Build script for nvjpeg-bridge.

Detects what this machine can build and compiles up to two native backends:

  * ``nvjpeg_bridge._jetson`` -- Jetson NVJPG + VIC (JetPack 5, 6 and 7, Multimedia API)
  * ``nvjpeg_bridge._cuda``   -- CUDA nvJPEG (any NVIDIA GPU with a CUDA toolkit; Linux and Windows)

Neither needs nvcc. If nothing can be built, the pure-Python package still
installs and falls back to torchvision / Pillow at runtime.

Environment variables
---------------------
NVJPEG_BRIDGE_BACKENDS   "auto" (default), "none", or a comma list such as
                         "jetson,cuda". Listed backends become mandatory:
                         the build fails loudly instead of skipping them.
CUDA_HOME / CUDA_PATH    CUDA toolkit root (default: from nvcc, /usr/local/cuda; on Windows
                         the installer's CUDA_PATH / CUDA_PATH_V*).
NVJPEG_HOME              Directory containing include/nvjpeg.h and lib*/libnvjpeg*.
JETSON_MMAPI             Multimedia API root (default /usr/src/jetson_multimedia_api).
TEGRA_LIB_DIR            Tegra driver libs (default: .../tegra or .../nvidia).
NVJPEG_BRIDGE_FORCE_JETSON=1   Treat the machine as a Jetson (cross builds, CI).
"""

import glob
import os
import platform
import shutil
import sys

import importlib.util

from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = sys.platform == "win32"


def _load_platform():
    """Import src/nvjpeg_bridge/_platform.py without importing the package."""
    path = os.path.join(HERE, "src", "nvjpeg_bridge", "_platform.py")
    spec = importlib.util.spec_from_file_location("_njb_platform", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLATFORM = _load_platform()


def log(msg):
    print("[nvjpeg-bridge] " + msg, file=sys.stderr)


# --------------------------------------------------------------- helpers ---


def _exists(*parts):
    return os.path.exists(os.path.join(*parts))


def _uniq(items):
    out = []
    for i in items:
        if i and i not in out:
            out.append(i)
    return out


def find_header(name, dirs):
    for d in dirs:
        if _exists(d, name):
            return d
    return None


def find_lib(name, dirs, prefer_static=False):
    """Return (dir, linker_name) for a library.

    Linux: lib<name>.so / lib<name>.so.N / lib<name>_static.a, e.g. ("/x", ":libnvjpeg.so.12").
    Windows: <name>.lib / <name>_static.lib (import libraries from the CUDA toolkit).
    """
    for d in dirs:
        if IS_WINDOWS:
            if prefer_static and _exists(d, "%s_static.lib" % name):
                return d, name + "_static"
            if _exists(d, "%s.lib" % name):
                return d, name
            continue
        if prefer_static and _exists(d, "lib%s_static.a" % name):
            return d, name + "_static"
        if _exists(d, "lib%s.so" % name):
            return d, name
        versioned = glob.glob(os.path.join(d, "lib%s.so.*" % name))
        if versioned:
            # Prefer the SONAME-like file (fewest dots): libnvjpeg.so.12 over .so.12.4.0
            best = sorted(versioned, key=lambda p: (os.path.basename(p).count("."), p))[0]
            return d, ":" + os.path.basename(best)
    return None, None


def pybind11_include():
    try:
        import pybind11
    except ImportError:
        sys.exit("[nvjpeg-bridge] pybind11 is required to build (uv pip install pybind11)")
    return pybind11.get_include()


# ----------------------------------------------------------------- CUDA ---


def cuda_candidates():
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    # The Windows installer also sets CUDA_PATH_V12_6, CUDA_PATH_V13_0, ... (newest first).
    roots += [os.environ[k] for k in sorted((k for k in os.environ if k.upper().startswith("CUDA_PATH_V")),
                                            reverse=True)]
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    roots += ["/usr/local/cuda", "/opt/cuda", "/usr"]
    return _uniq(roots)


def cuda_include_dirs(root):
    """Toolkit include dirs, including the targets/ layout (SBSA on JetPack 7 / ARM servers)."""
    arch = platform.machine()
    sub = ["include", "targets/%s-linux/include" % arch, "targets/sbsa-linux/include"]
    return [os.path.join(root, s) for s in sub if os.path.isdir(os.path.join(root, s))]


def cuda_lib_dirs(root):
    arch = platform.machine()
    if IS_WINDOWS:
        sub = ["lib/x64", "lib"]
    else:
        sub = ["lib64", "lib", "targets/%s-linux/lib" % arch, "targets/sbsa-linux/lib",
               "lib/%s-linux-gnu" % arch]
    return [os.path.join(root, s) for s in sub if os.path.isdir(os.path.join(root, s))]


def pip_wheel_cuda_dirs():
    """CUDA pieces shipped as pip wheels. Only visible with `uv pip install --no-build-isolation`.

    CUDA 12 wheels use one directory per package (nvidia/cuda_runtime, nvidia/nvjpeg,
    nvidia/cuda_nvcc); CUDA 13 wheels share one merged tree (nvidia/cu13).
    """
    incs, libs = [], []
    for p in sys.path:
        base = os.path.join(p, "nvidia")
        if not os.path.isdir(base):
            continue
        roots = [os.path.join(base, pkg) for pkg in ("cuda_runtime", "cuda_nvcc", "cuda_crt", "nvjpeg")]
        roots += sorted(glob.glob(os.path.join(base, "cu[0-9]*")), reverse=True)  # cu13, ...
        for root in roots:
            if os.path.isdir(os.path.join(root, "include")):
                incs.append(os.path.join(root, "include"))
            for sub in ("lib", os.path.join("lib", "x64")):
                if os.path.isdir(os.path.join(root, sub)):
                    libs.append(os.path.join(root, sub))
    return _uniq(incs), _uniq(libs)


def detect_cuda():
    inc_dirs, lib_dirs = [], []
    nvjpeg_home = os.environ.get("NVJPEG_HOME")
    if nvjpeg_home:
        inc_dirs += cuda_include_dirs(nvjpeg_home)
        lib_dirs += cuda_lib_dirs(nvjpeg_home)
    for root in cuda_candidates():
        incs = cuda_include_dirs(root)
        if find_header("cuda_runtime_api.h", incs):
            inc_dirs += incs
            lib_dirs += cuda_lib_dirs(root)
            break
    w_inc, w_lib = pip_wheel_cuda_dirs()
    inc_dirs = _uniq(inc_dirs + w_inc)
    lib_dirs = _uniq(lib_dirs + w_lib)

    missing = []
    if not find_header("cuda_runtime_api.h", inc_dirs):
        missing.append("cuda_runtime_api.h")
    if not find_header(os.path.join("crt", "host_defines.h"), inc_dirs):
        missing.append("crt/host_defines.h")
    if not find_header("nvjpeg.h", inc_dirs):
        missing.append("nvjpeg.h")
    # Linux: static cudart avoids clashing with torch's copy. Windows: link cudart.lib
    # (cudart64_NN.dll), the way PyTorch extensions do, to avoid CRT mismatches.
    cudart_dir, cudart = find_lib("cudart", lib_dirs, prefer_static=not IS_WINDOWS)
    if not cudart:
        missing.append("cudart.lib" if IS_WINDOWS else "libcudart")
    if missing:
        hint = ""
        if IS_WINDOWS:
            hint = ("; install the CUDA Toolkit (pip CUDA wheels for Windows ship DLLs but no .lib "
                    "import libraries, so they cannot be linked against)")
        return None, "CUDA toolkit not usable (missing: %s)%s" % (", ".join(missing), hint)
    return {
        "include_dirs": inc_dirs,
        "lib_dirs": lib_dirs,
        "cudart": (cudart_dir, cudart),
    }, "CUDA headers in %s" % find_header("cuda_runtime_api.h", inc_dirs)


def cudart_link(cuda):
    d, name = cuda["cudart"]
    libs = [name]
    if name.endswith("_static") and not IS_WINDOWS:
        libs += ["dl", "rt", "pthread"]  # static cudart avoids clashing with torch's cudart
    return d, libs


def rpath_args(dirs):
    # --disable-new-dtags emits DT_RPATH, which (unlike DT_RUNPATH) takes priority
    # over LD_LIBRARY_PATH. This pins the intended libnvjpeg: the Tegra libjpeg-API
    # library and CUDA's nvJPEG share the "libnvjpeg" name.
    if IS_WINDOWS:
        return []  # DLLs are located at import time (os.add_dll_directory), see _backends.py
    return ["-Wl,--disable-new-dtags"] + ["-Wl,-rpath," + d for d in _uniq(dirs)]


def cxx_flags():
    if IS_WINDOWS:
        # MSVC. /utf-8 keeps sources independent of the system code page (e.g. cp932).
        return ["/std:c++17", "/utf-8", "/EHsc", "/W3", "/D_CRT_SECURE_NO_WARNINGS"]
    return ["-O3", "-std=c++17", "-fvisibility=hidden", "-Wall", "-Wno-unused-parameter"]


def cuda_extension(cuda, mandatory):
    nvjpeg_dir, nvjpeg = find_lib("nvjpeg", cuda["lib_dirs"])
    if not nvjpeg:
        what = "nvjpeg.lib" if IS_WINDOWS else "libnvjpeg (CUDA)"
        return None, "%s not found in %s" % (what, cuda["lib_dirs"])
    cudart_dir, cudart_libs = cudart_link(cuda)
    ext = Extension(
        "nvjpeg_bridge._cuda",
        sources=["csrc/cuda/cuda_backend.cpp"],
        include_dirs=cuda["include_dirs"] + [pybind11_include()],
        library_dirs=_uniq([nvjpeg_dir, cudart_dir]),
        libraries=[nvjpeg] + cudart_libs,
        extra_compile_args=cxx_flags(),
        extra_link_args=rpath_args([nvjpeg_dir, cudart_dir]),
        language="c++",
        optional=not mandatory,
    )
    return ext, "nvJPEG: %s/%s, cudart: %s" % (nvjpeg_dir, nvjpeg, " ".join(cudart_libs))


# --------------------------------------------------------------- Jetson ---


def is_jetson():
    if os.environ.get("NVJPEG_BRIDGE_FORCE_JETSON") == "1":
        return True
    if platform.machine() != "aarch64":
        return False
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            return b"nvidia,tegra" in f.read()
    except OSError:
        return False


def find_tegra_jpeg_include(inc):
    """The Tegra libjpeg headers: include/libjpeg-8b on JetPack 5-7, any libjpeg* dir as fallback."""
    preferred = os.path.join(inc, "libjpeg-8b")
    if _exists(preferred, "jpeglib.h"):
        return preferred
    for d in sorted(glob.glob(os.path.join(inc, "libjpeg*"))):
        if _exists(d, "jpeglib.h"):
            return d
    return None


def jetson_extension(cuda, mandatory):
    info = PLATFORM.jetson_info()
    if info is not None and not info.supported:
        return None, ("%s: JetPack 4 uses the older nvbuf_utils API, which is not supported "
                      "(the 'cuda' backend may still work)" % info.describe())
    mmapi = os.environ.get("JETSON_MMAPI", "/usr/src/jetson_multimedia_api")
    inc = os.path.join(mmapi, "include")
    jpeg_inc = find_tegra_jpeg_include(inc)
    tegra_dirs = _uniq([
        os.environ.get("TEGRA_LIB_DIR"),
        "/usr/lib/aarch64-linux-gnu/tegra",   # JetPack 5
        "/usr/lib/aarch64-linux-gnu/nvidia",  # JetPack 6 and 7
    ])
    tegra_dirs = [d for d in tegra_dirs if os.path.isdir(d)]

    problems = []
    if not _exists(inc, "nvbufsurface.h") or not _exists(inc, "nvbufsurftransform.h"):
        problems.append("Multimedia API with NvBufSurface (JetPack 5+) not found at %s "
                        "(sudo apt install nvidia-l4t-jetson-multimedia-api)" % mmapi)
    if jpeg_inc is None:
        problems.append("Tegra libjpeg headers (%s/libjpeg-8b/jpeglib.h) missing" % inc)
    egl_dir = find_header(os.path.join("EGL", "egl.h"), [inc, "/usr/include"])
    if not egl_dir:
        problems.append("EGL headers missing (sudo apt install libegl-dev)")

    libs, lib_dirs = [], []
    for name in ("nvjpeg", "nvbufsurface", "nvbufsurftransform"):
        d, link = find_lib(name, tegra_dirs)
        if not link:
            problems.append("lib%s not found in %s" % (name, tegra_dirs or "Tegra lib dirs"))
        else:
            libs.append(link)
            lib_dirs.append(d)
    if problems:
        return None, "; ".join(problems)

    cudart_dir, cudart_libs = cudart_link(cuda)
    ext = Extension(
        "nvjpeg_bridge._jetson",
        sources=["csrc/jetson/jetson_backend.cpp", "csrc/jetson/tegra_jpeg.cpp"],
        # libjpeg-8b MUST come first: its jconfig.h defines TEGRA_ACCELERATE.
        include_dirs=_uniq([jpeg_inc, inc] + ([egl_dir] if egl_dir != "/usr/include" else [])
                           + cuda["include_dirs"] + [pybind11_include()]),
        # Tegra dirs first so the Tegra libnvjpeg wins over CUDA's at link time.
        library_dirs=_uniq(lib_dirs + [cudart_dir]),
        libraries=libs + cudart_libs,
        extra_compile_args=cxx_flags(),
        extra_link_args=rpath_args(lib_dirs + [cudart_dir]),
        language="c++",
        optional=not mandatory,
    )
    where = info.describe() if info else "Jetson (L4T release unknown)"
    return ext, "%s; MMAPI: %s; Tegra libs: %s" % (where, mmapi, _uniq(lib_dirs))


# ----------------------------------------------------------------- main ---


def select_extensions():
    request = os.environ.get("NVJPEG_BRIDGE_BACKENDS", "auto").strip().lower()
    if request in ("none", "0", "off"):
        log("native backends disabled by NVJPEG_BRIDGE_BACKENDS=none")
        return []
    wanted = {"jetson", "cuda"} if request == "auto" else {s.strip() for s in request.split(",") if s.strip()}
    unknown = wanted - {"jetson", "cuda"}
    if unknown:
        sys.exit("[nvjpeg-bridge] unknown backend(s) in NVJPEG_BRIDGE_BACKENDS: %s" % ", ".join(sorted(unknown)))
    mandatory = request != "auto"

    exts, report = [], []

    def fail_or_skip(name, why):
        if mandatory and name in wanted:
            sys.exit("[nvjpeg-bridge] cannot build required backend '%s': %s" % (name, why))
        report.append("  %-7s SKIPPED  (%s)" % (name, why))

    cuda, cuda_msg = detect_cuda()
    if cuda is None:
        for name in sorted(wanted):
            fail_or_skip(name, cuda_msg)
    else:
        if "jetson" in wanted:
            if is_jetson():
                ext, msg = jetson_extension(cuda, mandatory)
                if ext:
                    exts.append(ext)
                    report.append("  jetson  BUILD    (%s)" % msg)
                else:
                    fail_or_skip("jetson", msg)
            else:
                fail_or_skip("jetson", "not a Jetson device")
        if "cuda" in wanted:
            ext, msg = cuda_extension(cuda, mandatory)
            if ext:
                exts.append(ext)
                report.append("  cuda    BUILD    (%s)" % msg)
            else:
                fail_or_skip("cuda", msg)

    log("native backend plan (%s):" % cuda_msg)
    for line in report:
        log(line)
    if not exts:
        log("no native backend will be built; runtime will use torchvision/Pillow fallbacks")
    return exts


if __name__ == "__main__":  # setuptools' PEP 517 backend also runs this file as __main__
    setup(ext_modules=select_extensions())
