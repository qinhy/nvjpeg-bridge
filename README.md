# nvjpeg-bridge

Decode JPEG images straight into CUDA memory from Python, on **Jetson** (JetPack 5, 6 and 7, including Thor) and on **any NVIDIA GPU** on Linux or Windows (CUDA 11, 12 and 13), and hand them to PyTorch, CuPy and other libraries without a copy.

```
Jetson (JetPack 5-7):  JPEG ─NVJPG─▶ YUV surface ─VIC─▶ RGBA surface (pooled) ─EGL─▶ CUDA ptr ─▶ GpuImage
Any NVIDIA GPU:        JPEG ─nvJPEG (CUDA)──────────────────────────────────────────▶ CUDA ptr ─▶ GpuImage
Fallbacks:             torchvision.io.decode_jpeg(device="cuda")  →  Pillow on CPU + upload
```

`GpuImage` implements both `__dlpack__` and `__cuda_array_interface__` (v3), so `torch.from_dlpack(img)` and `cupy.from_dlpack(img)` share its memory.

## Highlights

The package picks the best available backend for each machine (`jetson → cuda → torchvision → pil`). When a backend can't handle a particular image, such as a progressive JPEG on NVJPG or CMYK on nvJPEG, the next backend decodes that image instead. Both native backends are host-only C++, so you don't need nvcc to build them.

A corrupt JPEG raises a Python exception instead of killing your process. NVIDIA's sample `NvJPEGDecoder` uses libjpeg's default error handler, which calls `exit()`.

Memory handling is safe with respect to CUDA streams:

- `decode_to_torch()` orders its work on `torch.cuda.current_stream()`.
- Pooled Jetson surfaces are reused only after their consumers have finished.
- The CUDA runtime API is used throughout, so the backends share PyTorch's primary context and work from any thread. The GIL is released during decoding.

## Support matrix

| Platform | L4T | CUDA / Python | Backends built |
|---|---|---|---|
| Jetson AGX Xavier / Orin, JetPack 5.x | R35 | 11.4 / 3.8 | `jetson` + `cuda` |
| Jetson Orin, JetPack 6.x | R36 | 12.x / 3.10 | `jetson` + `cuda` |
| Jetson Thor, JetPack 7.0 / 7.1 | R38 | 13.0 / 3.12 | `jetson` + `cuda` |
| Jetson Orin and Thor, JetPack 7.2 | R39 | 13.2 / 3.12 | `jetson` + `cuda` |
| Linux x86_64 / ARM SBSA with a CUDA 11, 12 or 13 toolkit | — | any ≥ 3.8 | `cuda` |
| Windows x64 with a CUDA 11, 12 or 13 toolkit and MSVC | — | any ≥ 3.8 | `cuda` |
| Linux with CUDA 12 or 13 from pip wheels (no system toolkit) | — | any ≥ 3.8 | `cuda` |
| Anything else with PyTorch + CUDA | — | — | `torchvision` / `pil` fallbacks only |
| JetPack 4.x (`nvbuf_utils` API, Python 3.6) | R32 | — | Not supported |

`python -m nvjpeg_bridge` prints what it detected, for example `L4T R38.2.1 -> JetPack 7.0, Thor (tegra264)`. The same information is available from `njb.jetson_info()`.

### JetPack 7 notes

JetPack 7 moved Jetson to Ubuntu 24.04, Python 3.12 and CUDA 13. On Thor, the GPU uses a new driver stack (OpenRM), and CUDA uses the ARM-server layout (`/usr/local/cuda/targets/sbsa-linux`). JetPack 7.2 brought Orin onto the same stack. The decode pipeline is unchanged: NVJPG, then VIC, then `NvBufSurfaceMapEglImage`, then CUDA EGL interop. NVIDIA's own r38 CUDA samples use this same path.

The build handles the differences automatically:

- It searches the `targets/sbsa-linux` CUDA layout even without the `include` and `lib64` symlinks.
- It finds the Tegra libraries in `/usr/lib/aarch64-linux-gnu/nvidia`.
- It finds CUDA 13 pip wheels, which use one merged `nvidia/cu13` tree instead of one directory per package.

**Build on the device, for the JetPack release it runs.** JetPack 7 widened several Multimedia API structs, including `NvBufSurface`, `NvBufSurfaceParams` and the transform rectangle. A build made against JetPack 5 or 6 headers must not run on JetPack 7, or the other way round. After upgrading JetPack, reinstall with `uv pip install --reinstall -v .`.

For PyTorch on JetPack 7, use NVIDIA's Jetson wheels for your release. Because CUDA there is the standard SBSA build, official aarch64 CUDA 13 wheels can also work, provided they include your GPU (Orin is `sm_87`, Thor is `sm_110`).

## Install

[uv](https://docs.astral.sh/uv/) is the default tool for everything below. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (the Jetson script does this for you). Plain pip still works; see the end of this section.

### Jetson (JetPack 5, 6, 7)

```bash
git clone <this repo> nvjpeg-bridge && cd nvjpeg-bridge
./scripts/install_jetson.sh          # apt deps, uv, .venv, build, environment report
source .venv/bin/activate
python -m pytest -v tests/
```

The script checks the JetPack release first. It then creates `.venv` from the system Python (3.8, 3.10 or 3.12 depending on JetPack) with `--system-site-packages`. NVIDIA's PyTorch wheels for Jetson are built for that Python, and this way they stay visible inside the venv. By hand:

```bash
sudo apt-get install -y nvidia-l4t-jetson-multimedia-api libegl-dev python3-dev
uv venv --python /usr/bin/python3 --system-site-packages
NVJPEG_BRIDGE_BACKENDS=jetson,cuda uv pip install -v .
uv pip install pytest pillow         # not numpy: keep JetPack's
.venv/bin/python -m nvjpeg_bridge    # shows which backends work
```

On Jetson, use `uv pip` rather than `uv sync`. uv doesn't count packages from the system site-packages as installed, so `uv sync` (or `uv pip install numpy`) would put a second NumPy in the venv. That copy can shadow the one NVIDIA's PyTorch was built against and break `import torch`.

### Desktop / server GPU

You need the CUDA toolkit (it includes `nvjpeg.h` and `libnvjpeg`). In the repo:

```bash
uv sync                              # creates .venv, builds, installs dev tools
uv run python -m nvjpeg_bridge       # environment report
uv run pytest -v                     # GPU tests also need: uv pip install torch
```

If the toolkit isn't auto-detected, set `CUDA_HOME=/usr/local/cuda-12.x` for `uv sync` and keep it set for `uv run`.

`uv sync` is exact: it removes packages that aren't in `uv.lock`, including a PyTorch you added with `uv pip install torch`. Use `uv sync --inexact` in that case, or just `uv run`, which never removes packages.

To use the package from another project:

```bash
uv add /path/to/nvjpeg-bridge        # or: uv pip install /path/to/nvjpeg-bridge
```

### Windows

You need two things:

- **The NVIDIA CUDA Toolkit.** The installer sets `CUDA_PATH` and provides the `.lib` import libraries that the build links against.
- **The MSVC C++ build tools.** Install Visual Studio or "Build Tools for Visual Studio" with the "Desktop development with C++" workload. setuptools finds MSVC automatically, so no developer prompt is needed.

```powershell
uv sync                              # or: uv pip install -v .
uv run python -m nvjpeg_bridge       # should list "native cuda built" and a "cuda dlls" line
uv run pytest -v
```

The build links against `cudart.lib` and `nvjpeg.lib`. At import time, the package registers the toolkit's `bin` directory with `os.add_dll_directory`, which finds `cudart64_NN.dll` and `nvjpeg64_NN.dll`. Python 3.8+ no longer searches `PATH` for these. The pip-wheel route below is not available on Windows, because NVIDIA's Windows CUDA wheels contain DLLs but no `.lib` files to link against.

If your project and uv's cache are on different drives (for example `D:` and `C:`), uv warns that it can't hardlink and falls back to copying. That's harmless. To silence it, set `UV_LINK_MODE=copy`, or move the cache to the same drive with `UV_CACHE_DIR=D:\uv-cache`.

### Install without a CUDA toolkit (pip wheels, Linux)

```bash
uv venv
uv pip install pybind11 setuptools wheel nvidia-cuda-runtime-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-nvcc-cu12
uv pip install -v --no-build-isolation .
```

The `nvcc` wheel is used only for its headers; nothing is compiled with nvcc.

### Build options (environment variables)

| Variable | Meaning |
|---|---|
| `NVJPEG_BRIDGE_BACKENDS` | `auto` (default), `none`, or e.g. `jetson,cuda`. Listed backends are mandatory and fail the build loudly. |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit root |
| `NVJPEG_HOME` | Directory with `include/nvjpeg.h` and `lib*/libnvjpeg*` |
| `JETSON_MMAPI` | Multimedia API root (default `/usr/src/jetson_multimedia_api`) |
| `TEGRA_LIB_DIR` | Tegra driver libs (default `/usr/lib/aarch64-linux-gnu/tegra` or `.../nvidia`) |

With `auto`, a backend that can't be built is skipped with a reason. Use `uv pip install -v .` or `uv sync -v` to see the build plan.

All of these variables are uv `cache-keys` (see `pyproject.toml`), along with the C++ and Python sources. Changing any of them makes `uv sync` and `uv run` rebuild instead of reusing a cached wheel built under different settings. The flip side: set them the same way every time, for example by exporting them in your shell profile. Otherwise each `uv run` with different values triggers a rebuild. `uv run --no-sync` skips that check.

### Plain pip

Everything works with pip as well: `pip install -v .`, or `pip install -v --no-build-isolation .` for the wheel route. On Jetson, create the venv with `python3 -m venv --system-site-packages .venv`.

## Usage

```python
import torch
import nvjpeg_bridge as njb

dec = njb.Decoder()                                   # best backend for this machine

# 1) Easiest: a normal contiguous tensor, stream-safe.
x = dec.decode_to_torch("photo.jpg", layout="CHW")    # (3, H, W) uint8 on cuda
x = x.unsqueeze(0).float().div_(255)                  # (1, 3, H, W) model input

# 2) Decode into a preallocated tensor (no allocation per image).
out = torch.empty((3, 1080, 1920), dtype=torch.uint8, device="cuda")
dec.decode_to_torch(jpeg_bytes, layout="CHW", out=out)

# 3) Zero-copy: keep the decoder's buffer and view it.
img = dec.decode(jpeg_bytes)                          # GpuImage, RGB uint8
t = torch.from_dlpack(img)                            # shares memory; keeps img alive
```

Inputs can be a path, `bytes`, `bytearray`, `memoryview`, a numpy `uint8` array, or a CPU tensor of encoded bytes. Output is always RGB `uint8`, with the layout `"HWC"` (H, W, 3) or `"CHW"` (3, H, W).

On Jetson, the zero-copy view is a strided view of an RGBA surface, so it is not contiguous; its strides are `(pitch, 4, 1)`. `decode_to_torch()` gives you a contiguous copy, which is what most models want.

### Which one should I use?

| Goal | Use |
|---|---|
| Feed a PyTorch model | `decode_to_torch()`. On dGPUs it decodes directly into the tensor. On Jetson it does one fast device copy and returns the surface to the pool immediately. |
| Absolute minimum work, short-lived use | `decode()` and `torch.from_dlpack()` |
| CuPy / Numba / other frameworks | `decode()` and `cupy.from_dlpack(img)` or `cupy.asarray(img)` |
| Host copy for debugging | `njb.to_numpy(img)` |

### Threads

`Decoder` is thread-safe; calls on one decoder are serialized and release the GIL. For parallel decoding, give each worker thread its own `Decoder`.

## API

`Decoder(backend="auto", device=0, layout="HWC", fallback=True, pool_size=4, sync_after_map=False)`

- `backend` is `"auto"`, a name (`"jetson"`, `"cuda"`, `"torchvision"`, `"pil"`) to prefer, or a sequence giving the order.
- `fallback=True` retries images that a backend can't handle on the next backend. `.last_backend` tells you which backend handled the last call.
- `.decode(src, layout=None)` returns a `GpuImage`; the torchvision and pil fallbacks return a `torch.Tensor`.
- `.decode_to_torch(src, layout=None, out=None)` returns a `torch.Tensor`.
- `.decode_batch(srcs, layout=None, to_torch=False)` returns a list.
- `.backend` is the preferred backend in use, and `.close()` releases the decoder.

`GpuImage` has these members:

- `width`, `height`, `layout`, `shape`, `strides` (bytes), `ptr`, `device`, `is_contiguous`, `backend`
- `__dlpack__()` and `__dlpack_device__()`, plus `__cuda_array_interface__`
- `release(stream=None)` returns the memory early. If you pass a `cudaStream_t` (int) on which all readers are enqueued, the memory is reused only after that stream reaches that point. It refuses to release while a DLPack export is alive.

The package also provides these helpers:

- `njb.read_header(data)` parses the frame header with no decoding.
- `njb.to_torch(img)` and `njb.to_numpy(img)` convert results.
- `njb.available_backends()` and `njb.report()` describe the environment; `python -m nvjpeg_bridge` prints the report.

It raises `njb.JpegDecodeError` when every backend failed (it carries `.attempts`) and `njb.BackendUnavailable` when no backend can start.

## Memory and synchronization model

When `decode()` returns, the image is completely written. Consumers never need to wait on a producer stream, so no `stream` key is set in the CUDA Array Interface.

The CUDA backend owns a `cudaMalloc` buffer per image and frees it with `cudaFree`. `cudaFree` implicitly synchronizes the device, so the buffer is never freed while a kernel still reads it.

The Jetson backend keeps a pool of RGBA surfaces, each with its EGLImage. A surface is CUDA-registered after VIC writes it. When an image is released, the backend records an event on the stream you pass, or, when the image is garbage collected, marks the slot as needing a device sync. Before reusing the slot it waits on that event or sync, unregisters it, and only then lets VIC write again. Waits happen inside `decode()` with the GIL released, never in destructors.

`decode_to_torch()` on Jetson copies on the current stream and releases with that stream. On dGPUs, nvJPEG writes straight into the tensor on the current stream.

## Changes from the original single-file snippet

| Original | Here | Why |
|---|---|---|
| `#include "NvBufSurface.h"` | `nvbufsurface.h` | The real header name; Linux paths are case-sensitive |
| System `jpeglib.h` could be picked up | `include/libjpeg-8b` first, plus a `#error` guard | The Tegra `jconfig.h` defines `TEGRA_ACCELERATE`, which changes struct layout |
| `NvJPEGDecoder` (libjpeg `exit()` on bad input) | Tegra libjpeg called directly with a `setjmp` error manager | A corrupt JPEG must not kill Python |
| `cuInit` with no current context | Runtime API on the primary context | Works in any thread; same context as PyTorch |
| Registered READ_ONLY, exported writable | Registered with no flags | Flags now match what the export allows |
| No `frameType` check | Requires pitch-linear | Avoids reading garbage pointers |
| Allocate, map, register, destroy per image | Pooled surfaces with stream-aware reuse | Much less per-image overhead |
| No source crop | `CROP_SRC` to the real size | Decoder buffers may be padded |
| `-lnvjpeg` resolved by search order | Tegra dir first plus `DT_RPATH` | Tegra and CUDA both ship a `libnvjpeg` |
| Shared cudart | Static cudart when available | No version clash with PyTorch's cudart |
| Jetson only | Jetson, any NVIDIA GPU, and Python fallbacks | Portability |

## Troubleshooting

Start with `uv run python -m nvjpeg_bridge` (Jetson: `.venv/bin/python -m nvjpeg_bridge`). It shows what was built and why each backend is or isn't usable.

- **Native backend "not built".** Rebuild with `uv pip install -v .` (or `uv sync -v --reinstall-package nvjpeg-bridge`) and read the `[nvjpeg-bridge]` plan lines. To make a backend mandatory, set `NVJPEG_BRIDGE_BACKENDS=jetson,cuda`.
- **Rebuilds on every `uv run`.** A build variable such as `CUDA_HOME` is set differently between runs; export it once in your shell.
- **Windows: `CUDA toolkit not usable (missing: cudart.lib)`.** The CUDA Toolkit isn't installed, or `CUDA_PATH` points somewhere without `lib\x64`. Reinstall the toolkit, or set `CUDA_PATH` to it.
- **Windows: `Microsoft Visual C++ 14.0 or greater is required`.** Install the "Desktop development with C++" workload, then rebuild.
- **Windows: `DLL load failed while importing _cuda`.** The CUDA DLLs weren't found. `python -m nvjpeg_bridge` prints the directories it searched. Make sure `CUDA_PATH` points to a toolkit with the same major version as the one used for the build.
- **JetPack 7: `nvjpeg.h` or `cuda_runtime_api.h` not found.** The Multimedia API package pulls in only part of CUDA 13. Install `nvidia-jetpack`, or just `cuda-cudart-dev-13-0 libnvjpeg-dev-13-0` (adjust the version to your CUDA). The install script prints the exact names.
- **Crashes or garbage right after a JetPack upgrade.** The extension was built against the old release's struct layouts. Reinstall with `uv pip install --reinstall -v .`.
- **`import torch` fails on Jetson after installing something.** A NumPy (or other package) installed into the venv is shadowing JetPack's. Run `uv pip uninstall numpy` in that venv.
- **`undefined symbol: jpeg_...` when importing `_jetson`.** The wrong `libnvjpeg` was loaded. Check `ldd $(python -c "import nvjpeg_bridge._jetson as m; print(m.__file__)")`. It should point into the Tegra lib dir.
- **`NvBufSurfaceMapEglImage failed`.** EGL couldn't initialize. Headless Jetsons normally work. Over SSH with X forwarding, try `unset DISPLAY`.
- **Colors look slightly off on Jetson.** Run `pytest tests/test_gpu.py -k matches_pillow`. It compares against Pillow with a tolerance that catches range or channel bugs.
- **Stale or corrupted frames (shouldn't happen).** Try `Decoder(sync_after_map=True)` and please report your JetPack version.
- **Progressive, CMYK or 12-bit JPEGs.** The hardware path rejects these and the decoder falls back automatically. With `fallback=False` they raise `JpegDecodeError`.

## Testing

```bash
uv run pytest -v                     # desktop/server: uv sync already installed the dev group
.venv/bin/python -m pytest -v tests/ # Jetson, after ./scripts/install_jetson.sh
```

`test_header.py` and `test_decoder_logic.py` run anywhere. `test_gpu.py` runs on a CUDA machine. It checks each available backend against Pillow and also covers:

- zero-copy views in both layouts, through DLPack and the CUDA Array Interface;
- pool aliasing under churn;
- side-stream ordering and `out=`;
- corrupt input;
- fallbacks;
- threading.

`uv run python examples/benchmark.py` compares throughput across backends.

### Validation status

What has been verified:

- Both backends compile warning-free (`-Wall`), link and import against the real runtime and nvJPEG headers and libraries of CUDA 11.7, 12.9, 13.0 (JetPack 7.0 / 7.1), 13.2 (JetPack 7.2) and 13.5. The CUDA backend links with static cudart and `DT_RPATH`, and reports a clean error without a GPU.
- A simulated JetPack 7 machine built correctly. It had an R38 release file, CUDA 13 in the SBSA `targets/` layout without symlinks, and Tegra libraries in `.../nvidia`. A simulated JetPack 4 machine was rejected with a clear message while the CUDA backend still built. Builds from CUDA 12 and CUDA 13 pip wheels both work.
- The Windows build logic is unit-tested on simulated Windows layouts: toolkit detection through `CUDA_PATH` and `CUDA_PATH_V*`, `.lib` import libraries, MSVC flags without rpath, the clear error for DLL-only wheels, and DLL directory discovery. The C++ sources are ASCII-only and avoid GCC-specific constructs.
- The Jetson backend compiles warning-free against the MMAPI Tegra libjpeg headers, with stand-ins for `nvbufsurface.h` and `nvbufsurftransform.h` written from NVIDIA's documented API.
- The libjpeg error path was exercised: repeated corrupt input returns errors and the process survives.
- Both modules import in one process.
- The DLPack ABI, ownership and release semantics were checked field by field.
- The pure-Python tests pass.
- The uv workflows were exercised with uv 0.11: `uv sync` (editable) with rebuilds triggered by the cache keys, `uv run pytest`, `uv pip install` with and without build isolation, and the pip-wheel CUDA route.

What has **not** been verified: this code has not yet been run on real GPU or Jetson hardware (Orin or Thor), and it has not been compiled with MSVC. Run the tests on your device first. If anything fails, the output of `python -m nvjpeg_bridge` plus the failing test makes it quick to diagnose.

## Limitations and roadmap

- EXIF orientation is not applied by any backend, so they all return identical pixels; rotate the result yourself if you need to.
- The nvJPEG hardware backend found on A100, H100 and similar GPUs (batched API) is not used yet; the default GPU and hybrid path is.
- Each Jetson `Decoder` uses one NVJPG session; to keep the hardware busier, run several decoders in threads.
- CUDA 13 and JetPack 7 have been compile-tested only. On Thor with Multi-Instance GPU (MIG) enabled, the `jetson` backend always uses CUDA device 0.
