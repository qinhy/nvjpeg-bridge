#!/usr/bin/env bash
# Install nvjpeg-bridge on a Jetson with uv.
# Supported: JetPack 5.x (L4T R35), 6.x (R36), 7.0/7.1 (R38, Thor), 7.2 (R39, Orin + Thor).
# Run from the repository root:  ./scripts/install_jetson.sh
#
# Creates ./.venv on top of the *system* Python with --system-site-packages, so
# NVIDIA's JetPack PyTorch / NumPy / OpenCV stay visible and are never shadowed.
#   VENV=/path/to/venv ./scripts/install_jetson.sh   # use another location
#   NVJPEG_BRIDGE_BACKENDS=jetson ...                 # require only the NVJPG backend
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f /etc/nv_tegra_release ]; then
  echo "This does not look like a Jetson (no /etc/nv_tegra_release)." >&2
  echo "On other machines run:  uv sync   (or: uv pip install -v .)" >&2
  exit 1
fi
if ! python3 src/nvjpeg_bridge/_platform.py; then
  echo "This JetPack release is not supported by the Jetson backend (JetPack 5 or newer is required)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y nvidia-l4t-jetson-multimedia-api libegl-dev python3-dev curl

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -f "$CUDA_HOME/include/cuda_runtime_api.h" ] || [ ! -f "$CUDA_HOME/include/nvjpeg.h" ]; then
  CUDA_VER=$(ls -d /usr/local/cuda-* 2>/dev/null | sed -n 's|.*/cuda-\([0-9]*\)\.\([0-9]*\)$|\1-\2|p' | sort -V | tail -n1)
  echo "CUDA toolkit headers (cuda_runtime_api.h, nvjpeg.h) not found under $CUDA_HOME." >&2
  echo "Install the JetPack CUDA components, for example:" >&2
  echo "    sudo apt-get install nvidia-jetpack" >&2
  if [ -n "$CUDA_VER" ]; then
    echo "or just the pieces needed here:" >&2
    echo "    sudo apt-get install cuda-cudart-dev-$CUDA_VER libnvjpeg-dev-$CUDA_VER" >&2
  fi
  echo "(or set CUDA_HOME if the toolkit lives elsewhere), then re-run this script." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  # System Python (3.8 on JetPack 5, 3.10 on 6, 3.12 on 7): Jetson PyTorch wheels target it.
  uv venv --python /usr/bin/python3 --system-site-packages "$VENV"
fi

# Both native backends are required here, so a problem fails loudly instead of being skipped.
export NVJPEG_BRIDGE_BACKENDS="${NVJPEG_BRIDGE_BACKENDS:-jetson,cuda}"
uv pip install --python "$VENV" -v .

# Test tools. NumPy is deliberately NOT installed: the system one must be used,
# because a different NumPy in the venv can break the installed PyTorch build.
uv pip install --python "$VENV" pytest pillow simplejpeg

"$VENV/bin/python" -m nvjpeg_bridge
echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Run the tests with:   $VENV/bin/python -m pytest -v tests/"
