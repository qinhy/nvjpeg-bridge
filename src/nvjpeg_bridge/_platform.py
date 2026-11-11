"""Jetson platform detection (pure Python, no dependencies).

Also loaded directly by setup.py and runnable as a script:
    python3 src/nvjpeg_bridge/_platform.py
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional, Tuple

NV_TEGRA_RELEASE = "/etc/nv_tegra_release"
DT_COMPATIBLE = "/proc/device-tree/compatible"

# L4T major -> JetPack series. Revision-specific entries refine it.
_JETPACK = {
    32: "4.x",
    34: "5.0 (developer preview)",
    35: "5.x",
    36: "6.x",
    38: "7.0 / 7.1",
    39: "7.2",
}
_JETPACK_REV = {
    (38, "2"): "7.0",
    (38, "4"): "7.1",
    (39, "2"): "7.2",
}
_SOC = {
    "tegra194": "Xavier",
    "tegra234": "Orin",
    "tegra264": "Thor",
}

# First L4T release with the NvBufSurface API this package uses (JetPack 5).
MIN_SUPPORTED_L4T = 34


class JetsonInfo(NamedTuple):
    l4t_major: int
    l4t_revision: str  # e.g. "2.1"
    jetpack: str  # e.g. "7.0"
    soc: str  # e.g. "Thor (tegra264)" or "unknown"

    @property
    def l4t(self) -> str:
        return "R%d.%s" % (self.l4t_major, self.l4t_revision)

    @property
    def supported(self) -> bool:
        return self.l4t_major >= MIN_SUPPORTED_L4T

    def describe(self) -> str:
        note = "" if self.supported else "  [NOT SUPPORTED: needs JetPack 5 or newer]"
        return "L4T %s -> JetPack %s, %s%s" % (self.l4t, self.jetpack, self.soc, note)


def parse_nv_tegra_release(text: str) -> Optional[Tuple[int, str]]:
    """'# R38 (release), REVISION: 2.1, GCID: ...' -> (38, '2.1')."""
    m = re.search(r"R(\d+)\s*\(\w+\),\s*REVISION:\s*([\d.]+)", text)
    if not m:
        return None
    return int(m.group(1)), m.group(2).rstrip(".")


def jetpack_for(major: int, revision: str) -> str:
    minor = revision.split(".")[0] if revision else ""
    if (major, minor) in _JETPACK_REV:
        return _JETPACK_REV[(major, minor)]
    if major in _JETPACK:
        return _JETPACK[major]
    if major > max(_JETPACK):
        return "newer than 7.2 (unknown to this version)"
    return "unknown"


def soc_from_compatible(raw: bytes) -> str:
    for key, name in _SOC.items():
        if b"nvidia," + key.encode() in raw:
            return "%s (%s)" % (name, key)
    return "unknown"


def jetson_info() -> Optional[JetsonInfo]:
    """Information about the running Jetson, or None when not on a Jetson."""
    try:
        with open(NV_TEGRA_RELEASE) as f:
            parsed = parse_nv_tegra_release(f.read())
    except OSError:
        return None
    if parsed is None:
        return None
    try:
        with open(DT_COMPATIBLE, "rb") as f:
            soc = soc_from_compatible(f.read())
    except OSError:
        soc = "unknown"
    major, rev = parsed
    return JetsonInfo(major, rev, jetpack_for(major, rev), soc)


if __name__ == "__main__":
    import sys

    info = jetson_info()
    if info is None:
        print("not a Jetson")
        sys.exit(2)
    print(info.describe())
    sys.exit(0 if info.supported else 1)
