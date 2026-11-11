"""JetPack / SoC detection: no Jetson needed."""

import pytest

from nvjpeg_bridge import _platform as P

RELEASES = [
    ("# R32 (release), REVISION: 7.4, GCID: 33514132, BOARD: t210ref, EABI: aarch64", 32, "7.4", "4.x", False),
    ("# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref, EABI: aarch64", 35, "4.1", "5.x", True),
    ("# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64", 36, "4.3", "6.x", True),
    ("# R38 (release), REVISION: 2.1, GCID: 41844464, BOARD: generic, EABI: aarch64", 38, "2.1", "7.0", True),
    ("# R38 (release), REVISION: 4.0, GCID: 42706000, BOARD: generic, EABI: aarch64", 38, "4.0", "7.1", True),
    ("# R39 (release), REVISION: 2.0, GCID: 43900000, BOARD: generic, EABI: aarch64", 39, "2.0", "7.2", True),
    ("# R39 (release), REVISION: 2.1, GCID: 44100000, BOARD: generic, EABI: aarch64", 39, "2.1", "7.2", True),
]


@pytest.mark.parametrize("text,major,rev,jetpack,supported", RELEASES)
def test_release_mapping(text, major, rev, jetpack, supported):
    assert P.parse_nv_tegra_release(text) == (major, rev)
    info = P.JetsonInfo(major, rev, P.jetpack_for(major, rev), "x")
    assert info.jetpack == jetpack
    assert info.supported is supported
    assert info.l4t == "R%d.%s" % (major, rev)


def test_future_release_is_reported_not_rejected():
    jp = P.jetpack_for(41, "1.0")
    assert "newer" in jp
    assert P.JetsonInfo(41, "1.0", jp, "x").supported


@pytest.mark.parametrize("compat,soc", [
    (b"nvidia,p3737-0000+p3701-0005\0nvidia,p3701-0005\0nvidia,tegra234\0", "Orin (tegra234)"),
    (b"nvidia,p3971-0000+p3834-0008\0nvidia,tegra264\0", "Thor (tegra264)"),
    (b"nvidia,p2822-0000+p2888-0001\0nvidia,tegra194\0", "Xavier (tegra194)"),
    (b"raspberrypi,4-model-b\0brcm,bcm2711\0", "unknown"),
])
def test_soc(compat, soc):
    assert P.soc_from_compatible(compat) == soc


def test_garbage_release():
    assert P.parse_nv_tegra_release("hello") is None


def test_not_jetson(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "NV_TEGRA_RELEASE", str(tmp_path / "missing"))
    assert P.jetson_info() is None


def test_fake_thor(monkeypatch, tmp_path):
    rel = tmp_path / "rel"
    rel.write_text(RELEASES[3][0] + "\n")
    dt = tmp_path / "dt"
    dt.write_bytes(b"nvidia,tegra264\0")
    monkeypatch.setattr(P, "NV_TEGRA_RELEASE", str(rel))
    monkeypatch.setattr(P, "DT_COMPATIBLE", str(dt))
    info = P.jetson_info()
    assert info.describe() == "L4T R38.2.1 -> JetPack 7.0, Thor (tegra264)"
