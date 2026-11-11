"""Header parser tests: no GPU needed."""

import pytest

import nvjpeg_bridge as njb


@pytest.mark.parametrize("sub,expect", [(0, "444"), (1, "422"), (2, "420")])
def test_subsampling(jpeg_factory, sub, expect):
    h = njb.read_header(jpeg_factory(subsampling=sub))
    assert (h.width, h.height, h.components, h.precision) == (257, 163, 3, 8)
    assert h.subsampling == expect
    assert not h.progressive and not h.arithmetic and not h.lossless


def test_progressive(jpeg_factory):
    assert njb.read_header(jpeg_factory(progressive=True)).progressive


def test_gray(jpeg_factory):
    h = njb.read_header(jpeg_factory(mode="L"))
    assert h.components == 1 and h.subsampling == "gray"


def test_cmyk(jpeg_factory):
    h = njb.read_header(jpeg_factory(mode="CMYK"))
    assert h.components == 4 and h.adobe_transform >= 0


@pytest.mark.parametrize("bad", [b"", b"\x00\x01\x02\x03", b"\xff\xd8", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\x00"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        njb.read_header(bad)


def test_buffer_types(jpeg_factory):
    np = pytest.importorskip("numpy")
    data = jpeg_factory()
    for src in (data, bytearray(data), memoryview(data), np.frombuffer(data, dtype=np.uint8)):
        assert njb.read_header(src).width == 257
