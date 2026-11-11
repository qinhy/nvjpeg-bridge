"""Pure-Python JPEG header reader (no decoding, no dependencies)."""

from __future__ import annotations

from typing import NamedTuple, Tuple


class JpegHeader(NamedTuple):
    width: int
    height: int
    components: int
    precision: int
    progressive: bool
    arithmetic: bool
    lossless: bool
    sampling: Tuple[Tuple[int, int], ...]  # (h, v) per component
    adobe_transform: int  # -1 when there is no Adobe APP14 marker

    @property
    def subsampling(self) -> str:
        """'444', '422', '420', '440', 'gray' or 'other'."""
        if self.components == 1:
            return "gray"
        if self.components != 3 or any(s != (1, 1) for s in self.sampling[1:]):
            return "other"
        return {(1, 1): "444", (2, 1): "422", (2, 2): "420", (1, 2): "440"}.get(self.sampling[0], "other")


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def read_header(data) -> JpegHeader:
    """Parse the frame header of a JPEG held in any bytes-like object.

    Raises ValueError if the data is not a well-formed JPEG header.
    """
    p = memoryview(data).cast("B")
    n = len(p)
    if n < 4 or p[0] != 0xFF or p[1] != 0xD8:
        raise ValueError("not a JPEG (missing SOI marker)")
    adobe = -1
    i = 2
    while i + 1 < n:
        if p[i] != 0xFF:
            raise ValueError("corrupt JPEG marker structure")
        m = p[i + 1]
        if m == 0xFF:
            i += 1
            continue
        i += 2
        if m == 0x01 or m == 0xD8 or 0xD0 <= m <= 0xD7:
            continue
        if m == 0xD9:
            raise ValueError("EOI before frame header")
        if i + 2 > n:
            raise ValueError("truncated JPEG segment")
        length = (p[i] << 8) | p[i + 1]
        if length < 2 or i + length > n:
            raise ValueError("truncated JPEG segment")
        seg = p[i + 2:i + length]
        if m == 0xEE and len(seg) >= 12 and bytes(seg[:5]) == b"Adobe":
            adobe = seg[11]
        if m in _SOF:
            if len(seg) < 6:
                raise ValueError("truncated frame header")
            nc = seg[5]
            if nc < 1 or len(seg) < 6 + 3 * nc:
                raise ValueError("invalid component table")
            sampling = tuple((seg[7 + 3 * c] >> 4, seg[7 + 3 * c] & 0x0F) for c in range(nc))
            h, w = (seg[1] << 8) | seg[2], (seg[3] << 8) | seg[4]
            if w == 0 or h == 0:
                raise ValueError("zero image dimension (DNL marker not supported)")
            return JpegHeader(
                width=w, height=h, components=nc, precision=seg[0],
                progressive=m in (0xC2, 0xC6, 0xCA, 0xCE),
                arithmetic=m >= 0xC9,
                lossless=m in (0xC3, 0xC7, 0xCB, 0xCF),
                sampling=sampling, adobe_transform=adobe,
            )
        if m == 0xDA:
            raise ValueError("scan data before frame header")
        i += length
    raise ValueError("no JPEG frame header found")
