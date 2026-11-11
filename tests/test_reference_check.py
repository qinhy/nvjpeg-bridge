"""Self-test of the GPU tests' image checker, on the CPU (no GPU needed).

Valid decoder differences must pass; typical decoding bugs must fail. libjpeg-turbo's
fast upsampling / fast IDCT (via simplejpeg) stand in for nvJPEG's behaviour.
"""

import pytest

from conftest import _ycc, compare_to_reference, make_jpeg, pil_reference

np = pytest.importorskip("numpy")
simplejpeg = pytest.importorskip("simplejpeg")


def _fast(data, fastdct=False):
    return simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=fastdct, fastupsample=True)


VALID = [
    dict(subsampling=0), dict(subsampling=1), dict(subsampling=2),
    dict(progressive=True), dict(mode="L"), dict(quality=60), dict(quality=98),
]


@pytest.mark.parametrize("kw", VALID, ids=lambda kw: ",".join("%s=%s" % i for i in kw.items()))
@pytest.mark.parametrize("fastdct", [False, True])
def test_valid_decoder_differences_pass(kw, fastdct):
    data = make_jpeg(**kw)
    assert compare_to_reference(_fast(data, fastdct), pil_reference(data)) == []


def _cbcr_swap(rgb):
    y, cb, cr = _ycc(rgb)
    out = np.stack([y + 1.402 * cb, y - 0.344136 * cr - 0.714136 * cb, y + 1.772 * cr], -1)
    return out.clip(0, 255).astype(np.uint8)


BUGS = {
    "bgr_swap": lambda g: g[..., ::-1],
    "limited_range": lambda g: (g.astype(np.float64) * 219 / 255 + 16).astype(np.uint8),
    "shift_1px": lambda g: np.roll(g, 1, axis=1),
    "cbcr_swap": _cbcr_swap,
    "half_missing": lambda g: np.concatenate([g[:80], np.zeros_like(g[80:])]),
    "gray_output": lambda g: np.repeat(_ycc(g)[0].clip(0, 255).astype(np.uint8)[..., None], 3, 2),
    "noise_8": lambda g: np.clip(g.astype(int) + np.random.default_rng(1).integers(-8, 9, g.shape),
                                 0, 255).astype(np.uint8),
    "transposed_hw": lambda g: np.ascontiguousarray(g.transpose(1, 0, 2)),
}


@pytest.mark.parametrize("bug", sorted(BUGS))
@pytest.mark.parametrize("sub", [0, 2])
def test_real_bugs_fail(bug, sub):
    data = make_jpeg(subsampling=sub)
    assert compare_to_reference(BUGS[bug](_fast(data)), pil_reference(data)) != []
