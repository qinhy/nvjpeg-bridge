import io

import pytest


def make_jpeg(w=257, h=163, subsampling=2, progressive=False, mode="RGB", seed=0, quality=92):
    """Synthetic JPEG with smooth gradients + texture (odd sizes on purpose)."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 3) % 256], -1)
    rgb = np.clip(rgb + rng.integers(-12, 12, rgb.shape), 0, 255).astype("uint8")
    im = Image.fromarray(rgb, "RGB")
    if mode != "RGB":
        im = im.convert(mode)
    buf = io.BytesIO()
    kw = {"quality": quality, "progressive": progressive}
    if mode in ("RGB", "YCbCr"):
        kw["subsampling"] = subsampling
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


def pil_reference(data):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def _ycc(rgb):
    np = pytest.importorskip("numpy")
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _box(a, k=4):
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# Decoders legitimately differ in chroma upsampling (libjpeg "fancy" interpolation vs
# nvJPEG / fast pixel replication) and slightly in IDCT rounding. Luma is stored at full
# resolution, so it is compared strictly; chroma is compared after 4x4 smoothing, which
# cancels upsampling differences but not real errors. Calibrated against libjpeg-turbo
# fast upsampling/fast IDCT (valid: luma p99 <= 1.9, chroma p99 <= 6.9) and injected bugs
# (BGR swap, limited range, 1-px shift, Cb/Cr swap, partial output, gray output, noise),
# each of which exceeds at least one limit; see tests/test_reference_check.py.
RGB_MEAN_TOL = 3.0
LUMA_P99_TOL = 5.0
CHROMA_P99_TOL = 10.0


def compare_to_reference(rgb, ref):
    """Return a list of failure messages (empty when `rgb` is a valid decode of `ref`)."""
    np = pytest.importorskip("numpy")
    rgb = np.asarray(rgb)
    ref = np.asarray(ref)
    if rgb.shape != ref.shape:
        return ["shape %s != %s" % (rgb.shape, ref.shape)]
    errors = []
    mean = np.abs(rgb.astype(np.float64) - ref).mean()
    if mean >= RGB_MEAN_TOL:
        errors.append("mean abs RGB diff %.2f >= %.1f" % (mean, RGB_MEAN_TOL))
    y1, cb1, cr1 = _ycc(rgb)
    y2, cb2, cr2 = _ycc(ref)
    luma = np.percentile(np.abs(y1 - y2), 99)
    if luma > LUMA_P99_TOL:
        errors.append("luma p99 diff %.2f > %.1f" % (luma, LUMA_P99_TOL))
    chroma = max(np.percentile(np.abs(_box(cb1) - _box(cb2)), 99),
                 np.percentile(np.abs(_box(cr1) - _box(cr2)), 99))
    if chroma > CHROMA_P99_TOL:
        errors.append("smoothed chroma p99 diff %.2f > %.1f" % (chroma, CHROMA_P99_TOL))
    return errors


def assert_close_to_reference(rgb, data):
    errors = compare_to_reference(rgb, pil_reference(data))
    assert not errors, "; ".join(errors)


@pytest.fixture
def jpeg_factory():
    return make_jpeg
