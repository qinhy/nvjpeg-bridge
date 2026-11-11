"""End-to-end tests on a real GPU. Each available backend is tested on its own.

Run on the target device:  uv run pytest -v   (Jetson: .venv/bin/python -m pytest -v tests/)
"""

import gc

import pytest

import nvjpeg_bridge as njb
from conftest import RGB_MEAN_TOL, assert_close_to_reference, make_jpeg, pil_reference

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

AVAILABLE = [n for n, (ok, _) in njb.available_backends().items() if ok]
NATIVE = [n for n in ("jetson", "cuda") if n in AVAILABLE]

MEAN_TOL = RGB_MEAN_TOL


def _decoder(name):
    return njb.Decoder(backend=name, fallback=False)


def _close_to_reference(rgb, data):
    assert_close_to_reference(rgb, data)


@pytest.mark.parametrize("backend", AVAILABLE)
@pytest.mark.parametrize("sub", [0, 1, 2])
def test_matches_pillow(backend, sub):
    data = make_jpeg(subsampling=sub, seed=sub)
    t = _decoder(backend).decode_to_torch(data)
    assert t.dtype == torch.uint8 and t.is_cuda and t.shape == (163, 257, 3) and t.is_contiguous()
    _close_to_reference(t.cpu().numpy(), data)


@pytest.mark.parametrize("backend", NATIVE)
def test_zero_copy_views_and_layouts(backend):
    data = make_jpeg()
    dec = _decoder(backend)
    hwc = dec.decode(data, layout="HWC")
    chw = dec.decode(data, layout="CHW")
    assert hwc.shape == (163, 257, 3) and chw.shape == (3, 163, 257)
    a = njb.to_numpy(hwc)
    b = njb.to_numpy(chw)
    assert np.array_equal(a, b.transpose(1, 2, 0))
    _close_to_reference(a, data)
    # DLPack and CUDA Array Interface both work and share memory.
    t1 = torch.from_dlpack(hwc)
    assert t1.data_ptr() == hwc.ptr
    assert np.array_equal(t1.cpu().numpy(), a)
    t2 = torch.as_tensor(hwc, device="cuda")
    assert np.array_equal(t2.cpu().numpy(), a)


@pytest.mark.parametrize("backend", NATIVE)
def test_images_survive_reuse(backend):
    """Many live images of the same size must never alias each other (pool safety)."""
    dec = _decoder(backend)
    datas = [make_jpeg(seed=i) for i in range(12)]
    imgs = [dec.decode(d) for d in datas]
    tensors = [torch.from_dlpack(i) for i in imgs]
    del imgs
    gc.collect()
    for _ in range(3):  # churn the pool while the tensors are alive
        [dec.decode(d) for d in datas]
    for t, d in zip(tensors, datas):
        _close_to_reference(t.cpu().numpy(), d)


@pytest.mark.parametrize("backend", AVAILABLE)
def test_decode_to_torch_side_stream_and_out(backend):
    dec = _decoder(backend)
    datas = [make_jpeg(seed=i) for i in range(6)]
    s = torch.cuda.Stream()
    outs = []
    with torch.cuda.stream(s):
        for d in datas:
            o = torch.empty((3, 163, 257), dtype=torch.uint8, device="cuda")
            outs.append(dec.decode_to_torch(d, layout="CHW", out=o).float().mean())
    torch.cuda.synchronize()
    for d, m in zip(datas, outs):
        assert abs(m.item() - pil_reference(d).mean()) < MEAN_TOL


@pytest.mark.parametrize("backend", AVAILABLE)
def test_out_validation(backend):
    dec = _decoder(backend)
    with pytest.raises(TypeError):
        dec.decode_to_torch(make_jpeg(), out=torch.empty((1, 1, 3), dtype=torch.uint8, device="cuda"))


def test_auto_fallback_handles_hard_images():
    dec = njb.Decoder()
    for data in (make_jpeg(progressive=True), make_jpeg(mode="L")):
        t = dec.decode_to_torch(data)
        assert t.shape == (163, 257, 3)
        _close_to_reference(t.cpu().numpy(), data)


@pytest.mark.parametrize("backend", AVAILABLE)
def test_corrupt_input_raises_not_crashes(backend):
    dec = _decoder(backend)
    for bad in (b"\xff\xd8" + bytes(range(256)) * 4, b"not a jpeg at all"):
        with pytest.raises(njb.JpegDecodeError):
            dec.decode(bad)
    # still healthy afterwards
    assert dec.decode_to_torch(make_jpeg()).shape == (163, 257, 3)


@pytest.mark.skipif("jetson" not in AVAILABLE, reason="not a Jetson build")
def test_jetson_rejects_progressive_without_fallback():
    with pytest.raises(njb.JpegDecodeError):
        _decoder("jetson").decode(make_jpeg(progressive=True))


@pytest.mark.parametrize("backend", NATIVE)
def test_threads(backend):
    from concurrent.futures import ThreadPoolExecutor

    dec = _decoder(backend)
    datas = [make_jpeg(seed=i) for i in range(16)]
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda d: dec.decode_to_torch(d).float().mean().item(), datas))
    for d, m in zip(datas, results):
        assert abs(m - pil_reference(d).mean()) < MEAN_TOL
