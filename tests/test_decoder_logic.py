"""Backend selection / fallback logic with fake backends: no GPU needed."""


import pytest

import nvjpeg_bridge as njb
from nvjpeg_bridge import _backends, _decoder


class _Unsupported(ValueError):
    pass


def _fake(name, fail_with=None, unavailable=False):
    class Fake(_backends.Backend):
        errors = (_Unsupported, RuntimeError)

        def __init__(self, device=0, **_):
            if unavailable:
                raise njb.BackendUnavailable(name + " not here")
            self.name = name

        def decode(self, data, layout):
            if fail_with:
                raise fail_with("%s cannot" % name)
            return (name, bytes(data), layout)

        def decode_to_torch(self, data, layout, out=None):
            return self.decode(data, layout)

    return Fake


@pytest.fixture
def registry(monkeypatch):
    def install(**backends):
        reg = dict(backends)
        monkeypatch.setattr(_backends, "REGISTRY", reg)
        monkeypatch.setattr(_decoder, "REGISTRY", reg)
        monkeypatch.setattr(_decoder, "AUTO_ORDER", tuple(reg))
    return install


def test_auto_picks_first_available(registry):
    registry(jetson=_fake("jetson", unavailable=True), cuda=_fake("cuda"), pil=_fake("pil"))
    d = njb.Decoder()
    assert d.backend == "cuda"
    assert d.decode(b"xx") == ("cuda", b"xx", "HWC")
    assert d.last_backend == "cuda"


def test_fallback_on_unsupported(registry):
    registry(jetson=_fake("jetson", fail_with=_Unsupported), cuda=_fake("cuda"))
    d = njb.Decoder()
    assert d.backend == "jetson"
    assert d.decode(b"xx", layout="chw")[0] == "cuda"
    assert d.last_backend == "cuda"


def test_no_fallback(registry):
    registry(jetson=_fake("jetson", fail_with=_Unsupported), cuda=_fake("cuda"))
    d = njb.Decoder(fallback=False)
    with pytest.raises(njb.JpegDecodeError) as ei:
        d.decode(b"xx")
    assert [n for n, _ in ei.value.attempts] == ["jetson"]


def test_explicit_backend_unavailable(registry):
    registry(jetson=_fake("jetson", unavailable=True), cuda=_fake("cuda"))
    with pytest.raises(njb.BackendUnavailable):
        njb.Decoder(backend="jetson", fallback=False)
    assert njb.Decoder(backend="jetson").backend == "cuda"


def test_all_fail_aggregates(registry):
    registry(a=_fake("a", fail_with=RuntimeError), b=_fake("b", fail_with=_Unsupported))
    with pytest.raises(njb.JpegDecodeError) as ei:
        njb.Decoder().decode(b"xx")
    assert [n for n, _ in ei.value.attempts] == ["a", "b"]


def test_nothing_available(registry):
    registry(a=_fake("a", unavailable=True))
    with pytest.raises(njb.BackendUnavailable):
        njb.Decoder()


def test_bad_layout_and_backend(registry):
    registry(a=_fake("a"))
    with pytest.raises(ValueError):
        njb.Decoder(layout="WHC")
    with pytest.raises(ValueError):
        njb.Decoder(backend="nope")


def test_as_buffer(tmp_path):
    np = pytest.importorskip("numpy")
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8abc")
    assert bytes(_decoder.as_buffer(str(p))) == b"\xff\xd8abc"
    assert bytes(_decoder.as_buffer(p)) == b"\xff\xd8abc"
    arr = np.frombuffer(b"abcdef", dtype=np.uint8)[::2]  # non-contiguous
    assert bytes(_decoder.as_buffer(arr)) == b"ace"
    assert bytes(_decoder.as_buffer(np.zeros(2, dtype=np.uint16))) == b"\0\0\0\0"
    with pytest.raises(TypeError):
        _decoder.as_buffer(12345)
