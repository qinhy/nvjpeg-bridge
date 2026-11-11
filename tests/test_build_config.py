"""Build-script detection logic (setup.py) on simulated Linux / Windows layouts. No GPU needed."""

import importlib.util
import os

import pytest

pytest.importorskip("setuptools")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def setup_mod(monkeypatch):
    spec = importlib.util.spec_from_file_location("njb_setup", os.path.join(ROOT, "setup.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__": setup() is not called
    monkeypatch.setattr(mod, "pip_wheel_cuda_dirs", lambda: ([], []))
    monkeypatch.setattr(mod, "pybind11_include", lambda: "/pybind11/include")
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    for k in list(os.environ):
        if k.upper().startswith(("CUDA_HOME", "CUDA_PATH", "NVJPEG_HOME")):
            monkeypatch.delenv(k)
    return mod


def _touch(root, *rel):
    for r in rel:
        p = os.path.join(root, *r.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()


HEADERS = ("include/cuda_runtime_api.h", "include/crt/host_defines.h", "include/nvjpeg.h")


def test_windows_toolkit(setup_mod, monkeypatch, tmp_path):
    tk = str(tmp_path / "CUDA" / "v12.6")
    _touch(tk, *HEADERS, "lib/x64/cudart.lib", "lib/x64/cudart_static.lib", "lib/x64/nvjpeg.lib",
           "bin/cudart64_12.dll", "bin/nvjpeg64_12.dll")
    monkeypatch.setattr(setup_mod, "IS_WINDOWS", True)
    monkeypatch.setenv("CUDA_PATH", tk)
    cuda, msg = setup_mod.detect_cuda()
    assert cuda is not None, msg
    assert cuda["cudart"] == (os.path.join(tk, "lib/x64"), "cudart")  # dynamic on Windows
    ext, msg = setup_mod.cuda_extension(cuda, mandatory=True)
    assert ext is not None, msg
    assert ext.libraries == ["nvjpeg", "cudart"]
    assert ext.extra_link_args == []
    assert "/utf-8" in ext.extra_compile_args and "/std:c++17" in ext.extra_compile_args


def test_windows_versioned_env_var(setup_mod, monkeypatch, tmp_path):
    tk = str(tmp_path / "v13.0")
    _touch(tk, *HEADERS, "lib/x64/cudart.lib", "lib/x64/nvjpeg.lib")
    monkeypatch.setattr(setup_mod, "IS_WINDOWS", True)
    monkeypatch.setenv("CUDA_PATH_V13_0", tk)
    cuda, msg = setup_mod.detect_cuda()
    assert cuda is not None, msg


def test_windows_wheels_without_import_libs(setup_mod, monkeypatch, tmp_path):
    # Windows CUDA wheels ship DLLs and headers but no .lib files.
    w = str(tmp_path / "nvidia" / "cu13")
    _touch(w, *HEADERS, "bin/x86_64/cudart64_13.dll", "bin/x86_64/nvjpeg64_13.dll")
    monkeypatch.setattr(setup_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(setup_mod, "pip_wheel_cuda_dirs", lambda: ([os.path.join(w, "include")], []))
    cuda, msg = setup_mod.detect_cuda()
    assert cuda is None
    assert "cudart.lib" in msg and "CUDA Toolkit" in msg


def test_linux_toolkit_unchanged(setup_mod, monkeypatch, tmp_path):
    tk = str(tmp_path / "cuda")
    _touch(tk, *HEADERS, "lib64/libcudart_static.a", "lib64/libnvjpeg.so.12")
    monkeypatch.setattr(setup_mod, "IS_WINDOWS", False)
    monkeypatch.setenv("CUDA_HOME", tk)
    cuda, msg = setup_mod.detect_cuda()
    ext, _ = setup_mod.cuda_extension(cuda, mandatory=True)
    assert ext.libraries == [":libnvjpeg.so.12", "cudart_static", "dl", "rt", "pthread"]
    assert "-Wl,--disable-new-dtags" in ext.extra_link_args
    assert "-O3" in ext.extra_compile_args


def test_windows_dll_dirs(monkeypatch, tmp_path):
    from nvjpeg_bridge import _backends

    tk = tmp_path / "v12.6"
    (tk / "bin").mkdir(parents=True)
    wheel = tmp_path / "site" / "nvidia" / "cu13" / "bin" / "x86_64"
    wheel.mkdir(parents=True)
    for k in list(os.environ):
        if k.upper().startswith(("CUDA_HOME", "CUDA_PATH")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("CUDA_PATH", str(tk))
    monkeypatch.setattr(_backends.sys, "path", [str(tmp_path / "site")])
    dirs = _backends.windows_cuda_dll_dirs()
    assert str(tk / "bin") in dirs and str(wheel) in dirs


def test_report_uses_dll_aware_loader(monkeypatch):
    """Regression: the environment report must load native modules through load_native()
    (which registers CUDA DLL directories on Windows), not with a plain import."""
    from nvjpeg_bridge import _env

    calls = []

    class FakeMod:
        cuda_runtime_version = 12060
        nvjpeg_version = (12, 3, 3)

    def fake_load(name):
        calls.append(name)
        return FakeMod

    monkeypatch.setattr(_env, "native_module_built", lambda name: True)
    monkeypatch.setattr(_env, "load_native", fake_load)
    monkeypatch.setattr(_env, "available_backends", lambda device=0: {})
    text = _env.report()
    assert calls == ["jetson", "cuda"]
    assert "native cuda   built (cudart 12060, nvjpeg 12.3.3)" in text


def test_jetson_unavailable_message_off_jetson(monkeypatch):
    from nvjpeg_bridge import _backends, _platform

    monkeypatch.setattr(_backends, "native_module_built", lambda name: False)
    monkeypatch.setattr(_platform, "jetson_info", lambda: None)
    with pytest.raises(_backends.BackendUnavailable, match="not a Jetson device"):
        _backends.load_native("jetson")
