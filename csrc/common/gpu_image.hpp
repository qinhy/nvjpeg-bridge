// Shared pieces for both native backends:
//   * GpuImage: a decoded RGB image living in CUDA device memory, exported to
//     Python through both __cuda_array_interface__ (v3) and DLPack.
//   * PyBufferView: RAII wrapper for reading any bytes-like object without a copy.
//   * Exception types that map to Python exceptions.
//
// Every backend fully finishes its GPU work before handing a GpuImage to
// Python, so consumers never need to synchronise with a producer stream.
#pragma once

#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "dlpack_min.h"

namespace njb {
namespace py = pybind11;

// ---------------------------------------------------------------- errors ---

// The stream is valid JPEG but this backend cannot decode it (e.g. progressive
// on the Jetson hardware path). Maps to a ValueError subclass in Python.
struct UnsupportedJpeg : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// Decoding or a driver call failed. Maps to a RuntimeError subclass in Python.
struct DecodeFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear sticky-free errors so later calls are not poisoned
        throw DecodeFailure(std::string(what) + " failed: " + cudaGetErrorName(e) + " (" +
                            cudaGetErrorString(e) + ")");
    }
}

// ---------------------------------------------------------------- layout ---

enum class Layout { HWC, CHW };

inline Layout parse_layout(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    if (s == "HWC") return Layout::HWC;
    if (s == "CHW") return Layout::CHW;
    throw std::invalid_argument("layout must be 'HWC' or 'CHW', got '" + s + "'");
}

inline const char* layout_name(Layout l) { return l == Layout::HWC ? "HWC" : "CHW"; }

// ----------------------------------------------------------- PyBufferView ---

// Holds a PyBUF_SIMPLE (C-contiguous, byte-addressable) view of a Python
// object. Must be constructed and destroyed with the GIL held; the pointer may
// be used without the GIL while this object is alive.
class PyBufferView {
public:
    explicit PyBufferView(py::handle obj) {
        if (PyObject_GetBuffer(obj.ptr(), &view_, PyBUF_SIMPLE) != 0) throw py::error_already_set();
        ok_ = true;
    }
    ~PyBufferView() {
        if (ok_) PyBuffer_Release(&view_);
    }
    PyBufferView(const PyBufferView&) = delete;
    PyBufferView& operator=(const PyBufferView&) = delete;

    const uint8_t* data() const { return static_cast<const uint8_t*>(view_.buf); }
    size_t size() const { return static_cast<size_t>(view_.len); }

private:
    Py_buffer view_{};
    bool ok_ = false;
};

// --------------------------------------------------------------- GpuImage ---

class GpuImage : public std::enable_shared_from_this<GpuImage> {
public:
    virtual ~GpuImage() = default;

    // Return the memory to its owner early. If `has_stream`, the caller
    // guarantees all work using the image has been *enqueued* on `stream`;
    // the backend will wait for that stream before reusing the memory.
    // Otherwise the backend conservatively waits for the whole device.
    virtual void release_impl(uintptr_t stream, bool has_stream) = 0;
    virtual const char* backend() const = 0;

    void release(py::object stream) {
        if (exports_.load() > 0)
            throw std::runtime_error(
                "GpuImage.release(): image is still exported via DLPack (a tensor created with "
                "from_dlpack() is alive). Delete that tensor first.");
        bool has = !stream.is_none();
        uintptr_t s = has ? stream.cast<uintptr_t>() : 0;
        release_impl(s, has);
        data_ = nullptr;
    }

    bool released() const { return data_ == nullptr; }
    void check_alive() const {
        if (data_ == nullptr) throw std::runtime_error("GpuImage has been released");
    }

    py::dict cuda_array_interface() const {
        check_alive();
        py::dict d;
        d["shape"] = py::make_tuple(shape_[0], shape_[1], shape_[2]);
        d["strides"] = py::make_tuple(strides_[0], strides_[1], strides_[2]);
        d["typestr"] = "|u1";
        d["data"] = py::make_tuple(reinterpret_cast<uintptr_t>(data_), false);
        d["version"] = 3;
        // No "stream" key: data is fully written before the object is returned.
        return d;
    }

    py::tuple dlpack_device() const { return py::make_tuple(static_cast<int>(kDLCUDA), device_); }

    static py::capsule to_dlpack(const std::shared_ptr<GpuImage>& self) {
        self->check_alive();
        auto* ctx = new DLCtx;
        ctx->keep = self;
        for (int i = 0; i < 3; ++i) {
            ctx->shape[i] = self->shape_[i];
            ctx->strides[i] = self->strides_[i];  // uint8: elements == bytes
        }
        DLTensor& t = ctx->mt.dl_tensor;
        t.data = self->data_;
        t.device = DLDevice{kDLCUDA, self->device_};
        t.ndim = 3;
        t.dtype = DLDataType{kDLUInt, 8, 1};
        t.shape = ctx->shape;
        t.strides = ctx->strides;
        t.byte_offset = 0;
        ctx->mt.manager_ctx = ctx;
        ctx->mt.deleter = &DLCtx::deleter;
        self->exports_.fetch_add(1);

        PyObject* cap = PyCapsule_New(&ctx->mt, "dltensor", &capsule_destructor);
        if (cap == nullptr) {
            DLCtx::deleter(&ctx->mt);
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::capsule>(cap);
    }

    // Copies the image to host memory. Returns the raw byte span covering the
    // strided view; Python wraps it with the correct shape/strides.
    py::bytes copy_span_to_host() const {
        check_alive();
        size_t span = 1;
        for (int i = 0; i < 3; ++i) span += static_cast<size_t>((shape_[i] - 1) * strides_[i]);
        PyObject* b = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(span));
        if (b == nullptr) throw py::error_already_set();
        py::bytes out = py::reinterpret_steal<py::bytes>(b);
        char* dst = PyBytes_AsString(out.ptr());
        {
            py::gil_scoped_release nogil;
            cuda_check(cudaSetDevice(device_), "cudaSetDevice");
            cuda_check(cudaMemcpy(dst, data_, span, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        }
        return out;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int device() const { return device_; }
    Layout layout() const { return layout_; }
    uintptr_t ptr() const { return reinterpret_cast<uintptr_t>(data_); }
    py::tuple shape() const { return py::make_tuple(shape_[0], shape_[1], shape_[2]); }
    py::tuple strides() const { return py::make_tuple(strides_[0], strides_[1], strides_[2]); }
    bool is_contiguous() const {
        return strides_[2] == 1 && strides_[1] == shape_[2] && strides_[0] == shape_[1] * shape_[2];
    }

protected:
    // Describe the view. `row_pitch` and `pixel_stride` are in bytes and
    // describe an interleaved buffer; `planar` describes three packed planes.
    void set_interleaved_view(void* data, int w, int h, size_t row_pitch, size_t pixel_stride,
                              Layout layout) {
        data_ = data;
        width_ = w;
        height_ = h;
        layout_ = layout;
        if (layout == Layout::HWC) {
            shape_ = {h, w, 3};
            strides_ = {static_cast<int64_t>(row_pitch), static_cast<int64_t>(pixel_stride), 1};
        } else {
            shape_ = {3, h, w};
            strides_ = {1, static_cast<int64_t>(row_pitch), static_cast<int64_t>(pixel_stride)};
        }
    }
    void set_planar_view(void* data, int w, int h) {
        data_ = data;
        width_ = w;
        height_ = h;
        layout_ = Layout::CHW;
        shape_ = {3, h, w};
        strides_ = {static_cast<int64_t>(w) * h, w, 1};
    }

    void* data_ = nullptr;
    int width_ = 0, height_ = 0, device_ = 0;
    Layout layout_ = Layout::HWC;
    std::array<int64_t, 3> shape_{};
    std::array<int64_t, 3> strides_{};
    std::atomic<int> exports_{0};

private:
    struct DLCtx {
        std::shared_ptr<GpuImage> keep;
        int64_t shape[3];
        int64_t strides[3];
        DLManagedTensor mt{};
        // May be called by a consumer without the GIL: touches only C++ state.
        static void deleter(DLManagedTensor* self) {
            auto* c = static_cast<DLCtx*>(self->manager_ctx);
            c->keep->exports_.fetch_sub(1);
            delete c;
        }
    };

    static void capsule_destructor(PyObject* cap) {
        // A consumer renames the capsule to "used_dltensor" and takes ownership.
        if (PyCapsule_IsValid(cap, "dltensor")) {
            auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
            if (mt != nullptr && mt->deleter != nullptr) mt->deleter(mt);
        } else {
            PyErr_Clear();
        }
    }
};

// Registers the Python-visible GpuImage class in a module. module_local so that
// both native backends can be imported in the same process.
inline void bind_gpu_image(py::module_& m) {
    py::class_<GpuImage, std::shared_ptr<GpuImage>>(m, "GpuImage", py::module_local(), R"doc(
RGB uint8 image in CUDA device memory.

Supports ``__cuda_array_interface__`` (v3) and DLPack (``__dlpack__``), so it
can be wrapped without a copy by PyTorch (``torch.from_dlpack``), CuPy, Numba, etc.
The data is fully written when the object is returned; no stream sync is needed.
)doc")
        .def_property_readonly("__cuda_array_interface__", &GpuImage::cuda_array_interface)
        .def("__dlpack__",
             [](const std::shared_ptr<GpuImage>& self, py::args, py::kwargs) {
                 return GpuImage::to_dlpack(self);
             })
        .def("__dlpack_device__", &GpuImage::dlpack_device)
        .def("release", &GpuImage::release, py::arg("stream") = py::none(), R"doc(
Return the GPU memory now instead of at garbage collection.

stream: optional ``cudaStream_t`` (int) on which all work reading this image has
been enqueued. The memory is reused only after that stream reaches this point.
Without a stream the backend waits for the whole device before reuse.
)doc")
        .def("_copy_span_to_host", &GpuImage::copy_span_to_host)
        .def_property_readonly("width", &GpuImage::width)
        .def_property_readonly("height", &GpuImage::height)
        .def_property_readonly("device", &GpuImage::device)
        .def_property_readonly("layout", [](const GpuImage& s) { return layout_name(s.layout()); })
        .def_property_readonly("shape", &GpuImage::shape)
        .def_property_readonly("strides", &GpuImage::strides)
        .def_property_readonly("ptr", &GpuImage::ptr)
        .def_property_readonly("is_contiguous", &GpuImage::is_contiguous)
        .def_property_readonly("released", &GpuImage::released)
        .def_property_readonly("backend", &GpuImage::backend)
        .def("__repr__", [](const GpuImage& s) {
            std::string r = "<GpuImage backend=" + std::string(s.backend()) + " " +
                            std::to_string(s.width()) + "x" + std::to_string(s.height()) + " " +
                            layout_name(s.layout()) + " cuda:" + std::to_string(s.device());
            return r + (s.released() ? " released>" : ">");
        });
}

inline void register_errors(py::module_& m) {
    py::register_local_exception<UnsupportedJpeg>(m, "UnsupportedJpegError", PyExc_ValueError);
    py::register_local_exception<DecodeFailure>(m, "DecodeError", PyExc_RuntimeError);
}

}  // namespace njb
