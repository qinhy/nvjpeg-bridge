// Generic backend: CUDA nvJPEG (part of the CUDA toolkit).
//
// Works on any NVIDIA GPU supported by the installed CUDA toolkit, including
// Jetson (where it runs on the GPU rather than the NVJPG engine). Decodes
// straight into device memory as RGB, either interleaved (HWC) or planar (CHW).
//
// Only host code: no nvcc is needed to build this file.
#include <pybind11/pybind11.h>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <mutex>
#include <string>

#include "../common/gpu_image.hpp"

namespace py = pybind11;
using namespace njb;

namespace {

const char* nvjpeg_status_name(nvjpegStatus_t s) {
    switch (s) {
        case NVJPEG_STATUS_SUCCESS: return "SUCCESS";
        case NVJPEG_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case NVJPEG_STATUS_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case NVJPEG_STATUS_BAD_JPEG: return "BAD_JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "JPEG_NOT_SUPPORTED";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "ALLOCATOR_FAILURE";
        case NVJPEG_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case NVJPEG_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case NVJPEG_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "IMPLEMENTATION_NOT_SUPPORTED";
        default: return "UNKNOWN";
    }
}

void nvjpeg_check(nvjpegStatus_t s, const char* what) {
    if (s == NVJPEG_STATUS_SUCCESS) return;
    std::string msg = std::string(what) + " failed: NVJPEG_STATUS_" + nvjpeg_status_name(s);
    if (s == NVJPEG_STATUS_BAD_JPEG || s == NVJPEG_STATUS_JPEG_NOT_SUPPORTED ||
        s == NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED)
        throw UnsupportedJpeg(msg);
    throw DecodeFailure(msg);
}

// Image that owns a cudaMalloc'd buffer. cudaFree() implicitly synchronises the
// device, so memory is never freed while a consumer kernel may still read it.
class CudaOwnedImage final : public GpuImage {
public:
    CudaOwnedImage(void* buf, int device, int w, int h, Layout layout) : buf_(buf) {
        device_ = device;
        if (layout == Layout::HWC)
            set_interleaved_view(buf, w, h, static_cast<size_t>(w) * 3, 3, Layout::HWC);
        else
            set_planar_view(buf, w, h);
    }
    ~CudaOwnedImage() override { free_buffer(); }

    void release_impl(uintptr_t, bool) override { free_buffer(); }
    const char* backend() const override { return "cuda"; }

private:
    void free_buffer() noexcept {
        if (buf_ == nullptr) return;
        int prev = -1;
        cudaGetDevice(&prev);
        cudaSetDevice(device_);
        cudaFree(buf_);  // errors ignored: may run at interpreter shutdown
        if (prev >= 0) cudaSetDevice(prev);
        buf_ = nullptr;
        data_ = nullptr;
    }
    void* buf_ = nullptr;
};

struct ImageInfo {
    int width = 0, height = 0, components = 0;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
};

const char* css_name(nvjpegChromaSubsampling_t c) {
    switch (c) {
        case NVJPEG_CSS_444: return "444";
        case NVJPEG_CSS_422: return "422";
        case NVJPEG_CSS_420: return "420";
        case NVJPEG_CSS_440: return "440";
        case NVJPEG_CSS_411: return "411";
        case NVJPEG_CSS_410: return "410";
        case NVJPEG_CSS_GRAY: return "gray";
        default: return "unknown";
    }
}

class CudaDecoder {
public:
    explicit CudaDecoder(int device) : device_(device) {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (device < 0 || device >= count)
            throw std::invalid_argument("CUDA device " + std::to_string(device) +
                                        " does not exist (found " + std::to_string(count) + ")");
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        nvjpeg_check(nvjpegCreateSimple(&handle_), "nvjpegCreateSimple");
        nvjpeg_check(nvjpegJpegStateCreate(handle_, &state_), "nvjpegJpegStateCreate");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    ~CudaDecoder() {
        cudaSetDevice(device_);
        if (stream_) cudaStreamDestroy(stream_);
        if (state_) nvjpegJpegStateDestroy(state_);
        if (handle_) nvjpegDestroy(handle_);
    }

    CudaDecoder(const CudaDecoder&) = delete;
    CudaDecoder& operator=(const CudaDecoder&) = delete;

    py::tuple info(py::object src) {
        PyBufferView buf(src);
        ImageInfo ii;
        {
            py::gil_scoped_release nogil;
            std::lock_guard<std::mutex> lk(mu_);
            ii = get_info(buf.data(), buf.size());
        }
        return py::make_tuple(ii.width, ii.height, ii.components, css_name(ii.subsampling));
    }

    std::shared_ptr<GpuImage> decode(py::object src, const std::string& layout_s) {
        Layout layout = parse_layout(layout_s);
        PyBufferView buf(src);
        void* dst = nullptr;
        ImageInfo ii;
        {
            py::gil_scoped_release nogil;
            std::lock_guard<std::mutex> lk(mu_);
            cuda_check(cudaSetDevice(device_), "cudaSetDevice");
            ii = get_info(buf.data(), buf.size());
            size_t bytes = static_cast<size_t>(ii.width) * ii.height * 3;
            cuda_check(cudaMalloc(&dst, bytes), "cudaMalloc");
            try {
                decode_impl(buf.data(), buf.size(), static_cast<uint8_t*>(dst), ii, layout, stream_);
                cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            } catch (...) {
                cudaFree(dst);
                throw;
            }
        }
        return std::make_shared<CudaOwnedImage>(dst, device_, ii.width, ii.height, layout);
    }

    // Decode into caller-owned, contiguous device memory (e.g. a torch tensor).
    // Work is ordered after everything already enqueued on `stream`, and the
    // call returns once the image is complete.
    void decode_into(py::object src, uintptr_t dst, int expect_w, int expect_h,
                     const std::string& layout_s, uintptr_t stream) {
        Layout layout = parse_layout(layout_s);
        if (dst == 0) throw std::invalid_argument("decode_into: null destination pointer");
        PyBufferView buf(src);
        {
            py::gil_scoped_release nogil;
            std::lock_guard<std::mutex> lk(mu_);
            cuda_check(cudaSetDevice(device_), "cudaSetDevice");
            ImageInfo ii = get_info(buf.data(), buf.size());
            if (ii.width != expect_w || ii.height != expect_h)
                throw std::invalid_argument("decode_into: destination is " + std::to_string(expect_w) +
                                            "x" + std::to_string(expect_h) + " but image is " +
                                            std::to_string(ii.width) + "x" + std::to_string(ii.height));
            auto s = reinterpret_cast<cudaStream_t>(stream);
            decode_impl(buf.data(), buf.size(), reinterpret_cast<uint8_t*>(dst), ii, layout, s);
            cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        }
    }

    int device() const { return device_; }

private:
    ImageInfo get_info(const uint8_t* data, size_t len) {
        ImageInfo ii;
        int widths[NVJPEG_MAX_COMPONENT] = {0};
        int heights[NVJPEG_MAX_COMPONENT] = {0};
        nvjpeg_check(nvjpegGetImageInfo(handle_, data, len, &ii.components, &ii.subsampling, widths,
                                        heights),
                     "nvjpegGetImageInfo");
        ii.width = widths[0];
        ii.height = heights[0];
        if (ii.subsampling == NVJPEG_CSS_UNKNOWN)
            throw UnsupportedJpeg("nvJPEG: unknown chroma subsampling");
        if (ii.components != 1 && ii.components != 3)
            throw UnsupportedJpeg("nvJPEG backend supports 1 or 3 components, got " +
                                  std::to_string(ii.components) + " (CMYK/YCCK is not supported)");
        if (ii.width <= 0 || ii.height <= 0) throw UnsupportedJpeg("nvJPEG: invalid image size");
        return ii;
    }

    void decode_impl(const uint8_t* data, size_t len, uint8_t* dst, const ImageInfo& ii, Layout layout,
                     cudaStream_t stream) {
        if (state_ == nullptr)  // a previous failure could not recreate it
            nvjpeg_check(nvjpegJpegStateCreate(handle_, &state_), "nvjpegJpegStateCreate");

        nvjpegImage_t out{};
        nvjpegOutputFormat_t fmt;
        if (layout == Layout::HWC) {
            fmt = NVJPEG_OUTPUT_RGBI;
            out.channel[0] = dst;
            out.pitch[0] = static_cast<size_t>(ii.width) * 3;
        } else {
            fmt = NVJPEG_OUTPUT_RGB;
            size_t plane = static_cast<size_t>(ii.width) * ii.height;
            for (int c = 0; c < 3; ++c) {
                out.channel[c] = dst + c * plane;
                out.pitch[c] = static_cast<size_t>(ii.width);
            }
        }
        nvjpegStatus_t st = nvjpegDecode(handle_, state_, data, len, fmt, &out, stream);
        if (st != NVJPEG_STATUS_SUCCESS) {
            // Reset the decode state so a bad image cannot poison later calls.
            cudaStreamSynchronize(stream);
            nvjpegJpegStateDestroy(state_);
            state_ = nullptr;
            if (nvjpegJpegStateCreate(handle_, &state_) != NVJPEG_STATUS_SUCCESS) state_ = nullptr;
            nvjpeg_check(st, "nvjpegDecode");
        }
    }

    int device_ = 0;
    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::mutex mu_;
};

}  // namespace

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "nvjpeg_bridge CUDA nvJPEG backend";
    register_errors(m);
    bind_gpu_image(m);

    py::class_<CudaDecoder>(m, "CudaDecoder", py::module_local())
        .def(py::init<int>(), py::arg("device") = 0)
        .def("info", &CudaDecoder::info, py::arg("data"),
             "Return (width, height, components, subsampling) without decoding.")
        .def("decode", &CudaDecoder::decode, py::arg("data"), py::arg("layout") = "HWC",
             "Decode JPEG bytes into a new GpuImage (RGB uint8).")
        .def("decode_into", &CudaDecoder::decode_into, py::arg("data"), py::arg("dst_ptr"),
             py::arg("width"), py::arg("height"), py::arg("layout"), py::arg("stream"),
             "Decode into caller-owned contiguous device memory, ordered on `stream`.")
        .def_property_readonly("device", &CudaDecoder::device);

    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    m.attr("cuda_runtime_version") = rt;
    int major = 0, minor = 0, patch = 0;
    nvjpegGetProperty(MAJOR_VERSION, &major);
    nvjpegGetProperty(MINOR_VERSION, &minor);
    nvjpegGetProperty(PATCH_LEVEL, &patch);
    m.attr("nvjpeg_version") = py::make_tuple(major, minor, patch);
}
