// Jetson backend (JetPack 5 / 6, NvBufSurface API):
//
//   JPEG bytes --NVJPG--> YUV NvBufSurface --VIC--> RGBA NvBufSurface (pooled)
//              --EGLImage--> CUDA pointer --> GpuImage (RGB view, alpha skipped)
//
// Design notes
// * The Tegra libjpeg API (libnvjpeg.so in /usr/lib/aarch64-linux-gnu/tegra)
//   is called directly instead of through NvJPEGDecoder from the MMAPI samples:
//   NvJPEGDecoder installs libjpeg's default error handler, which calls exit()
//   on a corrupt JPEG and would terminate the Python process. Here errors are
//   turned into exceptions (see tegra_jpeg.cpp); no sample sources are compiled.
// * Must be compiled against the MMAPI copy of jpeglib.h (include/libjpeg-8b):
//   its jconfig.h defines TEGRA_ACCELERATE, which changes the struct layout.
// * RGBA surfaces are pooled (allocation is expensive). The CUDA registration
//   of a surface is kept until the slot is reused; before reuse we wait for the
//   consumer (an event recorded on the consumer's stream, or the whole device),
//   then unregister, let VIC write, and register again. This mirrors the
//   register -> use -> sync -> unregister pattern of NVIDIA's own samples.
// * Uses the CUDA *runtime* API, so it shares the primary context with PyTorch,
//   CuPy, etc. and works from any thread.

#include <pybind11/pybind11.h>

#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_egl_interop.h>
#include <cuda_runtime_api.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "../common/gpu_image.hpp"
#include "../common/jpeg_header.hpp"
#include "tegra_jpeg.hpp"

namespace py = pybind11;
using namespace njb;

namespace {

// ------------------------------------------------------------ surface pool ---

struct Slot {
    enum class Pending { None, Event, Device };
    NvBufSurface* surf = nullptr;
    int w = 0, h = 0;
    cudaGraphicsResource_t res = nullptr;
    cudaEvent_t done = nullptr;
    Pending pending = Pending::None;
};

class SurfacePool {
public:
    explicit SurfacePool(size_t max_idle) : max_idle_(max_idle) {}
    ~SurfacePool() {
        for (auto& s : idle_) destroy(*s);
    }

    // Find an idle slot of this size or allocate one. Call without the GIL.
    std::unique_ptr<Slot> take(int w, int h) {
        std::unique_ptr<Slot> s;
        std::deque<std::unique_ptr<Slot>> evict;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto it = idle_.begin(); it != idle_.end(); ++it) {
                if ((*it)->w == w && (*it)->h == h) {
                    s = std::move(*it);
                    idle_.erase(it);
                    break;
                }
            }
            while (idle_.size() > max_idle_) {  // trim oldest
                evict.push_back(std::move(idle_.front()));
                idle_.pop_front();
            }
        }
        for (auto& e : evict) destroy(*e);
        if (!s) s = create(w, h);
        prepare_for_write(*s);
        return s;
    }

    // Hand a slot back. Cheap and non-blocking: safe in destructors / with the GIL.
    void give_back(std::unique_ptr<Slot> s) noexcept {
        std::lock_guard<std::mutex> lk(mu_);
        idle_.push_back(std::move(s));
    }

    // Wait until no consumer can still read the slot, then drop its CUDA mapping.
    static void prepare_for_write(Slot& s) noexcept {
        if (s.pending == Slot::Pending::Event) cudaEventSynchronize(s.done);
        if (s.pending == Slot::Pending::Device) cudaDeviceSynchronize();
        s.pending = Slot::Pending::None;
        if (s.res) {
            cudaGraphicsUnregisterResource(s.res);
            s.res = nullptr;
        }
    }

private:
    static std::unique_ptr<Slot> create(int w, int h) {
        auto s = std::make_unique<Slot>();
        s->w = w;
        s->h = h;
        NvBufSurfaceCreateParams p;
        std::memset(&p, 0, sizeof(p));
        p.gpuId = 0;
        p.width = static_cast<uint32_t>(w);
        p.height = static_cast<uint32_t>(h);
        p.size = 0;
        p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        p.layout = NVBUF_LAYOUT_PITCH;
        p.memType = NVBUF_MEM_SURFACE_ARRAY;
        if (NvBufSurfaceCreate(&s->surf, 1, &p) != 0 || s->surf == nullptr)
            throw DecodeFailure("NvBufSurfaceCreate (RGBA " + std::to_string(w) + "x" +
                                std::to_string(h) + ") failed");
        s->surf->numFilled = 1;
        if (NvBufSurfaceMapEglImage(s->surf, 0) != 0) {
            NvBufSurfaceDestroy(s->surf);
            throw DecodeFailure("NvBufSurfaceMapEglImage failed (is an EGL display available?)");
        }
        cudaError_t e = cudaEventCreateWithFlags(&s->done, cudaEventDisableTiming);
        if (e != cudaSuccess) {
            NvBufSurfaceUnMapEglImage(s->surf, 0);
            NvBufSurfaceDestroy(s->surf);
            cuda_check(e, "cudaEventCreate");
        }
        return s;
    }

    static void destroy(Slot& s) noexcept {
        prepare_for_write(s);
        if (s.surf) {
            NvBufSurfaceUnMapEglImage(s.surf, 0);
            NvBufSurfaceDestroy(s.surf);
            s.surf = nullptr;
        }
        if (s.done) {
            cudaEventDestroy(s.done);
            s.done = nullptr;
        }
    }

    std::mutex mu_;
    std::deque<std::unique_ptr<Slot>> idle_;
    size_t max_idle_;
};

// ---------------------------------------------------------------- image ---

class JetsonImage final : public GpuImage {
public:
    JetsonImage(std::shared_ptr<SurfacePool> pool, std::unique_ptr<Slot> slot, void* ptr, size_t pitch,
                Layout layout)
        : pool_(std::move(pool)), slot_(std::move(slot)) {
        device_ = 0;
        // RGBA memory exposed as an RGB view: pixel stride 4, alpha never read.
        set_interleaved_view(ptr, slot_->w, slot_->h, pitch, 4, layout);
    }
    ~JetsonImage() override { release_impl(0, false); }

    void release_impl(uintptr_t stream, bool has_stream) override {
        if (!slot_) return;
        if (has_stream && cudaEventRecord(slot_->done, reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess)
            slot_->pending = Slot::Pending::Event;
        else
            slot_->pending = Slot::Pending::Device;
        pool_->give_back(std::move(slot_));
        data_ = nullptr;
    }
    const char* backend() const override { return "jetson"; }

private:
    std::shared_ptr<SurfacePool> pool_;
    std::unique_ptr<Slot> slot_;
};

// -------------------------------------------------------------- decoder ---

std::string check_supported(const JpegHeader& h) {
    if (h.precision != 8) return "only 8-bit JPEG is supported by NVJPG";
    if (h.progressive) return "progressive JPEG is not supported by the NVJPG path";
    if (h.arithmetic) return "arithmetic-coded JPEG is not supported by the NVJPG path";
    if (h.lossless) return "lossless JPEG is not supported by the NVJPG path";
    if (h.components != 3) return "NVJPG path supports 3-component YCbCr only (got " +
                                  std::to_string(h.components) + " components)";
    if (h.adobe_transform == 0) return "RGB-coded (non-YCbCr) JPEG is not supported by the NVJPG path";
    const JpegComponent& y = h.comp[0];
    bool luma_ok = (y.h == 2 && y.v == 2) || (y.h == 2 && y.v == 1) || (y.h == 1 && y.v == 1) ||
                   (y.h == 1 && y.v == 2);
    bool chroma_ok = h.comp[1].h == 1 && h.comp[1].v == 1 && h.comp[2].h == 1 && h.comp[2].v == 1;
    if (!luma_ok || !chroma_ok) return "unsupported chroma subsampling for the NVJPG path";
    return "";
}

class JetsonDecoder {
public:
    JetsonDecoder(size_t pool_size, bool sync_after_map)
        : pool_(std::make_shared<SurfacePool>(pool_size)), sync_after_map_(sync_after_map) {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaFree(nullptr), "CUDA context init");  // create the primary context
    }

    std::shared_ptr<GpuImage> decode(py::object src, const std::string& layout_s) {
        Layout layout = parse_layout(layout_s);
        PyBufferView buf(src);

        JpegHeader hdr;
        std::string why = parse_jpeg_header(buf.data(), buf.size(), hdr);
        if (!why.empty()) throw UnsupportedJpeg("invalid JPEG: " + why);
        why = check_supported(hdr);
        if (!why.empty()) throw UnsupportedJpeg(why);

        std::unique_ptr<Slot> slot;
        void* ptr = nullptr;
        size_t pitch = 0;
        {
            py::gil_scoped_release nogil;
            std::lock_guard<std::mutex> lk(mu_);
            cuda_check(cudaSetDevice(0), "cudaSetDevice");

            int fd = -1;
            unsigned w = 0, h = 0;
            std::string err = jpeg_.decode_to_fd(buf.data(), buf.size(), fd, w, h);
            if (!err.empty()) throw DecodeFailure("NVJPG decode failed: " + err);
            if (static_cast<int>(w) != hdr.width || static_cast<int>(h) != hdr.height)
                throw DecodeFailure("NVJPG output size does not match the JPEG header");

            slot = pool_->take(hdr.width, hdr.height);
            try {
                convert_and_map(fd, *slot, ptr, pitch);
            } catch (...) {
                pool_->give_back(std::move(slot));
                throw;
            }
        }
        return std::make_shared<JetsonImage>(pool_, std::move(slot), ptr, pitch, layout);
    }

private:
    void convert_and_map(int src_fd, Slot& s, void*& ptr, size_t& pitch) {
        NvBufSurface* src = nullptr;
        if (NvBufSurfaceFromFd(src_fd, reinterpret_cast<void**>(&src)) != 0 || src == nullptr)
            throw DecodeFailure("NvBufSurfaceFromFd failed");

        // Session parameters are per thread; set them every call so any thread works.
        NvBufSurfTransformConfigParams cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.compute_mode = NvBufSurfTransformCompute_VIC;
        cfg.gpu_id = 0;
        if (NvBufSurfTransformSetSessionParams(&cfg) != NvBufSurfTransformError_Success)
            throw DecodeFailure("NvBufSurfTransformSetSessionParams(VIC) failed");

        // Crop the source explicitly: the decoder's buffer may be padded/aligned.
        NvBufSurfTransformRect src_rect;
        src_rect.top = 0;
        src_rect.left = 0;
        src_rect.width = static_cast<uint32_t>(s.w);
        src_rect.height = static_cast<uint32_t>(s.h);
        NvBufSurfTransformParams tp;
        std::memset(&tp, 0, sizeof(tp));
        tp.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC;
        tp.transform_filter = NvBufSurfTransformInter_Nearest;
        tp.src_rect = &src_rect;
        if (NvBufSurfTransform(src, s.surf, &tp) != NvBufSurfTransformError_Success)
            throw DecodeFailure("NvBufSurfTransform (YUV -> RGBA on VIC) failed");

        auto egl_image = static_cast<EGLImageKHR>(s.surf->surfaceList[0].mappedAddr.eglImage);
        cuda_check(cudaGraphicsEGLRegisterImage(&s.res, egl_image, cudaGraphicsRegisterFlagsNone),
                   "cudaGraphicsEGLRegisterImage");
        cudaEglFrame frame;
        std::memset(&frame, 0, sizeof(frame));
        cuda_check(cudaGraphicsResourceGetMappedEglFrame(&frame, s.res, 0, 0),
                   "cudaGraphicsResourceGetMappedEglFrame");
        if (frame.frameType != cudaEglFrameTypePitch)
            throw DecodeFailure("EGL frame is not pitch-linear (unexpected surface type)");
        ptr = frame.frame.pPitch[0].ptr;
        pitch = frame.frame.pPitch[0].pitch;
        if (ptr == nullptr || pitch < static_cast<size_t>(s.w) * 4)
            throw DecodeFailure("invalid mapped EGL frame");
        if (sync_after_map_) cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::mutex mu_;
    TegraJpeg jpeg_;
    std::shared_ptr<SurfacePool> pool_;
    bool sync_after_map_;
};

}  // namespace

PYBIND11_MODULE(_jetson, m) {
    m.doc() = "nvjpeg_bridge Jetson backend (NVJPG + VIC + EGL/CUDA interop)";
    register_errors(m);
    bind_gpu_image(m);

    py::class_<JetsonDecoder>(m, "JetsonDecoder", py::module_local())
        .def(py::init<size_t, bool>(), py::arg("pool_size") = 4, py::arg("sync_after_map") = false)
        .def("decode", &JetsonDecoder::decode, py::arg("data"), py::arg("layout") = "HWC",
             "Decode JPEG bytes on NVJPG into a pooled GpuImage (RGB view of an RGBA surface).");

    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    m.attr("cuda_runtime_version") = rt;
}
