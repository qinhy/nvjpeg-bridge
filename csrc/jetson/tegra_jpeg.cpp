// See tegra_jpeg.hpp. Must be compiled against <MMAPI>/include/libjpeg-8b.
#include "tegra_jpeg.hpp"

#include <csetjmp>
#include <cstdio>  // must precede jpeglib.h
#include <cstring>
#include <stdexcept>

#include "jpeglib.h"

#ifndef TEGRA_ACCELERATE
#error "Wrong jpeglib.h: compile against <jetson_multimedia_api>/include/libjpeg-8b (it defines TEGRA_ACCELERATE)."
#endif

namespace njb {
namespace {

struct ErrMgr {
    jpeg_error_mgr pub;
    jmp_buf jump;
    char msg[JMSG_LENGTH_MAX];
};

extern "C" void njb_error_exit(j_common_ptr cinfo) {
    ErrMgr* e = reinterpret_cast<ErrMgr*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, e->msg);
    longjmp(e->jump, 1);  // libjpeg's default handler would call exit()
}
extern "C" void njb_output_message(j_common_ptr) {}

// C-style on purpose: no object with a destructor lives between setjmp() and
// a possible longjmp(). Mirrors NvJPEGDecoder::decodeToFd from the MMAPI samples.
int decode_impl(jpeg_decompress_struct* cinfo, ErrMgr* err, const unsigned char* data, unsigned long len,
                int* fd, unsigned* w, unsigned* h) {
    if (setjmp(err->jump)) return -1;
    cinfo->out_color_space = JCS_YCbCr;
    jpeg_mem_src(cinfo, const_cast<unsigned char*>(data), len);
    cinfo->out_color_space = JCS_YCbCr;
    (void)jpeg_read_header(cinfo, TRUE);
    cinfo->out_color_space = JCS_YCbCr;
    cinfo->IsVendorbuf = TRUE;  // decode into a hardware (NvBufSurface) buffer
    jpeg_start_decompress(cinfo);
    jpeg_read_raw_data(cinfo, NULL, cinfo->comp_info[0].v_samp_factor * DCTSIZE);
    jpeg_finish_decompress(cinfo);
    *w = cinfo->image_width;
    *h = cinfo->image_height;
    *fd = cinfo->fd;
    return 0;
}

int create_impl(jpeg_decompress_struct* cinfo, ErrMgr* err) {
    std::memset(cinfo, 0, sizeof(*cinfo));
    std::memset(err, 0, sizeof(*err));
    cinfo->err = jpeg_std_error(&err->pub);
    err->pub.error_exit = njb_error_exit;
    err->pub.output_message = njb_output_message;
    if (setjmp(err->jump)) return -1;
    jpeg_create_decompress(cinfo);
    return 0;
}

void destroy_impl(jpeg_decompress_struct* cinfo, ErrMgr* err) {
    if (setjmp(err->jump) == 0) jpeg_destroy_decompress(cinfo);
}

}  // namespace

struct TegraJpeg::Impl {
    jpeg_decompress_struct cinfo;
    ErrMgr err;
    bool alive = false;

    std::string create() {
        if (create_impl(&cinfo, &err) != 0) return std::string("jpeg_create_decompress: ") + err.msg;
        alive = true;
        return "";
    }
    void destroy() {
        if (alive) destroy_impl(&cinfo, &err);
        alive = false;
    }
};

TegraJpeg::TegraJpeg() : impl_(new Impl) {
    std::string e = impl_->create();
    if (!e.empty()) throw std::runtime_error(e);
}

TegraJpeg::~TegraJpeg() { impl_->destroy(); }

std::string TegraJpeg::decode_to_fd(const uint8_t* data, size_t len, int& fd, unsigned& width,
                                    unsigned& height) {
    if (!impl_->alive) {
        std::string e = impl_->create();
        if (!e.empty()) return e;
    }
    int f = -1;
    unsigned w = 0, h = 0;
    if (decode_impl(&impl_->cinfo, &impl_->err, data, static_cast<unsigned long>(len), &f, &w, &h) != 0) {
        std::string msg = impl_->err.msg[0] ? impl_->err.msg : "unknown libjpeg error";
        // Recreate the decompressor so a bad image cannot poison later calls.
        impl_->destroy();
        impl_->create();
        return msg;
    }
    if (f < 0) return "decoder returned no hardware buffer";
    fd = f;
    width = w;
    height = h;
    return "";
}

}  // namespace njb
