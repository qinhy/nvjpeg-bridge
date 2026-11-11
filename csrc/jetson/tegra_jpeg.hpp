// Thin C++ wrapper around the Tegra libjpeg API (libnvjpeg.so from the L4T
// "tegra"/"nvidia" lib dir). Kept in its own translation unit so libjpeg's
// jconfig.h never meets Python's pyconfig.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace njb {

class TegraJpeg {
public:
    TegraJpeg();
    ~TegraJpeg();
    TegraJpeg(const TegraJpeg&) = delete;
    TegraJpeg& operator=(const TegraJpeg&) = delete;

    // Decode on the NVJPG engine into a hardware buffer. On success returns an
    // empty string and sets fd/width/height; the fd belongs to the decoder and
    // is overwritten by the next call. On failure returns the libjpeg message
    // and resets the decoder (the process is never terminated).
    std::string decode_to_fd(const uint8_t* data, size_t len, int& fd, unsigned& width, unsigned& height);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace njb
