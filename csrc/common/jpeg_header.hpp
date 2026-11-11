// Small, dependency-free JPEG header parser.
//
// Used to validate input *before* it reaches a hardware decoder, so that
// unsupported or malformed streams are rejected with a Python exception
// (and can fall back to another backend) instead of reaching code paths that
// may abort the process.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace njb {

struct JpegComponent {
    int id = 0;
    int h = 0;  // horizontal sampling factor
    int v = 0;  // vertical sampling factor
};

struct JpegHeader {
    int width = 0;
    int height = 0;
    int components = 0;
    int precision = 0;
    bool progressive = false;
    bool arithmetic = false;
    bool lossless = false;
    int adobe_transform = -1;  // -1: no Adobe APP14 marker
    JpegComponent comp[4];
};

// Returns an empty string on success, otherwise a human-readable reason.
inline std::string parse_jpeg_header(const uint8_t* p, size_t n, JpegHeader& out) {
    if (p == nullptr || n < 4) return "input is empty or too short";
    if (p[0] != 0xFF || p[1] != 0xD8) return "missing SOI marker (not a JPEG)";

    size_t i = 2;
    while (i + 1 < n) {
        if (p[i] != 0xFF) return "corrupt marker structure";
        uint8_t m = p[i + 1];
        if (m == 0xFF) {  // fill byte
            ++i;
            continue;
        }
        i += 2;
        // Stand-alone markers without a length field.
        if (m == 0x01 || m == 0xD8 || (m >= 0xD0 && m <= 0xD7)) continue;
        if (m == 0xD9) return "EOI reached before frame header";
        if (i + 2 > n) return "truncated segment length";
        size_t len = (static_cast<size_t>(p[i]) << 8) | p[i + 1];
        if (len < 2 || i + len > n) return "truncated segment";
        const uint8_t* s = p + i + 2;  // segment payload
        size_t sl = len - 2;

        if (m == 0xEE && sl >= 12 && s[0] == 'A' && s[1] == 'd' && s[2] == 'o' && s[3] == 'b' &&
            s[4] == 'e') {
            out.adobe_transform = s[11];
        }

        bool is_sof = (m >= 0xC0 && m <= 0xCF) && m != 0xC4 && m != 0xC8 && m != 0xCC;
        if (is_sof) {
            if (sl < 6) return "truncated frame header";
            out.precision = s[0];
            out.height = (s[1] << 8) | s[2];
            out.width = (s[3] << 8) | s[4];
            out.components = s[5];
            out.progressive = (m == 0xC2 || m == 0xC6 || m == 0xCA || m == 0xCE);
            out.arithmetic = (m >= 0xC9);
            out.lossless = (m == 0xC3 || m == 0xC7 || m == 0xCB || m == 0xCF);
            if (out.components < 1 || sl < 6 + 3 * static_cast<size_t>(out.components))
                return "invalid component table";
            for (int c = 0; c < out.components && c < 4; ++c) {
                out.comp[c].id = s[6 + 3 * c];
                out.comp[c].h = s[7 + 3 * c] >> 4;
                out.comp[c].v = s[7 + 3 * c] & 0x0F;
            }
            if (out.width <= 0 || out.height <= 0)
                return "zero image dimension (DNL marker not supported)";
            return "";
        }
        if (m == 0xDA) return "scan data before frame header";
        i += len;
    }
    return "no frame header found";
}

}  // namespace njb
