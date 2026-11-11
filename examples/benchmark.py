"""Compare decode throughput of every available backend.

    uv run python examples/benchmark.py [image.jpg] [--n 200]
"""

import argparse
import io
import time

import torch

import nvjpeg_bridge as njb


def synthetic(w=4000, h=3000):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([x % 256, y % 256, (x + y) % 256], -1) + rng.integers(0, 16, (h, w, 3))
    buf = io.BytesIO()
    Image.fromarray(rgb.clip(0, 255).astype("uint8")).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def bench(dec, data, n, zero_copy):
    op = dec.decode if zero_copy else dec.decode_to_torch
    for _ in range(10):  # warm-up (allocations, pools, lazy init)
        op(data)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        r = op(data)
        del r
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    data = open(args.image, "rb").read() if args.image else synthetic()
    hdr = njb.read_header(data)
    print("image: %dx%d %s, %.0f KB" % (hdr.width, hdr.height, hdr.subsampling, len(data) / 1024))
    for name, (ok, why) in njb.available_backends().items():
        if not ok:
            print("%-12s unavailable: %s" % (name, why))
            continue
        dec = njb.Decoder(backend=name, fallback=False)
        line = "%-12s to_torch %7.1f img/s" % (name, bench(dec, data, args.n, False))
        if name in ("jetson", "cuda"):
            line += "   zero-copy decode %7.1f img/s" % bench(dec, data, args.n, True)
        print(line)


if __name__ == "__main__":
    main()
