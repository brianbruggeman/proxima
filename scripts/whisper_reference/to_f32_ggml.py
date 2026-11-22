#!/usr/bin/env python3
"""Rewrite a ggml whisper checkpoint with every F16 tensor stored as F32.

whisper.cpp's published ggml-tiny.bin is ftype 1 (F16 weights), while the HF
safetensors this repo reads are F32. An F16 oracle and an F32 implementation
cannot be compared at the F32-derived ceilings in SPEC.md. This tool widens
the published file without changing a value: the reader still sees the same
tensor names, dims and order, and the data is the same f16 values exactly
representable in f32. It is a format rewrite, not a conversion or a re-train.

usage: to_f32_ggml.py <f16-or-f32.ggml> <out.ggml>
"""

import struct
import sys

MAGIC = 0x67676D6C
F32 = 0
F16 = 1
TYPE_SIZE = {F32: 4, F16: 2}


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise SystemExit(f"short read: wanted {n}, got {len(data)}")
    return data


def main(src_path, dst_path):
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        magic = struct.unpack("<I", read_exact(src, 4))[0]
        if magic != MAGIC:
            raise SystemExit(f"bad magic {magic:#x}")
        dst.write(struct.pack("<I", magic))

        hparams = struct.unpack("<11i", read_exact(src, 11 * 4))
        # (n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
        #  n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype)
        dst.write(struct.pack("<11i", *hparams[:10], F32))

        # mel filters
        n_mel, n_fft = struct.unpack("<2i", read_exact(src, 8))
        dst.write(struct.pack("<2i", n_mel, n_fft))
        dst.write(read_exact(src, n_mel * n_fft * 4))

        # vocab
        (n_vocab,) = struct.unpack("<i", read_exact(src, 4))
        dst.write(struct.pack("<i", n_vocab))
        for _ in range(n_vocab):
            (length,) = struct.unpack("<i", read_exact(src, 4))
            dst.write(struct.pack("<i", length))
            dst.write(read_exact(src, length))

        # tensors
        converted = 0
        while True:
            head = src.read(12)
            if not head:
                break
            if len(head) != 12:
                raise SystemExit("truncated tensor header")
            n_dims, length, ttype = struct.unpack("<3i", head)
            dims = struct.unpack(f"<{n_dims}i", read_exact(src, n_dims * 4))
            name = read_exact(src, length)
            elements = 1
            for dim in dims:
                elements *= dim
            if ttype == F16:
                raw = read_exact(src, elements * 2)
                values = struct.unpack(f"<{elements}e", raw)
                payload = struct.pack(f"<{elements}f", *values)
                ttype = F32
                converted += 1
            elif ttype == F32:
                payload = read_exact(src, elements * 4)
            else:
                raise SystemExit(f"tensor {name!r} has unsupported type {ttype}")
            dst.write(struct.pack("<3i", n_dims, length, ttype))
            dst.write(struct.pack(f"<{n_dims}i", *dims))
            dst.write(name)
            dst.write(payload)

        print(f"converted={converted} tensors; wrote {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
