#!/usr/bin/env python3
"""extract real IQ4_XS and IQ3_S blocks from the flint checkpoint's own GGUF.

The oracle for a dequantizer port is the incumbent's decoder over the incumbent's
own bytes (guiding-principles principle 9/14), so this reads the tensor directory
directly -- no proxima-gguf code -- and writes the exact block bytes the C oracle
in `iql_oracle.c` decodes.

usage: iql_extract.py <gguf> <out-dir> [blocks]
"""
import json
import pathlib
import struct
import sys

GGML_TYPE_NAMES = {23: "iq4_xs", 21: "iq3_s"}


def read_gguf_header(path):
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    version, = struct.unpack_from("<I", data, 4)
    tensor_count, kv_count = struct.unpack_from("<QQ", data, 8)
    cursor = 24

    def read_string():
        nonlocal cursor
        (length,) = struct.unpack_from("<Q", data, cursor)
        cursor += 8
        value = data[cursor : cursor + length]
        cursor += length
        return value.decode("utf-8", "replace")

    def skip_value(value_type):
        nonlocal cursor
        scalar = {
            0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
            10: 8, 11: 8, 12: 8,
        }
        if value_type == 8:
            read_string()
        elif value_type == 9:
            (item_type,) = struct.unpack_from("<I", data, cursor)
            cursor += 4
            (count,) = struct.unpack_from("<Q", data, cursor)
            cursor += 8
            if item_type == 8:
                for _ in range(count):
                    read_string()
            else:
                cursor += count * scalar[item_type]
        else:
            cursor += scalar[value_type]

    metadata = {}
    for _ in range(kv_count):
        key = read_string()
        (value_type,) = struct.unpack_from("<I", data, cursor)
        cursor += 4
        start = cursor
        skip_value(value_type)
        metadata.setdefault(key, (value_type, start, cursor))

    tensors = []
    for _ in range(tensor_count):
        name = read_string()
        (n_dims,) = struct.unpack_from("<I", data, cursor)
        cursor += 4
        dims = list(struct.unpack_from(f"<{n_dims}Q", data, cursor))
        cursor += 8 * n_dims
        (ggml_type,) = struct.unpack_from("<I", data, cursor)
        cursor += 4
        (offset,) = struct.unpack_from("<Q", data, cursor)
        cursor += 8
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})

    alignment = 32
    if "general.alignment" in metadata:
        value_type, start, _ = metadata["general.alignment"]
        if value_type == 4:
            alignment, = struct.unpack_from("<I", data, start)
    data_start = (cursor + alignment - 1) // alignment * alignment
    return data, data_start, version, tensors


def main():
    gguf, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    data, data_start, version, tensors = read_gguf_header(gguf)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = {}
    for tensor in tensors:
        name = GGML_TYPE_NAMES.get(tensor["type"])
        if name and name not in picks:
            picks[name] = tensor
    if len(picks) != 2:
        raise SystemExit(f"expected one iq4_xs and one iq3_s tensor, found {sorted(picks)}")

    block_bytes = {"iq4_xs": 136, "iq3_s": 110}
    manifest = {"gguf": str(gguf), "version": version, "blocks": blocks, "tensors": {}}
    for name, tensor in picks.items():
        run = block_bytes[name] * blocks
        start = data_start + tensor["offset"]
        chunk = data[start : start + run]
        if len(chunk) != run:
            raise SystemExit(f"{tensor['name']}: short read")
        (out_dir / f"{name}.bin").write_bytes(chunk)
        manifest["tensors"][name] = {
            "name": tensor["name"],
            "dims": tensor["dims"],
            "first_block_offset": tensor["offset"],
            "blocks": blocks,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
