"""Max-abs-delta between the Rust harness's dumped embeddings
(`ours_neon_S{length}.txt` / `ours_accel_S{length}.txt`) and this
directory's onnxruntime arm (`ort_S{length}.txt`), all written by
`write_embedding`/`traffic_bench.py`'s own identical plain-text format
(space-separated floats, one line) into the same `BGE_TRAFFIC_OUT_DIR`.
"""

import os
import sys

SEQUENCE_LENGTHS = [8, 32, 64, 128, 256, 512]


def read_embedding(path: str) -> list[float]:
    with open(path) as handle:
        return [float(value) for value in handle.read().split()]


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BGE_TRAFFIC_OUT_DIR", "")
    if not out_dir:
        print("usage: diff_embeddings.py <out_dir>", file=sys.stderr)
        sys.exit(1)

    print(f"{'S':>5} | {'max_abs_diff(neon, ort)':>24} | {'max_abs_diff(accel, ort)':>24}")
    print("-" * 65)
    for length in SEQUENCE_LENGTHS:
        ort_path = os.path.join(out_dir, f"ort_S{length}.txt")
        neon_path = os.path.join(out_dir, f"ours_neon_S{length}.txt")
        accel_path = os.path.join(out_dir, f"ours_accel_S{length}.txt")
        if not os.path.exists(ort_path):
            print(f"{length:>5} | missing ort embedding ({ort_path})")
            continue
        ort_embedding = read_embedding(ort_path)
        row = f"{length:>5} |"
        if os.path.exists(neon_path):
            neon_embedding = read_embedding(neon_path)
            row += f" {max_abs_diff(neon_embedding, ort_embedding):>24.3e} |"
        else:
            row += f" {'missing':>24} |"
        if os.path.exists(accel_path):
            accel_embedding = read_embedding(accel_path)
            row += f" {max_abs_diff(accel_embedding, ort_embedding):>24.3e}"
        else:
            row += f" {'missing':>24}"
        print(row)


if __name__ == "__main__":
    main()
