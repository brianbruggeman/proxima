"""Batch sweep, onnxruntime CPU EP arm -- companion to
`bge_traffic_sweep.rs`'s section (C). batch in {1, 8, 32} at a fixed
S=128, same synthetic token formula as `traffic_bench.py`/the Rust harness.
"""

import os
import sys
import time

MODEL_PATH_ENV = "BGE_MODEL_PATH"
SEQUENCE_LENGTH = 128
BATCH_SIZES = [1, 8, 32]
RUNS = int(os.environ.get("ONNX_REF_RUNS", "5"))


def coefficient_of_variation(samples: list[float], mean: float) -> float:
    if len(samples) < 2 or mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return (variance**0.5) / mean


def synthetic_tokens(length: int, seed: int) -> list[int]:
    tokens = [101]
    for index in range(length - 2):
        value = (index * 9301 + seed * 49297) % 26000
        tokens.append(2000 + value)
    tokens.append(102)
    return tokens


def main() -> None:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path or not os.path.exists(model_path):
        print(f"skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout", file=sys.stderr)
        sys.exit(1)

    import numpy as np
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])

    print(f"{'batch':>5} | {'total_ms/call':>14} | {'CoV%':>7} | {'ms/sentence':>12} | {'sentences/sec':>14}")
    print("-" * 80)
    for batch in BATCH_SIZES:
        dtype_by_name = {
            meta.name: (np.float32 if "float" in meta.type else np.int64) for meta in session.get_inputs()
        }
        rows = [synthetic_tokens(SEQUENCE_LENGTH, row_index * 31 + 128) for row_index in range(batch)]
        input_ids = np.array(rows, dtype=dtype_by_name["input_ids"])
        attention_mask = np.ones((batch, SEQUENCE_LENGTH), dtype=dtype_by_name["attention_mask"])
        token_type_ids = np.zeros((batch, SEQUENCE_LENGTH), dtype=dtype_by_name["token_type_ids"])
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

        session.run(None, feeds)  # warmup

        times_ms = []
        for _run in range(RUNS):
            start = time.perf_counter()
            session.run(None, feeds)
            times_ms.append((time.perf_counter() - start) * 1000.0)
        mean_ms = sum(times_ms) / len(times_ms)
        cov = coefficient_of_variation(times_ms, mean_ms) * 100.0
        per_sentence = mean_ms / batch
        throughput = batch / (mean_ms / 1000.0)
        print(f"{batch:>5} | {mean_ms:>14.4f} | {cov:>6.2f}% | {per_sentence:>12.4f} | {throughput:>14.1f}")

    print("\ntraffic_batch_bench.py (onnxruntime arm) complete", file=sys.stderr)


if __name__ == "__main__":
    main()
