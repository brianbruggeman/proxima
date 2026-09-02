"""Real-traffic shape sweep, onnxruntime CPU EP arm -- companion to
`proxima-onnx/examples/bge_traffic_sweep.rs`'s section (A). Same
`InferenceSession` protocol as `bench.py` (intra_op=1, inter_op=1,
sequential, single session reused across shapes), same synthetic token
formula as the Rust harness's own `synthetic_tokens` (see that file's own
doc) so both arms embed the EXACT same input at a given `(length, seed)`:

    interior_id(i, seed) = 2000 + ((i * 9301 + seed * 49297) % 26000)
    tokens(length, seed) = [101] + [interior_id(i, seed) for i in range(length - 2)] + [102]

Writes each S's embedding to `BGE_TRAFFIC_OUT_DIR/ort_S{length}.txt` (same
plain-text format the Rust harness's own `write_embedding` uses) so a
separate diff step can compute max-abs-delta without re-running either arm.
"""

import os
import sys
import time

MODEL_PATH_ENV = "BGE_MODEL_PATH"
OUT_DIR_ENV = "BGE_TRAFFIC_OUT_DIR"
SEQUENCE_LENGTHS = [8, 32, 64, 128, 256, 512]
RUNS = int(os.environ.get("ONNX_REF_RUNS", "5"))


def coefficient_of_variation(samples: list[float], mean: float) -> float:
    if len(samples) < 2 or mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return (variance**0.5) / mean


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def synthetic_tokens(length: int, seed: int) -> list[int]:
    assert length >= 3
    tokens = [101]
    for index in range(length - 2):
        value = (index * 9301 + seed * 49297) % 26000
        tokens.append(2000 + value)
    tokens.append(102)
    return tokens


def write_embedding(out_dir: str, label: str, embedding: list[float]) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.txt")
    with open(path, "w") as handle:
        handle.write(" ".join(f"{value:.9f}" for value in embedding) + "\n")


def build_ort_embed(model_path: str):
    import numpy as np
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])

    dtype_by_name = {meta.name: (np.float32 if "float" in meta.type else np.int64) for meta in session.get_inputs()}

    def embed(token_ids: list[int]) -> list[float]:
        sequence_length = len(token_ids)
        input_ids = np.array([token_ids], dtype=dtype_by_name["input_ids"])
        attention_mask = np.array([[1] * sequence_length], dtype=dtype_by_name["attention_mask"])
        token_type_ids = np.array([[0] * sequence_length], dtype=dtype_by_name["token_type_ids"])
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        outputs = session.run(None, feeds)
        last_hidden_state = outputs[0]
        cls = last_hidden_state[0][0].tolist()
        return l2_normalize(cls)

    print(
        f"onnxruntime thread settings in effect: intra_op_num_threads={session_options.intra_op_num_threads} "
        f"inter_op_num_threads={session_options.inter_op_num_threads} execution_mode=SEQUENTIAL",
        file=sys.stderr,
    )
    return embed


def main() -> None:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path or not os.path.exists(model_path):
        print(f"skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout", file=sys.stderr)
        sys.exit(1)
    out_dir = os.environ.get(OUT_DIR_ENV, "")

    embed = build_ort_embed(model_path)
    # 1-pass warmup, matching bench.py's own onnxruntime warmup depth.
    embed(synthetic_tokens(8, 8))

    print(f"{'S':>5} | {'ms/sentence':>12} | {'CoV%':>7} | {'ms/token':>10} | n_runs")
    print("-" * 70)
    for length in SEQUENCE_LENGTHS:
        tokens = synthetic_tokens(length, length)
        run_means_ms = []
        last_embedding: list[float] = []
        for _run in range(RUNS):
            start = time.perf_counter()
            embedding = embed(tokens)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            run_means_ms.append(elapsed_ms)
            last_embedding = embedding
        mean_ms = sum(run_means_ms) / len(run_means_ms)
        cov = coefficient_of_variation(run_means_ms, mean_ms) * 100.0
        print(f"{length:>5} | {mean_ms:>12.4f} | {cov:>6.2f}% | {mean_ms / length:>10.5f} | {RUNS}")
        write_embedding(out_dir, f"ort_S{length}", last_embedding)

    print("\ntraffic_bench.py (onnxruntime arm) complete", file=sys.stderr)


if __name__ == "__main__":
    main()
