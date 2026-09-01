"""Incumbent-arm harness for the BGE-small lane -- onnxruntime CPU EP and
torch/transformers, run on the SAME three real sentences and hardcoded
token-id arrays `proxima-onnx/examples/bge_eval.rs::sentences()` uses.

This replaces the throwaway `ort_bench.py` / `torch_bench.py` scripts that
produced `docs/discipline.md` ROW 195's numbers (5.6296 ms/sentence ort,
10.0360 ms/sentence torch) but were never committed -- an uncommitted
incumbent arm is a phantom cell (per that row's own honest admission). This
script is that harness, committed and reproducible.

Protocol mirrored from bge_eval.rs (read that file first):
  - three sentences, exact token-id arrays copied verbatim below
  - CLS pooling (first token's hidden state), L2-normalize
  - N runs (default 5, matching bge_eval.rs's own BGE_EVAL_RUNS default),
    each run embeds all three sentences once
  - per-run mean = mean of the three per-sentence latencies
  - reported mean/CoV are computed across the N per-run means, same
    population-variance CoV formula bge_eval.rs::coefficient_of_variation
    uses (sqrt(variance)/mean, not sample variance)
  - both arms pinned to single-thread where the API exposes the knob; the
    thread settings actually in effect are read back and printed, never
    assumed
"""

import os
import sys
import time

SENTENCES: list[tuple[str, list[int]]] = [
    ("the cat sat on the mat", [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
    ("a cat is sitting on a mat", [101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
    ("quantum physics explains atomic energy", [101, 8559, 5584, 7607, 9593, 2943, 102]),
]

MODEL_PATH_ENV = "BGE_MODEL_PATH"
HF_MODEL_ID = "BAAI/bge-small-en-v1.5"


def coefficient_of_variation(samples: list[float], mean: float) -> float:
    if len(samples) < 2 or mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return (variance**0.5) / mean


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def run_arm(name: str, threads_desc: str, embed_fn, runs: int) -> dict:
    run_means_ms = []
    last_embeddings: list[list[float]] = []
    for run_index in range(runs):
        per_sentence_ms = []
        embeddings = []
        for _, token_ids in SENTENCES:
            start = time.perf_counter()
            embedding = embed_fn(token_ids)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_sentence_ms.append(elapsed_ms)
            embeddings.append(embedding)
        run_mean = sum(per_sentence_ms) / len(per_sentence_ms)
        run_means_ms.append(run_mean)
        last_embeddings = embeddings
        print(f"{name}: run {run_index} per-sentence={[f'{value:.4f}ms' for value in per_sentence_ms]} mean={run_mean:.4f}ms", file=sys.stderr)

    mean_ms = sum(run_means_ms) / len(run_means_ms)
    cov = coefficient_of_variation(run_means_ms, mean_ms)

    similar = sum(a * b for a, b in zip(last_embeddings[0], last_embeddings[1]))
    dissimilar_a = sum(a * b for a, b in zip(last_embeddings[0], last_embeddings[2]))
    dissimilar_b = sum(a * b for a, b in zip(last_embeddings[1], last_embeddings[2]))

    return {
        "arm": name,
        "mean_ms": mean_ms,
        "cov_pct": cov * 100.0,
        "n_runs": runs,
        "threads": threads_desc,
        "run_means_ms": run_means_ms,
        "cosine_similar": similar,
        "cosine_dissimilar_a": dissimilar_a,
        "cosine_dissimilar_b": dissimilar_b,
        "embedding_preview": last_embeddings[0][:8],
    }


def build_ort_embed(model_path: str, warmup: int):
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])

    input_names = {node.name for node in session.get_inputs()}

    def embed(token_ids: list[int]) -> list[float]:
        sequence_length = len(token_ids)
        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = [[float(t) for t in token_ids]] if _wants_float(session, "input_ids") else [[int(t) for t in token_ids]]
        if "attention_mask" in input_names:
            feeds["attention_mask"] = [[1.0] * sequence_length] if _wants_float(session, "attention_mask") else [[1] * sequence_length]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = [[0.0] * sequence_length] if _wants_float(session, "token_type_ids") else [[0] * sequence_length]

        import numpy as np

        np_feeds = {}
        for input_meta in session.get_inputs():
            dtype = np.float32 if "float" in input_meta.type else np.int64
            np_feeds[input_meta.name] = np.array(feeds[input_meta.name], dtype=dtype)

        outputs = session.run(None, np_feeds)
        last_hidden_state = outputs[0]
        cls = last_hidden_state[0][0].tolist()
        return l2_normalize(cls)

    actual_intra = session_options.intra_op_num_threads
    actual_inter = session_options.inter_op_num_threads
    print(f"onnxruntime thread settings in effect: intra_op_num_threads={actual_intra} inter_op_num_threads={actual_inter} execution_mode=SEQUENTIAL", file=sys.stderr)

    for _ in range(warmup):
        embed(SENTENCES[0][1])

    return embed, f"intra_op=1 inter_op=1 sequential ({warmup}-pass warmup)"


def _wants_float(session, name: str) -> bool:
    for input_meta in session.get_inputs():
        if input_meta.name == name:
            return "float" in input_meta.type
    return False


def build_torch_embed(warmup: int):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModel

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        # torch refuses to change interop threads once parallel work has
        # already started in this process -- report what is actually in
        # effect rather than pretending the call succeeded.
        print(f"torch.set_num_interop_threads(1) rejected: {error}", file=sys.stderr)

    model = AutoModel.from_pretrained(HF_MODEL_ID)
    model.eval()

    def embed(token_ids: list[int]) -> list[float]:
        input_ids = torch.tensor([token_ids], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = output.last_hidden_state[0, 0, :].tolist()
        return l2_normalize(cls)

    actual_intra = torch.get_num_threads()
    actual_interop = torch.get_num_interop_threads()
    print(f"torch thread settings in effect: num_threads={actual_intra} num_interop_threads={actual_interop}", file=sys.stderr)

    for pass_index in range(warmup):
        embed(SENTENCES[0][1])
        print(f"torch: warmup pass {pass_index} complete", file=sys.stderr)

    return embed, f"num_threads=1 num_interop_threads={actual_interop} ({warmup}-pass warmup)"


def main() -> None:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path or not os.path.exists(model_path):
        print(f"skipping onnxruntime arm: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout", file=sys.stderr)
        sys.exit(1)

    runs = int(os.environ.get("ONNX_REF_RUNS", "5"))
    ort_warmup = int(os.environ.get("ONNX_REF_ORT_WARMUP", "1"))
    torch_warmup = int(os.environ.get("ONNX_REF_TORCH_WARMUP", "3"))

    results = []

    ort_embed, ort_threads_desc = build_ort_embed(model_path, ort_warmup)
    results.append(run_arm("onnxruntime", ort_threads_desc, ort_embed, runs))

    torch_embed, torch_threads_desc = build_torch_embed(torch_warmup)
    results.append(run_arm("torch", torch_threads_desc, torch_embed, runs))

    print()
    print("=== incumbent arm summary (BGE-small lane) ===")
    print(f"model.onnx = {model_path}")
    print(f"{'arm':<12} | {'ms/sentence':>12} | {'CoV%':>7} | {'n_runs':>6} | threads")
    print("-" * 80)
    for result in results:
        print(f"{result['arm']:<12} | {result['mean_ms']:>12.4f} | {result['cov_pct']:>6.2f}% | {result['n_runs']:>6} | {result['threads']}")
    print()
    for result in results:
        print(
            f"{result['arm']}: cosine(A,B similar)={result['cosine_similar']:.6f} "
            f"cosine(A,C dissimilar)={result['cosine_dissimilar_a']:.6f} "
            f"cosine(B,C dissimilar)={result['cosine_dissimilar_b']:.6f} "
            f"embedding[A][:8]={[f'{value:.6f}' for value in result['embedding_preview']]}"
        )
        assert result["cosine_similar"] > result["cosine_dissimilar_a"], f"{result['arm']}: similar pair should score higher than dissimilar pair A"
        assert result["cosine_similar"] > result["cosine_dissimilar_b"], f"{result['arm']}: similar pair should score higher than dissimilar pair B"
    print("sanity check passed for both arms: similar sentence pair scores higher than dissimilar pairs")


if __name__ == "__main__":
    main()
