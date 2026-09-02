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
    assumed, and the intra-op thread count is a hard assertion, not a
    request -- ROW 217's torch arm produced two irreconcilable numbers
    (~10ms contaminated by a concurrent build, ~16ms alone) on a nominally
    identical single-thread config, so a config that silently did not take
    is exactly the failure mode this harness now refuses to paper over

Process isolation (default, ONNX_REF_MODE=isolated):
  a follow-up investigation (thread config verified 1/1 in every condition)
  found the torch arm's remaining variance was NOT thread config -- it was
  arm ORDER inside one process. torch run after the ORT arm in the same
  process: 9.5136 ms/sentence. torch run alone in its own process: 16.7208
  ms/sentence. Same quiet box, same verified thread config, 1.76x apart,
  driven purely by ORT's process-local warmup effects on torch's later
  reads. An incumbent arm must measure the incumbent, not the incumbent
  warmed by a different framework's process state -- so each arm now runs
  in its own subprocess by default, and the parent asserts each child's PID
  to prove isolation actually happened rather than merely being requested.
  ONNX_REF_MODE=same-process reproduces the OLD one-process convention on
  demand, as a labeled control, never as the reported number.
"""

import os

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_thread_env_var] = "1"

import json
import pathlib
import subprocess
import sys
import time

SENTENCES: list[tuple[str, list[int]]] = [
    ("the cat sat on the mat", [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
    ("a cat is sitting on a mat", [101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
    ("quantum physics explains atomic energy", [101, 8559, 5584, 7607, 9593, 2943, 102]),
]

MODEL_PATH_ENV = "BGE_MODEL_PATH"
HF_MODEL_ID = "BAAI/bge-small-en-v1.5"

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def coefficient_of_variation(samples: list[float], mean: float) -> float:
    if len(samples) < 2 or mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return (variance**0.5) / mean


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def load_snapshot(label: str) -> None:
    one_min, five_min, fifteen_min = os.getloadavg()
    print(
        f"loadavg[{label}]: 1m={one_min:.2f} 5m={five_min:.2f} 15m={fifteen_min:.2f} "
        f"wall={time.strftime('%H:%M:%S')} pid={os.getpid()}",
        file=sys.stderr,
    )


def env_snapshot(label: str) -> None:
    values = {name: os.environ.get(name, "<unset>") for name in THREAD_ENV_VARS}
    print(f"thread env vars[{label}]: {values}", file=sys.stderr)


def print_sentence_tokens() -> None:
    for sentence, token_ids in SENTENCES:
        print(f"sentence tokens: {token_ids!r} <- {sentence!r}", file=sys.stderr)


def run_arm(name: str, threads_desc: str, embed_fn, runs: int, probe_fn=None) -> dict:
    run_means_ms = []
    last_embeddings: list[list[float]] = []
    load_snapshot(f"{name} start")
    for run_index in range(runs):
        if probe_fn is not None:
            probe_fn(f"{name} run {run_index} pre-timing")
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
    load_snapshot(f"{name} end")

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
        "worker_pid": os.getpid(),
    }


def build_ort_embed(model_path: str, warmup: int):
    import numpy as np
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

        np_feeds = {}
        for input_meta in session.get_inputs():
            dtype = np.float32 if "float" in input_meta.type else np.int64
            np_feeds[input_meta.name] = np.array(feeds[input_meta.name], dtype=dtype)

        outputs = session.run(None, np_feeds)
        last_hidden_state = outputs[0]
        cls = last_hidden_state[0][0].tolist()
        return l2_normalize(cls)

    # NOTE this is a tautological check, not an independent assertion:
    # onnxruntime's Python SessionOptions has no readback distinct from what
    # this function just assigned -- session_options.intra_op_num_threads
    # returns the same object we set two lines above, never a value read
    # from the C++ session's actual thread pool. torch's equivalent check
    # below (torch.get_num_threads()) IS independent -- it reads live
    # global interpreter state, not an echo of the setter. Keep this check
    # for symmetry/documentation, but do not report it as proof ORT's
    # thread pool honored the request; ORT's Python API exposes no such
    # proof.
    actual_intra = session_options.intra_op_num_threads
    actual_inter = session_options.inter_op_num_threads
    if actual_intra != 1 or actual_inter != 1:
        raise RuntimeError(
            f"onnxruntime thread config did not take: intra_op_num_threads={actual_intra} "
            f"inter_op_num_threads={actual_inter}, expected 1/1 -- refusing to report a number "
            f"taken under a config the harness did not verify"
        )
    print(f"onnxruntime thread settings requested (tautological readback, see comment above): intra_op_num_threads={actual_intra} inter_op_num_threads={actual_inter} execution_mode=SEQUENTIAL", file=sys.stderr)

    for _ in range(warmup):
        embed(SENTENCES[0][1])

    def probe(label: str) -> None:
        print(
            f"ort probe[{label}]: intra_op_num_threads={session_options.intra_op_num_threads} "
            f"inter_op_num_threads={session_options.inter_op_num_threads}",
            file=sys.stderr,
        )

    return embed, f"intra_op=1 inter_op=1 sequential ({warmup}-pass warmup)", probe


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
    if actual_intra != 1:
        raise RuntimeError(
            f"torch thread config did not take: torch.get_num_threads()={actual_intra}, "
            f"expected 1 -- refusing to report a number taken under a config the harness "
            f"did not verify (env at import time: "
            f"{ {name: os.environ.get(name, '<unset>') for name in THREAD_ENV_VARS} })"
        )
    print(f"torch thread settings in effect (verified live, independent readback): num_threads={actual_intra} num_interop_threads={actual_interop}", file=sys.stderr)
    print(f"torch.__config__.parallel_info():\n{torch.__config__.parallel_info()}", file=sys.stderr)
    env_snapshot("torch config time")

    for pass_index in range(warmup):
        embed(SENTENCES[0][1])
        print(f"torch: warmup pass {pass_index} complete", file=sys.stderr)

    def probe(label: str) -> None:
        live_intra = torch.get_num_threads()
        live_interop = torch.get_num_interop_threads()
        print(f"torch probe[{label}]: num_threads={live_intra} num_interop_threads={live_interop}", file=sys.stderr)
        env_snapshot(label)
        if live_intra != 1:
            raise RuntimeError(f"torch thread config drifted mid-run at {label}: num_threads={live_intra}")

    return embed, f"num_threads=1 num_interop_threads={actual_interop} ({warmup}-pass warmup)", probe


def compute_results(model_path: str, runs: int, ort_warmup: int, torch_warmup: int, only_arm: str | None) -> list[dict]:
    results = []
    if only_arm in (None, "ort"):
        ort_embed, ort_threads_desc, ort_probe = build_ort_embed(model_path, ort_warmup)
        results.append(run_arm("onnxruntime", ort_threads_desc, ort_embed, runs, probe_fn=ort_probe))
    if only_arm in (None, "torch"):
        torch_embed, torch_threads_desc, torch_probe = build_torch_embed(torch_warmup)
        results.append(run_arm("torch", torch_threads_desc, torch_embed, runs, probe_fn=torch_probe))
    return results


def print_summary(model_path: str, results: list[dict], pid_by_arm: dict[str, int] | None = None) -> None:
    print()
    print("=== incumbent arm summary (BGE-small lane) ===")
    print(f"model.onnx = {model_path}")
    if pid_by_arm:
        print(f"{'arm':<12} | {'ms/sentence':>12} | {'CoV%':>7} | {'n_runs':>6} | {'pid':>7} | threads")
        print("-" * 100)
        for result in results:
            pid = pid_by_arm.get(result["arm"], result.get("worker_pid", "-"))
            print(f"{result['arm']:<12} | {result['mean_ms']:>12.4f} | {result['cov_pct']:>6.2f}% | {result['n_runs']:>6} | {pid!s:>7} | {result['threads']}")
    else:
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
    print("sanity check passed for both arms (or the one arm run): similar sentence pair scores higher than dissimilar pairs" if results else "no arms run")


def run_isolated_arm(arm_name: str, model_path: str, runs: int, ort_warmup: int, torch_warmup: int) -> tuple[dict, int]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        result_path = pathlib.Path(tmp_dir) / f"{arm_name}_result.json"
        env = os.environ.copy()
        env["ONNX_REF_ONLY_ARM"] = arm_name
        env["ONNX_REF_RESULT_JSON"] = str(result_path)
        env["BGE_MODEL_PATH"] = model_path
        env["ONNX_REF_RUNS"] = str(runs)
        env["ONNX_REF_ORT_WARMUP"] = str(ort_warmup)
        env["ONNX_REF_TORCH_WARMUP"] = str(torch_warmup)
        env.pop("ONNX_REF_MODE", None)

        script_path = str(pathlib.Path(__file__).resolve())
        print(f"\n--- spawning isolated subprocess for arm={arm_name} ---", file=sys.stderr)
        process = subprocess.Popen([sys.executable, script_path], env=env)
        pid = process.pid
        print(f"arm={arm_name} subprocess pid={pid}", file=sys.stderr)
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"arm={arm_name} subprocess (pid={pid}) exited with code {returncode}")
        if not result_path.exists():
            raise RuntimeError(
                f"arm={arm_name} subprocess (pid={pid}) exited 0 but wrote no result json at "
                f"{result_path} -- N==0 masquerading as success, refusing to report a phantom result"
            )
        result = json.loads(result_path.read_text())
        if result.get("worker_pid") != pid:
            raise RuntimeError(
                f"arm={arm_name} result json worker_pid={result.get('worker_pid')} does not match "
                f"the subprocess pid the parent spawned ({pid}) -- isolation was not proven"
            )
        return result, pid


def run_all_isolated(model_path: str, runs: int, ort_warmup: int, torch_warmup: int, only_arm: str | None) -> tuple[list[dict], dict[str, int]]:
    arms = [arm for arm in ("ort", "torch") if only_arm in (None, arm)]
    results = []
    pid_by_arm: dict[str, int] = {}
    for arm in arms:
        result, pid = run_isolated_arm(arm, model_path, runs, ort_warmup, torch_warmup)
        results.append(result)
        pid_by_arm[result["arm"]] = pid
    return results, pid_by_arm


def main() -> None:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path or not os.path.exists(model_path):
        print(f"skipping onnxruntime arm: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout", file=sys.stderr)
        sys.exit(1)

    runs = int(os.environ.get("ONNX_REF_RUNS", "5"))
    ort_warmup = int(os.environ.get("ONNX_REF_ORT_WARMUP", "1"))
    torch_warmup = int(os.environ.get("ONNX_REF_TORCH_WARMUP", "3"))
    only_arm = os.environ.get("ONNX_REF_ONLY_ARM")
    result_json_path = os.environ.get("ONNX_REF_RESULT_JSON")
    mode = os.environ.get("ONNX_REF_MODE", "isolated")

    env_snapshot("process start")
    print(f"pid={os.getpid()}", file=sys.stderr)
    print_sentence_tokens()

    if only_arm is not None or result_json_path is not None:
        # worker mode -- either a manual diagnostic invocation
        # (ONNX_REF_ONLY_ARM=torch python bench.py) or the child a parent
        # isolator spawned via run_isolated_arm above.
        results = compute_results(model_path, runs, ort_warmup, torch_warmup, only_arm)
        if result_json_path:
            if len(results) != 1:
                raise RuntimeError("ONNX_REF_RESULT_JSON requires exactly one arm -- set ONNX_REF_ONLY_ARM too")
            pathlib.Path(result_json_path).write_text(json.dumps(results[0]))
        print_summary(model_path, results)
        return

    if mode == "same-process":
        print(
            "mode=same-process: reproducing the OLD ORT-then-torch single-process convention "
            "as a labeled control -- this is the artifact under test, NOT the canonical number",
            file=sys.stderr,
        )
        results = compute_results(model_path, runs, ort_warmup, torch_warmup, None)
        print_summary(model_path, results)
        return

    if mode != "isolated":
        raise RuntimeError(f"unknown ONNX_REF_MODE={mode!r}, expected 'isolated' or 'same-process'")

    print("mode=isolated (default, canonical): each arm runs in its own subprocess -- see README for why", file=sys.stderr)
    results, pid_by_arm = run_all_isolated(model_path, runs, ort_warmup, torch_warmup, None)
    print(f"isolation proof: {pid_by_arm}", file=sys.stderr)
    print_summary(model_path, results, pid_by_arm=pid_by_arm)


if __name__ == "__main__":
    main()
