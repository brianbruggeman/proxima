---
status: draft, 2026-10-06
---

# decode and prefill speed parity

## problem

Owner, 2026-10-02: gemma4 E2B decode "10ms is our floor, but ceiling is 7ms" (about 10 ms is llama.cpp
parity; 7 ms beats it). Owner, 2026-10-06: after the architecture-as-data cleanups, "we can start to
optimize".

Measured on main (architecture-as-data row 0b, 2026-10-05; 970-token prompt, 128 new tokens, arms
interleaved, Ollama quit):

| engine | decode ms/token | prefill ms (970 tokens) |
|---|---|---|
| proxima | 11.97 | 2534 |
| llama-server f1ea20621 | 9.06 | 569 |
| Ollama gemma4:e2b-it-qat | 9.13 | 578 (uncached request) |

Prefill is the larger gap: about 4.4 times llama, against about 1.3 times for decode. Time to first
token is prefill, so this is what a user waits on.

Granite moe 1b (row 0c, 1000 tokens): proxima 14.7 ms/token decode and a 6.5 s prefill. That is slower
than the larger gemma4 E2B on both, and it has not yet been measured against llama or Ollama.

Speculation verify (row 6) makes every family except gemma4 slower: granite 21.8 vs 15.2 ms/token,
qwen2 36.3 vs 20.3, qwen3 240 vs 29.2.

Load and bind time. The `generic_binder_` tests take seconds to minutes per model in the
test profile (2026-10-06, slice 9 gate): gemma4 26B 606 s, openchat 182 s, gemma4 E2B 161 s,
granite 65 s, qwen2 23 s. Owner: "if these were ms not s, I'd be happy."
- Unmeasured: how much of that is the unoptimized test build, how much is real work the binder does
  (dequantize, transpose, copy, or hash every weight byte), and what the release load time to first
  token is.
- A model load that maps the file and borrows quantized blocks in place should cost close to the page
  faults it takes, not a pass over 13 GB.

## requirements

- R1 Every change passes the three parity checks: correctness (tests, digests where the graph is meant
  to stay), semantic (llama_parity_, generic_verify_llama_parity_), performance (decode ms/token,
  prefill ms, TTFT, peak RSS, footprint, GPU bytes against the 0c baseline and against the previous
  landed change, interleaved arms, quiet box). A speed change that moves ids fails.
- R2 Every slice starts from an attribution, not a guess: a per-kernel GPU time census and a host/GPU
  timeline of the case being changed (the existing census and capture-replay hooks, 348dcf95,
  ab4782ce), compared with llama.cpp's own per-op timing on the same prompt. The slice states
  which measured cost it removes and by how much, before it is written.
- R3 Prefill: gemma4 E2B 970-token prefill at or below llama-server's on the same run.
- R4 Decode: gemma4 E2B at or below 10 ms/token (parity), then toward 7.
- R5 Granite moe: measured against llama-server and Ollama first; then the same prefill and decode
  parity bars.
- R7 Load and bind: measure release load-to-ready time and the test-profile binder time per model,
  attributed by phase (mmap, header, descriptor, lower, bind, upload, plan). Then:
  - remove any per-byte pass that the borrowed-block path does not need;
  - make the test profile build the hot crates optimized, so gates stop paying debug-build cost on
    real weights.

  Target: binder tests and release load in milliseconds to low seconds, with the measured floor
  stated.
- R6 Speculation verify: explain from a trace why verify costs more than it saves outside gemma4
  before any change to the per-family default.

## acceptance (each names a count)

- AC1 `decode_arms` reports llama-server and Ollama arms for granite moe as it does for gemma4 E2B:
  2 models x 3 engines x 7 runs recorded.
- AC2 prefill bound line `metric=prefill_ms arm=tip vs=llama` within=true on gemma4 E2B.
- AC3 decode bound line `metric=ms_per_token arm=tip vs=llama` within=true on gemma4 E2B, then the
  7 ms line.
- AC4 the same two for granite moe.
- AC5 `llama_parity_` 6 passed and `generic_verify_llama_parity_` 5 passed after every slice.

## slices

0. Granite incumbent arms (AC1), and an attribution census for E2B prefill and granite prefill and
   decode, against llama's per-op timing. Output: a ranked table of measured costs.
1. onward: one slice per ranked cost, largest first, each stating its measured target and passing
   R1. Prefill first (largest gap), then E2B decode, then granite.
