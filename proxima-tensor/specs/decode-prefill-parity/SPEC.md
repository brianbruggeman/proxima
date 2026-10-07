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
- AC5 `llama_parity_` 7 passed (the file defines gemma4_26b, gemma4_e2b, granite_moe, openchat, qwen2,
  lfm2, qwen3) and `generic_verify_llama_parity_` 5 passed after every slice. This line read 6 until
  slice 1; the count came from memory, not from `cargo nextest list`.

## slices

0. Granite incumbent arms (AC1), and an attribution census for E2B prefill and granite prefill and
   decode, against llama's per-op timing. Output: a ranked table of measured costs.
1. onward: one slice per ranked cost, largest first, each stating its measured target and passing
   R1. Prefill first (largest gap), then E2B decode, then granite.

## attribution (slice 0, measured 2026-10-06 to 2026-10-07)

Everything below is a measurement or a number derived from measurements, with its source. A cell that
says untraced names a cause nobody has measured. No row below is a verdict.

Evidence root: `evidence/slice0/` (this directory). Raw runs that were too large to commit (the debug
event logs, 413 MB) sit in `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/slice0/`;
the committed `timeline/*_token_breakdown.log` files are the `token_breakdown*` lines of those logs.

### what ran

- Host: Apple M1 Max, 32-core GPU, 68.7 GB, macOS 24.6. Ollama quit before every timed run (the driver
  starts and quits it per round). Box load at AC1 launch: `ac1/box_load_before.txt` (load average 4.75;
  a background daemon at 72% CPU, `suggestd` at 71%, WindowServer 27%). The same daemon was at 125.6% CPU
  in the 0c run (`evidence/speed_baseline_0c_granite/box_load.txt`). Load average was 4 to 10 during the census and timeline runs.
- Proxima: release `decode_gbps_baseline` built from `f76b4a97` (sha256 `9c2dca1a...3619`), `std,metal`.
  Timeline and census runs: the same tree built with `std,metal,instrument`.
- llama.cpp `f1ea20621`: `proxima-prefix-cache/scratchpad/bin/llama-f1ea2062/bin/llama-server` for AC1.
  For per-op time an out-of-tree Release+Metal build of the same commit with
  `evidence/slice0/llama_ops/llama_per_op.patch` applied (re-applies cleanly to `git archive f1ea20621`;
  built in `~/.cache/proxima-parity-perf/llama-build`).
- Ollama: `gemma4:e2b-it-qat` and `granite3.1-moe:1b`, `num_ctx 4096`, raw prompt, temperature 0, top_k 1.
- Prompt: `ac1/prompt1k.txt` (3641 bytes; 971 tokens on gemma4, 1000 on granite), 128 new tokens. The
  event-timeline runs passed it through the shell (`$(cat ...)`), which drops the trailing newline: step 0
  there is `new_count=970` (E2B) and 999 (granite). AC1, the census and llama use the file verbatim.

### AC1: six arms, one interleaved run

Command (run with no cargo, GPU or Ollama process on the box, an idle sccache server aside, `ac1/box_load_before.txt`;
`ac1/launches.log` has every launch time):

```
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 \
  --arm tip=decode_gbps_baseline_f76b4a97 --llama-server llama-server-f1ea20621 \
  --case gemma4_e2b=<E2B blob>=gemma4:e2b-it-qat --case granite_moe=<granite blob>=granite3.1-moe:1b
```

Per case, each of 3 rounds ran proxima, llama-server and Ollama in a rotated order, each with 1 warm-up
and 7 timed requests: 21 timed runs per arm, 126 in all (`ac1/decode_arms.out`: 144 `raw` lines, 24 per
arm, 3 of them warm-ups). Proxima ids equal llama ids on all 48 proxima generations (6 of them warm-ups). Medians over all 21 runs;
CoV over all 21 runs (outlier-removed figures in the file).

| arm | decode ms/token | CoV, range | prefill ms | CoV, range | TTFT ms | peak RSS (median of 3 procs) | peak footprint |
|---|---|---|---|---|---|---|---|
| E2B proxima | 13.348 | 3.11%, 12.477-14.403 | 2540.0 | 0.45%, 2528.9-2570.0 | 2540.0 | 3.93 GB | 715 MB |
| E2B llama-server | 9.018 | 5.98% (one 11.588 run), 8.967-11.588; kept n=20: 0.39% | 573.3 | 2.98%, 568.4-652.8 | 575.7 | 3.74 GB | 207 MB |
| E2B Ollama | 9.246 | 1.80%, 9.157-9.934 | 591.0 | 1.41%, 584.7-613.8 | 594.1 | 4.97 GB (sampled, see below) | not measured |
| granite proxima | 15.349 | 2.94%, 15.045-16.656 | 6546.0 | 0.94%, 6514.9-6748.0 | 6546.0 | 2.64 GB | 643 MB |
| granite llama-server | 5.210 | 1.10%, 5.062-5.278 | 151.4 | 0.32%, 150.2-152.4 | 153.2 | 1.76 GB | 285 MB |
| granite Ollama | 5.369 | 7.65% (one 7.180 run), 5.064-7.180; kept n=20: 2.77% | 176.7 | 2.94%, 167.3-184.8 | 180.0 | 1.91 GB (sampled) | not measured |

Proxima peak GPU allocation: 5.61 GB (E2B), 3.32 GB (granite). Range of the 7 timed E2B proxima decode runs
per process (`ac1/decode_arms.out`): 12.48 to 13.93 (process 0), 13.19 to 13.79 (1), 13.21 to 14.40 (2).

Bound lines (`bound metric=... arm=tip vs=llama-server`): `within=false` on ms/token, prefill, TTFT, peak
RSS and peak footprint for both models against llama-server; against Ollama the same except E2B peak RSS
(`within=true`, -1.04 GB), and peak footprint has no Ollama value (`n=0`). E2B proxima is 13.348 against 9.0169 (kept medians), delta +4.331 ms,
limit 0.180; prefill 2540.0 against 573.2, delta +1966.8, limit 11.46.

Definitions that differ across engines (each is in `decode_arms.rs`):

- Proxima prefill ms = run wall - decode ms/token x (tokens - 1); TTFT = the step-0 `elapsed_ms`
  (in-process). llama prefill = `timings.prompt_ms`; llama TTFT = client clock from request write to the
  first streamed chunk (loopback HTTP and tokenization included). Ollama prefill = `prompt_eval_duration`;
  Ollama TTFT = client clock to the first chunk minus the server's `load_duration`.
- Ollama is unloaded (`keep_alive 0`, then `/api/ps` empty) before every request. The previous driver did
  not: its Ollama prefill for runs 1 to 7 was 11.9 ms, a prefix-cache hit (row 0b's 578 ms was request 0
  only). `prompt_eval_count` is 971 in all 24 E2B Ollama requests of this run.
- Ollama RSS is the sum of `ps` rss over the Ollama processes (app, serve, runner) sampled every 50 ms; the
  others are `/usr/bin/time -l`. Shared pages can count twice in the Ollama figure.
- Ollama on granite loads a 131072-token context by default (`/api/ps`: size 8.7 GB, RSS 7.8 GB in a
  smoke run); every Ollama request in this run sets `num_ctx 4096`, the same as llama-server `-c 4096`.

### method for the per-op tables

- Proxima side: `gemma4_decode_kernel_census` (hook `348dcf95`, example `ab4782ce`) capturing one
  evaluation's live dispatches and replaying each distinct kernel alone. "own cb" is the median GPU span of
  one dispatch in its own command buffer, caches warm (`warm_ns`); "marginal" is the floor-amortised span
  of 16 back-to-back dispatches. The census's whole-sequence replay reconciles to the live step:
  prefill 2455 to 2479 ms replayed against 2460 to 2482 ms `gpu_busy_ms`; sum of own-cb 2381 ms
  (0.96 of busy). Decode E2B: sequence replay 10.7 to 11.96 ms against 11.47 ms busy; sum of own-cb
  16.27 ms (1.42 of busy).
- llama side: the patch gives every encoded op its own command buffer and records
  `GPUEndTime - GPUStartTime` per op key (`PROXIMA_OP_TIMING_OUT`). Tiny ops cost 4.0 to 4.6 us each in
  this mode (the `ADD` ops: min 4.0 us, per-key means 4.5 to 5.1), so floor-included numbers are comparable
  with proxima's own-cb numbers (tiny proxima dispatches, iota, constant and identity copy, measure 3.9 to 6.2 us own-cb), and the net-of-floor
  column subtracts 4.0 us per op. Profile mode perturbs the run: E2B prefill 614 to 639 ms against 573 ms
  unprofiled, E2B decode 21.8 ms/token against 9.0, because the graph no longer overlaps ops. The per-op
  numbers are GPU timestamps and do not carry that wall-clock cost. Controls: patched build with the env
  var unset gives 8.957 ms/token and 572.6 ms prefill against the reference binary's 8.961 and 571.5
  (7 runs each, `controls/llama_patched_vs_reference/`); the profiled run's 128 token ids hash equal to the
  reference binary's (E2B `2e96ec230837`, granite `7fc4d37ac7b1`).
- llama graphs per request: E2B prefill 3 graphs of 512, 455 and 4 tokens (`GRAPHS ntok=` lines in
  `llama_ops/e2b_ops.tsv`), granite prefill 2 graphs of 512 and 488; decode 381 graphs of 1 token over 3
  requests. Prefill rows sum the graphs of one request; decode rows are per graph. Proxima prefill is one
  evaluation of the whole prompt.
- Limits: own-cb sums exceed the real step on both sides, by different factors (decode E2B: proxima
  16.27 ms own-cb against 11.04 to 11.47 ms busy = 1.42x to 1.47x; llama 11.10 ms own-cb against 9.02 ms
  real = 1.23x), so decode class gaps below are on the own-cb basis and carry that bias. llama ops that
  overlap in the real graph (independent matmuls) are serialised here.
- Rank tables regenerate from the committed files: `cargo test -p proxima-model-interop --features std
  --example attribution_rank` (10 tests; 5 re-derive the saved tables and the step timelines from
  `census_*/census_groups.csv`, `census_*/census_dispatches.csv`, `llama_ops/*.tsv` and
  `timeline/*_token_breakdown.log`, 1 control asserts a wrong graph selection changes the table).

### the ranked table, prefill (ms per request, own-cb basis)

Gap = proxima - llama. E2B prefill measured 2540.0 ms (proxima) against 573.3 ms (llama): +1966.8 ms.
Granite prefill 6546.0 ms against 151.4 ms: +6394.6 ms.

| rank | phase / op class | proxima ms | llama ms | gap ms | ratio | measured mechanism | evidence |
|---|---|---|---|---|---|---|---|
| 1 | granite prefill, Q8_0 matmuls on 1024x512 weights (576 gathered expert dispatches plus 48 K/V projections) | 5873.64 (624 ops) | 114.41 (237 ops) | +5759.23 | 51.3x | Tiled-GEMM admission rejects all 576 expert dispatches with `GatheredOperand` and 97 more Q8_0 nodes with `NotQ4K`; admitted 0 of 748 classified reduce nodes in the plan. The 576 run four variants of one generic gather kernel (`..._g10`) at 9.67 to 15.62 ms each; 1000x1024x512 multiply-adds in 9.67 ms is 108 GFLOP/s (derived). llama runs 237 ops of that shape per request (MUL_MAT_ID for the experts). Why the generic kernel is that slow: untraced | `rank/granite_prefill.md`; `census_granite_prefill/census_groups.csv` (rows `g10`); `timeline/granite_tiled_gemm_classification_first_plan.log` (576 `GatheredOperand`, 97 `NotQ4K`, 0 `admitted=true`) |
| 2 | E2B prefill, Q4_0 weight matmuls (275 dispatches) | 1479.66 (275 ops) | 483.00 (825 ops over 3 graphs) | +996.66 | 3.06x | All 275 Q4_0 matmuls ARE on the tiled-GEMM path (275 `admitted=true`, 0 Q4_0 rejected). The full matmul class (rank/e2b_prefill.md, first table) is 1587.45 vs 487.35 because it also holds the F16 and F32 weights of rows 6 and below. Per shape: 1536x12288 60 ops 902.90 vs 284.91 (+618.0, 3.17x); 1536x6144 45 ops 333.67 vs 110.18 (+223.5); 1536x2048 56 ops 136.75 vs 47.83 (+88.9); 1536x4096 14 ops 68.34 vs 22.83 (+45.5). Rate on the 971x12288x1536 matmuls (36.6 GFLOP each): proxima 14.5 to 15.6 ms = 2.4 to 2.5 TFLOP/s; llama on 512 tokens (19.3 GFLOP) 2.33 to 2.43 ms = 7.9 to 8.3 TFLOP/s (all derived). Kernel-level cause: untraced | `rank/e2b_prefill.md`; `timeline/e2b_tiled_gemm_classification_first_plan.log` |
| 3 | E2B prefill, attention core | 749.70 (346 ops) | 41.31 (105 ops) | +708.38 | 18.2x | Global layers: 7 `cached attention partial` dispatches (`q971 c992 h1 g8 d512`) at 49.2 ms each = 344.7 ms. Sliding layers: 168 Q.K and P.V folds (112 + 56) are admitted to the dense-batched tiled path (168 `admitted=true`) and still cost 212.2 + 145.1 ms (1.9 and 2.6 ms per op); softmax max/exp/sum are 171 more dispatches = 47.7 ms. llama has one FLASH_ATTN_EXT per layer per graph, 0.39 ms mean. Why the tiled folds cost 1.9 to 2.6 ms: untraced | `rank/e2b_prefill.md` (label table); `census_e2b_prefill/stdout.log`; `timeline/e2b_dense_batched_gemm_classification_first_plan.log` |
| 4 | granite prefill, attention core | 286.28 (603 ops) | 12.93 (95 ops) | +273.36 | 22.1x | 24 `cached attention partial` dispatches at 11.76 ms each = 282.2 ms (`q1000 c1024 h8 g2 d64`); 386 softmax-max dispatches add 3.2 ms. Kernel-level cause: untraced | `rank/granite_prefill.md` |
| 5 | E2B prefill, per-layer F16 projection (1 dispatch, 1536x8960) | 107.77 | 4.35 (3 ops) | +103.42 | 24.8x | One dispatch; tiled admission refuses it (`NotPackedRowBlock(NotKQuantCodec)`, node 17: the operand codec is Float16); generic reduce kernel. 2x970x8960x1536 = 26.7 GFLOP in 107.8 ms = 0.25 TFLOP/s (derived) | `rank/e2b_prefill.md`; `timeline/e2b_tiled_gemm_classification_first_plan.log`, `e2b_dense_batched_gemm_classification_first_plan.log` (`OperandIsPacked` for node 17) |
| 6 | granite prefill, Q8_0 1024x1024 (q and o projections, 48 ops) | 118.21 | 15.08 | +103.13 | 7.8x | Rejected by tiled admission with `NotQ4K` (the rejection counts 97 Q8_0 nodes in the plan); runs generic reduce | `rank/granite_prefill.md`; `timeline/granite_tiled_gemm_classification_first_plan.log` |
| 7 | granite prefill, F32 1024x32 (router, 24 ops) | 11.62 | 2.17 | +9.44 | 5.3x | untraced | `rank/granite_prefill.md` |
| 8 | E2B prefill, rms norm / rope / head | 26.58 / 9.40 / 1.12 | 13.68 / 3.37 / 0.94 | +12.90 / +6.03 / +0.18 | 1.9x / 2.8x / 1.2x | 446 norm dispatches against llama's 726 ops (llama counts each fused pattern once per graph, 3 graphs); per-op cause untraced | `rank/e2b_prefill.md` |
| 9 | E2B prefill, elementwise / copy / other | 7.13 (370 ops) | 22.21 (634 ops) | -15.08 | 0.32x | Proxima folds the gate activation into matmul epilogues (labels `..._epi9_fused_..._tanh_...`); llama runs GLU as 105 separate ops per request (12.22 ms) and carries MUL 3.01, CONT 2.35, ADD 1.66, SET_ROWS 1.17, UNARY 1.14, SCALE 0.65 ms on top (22.21 ms in all). Only class where proxima is faster | `rank/e2b_prefill.md`; `llama_ops/e2b_ops.tsv` (sum `sum_us` per `op=` over `ntok` 512, 455, 4, divided by 3) |
| 10 | host, not GPU: plan prepare, pre-encode, readback, kv upload | 22.97 + 8.61 + 6.70 + 4.34 ms (E2B run 1); 26.46 + 10.81 + 20.20 + 8.74 (granite run 1) | n/a | n/a | n/a | The prefill plan is rebuilt on every request: step 0 of each of the 3 runs shows `plan_misses=1`, `prepare_calls=1`, `prepare_ms` 22.9 to 23.0 (E2B) and 26.5 to 26.7 (granite); that is 0.9% and 0.4% of the step. Pipelines are cached after run 0 (run 0: 65 pipeline misses, 74.9 ms compile, 95.6 ms commit-to-GPU-start; runs 1 and 2: 0 misses, 14.1 to 14.2 ms) | `rank/e2b_steps.md`, `rank/granite_steps.md` |

Timeline of the E2B prefill step (run 1, warm; `rank/e2b_steps.md`): step wall 2530.79 ms = `evaluate` 2522.21
+ named-block kv 4.34 + kv append 1.83 + the rest; `evaluate` holds `gpu_exec` 2474.85 (commit to wait),
of which `gpu_busy` 2460.43 ms; 1568 dispatches in 1 command buffer; encode 1.26 ms; prepare 22.97 ms;
readback 6.70 ms. GPU busy is 97.2% of the step wall. Granite (run 1): wall 6530.15, `gpu_busy` 6433.73
(98.5%), 1847 dispatches, 1 command buffer.

Answers to the three questions in the brief:

- Is prefill chunked at a small width? No. Step 0 evaluates `new_count=970` rows in one command buffer
  (`ubatch_size: 0` in `decode_gbps_baseline.rs`, `chunks=1`). llama-server evaluates 512, 455 and 4 rows
  per graph. Whether chunking at 512 would change proxima's time: not measured.
- Is the plan rebuilt per request? Yes, 23.0 ms (E2B) and 26.5 ms (granite) per request, row 10.
- Does matmul at prefill width use the tiled path? E2B: 275 of 275 Q4_0 matmuls do; the per-layer F16
  projection and the Q6_K head do not. Granite: 0 of 748 classified reduce nodes do (576 gathered, 97 Q8_0, the rest not
  weight matmuls).

### the ranked table, decode (ms per token)

E2B: proxima 13.348 (decode_arms, row above) against llama 9.018: +4.33. Granite: 15.349 against 5.210: +10.14.
Own-cb basis for class rows (see limits above); shapes with equal op counts compare like for like. Rows 1 and 2
are on different bases (serial host ms against own-cb ms) and are not ordered against each other.

| rank | phase / op class | proxima | llama | gap | measured mechanism | evidence |
|---|---|---|---|---|---|---|
| 1 | granite decode, serial host work before the GPU starts | evaluate 16.39 = host 5.45 + gpu_exec 10.94 (run 1); 17.41 = 5.94 + 11.47 (run 2) | n/a | n/a | 1 command buffer per step (`chunks=1`, `encode_overlap_ms=0`): all encoding happens before the commit. Stage counters (sibling spans of `encode_op`) inside that 5.45 ms: op setup 1.49, encode 1.96, expert-buffer lookup 0.59, loop head 0.15, retire scan 0.13, pre-encode 0.07 = 4.39; 1.06 ms of it is in no counter. E2B decode uses 8 command buffers and overlaps 2.2 ms of encode with the GPU: its `evaluate` 11.66 is `gpu_busy` 11.06 + 0.60 | `rank/granite_steps.md`, `rank/e2b_steps.md` |
| 2 | granite decode, weight matmuls | 9.68 ms own-cb (698 ops) | 3.99 (193 ops) | +5.69 (2.43x) | Dispatch structure: proxima issues 576 per-top-k-slot gathered expert matvecs per token (24 per layer, 1024x512 weights, 13.1 us each own-cb) where llama issues 120 ops of that shape (5 per layer, 25.5 us each). Matmul dispatches 698 against 193. Bytes read per token by those 624 matvecs: 624 x 1024x512 x 1.0625 B = 0.35 GB (derived), 0.87 ms at 400 GB/s (assumed spec); proxima's live GPU time is 10.3 to 10.8 ms, census replay 8.9 to 9.4 ms | `rank/granite_decode.md`; `census_granite_decode/stdout.log` |
| 3 | E2B decode, weight matmuls (Q4_0 shapes) | 8.71 own-cb (277 ops); sequence replay of the 275 Q4_0 dispatches 5.65 | 5.30 (276 ops) | +3.41 own-cb (1.64x) | Same op count both sides. Per op: 1536x12288 67.0 us vs 38.3 us (1.75x); 1536x6144 40.7 vs 23.6 (1.73x); 1536x2048 23.6 vs 12.7 (1.87x); 1536x256 10.8 vs 8.9 (1.22x). 1536x12288 weights are 10.6 MB: 158 GB/s proxima, 277 GB/s llama (derived; M1 Max spec 400 GB/s assumed). Kernel-level cause: untraced | `rank/e2b_decode.md` |
| 4 | E2B decode, rms norm | 3.97 own-cb (446 dispatches) | 1.80 (242 ops) | +2.17 (2.21x) | Dispatch count: 446 against 242; per op 8.9 us against 7.4 us. (Proxima's 170 `norm apply` plus 276 sumsq dispatches against llama's fused RMS_NORM patterns.) | `rank/e2b_decode.md` |
| 5 | granite decode, attention / rope / norm | 1.39 / 0.75 / 0.56 | 0.73 / 0.32 / 0.37 | +0.66 / +0.43 / +0.20 | untraced | `rank/granite_decode.md` |
| 6 | E2B decode, attention / rope / head | 1.56 / 0.61 / 1.12 | 1.21 / 0.32 / 0.94 | +0.35 / +0.29 / +0.18 | 73 vs 35 attention ops (partial + merge, 35 layers each, against one FLASH_ATTN_EXT per layer); untraced beyond that | `rank/e2b_decode.md` |
| 7 | E2B decode, elementwise / copy / other | 0.30 (44 ops) | 1.52 (214 ops) | -1.22 | Proxima fuses these into epilogues; the only class where it is faster | `rank/e2b_decode.md` |

Totals that frame the decode rows:

- E2B proxima decode is GPU-bound: steady steps (runs 1 and 2, 14 steps each) wall 11.70 and 11.67 ms,
  `gpu_busy` 11.06 and 11.04 ms, 941 dispatches in 8 command buffers. Census sequence replay by family
  (in program order, additive): Q4_0 matmuls 5.65 ms (275 dispatches), norms 2.46 (446), head 1.18 (1),
  attention 1.15 (70), rope/copy/elementwise 0.56 (147), other matvec 0.13 (2): 11.12 ms.
- `decode_arms` measures 13.35 ms/token against 11.7 ms steady step wall. In the 128-token event run the
  metric reads 12.14 to 12.32 while the decode step walls sum to 11.43 ms/token (run 1): 0.7 to 0.9 ms
  per token sits outside the step walls (cause untraced). Same-binary 128-token runs in one hour read
  12.14 to 14.44 ms/token (one 15.59 outlier): 12.14 to 12.72 in the event run, 12.25 to 13.41 in
  `controls/instrument_logging_control.txt` (with and without debug events, no difference), 13.43 to 14.44 in
  `controls/plain_128.err`. Drift between runs of 2 ms is on record and unexplained. Row 0b measured
  11.97 on 2026-10-05.
- Granite live GPU time per step (10.3 to 10.8 ms, run 0 14.3) is above the census replay (8.9 to 9.4 ms);
  cause untraced (GPU clock state is a candidate, not measured).

### why a 1B-parameter, 400M-active granite is slower than E2B

Measured, both sides, same prompt:

- Prefill 6546 ms against 2540 ms. Granite's 624 expert and K/V matmuls (1024x512 weights) take 5873.64 ms of its 6299.90 ms
  own-cb total (93%); none of the 576 expert ones reach the tiled path. E2B's 275 Q4_0 matmuls are tiled and total 1479.66 ms.
- Decode 15.35 against 13.35 ms/token. Per-token weight bytes (derived from shapes x op counts): granite
  about 0.46 GB (experts and K/V 0.35, q/o 0.05, head 0.05), E2B about 1.4 GB (Q4_0 1.05 GB, head 0.33 GB,
  f16 0.03 GB). Granite's live GPU time (10.3 to 10.8 ms) is about the same as E2B's (11.0 ms) for a third of
  the bytes: about 43 GB/s against about 127 GB/s (derived). The bytes do not set the step time. The step is 926 dispatches, 624 of them 1024x512 expert matvecs at
  13 us own-cb (whether dispatch count or per-kernel occupancy limits them is untraced), and its serial host
  encode (5.45 ms) is not overlapped, where E2B's is.

### what is not traced

- Kernel-level cause of every per-op gap above (tiled Q4_0 GEMM at width 971 reaching 2.5 of llama's
  8.3 TFLOP/s; the gathered expert kernel; the attention kernels; the decode matvec's 1.75x). No GPU counter
  capture or kernel variant toggle was run in this slice.
- 1.06 ms of granite's pre-commit host time; granite live GPU time above its replay; 0.7 to 1.7 ms per
  token of E2B decode between step walls and `decode_arms`; the run-to-run drift of proxima decode.
- Whether prefill chunked at 512 rows changes proxima's time.
- llama's decode per-class time on an additive (sequence) basis; only own-cb is available for llama.
- Ollama on the same per-op basis (its runner is llama.cpp inside the app; not instrumented).
- Peak footprint for Ollama (no `time -l` read; RSS is sampled).

### discipline rows for slice 0

| component | gate | result (N) | negative or surprising result kept |
|---|---|---|---|
| `decode_arms` multi-model driver, `292166ac` | clippy `-D warnings` std,metal `--all-targets`; alloc-tier and no-default checks; tensor and interop suites | clippy exit 0 (also with `instrument`); `proxima-tensor` 779 passed, 8 skipped; `proxima-model-interop` slice-gate 708 passed, 124 skipped | Ollama prefill for runs 1 to 7 was a cache hit in the old driver (11.9 ms against 578 to 591); Ollama defaulted to a 131072-token context on granite (RSS 7.8 GB) |
| kernel census knobs and tolerant replay, `bad027a6`, `e15876df` | same chain plus census unit tests | 21 census tests passed; granite decode replays 902 of 926 dispatches (24 `moe_topk` dispatches are refused by the capture hook and reported, not timed) | the first census build panicked on granite decode (`moe_topk binds extra outputs outside bindings`) |
| llama per-op patch (`llama_ops/llama_per_op.patch`) | controls: unset env equals reference; ids equal reference | 8.957 vs 8.961 ms/token, 572.6 vs 571.5 ms prefill, ids hash equal (E2B, granite) | first build aborted on `commit command buffer with uncommitted encoder`; graph width detected from `GET_ROWS` gave 8 for every granite graph (expert top-k) and 0 for E2B prefill, replaced by the widest MUL_MAT; an 8 us floor made llama nets negative, replaced by the measured 4.0 us |
| `attribution_rank` example | clippy, example tests | 10 passed (5 snapshot reproductions, 1 control that must differ, 4 mapping tests) | a first mapping keyed matmul shape by (K, N) read the reduction axis as the last extent; gate/up extents are `[tokens, K, N]`, so the key became the weight element count |

Designs abandoned: per-node host timing through llama's eval callback (it synchronises every split and
bills host latency to the op; replaced by one command buffer per op and GPU timestamps); shell and awk
aggregation of the llama file (replaced by the Rust example so the table re-proves in a test); prompt
perturbation to defeat Ollama's prefix cache (changes the token ids; replaced by unloading).

Re-prove commands: tables and timelines, `cargo test -p proxima-model-interop --features std --example
attribution_rank`; the census files, `gemma4_decode_kernel_census` with `M0_CAPTURE_STEPS=0` (prefill) or
the default (decode) and `M0_MODEL_GGUF`; the llama files, build the patched tree and run `llama-server`
with `PROXIMA_OP_TIMING_OUT=<file>`, three 128-token requests from `llama_ops/request.json`; AC1,
`decode_arms` as above. Missing for CI: no job runs example tests, and none runs a GPU bench on Apple
hardware, so the AC1 and census numbers have no saved baseline to diff against; the rank tables do re-prove
from committed files.

### proposed slices, largest measured gap first

Target numbers are in the unit the table above measured (own-cb ms for per-op classes, end-to-end for
prefill and decode). "Derived" targets subtract the table's class gap from the measured total.

1. Granite prefill, gathered expert matmuls onto a grouped tiled GEMM (llama's MUL_MAT_ID shape). Removes
   up to 5759 ms of the 6395 ms prefill gap. Target: 624-op Q8_0 1024x512 class 5873.64 -> at most 120 ms
   (llama 114.41); granite prefill 6546 -> about 790 ms after this slice alone (derived), toward llama
   151 / Ollama 177 once slices 3 and 4 land. LANDED 2026-10-07, targets not reached: class 5872.94 ->
   599.36 ms (llama 114.41), granite prefill 6458.99 -> 831.94 ms (llama 151.40). See "slice 1 result".
2. E2B prefill Q4_0 tiled GEMM throughput. Removes up to 980 ms. Target: Q4_0 class 1479.66 -> at most 500 ms
   (llama 483.00, the 7.9 to 8.3 TFLOP/s rate, derived); E2B prefill 2540 -> about 1560 ms (derived). First step: capture GPU counters
   or toggle the kernel's staging and fragment types to find the kernel-level cause, since it is untraced.
   LANDED 2026-10-07, class target not reached: Q4_0 class 1479.22 -> 577.77 ms (llama 483.00), E2B prefill 2397.95 ->
   1494.01 ms (llama 571.98). See "slice 2 result". Fixed 2026-10-07: class 489.99 ms own-cb against 482.99,
   E2B prefill 1413.94 ms against 569.69 (target met on the class row, AC2 not met). See "slice 2 fix result".
3. Prefill attention (E2B and granite): one fused kernel per layer. Target: E2B 749.70 -> at most 60 ms
   (llama 41.31), granite 286.28 -> at most 20 ms (llama 12.93). Removes about 690 ms from E2B prefill and 266
   ms from granite prefill.
4. Codec-generic tiled GEMM, dense and expert-grouped (rewritten 2026-10-07 by the main thread). Both fast
   paths are specialized by codec today:
   - the dense tiled path admits Q4_0 and Q4_K only (`emit_and_classify.rs` ~2548 whitelist and ~2859
     `NotQ4K`);
   - the expert-grouped path admits Q8_0 only (~3005, repeated at `expert_grouped_gemm.rs` ~66);
   - each decode is a hand-written arm (`tiled_gemm_cooperative_scan.rs` ~361, ~709, ~927;
     `expert_grouped_gemm.rs` ~393, ~442).

   That is why each test model's codec happens to be fast. gemma4 26B (experts Q3_K / Q5_0 / Q5_1; attn
   q/k and dense FFN Q3_K and Q5_0, per its `.bound` fixture) gets neither path. llama.cpp covers every
   codec with one `kernel_mul_mm` / `kernel_mul_mm_id` template plus a per-codec `dequantize_*` function
   (`ggml-metal/kernels/mul_mm.metal` :19, :151, :509; `kernels/dequantize.h`).

   Do the same:
   - one per-codec decode description (block elements, block bytes, a decode of 8 or 16 consecutive
     elements into half, the scale/min layout), selected by one function;
   - one stager consuming it for both the dense and the grouped path;
   - the three admission sites follow the description instead of whitelists;
   - the grouped K step derives from the codec's block size.

   Add codecs with a parity test against the CPU dequant path before each is turned on:
   - first the ones the models use: Q8_0 on the dense path (granite 1024x1024 118.21 -> at most 16 ms,
     llama 15.08), Q3_K, Q5_0, Q5_1, Q5_K, Q6_K (gemma4 26B prefill, then the E2B Q6_K head);
   - then F16 (E2B per-layer projection 107.77 -> at most 5 ms, llama 4.35).

   Measure gemma4 26B prefill before and after, against llama, alongside E2B and granite. A codec added
   without its own parity test does not land.
   After slices 2 to 4 E2B prefill is about 770 ms (derived: 2540 - 980 - 690 - 103) against llama 573.
   The 195 ms left: own-cb class gaps of about +40 ms (matmul +17, attention +19, norms +12.9, rope +6.0,
   elementwise -15.1 at those targets), and about 155 ms between the own-cb sum and the step wall (live GPU
   time 79 ms above the census sum, unexplained; host about 80 ms: plan prepare 23, commit-to-start 14,
   pre-encode 8.6, readback 6.7, kv 6.2, the rest untraced). Parity on R3 needs further slices after
   re-attribution.
5. Granite decode, command-buffer chunking so encode overlaps the GPU (E2B already runs 8 chunks). Target:
   step wall 16.5 -> at most 12 ms (derived: GPU 10.9 plus the first chunk's encode), and the 1.06 ms of
   untraced host time attributed. Then expert dispatch batching: 24 gathered matvecs per layer -> 3,
   dispatches 926 -> about 380; target live GPU time 10.3 -> at most 6 ms (llama total 5.2).
6. E2B decode: Q4_0 matvec per-op time (1536x12288 67.0 -> at most 45 us) and norm fusion (446 -> at
   most 250 dispatches). Target: `gpu_busy` 11.04 -> at most 10.0 ms (R4 parity), then toward 7. First step:
   re-attribute decode on an additive sequence basis for llama, since own-cb inflates both sides unevenly.
7. Prefill plan reuse across requests: 23.0 ms (E2B) and 26.5 ms (granite) per request to 0.

## slice 1 result (measured 2026-10-07)

Slice 1 of the list above: expert-grouped tiled GEMM for gathered Q8_0 expert weights. Every number
below is a measurement with its source; no row is a verdict. Evidence root: `evidence/slice1/` (this
directory); raw logs that were too large to commit sit in
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/slice1/`.

### what landed

- `0bdb132b` `omega` feature `metal-grouped-gemm` (passthrough in `proxima-model-interop`): classification
  in `omega/src/msl/emit_and_classify.rs` (`classify_tiled_gemm` now returns a `TiledGemmBlock` with
  `gathered: Some(ExpertGather)` for a gathered Q8_0 weight), the kernel body in
  `omega/src/msl/expert_grouped_gemm.rs`, sizing keys `[grouped_gemm] col_parts, scan_ahead` in
  `omega/omega-runtime.toml`, 6 unit tests (`msl::tests::expert_grouped_gemm`) and 8 Metal tests against the
  f32 CPU oracle (`omega/tests/expert_grouped_gemm_parity.rs`, includes a control that must fail and the
  out-of-range-expert fault).
- `37dc2980` `omega/examples/expert_grouped_gemm_speed_probe.rs`, the isolated GPU timer used for every
  kernel row below. `9eaa9b63` `decode_arms --dump-llama-ids`. `8c629a45` test
  `prefill_width_parity_with_llama_granite_moe` and its llama.cpp fixture (1000 prompt ids, 128 greedy ids).
- `b3f5e7aa` joins the feature into omega's `metal` set, so a default `std,metal` build takes the path.
- Scope: gathered, Q8_0, one token axis, route index constant off the token axis, activation unit-stride on
  the reduce axis, `TILED_GEMM_MIN_TOKENS` (160) rows or more, `metal-tiled-gemm` on. Every other gathered op
  (Q4_K/Q6_K experts, token axes `[sequence, selected]`, below 160 rows) still takes the cooperative gather
  kernel, now rejected by a named `TiledGemmRejection` variant (`GatheredCodecNotAdmitted`,
  `GatheredTokenAxesNotSingle`, `TokenExtentBelowMinimum`, ...) where it was `GatheredOperand` before.
  Non-gathered Q8_0 (the 97 `NotQ4K` nodes of the slice 0 plan) is untouched: slice 4.

### re-attribution on HEAD before any code

`evidence/slice1/census_head_granite_prefill/` (`gemma4_decode_kernel_census`, `M0_CAPTURE_STEPS=0`,
granite blob, `prompt1k.txt`, built from `f76b4a97`, whose omega, tensor and interop sources are identical to
`c4810cb7`: empty `git diff f76b4a97..c4810cb7 -- omega proxima-tensor/src proxima-model-interop/src`).

| quantity | slice 0 | HEAD re-measured | source |
|---|---|---|---|
| class `Q8_0 524288 (1024x512)`, 624 ops, own-cb | 5873.64 ms | 5872.94 ms | `rank.md` |
| `matvec Q8_0` 673 dispatches, cold own-cb | 5992.96 ms | 6013.47 ms | `stdout.log` |
| live step `gpu_busy_ms` | 6433.73 (run 1) | 6382.93 | `stdout.log` `m0 step` line |
| 698-dispatch matmul family, one command buffer, 7 runs | n/a | mean 6009.90, min 6004.08, CoV 0.04% | `stdout.log` family lines |

Mechanism removed, traced in `census_groups.csv`: each of the 576 gathered expert dispatches was the generic
cooperative gather kernel, `grid_threads=131072000` at threadgroup width 256 or 128, one threadgroup per
output element (1000 tokens x 512 rows), re-reading the expert row once per token: 9.67 ms (gate), 10.08 ms
(up with its silu epilogue), 9.77 ms (down), 15.62 ms (last round of down with the 25-step combine epilogue),
warm own-cb per dispatch (`warm_ns`; the cold column differs by at most 0.001 ms on these four rows,
`census_head_granite_prefill/census_groups.csv`). `classify_tiled_gemm` rejected all of them with `GatheredOperand`
(`timeline/granite_tiled_gemm_classification_first_plan.log`, slice 0).

### before and after

Interleaved final run, `evidence/slice1/interleaved/decode_arms.out`: 3 processes x (1 warm-up + 7 timed) =
21 timed runs per arm, arms rotated per process, Ollama quit (api refused, 0 processes), 1000-token granite
prompt and 971-token E2B prompt (`prompt1k.txt`), 128 new tokens. Arms: `base` =
`decode_gbps_baseline_f76b4a97` (sha256 `9c2dca1a...3619`), `tip` = the final tree, `tipcopy` = a byte copy of
`tip` (sha256 `e3db0a4b...f21`, both), llama-server `f1ea20621`. Box: load average 4.27 before and 3.75 after;
`suggestd` 90.6% CPU and a background daemon in `~/.local/bin` 89.8% CPU before (`box_load_before.txt`).
After the run only the date and the load averages were captured (`box_load_after.txt`, 93 bytes, no process
list), so the per-process CPU of the box during and after the run has no artifact; the 4.27 and 3.75 load
averages are the only recorded loadout figures beyond the before-run list. No cargo, GPU or Ollama process
of mine ran during it.

| arm | prefill ms (median, CoV, range) | TTFT ms | decode ms/token (median, CoV, range) | peak RSS (median of 3) | peak footprint | peak GPU bytes |
|---|---|---|---|---|---|---|
| granite base | 6458.99, 0.16%, 6454.0-6495.0 | 6459.0 | 15.004, 1.50%, 14.923-16.035 | 2.470 GB | 611.8 MB | 3,315,433,472 |
| granite tip | 831.94, 0.94%, 825.0-859.0 | 832.0 | 14.954, 0.85%, 14.548-15.093 | 2.501 GB | 597.7 MB | 3,315,433,472 |
| granite tipcopy (control) | 830.05, 1.60%, 827.0-884.0 | 830.0 | 14.933, 1.03%, 14.544-15.132 | 2.500 GB | 605.8 MB | 3,315,433,472 |
| granite llama-server | 151.40, 5.35%, 151.2-186.4 | 153.25 | 5.255, 5.43%, 5.116-6.337 | 1.764 GB | 279.7 MB | n/a |
| E2B base | 2402.95, 0.18% | 2403.0 | 12.203, 3.06% | 3.922 GB | 717.4 MB | 5,608,554,496 |
| E2B tip | 2401.05, 0.40% | 2401.0 | 12.180, 4.91% | 3.902 GB | 714.3 MB | 5,608,554,496 |
| E2B tipcopy (control) | 2399.06, 0.27% | 2399.0 | 12.230, 3.28% | 3.903 GB | 716.0 MB | 5,608,554,496 |
| E2B llama-server | 573.53, 7.99% | 576.9 | 9.002, 4.82% | 3.746 GB | 216.2 MB | n/a |

Bound lines (`bound metric=... arm=tip vs=base`, limit = max(2% of the base median, twin gap), printed by the
driver, computed from the outlier-filtered `median_kept` values of the `summary` lines, not from the
`median_all` figures in the table above: granite prefill kept medians 830.5205 against 6457.0170 give -5626.50
ms, where the table medians 831.94 and 6458.99 give -5627.05; ms/token tip 14.9570 against base 15.0010
gives -0.044, where the table gives -0.050; the llama-server figures below use kept medians as well
(prefill 151.375, ms/token 5.2359)): granite prefill delta -5626.50 ms within=true; TTFT -5626.00 within=true; decode -0.044 ms/token,
limit 0.300, within=true; RSS +30.8 MB, limit 49.4 MB, within=true; footprint -14.1 MB within=true; GPU bytes
delta 0. E2B (code path unchanged by this slice): prefill -1.93 ms, decode -0.040 ms/token, within=true on
all. Twin gap `tipcopy` vs `tip`: granite prefill -1.50 ms (limit 16.61), E2B prefill -2.98 ms. Against
llama-server, granite `tip` is `within=false` on every metric: prefill +679.1 ms, TTFT +677.8 ms,
ms/token +9.72, RSS +736 MB, footprint +318 MB (the AC4 prefill line of the spec: not reached).

Per-kernel, same census method as slice 0 (`evidence/slice1/census_tip_granite_prefill/`, final tree,
`rank.md`; warm own-cb (`warm_ns`) of one representative dispatch per kernel group, `census_groups.csv`;
the class total 599.36 reproduces on `warm_ns` (192 x 1.0347 + 192 x 1.0859 + 168 x 0.5907 + 24 x 1.1960 +
48 x 1.3390 = 599.36 ms) and not on `cold_ns`). The cold column and its CoV for the same dispatches:

| tip dispatch group | warm own-cb ms (table below) | cold own-cb ms | cold CoV |
|---|---|---|---|
| gate (192) | 1.0347 | 1.0211 | 10.99% |
| up + silu epilogue (192) | 1.0859 | 1.0786 | 6.69% |
| down (168) | 0.5907 | 0.5858 | 1.75% |
| down + combine epilogue (24) | 1.1960 | 1.2034 | 24.45% |

Three of the four cold CoVs exceed 5% (`cold_cov_pct` of the census), so the cold medians of those three groups
are not stable point estimates; the warm and cold values differ by 0.005 to 0.014 ms (0.6% to 1.3%) on each group. The
HEAD figures below agree between the two columns to 0.001 ms (cold CoV 8.05%, 0.01%, 0.01%, 0.03%).

| quantity | HEAD | tip | llama |
|---|---|---|---|
| class `Q8_0 524288 (1024x512)`, 624 ops, own-cb | 5872.94 ms | 599.36 ms | 114.41 ms (237 ops) |
| gate, per dispatch (192) | 9.670 ms, 131,072,000 threads | 1.035 ms, 2048 threads, depth 32 | n/a |
| up + silu epilogue (192) | 10.082 ms | 1.086 ms | n/a |
| down (168) | 9.770 ms | 0.591 ms | n/a |
| down + 25-step combine epilogue (24) | 15.622 ms | 1.196 ms | n/a |
| live step `gpu_busy_ms` | 6382.93 | 753.97 | n/a |
| 698-dispatch matmul family, one command buffer, 7 runs | 6009.90 mean, min 6004.08 | 747.92 mean, min 700.62, CoV 2.68% | n/a |

The class row is the number this slice named: 5872.94 -> 599.36 ms (-89.8%), against a target of at most
120 ms; the target is not met (5.24x llama). Granite prefill 6458.99 -> 831.94 ms (-87.1%), against the
derived 790 ms; not met (+42 ms), and 5.49x llama.

### the three checks

- correctness: omega suite 581 passed, 16 skipped (567 before plus 8 integration and 6 unit tests of this
  slice); `proxima-tensor` 779 passed, 8 skipped; `proxima-model-interop` slice-gate 709 passed, 124 skipped
  (708 before plus the new test); clippy `-D warnings` exit 0 for tensor, interop and omega with and without
  `metal-grouped-gemm`; alloc and no-default checks exit 0; omega `cargo check` exit 0 at `alloc`,
  `metal-core`, `cuda`, `metal-core,metal-tiled-gemm`, `metal-core,metal-grouped-gemm`, `std,metal`, 0
  warnings (`evidence/slice1/gate/*.tail`). The Metal tests compare the kernel with `proxima_tensor::cpu` on
  the same packed bytes: worst row error 8.1e-5 to 1.2e-4 of the row norm (half-precision weights and
  activations); a result compared against an oracle routed to other experts exceeds the 2e-3 tolerance (the
  control, asserted).
- semantic: ids equal to llama.cpp `f1ea20621` on 24 of 24 runs per arm for base, tip and tipcopy, on both
  models (`ids arm=... equal=true` lines, 128 ids, `common_prefix=128`); the owner gate sets on the final
  tree: `llama_parity_` 7 passed, `generic_verify_llama_parity_` 5 passed, and the new
  `prefill_width_parity_with_llama_granite_moe` 1 passed (13 run, `gate/7_ac5_llama_parity.tail`; the spec's
  AC5 read 6 for `llama_parity_` before this slice; the file defines 7 and the line now says 7). The existing llama fixtures carry prompts of 4 to 72
  tokens, below the 160 rows the tiled paths need, so none of them reaches this kernel; the new fixture is
  the 1000-token prompt with llama's own prompt ids. The new test takes 9.45 s with the feature off and
  3.86 s on (`evidence/slice1/parity/prefill_width_parity_grouped_off.log`, `..._on.log`): the 5.6 s
  difference is the evidence the path ran, the test itself asserts only the ids.
- performance: the table above. Decode ms/token, RSS, footprint and GPU bytes sit inside the base bounds;
  E2B is unchanged.

### incumbent, home-turf arm

llama.cpp `f1ea20621` `test-backend-ops perf -o MUL_MAT_ID -b MTL0` (built out of tree from
`git archive f1ea20621`, perf cases added for these shapes: `evidence/slice1/llama_mul_mat_id/`), `q8_0`
experts, 32 experts, 8 used, uniformly random routing per token. design-favors: incumbent: one op covers all
8 top-k slots, so tiles fill (250 tokens per expert at 1000 tokens) and one dispatch amortizes its ramp; the
per-route program issues 8 dispatches of one slot each. Ours: `expert_grouped_gemm_speed_probe`, uniform
routing, `col_parts` 2, 3 process runs (`evidence/slice1/probe/final_p2_run{1,2,3}.out`), per-slot median
times 8x to compare with the llama op.

| tokens | shape | llama 8-slot op | ours, 1 slot (3 runs) | ours x8 / llama |
|---|---|---|---|---|
| 160 | gate/up 1024->512 | 313.54 us | 330.8 to 513.0 us (CoV 14.6 to 16.7%) | 8.4 to 13.1 |
| 160 | down 512->1024 | 292.94 us | 199.1 to 201.0 us | 5.4 to 5.5 |
| 512 | gate/up | 752.74 us | 248.1, 251.4, 402.4 us | 2.6 to 4.3 |
| 512 | down | 752.91 us | 246.6 to 248.1 us | 2.6 |
| 1000 | gate/up | 1377.09 us (6.09 TFLOP/s) | 351.2 to 356.9 us (2.94 to 2.99 TFLOP/s useful) | 2.04 to 2.07 |
| 1000 | down | 1401.99 us (5.98 TFLOP/s) | 355.0 to 357.9 us | 2.03 to 2.04 |
| 2048 | gate/up | 2684.83 us | 595.5 to 601.5 us | 1.77 to 1.79 |
| 2048 | down | 2889.23 us | 604.4 to 611.7 us | 1.67 to 1.69 |

Frequency-weighted: the granite prefill is one 1000-token request per prompt; 576 gathered dispatches per
request, all at the 1000-token row of this table, so that row is the hot path, and there the kernel is 2.0x
llama's time for the same work.

### discipline log, one row per tweak

Probe rows are `expert_grouped_gemm_speed_probe` medians of 21 dispatches after 5 untimed ones (the first
arms ran without the warm-up), gate/up shape, 1000 tokens, one process unless stated; the kernel versions
other than the final one are not in the tree, their outputs are kept in `evidence/slice1/probe/` as the
record. Probe arms named `zipf` in v1 to v4 drew their routing from `[-1, 1]` instead of `[0, 1)`, which sent at
least half the tokens to expert 0 (`zipf (defective)`); fixed from v4c on. The `balanced` arm shows a
bimodal 15 to 68% CoV in every version (GPU clock state suspected, not traced), so it is not used for any
delta. Host loadout for all of them: the box described above (`suggestd`, a background daemon, `mds_stores` running), no
other bench of mine active.

| row | change | measurement | delta vs prior | CoV, runs | status |
|---|---|---|---|---|---|
| 0 | HEAD, generic cooperative gather kernel | granite prefill 6458.99 ms (interleaved), class 5872.94 ms | baseline | 0.16%, 21 | |
| 1 | v1: one threadgroup per (row tile, token tile, expert), each scanning the route, grid z = expert, Q8_0 tile decode, token-scattered write-back | prefill 6467.0 -> 1509.0 ms (2 runs, `sweeps/v1_vs_base_2runs`) | -76.7% | 0.07%, 2 | kept |
| 2 | ablation of v1 (`probe/ablate_*.out`, `PROXIMA_GROUPED_ABLATE`, zipf (defective)): full 1386.6 us (same run); scan only 322.2; scan + barriers + write-back 412.1; without weight staging 918.0; without activation staging 1165.6; without MMA 1160.8 | the scan alone is 23% of the dispatch | diagnostic | 0.09 to 2.4%, 1 each | informs row 3 |
| 3 | v2: token tiles of an expert walked inside one threadgroup (`col_parts` 4), scan once instead of per tile | zipf (defective) 1391.8 -> 803.7 us (v2 `col_parts` 4); `col_parts` 1 made `single` 899.5 -> 3486.3 us | -42% on zipf (defective) | 1.7%, 1 | kept |
| 4 | v3: route entries of `scan_ahead` 4 chunks loaded back to back, upper token half skipped when a tile has <= 16 tokens | zipf (defective) `col_parts` 4: 803.7 -> 803.3 us; gate/up `col_parts` 8: 822.5 -> 783.0 | no signal at 4, -4.8% at 8 | 2.9% and 3.2%, 1 | kept (sized key, ggml's own skip) |
| 5 | half-precision activation tile on the padded layout | zipf (defective) 807.7 -> 882.5 us (down 774.0 -> 770.4) | +9.3% on gate/up, none on down | 2.9% and 2.8%, 1 | no signal; half activations returned in row 7 with ggml's layout |
| 6 | weight levels loaded as aligned words instead of `packed_char4` | zipf (defective) 807.2 -> 828.0 us | +2.6% | 3.1% and 2.2%, 1 | no signal, ROLLED BACK |
| 7 | v4: ggml-metal's 8x8-block tile layout, half activations, token-major coalesced write-back | gate/up zipf (defective) `col_parts` 8: 783.0 -> 732.5 us; down 848.2 -> 771.7; prefill (`col_parts` 4) 1005.0 ms | -6.4% and -9.0% | 2.4% and 1.1%, 1; prefill 2 runs | kept |
| 8 | diagnostic: weight address held fixed across K (`weight_hot`, `probe/v2ab_weight_hot.out`) | 802 -> 496 us against 417 without any weight staging | about 300 us of the 385 us weight-staging cost is the K-streaming access, about 80 us decode | 2.5%, 1 | informs row 9 |
| 9 | v5: global loads and decode issued into registers before the barrier that waits for the previous MMA, per-tile weight and activation pointers advanced by 34 bytes and 32 floats per step | `col_parts` 4 uniform: gate/up 699.9 -> 402.4 us, down 666.9 -> 423.1; prefill 1005.0 -> 848.0 ms | -42.5% and -36.6% kernel; -15.6% prefill | 3.5% and 2.4%; 2.3% and 1.4%; prefill 0.72%, 3 | kept |
| 10 | v6: next step's loads issued before the MMA | uniform 402.4 -> 416.1 us, down 423.1 -> 459.9 | +3.4% and +8.7% | 2.5% and 1.1%, 1 | ROLLED BACK |
| 11 | `col_parts` sweep, whole granite prefill (`sweeps/v4_parts_*`, `sweeps/v5_parts_*`) | v4: 2/4/8/16/32 = 1051.0/1005.0/1030.0/1110.0/1243.5 ms; v5: 1/2/4/8 = 864.0/828.0/848.0/869.9 ms | v5 `col_parts` 2 is -2.4% vs 4 | 0.25 to 1.8%, 2 or 3 each | `col_parts` = 2 |
| 12 | diagnostic: threadgroup memory padded by 0/4/8/16 KB (`probe/tgpad_*.out`) | uniform 712.8/745.1/807.3/1013.6 us | +4.5%, +13.3%, +42.2% | 3.0 to 3.8%, 1 | occupancy depends on threadgroup memory; not tuned further |
| 13 | join into `metal` | final interleaved run, rows 0 and the table above | 6458.99 -> 831.94 ms | 0.16% and 0.94%, 21 each | landed |

### not met, not traced, not done

- The class target (at most 120 ms) and the derived granite prefill target (about 790 ms) are not met:
  599.36 ms and 831.94 ms. Granite prefill is 5.49x llama (151.40 ms) and its decode is untouched (14.954
  against llama 5.255 ms/token).
- Why the per-slot kernel is 2.04x llama's time at 1000 tokens: not decomposed. Two facts that bear on it, both
  stated as derived or measured: at 31 tokens per expert per slot it runs about 46 partial tiles where llama
  runs about 32 full ones for the same tokens (derived: expected `ceil(n / 32)` over 32 experts at uniform
  routing, not counted on the device), and each of the 576 dispatches ramps and drains on its own
  (measured: 351 to 357 us per dispatch in the probe). Fusing the 8 slots into one dispatch is not done in
  this slice: every production builder emits `MoeProjectionStrategy::PerRoute`; `GroupedGateUp` exists in
  `gqa_layer_routed.rs` but no production caller selects it, and its `[sequence, selected]` token axes are
  rejected by this kernel (`GatheredTokenAxesNotSingle`). Spec slice 5's "expert dispatch batching" is the
  same structural change for decode.
- The census times one representative dispatch per kernel group (layer 0, round 0). The live step
  (`gpu_busy_ms` 753.97) and the family replay (747.92) are the aggregates; the group rows above are not
  additive to them.
- The probe's `balanced` arm bimodality (15 to 68% CoV) and the high CoV of the 160-token rows (14.6 to
  16.7%) are unexplained.
- `ServingConfig::default().ubatch_size` is 32 (`omega`-side `TILED_GEMM_MIN_TOKENS` is 160), so a prefill
  run through the default serving config is chunked below the threshold and never reaches this kernel or the
  dense tiled one; only `ubatch_size: 0` (what `decode_gbps_baseline` sets) or a value of at least 160 does.
  Not changed here.
- The threshold itself (160) was not re-measured for the gathered kernel: at 160 tokens the kernel takes
  199 to 513 us per slot against about 1.55 ms for the generic path scaled linearly from its 9.67 ms at
  1000 tokens (derived, not measured).
- Q4_K, Q6_K and Q4_0 experts (qwen35moe, gemma4 26B) were not run through this kernel and are not admitted.

### corrections after verification

- The per-kernel tables were labelled cold own-cb; the figures are `warm_ns`. Relabelled, cold column and cold
  CoV added beside them (above).
- The after-run process list quoted for the interleaved run had no artifact; the sentence now states only
  what `box_load_after.txt` holds (date and load averages).
- AC5 said 6 for `llama_parity_`; `cargo nextest` lists 7 (13 with the other two sets), the line now says 7.
- The driver's bound lines use `median_kept`; the headline table uses `median_all`. Both bases are now named
  next to the bound lines.
- `expert_grouped_gemm_speed_probe.rs` imports moved from the body of `run()` to module scope behind the
  same cfg gate; the `metal-grouped-gemm` comment in `omega/Cargo.toml` no longer cites a slice number.
  Clippy `-D warnings` on the example exits 0 with and without the probe features
  (`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/slice1/fix/`); the probe still runs
  (44 output lines, 40 measurements, box not quiet: smoke run, not a timing record).

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate                      # 581 tests
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate  # 709
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate \
  -E 'test(llama_parity_) or test(generic_verify_llama_parity_) or test(prefill_width_parity_with_llama_granite_moe)'  # 13
cargo run --release -p omega --example expert_grouped_gemm_speed_probe --features metal,instrument
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm base=<f76b4a97 binary> \
  --arm tip=<final binary> --arm tipcopy=<copy of tip> --llama-server <llama-server f1ea20621> \
  --case granite_moe=<granite blob> --case gemma4_e2b=<E2B blob>
gemma4_decode_kernel_census with M0_CAPTURE_STEPS=0 M0_MODEL_GGUF=<granite blob> PROXIMA_PROMPT_FILE=prompt1k.txt
attribution_rank rank --census DIR --llama evidence/slice0/llama_ops/granite_ops.tsv --ntok 512,488 --requests 3 --floor-us 4.0
test-backend-ops perf -o MUL_MAT_ID -b MTL0   # llama.cpp f1ea20621 with evidence/slice1/llama_mul_mat_id/perf_cases.patch
```

Missing for CI, as in slice 0: no job runs the example tests or a GPU bench on Apple hardware, so the
interleaved numbers, the census and the probe have no saved baseline to diff against; the Metal tests and the
gates above are the part that re-proves from the tree. The llama perf cases are
`evidence/slice1/llama_mul_mat_id/perf_cases.patch`, applied to `tests/test-backend-ops.cpp` of a
`git archive f1ea20621` tree built with `-DLLAMA_BUILD_TESTS=ON -DGGML_METAL=ON` (Release); run with
`-p "type_a=q8_0,type_b=f32,n_mats=32,n_used=8"`.

## slice 2 result (measured 2026-10-07)

Slice 2 of the list above: E2B prefill Q4_0 tiled GEMM throughput. Every number below is a measurement with its
source; no row is a verdict. Evidence root: `evidence/slice2/` (this directory); raw logs too large to commit (the
census telemetry logs of 16 MB each, the census timing samples of 4 MB each, the two xctrace recordings and their
exports) sit in `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/slice2/`. In the committed box-load files
the command line of a private background daemon is replaced by its description, as in slice 1; the raw files are in
that directory.

### what landed

- `bf740944` `omega/src/msl/kernel_types_identity.rs`: `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE` and
  `PROXIMA_TILED_GEMM_GRID2D` now default ON (unset admits, explicit `0` falls back), the same posture as
  `PROXIMA_TILED_GEMM_SLIM_TGMEM`. Both levers existed with byte-identity tests and isolated-kernel numbers (docs
  `docs/model-interop/discipline.md` rows C4.12 and C4.19) and had been left default off; the existing parity tests
  pin their baseline arm to `0` instead of relying on unset. `PROXIMA_TILED_GEMM_DIRECT_STORE` stays default off.
- `96eb42d1`, `7feb78e3` `omega/examples/tiled_gemm_q4_0_speed_probe.rs`: GPU time of one dense Q4_0 matmul through
  `omega::metal::execute_plan_named_op_timed` at six E2B weight shapes (K 1536 against 12288, 6144, 4096 and 2048
  rows; 1536 rows against K 6144 and 12288) and 160, 512 and 971 tokens, with an FNV-1a hash of the output bits.
- `6e8729fe` `PROXIMA_TILED_GEMM_MM_LAYOUT` (`omega/src/msl/tiled_gemm_cooperative_scan.rs`
  `push_mm_layout_k_loop`, identity suffix `_mml`, default ON): the tiled GEMM stages and multiplies in ggml's
  `kernel_mul_mm` tile layout (`mul_mm.metal:164-316` of llama.cpp `f1ea20621`): weight tile in 8x8 `[k][feature]`
  blocks, activation tile in 8x8 `[token][k]` blocks, `token x feature` accumulators stored transposed into the
  existing `[feature][token]` copy-out tile, rows and tokens past the extent clamped instead of zero-filled, a Q4_0
  half-block decoded with ggml's fused multiply-add form (`q4_0_dequant_half16`), and the K offset of a unit-stride
  activation read as `k0`. 11 new tests: 8 byte-identity tests against the row-major layout on real gemma4-E2B Q4_0
  and openchat Q4_K weights (`omega/tests/tiled_gemm_mm_layout_parity.rs`) and 3 emitter tests
  (`msl::tests::mm_layout_*`).
- Scope: applies where the wide weight stage applies (Q4_0 and Q4_K packed weights at `TILED_GEMM_MIN_TOKENS` (160)
  rows or more) and the build-time tile sizing is ggml's 64 x 32 x 32 (`mm_layout_geometry_supported`); any other
  sizing keeps the row-major tile. The dense-batched attention folds, the grouped Q8_0 expert kernel and every
  decode kernel are untouched by the layout.

### re-attribution on HEAD before any code

Census (`gemma4_decode_kernel_census`, `M0_CAPTURE_STEPS=0`, `prompt1k.txt`) built from the slice 2 tree with
`PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=0 PROXIMA_TILED_GEMM_GRID2D=0` pinned (`bf740944` binary; the off path is the
row-major code the flip did not touch, and the class total below reproduces slice 0's within 0.03%),
`evidence/slice2/census_head_flags_off/`:

| quantity | slice 0 | HEAD re-measured | source |
|---|---|---|---|
| `matvec Q4_0` class, 275 dispatches, own-cb | 1479.66 ms | 1479.22 ms | `census_head_flags_off/rank.md` |
| matmul class (277 ops) | 1587.45 ms | 1587.01 ms | same |
| attention core class (346 ops) | 749.70 ms | 691.59 ms | same; 58.1 ms lower than slice 0, cause not traced |
| one-command-buffer sequence replay | 2455 to 2479 ms | 2336.46, 2339.32, 2338.61 ms | `census_head_flags_off/stdout.log` |
| live step `gpu_busy_ms` | 2460.43 (run 1) | 2340.22 | same |

Toggle experiment on the unchanged slice 1 tip binary (`decode_gbps_baseline_base_s2`, sha256 `e3db0a4b...f21`),
971-token prompt, 4 tokens, 4 runs per process (run 0 compiles pipelines; runs 1 to 3 below), `ttft_ms`:
`evidence/slice2/toggle1_control_off_off/` is a control (a shell word-split slip made both arms the default; kept as
the twin measurement): 2398 to 2407 ms over 4 processes. Warm TTFT with the env switches set, 2 processes each
(`toggle2_all_three/`, `toggle3_single_flags/`), base 2398 to 2409 ms:

| switches on | warm ttft_ms | delta vs base (about 2403) |
|---|---|---|
| `DIRECT_STORE` alone | 2369 to 2394 | about -30 |
| `GRID2D` alone | 2323 to 2331 | about -78 |
| `WIDE_WEIGHT_STAGE` alone | 1722 to 1752 | about -678 |
| wide stage + `DIRECT_STORE` | 1708 to 1721 | about -687 |
| wide stage + `GRID2D` | 1671 to 1674 | about -730 |
| all three | 1674 to 1685 | about -725 |

The text hash of the 4 generated tokens equals the base's in all of them. `DIRECT_STORE` adds nothing on top of the
other two (1678 against 1673 ms), so it was not flipped.

Mechanism removed by the flip, from the rows that built the levers: `WIDE_WEIGHT_STAGE` fixes "only 64 of 128 threads
staged weights", the 1-byte `uchar` loads (now `ushort`), the per-K-step `slot_off / block_elements` division and the
scalar `half` stores (C4.12, IR read); `GRID2D` reads tile coordinates from `threadgroup_position_in_grid` and narrows
the K counter to `int` (C4.19). This slice re-measured the effect (above and the census below), not the IR claims.

### before and after

Interleaved final run, `evidence/slice2/interleaved/decode_arms.out`: 3 processes x (1 warm-up + 7 timed) = 21 timed
runs per arm, arms rotated per process, Ollama quit (SIGTERM of the app and the server after osascript quit was
cancelled; api refused, no Ollama process), 971-token E2B prompt and 1000-token granite prompt (`prompt1k.txt`), 128
new tokens. Arms: `base` = `decode_gbps_baseline_base_s2` (HEAD before the slice, sha256 `e3db0a4b...f21`), `tip` =
`decode_gbps_baseline_s2_mml` (final tree), `tipcopy` = a byte copy of `tip` (`interleaved/binaries.sha256`),
llama-server `f1ea20621`. Box before: load average 3.80, `spotlightknowledged` 68.3% CPU, a background daemon in `~/.local/bin` 66.0% CPU,
iTerm2 30.3% (`box_load_before.txt`); after: load average 2.86, `CoreSuggestions` 82.0%, the same daemon 80.5%
(`box_load_after.txt`). No cargo, GPU or Ollama process of mine ran during it.

| arm | prefill ms (median kept, CoV all, range all) | TTFT ms | decode ms/token (median kept, CoV all, range all) | peak RSS (3 processes) | peak footprint | peak GPU bytes |
|---|---|---|---|---|---|---|
| E2B base | 2397.95, 0.26%, 2394.02-2420.06 | 2398.0 | 12.225, 4.91% (one 15.079 run), 12.169-15.079 | 3.859 / 3.865 / 3.950 GB | 716.5 MB | 5,608,554,496 |
| E2B tip | 1494.01, 0.29%, 1489.05-1510.06 | 1494.0 | 12.213, 3.80% (one 14.282 run), 12.096-14.282 | 3.903 / 3.913 / 3.884 GB | 710.8 MB | 5,608,554,496 |
| E2B tipcopy (control) | 1492.03, 0.34%, 1489.97-1510.95 | 1492.0 | 12.214, 4.01%, 11.430-14.163 | 3.921 / 3.937 / 3.898 GB | 714.0 MB | 5,608,554,496 |
| E2B llama-server | 571.98, 0.23%, 569.03-574.25 | 574.57 | 8.952, 0.67%, 8.916-9.179 | 3.721 GB | 211.4 MB | n/a |
| granite base | 830.00, 0.79%, 823.98-851.05 | 830.0 | 14.9445, 1.06%, 14.635-15.342 | 2.358 / 2.484 / 2.487 GB | 599.6 MB | 3,315,433,472 |
| granite tip | 831.00, 1.07%, 827.00-862.97 | 831.0 | 14.9550, 1.11%, 14.477-15.015 | 2.477 / 2.579 / 2.573 GB | 616.4 MB | 3,315,433,472 |
| granite tipcopy (control) | 831.97, 1.05%, 827.05-855.99 | 832.0 | 14.9520, 0.61%, 14.659-15.145 | 2.527 / 2.515 / 2.617 GB | 608.1 MB | 3,315,433,472 |
| granite llama-server | 151.28, 0.56% | 153.08 | 5.2057, 1.71%, 5.048-5.454 | 1.762 GB | 280.8 MB | n/a |

Medians are the driver's `median_kept` (outlier-filtered; the table's CoV and range are over all 21 runs).

Bound lines (`bound metric=... arm=tip vs=base`, limit = max(2% of the base median, twin gap)):

- E2B: prefill -903.94 ms (limit 47.96) within=true; TTFT -904.00 within=true; decode -0.012 ms/token (limit 0.2445)
  within=true; RSS +37.6 MB (limit 77.3) within=true; footprint -5.7 MB within=true; GPU bytes 0 within=true.
- granite (prefill kernels it does not take): prefill +1.00 ms (limit 16.6) within=true; decode +0.0105 ms/token
  (limit 0.299) within=true; GPU bytes 0 within=true; **peak RSS +88.2 MB (limit 49.7) within=false; peak footprint
  +16.7 MB (limit 12.0) within=false**.
- Against llama-server, E2B `tip` is `within=false` on every metric: prefill +922.03 ms (limit 11.44), TTFT +919.43,
  ms/token +3.261 (limit 0.179), RSS +181.8 MB, footprint +499.4 MB. The AC2 line is not met.
- Token ids: `ids ... equal=true` on 144 of 144 proxima generations (24 per arm, 3 proxima arms, 2 models), 128 ids each,
  `common_prefix=128`; `equal=false` 0.

The two red granite memory lines were investigated, not dismissed. Per-process peak RSS of the granite arms
(`interleaved/decode_arms.out` `memory` lines): base 2.358, 2.484, 2.487 GB; tip 2.477, 2.579, 2.573; tipcopy 2.527,
2.515, 2.617: both `tip` and `tipcopy` are above `base` in all three process rounds (+31 to +169 MB), and the same
binary differs from itself by up to 64 MB between processes. Attribution runs with `/usr/bin/time -l` on the tip
binary with the env switches pinned (`granite_rss_2runs/`, 4 processes per arm, 2 runs each: base 2102 to 2220 MB,
tip 2110 to 2216, every pinned variant 2100 to 2278; `granite_rss_8runs/`, 3 or 6 processes per arm, 8 runs each:
base 2282 to 2496 MB, tip 2368 to 2495, `GRID2D=0` 2260 to 2373, `GRID2D=0 WIDE_WEIGHT_STAGE=0` 2284 to 2372):
the spread inside one arm (up to 214 MB) is larger than the tip-versus-base gap (median 2359 against 2425 MB,
+66 MB), and the two pinned arms sit with base. Cause of the process-to-process spread: not traced. Cause of a
+66 to +88 MB shift on granite: not traced; the granite kernels the change touches are the dense-batched attention
folds through `GRID2D`, and the 2- and 8-run attribution does not isolate it from the spread.

### per-kernel and per-class

Census, same method as slice 0 (`census_tip/`, final tree; `census_flip/`, tree at `bf740944`; rank tables from
`attribution_rank`, llama column from `evidence/slice0/llama_ops/e2b_ops.tsv`):

| quantity | HEAD (flags pinned off) | `bf740944` | final | llama |
|---|---|---|---|---|
| `matvec Q4_0` class, 275 dispatches, own-cb | 1479.22 ms | 753.68 ms | 577.77 ms | 483.00 ms (825 ops over 3 graphs) |
| same 275 dispatches, one command buffer, 7 replays | 1481.66 ms (CoV 0.05%) | 756.33 ms (CoV 0.18%) | 577.75 ms (CoV 0.04%) | n/a |
| matmul class including the F16 projection | 1587.01 ms | 861.46 ms | 685.52 ms | 487.35 ms |
| one-command-buffer sequence replay of all 1568 dispatches | 2338.13 ms | 1606.91 ms | 1429.98 ms | n/a |
| live step `gpu_busy_ms` (one step) | 2340.22 | 1613.82 | 1439.23 | n/a |
| attention core class | 691.59 ms | 688.25 ms | 691.31 ms | 41.31 ms |

The Q4_0 class changed by -901.45 ms; the interleaved E2B prefill changed by -903.94 ms.

Largest kernel groups by final total (warm own-cb per dispatch, `census_groups.csv`, representative node; the cold
CoV of the final run is 0.04% to 0.12% on these six):

| node | dispatches | grid threads | HEAD ms | `bf740944` ms | final ms |
|---|---|---|---|---|---|
| 3010 | 20 | 761856 | 15.054 | 7.695 | 6.092 |
| 3012 | 20 | 95232 | 15.527 | 7.644 | 5.732 |
| 2995 | 20 | 761856 | 14.522 | 7.229 | 5.609 |
| 206 | 15 | 380928 | 7.432 | 3.896 | 3.061 |
| 191 | 15 | 380928 | 7.176 | 3.659 | 2.819 |
| 208 | 15 | 95232 | 7.573 | 3.853 | 2.875 |

Isolated kernel, `tiled_gemm_q4_0_speed_probe`, 971 tokens, p25 of 21 dispatches in each of 3 processes (60 untimed
dispatches first), `evidence/slice2/probe_final/`. The median of the 21 is polluted on the 971-token cells by GPU
contention outliers (cell CoV 2% to 63% in the `tip` arm, 0.2% to 19% at HEAD, 1.7% to 54% with the flags on; max
up to 3.4x the min), so the rows use p25, which repeats to within 0.02% to 1.5% across the 3 processes (the widest is
the HEAD 1536 x 12288 cell); the output hash is identical across the three arms and the three processes at every one
of the 18 shape and token cells (`output_fnv`, checked by `sort -u` over the 9 files):

| weight shape (rows x K) | HEAD (wide stage and `GRID2D` off), us | flags on, row-major layout, us | final, us | llama op, us |
|---|---|---|---|---|
| 12288 x 1536 | 14508.2, 14500.2, 14499.6 | 7234.1, 7240.2, 7239.3 | 5620.2, 5626.8, 5619.4 | 4620.9, 4651.0 |
| 6144 x 1536 | 7155.9, 7155.4, 7170.6 | 3661.6, 3664.4, 3663.2 | 2823.1, 2823.3, 2823.4 | 2324.0, 2322.3 |
| 4096 x 1536 | 4753.9, 4758.4, 4760.6 | 2478.5, 2479.3, 2471.5 | 1893.1, 1894.6, 1893.6 | 1555.8, 1556.7 |
| 2048 x 1536 | 2414.6, 2413.6, 2408.6 | 1299.8, 1297.0, 1294.1 | 964.4, 964.6, 965.1 | 791.3, 791.7 |
| 1536 x 6144 | 7562.2, 7581.2, 7588.1 | 3792.5, 3810.4, 3807.0 | 2880.6, 2879.9, 2879.9 | 2433.2, 2431.2 |
| 1536 x 12288 | 15670.6, 15432.5, 15439.5 | 7540.1, 7583.6, 7573.3 | 5741.6, 5738.0, 5737.4 | 4906.0, 4924.6 |

Final over llama op: 1.216, 1.215, 1.217, 1.219, 1.184, 1.170 (first llama run); the flags-on row-major layout was
1.567, 1.576, 1.593, 1.639, 1.565, 1.544, and HEAD 3.138, 3.079, 3.058, 3.050, 3.116, 3.147. The same ratio holds at
512 and 160 tokens (`probe_final/*.out`).

### the three checks

- correctness: omega suite 592 passed, 16 skipped (581 before plus the 11 new tests;
  `evidence/slice2/gate/13_omega_full.tail`); `proxima-tensor` 779 passed, 8 skipped; `proxima-model-interop`
  slice-gate 709 passed, 124 skipped; clippy `-D warnings` exit 0 for `proxima-tensor`, `proxima-model-interop` and
  `omega` (with `omega/metal`), and `omega` again with `instrument`; tier builds exit 0: `proxima-tensor` at `alloc`,
  `proxima-model-interop` at `--no-default-features`, `omega` at `metal-core`, `metal-core,metal-tiled-gemm` and
  `alloc` (the `metal-core` and `alloc` builds compile none of the new tiled-GEMM code, which is behind
  `metal-tiled-gemm`; that feature implies `metal`, so the restricted-tier claim for the new code is the `std`
  build's). The mm layout against the row-major layout: 0 differing words of 248,576 (971 tokens x 256 rows, real
  gemma4 Q4_0 weights) and of 51,000 (Q4_K, 510 tokens x 100 rows), and 0 in the other six cases, each printing its
  word count; the probe hash equal at 18 of 18 cells.
- semantic: ids equal to llama.cpp `f1ea20621` on 144 of 144 proxima generations; the owner gate sets on the final
  tree: `llama_parity_` 7 passed, `generic_verify_llama_parity_` 5 passed, `prefill_width_parity_with_llama_granite_moe`
  1 passed (13 run, `gate/20_owner_parity.tail`). The llama fixtures carry prompts of 4 to 72 tokens, below the 160
  rows the tiled path needs, so none of them reaches this kernel; the E2B evidence at prefill width is the
  interleaved run's 72 E2B ids lines, and a 128-token, 3-run check of the base, `bf740944` and final binaries
  produced the same token-id hash `4ce6df64e9a9` (`quick_e2e/`).
- performance: the tables above. E2B prefill and TTFT moved by -904 ms; E2B decode, RSS, footprint and GPU bytes sit
  inside the base bounds; granite prefill and decode are unchanged and its peak RSS and footprint are over the bound
  (above).

### incumbent, home-turf arm

llama.cpp `f1ea20621` `test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32"`, built out of tree (the
slice 1 test tree with `evidence/slice2/llama_mul_mat/perf_cases.patch` applied on top of slice 1's patch; Release,
Metal), `q4_0` weight times `f32` activation at the E2B shapes, 160, 512 and 971 tokens, 2 runs
(`llama_mul_mat/test_backend_ops_perf_run{1,2}.out`, `..._longk_run{1,2}.out`). design-favors: incumbent: ggml's own
kernel at its own design point, tokens 971 so its output-bounds variant is in play. The llama column is wall-clock
microseconds per run of a one-op graph (the harness's figure); ours is the command buffer's GPU span, so the ratio
is not like for like (the wall clock of a graph run is expected to include launch overhead the GPU span does not;
not measured). Frequency: all 275
Q4_0 dispatches of one 971-token request run this op family (class 1479.22 ms of the 2340 ms step at HEAD), so the
rows above are the hot path.

### discipline log, one row per tweak

Probe rows are p25 of 21 dispatches per process, 971 tokens, 2 or 3 processes interleaved (the first probe draft
printed medians only: rows 2 and 3). The raw files label the four K = 1536 shapes `ffn_gate_up` (12288 rows),
`ffn_down` (6144 rows), `attn_q` (4096) and `attn_o` (2048) from an earlier draft; the labels were wrong for three of
them and the committed probe names them by rows. Variants not in the tree are in
`evidence/slice2/patches/` (`experiments_blocked_all_variants.patch` and `stage_ahead_v1_with_btrans_hook.patch`, both
apply to the tree at `bf740944`; the hook is env `PROXIMA_TILED_GEMM_EXP`, tokens `blk`, `fma`, `unitk`, `ahead`,
`half`, `intidx`, `nostage`, `pipe`, `btrans0`). Host loadout for all of them: the box above, no other bench of mine
active, `evidence/slice2/probe/*box_load*.txt`.

| row | change | measurement | delta vs prior | CoV, runs | status |
|---|---|---|---|---|---|
| 0 | HEAD (slice 1 tip) | E2B prefill 2397.95 ms interleaved; Q4_0 class 1479.22 ms; kernel 14500.2 us (12288 x 1536) | baseline | 0.26%, 21; 0.05%, 7 replays; p25 spread 0.06% over 3 | |
| 1 | flip `WIDE_WEIGHT_STAGE` and `GRID2D` on (`bf740944`) | warm TTFT 2403 -> 1673 ms (toggles); Q4_0 class 1479.22 -> 753.68; kernel 14500.2 -> 7239.3 us, rows 6144 7155.9 -> 3663.2 | -46.3% to -51.0% across the six probe shapes; -49.0% class | toggle runs within 5 ms; probe p25 spread 0.1%, 3 | kept |
| 1a | `DIRECT_STORE` on top of the two | warm TTFT 1673 -> 1678 ms | no signal | within 5 ms, 2 processes | not flipped |
| 2 | stage-ahead v1: global weight and activation reads and the weight decode issued before the barrier that fences the previous multiplies (row-major layout; `sah_off` against `sah_on`) | median 7252.6, 7244.7 -> 7542.6, 7551.0 us (12288 x 1536); +3.4% to +4.4% on the four shapes | +4.0% (slower) | median of 21, 2 processes agree to 0.2%; output hash equal | ROLLBACK, patch kept |
| 3 | ablation: activation fragment loaded untransposed (timing only, output wrong; `exp_base` against `exp_btrans0`) | 7248.5 -> 7059.0 us and 3673.7 -> 3574.5 us (run 2); -1.8% to -2.8% over the four shapes | -2.6% | 2 processes | informs row 4, not a change |
| 4 | mm layout, activation stored as `float`, original decode (`blk4_base` against `blk4_blk`) | 7229.3 -> 5929.8, 3663.1 -> 2978.3, 2475.6 -> 1995.1, 1304.4 -> 1015.3 us | -18.0%, -18.7%, -19.4%, -22.2%; output hash equal | p25 spread 0.0% to 0.8%, 2 | kept |
| 5 | out tile stored token-major (`blk5` against `blk4_blk`) | p25 5924.0, 5925.0 against row 4's 5929.6, 5930.0 us; the other three shapes within 0.1% | -0.1% to +0.1% | 2 | not kept |
| 6 | load-before-barrier on top of the mm layout (`blk4_ahead`) | 5929.8 -> 5887.0 us; -0.6% to -0.7% on the four shapes | -0.7% | 2 | not kept (simpler form, below the 2% bound) |
| 7 | ggml's fused Q4_0 decode, `q4_0_dequant_half16` (`blk7_blk` against `blk7_fma`) | 5924.7 -> 5669.3, 2974.0 -> 2849.7, 1993.7 -> 1910.8, 1014.1 -> 973.9 us | -4.3%, -4.2%, -4.2%, -4.0%; output hash equal | p25 spread 0.0% to 0.1%, 2 | kept |
| 8 | K offset `k0` for a unit-stride activation (`blk8_fma` against `blk8_unitk`) | 5670.3 -> 5613.9, 2849.5 -> 2820.6, 1911.6 -> 1891.1, 974.1 -> 963.8 us | -1.0%, -1.0%, -1.1%, -1.1% | p25 spread under 0.1%, 2 | kept |
| 9 | 32-bit tile indices (`blk9_intidx`) | 5611.1 -> 5612.2 us | 0.0% | 2 | not kept |
| 10 | direct device store on the mm layout, interior tiles with an identity epilogue (`blk6_blkds`) | 5924.7 -> 5824.7, 2976.6 -> 2924.5, 1994.8 -> 1959.9, 1014.5 -> 997.8 us | -1.7%, -1.7%, -1.7%, -1.6% on that subset | 2 | not flipped, off |
| 11 | half-precision activation tile (`blk9_half`) | 5612.4 -> 5525.0, 2820.6 -> 2776.5, 1891.5 -> 1861.8, 963.8 -> 948.7 us | -1.5%, -1.6%, -1.6%, -1.6%; output hash differs | 2 | not kept (moves numerics for 1.5%) |
| 12 | load-before-barrier on the final loop (`blk9_ahead`) | 5612.4 -> 5562.5, 2820.6 -> 2795.5 us | -0.9% | 2 | not kept |
| 13 | software-pipelined loads: next step's raw global reads issued before the multiplies (`blk11_pipe`) | 5612.4 -> 5911.4, 2819.6 -> 2970.5, 1891.4 -> 1990.4, 964.2 -> 1011.3 us; `maxTotalThreadsPerThreadgroup` 896 -> 704 | +5.3%, +5.4%, +5.2%, +4.9% (slower) | p25 spread 0.0% to 0.1%, 2 | ROLLBACK, patch kept |
| 14 | landing of rows 4, 7, 8 as `PROXIMA_TILED_GEMM_MM_LAYOUT` (`6e8729fe`) | Q4_0 class 753.68 -> 577.77 ms; E2B prefill 2397.95 -> 1494.01 ms; kernel 7239.3 -> 5620.2 us (12288 x 1536) | -23.3% class, -22.4% to -25.6% kernel; -37.7% prefill over both commits | 0.04%, 7 replays; 0.29%, 21 | landed |
| 15 | ablation: both staging steps removed from the mm loop, barriers and copy-out kept (`blk10_nostage`; output meaningless) | 4742.3, 2385.8, 1600.1, 814.5 us (float tile); 4626.9, 2327.0, 1561.0, 794.9 us (half tile) against llama op (mean of 2 runs) 4635.9, 2323.2, 1556.3, 791.5 us | informs: the multiply loop alone costs as much as llama's whole op | 2 | informs |

Pipeline footprint (`probe/pipeline_footprint.txt`): `staticThreadgroupMemoryLength` 8192 bytes in every variant;
`maxTotalThreadsPerThreadgroup` HEAD kernel 832, flags on 1024, mm layout 896, MMA-only ablation 1024,
software-pipelined 704. Host allocations per operation: the hot path allocates nothing before or after (the kernel
source is emitted at plan time); peak GPU bytes are equal in `base` and `tip` for both models.

### not met, not traced, not done

- The class target (at most 500 ms) is not met: 577.77 ms against llama 483.00 (1.196x); the E2B prefill is 1494.01
  ms against llama 571.98 ms (2.61x, AC2 not met). The remaining prefill gap is the attention core (691.31 ms against
  41.31), the F16 per-layer projection (107.73 ms against 4.35) and the 94.8 ms left in the Q4_0 class (slices 3 and 4).
- Why the mm layout is 18% to 22% faster than the row-major layout: not decomposed. The layouts differ in the weight
  scatter pattern, the fragment load addresses, the multiply operand order and the accumulator orientation. The only
  piece isolated is the transposed fragment load (row 3: 1.8% to 2.8%). The fused decode (row 7) is explained by the
  source (one fused multiply-add per element against shift, convert, subtract, multiply), not by a disassembly.
- Why the kernel is still 1.17x to 1.22x llama's `mul_mm` on the same shapes: not explained. (Superseded: the ladder of "slice 2 fix result" toggles
  it; the kernel is now 1.0131x to 1.0367x of ggml at 971 tokens.) Facts: the geometry,
  the weight and activation layouts, the decode form and the loop order now follow `mul_mm.metal`; `staticThreadgroup
  MemoryLength` is 8192 in both for 971 tokens; the multiply loop with no staging at all (row 15) already costs 1.03x
  llama's whole op at 6144 x 1536. The ratio is constant over the four K = 1536 shapes (1.215 to 1.219), which points
  at a per-K-step cost, not a tile-edge or occupancy-tail effect. Two remaining differences from ggml are measured
  small (half tile, rows 11 and 12: 1.5%, direct store, row 10: 1.7%) and the others are not isolated.
- GPU counters: `xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters'` on the probe
  recorded the `gpu-counter-value` table (1.1 GB exported; counter ids 0 to 30 with no name table in the export), and
  was not decoded; nothing in this slice rests on it.
- The attention class read 749.70 ms in slice 0 and reads 691.59 ms on the same kernels here, 58.1 ms lower; cause not
  traced (slice 3 re-attributes it). (Traced in "slice 2 fix result": the same source tree measures 691 ms now; the slice 0 record sits in the slice 0 measurement.)
- The granite peak RSS (+88.2 MB, limit 49.7) and footprint (+16.7 MB, limit 12.0) bound lines are red, with the
  attribution above not resolving them from process-to-process spread. (Re-measured with 5 processes per arm in "slice 2 fix result": both lines within=true; the spread traced to resident `MALLOC_MEDIUM` pages in both binaries.)
- `DIRECT_STORE` stays off: (superseded by `888deca1`, which flips it on, with the census and ladder rows of "slice 2 fix result".) the kernel gain is 1.7% on interior tiles with an identity epilogue; 240 of the 275 Q4_0
  dispatches have one (410.0 of 577.8 warm ms in `census_tip/census_groups.csv`) and 35 carry a fused epilogue
  (167.8 ms), so the effect is about 7 ms of the 1494 ms prefill (derived), under the 2% bound; its default also
  covers the dense-batched attention path, where an earlier row measured no gain.
- The probe covers K = 1536 against four row counts and the two long-K projections; the census covers all 275
  dispatches. The Q6_K output head and the F16 projection are not on this path.

Designs abandoned: further toggling of the row-major kernel (the earlier harness had spent six rows on hybrids and
retired nested-loop control flow, fragment precision and the blocked layout one at a time; replaced by transcribing
`kernel_mul_mm`'s layout and decode as a whole); load-before-barrier and software-pipelined loads (rows 2, 12 and
13); half-precision activations (row 11, numerics); a token-major copy-out tile (row 5); GPU-counter attribution
through xctrace (recorded, undecoded).

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate                                   # 592 tests
cargo nextest run -p omega --features metal --cargo-profile gate -E 'test(mm_layout)'              # 11
cargo clippy -p proxima-tensor -p proxima-model-interop -p omega --features proxima-model-interop/std,proxima-model-interop/metal,omega/metal --all-targets -- -D warnings
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate   # 709
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate \
  -E 'test(llama_parity_) or test(generic_verify_llama_parity_) or test(prefill_width_parity_with_llama_granite_moe)'   # 13
cargo run --release -p omega --example tiled_gemm_q4_0_speed_probe --features metal,instrument
  # HEAD kernels: PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=0 PROXIMA_TILED_GEMM_GRID2D=0; row-major layout: PROXIMA_TILED_GEMM_MM_LAYOUT=0
gemma4_decode_kernel_census with M0_CAPTURE_STEPS=0 M0_MAX_TOKENS=2 PROXIMA_PROMPT_FILE=prompt1k.txt (same two env pins for HEAD)
attribution_rank rank --census DIR --llama evidence/slice0/llama_ops/e2b_ops.tsv --ntok 512,455,4 --requests 3 --floor-us 4.0
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm base=<e3db0a4b binary> --arm tip=<final binary> \
  --arm tipcopy=<copy of tip> --llama-server <llama-server f1ea20621> --case granite_moe=<granite blob> --case gemma4_e2b=<E2B blob>
test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32"   # slice 1 test tree + evidence/slice2/llama_mul_mat/perf_cases.patch
```

Missing for CI, as in slices 0 and 1: no job runs a GPU test or bench on Apple hardware, so the interleaved numbers, the
census and the probe have no saved baseline to diff against; the byte-identity tests, the emitter tests and the
clippy and tier builds are the part that re-proves from the tree on a Mac. The variant rows (2, 3, 5, 6, 9 to 13, 15)
re-prove only by applying the patches in `evidence/slice2/patches/` to `bf740944` and running the probe with the
env tokens above.

## slice 2 fix result (measured 2026-10-07)

Slice 2 was returned with seven findings. This section is what was changed for each, with the measurement and its
source. Every number is a measurement or a number derived from measurements; no row is a verdict. Evidence root:
`evidence/slice2fix/` (this directory). Raw logs too large to commit (the census telemetry logs and timing samples,
the per-process `time -l` raw reports, 123 MB of attention-trace censuses) sit in
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/slice2fix/`. Commits, in order: `2b0e56d2` (refactor), `7c5e25c4` (test),
`ee640ea2` (ladder), `6e637861`, `888deca1`, `0c4ff281` (perf), `3a933038` (test), `6875fe1c` (render check), `07e261b2` (ladder
shared memory), `2e2933e6` (rename); the docs commit that carries this section follows them. `ae2c5847` (the main thread's rewrite of the slice 4
entry in `proposed slices`) landed between `3a933038` and `6875fe1c` and is not part of this fix.

### the seven findings

| finding | what was done | evidence |
|---|---|---|
| `#[allow(clippy::too_many_arguments)]` on `push_tiled_gemm_tile_writeback` (10 parameters), file allow count 14 to 15; `#![allow(clippy::unwrap_used, clippy::expect_used)]` in the new test file | the write-back tail is inlined back into `push_tiled_gemm_body` (the mm path now sets `mm_layout_active` and falls through to the one shared tail), so the helper and its allow are gone (`2b0e56d2`); the test helpers return `Result<_, String>` and the tests return it, so the file carries no allow (`7c5e25c4`). File allow count `grep -c 'allow(' omega/src/msl/tiled_gemm_cooperative_scan.rs`: 14 at `29b097b7` (before the slice), 14 now. `git diff 29b097b7..HEAD -- omega proxima-model-interop proxima-tensor/src` adds no `allow(` line | `gate/omega_clippy.tail` (exit 0, `-D warnings`, `omega --features metal --all-targets`), `gate/clippy_all.tail`, `gate/clippy_instrument.tail` |
| AC2 not met: E2B prefill 1494.01 ms against llama 571.98 ms | not met, and not reachable by this slice: 1413.94 ms against 569.69 ms (+844.25 ms) after the fix; see "AC2" below | `interleaved/decode_arms.out` bound line `metric=prefill_ms arm=gemma4_e2b.tip vs=gemma4_e2b.llama-server` |
| Q4_0 class target (at most 500 ms) not met: 577.77 ms against llama 483.00 | met on the own-cb basis the target was stated on: 489.99 ms against llama 482.99 (same basis, the six Q4_0 shapes), by three emitter changes found with the ladder below | `census_fix_tip/rank.md`, `census_fix_tip/stdout.log` |
| granite peak RSS (+88.2 MB, limit 49.7) and footprint (+16.7 MB, limit 12.0) red, cause not traced | re-measured with 5 processes per arm and traced to a region class; see "granite memory" below | `interleaved/decode_arms.out`, `interleaved/granite_time_l_per_process.txt`, `granite_vmmap/` |
| the timed tip binary was placed by mtime only | the timed binary is built from a clean tree at a named commit and rebuilt from scratch to the same sha256; see "provenance" | `interleaved/binaries.sha256`, `interleaved/tip_commit.txt` |
| llama fixtures (4 to 72 prompt tokens) never reach the tiled kernel | new fixture `tests/fixtures/llama-parity/gemma4_e2b/long_prompt_llama_ids.json` (971 prompt ids, 128 greedy ids, produced by `decode_arms --dump-llama-ids` against llama-server `f1ea20621`) and test `prefill_width_parity_with_llama_gemma4_e2b` (`3a933038`); the test asserts at least 160 prompt tokens so it prefills on the tiled path | `fixture_gen/`, `gate/prefill_width_parity.tail` |
| untraced mechanisms (why the layout is faster, why 1.17x to 1.22x remained, GPU counters, attention class 58.1 ms lower) | the first two are now toggled and measured (ladder); the attention difference is traced to the measurement environment; GPU counters remain undecoded | "mechanism ladder", "attention class", "not met, not traced, not done" below |

### tool: the mm kernel ladder

`omega/examples/mm_kernel_ladder.rs` (`ee640ea2`) times, under one GPU clock and interleaved, the production tiled
`Q4_0` kernel (emitted by `omega::emit`), any number of hand-edited MSL variants of it, and ggml's own
`kernel_mul_mm_q4_0_f32` assembled at run time from a llama.cpp `f1ea20621` checkout (local includes pasted in order,
the six `function_constant`s replaced by the values ggml's host code computes for the shape, everything after the legacy
`kernel_mul_mm` except the `q4_0_f32` instantiation dropped; nothing is copied into this repository). Timing is
`GPUStartTime`/`GPUEndTime` of one dispatch; each cell is the 25th percentile of 21 dispatches after 30 warm-up
dispatches; every variant prints how many output words differ from the production kernel, and ggml (which stages the
activation as `half`) prints its scaled error. ggml's launch is `dispatchThreadgroups` with dynamic threadgroup memory of the length its host code picks (`ggml-metal-device.cpp`:
`bc_out ? 8192 : (4096 + 2048)`, so 8192 where the output needs bounds checks, as at 971 tokens, and 6144 otherwise; the
`float`-activation variants need 8192 and get it). Weights are synthetic Q4_0 blocks, activations random floats.
The slice 2 probe (`tiled_gemm_q4_0_speed_probe`) timed the same kernel through the production plan path: 5620.2 us
(12288 x 1536 x 971, final row, p25) against 5607.9 us for the ladder's `prod` at the slice 2 tip, 0.2% apart.

### mechanism ladder (12288 x 1536 weights, 971 tokens)

Raw: `evidence/slice2fix/ladder/run*.out` (CoV of every cell below is 0.02% to 0.13%, n = 21, except where stated);
variants as diffs against the dumped emissions in `ladder/variants/` (`prod_slice2_tip.metal`, the slice 2 tip emission;
the ggml diffs apply to the dump `LADDER_DUMP` writes). Every variant listed as "0 words" produced output identical
to the production kernel it was derived from, word for word (11,931,648 words).

First fact: ggml's kernel with its activation tile made `float` (one line, the template instantiation) produces output
identical to the slice 2 tip kernel in 0 of 11,931,648 words and runs at 4575.3 us against 4520.9 us for ggml's own
`half` form (+1.2%). The numerics of this kernel are therefore ggml's structure with a `float` activation, and the
remaining 1.24x (5607.9 us against 4520.9 us) is structure, not precision or rounding.

Toward the fast kernel, from the slice 2 tip (p25 us, one change added at a time except where marked):

| rung | change | us | delta | words differing | pipeline `maxTotalThreadsPerThreadgroup` | run |
|---|---|---|---|---|---|---|
| slice 2 tip (`prod`) | none | 5607.9 | baseline | n/a | 896 | run0 |
| p1 | `_Pragma("clang loop unroll(full)")` on the multiply loops | 5485.7 | -2.2% | 0 | 832 | run3 |
| p2 | and on the decode and store loops | 5484.5 | -0.0% vs p1 | 0 | 832 | run3 |
| p3 | p1 with ggml's multiply order (`mb[i/4]` outer) | 5484.8 | -0.0% vs p1 | 0 | 832 | run4 |
| p4 | p1 with the activation moved as one `float2x4` | 5485.6 | 0.0% vs p1 | 0 | 832 | run5 |
| p5 | p1 with the activation read through one carried pointer | 5420.5 | -1.2% vs p1 | 0 | 832 | run5 |
| p6 | the multiply section transcribed from `mul_mm.metal:290-314`: fragment pointers walked by addition, `ma`/`mb` declared before the loop, three `simdgroup_barrier`s | 5143.4 | -8.3% vs prod, -6.2% vs p1 | 0 | 832 | run11 |
| p7 | p6 and the carried activation pointer | 5066.9 | -1.5% vs p6 | 0 | 832 | run12 |
| p6_dyn, p7_dyn | p6, p7 with the tile array as the `[[threadgroup(0)]]` argument and `setThreadgroupMemoryLength` at launch | 5007.4, 4929.0 | -2.6%, -2.7% vs p6, p7 | 0 | 832 | run12 |
| p8_dyn | p7_dyn with ggml's barrier order: decode, barrier, store, barrier, multiply, no trailing barrier, one barrier after the loop | 4804.0 | -2.5% vs p7_dyn | 0 | 832 | run13 |
| p9_dyn | p8_dyn with the direct device store for interior tiles | 4661.5 | -3.0% vs p8_dyn | 0 | 832 | run14 |
| p9_static | p9_dyn with the array declared in the kernel again | 4803.5 | +3.0% vs p9_dyn | 0 | 832 | run15 |
| ggml (`half` activation) | reference | 4520.9 | | n/a | 832 | run0 |

Away from the fast kernel, from ggml's kernel with a `float` activation (4575.3 us), which is the same numerics:

| change | us | delta vs 4575.3 | words differing | max threads | run |
|---|---|---|---|---|---|
| ggml's own restage-through-threadgroup branch forced for every tile (not our epilogue) | 4561.1 | -0.3% | 0 | 832 | run2 |
| unroll pragmas removed from the multiply loops | 4920.6 | +7.5% | 0 | 896 | run2 |
| the tile array declared in the kernel (static) | 4963.0 | +8.5% | 0 | 768 | run6 |
| our weight decode and store | 4524.9 | -1.1% | 0 | 832 | run8 |
| our activation transfer | 4575.0 | 0.0% | 0 | 832 | run8 |
| both of ours | 4525.4 | -1.1% | 0 | 832 | run8 |
| our multiply section (with `FOR_UNROLL`) | 4918.5 | +7.5% | 0 | 832 | run9 |
| our multiply section plus ggml's two extra `simdgroup_barrier`s | 4910.1 | +7.3% | 0 | 832 | run10 |

Reading the two tables together: weight decode, weight store and activation transfer do not move ggml's kernel; its
epilogue form does not either (restage-always -0.3%); the multiply section's form moves it 7.5% in the direction the
transcription moved ours, and the static tile array moves it 8.5% and ours 3.0%. What the ladder does not say, and the
IR below confirms it cannot: why those forms cost what they cost on the device.

IR read (`ladder/ir/ir_counts.txt`: `xcrun -sdk macosx metal -std=metal3.0 -fmetal-math-mode=relaxed -O2 -S -emit-llvm`,
kernel function only). Every rung has one `simdgroup_matrix_8x8_multiply_accumulate` call site and one load site per
type, so the AIR keeps the loops rolled and unrolling happens in the device compiler. The slice 2 tip IR carries 0
`llvm.loop.unroll.full` nodes and every later rung 1; `ggml_float` and `ggml_float_static` differ by 4 lines
(461, 457) with identical block, phi, barrier and call-site counts, and run 4575.3 us and 4963.0 us. So the unroll
toggle (-2.2% and +7.5%) is a metadata fact the device compiler honours (the pipeline's `maxTotalThreadsPerThreadgroup`
changes with it), and the static-versus-dynamic toggle is not visible in AIR at all. The cause of the static-array cost
is untraced below the AIR.

### what landed

- `2b0e56d2` and `7c5e25c4`: the two lint findings (above). The refactor is emission-neutral: the render check
  (`omega/examples/tiled_gemm_render_diff_check.rs`, fixed in `6875fe1c`) renders 12 kernels (Q4_0 and Q4_K tiled at 971 tokens, the dense-batched fold, each at the four explicit
  `WIDE_WEIGHT_STAGE` x `DIRECT_STORE` combinations) at the slice 2 tip (`41736dc4`) and at `7c5e25c4`, and
  `diff -r` of the two output directories exits 0 with 12 files in each (`evidence/slice2fix/render_identity/`, sha256
  lists equal). That covers both arms of the restructured branch: the mm path (`WIDE_WEIGHT_STAGE=1`) and the row-major
  path (`=0`), with the direct-store tail on and off.
- the render check itself: before this fix it swept a 64-token program the tiled path does not admit, so all four of
  its "combinations" rendered the same text (the four Q4_0 files shared one sha256) and a `diff` of two commits said
  nothing; it now uses tiled-admitted shapes, pins each switch to an explicit `"0"` or `"1"` (an unset switch changed
  meaning at slice 2), also renders Q4_K, and fails if any switch under test leaves the text unchanged.
- `ee640ea2` the ladder (above).
- `6e637861` the mm-layout K loop in `kernel_mul_mm`'s schedule: decode before the first barrier, one barrier after the
  stores, multiply section transcribed (p8 minus the dynamic memory and the store), a barrier after the loop for the
  epilogue (`push_mm_layout_k_loop`, `push_mm_layout_weight_decode`, `push_mm_layout_activation_stage`,
  `push_mm_layout_multiply`). Tests: the 8 byte-identity cases of `tiled_gemm_mm_layout_parity` (Q4_0 and Q4_K, real
  weights) pass; two emitter assertions updated; new `mm_layout_decodes_before_the_first_barrier_and_ends_the_loop_on_the_multiplies`,
  which fails on the previous emitter (control: previous file restored, test run, `EXIT=100`, then reverted).
- `888deca1` `PROXIMA_TILED_GEMM_DIRECT_STORE` default ON (unset admits; explicit `"0"` falls back), the existing arm
  (`push_tiled_gemm_direct_store_arm`) and its identity-epilogue and `float` gates unchanged. The byte-identity tests
  that named unset as off now pin `"0"`; `tiled_gemm_mm_layout_parity` gains the 971-token prefill-width case with the
  default.
- `0c4ff281` `PROXIMA_TILED_GEMM_DYNAMIC_TGMEM` (default ON; `_dyn` identity suffix): the packed tiled kernel takes its
  slim tile array as `threadgroup uchar *tg_shared [[threadgroup(0)]]`, and `Grid2DSpec::threadgroup_bytes`
  (`tiled_gemm_shared_bytes()`, 8192 for the `64 x 32 x 32` tile) rides the launch to the one physical dispatch site
  (`resident_nocopy_cache::dispatch`, `setThreadgroupMemoryLength`). One predicate (`dynamic_tgmem_active`: lever, packed
  tiled path, slim store, grid launch) decides the kernel text, the launch length and the cache identity. Tests: 4 byte
  identity cases of the default against `PROXIMA_TILED_GEMM_DYNAMIC_TGMEM=0` (Q4_0 at the prefill width, Q4_0 with
  partial tiles, Q4_0 on the row-major layout, Q4_K with partial tiles) and
  `dynamic_tgmem_kernel_argument_and_launch_length_always_agree` (kernel argument present exactly when the spec's length
  is 8192, for Q4_0 and Q4_K, absent when any of the three things it rides on is off, absent for the dense-batched
  path). Control: with the spec forced to `0` and the kernel still declaring the argument, the prefill-width case fails
  (`EXIT=100`).
- `3a933038` the E2B prefill-width fixture and test (above).

Abandoned: reading the 20% as an epilogue cost (ggml's own kernel with the restage epilogue forced is -0.3%, so in
ggml's structure the epilogue form is free, while direct store is worth 3.0% on ours; both numbers are measured and the
difference is not explained); a half-precision activation tile (slice 2 row 11; the `float`-activation ggml kernel is
bit-identical to ours, so precision was never the gap); a toggle search over the old row-major kernel (replaced, as in
slice 2, by transcription, now with a bit-exact `float` ggml arm as the oracle for each step); a new type to carry the
write-back parameters (the helper was removed instead, so there is neither an `allow` nor a type).

### before and after

Isolated kernel, `mm_kernel_ladder`, p25 of 21 dispatches, 2 processes per cell, fix tip =
`evidence/slice2fix/ladder/sweep_final_run{1,2}/`. The slice 2 tip column is the slice 2 probe
(`tiled_gemm_q4_0_speed_probe`, p25 over 3 processes, `evidence/slice2/probe_final/`), a different harness timing the same
production kernel with the same GPU-span timer. Ratio is production over ggml's own `half`-activation kernel
(design-favors: incumbent), same dispatch, same data, same clock, from the ladder; ggml's launch binds the threadgroup
memory its host code picks (`ggml-metal-device.cpp`: 8192 bytes when the output needs bounds checks, 6144 when not;
the first sweeps of this fix gave ggml 8192 in every cell, and re-running the 512- and 160-token cells with 6144 moved
ggml's microseconds by 0.0% to 0.1% in the cells with a CoV under 5%):

| weight (rows x K) | tokens | slice 2 tip us (probe) | fix tip us (pass 1 / pass 2) | ggml us (pass 1 / pass 2) | fix tip / ggml (pass 1 / pass 2) |
|---|---|---|---|---|---|
| 12288 x 1536 | 971 | 5620.2, 5626.8, 5619.4 | 4686.9 / 4686.8 | 4520.9 / 4520.7 | 1.0367 / 1.0367 |
| 6144 x 1536 | 971 | 2823.1, 2823.3, 2823.4 | 2354.7 / 2354.7 | 2272.7 / 2272.7 | 1.0361 / 1.0361 |
| 4096 x 1536 | 971 | 1893.1, 1894.6, 1893.6 | 1578.3 / 1578.2 | 1523.2 / 1523.1 | 1.0362 / 1.0362 |
| 2048 x 1536 | 971 | 964.4, 964.6, 965.1 | 801.6 / 801.9 | 774.2 / 773.8 | 1.0354 / 1.0363 |
| 1536 x 6144 | 971 | 2880.6, 2879.9, 2879.9 | 2417.0 / 2419.5 | 2378.6 / 2381.6 | 1.0162 / 1.0159 |
| 1536 x 12288 | 971 | 5741.6, 5738.0, 5737.4 | 4843.6 / 4839.8 | 4775.7 / 4777.4 | 1.0142 / 1.0131 |
| 12288 x 1536 | 512 | n/a | 2430.2 / 2429.7 | 2320.8 / 2320.7 | 1.0471 / 1.0470 |
| 12288 x 1536 | 160 | n/a | 776.6 / 776.0 | 741.7 / 741.4 | 1.0471 / 1.0466 |

At the slice 2 tip the ladder's `prod` for 12288 x 1536 x 971 is 5607.9 us, 1.2404 of ggml (run0). The slice 2 spec
table put the six shapes at 1.216, 1.215, 1.217, 1.219, 1.184 and 1.170 of llama's op, from `test-backend-ops` wall clock
rather than a GPU span. The sweep after the loop schedule alone (`6e637861`, direct store off, static array;
`ladder/sweep_after_k_loop_schedule/`, one pass, ggml given 8192 in every cell) reads 1.0494 to 1.0973 at 971 tokens;
with direct store on (`ladder/sweep_direct_store/`, 12 cells at 512 and 971 tokens) every cell is 0.7% to 3.0% faster
than without. All 18 fix-tip cells (6 shapes x 160, 512, 971 tokens) over both passes: 1.0131 to 1.0471 in the 28 cell
measurements where neither arm has a CoV above 5%, and 1.0131 to 1.0988 over all 36. Eight of the 36 cell measurements
have a production or ggml CoV above 5%, all at 160 or 512 tokens: pass 1, 2048 x 1536 x 160 (production 3.16%, ggml
5.21%), 2048 x 1536 x 512 (11.38%, 11.40%), 4096 x 1536 x 160 (13.73%, 14.06%), 6144 x 1536 x 160 (10.87%, 12.75%, ratio
1.0988); pass 2, 1536 x 6144 x 160 (10.94%, 9.65%), 2048 x 1536 x 160 (8.80%, 6.99%), 2048 x 1536 x 512 (10.47%,
10.29%), 4096 x 1536 x 160 (13.82%, 13.26%). In all eight the ggml arm has a CoV above 5% (production does in seven), and the same
cell differs between passes in absolute terms (6144 x 1536 x 160: 662.8 us in pass 1, 401.0 us in pass 2; 2048 x 1536
x 512: 426.2 us and 566.1 us) while the ratios of the two arms stay near each other except the one 1.0988 cell.

Census, `gemma4_decode_kernel_census` (`M0_CAPTURE_STEPS=0`, `prompt1k.txt`), one-command-buffer sequence replay, 7
replays, Q4_0 class of 275 dispatches (`evidence/slice2fix/census_*/stdout.log`):

| tree | Q4_0 class ms (CoV) | all 1568 dispatches ms (3 replays) | own-cb class (`rank.md`) |
|---|---|---|---|
| slice 2 tip (`6e8729fe`, slice 2 evidence) | 577.75 (0.04%) | 1429.34, 1430.86, 1429.73 | 577.77 |
| `6e637861` (loop schedule) | 510.27 (0.02%) | 1361.91, 1361.47, 1362.85 | 510.70 |
| `6e637861` with `DIRECT_STORE=1` | 502.77 (0.03%) | 1356.31, 1353.48, 1355.76 | 503.03 |
| `0c4ff281` (fix tip, all defaults) | 489.41 (0.02%) | 1341.27, 1342.29, 1342.45 | 489.99 |

At the fix tip the six Q4_0 shapes, own-cb ms against llama's (`census_fix_tip/rank.md`): 1536 x 12288 296.87 against
284.91; 1536 x 6144 111.65 against 110.18; 1536 x 4096 22.39 against 22.83; 1536 x 2048 45.60 against 47.83; 1536 x 512
1.38 against 1.57; 1536 x 256 12.10 against 15.67; class 489.99 against 482.99 (target at most 500). Direct store on the dense-batched folds: the census family
`rope_copy_elementwise` (it holds the 168 dot and AV dispatches, the softmax and rope dispatches; `sequence_family` in
`gemma4_decode_kernel_census.rs`) reads 377.24 ms with direct store off and 377.16 ms with it on, in the two censuses of
`6e637861`: no signal there.

Interleaved final run, `evidence/slice2fix/interleaved/decode_arms.out`: 5 processes x (1 warm-up + 7 timed) = 35 timed
runs per arm, arms rotated per process, Ollama stopped (osascript refused; SIGTERM of the app and the server, `/api/ps`
refused, no Ollama process), prompt `prompt1k.txt` (971 tokens E2B, 1000 granite), 128 new tokens. Arms: `base` =
`decode_gbps_baseline_base_s2` (sha256 `e3db0a4b...f21`, HEAD before slice 2), `prev` = `decode_gbps_baseline_s2_mml`
(sha256 `3f560083...b54`, the slice 2 tip), `tip` = `decode_gbps_baseline_s2fix_3a933038` (sha256 `4bb13cb5...eac6`),
`tipcopy` = a byte copy of `tip` (same sha256), llama-server `f1ea20621`. Box before: load average 5.73;
a background daemon in `~/.local/bin` 69.3% CPU, `mediaanalysisd` 61.0%, `mds_stores` 52.6%, iTerm2 30.7% (`box_load_before.txt`); after: load average 2.92,
`suggestd` 90.3%, the same daemon 90.2% (`box_load_after.txt`). No cargo, GPU or Ollama process of mine ran during it.

| arm | prefill ms (median kept, CoV all, range all) | TTFT ms | decode ms/token (median kept, CoV all) | peak RSS (median of 5) | peak footprint | peak GPU bytes |
|---|---|---|---|---|---|---|
| E2B base | 2403.96, 0.27%, 2395.97-2420.98 | 2404.0 | 11.910, 4.96% (one 15.05 run) | 3.964 GB | 709.9 MB | 5,608,554,496 |
| E2B prev (slice 2 tip) | 1497.00, 0.46%, 1491.03-1516.01 | 1497.0 | 11.876, 4.71% | 3.949 GB | 718.4 MB | 5,608,554,496 |
| E2B tip | 1413.94, 0.47%, 1404.05-1432.03 | 1413.5 | 11.9005, 6.19% (kept n=24: 0.34%) | 3.997 GB | 712.3 MB | 5,608,554,496 |
| E2B tipcopy (control) | 1412.94, 0.43%, 1405.02-1425.99 | 1413.0 | 11.8905, 6.89% (kept n=24: 0.96%) | 3.968 GB | 725.0 MB | 5,608,554,496 |
| E2B llama-server | 569.69, 0.39%, 567.58-576.87 | 572.77 | 8.9889, 0.50% | 3.751 GB | 219.5 MB | n/a |
| granite base | 833.96, 1.82%, 827.97-889.06 | 834.0 | 14.804, 0.63% | 2.531 GB | 609.4 MB | 3,315,433,472 |
| granite prev | 836.06, 1.06%, 830.95-872.05 | 836.0 | 14.823, 2.17% | 2.520 GB | 630.8 MB | 3,315,433,472 |
| granite tip | 837.02, 1.47%, 832.00-874.95 | 837.0 | 14.810, 0.59% | 2.558 GB | 616.5 MB | 3,315,433,472 |
| granite tipcopy (control) | 837.02, 0.96%, 829.98-864.96 | 837.0 | 14.8075, 0.74% | 2.491 GB | 601.1 MB | 3,315,433,472 |
| granite llama-server | 150.19, 0.44% | 152.06 | 5.2336, 2.12% | 1.763 GB | 280.5 MB | n/a |

Bound lines, E2B (`bound metric=... arm=tip vs=..., limit = max(2% of the reference median, twin gap)`): tip against
base, prefill -990.02 ms (limit 48.08) within=true, TTFT -990.50 within=true, decode -0.0095 ms/token (limit 0.2382)
within=true, RSS +33.0 MB (limit 79.3) within=true, footprint +2.4 MB (limit 14.2) within=true, GPU bytes 0; tip against
prev, prefill -83.06 ms (limit 29.94) within=true, decode +0.0245 within=true, RSS +47.6 MB within=true, footprint -6.1
MB within=true. Tip against llama-server: prefill +844.25 ms (limit 11.39) within=false, TTFT +840.73 within=false,
decode +2.9116 ms/token (limit 0.1798) within=false, RSS +245.7 MB within=false, footprint +492.8 MB within=false. Granite,
tip against base: prefill +3.06 (limit 16.68) within=true, decode +0.006 (limit 0.2961) within=true, RSS +27.2 MB
(limit 50.6) within=true, footprint +7.1 MB (limit 12.2) within=true, GPU bytes 0; tip against llama-server `within=false`
on every metric that has a llama value (prefill +686.83 ms, decode +9.58 ms/token). Token ids: `ids ... equal=true` on 320 of 320 proxima
generations (2 models x 4 arms x 5 processes x 8 runs including the warm-ups), 128 ids each; `equal=false` 0.

### AC2

AC2 (`metric=prefill_ms arm=tip vs=llama` within=true on E2B) is not met: tip 1413.94 ms, llama 569.69 ms, +844.25 ms
(2.48x), limit 11.39. What the census says the remaining gap is made of, own-cb ms at the fix tip against llama
(`census_fix_tip/rank.md`, llama column from `evidence/slice0/llama_ops/e2b_ops.tsv`):

| class | proxima ms | llama ms | gap ms |
|---|---|---|---|
| attention core (7 `cached attention partial` 337.7, 112 dot 177.1, 56 AV 140.8, softmax, rest) | 686.30 | 41.31 | +644.98 |
| matmul (weights), of which Q4_0 489.99 against 482.99, F16 projection 107.73 against 4.35 | 597.74 | 487.35 | +110.39 |
| rms norm | 24.03 | 13.68 | +10.35 |
| rope | 9.34 | 3.37 | +5.98 |
| output head | 1.12 | 0.94 | +0.18 |
| elementwise, copy, other | 6.75 | 22.21 | -15.46 |
| sum of classes | 1325.28 | 568.86 | +756.42 |

The measured prefill gap is 844.25 ms and the class sum is 756.42 ms; the 87.83 ms between them is time the own-cb sum does not
hold; the slice 0 timeline's host figures (plan prepare 23.0, pre-encode 8.6, readback 6.7, kv 6.2, commit to GPU start
14.1) sum to 58.6 ms, and the rest of the 87.83 ms is untraced. The spec's own slice list (`proposed slices`, item 4) already states that
after slices 2 to 4 E2B prefill is about 770 ms against llama's 573 and that parity "needs further slices after
re-attribution". Slice 3 is the 644.98 ms attention row; the F16 projection (103.38 ms of the matmul row) is the last codec of slice 4
as rewritten in `ae2c5847`. So AC2 is a program-level criterion reached by slices 3 and 4 and the further slices the
spec names; slice 2's own measured target in the spec is the Q4_0 class row, which is met. Within slice 2, the
matmul-class gap is +110.39 ms, of which +103.38 ms is the F16 projection and +7.0 ms the Q4_0 shapes.

### granite memory

The slice 2 interleaved run (3 processes per arm) read granite peak RSS +88.2 MB (limit 49.7) and footprint +16.7 MB
(limit 12.0) for tip against base, `within=false`. In this run (5 processes per arm: the same two binaries plus the fix
tip) the same two lines read +27.2 MB (limit 50.6) and +7.1 MB (limit 12.2), `within=true`. The same binary against
its own copy moves as far as the binaries do: tipcopy against tip RSS -66.3 MB and footprint -15.4 MB; and on E2B,
tipcopy against base footprint reads +15.04 MB (limit 14.20) `within=false` while tip against base reads +2.38 MB
`within=true`, so one binary sits on both sides of the bound. The only other `within=false` between proxima arms in
this run is granite `prev` against `base` footprint (+21.4 MB, limit 12.2). Per-process peak RSS
(`interleaved/granite_time_l_per_process.txt`, `/usr/bin/time -l`): base 2475 to 2637 MB (span 161), prev 2492 to 2596
(104), tip 2423 to 2726 (303), tipcopy 2471 to 2602 (131); footprint base 570 to 624 MB, prev 591 to 649, tip 585 to 651,
tipcopy 565 to 640. The four arm medians span 2491 to 2558 MB (RSS) and 601 to 631 MB (footprint), 66 MB and 30 MB, and
the identical-binary pair (tip, tipcopy) accounts for 66.3 MB and 15.4 MB of those; within one arm the spread is larger
than any between-arm shift of a median.

Where the spread lives, from `vmmap -summary` on live processes (`granite_vmmap/`: 3 rounds of base then tip, snapshots
at 5, 11 and 17 seconds after launch, 6 processes, 18 snapshots, plus one more process at 9 seconds): `mapped file` is
1.3 GB resident in all 19 snapshots (the model); `IOAccelerator (graphics)` is 256.5 MB resident at 5 and 9 seconds and
119.9 MB at 11 and 17 seconds in every snapshot; `MALLOC_NANO` is 27.2 to 28.0 MB; `MALLOC_MEDIUM` is the class that
moves: resident 354.3 to 629.0 MB across the 18 (base 378.9 to 572.0 MB, tip 354.3 to 629.0 MB), while its dirty size is
111.8 to 132.6 MB in the four rows read in full (`tip_r1_t17` 365.9 resident / 132.6 dirty; `tip_r2_t17` 629.0 / 120.0;
`base_r2_t17` 382.5 / 121.4; `base_r3_t17` 572.0 / 111.8). The `TOTAL` resident row moves with it (2.2 GB for
`tip_r1_t11`, 2.6 GB for `tip_r2_t17`). Across the five tip processes RSS tracks `page reclaims` (321,809 to 343,214
reclaims for 2422.2 to 2725.9 MB, 14.2 KB per reclaim, a 16 KB page being the unit). The same `MALLOC_MEDIUM` row also
moves inside one process (`tip_r2`: 541.0, 607.0, 629.0 MB at 5, 11, 17 s; `base_r1`: 499.7, 381.0, 509.5 MB). So the
spread is resident, mostly clean, medium-size malloc pages, in both binaries; why one process keeps more of those pages
resident than another is not traced, and no between-binary signal was found.

### provenance

The timed `tip` binary is `decode_gbps_baseline` built with `cargo build --release -p proxima-model-interop --features
std,metal --example decode_gbps_baseline`, `CARGO_TARGET_DIR=/private/tmp/cargo_target_arch`, from commit
`3a9330380f9f04f9a9adccfcd0120928f551254f` with `git status --short` showing only the untracked
`proxima-tensor/specs/decode-as-data/` (`interleaved/tip_commit.txt`); sha256 `4bb13cb53f7771c3cabd0f5d249c28ca8967e8566468fa62f862bc7c6dfeeac6`. After
touching `omega/src/lib.rs` and `proxima-model-interop/src/lib.rs`, a rebuild recompiled both crates and produced the same
sha256, and a second rebuild at `2e2933e6` (after a local-variable rename in `tiled_gemm_cooperative_scan.rs` and the example
fixes) did too (`interleaved/rebuild_at_2e2933e6.sha256`), so the build re-proves the binary from the commit. `base` and `prev` hashes equal the ones the slice 2 run recorded
(`slice2/interleaved/binaries.sha256`). `git diff --stat 3a933038..2e2933e6` lists four files: `omega/examples/mm_kernel_ladder.rs`,
`omega/examples/tiled_gemm_render_diff_check.rs` (examples that `decode_gbps_baseline` does not link),
`omega/src/msl/tiled_gemm_cooperative_scan.rs` (a local variable renamed and one line-number citation in a doc comment,
10 lines) and this `SPEC.md` (`ae2c5847`); the rebuilt binary from that tree has the same sha256.

### attention class

Slice 0 recorded the attention core class at 749.70 ms and slice 2's re-attribution at 691.59 ms, with the cause of the
58.1 ms not traced. Re-measurement (`attn_trace/`, interleaved: slice 0 source tree built from `c4810cb7`, whose omega,
tensor and interop sources equal `f76b4a97`'s; the fix tip with every slice 1 and 2 switch pinned to `0`
(`WIDE_WEIGHT_STAGE`, `GRID2D`, `MM_LAYOUT`, `DIRECT_STORE`, `DYNAMIC_TGMEM`); the fix tip with defaults; two rounds
each, census as above, `attribution_rank rank` with the slice 0 llama file):

| binary | attention core class ms (own-cb) | Q4_0 class ms (own-cb) | sequence replay ms |
|---|---|---|---|
| slice 0 source tree, round a / b | 691.66 / 691.07 | 1483.25 / 1478.63 | 2335.8, 2337.2, 2335.2 / 2333.0, 2334.5, 2335.5 |
| fix tip, switches pinned to 0, a / b | 691.02 / 690.73 | 1479.02 / 1482.16 | 2336.2, 2333.9, 2333.6 / 2334.0, 2336.3, 2337.9 |
| fix tip, defaults, a / b | 686.55 / 686.30 | 490.17 / 489.99 | 1342.1, 1342.4, 1342.6 / 1342.1, 1345.9, 1346.1 |

The slice 0 source tree measures 691.66 and 691.07 ms now, where the slice 0 record was 749.70; the Q4_0 class of the
same tree reads 1483.25 and 1478.63 now against 1479.66 recorded. The fix tip with the switches pinned off equals the slice
0 tree to within 0.1% on the attention class and 0.3% on the Q4_0 class, so the code did not change the attention class
between slice 0 and slice 2; the difference sits in the slice 0 measurement. By label, on the same own-cb basis (slice 0
record `evidence/slice0/rank/e2b_prefill.md` against `attn_trace/slice0_a/rank.md`; the six labels are the whole
346-dispatch class):

| label | dispatches | slice 0 record ms (own-cb / marginal) | slice 0 tree now ms (own-cb / marginal) | difference ms (own-cb) |
|---|---|---|---|---|
| cached attention partial | 7 | 344.70 / 343.56 | 337.79 / 337.85 | -6.91 |
| attention dot | 112 | 212.16 / 178.72 | 178.40 / 177.92 | -33.76 |
| attention AV | 56 | 145.11 / 145.16 | 144.74 / 144.32 | -0.37 |
| softmax exp | 56 | 22.87 / 22.71 | 13.83 / 13.82 | -9.04 |
| softmax sum | 57 | 16.30 / 15.99 | 8.36 / 8.21 | -7.94 |
| softmax max | 58 | 8.56 / 8.58 | 8.54 / 8.44 | -0.02 |
| class | 346 | 749.70 | 691.66 | -58.04 |

Inside the slice 0 record the dot row's own-cb figure (212.16) is 33.4 ms above its own marginal figure (178.72), where
now they agree to 0.5 ms; the softmax exp and sum rows are 40% and 49% lower per dispatch now in both columns (408.3 and
285.9 us then, 246.9 and 146.6 us now). Why those rows read higher in the slice 0 session is unmeasured (GPU clock
state or another GPU tenant at the time are candidates; no clock or tenant record from that session exists). The fix
tip's defaults take 4.5 ms (0.65%) off the class against the five switches pinned off; which of the switches does it is
not isolated (direct store alone moved the `rope_copy_elementwise` family by 0.08 ms in the `6e637861` pair).

### the three checks

- correctness: `omega` 599 passed, 16 skipped (592 before this fix, plus the emitter assertion, 4 dynamic memory cases,
  1 prefill-width direct-store case, 1 agreement test; `gate/omega_full.tail`); `proxima-tensor` 779 passed, 8 skipped
  (`gate/tensor_suite.tail`); `proxima-model-interop` slice-gate 710 passed, 124 skipped (709 before plus the E2B
  prefill-width test; `gate/interop_slice_gate.tail`); clippy `-D warnings` exit 0 for `proxima-tensor`,
  `proxima-model-interop` and `omega` (`--features proxima-model-interop/std,proxima-model-interop/metal,omega/metal
  --all-targets`) and for `omega` with `instrument`; tier builds exit 0: `omega --no-default-features --features
  metal-core`, `metal-core,metal-tiled-gemm`, `alloc`; `proxima-model-interop --no-default-features`; `proxima-tensor
  --no-default-features --features alloc` (the new code is behind `metal-tiled-gemm`, which implies `metal`; the
  `metal-core` and `alloc` builds compile none of it, and `dynamic_tgmem_active` has a `not(metal-tiled-gemm)` arm that
  returns false, which those builds do compile). The feature-gated allocation-count tests (`named_placement_alloc_count`, `plan_pipeline_alloc_count`, run with
  `--features metal,alloc-count`: 6 passed, `gate/alloc_count.tail`, among them `a_warm_call_s_allocation_count_does_not_grow_with_extra_steps`)
  and `decode_step_telemetry_budget` (in the 599) pass. Mm layout and dynamic memory against their
  off form: 0 differing words of 248,576 (971 tokens x 256 rows) and of 51,000, 65,280, 65,536 in the other cases, each
  printing its count (13 lines, 13 of 13 at `differing_words=0`, `gate/byte_identity_words.txt`). Allocation: the emitter allocates source text at plan time as
  before; the launch adds one `setThreadgroupMemoryLength` call and no allocation per dispatch; hot-path budget stated 0,
  measured: not separately instrumented (the existing tests above are the evidence).
  The write-back refactor is emission-neutral: 12 rendered kernels byte-identical at `41736dc4` and `7c5e25c4`
  (`render_identity/`, above).- semantic: ids equal to llama.cpp `f1ea20621` on 320 of 320 proxima generations of the interleaved run; owner gate sets
  on the fix tree: `llama_parity_` 7 passed, `generic_verify_llama_parity_` 5 passed,
  `prefill_width_parity_with_llama_granite_moe` and `prefill_width_parity_with_llama_gemma4_e2b` 2 passed (14 run,
  `gate/owner_parity.tail`); the E2B fixture has 971 prompt ids so it prefills through the mm-layout, direct-store and
  dynamic-memory kernel. Control: with one oracle id incremented the E2B test fails (`EXIT=100`), fixture restored
  byte for byte.
- performance: the tables above. E2B decode, RSS, footprint and GPU bytes sit inside the base and prev bounds; E2B
  prefill and TTFT -990 ms against base and -83 ms against the slice 2 tip; granite prefill, decode, RSS, footprint and
  GPU bytes sit inside the base bounds.

### incumbent, home-turf arm

llama.cpp `f1ea20621` `kernel_mul_mm_q4_0_f32`, run inside the ladder (not `test-backend-ops`): ggml's own kernel, grid,
kargs and function-constant values for each shape, same dispatch clock as the production kernel, same data, interleaved,
design-favors: incumbent. At 971 tokens the fix tip is 1.0131 to 1.0367 of ggml's time on the six E2B shapes; ggml with a
`float` activation, which is bit-identical to production, is 1.0122 of ggml's own. Frequency: all 275 Q4_0 dispatches
of one 971-token request run this op family (class 489.99 ms of the 1342 ms step at the fix tip).

### discipline log, one row per tweak

Ladder rows are p25 of 21 dispatches at 12288 x 1536 x 971, interleaved against the production kernel and ggml in one
process, CoV 0.02% to 0.13%, box load 3 to 6 with the background processes listed above, release build of the example.
Rows 1 to 10 are variants in `ladder/variants/`; rows 11 to 13 are the three landed commits; rows 14 and 15 are the
production measurements.

| row | change | measurement | delta vs prior | CoV, runs | status |
|---|---|---|---|---|---|
| 0 | slice 2 tip, ladder baseline | 5607.9 us, 1.2404x ggml (4520.9 us) | baseline | 0.02%, 21 | |
| 1 | ggml with a `float` activation tile, the numerics oracle | 4575.3 us; 0 of 11,931,648 words differ from row 0 | n/a | 0.06%, 21 | informs |
| 2 | unroll metadata on the multiply loops (p1) | 5485.7 us | -2.2% | 0.03%, 21 | kept, inside row 11 |
| 3 | ggml's multiply order (p3) | 5484.8 us | -0.0% vs row 2 | 0.05%, 21 | not kept, no signal |
| 4 | activation as one `float2x4` (p4) | 5485.6 us | 0.0% vs row 2 | 0.06%, 21 | not kept, no signal |
| 5 | carried activation pointer (p5) | 5420.5 us | -1.2% vs row 2 | 0.03%, 21 | kept, inside row 11 |
| 6 | multiply section transcribed from `mul_mm.metal` (p6) | 5143.4 us | -6.2% vs row 2 | 0.08%, 21 | kept, inside row 11 |
| 7 | tile array as launch-bound dynamic memory (p6_dyn, p7_dyn) | 5007.4 us, 4929.0 us | -2.6%, -2.7% | 0.07%, 0.05%, 21 | kept, row 13 |
| 8 | ggml barrier order (p8_dyn) | 4804.0 us | -2.5% vs p7_dyn | 0.03%, 21 | kept, inside row 11 |
| 9 | direct device store (p9_dyn) | 4661.5 us | -3.0% vs p8_dyn | 0.08%, 21 | kept, row 12 |
| 10 | same, static array (p9_static) | 4803.5 us | +3.0% vs p9_dyn | 0.05%, 21 | the static array is what row 7 removes |
| 11 | `6e637861` loop schedule (rows 2, 5, 6, 8 minus the dynamic memory and the store) | emitter 4959.9 us; Q4_0 class census 577.75 -> 510.27 ms | -11.5% kernel, -11.7% class | 0.03%, 21; 0.02%, 7 | landed |
| 12 | `888deca1` direct store default on | Q4_0 class 510.27 -> 502.77 ms; kernel cells -0.7% to -3.0% in 12 of 12 cells | -1.5% class | 0.03%, 7; ladder 21 | landed |
| 13 | `0c4ff281` dynamic threadgroup memory | emitter 4687.3 us; class 502.77 -> 489.41 ms | -2.7% class, -16.4% kernel against row 0 | 0.06%, 21; 0.02%, 7 | landed |
| 14 | production, 6 shapes x 3 token counts, 2 passes | 1.0131 to 1.0471 of ggml in the 28 cell measurements with both CoVs at or below 5%, 1.0131 to 1.0988 over all 36 | 1.170 to 1.219 before (spec table, llama op wall-clock timer, not the same timer) | 8 of 36 cell measurements have ggml CoV above 5% (production above 5% in 7 of them), all at 160 or 512 tokens | measured |
| 15 | E2B prefill, interleaved, 35 runs per arm | 1497.00 -> 1413.94 ms against the slice 2 tip; 2403.96 against base | -83.1 ms (-5.5%) | 0.47%, 35 | landed |

Rows that moved nothing are kept: p2, p3, p4 (rows 3, 4), the epilogue ablation (`ggml_grestage`, -0.3%), the
decode and activation swaps inside ggml's kernel (-1.1%, 0.0%): no signal at the 21-sample CoV of 0.02% to 0.13%, so
they are not the lever.

### not met, not traced, not done

- AC2 is not met (above), and cannot be met by this slice: the classes that hold it are the attention core (+644.98 ms,
  slice 3) and the F16 projection (+103.38 ms, the last codec of slice 4 as rewritten in `ae2c5847`).
- Why the multiply section's form, the unroll metadata and the static tile array cost what they cost on the device:
  not traced. The AIR is the same size across them; the device compiler's allocation differs (`maxTotalThreadsPerThreadgroup`
  768, 832, 896 across rungs) but nothing here decodes what it emits. GPU counters: the slice 2 `xctrace` recording is
  still undecoded; nothing in this fix rests on it.
- The remaining 1.3% to 3.7% against ggml's `half` kernel at 971 tokens (ggml's own `float`-activation form measures
  1.2% above its `half` form): not isolated. Candidates not toggled: `short` against `long` index types in the loop prologue, the `u.*` uniform reads
  against ggml's kargs, ggml's `il` state machine against the carried block pointer.
- The dense-batched attention folds (`attention dot`, `attention AV`, 168 dispatches, 317.8 ms own-cb) still run the
  row-major layout, static memory and the old multiply section; the ladder technique applies to them (slice 3).
- Why one granite process keeps more resident clean `MALLOC_MEDIUM` pages than another (above).
- The Q4_K path of the mm layout gets the same loop schedule and dynamic memory (byte-identity cases pass); no E2B
  dispatch uses it, so its timing is not measured.
- Large-N and `K`-tail shapes: the ladder covers K = 1536, 6144 and 12288 and 160, 512 and 971 tokens; the 275 dispatches of the census cover the real shapes.

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate                                   # 599 tests
cargo nextest run -p omega --features metal --cargo-profile gate -E 'binary(tiled_gemm_mm_layout_parity)'   # 13
cargo clippy -p proxima-tensor -p proxima-model-interop -p omega --features proxima-model-interop/std,proxima-model-interop/metal,omega/metal --all-targets -- -D warnings
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate   # 710
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate \
  -E 'test(llama_parity_) or test(generic_verify_llama_parity_) or test(prefill_width_parity_with_llama)'   # 14
LADDER_ROWS=12288 LADDER_K=1536 LADDER_TOKENS=971 LADDER_VARIANTS=<variant.metal,...> LADDER_DUMP=<dir>/prod.metal \
  cargo run --release -p omega --example mm_kernel_ladder --features metal    # needs a llama.cpp f1ea20621 checkout at LADDER_GGML_DIR
xcrun -sdk macosx metal -std=metal3.0 -fmetal-math-mode=relaxed -O2 -S -emit-llvm -w -o out.ll <source>   # IR counts
gemma4_decode_kernel_census with M0_CAPTURE_STEPS=0 M0_MAX_TOKENS=2 PROXIMA_PROMPT_FILE=prompt1k.txt   # switches pinned off: PROXIMA_TILED_GEMM_{WIDE_WEIGHT_STAGE,GRID2D,MM_LAYOUT,DIRECT_STORE,DYNAMIC_TGMEM}=0
attribution_rank rank --census DIR --llama evidence/slice0/llama_ops/e2b_ops.tsv --ntok 512,455,4 --requests 3 --floor-us 4.0
decode_arms --prompt-file prompt1k.txt --processes 5 --runs 7 --arm base=<e3db0a4b binary> --arm prev=<3f560083 binary> \
  --arm tip=<4bb13cb5 binary> --arm tipcopy=<copy of tip> --llama-server <llama-server f1ea20621> --case granite_moe=<granite blob> --case gemma4_e2b=<E2B blob>
decode_arms --processes 1 --runs 1 --llama-server <llama-server f1ea20621> --ignore-ollama --case gemma4_e2b=<E2B blob> --dump-llama-ids DIR   # the E2B fixture
vmmap -summary <pid>   # granite snapshots, 5, 11 and 17 s after launch
```

Missing for CI, as in slices 0 to 2: no job runs a GPU test or bench on Apple hardware, so the ladder, census and
interleaved numbers have no saved baseline to diff against; the byte-identity tests, the emitter tests, the clippy and
tier builds, and the E2B prefill-width fixture are the part that re-proves from the tree on a Mac with the E2B blob. The
ladder variants re-prove only by applying the diffs in `ladder/variants/` and running the ladder against a llama.cpp
checkout.
