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

## decode drift (measured 2026-10-07)

Owner question: the E2B decode numbers recorded on main moved (11.97, 12.2, 13.35, 12.21 ms/token); is that a code
regression or measurement contamination. Everything below is a measurement or a number derived from measurements, with
its source. No row is a verdict. Evidence root: `evidence/decode_drift/` (this directory). Raw per-process child
stderr (`launches.raw.*.err`, one per arm per process) sits in
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/regression_hunt/` next to the binaries (`bin/`) and the log dir
for every run; the committed `decode_arms.out` files carry every `raw` line.

### what ran

- Host: Apple M1 Max, macOS 24.6, AC power. HEAD `8ef12e94` release `std,metal` `decode_gbps_baseline` built for this
  task (`build_head.log`); its sha256 `4bb13cb5...eac6` equals the `3a933038` binary and the `2e2933e6` rebuild
  recorded in `evidence/slice2fix/interleaved/rebuild_at_2e2933e6.sha256`. `a16` (`3a933038`), `a17` (HEAD) and `a18`
  (byte copy of HEAD) are therefore three copies of one executable.
- Driver: `decode_arms` with the new `--new-tokens` flag (`4ea3fa38`, the only code change of this task), arms
  rotated per process, 1 warm-up + 7 timed runs per process per arm, Ollama not running for any run
  except the discarded condition `ollama_ctl/O1` (see below); the checks
  (`curl -s -m 1 http://127.0.0.1:11434/api/ps` refused, exit 7, and no Ollama process, before and after:
  `run1/box_load_before.txt`, `final_box.txt`, `run5_probe/box_load_after.txt`). One llama-server `f1ea20621` arm per
  full run, as that run's anchor.
- Box: load average 4.47 before run 1, 4.30 after. Top CPU before: the `~/.local/bin` daemon
  (a background daemon in `~/.local/bin`, pid 17029, started 2026-09-30 17:56; its full path and arguments are in the local copy of `run1/box_load_before.txt` under `regression_hunt/`, the committed copy redacts the name)
  74.2%, WindowServer 52.0%, iTerm2 10.6%, claude 5.4% (`run1/box_load_before.txt`). The daemon was running for every
  run except the two paused conditions below. `pmset -g therm`: no thermal, performance or CPU power warning recorded
  (before, `run1/box_load_before.txt`; after, `final_box.txt`). A `ps` sample every 60 s is in `run1/box_samples.txt`
  (77 samples, 11:49Z to 13:06Z, covers runs 1, 2, 3, the control conditions and `ollama_ctl`), `run4_long_subset/box_samples.txt`,
  `run5_probe/box_samples.txt`.
- Prompts: `prompt1k.txt` (971 tokens, 128 new tokens, sha256 in `prompt1k.sha256`; the file is
  `evidence/slice0/ac1/prompt1k.txt`); short chat prompts `prompt_short_hippo.txt` (26 tokens; the default prompt of
  `decode_gbps_baseline`) and `prompt_short_france.txt` (22 tokens), 64 new tokens, gemma4 chat template
  `<|turn>user\n...<turn|>\n<|turn>model\n`; both generated the full 64 tokens (`tokens_generated=64`, `stopped_by_eos=false`).
- Binary to commit mapping. Named by file or build log: `a00` 9f0647da, `a01` 152467a9, `a13` f76b4a97, `a14`
  (sha256 equals `base_s2`, the head before slice 2), `a15` slice 2 tip, `a16` 3a933038, `a19` the 9f0647da probe build.
  Inferred (no build log names the commit; matched by the binary's mtime against commit times): `a02`..`a12`, the
  `mapping` column below. `a03`, `a04`, `a07`, `a08` were built from uncommitted trees ahead of the named commit.

### the table: long prompt, 971 tokens, 128 new tokens (`run1/`)

Run 1: 19 proxima arms plus llama-server, 3 processes x 7 timed runs = 21 per arm, 2026-10-07 11:52Z to 12:24Z.
Median is over all 21 runs, MAD over the runs left after the driver's fixed outlier rule, CoV over all 21
(`run1/decode_arms.out` `summary` lines; `run1/table.md` is this table). Prefill is the headline column.

| arm | commit | mapping | prefill ms: median / MAD / CoV% | decode ms/token: median / MAD / CoV% |
|---|---|---|---|---|
| a00_0b_9f0647da | 9f0647da (row 0b) | named | 2533.0 / 3.0 / 0.23 | 12.245 / 0.138 / 1.86 |
| a01_0c_152467a9 | 152467a9 (row 0c) | named | 2538.0 / 2.0 / 0.21 | 12.288 / 0.185 / 1.77 |
| a02_slice5 | ecdd1bdd | inferred from mtime 10-05 11:45 | 2537.0 / 2.9 / 0.16 | 12.216 / 0.086 / 1.33 |
| a03_slice6 | tree before ce94f2d0 | inferred from mtime 10-05 15:20 | 2539.1 / 3.9 / 0.21 | 12.181 / 0.040 / 1.01 |
| a04_slice7 | tree before 53064bff | inferred from mtime 10-05 21:04 | 2537.0 / 3.0 / 0.17 | 12.147 / 0.037 / 0.71 |
| a05_slice8 | 3c42090d | inferred from mtime 10-06 04:25 | 2539.1 / 3.9 / 0.16 | 12.189 / 0.030 / 1.13 |
| a06_slice9 | bc686691 | inferred from mtime 10-06 09:48 | 2540.0 / 2.9 / 0.17 | 12.166 / 0.088 / 1.10 |
| a07_slice10b | tree before 5576a197 | inferred from mtime 10-06 13:15 | 2538.1 / 2.0 / 0.19 | 12.174 / 0.054 / 1.12 |
| a08_slice10b_fix | tree before 51df2ee7 | inferred from mtime 10-06 14:00 | 2540.0 / 4.0 / 0.17 | 12.200 / 0.098 / 1.29 |
| a09_slice10c | 74bdbb87 | inferred from mtime 10-06 15:29 | 2541.0 / 4.0 / 0.18 | 12.141 / 0.114 / 1.46 |
| a10_slice11 | 44f48761 / f2aea8bc | inferred from mtime 10-06 16:50 | 2540.0 / 4.1 / 0.24 | 12.178 / 0.051 / 1.16 |
| a11_slice12 | e9f94b2e | inferred from mtime 10-06 18:02 | 2542.0 / 1.5 / 0.27 | 12.227 / 0.087 / 2.59 |
| a12_eor | 42f375e2 | inferred from mtime 10-06 20:12 | 2540.0 / 4.0 / 0.20 | 12.119 / 0.031 / 0.95 |
| a13_f76b4a97 | f76b4a97 | named | 2543.0 / 3.9 / 0.20 | 12.118 / 0.061 / 0.57 |
| a14_tip_s1 | slice 1 tip, sha equals base_s2 | named, sha-identical to base_s2 | 2543.0 / 4.0 / 0.20 | 12.156 / 0.023 / 0.53 |
| a15_s2_mml | slice 2 tip 6e8729fe | named s2_mml | 1619.0 / 3.0 / 0.28 | 12.182 / 0.141 / 1.82 |
| a16_s2fix_3a933038 | 3a933038 | named | 1438.1 / 2.9 / 0.25 | 12.151 / 0.066 / 1.64 |
| a17_head_8ef12e94 | 8ef12e94 | built this task | 1439.0 / 3.0 / 0.32 | 12.153 / 0.035 / 0.81 |
| a18_head_copy | 8ef12e94 byte copy | control | 1438.1 / 2.0 / 0.26 | 12.122 / 0.030 / 1.36 |
| llama-server | reference | llama-server f1ea20621 | 571.6 / 2.4 / 2.23 | 9.154 / 0.061 / 1.32 |

Control (HEAD against its byte copy, against 0b): prefill 1439.0 against 1438.1 (difference 0.96 ms), against 0b 2533.0
(difference 1094.0); decode 12.153 against 12.122 (difference 0.031), against 0b 12.245 (difference 0.092). In run 1
the HEAD-to-copy difference is smaller than the HEAD-to-0b difference on both metrics.

### what the table shows, hurt first

- Decode: arm medians span 12.119 (`a12`) to 12.288 (`a01`), 0.169 ms, in run 1. No adjacent pair differs by
  more than 0.11 ms in either direction (largest rise `a10` to `a11` +0.049, `a00` to `a01` +0.043; largest fall `a11` to
  `a12` -0.108). That span is smaller
  than the 0.399 ms HEAD-to-copy difference in run 4 (below), so run 1 contains no step the harness can resolve.
- Prefill: 2533.0 (0b) to 2543.0 (`a13`, `a14`), +10.0 ms (+0.39%), a slow creep across `a01`..`a14` with no neighbour
  step above +5.1 ms (`a00` to `a01`); the HEAD-to-copy control is 0.96 ms. This is not bisected. It is inside the driver's
  own bound (median <= reference + max(MAD, 2% of reference), 2% of 2533 is 50.7 ms). In run 5 (below), the clean
  processes read 0b 2528.0, 0b probe 2530.0, 0c 2529.5 (`tools/metric_median`, 14 runs each), so the `a00` to `a01` step does not reproduce there.
- Prefill after 0b falls at slice 2 (`a15` 1619.0, `a16` 1438.1, HEAD 1439.0), 43% below 0b; llama-server 571.6.
- Decode against llama-server: HEAD 12.153 against 9.154, +2.999 ms; prefill 1439.0 against 571.6.

### short chat prompts (`run2_short_hippo/`, `run3_short_france/`)

Run 2: the same 19 arms plus the 0b probe build `a19` plus llama-server, 26-token prompt, 64 new tokens, 21 timed runs
per arm. Run 3: 6 arms, 22-token prompt, 64 new tokens, 4 processes = 28 timed runs per arm, no llama arm.

Run 2 (the CoV values above 5% in this table trace to process 1, see the noise section; `run2_short_hippo/table.md` has the commit column):

| arm | prefill ms: median / MAD / CoV% | decode ms/token: median / MAD / CoV% |
|---|---|---|
| a00_0b_9f0647da | 127.0 / 2.0 / 11.10 | 11.708 / 0.179 / 9.11 |
| a01_0c_152467a9 | 125.0 / 2.0 / 4.96 | 11.614 / 0.138 / 2.80 |
| a02_slice5 | 127.0 / 5.0 / 7.59 | 11.829 / 0.142 / 4.40 |
| a03_slice6 | 130.0 / 7.0 / 7.65 | 11.599 / 0.130 / 3.85 |
| a04_slice7 | 129.0 / 6.0 / 5.47 | 11.701 / 0.114 / 2.90 |
| a05_slice8 | 127.0 / 5.0 / 5.47 | 11.785 / 0.157 / 3.86 |
| a06_slice9 | 127.0 / 4.0 / 5.06 | 11.679 / 0.087 / 3.18 |
| a07_slice10b | 127.0 / 5.0 / 5.21 | 11.751 / 0.133 / 3.15 |
| a08_slice10b_fix | 132.0 / 8.0 / 7.53 | 11.715 / 0.106 / 4.95 |
| a09_slice10c | 123.0 / 0.0 / 6.98 | 11.627 / 0.059 / 2.94 |
| a10_slice11 | 124.0 / 1.0 / 6.34 | 11.620 / 0.013 / 1.73 |
| a11_slice12 | 128.0 / 6.0 / 8.92 | 11.612 / 0.038 / 2.35 |
| a12_eor | 123.0 / 1.0 / 4.94 | 11.617 / 0.064 / 2.79 |
| a13_f76b4a97 | 123.0 / 0.0 / 5.92 | 11.626 / 0.042 / 2.05 |
| a14_tip_s1 | 127.0 / 4.0 / 5.04 | 11.641 / 0.062 / 1.65 |
| a15_s2_mml | 126.0 / 1.0 / 8.85 | 11.558 / 0.041 / 8.59 |
| a16_s2fix_3a933038 | 129.0 / 5.0 / 6.58 | 11.573 / 0.048 / 1.16 |
| a17_head_8ef12e94 | 129.0 / 5.0 / 12.05 | 11.597 / 0.242 / 15.38 |
| a18_head_copy | 132.0 / 5.5 / 8.20 | 11.806 / 0.127 / 20.80 |
| a19_0b_probe_9f0647da | 134.0 / 10.0 / 7.64 | 11.863 / 0.347 / 4.19 |
| llama-server | 57.8 / 0.4 / 9.18 | 9.080 / 0.065 / 13.97 |

Run 3 (`run3_short_france/table.md`):

| arm | prefill ms: median / MAD / CoV% | decode ms/token: median / MAD / CoV% |
|---|---|---|
| a00_0b_9f0647da | 117.0 / 12.0 / 10.17 | 11.867 / 0.318 / 4.10 |
| a01_0c_152467a9 | 108.5 / 1.0 / 10.97 | 11.868 / 0.281 / 4.17 |
| a12_eor | 120.5 / 10.0 / 9.59 | 12.540 / 0.592 / 5.11 |
| a16_s2fix_3a933038 | 107.0 / 0.0 / 11.52 | 11.668 / 0.060 / 5.07 |
| a17_head_8ef12e94 | 106.0 / 0.0 / 8.67 | 11.620 / 0.033 / 3.81 |
| a18_head_copy | 120.0 / 14.0 / 11.60 | 11.627 / 0.022 / 3.50 |

- Short-prompt decode: arm medians 11.558 (`a15`) to 11.863 (`a19`) in run 2, 11.620 (HEAD) to 12.540 (`a12`) in run 3.
  Long-prompt decode over the same arms is 12.119 to 12.288. For the byte-identical HEAD: 12.153 (long, run 1), 11.597
  (short hippo, run 2), 11.620 (short France, run 3). The long-minus-short gap for HEAD is 0.556 and 0.533 ms.
  No short-prompt timed run is below 11.121 (`a15`, run 2).
- Short-prompt prefill: the minimum per arm is 120.99 to 124.01 ms in run 2 (hippo) and 103.97 to
  105.02 in run 3, with no trend across the arms; llama-server 57.75 median in run 2. The medians in the table are higher and noisy (CoV 5 to 12%) because
  of the bursts below.
- Cold first request of each process (`run_index=0`, the warm-up the table excludes), TTFT of HEAD and its copy: 220-229 ms
  for the 22-token prompt (8 values) and 239-254 ms for the 26-token prompt (6 values), against timed-run medians of 106.0
  and 120.0 ms (France, HEAD and copy) and 129.0 and 132.0 ms (hippo). The owner's recorded short-prompt TTFT of 207-491 ms was not produced by any run here; the
  nearest figures are these cold first requests. A repo search (`grep` over `proxima-tensor`, `omega`,
  `proxima-model-interop`, `docs` for those figures and for 10-11 ms/token) found no record of a 10-11 ms decode.

### control: does the measurement see a difference

Rule: if HEAD against its copy differs by more than HEAD against 0b, the measurement cannot see the regression.

| run | HEAD to copy | HEAD to 0b | decode metric | result |
|---|---|---|---|---|
| run 1, long, 19 arms (`run1/`) | 0.031 | 0.092 | median ms/token | copy difference smaller |
| run 4, long, 4 arms (`run4_long_subset/`) | 0.399 (HEAD 12.580, copy 12.181) | 0.360 (0b 12.220) | median ms/token | copy difference larger |
| run 2, short hippo, 20 arms | 0.209 | 0.111 | median ms/token | copy difference larger |
| run 3, short France, 6 arms | 0.007 | 0.247 | median ms/token | copy difference smaller |
| control U1 (daemon running) | 0.163 | 0.308 | median ms/token | copy difference smaller |
| control P1 (daemon paused) | 0.012 | 0.002 | median ms/token | copy difference larger (both below 0.02) |
| control P2 (daemon paused) | 0.025 | 0.128 | median ms/token | copy difference smaller |
| control U2 (daemon running) | 0.524 | 0.592 | median ms/token | copy difference smaller |

Prefill passes the control in every long run (copy difference 0.96 ms in run 1, 2.0 ms in run 4, against differences of
1000+ ms to 0b). For decode the largest byte-identical difference observed is 0.524 ms (U2), so differences below about
0.5 ms in a decode median are not resolvable by these runs.

### noise source

- Per-process state. For one binary the per-process medians (7 timed runs each) differ more than the within-process
  MAD. Run 4, HEAD: 12.687, 12.506, 12.406; its byte copy: 12.137, 12.258, 12.088. Run 1, `a16`: 12.462, 12.126, 12.151
  (`tools/per_process`). Run 5, 0b: 12.105, 12.130, 13.519, 12.938. Within a process the values are bimodal, about 11.5
  to 11.7 and 12.2 to 13.1 on the short prompt.
- The background daemon does not account for it. Pausing pid 17029 (`kill -STOP`, resumed by `kill -CONT` from a drop
  guard, `tools/pause_run.rs`; `pause_run: SIGSTOP`, `SIGCONT` and the process state before and after are in
  `control/P1/decode_arms.err` and `control/P2/decode_arms.err`; the daemon was in state R before and after, and R in
  `final_box.txt`). Short-prompt timed runs at or above 12.0 ms, out of 28 per arm
  (0b, eor, HEAD, copy): U1 daemon running 7, 10, 13, 11; P1 paused 12, 10, 7, 6; P2 paused 18, 12, 24, 21; U2 running
  4, 6, 16, 9 (`control/*/decode_arms.out`). The paused and running ranges overlap (paused 6 to 24, running 4 to 13 and
  4 to 16); the highest fractions of the four conditions are in paused P2.
- A system burst contaminated run 2, process 1. `run1/box_samples.txt:604` (2026-10-07T12:32:51Z): load average 46.43,
  `contactsd` 161.9% CPU, `contactsdonationagent` 32.9%, `AddressBookManager` 29.7%, Firefox media helper 22.6%, while `a00`
  ran (its process-1 runs: 12.016, 13.721, 12.906, 14.761, 15.343, 12.796, 13.205, 13.110). The previous sample, 12:31:51Z
  (`run1/box_samples.txt`), has load average 4.85. HEAD and its copy ran 12:32:06Z to 12:32:29Z in the same process,
  between the two samples; their process-1 timed runs span 11.263 to 17.043 (HEAD) and 11.806 to 20.943 (copy)
  (`run2_short_hippo/decode_arms.out`, `process=1`). The driver flags 4 of HEAD's 5 outliers and all 4 of the copy's
  outliers in process 1; HEAD CoV 15.4% and copy CoV 20.8% in run 2.
- External compiles contaminated run 5, processes 2 and 3: `cargo doc --workspace` (pid 69181) and
  `cargo test -p proxima ...` (pid 69849) were running from about 13:21Z (`run5_probe/box_samples.txt`, the 13:23:18Z
  sample lists `rustc` at 20% each; `ps` at 13:24Z); prefill in those processes rose to 2800-3415 ms
  (`run5_probe/decode_arms.out`). The 0b, probe and 0c comparison above uses processes 0 and 1 only.
- Mechanism of the per-process slow state: not traced. The instrument that would show it (GPU and CPU clocks via
  `powermetrics`) needs sudo (`sudo -n true` asks for a password), `ioreg` `AGXAccelerator` `PerformanceStatistics`
  carries utilization and memory but no clock. No thermal warning was recorded.

### the same binary, different launch contexts (long prompt, decode ms/token median)

`a13` (`f76b4a97`, sha256 `9c2dca1a...`), 21 timed runs unless noted:

| context | median | per-process medians | source |
|---|---|---|---|
| 19-arm rotation, run 1 | 12.118 | 12.111, 12.044, 12.149 | `run1/` |
| 4-arm rotation, run 4 | 12.188 | 12.113, 12.179, 12.200 | `run4_long_subset/` |
| alone, 3 processes back to back, C1 | 12.449 | 12.507, 12.464, 12.446 | `ollama_ctl/C1/` |
| alone, C2 | 12.467 | 12.423, 12.458, 12.954 | `ollama_ctl/C2/` |
| each process: arm then llama-server, N1 | 12.500 | 12.217, 12.554, 12.806 | `ollama_ctl/N1/` |
| each process: arm then llama-server, N2 | 12.563 | 13.166, 12.588, 12.215 (process 0 began 12:53:35Z, the minute of the Ollama stop below) | `ollama_ctl/N2/` |
| slice 0 AC1: arm, llama-server, Ollama (Ollama started and quit by the driver each process) | 13.348 (range 12.477-14.403) | p0 12.48-13.93, p1 13.19-13.79, p2 13.21-14.40 (ranges) | `evidence/slice0/ac1/decode_arms.out`, text above |

The row 0b recorded on 2026-10-05 (11.9735, 14 runs, `evidence/speed_baseline/decode_arms.out` in the architecture-as-data
spec) came from the same executable as `a00` (sha256 `07e6cc6c...`); `a00` reads 12.245 (run 1) and 12.220 (run 4) today.
The AC1 process 0 ran first in its run, before any Ollama request of that run, and read 12.48-13.93, so Ollama preceding
the proxima process does not by itself explain AC1's 13.348. Whether Ollama running in the rotation raises decode was
not tested: the condition O1 (driver `--ollama`) opened the Ollama app at 12:52:04Z and 12:53:02Z, the main thread
stopped it at about 12:53:35Z, the driver panicked at `decode_arms.rs:1034`, and O1 is discarded (`ollama_ctl/O1/`). Ollama
was not started again and is stopped. The cause of the lower reading inside a rotation than alone is untraced.

### findings

- Decode between 0b and HEAD: no adjacent arms differ by more than 0.11 ms in run 1; the same-binary spread across
  launch contexts today is 0.445 ms (12.118 to 12.563) and the byte-identical spread is up to 0.524 ms. No arm is
  worse than the noise by these controls, so there is no first-worse arm, no bisect, and no per-kernel census of two
  commits was taken (`gemma4_decode_kernel_census_*` binaries exist in `parity_perf/bin/`; none was run).
- Prefill: +10.0 ms (+0.39%) from 0b to f76b4a97 in run 1, not reproduced between 0b and 0c in run 5; a 1094 ms fall
  at slice 2 and slice 2 fix.
- Unmeasured: the cause of the per-process slow state; Ollama in the rotation; whether `a02`..`a12` are built from the
  commits in the mapping column; any commit before 9f0647da (no binary exists, none was built).

### re-prove

```
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm <label>=<binary> ... --llama-server <llama-server f1ea20621>    # run 1 (19 arms), run 4 (4 arms)
decode_arms --prompt-file prompt_short_hippo.txt --new-tokens 64 --processes 3 --runs 7 --arm ... --llama-server ...                 # run 2
decode_arms --prompt-file prompt_short_france.txt --new-tokens 64 --processes 4 --runs 7 --arm ...                                   # run 3
pause_run 17029 decode_arms --prompt-file prompt_short_hippo.txt --new-tokens 64 --processes 4 --runs 7 --arm ...                    # P1, P2
drift_table <decode_arms.out>      # the tables; summary_table, per_process give the cells quoted above
sha256: binaries.sha256 lists the 20 binaries; `shasum -a 256 -c` against the copies in regression_hunt/bin/
```

Missing for CI, as in slices 0 to 2: no job runs a GPU bench on Apple hardware, so none of these numbers has a saved
baseline to diff against; the numbers re-prove only on a Mac with the E2B blob and the 20 binaries.
# re-plan from the fusion, reduction and caching audit (2026-10-07, main 8ef12e94)

Replaces slices 3-7 of decode-prefill-parity. Ranked by milliseconds recovered on the own-cb basis,
measured against llama f1ea20621. Each slice keeps the existing three checks and must not slow decode.

1. Prefill attention as an MMA flash-attention kernel (about 645 ms on E2B, 274 ms on granite). Both
   parts are the same kernel problem:
   - 28 sliding layers are refused by the fusion at `dead_code_cached_attention.rs:1202` (window 512
     < 970 rows), so they run materialized dot, max, exp, sum and AV through device memory: 349 ms.
     35% of the exp work is on the empty, bucketed cached half.
   - 7 global layers (and all 24 granite layers) fuse onto the legacy per-row scalar kernel, because the
     row-tiled MMA kernel caps at 64 query rows (`omega-runtime.toml` `max_query_rows=64`). That kernel
     runs 48 ms per dispatch, at 0.16 TFLOP/s and 1.5% of peak.

   The fix, against llama `fa_*.metal`:
   - one row-tiled, key-tiled, `simdgroup_matrix` kernel with causal block skip and a windowed band
     form for the sliding layers;
   - admission of windowed prefill;
   - the row cap removed;
   - KV-head sharing across query heads.
2. One expert dispatch per (layer, projection) for MoE, `mul_mat_id` and `mul_mv_id` style (granite
   prefill 485 ms, decode part of 9.75 ms). Today `append_moe_ffn` unrolls one round per top-k slot
   (`gqa_layer_routed.rs:983-1010`, `PerRoute`), so 576 dispatches. `apply_moe_round_group_fusion`
   exists but is default-off with no interop passthrough. Admit top-k at prefill
   (`gdn_moe_fusion_apply.rs:966`, about 6 ms of the router chain).
3. Codec-generic tiled GEMM (the existing slice 4 text): F16 (107.7 vs 4.35 ms), Q8_0 dense (118 vs 15
   ms), Q3_K, Q5_0, Q5_1, Q5_K, Q6_K. One per-codec decode description consumed by one stager.
4. Granite decode host path:
   - set `command_buffer_chunks` in `profiles/granitemoe.toml`; it is config, and E2B uses 8;
   - trace and remove the 360 output-buffer allocations per step (1.5 ms of op setup that E2B does not
     pay).
   Target: granite decode at llama's 5.25 ms/token, together with slice 2.
5. Caching fusions:
   - a resident prefill plan keyed by (new_count, kv bucket, PlanIdentity), 23-27 ms per request;
   - an on-disk pipeline cache (MTLBinaryArchive), 75-93 ms per process start;
   - the first decode plan, 27-47 ms per process.
   Every key is a content digest that includes device, OS and Metal compiler version.
6. Norm-apply and rope dispatches:
   - 170 norm-apply stay unfused (cause untraced);
   - rope is 2 dispatches where llama uses 1;
   - about 16 ms of prefill, and part of decode.
7. Real serving prefill width: `ServingConfig::default().ubatch_size = 32` (`serving.rs:1175`) is below
   `TILED_GEMM_MIN_TOKENS = 160`, so real serving never reaches the tiled kernels the benches measure.
   Measure TTFT in the real serving path, and make ubatch and the kernel thresholds agree.
8. E2B decode, Q4_0 matvec per-op cost, plus norm-apply and rope dispatch counts at decode:
   - Q4_0 matvec per op, 1536x12288 67.0 us against llama 38.3 us (decode matmul class 8.71 against
     5.30 ms own-cb, slice 0 decode table row 3); the kernel-level cause is untraced;
   - 446 norm dispatches against llama's 242 (170 `norm apply` stay unfused), and rope at 2 dispatches
     where llama uses 1;
   - read the real serving path (speculation default on, `ServingConfig::default()`), so the owner's chat
     numbers are reproduced.

   Target: E2B decode at or below llama 9.15 ms/token at 971 tokens, and below 11 ms on the short chat
   prompts of `evidence/decode_drift`. Re-attribute on HEAD first. Every change is expressed on the lowered
   program and kernel description, and reports its gain on every test model whose program carries the op.

Targets:
- E2B: prefill at or below llama 570 ms; decode about 9 ms at 970 tokens, and the owner's 10-11 ms
  short-prompt figure re-established.
- granite: prefill at or below 151 ms; decode at or below 5.25 ms/token.

## r8 result (measured 2026-10-07, main 14b8389d to HEAD of this slice)

Slice r8 of the re-plan: E2B decode, the Q4_0 matvec per-op cost, the norm-apply and rope dispatch counts, and the real
serving path. The task text named "r1" and defined "r8"; the spec had no item 8, so the r8 text was added first
(`14b8389d`) and this section answers it. Evidence root: `evidence/r8/` (this directory). Raw per-process logs, the
binaries and every variant kernel are under
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/parity_perf/r8/` (`bin/` holds the executables, `variants/` the
kernel files, `logs/` every build and gate log). Every number below is a measurement with its source; the status of a
statement is the status of its weakest cell.

### what landed

| commit | change | status of its number |
|---|---|---|
| `7c531e84` | `CapturedDispatch::with_pipeline_of`, `describe_buffers`, `replay_output` poisons `output_total` elements only (3 Metal tests, `omega/tests/captured_dispatch_replay.rs`) | measurement tool; no kernel text changes |
| `4d29bc87` | `norm_variant_ab` times a kernel variant, or the omission of a group, over the whole captured step (`AB_STEP_SEQUENCE`, `AB_OMIT_ALL`, `AB_SEQUENCE`, `AB_DESCRIBE`) | measurement tool |
| `10bae12f` | a fused body that mentions one tensor several times loads it once: epilogue operands (`push_epilogue_operand_reads`, `elementwise_reduce_core.rs:484`), the cooperative fold's batched loads (`push_batched_accumulate_loop`, `tiled_gemm_cooperative_scan.rs:2865`), the broadcast prefetch (`prefetch_loaded_operands`, `:2971`); the repeat pattern is part of the pipeline identity (`operand_alias_cache_token`, `kernel_types_identity.rs:2601`, `identity.rs:856`) | E2B decode -0.30 ms short, -0.34 ms long (arm `c1_dedupe` against `base`, "earlier interleaved runs" below); 26B prefill +88 ms (+1.1%), which `c23e9a2c` removed |
| `962d9351` | a matvec simdgroup with a fused epilogue finishes row `q` on lane `q` (`push_packed_row_lane_parallel_tail`, `elementwise_reduce_core.rs:2319`); a matvec with no epilogue keeps its single-lane write | E2B decode -0.32 ms short, -0.29 ms long (arm `c2_lanes` against `c1_dedupe`, same table) |
| `c23e9a2c` | the cooperative fold's batched loads (`push_batched_accumulate_loop`) are back to one load per operand slot; the repeat pattern stays in the pipeline identity for the epilogue and prefetch paths | 26B prefill 8014.0 to 7904.0 ms (arm `c2_lanes` against `c3_nofold`, same table); E2B decode not separately timed |
| `064584d1` | a round-batched reduce is keyed by its epilogue repeats (`omega/src/msl/kernel_types_identity.rs`, `omega/src/identity.rs`, 61 added test lines in `omega/src/msl/tests.rs`) | correctness fix to the identity of `10bae12f`; no timing |
| `35f63a6d` | `norm_variant_ab` times one member of a group or a span window of a step | measurement tool |

Neither perf commit touches `proxima-tensor` or the dispatch path (`git show --stat 10bae12f 962d9351`: `omega/src/msl/*`,
`omega/src/identity.rs`, tests). The Q4_0 matvec body itself (`push_q4_0_native_body`) is unchanged; no model name, trait
or registry was added.

### re-attribution on HEAD before any code

The r8 premise (1536x12288 Q4_0 67.0 us against llama 38.3 us, matmul class 8.71 against 5.30 ms) is slice 0's census at
`f76b4a97`. Re-run on HEAD (`gemma4_decode_kernel_census`, release `std,metal,instrument`, 971-token prompt, box load average
5.06, no Ollama, `evidence/r8/census_head_long/`), through the same `attribution_rank` file:

| class | proxima ops | proxima ms own-cb | llama ms own-cb | gap | slice 0 gap |
|---|---|---|---|---|---|
| matmul (weights) | 277 | 6.39 | 5.30 | +1.09 (1.21x) | +3.41 (1.64x) |
| Q4_0 1536x12288 (60 ops) | 60 | 2.52 = 42.0 us/op | 2.30 = 38.3 us/op | +0.22 (1.10x) | +1.73 (1.75x), 67.0 us/op |
| Q4_0 1536x6144 (45 ops) | 45 | 1.19 | 1.06 | +0.14 (1.13x) | +0.78 (1.73x) |
| Q4_0 1536x2048 (56 ops) | 56 | 1.31 | 0.71 | +0.60 (1.85x) | +0.61 (1.87x) |
| rms norm (446 against 242 dispatches) | 446 | 3.90 | 1.80 | +2.10 (2.17x) | +2.17 (2.21x) |
| rope (100 against 50) | 100 | 0.62 | 0.32 | +0.30 | +0.29 |
| attention core | 73 | 1.48 | 1.21 | +0.27 | +0.35 |

`evidence/r8/census_head_long/rank_vs_llama.md`. Control, same session: the slice-2 era census binary (`census_s2_mml`)
reads 5676.5 us for the 275 Q4_0 dispatches (20.64 us/op cold) against HEAD's 6081.1 us (22.11 us/op), so slice 0's
8.58 ms / 31.2 us/op is not reproduced by either binary
(`evidence/r8/census_head_long/control_s2_binary_stdout.log`). In slice 0's census the ffn_down group (`count=20 grid=12288`)
read 70.6 cold / 113.2 warm / 104.9 us marginal; on HEAD it reads 40.2 / 37.8 / 37.3 (same group, `census_groups.csv`).

What the census measures differs from what a step spends. The 1536x2048 attn_output group
(`omega_reduce_r5_o2_n2_multiply_add_zero`, 28 dispatches) reads 21.4 cold / 33.2 warm / 26.7 us marginal in the census
and 9.3 us marginal, 12.0 us single in a batch-16 replay inside a process that keeps the GPU busy
(`evidence/r8/ab/attn_out_run1.out`, `attn_out_run3.out`); the 28-dispatch sequence with each member on its own weights
is 9.2 us per dispatch with the system cache warm and 26.1 us per dispatch after 384 MiB of CPU writes
(`ab/single_tg.out`, `ab/seq_tg.out` are not kept; `attn_out_run3.out` holds the sequence). The census issues one
single-dispatch command buffer per sample with the CPU waiting in between; GPU clock state during that is not read
(`sudo` for `powermetrics`). Its per-op numbers were therefore not used to rank work. The ranking below is by omission:
the same captured step, replayed in one command buffer in program order, with one kernel group's dispatches removed
(`AB_OMIT_ALL`), so the cost is what the step loses without the group.

### where a decode step goes, by omission (E2B HEAD, step 5, 971-token prompt)

`evidence/r8/ab/omit_all_ranked.txt` (ms the step loses, group size, us per dispatch; replay step 10.3 to 11.4 ms by box
state, small groups carry +-0.1 ms of noise). The largest:

| group | n | ms | us/dispatch |
|---|---|---|---|
| ffn_gate + fused gelu epilogue, K=1536 N=12288 | 20 | 1.460 | 73.0 |
| head (Q6_K, 262144 rows) | 1 | 1.092 | 1092 |
| RMSNorm sumsq + fused epilogue, [1,1536] | 71 | 0.869 | 12.2 |
| ffn_up plain, K=1536 N=12288 | 20 | 0.861 | 43.1 |
| ffn_down, K=12288 N=1536 | 20 | 0.829 | 41.5 |
| sliding cached attention partial | 28 | 0.719 | 25.7 |
| RMSNorm sumsq, [1,1536] | 105 | 0.544 | 5.2 |
| ffn_gate + fused gelu epilogue, N=6144 | 15 | 0.520 | 34.7 |
| ffn_down N=6144 | 15 | 0.459 | 30.6 |
| ffn_up plain N=6144 | 15 | 0.400 | 26.7 |

ffn_gate and ffn_up read the same 10.6 MB of weights. The fused-epilogue form costs 73.0 us against 43.1 us (N=12288) and
34.7 against 26.7 (N=6144): 20 x 29.9 + 15 x 8.0 = 0.72 ms per token that is not weight traffic.

### mechanism

1. Operand slots. `AB_DESCRIBE` prints every buffer a captured dispatch is bound to, before any replay writes
   (`evidence/r8/ab/describe_all.out`). The gate-epilogue kernel (`omega_reduce_r3_o2_n2_multiply_add_zero_epi9_...`,
   N=12288) binds nine epilogue operands: slots 4, 6, 7, 8 and 9 are one buffer (`0x121eb4000`, 49152 bytes, the gate
   vector, offset 0); slots 2, 3, 5 and 10 are four-byte buffers holding 1.0, 0.7978846, 0.044715 and 0.5. The emitted
   epilogue loads every slot from device memory, per row, on lane 0. 19 of the 56 kernel groups of the step bind a buffer
   twice (`describe_all.out`): the sumsq `x * x` binds `x` twice (105 + 71 dispatches), the 15-operand per-layer gate binds
   one vector five times and carries seven scalar constants. `proxima-tensor/src/cpu/epilogue.rs:433` records that
   `bind` does not deduplicate a repeated operand; the duplicate is in the lowered program, not in one kernel.
2. Lane-0 tail. `push_packed_row_combine_and_write` combined the four rows of a simdgroup and then ran, for each row in
   turn, the epilogue operand loads, the arithmetic and the store on lane 0. The multi-token form already writes
   one pair per lane for the same reason (`push_multi_row_lane_epilogue`, `elementwise_reduce_core.rs:1408`: "a fused
   `ffn_up` at 8 tokens spent half its time there"); the single-token form did not.

Variant evidence (replay of the whole captured step, arms interleaved per round, 60 rounds, 3 passes; the state of the
box moves the absolute step by about 1 ms between passes, so the delta is read inside a pass; every variant is a diff of
the production kernel: `evidence/r8/variants_diff/`). Gate group N=12288, ms of step against the production kernel:

| variant | -ms (N=12288) | -ms (N=6144) | output bits against production |
|---|---|---|---|
| duplicate loads replaced by the first (`dedupe`) | 0.12 | 0.04 to 0.07 | 0 of 12288 differ; 10 of 6144 differ, 9 ulp at most |
| `dedupe` + scalars as literals (`immed`) | 0.28 | 0.13 to 0.15 | 0 / 8 differ, 6 ulp at most |
| the four gate loads issued together before the stores (`hoist`) | 0.27 | 0.14 | 0 / 8 |
| `dedupe` + epilogue on lane `q` (`lanepar_loaded`) | 0.33 | 0.17 | 0 / 10, 9 ulp at most |
| `immed` + lane `q` (`lanepar`) | 0.35 | 0.18 | 0 / 8 |
| epilogue operands preloaded before the K loop | worse than `dedupe` (11.31 against 11.25) | | |
| scalar operands in the `constant` address space | `dedupe`-level | | |

(`ab/epi3.out`, `epi4.out`, `epi5.out`, `epi6.out`, `epi7.out`.) `lanepar_loaded` equals `lanepar`, so the scalar
constants need not be literals, which would need the value of a constant node at emit (the `PackedOperands` map is
threaded through 225 sites in `omega/src`); the landed form keeps them loaded.

### interleaved timing of the landed tree (lander run, 2026-10-07)

Command (`evidence/r8/land/`; the binaries and their sha256 are in `land/binaries.sha256`; `base` is the
`cc700bea` build, which has no non-markdown difference from `14b8389d` (`git diff --stat cc700bea 14b8389d -- . ':!*.md'`
printed nothing), `tip` is a release build of HEAD `35f63a6d` whose sha256 equals the earlier `decode_gbps_baseline_tip`,
`tipcopy` is a byte copy of `tip`, the same-binary control). `tip` also contains `e38ce6c3` (the `sha2` dev-dependency
layout in `proxima-model-interop/Cargo.toml`, `asm` kept on non-Windows targets), which `base` lacks. `git diff --stat
14b8389d HEAD -- . ':!*.md'` lists 13 files: 10 from the r8 commits (9 in `omega/`, `norm_variant_ab.rs`) and 3 from
`e38ce6c3` (`ai_docs/index.jsonl`, the interop `Cargo.toml`, a 3-line test import in `serving_grammar.rs`). The commands:

```
decode_arms --prompt-file prompt1k.txt --processes 2 --runs 3 --arm base=<base> --arm tip=<tip> --arm tipcopy=<tipcopy> \
  --llama-server <llama-server f1ea20621> --ignore-ollama --case gemma4_e2b=<E2B blob> --case granite_moe=<granite blob>        # long
decode_arms --prompt-file prompt_short_hippo.txt --new-tokens 64 --processes 2 --runs 3 --arm ... (same arms and cases)         # short
```

Each arm ran 2 processes of 1 warm-up and 3 timed generations, so n = 6 timed per cell (`n_all` in the file), rounds
interleaved. Long prompt: `prompt1k.txt` (970 tokens on gemma4, 1000 on granite), 128 new tokens. Short chat prompt:
`prompt_short_hippo.txt` (26 tokens on gemma4, 36 on granite), 64 new tokens. There is no Ollama arm and no `ollama` CLI call (`--ignore-ollama`; an Ollama server process was already resident on the box, listed by `pgrep` at 14:10 local, and `curl localhost:11434/api/ps` in `long/box_before.txt` returns `{"models":[]}`, no model loaded);
the llama-server arm is the one allowed anchor.
Quiet lock and GPU lock held from `job3.log` `quiet-held 19:26:37Z`, `gpu-held 19:26:42Z` for the long run and until
`quiet-released 19:30:11Z`; the short run ran under the locks of the relaunched job recorded in `job.log` (`quiet-held
19:17:50Z`, `gpu-held 19:17:55Z`, `short-exit 19:22:14Z`, `released 19:22:14Z`). Cells are
median over the 6 timed runs (min to max; CoV of all 6, percent). A CoV above 5 percent would be reported as a range
only; none of the decode cells exceeds 3.8 percent. Token text: every generation of `base`, `tip` and `tipcopy` in a
case printed one `text_hash` (`land/text_hash_counts.txt`: 24 rows, each 4 generations with one hash per model and prompt).

decode ms/token:

| case | base | tip | tipcopy (control) | tip against base | tip against tipcopy | llama-server (same run) |
|---|---|---|---|---|---|---|
| E2B long | 11.981 (11.673 to 12.867; 3.71) | 11.584 (11.150 to 12.036; 3.13) | 11.728 (11.635 to 12.021; 1.16) | -0.397 (-3.3%) | -0.144 | 9.173 (9.082 to 9.527; 1.83) |
| E2B short | 11.709 (11.440 to 11.837; 1.18) | 11.102 (11.017 to 11.232; 0.67) | 11.108 (10.975 to 11.151; 0.64) | -0.607 (-5.2%) | -0.006 | 9.028 (8.961 to 9.295; 1.36) |
| granite long | 14.860 (14.739 to 14.982; 0.52) | 14.484 (14.373 to 14.621; 0.60) | 14.409 (14.353 to 14.755; 0.92) | -0.376 (-2.5%) | +0.075 | 5.254 (5.106 to 5.484; 2.10) |
| granite short | 13.850 (13.564 to 13.928; 0.87) | 13.348 (13.273 to 13.711; 1.11) | 13.406 (13.307 to 13.902; 1.55) | -0.502 (-3.6%) | -0.058 | 5.134 (5.101 to 5.191; 0.61) |

prefill ms (TTFT is the same figure to the millisecond in the file):

| case | base | tip | tipcopy | llama-server |
|---|---|---|---|---|
| E2B long | 1429.5 (1425.0 to 1436.0; 0.27) | 1428.5 (1417.0 to 1436.0; 0.42) | 1426.0 (1423.0 to 1430.0; 0.19) | 569.7 (566.8 to 570.8; 0.26) |
| E2B short | 125.0 (124.0 to 143.0; 6.21) | 124.5 (124.0 to 132.0; 2.27) | 132.0 (125.0 to 139.0; 3.88) | 57.4 (56.8 to 59.2; 1.47) |
| granite long | 897.5 (889.0 to 903.0; 0.57) | 891.0 (831.0 to 902.1; 2.71) | 891.5 (882.0 to 894.0; 0.43) | 151.3 (150.6 to 151.6; 0.22) |
| granite short | 279.5 (278.0 to 288.0; 1.27) | 278.0 (277.0 to 286.0; 1.13) | 277.0 (275.0 to 292.0; 2.09) | 19.1 (19.0 to 19.5; 1.05) |

The E2B short `base` prefill cell has CoV 6.21 percent over six runs (one run at 143.0); it is a range, not a point.

Memory, median over the 2 processes (`peak_rss_bytes`, `peak_footprint_bytes`, `peak_gpu_bytes`; tip against base):

| case | RSS base to tip | footprint base to tip | GPU allocation base to tip |
|---|---|---|---|
| E2B long | 3 888 644 096 to 3 929 088 000 | 719 375 136 to 725 912 576 | 5 608 554 496 to 5 608 554 496 |
| E2B short | 3 621 232 640 to 3 624 525 824 | 212 265 280 to 205 285 568 | 3 386 310 656 to 3 386 187 776 |
| granite long | 2 415 247 360 to 2 444 632 064 | 608 639 264 to 634 960 160 | 3 315 433 472 to 3 315 433 472 |
| granite short | 1 592 614 912 to 1 597 825 024 | 124 556 032 to 125 391 712 | 1 482 113 024 to 1 482 113 024 |

Against the owner's bounds, from the tip rows above (llama figures are the same run's llama-server arm, E2B decode 9.173
against the recorded 9.15 and 9.0178, `evidence/slice0/ac1`):

| bound | tip | gap |
|---|---|---|
| E2B decode at or below llama on the 971-token prompt | 11.584 ms | +2.41 ms, 1.26x llama-server in the same run |
| E2B decode under 11 ms on the short chat prompt | 11.102 ms (range 11.017 to 11.232) | +0.10 ms over the bound; all 6 timed runs are above 11 |
| E2B prefill at or below llama (572 ms recorded; 569.7 ms in this run) | 1428.5 ms | +858.8 ms against this run, 2.51x |
| granite decode at or below 5.25 ms/token | 14.484 ms | +9.23 ms, 2.76x |
| granite prefill at or below 151 ms | 891.0 ms | +740 ms, 5.9x |

None of the five bounds is met. This slice moves E2B decode by 0.40 ms (long) and 0.61 ms (short) and granite decode by
0.38 and 0.50 ms. Prefill base to tip differs by 1.0, 0.5, 6.5 and 1.5 ms (E2B long, E2B short, granite long, granite
short; 0.1 to 0.7 percent); granite long's 6.5 ms is outside the 0.5 ms between `tip` and `tipcopy` and inside the min to
max range of every arm. Memory, per-process values in `long/decode_arms.out` (`memory` lines): RSS base to tip is +40.4 MB
(E2B long) and +29.4 MB (granite long) on the two-process medians, inside the 43 MB and 69 MB between the byte-identical
`tip` and `tipcopy`. Peak footprint is the one that moves the wrong way and does not sit inside that spread on granite
long: base 612.0 and 605.3 MB, tip 652.6 and 617.3 MB, tipcopy 646.5 and 659.6 MB (all four tip-family processes above both
base processes, +12 to +54 MB; +4.3 percent on the medians); on E2B long base is 730.0 and 708.7 MB, tip 729.4 and 722.4,
tipcopy 731.5 and 723.4, within base's own 21 MB process spread. Why granite footprint reads higher at the tip is not traced
(the tip contains `e38ce6c3`'s dev-dependency change as well as the r8 kernels; nothing here separates them); the short
cases show +0.8 MB (granite) and -7.0 MB (E2B). GPU allocation is equal to the byte on the long cases and 122 880 bytes apart
on E2B short.

Box state. Long run: `land/long/box_before.txt` (14:26 local, load averages 6.36 8.23 8.52) lists
`packed_row_multi_row_index32_ab-4cf6cbe2b1987eb6` at 100 percent CPU from `/private/tmp/cargo_target_arch/gate/deps/`; it is
absent from `land/long/box_after.txt` (14:30, load 6.04 6.94 7.88). That binary is an `omega` test from the gate profile;
which process started it, and how long it overlapped the first minutes of the run, was not recorded. Short run: its
`box_before.txt` was lost (the stale process described next rewrote it at 19:22:57Z, while a clippy compile of mine was
running, load 20.60; that file is not kept); `land/short/box_after.txt` (14:22, load 5.69 7.01 7.97) is the only box record
for it, and `job.log` shows the compile-drain loop found no `rustc` or `cargo` process before the job's runs started
(`compile-drain-waits=0` at 19:17:55Z). a background daemon in `~/.local/bin` (named so in the box files, as in earlier evidence) at 52 to 68 percent CPU and WindowServer at 32 to 47 percent in the `ps`
samples that list them.

A first long run (job `job.log`, 19:17:55Z to 19:20:57Z) was lost: the very first launch of the job (`quiet-held
19:11:35Z`) had been sent SIGTERM at 19:17:37Z while it sat in a load-wait loop; its `sh` trap released the locks and the
script then continued, and at 19:22:57Z it started `decode_arms` again without locks, overwriting `long/decode_arms.out`
and the raw per-process logs after I had read the summary lines. The summary medians
read from the terminal before the overwrite were E2B 12.0885, 11.805 and 11.8245 ms/token (base, tip, tipcopy) and granite
14.945, 14.5765 and 14.614; they are not an artifact and are not used above. The stale process was killed with SIGKILL, the
run was repeated as `long2` (copied to `land/long/`), and the short run's files were not touched (`land/short/decode_arms.out`
timestamp 19:22:14Z, 446 lines).

### earlier interleaved runs (development builds, not the landed tree)

Seven interleaved runs from the working session, kept for the per-commit deltas the landed-tree run cannot give. `base` is
`decode_gbps_baseline_base_cc700bea`; `c1_dedupe` (sha256 `6573e46c`, byte-identical to `c1_copy`) is the tree of `10bae12f`;
`c2_lanes` (sha256 `0cadeb16`, byte-identical to `c2_copy`) adds `962d9351`; `c3_nofold` and `c3_copy` are byte-identical to
each other and to `tip` (sha256 `78095f90`). Binary names are the earlier session's; the mapping to commits was not rebuilt
here. Medians over all timed runs (`n_all`), CoV in parentheses; the file is `evidence/r8/live/<run>/stdout.log` (the 26B
short runs: `live7_26b_short`, `live8_26b_short`, `live9_26b_short`).

| run | model, prompt | base | c1_dedupe | c2_lanes | c3 (tip bytes) | same-binary control | prefill ms base / c1 / c2 / c3 |
|---|---|---|---|---|---|---|---|
| live3 | E2B, 26 tokens, n=42 | 11.4145 (0.72) | 11.1155 (4.98) | 10.7935 (2.18) | | base_copy 11.4785 (1.49); c2_copy 10.7925 (1.33) | 124.0 / 124.0 / 124.0 / - |
| live4 | E2B, 970 tokens, n=35 | 12.178 (2.21) | 11.839 (4.13) | 11.547 (3.17) | | base_copy 12.228 (5.14) | 1431.0 / 1427.0 / 1420.0 / - |
| live5 | granite, 1000 tokens, n=20 | 14.8435 (0.49) | 14.7010 (0.60) | 14.3725 (0.83) | | base_copy 14.854 (0.49) | 890.4 / 880.5 / 882.0 / - |
| live6 | 26B, 970 tokens, n=9 | 51.860 (3.08) | | 50.439 (1.70) | | base_copy 51.598 (1.13) | 40132.0 / - / 40596.0 / - |
| live7 | 26B, 192 tokens, n=9 | 47.725 (2.82) | | 47.319 (2.25) | | base_copy 47.921 (2.40); c2_copy 47.381 (2.57) | 7931.0 / - / 8016.0 / - |
| live8 | 26B, 192 tokens, n=9 | 47.667 (2.15) | 47.268 (2.07) | 46.602 (1.72) | | c1_copy 46.527 (2.08) | 7927.0 / 8015.0 / 8015.0 / - |
| live9 | 26B, 192 tokens, n=9 | 48.582 (2.21) | | 47.307 (3.51) | 46.431 (2.34) | c3_copy 45.988 (3.45) | 7931.0 / - / 8014.0 / 7904.0 |

Readings that hurt first.

- The cooperative fold's dedupe (the part of `10bae12f` in `push_batched_accumulate_loop`) raised 26B prefill by 88.0 ms
  (+1.1 percent, CoV 0.05 and 0.18 percent on the two sides) in `live8` (7927.0 to 8015.0), 85.0 in `live7`, 83.0 in `live9`;
  the tip bytes read 7904.0 and 7902.0 (`live9`), 27 ms under `base`. E2B and granite prefill did not move with the same
  binaries (`live3`, `live4`, `live5` prefill columns, and the landed-tree table above). Why 26B pays and E2B does not is
  not traced: no census of the 26B prefill was taken with and without the fold change. `c23e9a2c` takes the dedupe out of
  the fold (`git show c23e9a2c`, 9 changed lines in `tiled_gemm_cooperative_scan.rs`); the epilogue and prefetch dedupe stay.
- The same-binary control moves by as much as the effect in three cells: `live4` `base_copy` CoV 5.14 percent, `live9` the
  two byte-identical c3 binaries differ by 0.443 ms (46.431 against 45.988), `live7` `base` and `base_copy` differ by 0.196.
  26B decode deltas (-0.4 to -2.6 ms against 45 to 50) are inside two to three times that control spread and are
  `plausible`, not `proven`.
- 26B long-prompt prefill moved +464 ms (40132.0 to 40596.0, `live6`, CoV 0.12 and 0.03 percent) with `c2_lanes`; 40 seconds
  for 970 tokens is itself a figure with no attribution in this spec. `c3` was not run on the long 26B prompt.

### norm and rope: variants tried over the captured step, none landed

The norm family is 446 of the 941 dispatches in the captured step (`family=norms` in `census_head_long/stdout.log`), 3.90 ms
own-cb against llama's 1.80 in the re-attribution above. Omitting the `RMSNorm sumsq + fused epilogue` group (71 dispatches,
`b58c6089`) takes the replayed step from 11.37 to 10.64 ms (`ab/norm5.out`, pass 0), so the group costs about 0.73 ms of the
step; every variant below was timed in that replay, arms interleaved, 3 passes, against the `copy` control (the production
kernel recompiled from its own text):

| variant (`evidence/r8/variants_diff/`) | step ms against `copy` (pass 0) | output bits | source |
|---|---|---|---|
| `tg1024` threadgroup width 1024 | +0.112 (11.479 against 11.367) | differs, 1094584677 ulp in the one element checked | `ab/norm5.out` |
| `tg512` | -0.012 (11.355 against 11.367) | 1 ulp | `ab/norm5.out` |
| `tg128` | +0.379 (11.687 against 11.308) | 0 differ | `ab/norm4.out` |
| `tg64` | +0.563 (11.871 against 11.308) | 1 ulp | `ab/norm4.out` |
| one barrier in the reduction (`onebarrier`), 1536-wide group | +0.047 (11.075 against 11.028) | 0 differ | `ab/norm1.out` |
| one barrier, 256-wide group (`8a0df5ee`, 35 dispatches) | -0.010 (11.246 against 11.256) | 0 differ | `ab/norm1.out` |
| epilogue removed (`noepi`; wrong output by construction) | -0.256 (10.765 against 11.021) | differs | `ab/norm2.out` |
| reduction removed (`noreduce`; wrong output by construction) | -0.090 (10.931 against 11.021) | differs | `ab/norm2.out` |
| duplicate operand loads replaced by the first, 71-dispatch group (`dedupe`) | -0.057 (11.275 against 11.333) | 0 differ | `ab/norm3.out` |
| `dedupe`, 105-dispatch sumsq group (`97ab4307`) | -0.090 (11.249 against 11.339) | 0 differ | `ab/norm3.out` |

The omission costs 0.73 ms and no variant with correct output recovers more than 0.09 ms of it (the best is the `dedupe`
of the 105-dispatch group). Threadgroup width, barrier count, the epilogue and the reduction each account for under
0.27 ms, and the two largest deltas belong to the two variants that produce wrong output. The dedupe rows are what
`10bae12f` generalises. Per dispatch the group is 12.2 us (`ab/omit_all_ranked.txt`, 71 dispatches, 0.869 ms) for a
1536-float, 6 KB operand, so the cost is not a bandwidth cost; which fixed per-dispatch cost it is (launch, the barrier
ladder, the epilogue loads) the variants did not isolate, and that is unexplained. The dispatch count itself (446 norm
family dispatches against llama's 242, `rank_vs_llama.md`) was not changed by this slice, and no change to the lowered
program that merges norm dispatches was written or timed. The rope class (100 against 50 dispatches, +0.30 ms in the
census) was not varied: no rope variant was timed.

### reach: which dispatches of each test model carry the changed kernels

Census at base (`census_base`, the `cc700bea` instrument build) and at tip (`census_tip`, release `std,metal,instrument`
of HEAD), same prompt (`prompt_short_hippo.txt`), `M0_MAX_TOKENS=2 M0_CAPTURE_STEPS=1 M0_ITERS=1 M0_BATCH=2`, one process
each, under the GPU lock (`job3.log` `census-*-exit=0`; commands in `job3.log`'s script, groups in
`evidence/r8/land/reach/<model>/groups_{base,tip}.csv`, the `msl_sha256` column). A group counts as reached when the
tuple (class, count, entry, grid threads, `msl_sha256`) exists on the tip side and not on the base side; the same count of
dispatches is unmatched on the base side in all three models (`changed_groups_base_side.txt`). This counts kernels whose
text changed. It does not time them. The captured step is decode step 1, whose plan differs from the steady-state step
(E2B: 1199 dispatches in 140 groups here, 941 in 124 at step 23 in `census_head_long`).

| model | dispatches in the step | dispatches with changed kernel text | groups changed / total | what they are |
|---|---|---|---|---|
| gemma4 E2B | 1199 | 71 | 38 / 140 | Q4_0 matvec with the gelu-gate epilogue (20 + 15), 35 per-layer norm-plus-epilogue groups (`epi15`), the head (1) |
| granite moe | 935 | 241 | 4 / 38 | Q8_0 matvec with a fused epilogue (24 + 24 + 192 + 1) |
| gemma4 26B | 2599 | 271 | 3 / 75 | Q3K matvec with a fused epilogue (30 + 240), the head (1) |

Decode ms/token moved on E2B (-0.40 long, -0.61 short) and granite (-0.38 long, -0.50 short) in the landed-tree table; 26B
was timed only in the development runs above (`live6` to `live9`, within the control's spread for decode). The sum of the
replayed `lanepar_loaded` deltas for the two E2B gate groups (0.33 + 0.17 = 0.50 ms, variant table above) sits inside the
range of the two measured E2B deltas; the 35 norm-epilogue groups and the head have no replay delta of their own, so the
split of the measured -0.40 to -0.61 between them is not attributed.

What the reach does not include, from the same census: the `RMSNorm sumsq` groups (`97ab4307`, 105 dispatches; `b58c6089`,
71 dispatches; and the others in `groups_tip.csv`) carry the same `msl_sha256` at base and tip. They are the groups where
the `dedupe` variant timed -0.057 and -0.090 ms (norm table above); `c23e9a2c` took that dedupe out of the cooperative
fold, so those two gains are not in the tip. Their sum is 0.147 ms of an 11.58 ms step if the two are additive, which was not measured; both are replay figures on E2B only.
Whether the fold dedupe can be restored for a decode-only shape without the 26B prefill cost (+85 ms, `live7` to `live9`)
was not tried.

### three checks

1. Correctness (tests, run under the GPU lock for the Metal ones; logs in `evidence/r8/land/gate/`).
   - `cargo nextest run -p omega --features metal --cargo-profile gate`: 613 tests run, 613 passed, 16 skipped
     (`nextest_omega.head_tail.txt`; 613 PASS lines, 0 FAIL lines in the full log). The last count recorded in this spec is 599 (slice 2 fix); this slice adds 14, which makes 613:
     5 integration cases in `omega/tests/packed_row_single_token_epilogue.rs` (rows 6, 1536, 3, 5 and 7 over Q4_0 and Q6_K
     lane masking) and 9 unit cases in `omega/src/msl/tests.rs`. The 16 skipped are `#[ignore]`d probes and golden recorders
     already present before this slice: `record_main_goldens`, `record_main_softmax_weights_goldens`,
     `device_streaming_ceiling_across_three_sources_and_two_sizes`, `gpu_load_generator`,
     `real_per_layer_model_proj_weight_is_fully_populated_at_the_overflow_row_counts`,
     `ladder_eight_dispatches_vs_one_merged_dispatch_gpu_time`, eight `matvec_roofline_ladder` probes,
     `upload_path_totals_report_after_the_full_parity_suite` and
     `metal_matmul_on_real_ffn_up_q3k_bytes_matches_the_dequantized_f32_cpu_path` (a real-checkpoint test, skipped by
     `#[ignore]`; it is not run by this gate and was not run here).
   - `cargo nextest run -p omega --features metal,instrument --cargo-profile gate -E 'binary(captured_dispatch_replay)'`:
     3 run, 3 passed (`nextest_replay.log`); the default gate does not build those three because they need `instrument`.
   - `cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate`: 710 run,
     710 passed, 124 skipped (`nextest_interop_slice_gate.head_tail.txt`; 710 PASS, 0 FAIL).
   - `cargo clippy -p proxima-tensor -p proxima-model-interop -p omega --features
     proxima-model-interop/std,proxima-model-interop/metal,omega/metal --all-targets -- -D warnings`: exit 0
     (`clippy.log`); with `omega/instrument` and `proxima-model-interop/instrument` added for omega and interop: exit 0
     (`clippy_instr.log`).
   - Not run: the alloc-tier and `--no-default-features` builds of `proxima-tensor`, which `git diff --stat 14b8389d HEAD`
     shows untouched (the 13 files are in `omega/`, `proxima-model-interop/`, `ai_docs/`); `cargo test --doc`.
2. Semantic (the incumbent's token ids; `evidence/r8/land/gate/nextest_parity.log`).
   `cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate -E 'test(llama_parity_) or
   test(generic_verify_llama_parity_) or test(prefill_width_parity_with_llama)'`: 14 run, 14 passed, 820 skipped, 3 slow
   (340.6 s). The 14 are 7 `llama_parity_` (gemma4_26b, gemma4_e2b, granite_moe, lfm2, openchat, qwen2, qwen3), 5
   `generic_verify_llama_parity_` (gemma4_e2b, granite_moe, openchat, qwen2, qwen3) and 2
   `prefill_width_parity_with_llama_` (gemma4_e2b, granite_moe). `lfm2` is excluded from the slice-gate profile and ran here.
   The decode_arms run printed one `text_hash` per case across `base`, `tip` and `tipcopy`
   (`text_hash_counts.txt`: 24 files, 4 generations each, 96 generations; E2B long `8fec363180a250e0`, E2B short
   `1ab772cf44dbbf7b`, granite long `c4625c1fb93f28b7`, granite short `8cf1c359770788c7`). That compares base to tip
   text, not ids to the incumbent; the ids-to-incumbent comparison is the `llama_parity_` line.
3. Performance: the landed-tree tables above (decode, prefill, TTFT, peak RSS, footprint, GPU allocation; E2B and granite,
   long and short; 2 processes x 3 timed runs, same-binary control). Not measured here: 26B at the tip (the development runs
   `live6` to `live9` stand for it), and p99 per-token latency (the driver prints medians and ranges).

### what the three checks do not establish

- The five bounds of the goal are all unmet at the tip (bounds table above). The slice moved E2B decode by 0.40 and
  0.61 ms and granite decode by 0.38 and 0.50 ms; the remaining gap to llama-server in the same run is 2.41 ms (E2B long),
  2.07 ms (E2B short, against 9.028), 9.23 ms (granite long) and 8.21 ms (granite short, against 5.134).
- E2B decode deltas are 3 to 5 percent of the step and the long-prompt cells have CoV 3.1 and 3.7 percent over six runs;
  the `tip` to `tipcopy` difference on E2B long (0.144 ms) is more than a third of the base to tip difference (0.397 ms).
  The short-prompt delta (0.607 ms, control 0.006 ms) is the clearer of the two.
- The omission ranking ("where a decode step goes") was taken on the pre-slice tree. The ranking at the tip was not
  re-taken; the next slice starts from that table and from the fact that the norm family (0.73 ms for the 71-dispatch group
  alone), the head (1.09 ms) and the Q4_0 `ffn_*` groups (0.83 to 1.46 ms each) were the largest entries before it.
- Granite prefill (891 ms against 151) and granite short-prompt prefill (278 ms for 36 tokens against llama's
  19.1 ms) are not touched by this slice and have no attribution in this section.
- Unexplained: why the cooperative fold dedupe costs 26B prefill 85 to 88 ms and E2B and granite prefill nothing; what the
  fixed per-dispatch cost of the 12.2 us norm group is; what overlapped the first minutes of the long run (the
  `packed_row_multi_row_index32_ab` process in `box_before.txt`).

### re-prove

```
git diff --stat 14b8389d HEAD -- . ':!*.md'                                   # 13 files, none in proxima-tensor
shasum -a 256 -c evidence/r8/land/binaries.sha256                             # from .long_ctx_backups/parity_perf/r8/land (bin/, prompts)
decode_arms --prompt-file prompt1k.txt --processes 2 --runs 3 --arm base=<land/bin/base> --arm tip=<land/bin/tip> \
  --arm tipcopy=<land/bin/tipcopy> --llama-server <llama-server f1ea20621> --ignore-ollama \
  --case gemma4_e2b=<E2B blob> --case granite_moe=<granite blob>               # land/long
decode_arms --prompt-file prompt_short_hippo.txt --new-tokens 64 ... (same arms)   # land/short
cargo nextest run -p omega --features metal --cargo-profile gate                                # 613 passed, 16 skipped
cargo nextest run -p omega --features metal,instrument --cargo-profile gate -E 'binary(captured_dispatch_replay)'   # 3
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate           # 710
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate \
  -E 'test(llama_parity_) or test(generic_verify_llama_parity_) or test(prefill_width_parity_with_llama)'          # 14
M0_OUT_DIR=<dir> M0_MODEL_GGUF=<blob> M0_MAX_TOKENS=2 M0_CAPTURE_STEPS=1 M0_ITERS=1 M0_BATCH=2 \
  PROXIMA_PROMPT_FILE=prompt_short_hippo.txt gemma4_decode_kernel_census                                         # reach; compare msl_sha256
```

## r9 written changes (written 2026-10-07, nothing below was run)

Slice r9: the E2B Q4_0 attention-output matvec and the 34 untraced norm-family dispatches. Owner's order: write, then bench
and test. Verification so far is `cargo check` and `cargo clippy -D warnings` only. Every number is a reading of the
artifact named beside it.

### 1536x2048 Q4_0 matvec (read)

- The class row `Q4_0 3145728 (1536x2048)`, 56 ops, 1.31 ms against llama 0.71 ms (`rank_vs_llama.md`) holds two
  different matvecs of the same 3145728 elements: 28 Q projections (K=1536, rows 2048, entry `omega_reduce_r4_o3_...`,
  census marginal 12.4 us) and 28 attention outputs (K=2048 as heads 8 x head_dim 256, rows 1536, entry
  `omega_reduce_r5_o2_...`, census marginal 26.7 us); `census_groups.csv`. llama times the two at 12.34 and 12.94 us own-cb
  (`slice0/llama_ops/e2b_ops.tsv`, ntok=1, `Qcur` and `node_32`...).
- Geometry is already llama's: 4 rows per simdgroup (`PACKED_ROWS_PER_GROUP`, `emit_and_classify.rs:2288`), 2 simdgroups per
  threadgroup (`packed_row_nsg_factor`, `tiled_gemm_cooperative_scan.rs:1876`), 64 threads, 192 threadgroups for 1536 rows;
  llama `N_R0_Q4_0 4`, `N_SG_Q4_0 2` (`ggml-metal-impl.h:32-33`), dispatch `(ne01 + nr0*nsg - 1)/(nr0*nsg)`
  (`ggml-metal-ops.cpp:2896`). llama's tuning table (`ggml-metal-tuning.cpp`) is flash-attention only; no matvec geometry is
  shape-keyed there, so there is no shape entry to copy and a proxima geometry table would reproduce the constants.
- The hot loop is the same text in both proxima kernels (`diff` of `variants/6ccd88ae...copy.metal` and
  `variants/2c9c96c7...copy.metal`: coordinate prologue and the store tail only) and the same arithmetic as llama's
  `mul_vec_q_n_f32_impl` (`mul_mv.metal:219`): `ix = lane/2`, `il = (lane%2)*8`, 16 blocks in flight, `sumy * -8`.
- The difference is the write tail. Before: four serial lane-0 stores, each addressed by a sum over all 5 iteration axes
  (`coord_q_cache[q][0..4] * u.out_strides[0..4]`, 64-bit multiplies); the Q kernel pays 4 terms, the attention output 5,
  the census marginal reads 12.4 and 26.7 us; the fused-epilogue tail was made lane-parallel in 962d9351 and the plain tail
  was left as it was. llama's tail is `dst[r0 + row] = tot`.
- Written: the plain tail is lane-parallel too (lane `q` stores row `q`) and, when `packed_row_direct_output_axis`
  finds one non-unit output axis, addresses the store as `out_base + flat * out_strides[axis]`
  (`push_packed_row_lane_parallel_tail`, `elementwise_reduce_core.rs`). Values are the same `simd_sum` results; only the
  lane that stores them and the address arithmetic change. What this does to the 26.7 us is unmeasured.
- Not established: whether the tail is what separates 12.4 from 26.7 us. A single-dispatch census cb and a batch replay of the
  same group read 26.7 and 9.3 us (SPEC r8 section), so the census figure also carries GPU clock state.

### the 34 norm-family dispatches (read)

- Norm family after r6's fold: 170 `RMSNorm sumsq` + 106 `RMSNorm sumsq + fused epilogue` = 276 against llama's 242.
- llama (ntok=1, count/381 graphs): hidden-width 106 + 70 = 176, Q-norm 28 + 7 = 35, K-norm 12 + 3 = 15, V-norm 12 + 3 = 15,
  per-layer-input 1 (`[256,35]`, `fuse=3`) = 242.
- proxima (`census_groups.csv`): hidden-width 105 + 71 = 176, Q-norm 28 + 7 = 35, K and V 24 + 6 = 30, per-layer-input 35
  (`RMSNorm sumsq + fused epilogue`, `epi15`, 1 each, grid 64) = 276. Every class matches except the last: 35 against 1 = 34.
- Cause: `ple_layer_input` (`proxima-tensor/src/spec/attention_forward.rs:1695`) takes a `d + layer*256` window of the flat
  `[s, 35*256]` projection and builds `rmsnorm` on it once per layer. The norm is not left unfused by a rule; the program
  contains 35 of them.
- Fusion route, not written: one RMSNorm over `[s, 35, 256]` (`rmsnorm_per_head` with a layer axis) needs the layer axis to be
  sized by an operand. `per_layer_model_proj.weight` is declared `[embedding, ple_total]` (`attention_forward.rs:1634`), so
  the grammar cannot size a split axis (`map.rs:107`, `len_target_axis`); declaring the weight `[embedding, 35, 256]`
  changes the leaf shape that `declared_leaves_match_bound_leaves_tests` and the GGUF binder compare. The per-layer gate
  chain (norm, add, scale, gelu gate, `epi15`) would then be an elementwise consumer of the shared norm: norm reduces 35 ->
  1, elementwise +35, so total dispatches stay level and the saving is the reduce launches only.

### tests written (not run)

- `omega/tests/packed_row_attn_output_shape_parity.rs`: Q4_0, K = 8 x 256 folded, rows 1536 / 6 / 4, f64 dequantize-and-dot
  oracle, tolerance 1e-5 relative.
- `proxima-model-interop/tests/gemma4_norm_family_census.rs` (`#[ignore]`, needs the E2B gguf): 35 reduces over `[1, 256]`
  in the bound decode program, 34 beyond llama's 1.
- `omega/src/msl/tests.rs`: `a_plain_matvec_finishes_each_row_of_the_simdgroup_on_its_own_lane` replaces the single-lane pin;
  `omega/tests/fixtures/packed_row_blocked_s1_q4k.msl` carries the new tail (written by hand, to be compared by
  `packed_row_blocked_s1_byte_identity`).

### run after

```
cargo nextest run -p omega --features metal -E 'binary(packed_row_attn_output_shape_parity) or binary(packed_row_blocked_s1_byte_identity) or binary(packed_row_single_token_epilogue)'
cargo nextest run -p omega --features metal -E 'test(a_plain_matvec_finishes_each_row) or test(a_fused_epilogue_finishes_each_row)'
cargo nextest run -p proxima-model-interop --features std,metal --run-ignored all -E 'test(gemma4_decode_holds_one_per_layer_input_norm)'
```

## r9 follow-up: the per-layer-input norm as one reduce (written 2026-10-07, nothing below was run)

Supersedes the "Fusion route, not written" paragraph and the `#[ignore]`d census test described above.

### what changed

- Grammar: a multi-term operand axis `tile*outer + inner@tile` now sizes `outer` as the operand extent divided by `tile`
  (`AxisIndex::split_outer_axis`, `proxima-tensor/src/map.rs`; `split_outer_extent`, `shape.rs`). It is a fallback
  applied after every other operand has had its say, so a program that already resolved `outer` from an anchor resolves
  the same value; only programs that previously failed with `UnconstrainedDim` are newly accepted. A tile that is not
  the stride, or an extent that is not a whole number of tiles, still fails `UnconstrainedDim`. The symbol-mask twin
  (`fold_split_into_iteration_masks`) follows the same rule.
- Notation: an axis of constants alone (`"s,3,d->sd"`) is a fixed index with no iteration term
  (`parse_axis_expr`, `spec/primitives.rs`).
- Program: `append_ple_shared_projections` views the flat `[s, 35*256]` projection and embedding gather as
  `[s, 35, 256]` through `"s,256*l+d@256->sld"`, applies `rmsnorm_per_head(.., "l")` (one `sumsq` reduce over
  `[s, 35, 256]`), adds, scales, and `ple_layer_input` reads layer `n` as `"s,n,d->sd"`. The 2D
  `per_layer_model_proj.weight` leaf, the GGUF binder and `declared_leaves_match_bound_leaves_tests` are untouched; leaf
  declaration order is unchanged (`eps` was already declared before the preamble in both callers).
- Everything downstream of `ple_layer_input` (gate matmul, GeLU, `gated`, `proj`, post norm) still consumes a `[s, 256]` node.

### digests expected to change (derived, to be confirmed by the recapture)

Only `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b.digest` hashes a program that contains the preamble
(`describe_program` hashes the raw `Op` list, `tests/arch_data_baseline.rs:166`):

- `bind.ops` 6074 -> 5632 and `verify.ops` 6072 -> 5630: old form 14 ops per layer x 35 = 490, new form 13 shared + 35
  slices = 48, delta -442 (derived from reading the two builders, not measured).
- `bind.ops_sha256`, `verify.ops_sha256`: every op after the preamble moves and the preamble ops differ.
- `bind.logits_root`, `verify.logits_root`: NodeId = op index, shifted by the op-count change.
- `bind.layer_roots` / `verify.layer_roots` sha256: NodeIds of every layer root shift (the 35 per-layer preamble blocks
  used to interleave with the layers and now all precede layer 0).

Unchanged by construction: `residual_roots`, `router_roots`, `hidden_root`, `single_position_step`, every `.bound` file
(same leaves), and the other seven `.digest` files (`gemma4_26b` declares `embedding_length_per_layer_input = 0`,
`fixtures/llama-parity/gemma4_26b/gguf_kv.txt:37`; the rest are non-gemma4 architectures).

### tests written (not run)

- `proxima-tensor/src/spec/attention_forward.rs` `ple_single_reduce_parity_tests`: 5 layers, ple_dim 8, embedding 12,
  3 tokens, CPU evaluator, one-reduce form against the per-layer windowed form over the same flat nodes, 1e-6; a control
  that layer 0 of one form differs from layer 1 of the other by more than 1e-3; a count of `Keep::Reduce` ops
  (matmul 1 + shared norm 1 + windowed 5).
- `proxima-tensor/src/shape.rs`: tiled split sizes its outer axis; partial tile and tile != stride stay `UnconstrainedDim`.
- `proxima-tensor/src/spec/tests.rs`: a constant axis parses with no terms; a constant with `@len` is malformed.
- `proxima-model-interop/tests/gemma4_norm_family_census.rs`: not ignored, fails when the checkpoint is missing,
  asserts 1 reduce over `[1, 35, 256]` and 0 over `[1, 256]`.

### run after

```
cargo nextest run -p proxima-tensor -E 'test(ple_) or test(tiled_split) or test(split_over) or test(split_whose) or test(constant_axis)'
cargo nextest run -p proxima-model-interop --features std,metal -E 'test(gemma4_decode_holds_one_per_layer_input_norm_reduce)'
# then recapture the gemma4_e2b digest with the arch_data_baseline capture path and diff it against the list above
```

## r6 written changes (written 2026-10-07, nothing below was run)

Slice r6 of the re-plan, item 6: norm-apply and rope dispatches. The owner's order for this pass was to write all of r2 to
r6 first and bench and test afterwards, so this section holds what was read and what was written. Every number is a
reading of an artifact named next to it; no cell here is a measurement taken for this slice. Verification so far is
`cargo check` and `cargo clippy` only.

### the declining condition (read)

- The 170 unfused applies are the `norm apply` rows of `evidence/r8/census_head_long/census_groups.csv`: 12 + 12 + 28 + 70
  + 35 + 3 + 3 + 7 = 170. The class histogram of `census_dispatches.csv` (column 5) reads 170 `RMSNorm sumsq`, 170
  `norm apply`, 106 `RMSNorm sumsq + fused epilogue`, 100 `RoPE`, 36 `identity copy`: 170 + 170 + 106 = 446, the norm
  family of the r8 table. In the dispatch order each plain `RMSNorm sumsq` is followed by its `norm apply` (rows 89-90,
  94 and 96, 114-115, 120-121 of the same file).
- The pass: `find_epilogue_source` returns the first reduce-fold operand of the consumer
  (`proxima-tensor/src/bind/cached_attention_epilogue_liveness.rs:317`). An apply lists the projection output `x` before the
  sum-of-squares reduce. `x` is read by the reduce and by the apply, and the candidate gate drops any source with other
  readers (`:414`, `reference_counts != 1`), so the pair is never formed and the reduce behind `x` is not considered.
  `find_epilogue_sources` (`:362`) considers every flagged operand; `bind_with_fusion` selects it only when the policy
  grants `WidenedReduceEpilogueFusion`, that is `epilogue_sources` (`bind/gdn_moe_fusion_apply.rs:159-161`).
- The serving default had `epilogue_sources: false` (`serving_settings.rs` `default_str`, `ServingConfig::default`). The
  bind test `projection_output_rmsnorm_tail_fuses_only_with_the_epilogue_sources_switch`
  (`bind/tests.rs`, written before this slice) asserts zero real epilogues with the switch off and at least one with it on
  for a projection-fed RMSNorm. The ignored census test `gemma4_epilogue_sources_census` names the E2B total: 35 x (3
  hidden norms + Q norm + attention combine) + 15 x (K norm, V norm) = 205 ops absorbed.
- The kernel the fold reaches is one dispatch per row: a cooperative fold, the scalar published through shared memory,
  every lane writing its share of the row, with the operand loads issued before the fold
  (`omega/src/msl/tiled_gemm_cooperative_scan.rs:2576` `push_cooperative_reduce_tail`, `:2729`
  `push_broadcast_epilogue_write`, `:3048` `push_broadcast_epilogue_preload`). The gamma multiply and the residual add of
  the hidden-width post-norms are part of the epilogue body (`omega/tests/rmsnorm_epilogue_bit_identity.rs`, shape
  `hidden_residual`).

### what was written

| commit | change |
|---|---|
| `feat(interop): grant epilogue_sources in the serving default` | `ServingConfig::default`, the `ServingSettings` builder default and its `default_str` now grant `epilogue_sources`; tests: `default_numeric_policy_is_llama_relaxed_with_epilogue_sources`, `default_numeric_policy_admits_the_widened_reduce_epilogue_fusion`, the three `*_agrees_across_literal_and_default_override` literals, and the settings default assertion |
| `test(tensor): pin the gamma and residual tail fold under epilogue_sources` | `projection_output_rmsnorm_gamma_residual_tail_folds_into_one_broadcast_reduce_with_the_switch`: switch off leaves no broadcast reduce; switch on leaves exactly one whose epilogue reads `gamma` and `residual`, drops ops, and matches the unfused CPU result within 1e-6 relative |

The switch stays a field of `NumericPolicy`, so `PROXIMA_EPILOGUE_SOURCES=0` in `decode_gbps_baseline` and
`gemma4_decode_kernel_census` is the off arm of the A/B (`examples/decode_gbps_baseline.rs:218-222`).

### what the arithmetic says about the target (derived, not measured)

Removing 170 applies from the 446-dispatch norm family leaves 276, not llama's 242 (`rank_vs_llama.md`). The 34 beyond
242 are not traced. The earlier decode measurement recorded in `NumericPolicy::epilogue_sources` (15.72 ms off, 15.93 ms
on) was taken before the operand preload (`c3d924e3`, 2026-10-04 08:55 -0500, against the doc commit `810bc46b` at 01:50
-0500), whose own doc reports 8.6 us to 4.6 us on a `[1, 1536]` row; whether the fold now pays is what the bench step
decides. The same doc records logits differing in about 85% of their bits with row-norm-relative error at most 6.7e-8 and
the same argmax at every step on E2B; no other model was measured.

### rope: one dispatch per rotated tensor (written, nothing below was run)

The single-op form over a parity axis stays blocked for the three reasons read in the first pass of this section
(`unify_iteration_space` cannot size a split-half parity axis, a sign operand costs shared iota dispatches, and the
zero-copy window alias is a default-off switch of its own). The twin-output elementwise the first pass proposed is
written instead, as a bind-level fusion that leaves the graph at two ordinary nodes:

- `BoundOpKind::ElementwiseTwin { body, operands, twin_node, twin_body }` (`bind/types_layout_boundop.rs`): one shared
  operand list, two bodies, two output nodes. A separate variant rather than an `Option` on `Elementwise`, the convention
  `RoundBatchedReduce` documents, so every exhaustive backend match decides for itself.
- `fuse_twin_elementwise(built, program)` (`bind/twin_elementwise.rs`): groups `Elementwise` ops by iteration space plus
  the multiset of (source, layout) reads and merges each pair, remapping the second body's operand indices onto the first
  op's list. It keys on data (extents, layouts), names no model and no RoPE: `fused_rope_pair` emits `x_same*cos -
  x_partner*sin` and `x_partner*cos + x_same*sin` over the same four reads, for split-half (`i@pairs`, `i+pairs`) and
  adjacent (`2*i`, `2*i+1`) alike. It runs after `prune_dead` and `promote_output_placed_nodes`, in
  `prepare_uniforms_pack.rs`, so both siblings are live, and `bind_with_fusion` never runs it: the CPU, wgpu and cuda
  paths keep two plain ops (they reject the kind by name if handed one). `BoundOp::twin_halves` is the inverse.
- Metal: `render_elementwise` emits the second store (`extra_out0[gid]`) from `kernel_signature_with_extra_outputs`,
  declared at buffer index `Kernel::bindings.len()`, where `encode_op` already binds an op's extra outputs. The arena side
  is the r4 commit "bind a multi-output op's extra outputs from the plan arena": `extra_output_nodes` gives the twin node a
  plan-owned slot. A placed twin node (the K cache roots, `LayerCacheRoots`) is seeded into `device_buffers` from
  `output_placed` ahead of the arena's slot (`placements_execute_named.rs`), so the kernel writes the caller's KV buffer
  at the caller's offset, exactly as it does for the primary node.
- Switch: cargo feature `twin-elementwise-fusion` (in omega's `metal` list, so interop gets it), and the env var
  `PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION` is the off arm of the A/B, the shape the other `PROXIMA_DISABLE_*` switches take.

Expected dispatch change, derived from the layer schedule and not measured: 35 Q rotations + 15 own-KV K rotations = 50
tensors, 100 RoPE dispatches before, 50 after (`gemma4_rope_twin_census`).

Unmeasured and unverified by anything in this pass: that Metal's compiler contracts `a*b - c*d` identically in the
two-store kernel and in the two single-store kernels (the Metal bit-identity test below is the check); what 50 fewer
dispatches do to decode ms; any other sibling pair the pass merges beyond the 50 (the census asserts exactly 50 twins and
prints the kind histogram).

### bench and test to run after r2 to r6 are written

```
cargo nextest run -p proxima-tensor --features reduce-epilogue-fusion,cached-attention-streaming -E 'test(projection_output_rmsnorm)'   # 2
cargo nextest run -p proxima-model-interop --features std,metal -E 'test(default_numeric_policy)'                                      # 2, both new
cargo nextest run -p omega --features metal -E 'binary(rmsnorm_epilogue_bit_identity)'                                                  # 2
cargo nextest run -p proxima-model-interop --features std,metal -E 'test(llama_parity_) or test(generic_verify_llama_parity_)'          # 12
cargo nextest run -p proxima-model-interop --features std,metal -E 'test(epilogue_sources_drift)' --run-ignored all                      # needs the E2B gguf
PROXIMA_EPILOGUE_SOURCES=0|1 decode_gbps_baseline / gemma4_decode_kernel_census   # off and on arms; the census reads the norm family 446 -> ?
cargo nextest run -p proxima-tensor -E 'test(twin_elementwise_tests)'                                                                   # 8: CPU parity per pairing, swapped-body control, no-merge cases
cargo nextest run -p omega --features metal -E 'test(twin_elementwise_tests)'                                                           # 6: kernel source, extra-output buffer index
cargo nextest run -p omega --features metal -E 'binary(elementwise_twin_dispatch)'                                                      # 9: Metal vs CPU, twin vs two-dispatch bits, placed KV outputs
cargo nextest run -p proxima-model-interop --features std,metal -E 'test(gemma4_rope_twin_census)' --run-ignored all                   # needs the E2B gguf: 50 twins, 100 -> 50
PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1|unset decode_gbps_baseline / gemma4_decode_kernel_census   # off and on arms for the rope fold
```

## combined r1-r9 result (measured 2026-10-07, origin/main ab69ec03 to HEAD of this section)

Order executed: apply the r9, r3, r2, r4, r6, r5, r7 patch sets onto the r1-applied main (`7d693c1a`), remove model names from library
source, run every gate once at the integrated tip, bench once, push. Evidence root: `evidence/combine/` (this directory); raw per-process
logs, the base source export and the binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/combine/`. Every number below
names its artifact; the status of a sentence is the status of its weakest cell. Test models were gemma4 E2B and granite moe 1b only (the order);
no result here covers any other checkpoint.

### what landed (`git log ab69ec03..HEAD`, 67 commits before the commit that adds this section)

| slice | commits at the tip | note |
|---|---|---|
| r1 (on main, unpushed) | `2d7a8e1a e0423838 e80df86b 6c99b40b a035824d 7d693c1a 16f4851c` | row-tiled and windowed prefill attention |
| r9 | `210afc05 b824a691 fc3d8e05 38dbf823 bfa949c4 a698dfa7 63f4bca6 f1bb9b08 dbd8b49c` | Q4_0 matvec tail on its own lanes; one per-layer-input norm over the layer axis |
| r3 | `b06bdf10 966e0f70 0712bc78 559f6c38 84cfba10` | every described codec on the tiled GEMM; disable switch |
| r2 | `f582a26b a0c75ba4 b05a0393 8645f749 ffc69cff 102343f8 fefbd8d8 e6cded5e bfb62f6f d8ede9d5 e365e191 60c5e0f6`; `9aa0833c` reverted by `b18a6537` | stacked routed experts as an opt-in strategy |
| r4 | `3af7e47c fdca26ce c01209d5 ba01bb44 3026cd8a 2d04a25d eb83c913 401f2003` | gather fault buffer pool, arena extras, 8 command buffers for granite |
| r6 | `96a64c5f 0b3aa7a0 76593bda d5b83512 db88f4e2 40c78ce8`; r6 0004 skipped (same hunks as r4 0004) | epilogue_sources grant; twin-output elementwise |
| r5 | `55940cd2 3127bc55` | pipeline archive cache on disk; resident prefill plans |
| r7 | `f2f273d8 fa405b6a f61d6fb1 a79513bd 5ae7d822` | arena allowance per prefill row; default ubatch 512 |
| this pass | `96d24dc4` (model names out of library source), `0780fb6c 5cf74898 b8b036b1 df9e38a1 6868398c b18a6537 8eca5477 a2edd376 04c67331 85f1f23e 1989bdc1` | fixes, one revert, the e2b digest, an example hook; each is described in `evidence/combine/conflicts.md` |

Patch application: 10 conflict sites in 7 patches resolved keeping both intents; 1 patch (r6 0004) skipped as already applied (the same source hunks as r4 0004); 2 patches
(r4 0004, r6 0005) needed their blobs fetched from the writer's tree to merge (`evidence/combine/conflicts.md`). While staging one resolution, `git add omega proxima-tensor` swept two
untracked trees owned by others into one commit; the 16 commits from it upward were rebuilt with `git commit-tree` without those files, and the two trees are byte-identical to their backup (`diff -r` clean).

### before and after, per model (`decode_arms`, release std+metal, 3 processes x (1 warmup + 7 runs), arms interleaved, `evidence/combine/bench/`)

Arms: `base` = the tree at ab69ec03 built release; `tip` = the final tree (`decode_gbps_baseline`); `control` = a byte copy of the tip binary (same sha256, `bench/binaries.sha256`).
The example forces `batch_size: 0, ubatch_size: 0`, so r7's default ubatch of 512 is not in these timings (its effect is covered by its parity tests only). llama-server and Ollama were not re-run;
the recording (`evidence/slice0/ac1`) reads E2B prefill 572 ms and decode 9.15 ms/token, granite prefill 151 ms and decode 5.25 ms/token, from a different session. Box: Ollama down, no other cargo job,
`mds_stores` 70-106% CPU and a background daemon in ~/.local/bin at 50-70% CPU during every run (`box_load_before.txt`), one-minute load average 4.3 to 12.8 across the runs. Medians over all 21 runs per arm; CoV over all 21; where CoV
exceeds 5% the range and the outlier-removed median (kept) are given.

Three complete matrices exist; they differ in the tip binary and the box state. Run A/B: tip before the arena-extras commit (`bench/runA`, `runB`). Run A2/B2: final tip, started at load average 12.8,
CoV above 5% in 5 cells, all in B2 (`runA2`, `runB2`). Run A3/B3: final tip, settled box (the tables below).

E2B, 971-token prompt (`runA3`, `gemma4_e2b.*`):

| metric | base | tip | control | tip vs base |
|---|---|---|---|---|
| prefill ms (= ttft) | 1411.0 (CoV 0.45%, 1400.0-1427.0) | 594.0 (0.41%, 591.0-604.0) | 595.0 (1.09%, 592.0-619.0) | -817.0 ms, -57.9% |
| decode ms/token | 11.875 (CoV 6.67%, 11.482-15.175; kept 11.763, n=19) | 11.063 (1.24%, 10.958-11.683) | 11.089 (4.98%, 10.999-13.389; kept 11.077) | -0.812 (-6.8%); kept -0.700 (-6.0%) |

E2B, short chat prompt (`prompt_short_hippo.txt`, 25 tokens; `runB3`):

| metric | base | tip | control | tip vs base |
|---|---|---|---|---|
| prefill ms (= ttft) | 124.0 (CoV 8.26%, 123.0-165.1; kept 124.0, n=13) | 88.0 (0.85%, 87.0-91.0) | 88.0 (8.68%, 87.0-113.0) | -36.0 ms, -29.0% |
| decode ms/token | 11.139 (2.97%, 10.695-12.510) | 10.874 (1.19%, 10.455-10.945) | 10.896 (6.01%, 10.540-13.487; kept 10.879) | -0.265 (-2.4%) |

granite moe 1b, the same prompt file as the 971-token E2B run (`runA3`, `granite_moe.*`):

| metric | base | tip | control | tip vs base |
|---|---|---|---|---|
| prefill ms (= ttft) | 866.0 (2.83%, 828.0-925.0) | 374.0 (3.38%, 364.0-414.0) | 385.0 (2.89%, 364.0-404.0) | -492.0 ms, -56.8% |
| decode ms/token | 14.454 (1.03%, 14.267-14.853) | 9.752 (CoV 10.39%, 9.324-13.914; kept 9.749, n=18) | 9.829 (9.32%, 9.455-12.944; kept 9.740) | -4.702 (-32.5%); kept -4.705 |

Across the three matrices (the tip is the pre-arena-commit binary in A/B):

| cell | base median | tip median | tip vs base |
|---|---|---|---|
| E2B 971-token decode ms/token | 11.665 (A), 11.681 (A2), 11.875 (A3) | 11.379 (A), 11.323 (A2), 11.063 (A3) | -2.5%, -3.1%, -6.8% |
| E2B 971-token prefill ms | 1455.0, 1417.0, 1411.0 | 609.0, 600.0, 594.0 | -58.1%, -57.7%, -57.9% |
| E2B short decode ms/token | 10.857 (B), 11.065 (B2, CoV 3.36%), 11.139 (B3) | 10.862 (B), 10.602 (B2, CoV 11.52%; kept 10.571), 10.874 (B3) | +0.05%, -4.2%, -2.4% |
| E2B short prefill ms | 138.0 (CoV 3.83%), 136.1 (8.79%), 124.0 (8.26%) | 90.0, 88.0, 88.0 | -34.8%, -35.3%, -29.0% |
| granite decode ms/token | 14.813 (A), 14.445 (A2), 14.454 (A3) | 9.427, 9.660, 9.752 | -36.4%, -33.1%, -32.5% |
| granite prefill ms | 911.0, 879.0, 866.0 | 377.0, 380.0, 374.0 | -58.6%, -56.8%, -56.8% |

Against the recorded llama-server numbers (a different session; ratio = tip / recording): E2B prefill 594.0 / 572 = 1.04, E2B decode 11.063 / 9.15 = 1.21, granite prefill 374.0 / 151 = 2.48, granite decode 9.752 / 5.25 = 1.86.

### memory, with the bound lines (`peak_rss_bytes`, `peak_footprint_bytes`, `peak_gpu_bytes`; median of 3 processes; bound = max(2% of base, |control - tip|))

| cell | metric | base | tip | control | bound | tip - base | against the bound |
|---|---|---|---|---|---|---|---|
| E2B 971 | footprint | 713,714,816 | 676,359,296 | 683,453,632 | 14,274,296 | -37,355,520 | inside |
| E2B 971 | RSS | 3,946,692,608 | 3,960,553,472 | 3,963,142,144 | 78,933,852 | +13,860,864 | inside |
| E2B 971 | GPU bytes | 5,608,554,496 | 3,838,328,832 | 3,838,328,832 | 112,171,090 | -1,770,225,664 | below base |
| E2B short | footprint | 218,491,072 | 218,474,816 | 222,177,536 | 4,369,821 | -16,256 | inside |
| E2B short | RSS | 3,651,534,848 | 3,629,268,992 | 3,649,667,072 | 73,030,697 | -22,265,856 | inside |
| E2B short | GPU bytes | 3,389,702,144 | 3,391,127,552 | 3,391,324,160 | 67,794,043 | +1,425,408 | inside |
| granite 1000 | footprint | 634,427,584 | 653,367,488 | 663,705,664 | 12,688,552 | +18,939,904 | OUTSIDE (+3.0%) |
| granite 1000 | RSS | 2,622,062,592 | 2,761,949,184 | 2,730,508,288 | 52,441,252 | +139,886,592 | OUTSIDE (+5.3%) |
| granite 1000 | GPU bytes | 3,315,433,472 | 1,697,382,400 | 1,697,382,400 | 66,308,669 | -1,618,051,072 | below base |

The two granite cells outside the bound are not explained. Spread inside the same matrix: granite RSS max over the 3 processes is 2,825,453,568 for base, 2,888,925,184 for the tip and 3,077,783,552 for the control,
so the base-to-tip step sits inside the control's own process-to-process spread; in run A (tip before the arena commit) the same cell read base 2,727,591,936 and tip 2,631,319,552 (-96 MB), the opposite sign.
n = 3 processes per cell.

Footprint attribution on E2B 971 (measured, `runD`, tip before the arena-extras commit, 3 processes x 2 runs): default 844,885,056; with `PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1` 680,635,328 (-164 MB, decode ms/token
11.285 -> 11.390); with the pipeline cache off 843,934,976; with the f16 tiled path off 835,546,048. With the r5 resident plan budget at 0 versus the default (`runC`) the E2B footprint was 849,456,064 versus
843,770,752. Read from `arena_encode_dispatch_finish.rs` before `85f1f23e`: each extra output (the twin's second node, top-k routes and weights, softmax weights) took a dedicated arena slot for the plan's lifetime.
`85f1f23e` returns them to the free list; the final tip's footprint on the same prompt reads 676,359,296 (`runA3`), against 844.9 MB before that commit (`runD`, a different matrix).

### attribution that the knobs allow (same binary, environment switch, interleaved)

| slice | switch | cell | with / without | effect | source |
|---|---|---|---|---|---|
| r6 twin | `PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1` | E2B 971 decode | 11.285 / 11.390 ms | -0.105 ms (-0.9%) | `runD` |
| r6 twin | same | granite 1000 decode | 9.606 / 9.859 ms | -0.253 ms (-2.6%) | `runE` |
| r3 tiled f16 | `PROXIMA_TILED_GEMM_DISABLE=f16` | E2B 971 prefill | 606.5 / 713.5 ms | -107.0 ms (-15.0%) | `runD` |
| r3 tiled q8_0 (with the grouped path) | `PROXIMA_TILED_GEMM_DISABLE=q8_0` | granite 1000 prefill | 372.0 / 6133.0 ms | 16.5x | `runE` |
| r4 8 command buffers | `PROXIMA_COMMAND_BUFFER_CHUNKS=1` | granite 1000 decode | 9.606 / 11.160 ms | -1.554 ms (-13.9%) | `runE` |
| r5 pipeline archive | fresh cache dir, 3 processes | E2B 25-token prompt, 16 tokens | ttft 599 -> 253 -> 242 ms | see below | `r5cache` |
| r5 resident plans | `PROXIMA_RESIDENT_PREFILL_PLAN_BYTES=0` | E2B / granite 971 | decode 11.362 / 11.377, 9.425 / 9.415 ms | no decode or prefill change in 9 runs per cell | `runC` |

r9 (matvec tail, one per-layer-input norm), r1 (row-tiled attention), r2's remaining commits, r4's fault-buffer pool and arena extras, r6's epilogue_sources grant, and r7 have no switch in this binary, so their share of
the totals above is unmeasured. The granite decode step from 14.454 to 9.752 ms is attributed by these knobs for -1.554 (r4) and -0.253 (r6) only; the remaining -2.9 ms is not attributed. The E2B decode step of
-0.8 ms and the prefill step of -817 ms are attributed only for the r3 f16 share above.

r5 cache check, `evidence/combine/r5cache/` (E2B, 25-token prompt, `PROXIMA_MAX_TOKENS=16`, `OMEGA_PIPELINE_CACHE_DIR` pointing at an empty directory, three consecutive processes of the tip example): process 1: archive
hits=4 stores=61 (61 backend compiles), ttft 599 ms, 61 files in the directory; process 2: hits=65 stores=0, ttft 253 ms; process 3: hits=65 stores=0, ttft 242 ms. The generated text hash is `4e361803a2e2d8fc` in all three.

### llama parity and digests

At the final tip (`final/nextest_interop.log`): `llama_parity_` 2 passed (gemma4_e2b, granite_moe), `generic_verify_llama_parity_` 2, `prefill_width_parity_with_llama_` 2, r7's `serving_default_ubatch_prefill_parity` 2 (971 tokens
each). The other checkpoints' parity tests were not run (order).

Digests (`evidence/combine/digests.md`): `gemma4_e2b.digest` recaptured (8 lines: ops 6074 -> 5667, ops_sha256, logits_root 6073 -> 5666, layer_roots sha256, and the same four for the verify program), explained by r9;
r9's derived estimate was 5632, and measured builder op counts (11 flat + 13 shared + 35 x 1 against 11 + 35 x 13) give -407, so the recapture is 35 ops above r9's arithmetic. `granite_moe.digest` is unchanged at the
tip (it moved by -792 only while the stacked default was on). The other six fixtures were not run.

### failed, reverted, fixed, open

- REVERTED `9aa0833c` (r2 0013, stacked experts in the interop `metal` set): `external_expert_paging` (2 tests) fails with it and passes at base; the CPU quantized reduce requires one activation row per leading position and a
  stacked gate/up reads one row for several gathered experts (`run_reduce_scan.rs:436-439`). The strategy stays behind its own feature. The granite numbers above are therefore measured without the stacked default; r2's own
  benefit from that default is not in them.
- FIXED `df9e38a1`: `dead_resolved_nodes` dropped a top-k whose consumers read only its stacked outputs (NaN logits under `moe-stacked-experts`; the synthetic parity diff could not see NaN, `6868398c`).
- FIXED `8eca5477`: r9's layer-axis view made the CPU packed reduce reject `blk.0.proj`; found by `gemma4_e2b_tiled_gemm_defaults_vs_all_off_full_logit_vector_diff`, which passes at base.
- CHANGED TEST `5cf74898` (r3 tiled f16 arms: error 0.0043 against an f64 dot, bound n x 2^-24 x sum|a w|, with a control that rejects a +10.0 corruption), `b8b036b1` (r4 allocation test on the placements executor),
  `a2edd376` (E2B census 1661 -> 1559 = 34 x 3 ops).
- FIXED `85f1f23e`: arena extras took dedicated slots (+164 MB measured on E2B 971 with the twin pass on).
- OPEN, same on base: `omega --features metal,instrument` 3 failures (`gates.md`); granite RSS and footprint outside the bound in `runA3` (unexplained, above); the six digests and all parity tests of the checkpoints the order excluded.

### reach

Measured: gemma4 E2B and granite moe 1b on Apple Metal. By construction (not measured): r3 reaches every model with a codec that has a `tiled_decode` description; r9's per-layer-input form reaches models with
`embedding_length_per_layer_input > 0` (gemma4 26B declares 0); r6's twin pass reaches any program with a `fused_rope_pair`; r4's 8-command-buffer setting is in the granite profile only; r5's archive cache reaches every Metal
pipeline; r2's stacked strategy reaches nothing by default.

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate --no-fail-fast                      # 763 passed, 16 skipped
cargo nextest run -p proxima-tensor --cargo-profile gate --no-fail-fast                               # 798 passed, 8 skipped
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate --no-fail-fast -E 'not test(~gemma4_26b) and not (binary(arch_data_baseline) and (test(~openchat) or test(~qwen) or test(~lfm2)))'   # 716 passed, 143 skipped
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm base=<ab69ec03 build> --arm tip=<tip build> --arm control=<byte copy of tip> --case gemma4_e2b=<gguf> --case granite_moe=<gguf>
```

## r2 stacked default restored (measured 2026-10-07, c63d839d to the commit that adds this section)

Evidence root: `evidence/r2fix/` (sibling of `evidence/combine/`). Raw per-process bench logs, the gate logs, the digest diff and the CPU fix patch are in it; the
binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/r2fix/bench/bin/` with sha256 in `evidence/r2fix/bench/binaries.sha256`. Test models: gemma4 E2B and granite moe 1b only.
The status of a sentence is the status of its weakest cell; no verdict is stated.

### what changed

| commit | change |
|---|---|
| `3158dbf0` | the CPU quantized reduce reads a gathered weight's activation row through the activation's own layout; test `stacked_projection_over_packed_q4k_experts_matches_the_per_route_graph_at_every_token_count` |
| `9c7a8819` | `b18a6537` reverted (stacked experts are in `metal` again) plus the recapture of three op-graph fixtures the default moves |

### CPU mechanism (read, then reproduced)

Reproduced before the fix, `evidence/r2fix/repro_paging.log`: with `std,metal,moe-stacked-experts`, `external_expert_paging` ran 4, 2 passed, 2 failed
(`paging_an_expert_between_steps_bumps_its_epoch_and_decode_continues`, `q2k_paging_actually_changes_the_decoded_ids`), both with `NotLowerable { node: NodeId(132), reason: "quantized matmul batch shape does not evenly divide by its packed weight rows" }`.

Cause, `run_reduce_quantized` in `proxima-tensor/src/cpu/run_reduce_scan.rs`: `leading_total` is the product of the output axes the weight does not vary over (`:330-358`). A stacked gate or up has output axes
`[token, selected, row]` with the expert gathered on `selected`, so `leading_total = tokens x selected`, while the activation holds `tokens x k` elements (stride 0 on `selected`; inferred from the layout-indexed read passing, the stride itself was not printed). The check at c63d839d
`:445` (`activation.len() != leading_total * k`) rejected it, and the loop at c63d839d `:588` (`&activation[position * k..(position + 1) * k]`) would have read the wrong row had the check passed.

Fix: `:448-452` keeps the dense length check only when the weight is not gathered; `:607-616` computes the activation row start from `activation_layout.offset_of(&full_coordinate)` (the same coordinate the gather
index already used) and slices `k` elements with a bounds check. No new type, no model name, no special case: a gathered weight reads each position's activation through the layout that owns the activation, so one row may feed
every expert a token selects. The non-gathered paths (wide folds, per-position loop) are unchanged and still index by position.

Test (`proxima-tensor/src/spec/tests.rs:3323`): `append_moe_ffn` with `Stacked` against `PerRoute` over packed Q4_K gate/up/down stacks (4 experts, 3 used, 256 x 256), tokens 1, 3 and 8, worst relative difference
(denominator floored at 1e-3) bound 1e-6. Passes with the fix (`cpu_parity.log`: 2 run, 2 passed, together with the existing f32 stacked test). Control, the same test with the fix patch reverse-applied
(`cpu_parity_control.log`): FAILS with `NotLowerable { node: NodeId(55), ... }`. The measured drift value was not printed; only the bound is asserted.

### gates at the re-applied tip (`evidence/r2fix/`)

| gate | result | log |
|---|---|---|
| `external_expert_paging` | 4 run, 4 passed (was 2 passed, 2 failed) | `after_fix_paging.log`, `gate_targeted.log` |
| clippy `-p proxima-tensor -p proxima-model-interop --features .../std,.../metal --all-targets -D warnings` | exit 0 | `gate_head_static.log` |
| `cargo check -p proxima-tensor --no-default-features --features alloc` | exit 0 | `gate_head_static.log` |
| `cargo check -p proxima-model-interop --no-default-features` | exit 0 | `gate_head_static.log` |
| `cargo nextest run -p proxima-tensor --cargo-profile gate` | 800 passed, 8 skipped (799 passed at `3158dbf0`, `tensor1.log`) | `gate_tip.log` |
| interop `slice-gate` at the re-applied default before the digest recapture | fail-fast run: 618 of 736 run, 617 passed, 1 failed; `--no-fail-fast`: 736 run, 730 passed, 6 failed, 124 skipped | `gate_tip.log`, `gate_tip_slice_nofailfast.log` |
| interop `slice-gate` after the recapture | 736 run, 736 passed, 124 skipped | `gate_tip_slice2.log` |
| targeted: `llama_parity_` (gemma4_e2b, granite_moe), `generic_verify_llama_parity_` (2), `prefill_width_parity_with_llama_` (2), r7 `serving_default_ubatch_prefill_parity` (2, 971 tokens each), `external_expert_paging` (4), `moe_stacked_default` (1) | 13 run, 13 passed | `gate_targeted.log` |

The 6 failures with the default on were `arch_data_digest_` and `model_config_roundtrip_` of granite moe, qwen35moe and gemma4 26B: their op-graph fixtures pin the per-route program. No parity test failed.

### digests recaptured (`PROXIMA_ARCH_DATA_CAPTURE=1`, `capture.log`; before and after in `digests_before/` and `digest_diff.patch`)

| fixture | ops before -> after | delta | per layer | logits_root |
|---|---|---|---|---|
| `granite_moe.digest` bind | 6394 -> 5602 | -792 | 24 layers x 33 | 6393 -> 5601 |
| `qwen35moe.digest` bind | 14982 -> 13662 | -1320 | 40 layers x 33 | 14981 -> 13661 |
| `gemma4_26b.digest` bind / verify | 13314 -> 10434 / 13312 -> 10432 | -2880 / -2880 | 30 layers x 96 | 13313 -> 10433 / 13311 -> 10431 |

granite matches the expected -792 = 24 x 33. The two other fixtures are the same mechanism and were recaptured because the gate cannot pass otherwise; `arch_data_digest_gemma4_26b` also hardcoded 13314 / 13313,
now 10434 / 10433. No parity test of qwen35moe or gemma4 26B was run (order: E2B and granite only). `gemma4_e2b.digest` did not change.

### bench (`decode_arms`, release std+metal, granite moe 1b, `prompt1k.txt` sha256 `46cb9a5b...`, 3 processes x (1 warmup + 7 runs), arms interleaved; `evidence/r2fix/bench/run1/`)

Arms: base = c63d839d (`git archive` export built release), tip = `9c7a8819`, control = byte copy of the tip (identical sha256), ref = the ab69ec03 build from `evidence/combine` (added for the memory question).
Box: Ollama down (curl :11434 refused), no cargo or nextest job (only an idle `sccache`), load average 3.84 / 5.67 / 7.18 before and 3.34 / 4.66 / 6.47 after, `suggestd` 78-82% and a background daemon in ~/.local/bin 78-81% CPU during the run (its name is replaced by `<background daemon>` in the two box_load files)
(`box_load_before.txt`, `box_load_after.txt`), GPU device utilization 0 before. The example forces `batch_size: 0, ubatch_size: 0`.

| metric | base | tip | control | ref (ab69ec03) | tip vs base |
|---|---|---|---|---|---|
| decode ms/token, all 21 runs | 9.618 (CoV 1.12%) | 6.822 (0.89%) | 6.892 (1.03%) | 14.545 (0.43%) | -2.796 (-29.1%) |
| decode ms/token, outliers removed | 9.614 (n=18, 0.40%) | 6.820 (n=20, 0.74%) | 6.892 (n=17, 0.45%) | 14.5445 (n=20, 0.28%) | -2.794 (-29.1%) |
| prefill ms (= ttft), all 21 runs | 368.061 (2.41%) | 371.054 (2.14%) | 385.035 (2.19%) | 853.960 (2.21%) | +2.993 (+0.8%) |
| prefill ms, outliers removed | 368.061 (n=21) | 369.527 (n=16, 1.40%) | 386.953 (n=19, 1.89%) | 853.960 (n=21) | +1.466 |

Bound for decode and prefill: max(2% of base, |control - tip|). Decode: max(0.192, 0.072) = 0.192; the tip is 2.794 below base, outside the bound in the faster direction. Prefill: max(7.361, 17.426 kept) = 17.426;
the tip is +1.466 (kept), inside. The two same-binary arms differ by 14.0 ms over all runs because prefill is bimodal (values near 367-370 and near 386-390 in both tip and control, `decode_arms.out`).

Memory, per process (bytes):

| metric | arm | process 0 | process 1 | process 2 | median | max - min |
|---|---|---|---|---|---|---|
| RSS | base | 2,522,251,264 | 2,491,465,728 | 2,648,621,056 | 2,522,251,264 | 157,155,328 |
| RSS | tip | 2,357,886,976 | 2,351,579,136 | 2,502,868,992 | 2,357,886,976 | 151,289,856 |
| RSS | control | 2,358,427,648 | 2,236,907,520 | 2,389,639,168 | 2,358,427,648 | 152,731,648 |
| RSS | ref | 2,477,342,720 | 2,619,146,240 | 2,487,910,400 | 2,487,910,400 | 141,803,520 |
| footprint | base | 593,926,592 | 617,126,144 | 598,170,048 | 598,170,048 | 23,199,552 |
| footprint | tip | 664,836,032 | 701,765,696 | 649,517,376 | 664,836,032 | 52,248,320 |
| footprint | control | 654,825,344 | 626,480,960 | 642,242,560 | 642,242,560 | 28,344,384 |
| footprint | ref | 610,769,216 | 655,399,680 | 673,585,792 | 655,399,680 | 62,816,576 |
| GPU bytes | base | 1,697,382,400 | 1,697,382,400 | 1,697,382,400 | 1,697,382,400 | 0 |
| GPU bytes | tip and control | 1,736,818,688 | 1,736,818,688 | 1,736,818,688 | 1,736,818,688 | 0 |
| GPU bytes | ref | 3,315,433,472 | 3,315,433,472 | 3,315,433,472 | 3,315,433,472 | 0 |

Bound lines (bound = max(2% of base, |control - tip|), medians of 3):

| metric | base | tip | control | bound | tip - base | against the bound |
|---|---|---|---|---|---|---|
| RSS | 2,522,251,264 | 2,357,886,976 | 2,358,427,648 | 50,445,025 | -164,364,288 | below base, but the per-arm spread is 151-157 MB and the ranges overlap (tip max 2,502,868,992 exceeds base min 2,491,465,728) |
| footprint | 598,170,048 | 664,836,032 | 642,242,560 | 22,593,472 | +66,665,984 | OUTSIDE (+11.1%); every tip and control value (626.5-701.8 MB) exceeds every base value (593.9-617.1 MB) |
| GPU bytes | 1,697,382,400 | 1,736,818,688 | 1,736,818,688 | 33,947,648 | +39,436,288 | OUTSIDE (+2.3%); identical in all 3 processes |

The open memory question (granite RSS +140 MB and footprint +19 MB outside the bound at c63d839d against ab69ec03), re-measured with ref = ab69ec03 and base = c63d839d in one matrix:
RSS c63d839d - ab69ec03 = 2,522,251,264 - 2,487,910,400 = +34,340,864 (bound 2% of ref = 49,758,208: inside); footprint = 598,170,048 - 655,399,680 = -57,229,632 (c63d839d lower, the opposite sign to the +18,939,904 seen before).
The earlier +139,886,592 RSS is not reproduced; the per-arm process-to-process spread in this matrix is 141.8-157.2 MB for RSS and 23.2-62.8 MB for footprint, the same size as that earlier step. n = 3 processes per cell: plausible (spread), not proven.

Does the stacked path change them: the GPU allocation does (+39,436,288 bytes, deterministic across 3 processes), and the footprint moves by +44 to +67 MB (control and tip against base) on the 1000-token prompt. RSS does not separate from the spread.
Short-prompt check (`prompt_short_hippo.txt`, 3 processes x (1 warmup + 3 runs), base and tip, `evidence/r2fix/bench/run2/`): GPU bytes 1,463,025,664 (base) -> 1,464,090,624 (tip), +1,064,960; footprint medians 145,396,480 -> 132,846,208
(tip lower); RSS 1,624,342,528 -> 1,607,548,928. The +39 MB GPU and the footprint increase therefore scale with prompt length; which buffers they are was not itemized, so the mechanism is open. Short-prompt decode ms/token 8.343 (CoV 17.21%, kept 8.324, n=6) -> 5.631 (0.24%); prefill 72.972 -> 38.987.

Against the recorded llama-server numbers (a different session, `evidence/slice0/ac1`): granite decode tip / recording = 6.822 / 5.25 = 1.30 (1.86 at the combined tip), granite prefill 371.054 / 151 = 2.46. Dispatch counts per prefill and per decode token were not re-counted here;
the program op count (-792) is the only structural measure.

### what this does not establish

- Which of the removed ops carry the -2.79 ms decode step: no run-time switch exists for the stacked default, so the base-to-tip step is the whole effect and is not split.
- qwen35moe and gemma4 26B execution under the default: only their programs' op counts changed (digests); no run.
- The mechanism of the GPU and footprint increase (above).
- E2B timing under this default: E2B has no routed experts, its digest did not change, and it was not re-benched.

### re-prove

```
cargo nextest run -p proxima-tensor --cargo-profile gate -E 'test(stacked_projection)'                                      # 2 passed
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate -E 'binary(external_expert_paging)'    # 4 passed
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate                   # 736 passed, 124 skipped
decode_arms --prompt-file evidence/r2fix/bench/prompt1k.txt --processes 3 --runs 7 --arm base=<c63d839d build> --arm tip=<tip build> --arm control=<byte copy of tip> --arm ref=<ab69ec03 build> --case granite_moe=<gguf>
```

## stacked experts memory (measured 2026-10-08, e9375ac2 to the commit that adds this section)

Evidence root: `evidence/memfix/` (`census/` per-plan arena itemization, `bench/` the interleaved matrix, `gate/` the gate logs, `rejected/` the three attempts that were rolled back, with their patches).
Raw per-process stderr, the binaries and the 80k-line census logs are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/next/`. Test models: gemma4 E2B and granite moe 1b only.
The status of a sentence is the status of its weakest cell; no verdict is stated.

### allocation budget (stated before the change)

Hot path (a warm step after the plan's arena exists): zero device buffers. Setup (the first placed call of a plan): one device buffer per output that is never released, plus one shared buffer
for every released output. Cold or error path: none beyond setup. Measured: `output_buffer_allocations` is 0 at steps 2 and 3 for granite and for E2B in `evidence/memfix/census/*_after_packed.txt`, and the new
`omega/tests/step_buffer_allocations.rs` test `a_warm_serial_routed_moe_step_allocates_no_device_buffers_and_matches_the_concurrent_step` asserts it (3 of 3 tests in the binary pass).

### which buffers grew (granite moe 1b, 999-token prefill plan, whole-slot arena, `census/granite_before_*.txt`)

`placed_arena_allocated_bytes` at step 0: 181,530,712 at e9375ac2 (stacked) against 141,858,424 at c63d839d (per route): +39,672,288. The per-size-class event (`buffer arena size class`, emitted by `emit_arena_census`,
`omega/src/metal/arena_encode_dispatch_finish.rs:352`) lists, for every output size, how many outputs of that size the plan writes, how many are live at once, and which op writes them:

| output size (bytes) | writer | e9375ac2 slots | c63d839d slots | held bytes, e9375ac2 - c63d839d |
|---|---|---|---|---|
| 16,367,616 (`[999, 8, 512]` f32) | `keep::reduce fold [999, 8, 1024, 512]` x48 (gate, and up with the activation epilogue, 24 layers) | 2 | 0 | +32,735,232 |
| 32,735,232 (`[999, 8, 1024]` f32) | `keep::reduce fold [999, 8, 512, 1024]` x24 (down) | 1 | 0 | +32,735,232 |
| 4,091,904 (`[999, 1024]` f32) | residual-width outputs | 3 | 9 | -24,551,424 |
| 2,045,952 (`[999, 512]` f32) | | 26 | 27 | -2,045,952 |
| 1,022,976 (`[999, 8, 32]` f32) | rope halves, pinned outputs | 48 | 48 | 0 |
| every class under 1 MiB | includes the top-k extras | | | +799,200 |

The classes sum to the +39,672,288. The top-k extra outputs the brief named are `moe_topk[999]` x408 of 3,996 bytes and x48 of 31,968 bytes: 3,164,832 bytes if every one were held for the plan's lifetime, and the
census shows `peak_live` 224 and 2, so they are recycled. They are not the growth. The growth is the three `[tokens, selected, width]` outputs of the stacked projections, which are live: at the position where the
live total peaks (`live peak by size class`: position 363, a down fold, 152,359,504 bytes) one 16,367,616-byte hidden buffer and the 32,735,232-byte down output are live together with 98,205,696 bytes of pinned
K/V rows. The per-route plan's peak (137,214,664 bytes, position 842) holds eight 4,091,904-byte buffers instead. Source of the shape: `proxima-tensor/src/spec/gqa_layer_routed.rs:1205-1271` (gate and up over `skio`, down at `:1235`,
the weighted reduce over the selected axis at `:1257-1271`).

Allocation against liveness: e9375ac2's arena held 181,530,712 bytes for a live peak of 152,359,504 (29,171,208 held beyond the peak, 16.0%); c63d839d's held 141,858,424 for 137,214,664 (4,643,760). The excess is
size-class fragmentation: gate/up and down are different lengths, so the whole-slot free list could not hand one to the other.

### what landed

| commit | change |
|---|---|
| `838d4962` | `omega/src/metal/arena_layout.rs` (new, device-free, 8 unit tests): `lay_out_packed` gives every output that some later op retires a byte range in one shared buffer, largest first at the lowest aligned offset no simultaneously live output occupies; outputs never retired keep a buffer of their own at offset 0 (the readback invariant); `lay_out_whole_slots` is the previous free-list policy unchanged. `build_buffer_arena` (`arena_encode_dispatch_finish.rs:251`) packs when the plan is `DispatchType::Serial` (`:303`) and falls back to whole slots otherwise or when a slot exceeds `maxBufferLength`; it now refuses an over-cap plan before allocating. `BufferArena` carries a byte offset per position and per extra output; `encode_op` already bound `(buffer, offset)`. Two `arena_tests`: packed arena equals the CPU oracle with one shared buffer for four released outputs of four lengths, and serial-packed equals concurrent-whole-slot bit for bit. |
| `11edadd1` | the size-class itemization above as a `debug!` event under `instrument` |
| `f48386a0` | warm serial routed step allocates no device buffers and matches the concurrent result |

Why Serial only: `HazardTracker` (`execute_and_hazards.rs:1217`) names a buffer by pointer, so a shared buffer would read as one resource and every concurrent dispatch would get a barrier. `DispatchType::default()` is
`Concurrent`; `ServingConfig::default()` sets `Serial`. Constants: `RANGE_ALIGNMENT` 256 (offset alignment) and `RANGE_GUARD` 256 (a whole-slot buffer is page-rounded and absorbed a kernel store past its last element;
a range with a live neighbour has no such slack) are hardware and safety values, not per-system caps; no sizing-TOML axis.

### before and after (instrumented release build, one process, 999-token prompt, `census/`)

| model | metric | e9375ac2 | this tree | delta |
|---|---|---|---|---|
| granite | prefill arena bytes | 181,530,712 | 152,560,344 | -28,970,368 (-16.0%) |
| granite | `device_allocated_bytes` at step 1 and later | 1,723,957,248 | 1,694,793,728 | -29,163,520 |
| granite | `phys_footprint_bytes` at step 1 and later (n=1) | 1,934,827,840 | 1,898,438,912 | -36,388,928 |
| granite | live peak / arena after | 152,359,504 | 152,359,504 / 152,560,344 | arena 200,840 above the peak |
| E2B | prefill arena bytes | 361,677,704 | 181,827,720 | -179,849,984 (-49.7%) |
| E2B | `device_allocated_bytes` at step 1 and later | 3,782,934,528 | 3,602,726,912 | -180,207,616 |
| E2B | `phys_footprint_bytes` at step 1 and later (n=1) | 2,060,481,600 | 1,886,614,400 | -173,867,200 |
| E2B | live peak / arena after | 177,797,984 | 177,797,984 / 181,827,720 | arena 4,029,736 above the peak |

Generated text hash is identical before and after: granite `fbe77b83cee157bf`, E2B `a032fc72c57a69e4` (4 tokens, `PROXIMA_SPECULATIVE_TYPES=none`).

### bench (`decode_arms`, release std+metal, granite moe 1b, `prompt1k.txt`, 3 processes x (1 warmup + 7 runs), arms interleaved; `evidence/memfix/bench/`)

Arms: base = e9375ac2 build (sha256 `b160c44e...`, byte-identical to the r2fix `tip` binary), tip = this tree, control = byte copy of tip (`3c600c45...`), ref = c63d839d per-route build (the r2fix binary), refpacked = c63d839d
with this commit's arena files copied in (the per-route program under the same layout). The example forces `batch_size: 0, ubatch_size: 0`. Box: Ollama down (curl :11434 refused), no cargo job during the run, load average 10.59 / 15.72 / 13.20 before and 8.76 / 11.73 / 12.01 after,
`mds_stores` 111%, `mediaanalysisd` 64%, a background daemon 40% CPU (`box_load_before.txt`, `box_load_after.txt`).

| metric | base | tip | control | ref (c63d839d) | refpacked |
|---|---|---|---|---|---|
| decode ms/token, 21 runs | 6.7210 (CoV 3.34%, 6.585-7.709; kept 6.7155, n=20, 1.42%) | 6.8300 (1.91%, 6.638-7.276; kept 6.8165, n=20, 1.35%) | 6.7450 (3.56%; kept 6.7450, n=20, 1.18%) | 9.6910 (1.15%) | 9.7660 (2.02%) |
| prefill ms | 378.978 (3.11%) | 381.047 (2.39%) | 385.012 (3.74%) | 382.022 (3.64%) | 376.001 (3.52%) |
| `peak_gpu_bytes`, 3 of 3 processes identical | 1,736,818,688 | 1,707,655,168 | 1,707,655,168 | 1,697,382,400 | 1,694,908,416 |
| `peak_footprint_bytes` per process | 691.9 / 709.0 / 638.3 MB | 635.7 / 673.3 / 600.8 MB | 706.2 / 601.0 / 603.2 MB | 637.2 / 653.9 / 660.1 MB | 648.1 / 650.0 / 627.0 MB |
| `peak_rss_bytes` per process | 2.744 / 2.601 / 2.476 GB | 2.560 / 2.492 / 2.729 GB | 2.880 / 2.490 / 2.522 GB | 2.541 / 2.658 / 2.729 GB | 2.748 / 2.768 / 2.762 GB |

Bound lines (`limit` as printed by `decode_arms` in its `bound` rows, 2% of the reference arm for memory and its outlier-aware spread for time; `control - tip` is the same-binary difference, the noise indicator):

| comparison | metric | delta | bound | against the bound |
|---|---|---|---|---|
| tip - base | decode ms/token (kept medians) | +0.1010 (+1.5%) | 0.1343 | inside |
| tip - base | prefill ms | +2.069 | 8.940 | inside |
| tip - base | `peak_gpu_bytes` | -29,163,520 (-1.68%) | 34,736,374 | inside, lower |
| tip - ref | `peak_gpu_bytes` | +10,272,768 (+0.605%) | 34,153,103 | inside (was +39,436,288, outside, at e9375ac2) |
| tip - refpacked | `peak_gpu_bytes` | +12,746,752 (+0.752%) | 34,153,103 | inside |
| tip - base | `peak_footprint_bytes` (medians) | -56,213,696 | 13,838,710 | outside, lower; control - tip is -32,522,176, per-arm spread 72-105 MB, ranges overlap, n=3: plausible, not proven |
| tip - ref | `peak_footprint_bytes` (medians) | -18,153,856 | 12,714,436 | outside, lower; smaller than the same-binary control - tip difference of 32,522,176 |
| control - tip | decode ms/token | -0.0715 | 0.1363 | inside |

Decode ms/token is 6.83 (tip, all 21 runs); the same binary under the control label reads 6.745, which is the noise floor of this matrix on this box. The 9.69 of the per-route build against 6.82 is the r2 effect, unchanged.
RSS does not separate from its 250 MB per-arm spread.

### rejected, with the numbers (each patch is in `evidence/memfix/rejected/`)

- ROLLED BACK, reuse a retired slot for a smaller output across size classes (smallest retired slot at least as long, outputs read back kept exact). Arena at step 0, granite: 185,750,784 whole-slot, 183,704,832 (ratio cap 2),
  183,703,092 (ratio 3 and 4), 182,939,756 (ratio 8): at most -2,811,028. With no cap the arena grew to 1,307,571,980 and the prefill plan fell out of the resident budget. The gap is not idle slots of the wrong size.
- ROLLED BACK, allocate no arena slot for a node the caller output-places. Granite prefill arena 181,530,712 -> 181,334,092 (-196,620); E2B 361,677,704 -> 360,629,128 (-1,048,576); decode plan arena 577,232 -> 282,308 bytes (peak 495,084 -> 200,160). The prefill
  K/V rows are not output-placed on this path (they are read back), so there is nothing to skip; the 98 MB of pinned K/V in the arena is untouched.
- ROLLED BACK, drop the `[tokens, selected, 1024]` down output by reducing the selected and hidden axes in one gathered fold (`probe_multiaxis_combine.patch`, `gqa_layer_routed.rs` stacked branch). Metal run:
  `Metal(Tensor(GatherIndexOutOfRange { node: NodeId(2536), index: 0, extent: 32 }))`; CPU: `NotLowerable { node: NodeId(66), reason: "quantized matmul batch shape does not evenly divide by its packed weight rows" }` in
  `stacked_projection_over_packed_q4k_experts_matches_the_per_route_graph_at_every_token_count` and a drift of 4.73e-7 over the f32 stacked test's bound. The gathered quantized fold does not accept a second reduced axis on either executor.

### gates (`evidence/memfix/gate/`)

| gate | result |
|---|---|
| clippy `-p proxima-tensor -p proxima-model-interop --features .../std,.../metal --all-targets -D warnings`; `-p omega --features metal` and `metal,instrument` | exit 0 (4 invocations) |
| `cargo check -p proxima-tensor --no-default-features --features alloc`; `-p proxima-model-interop --no-default-features`; `-p omega --no-default-features` | exit 0 (the arena modules are `metal` and macOS only, so no restricted-tier module was built) |
| nextest `-p proxima-tensor --cargo-profile gate` | 800 passed, 8 skipped |
| nextest `-p omega --features metal --cargo-profile gate` | 773 passed, 16 skipped (763 + the 10 new tests) |
| omega `--features metal,instrument` | 826 run, 823 passed, 3 failed: `q4_0_two_token_index32_dispatch_classifies_as_packed_row_blocked`, `rmsnorm_fused_epilogue_air_division_count_decode_shape`, `..._prefill_shape` (the same three as `evidence/combine/gates.md`) |
| omega gated features (`binary(step_buffer_allocations)` and 6 other alloc and parity binaries) | 20 passed |
| interop `slice-gate` | 736 passed, 124 skipped |
| `external_expert_paging`, `llama_parity_`, `generic_verify_llama_parity_`, `prefill_width_parity_with_llama_` (gemma4_e2b, granite_moe), r7 `serving_default_ubatch_prefill_parity` (971 tokens), `moe_stacked_default` | 13 run, 13 passed |
| control: packing test with the plan left `Concurrent` | fails, `left: 12 right: 9` (`gate/control_packing_disabled.log`) |

`ServingConfig::default()` is `Serial`, so the interop parity tests ran the packed arena; the omega unit and integration tests default to `Concurrent` and run the whole-slot layout except the three tests that set `Serial`.

### what this does not establish

- Stacked experts still cost more than per-route at the same layout: +12,746,752 `peak_gpu_bytes` (refpacked to tip), which is the 32,735,232-byte down output held with the 16,367,616-byte hidden buffer. It is inside the 2% bound; it is not zero.
- Footprint: n=3 per arm with a 72-105 MB spread; the -56 MB and -18 MB medians are not separated from it.
- `DispatchType::Concurrent` plans keep the previous layout; no Concurrent timing or memory was taken.
- Only granite moe 1b and gemma4 E2B were executed. Every other checkpoint under `Serial` now gets a packed arena and was not run. The digests do not change (the arena is not part of the op graph).
- The example forces `ubatch_size: 0`; memory at the r7 default ubatch of 512 was not benched.
- Ollama and llama-server were not run.

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate -E 'test(arena_layout) or test(arena_tests)'      # 18 passed
cargo nextest run -p omega --features metal,instrument,moe-topk-fusion --cargo-profile gate -E 'binary(step_buffer_allocations)'   # 3 passed
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate           # 736 passed, 124 skipped
cargo build --release -p proxima-model-interop --example decode_gbps_baseline --features std,metal,instrument
RUST_LOG=debug PROXIMA_DECODE_MODEL_GGUF=<granite> PROXIMA_PROMPT="$(cat evidence/memfix/bench/prompt1k.txt)" PROXIMA_MAX_TOKENS=4 PROXIMA_RUNS=1 decode_gbps_baseline   # grep 'buffer arena'
decode_arms --prompt-file evidence/memfix/bench/prompt1k.txt --processes 3 --runs 7 --arm base=<e9375ac2 build> --arm tip=<tip build> --arm control=<byte copy of tip> --arm ref=<c63d839d build> --arm refpacked=<c63d839d with the arena files> --case granite_moe=<gguf>
```

## attribution after the combined landing (measured 2026-10-08 on `0ca915e8`; the omission binary adds `eace68a8`, a tool-only change)

Evidence root: `evidence/attr2/` (`census_*` the kernel censuses, `census_*_minlen9` the same with one build constant changed, `rank/` the `attribution_rank` tables and step timelines, `timeline/` the `token_breakdown` event lines,
`omission/` the captured-step omission runs, `binaries.sha256`). Raw stderr and the binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/next/`. Test models: gemma4 E2B and granite moe 1b only.
llama.cpp and Ollama were not run: the llama side of every comparison is `evidence/slice0/llama_ops/*.tsv` and `evidence/slice0/ac1/decode_arms.out`, recorded 2026-10-07 on a different box state.
Every number names its basis: **own-cb** = median GPU span of one dispatch alone in its own command buffer (floor of about 4 us included on both sides), **in-situ** = a captured step replayed in one command buffer with the group's dispatches removed
(`norm_variant_ab`, `AB_OMIT_ALL`, the r8 method) or a family replayed in program order, **wall** = host clock. DERIVED marks arithmetic on measured numbers; ASSUMED marks a spec-sheet or upstream-default value.

### the four gaps, with the numbers they come from

| model, phase | proxima | llama (recording) | ratio | proxima source |
|---|---|---|---|---|
| granite prefill, 1000 tokens | 381.0 ms (CoV 2.39%, 370.0-406.1, 21 runs) | 151 | 2.52 | `evidence/memfix/bench/decode_arms.out` arm `tip` (this session, `ubatch 0`); 371 at e9375ac2 in `evidence/r2fix` |
| granite decode | 6.83 ms/token (all 21 runs, CoV 1.91%) | 5.25 | 1.30 | same file |
| E2B decode, 971 tokens | 11.06 ms/token | 9.15 | 1.21 | `evidence/combine/bench/runA3` (not re-run this session) |
| E2B prefill, 971 tokens | 594 ms | 572 | 1.04 | same |

Box during the censuses and timelines: Ollama down (curl :11434 refused), no cargo job, `mds_stores`, `mediaanalysisd` and a background daemon at 40-110% CPU, GPU "Device Utilization %" sampled 0 before each run with
one-sample spikes of 80-99% from other processes (`census_*/box_load.txt`, `timeline/*_box_load.txt`). The `decode_arms` bench above ran with the same background load (`evidence/memfix/bench/box_load_before.txt`).

### GPU stalls during this pass (kept, not buried)

Four `norm_variant_ab` omission runs stopped producing output while the process sat at 0% CPU: the full run (00:35, five groups printed, killed 00:46), the `AB_SHA=49f4a864` probe (00:50), a run with the first group skipped (00:56, no output in 135 s because the GPU was already busy),
and the same run again (01:04, 25 groups printed, killed 01:07). After each kill "Device Utilization %" read 100 until 01:04:22 (17 minutes after the first kill) and until 01:25:17 (18 minutes after the last), with 4.1 GB of GPU memory in use that fell back to 0.1 GB.
While it read 100, a tiny omega test ran in 0.12 s and an E2B decode ran at 29.7-32.5 ms/token (3x) with 2.4-2.8 s TTFT (`decode_gbps_tip`, 24 tokens). The runs that printed a stall point stopped at the omission of a q-norm group: `49f4a864... [1, 8, 256]` (28 dispatches)
and `cdf3715b... [1, 8, 512]` (7 dispatches); omitting the `[1, 1, N]` groups of the same kernels did not stall. Cause: not isolated. One hypothesis, untested: the arena now packs released outputs into shared ranges, so an omitted producer leaves its output range holding another node's
bytes. The final omission runs skip those two groups (`AB_SKIP_GROUPS`, `eace68a8`) and completed in 45 s each (E2B) and 10 s (granite). The reported timings were taken with the GPU reading 0 before each run except the first granite prefill census, whose pre-run sample read 94 and whose following eight samples read 0 with one 98
(`census_granite_prefill/` has no box file; the sample is in the session log only, so treat that census as plausible, not proven); the sequence replay of that census (361.9 / 360.4 / 362.0 ms) sits 8% above the live `gpu_busy` (335-356 ms).
`omission/probe49f4.out` and `omission/e2b_decode_unpacked_partial_run1.out` are the stalled runs.

### granite prefill, one 1000-token evaluation (`census_granite_prefill/`, `rank/granite_prefill.md`)

Dispatches 383 (1847 at slice 0). Census sequence replay 361.9 / 360.4 / 362.0 ms for 359 replayable dispatches (24 `moe_topk_stacked` are not replayable); live step 0: `gpu_busy` 335.3-355.5 ms, wall 378.9-401.5 ms
(`rank/granite_steps.md` runs 1-2, `rank/granite128_steps.md`). Sum of own-cb 396.3 ms against `gpu_busy` 335 (1.18x). Ranked by own-cb milliseconds recovered if the row reached its reference:

| rank | row | ops | proxima ms (us/op) | reference | recovered ms | efficiency (DERIVED) | mechanism |
|---|---|---|---|---|---|---|---|
| 1 | MoE combine fold: reduce over the 8 selected experts, epilogue of 10 operands (`omega_reduce_r3_o2_n2_multiply_add_zero_epi10_...`, extents `[1000, 8, 1024]`) | 24 | 99.23 (4134) | **567.7 us/op measured with the serial route** (13.6 ms) | **85.6** | 36.9 MB in 4134 us = 8.9 GB/s; 2.2% of the 400 GB/s spec (ASSUMED); at 568 us 65 GB/s | The fold is 8 long, so `reduce_is_cooperative` (`omega/src/msl/emit_and_classify.rs:1669`, length gate `meets_cooperative_min_len` `:1992`, `min_len = 2` at `omega/omega-runtime.toml:125`) takes the cooperative body (`tiled_gemm_cooperative_scan.rs:1994`) and the grid is 32,768,000 = 1000 x 1024 x `SIMD_WIDTH` 32 (`census_dispatches.csv`, node 249): one 32-lane simdgroup per output, 24 lanes idle. Toggle, measured: build with `OMEGA_COOPERATIVE_REDUCE_MIN_LEN=9`: grid 1,024,000, 567.7 us/op, whole-step replay 361.9 / 360.4 / 362.0 -> 299.4 / 278.2 / 277.0 ms (-82 to -85 on the last two samples) |
| 2 | expert gate, up (+ silu epilogue) and down, stacked `[1000, 8, .., ..]`, plus the K/V projections, Q8_0 1024x512 | 72 + 48 | 203.05 (gate 2921, up 2737, down 2430 us/op cold; K/V 170 us/op) | llama 114.41 ms for the same shape class (237 ops) | 88.6 | 8.39 GFLOP per expert op: 2.87 / 3.06 / 3.45 TFLOP/s; per layer proxima 8.46 ms against llama 4.77 ms (the class includes the K/V projections on both sides), llama 25.2 GFLOP of experts per layer = 5.3 TFLOP/s | grouped-GEMM body `push_expert_grouped_gemm_body` (`omega/src/msl/expert_grouped_gemm.rs:60`); the tile and occupancy cause of the 1.7x per layer is untraced |
| 3 | rope as two fused elementwise "twin" kernels (`omega_elementwise_twin_r3_n4_..._subtract/_add_...`) on Q `[1000, 16, 32]` and K `[1000, 8, 32]` | 48 | 35.20 (Q 974, K 493) | llama 1.71 ms, 96 ops | 33.5 | Q: 8.2 MB in 974 us = 8.4 GB/s; K: 4.1 MB in 493 us = 8.3 GB/s | `ElementwiseTwin` (`proxima-tensor/src/bind/types_layout_boundop.rs`) renders through `signature_tokens_prelude.rs:404`; 512,000 threads, `tg=None`; why one pass over 4 MB takes 974 us: untraced |
| 4 | cached attention partial (`q1000 c1024 h8 g2 d64`) | 24 | 27.66 (1153) | llama 12.93 ms, 95 ops | 14.8 | 4.10 GFLOP per layer: 3.6 TFLOP/s; llama 7.6 (crude, DERIVED with one non-causal count for both; llama runs two graphs of 512 and 488 tokens) | row-tiled kernel from r1; untraced below the kernel |
| 5 | router matvec F32 1024x32 | 24 | 11.55 (481) | llama 2.17 ms, 47 ops | 9.4 | 65.5 MFLOP in 481 us = 0.14 TFLOP/s; 4.4 MB = 9.1 GB/s | grid 4096 threads in threadgroups of 128 for 32,000 outputs; occupancy, untraced |
| 6 | Q8_0 1024x1024 projections, rms norm, head | 48 + 49 + 1 | 16.09 + 2.87 + 0.16 | llama 15.08 + 1.77 + 0.22 | 1.0 + 1.1 | | within 1.07x and 1.62x |
| host | outside the GPU: `wall - gpu_busy` | | 31-54 ms (readback 10.2-20.2, kv append 8.0-19.7, kv named 5.5-8.9; the fields overlap) | | up to 40 | | the 98 MB of pinned K/V rows (`evidence/memfix`) are read back and appended on the host |

Own-cb recoverable total of rows 1-5: 232 ms; scaled by the 0.85 ratio of `gpu_busy` to own-cb it is about 196 ms, which would put the step near 185 ms against llama 151, the rest being the host row. Rows 1 and 2 are 75% of it.

### granite decode, one step after the 1000-token prompt (census step 23, omission step 5; `census_granite_decode/`, `omission/granite_decode.out`, `rank/granite_decode.md`)

Dispatches 398 (926 at slice 0). Own-cb sum 6.27 ms against llama 6.49: the kernels are at parity by that measure (matmul 3.87 against 3.99, `-0.12`). Census sequence replay of 374 dispatches 5.14-5.20 ms; omission harness base replay
5.81-6.04 ms (median 5.92, 26 groups; it replays the same dispatches without the census's flush, a 0.7 ms difference between the two replays that is not explained). Live step, 128-token run (`rank/granite128_steps.md`, n=112-126 steps per run, instrumented build): wall 6.48-6.71,
evaluate 6.42-6.63, `gpu_busy` 5.92-5.94, `gpu_exec` 5.25-5.29, encode 0.46-0.47, 8 chunks; 16-token run: wall 6.21-6.25, `gpu_busy` 5.65-5.68. Release bench: 6.83 ms/token.

By omission (ms the step loses without the group; base 5.92 ms; the noise is the spread of the 26 base figures, 0.22 ms, so rows under 0.2 ms are unresolved):

| rank | group | ops | ms | us/op | llama own-cb class | note |
|---|---|---|---|---|---|---|
| 1 | cached attention partial `[1, 8, 2, 64]` | 24 | 1.231 | 51.3 | attention 0.73 ms, 48 ops (15 us/op) | the one class above llama on own-cb (+0.66 ms) |
| 2 | expert down `[1, 8, 512, 1024]` | 24 | 0.869 | 36.2 | matmul class 3.99 ms vs 3.87 | 4.46 MB per op: 123 GB/s (DERIVED) |
| 3 | expert up + silu `[1, 8, 1024, 512]` | 24 | 0.791 | 33.0 | | 135 GB/s |
| 4 | expert gate `[1, 8, 1024, 512]` | 24 | 0.624 | 26.0 | | 172 GB/s |
| 5 | rms norm + epilogue `[1, 1024]` | 49 | 0.557 | 11.4 | norm 0.37 ms | +0.19 ms on own-cb |
| 6 | K/V projections `[1, 8, 64, 1024]` | 48 | 0.426 | 8.9 | | |
| 7 | combine fold, 10 epilogue operands `[1, 8, 1024]` | 24 | 0.298 | 12.4 | | at decode the cooperative route is the right one: the same constant at 9 slows the step (below) |
| 8 | attention out, q projection, router, head, rope twins, attention merge | | 0.277, 0.240, 0.214, 0.176, 0.176 + 0.154, 0.159 | | | |

Where the 6.83 ms goes against llama's 5.25 (wall basis, 128-token timeline, n=3 runs): `gpu_busy` 5.92-5.94 is 0.67-0.69 above llama's whole step; `evaluate - gpu_busy` is 0.49-0.69 ms of host time that is not hidden behind the GPU; `wall - evaluate` is 0.06-0.08;
and the release per-token mean (6.83) sits 0.12-0.35 ms above the instrumented steps' median wall (6.48-6.71). The kernels do not carry this gap on the own-cb basis; the structure around them does, and nothing in this pass attributes the 0.49-0.69 ms.

### E2B decode, one step after the 971-token prompt (census step 23, omission step 5; `census_e2b_decode/`, `omission/e2b_decode_packed.out`, `rank/e2b_decode.md`)

Dispatches 653 (941 at slice 0). Own-cb sum 11.96 ms against llama 11.10 (+0.86). Census sequence replay 10.58-10.67 ms; omission base replay 10.61-10.71 (45 groups, spread 0.095 ms, 30 rounds); live step wall 10.61-11.00, `gpu_busy` 9.91-10.32,
`evaluate` 10.56-10.95 (`rank/e2b_steps.md`, 16-token runs). Family replays in program order sum to 10.55 ms: Q4_0 matvec 5.16, norms 2.48, attention 1.11, head 1.10, rope/copy/elementwise 0.575, other 0.13.

| rank | group | ops | omission ms | us/op | llama own-cb | bytes rate (DERIVED) | recoverable ms |
|---|---|---|---|---|---|---|---|
| 1 | head Q6_K `[1, 1536, 262144]` | 1 | 1.061 | 1060.7 | 0.94 ms | 330 MB in 1061 us = 311 GB/s; llama 351 | 0.12 |
| 2 | ffn_down Q4_0 K=12288 | 20 | 0.943 | 47.15 | 38.3 us/op (60 ops of 1536x12288, 2.30 ms) | 10.6 MB in 47.2 us = 225 GB/s; llama 277 | 0.18 |
| 3 | ffn_gate + gelu epilogue K=1536 N=12288 | 20 | 0.884 | 44.21 (73.0 before r8) | 38.3 | 240 GB/s | 0.12 |
| 4 | rms norm epilogue `[1, 1536]` (epi5) | 70 | 0.864 | 12.34 | 7.4 us/op (242 ops, 1.80 ms) | | |
| 5 | ffn_up plain N=12288 | 20 | 0.842 | 42.09 | 38.3 | 252 GB/s | 0.08 |
| 6 | rms norm epilogue `[1, 1536]` (epi4) | 71 | 0.728 | 10.26 | 7.4 | | |
| 7 | sliding attention partial `[1, 1, 8, 256]` | 28 | 0.727 | 25.96 | attention 1.21 ms, 35 ops (all) | | |
| 8 | attn q projection `[1, 8, 256, 1536]`, attn output `[1, 1, 8, 256, 1536]` | 28 + 28 | 0.404 + 0.389 | 14.4, 13.9 | | | |
| 9 | global attention partial `[1, 1, 8, 512]` | 7 | 0.389 | 55.5 | | | |
| 10 | per-layer norms `[1, 1536]` epi6, `[1, 1536, 256]` epi5 | 35 + 35 | 0.366 + 0.340 | 10.4, 9.7 | | | |

The four rms-norm groups in ranks 4, 6 and 10 add to 2.30 ms in situ over 211 dispatches (10.9 us each) against llama's 1.80 ms own-cb over 242 (7.4 us each, floor included on llama's side only): about 0.7 ms at llama's per-dispatch cost. The 12288-wide Q4_0 groups (gate, up, down) total 2.67 ms and would lose 0.37 ms at llama's
38.3 us/op; the 6144-wide ones (21.7-24.0 us/op, 1.03 ms) are at llama's 23.6 us/op. Two q-norm groups (28 + 7 dispatches) are not in this table (stalled, above); their cost is inside the 2.48 ms norm family replay.
Wall basis: `gpu_busy` 9.91-10.32 is 0.8-1.2 ms above llama's whole step of 9.15; wall 10.6-11.0.

### E2B prefill

No census was taken (not requested). Timeline (`rank/e2b_steps.md`, step 0, warm, runs 1-2): wall 589.2-591.4, `gpu_busy` 567.3-568.6, readback 4.8-4.9, kv_named 5.7-6.4, kv_append 3.6-3.9. The GPU span is at llama's whole prefill (572); the 22 ms above it is host time.

### one build constant, three censuses (`census_*_minlen9/`)

`OMEGA_COOPERATIVE_REDUCE_MIN_LEN=9` (default 2; `omega/omega-runtime.toml:125`) sends reduces shorter than 9 to the one-thread-per-output body. Whole-step replay, three samples each, default -> 9:

| step | default | min_len 9 | change |
|---|---|---|---|
| granite prefill | 361.9 / 360.4 / 362.0 | 299.4 / 278.2 / 277.0 | -63 to -85 ms |
| granite decode | 5.14 / 5.20 / 5.15 | 6.00 / 6.01 / 5.90 | +0.7 to +0.9 ms |
| E2B decode | 10.67 / 10.63 / 10.58 | 11.84 / 11.73 / 12.52 | +1.1 to +1.9 ms |

The length gate cannot serve both: the same constant that removes 85 ms from prefill adds about 1 ms to a decode step. What differs is the number of outputs (1,024,000 in the prefill fold; a few thousand at decode), which the gate does not read.
`omega-runtime.toml` already records that a global threshold made 64-element attention folds slower at decode.

### proposed next slices, each with a target and the number it moves

1. **Short cooperative folds by output count** (rank 1, granite prefill). Choose the serial body for a fold shorter than `min_len`-class lengths only when it has at least about 64K outputs (the prefill combine has 1,024,000; decode folds have under 10,000); keep the cooperative body otherwise. Target: combine 4134 -> at most 600 us/op, granite prefill own-cb -85 ms,
   step 381 -> at most 300 ms (derived from the toggle, whole-step replay -82 to -85 ms). Gates: the three censuses above must show decode replay unchanged (5.14-5.20, 10.58-10.67), `llama_parity_`, `generic_verify_llama_parity_`, the digests, and a `decode_arms` run with a control arm. The fold order of 8 elements changes from a 32-lane tree to a sequential sum, so a parity test over the combine at 1000 tokens is part of the slice.
2. **Fuse the combine into the down projection** (rank 1 and the memory finding). The combine reads a `[1000, 8, 1024]` buffer that exists only to be reduced over the selected axis. Removing it removes 32,735,232 bytes of arena (`evidence/memfix`) and the 568 us/op after slice 1. Needs a gathered quantized fold with a second reduced axis, which both executors reject today (`evidence/memfix/rejected/probe_multiaxis_combine.patch`).
   Target: arena -32.7 MB, prefill a further -13 ms. Larger and later than slice 1.
3. **Stacked expert GEMM occupancy** (rank 2). Trace first: read the per-threadgroup occupancy and tile shape of `push_expert_grouped_gemm_body` for the three projections against llama's `kernel_mul_mm_id`, then change one parameter at a time under the r3 parity test. Target: 203 -> at most 115 ms own-cb (llama 114.4), -88 ms.
4. **Rope twin kernels** (rank 3). Capture one Q twin dispatch and replay variants with `norm_variant_ab` (the tool already compiles a variant against a captured dispatch's own buffers). Target 35.2 -> at most 3 ms, -32 ms; the mechanism is untraced, so the first deliverable is the trace.
   Written, not run (rope kernel slice): `render_elementwise` (`omega/src/msl/elementwise_reduce_core.rs`) decodes the thread index of a 512,000-thread twin with two runtime `%` and two runtime `/` per thread (rank 3, no operand dense), and the repo's own note on that decode is that Apple GPUs emulate integer division as a long dependent sequence. That is a code-read candidate, not a measured cause: the 1.9 ns/thread rate is the same for Q (512,000 threads, 974 us) and K (256,000, 493 us), which fits any per-thread cost. The change replaces the divides with a float reciprocal plus one signed correction for grids in `[omega-runtime.toml [elementwise].reciprocal_min_elements, 2^22]`, keyed into the pipeline identity (`_ear`). Unmeasured: whether the twin moves at all. Falsifier, to run with the other slices: the Q twin stays at 974 us +/- noise with `_ear` in its pipeline identity, in which case the divide is not the cost and the next probe is occupancy (`maxTotalThreadsPerThreadgroup` of the twin pipeline) and the operand cache-line census.
5. **Place the prefill K/V rows** (host row; also memory). Write the K even/odd and V outputs of the prefill plan straight into the device KV buffers, as the decode plan does (`decode.rs:6702-6726`), instead of reading back 98 MB and appending. Target: wall - `gpu_busy` 31-54 ms -> at most 15 ms, and the 98,205,696 pinned bytes leave the prefill arena.
6. **Prefill attention and router** (ranks 4 and 5): 27.7 -> 13 ms and 11.6 -> 3 ms own-cb, -24 ms together; trace before changing.
7. **Granite decode structure** (the 0.49-0.69 ms of `evaluate - gpu_busy`, the 0.67 ms of `gpu_busy` above llama, and attention at 51 us/op). First slice is instrumentation only: timestamps of the first dispatch start and last end against the host commit for each of the 8 chunks, to split the 0.5-0.7 ms; the attention row has one stated hypothesis to test,
   that proxima's decode example stores K and V as `F32` (`decode_gbps_baseline.rs`, `kv_cache_key_quant: GgmlType::F32`) while `llama-server` was launched without `-ctk`/`-ctv` (`decode_arms.rs:838-856`) and the upstream default is f16 (ASSUMED, not read from the recorded server log). Target: step 6.83 -> at most 6.0 ms.
8. **E2B decode norms and Q4_0 widths** (ranks 1-6). 211 measured norm dispatches at 10.9 us in situ against llama's 7.4 us own-cb: target -0.7 ms; Q4_0 1536x12288 at 38.3 us/op like llama: -0.37 ms; head: -0.12 ms (in-situ against own-cb, an unequal pair). Target: 11.06 -> at most 9.9 ms.

### what this section does not establish

- Rows 2-5 of the prefill table have a reference (llama's recorded class time) and no cause; only row 1 has a toggle that moves the number. The recovered milliseconds of rows 2-5 are the distance to llama, not a measured change.
- The omission runs exclude two q-norm groups (35 dispatches) and the 24 `moe_topk_stacked` dispatches the capture cannot replay; their cost is inside the family replays only.
- The omission base replay of granite decode (5.92 ms) is 0.7 ms above the census sequence replay of the same dispatches (5.14-5.20); the difference is unexplained.
- E2B decode and prefill wall figures are from the 16-token instrumented timeline and from the previous session's matrix; E2B was not re-benched here.
- Own-cb sums exceed `gpu_busy` (1.18x for granite prefill), so own-cb recoveries overstate wall time by about that factor.
- Efficiency percentages use 400 GB/s (ASSUMED, the spec sheet) and llama's recorded per-op rates; no ceiling for this GPU was measured in this pass (`proxima-tensor/docs/rooflines.md:411` records the GPU streaming bandwidth as not measured).
- The stalls above: cause not isolated.

### re-prove

```
attribution_rank rank --census evidence/attr2/census_granite_prefill --llama evidence/slice0/llama_ops/granite_ops.tsv --ntok 512,488 --requests 3 --floor-us 4.0
attribution_rank rank --census evidence/attr2/census_granite_decode --llama evidence/slice0/llama_ops/granite_ops.tsv --ntok 1 --requests 381 --floor-us 4.0
attribution_rank rank --census evidence/attr2/census_e2b_decode --llama evidence/slice0/llama_ops/e2b_ops.tsv --ntok 1 --requests 381 --floor-us 4.0
attribution_rank steps --events evidence/attr2/timeline/granite128_token_breakdown.log
M0_OUT_DIR=<dir> M0_MODEL_GGUF=<granite> M0_MAX_TOKENS=2 M0_CAPTURE_STEPS=0 PROXIMA_PROMPT_FILE=prompt1k.txt gemma4_decode_kernel_census      # granite prefill; decode: omit M0_MAX_TOKENS and M0_CAPTURE_STEPS
OMEGA_COOPERATIVE_REDUCE_MIN_LEN=9 cargo build --release -p proxima-model-interop --example gemma4_decode_kernel_census --features std,metal,instrument   # the toggle
PROXIMA_GEMMA4_E2B_GGUF=<blob> PROXIMA_PROMPT_FILE=prompt1k.txt AB_VARIANT_DIR=<empty dir> AB_OMIT_ALL=1 AB_PACKED=1 AB_STEP=5 AB_ROUNDS=30 AB_FLUSH_MIB=0 AB_SKIP_GROUPS='49f4a86451846a3b:[1, 8, 256];cdf3715b4db55454:[1, 8, 512]' norm_variant_ab   # E2B; granite: same with its blob and no skip list
cargo test -p proxima-model-interop --features std --example attribution_rank                                                              # the table generator's own tests
```

### the three omega `instrument` failures (fixed, 2026-10-08)

The three failures listed in the gates table of the memory section and in `evidence/combine/gates.md` were stale text expectations, not kernel defects. `classify_kind_packed_row_marker_tests::q4_0_two_token_index32_dispatch_classifies_as_packed_row_blocked`
asserted the spelling `q4_0_element(wblk0`; the two-token Q4_0 kernel now reads its weight through `device const uchar *wblk0 = in0 + ...` and per-row `blk0..blk3` pointers (printed from the emitted source), and `classify_kind` still returns
`reduce-packed-row-blocked` for it, so only the precondition string changed (`b1e102a0`). `rmsnorm_fused_epilogue_air_division_count_{decode,prefill}_shape` rewrote the emitted `%`/`/` coordinate pair into `full_coord[1] = r;` to count AIR division instructions; the emitter
now writes `full_coord[1] = r;` itself (the emitted source has no `remaining_r`), so the test restores the pair, compiles both forms to AIR, and asserts the emitted form has strictly fewer division-class instructions (`3fdc667d`).
After both: `cargo nextest run -p omega --features metal,instrument --cargo-profile gate --no-fail-fast` 826 run, 826 passed, 22 skipped (`evidence/attr2/omega_instrument_after_test_fixes.log`); clippy `-p omega --features metal,instrument --all-targets -D warnings` exit 0.

## moe combine (written, not yet run)

Written without a GPU, a model load, a test run or a bench; the numbers below are targets from the attribution section, not measurements of this tree.

1. Short folds by shape. `omega-runtime.toml` `[cooperative_reduce]` gains `serial_below_len` (32) and `serial_min_outputs` (65536); `msl::short_fold_prefers_serial` (`omega/src/msl/emit_and_classify.rs`) sends a fold shorter than the first with at least the second many outputs to the serial body, read through `reduce_is_cooperative` so dispatch geometry and emitted body agree. Prefill combine (8 long, 1,024,000 outputs) goes serial; decode combine (8 long, 1,024 outputs) and the 34/64-long attention folds keep the cooperative body. Both keys take 0 to disable and accept `OMEGA_COOPERATIVE_REDUCE_SERIAL_BELOW_LEN` / `OMEGA_COOPERATIVE_REDUCE_SERIAL_MIN_OUTPUTS`. Target: combine 4134 -> at most 600 us/op, granite prefill 381 -> at most 300 ms. Re-prove: the three censuses of the attribution section, `llama_parity_`, `generic_verify_llama_parity_`, a `decode_arms` run with a control arm.
2. Combine folded into the down projection. `MoeProjectionStrategy::StackedCombined` (feature `moe-stacked-combine`, default off, implies `moe-stacked-experts`) applies the routing weights to the hidden activation and reduces the selected and hidden axes of the slotwise down product together, so the `[tokens, selected, width]` output is not materialized. CPU: `run_reduce_quantized` (`proxima-tensor/src/cpu/run_reduce_scan.rs`) now separates reduced axes the route index moves along (selection axes) from the contraction one expert slab holds, runs one matvec per (position, selection) and adds the partial rows. Metal: such a fold is not cooperative (`gather_is_reduction_invariant`), is rejected by `classify_packed_row_block_with` and so renders `push_serial_reduce_body`, which fetches the route per reduction step.
   Not established: the Metal serial body for this fold is one thread per output walking `selected x width` elements (4096 per output at granite shapes); no kernel competing with the expert-grouped GEMM exists for it, so the default stays `Stacked` until a bench says otherwise. The earlier `GatherIndexOutOfRange` was not reproduced (no GPU run); `omega/tests/selection_fold_parity.rs` is the test that decides it.

## round two result (measured 2026-10-08, main 9015ad1a to the commit that adds this section)

Six written slices (combine, gemm, rope, kvplace, e2bdecode, granitehost; the "moe combine (written, not yet run)" section above is the first) applied to main in that order, tested once at the integrated tip, and benched once. Every number below names its source under `evidence/round2/` (`conflicts.md`, `gates/`, `bench/`). Test models: gemma4 E2B and granite moe 1b. gemma4 26B was not loaded. llama.cpp and Ollama were not run: the incumbent side is `evidence/slice0/ac1/decode_arms.out` (E2B 9.018 ms/token, prefill 573.3 ms; granite 5.210, 151.4; recorded 2026-10-07 on a different box state).
Nothing in this section is a verdict; the rows are measurements with the mechanism where one was traced, and the unexplained ones are listed last.

### commits (30 on top of 9015ad1a, no force, no trailer)

Slices as applied (`git am --3way`, 22 commits): combine `0cd6c7ef 29ef2def 6fd4a8c6`; gemm `f920c82d`; rope `7c042379 a14413c0 d946f343`; kvplace `748e3fa8 629ff850`; e2bdecode `51d6543b 230f755c efe85bdc`; granitehost `a38c17a9 59a59b84 a41b48e3 aef3dae3 3a24001c 746b1288 14f4dbab 73f0bc6a 3f536b1f 0db7f484`.
Integration fixes, one change each (`conflicts.md` items 5-11, 13): `7906a909` a comment named a model family (AC4 count 1 to 0); `94adb2da` the packed weight row width counted a broadcast axis (below); `3cf7e4ea` the q6k accumulator test follows the configured row count; `2b6370e3` the plain decode-split test asserts no half pointer instead of no substring "half"; `e38cc8c6` nextest exclusive override for the selection fold binary; `242cfe09` `OUTPUT_BUFFER_POOL` visibility (feature `metal-buffer-pool` did not compile at 9015ad1a); `256c9d23` the fold-route decision without a heap vector; `a4ceb193` the decode baseline example can turn speculation off.
Conflicts: 1 textual in `omega/build.rs` (rope and e2bdecode each append a sizing constant at the same place; both kept), 1 textual in `device_kv.rs` (`DeviceKv::adopt` doc and `#[allow]`; both kept), 1 compile interaction (a kvplace test calls `adopt` with 7 arguments, granitehost's takes 8) and 1 semantic interaction in `decode.rs` (kvplace adopts the device KV before prefill; granitehost's half-width cache is read only by the decode-split kernel; resolved as: an f32 cache adopts before the first evaluation, an f16 cache keeps `cached_len > 0 && is_last_step_batch`). No slice was aborted or reverted.

### gates at the integrated tip (`gates/`; N is the count the run printed)

| gate | command (tests: `--cargo-profile gate`) | N | source |
|---|---|---|---|
| tensor tests | `nextest run -p proxima-tensor` | 803 passed, 8 skipped | `final_nextest_tensor.log` |
| tensor, combine feature | `... --features moe-stacked-combine` (before the test added by `94adb2da`) | 803 passed, 8 skipped | `nextest_tensor_combine.log` |
| omega | `nextest run -p omega --features metal` | 809 passed, 16 skipped | `final_nextest_omega_metal.log` |
| omega instrument | `... --features metal,instrument` | 862 passed, 22 skipped | `final_nextest_omega_instrument.log` |
| omega feature-gated binaries | `... --features metal,metal-buffer-pool,metal-moe-mul-mat-id,moe-topk-fusion,top-fraction-fusion,alloc-count` | 830 passed, 16 skipped | `final_nextest_omega_gated_a.log` |
| omega split-k | `... --features metal,metal-q4k-split-k` | tip: 801 run, 793 passed, **8 failed**; 9015ad1a: 765 run, 758 passed, **7 failed** | `nextest_omega_gated_b.log`, `base_omega_gated_b.log` |
| interop slice-gate, 26B excluded | `nextest run -p proxima-model-interop --features std,metal --profile slice-gate -E 'not test(/gemma4_26b/)'` | 741 passed, 128 skipped | `final_nextest_interop_slice.log` |
| interop descriptor tests | `... --features std,metal,conflaguration` on `model_config_roundtrip_`, `serving_fsm_drives_`, `zero_rust_variant_`, `window_ring_layers_`, `swa_rope_from_metadata`, `generic_binder_`, E2B and granite only | 10 passed | `nextest_interop_conflaguration.log` |
| clippy `-D warnings --all-targets` | tensor+interop (std,metal), omega metal, omega metal+instrument, omega metal+buffer-pool+alloc-count, interop std,metal,instrument, the example; earlier also interop and tensor with `moe-stacked-combine` | exit 0 each | `clippy_final_*.log`, `clippy_*combine.log` |
| tiers | `check -p proxima-tensor --no-default-features --features alloc` (it builds none of the changed code: the CPU fold and the graph builder are std-side and omega is metal-gated), `-p proxima-model-interop --no-default-features`, `-p omega --no-default-features`, `--workspace --all-targets` | exit 0 each | `check_final_*.log`, `check_omega_nodefault.log` |
| AC4 | the `git grep ... \| wc -l` of the architecture-as-data table | 0 (1 before `7906a909`) | session |
| AC5 first command, AC11 grep | `git grep` counts | 0, 0 | session |

AC mapping inside the 741 and the 10: AC0 `arch_data_digest_` 7 passed (gemma4 E2B, openchat, qwen2, qwen3, qwen35, qwen35moe, granite moe; **the 26B row is not run**); AC2 `generic_verify_llama_parity_` E2B and granite, 2 passed; AC3 `generic_binder_` E2B and granite, 2 passed; AC6 `llama_parity_` E2B and granite, 2 passed; AC7 `window_ring_layers_`, 2 passed; `prefill_width_parity_with_llama_` E2B and granite, 2 passed; `serving_default_ubatch_prefill_parity`, 2 passed; `external_expert_paging`, 4 passed; AC5 second command 4 of 10 (E2B and granite round trips, the 2 FSM tests); AC9 2 of 6; AC10 (b) passed, (a) needs qwen2.
**Not run, 16 tests** (`gates/interop_complement_list.txt`: generic_binder, generic_verify and llama_parity for openchat/qwen2/qwen3, generic_binder for qwen35 and qwen35moe, llama_parity_lfm2, three lfm2 descriptor tests): the task names E2B and granite as the only test models, so the large-checkpoint end-of-run gate (727 passed in 607 s at the previous measurement) was not repeated; every 26B test was excluded.
Digests: the 7 digests above equal the vendored fixtures; none moved. Under the default-off feature `moe-stacked-combine` the granite digest moves from `bind.ops=5602` to `5578` (one slotwise-sum op per layer, 24, no longer materialized) and `model_config_roundtrip_granite_moe` fails with it (`nextest_interop_combine_feature.log`); that is the feature's graph, not the default's.

### failures the gate found, mechanism, fix (`conflicts.md` 5-11)

1. `selection_fold_parity`: 4 of 6 failed on the first run. Metal error 2.08e4, 4.96e4 and 4.77e5 relative to the row norm against CPU 2.7e-7, 5.3e-7 and 4.2e-7; the 120-token chunk raised `CommandBufferFailed`. Probes (`gates/probe_*.log`): the same bound op over f32 weights matched (2.74e-7); packed Q8_0 matched at selected=1 and at selected=2 returned `metal[1] = definition[2]`. The two emitted kernels differ by one line (`gates/fold_f32.metal`, `fold_q8_0.metal`), so the cause was in the uniforms. `native_packed_layout` (`proxima-tensor/src/bind/dead_code_cached_attention.rs`) sized a packed weight row as the product of every non-output axis, including `selected`, which the weight does not vary along, so rows were read `selected` rows apart. `94adb2da` leaves out axes on which the operand has stride 0. After: 6 of 6 (`nextest_selection_fold.log`), and a tensor unit test pins the layout. Mechanism evidence: the result goes from garbage to 2.7e-7 with that change alone, and the value pattern fits the stride.
2. Encode allocation budget (`alloc-count`): at 9015ad1a q4k/q6k/tiled/f32 = 287/287/290/154 against pins 297/258/254/179 (q6k and tiled already over); at `0cd6c7ef` 315/315/318/190, because `reduce_is_cooperative` called the heap-allocating `reduction_len` twice per query. After `256c9d23`: 231/231/234/82, all under the pins (`alloc_budget_after_fix2.log`; `bisect_alloc_*.log` give each step).
3. The text-expectation failures and the buffer-pool compile error are `conflicts.md` 7-9.
4. Not closed: 8 omega tests fail under `--features metal-q4k-split-k` (7 at 9015ad1a): `epilogue_operand_reuse::*` (2), `multi_row_kernel_folds_only_the_activation_rows_the_op_has`, `ported_matvec_bodies_unroll_their_row_and_lane_loops_fully`, `push_packed_row_blocked_body_emits_a_q4_0_row_blocked_kernel`, `q4_0_codec_takes_the_row_blocked_path_at_a_256_extent`, `q6k_single_token_matvec_folds_the_configured_rows_per_simdgroup`, `packed_row_rows_per_simdgroup_ab::default_env_emits_node_94_shape_for_diffing_against_the_capture`. They assert the unsplit single-token route, which split-k replaces (`(groups, split)` is `(4, 8)`, expected `(4, 1)`). The fix I tried, a `cfg(not(feature = "metal-q4k-split-k"))` on each, was refused by the permission system as a CI bypass and is not applied. `packed_row_q4_0_multi_row_hoist_ab::tokens27_k1536_rows12288_bit_exact_across_hoist` failed at 9015ad1a under split-k and passed at the tip; not explained.

### bench (release `decode_gbps_baseline`, `decode_arms`, 3 processes x (1 warm-up + 7 timed) = 21 timed runs per arm, arms interleaved per process)

Binaries (`bench/binaries.sha256`): `base` = 9015ad1a (sha256 `53baf7de...`), `tip` = 256c9d23 (`f23a1819...`), `control` = a byte copy of `tip` (same sha256). Prompts: `prompt1k.txt` (971 tokens E2B, 1000 granite, 128 new tokens), `prompt_short_hippo.txt` (25 tokens). Speculation is at the example's default (ngram-simple) in every arm; the `PROXIMA_SPECULATIVE_TYPES=none` that `decode_arms` exports is not read by the example. Ollama refused on :11434; no cargo or GPU peer during a timed run (`peers_present_at_exit=[]` on every process, `launches.log`). Box (`bench/run*/box_load_before.txt`): a private background daemon at 67-80% CPU, `mds_stores` 58-68%, `mediaanalysisd` 53-61% in every run; load average 4.8 at the launch of run A (the saved probe is 10.06 at 08:35, just after the builds; a recheck at 08:37 read 4.78 and is in the session only). GPU "Device Utilization %": two single probes read 98 and 95 (other processes); 18 repeated samples around them read 0. Medians are the driver's kept-run medians (outlier rule 3 x 1.4826 x MAD, fixed before the run); CoV is over all 21 runs, the kept CoV is in the file. Generated-text hashes are equal across `base`, `tip` and `control` on all 72 generations per model (`launches.raw.*.err`, `text_hash`).

Run A, `bench/runA_long/decode_arms.out` (144 raw lines = 6 arms x 24):

| arm | decode ms/token (kept median; CoV all, range all) | prefill ms (kept median; CoV all, range) | TTFT ms | peak RSS median of 3 | peak footprint | peak GPU bytes |
|---|---|---|---|---|---|---|
| E2B base | 11.029 (6.71%, 10.948-14.266; kept n=17) | 590.04 (1.56%, 587.95-625.95) | 590.0 | 3.958 GB | 505.1 MB | 3,657,105,408 |
| E2B tip | 11.1025 (4.34%, 11.061-13.359; kept n=18) | 592.02 (1.87%, 591.02-635.96) | 592.0 | 3.815 GB | 422.6 MB | 3,622,256,640 |
| E2B control | 11.094 (7.11%, 11.025-14.240) | 591.99 (0.94%) | 592.0 | 3.758 GB | 424.4 MB | 3,622,256,640 |
| granite base | 6.728 (1.11%, 6.616-6.898) | 372.54 (2.90%, 363.0-411.0; kept n=18) | 372.5 | 2.530 GB | 619.9 MB | 1,707,655,168 |
| granite tip | 6.521 (1.14%, 6.463-6.746) | 264.99 (0.37%, 263.96-267.98) | 265.0 | 2.100 GB | 367.5 MB | 1,728,069,632 |
| granite control | 6.576 (4.47%, 6.450-7.862) | 265.01 (0.88%) | 265.0 | 2.110 GB | 367.8 MB | 1,728,069,632 |

Bound lines (`bound metric=... arm=tip vs=base`; limit = max(MAD, 2% of base) for time, max(2% of base, |control - tip|) for memory): E2B decode +0.0735 (limit 0.2206, within); prefill +1.974 (limit 11.80, within); granite decode -0.207 (limit 0.1346); prefill -107.549 (limit 7.45); RSS -429.5 MB (limit 50.6); footprint -252.4 MB (limit 12.4); **GPU bytes +20.4 MB (limit 34.2, within; the sign is against the kvplace target of a smaller arena)**. control against tip: every metric within its limit (granite decode +0.055, limit 0.1304).

Run B, `bench/runB_short/decode_arms.out`, E2B, 25-token prompt: decode base 10.583 (CoV 10.76% all, 10.515-14.656; kept n=16), tip **10.833 (+0.250, limit 0.2117, within=false)**, control 10.844 (+0.261 against base, +0.011 against tip). Prefill 86.01 to 87.50 (+1.49, limit 1.72, within); control 87.97 (+1.96, limit 1.72, within=false against base). RSS 3.641 to 3.635 GB, footprint 208.8 to 204.8 MB, GPU bytes 3,384,639,488 to 3,389,947,904.
The E2B decode figures differ between runs by more than the within-run differences: `base` reads 11.029 (run A) and 11.277 (run C3) on the same binary; `tip` 11.1025 and 11.432; the short prompt's `base` 10.583 (B), 10.5775 (C1), 10.8805 (C2). Cause of the box drift not isolated; only within-run differences are used below.

### per-slice attribution

Boundary builds: `s1`..`s6` are the six slice tips (`6fd4a8c6 f920c82d d946f343 629ff850 efe85bdc 0db7f484`), `tip` the final tree; `off_*` are the tip rebuilt with the slice's build-time override (`OMEGA_*`, principle 12). Kept medians, ms; one interleaved run each.

| arm | E2B 971-token decode | E2B 971-token prefill | granite decode | granite prefill | source |
|---|---|---|---|---|---|
| base | 11.277 | 588.98 | 6.897 | 373.96 | `runC3_long_attr` |
| s1 combine | 11.3185 | 588.49 | 6.849 | 292.98 | |
| s2 gemm | 11.336 | 589.49 | 6.926 | 293.96 | |
| s3 rope | 11.312 | 587.97 | 6.858 | 291.97 | |
| s4 kvplace | 11.266 | 582.02 | 6.6695 | 265.98 | |
| s5 e2bdecode | 11.443 | 592.03 | 6.662 | 265.06 | |
| s6 granitehost | 11.411 | 592.00 | 6.6815 | 265.04 | |
| tip | 11.432 | 592.02 | 6.663 | 265.98 | |
| tip, `OMEGA_COOPERATIVE_REDUCE_SERIAL_MIN_OUTPUTS=0` | 11.410 | 591.97 | 6.6925 | **350.995** | |
| tip, `OMEGA_ELEMENTWISE_RECIPROCAL_MIN_ELEMENTS=4294967296` | 11.438 | 593.00 | 6.684 | 266.01 | |
| tip, `OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=256` | **11.250** | **586.00** | 6.675 | 265.98 | |
| tip, `OMEGA_COOPERATIVE_REDUCE_BROADCAST_SIMD_FOLD=0` | 11.462 | 591.98 | 6.723 | 266.02 | |
| tip, `OMEGA_PACKED_ROW_BLOCK_Q6K_ROWS=1` | 11.451 | 592.01 | 6.699 | 265.03 | |

E2B 25-token prompt, decode ms/token (`runC_short_attr`; then `runC2_short_e2b_switches`, whose box read 0.3 ms slower). C1: base 10.5775, s1 10.575, s2 10.563, s3 10.581, s4 10.589, **s5 10.816**, s6 10.8385, tip 10.8485, off combine 10.8685, off rope 10.856, off the three e2bdecode keys 10.623. C2: base 10.8805, s4 10.896, s5 11.158, tip 11.1085, off the three keys 10.904, off simd fold 11.143, **off broadcast width 10.926**, off q6k rows 11.123.

Rows, each a measurement with what moved it:

- combine, key `serial_min_outputs`: granite prefill 373.96 to 292.98 at `s1` (-81.0), and +85.0 when the key is 0 on the tip (350.995 against 265.98); decode -0.05 (inside CoV 1.0%). Mechanism: the toggle moves the number in both directions; the 1,024,000-output length-8 fold takes the serial body (attribution record: 4134 us/op cooperative, 567.7 us/op serial). Second part, default-off feature `moe-stacked-combine`: granite `llama_parity_granite_moe` ids equal llama in 159.9 s against 1.4 s on the default, `generic_verify` 319.5 s against 5.2 s, `prefill_width_parity` 255.2 s against 1.9 s (61x to 137x), and `serving_default_ubatch_prefill_parity` (971 tokens) stopped at the 60 s test body limit (`nextest_interop_combine_feature.log`). After `94adb2da` the Metal fold matches the definition and the CPU evaluator within 1e-4 (6 of 6); its cost is the serial body walking `selected x width` per output (the writer's prediction); the default stays `Stacked`.
- gemm (`scan_ahead`, no switch; the key's meaning changed): granite prefill 292.98 to 293.96 (+0.98; CoV 2.3% and 2.2%). The 203 to 115 ms target is a grouped-GEMM time; no census was taken in this pass, so the target is measured neither way. The wall effect is not distinguishable from zero.
- rope (key `reciprocal_min_elements`): `s3` against `s2`: granite prefill -1.98, E2B -1.53; the switch-off arm differs from the tip by +0.03 (granite) and +0.98 (E2B). The 35.2 to 3 ms per-op target is not measured here.
- kvplace (no switch): granite prefill 291.97 to 265.98 (-26.0), decode -0.19; E2B prefill 587.97 to 582.02 (-5.9), decode -0.05. Memory: granite footprint 611.6 to 367.2 MB (-244.4), RSS 2.484 to 1.878 GB; E2B footprint 498.3 to 423.1 MB (-75.2), RSS 3.871 to 3.765 GB; GPU bytes E2B -36 MB, **granite +20.4 MB**. The targets "no K/V readback during prefill, -98 MB arena, wall minus gpu_busy at most 15 ms": the footprint moved by -244 MB and -75 MB; the arena size, the readback and the wall-minus-gpu_busy figure were not instrumented.
- e2bdecode: E2B decode **+0.227 (short) and +0.177 (long) at `s5` against `s4`, long prefill +10.0**; granite decode -0.008, prefill -0.93. Inside the slice, `OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=256` returns the E2B numbers: short decode 11.1085 to 10.926 (-0.18, C2), long decode 11.432 to 11.250 (-0.18), long prefill 592.02 to 586.00 (-6.0), and with all three keys the short decode goes 10.8485 to 10.623 (C1); granite is unchanged (6.663 to 6.675). The simd-fold key and the q6k row key move no E2B or granite number by more than 0.04 ms. The unroll commit has no switch and stays in every `off_*` arm. Target (11.06 to 9.9 ms, -1.16): not reached; the slice moved E2B decode the other way, and the cause of the width cap's cost is not traced.
- granitehost (default f32, no change expected): `s6` against `s5`: E2B decode -0.03, granite +0.02 (inside noise). Switch `PROXIMA_KV_CACHE_TYPE=f16` (needs speculation off, `a4ceb193`; `runD3_f16`, speculation off in both arms, equal text hashes): E2B decode 11.4305 to **11.970 (+0.54)**, prefill 591.99 to 598.98 (+7.0), footprint 416.2 to 482.3 MB (+66), RSS +113 MB; granite decode 6.687 to **7.5905 (+0.90)**, prefill 265.03 to 288.00 (+23.0), footprint 360.0 to 594.7 MB (+235), RSS +272 MB; GPU bytes +6.6 MB and +22.8 MB. The target 6.83 to 6.0 ms has the opposite sign. Mechanism, instrumented run (`runE_f16_mechanism`, one process, debug events on): host narrowing 0.027 ms per step (`device_kv_narrow` steps 2-3; 0.110 at step 1); decode `gpu_exec_ms` at step 20 is 5.207 (f32) against 5.517 (f16), +0.31; host-unhidden time per decode step is equal (median 0.379 and 0.382). That accounts for about 0.3 of the 0.9 ms; the rest is unexplained. The first f16 attempt (`runD_f16`) used an unqualified `--arm-env` label, applied no environment, and is two same-binary arms (granite 6.725 against 6.674, E2B 11.4575 against 11.443).

### frequency-weighted read

Granite decode and prefill: tip is below base on both (-0.207 ms/token, -107.5 ms prefill), with RSS down 429 MB and footprint down 252 MB. E2B decode is the 100%-frequency path of the owner's target and it is above base by 0.07 (long, run A), 0.155 (long, run C3) and 0.250 (short, run B) ms/token; the width cap accounts for 0.18 of that at both prompt lengths.

### unexplained, unmeasured, assumed

- Why a 512-lane broadcast-epilogue threadgroup costs E2B 0.18 ms/token and 6 ms of prefill and changes nothing on granite: not traced (no census or AIR read in this pass). The toggle is the evidence.
- The f16 path's remaining ~0.6 ms/token on granite, and its +23 ms prefill and +235 MB footprint (host-side seeding and the staging buffers are the candidates; unmeasured).
- Run-to-run drift of the E2B decode figures (0.25-0.33 ms between runs on the same binary).
- Per-op targets of gemm (203 to 115 ms), rope (35.2 to 3 ms) and kvplace (arena -98 MB, readback, 15 ms): not instrumented here.
- The `StackedCombined` slowdown is a wall figure over whole tests, not a per-dispatch census.
- CUDA and WGSL `reduce_is_cooperative` do not read the new serial-fold keys (the writer's note); not touched. 16 large-checkpoint tests and every 26B test were not run.
- GPU contention can fail a command buffer (`0000000e`) in the long selection-fold dispatch (1 of 3 full-suite omega runs before `e38cc8c6`); not seen alone in 6 of 6 runs.
- Decisions that belong to the owner and are not taken here: the default of `broadcast_max_width` (512 against the previous 256), whether the f16 cache stays in tree with these numbers, and the `moe-stacked-combine` feature.

### re-prove

```
cargo nextest run -p proxima-tensor --cargo-profile gate                                                     # 803
cargo nextest run -p omega --features metal --cargo-profile gate                                              # 809
cargo nextest run -p omega --features metal,instrument --cargo-profile gate                                   # 862
cargo nextest run -p omega --features metal,metal-buffer-pool,metal-moe-mul-mat-id,moe-topk-fusion,top-fraction-fusion,alloc-count --cargo-profile gate   # 830
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate -E 'not test(/gemma4_26b/)'   # 741
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm base=<9015ad1a binary> --arm tip=<256c9d23 binary> --arm control=<copy of tip> --case gemma4_e2b=<E2B blob> --case granite_moe=<granite blob>
decode_arms --prompt-file prompt_short_hippo.txt ... --case gemma4_e2b=<E2B blob>
OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=256 cargo build --release -p proxima-model-interop --example decode_gbps_baseline --features std,metal   # the width-cap toggle
decode_arms ... --arm nospec=<tipx> --arm nospec_f16=<tipx> --arm-env gemma4_e2b.nospec:PROXIMA_DECODE_SPECULATIVE=none --arm-env gemma4_e2b.nospec_f16:PROXIMA_DECODE_SPECULATIVE=none --arm-env gemma4_e2b.nospec_f16:PROXIMA_KV_CACHE_TYPE=f16   # arm-env labels are case-qualified
```

Missing for CI: no job runs the Metal tests, a GPU bench or `decode_arms`; the nextest runs and the bench numbers re-prove only on this box. The bench binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/combine2/bin/` (sha256 in `bench/binaries.sha256`), not in the tree.

## round two follow-up (measured 2026-10-08, main f6a85722; commits `356e44e4`, `bbdbef55` and the one adding this section)

Two items from the round two result: the E2B decode regression traced to the broadcast width cap, and the 8 omega tests that failed under `--features metal-q4k-split-k`. Test models: gemma4 E2B and granite moe 1b; Ollama was not running (a process listing for ollama, cargo, rustc, nextest and decode_gbps printed only `sccache` before run A; session only) and no cargo or GPU peer ran during a timed run (the driver gates every launch on its peer list). Evidence is `evidence/round2fix/` (`bench/`, `census/`, `kernel/`, `gates/`). Nothing here is a verdict.

### 1. broadcast width cap back to 256 (`bbdbef55`, `omega/omega-runtime.toml` `[wide_cooperative_reduce] broadcast_max_width`)

The default went from 512 to 256; the key and its build-time override are unchanged, so `OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=512` rebuilds the previous shape. Arms: `base` = the working tree built with that override at 512 (the codegen of f6a85722; the commit between them changes only `cfg(test)` code and the toml, and the example binary has no other input), `tip` = 256, `control` = a byte copy of `tip` (same sha256, `bench/binaries.sha256`). `decode_arms`, 3 processes x (1 warm-up + 7 timed) = 21 timed runs per arm, arms interleaved per process, run twice per prompt with the arm order rotated (base,tip,control; tip,control,base). Medians are the driver's kept-run medians (outlier rule 3 x 1.4826 x MAD, fixed before the run); the CoV and range are over all 21 runs, because five arms exceed 5% over all 21 on a few spikes (the max column) and are reported as ranges. Box: load average 4.2 to 5.9 at launch, a private background daemon 55-70% CPU and `suggestd` 70% in every run (`box_load_before.txt`); GPU utilization was not sampled this pass.

Decode ms/token (kept median; CoV all, range all):

| prompt | order | base (512) | tip (256) | control (copy of tip) | tip - base | limit | control - tip |
|---|---|---|---|---|---|---|---|
| E2B 971 tokens | base,tip,control | 11.366 (2.68%, 11.012-12.103) | 10.9565 (5.22%, 10.796-13.646) | 11.229 (3.91%, 10.814-12.794) | **-0.4095** | 0.2273 | **+0.2725, limit 0.2191, within=false** |
| E2B 971 tokens | tip,control,base | 11.138 (0.81%, 11.051-11.413) | 10.9825 (5.49%, 10.847-13.841) | 10.9535 (5.73%, 10.888-13.447) | -0.1555 | 0.2197 | -0.029 |
| E2B 25 tokens | base,tip,control | 10.838 (4.82%, 10.797-12.834) | 10.6715 (6.49%, 10.615-13.737) | 10.6675 (7.46%, 10.602-14.465) | -0.1665 | 0.2168 | -0.004 |
| E2B 25 tokens | tip,control,base | 10.897 (6.79%, 10.834-13.933) | 10.6415 (4.96%, 10.596-13.151) | 10.654 (0.24%, 10.587-10.715) | **-0.2555, limit 0.2128, within=false** | 0.2128 | +0.0125 |
| granite 1000 tokens | base,tip,control | 6.640 (1.77%, 6.451-6.809) | 6.521 (1.37%, 6.423-6.745) | 6.579 (1.53%, 6.450-6.835) | -0.119 | 0.1328 | +0.058 |
| granite 1000 tokens | tip,control,base | 6.522 (2.44%, 6.442-7.225) | 6.5495 (2.54%, 6.470-7.267) | 6.541 (0.78%, 6.459-6.639) | +0.0275 | 0.1310 | -0.0085 |

Prefill ms (kept median): E2B 971 tokens base 591.989 / 592.0085, tip 585.993 / 585.975 (tip - base -5.996 and -6.034; control 586.024 and 586.023); E2B 25 tokens base 87.998 / 87.985, tip 87.028 / 87.503 (-0.970 and -0.482, limits 1.76 and 1.75); granite base 265.952 / 265.045, tip 265.015 / 265.504 (-0.937 and +0.459, limit about 5.3). Peak RSS, footprint and GPU bytes moved by less than their limits in the one order that printed them (`runA_long`: E2B RSS +15.4 MB, footprint +0.5 MB, GPU +0.5 MB; granite +4.3 MB, +0.9 MB, 0 B). Generated-text hashes: one hash per model across all 72 generations of every arm in each of the four runs (`launches.raw.*.err`, `text_hash`), so 256 and 512 produce the same text on these prompts.

What the table supports: tip is below base on E2B decode in 4 of 4 runs (-0.4095, -0.1555, -0.1665, -0.2555), and the 971-token prefill by 6.0 ms in both orders with the copy of tip agreeing to 0.03 ms. Two runs breach the control bound (E2B 971 first order, control above tip by 0.2725; E2B 25 tokens rotated, base above tip by 0.2555 against a 0.2128 limit), so the size of the decode delta is uncertain by about 0.25 ms between copies of one binary in the worst run; the sign is stable and the magnitude lies between -0.16 and -0.41. Granite decode: tip - base is -0.119 and +0.0275 against limits 0.131-0.133, both within, with the sign changing. Mechanism for granite: its hidden width is 1024 (`embedding_length` u32 at the blob's header, bytes `00 04 00 00`), `quarter_width` gives 1024/4 = 256 lanes, which both caps leave alone (`tiled_gemm_cooperative_scan.rs:1926-1941`), so the broadcast-reduce kernels of granite's hidden-width norms are the same at 256 and 512 (derived from that formula; not diffed for granite), and its delta is the run-to-run spread.

### mechanism: where the 0.18 ms goes (traced to the kernel group; the last link is not traced)

1. Kernel source: the hidden-width norm kernel for a 1536-wide row emitted at cap 256 and at 512 differs in constants only (`kernel/kernel_cap256.metal` against `kernel_cap512.metal`, `diff` is 15 hunks, 30 changed lines): `gid / 256` against `gid / 384`, `lane = gid % N`, `slot * N`, the walk stride `8N` (2048 against 3072), `partials[8]` against `partials[12]`, and the second-level `simd_sum((lane % 32u) < 8u ? ...)` against `< 12u`. 384 lanes is `quarter_width(1536) = 1536/4` held to the cap; 256 lanes is the cap.
2. Census (`gemma4_decode_kernel_census`, the 25-token prompt, last decode step, 653 dispatches, groups replayed alone in a cold command buffer; 4 runs per cap interleaved 512,256, `census/census_{512,256}_r{1..4}.out`): the class "RMSNorm sumsq + fused epilogue" (242 dispatches per step) reads 12.63 us per dispatch cold at 512 (12.63, 12.68, 12.54, 12.67) against 11.81 at 256 (11.65, 11.83, 11.86, 11.91); the norms family replayed in one command buffer (242 dispatches, 7 runs each) reads gpu_ms_mean 2.5955 at 512 against 2.3088 at 256 (+0.287 ms). The Q4_0 matvec and head rows sit inside their run-to-run spread between the two (Q4_0 matvec 18.60-18.83 us per dispatch at 512 against 18.59-19.19 at 256; head 1140.7-1142.1 against 1142.4-1145.7); the other classes were not tabulated.
3. Group level (`census_groups.csv`): the groups whose threadgroup width changes are the three hidden-width groups (71, 70 and 35 dispatches per step, grid = threadgroup width = 384 at 512, 256 at 256); every other norm group has an identical width in both builds and a cold time within about 0.5 us in run pair 1 (`census_512_r1` against `census_256_r1`). Cold ns, mean of 3 interleaved runs per cap (`census/sweep_<cap>_r{1..3}.groups.csv`), width 256 then 384: group of 71 dispatches 11681 then 12361; of 70, 12583 then 13917; of 35, 12514 then 13410. Weighted by dispatch count this is +173 us per step cold (derived from those means; the end-to-end figure above is 0.16-0.41 ms per token, measured).
4. Width sweep on the same three groups (cap 128, 192, 256, 320, 512, 3 runs each; widths 128, 192, 256, 320, 384), cold ns for the 70-dispatch group: 19035, 14465, 12583, 13785, 13917. Per-lane element count falls monotonically with width (12, 8, 6, 5, 4 of the 1536 elements per lane), yet the time is minimal at 256 and rises by about 1.2 us at 320 and 384. The other two groups show the same shape (71-dispatch group 16326, 12292, 11681, 12215, 12361; 35-dispatch group 18660, 14049, 12514, 13681, 13410).
5. Not traced: why a wider group is slower once the per-lane element count is already down to 5 or 4. Three candidates are not separated by any run here: the cross-simdgroup combine growing from 8 to 10 or 12 partials behind the same barrier, the integer divide and modulo by a non-power-of-two (`gid / 384`, `gid % 384u`; 320 and 192 are non-powers of two as well, and 192 sits on the falling side), and threadgroup residency of a 320 or 384-thread group on one core. Separating them needs an ISA read or a counter capture of the 384-lane kernel, or a kernel variant that holds the divisor a power of two while keeping 12 simdgroups; none was done.

The cap's own doc in the toml now records that 512 costs 0.18 ms/token and 6 ms of prefill on E2B; 1024 stays unmeasured.

### 2. the 8 omega tests under `--features metal-q4k-split-k` (`356e44e4`)

Before: `nextest run -p omega --features metal,metal-q4k-split-k` on the 8 names printed 15 run, 7 passed, 8 failed (`gates/splitk_8_before.log`). The route the classifier selects when the feature is on is the split-aware body: `packed_row_dispatch` returns `(groups, split)` with `split = target_simdgroups(2048) / base_simdgroups` held to `max_split(8)` and to `split_k_max_rows(4096)`, `use_q4_0_native` and the ggml-port bodies require split-k off (`packed_row_blocked_ggml.rs:176,204,219`), the `ib` walk is strided by `sgitg` and `split`, and the tail is `push_packed_row_combine_and_write`'s split arm: `partial_sums` in threadgroup memory, a barrier, simdgroup 0 lane 0 adds the partials. Each test now asserts that route when the feature is on and its previous expectation when it is off; none was skipped, no assertion was removed:

| test | feature off (unchanged) | feature on (new) |
|---|---|---|
| `q6k_single_token_matvec_folds_the_configured_rows_per_simdgroup` | dispatch `(groups, 1)` | `(groups, split)` with split from the build-time keys (4, 8 at 8 rows), plus the partial/barrier/combine tail in the source |
| `multi_row_kernel_folds_only_the_activation_rows_the_op_has` | threads = simdgroups x token groups x 32 | the same times `split` (16384 for 2 tokens, 64 simdgroups, split 8) and `threadgroup_width == 32 x split` |
| `q4_0_codec_takes_the_row_blocked_path_at_a_256_extent`, `push_packed_row_blocked_body_emits_a_q4_0_row_blocked_kernel`, `default_env_emits_node_94_shape_for_diffing_against_the_capture` | native inline dot (`sumy * -8.0f`), no `q4_0_pair_dot` | `q4_0_pair_dot(blk` and no `sumy * -8.0f`; the combine tail present (node 94's shape: `ib_first` strided by `sgitg`, `partial_sums`, the sgitg-0 lane-0 combine); off: none of `partial_sums`, `sgitg`, `split` in the source |
| `ported_matvec_bodies_unroll_their_row_and_lane_loops_fully` | the unroll pragmas on the row loop, the block-pointer setup and advance | neither ported body renders (`sumy * -8.0f`, `int ib_step = 2;` absent), `ib_first`/`ib_step` strided by `sgitg`/`split`, the row loop still opens one line above `blk_ptr[q]`, the block pointers set up and advanced, the combine tail present |
| `epilogue_operand_reuse::a_fused_epilogue_finishes_each_row_of_the_simdgroup_on_its_own_lane` | `if (lane < 4u)`, `reduced_row3 = simd_sum(sumf[3])`, no lane-0 row loop | each row's `simd_sum` parked in `partial_sums[q][sgitg]`, the combine after the barrier on simdgroup 0 lane 0, `epi_scratch[3] = total` feeding the fused epilogue, no `if (lane < 4u)` |
| `epilogue_operand_reuse::a_plain_matvec_finishes_each_row_of_the_simdgroup_on_its_own_lane` | `if (lane < 4u)`, one-term store | the combine tail, the store at the cached coordinate (`out[out_offset] = total;`), no `if (lane < 4u)` and no per-row lane-0 test |

Not asserted for the feature-on route: that the generic body carries the unroll pragma the ported bodies have; it does not (`splitk_8_before.log` prints the kernel), and pinning its absence would pin a missing optimization as expected. Observed by reading, not executed: under split-k a multi-token op (2 to 7 tokens, `push_packed_row_multi_row_body`) is dispatched `split` simdgroups per group (the 16384 above) while that body reads no `sgitg`, so each simdgroup of a group appears to compute and store the same rows; the cost and whether it matters for any model path are unmeasured, and `metal-q4k-split-k` is not in the `metal` feature list.

### gates at the tip (`gates/`; N is the count the run printed)

| gate | command | N | source |
|---|---|---|---|
| clippy `-D warnings --all-targets` | tensor + interop (`std,metal`) | exit 0 | `gate_clippy.log` |
| clippy | omega `metal`; omega `metal,metal-q4k-split-k` | exit 0 each | `clippy_omega_metal.log`, `clippy_omega_splitk.log` |
| tiers | `check -p proxima-tensor --no-default-features --features alloc` (builds none of the changed code); `check -p proxima-model-interop --no-default-features` | exit 0 each | `gate_check_alloc.log`, `gate_check_interop_nd.log` |
| tensor | `nextest run -p proxima-tensor --cargo-profile gate` | 803 passed, 8 skipped | `gate_nextest_tensor.log` |
| omega metal | `nextest run -p omega --features metal --cargo-profile gate` | 809 passed, 16 skipped | `nextest_omega_metal.log` |
| omega split-k | `... --features metal,metal-q4k-split-k` | run 1: 801 run, 800 passed, **1 failed** (`packed_row_multi_row_unroll_ab::q8_0_tokens600_k12288_rows1536`: `CommandBufferFailed` `0000000e`); run 2: 801 passed, 16 skipped; the failed test alone: 2 run, 2 passed | `nextest_omega_splitk.log`, `nextest_omega_splitk_run2.log`, `rerun_q8_alone.log` |
| interop slice-gate, 26B excluded | `nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate -E 'not test(/gemma4_26b/)'` | 741 passed, 128 skipped | `gate_nextest_interop_slice.log` |
| parity, E2B and granite | `llama_parity_`, `generic_verify_llama_parity_`, `prefill_width_parity_with_llama_` for `gemma4_e2b` and `granite_moe` | 6 of 6 passed | `gate_parity.log` |

The one split-k failure is the `0000000e` command-buffer failure the round two section recorded under GPU contention (a heavy 600-token Q8_0 dispatch while other omega tests run); load average was 8.9 at the rerun. It passed alone and in the second full run, and its mechanism was not isolated here. Before this pass split-k read 801 run, 793 passed, 8 failed; it now reads 801 run, 801 passed in one of two full runs.

### re-prove

```
cargo nextest run -p omega --features metal --cargo-profile gate                                   # 809
cargo nextest run -p omega --features metal,metal-q4k-split-k --cargo-profile gate                 # 801
OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=512 cargo build --release -p proxima-model-interop --example decode_gbps_baseline --features std,metal   # base arm
cargo build --release -p proxima-model-interop --example decode_gbps_baseline --features std,metal                                                          # tip arm
decode_arms --prompt-file prompt1k.txt --processes 3 --runs 7 --arm base=<512 binary> --arm tip=<256 binary> --arm control=<copy of tip> --case gemma4_e2b=<E2B blob> --case granite_moe=<granite blob>
OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=<128|192|320|512> cargo build --release -p proxima-model-interop --features std,metal,instrument,metal-fuse-attn-decode --example gemma4_decode_kernel_census
```

Missing for CI: no job runs the Metal tests, `decode_arms` or the census; they re-prove only on this box. The binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/round2fix/bin/` (sha256 in `evidence/round2fix/bench/binaries.sha256`).

## round three attribution (measured 2026-10-08 on `56d7d21d`, no library code changed)

Evidence root: `evidence/attr3/` (`rank/` the `attribution_rank` tables, `census/` the group CSVs, dispatch CSVs, box files and the step/chunk event lines of four censuses, `omission/` the captured-step omission runs and their summaries, `timeline/` the per-step host/GPU timelines, `probes/` the kernel-variant experiments with their emitted sources, `kv_f16_probe/`, `tools/` the two Rust summarizers, `bin.sha256`). Raw logs and the four release binaries are under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/attr3/` (the 21-33 MB `decode_telemetry.log` files stay there; `census/*/telemetry_events.log` is the `chunk_record` and `token_breakdown` subset the timeline tool reads, and reproduces `timeline/granite_decode.md` apart from the header line).
Test models: gemma4 E2B (`gemma4:e2b-it-qat`, blob `sha256-3646b4c1...`) and granite 3.1 moe 1b (blob `sha256-cd60b3e8...`). Ollama was not running; llama.cpp was not run: the llama side of every comparison is `evidence/slice0/llama_ops/*.tsv` and `evidence/slice0/ac1/decode_arms.out`, recorded 2026-10-07 on a different box state.
Bases (as in the attribution section): **own-cb** = median GPU span of one dispatch alone in its own command buffer (about 4 us of floor on both sides), **in-situ** = the captured step replayed in one command buffer with the group removed (`norm_variant_ab`, `AB_OMIT_ALL`), **live** = the step as the decode loop runs it. DERIVED marks arithmetic on measured numbers.
Box: load average 3.3 to 6.7 at launches, a background daemon 40-80% CPU, `mds_stores`/`mediaanalysisd`/`WindowServer` 30-130% CPU in the per-run `box*.txt` files. GPU "Device Utilization %" read 0 before every timed run except where a sample taken within a second of the previous process exiting read 93-99 (`kv_f16_probe/box_*.txt` all read 93-95 for that reason; the runs before them were back to back); the settled samples after each read 0. The census binaries were built at `56d7d21d`, sha256 in `evidence/attr3/bin.sha256` (all four matched before reuse).
Nothing in this section is a verdict.

### 1. ranked class tables (own-cb; llama per request from the recordings)

`evidence/attr3/rank/granite_prefill.md` (359 timed dispatches, sum 285.10 ms against llama 157.56 ms; live GPU span of the same step 258.5-260.3 ms, `timeline/`):

| class | ours ops / ms | llama ops / ms | gap ms | ratio |
|---|---|---|---|---|
| matmul (weights) | 218 / 218.41 | 381 / 131.89 | +86.52 | 1.66x |
| elementwise, copy, other | 65 / 36.19 | 383 / 9.27 | +26.92 | 3.91x |
| attention core | 27 / 27.59 | 95 / 12.93 | +14.67 | 2.13x |
| rms norm | 49 / 2.90 | 96 / 1.77 | +1.13 | 1.64x |
| moe routing | 24 / 0.00 (not replayable) | 0 | 0 | n/a |
| rope | 0 / 0.00 | 96 / 1.71 | -1.71 | 0.00x (the work sits in the 36.19 row as two twin kernels) |

Within matmul: `Q8_0 4194304` 72 ops 169.99 ms (llama has no op of that shape; its stacked expert ops are inside `Q8_0 524288`), `Q8_0 524288` 48 ops 7.97 ms against llama 237 ops 114.41 ms (-106.44), `F32 8192` 24 ops 13.63 ms, `F32 32768 (1024x32)` 24 ops 11.52 ms against 47 ops 2.17 ms (+9.35, 5.30x), `Q8_0 1048576` 48 ops 15.13 ms against 96 ops 15.08 ms (1.00x), head `Q8_0 50334720` 0.16 against 0.22 ms.

`evidence/attr3/rank/granite_decode.md` (374 timed dispatches, 6.28 ms against llama 6.49 ms):

| class | ours ops / ms | llama ops / ms | gap ms | ratio |
|---|---|---|---|---|
| attention core | 51 / 1.38 | 48 / 0.73 | +0.66 | 1.90x |
| rms norm | 49 / 0.56 | 49 / 0.37 | +0.19 | 1.52x |
| matmul (weights) | 218 / 3.87 | 193 / 3.99 | -0.12 | 0.97x |
| elementwise, copy, other | 56 / 0.47 | 196 / 1.09 | -0.62 | 0.43x |
| rope | 0 / 0.00 | 48 / 0.32 | -0.32 | 0.00x |
| moe routing | 24 / 0.00 (not replayable) | 0 | 0 | n/a |

`evidence/attr3/rank/e2b_decode.md` (653 timed dispatches, 11.32 ms against llama 11.10 ms):

| class | ours ops / ms | llama ops / ms | gap ms | ratio |
|---|---|---|---|---|
| rms norm | 242 / 2.92 | 242 / 1.80 | +1.12 | 1.62x |
| output head | 1 / 1.08 | 1 / 0.94 | +0.14 | 1.15x |
| attention core | 73 / 1.29 | 35 / 1.21 | +0.08 | 1.07x |
| matmul (weights) | 277 / 5.29 | 276 / 5.30 | -0.01 | 1.00x |
| rope | 0 / 0.00 | 50 / 0.32 | -0.32 | 0.00x |
| elementwise, copy, other | 60 / 0.74 | 214 / 1.52 | -0.79 | 0.48x |

### 2. omission, in-situ cost per group (`evidence/attr3/omission/summary_*.md`; median of the runs, range and per-dispatch us in the files)

Granite decode, 3 runs (`granite_decode_r1..r3.out`), base replay median 5.576 ms, sum of medians 5.571 ms: attention partial `[1,8,2,64]` 24 ops 1.211 ms (50.5 us/op); expert down `[1,8,512,1024]` 0.765 ms (31.9 us); expert up+silu 0.727 ms (30.3 us, one of three runs at 1.015); expert gate 0.552 ms (23.0 us); rms norm epi4 `[1,1024]` 49 ops 0.489 ms (10.0 us); K/V `[1,8,64,1024]` 48 ops 0.365 ms (7.6 us); attention out 0.261 ms; combine fold 0.254 ms (10.6 us); attention merge 0.162 ms (6.8 us); the two rope twins 0.107 + 0.099 ms (4.5 and 4.1 us). 15 of the 26 rows have a range wider than their median (all of them rows under 0.3 ms; E2B 18 of 45, prefill 6 of 20, counted from the `cost range ms` column); rows under about 0.2 ms are unresolved at the run spread.
E2B decode, runs r2-r4 complete (45 groups, 2 groups skipped by `AB_SKIP_GROUPS`: `a638c75f...:[1, 8, 256]` 28 dispatches and `2417697c...:[1, 8, 512]` 7 dispatches, the q-norm groups; the hashes changed with the build, the old ones were `49f4a864`/`cdf3715b`), r1 stopped after 16 groups with an empty `.err` and no exit record (the previous session ended; the 16 lines are included in the medians, `runs` column = 4 for them), base replay median 10.539 ms: head `[1,1536,262144]` 1.102 ms; `ffn_down` Q4_0 K=12288 20 ops 0.944 ms (47.2 us); ffn_gate + gelu 0.881 ms (44.0 us); rms norm epi5 `[1,1536]` 70 ops 0.834 ms (11.9 us); ffn_up 0.808 ms (40.4 us); sliding attention partial 28 ops 0.721 ms (25.8 us); rms norm epi4 71 ops 0.686 ms (9.7 us); epi6 35 ops 0.391 ms (11.2 us).
Granite prefill (not asked; run to separate replay artifacts from cost), 2 runs `granite_prefill_r1/r2.out` at `AB_STEP=0`, `AB_ROUNDS=15`, base 253.3 ms: up+silu 58.76 ms (2448 us/op), down 58.04 (2418), gate 54.14 (2256), attention partial 27.23 (1135), combine fold 13.51 (563), router `[1000,1024,32]` 12.28 (512), attention out 8.73, K/V 7.97, Q 7.06, **Q twin 0.936 ms (39 us/op, range 0.787-1.086), K twin 0.562 ms (23 us/op)**, rms norm 49 ops **-7.86 ms** (a negative cost: removing the group made the replay slower, range -8.24 to -7.49; unexplained, see section 7).

### 3. timelines (`evidence/attr3/timeline/*.md`, from `chunk_record` and `token_breakdown` events)

The live log carries per-chunk (one command buffer) GPU start and end, the encode window and the commit instant; **it carries no per-dispatch timestamps**. The finest emitted unit is the chunk (`omega/src/metal/placements_execute_named.rs:1772-1813`, `chunk_record`); a dispatch-level timeline would need a per-dispatch counter sample, which only the stage-boundary path has (`omega/src/metal/dispatch_timed_and_classify.rs:698`, documented to inflate the step 50-100x). The "dispatch before / after" of each gap below is therefore the last dispatch of the previous chunk and the first of the next, from `census_dispatches.csv` `chunk_index`.

| case | wall | evaluate | chunk busy sum | lead idle | inter-chunk idle | last gpu end | residual after it |
|---|---|---|---|---|---|---|---|
| granite decode, median of 22 steps (steps 2-23) | 6.468 | 6.398 | 5.375 | 0.341 | 0.372 | 6.108 | 0.208 |
| E2B decode, median of 22 steps | 11.347 | 11.290 | 9.896 | 0.286 | 0.713 | 10.901 | 0.285 |
| granite prefill 1000 tokens, step 0 of the decode census (`census_granite_decode`) | 399.854 | 389.197 | 258.514 | 62.043 | 0 (one chunk) | 320.557 | 0.785 |
| same step, cold census (`census_granite_prefill`, 33 pipeline misses, 455.1 ms compile) | 836.776 | 826.572 | 260.333 | 65.763 | 0 | 326.096 | 0.872 |

(ms; "lead idle" is the time from the start of encoding to the first GPU start; "residual" is evaluate minus prepare, pre_encode and the last GPU end.)
Largest gaps, census step 23 (`timeline/granite_decode.md`, `timeline/e2b_decode.md`):
- granite decode: 0.409 ms before chunk 1 (host: commit at 0.241 ms, GPU start 0.409); 0.240 ms residual; **0.212 ms between #20 `matvec Q8_0 [1,8,64,1024]` (chunk 1) and #21 the K rope twin (chunk 2)**; the other six chunk boundaries 0.026-0.035 ms each.
- E2B decode: **0.552 ms between #15 cached attention merge `[1,1,8,256]` and #16 `matvec Q4_0 [1,1,8,256,1536]`**; 0.284 ms residual; 0.283 ms before chunk 1; the other six boundaries 0.028-0.034 ms.
- Mechanism of the chunk-1 to chunk-2 gap (read from the same records): the first chunk is 24 ops (`first_chunk_ops = 24`, `omega/omega-runtime.toml:289`); its GPU busy time is 0.273 ms (granite) and 0.262 ms (E2B), while encoding chunk 2 takes 0.488 ms (51 ops) and 0.823 ms (116 ops) and commits at 0.743 and 0.977 ms, after chunk 1 ended at 0.682 and 0.545 ms. Every later chunk's GPU time (0.7-2.3 ms) exceeds the next chunk's encode time (0.48-0.83 ms), so the later boundaries are 0.03 ms.
- Prefill: 141.3 ms of the 399.9 ms step is outside the GPU busy time: lead idle 62.0 (of which commit to GPU start 58.2 ms; steady-state decode shows 0.17 ms for the same field), prepare 38.4, pre_encode 29.4 (pipeline compile 24.6 ms for 33 misses). Why the GPU starts 58.2 ms after the commit: untraced.

### 4. the two round-two targets that missed

**Rope twin kernels at granite prefill (35.75 ms own-cb, target 35 -> 3).** The two own-cb rows (`census_granite_prefill/census_groups.csv`): Q twin `[1000,16,32]` 512,000 threads 989.5 us warm, K twin `[1000,8,32]` 256,000 threads 500.1 us; 24 ops each, 35.75 ms together.
- Bytes (DERIVED from `omission/granite_prefill_describe.out` and the emitted source `probes/q_twin_emitted.metal`): operands are the cos table (128,000 B), the Q region read as even and odd elements (4,096,000 B once), the sin table (128,000 B); two outputs of 2,048,000 B each. Q twin 8.448 MB, K twin 4.352 MB. Own-cb: 8.5 and 8.7 GB/s. In the same run family the Q8_0 expert matvecs at decode run 139.8 / 193.8 / 147.1 GB/s in situ (4.456 MB per op over 31.9 / 23.0 / 30.3 us, `omission/summary_granite_decode.md`), the E2B Q4_0 `ffn_down` 224.9 GB/s (10.617 MB over 47.2 us) and the E2B head 299.8 GB/s (330.3 MB over 1101.7 us).
- The same arithmetic as four separate single-output elementwise kernels (`PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1`, `census_granite_prefill_notwin/`): 31.1 + 31.0 us per layer for Q and 19.1 + 19.3 us for K, 2.71 ms for the whole rope/copy family against 35.98 ms with the twins (`stdout.log` family replay lines); the whole-step sequence replay reads 252.1 / 253.3 / 252.2 ms with the twins and 255.3 / 254.5 / 253.8 without; the live GPU span of the 1000-token step reads 260.3 ms (cold census) and 258.5 ms (decode census) with the twins and 259.8 ms without (`timeline/`).
- Variant experiments on the captured Q twin (`norm_variant_ab`, `AB_STEP=0`, `AB_SHA=17b21f30`, `probes/q_variants_r1.out`, `_r2.out`): control 922.7 / 946.8 us, extra output write dropped **30.4 / 29.6 us**, first output write dropped 921.8 / 947.3 us, extra output redirected into the (bound) `out` buffer at the same physical offset **31.0 us**; all with `differing=0` on the output they kept. The cost is the write of `extra_out0`, and it vanishes when the same bytes go through a bound slot.
- Mechanism, traced: the live encoder binds the twin's second output at buffer index `bindings.len() + offset` (`omega/src/metal/arena_encode_dispatch_finish.rs:2451-2463`; the node comes from `extra_output_nodes`, `:108-134`), but the capture used by the censuses and the replay tool records only `bindings` plus whatever `live_extra_buffers` adds (`:1512-1528`), and its match arm for `ElementwiseTwin` is `_ => return None` (`:1528`). A replay of the twin therefore runs with buffer(6) unset. A window test fixes the boundary: with the twin and its neighbours replayed alone (`AB_WINDOW`, `probes/q_window*.out`), the twin (control arm minus the arm without the extra-output write, same window) costs +839 us at radius 1, +794 at radius 3, +750 at radius 4 (window = dispatches 6-14, none of which has a seventh binding) and **+15 us at radius 5**, where the window reaches dispatch 5, the rms norm whose describe output lists a valid buffer at binding 6 (`granite_prefill_describe.out`, sha `23d555d5`); radius 6 reads -3 us and radius 14 +10 us (the separately run `base` arm differs from `control` by up to 0.13 ms in an 8-12 ms window, which is the window noise). That is the replay inheriting a stale binding from an earlier dispatch in the same command buffer, which the one-twin-per-command-buffer census cannot have.
- What follows: in situ, in the whole prefill step, the Q twin costs 0.936 ms and the K twin 0.562 ms (1.50 ms for 48 dispatches, 39 and 23 us per dispatch, 217 and 186 GB/s DERIVED) against llama's 1.71 ms for 96 rope ops (`llama_ops/granite_ops.tsv`). The 35 -> 3 ms target was set against the own-cb rows, which are this binding artifact; the live step with the twins is not longer than the step without them (258.5-260.3 against 259.8 ms). Status: the cost the target named is not present in the live step; the row is a harness defect.

**Stacked expert GEMM at granite prefill (`Q8_0 4194304`, 72 ops, 169.99 ms own-cb; llama `Q8_0 524288`, 237 ops, 114.41 ms).**
- Groups (`census_groups.csv`, warm ns): gate `[1000,8,1024,512]` 2268.4 us, up + silu epilogue 2411.0 us, down `[1000,8,512,1024]` 2403.7 us, 24 ops each; in situ (two omission runs) 2255.7 / 2448.4 / 2418.2 us. 8,388,608,000 FLOP per op (2 x 1000 tokens x 8 selected x 1024 x 512): own-cb **3.70 / 3.48 / 3.49 TFLOP/s**, in situ 3.72 / 3.43 / 3.47.
- llama, same math (`llama_ops/granite_ops.tsv`, `MUL_MAT_ID`, 69 full-size ops per graph over 3 requests): gate 820.7 us at 512 tokens + 716.0 us at 488 = 1536.7 us per layer, up 1538.4, down 1512.7, i.e. **5.46 / 5.45 / 5.55 TFLOP/s**. Per op ours/llama 1.48 / 1.57 / 1.59. The class numbers differ in scope: ours 177.96 ms (170.0 stacked + 7.97 K/V) against llama 114.41 ms including its K/V is 1.56x; llama prunes the last layer to the output row (the 3 one-token ops per tensor), ours runs 24 full layers, 1/24 more work.
- Kernel: not a matvec. The stacked path lowers to the expert-grouped tiled GEMM (entry suffix `_g10`; `omega/src/msl/expert_grouped_gemm.rs:66`, emitter `push_expert_grouped_gemm_body`). Tile 64 weight rows x 32 tokens x 32 K, 4 simdgroups (128 threads, `GROUPED_THREADS` `:133`), half-staged tiles, 8x8 simdgroup MMA, the same tile as `kernel_mul_mm_id` (`:5-31`, doc). Grid 2048 threads x depth 32 experts for N=512 and 4096 for N=1024 = (row tiles 8 or 16) x `col_parts` 2 threadgroups of 128 per expert; emitted source `probes/gate_emitted.metal:895-896,954`.
- Measured with variants of the captured gate and down dispatches (`probes/gemm_*.out`, all `differing=0` where output was kept): `col_parts` 1 / 2 / 4 / 8 = **2943.7 / 2252.8 / 2324.3 / 2498.8 us** (gate), **2185.5 / 2397.9 / 2592.8 / 2984.4 us** (down): the optimum differs by projection and the spread is 10-30%, not 1.5x (the sweep in `omega-runtime.toml:96-99` is whole-prefill 864 / 828 / 848 / 870 ms). Phase ablations, gate (output wrong where noted, timing only): MMA removed **1074.8 us** (-1175), weight dequant replaced by a constant 2100.4 (-150), activation loads replaced 2247.9 (-2), both loads 2037.8 (-212), loads and MMA both 542.4; down: 1249.6 / 2171.6 / 2260.8 / 2037.9 / 765.9. Route loads replaced by a balanced synthetic assignment (`(token*7)&31`, 250 tokens per expert): gate **2283.8 -> 1734.7 us**, down **2383.7 -> 1875.0 us** (-549 and -509 us, -24% and -21%).
- Mechanism so far: the MMA phase (with its threadgroup fragment loads) is 52% of the gate dispatch, 1.175 ms for the 8.39 GFLOP (7.1 TFLOP/s for that phase alone); the part that does not involve MMA is 24% (542 us: the in-kernel route scan, barriers, write-back); the route itself is worth 0.51-0.55 ms per op, and that figure mixes two causes that this pass did not separate: the global loads of the 8000 route entries by each of the 512 threadgroups (each scans the whole route, `expert_grouped_gemm.rs:328`, `push_grouped_refill`) and the imbalance of the real routing against the balanced synthetic one. Per-expert token counts of the real routing were not captured. llama gives every (token tile, row tile, expert) its own threadgroup after a separate route compaction (`expert_grouped_gemm.rs:24-31` doc); ours gives 16 threadgroups per expert, each walking that expert's token tiles in turn. Whether the remainder (2250 - 1735 = 515 us above the balanced-route variant, against llama's 1537) is MMA issue, staging overlap or occupancy: untraced.

### 5. mechanism per remaining row

- **Router matvec `F32 32768 (1024x32)`, 24 ops, 11.52 ms own-cb (480.1 us/op; in situ 511.7; llama 48.8 us at 512 tokens + 45.4 at 488 = 94.3 us per layer, 5.1x).** Kernel: the dense batched GEMM (`classification ... node=103 dense-batched-gemm admitted=true` in `census_granite_prefill/decode_telemetry.log`; `push_dense_batched_gemm_body`, `omega/src/msl/tiled_gemm_cooperative_scan.rs:1269`), 32 threadgroups of 32 x 4 threads for 32,000 outputs, float tiles. 65.5 MFLOP in 480 us = 0.136 TFLOP/s; llama 0.695 TFLOP/s DERIVED. Variants of the captured dispatch (`probes/router_variants_r1.out`, source `router_emitted.metal`): control 475.7 us; weight loads removed **148.0**; activation loads removed 434.3; both removed **90.2**; MMA removed 435.7; weight staging rewritten to load its 32 elements into a register array before storing them to the tile (`#pragma unroll`), **207.8 us with `differing=0`**. Mechanism: the weight staging loop (`tiled_gemm_cooperative_scan.rs:1451-1475`, emitted at `router_emitted.metal:886-897`) is 64 threads x 32 serial `if`-guarded scalar loads with 64-bit address arithmetic per K step (128 threads exist, 64 rows, 32 of them real features), 1024 dependent global loads per thread over the K loop; those loads are 328 of the 476 us.
- **`F32 8192` combine fold, 24 ops, 13.63 ms own-cb (567.9 us/op; in situ 562.8; no llama op of that shape).** Kernel: the serial reduce body, grid 1,024,000 = one thread per output of `[1000,8,1024] -> [1000,1024]` (`omega/src/msl/elementwise_reduce_core.rs:655`; the dense GEMM declined with `AxisOwnershipAmbiguous`, the packed row with `NotCooperativeReduce`, telemetry node 249). At least 36.9 MB moved (32.77 MB read + 4.10 MB written, DERIVED from extents): 65 GB/s. Variants (`probes/combine_variants_r1.out`): control 566.5 us; the 64-bit `%` and `/` that decode the reduction coordinate inside the 8-step loop (`elementwise_reduce_core.rs:705-709`, one extent of 8, so the decode is the identity) replaced by `reduction_coord[0] = r`: **453.8 us**; plus the output coordinate decode (`:683-692`) in 32-bit: **220.8 us**; both `differing=0`. The repo's own note on this hardware (`elementwise_reduce_core.rs`, `push_coordinate_decomposition` doc) says integer divide is emulated; here 64-bit divides are 61% of the kernel.
- **Attention core at granite prefill, 27.59 ms for 24 ops (1149 us/op own-cb, 1135 in situ) against llama 12.93 ms for 95 ops.** Kernel: the row-tiled flash-style partial `omega_cached_attention_h8_g2_d64_..._r8_n2_b64_rt` (`omega/src/msl/cached_attention_row_tiled.rs:35`, template `:95`), f32 K/V and f32 simdgroup MMA, 1000 threadgroups of 64 threads (8 kv heads x 125 row tiles of 8 rows), causal blocks above the diagonal not visited. Visited key-query pairs, DERIVED from tile 8 and block 64: 532,480 (53% of 1,000,000); 4096 FLOP per pair across 16 heads: 2.18 GFLOP per layer, **1.92 TFLOP/s** at 1135 us. llama: `FLASH_ATTN_EXT` on **f16** K/V, two graphs, ~160 us (512 tokens, 512 keys) + ~345 us (488 tokens, 1024 keys) per layer = 505-539 us; useful causal pairs 500,256, 2.05 GFLOP, 3.8-4.1 TFLOP/s DERIVED (llama's executed pairs unknown). Which part of the kernel costs the extra time (Q.K^T, softmax, P.V, the f32 versus f16 operand width): untraced; no variant was run on it.
- **Attention core at granite decode, 51 ops 1.38 ms own-cb (partial 50.3 us/op, merge 6.5) against llama 48 ops 0.73 ms (`FLASH_ATTN_EXT` alone 200.1 ms / 9144 ops = 21.9 us).** Kernel: the decode split partial (`cached_attention_decode_split.rs:32`), 512 threadgroups of 128 threads, f32 K and V (`gemma4_decode_kernel_census.rs:488-489`), about 4.19 MB of K/V per layer at 1023 keys = 83 GB/s over 50.5 us; llama reads f16 [64,1024,8] x2 = 2.10 MB in 21.9 us = 96 GB/s. The hypothesis of the attribution section (f32 KV is the cause) was tested by running the same decode with an f16 cache: `decode_gbps_baseline`, speculation off, 999-token prompt, 96 tokens, 3 interleaved pairs (`kv_f16_probe/`): f32 **7.121 / 7.119 / 7.133 ms/token**, f16 **8.364 / 7.820 / 7.710 ms/token**, same text hash `458fa079d17c40ab` in all six. The f16 cache is 0.59-1.24 ms/token slower; the byte count is not what the attention row pays for. Mechanism: untraced.
- **E2B decode rms norm, 242 ops, 2.92 ms own-cb (12.1 us/op) against llama 1.80 ms (7.4 us/op, 4 us floor in both).** In situ: 70 ops 11.9 us, 71 ops 9.7 us, 35 ops 11.2 us (`summary_e2b_decode.md`); the 242-dispatch family replayed in one command buffer 2.216 ms mean = 9.16 us per dispatch (`census_e2b_decode/stdout.log`). Per op the data is 6.1 KB in, 12.3 KB of two weight vectors, 6.1 KB out = 24.6 KB: 2.7 GB/s at 9.16 us, so the row is latency, not bandwidth. Kernel (the form dumped in `evidence/round2fix/kernel/kernel_cap256.metal`): one 256-thread threadgroup per row, per-lane preloads of 8 slots from each epilogue operand, a strided walk, `simd_sum`, a threadgroup partial combine behind a barrier, the epilogue. The width sweep already recorded (`round2fix` section 4: 128, 192, 256, 320, 384 lanes, minimum at 256) is the only structure measured; which phase holds the 9.16 us: untraced.
- **`omega_moe_topk_e32_k8_stacked`, 24 ops, `failure=metal driver error: moe_topk binds extra outputs outside bindings` in `census_granite_prefill/census_groups.csv`.** The live encoder binds a `MoeTopK`'s 16 extra outputs after `bindings` (`arena_encode_dispatch_finish.rs:2451-2463`, outputs from `moe_topk_extra_outputs_iter`, `:108-134`); the capture records `bindings` and, for extras, only the `CachedSoftmaxWeights` triple; for `GatedDeltaNet | MoeTopK` it returns the reason string and marks the dispatch unreplayable (`:1512-1528`, the string at `:1526`, the flag at `:1258`). The census therefore cannot time or replay it (`m0 unreplayable by capture: groups=1 dispatches=24`), and the row is absent from every own-cb and in-situ total. A bound from the totals, DERIVED: live GPU span 258.5-260.3 ms minus whole-step replay without them 252.1-253.3 ms = 5.2-8.2 ms, 0.22-0.34 ms per op, which also contains any live-versus-replay difference unrelated to the router.

### 6. proposed round three slices (ranked by gap; each changes one cost; no model names; numbers are own-cb ms at granite prefill unless stated)

1. **Route-dependent term of the stacked expert GEMM** (gap +59.9 ms to llama's 110.1 ms at 24 layers, DERIVED from 24 x (1536.7 + 1538.4 + 1512.7) us). Change: stop every threadgroup scanning the full route; compact the route once per projection and give token tiles their own threadgroups (`omega/src/msl/expert_grouped_gemm.rs:328`, `push_grouped_refill`, and the grid in `push_grouped_entry` `:250`). Target 169.99 -> 131.4 ms (-38.6 ms = 24 x (549 + 549 + 509) us, from variants `b1` on gate and down; up+silu assumed equal to gate, untested). Remainder to llama (21 ms) is the MMA/staging term, untraced. Gate: the output must stay `differing=0` against the current kernel on the captured dispatches (the variants held it), `llama_parity_`, the three censuses.
2. **Attention core at prefill** (gap +14.67 ms). First deliverable: ablation variants of `ROW_TILED_KERNEL` (`cached_attention_row_tiled.rs:95`) with the AB tool, as done for the router and the GEMM (Q.K^T only, P.V only, no softmax). Target 27.59 -> 12.93 ms (llama's class), the change being whichever single term the ablation names.
3. **Combine fold coordinate decode** (gap vs the measured variant 8.3 ms). Change: decode the output and reduction coordinates in 32-bit when the grid fits, drop the identity decode of a single reduction axis (`elementwise_reduce_core.rs:683-709`). Target 13.63 -> 5.30 ms (24 x 220.8 us, variant `c2`, bit-identical on the captured dispatch).
4. **Router weight staging** (gap +9.35 ms to llama). Change: the dense GEMM weight staging loop loads its row into registers before the tile stores (`tiled_gemm_cooperative_scan.rs:1451-1475`). Target 11.52 -> 5.0 ms (24 x 207.8 us, variant `r5`, bit-identical); the further step to 2.3 ms (llama 94.3 us per layer, the stage-free floor 90.2 us) is a second slice on the activation loop and the idle half of the weight threads.
5. **First command buffer size at decode** (host row; E2B -0.55 ms, granite -0.21 ms per step). Change: `first_chunk_ops` (`omega/omega-runtime.toml:289`, consumed at `placements_execute_named.rs:226`) chosen so chunk 1's GPU time covers chunk 2's encode time (0.262 ms of GPU against 0.823 ms of encode for E2B). Target: the chunk-2 gap 0.552 -> 0.03 ms (E2B) and 0.212 -> 0.03 ms (granite), net of the lead-idle increase a larger head causes (the encode cost of the head ops, about 6 us per op, is not netted here; the net is the measurement).
6. **Replay of extra outputs** (measurement defect; no step time moves). Change: record the `ElementwiseTwin` extra output in `live_extra_buffers` (`arena_encode_dispatch_finish.rs:1528`) and the `MoeTopK` ones, so the census times them. Target: the twin row 35.75 ms -> about 1.7 ms (the in-situ 0.936 + 0.562 ms plus 48 dispatches x 4 us of own-cb floor), and `omega_moe_topk_e32_k8_stacked` gets an own-cb number in place of `failure=`.
7. **E2B decode rms norm** (gap +1.12 ms to llama). First deliverable: phase ablation of the 256-lane norm kernel with the AB tool. Target 2.92 -> 1.80 ms (llama's class).
8. **Prefill step host time** (141.3 ms of 399.9 ms outside GPU busy). First deliverable: a timestamp at each of prepare, pre_encode, commit and GPU start to split the 58.2 ms commit-to-start; target commit-to-start 58.2 -> 0.2 ms (the steady-state chunk-1 value is 0.17 ms).

### 7. what this section does not establish

- Slices 2, 7 and 8 have a reference and no cause. The granite decode attention row has a refuted hypothesis (f16 cache slower) and no cause, so it has no slice.
- The real per-expert token counts were not captured, so the share of the route term due to imbalance versus route loads is not separated; the balanced synthetic route is not the model's routing.
- The twin artifact: the stale-binding explanation is supported by the radius 4 / 5 boundary and the redirected-write variant; it was not shown by reading the Metal driver. The live step's twin cost is bounded by the equal GPU spans, not measured per dispatch.
- The `rms norm` row of the granite prefill omission reads -7.86 ms (49 dispatches): removing it speeds the replay. Unexplained; it does not appear in the own-cb table (2.90 ms).
- Omission runs: granite decode 3 runs, E2B 3 complete runs plus a truncated first, prefill 2 runs; ranges are in the summaries, several rows under 0.2 ms are inside them. The omission base replay of granite decode (5.576 ms) is 0.4 ms above the census sequence replay of the same dispatches (5.14-5.20 ms in the previous pass); not explained.
- Own-cb sums exceed `gpu_busy` by 1.10x for granite prefill (285.1 against 258.5-260.3), so own-cb recoveries overstate wall time by about that factor. The whole-step replay of the censuses (252-255 ms) is 2.7% below the live span (258.5-260.3).
- The kernel variants are timing instruments on one captured dispatch each (member 0 of the group); the f16/f32 cache pair was not interleaved with a control arm and the GPU-utilization samples before it read 93-95.
- llama numbers are recordings from another box state; llama's 23 full layers per request against ours 24 is stated above, not corrected in the tables.

### re-prove

```
attribution_rank rank --census evidence/attr3/census/census_granite_prefill --llama evidence/slice0/llama_ops/granite_ops.tsv --ntok 512,488 --requests 3 --floor-us 4.0   # rank/granite_prefill.md
omission_summary evidence/attr3/omission/granite_decode_r1.out evidence/attr3/omission/granite_decode_r2.out evidence/attr3/omission/granite_decode_r3.out   # tools/omission_summary.rs
step_timeline evidence/attr3/census/census_granite_decode/telemetry_events.log evidence/attr3/census/census_granite_decode/census_dispatches.csv 23   # tools/step_timeline.rs
PROXIMA_GEMMA4_E2B_GGUF=<granite blob> PROXIMA_PROMPT_FILE=prompt1k.txt AB_VARIANT_DIR=evidence/attr3/probes/variants_q AB_SHA=17b21f30 AB_STEP=0 AB_ROUNDS=30 norm_variant_ab        # twin variants; variants_router (AB_SHA=7cc38fcb), variants_combine (2c18a81c), variants_gemm/2/3 (6cd35d13, AB_PACKED=1)
PROXIMA_GEMMA4_E2B_GGUF=<E2B blob> PROXIMA_PROMPT_FILE=prompt1k.txt AB_VARIANT_DIR=<empty dir> AB_OMIT_ALL=1 AB_PACKED=1 AB_STEP=5 AB_ROUNDS=30 AB_FLUSH_MIB=0 AB_SKIP_GROUPS='a638c75f896670e1:[1, 8, 256];2417697c78307ab2:[1, 8, 512]' norm_variant_ab   # E2B; granite: no skip list; prefill: AB_STEP=0 AB_ROUNDS=15
PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1 M0_OUT_DIR=<dir> M0_MODEL_GGUF=<granite blob> M0_MAX_TOKENS=2 M0_CAPTURE_STEPS=0 PROXIMA_PROMPT_FILE=prompt1k.txt gemma4_decode_kernel_census
PROXIMA_DECODE_MODEL_GGUF=<granite blob> PROXIMA_PROMPT="$(cat prompt1k.txt)" PROXIMA_MAX_TOKENS=96 PROXIMA_DECODE_SPECULATIVE=none PROXIMA_KV_CACHE_TYPE=<f32|f16> decode_gbps_baseline
PROXIMA_DEBUG_METAL_SOURCE=1 PROXIMA_METAL_COMPARE_BOUND_NODE=<node> ... AB_DESCRIBE=1 AB_STEP=0 norm_variant_ab 2> source.txt   # emitted MSL of one node (nodes: 47 Q twin, 103 router, 226 gate, 249 combine)
```
Missing for CI: no job runs the Metal tests, the censuses or the AB tool; every row re-proves on this box only. The binaries are under `.long_ctx_backups/attr3/bin/` (sha256 in `evidence/attr3/bin.sha256`).
The two summarizers are single-file Rust programs, built with `rustc -O --edition 2024 evidence/attr3/tools/omission_summary.rs` and `.../step_timeline.rs`; they read the `.out` and event files in this directory and print the markdown tables kept beside them.
