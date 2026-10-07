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
   151 / Ollama 177 once slices 3 and 4 land.
2. E2B prefill Q4_0 tiled GEMM throughput. Removes up to 980 ms. Target: Q4_0 class 1479.66 -> at most 500 ms
   (llama 483.00, the 7.9 to 8.3 TFLOP/s rate, derived); E2B prefill 2540 -> about 1560 ms (derived). First step: capture GPU counters
   or toggle the kernel's staging and fragment types to find the kernel-level cause, since it is untraced.
3. Prefill attention (E2B and granite): one fused kernel per layer. Target: E2B 749.70 -> at most 60 ms
   (llama 41.31), granite 286.28 -> at most 20 ms (llama 12.93). Removes about 690 ms from E2B prefill and 266
   ms from granite prefill.
4. Weights outside the K-quant tiled path: E2B F16 per-layer projection 107.77 -> at most 5 ms (llama 4.35),
   granite Q8_0 1024x1024 118.21 -> at most 16 ms (llama 15.08). Removes about 103 ms each.
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
