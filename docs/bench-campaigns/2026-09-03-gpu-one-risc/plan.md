# GPU parity through ONE RISC — the plan, the evidence, and the Luna task cards

Date opened: 2026-09-03. Main at plan time: `4be2f3a`. Owner: Brian Bruggeman.
Lane: proxima-tensor -> omega -> Metal (and wgpu/CUDA as the same lowering must cover).
Incumbents: llama.cpp-Metal (deployed arm for quantized LLM decode), torch-MPS (f32 training),
onnxruntime-CoreML (ONNX inference), burn (secondary reference only).

This document is three things in one file, in this order, because the reader who executes it
(Luna, the cheap hands model) needs each in exactly this order:

1. **Section 0 — the execution protocol.** How a task card is read, what a task may and may not
   do, what "done" means for a card, and how a card reports. Read once, obey always.
2. **Sections 1-4 — the diagnosis and its evidence.** What is going wrong, with every claim
   tagged by provenance and bound to `file:line` on main `4be2f3a`. This is what the cards act
   on. A card that contradicts the evidence stops and reports; it does not improvise.
3. **Sections 5-9 — the plan.** The phases, the task cards, the dependency graph, the rollback
   map, the discipline-log row templates, the ai_docs records, and the tournament trail that
   produced the ordering.

The rules this plan is built under are binding and are not restated here beyond what a card
needs: `~/.claude/skills/guiding-principles/SKILL.md` (§1 reuse-first, §3 tiers, §4 config+
builder, §6 read the code, §11 sans-IO, §12 no magic numbers, §14 incumbent wins on
correctness, §15 no punt, §16 re-provable, §18/§19 provenance and the evidence ladder, §20
box-free, §21 lock-free), `~/.claude/skills/disciplined-component/SKILL.md` (the 16-point gate,
home-turf arms, the frequency-weighted scorecard, one measurer on the box), and the workspace
`AGENTS.md` (directive compliance; "do not add arbitrary rules/code for specific instances";
never commit without asking the owner; no `gh`; no time estimates; no verdicts).

---

## 0. Execution protocol for task cards

### 0.1 Who executes what

| role | model | does | never does |
|---|---|---|---|
| hands | Luna (GPT-5.6 Luna) or Haiku 4.5 | runs the card's commands verbatim, opens the named `file:line`, applies the card's bounded edit, records the card's N and numbers into the card's log row, reports | designs, chooses between two shapes, widens scope, "improves" a command, skips a gate, commits |
| worker | Terra / Sonnet 5 | cards marked `tier: worker` — implementation whose shape the card fixes exactly (a named function with a named signature and a named test) | new types, new traits, new Op/BoundOp variants, any change to a file the card does not name |
| judge | Sol / Opus (main thread) | every card marked `tier: judge` — adjudications, the fork resolutions in §8, the log rows' "honest read", landing order | typing code, running builds, grepping |

A card's `tier:` field is the routing decision. Luna executes `tier: hands` cards. A `tier:
worker` card is dispatched by the main thread to a worker with the card as the whole brief. A
`tier: judge` card is never executed by hands; hands stop at it and report.

### 0.2 The card contract (every card has every field; a blank field is a defect in the plan)

```
CARD <phase>.<n> — <imperative title>
tier: hands | worker | judge
depends_on: [<card ids>]            # all must be DONE (row sealed) before this card starts
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>  # one per phase-branch (§5 G2); NEVER an existing name (§3.7)
branch: risc/<PH>-<slug>            # created from the card's stated base at the card's start
target_dir: <worktree>/target       # own dir, never shared
lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock       # card 0.1's shim; every build/test/bench/gate welded through it (§5 G3)
opens: <file:line> [, <file:line>]  # the only files the card may edit
commands:                            # verbatim, in order, each welded with `cd <worktree> &&`
expect:                              # the N and the numbers; N==0 is RED, always
predict:                             # the ONE-rung-ahead prediction, written before running
kill:                                # the observation that ends the card as a NEGATIVE
memory gate: MG-1 | MG-2 | MG-3      # §0.3a; MG-3 is the five clauses (two slopes, prefill cap, steady cap, plan_cache_len, uniform-cache gauge)
rollback:                            # the exact command that restores the prior state
blast:                               # every file/crate/backend the change can reach
observe:                             # the counter or record that proves the mechanism fired
row:                                 # the discipline-log row template this card fills
reprove:                             # the exact command that regenerates the card's numbers
```

### 0.3 Rules a hands card obeys without exception

1. **Weld the worktree into every command.** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-<name> && <command>`. If `pwd` ever shows a different `proxima-wt-*`, stop and report drift.
2. **Never pipe a build/test into a filter.** `cargo ... 2>&1 | tee <log>; echo "EXIT=${PIPESTATUS[0]}"` and grep the log afterwards. The compiler's exit code is the gate, not grep's.
3. **Assert N.** Every test command reports `N run / N passed`. `N==0` is RED and the card stops. `cargo test --doc` and `nextest -p <crate>` both exit 0 on zero matches — that is the trap.
4. **One measurer on the box.** Before any timed cell: `pgrep -fl 'cargo|rustc|llama|criterion|nextest'` must show only your own processes and `uptime` 1-minute load must be under the card's stated gate. Record `uptime` before and after every cell into the cell's log.
5. **Interleave arms.** A/B/A/B/A/B, never A/A/A/B/B/B. The control is measured against itself first; if the control's own spread exceeds the card's noise line, the cell is VOID and the card reports that, not a number.
6. **Compare by family, never by bucket.** `op_profile_bucket` is a source-text label; `op_profile_family` (weight name) is structural.
7. **No landing.** Nothing reaches `main` except through a `tier: judge` landing card the owner authorizes per phase. Commits on a `risc/*` branch inside the card's own worktree are the one authorization this plan asks for up front (§5 G13): the cherry-picks, the ON/OFF arms built from the previous commit and the phase-branch bases all need them. Until the owner records that acceptance, a card ends at `git add` with its message in `<worktree>/runs/<card>.commitmsg` and says so in its report.
8. **No new types.** If a card cannot be completed without a new struct/enum/trait/variant, the card is wrong; stop and report which line forced it.
9. **No estimates, no verdicts.** Report what ran, what it printed, the `file:line`, the N.
10. **A miss is a work item.** If `expect` or `predict` is missed, the report names the category (inconsistency between two measurements, or a cost term the model did not know) and the instrumentation that would make it visible.

### 0.3a Memory and CPU% are mandatory cell columns (owner, 2026-09-03: "are you measuring memory?")

Every timed cell records, per arm, in the same run as the timing:
- ours: `phys_footprint_bytes` (task RSS) and `device_allocated_bytes` (Metal allocation) from
  `token_breakdown_metal` (`generate.rs:1721-1750`), plus peak RSS via `/usr/bin/time -l` on the
  test binary, plus CPU% (`ps -o %cpu` sampled, or `/usr/bin/time -l`'s user+sys over wall).
- incumbent: `/usr/bin/time -l` max RSS and user+sys around the `llama-bench`/`llama-cli` run;
  `llama-bench` itself reports no memory.
- the row states the SCOPE of every memory number (process RSS vs device allocation vs peak);
  two numbers of different scope are never placed in one ratio. Today's seal (§1.1) recorded
  ours (RSS 310-357 MB, device 4.30 GB) and nothing for the incumbent; the only incumbent
  memory cell in the log is the parallel branch's max RSS 4.27-4.78 GB (ROWs 243/250/257).
  Card 0.5 re-runs the seal with both arms under `/usr/bin/time -l`.

**Memory is a kill criterion on every card, not a column** (owner, 2026-09-03: "if you've
fucked up memory again, you lose"). The two prior failures: a 34 GB KV allocation sized from
`context_length`'s 131,072 default (found in a worktree, never landed on main), and a per-token
plan-cache growth on the Rust heap (bounded on main by ff749a0). Every card whose cell runs the
decode harness asserts, from the same log:
1. steady-state task RSS slope over steps 1..N is within jitter (today: 48-66 MB, no
   monotonic trend; `phys_footprint_bytes` at `generate.rs:1721-1750`);
2. `device_allocated_bytes` slope ≤ the KV row bytes the graph legitimately adds per token
   (today: +1-2 MB/token at 4.152-4.163 GB; a KV-resident card may move the base, never the slope);
3. `plan_cache_len` stays ≤ 1 on every step, bucketed or not — the cache clears on miss
   (`generate.rs:973`), so a bucketed key that fills the map is the ff749a0 leak re-opened;
4. two absolute caps built only from MEASURED numbers and one labelled CHOSEN headroom (§5 G8,
   clauses 3a/3b): prefill (step 0) peak `device_allocated_bytes` ≤ `PREFILL_CAP = 4_305_000_000 ×
   1.05 + kv_capacity_tokens × 262_144` (R13's prefill peak 4.299-4.305e9, top of range; 4,520,250,000
   at kv=0, 4,654,467,728 at kv=512, DERIVED) and steady peak over steps 3..S ≤ `STEADY_CAP =
   4_163_000_000 × 1.05 + kv_capacity_tokens × 262_144` (4,371,150,000 at kv=0, 4,505,367,728 at
   kv=512). The per-token KV term is 32 layers × (2048 + 2048 + 4096) B (R17). A single cap that did
   not separate prefill from steady state would fire on unmodified main today (round-4 critique
   B7), and the earlier 40 MiB "activation term" had no derivation (B14) — both are gone. The
   transient headroom for arenas is `ARENA_TRANSIENT_CAP = (4_305_000_000 − 4_140_417_024) × 1.05 =
   172_812_125 B` (DERIVED). `kv_capacity_tokens` is a build-time key with a `build.rs` byte
   assertion; `context_length` (default 131_072, `serving.rs:161`) never sizes an allocation —
   131_072 × 262_144 = 34_359_738_368 B is the trap reproduced from source;
5. `UNIFORM_CACHE_LEN` (a gauge card 0.2 adds over `UNIFORM_BUFFERS`, `omega/src/metal.rs:2056-2078`)
   does not grow after step 3 — that map is keyed by the uniform BYTES and is unbounded on main, and
   because every `Uniforms` blob carries `reduction_total`, a function of `cached_len`, it is the
   pre-registered candidate mechanism for today's +1-2 MB/token device slope (round-4 finding D6;
   card 0.2 measures it before any card designs against it).
A card that breaks any of the five is NEGATIVE regardless of its timing delta. The three named
gates (§5 G8): MG-1 build/lint only; MG-2 probe/bench without a checkpoint (peak RSS ≤ 400 MB,
raised only with the arithmetic on the card); MG-3 the decode harness, all five clauses.

Second instrument defect found in today's cell: `block_upload_bytes` = 4.147 GB per steady
token — it counts the mapping-offset BINDING of the whole checkpoint (`mapping_offset_uploads=291`),
not copied bytes; the copied bytes are the KV's 8-10 MB (`copying_uploads=4`). Card 0.2 fixes
`block_upload_bytes` and `operand_bytes` together, in BOTH upload loops (`execute_plan` at
`metal.rs:465-486` and `execute_plan_op_timed`'s own loop at `:663-690`, which is where every
`op_profile_family` number comes from): report copied bytes and tensor bytes, never buffer lengths.

### 0.4 How a card reports (the only accepted shape)

```
CARD <id> — <STATUS: DONE | NEGATIVE | VOID | STOPPED>
ran: <each command, EXIT code>
N: <tests run/passed per crate>  <dispatch/op counts>  <cells x repeats>
numbers: <table, with CoV per cell, load before/after per cell>
predict vs observed: <one line; if miss: category + work item>
files: <path:line ranges changed>  diff --stat: <...>
row: <the filled row text, ready to paste>
reprove: <command>
open: <anything the card could not do, by name>
```

---

## 1. Scoreboard at plan time

Every number carries provenance: **MEASURED** (a record produced this session or a discipline
row opened this session), **READ** (source opened this session with `file:line`), **MEMORY**
(from the 2026-09-02 session notes; must be re-measured before any card depends on it),
**DERIVED** (computed from other rows; never a mechanism claim).

| lane | machine roofline | deployed incumbent | us (main 4be2f3a) | gap | provenance |
|---|---|---|---|---|---|
| quantized 7B decode, Metal, openchat-3.5 Q4_K_S, ms/token | **DEBT** — no GPU streaming-bandwidth constant exists (rooflines.md:396-479, summary row :751) | llama.cpp-Metal b25346221 **17.354-17.470** CoV 0.36-0.45% (parallel-branch ROWs 243/250; MEMORY 17.470 quiet-box) | see §1.1 for today's sealed cell; parallel-branch control 51.571 wall / 35.117 GPU CoV 1.75/2.00% (ROW 267, feature off, paired-Q4_K body present) | **2.92-3.98x** | ledger R1, R12 |
| same lane, dispatches per token | — | ~740 (READ, 23 real ops/layer x 32 + 4, ggml b25346221) | 1194-1196 main; 616 with the parallel branch's macro-op | 1.62x / 0.83x | R8, R12 |
| Q4_K matvec effective GB/s, ffn family | DEBT | 228.9 whole-token (MEMORY) | 96-98 (MEMORY, main) / ~140 with a rewritten body (DERIVED from -29..-36%) | 2.4x -> ~1.6x | R2, R3, R12 |
| mnist f32 inference /image, GPU | NEON 0.057 ms is the CPU physics; GPU constant DEBT | torch-MPS **no cell**; ORT-CoreML **no cell** | **no Metal arm exists** (R9) | unmeasured | R9 |
| MLP 784-128-10 b32 train step, GPU | DEBT | torch-MPS **no cell** | Metal parity test only, untimed (`omega/tests/training_step_parity.rs:400-607`) | unmeasured | R9 |
| BGE-small /sentence, GPU | DEBT | ORT-CoreML **no cell** | **no Metal arm exists** | unmeasured | R9 |
| BGE-small /sentence, CPU (context) | AMX 1.94-2.75 ms | ORT 5.724 CoV 0.63% | 6.037 CoV 0.40% (Accelerate on) | 1.055x parity | MEMORY |

**Out of this plan's scope, recorded so it is not lost** (owner, 2026-09-03: "we were at the ORT
level which was 5 ms, floor is 1 ms for hardware, and we were at about 7-9 ms"): the BGE-small
CPU lane sits at 6.037 ms/sentence against ORT's 5.724 and a machine roofline the rooflines doc
derives at 1.94-2.75 ms on AMX (`rooflines.md:481-733`); the 7-9 ms cells were the NEON-only
and pre-arena rows; no row in the log supports a 1 ms floor. That 2.2-3.1x gap to silicon is a
CPU-lane campaign (per-node dispatch structure, the fixpoint rewrite engine in
`rewrite-algebra.md`) and shares nothing with the Metal decode path but the IR. It is not
addressed here.

The owner's statement "llama, ggml, ort and torch can beat us on gpu" is MEASURED for llama.cpp
(which is ggml-Metal under another binary — one incumbent, two names) and UNMEASURED for ORT
and torch: no GPU cell exists on either side for the lanes they deploy. Cards 10.3 and 10.4
build those cells with their scope stated on the card: neither torch nor ORT has a Q4_K/GGUF
path, so neither is a decode competitor; they are a roofline companion (matvec shapes) and an
embedding-lane arm (BGE-small f32). The CPU lane "we have if we turn on acceleration" is
MEASURED at 1.055x ORT with Accelerate engaged (MEMORY, quiet box 2026-09-01) — the valve
`ACCELERATE_GEMM_ENABLED` is `new(false)` at `proxima-tensor/src/cpu.rs:1915` (MEMORY; out of
this plan's scope).

### 1.1 Today's sealed cell (sealed this session; card 0.5 replicates it on fixed instruments; raw records in §3.9)

MEASURED 2026-09-03 on main `4be2f3a`. Box LOADED (1-minute load 4.7-5.7 through every cell,
a `cdb-daemon` resident; no cargo/rustc/llama concurrent). Arms interleaved A B A B A B. Raw
logs: `docs/bench-campaigns/2026-09-03-gpu-one-risc/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile,build,quiet_gate}.log` (copied
into the ROW 234 record by card 11.1).

| arm | per run | mean | CoV across runs | ms/token | ratio |
|---|---|---|---|---|---|
| llama.cpp-Metal `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99`, tg32 t/s | 56.70±0.35 / 56.89±0.54 / 57.66±0.14 | **57.08 t/s** | 0.89% | **17.52** | 1x |
| ours `step_wall_ms`, steps 1-7 (7 per run) | 67.55 / 67.96 / 68.25 | 67.92 | 0.5% | 67.92 | **3.88x** |
| ours `gpu_exec_ms`, steps 1-7 | 57.07 / 56.49 / 57.22 | 56.93 | 0.7% | | 3.25x kernel-only |

`generated_text` identical across all three runs. `op_count=1196` every step. Per-phase means
(ms/token): prepare 1.97, emit 0.81, block_upload 2.0, op_setup 3.9, pipeline_lookup 0.04,
encode_dispatch 0.47, readback 0.22 (Σ 9.4; wall − gpu_exec = 11.0; ~1.6 residual in sampling
and cache append). **`plan_hits=0 plan_misses=8` every run** (P4 confirmed on today's main;
there is exactly one `plan_hits` field, `generate.rs:905`, and the harness asserts it is zero
at `proxima-model-interop/src/bind.rs:3052-3055` — card 6.2 inverts that assertion, `#[cfg]`-paired, as the
consecutive-key formula of §5 G5, never as a literal token count).

Per-op profile, step 3 (diagnostic mode; Σ 61.082 ms over 1196 ops vs batched 56.93 = 7.3%
excess, admissible):

| bucket | ops | gpu_ms | ns/op |
|---|---|---|---|
| reduce-packed-row-blocked (Q4_K/Q6_K matvec) | 225 | **44.450** | 197,556 |
| reduce-cooperative | 385 | 9.113 | 23,670 |
| elementwise | 547 | 7.350 | 13,437 |
| constant / iota (the zero-byte degenerate control) | 37 / 2 | 0.161 / 0.008 | 4,352 / 4,208 |

| family | ops | gpu_ms | true bytes/op | GB/s |
|---|---|---|---|---|
| ffn_up / ffn_gate | 32 / 32 | 10.862 / 10.843 | 33.05 MB | 97.4 / 97.5 |
| ffn_down | 32 | 10.005 | 34.00 MB | 108.7 |
| attn_q / attn_output | 32 / 32 | 5.172 / 3.846 | 9.45 MB | 58.5 / 78.6 |
| attn_v / attn_k | 32 / 32 | 1.523 / 1.462 | 2.4 MB | 50-53 |
| output.weight (Q6_K) | 1 | 0.737 | 107.5 MB | 145.9 |
| "(no named operand)" — attention, norms, elementwise | 681 | 11.382 | — | — |
| kv_cache.v / k_odd / k_even (cached-range attention reduces) | 32 each | 1.559 / 0.783 / 0.774 | | |
| rope_cos / rope_sin | 64 / 64 | 0.779 / 0.762 | | |
| eps (rms-norm) | 65 | 0.571 | | |

Gap decomposition on today's main (67.92 − 17.52 = 50.40 ms/token): weight streaming at the
incumbent's 228 GB/s ≈ 17.5 (irreducible); Q4_K matvec above that rate ≈ **26.9**; non-matmul
GPU ≈ **16.6** (9.1 cooperative + 7.35 elementwise + 0.17 floor); orchestration ≈ **11.0**.
(Σ 54.5 vs 50.4 — the difference is per-op-mode inflation.) The MEMORY numbers in §3 are
confirmed within 1-3%; the diagnosis in §2 stands on today's main.

**Instrument defect found by this cell:** `operand_bytes` per matvec op reports 4,140,417,024
= the whole checkpoint mapping buffer (since 7d09145 addresses weights by offset into ONE
buffer), so `gpu_ns_per_byte` and `total_operand_bytes` (1.2 TB) are wrong. True bytes above
are derived from shapes (`rows * k * 0.5625`). Card 0.2 fixes the profile to report the
tensor's byte length before any GB/s row is written from it.

---

### 1.2 What landed on 2026-09-04 (after this plan was written; supersedes §8.1's rejection)

The owner overrode §8.1 and ordered every unmerged branch and dirty worktree merged. Ninety
commits landed on main between `4be2f3a` and `f2d5094`, every one behind the five gates plus
`cargo nextest run -p omega --all-features` and the six-step `scripts/omega-gate.sh`; every perf
feature landed default-off, then two interleaved sweeps against `llama-bench` on a quiet box
decided the defaults. Raw logs: `tournament/../sweep-logs` in the session scratchpad; the
discipline row that carries the tables is the one the flip commit adds after ROW 270.

| arm (main `f2d5094`, 3 rounds × 7 decode steps, `PROXIMA_MAX_TOKENS=8`) | ms/token | CoV | ops | ratio vs llama.cpp 17.62 |
|---|---|---|---|---|
| this morning's seal, `4be2f3a` | 67.92 | 0.5% | 1196 | 3.88x |
| default after the cached-attention merge (`18d4ab0`+) | 52.66 | 0.31% | 1194 | 2.99x |
| `metal-wide-cooperative-reduce` | 49.74 | 0.48% | 1194 | 2.82x |
| `metal-output-placement` (device-resident KV, zero KV upload) | 45.32 | 0.46% | 938 | 2.57x |
| **placement + wide reduce (the new default)** | **43.05** | 0.56% | 938 | **2.44x** |
| + single-fetch / + mask-fma on top | 43.06 / 43.32 | 0.5% | 938 | no further gain |
| `metal-buffer-pool` / `metal-q4k-split-k` | 56.24 / 56.50 | | 1194 | losses, stay default-off |

`generated_text` identical on every arm; device bytes 3970-3977 MiB and RSS 54-77 MiB on every
arm (the memory rule held).

Second wave, same day, each on the then-current default and each landed or recorded:

| change | ms/token (default oracle) | what the measurement says |
|---|---|---|
| KV extent bucketed to 32 tokens (cards 6.1-6.3; ROW 273) | **40.88** (2.32x), `plan_hits` 5 of 8, text identical | prepare 3.7 → 1.1 ms on hit steps; bucket 256 is a loss (+5.6 ms) because `op_setup` scales with the padded extent; the placed path already carried a `cached_len` leaf and its causal mask masks padded keys, so ZERO new ops were needed (938 on every arm) |
| byte counters (card 0.2; ROW 272) | unchanged | per-family bytes are tensor bytes (ffn_up 33.05 MB, output.weight 107.5 MB, 4.17 GB/token); `uniform_cache_len` = 57, which REFUTES the D6 candidate mechanism for the device slope |
| op-timed executor for the placed path (main `80c526a`) | unchanged | per-op attribution restored on the default: packed Q4_K 225 ops 25.8 ms, cooperative 225 ops 4.4, elementwise 451 ops 5.8; ffn families stream at 172-186 GB/s vs the incumbent's 229, attn_k/v at 66-78 |
| verbatim port of ggml's `kernel_mul_mv_q4_K_f32_impl<4,2,32>` (`metal-q4k-ggml-port`) | +16% `gpu_exec` on a contended box, direction confirmed twice at the ffn_up shape | a fifth negative for the incumbent's geometry on our tree; and it exposed that the `metal-packed-row-nsg2` arm was unreachable dead code, so every earlier nsg=2 "negative" here measured nothing |
| split-K gated to rows ≤ N (card 10.2's entry-gated form) | loss on every arm | attn_k/v go 79 → 31 GB/s under any cap; compiling the feature in taxes every packed matvec ~5% (a runtime division at `split == 1`) |
| fused `CachedAttention` on the placed path | no signal (42.80 vs 42.88) | its matcher returns zero candidates on the single-range program; the kernel §8.1 rejected and the owner merged is dead code on the default |
| removing the 64 `Identity` regroup copies and fusing the RoPE halves (graph) | not landable | blocked by the algebra: `unify_iteration_space` (`shape.rs:225-229`) pins an axis only from a single-term coefficient-1 operand, `Multiply` is arity 2 (`op.rs:95-103`), and `CachedLayerRoots = (even, odd, value)` is consumed at 15+ sites — a design decision for the owner, not a card |

| serial route for reductions shorter than `[cooperative_reduce] min_len` | +10% wall; the 64-long score reduces 3x slower, `eps` control flat | these reductions are memory-LATENCY-bound: 32 lanes per output hide DRAM latency, one thread per output does not; the knob lands inert at 0 |

What is left after the second wave, per token on the default: matvec ≈23.5 ms at ≈170 GB/s
aggregate against the incumbent's 229 (≈6 ms), attention in latency-bound ops ≈8 ms (a
34-long and a 64-long reduction per head; more threads per output helped, fewer hurt), RoPE and
regroup copies ≈3.5 ms (algebra-blocked), orchestration ≈6 ms after bucketing.

Two decisions only the owner can make, because each changes the algebra or adds a kernel the
RISC does not currently express:
1. **Attention.** The measured facts point at one fused per-layer attention kernel (Q·K over the
   cached range, softmax, ·V) with threadgroup-cooperative K/V row loads — the incumbent's
   flash-attention-vector shape — replacing 7 launch-bound ops per layer (≈8 ms/token). The
   merged `CachedAttention` kernel is not that kernel: its matcher never fires on the
   single-range program and, where it did fire on the old tree, its GPU time was worse. Doing
   this right is a new emitter body for an existing `Reduce` shape (one bound kind, no new `Op`),
   plus the graph writing attention as that shape.
2. **Extent inference and arity.** The 64 `Identity` regroup copies and the four RoPE ops per
   layer exist because `unify_iteration_space` pins an axis only from a single-term
   coefficient-1 operand axis and elementwise bodies have arity 2. Relaxing either is an IR
   change (`shape.rs:225-229`, `op.rs:95-103`) with a workspace-wide blast radius; the gain is
   ≈3.5 ms/token. What the sweeps refute in this document: the buffer pool (§5 card
6.5's premise) is a measured loss on the current tree, and split-K (card 10.2) is a measured
loss at this shape; both stay selectable and recorded. What they confirm: the wide cooperative
reduce (card 7.2, −2.9 ms alone, −2.1 ms on top of placement) and device-resident KV with
in-graph placement (cards 5.2/9.2, −7.3 ms). The remaining 25 ms to llama.cpp is the Q4_K body
(`gpu_exec` ≈36 ms on the placed path against the incumbent's 17.6 total) and ≈6 ms of
orchestration — Phases 3, 4 and 6 of §5, in that order.

Third wave, 2026-09-04, continuing on the then-current default (KV-bucket-32, ROW 273): three
negatives already named in the second wave (split-K, serial-reduce, ggml-port) got their formal
discipline.md rows and quiet-box numbers, one dispatch-reachability bug was found and fixed, and
three features flipped the default forward.

| change | ms/token (default oracle) | what the measurement says |
|---|---|---|
| operand-byte accounting corrected (ROW 272 correction, `c05bf1e`) | unchanged (instrument-only) | `operand_bytes` summed the shared checkpoint buffer's own length per operand instead of `element_count × dtype size`; replaced with `operand_tensor_bytes`; `total_operand_bytes` now 4.169 GB/step, within 2.4% of the ~4.07 GB/step prediction |
| split-K row-count gate, formal numbers (ROW 274) | quiet `gpu_exec_ms`: default 34.733 vs split-K 38.17-38.77 (~11% slower); loaded box (host under concurrent load, same rounds), end-to-end wall: default ~177 vs split-K ~229-232 ms/token | gate is reachable and causal but recovers no win on any family it reaches, including its intended target; stays default-off, landed as infrastructure for a future kernel redesign |
| packed-row nsg2 dispatch made reachable (ROW 275) | corrected steady-state read: default 41.09 vs nsg2 41.29 ms/token (flat) | the earlier +23% reading averaged the cold pipeline-compile step into the mean; per-family GB/s flat on every family — the second wave's nsg=2 negative had measured unreachable dead code |
| short-reduce serial route, formal numbers (ROW 276) | min_len=0 (default) 41.920 vs min_len=64/128/256 43.315/46.164/46.020 ms/token | confirms the second wave's read: these reduces are memory-latency-bound, not compute-bound; the knob lands inert at 0 |
| verbatim ggml Q4_K port, formal numbers (ROW 277) | steady-state default 41.414 vs ggml-port 45.567 ms/token (+10.0%); packed-row-blocked bucket 25.773 vs 29.769 ms (+15.5%) | confirms the second wave's read as a fifth negative for the incumbent's geometry on this tree; the same commit exposed nsg2 as unreachable dead code |
| paired-nibble Q5_K matvec, default flip (ROW 278) | per-op 202,083 → 124,265 ns/op (1.63x) on the 8 real Q5_K ops; decode wall flat within noise | text identical every arm/round; default oracle at this point: 40.521 ms/token |
| uniform buffer cache bounded with LRU (ROW 279) | not a perf change | `UNIFORM_BUFFERS` grew unbounded on varying uniform bytes; bounded at `[spans].uniform_cache_entries=4096`; `uniform_cache_len` plateaus at 50 on the default decode |
| plan-stable device buffers, default flip (ROW 280) | quiet 3-round table: OFF 40.827/40.090/41.107 vs ON 39.688/37.476/40.780 (ON ≤ OFF every round) | `op_setup` fell 3.94-5.42 → 0.39-0.60 ms/step; `OUTPUT_BUFFER_ALLOCATIONS` 842/step → 0 on plan-cache hits; device bytes flat (~4.163 GB OFF, ~4.153 GB ON); becomes the default |
| fused `CachedAttention` reaches the single-range program (ROW 281, `2f6f12d`/`15c469b`/`563fc0c`/`a943390`); flipped default-on (ROW 282, `land/fattn-on`) | quiet bake-off, 3 rounds: unfused 37.096/37.506/38.786 vs fused 39.796/36.363/37.154 (means 37.796 vs 37.771, CoV 2.33%/4.76%) | delta is inside both arms' noise, not a confirmed wall win; `emit_calls` 938→616/step, per-op GPU sum 35.266→32.356 ms (-8.2%) localized to the fused bucket, device bytes -131,072 B, text identical; ROW 282 flips the feature into `metal`'s default per the owner's less-work rule (output identical, work down, wall not worse beyond CoV) even without a confirmed wall win |
| cooperative K/V loads inside the fused attention kernel, default-on (ROW 283, `4d5bb08`/`7b7defe`) | quiet bake-off, 3 interleaved rounds, contended box, no overlap: `gpu_exec_ms` 30.643 → 29.300 (-4.4%, `discipline.md:20857`) | one threadgroup per `(query_row, kv_head)`, K/V rows loaded once into threadgroup memory (`discipline.md:20836`); wall neutral within CoV (`discipline.md:20857`); parity `relative=5.29e-7` at kv-capacity-bucket paddings 0/1/5 (`discipline.md:20838`); text identical; landed default under the owner's less-work rule (`discipline.md:20867`) |
| placed outputs skip post-wait readback, default (ROW 284, `aa7905d`/`04360a3`/`65c2117`) | not a decode-time perf change; no bake-off run | a latent buffer-offset bug fixed first (`discipline.md:20891`); `readback_calls` 97 → 1 and `readback_bytes` 390,152 → 128,008 per steady step (step 0: 12,094,712 → 3,968,248, `discipline.md:20897-20898`); text identical; landed default -- post-wait memcpy removal makes a wall-clock regression structurally impossible (`discipline.md:20903`) |

`generated_text` identical on every arm across every row above; device bytes stayed within the
~4.15-4.17 GB band on every measured arm.

Concurrent dispatch with dataflow barriers (`metal-concurrent-dispatch`, ROW 285): quiet 3-round
bake-off, `gpu_exec_ms` mean 29.267 → 27.258 ms (-6.9%), `step_wall_ms` mean 36.556 → 34.725 ms
(CoV 8.3%/8.2%), 419 barriers over 616 ops/step, `generated_text` identical across 6 ON + 3 OFF
runs; flips `metal`'s default feature list on (owner's less-work rule: work down, wall not worse
beyond CoV).

What is left after the third wave, per token on the default at main `af918bb` (ROW 286, includes
`metal-concurrent-dispatch`, corrected to steady state by ROW 288): steady-state `step_wall_ms`
(steps 3..7 only, `plan_hits` nonzero and rising every step) 28.82 ms/token, mean of 3 rounds
28.542 (CoV 0.70%), against `llama-bench` 57.32 t/s mean (17.445 ms/token, CoV 0.55%) —
**1.652x** (1.636x on the 3-round mean). ROW 286's steps-1..7 mean (33.032 ms/token, 1.8935x) is
retired: steps 1-2 are plan misses (bind + pipeline compile, ~45/43 ms) that `llama-bench`'s
`tg32` never pays, so the two numbers were not measuring the same thing (`discipline.md`, ROW
288). Progression across all three waves, ms/token (steps-1..7 convention, pre-correction; the
final entry is superseded by the steady-state figure above): 67.9 → 52.7 → 43.0 → 40.9 → 40.5 →
~37.8 → 33.0 (steps 1..7, retired) / 28.82 (steps 3..7 steady state, ROW 288).
`cached-attention-streaming` joins the `metal` default feature list (ROW 282, `land/fattn-on`)
on top of this same tree by the owner's less-work rule -- `emit_calls` 938 → 616/step, text and
wall unchanged within CoV -- so the default's feature set as of `land/fattn-on` is `metal`'s
full list (`metal-output-placement`, `metal-wide-cooperative-reduce`, `kv-capacity-bucket`,
`metal-q5k-pair-dot`, `metal-plan-stable-buffers`, `cached-attention-streaming`), not just the
subset named above. Two more work metrics drop on the same tree without a numbers change here
(quiet re-measure pending): `gpu_exec_ms` 30.643 → 29.300/step (ROW 283) and readback 97 → 1
copies/token (ROW 284, `discipline.md:20897`).

## 1.3 2026-09-04 reframing: the floor, the bytes, and the audit

**Owner rules, verbatim, 2026-09-04** (session record; the less-work rule is already applied at
ROWs 282/283/285):
- "if we reduce the amount of work, even if that doesn't seem to move the wall clock, we should
  keep it assuming the quality is the same"
- "llama is not the floor"
- "the hardware hit like 400GB/s"
- "realistically, I'd love to 5x llama" / "that would require something insane and we'd need to
  change the problem"
- "pipe shaped, fsm x sansio + fsm x orchestration over pipes and also I want you to make sure
  that we are using our risc architecture and algebra. it should be _generic_"
  (`design-task.md:3-7`)

These replace "close the gap to llama.cpp" as the standing target: llama.cpp is one measured
incumbent, not the floor; the floor is bytes moved against the device's spec bandwidth.

### 1.3.1 The floor

| lane | ms/token | GB/s | % of 400 GB/s spec | provenance |
|---|---|---|---|---|
| hardware floor (4.169 GB/token at 400 GB/s spec) | 10.4 | 400 (spec) | 100% | DERIVED, `design-task.md:92`; bytes/token from the ROW 272 correction, `discipline.md:312` (`total_operand_bytes` 4.169 GB/step) |
| llama.cpp b25346221 | 17.45 | 239 | 60% | MEASURED, `discipline.md:333-336` (17.445 ms/token, CoV 0.55%) |
| default, main `af918bb` (ROW 286, steady-state correction ROW 288) | 28.82 | 145 | 36% | MEASURED, ROW 288 (`step_wall_ms` steps 3..7 only, `plan_hits` nonzero every step, 28.82 ms/token r3 / 28.542 mean of 3 rounds); supersedes the steps-1..7 mean (33.032 ms/token, CoV 0.52%) previously cited at `discipline.md:333-334` |
| matvecs alone, in-buffer ablation (ROW 287, landing) | 23.24 | 179 | 45% | MEASURED, `audit-2026-09-04.md:22` pointer; ALL 27.04 ms / MATVEC 23.24 ms (225 ops) / NOT-MATVEC 4.05 ms, residual −0.25 ms |

Owner target: "5x llama" = 3.5 ms/token. At the 400 GB/s spec that is ≤ 1.4 GB/token; at a
realistic 300 GB/s ceiling, ≤ 1.0 GB/token (`design-task.md:90-93`). 3.5 ms is below the 10.4 ms
hardware floor for the current 4.169 GB/token payload — the floor moves only if the payload
does. This makes the standing work a bytes campaign, not a kernel campaign: multi-token passes
(draft/verify amortizing the weight stream across k generated tokens), dynamic row elision /
contextual sparsity on the FFN and projection matvecs, lower-bit codecs (Q4 → Q3/Q2/ternary), and
`output.weight` (107.5 MB) top-k or tying. Each lever is lossy; `generated_text` byte-identical
is not an available gate past this point. The gate becomes a quality metric (exact-match rate
against the full model on a held-out prompt set) with a stated kill criterion, evaluated per
lever, before the lever's bytes saved are counted (`design-task.md:103-106`).

### 1.3.2 Dispatch census: 19 bound ops/layer, 616/token, why each stays separate

Source: `proxima-tensor/src/spec.rs:3517-3921` (68 raw RISC ops/layer folded by
`BoundOpBuilder`), reconciled to 616 on `af918bb` (`dispatch-census.md:1-3`).

| # | node | kind | why separate |
|---|---|---|---|
| 1 | sum_squares (attn norm) | reduce-cooperative | reduce boundary, reduces never fuse (`spec.rs:12700`) |
| 2 | normed | elementwise (6-op chain) | consumes a materialized reduce; broadcast s→sd |
| 3-5 | q, k_new, v_new | reduce-packed-row-blocked | matvec |
| 6-9 | rotated_{q,k}_{even,odd} | elementwise (3-op RoPE) | `is_identity_projection` fails on the 2i/2i+1 stride and the GQA group map `h=group*u+g` (`bind.rs:1153-1164`) |
| 10 | attended | cached-attention | matcher `bind.rs:2295-2517` |
| 11 | attn_out | reduce-packed-row-blocked | matvec |
| 12 | residual1 | elementwise | `StillLive`: x read twice (`bind.rs:701,778`) |
| 13 | sum_squares (ffn norm) | reduce-cooperative | as 1 |
| 14 | normed2 | elementwise | as 2 |
| 15-16 | gate, up | reduce-packed-row-blocked | matvec |
| 17 | ffn_hidden (SwiGLU) | elementwise (6-op) | `quarantine_broadcast_operands`: child extent 14336 < reduce extent 14336x4096 (`bind.rs:935-973`) |
| 18 | ffn_out | reduce-packed-row-blocked | matvec |
| 19 | x_next | elementwise | `StillLive` |

Tail: 2 iota, 1 mask elementwise, final norm (1 coop + 1 elementwise), lm_head matvec, 2
constants. Totals per token: reduce-cooperative 65, packed-row 225, cached-attention 32,
elementwise 290, constant 2, iota 2 (`dispatch-census.md:23`). Measured per-op profile (ROW
281/282, serialized command buffers, biased high for small ops): cached-attention 3.515 ms,
elementwise 2.978 ms, reduce-cooperative 0.851 ms, packed-row 24.991 ms (`dispatch-census.md:26-27`).

llama.cpp b25346221 (zero fusion on ggml-Metal at this checkout): 23 ggml ops/layer × 32 + tail
= 740 dispatches/token. We already dispatch fewer (616) with more work fused per dispatch
(`dispatch-census.md:29-30`).

Four fusion levers, stated as generic rules over the algebra, not model-specific matchers
(`dispatch-census.md:32-49`):

| lever | rule | removes/layer | removes/token | blocked by |
|---|---|---|---|---|
| epilogue fusion | an elementwise consumer of a Reduce whose iteration space equals the reduce's output space, with no other consumer, becomes the reduce's epilogue | 3 (residual1, x_next, ffn_hidden) | 96 | `BoundOpBuilder` only fuses elementwise INTO reduce operands (prologue), never out of them |
| prologue-with-broadcast | relax `quarantine_broadcast_operands` for a per-row recompute cheaper than a materialization + dispatch | 2 (normed, normed2) | 64 | needs a cost bound comparing recompute against materialization |
| RoPE fusion | fuse even/odd into one dispatch, then into the q/k matvec epilogue | 2 (one dispatch) / 4 (epilogue) | 64-128 | stride/group-map identity rule; `CachedLayerRoots` needs even/odd as separate outputs, consumed at 15+ sites (`spec.rs:2333`) |
| reduce-with-broadcast epilogue | a two-phase threadgroup op: reduce then normalize in one dispatch | 2 (sum_squares ×2) | 64 | needs an IndexMap-aware epilogue, the same extension as epilogue fusion generalized |

Ceiling if all four land: 19 → 8/layer (7 matvec + 1 attention) = 264/token (`dispatch-census.md:49`).

### 1.3.3 MSL kernels against ggml: five structural differences, everything else identical

Read on main `3b9735e`: dispatch shape, buffer binding, activation loads, uniform hoisting, and
epilogue structure are structurally identical to ggml's Metal kernels. Five differences remain:

1. per-iteration 64-bit address recompute inside the row loop, not hoisted (`omega/src/msl.rs:3363,3374`)
2. two-`uchar` word loads where ggml loads one packed word (`omega/src/msl.rs:275-276`)
3. no fast arm for Q6_K; every element goes through the general decode path (`omega/src/msl.rs:473-493`)
4. `MTLMathMode::Safe` for parity (`omega/src/metal.rs:2339`) against ggml's fast-math default
5. `other_stride` read at runtime per dispatch instead of baked into the generated source (`omega/src/msl.rs:3256`)

### 1.3.4 Audit, main `3b9735e` (`audit-2026-09-04.md`, condensed, ranked by blast radius)

Status column: FIX = a worker is fixing it on a named branch (§1.3.5); DESIGN = goes to the
design tournament (§1.3.5); OWNER = the less-work rule already decides it.

| # | finding | file:line | status |
|---|---|---|---|
| 1 | `BoundOpKind::CachedAttention` is a 10-field macro-op (attention semantics), not Op/ScalarOp/IndexMap structure; any other layout silently falls back | bind.rs:225-251 | DESIGN A |
| 2 | `render_cached_attention` ignores operand `Layout.strides`; the matcher compensates with 8 literal stride tuples + rank gates | msl.rs:2578; bind.rs:2417-2474 | DESIGN A |
| 3 | HazardTracker checks output identity from `placement` but records from `device_buffers` → WAW/WAR skipped when placement is None; `hazard_inputs` drops missing operands; stale pointers (ABA) after retire | metal.rs:1103 vs 1121; 1095-1100; 1131 | FIX fix/hazard-identity |
| 4 | Hazard tests drive a hand-written copy of the loop (`encode_with_barrier_bookkeeping`) | metal.rs:4783-4795 | FIX fix/hazard-identity |
| 5 | Single-range fused kernel runs 2t iterations for t of work (cached_lower=i64::MAX, duplicated operands 4/5/7) | bind.rs:2504-2510; msl.rs:2586 | DESIGN A |
| 6 | `operands.len() == 8 \| 9` is a runtime state discriminator agreed across 4 files | bind.rs:237; cpu.rs:4847; msl.rs:2030,2543 | DESIGN A |
| 7 | `classify_kind` substring-greps generated MSL to recover the emitter's routing; profiler groups by &str | metal.rs:1631-1727 | DESIGN B |
| 8 | `RMS_EPSILON = 1e-5` hardcoded, ignores checkpoint metadata (qwen35/lfm2 read it) | generate.rs:115; bind.rs:3306 | FIX fix/decode-path-correctness |
| 9 | Nine executor entry points = one driver × {named, placed, timed}; upload loop copy-pasted; none a Pipe | metal.rs:519..1613 | DESIGN B |
| 10 | Per-token allocation storm in the decode closure; no allocation-counter test | generate.rs:2301-2306,2323,2361,2400,2504; metal.rs:1032,1073,1095 | DESIGN B |
| 11 | `plan()` performs device IO (device_and_queue, arena, uniform buffers); `Plan` two-phase init via `mark_resident`; no builder/config surface | metal.rs:490-500, 373 | DESIGN B (arena scoping: FIX refactor/plan-cache-and-arena-scope) |
| 12 | Arena built for every plan, used by 2 of 4 executors | metal.rs:490-500, 643-651 | FIX refactor/plan-cache-and-arena-scope |
| 13 | Plan cache = one-entry map cleared on miss, copied ×3; `PlanCacheEntryVanished` is an impossible-state error | generate.rs:1417,1325,1373 | FIX refactor/plan-cache-and-arena-scope |
| 14 | `cached_len_before` printed after the increment (both loops) | generate.rs:2049/2076, 2483/2510 | FIX fix/decode-path-correctness |
| 15 | Op-timed profiler measures one-command-buffer-per-op, not production; decompositions built from it measure a different program | metal.rs:1509 | superseded by the in-buffer ablation (ROW 287) |
| 16 | println!/eprintln! and per-step env reads in library code | metal.rs:1072-1082,3567,3617; generate.rs:135..2132 | FIX chore/metal-path-hygiene |
| 17 | Arena cap prints "MG-3 KILL" and returns Ok | metal.rs:3617-3626 | FIX fix/decode-path-correctness |
| 18 | ggml Q4_K port not in THIRD_PARTY.md | msl.rs:3728-3734 | FIX chore/metal-path-hygiene |
| 19 | Bare tunables: ARENA_TRANSIENT_CAP, OUTPUT_POOL_MAX_PER_BUCKET, PACKED_ROWS_PER_GROUP, PACKED_ROW_NSG, TILED_GEMM_NSG, OP_PROFILE_TOP_N | metal.rs:3481,2474; msl.rs:1261,4433,1290; generate.rs:120 | FIX chore/metal-path-hygiene |
| 20 | `KV_BUCKET_TOKENS` (Metal cache-key policy) lives in the IR crate | proxima-tensor/src/sized.rs | FIX refactor/plan-cache-and-arena-scope |
| 21 | Seven `unreachable!` in production render/pack paths | metal.rs:2199,2259,2302; msl.rs:1110,1143,1378,2527 | FIX chore/metal-path-hygiene |
| 22 | `fuse_cached_attention: bool` collapses "which fused kinds"; bind_plain + matchers run twice per plan | bind.rs:2635-2683 | DESIGN C |
| 23 | Model-named program builders in library crates; TOML spec path exists and production ignores it; `CachedLayerRoots` positional; 23-arg builders; `cached_len` as Float32 | spec.rs:898..7184, 2333, 3517; generate.rs:624,731 | DESIGN C |
| 24 | `finish` silently drops placed outputs from `Evaluated` | metal.rs:3985-3991 | FIX fix/decode-path-correctness |
| 25 | Eight `thread_local! RefCell` globals + `register_checkpoint_mapping` side channel; counters' "exactly once per step" protocol unenforced | metal.rs:266..3266, 2957, 2719 | DESIGN B |
| 26 | HazardTracker generic `Id` + hand `Default` exist only for the mirror test | metal.rs:839-857 | FIX fix/hazard-identity |
| 27 | `metal` is a super-feature carrying 7 experiments; 13 `metal-*` flags with no matrix gate; cross-feature correctness in prose | omega/Cargo.toml; interop Cargo.toml:94-104 | OWNER (less-work rule keeps them default-on); matrix gate = open |
| 28 | Positional/type-unsafe: `CachedLayerRoots` tuple, 23 NodeId params, 25+ `allow(too_many_arguments)` without why, `cached_len` Float32, per-token `zip` without re-validation | spec.rs:2333,3517; metal.rs:1039 | DESIGN C |
| 29 | Barrier policy: reset both sets on any barrier, scope = all buffers; schedule is static per plan but recomputed per token; `memoryBarrierWithResources` exists | metal.rs:874-877, 1105 | measured card after the hazard fix lands |
| 30 | `read_back` int dtypes reinterpreted as f32; `zero_placed_buffer` memsets full capacity on step 0; matcher operand lookup is a linear scan ×9 ×2 binds; `PlanUniforms` unsafe write invariant in prose only; `BufferArena::placement_for` parallel-array invariant in a comment | metal.rs:3898-3920, 3679-3694, 3525; generate.rs:2226 | FIX (read_back) fix/decode-path-correctness; rest DESIGN B |

**Not a pipe** (central-claim lint, `audit-2026-09-04.md:39-43`): every
executor/encode/finish/read_back/prepare/plan/arena/uniform/upload function, `bind_plain`/
`bind_with_fusion` and both matchers, every `render_*`/`push_*_body`, `kernel_dispatch_shape`,
`classify_kind`, `run_decode_loop*`, `decode_until_stop_or_budget`, `build_position_inputs`,
`sample_next_token`, `report_op_timings`, `print_token_breakdown*`. The one `Pipe` impl on the
path is `LoadedModel::call` (`generate.rs:1533`).

**Hidden state machines, none an enum** (`audit-2026-09-04.md:45-49`): `HazardTracker`; the
encode loop; `BufferArena` construction; `OUTPUT_BUFFER_POOL` lifecycle; `UNIFORM_BUFFERS` LRU;
`Plan` two-phase init; the one-entry plan cache; the operand-count discriminator;
`dynamic_cached_len`; band sentinels `{MIN, MAX, real}`; the decode step closure; the
snapshot-and-reset counter protocol. Counter-example that the shape exists:
`LayerCacheState`/`LayerCacheNames` (`generate.rs:1090,1100`) are enums.

### 1.3.5 In flight

Nine branches, each addressing one or more findings above: `fix/hazard-identity` (findings 3, 4,
26), `fix/decode-path-correctness` (findings 8, 14, 17, 24, 30 read_back), `chore/metal-path-hygiene`
(findings 16, 18, 19, 21), `refactor/plan-cache-and-arena-scope` (findings 11 arena scoping, 12,
13, 20), `perf/reduce-epilogue-fusion`, `perf/packed-row-addressing`, `feat/decode-quality-harness`,
`perf/device-streaming-ceiling`, `docs/thread-envelope`.

The design tournament (`design-task.md`, sections A-E) covers what a branch cannot: A (attention
as RISC algebra, replacing `BoundOpKind::CachedAttention`, findings 1, 2, 5, 6), B (the decode
step as FSM × orchestration over pipes, collapsing nine executors to one driver, findings 7, 9,
10, 25, 30 non-read_back), C (generic model programs off the TOML spec path, findings 22, 23,
28), D — primary, per §1.3.1 — bytes (multi-token passes, dynamic row elision, lower-bit codecs,
`output.weight` tricks, KV bytes), E (ordering, gates per card, and at least one design abandoned
per constraint). Output: a stepwise plan with signatures, ≤ 1500 lines, every claim citing
`file:line` opened this session, no adjectives, no verdict words (`design-task.md:71-114`).

## 1.4 Design of record (2026-09-04)

Full trail and design of record: `design-2026-09-04/` in this directory (`README.md` indexes the
round-1 and round-2 tournaments, the judge scores, and the stop decision).

The tournament answers three owner rules, quoted verbatim: "pipe shaped, fsm x sansio + fsm x
orchestration over pipes and also I want you to make sure that we are using our risc architecture
and algebra. it should be _generic_"; "omega should support cpu, gpu and _mixed_ backends"; "how is
there any more than 2 backends?"

Round 1 (design-A, design-B, design-AB) selected design-AB, unanimous across three judges under
three different anonymizations. Round 2 (design-AB, design-B2, design-AB2) selected design-AB2,
unanimous across three judges. `design-final.md` is design-AB2 with the round-2 panel's named holes
closed against the shipped source at `HEAD ce05362`.

The first five cards, ordered on the measured gap (`design-final.md` §E): D0a repeats the device
streaming ceiling on a quiet box; D0b adds a ceiling arm at the Q4_K superblock stride; D0c measures
concurrent CPU+GPU streaming against the GPU-alone ceiling; D1 is the matvec roofline ladder; D1b is
the packed-row addressing arms.

`Backend`'s seven variants collapse to `Engine::{Cpu, Gpu}` plus a `GpuDriver` resolved once per
target; there is no `Backend::Mixed` variant — mixing is a placement field on the scheduled op, not
a third engine.

<!-- measured 2026-09-04: k' --> Draft acceptance is now measured on real greedy Metal streams
(n-gram prompt-lookup, `test/draft-acceptance-harness`): mean k' = 1.36 at k=4 and 1.49 at k=8, both
below the design's own `A < 1.5` multi-token kill criterion (`design-final.md` §D.4, §D.4a, card D2).

## 2. The diagnosis, built formally (V0-V8)

The default is no verdict. What follows is a proposal built by the admissibility procedure so
the owner can audit it line by line and decide. Status per claim is stated; nothing below is
carried forward as settled.

### V0 — propositions (three, each falsifiable, every field bound)

- **P1 (graph).** On main `4be2f3a`, the Llama-arch cached-decode graph `mistral_cached_forward_program` emits ≥1194 `BoundOp`s per token against the incumbent's ~740 real dispatches (ggml b25346221, same model, same silicon), and the excess is attention: two-range online-softmax over `(cached, new)` K/V because `Reduce::out_map` must be a pure projection, so K/V cannot be written in place into a persistent device buffer.
- **P2 (kernel).** On main, the Q4_K row-blocked matvec body performs ~6x the ALU operations per 8 weights of `kernel_mul_mv_q4_K_f32_impl` (explicit shift+mask+cast vs mask-and-fold-into-scale), branches on `sub_block < 4` inside a simdgroup, and the cooperative reduce launches 32 threads for a 4096-wide row where the incumbent launches up to 1024; together these hold `gpu_exec_ms` at ~57 ms/token where a body-only rewrite measures ~35-47 ms (two independent rewrites, R3/R12).
- **P3 (lowering).** On main, the GPU lowering is three separately hand-written emitters sharing 3 types + 2 functions; the route a `Reduce` takes is decided by hand-ordered gates inside `push_cooperative_reduce_body` and is recoverable only by grepping emitted MSL; geometry constants are bare source consts; CUDA rejects 2 of 4 `BoundOpKind`s. This is not a per-token cost; it is why P1/P2 fixes are slow to land, cannot be censused, and diverge across backends.
- **P4 (orchestration).** On main, `cached_len` is `Extent::Symbolic(1)` on every KV leaf, the plan-cache key is `(new_count, cached_len)`, and `ff749a0` clears the cache on every miss — so every token re-runs `plan_named` (infer+bind+pack+retire), `mark_resident`, and per-op `allocate_buffer`+`upload_uniforms` for ~1196 ops, costing the `prepare`+`op_setup` slices (MEMORY 2.1 + 4.4 ms/token; parallel-branch `prepare` 11.6 ms with its matcher).

### V1 — refutation conditions (written before the evidence was read)

- P1 false if: main's per-token BoundOp count is within 10% of the incumbent's, OR the attention ops per layer are ≤ 8, OR `Reduce::out_map` already accepts a non-zero write offset.
- P2 false if: a body-only rewrite (no geometry change) moves `gpu_exec_ms` by < 5% on the real graph, OR the incumbent's body performs a comparable op count per 8 weights.
- P3 false if: a single route decision exists as a value before emission, OR wgsl/cuda share the render functions with msl.
- P4 false if: `plan_hits > 0` on steady tokens, OR `prepare_ms` + `op_setup_ms` < 1 ms/token on main.

### V2 — artifact register (every artifact opened this session, by `file:line`)

| # | artifact | what it showed |
|---|---|---|
| A1 | `proxima-tensor/src/op.rs:175-266` | `Op` has 5 variants (doc at :166 still says "four"); `Reduce` at :153-164 carries `in_map` and `out_map: IndexMap` — "Addresses the result. Data-dependent here is what makes a scatter." |
| A2 | `proxima-tensor/src/map.rs:1-60,134-152` | `IndexMap::{Affine(IndexPattern), Computed{..}}`; slice = non-zero `offset` on the read side (doc table :8-15). |
| A3 | `proxima-tensor/src/bind.rs:82,95-98,127-134,160-192,200-264` | `BoundOp{node,dtype,extents,kind}`; `BoundOpKind` 4 variants; `Reduce{.., out_layout: Layout{base,strides}, out_scatter: Option<Lookup>}` — a write base offset exists at the bound level. |
| A4 | `proxima-tensor/src/spec.rs:2303-2319, 2336-2362, 2616-2617, 6216-6245` | the cached layer (530 lines, 25 args); "`Reduce::out_map` must stay a pure projection ... so nothing upstream of a reduce can splice two tensors into one axis"; "no literal concatenation anywhere"; `cached_len` = `Extent::Symbolic(1)` on `kv_cache.{layer}.{k_even,k_odd,v}`. |
| A5 | `proxima-model-interop/src/generate.rs:621-654, 880-907, 958-977, 1175-1181, 1298-1649` | `LayerCache` grows by `extend_from_slice`; plan cache `BTreeMap<(usize,usize),Plan>` keyed `(symbols[0],symbols[1])`; `plans.clear()` on every miss (:973, from ff749a0); the per-token call sequence. |
| A6 | `omega/src/metal.rs:327-363, 449-568, 785-826, 835-854, 859-870, 945-1000, 1402, 1744-1815, 1848-1892, 2179-2252` | `mark_resident` by name; `execute_plan` one encoder/commit/wait; `classify_kind` by MSL substring; `diagnose_kind`; `Prepared`; `prepare` strict `found != expected`; `pipeline_for` cache; checkpoint mapping (7d09145); `NOCOPY_BUFFERS` keyed `(ptr,len)`; `encode_op` allocating output + uniforms per op per token. |
| A7 | `omega/src/msl.rs:673-697, 824, 1017, 1030, 1046, 2178-2257, 3140-3194` | `emit` routing; `reduce_is_cooperative`; `PACKED_ROWS_PER_GROUP=4`, `TILE_DIM=8`, `TILED_GEMM_NSG=4` as bare consts; serial/cooperative split; tiled/packed/generic split — 8 body shapes. |
| A8 | `omega/src/sized.rs:1-45`, `omega/omega-runtime.toml` | `SIMD_WIDTH=32` hardware fact; only `[tiled_gemm]` sizing exists in the toml. |
| A9 | `omega/src/cuda.rs:146-183`, `omega/src/wgsl.rs:105`, `omega/src/cuda.rs:66`, `omega/src/lib.rs:57-63` | CUDA rejects `Iota`/`Constant`; the shared surface is `Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots`. |
| A10 | `omega/src/backend.rs:1-52, 402-414` | six backends, two implemented; plan/execute not a pipe (relocation test failed 2026-08-30); `register_checkpoint_mapping`. |
| A11 | `omega/Cargo.toml` (full) | features: default/std/alloc/cpu/metal/vulkan/cuda/npu/ane/instrument/metal-tiled-gemm/wgpu-backend. No `metal-q4k-*`, no `metal-buffer-pool`, no `metal-output-placement`. |
| A12 | `proxima-tensor/src/instrument.rs:809-828, 848-864, 1430-1435`, `lib.rs:213-214` | `WidthDeclineReason` (8), `record_width_tile_decline` keyed `(node, reason)`, `Path` (4); the census pattern to mirror. `FuseSite`/`FuseDeclineReason` do not exist on main. |
| A13 | `proxima-tensor/src/align.rs:33-46,69` + grep | `AlignedBuffer::new(min_elements, page_size)`; zero production callers. |
| A14 | `proxima-tensor/docs/discipline.md` (18766 lines; ROW 193 :17258-17338, ROW 221 :18390-18430, ROW 223 :18460-18487, last ROW 233 :18736) | main's log ends at ROW 233; no `metal-q4k-*`, `17.470`, `3.54x`, `RISC` strings anywhere. |
| A15 | `proxima-tensor/docs/rooflines.md:396-479, 751, 766-773` | GPU lane ceiling = DEBT; the only ratio is vs incumbent achieved. |
| A16 | `proxima-tensor/docs/rewrite-algebra.md:1-120, 389-519, 540-694` | six laws, fixpoint engine spec (not built); band-streaming named as "later"; "5 kernels/layer" is a derivation, PROPOSED. |
| A17 | `git log 2b95210..main`, `git diff --stat` per branch, `git status` per worktree (R0, R6, R7) | 9 commits on main since 2b95210; the 2026-09-02 wins are uncommitted diffs on 2b95210 in 10 worktrees; a 42-commit branch on today's main from another agent. |
| A18 | `git show perf/cached-attention-streaming:proxima-tensor/docs/discipline.md` ROWs 234-267, `physical.rs:1-140`, bind.rs/msl.rs/Cargo.toml diffs, `failure-cached-attention-matcher.md` | a new `BoundOpKind::CachedAttention`, a structural matcher, a paired-nibble Q4_K body, and 12 measured negatives. |
| A19 | `~/repos/others/llama.cpp` @ b25346221: `ggml-metal.m:1835-1847, 1928-5190, 2497-2615, 3215-3220, 3327-3330, 3787-3823, 3934-4047, 4462-4469, 5216-5289, 5783`; `ggml-metal.metal:1051-1145, 1679-1721, 5086-5206`; `ggml-metal-impl.h:32-33`; `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`; `llama-model.cpp:4691-4845, 14250-14273`; `llama-kv-cache-unified.cpp:74-118, 715-788`; `common/common.h:328` | the incumbent's per-layer op list (23), no fusion at this checkout, KV cache written by `ggml_cpy` into a view at a byte offset of a persistent device tensor, `soft_max` fused in one kernel, `rms_norm` at up to 1024 threads, Q4_K body, `n_cb=1`, flash-attn off by default, RoPE NORM. |
| A20 | `omega/benches/metal_vs_cpu.rs:1-44, 232-326`; grep over all benches/tests for metal/wgpu | the only non-decode GPU arm; mnist/BGE/train have none. |
| A21 | `proxima-onnx/scripts/torch_reference/venv` python probes | torch 2.13.0, MPS available; onnxruntime not installed. |

### V3 — both directions

**For P1.** A4 names the constraint in the source's own words; A19 shows the incumbent writes K/V in place (`ggml_cpy` into a view at `row_size*head_cur`) and emits 23 ops/layer; R12's branch reaches 616 dispatches by fusing exactly that cluster. **Against P1:** halving the dispatch count (1194 -> 616) moved wall by 0% and GPU by +13% in R12 (ROW 262 vs 267) — so the dispatch COUNT is not the cost; H2 (per-dispatch fixed cost) was refuted at 3.6% agreement of Σper-op vs batched (MEMORY). The excess is real work (extra reduces at 32 threads, extra elementwise passes), not launch overhead. P1's mechanism stands; its MASS is the per-op work of the extra ops (reduce-cooperative 8.6 + elementwise 6.65 = 15.4 ms MEMORY), not the count.

**For P2.** Two independent rewrites of the body measured -29% (R12 ROW 257) and -36% (R3 mask-fma), both with parity ≤ 3.1e-6 relative on real weights; A19 quotes the incumbent's 8-op inner loop. **Against P2:** the rising-marginal-with-simdgroups table (R3 M5) says low-row shapes are starved regardless of body — a second mechanism (occupancy) the body rewrite does not touch; and the parallel branch's GPU time went UP 35.1 -> 39.8 when its macro-op was enabled, so the body is not the only mover. P2 stands as the LARGEST single term, not the only one.

**For P3.** A7/A9 are direct reads. **Against P3:** it costs zero ms per token; the plan must not schedule it ahead of P1/P2 on mass. Its justification is the owner's directive ("one RISC") and the observed cost of the current shape: two agents rewrote the same Q4_K body twice in two days; a classifier mislabeled 216 ops (R12 ROW 263); nsg=2 was tried four times.

**For P4.** A5 :966 + A4 :6216-6245 are a direct causal chain: key contains `cached_len`, `cached_len` increments per token, therefore miss per token. **Against P4:** 7d09145 already removed per-token WEIGHT upload (A6 :1744-1815) and ff749a0 bounded the leak; `prepare` on main may be lower than MEMORY's 2.087 ms — today's seal measured 1.97 and card 0.5 replicates it. The incumbent avoids the class entirely by padding n_kv to 256 (A19 `ggml-metal.m:2508` `nth*ne01...<256` is the softmax shape; the KV padding lives in llama.cpp's kv-cache sizing, not read this session — marked as a residual below).

### V4 — mechanism per row

- A4 -> P1: `project_output_shape` requires `out_map` to be a projection; a concat needs two producers writing disjoint offsets of one buffer. The bound level already has `out_layout.base` (A3); the block-input check requires `found == expected` (A6 :991-1000), so an over-allocated cache buffer is rejected; the driver's no-copy cache is keyed on `(ptr, len)` (A6 :1848) so a growing Vec misses. Three lines, one root: no notion of "a caller-owned persistent output buffer written at an offset".
- A7 -> P2: `push_packed_row_blocked_body` (msl.rs:2425 region, MEMORY :269-285 for `q4k_run8`) emits shift+mask+cast per element and a separate accumulate loop; ggml folds (A19 `:5158-5175`). `reduce_is_cooperative(..).then_some(SIMD_WIDTH)` caps threadgroup width at 32 (MEMORY :3133; card 7.2 re-binds the line).
- A5 -> P4: `resolve_plan` :966-975 is the whole mechanism; `encode_op` :2210-2211 is the per-op allocation.
- A9/A7 -> P3: direct structure.

### V5 — adversarial pass + degenerate control

- **Correctness lens.** Every kernel change in this plan gates on real-checkpoint parity (`omega/tests/q4k_real_checkpoint_parity.rs` exists on the parallel branch; main has `omega/tests/metal_real_forward.rs` and `backend_parity.rs`) normalized to BATCH PEAK, never per-row relative (MEMORY: the 872% artifact).
- **Reproduction lens.** Today's seal (§1.1) ran both arms interleaved, on a loaded box with the loadout recorded, N=3 each plus the per-op profile; card 0.5 replicates it on fixed instruments; the MEMORY numbers are not used as a baseline for any delta.
- **Wrong-thing-measured lens.** `gpu_exec_ms` vs `step_wall_ms`: the headline is `step_wall_ms` (MEMORY: five wins were measured on components and the headline never moved). Every card's `expect` names the wall cell.
- **Degenerate control.** The per-op profile's zero-byte `constant`/`iota` ops (MEMORY ~4 us each) are the floor control; a kernel change that "improves" those is measuring the harness. Today's seal recorded them (37 constant / 2 iota ops, 4,352 / 4,208 ns each); card 0.5 records them again.

### V6 — residual

- The llama.cpp KV padding granularity (n_kv rounding) was not read this session (A19 covers the cache write, not the sizing); the plan's "capacity bucket" size is chosen by measurement in card 6.3, not copied.
- `ffn_down`/`attn_v` 0.5781 B/weight vs 0.5625 (rooflines.md:461-479) has a candidate explanation, not a confirmed one: a Q4_K_M checkpoint carries Q5_K/Q6_K for some `ffn_down`/`attn_v` layers, so a per-family constant is wrong by construction; card 0.2 recomputes bytes per tensor from the codec the GGUF header declares and the gap is either explained or RED. It does not flip any proposition.
- The device slope of +1-2 MB/token has a candidate mechanism found by the round-4 critique and not yet measured: `UNIFORM_BUFFERS` (`omega/src/metal.rs:2056-2078`) is a content-keyed, unbounded map and every `Uniforms` blob carries `reduction_total`, a function of `cached_len`, so on main the key changes every token and the map inserts forever. Card 0.2 adds the `UNIFORM_CACHE_LEN` gauge and pre-registers growth ≈ `op_count` per token; if the gauge is flat, the mechanism is wrong and 6.5/9.2 re-scope.
- `packed_operands_of` (`omega/src/metal.rs:375`) returns `PackedOperands = BTreeMap<NodeId, PackedCodec>` (`omega/src/msl.rs:656`, `:591`), both omega types; the earlier idea of moving it into proxima-tensor does not type-check and is unnecessary because `correct_packed_matmul_layouts` already takes `&BTreeSet<NodeId>` (`proxima-tensor/src/bind.rs:1648`) — card 2.2 is a signature change on `bind`, not a relocation.
- The GPU bandwidth ceiling is unmeasured; every "GB/s" here is a rate against the incumbent's achieved number, not against physics (card 0.6).
- Whether `prepare`/`op_setup` on main today are the MEMORY values was unknown when this register was opened; today's seal answered it (1.97 / 3.90 vs MEMORY 2.087 / 4.394) and card 0.5 replicates it on fixed instruments.

### V7 — status

P1 **plausible** (mechanism read in source both sides; mass attribution rests on MEMORY per-op buckets pending re-seal). P2 **plausible-strong** (two independent measured rewrites agree on direction and magnitude; occupancy is a second unmeasured term). P3 **proven as structure** (direct reads; no measurement claimed). P4 **proven as mechanism** (`:966` + `:6216`), mass **unmeasured on today's main**.

### V8 — hand-off

The owner decides the order in §5. The plan below is the proposal built on this register; the
tournament in §9 stress-tested its ordering.

---

## 3. Evidence ledger (the register the cards cite; repo-relative paths, main `4be2f3a`)

Two files are named `bind.rs` and every cite below and in §5 carries its crate: `proxima-tensor/src/bind.rs`
(3089 lines) owns `Layout` `:95-98`, `BoundOp` `:200-215`, `BoundOpKind` `:221-264`, `layout_of`
`:1594-1606`, `correct_packed_matmul_layouts` `:1618-1707`; `proxima-model-interop/src/bind.rs` (4257 lines)
owns the harnesses — `PROXIMA_MAX_TOKENS` `:2719`, the ORACLE test `:2765`, the BENCH test `:3002`
(`#[ignore]` at `:3001`, `#[cfg(feature = "metal")]` at `:2999`), the two assertions `:3052-3054`
(`plan_hits == 0`) and `:3056-3058` (`plan_misses == forward_calls_taken`), the MILLI test `:3084`
(`max_tokens = 5` hardcoded at `:3103`, sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself at `:3122`). A bare
`bind.rs:3052` opened in the wrong crate lands on an unrelated test. `scripts/omega-gate.sh` has six steps,
two of them count-asserted (`[3/6]` `ran_count`, `[6/6]` `passed_count`); the `--all-features` build and
test are `[2/6]` and `[3/6]`, clippy runs two arms at `[4/6]`, rustdoc two arms at `[5/6]`.

### 3.1 Repo state at plan time (READ)
- main == origin/main == `4be2f3a`; clean.
- Since `2b95210`: `0c3bd4f` qwen3.5 hybrid attention (spec.rs +8735/-2836), `7d09145` bind checkpoint mapping once (metal.rs +177), `e64992b` packed-byte embedding gather, `6745962`, `23e2e5e` gate no-copy cache on resident classification, `ff749a0` bound plan cache + memory instrumentation, three example commits.
- `omega/Cargo.toml` features: `default/std/alloc/cpu/metal/vulkan/cuda/npu/ane/instrument/metal-tiled-gemm/wgpu-backend`. Nothing else.
- Machine: Apple M1 Max, 32-core GPU, Metal 3, 64 GiB. Incumbent binary `~/repos/others/llama.cpp/build/bin/llama-bench` @ b25346221. Model `~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf` (3.9 GB).
- torch 2.13.0 + MPS in `proxima-onnx/scripts/torch_reference/venv`; onnxruntime absent there; ORT source at `~/repos/others/onnxruntime`.
- `ai_docs/task-routes.jsonl` and `invariants.jsonl`: zero tensor/omega/GPU records.

### 3.2 The instruction set (READ) — it IS one RISC
- `Op`: Input, Elementwise, Reduce(Reduce), Iota, Constant (`op.rs:175-266`). `ScalarOp`: 17 (`op.rs:60-78`). `Keep`: Reduce | Scan (`:142`). `ReduceInit`: 5 (`:124`). `Extent`: Static | Symbolic (`:45`). `IndexMap`: Affine | Computed (`map.rs:134-152`).
- `Reduce{dtype, body, init, operand, in_map, out_map, keep, name}` (`op.rs:153-164`).
- `BoundOpKind`: Elementwise{body: ComposedBody, operands} | Reduce{element_body, reduce_op, init, keep, operands, output_axes, out_layout: Layout, out_scatter: Option<Lookup>} | Iota | Constant{value} (`proxima-tensor/src/bind.rs:221-264`). `Layout{base: i64, strides}` (`:95-98`). `MAX_INLINE_RANK = 4` (`:82`).
- No `Concat`, `Pad`, `Tile`, `PlacedBuffer`, `write_placement` anywhere.

### 3.3 The lowering (READ) — it is NOT one RISC
- omega/src: msl.rs 4712, wgsl.rs 1929, cuda.rs 1838, metal.rs 2385, wgpu_driver.rs 872, backend.rs 615, sized.rs 45, error.rs 84, lib.rs 73.
- `emit` (`msl.rs:673-697`) -> render_elementwise | render_reduce | render_scan | render_iota | render_constant. `render_reduce` (`:2178-2257`) -> serial | cooperative (`reduce_is_cooperative` `:824`). `push_cooperative_reduce_body` (`:3140-3194`) -> tiled_gemm_block | packed_row_block | generic simd fold; ordering load-bearing (`:751-754`). 8 kernel-body shapes.
- `classify_kind` (`metal.rs:785-826`) buckets by substring of emitted MSL; its doc (`:777-783`) says the route decision "is not exposed as its own accessor". `diagnose_kind` (`:835-854`).
- Shared across msl/wgsl/cuda: `Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots` only (`wgsl.rs:105`, `cuda.rs:66`). ~26 functions reimplemented per backend. CUDA rejects Iota/Constant (`cuda.rs:161-172`); wgsl and cuda have no tiled-GEMM/packed-row-block path.
- Consts: `SIMD_WIDTH=32` (`sized.rs:45`); `PACKED_ROWS_PER_GROUP=4` (`msl.rs:1017`), `TILE_DIM=8` (`:1030`), `TILED_GEMM_NSG=4` (`:1046`), codec block consts (`:294-556`); `omega-runtime.toml` carries only `[tiled_gemm] min_tokens=8 block_m=64 block_n=32 block_k=32`.
- Phases (instrument-gated): prepare `metal.rs:405-410`; block_upload `:457-491`; emit `:2187-2199`; pipeline_lookup `:2201-2207`; op_setup `:2209-2220`; encode_dispatch `:2230-2248`; gpu_exec `:547-554`; readback `:2358-2378`. `execute_plan` `:449-568` = one encoder, one commit, one wait. Diagnostic per-op twin `execute_plan_op_timed` `:654-761`.
- `encode_op` (`:2179-2252`): per op per token — `kernel_cache_key`, `kernel_dispatch_shape`, `pipeline_for` (cached), **`allocate_buffer` for the output (:2210)**, **`upload_uniforms` (:2211)**, fault buffer, bind, dispatch.

### 3.4 The orchestration (READ)
- One driver `run_decode_loop` (`generate.rs:1175-1181`); per token: `apply_serving_config` :1298 -> `build_position_inputs` :1304 -> `named_blocks` assembly :1313-1391 -> `runtime.evaluate` :1443-1449 (= `resolve_plan` + `execute_plan_named`, :927-941) -> `cache.append/advance` :1477-1556 -> `cached_len += new_count` :1559 -> `sample_next_token` :1643.
- `LayerCache{k_even, k_odd, v: Vec<f32>}` :621-625; `append` = 3x `extend_from_slice` :636-640; `named_blocks` hands the whole Vec :642-653.
- Plan cache `BTreeMap<(usize,usize), Plan>` :904; key `(symbols[0], symbols[1])` = `(new_count, cached_len)` :966; miss -> `plan_named` + `mark_resident` + `plans.clear()` + insert :969-975. **Misses every token by construction.**
- `mark_resident` by NAME (`metal.rs:350-362`). `NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed `(ptr, byte_len)` :1848-1849; non-resident blocks go to `upload_block_no_copy_uncached` (23e2e5e) — a fresh no-copy buffer per KV block per token.
- Strict `found != expected` at `metal.rs:991-1000` and `cpu.rs:346-356`.
- `register_checkpoint_mapping` (`backend.rs:402-414`, `metal.rs:1744-1815`): one no-copy buffer for the whole checkpoint mmap; weights addressed by `(buffer, offset)`. Weight re-upload per token is closed on main.
- `AlignedBuffer` (`align.rs:42-46`, `new` :69) — zero production callers.
- `select_backend` picks Metal only when `gpu_layers == GPU_LAYERS_ALL` (ROW 223).

### 3.5 The attention graph (READ)
- `append_mistral_cached_layer` `spec.rs:2336-2865` (530 lines, 25 args). Doc :2303-2319 and comment :2616-2617 name the constraint: `Reduce::out_map` must stay a pure projection. `cached_len` = `Extent::Symbolic(1)` on `kv_cache.{layer}.{k_even,k_odd,v}` :6216-6245; caller passes `symbols = [new_positions, cached_len]` :6012-6013.
- MEMORY (to re-count in card 9.3's ops/layer census): ~26 attention ops/layer; 95 raw nodes/layer; 1196 BoundOps/token; reduces never fuse (610 raw = 610 bound); 2026 raw elementwise fuse to 547.

### 3.6 The incumbent at the exact checkout llama-bench runs (READ, b25346221)
- 23 real ops per Llama layer at decode (list in §4.3); ~740 dispatches/token; views/reshape/permute are no-ops (`ggml-metal.m:1835-1847`).
- NO fusion at this checkout (`grep -rln fuse ggml/src` empty). Parity is reachable without a fusion engine.
- KV cache: device tensors sized `kv_size` once (`llama-kv-cache-unified.cpp:74-118`); written per token by `ggml_cpy(k_cur, ggml_view_1d(k, n_tokens*n_embd_k_gqa, row_size*head_cur))` (`:749-788`); read as `ggml_view_3d` (`:715-747`); V stored transposed (`v_trans = !flash_attn`).
- Decode attention: `mul_mat(K,Q)` -> `soft_max_ext` (ONE kernel: scale+mask+max+exp+sum+normalize, `ggml-metal.metal:1051-1145`; nth from 32 doubling while `nth*ne01*ne02*ne03 < 256`, `ggml-metal.m:2508`) -> `mul_mat(V, kq)`. Flash attention off by default (`common/common.h:328`).
- `rms_norm`: nth doubles to `min(ne00/4, maxTotalThreadsPerThreadgroup)` (`ggml-metal.m:3797-3804`), `float4`, two-level `simd_sum` (`ggml-metal.metal:1679-1721`).
- RoPE NORM for Llama/Mistral (`llama-model.cpp:14250-14273`), nth=min(1024, ne00), one dispatch per tensor (`ggml-metal.m:3941,4046`).
- Q4_K matvec `<nr0=4, nsg=2, nw=32>` (`ggml-metal.metal:5086-5206`, `ggml-metal-impl.h:32-33`): mask without shift (`:5158-5165`), fold 1/256 and 1/16 at combine (`:5171-5175`), `kmask1/2/3` branch-free (`:5147-5150`); `dispatchThreadgroups((ne01+7)/8, 1, ne12*ne13)` x `(32,2,1)` (`ggml-metal.m:3330`).
- `n_cb=1` (`ggml-metal.m:5783`); first 128 nodes on the main thread, rest in one more command buffer (`:5222-5289`).

### 3.7 Unlanded and parallel work (READ) — what Phase 0 reconciles
| where | branch | state | carries |
|---|---|---|---|
| proxima-wt-all | perf/gpu-all-wins @2b95210 | UNCOMMITTED 13 files +3859/-197 | features tiled-gemm, packed-row-nsg2, q4k-mask-fma, q4k-single-fetch, wide-cooperative-reduce, buffer-pool, output-placement |
| proxima-wt-drive | perf/kv-device-resident @2b95210 | UNCOMMITTED 7 files +2375/-96 | output-placement + KV driver (`PlacedBuffer`, `positions_needed` fix) |
| proxima-wt-rules | perf/op-rule-census @2b95210 | UNCOMMITTED 3 files +842/-9 | `FuseSite`/`FuseDeclineReason` census |
| proxima-wt-splitk / -lat | perf/q4k-split-k, perf/kernel-latency | UNCOMMITTED 5 files +352/-48 each | metal-q4k-split-k (unmeasured) |
| proxima-wt-place | perf/output-placement @2b95210 | UNCOMMITTED 3 files +322/-14 | output-placement |
| proxima-wt-merge | perf/attention-single-range @2b95210 | UNCOMMITTED spec.rs +1181/-82 | `append_mistral_single_range_cached_layer`, 488/488, 1196 -> 939 |
| proxima-wt-gpudisp | perf/gpu-dispatch-count | UNCOMMITTED 4 files +244/-28 | wide-cooperative-reduce |
| proxima-wt-gpuker | perf/metal-simdgroup-geometry @bfc150d (+2) | +237/-41 | nsg2 (NEGATIVE), mask-fma |
| proxima-wt-q4k | perf/q4k-orchestration @a2175c2 | +589/-182 | q4k-single-fetch |
| main branch list | perf/decode-orchestration-2 (1 commit) | committed | `PROXIMA_ORCH_THREADS` (unmeasured) |
| main branch list | bench/sealed-pass (2 commits) | committed | `scripts/sealed-pass.sh` (280 lines) |
| main branch list | perf/gpu-decode-ladder (2 commits off 3076d81) | committed | `membw_probe` streaming fix + `scripts/llama_reference/` |
| proxima-wt-cattn | **perf/cached-attention-streaming (42 commits on 4be2f3a)** | committed, another agent, today | `BoundOpKind::CachedAttention` + matcher + `physical.rs` + paired-nibble Q4_K + `prune_dead` + 34 log rows |
All 2b95210-based diffs will conflict on rebase with `0c3bd4f` (spec.rs) and `7d09145`/`23e2e5e` (metal.rs).
Existing worktree names (do not reuse): 3ax access accs all amx anchor attn batch bind bseal bsel build cattn census ci comp const cross cut doc drive gate geo gpu gpucombo gpudisp gpuker green hit integrate land lat merge nanofix narrowtile o2g opcount orch orch2 parity place plancache q06 q4b q4k q4k-accum q4kfix qload qmmap qwen3 qwen35 rebind resolve rules seal sealpath splitk tafter tbase tier torcharms traffic train transa unify verify.

### 3.8 Dead levers (measured negatives; a card that re-proposes one is defective)
nsg=2 / two-simdgroup regrouping (4 measurements, both lanes); encoder churn; per-dispatch fixed cost as the gap; `-t` thread count on the incumbent; rematerialize all ≤2-consumer nodes; the quarantine heuristic as a defect; float4 accumulation in the Q4_K helper; explicit unroll; width-64 dispatch; row-batch 8; tiled-gemm at decode (min_tokens=8 gate, and measured 1.03x slower when forced).

### 3.9 Measured wins not on main (each a Phase 0 landing candidate)
| lever | measured | where | status |
|---|---|---|---|
| Q4_K body: mask-without-shift + fold + branch-free scale/min | gpu_exec 56.618 -> 46.875 (-17.2%); ffn -36% | proxima-wt-all (uncommitted) | MEMORY, N=1/arm + stacked N=3 |
| Q4_K body: paired nibbles (`q4k_pair_dot`) | family 47.8 -> 33.9 ms (-29%); parity 3.1e-6 | perf/cached-attention-streaming ROW 257 | READ, N=3 |
| wide cooperative reduce | bucket 8.918 -> 7.136; gpu_exec -2.80 ms | proxima-wt-gpudisp / -all | MEMORY |
| buffer pool (persistent per-op output/uniform buffers) | in the -11.1% stack | proxima-wt-all | MEMORY, not isolated |
| stacked three | step_wall 69.60 -> 61.85 (-11.1%), gpu_exec 57.0 -> 44.8 | proxima-wt-all | MEMORY, N=3 interleaved |
| single-range attention graph | 1196 -> 939 BoundOps, 488/488, parity 7.35e-8 | proxima-wt-merge | MEMORY; needs write placement |
| `prune_dead` before GPU dispatch | generic | 216d925 on the parallel branch | READ |
| consumer index in the matcher | prepare 150.7 -> 11.6 ms | parallel branch ROW 248 | READ (only matters if the matcher survives §8.1) |

---

## 4. The one-RISC binding

### 4.1 What "one RISC" means here, bound to the code
1. **One instruction set.** `Op` (5) x `ScalarOp` (17) x `IndexMap` (2) x `Keep` (2) x `ReduceInit` (5). Unchanged by this plan. The doc header at `op.rs:166` is corrected to "The five generators".
2. **One bound plan.** `&[BoundOp]` with exactly the 4 `BoundOpKind`s. **No fifth kind.** A fused attention is what the rewrite engine's laws produce over Elementwise/Reduce (rewrite-algebra.md §2 softmax instance + §8 band-streaming), not a variant. §8.1 adjudicates the parallel branch's `CachedAttention` against this line.
3. **One write model.** A `Reduce`'s `out_layout.base` (already in `BoundOpKind::Reduce`) plus a caller-owned persistent OUTPUT buffer is placement. Concat = two reduces writing disjoint `base` ranges of one buffer. The KV cache is a persistent buffer of capacity `C`; each token's K/V reduce writes at `base = cached_len * row_stride`. This changes `shape::project_output_shape`'s contract (offset allowed, extent ≤ buffer), the block-input check (`found >= expected` for OUTPUT-placed buffers only, never for inputs), and the driver (a named output buffer that survives across `execute_plan` calls). No new Op; the change is in `bind`, `shape`, and the two drivers.
4. **One shape across tokens.** `cached_len` stops being the plan key. The graph reduces over the cache CAPACITY bucket `C` (a multiple of a build-time `KV_BUCKET`), masking positions ≥ `cached_len` with the existing `Iota`/`Greater`/`Select` composition (`op.rs:211-216` documents the construction; `spec.rs:823-845` builds today's causal mask). This IS a graph change, not a driver change: today's `is_future` is `[s,w]` over symbol 0 on both axes and the cached block is deliberately never masked (`spec.rs:2314-2319`: "correct without a `cached_len` scalar"). Bucketing adds one `Iota` over the cache axis (`C`), one scalar `Input` leaf carrying `cached_len`, one `Greater` and one `Select` per layer — zero new `Op` variants. The plan key becomes `(new_count, C)`; hits on `KV_BUCKET-1` of every `KV_BUCKET` tokens. The harness assertion `plan_hits == 0` (`proxima-model-interop/src/bind.rs:3052-3058`) is inverted by the same card.
5. **One route.** `enum ReduceRoute { Serial, CooperativeGeneric, PackedRowBlock, TiledGemm }` decided once per `BoundOp` before any emitter runs, with `enum RouteDecline` reasons censused `(NodeId, reason)` mirroring `WidthDeclineReason`/`record_width_tile_decline`. `classify_kind` reads the route, never the source text. All three emitters consume the same `Route`.
6. **One emitter core.** The ~26 per-backend duplicates (R5's census: `msl.rs` 4712 + `wgsl.rs` 1929 + `cuda.rs` 1838 lines, exactly 3 types + 2 fns shared) become one generic structural core over a `Dialect` of seven TEXT methods (`scalar_op_expr`, `preamble`, `kernel_signature`, `entry_name(&BoundOp)` — signature unchanged so the route never enters the emitted name, `simd_reduce_intrinsic`, `threadgroup_barrier`, `atomic_fold`), consumed through a generic parameter over the closed compile-time set of three backends — no `Box<dyn>`, no runtime table. Every backend renders all 4 kinds (CUDA gains Iota/Constant, card 8.2). Card 8.1 classifies all 78 function×backend cells with their MEASURED line counts before one line moves; card 8.3 lands one kind (`Elementwise`) byte-identically and continues only if the deleted-duplicate count equals 8.1's measured STRUCTURE count for that kind (423 lines across three backends on main) — a measured gate, not a threshold. Phase 8 is scheduled AFTER the final board (11.2): it has no measured payoff by its own words and every one of its commands takes the single measurement mutex.
7. **One sizing config.** `omega-runtime.toml` gains `[packed_row_block] rows_per_group=4 lanes_per_block=8`, `[cooperative_reduce] max_width=1024 min_elements_per_thread=4`, `[kv] bucket=256`, `[buffer_pool] ...`; `build.rs` emits them; the bare consts at `msl.rs:1017,1030,1046` and the `SIMD_WIDTH` cap at the cooperative reduce reference generated consts. `SIMD_WIDTH` stays a hardware fact.
8. **One census.** `ReduceRoute` + `RouteDecline` + the existing phase counters + per-family op profile, all under the existing `instrument` feature, all asserting their N.

### 4.2 What does NOT change
No new `Op` variant. No new `BoundOpKind`. No trait. No `Box<dyn>`. No runtime flag for anything that is a compile-time geometry. No per-model macro-op. No change to `ScalarOp`.

### 4.3 The target graph, per Llama layer at decode (the incumbent's own count is the bar)
```
 1 rms_norm(sum)        2 mul(norm w)      3 wq   4 wk   5 wv
 6 rope(Q)              7 rope(K)          8 K -> cache[base]   9 V -> cache[base]
10 scores = K·Q over C  11 softmax(scale, mask, max, exp, sum, normalize)  12 V·p
13 wo                   14 add(residual)   15 rms_norm  16 mul  17 up  18 gate
19 silu                 20 mul             21 down      22 add
```
22-23 ops (the incumbent has a `cont` we do not need). Ours today: ~37 (MEMORY), of which
~26 are attention. The plan's Phase 3 target is ≤ 24 bound ops/layer measured by the census.

---

## 5. The cards (the deliverable Luna executes; 44 cards, 13 phase worktrees + 1 contingent)

This section is the plan-rigor tournament's winning card plan (synthesis_3, §9), expanded so that every
command is verbatim and absolute, then patched with every residual the round-3 judges and the round-4
critique found (§9.1). The four sub-sections are: 5.1 the global protocol G1-G14 (binding on every
card), 5.2 the band ladder (every band derived from its predecessor's delta), 5.3 the cards by phase,
5.4 the appendices the cards refer to (dependency graph, rollback map, abandoned designs, open
questions, conflict resolutions). A card's `tier:` is the routing decision (§0.1); a `worker` card is a
whole brief for a Sonnet-class worker; a `hands` card is what Luna types; a `judge` card stops hands.

How to read one card: `opens` is the only file set the card may edit; `commands` are typed in order
with the real exit code echoed after every build or test; `expect` is the N and is RED at zero;
`predict` is exactly one rung ahead and is written before the first command runs; `kill` ends the
card as a NEGATIVE row, never as a silent drop; `memory gate` is MG-1/MG-2/MG-3 with the byte formula
of §0.3a and G8; `row` is the discipline-log row the card fills, with `ROW <NEXT>` assigned at land
time; `reprove` regenerates the card's numbers today. Test-filter names that a card itself creates
(for example `test(risc_cardinality)`, `test(plan_fingerprint)`, `test(sizing_config)`) are the names
the worker MUST give those tests, so the filter is never a guess.

### 5.0 Card index

| card | title | tier | depends_on | worktree | feature |
|---|---|---|---|---|---|
| 0.1 | The measurement mutex, which does not exist | worker | [] | proxima-wt-risc00 | none |
| 0.2 | Both byte counters, fixed in BOTH upload loops | worker | [0.1] | proxima-wt-risc00 | none (instrument-gated only) |
| 0.3 | A device-side `gpu_exec` window for the batched path | worker | [0.2] | proxima-wt-risc00 | none (instrument-gated only) |
| 0.4 | `scripts/gpu-cell.sh`: parameterized, budget-pinned, N-asserting, memory-gated | worker | [0.3] | proxima-wt-risc00 | none |
| 0.5 | Re-seal R13 on fixed instruments, and add the `-fa 1` incumbent arm | hands | [0.4] | proxima-wt-risc00 | none |
| 0.6 | The streaming ceiling: a copy arm, device-timed, readback outside the window | worker | [0.3] | proxima-wt-risc00 | none |
| 0.7 | Quarantine the uncommitted worktree diffs INSIDE git | hands | [0.1] | proxima-wt-risc00 | none |
| 0.8 | The harness N-contract, the hit formula, and the `symbols` dump | worker | [0.5] | proxima-wt-risc00 | none (dump is instrument-gated) |
| 0.9 | The RISC's cardinality, the row protocol, ai_docs, and a clean tree | worker | [0.5, 0.6] | proxima-wt-risc00 | none |
| 1.1 | Adjudicate `BoundOpKind::CachedAttention` | judge | [0.7] | none | none |
| 1.2 | Recover the mask-fma Q4_K body onto today's main | worker | [1.1, 0.7] | proxima-wt-risc01 | none |
| 1.3 | Cherry-pick the paired body, `prune_dead`, and the consumer index — without their row numbers | worker | [1.1] | proxima-wt-risc02 | none |
| 2.1 | The plan-identity test: a per-op fingerprint vector through `pub` accessors, pre-registered RED | worker | [0.9, 1.3] | proxima-wt-risc03 | `instrument` (existing feature; gates the two new `pub` a... |
| 2.2 | `bind` owns the packed layout; the post-bind rewrite is deleted | worker | [2.1] | proxima-wt-risc03 | none |
| 2.3 | Re-anchor: same plan, same numbers | hands | [2.2, 0.5] | proxima-wt-risc03 | none |
| 3.1 | `Route`, unit-only, with `fn slot(&self) -> usize`, decided by `emit` itself | worker | [2.3] | proxima-wt-risc04 | none — `route::of` is **not** feature-gated (a gated ro... |
| 3.2 | Delete `classify_kind`; the census, lock-free, cost-bounded | worker | [3.1] | proxima-wt-risc04 | `instrument` (existing feature; gates the 8 counters — ... |
| 3.3 | Route the other two backends; the coverage matrix | worker | [3.2] | proxima-wt-risc04 | none |
| 3.4 | The census cell, the census gate, the fingerprint gate, and the elementwise concentration | worker | [3.3, 0.8, 2.2] | proxima-wt-risc04 | none — the gate's named feature set is not `--all-featu... |
| 4.1 | The build-time body selector, not two cargo features | worker | [3.4, 1.2, 1.3] | proxima-wt-risc05 | none — build-time profile key `[q4k] body` (never a car... |
| 4.2 | The bake-off: batched AND per-op arms, terminal tie-break, nothing deleted | hands | [4.1] | proxima-wt-risc05 | `OMEGA_Q4K_BODY` env override (mask_fma | pair_dot) selec... |
| 4.3 | Land one; the loser is a recorded negative, kept selectable | judge | [4.2] | proxima-wt-risc05 | `[q4k] body` toml key |
| 5.1 | Generalize the registered host span to N slots | worker | [4.3] | proxima-wt-risc06 | none — extends the existing checkpoint-mapping primitiv... |
| 5.2 | A capacity-reserved, page-aligned KV arena; `found == expected` preserved | worker | [5.1] | proxima-wt-risc06 | `metal-kv-resident` |
| 6.1 | Bucket the KV leaf extent; the tail mask spelled with the ops that exist | worker | [5.2, 0.8] | proxima-wt-risc07 | `kv-capacity-bucket` |
| 6.2 | Invert the `plan_hits == 0` assertion, `#[cfg]`-paired, as a formula | worker | [6.1] | proxima-wt-risc07 | `kv-capacity-bucket` (the `#[cfg]`-paired assertion pair) |
| 6.3 | The bucketing trade cell: orchestration saved against padding added | hands | [6.2] | proxima-wt-risc07 | `kv-capacity-bucket`, swept via `OMEGA_KV_BUCKET_TOKENS` |
| 6.4 | *(contingent on 6.3's kill)* The shape-invariant plan | worker | [6.3] (only if 6.3's decisive kill fired at every bucket value) | proxima-wt-risc14 | the same feature flag that gates 6.2's `#[cfg]` pair (`kv... |
| 6.5 | The device output/uniform arena: whole-buffer sharing only | worker | [6.3] (or 6.4 if reached) | proxima-wt-risc07 | `metal-plan-stable-buffers` |
| 6.6 | Orchestration re-seal | hands | [6.5] | proxima-wt-risc07 | all landed Phase 4-6 features enabled together (the Q4_K ... |
| 7.1 | Every geometry constant into `omega-runtime.toml` | worker | [3.4] | proxima-wt-risc08 | none — build-time consts via `omega-runtime.toml`, not ... |
| 7.2 | The cooperative reduce stops being 32 lanes wide for every size | worker | [7.1, 6.6] | proxima-wt-risc08 | metal-wide-reduce, forwarded from `proxima-tensor` throug... |
| 8.1 | The `Dialect` classification: which of the ~26 functions are TEXT and which are STRUCTURE | judge | [3.4] | proxima-wt-risc09 | none — docs only, no source change |
| 8.2 | CUDA covers `Iota` and `Constant`, in two commits | worker | [8.1] | proxima-wt-risc09 | none — CUDA is an existing backend feature, not a new one |
| 8.3 | The core, one kind (`Elementwise`), with the continuation gate | worker | [8.2] | proxima-wt-risc09 | none — **not feature-gated**; a gate on a refactor mean... |
| 9.1 | Structural injectivity at bind; the affine scatter degenerates to a strided store | worker | [7.2] | proxima-wt-risc10 | none — the prover is unconditional, not feature-gated |
| 9.2 | The per-token write base, and the KV write moves into the graph | worker | [9.1, 6.5] | proxima-wt-risc10 | kv-scatter-write, in `proxima-tensor`, forwarded |
| 9.3 | Single-range attention; the op-count cell, with wall in the criterion | worker | [9.2] | proxima-wt-risc10 | attention-single-range, default-off |
| 10.1 | The geometry sweeps, re-run against the graph that now exists | hands | [9.3, 7.1, 7.2] | proxima-wt-risc11 | none — two build-time sizing-config env overrides ([pac... |
| 10.2 | Split-K for the starving low-row shapes — with a delete-the-card entry gate | worker | [10.1] | proxima-wt-risc11 | metal-packed-split-k |
| 10.3 | torch-MPS, honest scope | worker | [0.6] | proxima-wt-risc12 | none — `--device` flag added to an existing script, def... |
| 10.4 | ORT-CoreML, honest scope, and the one unbounded build | worker | [0.9]        scheduled terminal within Phase 10 | proxima-wt-risc12 | none — `--provider` flag added to an existing script, d... |
| 11.1 | Discipline rows, rooflines, ai_docs closure | hands | [0.9, 1.1, 3.4, 0.6, 10.1] | proxima-wt-risc13 | none |
| 11.2 | The final board, sealed AFTER every sweep | hands | [11.1, 9.3, 8.3, 10.1, 10.2, 10.3, 10.4, 6.6] | proxima-wt-risc13 | none — composed sweep of every already-landed feature a... |

Tiers: 32 worker, 9 hands, 3 judge. 33 of the 44 cards run a process on the GPU box and therefore
serialise on the one measurement mutex (G3); the schedule length is that count, not the 30-card
longest chain (5.4 §VI). Cards 6.4 and 10.2 are contingent: 6.4 runs only if 6.3's kill fires, 10.2
deletes itself if its entry gate does not fire.

### 5.1 Global protocol (binding on every card)

**G0 — every citation carries its crate path** [B4 adoption C-A]. There are two `bind.rs` (`proxima-tensor/src/bind.rs`, 3089 lines; `proxima-model-interop/src/bind.rs`, 4257 lines) and `plan_hits` has **zero** hits in the tensor one. A hands model opening `proxima-tensor/src/bind.rs:3052` lands in an unrelated windowed-reduce test. Every `file:line` in every card below is crate-qualified; a card carrying a bare `bind.rs` is RED before it is executed.

**G1 — tiers, no hybrids.** `hands` = runs the given commands verbatim, opens the named `file:line`, records the printed N into the row placeholder. Hands never designs, never writes new code, and **never improvises a substitute command: if a command errors or an N is absent, hands STOPS and reports** [B4 adoption]. `worker` = writes source, tests, scripts, build steps inside one named file set against a fixed design; may not add a type, feature or file the card does not name. `judge` = adjudicates a pre-registered rule; produces no code. A card needing two tiers is split.

**G2 — worktrees: one per phase-branch, cards on it strictly sequential.** Verified collision-free this pass: `git branch --list 'risc/*'` = 0; `git worktree list` = **68** and none matches `risc`; no `proxima-wt-risc*` exists. **13 worktrees, 44 cards.** **Worktree creation is an owner-authorized action** [B4 adoption]: the card names it, the owner creates it, hands then runs inside it. Every card's first block:
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>
TD=$WT/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add $WT -b risc/<PH>-<slug> <base-ref>
mkdir -p $TD
```
Teardown: `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree remove --force $WT && git -C /Users/brianbruggeman/repos/slot-0/proxima branch -D risc/<PH>-<slug>`. **Never enter another card's worktree; never enter an R7/R15 `proxima-wt-*`** — cross-worktree reads use `git -C <path>`. Repo-wide greps are scoped to crate directories: a nested checkout `.claude/worktrees/agent-a6600bd71b04b05fe` lives inside the repo and double-counts otherwise [B4 adoption]. The block above is the *definition*; **every command line on every card below is written with the absolute paths already substituted** (`$WT`, `$TD`, `$LOCK` appear only in this definition), because a hands model's shell state does not survive between its tool calls and a command that depends on an earlier `export` is a command that runs in the wrong tree [round-3 j9]. Fourteen worktrees exist in total: thirteen phase-branch worktrees `risc00`–`risc13` plus `risc14`, minted only if card 6.3's kill unparks card 6.4.

**G3 — the measurement mutex, which does not exist and must be built.** `flock(1)` is **absent** (re-verified: `command -v flock` → exit 1). Card **0.1** writes `scripts/gpu-measure-lock.sh` (python `fcntl.flock` + `os.execvp`), because a repo-local script lands in git, is re-provable by §16, and mutates no host state; `brew install flock` is recorded as the alternative and is **not** the mechanism. `--wait <seconds>`, exit **75** on timeout. Every command that **builds, probes, benches, runs a harness, or runs any `scripts/*-gate.sh`** is welded — `omega-gate.sh [2/6]`/`[3/6]` build and run `--all-features`, i.e. real GPU work:
```
cd $WT && CARGO_TARGET_DIR=$TD CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- <command>
```
**Any command in any card not written in this welded form is RED.** The two cards that hold the lock through an unbounded build (10.4's ORT source build, 4.2's per-arm rebuilds) cap parallelism, record peak RSS via `/usr/bin/time -l`, and are scheduled terminal in their phase; **no other card is scheduled while they hold it**, because a queued card exits 75 rather than waiting [crit residual on 10.4]. **Exit 75 has a policy, not only a report** [round-4 j10]: a hands card whose welded command exits 75 re-issues that same command after a 300-second pause, up to six times, appending each attempt's timestamp to `<worktree>/runs/<card>-waits.log`; the seventh 75 is a STOP with that log attached. The pause is a scheduling loop outside any test or measurement window and is recorded on the row, never hidden.

**G4 — the harness commands, budget pinned, crate-qualified.** `decode_loop_max_tokens()` defaults to 24 (`proxima-model-interop/src/bind.rs:2719`); **every** command sets `PROXIMA_MAX_TOKENS=8` explicitly. The three cells, all `#[ignore]`d (BENCH at `proxima-model-interop/src/bind.rs:3001` under `#[cfg(feature = "metal")]` at `:2999`; MILLI at `:3083` under `#[cfg(all(feature = "metal", feature = "instrument"))]` at `:3081`; ORACLE at `:2764` with NO feature cfg — it is the CPU path and runs under any feature set) and therefore requiring `--run-ignored all`:
```
# BENCH   cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all \
#           -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
# MILLI   PROXIMA_METAL_OP_PROFILE_STEP=3 <same> -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)'
# ORACLE  <same> -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)'
```
The test `return`s silently if the gguf is absent (`:3004-3011`) — a missing fixture exits 0, so **N==0 is RED everywhere**. The seal's binary is `proxima_model_interop-…` under the **release** profile (`baseline-2026-09-03/build.log`), so every cell carries `--release`; `cpu` is not a feature of `proxima-model-interop` (its `[features]` are `std`, `interop-bgpool`, `instrument`, `metal`, `metal-tiled-gemm`), so `--features metal,instrument` plus the card's own forwarded arm is the whole feature list. **The MILLI rung does not read `PROXIMA_MAX_TOKENS`:** `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`) [round-3 j9]; the env is still set on every command for uniformity, and only G5's `F`-from-`generated.0.len()` contract governs any count on that rung.

**G5 — the N contract, EOS-invariant, as a formula; never a literal token count.** From the harness's own quantities (`proxima-model-interop/src/bind.rs:3051`):
```
F := generated.0.len() + usize::from(generated.2)      # forward calls taken
S := F - 1                                             # steady rows; step 0 is prefill, excluded from every mean
plan_hits + plan_misses == F
plan_misses == 1 + #{ steps k in 1..F : key(k) != key(k-1) }   # the cache is CLEARED on miss
plan_hits   == F - plan_misses                                 # (generate.rs:973), so a hit needs the PRECEDING key
```
On main both assertions hold as `plan_hits == 0 && plan_misses == F`. `F >= 2`, `S >= 1`, `op_count > 0`, `ENCODE_DISPATCH_CALLS / F == op_count`, else RED. No card assumes the prompt length; **0.8's `symbols` dump observes it.**

**G6 — arms, interleaving, CoV, clocks.** A B A B A B, never before-block/after-block; ≥3 runs at milli, ≥5 at bench; mean + CoV; the llama.cpp-Metal home-turf arm on every compare row, at **both** `-fa 0` and `-fa 1` (0.5). Every measuring card prints the loadout first (`pgrep -x` per named process + `sysctl -n vm.loadavg`). A loaded box is admissible provided the loadout is on the row. **No kill criterion may be set inside a CoV band** — and **0.5 therefore measures CoV per phase counter as well as on wall and gpu**, because 3.2, 6.2 and 6.5 all set per-phase kills and nothing else produces those bands [crit O-A]. **Every number names its clock**: host ticks (`gpu_exec_ms`, phase counters) or GPU timestamps (`gpu_device_ms`, per-op) [B4 adoption Q15].

**G7 — the bench ladder.** nano (counts, hashes, emitted text, no device) → micro (one kernel: `omega/examples/{q4k_matvec_probe, membw_probe}`, `omega/benches/metal_vs_cpu.rs`) → milli (`profiles_one_real_decode_step_by_per_op_gpu_time`, **+7.3% per-op inflation quoted beside every per-op number and deflated before any band arithmetic**) → bench (the interleaved cell vs `llama-bench`). Every prediction is **exactly one rung ahead**; a card with nothing to measure writes `predict: none — this card produces records, not a measurement`. A miss kills the climb and is decomposed into *inconsistency* vs *understanding-gap* with a named work item.

**G8 — memory is a KILL on every card that runs a process, with the byte formula** (owner rule 2026-09-03). Observables: `phys_footprint_bytes()` (`proxima-model-interop/src/generate.rs:248`), `omega::metal::current_allocated_size()` (`omega/src/metal.rs:270`), `kv_cache_upload_bytes` (`generate.rs:1657`), `device_allocated_bytes`/`plan_cache_len` (`:1731`).
- **MG-1 (build/lint only):** build exit 0; no new heap-holding `static`/`thread_local` without a bound stated at the site.
- **MG-2 (probe/bench, no checkpoint):** peak task RSS ≤ **400 MB** (R13 prefill 310–357 MB), raised only with the arithmetic on the card; `current_allocated_size()` returns to its pre-probe value ±2 MB.
- **MG-3 (decode harness), all five, any one failing is a KILL:**
```
(1) phys_footprint slope over steps 3..S <= 1_000_000 B/step               (R13: "no monotonic trend")
(2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step             (R13 +1-2 MB/tok; the 262_144
                                                                            term goes to 0 after 5.2)
(3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP
     PREFILL_CAP = 4_305_000_000                      # MEASURED R13 prefill peak, TOP of 4.299-4.305
                 * 1.05                               # CHOSEN headroom -- chosen, not derived, and labelled
                 + kv_capacity_tokens * 262_144        # 32 x (2048 k_even + 2048 k_odd + 4096 v)
     kv=0   => 4_520_250_000        kv=512 => 4_654_467_728                (DERIVED)
(3b) STEADY peak device_allocated_bytes <= STEADY_CAP
     STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
     kv=0   => 4_371_150_000        kv=512 => 4_505_367_728                (DERIVED)
(4) plan_cache_len <= 1 on every step
(5) UNIFORM_CACHE_LEN (NEW, 0.2) does not grow after step 3                (D6: the content-keyed uniform
                                                                            cache is unbounded on main)
ARENA_TRANSIENT_CAP = (4_305_000_000 - 4_140_417_024) * 1.05 = 172_812_125 B
     -- the transient (activations + uniforms + outputs) headroom, DERIVED from two MEASURED R13 numbers.
        6.5's ARENA_PEAK_BYTES and 10.2's split-K partials are both asserted against it.
```
**The 34 GB trap, closed at build time:** `ServingConfig::context_length` defaults to **131_072** (`proxima-model-interop/src/serving.rs:161`, re-verified) × 262,144 = **34,359,738,368 B**. `kv_capacity_tokens` is a **build-time key** with a `build.rs` byte assertion in the style of `require_nonzero` (`omega/build.rs:16`) and `require_multiple_of_eight` (`:59`, so `capacity × 2048` is a 16 KiB multiple); **`context_length` never sizes an allocation**, and a runaway is a compile error. RSS ceiling rises to **540 MB** from 5.2 onward (400 + 512×262,144 = 134,217,728 B), stated once with its arithmetic and inherited. Non-measuring cards record `memory gate: MG-1 — <reason>`, never blank.
*Provenance note, stated because the previous round's term was back-fit:* there is **no** invented 41,943,040 activation constant. Every term above is a MEASURED R13 figure or a labelled CHOSEN headroom [crit B14, SD-A].

**G9 — the crate gates, all of them.** `scripts/omega-gate.sh` has **six** steps: [1/6] `--no-default-features --features alloc` build; [2/6] `--all-targets --all-features` build; [3/6] nextest `--all-features` with **`ran_count` asserted nonzero**; [4/6] clippy pedantic on **two** arms (`--lib --no-default-features --features alloc`, `:56`; `--all-targets --all-features`, `:57`); [5/6] `cargo doc` on two arms; [6/6] `cargo test --doc --all-features` with **`passed_count` asserted nonzero** [B4 adoption C-C]. **Either count == 0 is RED**; `ran_count` must not decrease card to card. Additionally, and this is new: a card touching `proxima-tensor` also runs `bash scripts/proxima-tensor-gate.sh`; a card touching `proxima-tensor/src/{shape,map,bind}.rs` also runs **`bash scripts/proxima-autograd-gate.sh`**, because `proxima-autograd` consumes `Reduce::out_map` semantics (`proxima-autograd/src/adjoint.rs:835`, `:1033`; `error.rs:73-88`) and `omega/Cargo.toml:196` depends on it [crit RS-A]; and **no gate runs the decode harness** (D7), so every card that touches it re-proves it by name.

**G10 — correctness oracle (§14).** `generated_text` stays `"Here is a simple Python function that returns"` and the single-token oracle stays `2651` / `"known"` (`proxima-model-interop/src/bind.rs:2797-2803`). Any drift is a KILL regardless of the number.

**G11 — row placeholders.** Main's last row is **ROW 233** (`proxima-tensor/docs/discipline.md:18736`). Branches carry `## ROW <NEXT> -- <title>` only; the number is assigned at land time from `grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1`. **Every cherry-pick uses `--no-commit` and drops the `discipline.md` hunk**:
```
git cherry-pick -x --no-commit <sha>
git checkout HEAD -- proxima-tensor/docs/discipline.md
git diff --cached --name-only | grep -c discipline.md    # must be 0, else RED
```

**G12 — provenance.** Every number cites its ledger section; computed numbers are tagged **DERIVED** and never anchor a mechanism claim (§18). MEMORY figures appear only as "MEMORY, superseded by R13". R2's "KV re-upload ~3.3%" has no counterpart in R13 and may not anchor a prediction. R1's `228.9 GB/s` is MEMORY and may not be a kill threshold until 0.6 measures a ceiling. **A number measured on another tree is labelled cross-tree and may not be called "a MEASURED card delta" until a card on this tree re-earns it** [crit residual on 4.3].

**G13 — commits and CI.** Conventional, lowercase, imperative, <72 chars, one logical change, **every commit a green bisect point**. Behaviour-changing cards ship a default-off feature (or a build-time profile key) and `#[cfg]`-pair any assertion the change invalidates. **Every new cargo feature or `rustc-cfg` value is added to `.github/workflows/proxima-tensor.yml`'s omega jobs in the same commit that introduces it, or the card names it as deliberately out of CI with the reason** — two macOS jobs exist (`omega-gate` running `scripts/omega-gate.sh`, and `omega-compare-bench` running `cargo bench -p omega --features metal --bench metal_vs_cpu -- --quick`, whose comment reads "Smoke-runs the incumbent bench so it can never again sit registered-but-unexecuted") and a feature invisible to them is a level failure the workspace has recorded before [crit HC-C]. **Nothing lands on `main` without owner authorization, per phase, at that phase's judge card.** Inside a `risc/*` worktree branch the cards DO commit — the cherry-picks (1.3), the green bisect points, the ON/OFF arms built from the previous commit (3.2) and the phase-branch bases all require it — and that is the one authorization this plan asks for up front: **the owner's acceptance of this plan authorizes local commits on `risc/*` branches inside `proxima-wt-risc<NN>` worktrees, and the creation of those fourteen worktrees, and nothing else.** Until that acceptance is recorded, every card ends at `git add` with its message written to `<worktree>/runs/<card>.commitmsg`, and the report says so.

**G14 — no time estimates, no verdicts.** Cards produce evidence rows.

### 5.2 The band ladder (every band DERIVED from its predecessor's own predicted delta)

Round 4's arithmetic correction: R13's bucket figures are **per-op-mode** with **+7.3% inflation** (R13's own words), and the previous ladder subtracted them from a **batched** `gpu_exec_ms`. Every bucket is deflated by 1.073 before it enters a band [crit residual]. `δ_b` is a **MEASURED** quantity produced by 6.3 and is carried explicitly to the board rather than capped by assertion [crit O-B].

| after | `gpu_exec_ms` (host ticks) | `step_wall_ms` | ratio vs 17.52 | the delta and its source |
|---|---|---|---|---|
| R13 baseline | **56.93** | **67.92** | 3.88x | MEASURED R13 |
| **4.3** Q4_K body | [44.9, 48.6] | [55.9, 59.6] | 3.19–3.40x | packed bucket 44.450 per-op ÷1.073 = **41.43 batched-equiv**; −29% (R12 ROW 257, **cross-tree**) to a −20% floor — **DERIVED** |
| **5.2** KV residency | unchanged | [54.4, 58.1] | 3.11–3.32x | `block_upload` 2.00 → ≤0.5 (R13: 0.4 on step 2, weights only) — **DERIVED** |
| **6.3** bucketing at the swept optimum | +δ_b | [52.7, 56.4] + δ_b | — | `prepare` 1.97 → ≤0.2 on hit steps (−1.7); **δ_b MEASURED at 6.3, pre-registered [0.3, 2.8]** |
| **6.5** device arena | +δ_b | [49.2, 53.3] + δ_b | — | `op_setup` 3.90 → [0.4, 0.8] — **DERIVED** |
| **7.2** wide cooperative reduce | [43.2, 47.8] + δ_b | [47.5, 52.5] + δ_b | — | coop bucket 9.113 ÷1.073 = 8.49; −20% (R3/M4, **MEMORY**, flagged) to a −10% floor — **DERIVED** |
| **9.3** single-range attention | ±1.0 | ±1.0 | — | pre-registered ≈ **zero** wall movement (R12's control: 1194→616 moved 0.036 ms) |
| **11.2** the board, one prediction | **[43.5, 50.6]** | **[46.8, 57.3]** | **2.67–3.27x** | the sum above with δ_b at its measured endpoints and the 9.3 widening — **DERIVED, the weakest number in this document** |

**The board's kill:** `step_wall_ms` **> 59.0** — outside the band the cards' own terms produce (57.3), so it cannot fire on a tree behaving exactly as designed; a breach means R13's decomposition is wrong somewhere and the row names **which bucket did not move, by counter**, before any further card is scheduled.


### 5.3 The cards, by phase

### 5.3.0 PHASE 0 — measurement truth

*Worktree `proxima-wt-risc00`, branch `risc/0-measure-truth`, base `4be2f3a`. Cards 0.1–0.9 are strictly sequential on it. Nothing downstream is attributable until this phase closes.*

---

#### CARD 0.1 — The measurement mutex, which does not exist `[B3 P0.1, crit SD-1, R18]`

tier: worker
depends_on: []
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- nothing in-repo — this card writes a new script; no existing repo source is read for its body
- `scripts/omega-gate.sh` — the weld target: every card's crate gate runs through this mutex
- `scripts/proxima-tensor-gate.sh` — the weld target: every card's tensor gate runs through this mutex

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2.
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 -b risc/0-measure-truth 4be2f3a
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
```
3. `edit: scripts/gpu-measure-lock.sh (new file) — a python shim: fcntl.flock(fd, LOCK_EX) on /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock then os.execvp(the wrapped command); takes --wait <seconds>, exits 75 on timeout`
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima && command -v flock; echo "flock_present=$?"
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash -n scripts/gpu-measure-lock.sh && shellcheck scripts/gpu-measure-lock.sh
```
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && touch /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 10 -- echo lock-ok
```
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && ( bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 60 -- sleep 5 & sleep 1; \
            bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 2 -- echo second; echo "exit=$?" )
```

##### expect
- N1 `command -v flock` reports **absent** — that is the finding and the reason this card is card one
- N2 the shim prints `lock-ok`
- N3 the contention test: the second invocation **blocks then runs**, and with `--wait 2` against a 5 s holder it **exits 75** and prints nothing
- N==0 is RED

##### predict (nano → micro)
the shim adds < 20 ms to a welded command (one `open` + one `flock` + `execvp`), i.e. below one part in 10³ of any measuring cell.

##### kill
- the shim cannot obtain an exclusive lock from two processes ⇒ **the plan does not proceed**; an unserialised GPU box makes every CoV in this document a lie
- fallback recorded on the row: `brew install flock` (host mutation, not preferred — it is not in git and cannot be re-proved by §16)

##### memory gate
- gate: MG-1
- what this card allocates, in bytes: zero — build/lint only, no process runs the model

##### rollback
`git revert`; `rm /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`.

##### blast
one new script; zero library code.

##### observe
the shim's exit codes; the lock file; the loadout capture that every later card reuses.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 10 -- echo lock-ok`

##### row
```
## ROW <NEXT> -- the GPU box had no measurement mutex and macOS ships no flock(1): the lock is a repo script, not a homebrew formula

**Card:** 0.1. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-1 — no process runs the model; zero bytes.
**Predict (one rung ahead, written before running):** shim overhead < 20 ms, below one part in 10³ of any measuring cell. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| shim | lock-present / blocks-then-runs / timeout-exit-75 | 3 | n/a — pass/fail assertions, not a rate | <...> |
**Gates:** n/a — no crate gate run this card (mutex construction only).
**Parity:** n/a — no model forward pass runs in this card.
**Census:** the shim's exit codes (lock-ok / blocks-then-runs / exit=75); the lock file at /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock.
**Home-turf arm:** none — this card builds tooling, not a decode cell.
**Principles engaged and what each changed:** §III G3 (the measurement mutex, which does not exist and must be built); §X conflict 1 (flock absent on this Mac; repo-local shim wins over brew). **Abandoned:** none.
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 10 -- echo lock-ok`.
```

##### report skeleton
```
CARD 0.1 — <STATUS>
ran: <each command above, EXIT>
N: N1 flock_present, N2 lock-ok, N3 contention (blocks-then-runs, exit=75)
numbers: <shim overhead ms; lock/contention pass-fail table>
predict vs observed:
files: scripts/gpu-measure-lock.sh (new)  diff --stat:
row: <see above>
reprove: cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 10 -- echo lock-ok
open:
```

---

#### CARD 0.2 — Three byte counters in two loops, the uniform cache read for the first time `[S2 0.1, B3 P0.2+P0.3, crit HC-3, l]` [round-4 synth S2]

tier: worker
depends_on: [0.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none (instrument-gated only)          default: n/a

##### opens
- `omega/src/metal.rs:612` — `pub operand_bytes: u64`
- `omega/src/metal.rs:694-700` — defect 1, verified verbatim: `.map(|(buffer,_offset)| buffer.length() as u64)`
- `omega/src/metal.rs:465-468` — defect 2, verified: `counter!(BLOCK_UPLOAD_BYTES, …)` fires **before** the path match at `:469-486`
- `omega/src/metal.rs:663-690` — `execute_plan_op_timed`'s **own** block-upload loop; every `op_profile_family` number comes from this path `[crit HC-3]`
- `omega/src/metal.rs:1786-1815` — `checkpoint_mapping_offset`
- `omega/src/metal.rs:1879` / `:1903` / `:1914` — the no-copy upload paths this counter must cover
- `omega/src/metal.rs:2056-2078` — `UNIFORM_BUFFERS`, content-keyed and unbounded (D6), and `:2064`/`:2069` (`UNIFORM_CACHE_LEN`/`UNIFORM_BUFFER_REUSES`) [round-4 synth S2]
- `omega/src/msl.rs:294-556` — codec block constants (Q4_K/Q5_K/Q6_K bytes-per-element siblings)
- `proxima-model-interop/src/generate.rs:109-212` — every consumer of these counters
- `proxima-model-interop/src/generate.rs:1723` — the printed field list

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: omega/src/metal.rs:694-700 — compute the operand's tensor bytes from element_count(prepared.shapes.of(*source)) × dtype/codec bytes-per-element (f32 = 4; Q4_K = 144/256 = 0.5625, siblings at msl.rs:294-556); keep bound_buffer_bytes as a separate field so mapping-offset behaviour stays observable`
3. `edit: omega/src/metal.rs:465-468 and :469-486 — delete the unconditional BLOCK_UPLOAD_BYTES counter fire; RETIRE BLOCK_UPLOAD_BYTES and RENAME it BLOCK_OFFERED_BYTES at its existing site (not kept as "their sum" — that would preserve the old headline artefact under a new name); record BLOCK_COPIED_BYTES / BLOCK_NOCOPY_BOUND_BYTES / BLOCK_OFFSET_BOUND_BYTES inside each terminal path [round-4 synth S2]`
4. `edit: omega/src/metal.rs:663-690 — apply the same three counters to execute_plan_op_timed's own block-upload loop, in this same commit`
4a. `edit: omega/src/metal.rs:2064 region — add UNIFORM_CACHE_LEN gauge, printed per step; pre-registered prediction: it grows by approximately op_count per token because every Uniforms blob carries reduction_total, a function of cached_len; if flat, D6 is wrong and 6.5/9.2 re-scope [round-4 synth S2]`
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.2-gate.log; echo "EXIT=${PIPESTATUS[0]}"
```
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.2-milli.log; echo "EXIT=${PIPESTATUS[0]}"
```
This is a **5-token cell**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` and the env profile step above are inert on this rung — milli budget 5, bench budget 8, the two rungs are never read as the same cell [round-4 judge J4].
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.2-bench.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1 `COPIED + NOCOPY_BOUND + OFFSET_BOUND == BLOCK_OFFERED_BYTES` on **every** step in **both** paths — stated on the row as a partition check, true by construction, whose only job is to prove no path is uninstrumented, not as evidence of anything (supersedes F1's wording) [round-4 fix F1] [round-4 synth S2]
- N2 the load-bearing number: `BLOCK_COPIED_BYTES` per steady token is 8-10 MB (R13's `kv_cache_upload_bytes` 8.13-9.70 MB, the only real copies) and `BLOCK_OFFSET_BOUND_BYTES` ≈ 4.14 GB, with `mapping_offset_uploads=291` / `copying_uploads=4`; `BLOCK_COPIED_BYTES` above 100 MB/token is RED [round-4 synth S2]
- N3 all **8** `op_profile_family` rows within **1%** of R13's shape-derived column (`ffn_up` 33.05 MB, `ffn_down` 34.00, `attn_q` 9.45, `attn_v`/`attn_k` 2.40, `output.weight` 107.5 MB) — where the column is re-derived per tensor from the codec the GGUF header declares for THAT tensor (Q4_K_M checkpoints carry Q6_K for some `ffn_down`/`attn_v` layers, which is why R13's `ffn_down` 34.00 MB differs from `ffn_up`'s 33.05 MB at identical element counts; a uniform `rows*k*0.5625` would put the two 2.9% apart, inside the 1%-5% band this card assigns no action to). The per-tensor codec is read, never assumed [round-4 fix F1]; N < 8 is RED
- N4 `total_operand_bytes` falls from ~1.2 TB to **≈ 4.07 GB/step** (DERIVED from R13's per-family column)
- N5 ≥3 unit tests (Q4_K operand == `rows*k*0.5625`; f32 == `elements*4`; an operand bound at a nonzero mapping offset reports the **tensor** length)
- N==0 is RED

##### predict (milli → bench)
`step_wall_ms` unchanged within R13's CoV, mean in **[67.6, 68.3]** — this card changes accounting, not work. A timing move means the byte computation is on the hot path and is itself the finding.

##### kill
- `step_wall_ms` moves > 2× R13's CoV (>1.0%) ⇒ hoist the computation to once-per-plan over `prepared.resolved` and re-measure; still moving ⇒ revert
- corrected per-family bytes disagreeing with R13's derived column by >5% ⇒ the shape derivation is wrong, **no GB/s row may be written anywhere and 0.6 is blocked**

##### memory gate
- gate: MG-3
  1. `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  2. `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  3a. PREFILL peak `device_allocated_bytes` <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  3b. STEADY peak `device_allocated_bytes` over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  4. `plan_cache_len` <= 1 on every step
  5. `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
- what this card allocates: six `AtomicU64` statics (+336 B static, 56 B each per `counter.rs:98`); zero heap

##### rollback
`git revert`; instrument-gated only; reverting restores R13's (wrong) numbers exactly.

##### blast
`omega/src/metal.rs` (2 defects × 2 loops, 4 counter decls, 2 struct fields), `generate.rs` printers. Zero kernel, zero IR, zero feature.

##### observe
`operand_bytes`, `bound_buffer_bytes` (**NEW**, `metal.rs:694` and `:686`), `total_operand_bytes`, `gpu_ns_per_byte`, the three block counters (**NEW**, inside each terminal path in **both** loops), `mapping_offset_uploads`, `copying_uploads`.

##### reprove
the welded MILLI cell (command 6 above); the row's claim is N1's identity in both paths and N3's 8-family table.

##### row
```
## ROW <NEXT> -- two byte counters measured the wrong thing in two loops: operand_bytes was the checkpoint mapping, block_upload_bytes was the binding, and the per-op path had its own copy of the defect

**Card:** 0.2. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none (instrument-gated only), default: n/a.
**Allocation budget (hot/setup/cold):** MG-3, six AtomicU64 statics (+336 B static), zero heap; DEVICE_CAP_BYTES formula above.
**Predict (one rung ahead, written before running):** step_wall_ms unchanged within R13's CoV, mean in [67.6, 68.3]. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| ours | MILLI (op-profile step 3) | <...> | <...> | <...> |
| ours | BENCH (cached decode loop) | <...> | <...> | <...> |
**Gates:** omega gate `ran_count`/`passed_count` <N run/N passed>.
**Parity:** n/a — instrument-only change; no oracle drift is asserted here (see 0.5/0.9 for oracle gates).
**Census:** BLOCK_COPIED_BYTES / BLOCK_NOCOPY_BOUND_BYTES / BLOCK_OFFSET_BOUND_BYTES in both loops, N=<...> steps each; the 8-family table.
**Home-turf arm:** none — this card is accounting-only, no llama.cpp cell.
**Principles engaged and what each changed:** §I.2 D1 (the two instrument defects, now fixed in both loops); §III G12 (provenance — no GB/s row is valid until this card closes). **Abandoned:** none.
**Re-prove:** the welded MILLI cell (command 6 above).
```

##### report skeleton
```
CARD 0.2 — <STATUS>
ran: <each command above, EXIT>
N: N1 identity, N2 offset/copied ratio, N3 8-family table, N4 total_operand_bytes, N5 unit tests
numbers: <8-family table, both loops' identity sums>
predict vs observed:
files: omega/src/metal.rs, proxima-model-interop/src/generate.rs  diff --stat:
row: <see above>
reprove: the welded MILLI cell
open:
```

---

#### CARD 0.3 — A device-side `gpu_exec` window for the batched path `[B3 P0.4, R18]`

tier: worker
depends_on: [0.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none (instrument-gated only)          default: n/a

##### opens
- `omega/src/metal.rs:545-555` — host ticks around `commit()`/`waitUntilCompleted()`
- `omega/src/metal.rs:734` — the device window the op-timed path already uses

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: omega/src/metal.rs:546-554 — add GPU_DEVICE_NS from GPUEndTime()-GPUStartTime() on the one batched command buffer; report both gpu_exec_ms and gpu_device_ms, neither replacing the other`
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.3-gate.log; echo "EXIT=${PIPESTATUS[0]}"
```
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.3-bench.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1 `gpu_device_ms <= gpu_exec_ms` on every step (identity assertion; a violation is RED)
- N2 `gpu_exec_ms` mean reproduces R13's **56.93 ± 0.7%**
- N==0 (no steps parsed) is RED

##### predict (milli → bench)
`gpu_exec_ms − gpu_device_ms` ≤ **1.0 ms/token**. A larger gap relocates mass from the GPU bucket to orchestration and **rewrites §I.1's decomposition** — which is the finding, reported first.

##### kill
`gpu_device_ms` returns 0 or negative on any step (`GPUStartTime` unpopulated) ⇒ unusable; revert and record the negative.

##### memory gate
- gate: MG-3
  1. `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  2. `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  3a. PREFILL peak `device_allocated_bytes` <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  3b. STEADY peak `device_allocated_bytes` over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  4. `plan_cache_len` <= 1 on every step
  5. `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
- what this card allocates: instrument-only, zero heap; one new `u64` field per timed step

##### rollback
`git revert`; instrument-only.

##### blast
instrument-only.

##### observe
`gpu_exec_ms`, `gpu_device_ms` (**NEW**, `metal.rs:546-554`).

##### reprove
the welded BENCH cell (command 4 above).

##### row
```
## ROW <NEXT> -- the batched gpu_exec window was host ticks; the device window is <N> ms narrower

**Card:** 0.3. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none (instrument-gated only), default: n/a.
**Allocation budget (hot/setup/cold):** MG-3, instrument-only, zero heap.
**Predict (one rung ahead, written before running):** gpu_exec_ms − gpu_device_ms <= 1.0 ms/token. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| ours | BENCH (cached decode loop) | <...> | <...> | <...> |
**Gates:** omega gate `ran_count`/`passed_count` <N run/N passed>.
**Parity:** n/a — instrument-only change.
**Census:** gpu_exec_ms, gpu_device_ms per step, N=<...> steps.
**Home-turf arm:** none — this card measures our own device window only.
**Principles engaged and what each changed:** §I.2 D1 (R18's addendum: the batched window is host ticks around commit/wait); §IX Q2. **Abandoned:** none.
**Re-prove:** the welded BENCH cell (command 4 above).
```

##### report skeleton
```
CARD 0.3 — <STATUS>
ran: <each command above, EXIT>
N: N1 gpu_device_ms<=gpu_exec_ms identity, N2 gpu_exec_ms mean vs R13
numbers: <gpu_exec_ms, gpu_device_ms, delta per step>
predict vs observed:
files: omega/src/metal.rs  diff --stat:
row: <see above>
reprove: the welded BENCH cell
open:
```

---

#### CARD 0.4 — `scripts/gpu-cell.sh`: parameterized, budget-pinned, N-asserting, memory-gated `[S2 0.2, B3 P0.5, crit MS-4]`

tier: worker
depends_on: [0.3]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- `git show bench/sealed-pass:scripts/sealed-pass.sh` — verified hardcoded (`REPO_ROOT` = `proxima-wt-seal` at `:4`, four sibling worktrees `:25-28`, `MACS_PER_TOKEN=7110402048`, `WEIGHT_BYTES_PER_TOKEN_GB=3.9996`)
- `scripts/omega-gate.sh:37-47` — the assert-nonzero pattern to mirror
- `proxima-model-interop/src/bind.rs:2719`, `:3002`, `:3084` — the budget symbol and the two harness entry points
- `proxima-model-interop/src/generate.rs:107, 248, 1657, 1723-1764` — the field prints this script extracts
- `proxima-model-interop/src/generate.rs:1640-1670` — `greedy_pick_ms`, printed per step alongside the seven phase counters [round-4 synth S3]

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: scripts/gpu-cell.sh (new file) — cargo-cell.sh <worktree-abs> <target-dir> <feature-list> <runs>; write fresh, do not port bench/sealed-pass's hardcoded script. It: (1) prints the G6 loadout; (2) takes the 0.1 mutex; (3) exports PROXIMA_MAX_TOKENS=8 inside the script; (4) interleaves incumbent and ours A B A B; (5) extracts named key=value fields with grep -oE, never a sed backreference; (6) asserts G5's contract and exits nonzero on F<2 || S<1 || op_count==0; (7) computes MG-3's five clauses, asserting clause 3a at kv_capacity_tokens=0 (4,520,250,000) and clause 3b at kv_capacity_tokens=0 (4,371,150,000), and exits nonzero on a breach; (8) emits mean and CoV PER PHASE COUNTER (prepare, emit, block_upload, op_setup, pipeline_lookup, encode_dispatch, readback) plus greedy_pick_ms (generate.rs:1640-1670); (9) emits one machine-readable line per arm; sealed-pass.sh's MACS_PER_TOKEN/WEIGHT_BYTES_PER_TOKEN_GB are named as magic numbers that belong in a config, not ported into this script [round-4 synth S3]`
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash -n scripts/gpu-cell.sh
```
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && shellcheck scripts/gpu-cell.sh
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target std,metal,instrument 5
```

##### expect
- N1 `grep -c 'proxima-wt-' scripts/gpu-cell.sh == 0`
- N2 ≥2 arms enumerated
- N3 `runs × S` rows per arm
- N4 a run breaching a memory clause exits nonzero
- N5 mean and CoV emitted per phase counter (prepare, emit, block_upload, op_setup, pipeline_lookup, encode_dispatch, readback) plus `greedy_pick_ms` — not only for wall and gpu [round-4 synth S3]
- N==0 is RED

##### predict (milli → bench)
the script's own ours/llama cells reproduce 0.5's numbers inside 0.5's measured CoV band.

##### kill
the script's numbers disagree with 0.5 beyond that band ⇒ it is measuring something else (budget, arm order); fix before any later card uses it.

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1] — the script *implements* it; its own run asserts clause 3a at `kv_capacity_tokens = 0` (4,520,250,000) against R13's prefill 4.299-4.305 GB and clause 3b at `kv_capacity_tokens = 0` (4,371,150,000) against R13's steady 4.152-4.163 GB [round-4 synth S3]
- what this card allocates: zero — it is a shell script that wraps existing harness binaries

##### rollback
`git revert`; `scripts/` only.

##### blast
one new script.

##### observe
`plan_hits`/`plan_misses` (`generate.rs:1764`, `proxima-model-interop/src/bind.rs:3046`), `op_count` (`:107`), `phys_footprint_bytes`, `device_allocated_bytes`, `kv_cache_upload_bytes`, `UNIFORM_CACHE_LEN`, `greedy_pick_ms` (`generate.rs:1640-1670`), per-arm and per-phase CoV, loadout. [round-4 synth S3]

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target std,metal,instrument 5`

##### row
```
## ROW <NEXT> -- the GPU cell is a script, not a memory: parameterized, budget-pinned, N-asserting, memory-gated

**Card:** 0.4. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1] as implemented by the script; clause 3a asserted at kv_capacity_tokens=0 (4,520,250,000 B) against R13's prefill 4.299-4.305 GB and clause 3b at kv_capacity_tokens=0 (4,371,150,000 B) against R13's steady 4.152-4.163 GB [round-4 synth S3].
**Predict (one rung ahead, written before running):** the script's ours/llama cells reproduce 0.5's numbers inside 0.5's CoV band. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| llama.cpp-Metal | tg32 via gpu-cell.sh | <...> | <...> | <...> |
| ours | BENCH via gpu-cell.sh | <...> | <...> | <...> |
**Gates:** `bash -n` + `shellcheck` both EXIT 0; script's own memory-clause exit code.
**Parity:** n/a — no oracle assertion inside this script's remit.
**Census:** plan_hits/plan_misses, op_count, phys_footprint_bytes, device_allocated_bytes, kv_cache_upload_bytes per arm.
**Home-turf arm:** llama.cpp-Metal cell, run through the same script.
**Principles engaged and what each changed:** §III G3 (mutex reuse) and G5 (the N contract asserted inside the script); §X conflict 1 (sealed-pass.sh's hardcoding ruled out as the vehicle). **Abandoned:** porting `bench/sealed-pass:scripts/sealed-pass.sh` verbatim — its hardcoded worktree paths and constants are not reused (§VIII item 16 analog for tooling).
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target std,metal,instrument 5`.
```

##### report skeleton
```
CARD 0.4 — <STATUS>
ran: <each command above, EXIT>
N: N1 grep proxima-wt- count, N2 arms enumerated, N3 rows per arm, N4 memory-breach exit
numbers: <machine-readable per-arm lines the script emits>
predict vs observed:
files: scripts/gpu-cell.sh (new)  diff --stat:
row: <see above>
reprove: cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target std,metal,instrument 5
open:
```

---

#### CARD 0.5 — Re-seal R13 on fixed instruments, and add the `-fa 1` incumbent arm `[S2 0.3+0.4, B3 P0.5, crit b, O-2]`

tier: hands
depends_on: [0.4]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- `docs/bench-campaigns/2026-09-03-gpu-one-risc/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log` — the R13 baseline logs this card re-seals against
- `common/common.h:328` at checkout `b25346221` — flash attention **OFF** by default, so R13's 17.52 is the incumbent's *default*, not its *best*
- `-fa 1` also changes the KV layout (`v_trans = !flash_attn`), so it is a **second incumbent**, not a variant (R8)

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: scripts/gpu-cell.sh — add the llama.cpp -fa 1 arm alongside the existing llama -fa 0 / ours arms`
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg
```
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target std,metal,instrument 5   # three arms: llama -fa 0 / ours / llama -fa 1, interleaved A B C A B C
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg
```

##### expect
- N1 3 arms × 5 runs, all parsed
- N2 `op_count == 1196`
- N3 `plan_hits == 0 && plan_misses == F` **written as the formula** [round-2 j6]
- N4 `generated_text` identical
- N5 a CoV band recorded for each phase counter (`prepare_ms`, `emit_ms`, `block_upload_ms`, `op_setup_ms`, `pipeline_lookup_ms`, `encode_dispatch_ms`, `readback_ms`) and for `greedy_pick_ticks` (`generate.rs:1640-1670`) — the bands 3.2, 6.2 and 6.5 set kills outside of; a phase with no band cannot carry a kill [round-4 fix F2] [round-4 synth S3]
- N==0 is RED

##### predict (bench, the anchor rung)
`step_wall_ms` within ±3% of **67.92**, `gpu_exec_ms` within ±3% of **56.93**, and the CoV **tightens** if the box is quieter rather than the means moving. Pre-registered on the second incumbent: **`-fa 1` is faster** — their decode path at this checkout is `mul_mat(K,Q)` → `soft_max_ext` → `mul_mat(V)`, three dispatches per layer against flash attention's one (R8).

##### kill
- `op_count != 1196` or `step_wall_ms` outside ±5% of 67.92 ⇒ the box or the tree is not what R13 measured; STOP and name the difference. **This is not a re-baseline** — R13 stays THE baseline; this card produces the band
- if `-fa 1` wins, **it becomes the home-turf incumbent for every later ratio, the standing gap is worse than 3.88x, and R13/R1/R10's ratios re-base against it, loudly**
- if the build rejects `-fa 1`, record the exact stderr and proceed with `-fa 0` as sole incumbent with the reason on the row

##### memory gate
- gate: MG-3 per run, all five clauses [round-4 synth S1]; a breach on **unmodified main** is RED and stops the plan
  1. `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  2. `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  3a. PREFILL peak `device_allocated_bytes` <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  3b. STEADY peak `device_allocated_bytes` over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  4. `plan_cache_len` <= 1 on every step
  5. `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
- the incumbent arms' peak RSS via `/usr/bin/time -l` (an incumbent that swaps invalidates interleaved cells sharing the box)
- what this card allocates: measurement only, no new allocation

##### rollback
none (measurement); one script line for the arm.

##### blast
`scripts/gpu-cell.sh` and the denominator of every ratio in `discipline.md`/`rooflines.md`.

##### observe
all seven phase counters, `gpu_device_ms` (0.3), `op_count`, `plan_hits`/`plan_misses`/`plan_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, `UNIFORM_CACHE_LEN`, `greedy_pick_ms`, RSS, CoV, loadout. `greedy_pick_ticks` — the ~1.6 ms residual R13 attributes to sampling and cache append gets a counter, so §VIII.20's un-park condition is measurable [round-4 fix F2] [round-4 synth S3].

##### reprove
the three-arm interleaved sweep (command 4 above).

##### row
```
## ROW <NEXT> -- R13 replicated on fixed instruments, and the second incumbent arm lands before any board prediction: flash attention is off by default at b25346221

**Card:** 0.5. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1], breach on unmodified main is RED and stops the plan; incumbent peak RSS via /usr/bin/time -l.
**Predict (one rung ahead, written before running):** step_wall_ms within ±3% of 67.92, gpu_exec_ms within ±3% of 56.93, -fa 1 faster than -fa 0. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| llama.cpp-Metal -fa 0 | tg32 t/s | 5 | <...> | <...> |
| ours | step_wall_ms / gpu_exec_ms | 5 | <...> | <...> |
| llama.cpp-Metal -fa 1 | tg32 t/s | 5 | <...> | <...> |
**Gates:** n/a — this card is a measurement sweep, not a crate gate run.
**Parity:** generated_text == "Here is a simple Python function that returns", oracle 2651/"known" per G10.
**Census:** op_count=1196; plan_hits/plan_misses per the G5 formula; all seven phase counters.
**Home-turf arm:** llama.cpp-Metal, both -fa 0 and -fa 1, per G6.
**Principles engaged and what each changed:** §III G6 (arms/interleaving/CoV); §IX Q3, Q4; §X conflict 6 (the board's denominator). **Abandoned:** none.
**Re-prove:** the three-arm interleaved sweep (command 4 above).
```

##### report skeleton
```
CARD 0.5 — <STATUS>
ran: <each command above, EXIT>
N: N1 3x5 arms parsed, N2 op_count==1196, N3 plan_hits/misses formula, N4 generated_text identical
numbers: <step_wall_ms, gpu_exec_ms, tg32 t/s for -fa 0 and -fa 1, CoV per arm>
predict vs observed:
files: scripts/gpu-cell.sh  diff --stat:
row: <see above>
reprove: the three-arm interleaved sweep
open:
```

---

#### CARD 0.6 — The streaming ceiling: a copy arm, device-timed, readback outside the window `[S2 0.5, B3 P0.6, crit V-4, R18]`

tier: worker
depends_on: [0.3]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- `omega/examples/membw_probe.rs:139-146` — the Metal arm is a **reduce-to-scalar**, which is why `rooflines.md:411` records the ceiling as **DEBT**
- `omega/examples/membw_probe.rs:165-166` — `Instant::now()` wraps the whole `execute_plan`, so upload, commit, wait **and readback** are inside the window (R18)
- `omega/src/metal.rs:1496` — `READBACK_BYTES` [round-4 synth S4]
- `omega/src/metal.rs:1583` — `snapshot_and_reset` [round-4 synth S4]
- `omega/src/metal.rs:2369-2373` — the readback site `READBACK_BYTES` increments at [round-4 synth S4]
- `rooflines.md:396-479`, `:751`, `:766-773` — the GPU lane's DEBT marker and its closing caveat

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: omega/examples/membw_probe.rs — add a copy arm, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: vec![(src, <the affine identity IndexMap, built the way causal_mask builds its maps at spec.rs:823-845>)], name: Some("membw_copy") } (op.rs:191-196 — dtype and name are required fields, operands is a Vec; the literal names all four fields) [round-4 fix F3] [round-4 synth S4] over N f32 into an N f32 output, at two sizes (64 MiB, 256 MiB), 21 runs, min reported, two-size marginal; time with 0.3's GPUStartTime/GPUEndTime so readback is outside the window; validate correctness on a separate untimed run; print both read_only_gbs = 4N/gpu_s and traffic_gbs = 8N/gpu_s on every line; sweep the simdgroup count and take the ceiling at simdgroup saturation (R3/M5: 52 -> 147 GB/s from 256 -> 8001 simdgroups) [round-4 synth S4]`
3. `edit: rooflines.md:396-479, :751, :766-773 — replace the DEBT marker with the copy-arm cell once it passes`
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo run --release -p omega --features metal,cpu,instrument --example membw_probe \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.6-probe.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N = 3 existing arms + 2 copy arms × 2 denominators + 1 marginal = **8 rows**
- the window-cleanliness proof is `READBACK_BYTES.snapshot_and_reset()` delta == exactly 4N per run AND `gpu_device_ms < host_window_ms − readback_ms` on every run; the timed window is `GPUEndTime − GPUStartTime` of the command buffer, which by construction ends before `finish` runs; `READBACK_BYTES` (`metal.rs:1496`, incremented at `:2369-2373`, reset via `snapshot_and_reset` at `:1583`) is a monotonic counter, so the assertion is a DELTA: `READBACK_BYTES.snapshot_and_reset() == 4N` per run (the copy's whole output was read back, outside the window), and `readback_ticks` are not inside `[GPUStartTime, GPUEndTime]` (supersedes F3's readback sentence) [round-4 fix F3] [round-4 synth S4]
- a marginal row with `delta_ms <= 0` is RED
- `read_only_gbs` below the CPU multi-thread triad (69.95/81.21 GB/s, ROW 176) is RED
- the ceiling is taken at simdgroup saturation, not at any fixed simdgroup count (sweep the simdgroup count; R3/M5 52 → 147 GB/s from 256 → 8001) [round-4 synth S4]
- N==0 is RED

##### predict (micro → milli)
the copy arm's `traffic_gbs` **exceeds 228.9** (R1's MEMORY figure for the incumbent's *achieved* decode rate), establishing for the first time that 228.9 is not the machine ceiling. If it lands **below** 228.9, the reduce probe was never measuring bandwidth and 228.9 becomes the ceiling estimate by default — the more interesting outcome, reported first (§19).

##### kill
copy-arm CoV > 5% over 21 runs ⇒ raise both sizes 4× and re-run once; still >5% ⇒ report single-size numbers with the contamination stated and leave the ceiling DEBT. **No spec-sheet figure is ever substituted** (§18; `rooflines.md:411` refused that once already).

##### memory gate
- gate: MG-2, **raised to 1.2 GB with the arithmetic on the row**: 256 MB source + 256 MB destination + 256 MB host mirror + 400 MB baseline; asserted via `current_allocated_size()` before and after
- the decode caps are deliberately exempt and the exemption is stated
- what this card allocates: 512 MB device (source+destination at the 256 MiB size) plus a 256 MB host mirror for the untimed correctness pass

##### rollback
`git revert`; example + one doc section.

##### blast
`omega/examples/membw_probe.rs`, `rooflines.md:396-479, :751, :766-773`.

##### observe
`gpu_device_ms` per arm, `readback_bytes == 0`, both GB/s columns with CoV.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo run --release -p omega --features metal,cpu,instrument --example membw_probe`

##### row
```
## ROW <NEXT> -- the GPU streaming ceiling stops being DEBT: a copy, device-timestamped, readback outside the window, both denominators printed

**Card:** 0.6. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-2 raised to 1.2 GB (256 MB source + 256 MB dest + 256 MB host mirror + 400 MB baseline).
**Predict (one rung ahead, written before running):** copy-arm traffic_gbs exceeds 228.9 GB/s. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| copy (64 MiB) | read_only_gbs / traffic_gbs | 21 | <...> | <...> |
| copy (256 MiB) | read_only_gbs / traffic_gbs | 21 | <...> | <...> |
| marginal (two-size) | delta_ms | 1 | n/a | <...> |
**Gates:** n/a — example binary, no crate-gate run this card.
**Parity:** the copy arm's untimed run validated for correctness (source == destination), stated separately from the timed cell.
**Census:** readback_bytes==0 assertion per arm; both GB/s columns.
**Home-turf arm:** the CPU multi-thread streaming triad (69.95/81.21 GB/s, ROW 176) as the floor check, not a decode-cell comparison.
**Principles engaged and what each changed:** §III G7 (the bench ladder, micro rung); §VIII item 16 (a spec-sheet figure ruled out, this card is the real probe instead); §IX Q5. **Abandoned:** substituting a spec-sheet GPU bandwidth figure (§VIII item 16).
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo run --release -p omega --features metal,cpu,instrument --example membw_probe`.
```

##### report skeleton
```
CARD 0.6 — <STATUS>
ran: <each command above, EXIT>
N: N=8 rows (3 existing + 2 copy arms x 2 denominators + 1 marginal)
numbers: <read_only_gbs, traffic_gbs per size, marginal delta_ms, CoV>
predict vs observed:
files: omega/examples/membw_probe.rs, rooflines.md  diff --stat:
row: <see above>
reprove: cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo run --release -p omega --features metal,cpu,instrument --example membw_probe
open:
```

---

#### CARD 0.7 — Quarantine the uncommitted worktree diffs INSIDE git `[S2 0.6, crit RS-6, round-2 j6]`

tier: hands
depends_on: [0.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- R7's table — 10 worktrees; `proxima-wt-all` 13 files +3859/−197 **and 6 untracked**
- the three worktrees based on other HEADs: `gpuker@bfc150d`, `q4k@a2175c2`, `lat@14f1304`

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: docs/bench-campaigns/2026-09-03-gpu-one-risc/quarantine.sh (new file, committed with the patches) — the capture loop below, verbatim, under `#!/usr/bin/env bash` + `set -euo pipefail`.` The loop uses `read -r -d ''` and process substitution, which are bash constructs; the host's login shell is zsh, so the loop is never typed at a prompt — it is a bash script run by `bash` [round-3 j9]:
```
#!/usr/bin/env bash
set -euo pipefail
R=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered
mkdir -p "$R"
for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
  W=/Users/brianbruggeman/repos/slot-0/proxima-wt-$wt
  git -C "$W" rev-parse HEAD                       > "$R/$wt-head.txt"
  git -C "$W" diff                                 > "$R/$wt-tracked.patch"
  git -C "$W" diff --stat                          > "$R/$wt-stat.txt"
  git -C "$W" status --porcelain                   > "$R/$wt-status.txt"
  mkdir -p "$R/$wt-untracked"
  while IFS= read -r -d '' f; do
    mkdir -p "$R/$wt-untracked/$(dirname "$f")"
    cp "$W/$f" "$R/$wt-untracked/$f"
  done < <(git -C "$W" ls-files --others --exclude-standard -z)
done
```
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash docs/bench-campaigns/2026-09-03-gpu-one-risc/quarantine.sh; echo "EXIT=$?"
```
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && git add docs/bench-campaigns && git status --porcelain
```
5. `edit: docs/bench-campaigns/2026-09-03-gpu-one-risc/quarantine-check.sh (new file) — the apply-check loop below, verbatim, under `#!/usr/bin/env bash`; each check runs against that worktree's OWN recorded HEAD, never against main [round-2 j6]:`
```
#!/usr/bin/env bash
R=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered
for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-$wt apply --check \
      "$R/$wt-tracked.patch" 2>&1 | sed "s/^/$wt /"
  echo "$wt apply-check EXIT=${PIPESTATUS[0]}"
done
```
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash docs/bench-campaigns/2026-09-03-gpu-one-risc/quarantine-check.sh
```

##### expect
- N = **10** non-empty `*-tracked.patch`, **10** `*-head.txt`, **10** `*-stat.txt`, and a distinct `$wt-untracked/` tree for **every** worktree whose `git status` shows `??` (R7 records 6 files under `all`)
- **N==0 is RED**; an empty patch for a worktree R7 lists as dirty is RED; a shared or missing untracked tree is RED
- each `*-stat.txt` is reconciled against R7's table (`proxima-wt-all` = 13 files +3859/-197, 6 untracked); a mismatch means the tree moved since the ledger [round-4 synth S5]
- patches and untracked **contents** are committed to this branch — never `/tmp`, which is OS-cleared

##### predict
none — this card produces records, not a measurement.

##### kill
a patch fails `git apply --check` **at its own recorded HEAD** ⇒ that worktree mutated since the R7 audit; re-audit before anything is landed from it.

##### memory gate
- gate: MG-1 — no build, no process
- what this card allocates: zero — it copies patch/status text and untracked file contents into the worktree's own docs tree

##### rollback
`git revert` one commit; the patches remain in history, which is the point.

##### blast
`docs/bench-campaigns/` only.

##### observe
patch count, per-patch line counts vs R7's table, per-worktree HEAD, `*-stat.txt` reconciliation against R7's table [round-4 synth S5], ten `apply --check` exit codes.

##### reprove
the apply-check loop above (command 4).

##### row
```
## ROW <NEXT> -- the measured-but-uncommitted wins enter git from each worktree's own HEAD, untracked files included and not colliding

**Card:** 0.7. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-1 — no build, no process; zero bytes.
**Predict (one rung ahead, written before running):** none — this card produces records, not a measurement. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| n/a | git apply --check per worktree | 10 | n/a — pass/fail, not a rate | <...> |
**Gates:** n/a — no build, no crate gate.
**Parity:** n/a — no model forward pass runs in this card.
**Census:** 10 tracked patches, 10 head files, per-worktree untracked-file counts, 10 apply-check exit codes.
**Home-turf arm:** none — this card is a git-quarantine operation.
**Principles engaged and what each changed:** §III G2 (never enter another card's worktree; cross-worktree reads use `git -C`); §X conflict 7 (13 worktrees, not 49). **Abandoned:** none.
**Re-prove:** the apply-check loop (command 4).
```

##### report skeleton
```
CARD 0.7 — <STATUS>
ran: <each command above, EXIT>
N: N=10 tracked patches, N=10 head files, N=10 apply-check exit codes
numbers: <per-worktree patch line counts vs R7's table>
files: docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/*  diff --stat:
row: <see above>
reprove: the apply-check loop
open:
```

---

#### CARD 0.8 — The harness N-contract, the hit formula, and the `symbols` dump `[S2 0.12, B3 G5, crit k, round-2 j6]`

tier: worker
depends_on: [0.5]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none (dump is instrument-gated)          default: n/a

##### opens
- `proxima-model-interop/src/bind.rs:3040-3059` — `let forward_calls_taken = generated.0.len() + usize::from(generated.2);` at `:3051`
- `proxima-model-interop/src/bind.rs:3053-3055` — `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, …")`
- `proxima-model-interop/src/bind.rs:3057-3059` — `assert_eq!(runtime.plan_misses, forward_calls_taken, …)`
- `proxima-model-interop/src/generate.rs:905` — the **single** `plan_hits` definition
- `proxima-model-interop/src/generate.rs:968` — the increment
- `proxima-model-interop/src/generate.rs:973` — `self.plans.clear()` — the fact that makes the hit formula *consecutive-key*, not *distinct-key* [correction to S2 G4]
- `proxima-model-interop/src/generate.rs:966` — the key
- `proxima-model-interop/src/generate.rs:1393` — `symbols`

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: proxima-model-interop/src/bind.rs:3052-3055 and proxima-model-interop/src/bind.rs:3056-3059 — TWO assertions restated as G5's formula: the plan_hits == 0 assertion at proxima-model-interop/src/bind.rs:3052-3055 and the plan_misses == forward_calls_taken assertion at proxima-model-interop/src/bind.rs:3056-3059 both become plan_hits + plan_misses == F; plan_misses == 1 + #{consecutive key changes}; plan_misses >= 1, without weakening either assertion; on main the formula reduces to plan_hits == 0, so today's behaviour is preserved exactly [round-4 synth S6]`
2a. `edit: proxima-model-interop/src/bind.rs (new test module) — two negative-path tests: an injected hit trips the plan_hits == 0 assertion (proxima-model-interop/src/bind.rs:3052-3055); an injected extra miss trips the plan_misses == forward_calls_taken assertion (proxima-model-interop/src/bind.rs:3056-3059) [round-4 synth S6]`
3. `edit: proxima-model-interop/src/generate.rs:966 region — add a symbols-per-step dump behind the instrument feature, printing one (new_count, symbol1) pair per step, so the key sequence is observed, not assumed`
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.8-gate.log; echo "EXIT=${PIPESTATUS[0]}"
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.8-cell-08.log; echo "EXIT=${PIPESTATUS[0]}"
```
6.
```
grep -o 'symbols=([0-9]*,[0-9]*)' /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.8-cell-08.log
```

##### expect
- N1 the harness prints exactly `F` breakdown lines
- N2 on main `plan_hits == 0`, `plan_misses == F`, `stopped_by_eos` recorded
- N3 both negative-path tests fire: an injected hit trips the plan_hits == 0 assertion (proxima-model-interop/src/bind.rs:3052-3055) and an injected extra miss trips the plan_misses == forward_calls_taken assertion (proxima-model-interop/src/bind.rs:3056-3059) [round-4 synth S6]
- N4 the dump prints one `(new_count, symbol1)` pair per step
- **No card anywhere assumes the prompt length** — it is read from the dump
- N==0 is RED

##### predict (nano → micro)
the dumped pairs on main are `(prompt_tokens, 0)` then `(1, cached_len)` with `cached_len` strictly increasing, so every consecutive key differs — which *is* the mechanism `plan_hits=0` is, and the sequence 6.1 changes.

##### kill
`plan_hits != 0` on unmodified main ⇒ the counter's semantics moved since R13; stop and re-derive before 6.1 is designed.

##### memory gate
- gate: MG-3 (it runs the harness)
  1. `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  2. `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  3a. PREFILL peak `device_allocated_bytes` <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  3b. STEADY peak `device_allocated_bytes` over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
      (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  4. `plan_cache_len` <= 1 on every step
  5. `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
- what this card allocates: test-only; the `symbols` dump is instrument-gated text output, zero heap growth

##### rollback
`git revert`; test-only.

##### blast
the `proxima-model-interop/src/bind.rs` test module only, edited once, here. [round-4 synth S6]

##### observe
`plan_hits`, `plan_misses`, `plan_cache_len`, `tokens_generated`, `stopped_by_eos`, the `symbols` dump (**NEW**, `generate.rs:966` region under `instrument`).

##### reprove
the welded BENCH cell (command 5) + `grep -o 'symbols=([0-9]*,[0-9]*)' runs/0.8-cell-08.log` (command 6).

##### row
```
## ROW <NEXT> -- the harness asserts the defect as a formula: the cache is cleared on miss, so a hit needs the immediately preceding key

**Card:** 0.8. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none (dump is instrument-gated), default: n/a.
**Allocation budget (hot/setup/cold):** MG-3, test-only, zero heap growth from the dump.
**Predict (one rung ahead, written before running):** dumped pairs are (prompt_tokens,0) then (1,cached_len) strictly increasing, every consecutive key differs. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| ours | BENCH (cached decode loop) | <...> | <...> | <...> |
**Gates:** omega gate `ran_count`/`passed_count` <N run/N passed>.
**Parity:** n/a — assertion/formula restatement, no oracle drift claimed.
**Census:** plan_hits, plan_misses, plan_cache_len, tokens_generated, stopped_by_eos, symbols dump, N=<...> steps.
**Home-turf arm:** none — this card is a harness-contract card, not a comparison cell.
**Principles engaged and what each changed:** §III G5 (the N contract, EOS-invariant, as a formula, never a literal token count); §I.2 D4. **Abandoned:** none.
**Re-prove:** the welded BENCH cell + `grep -o 'symbols=([0-9]*,[0-9]*)' runs/0.8-cell-08.log`.
```

##### report skeleton
```
CARD 0.8 — <STATUS>
ran: <each command above, EXIT>
N: N1 F breakdown lines, N2 plan_hits==0/plan_misses==F, N3 injected-hit assertion fires, N4 symbols dump pairs
numbers: <F, S, plan_hits, plan_misses, the (new_count,symbol1) sequence>
predict vs observed:
files: proxima-model-interop/src/bind.rs, proxima-model-interop/src/generate.rs  diff --stat:
row: <see above>
reprove: the welded BENCH cell + grep -o 'symbols=([0-9]*,[0-9]*)' runs/0.8-cell-08.log
open:
```

---

#### CARD 0.9 — The RISC's cardinality, the row protocol, ai_docs, and a clean tree `[S2 0.9+0.10+0.13, B3 P0.7, crit RS-4, SD-3]`

tier: worker
depends_on: [0.5, 0.6]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00          branch: risc/0-measure-truth          base: 4be2f3a
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none          default: n/a

##### opens
- `proxima-tensor/src/op.rs:166` — the doc header reads **"The four generators"** directly above `pub enum Op` at `:175`, which has **5** variants
- `proxima-tensor/src/op.rs:60-78` — `ScalarOp`, **17**, verified this session
- `proxima-tensor/src/bind.rs:221-264` — `BoundOpKind`, 4
- `proxima-tensor/src/map.rs:134-152` — `IndexMap`, 2
- `proxima-tensor/docs/discipline.md:18736` — **ROW 233**
- `ai_docs/AGENT.md` — "add records … instead of bypassing"
- `ai_docs/{index,examples-index,task-routes,invariants}.jsonl` — ai_docs is FOUR JSONL files, 95 lines total on main; R0: zero tensor/omega/GPU records [round-4 synth S7]
- `git status --porcelain` on main — `?? proxima-onnx/scripts/torch_reference/venv/`

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs
```
2. `edit: proxima-tensor/src/op.rs:166 — correct the doc header from "The four generators" to name all 5 variants`
3. `edit: proxima-tensor/src/op.rs (new test module) — add four tests written as exhaustive match statements over constructed values with no _ arm, over Op, BoundOpKind, ScalarOp, and IndexMap, so adding a variant to any of the four breaks the build`
4. `edit: docs/bench-campaigns/2026-09-03-gpu-one-risc/row-protocol.md (new file) — write the row protocol per G11`
5. `edit: ai_docs/index.jsonl — append 1 record (proxima.omega.gpu_lane)`
6. `edit: ai_docs/task-routes.jsonl — append 1 record (gpu-decode-perf, done_when = ["every GB/s row cites a measured denominator", "every route cites a Route value, never a source substring", "MG-3 recorded on every decode cell", "every geometry constant traces to omega-runtime.toml", "the census sum equals ENCODE_DISPATCH_CALLS"])`
7. `edit: ai_docs/invariants.jsonl — append 6 records (one bound plan; route-is-a-value; no lock on a per-dispatch path; KV arena bytes are a build-time bound; profile bytes are tensor bytes; a sixth invariant "the uniform buffer cache is content-keyed and must not grow per token" with evidence_required = UNIFORM_CACHE_LEN flat over steady steps), each with evidence_required [round-4 synth S7]`
8. `edit: proxima-onnx/scripts/torch_reference/.gitignore — add venv/`
9.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo nextest run -p proxima-tensor \
  --features std -E 'test(risc_cardinality)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/runs/0.9-cardinality.log; echo "EXIT=${PIPESTATUS[0]}"
```
10.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/index.jsonl > /dev/null && echo "index_ok=$?"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/examples-index.jsonl > /dev/null && echo "examples-index_ok=$?"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/task-routes.jsonl > /dev/null && echo "task-routes_ok=$?"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/invariants.jsonl > /dev/null && echo "invariants_ok=$?"
```
11.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash ai_docs/query.sh gpu-decode-perf
```
12.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && git status --porcelain | wc -l
```
13. (e) open `.github/workflows/proxima-tensor.yml:176-205` (the `omega-gate.sh` job and the `omega-compare-bench` job that runs `cargo bench -p omega --features metal --bench metal_vs_cpu -- --quick` on `macos-latest`) and record in `row-protocol.md` the rule: every feature this plan declares (`metal-kv-resident`, `kv-capacity-bucket`, `metal-plan-stable-buffers`, `kv-scatter-write`, `attention-single-range`, `metal-wide-reduce`, `metal-packed-split-k`) and the `omega_q4k_body` cfg is added to the CI matrix by the card that lands it — the CI job-set is not the gate-glob (memory: hand-picked gates miss CI's config) [round-4 fix F4]

##### expect
- N1 four exhaustive-match tests compile with no wildcard; `grep -c "The four generators" proxima-tensor/src/op.rs == 0`
- N2 `jq -c .` parses all four JSONL files; **8 new records**, not 7; `bash ai_docs/query.sh gpu-decode-perf` returns ≥1 row [round-4 synth S7]
- N3 `git status --porcelain | wc -l == 0`
- **N==0 on any of the three is RED**

##### predict (nano → micro)
the four matches compile today, i.e. the cardinalities are 5/4/17/2 as R5 and this session read them, and `grep -ci "omega\|metal\|gpu" ai_docs/task-routes.jsonl` moves from **0** to ≥1.

##### kill
- a match needs a wildcard ⇒ a cardinality is not what was read; stop and re-derive the one-RISC binding
- malformed JSONL or an empty query ⇒ read `query.sh` and fix the record shape rather than bypassing the index

##### memory gate
- gate: MG-1 — doc, tests and JSONL
- what this card allocates: zero — doc edits, four match-exhaustiveness tests, JSONL appends, one gitignore line

##### rollback
`git revert`.

##### blast
`op.rs` doc + tests, four JSONL files [round-4 synth S7], one ignore file, one docs file.

##### observe
the four variant counts; record counts per file; query hit count; the porcelain line count.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo nextest run -p proxima-tensor --features std -E 'test(risc_cardinality)'` + the four `jq` commands [round-4 synth S7] (command 10).

##### row
```
## ROW <NEXT> -- the RISC's doc said four over five variants and its tripwire watched two of four closed sets; now all four are compile errors to change, and the GPU lane enters ai_docs

**Card:** 0.9. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none, default: n/a.
**Allocation budget (hot/setup/cold):** MG-1 — doc, tests, JSONL only; zero bytes.
**Predict (one rung ahead, written before running):** four matches compile at cardinalities 5/4/17/2; task-routes.jsonl gpu/metal/omega grep moves 0 -> >=1. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
|---|---|---|---|---|
| n/a | risc_cardinality test + 3x jq parse + query.sh | 5 | n/a — pass/fail, not a rate | <...> |
**Gates:** proxima-tensor `cargo nextest -E 'test(risc_cardinality)'` <N run/N passed>.
**Parity:** n/a — no model forward pass runs in this card.
**Census:** Op=5, BoundOpKind=4, ScalarOp=17, IndexMap=2 variant counts; 8 new JSONL records across four files; 1 query.sh hit; porcelain line count == 0. [round-4 synth S7]
**Home-turf arm:** none — this card is doc/test/index hygiene, not a decode cell.
**Principles engaged and what each changed:** §II non-negotiables (Op stays 5, BoundOpKind stays 4, ScalarOp stays 17, the tripwire in 0.9 covers all four sets [crit RS-4, SD-3]); ai_docs/AGENT.md ("add records … instead of bypassing"). **Abandoned:** none.
**Re-prove:** the `risc_cardinality` test + the four `jq` commands [round-4 synth S7].
```

##### report skeleton
```
CARD 0.9 — <STATUS>
ran: <each command above, EXIT>
N: N1 four exhaustive matches + header grep, N2 jq parse x4 + 8 records + query hit, N3 porcelain==0
numbers: <variant counts 5/4/17/2; record counts per JSONL file; query hit count>
predict vs observed:
files: proxima-tensor/src/op.rs, docs/bench-campaigns/2026-09-03-gpu-one-risc/row-protocol.md, ai_docs/index.jsonl, ai_docs/task-routes.jsonl, ai_docs/invariants.jsonl, proxima-onnx/scripts/torch_reference/.gitignore  diff --stat:
row: <see above>
reprove: the risc_cardinality test + the three jq commands
open:
```

---

cards: 9 | commands: 26 | citations: 43
### 5.3.1 PHASE 1 — recover what exists into git; adjudicate the parallel branch

#### CARD 1.1 — Adjudicate `BoundOpKind::CachedAttention` `[S2 0.7, B3 P1.1]`

tier: judge
depends_on: [0.7]
worktree: none — read-only adjudication via `git -C /Users/brianbruggeman/repos/slot-0/proxima show …`; never enter a `proxima-wt-*`          branch: n/a          base: n/a
target_dir: n/a — this card runs no build          lock: n/a — this card runs no build/probe/bench command
feature: none       default: n/a

##### opens
- `git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs` — the diff hunks under adjudication; read the hunks themselves, never the commit list.
- `git show perf/cached-attention-streaming:failure-cached-attention-matcher.md` — the branch's own record: a BoundOp-only matcher was abandoned as "a heuristic" that "cannot prove the semantic roles", superseded by the structural matcher this ruling still rejects.
- `proxima-tensor/src/bind.rs:221-264` — `BoundOpKind`, the closed 4-variant set `CachedAttention` would become a fifth member of.
- `op.rs:175-266` — `Op`, the sibling closed 5-variant set the same constraint binds.
- workspace `AGENTS.md` — "we should not be adding arbitrary rules/code for specific instances."

##### the ruling, four independently sufficient grounds
(1) a fifth variant of a closed 4-variant set minted for one model's attention shape (AGENTS.md); (2) §1 — the expression already exists (two `Reduce`s with an online-softmax combine, `spec.rs:2596-2720`), so the type buys no caller capability; (3) MEASURED not to buy wall time — R12: 51.535 ON vs 51.571 OFF (CoV 1.75–2.16%) with dispatches 1194 → 616 and GPU **worse** (39.841 vs 35.117); (4) the matcher is itself a per-token CPU cost — their ROW 247: `prepare` 150.7 ms/token before indexing, 11.6 after, because `plan_hits=0`. **REJECT** `CachedAttention`, `physical.rs`, `render_cached_attention`, the bind.rs matcher, the `libm` dep. **KEEP** as separate cards: `prune_dead` (1.3), the consumer index (1.3), the paired Q4_K body (1.3 → 4.2), and their ROW 263 classifier-mislabel **finding** as the third witness on 3.2's row (their fix — a second marker string — is **not** ported). **KEEP as recorded negatives, never re-proposed:** float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode, nsg=2 (now a **fourth**-time negative across both lanes, R4 + R12 ROW 267).

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00` (Phase 0's already-created worktree; not created here)
2. `mkdir -p none/runs`
3. `cd /Users/brianbruggeman/repos/slot-0/proxima && git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs > none/runs/1.1-diff.patch`
4. `cd /Users/brianbruggeman/repos/slot-0/proxima && git show perf/cached-attention-streaming:failure-cached-attention-matcher.md > none/runs/1.1-failure-record.md`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming > none/runs/1.1-commits.txt && wc -l < none/runs/1.1-commits.txt`
5a. `cd /Users/brianbruggeman/repos/slot-0/proxima && git diff --stat main..perf/cached-attention-streaming > none/runs/1.1-diffstat.txt && git show 216d925 --stat > none/runs/1.1-216d925-stat.txt` [round-4 synth S8]
6. edit: `none/docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md` — extract R12's ROWs 234-267 verbatim, tag each `keep` / `renumber` / `supersede` against the four-ground ruling above.

##### expect
N1 = **34** rows extracted, each tagged keep / renumber / supersede. N2 = six rulings, each with its constraint named and its premise confirmed by a `git show` line. N3: `git log --oneline main..perf/cached-attention-streaming | wc -l == 42`, accounted member-by-member against the ruling. N4 (B4's diff-stat pins): `git diff --stat main..perf/cached-attention-streaming` shows `physical.rs` +576, `bind.rs` +666, `discipline.md` +938; `git show 216d925 --stat` names `prune_dead`/`dead_resolved_nodes`; a mismatch means the branch moved [round-4 synth S8]. **N==0 is RED.**

##### predict
none — a recorded decision boundary.

##### kill
- if 9.3 measures that ≤23 ops/layer is unreachable through affine write placement, this adjudication re-opens **with that number attached** (pre-registered re-open condition).
- if `git show` contradicts a premise (e.g. `prune_dead` is entangled with the macro-op), that item's ruling is re-derived from the diff and recorded as a correction.

##### memory gate
- gate: MG-1 — no build, no process runs the model.
- this card allocates nothing: no build, no process.

##### rollback
docs revert; a ruling is reversed only by new evidence in its own row.
##### blast
docs only (`docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md`).
##### observe
the 42-commit accounting; the row count (34).
##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming | wc -l`

##### row
```
## ROW <NEXT> -- the fifth bound kind the binding does not admit, rejected on four grounds, four pieces kept
**Card:** 1.1. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** n/a — no process runs.
**Predict (one rung ahead, written before running):** none — a recorded decision boundary. **Observed:** <the four-ground ruling text with each premise's git-show line>. **Miss category + work item:** none.
| arm | cell | n | CoV | load before/after |
| none — adjudication only, no timed cell | — | — | — | — |
**Gates:** none run (MG-1, no build).
**Parity:** n/a — no code path executed.
**Census:** 42-commit accounting (member-by-member); 34-row extraction count.
**Home-turf arm:** none — no timed cell.
**Principles engaged and what each changed:** §1 (new-caller-capability question — `CachedAttention` buys nothing a caller could not already do); AGENTS.md's closed-set invariant. **Abandoned:** `BoundOpKind::CachedAttention`, `physical.rs`, `render_cached_attention`, the bind.rs structural matcher, the `libm` dep — ruled out by AGENTS.md's hard invariant, §1's binary question, R12's own measurement (gpu_exec worse, 39.841 vs 35.117), and the matcher's own per-token cost (ROW 247: prepare 150.7 → 11.6 ms/token); re-open condition: 9.3 measuring ≤23 ops/layer unreachable through affine write placement.
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming | wc -l`.
```
```
## ROW <NEXT> -- the parallel lane's measured negatives, renumbered onto main
**Card:** 1.1. **Worktree/branch/commit:** proxima-wt-risc00/risc/0-measure-truth/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** n/a — no process runs.
**Predict (one rung ahead, written before running):** none — a recorded decision boundary. **Observed:** <the 34 tagged ROW extractions>. **Miss category + work item:** none.
| arm | cell | n | CoV | load before/after |
| none — adjudication only, no timed cell | — | — | — | — |
**Gates:** none run (MG-1, no build).
**Parity:** n/a.
**Census:** 34 extracted rows tagged keep/renumber/supersede; float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm-at-decode, nsg=2 recorded as negatives never to re-propose.
**Home-turf arm:** none — no timed cell.
**Principles engaged and what each changed:** §21 (do not re-propose a dead lever); R4's dead-levers register extended by R12's fourth nsg=2 negative. **Abandoned:** nsg=2 (fourth-time negative, R4 + R12 ROW 267); float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode (R12 ROWs 249, 251-254, 259-260, 265-267).
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima && git show perf/cached-attention-streaming:failure-cached-attention-matcher.md`.
```

##### report skeleton
```
CARD 1.1 — <STATUS>
ran: <each command above, EXIT>
N: 34 rows extracted / 42 commits accounted
numbers: <34-row keep/renumber/supersede table>
predict vs observed: none — a recorded decision boundary
files: docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md  diff --stat:
row: see above (two rows)
reprove: cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming | wc -l
open:
```

---

#### CARD 1.2 — Recover the mask-fma Q4_K body onto today's main `[S2 2.1, B3 P1.2]`

tier: worker
depends_on: [1.1, 0.7]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01          branch: risc/1-q4k-mask-fma          base: risc/0-measure-truth
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none       default: n/a

##### opens
- `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`) — the codec-unpack block whose body this card replaces.
- `omega/src/msl.rs:2452-2530` (`push_packed_row_blocked_body`, `lanes_per_block` at `:2516`, loop step at `:2527`) — the packed-row-blocked emission the recovered body slots into.
- `ggml-metal.metal:5086-5193` (incumbent `kernel_mul_mv_q4_K_f32_impl<4,2,32>`) — mask **without** shift, branch-free `kmask1/2/3` at `:5147-5150`, 1/16 and 1/256 folded into the scale at combine `:5171-5175` (R8) — the mechanism being ported.
- `$R/all-tracked.patch` and `$R/all-head.txt` (0.7's recovery of `proxima-wt-all` / `perf/gpu-all-wins`) — the source of the hunks this card applies.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/runs`
5. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 -b risc/1-q4k-mask-fma risc/0-measure-truth`
6. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. `R=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered`
9. edit: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/omega/src/msl.rs:190-311,2452-2530` — `git apply --3way $R/all-tracked.patch` restricted to the Q4_K hunks in `msl.rs`, then strip to the one mask-fma body only (`proxima-wt-all` carries seven features per R7; every other feature's hunk is discarded); resolve the 9-commit conflict against `spec.rs +8735/-2836` and the rest — the conflicts are the work, not a blocker.
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/runs/1.2-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/runs/1.2-oracle.log; echo "EXIT=${PIPESTATUS[0]}"`
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target CARGO_TERM_COLOR=never bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo run --release -p omega --features metal,cpu,instrument --example q4k_matvec_probe 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/runs/1.2-q4k-probe.log; echo "EXIT=${PIPESTATUS[0]}"`
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
N1 gate PASS with `ran_count`/`passed_count` recorded (either ==0 is RED). N2 the oracle asserts `2651`/`"known"`. N3 `omega/tests/q4k_real_checkpoint_parity.rs` runs ≥1 case with the mask-fma body selected. **N==0 is RED.**

##### predict (nano → micro)
`q4k_matvec_probe` at the `ffn_up` shape shows **≥20% lower** ns/op than main's body (R7's −36% on ffn families is MEMORY and the floor is conservative; the row states the anchor is MEMORY, not this card's own measurement).

##### kill
- oracle drift (G10 — `generated_text` or the single-token oracle `2651`/`"known"` moves).
- parity max-abs error vs `cpu::evaluate` on real `blk.0.attn_q.weight` > **1e-4** (§14 — the body does not land at any speed).
- `ran_count` falls below 0.5's recorded value.

##### memory gate
- gate: MG-1 for the gate step; MG-3 for the oracle step:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2; R13 prefill 310-357 MB).
- what this card allocates: a kernel-body change allocates nothing; any device increase is a **NEGATIVE**.

##### rollback
`git -C /Users/brianbruggeman/repos/slot-0/proxima worktree remove --force /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && git -C /Users/brianbruggeman/repos/slot-0/proxima branch -D risc/1-q4k-mask-fma`; nothing depends on this branch until 4.1.
##### blast
`omega/src/msl.rs` Q4_K body only.
##### observe
`ran_count`, `passed_count`, oracle token/text, `q4k_macs` (`proxima-model-interop/src/bind.rs:2856-2860`).
##### reprove
the gate + oracle commands above, welded (steps 10-11).

##### row
```
## ROW <NEXT> -- the measured -17.2% mask-fma Q4_K body, rebased nine commits forward and committed
**Card:** 1.2. **Worktree/branch/commit:** proxima-wt-risc01/risc/1-q4k-mask-fma/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-1 (gate)/MG-3 (oracle); RSS ceiling 400 MB; a kernel-body change allocates nothing, any device increase is a NEGATIVE.
**Predict (one rung ahead, written before running):** q4k_matvec_probe at ffn_up shows >=20% lower ns/op than main's body (MEMORY anchor, R7 -36%). **Observed:** <ns/op from step 12>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| ours mask-fma | q4k_matvec_probe @ ffn_up | <n> | <CoV> | <load> |
| ours mask-fma | ORACLE (single forward pass) | 1 | n/a | <load> |
**Gates:** omega <ran_count/passed_count> (features: metal,cpu,instrument).
**Parity:** q4k_real_checkpoint_parity vs f32 oracle on real blk.0.attn_q.weight, max-abs error <= 1e-4.
**Census:** q4k_macs (proxima-model-interop/src/bind.rs:2856-2860).
**Home-turf arm:** none — no decode-cell comparison in this card (micro rung only).
**Principles engaged and what each changed:** §14 (correctness oracle stays the gate on every kernel-body change); recovery-not-reinvention (§1 relocation — the body already exists on `perf/gpu-all-wins`, only the rebase is new work). **Abandoned:** none this card.
**Re-prove:** the gate + oracle commands, welded (steps 10-11 above).
```

##### report skeleton
```
CARD 1.2 — <STATUS>
ran: <each command above, EXIT>
N: gate <ran/passed>; oracle 1/1; parity suite <n/n>
numbers: q4k_matvec_probe ns/op (mask-fma) vs R7's MEMORY -36% anchor
predict vs observed: <one line; if miss: category + work item>
files: omega/src/msl.rs:190-311,2452-2530  diff --stat:
row: see above
reprove: steps 10-11
open:
```

---

#### CARD 1.3 — Cherry-pick the paired body, `prune_dead`, and the consumer index — without their row numbers `[S2 0.8+0.9+2.2, B3 P1.3, crit RB-1]`

tier: worker
depends_on: [1.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02          branch: risc/1-recovered-picks          base: risc/0-measure-truth
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none       default: n/a

##### opens
- `git log --oneline --reverse main..perf/q4k-independent-accumulators` — the ordered commit list this card cherry-picks from.
- `git show 216d925` ("drop dead resolved nodes before GPU dispatch") — the `prune_dead` commit, generic and RISC-conformant.
- R12 ROW 257 (paired Q4_K body: family 47.8 → 33.9 ms, −29%, parity 3.1e-6) — the second pick's measured result.
- R12 ROW 248 (consumer index, behind ROW 247's `prepare` 150.7 → 11.6) — the third pick's measured result.
- `proxima-tensor/src/bind.rs:200-215` (`BoundOp`) — the struct the consumer index and `prune_dead` both operate over.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/runs`
5. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 -b risc/1-recovered-picks risc/0-measure-truth`
6. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick -x --no-commit 216d925` — pick 1: `prune_dead` ("drop dead resolved nodes before GPU dispatch"); excludes `physical.rs`, `CachedAttention`, the matcher, the `libm` dep per 1.1's ruling.
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git checkout HEAD -- proxima-tensor/docs/discipline.md`
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git diff --cached --name-only | grep -c discipline.md` — must be 0, else RED.
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git commit -m "perf(bind): drop dead resolved nodes before GPU dispatch"` — subject to G13's owner-authorization gate; the diff is prepared and staged for the owner's commit if not pre-authorized.
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick -x --no-commit <sha>` (locate: `git log --oneline --reverse main..perf/q4k-independent-accumulators` — the paired-Q4_K-body commit, R12 ROW 257) — pick 2.
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git checkout HEAD -- proxima-tensor/docs/discipline.md`
14. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git diff --cached --name-only | grep -c discipline.md` — must be 0, else RED.
15. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git commit -m "perf(msl): pair Q4_K nibble accumulation"` — same authorization note as step 11.
16. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick -x --no-commit <sha>` (locate: `git log --oneline --reverse main..perf/q4k-independent-accumulators` — the consumer-index commit, R12 ROW 248) — pick 3.
17. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git checkout HEAD -- proxima-tensor/docs/discipline.md`
18. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git diff --cached --name-only | grep -c discipline.md` — must be 0, else RED.
19. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git commit -m "perf(bind): consumer index for plan preparation"` — same authorization note as step 11.
20. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/runs/1.3-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
21. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/proxima-tensor-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/runs/1.3-tensor-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
22. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/runs/1.3-bench.log; echo "EXIT=${PIPESTATUS[0]}"`
23. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/runs/1.3-oracle.log; echo "EXIT=${PIPESTATUS[0]}"`
24. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
exactly **3** commits, zero `discipline.md` hunks; `op_count` **strictly less than 1196** — this card publishes **`OPS_AFTER_PRUNE`** as a MEASURED number every downstream card quotes, never the expression "1196 minus the pruned count" [crit O-5]; `prepare_ms` **below** R13's 1.97; `generated_text` unchanged; ≥2 new `prune_dead` tests. **`op_count == 1196` is RED for this card specifically.** **N==0 is RED.**

##### predict (milli → bench)
`encode_dispatch_calls` falls from 1196 by exactly the pruned-node count and every removed node is nameable; `gpu_exec_ms` moves **< 0.2 ms** (R13: the 39 degenerate constant/iota control ops total 0.169 ms); `prepare_ms` **≤ 1.4** (−30%, R12 ROW 248 anchored).

##### kill
- `gpu_exec_ms` moves > 0.2 ms ⇒ live nodes were removed — a correctness event; diff `generated_text` and stop.
- `prepare_ms` unchanged ⇒ the consumer index does not bind here; record the negative and drop that commit.
- oracle drift ⇒ revert (§14).

##### memory gate
- gate: MG-3 —
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2).
- `prune_dead` **removes** buffers, so `device_allocated_bytes` must be `<=` 0.5's steady value — an increase is a NEGATIVE and rolls back regardless of the dispatch win.

##### rollback
per-commit `git revert`; the three picks are independent by construction.
##### blast
`proxima-tensor/src/bind.rs` — **cross-backend**, so the CPU oracle runs on this branch, not only Metal.
##### observe
`OPS_AFTER_PRUNE` (**NEW**, printed via `op_count` at `generate.rs:107`), `encode_dispatch_calls`, `prepare_ms`, `device_allocated_bytes`, `generated_text`.
##### reprove
the welded BENCH + ORACLE cells (steps 22-23).

##### row
```
## ROW <NEXT> -- three generic pieces recovered from the parallel branch, without its row numbers; the post-prune op count becomes the number every later card quotes
**Card:** 1.3. **Worktree/branch/commit:** proxima-wt-risc02/risc/1-recovered-picks/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB; device_allocated_bytes must be <= 0.5's steady value (prune_dead removes buffers, an increase is a NEGATIVE).
**Predict (one rung ahead, written before running):** encode_dispatch_calls falls by exactly the pruned-node count; gpu_exec_ms moves < 0.2 ms; prepare_ms <= 1.4. **Observed:** OPS_AFTER_PRUNE = <n>; gpu_exec_ms delta = <n>; prepare_ms = <n>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| ours (3 picks) | BENCH (cached decode loop) | <n> | <CoV> | <load> |
| ours (3 picks) | ORACLE (single forward pass) | 1 | n/a | <load> |
**Gates:** omega <ran_count/passed_count>; proxima-tensor <ran/passed>.
**Parity:** oracle 2651/"known" unchanged (§14); n/a fixture beyond that.
**Census:** OPS_AFTER_PRUNE, encode_dispatch_calls, prepare_ms, device_allocated_bytes.
**Home-turf arm:** none — no incumbent comparison in this card.
**Principles engaged and what each changed:** crit O-5 (a MEASURED number, never a subtraction expression); crit RB-1 / G11 (no literal ROW from the parallel branch's 234-267 numbering lands beside main's own). **Abandoned:** none this card (physical.rs, CachedAttention, the matcher, and libm excluded per 1.1's ruling, not abandoned here).
**Re-prove:** steps 22-23 above.
```

##### report skeleton
```
CARD 1.3 — <STATUS>
ran: <each command above, EXIT>
N: 3 commits / 0 discipline.md hunks; gate <ran/passed> x2; BENCH+ORACLE 1/1 each
numbers: OPS_AFTER_PRUNE, encode_dispatch_calls delta, gpu_exec_ms delta, prepare_ms
predict vs observed: <one line; if miss: category + work item>
files: proxima-tensor/src/bind.rs  diff --stat:
row: see above
reprove: steps 22-23
open:
```

---

### 5.3.2 PHASE 2 — one bound plan (brief item 1), closed EARLY
*Conflict 8: this lands in Phase 2, not behind the emitter reorganisation. Every claim about "the plan" downstream is unfalsifiable while D3 stands, and this is the widest-signature change in the plan — cheapest when nothing else is in flight. Worktree `proxima-wt-risc03` / `risc/2-one-bound-plan`, base `risc/1-recovered-picks`.*

#### CARD 2.1 — The plan-identity test: a per-op fingerprint vector through `pub` accessors, pre-registered RED `[B3 P2.1, S2 0.14, crit RS-5, MS-6, e]`

tier: worker
depends_on: [0.9, 1.3]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03          branch: risc/2-one-bound-plan          base: risc/1-recovered-picks
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: `instrument` (existing feature; gates the two new `pub` accessors)       default: n/a — existing feature, not new

##### opens
- `omega/src/metal.rs:1003` (`bind(...)`) — where the Metal driver binds the plan.
- `omega/src/metal.rs:1013` (`correct_packed_matmul_layouts(&mut resolved, …)`) — the post-bind rewrite this test proves exists.
- `omega/src/metal.rs:1005-1012` (the comment: `layout_of` "assumes every operand is stored row-major in its DECLARED axis order … never true for a packed Q4_K/Q5_K/Q6_K weight") — the rewrite's own stated reason.
- `omega/src/metal.rs:859-864` — `struct Prepared` and its `resolved` field are **private**, verified this session.
- `omega/src/metal.rs:313-325` — `Plan`'s fields private.
- `proxima-tensor/src/cpu.rs:358` (`bind::bind`, no rewrite) — the CPU path that skips the Metal-only rewrite.
- `proxima-tensor/src/cpu.rs:280` (`Prepared<'block>`) — the CPU-side equivalent struct.
- `proxima-tensor/src/bind.rs:1718-1722` (`bind` has **no** backend parameter).
- `proxima-tensor/src/bind.rs:200-215` (`BoundOp`), `:95-98` (`Layout {base, strides}`).
- `proxima-tensor/src/bind.rs:1594-1606` (`layout_of`).
- `omega/src/msl.rs:4656` (`emit_is_deterministic_byte_equal`) — the golden-source test pattern this card's fingerprint pairs with in 3.1.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs`
5. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 -b risc/2-one-bound-plan risc/1-recovered-picks`
6. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. edit: `proxima-tensor/src/bind.rs` — add `pub fn fingerprint_ops(plan: &[BoundOp]) -> Vec<u64>`, one FNV-1a-64 per op (not one per plan), over a canonical serialization: node id, dtype, extents, kind discriminant; for `Elementwise` every `ComposedBody` step's `ScalarOp` + operand indices; for `Reduce` `element_body`, `reduce_op`, `init`, `keep`, `output_axes`, `out_layout.base`/`strides`, `out_scatter` fields; per operand `(source, Layout, Option<Lookup>)`; for `Constant` `value.to_bits()`.
9. edit: `omega/src/metal.rs:859-864` — add `#[cfg(feature="instrument")] pub fn bound_ops(&self) -> &[BoundOp]` on `Plan`.
10. edit: `proxima-tensor/src/cpu.rs:280` — add the CPU-side equivalent `#[cfg(feature="instrument")] pub fn bound_ops(&self) -> &[BoundOp]` on `Prepared<'block>`.
11. edit: new test file — build the real cached-forward program with real symbols and real blocks, capture each backend's ops **after that backend's own rewrites** (Metal after `metal.rs:1013`, CPU after `cpu.rs:358`), compare vector to vector, report the differing node ids. Land `#[ignore]`d with a `// EXPECTED RED, see R16 / metal.rs:1013` marker, plus a companion always-green test asserting the differing node set == `packed_operands.keys()`.
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features metal,cpu,instrument --run-ignored all -E 'test(plan_fingerprint)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.1-fingerprint.log; echo "EXIT=${PIPESTATUS[0]}"`
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
the test **FAILS**, naming every differing op's node id and both `Layout{base,strides}` values. **A PASS is RED** — it would mean the fingerprint does not cover `Layout.strides`. Differing ops expected = **exactly the packed matmul weight operands' consumers**; `N differing == 0` is RED. Lands `#[ignore]`d with the `EXPECTED RED` marker plus the companion always-green test. **N==0 (test not run) is RED** — the test is gated on `metal` **and** `cpu` together, the likeliest trap.

##### predict (nano → micro)
`fingerprint_ops` over `OPS_AFTER_PRUNE` ops costs **< 100 µs**, i.e. < 0.01% of R13's `prepare_ms` 1.97.

##### kill
- the differing set includes a node **not** in `packed_operands` ⇒ there is a **third** rewrite; that finding outranks every performance card and 2.2 is re-scoped to name it.
- a card that makes the test green by loosening the field list has inverted it; **the field list is fixed here and is not negotiable downstream.**

##### memory gate
- gate: MG-1; an allocation-counter assertion over 1000 calls asserts zero heap in `fingerprint_ops`.
- what this card allocates: zero heap in `fingerprint_ops` itself (asserted); the pub accessors add no allocation.

##### rollback
`git revert`; one pure function, two `#[cfg]` accessors, one test file.
##### blast
`proxima-tensor/src/bind.rs` +1 function; `omega/src/metal.rs` +1 accessor; `cpu.rs` +1 accessor. Zero hot path.
##### observe
the two op-vectors and the differing node set (**NEW**, the test's own output).
##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features metal,cpu,instrument --run-ignored all -E 'test(plan_fingerprint)'`

##### row
```
## ROW <NEXT> -- brief item 1 is false on main today: the Metal driver rewrites the bound plan after bind, and here are the node ids it rewrites
**Card:** 2.1. **Worktree/branch/commit:** proxima-wt-risc03/risc/2-one-bound-plan/<sha at report time>. **Feature:** instrument (existing).
**Allocation budget (hot/setup/cold):** MG-1; zero heap in fingerprint_ops asserted over 1000 calls.
**Predict (one rung ahead, written before running):** fingerprint_ops over OPS_AFTER_PRUNE ops costs < 100 us (<0.01% of prepare_ms 1.97). **Observed:** <measured µs>; differing node set = <list>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| omega (metal,cpu,instrument) | plan_fingerprint nextest | 1 | n/a | <load> |
**Gates:** omega <ran/passed> (features: metal,cpu,instrument).
**Parity:** n/a — the test is a structural comparison, not an oracle run.
**Census:** the two per-op fingerprint vectors; the differing node-id set == packed_operands.keys() (companion always-green test).
**Home-turf arm:** none — structural test, no timed cell.
**Principles engaged and what each changed:** §VIII item 11 — a single u64 plan fingerprint was ruled out (crit e: three separate requirements need node identities), so this is a per-op Vec<u64> plus the pub accessors crit RS-5 requires. **Abandoned:** a single u64 plan fingerprint (§VIII.11).
**Re-prove:** cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features metal,cpu,instrument --run-ignored all -E 'test(plan_fingerprint)'.
```

##### report skeleton
```
CARD 2.1 — <STATUS>
ran: <each command above, EXIT>
N: fingerprint test 1/0 (EXPECTED RED) + companion always-green 1/1
numbers: differing node ids; both Layout{base,strides} values; fingerprint_ops timing (µs)
predict vs observed: <one line; if miss: category + work item>
files: proxima-tensor/src/bind.rs +fn, omega/src/metal.rs +accessor, proxima-tensor/src/cpu.rs +accessor, new test file  diff --stat:
row: see above
reprove: step 12
open:
```

---

#### CARD 2.2 — `bind` owns the packed layout; the post-bind rewrite is deleted `[B3 P2.2, S2 7.6, R16, R18]`

tier: worker
depends_on: [2.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 (continues from 2.1)          branch: risc/2-one-bound-plan          base: risc/1-recovered-picks
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none       default: n/a

##### opens
- `omega/src/metal.rs:375` (`packed_operands_of`, verified this session — it lives in omega while `QuantizedBlock` lives in `proxima-tensor/src/cpu.rs:3084-3110`).
- `omega/src/metal.rs:412` (its call site), `:1003`, `:1013`, `:191` (the import).
- `proxima-tensor/src/bind.rs:1618-1707` (the rewrite and its doc), `:1718` (`pub fn bind`).
- `proxima-tensor/src/cpu.rs:358` (the CPU path that does not apply the rewrite).

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
6. edit, COMMIT 1 of three, each green: `proxima-tensor/src/bind.rs:1718-1722` — `bind` gains `packed: &BTreeSet<NodeId>` as a parameter; `correct_packed_matmul_layouts` (`:1618-1707`) becomes **private**; the public function is **removed**, not deprecated (§15) — together with the re-export at `proxima-tensor/src/lib.rs:244` and the four omega TARGET call sites migrated in the SAME commit (`omega/tests/attn_multi_axis_tiled_gemm_parity.rs:223`, `:312`, `omega/examples/attention_tiled_gemm_probe.rs:121`, `omega/examples/real_forward_packed_probe.rs:124`), which `omega-gate.sh [2/6] --all-targets --all-features` builds, plus the doc references at `proxima-model-interop/src/bind.rs:719, :729` and `hf_bind.rs:315` — a commit that deletes the function and leaves the targets is not a green bisect point [round-4 fix F5] [round-4 synth S10].
6a. edit, COMMIT 2 of three: `omega/src/metal.rs:1013` — delete the call to `correct_packed_matmul_layouts`; `cpu.rs:358` passes its own packed set, derived from its own `QuantizedBlock`s, to `bind`. `PackedOperands` stays `BTreeMap<NodeId, PackedCodec>` at `omega/src/msl.rs:656` (`PackedCodec` an omega enum at `:591`), so proxima-tensor never names that return type — `bind` takes exactly the `&BTreeSet<NodeId>` `correct_packed_matmul_layouts` already took (`proxima-tensor/src/bind.rs:1648`), which `metal.rs:1013` built as `packed_operands.keys().copied().collect()` [round-4 synth S10].
6b. edit, COMMIT 3 of three: `proxima-tensor/src/bind.rs` (2.1's test) — remove the `#[ignore]`/`EXPECTED RED` marker now that the fingerprint test is green [round-4 synth S10].
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.2-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
7a. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/proxima-tensor-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.2-proxima-tensor-gate.log; echo "EXIT=${PIPESTATUS[0]}"` [round-4 synth S10]
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo test -p proxima-tensor --all-features 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.2-tensor-test.log; echo "EXIT=${PIPESTATUS[0]}"`
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.2-oracle.log; echo "EXIT=${PIPESTATUS[0]}"`
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && grep -rn correct_packed_matmul_layouts --include='*.rs' proxima-tensor omega proxima-model-interop | grep -v 'proxima-tensor/src/bind.rs'` [round-4 synth S10]
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
N1 2.1's fingerprint test turns **GREEN** and its `#[ignore]`/`EXPECTED RED` marker is removed in commit 3. N2 `grep -rn correct_packed_matmul_layouts --include='*.rs' proxima-tensor omega proxima-model-interop | grep -v 'proxima-tensor/src/bind.rs'` returns **0** (crate-scoped, per G2) — main returns 22 hits in 9 files today; hits inside `proxima-tensor/src/bind.rs` (the private definition, its doc and its own tests such as `:2267`) are expected and are not counted [round-4 fix F5] [round-4 synth S10]. N3 every Q4_K/Q5_K/Q6_K parity suite green (`q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward`) with counts recorded. N4 `generated_text` and `2651`/`"known"` unchanged. N5 `scripts/proxima-tensor-gate.sh` `ran_count`/`passed_count`, either 0 is RED [round-4 synth S10]. **N==0 is RED.**

##### predict (nano → micro)
emitted MSL is **byte-identical** (`emit_is_deterministic_byte_equal`, `msl.rs:4656`) and `q4k_matvec_probe` output is byte-identical — the corrected layout **is** the layout Metal already executed; only *where* the correction happens changes.

##### kill
- any byte drift, any parity regression, or a still-unequal fingerprint ⇒ a **third** rewrite; name it from the node set and stop.
- if the CPU oracle drifts, the CPU path **does** read packed bytes through `layout_of`, contradicting R16 — the finding supersedes the design and the correction must live behind a per-backend physical-layout query instead.

##### memory gate
- gate: MG-3 (the oracle run):
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2).
- what this card allocates: no allocation change (behaviour-preserving relocation).

##### rollback
three `git revert`s, in reverse commit order: revert commit 3, then commit 2, then commit 1 [round-4 synth S10]; 2.1 returns to its documented RED once commit 1 is reverted [round-4 fix F5].
##### blast
**cross-crate and cross-backend**: `bind`'s signature, every caller in proxima-tensor / omega / proxima-model-interop, `metal.rs:1003-1013`. Deliberately early, when nothing else is in flight.
##### observe
the fingerprint vectors, the grep count, `ran_count`, the six parity suites, the oracle, `scripts/proxima-tensor-gate.sh`'s `ran_count`/`passed_count` [round-4 synth S10].
##### reprove
steps 7-10 above, welded [round-4 synth S10].

##### row
```
## ROW <NEXT> -- bind owns the packed layout; no backend rewrites the plan after bind, and metal.rs:1013 is gone
**Card:** 2.2. **Worktree/branch/commit:** proxima-wt-risc03/risc/2-one-bound-plan/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB; no allocation change expected.
**Predict (one rung ahead, written before running):** emitted MSL byte-identical; q4k_matvec_probe output byte-identical. **Observed:** <byte-diff result>; grep count = <n>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| omega (metal,cpu,instrument) | plan_fingerprint (now green) | 1 | n/a | <load> |
| proxima-model-interop | ORACLE (single forward pass) | 1 | n/a | <load> |
**Gates:** omega <ran/passed>; proxima-tensor-gate.sh <ran/passed> [round-4 synth S10]; proxima-tensor <ran/passed> (cargo test --all-features).
**Parity:** q4k_real_checkpoint_parity, q4k_matmul_layout, q4k_unpack, metal_parity, backend_parity, metal_real_forward — all green, counts recorded.
**Census:** grep -rn correct_packed_matmul_layouts (crate-scoped: proxima-tensor omega proxima-model-interop, excluding proxima-tensor/src/bind.rs) count == 0 [round-4 synth S10]; fingerprint vectors equal.
**Home-turf arm:** none — structural/parity card, no timed decode cell.
**Principles engaged and what each changed:** §1 relocation (packed_operands_of moves to the crate owning both argument types, nothing minted); §15 (removed, not deprecated); §VIII item 12 — landing this fix in Phase 2 rather than behind the emitter reorganisation, because bind's signature is the widest change in the plan and cheapest when nothing else is in flight. **Abandoned:** landing the second-rewrite fix late, behind the emitter reorganisation (§VIII.12).
**Re-prove:** steps 7-10 above [round-4 synth S10].
```

##### report skeleton
```
CARD 2.2 — <STATUS>
ran: <each command above, EXIT>
N: 3 commits, each green; fingerprint 1/1 (now green); grep count 0 (crate-scoped); parity suites <n/n> x6; oracle 1/1; proxima-tensor-gate.sh <ran/passed> [round-4 synth S10]
numbers: grep hit count; byte-diff result for emitted MSL and q4k_matvec_probe output
predict vs observed: <one line; if miss: category + work item>
files: proxima-tensor/src/bind.rs, proxima-tensor/src/cpu.rs, omega/src/metal.rs:375,412,1003,1013,191  diff --stat:
row: see above
reprove: steps 7-10 [round-4 synth S10]
open:
```

---

#### CARD 2.3 — Re-anchor: same plan, same numbers `[B3 P2.3]`

tier: hands
depends_on: [2.2, 0.5]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 (continues from 2.2)          branch: risc/2-one-bound-plan          base: risc/1-recovered-picks
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none       default: n/a

##### opens
(none named beyond 0.5's own opens, which this card reruns — see 0.5's card for `docs/bench-campaigns/2026-09-03-gpu-one-risc/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log` and `common/common.h:328`.)

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
6. run the 0.5 three-arm sweep on this branch, interleaved A B C A B C against the 0.5 anchor binary — `llama -fa 0` / ours / `llama -fa 1`, 5 rounds:
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target std,metal,instrument 5 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/runs/2.3-sweep.log; echo "EXIT=${PIPESTATUS[0]}"`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
`step_wall_ms`, `gpu_exec_ms`, `op_count` (== `OPS_AFTER_PRUNE`) and `generated_text` all **within 0.5's CoV band**; EVERY phase counter (`prepare_ms`, `emit_ms`, `block_upload_ms`, `op_setup_ms`, `pipeline_lookup_ms`, `encode_dispatch_ms`, `readback_ms`) and `greedy_pick_ms` also within its own 0.5 band, not only wall/gpu/op_count [round-4 synth S11]. A no-op cell by construction; its value is proving a three-crate signature change moved zero milliseconds. **N==0 is RED.**

##### predict (bench, anchor)
predicts nothing further; it is the anchor for Phase 3 onward.

##### kill
any metric outside the band ⇒ the move was not behaviour-preserving; bisect 2.2's three sub-moves.

##### memory gate
- gate: MG-3, every 0.5 clause:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2); incumbent arms' peak RSS via `/usr/bin/time -l`.
- what this card allocates: none (measurement only, no code change).

##### rollback
see 2.2. **blast** none (measurement).
##### observe
every 0.5 counter: `step_wall_ms`, `gpu_exec_ms`, `gpu_device_ms` (0.3), `op_count`, `plan_hits`/`plan_misses`/`plan_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, RSS, CoV, loadout.
##### reprove
the 0.5 sweep on this branch (step 7).

##### row
```
## ROW <NEXT> -- one bound plan lands with zero measured cost
**Card:** 2.3. **Worktree/branch/commit:** proxima-wt-risc03/risc/2-one-bound-plan/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB; no allocation change (measurement only).
**Predict (one rung ahead, written before running):** predicts nothing further; anchor for Phase 3 onward. **Observed:** step_wall_ms = <n>, gpu_exec_ms = <n>, op_count = <OPS_AFTER_PRUNE>, generated_text unchanged. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| llama.cpp-Metal -fa 0 | tg32 t/s | 5 | <CoV> | <load> |
| ours | BENCH (step_wall_ms/gpu_exec_ms) | 5 | <CoV> | <load> |
| llama.cpp-Metal -fa 1 | tg32 t/s | 5 | <CoV> | <load> |
**Gates:** none new this card (measurement only, 2.2's gates already recorded).
**Parity:** generated_text identical to 0.5's anchor; n/a fixture beyond that.
**Census:** plan_hits/plan_misses/plan_cache_len, kv_cache_upload_bytes, device_allocated_bytes.
**Home-turf arm:** llama.cpp-Metal -fa 0 and -fa 1, both arms of 0.5's three-arm sweep.
**Principles engaged and what each changed:** G6 (interleaved A B C A B C arms, CoV banded); this card is the re-anchor proving 2.1+2.2's cross-crate signature change was behaviour-preserving. **Abandoned:** none this card.
**Re-prove:** step 7 above (the 0.5 sweep on this branch).
```

##### report skeleton
```
CARD 2.3 — <STATUS>
ran: <each command above, EXIT>
N: 3 arms x 5 rounds = 15 cells
numbers: step_wall_ms, gpu_exec_ms, op_count, CoV per arm, load before/after
predict vs observed: <one line; if miss: category + work item>
files: none (measurement only)  diff --stat: none
row: see above
reprove: step 7
open:
```

---

### 5.3.3 PHASE 3 — the route becomes a value
*Worktree `proxima-wt-risc04` / `risc/3-route-value`, base `risc/2-one-bound-plan`. R13's 225/385/547 split is `classify_kind`'s substring output and R12 ROW 263 measured that instrument relabelling 9/601 → 225/385 when a body changed — so the census lands **before** any body swap.*

#### CARD 3.1 — `Route`, unit-only, with `fn slot(&self) -> usize`, decided by `emit` itself `[S2 1.1, B3 P3.1, crit RS-2, f, OB-2]`

tier: worker
depends_on: [2.3]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04          branch: risc/3-route-value          base: risc/2-one-bound-plan
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — `route::of` is **not** feature-gated (a gated route means two routers)       default: n/a

##### opens, in this order
- `omega/src/msl.rs:673-697` — `emit`'s 4-kind + `Keep` match.
- `omega/src/msl.rs:731-775` — `kernel_cache_key`: three route characters, `'G'` if `tiled_gemm_block(..).is_some()` else `'B'` if `packed_row_block(..).is_some()` else `'S'`, with `:751-758` recording the ordering as load-bearing.
- `omega/src/msl.rs:730-736` — `kernel_cache_key` is `pub(crate)` and `#[cfg(any(test, all(feature = "metal", target_os = "macos")))]`, so N2 runs under a macOS metal build while N4 runs under the alloc-tier build — two build configurations, stated [round-4 synth S12]
- `omega/src/msl.rs:1690` — `entry_name(&BoundOp)` keeps its signature; the `Route` must never enter the emitted name because `kernel_cache_key` starts from it [round-4 synth S12]
- `omega/src/msl.rs:3164-3187` — `push_cooperative_reduce_body`'s **third** re-derivation of the same gates.
- `omega/src/msl.rs:797`, `:1517-1560`, `:824`, `:1235`, `:1450`, `:1487` — `kernel_dispatch_shape`/`grid_threads`, `reduce_is_cooperative`, `packed_row_block`, `tiled_gemm_block`, `diagnose_packed_row_block`.
- `omega/src/metal.rs:777-826` and `:835-854` and `:709-710` — `classify_kind`, `diagnose_kind`, and their op-timed call site.
- `proxima-tensor/src/instrument.rs:809-828`/`:842`/`:848-864` — the `(NodeId, reason)` **shape** to mirror, **explicitly not its `Mutex<BTreeMap>`**.
- `proxima-telemetry/src/metric/counter.rs:12-17` — **`Counter` holds an `AtomicU64` and is not `Copy`**, verified.

##### the design, both questions answered
*Pipe question:* `Route` is a decision value computed once per `BoundOp` and consumed once by `emit` — no stages, no backpressure; `backend.rs:1-52` already adjudicated this boundary not-a-pipe. *Relocation question, call site both ways:* Way A `let (route, reason) = route::of(bound, packed); match route { … }` makes `assert_eq!(route::of(bound).0, recorded(node))` and `assert_eq!(Σ ROUTE_DISPATCHES, ENCODE_DISPATCH_CALLS)` **possible**; Way B (today) is four hand-ordered `if let` gates in three places plus a substring recovery MEASURED to mislabel. New caller capability ⇒ **the type is earned.**

##### the enum, and why `Declined` carries no data [conflict 9, crit RS-2/f/OB-2]
`#[derive(Clone,Copy,PartialEq,Eq)] #[repr(u8)] pub enum Route { Elementwise, Scan, Iota, Constant, ReduceSerial, ReduceCooperative, ReduceRowBlockedPacked, ReduceTiledGemm, Declined }` — **unit-only**, so no `route as usize` (E0605 on a data-carrying variant) ever appears; the mapping is an explicit `pub fn slot(&self) -> usize { match self { Elementwise => 0, … Declined => 8 } }`. The **reason** travels beside the route as a separate `DeclineReason` (reusing `diagnose_packed_row_block`'s existing reason set), never inside the variant. `route::of` returns `(Route, DeclineReason)` and is **the function `emit` branches on**, so census and emission cannot disagree.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs`
5. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 -b risc/3-route-value risc/2-one-bound-plan`
6. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. edit: new `omega/src/route.rs` — the `Route`/`DeclineReason` enum and `route::of`, per the design above.
9. edit: `omega/src/msl.rs:673-697,731-775,3164-3187,797,1517-1560,824,1235,1450,1487` — `emit`, `kernel_cache_key`, `kernel_dispatch_shape`/`grid_threads`, and `push_cooperative_reduce_body` all consume `route::of` instead of their own re-derivations.
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build -p omega --no-default-features --features alloc 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.1-alloc-build.log; echo "EXIT=${PIPESTATUS[0]}"`
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.1-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(route) + test(emit_is_deterministic_byte_equal)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.1-route-tests.log; echo "EXIT=${PIPESTATUS[0]}"`
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
N1 a table-driven test returning each of the **8 dispatching variants** for a hand-built `BoundOp` at that shape — a variant with no case is RED (an unroutable variant is an invented variant). N2 **`kernel_cache_key` byte-stability**, run under the macOS metal build (`kernel_cache_key` is `pub(crate)` and `#[cfg(any(test, all(feature = "metal", target_os = "macos")))]`) [round-4 synth S12]: for every op the key through `route::of` is byte-identical to main's three-character logic, G-before-B preserved. N3 **golden emitted source**: the emitted MSL for every op in the real program hashes identically before and after, extending `emit_is_deterministic_byte_equal` to a cross-commit golden. N4 the alloc-tier build compiles `route.rs` and **states which modules it built** — a second, distinct build configuration from N2's [round-4 synth S12]. N5 an `emit_ms` control within ±5% of R13's 0.81 ms [round-4 synth S12]. **N==0 is RED.**

##### predict (nano → micro)
behaviour-neutral: golden hashes match for all `OPS_AFTER_PRUNE` ops; `q4k_matvec_probe` ns/op unchanged within CoV, because no kernel text changed.

##### kill
one golden hash drifts or one `kernel_cache_key` differs ⇒ the refactor changed routing, which is a different card; **do not proceed to 3.2.**

##### memory gate
- gate: MG-1.
- what this card allocates: `route::of` is a pure decision function; no new heap-holding static or thread_local.

##### rollback
`git revert`; `route::of` is **not** feature-gated (a gated route means two routers).
##### blast
new `omega/src/route.rs`; `msl.rs` three decision sites collapse to one. `wgsl.rs`/`cuda.rs` untouched here (3.3 gives them the route).
##### observe
the 8-case table; the golden hashes; `kernel_cache_key` equality; `emit_ms` against R13's 0.81 ms control [round-4 synth S12].
##### reprove
steps 10-12 above.

##### row
```
## ROW <NEXT> -- the route stops being a substring of the kernel it selected: one unit-only enum with an explicit slot map, and emitted MSL byte-identical
**Card:** 3.1. **Worktree/branch/commit:** proxima-wt-risc04/risc/3-route-value/<sha at report time>. **Feature:** none (route::of not feature-gated).
**Allocation budget (hot/setup/cold):** MG-1; no new heap-holding static/thread_local.
**Predict (one rung ahead, written before running):** golden hashes match for all OPS_AFTER_PRUNE ops; q4k_matvec_probe ns/op unchanged within CoV. **Observed:** <golden hash diff result>; <ns/op delta>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| omega (all-features) | route + emit_is_deterministic_byte_equal nextest | <n> | n/a | <load> |
**Gates:** omega alloc-tier build EXIT <..> (modules built: msl, wgsl, cuda); omega full gate <ran/passed>.
**Parity:** emit_is_deterministic_byte_equal golden hash, cross-commit.
**Census:** 8-case dispatching-variant table; kernel_cache_key byte-stability check.
**Home-turf arm:** none — structural/golden-hash card, no timed decode cell.
**Principles engaged and what each changed:** §VIII item 10 — route as usize over a data-carrying Declined(reason) ruled out by E0605 and OB-2, replaced by unit-only Route + explicit slot(); conflict 9 resolution (reason carried beside the route, never inside the variant). **Abandoned:** route as usize over a data-carrying variant; a hot counter slot for Declined (§VIII.10).
**Re-prove:** steps 10-12 above.
```

##### report skeleton
```
CARD 3.1 — <STATUS>
ran: <each command above, EXIT>
N: 8-case table 8/8; kernel_cache_key stability N/N; golden hash N/N; alloc-tier build EXIT
numbers: golden hash equality; kernel_cache_key byte-diff count; q4k_matvec_probe ns/op delta
predict vs observed: <one line; if miss: category + work item>
files: omega/src/route.rs (new), omega/src/msl.rs:673-697,731-775,3164-3187,797,1517-1560,824,1235,1450,1487  diff --stat:
row: see above
reprove: steps 10-12
open:
```

---

#### CARD 3.2 — Delete `classify_kind`; the census, lock-free, cost-bounded `[S2 1.1 census half, B3 P3.2, crit RS-1, OB-1, OB-2]`

tier: worker
depends_on: [3.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 (continues from 3.1)          branch: risc/3-route-value          base: risc/2-one-bound-plan
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: `instrument` (existing feature; gates the 8 counters — the route itself is not gated)       default: n/a

##### opens
- `omega/src/metal.rs:709-710`, `:785-826`, `:835-854` — `classify_kind`/`diagnose_kind` and the op-timed call site being deleted.
- `omega/src/metal.rs:2243` (`counter!(ENCODE_DISPATCH_CALLS, 1)` inside `#[cfg(feature="instrument")]`), `:2244-2246` (`ENCODE_DISPATCH_TICKS`).
- `omega/src/metal.rs:1484` (`Counter::new`).
- `omega/src/metal.rs:578-585` (`plan_named`) — where the cold table is filled.
- `omega/src/metal.rs:1577` (`snapshot_and_reset`).
- `proxima-tensor/src/instrument.rs:842-864` — **the pattern NOT to copy** (`Mutex<BTreeMap>`).

##### the census, hot and cold
*Cold, once per plan* at `plan_named`: `Plan.routes: Vec<(Route, DeclineReason)>` filled at plan build, single-owner, no synchronisation — **the census of record**, keyed `(NodeId, Route, reason)`. *Hot, per dispatch* at `:2243`: `ROUTE_DISPATCHES[route.slot()].add(1)` on a **`[Counter; 8]`** written as **eight explicit `Counter::new("omega.metal.route_dispatches.<variant>")` const initializers** (`Counter` is non-`Copy`, verified) with eight matching `snapshot_and_reset` fields [crit OB-1]. **`Declined` gets NO hot slot** [conflict 9, crit OB-2]: a declined op never dispatches, so its counter would be structurally 0 and would make the sum identity vacuous — declines are observable only in the cold table, and 3.4's gate asserts **both** the hot sum identity and the cold decline count. `classify_kind` and `diagnose_kind` are **deleted**, not kept alongside; call sites read `plan.routes[position]`.

##### the cost bound, MEASURED not assumed
R13: `encode_dispatch_ms = 0.47` over 1196 dispatches = **393 ns/dispatch** (DERIVED). Budget **≤ 5% = 19.6 ns/dispatch = 0.0235 ms/token**. The card runs census-ON vs census-OFF interleaved and asserts the `encode_dispatch_ms`, `op_setup_ms` and `prepare_ms` deltas are each ≤ 0.0235. The guard is **`ENCODE_DISPATCH_TICKS`**, not `gpu_exec_ms` — `gpu_exec` is the device window (`metal.rs:546-554`) and cannot see a CPU-side cost [crit OB-1].

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
6. edit: `omega/src/metal.rs` — delete `classify_kind`/`diagnose_kind` (incl. `:709-710`); add the cold `Plan.routes` table in `plan_named` (`:578-585`); add the 8 hot `Counter`s + one `.add(1)` line at `:2243`.
7. edit: `omega/examples/real_forward_packed_probe.rs`, `proxima-model-interop/src/generate.rs` — `op_profile_bucket` sources `kind` from `Route` instead of `classify_kind`.
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && grep -rn "classify_kind\|diagnose_kind" --include='*.rs' proxima-tensor omega proxima-model-interop` — must be 0 (crate-scoped, never `.`, per G2) [round-4 synth S13].
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && grep -rn "Mutex" --include='*.rs' omega/src` — must be 0.
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.2-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.2-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"` (census ON)
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.2-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"` (census OFF). **The OFF arm is the 3.1 tree with IDENTICAL features, not an instrument-off build** — `encode_dispatch_ms` and `ENCODE_DISPATCH_TICKS` only exist under `instrument`, so an instrument-off arm could not report the very counter the budget is measured on. Build the OFF binary BEFORE applying step 6's edit (`git stash` is forbidden by G2; instead run step 12 first, from the 3.1 commit, and keep its log as `3.2-bench-off.log`), then apply the edit and run step 11; interleave by alternating the two prebuilt test binaries under `target/release/deps/proxima_model_interop-*` with `--exact --nocapture --ignored bind::real_openchat_file::runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache` (the technique of `discipline.md:9904`), 3 rounds A B A B A B.
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
both greps **0 hits** (any hit is RED; §21). `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS` exactly, every step. `ENCODE_DISPATCH_CALLS / F == op_count == OPS_AFTER_PRUNE`. The per-route counts against R13's buckets (225/385/547/37/2): **a difference is not automatically RED, it is the finding** — R12 ROW 263 proves the substring classifier mislabels; any difference is reported as "the census disagrees with `classify_kind` at N ops, here are their nodes", and **every bucket number in R13 is then restated against routes**. **N==0 is RED.**

##### predict (milli → bench)
`encode_dispatch_ms` delta ON−OFF ≤ **0.0235 ms**; `step_wall_ms` ON within R13's 0.5% CoV of OFF.

##### kill
- the budget is exceeded — where the budget is `max(0.0235 ms, 2 × CoV(encode_dispatch_ms) as 0.5 measured it)`, so the kill sits outside a band that exists (G6) [round-4 fix F6] ⇒ collapse to the cold table only and re-measure; still over ⇒ the census does not go on the dispatch path at all and the row says so.
- sum ≠ `ENCODE_DISPATCH_CALLS` ⇒ a path bypassed `route::of`; **do not proceed to Phase 4.**

##### memory gate
- gate: MG-3, both ON and OFF arms:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2).
- what this card allocates: 8 `Counter`s = 448 B static; the cold vector is `2 B × op_count` ≈ 2.4 KB/plan, bounded by `plan_cache_len == 1`; anything that makes it grow per token is a KILL.

##### rollback
`git revert`; the counters are behind `instrument`, the route is not.
##### blast
`omega/src/metal.rs` (`classify_kind` deleted incl. `:709-710`; +8 counters + one line at `:2243`; the cold table in `plan_named`), `omega/examples/real_forward_packed_probe.rs`, `generate.rs` (`op_profile_bucket` sources `kind` from `Route`).
##### observe
`ROUTE_DISPATCHES[8]` (**NEW**, `metal.rs:2243`), the cold table (**NEW**, `plan_named` `:578-585`), `ENCODE_DISPATCH_CALLS`, `ENCODE_DISPATCH_TICKS`, `op_setup_ms`, `prepare_ms`.
##### reprove
the ON/OFF interleaved pair (steps 11-12) + both greps (steps 8-9).

##### row
```
## ROW <NEXT> -- the route census is a plan property: zero locks, one atomic add on a line that already had one, and a measured 5% budget
**Card:** 3.2. **Worktree/branch/commit:** proxima-wt-risc04/risc/3-route-value/<sha at report time>. **Feature:** instrument (existing, counters only).
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB; hot = 8 Counters = 448 B static; cold = 2 B x op_count ~= 2.4 KB/plan, bounded by plan_cache_len == 1.
**Predict (one rung ahead, written before running):** encode_dispatch_ms delta ON-OFF <= 0.0235 ms; step_wall_ms ON within R13's 0.5% CoV of OFF. **Observed:** <delta ms>; <step_wall_ms ON vs OFF>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| ours, census ON (instrument) | BENCH (cached decode loop) | <n> | <CoV> | <load> |
| ours, census OFF (the 3.1 commit's binary, same features) | BENCH (cached decode loop) | <n> | <CoV> | <load> |
**Gates:** omega <ran/passed>.
**Parity:** n/a — census card, no oracle-affecting change.
**Census:** ROUTE_DISPATCHES[8] sum vs ENCODE_DISPATCH_CALLS; per-route counts vs R13's 225/385/547/37/2 substring buckets.
**Home-turf arm:** none — ON/OFF self-comparison card, no incumbent arm.
**Principles engaged and what each changed:** §21 (no lock on a per-dispatch path) — §VIII item 9, a Mutex<BTreeMap> mirroring WIDTH_TILE_DECLINE ruled out by arithmetic (1196 lock/unlock sit inside the very slice Phase 6 measures); §VIII item 10, Declined gets no hot slot (OB-2, a declined op never dispatches so its slot would be structurally 0). **Abandoned:** a Mutex<BTreeMap> route census (§VIII.9); a hot counter slot for Declined (§VIII.10).
**Re-prove:** steps 11-12 (ON/OFF pair) + steps 8-9 (both greps).
```

##### report skeleton
```
CARD 3.2 — <STATUS>
ran: <each command above, EXIT>
N: grep classify_kind/diagnose_kind = 0; grep Mutex omega/src = 0; BENCH ON/OFF x <n> runs
numbers: ROUTE_DISPATCHES sum vs ENCODE_DISPATCH_CALLS; per-route counts vs R13 buckets; encode_dispatch_ms/op_setup_ms/prepare_ms deltas
predict vs observed: <one line; if miss: category + work item>
files: omega/src/metal.rs:709-710,578-585,2243, omega/examples/real_forward_packed_probe.rs, proxima-model-interop/src/generate.rs  diff --stat:
row: see above
reprove: steps 11-12, 8-9
open:
```

---

#### CARD 3.3 — Route the other two backends; the coverage matrix `[B3 P3.3, S2 7.2 partial]`

tier: worker
depends_on: [3.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 (continues from 3.2)          branch: risc/3-route-value          base: risc/2-one-bound-plan
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none       default: n/a

##### opens
- `omega/src/wgsl.rs:105`, `:364` — `PackedOperands`/`gather_count` shared type site, and `EmitError::ScatterNotSupported` raise site.
- `omega/src/cuda.rs:66`, `:146-183` (`emit_cuda` **rejects Iota and Constant** via `CudaUnsupportedOpKind`), `:241`.
- `omega/src/error.rs:53` — `EmitError::ScatterNotSupported` definition.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
6. edit: `omega/src/wgsl.rs`, `omega/src/cuda.rs` — `route::of` moves to the shared surface; `wgsl` and `cuda` each return a `Route` for every `BoundOpKind`.
7. edit: `omega/src/cuda.rs:146-183` — CUDA's `CudaUnsupportedOpKind` rejection of Iota/Constant becomes `Route::Declined(BackendLacksKind)` — a recorded, censused coverage hole, not a silent `Err`.
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.3-coverage.log; echo "EXIT=${PIPESTATUS[0]}"`
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
a coverage-matrix test, 3 backends × 5 emission shapes (the 4 `BoundOpKind`s plus `Reduce` under `Keep::Scan`, which is a field of `Reduce`, not a fifth kind — `BoundOpKind` stays 4) = **15 cells** [round-4 fix F7]; N==0 is RED. Pre-registered: Metal 4/4; WGSL 4/4 with no tiled-GEMM and no packed-row-block route; CUDA **2/4** with two named declines. The test asserts the matrix **equals** the pre-registration — a cell changing without a card is RED.

##### predict (nano → micro)
no device effect; WGSL and CUDA have no decode driver.

##### kill
the matrix disagrees with the pre-registration ⇒ R5's coverage claim is stale; re-read before the matrix is written.

##### memory gate
- gate: MG-1.
- what this card allocates: none — no process runs the model; WGSL/CUDA have no decode driver.

##### rollback
`git revert`.
##### blast
`wgsl.rs`, `cuda.rs` emit entry points.
##### observe
the 15-cell matrix; per-backend decline reasons.
##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`

##### row
```
## ROW <NEXT> -- backend route coverage, censused: CUDA is two of four kinds and now says so with a reason
**Card:** 3.3. **Worktree/branch/commit:** proxima-wt-risc04/risc/3-route-value/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-1; no process runs the model.
**Predict (one rung ahead, written before running):** no device effect; WGSL and CUDA have no decode driver. **Observed:** 15-cell matrix = <result>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| omega (all-features) | backend_route_coverage nextest | 1 | n/a | <load> |
**Gates:** omega <ran/passed> (backend_route_coverage).
**Parity:** n/a — coverage matrix, no oracle run.
**Census:** 15-cell matrix (3 backends x 4 BoundOpKind + Keep::Scan); per-backend decline reasons.
**Home-turf arm:** none — no decode driver on WGSL/CUDA.
**Principles engaged and what each changed:** brief item 4 (every backend covers every kind, or a censused Route::Declined(reason), never a silent Err). **Abandoned:** none this card.
**Re-prove:** cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'.
```

##### report skeleton
```
CARD 3.3 — <STATUS>
ran: <each command above, EXIT>
N: 15-cell coverage matrix
numbers: Metal 4/4, WGSL 4/4, CUDA 2/4 with 2 named declines
predict vs observed: <one line; if miss: category + work item>
files: omega/src/wgsl.rs:105,364, omega/src/cuda.rs:66,146-183,241  diff --stat:
row: see above
reprove: step 8
open:
```

---

#### CARD 3.4 — The census cell, the census gate, the fingerprint gate, and the elementwise concentration `[S2 1.2+3.2+7.7, B3 P3.4]`

tier: worker
depends_on: [3.3, 0.8, 2.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 (continues from 3.3)          branch: risc/3-route-value          base: risc/2-one-bound-plan
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target          lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — the gate's named feature set is not `--all-features` (which also enables `cuda`/`wgpu-backend`/`vulkan`/`npu`/`ane`)       default: n/a

##### opens
- `scripts/omega-gate.sh` — the gate steps this card extends.
- `proxima-model-interop/src/generate.rs:1721-1750` — `token_breakdown_metal`, the printer this card's histogram sources from.
- R13's elementwise bucket (**547 ops / 7.350 ms** per-op = **6.85 ms batched-equivalent** (÷1.073), 13,437 ns/op) — the bucket this card concentrates [round-4 synth S14].

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04`
2. `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
3. `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
4. `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.4-bench-route-dump.log; echo "EXIT=${PIPESTATUS[0]}"` (the route dump)
7. edit: `scripts/omega-gate.sh` — add gate steps [7/6], following the script's `[n/6]` numbering [round-4 synth S14], under a **named** feature set (never `--all-features`): `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS`; cold-table decline count == the census's; **2.1's fingerprint equality**; `grep -c correct_packed_matmul_layouts == 1`.
8. edit: `omega/src/metal.rs` (`plan_named`'s cold table) — extend with `(extents, operand_count)` per elementwise node — cold, so 3.2's budget is not re-spent — and rank by tick share.
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/runs/3.4-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && grep -c correct_packed_matmul_layouts --include='*.rs' -r .`
11. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
every op carries a `(NodeId, Route, DeclineReason)` triple; sum == `op_count`; the gate is RED on any mismatch and prints both sides; ≥1 cold row per elementwise node with the count equal to the `Elementwise` share of `ENCODE_DISPATCH_CALLS`. **N==0 is RED; a mismatch is RED.**

##### predict (nano → micro)
the route histogram **matches R13's substring buckets exactly** (225/385/547/37/2 scaled to `OPS_AFTER_PRUNE`); a mismatch means `classify_kind` mislabelled on main too and every R13 bucket is restated. Separately: the **top-5 elementwise nodes carry ≥50%** of the 7.350 ms bucket — the bucket is concentrated, not uniform.

##### kill
- any op landing on `ReduceSerial` that R13 attributes to a Q4_K/Q5_K/Q6_K family — a quantized matvec on the serial path is a route bug worth more than any kernel tweak.
- if the elementwise bucket is uniform across >200 nodes, no single-node lever exists and the only remaining lever is *fewer nodes* (Phase 9); this card closes with that pointer. Standing reason not to reach for fusion: `grep -rln fuse ggml/src` is **empty** at `b25346221` (R8).

##### memory gate
- gate: MG-3:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: **400 MB** (this card is before 5.2).
- what this card allocates: the cold table grows by ~3 fields × plan length ≈ 15 KB, bounded by `plan_cache_len == 1`.

##### rollback
`git revert` the gate steps and the census extension.
##### blast
`scripts/omega-gate.sh`, the cold recorder (`omega/src/metal.rs`, `plan_named`).
##### observe
the route histogram; the gate output; per-node elementwise ticks.
##### reprove
the welded BENCH cell (step 6) + `bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh` (step 9).

##### row
```
## ROW <NEXT> -- 1196 ops censused by route with a reason, not by grepping MSL
**Card:** 3.4. **Worktree/branch/commit:** proxima-wt-risc04/risc/3-route-value/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB; cold table grows ~3 fields x plan length ~= 15 KB, bounded by plan_cache_len == 1.
**Predict (one rung ahead, written before running):** route histogram matches R13's 225/385/547/37/2 buckets scaled to OPS_AFTER_PRUNE; top-5 elementwise nodes carry >=50% of the 7.350 ms bucket. **Observed:** <histogram>; <top-5 share>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| ours | BENCH (cached decode loop, route dump) | <n> | <CoV> | <load> |
**Gates:** omega <ran/passed> (named feature set, not --all-features); route-dispatch sum == ENCODE_DISPATCH_CALLS; cold-table decline count == census; 2.1 fingerprint equality; grep -c correct_packed_matmul_layouts == 1.
**Parity:** n/a — census/gate card.
**Census:** (NodeId, Route, DeclineReason) triple per op; elementwise per-node (extents, operand_count) ranked by tick share.
**Home-turf arm:** none — no incumbent comparison in this card.
**Principles engaged and what each changed:** brief item 2 (one route decision, censused (NodeId, reason), not recovered by grepping emitted source); §VIII item 17 (a second marker string was rejected as the fix for the ROW 263 mislabel — 3.1/3.2 make the route a value instead, and ROW 263 becomes a witness here). **Abandoned:** a second marker string to fix the classifier mislabel (§VIII.17).
**Re-prove:** step 6 (BENCH cell) + step 9 (omega-gate.sh).
```
```
## ROW <NEXT> -- one bound plan and one route are gates, not claims
**Card:** 3.4. **Worktree/branch/commit:** proxima-wt-risc04/risc/3-route-value/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** MG-3, all five clauses [round-4 synth S1], RSS ceiling 400 MB (see above row for arithmetic).
**Predict (one rung ahead, written before running):** grep -c correct_packed_matmul_layouts == 1; 2.1's fingerprint equality holds under the gate. **Observed:** <grep count>; <fingerprint gate result>. **Miss category + work item:** <inconsistency|understanding-gap|none>.
| arm | cell | n | CoV | load before/after |
| omega | omega-gate.sh (named feature set) | 1 | n/a | <load> |
**Gates:** omega <ran/passed>; Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS; cold-table decline count == census; fingerprint gate; grep -c correct_packed_matmul_layouts == 1.
**Parity:** n/a — gate card.
**Census:** the four gate assertions above, each with its pass/fail.
**Home-turf arm:** none.
**Principles engaged and what each changed:** brief item 1 (one bound plan, gated by 2.1's fingerprint equality) and brief item 2 (one route, gated by the dispatch-sum identity) both become CI gates in this card, not narrative claims. **Abandoned:** none this row.
**Re-prove:** step 9 (omega-gate.sh under the named feature set).
```

##### report skeleton
```
CARD 3.4 — <STATUS>
ran: <each command above, EXIT>
N: route dump 1 run; gate 4 assertions; grep count 1
numbers: route histogram (225/385/547/37/2 scaled); top-5 elementwise tick share; ROUTE_DISPATCHES sum vs ENCODE_DISPATCH_CALLS
predict vs observed: <one line; if miss: category + work item>
files: scripts/omega-gate.sh, omega/src/metal.rs (plan_named cold table)  diff --stat:
row: see above (two rows)
reprove: step 6, step 9
open:
```

---

cards: 10 | commands: 123 | citations: 48
### 5.3.4 PHASE 4 — the Q4_K body (R13's largest measured mass: 225 ops / 44.450 ms)
*Worktree `proxima-wt-risc05` / `risc/4-q4k-bakeoff`, base `risc/3-route-value`, with `risc/1-q4k-mask-fma` and `risc/1-recovered-picks` merged in as the two bodies.*

#### CARD 4.1 — The build-time body selector, not two cargo features `[B3 P4.1, R18, conflict 4]`

tier: worker
depends_on: [3.4, 1.2, 1.3]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05`          branch: `risc/4-q4k-bakeoff`          base: `risc/3-route-value` (merges `risc/1-q4k-mask-fma`, `risc/1-recovered-picks`)
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: none — build-time profile key `[q4k] body` (never a cargo feature, conflict 4)       default: n/a

##### opens
- `scripts/omega-gate.sh` step **[2/6]** `cargo build -p omega --all-targets --all-features` — the constraint that forbids two exclusive features
- `scripts/omega-gate.sh` step **[3/6]** `nextest --all-features`
- `omega/omega-runtime.toml` — gains the `[q4k]` section
- `omega/build.rs:16` `require_nonzero`
- `omega/build.rs:35`
- `omega/build.rs:43`
- `omega/build.rs:59`
- `omega/build.rs:79-85` `resolve_int` + `rerun-if-env-changed`
- `omega/build.rs:105` `emit_sizing_consts`
- `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`)
- `omega/src/msl.rs:2452-2530` (`push_packed_row_blocked_body`)

##### the ruling (conflict 4, §X.4)
Two mutually-exclusive cargo features + a precedence rule would make `omega-gate.sh`'s own `--all-features` gate exercise only the precedence winner, so the two arms could never be distinguished by the crate's own gate, and `compile_error!`-guarded exclusivity would turn it RED outright. Exclusive cargo features would break FIVE of six gate steps ([2/6], [3/6], [4/6]-arm-2, [5/6]-arm-2, [6/6]), not one [round-4 synth S15]. `[q4k] body` → `cargo:rustc-cfg` (guiding-principles §8's profile input) selects exactly one body under **any** feature set, makes the bake-off a rebuild, and makes the rollback one toml line. No new cargo feature. The `omega_q4k_body` cfg enters the CI matrix (G13) [round-4 synth S15].

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs
```
2.
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 -b risc/4-q4k-bakeoff risc/3-route-value
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && git merge --no-ff risc/1-q4k-mask-fma 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.1-merge-maskfma.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && git merge --no-ff risc/1-recovered-picks 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.1-merge-picks.log; echo "EXIT=${PIPESTATUS[0]}"
```
3. `edit: omega/omega-runtime.toml — add [q4k] section, key body = "pair_dot", all three values ("main","mask_fma","pair_dot") documented at the key — "main" is the body on 4be2f3a today and is the control arm C of card 4.2`
4. `edit: omega/build.rs:105 region — do NOT add proxima-build as a build-dependency and do NOT add the axis to the workspace Profile (its axes are a fixed workspace-runtime table, proxima-build/src/profile.rs:50, src/lib.rs:75-96, and omega does not depend on it); instead omega/build.rs emits the SAME directive form proxima-build uses (cargo:rustc-check-cfg=cfg(omega_q4k_body, values("main","mask_fma","pair_dot")) + cargo:rustc-cfg=omega_q4k_body="<value>" + cargo:rerun-if-env-changed=OMEGA_Q4K_BODY) beside emit_sizing_consts; the row names proxima-build/src/lib.rs:205-234 and this reason so the second mechanism is not unexplained (supersedes F8) [round-4 fix F8] [round-4 synth S15]`
5. `edit: omega/src/msl.rs:190-311, :2452-2530 — select the Q4_K body under #[cfg(omega_q4k_body="…")], no new cargo feature`
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_Q4K_BODY=mask_fma \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.1-gate-mask_fma.log; echo "EXIT=${PIPESTATUS[0]}"
```
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_Q4K_BODY=pair_dot \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.1-gate-pair_dot.log; echo "EXIT=${PIPESTATUS[0]}"
```
8.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_Q4K_BODY=nonsense \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build -p omega --all-targets --all-features \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.1-build-nonsense.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1: both valid values PASS the full gate **including** `--all-features` — the card's whole point — `ran_count`/`passed_count` recorded per value, either ==0 is RED.
- N2: the nonsense value FAILS the build with a named `build.rs` panic; a successful build on a nonsense value is RED.
- N==0 is RED.

##### predict (nano → micro)
`q4k_matvec_probe` under `pair_dot` shows lower ns/op than under `mask_fma` at the `attn_q` shape (R12 measured pair-dot there) and the reverse or a tie at `ffn_up` (R7 measured mask-fma there). If one body wins both micro shapes, the milli rung is expected to agree; a disagreement is the finding.

##### kill
- `--all-features` fails under either valid value.

##### memory gate
- gate: MG-1
- what this card allocates: none — build-time cfg selection, zero runtime allocation.

##### rollback
`git revert` the `build.rs` + toml hunks; both bodies remain in source, unselectable.

##### blast
`omega/build.rs`, `omega-runtime.toml`, `msl.rs` body selection. Nothing outside omega.

##### observe
Two gate runs' `ran_count`/`passed_count`; the nonsense-value `build.rs` panic message.

##### reprove
The three welded gate commands above with `OMEGA_Q4K_BODY` set.

##### row
```
## ROW <NEXT> -- two Q4_K bodies, one build-time selector: exclusive without breaking --all-features
**Card:** 4.1. **Worktree/branch/commit:** proxima-wt-risc05/risc/4-q4k-bakeoff/<sha at report time>. **Feature:** none — build-time profile key `[q4k] body`.
**Allocation budget (hot/setup/cold):** none — build-time selection only, zero runtime allocation (MG-1).
**Predict (one rung ahead, written before running):** pair_dot wins the attn_q micro shape, mask_fma wins or ties ffn_up (not run by this card — deferred to 4.2). **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| mask_fma | omega-gate.sh full | <ran_count>/<passed_count> | n/a | <...>/<...> |
| pair_dot | omega-gate.sh full | <ran_count>/<passed_count> | n/a | <...>/<...> |
| nonsense | cargo build --all-features | build FAIL (named panic) | n/a | <...>/<...> |
**Gates:** omega <ran_count run/passed_count passed> (features: --all-features) under mask_fma and under pair_dot; nonsense value build FAILS with a named panic.
**Parity:** n/a — no oracle run in this card.
**Census:** none new.
**Home-turf arm:** none — this card produces build-gate records, not a measurement.
**Principles engaged and what each changed:** §12 (sizing config owns every geometry constant, no bare source const); conflict 4 (build-time profile axis, not two cargo features). **Abandoned:** VIII.8 — two mutually-exclusive cargo features with a precedence rule.
**Re-prove:** the three welded gate commands above with OMEGA_Q4K_BODY set.
```

##### report skeleton
```
CARD 4.1 — <STATUS>
ran: <each command above, EXIT>
N: N1 (both values full-gate pass), N2 (nonsense value build fails)
numbers: ran_count/passed_count per value; nonsense-value panic message text
predict vs observed: deferred to 4.2 (this card produces no timing cell)
files: omega/build.rs, omega/omega-runtime.toml, omega/src/msl.rs:190-311,2452-2530  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

#### CARD 4.2 — The bake-off: batched AND per-op arms, terminal tie-break, nothing deleted `[S2 2.3, B3 P4.2, crit HC-4, O-6, g, d]`

tier: hands
depends_on: [4.1]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05`          branch: `risc/4-q4k-bakeoff`          base: `risc/3-route-value`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `OMEGA_Q4K_BODY` env override (mask_fma | pair_dot) selects the arm under test; both incumbent arms present per 0.5       default: n/a — measurement card, no default landed here

##### opens
- `msl.rs:1978-1982` — `Q4K_UNPACK_MSL`/`Q5K`/`Q6K` concatenated with no delimiter, why a "grep the region" tie-break is undecidable [R16]
- `proxima-model-interop/src/bind.rs:3084` (per-op)
- `proxima-model-interop/src/bind.rs:3002` (decode cell)
- incumbent geometry `<4,2,32>` (`ggml-metal.m:3330`, `:3215-3220`)

##### the rule — the tie-break, pre-registered, terminal, every rung decidable without reading emitted text [crit d]
1. **Parity gate.** Max-abs error vs `cpu::evaluate` on real `blk.0.attn_q.weight` > **1e-4** ⇒ out (§14; R12's recorded 3.1e-6 is the standard). All six parity suites green under each value.
2. **Route-count pin.** The comparison is void unless both arms report the same `Route::ReduceRowBlockedPacked` census count. Unequal counts mean different op sets; a body change that moves the route is RED.
3. **Primary metric — batched `gpu_exec_ms`**, mean over interleaved runs, winner iff the difference exceeds `max(CoV_A, CoV_B) × max(mean_A, mean_B)`. This rung is batched precisely because `op_profile_family` exists only in per-op mode (`generate.rs:184`, produced only by `execute_plan_op_timed`), and per-op mode inserts a commit/wait between every op, removing all inter-op overlap [crit HC-4, g].
4. Tie → summed per-op family `gpu_ms` over the seven weight families, with 0.2's true bytes and the +7.3% inflation quoted.
5. Tie → lower max-abs parity error on the same real tensor.
6. Tie → smaller total emitted MSL byte length for the full decode program (`emit_is_deterministic_byte_equal`, `msl.rs:4656`; needs no region parsing).
7. Terminal: `pair_dot` wins — it is a commit on `4be2f3a` while mask-fma was an unrebased diff off `2b95210` (R7); lower landing risk breaks the last tie. No rung can fail to decide.
Arms interleaved **A B C A B C** (one sweep per round, three rounds), where **C is the control body** = `OMEGA_Q4K_BODY=main`, the body on 4be2f3a today (4.1 keeps it selectable as the third value precisely so the control can be measured against itself first, per G6).

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs
```
`OMEGA_Q4K_BODY` selects the arm; features are identical across arms. **Selecting the arm is a rebuild** (the value is a `rustc-cfg`), so the arms are NOT rebuilt between measurements: build the three test binaries once, one per value, with `--no-run` (`cargo test --release -p proxima-model-interop --features metal,instrument --lib --no-run` under `OMEGA_Q4K_BODY=<value>` and a distinct `CARGO_TARGET_DIR` per value: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-main`, `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-mask_fma`, `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-pair_dot`), record the three binary paths, then interleave the prebuilt binaries A B C A B C with `--exact --nocapture --ignored bind::real_openchat_file::<test>` (the technique of `discipline.md:9904`); the mutex is held per run, never across a build [round-4 fix F9].
2. Per round (3 rounds), per body (mask_fma, pair_dot), the welded MILLI cell (per-family `gpu_ms`):
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_Q4K_BODY=<body> PROXIMA_METAL_OP_PROFILE_STEP=3 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.2-milli-<body>-round<N>.log; echo "EXIT=${PIPESTATUS[0]}"
```
The MILLI cell above is a **5-token cell**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` and the env profile step above are inert on this rung — milli budget 5, bench budget 8, the two rungs are never read as the same cell [round-4 judge J4].
3. Per round, per body, the welded BENCH cell (`gpu_exec_ms`, `gpu_device_ms`):
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_Q4K_BODY=<body> \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.2-bench-<body>-round<N>.log; echo "EXIT=${PIPESTATUS[0]}"
```
4. Per round, both incumbent arms (0.5's second arm included):
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99 -fa 0 -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.2-llama-fa0-round<N>.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99 -fa 1 -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.2-llama-fa1-round<N>.log; echo "EXIT=${PIPESTATUS[0]}"
```
5. Six parity suites, each welded, per value of `OMEGA_Q4K_BODY`: `q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward` — `cargo nextest run -p proxima-model-interop --release --features metal,cpu,instrument --run-ignored all -E 'test(<suite>)'`, `--wait 900` each.

##### expect
- N: 3 bodies (mask_fma, pair_dot, and control C = `OMEGA_Q4K_BODY=main`, already present per the tie-break rule above) × 3 rounds × 2 rungs + 2 incumbent arms = **≥20 cells** [round-4 synth S16]; every cell carries `generated_text` == R13's string and the parity number.
- N==0 is RED.

##### predict (milli → bench)
The packed-row-blocked bucket enters the band ladder as **44.450 ms per-op = 41.43 ms batched-equivalent (÷1.073)** [round-4 synth S16]. The winner drops `ReduceRowBlockedPacked` from **44.450** to **≤33.0 ms** (−29%, R12 ROW 257 applied to R13's mass, **DERIVED**) ⇒ `gpu_exec_ms` **56.93 → [44.9, 48.6]** and `step_wall_ms` **67.92 → [55.9, 59.6]** (predict band per synth4 §IV) [round-4 synth S16].

##### kill
- Neither body clears **−10%** on the packed-row-blocked bucket beyond both CoV bands ⇒ the −17.2%/−29% micro figures did not transfer; decompose — *inconsistency* if the bucket moved but wall did not (Phase 6 owns that), *understanding-gap* if the bucket itself missed (work item: "what else was in R12's feature-off control", which R12 records already carried the paired body). Stop the climb.
- Parity > 1e-4, or `generated_text` drift.

##### memory gate
- gate: MG-3, both arms, every run
- what this card allocates: any device increase is a NEGATIVE that rolls back the winning arm — a kernel-body change allocates nothing.
- three target dirs ≈ 3× one release build on disk; recorded, not gated [round-4 fix F9]

##### rollback
Flip `[q4k] body` — one toml line, no code fork.

##### blast
The packed-row-blocked route only: 225 of 1196 ops (R13).

##### observe
`ReduceRowBlockedPacked` census count and bucket ms, per-family ms with 0.2's bytes, `gpu_exec_ms`, `gpu_device_ms`, parity error, `generated_text`.

##### reprove
The interleaved sweep (all commands above).

##### row
```
## ROW <NEXT> -- Q4_K bake-off: two independent re-derivations of ggml's mask-without-shift, decided on a batched arm before anything was deleted
**Card:** 4.2. **Worktree/branch/commit:** proxima-wt-risc05/risc/4-q4k-bakeoff/<sha at report time>. **Feature:** OMEGA_Q4K_BODY env override, both values swept.
**Allocation budget (hot/setup/cold):** any device increase vs 0.5's steady value is a NEGATIVE (MG-3).
**Predict (one rung ahead, written before running):** packed-row-blocked bucket 44.450 ms per-op = 41.43 ms batched-equivalent (÷1.073); gpu_exec_ms 56.93 -> [44.9, 48.6]; step_wall_ms 67.92 -> [55.9, 59.6]. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S16]
| arm | cell | n | CoV | load before/after |
| mask_fma | MILLI | 3 rounds | <...> | <...>/<...> |
| mask_fma | BENCH | 5 runs x 3 rounds | <...> | <...>/<...> |
| pair_dot | MILLI | 3 rounds | <...> | <...>/<...> |
| pair_dot | BENCH | 5 runs x 3 rounds | <...> | <...>/<...> |
| llama -fa 0 | llama-bench tg32 | 5 x 3 rounds | <...> | <...>/<...> |
| llama -fa 1 | llama-bench tg32 | 5 x 3 rounds | <...> | <...>/<...> |
**Gates:** six parity suites (q4k_real_checkpoint_parity, q4k_matmul_layout, q4k_unpack, metal_parity, backend_parity, metal_real_forward), each <N run/N passed>, under both body values.
**Parity:** blk.0.attn_q.weight, max-abs error vs cpu::evaluate, <...> vs 1e-4 threshold.
**Census:** ReduceRowBlockedPacked count per arm (route-count pin, must match).
**Home-turf arm:** llama.cpp-Metal -fa 0 and -fa 1, both present, per 0.5's frequency-weighted read.
**Principles engaged and what each changed:** §II item 3/5 (one route, one sizing config); the tie-break's seven pre-registered rungs. **Abandoned:** none.
**Re-prove:** the interleaved sweep.
```

##### report skeleton
```
CARD 4.2 — <STATUS>
ran: <each command above, EXIT>
N: >=20 cells (3 bodies incl. control x 3 rounds x 2 rungs + 2 incumbent arms); six parity suites x 2 values [round-4 synth S16]
numbers: gpu_exec_ms / gpu_device_ms / MILLI family ms per body per round; llama tg32 t/s -fa 0/-fa 1
predict vs observed: winner in [44.9,48.6] gpu_exec_ms / [55.9,59.6] step_wall_ms vs <observed> [round-4 synth S16]
files: none edited — measurement card  diff --stat: n/a
row: see above
reprove: see above
open:
```

---

#### CARD 4.3 — Land one; the loser is a recorded negative, kept selectable `[S2 2.4, B3 P4.3, crit O-6]`

tier: judge
depends_on: [4.2]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05`          branch: `risc/4-q4k-bakeoff`          base: `risc/3-route-value`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `[q4k] body` toml key       default: set to the winner named by the mechanism (value stated on the row, not decided here)

##### opens
- (none new — this card adjudicates 4.2's own cells; no additional `file:line` read)

##### the ruling
Set `[q4k] body` to the winner named by the **mechanism**, not the number alone; write two rows: one for the winner with its delta vs the 2.3 anchor and against 4.2's same deflated band (`gpu_exec_ms` [44.9, 48.6], `step_wall_ms` [55.9, 59.6]) [round-4 synth S16], one for the loser with its measured number and the rung that decided it. The losing body's source is **kept and selectable** — a body that lost by 3% on one machine is the first thing to try on the next one, and nothing is deleted before a batched confirmation exists [crit O-6]. The three target dirs 4.2 built (`target-main`, `target-mask_fma`, `target-pair_dot`) are reclaimed after the row is written [round-4 synth S16].

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs
```
2. `edit: omega/omega-runtime.toml [q4k] body — set to the winner named by 4.2's mechanism`
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/runs/4.3-bench-landed.log; echo "EXIT=${PIPESTATUS[0]}"
```
4. `rm -rf /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-main /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-mask_fma /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target-pair_dot` — the three target dirs 4.2 built are reclaimed after the row is written [round-4 synth S16]

##### expect
- One row for the winner with its delta vs the 2.3 anchor; one row for the loser with its measured number and the rung that decided it. A loser row with a blank number is RED.
- N==0 is RED.

##### predict
None — a ruling over 4.2's cells.

##### kill
n/a.

##### memory gate
- gate: MG-1
- what this card allocates: none — a toml default flip.

##### rollback
The toml key.

##### blast
One toml default.

##### observe
The two rows; `encode_dispatch_calls` unchanged at `OPS_AFTER_PRUNE` — a body change that moves the dispatch count changed the route.

##### reprove
4.2's sweep at the landed value.

##### row
```
## ROW <NEXT> -- <winner> lands as the default Q4_K body; <loser> recorded at <delta> and still selectable
**Card:** 4.3. **Worktree/branch/commit:** proxima-wt-risc05/risc/4-q4k-bakeoff/<sha at report time>. **Feature:** [q4k] body toml key, landed value <winner>.
**Allocation budget (hot/setup/cold):** none — toml default only (MG-1).
**Predict (one rung ahead, written before running):** none — a ruling over 4.2's cells. **Observed:** <winner delta vs 2.3 anchor>; <loser measured number + deciding rung>. **Miss category + work item:** none.
| arm | cell | n | CoV | load before/after |
| <winner>, landed | BENCH | <...> | <...> | <...>/<...> |
**Gates:** the welded BENCH cell at the landed value, <N run/N passed>.
**Parity:** carried from 4.2's parity gate, not re-run here.
**Census:** encode_dispatch_calls == OPS_AFTER_PRUNE, unchanged.
**Home-turf arm:** carried from 4.2, not re-run here.
**Principles engaged and what each changed:** crit O-6 (nothing deleted, the loser stays selectable). **Abandoned:** VIII.8 — two mutually-exclusive cargo features with a precedence rule.
**Re-prove:** 4.2's sweep at the landed value.
```

##### report skeleton
```
CARD 4.3 — <STATUS>
ran: <each command above, EXIT>
N: two rows produced (winner, loser)
numbers: winner's delta vs 2.3 anchor; loser's measured number + deciding rung
predict vs observed: none — a ruling
files: omega/omega-runtime.toml  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

### 5.3.5 PHASE 5 — KV device residency, BEFORE bucketing
*Conflict 2: **residency first.** Worktree `proxima-wt-risc06` / `risc/5-kv-residency`, base `risc/4-q4k-bakeoff`. Residency changes no graph — the block handed to omega stays a `cached_len`-sized slice, so `element_count(shapes.of(node)) == block_element_count(block)` and the strict checks at `metal.rs:991-1000` / `cpu.rs:346-356` are **preserved, not relaxed**. Doing it the other way round is what produced the 4.1↔5.3 code-level cycle [crit RS-1, j].*

#### CARD 5.1 — Generalize the registered host span to N slots `[B3 P5.1, R18]`

tier: worker
depends_on: [4.3]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06`          branch: `risc/5-kv-residency`          base: `risc/4-q4k-bakeoff`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: none — extends the existing checkpoint-mapping primitive (§1: extend, don't add a peer)       default: n/a (not gated)

##### the two binary questions for `SpanSlot` (answered by writing the code, not by arguing) [round-3 j9]
*Can an existing primitive express it?* The expression is `register_host_span(name, ptr, bytes)` returning the index into a fixed `[Option<HostSpan>; OMEGA_RESIDENT_SPAN_SLOTS]` that today holds exactly one entry (`CHECKPOINT_MAPPING`, `metal.rs:1744-1752`); `SpanSlot` is that index, `#[repr(transparent)]` over `u8`, so a caller cannot pass a buffer length where a slot belongs. Nothing new is minted beyond the index newtype, and if the index can be a bare `usize` without a single call site confusing it with a byte count, it is a bare `usize` — the card writes the two call sites and keeps whichever compiles with fewer casts. *What can a caller do that it could not before?* Before: `register_checkpoint_mapping(ptr, bytes)` — one span, the checkpoint, and every other host buffer is copied per token (`:1903`). After: `register_host_span("kv.layer.7.v", ptr, bytes)` — the KV arena resolves through `host_span_offset` exactly as the checkpoint does, with zero copies (5.2's N2). Both lines go on the row.

##### opens
- `omega/src/metal.rs:1744-1752` (`CHECKPOINT_MAPPING`)
- `omega/src/metal.rs:1768-1773` (`register_checkpoint_mapping`)
- `omega/src/metal.rs:1786-1815` `checkpoint_mapping_offset`, whose own doc says the scratch and KV-cache buffers "never live inside the checkpoint's own mmap, so they fall through unchanged" [R18]
- `backend.rs:402-414`
- `metal.rs:1616` `is_page_aligned`
- `metal.rs:1914` `create_no_copy_buffer` (page-aligned pointer and length required)
- `omega/build.rs:99-104` — the feature-gated-consts-only-under-CARGO_FEATURE_* convention this card's unconditional `[spans] slots` const is distinguished from [round-4 synth S17]

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs
```
2.
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 -b risc/5-kv-residency risc/4-q4k-bakeoff
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target
```
3. `edit: omega/src/metal.rs:1744-1752 — the single Option slot becomes a fixed-size array of OMEGA_RESIDENT_SPAN_SLOTS, that constant from a new [spans] slots key in omega-runtime.toml via emit_sizing_consts (§12 — no bare source const); [spans] slots is emitted UNCONDITIONALLY (it feeds the default upload path), unlike omega/build.rs:99-104's convention of feature-gated consts only under CARGO_FEATURE_* — that convention governs conditionally-compiled consts, not this always-compiled one, and the row states the distinction [round-4 synth S17]`
4. `edit: omega/src/metal.rs:1768-1773 — register_checkpoint_mapping becomes register_host_span(name, ptr, bytes) -> SpanSlot; the checkpoint is slot 0 and its call site is updated`
5. `edit: omega/src/metal.rs:1786-1815 — checkpoint_mapping_offset becomes host_span_offset, scanning slots`
6. `edit: MAPPING_OFFSET_UPLOADS — gains a per-slot breakdown`
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && grep -rn 'register_checkpoint_mapping\|CHECKPOINT_MAPPING' --include='*.rs' . \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.1-grep-old-names.log; echo "EXIT=${PIPESTATUS[0]}"
```
8.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.1-gate.log; echo "EXIT=${PIPESTATUS[0]}"
```
9.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo nextest run -p omega --release \
  --features metal,cpu,instrument --run-ignored all -E 'test(host_span_slot_exhaustion)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.1-negative-path.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- `grep -rn 'register_checkpoint_mapping\|CHECKPOINT_MAPPING' --include='*.rs' .` = **0**.
- `register_host_span` present at exactly the loader call site (and the KV site after 5.2).
- A slot-exhaustion negative-path test returns a **named error**, never silently overwriting.
- Gate PASS with `ran_count` ≥ prior card's recorded value.
- N==0 is RED.

##### predict (nano → micro)
`mapping_offset_uploads` on a decode step is **unchanged at 291** (R13), because slot 0 behaves exactly as the old singleton did.

##### kill
`mapping_offset_uploads != 291` before 5.2 lands ⇒ the generalisation changed the checkpoint path.

##### memory gate
- gate: MG-3
- what this card allocates: the span table is `slots × 24 B`, fixed at build time — the number goes on the row.

##### rollback
`git revert`; the singleton returns.

##### blast
`metal.rs` upload path, `backend.rs` re-export, the loader call site.

##### observe
The two greps (old-name absence); `mapping_offset_uploads` per slot; `nocopy_cache_len`.

##### reprove
The grep + the welded gate.

##### row
```
## ROW <NEXT> -- one registered-span primitive, N slots, the checkpoint is slot 0
**Card:** 5.1. **Worktree/branch/commit:** proxima-wt-risc06/risc/5-kv-residency/<sha at report time>. **Feature:** none — extends register_checkpoint_mapping in place.
**Allocation budget (hot/setup/cold):** span table = slots x 24 B, fixed at build time (MG-3).
**Predict (one rung ahead, written before running):** mapping_offset_uploads unchanged at 291. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| generalized span | BENCH (mapping_offset_uploads) | <...> | <...> | <...>/<...> |
**Gates:** omega <ran_count run/passed_count passed> (features: --all-features).
**Parity:** n/a — instrument-only card.
**Census:** grep(register_checkpoint_mapping|CHECKPOINT_MAPPING) == 0; mapping_offset_uploads per slot.
**Home-turf arm:** none — this card produces records, not a comparison.
**Principles engaged and what each changed:** §1 (extend, don't add a peer) — register_host_span extends register_checkpoint_mapping. **Abandoned:** none.
**Re-prove:** the grep + the welded gate.
```

##### report skeleton
```
CARD 5.1 — <STATUS>
ran: <each command above, EXIT>
N: 2 greps (0 hits each expected), 1 negative-path test, gate ran_count/passed_count
numbers: mapping_offset_uploads per slot; slots x 24 B span table size
predict vs observed: mapping_offset_uploads == 291 vs <observed>
files: omega/src/metal.rs:1744-1815, omega/src/backend.rs:402-414, omega/omega-runtime.toml  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

#### CARD 5.2 — A capacity-reserved, page-aligned KV arena; `found == expected` preserved `[B3 P5.2, S2 5.3 driver half, crit RS-1, MS-4, d]`

tier: worker
depends_on: [5.1]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06`          branch: `risc/5-kv-residency`          base: `risc/4-q4k-bakeoff`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `metal-kv-resident`       default: off

##### opens
- `proxima-model-interop/src/generate.rs:621-654` (`LayerCache`, `Vec::new()`, three `extend_from_slice` `:636-640`)
- `proxima-model-interop/src/generate.rs:1804-1860` — `forward_node_values` has its OWN `LayerCache::new()` and its own `named_blocks` assembly, a second site the arena re-types too [round-4 synth S18]
- `generate.rs:1364-1389` (the per-layer KV push loop and its "full `cached_len`-sized array re-bound every step" comment)
- `generate.rs:1559` (`cached_len += new_count`)
- `omega/src/metal.rs:1903` (the fresh-buffer-every-token path)
- `metal.rs:991-1000` (`InputSizeMismatch`, **preserved untouched**)
- `cpu.rs:346-356` (its twin)
- `metal.rs:350-362` `mark_resident` (classifies by **name**)
- `metal.rs:1606` `page_size()`
- `proxima-tensor/src/align.rs:42-46`, `:56-58` ("a real host page size the caller queried itself … never hard-coded here", verified), `:69`, `:78-79` (`next_multiple_of(page_size).max(page_size)`)
- `proxima-tensor/src/align.rs` (locate: `grep -n 'fn as_slice\|fn as_ref\|impl.*Deref' proxima-tensor/src/align.rs`) — the slice the arena hands to `named_blocks` as `QuantizedBlock::Float32` needs an accessor yielding `&[f32]` from `AlignedBuffer`; the card cites `align.rs` for page size only above, this is the second, distinct need [round-4 judge J5]
- incumbent `llama-kv-cache-unified.cpp:74-118` (allocated once), `:749-788`

##### the design — the page-size source, which S2 had no answer for [crit MS-4]
`omega::metal::page_size()` exists only inside the metal+macOS module, and `metal` is optional in `proxima-model-interop`. `proxima-tensor` already depends on `libc` under `std`, so the non-Metal source is **`libc::sysconf(_SC_PAGESIZE)`**, exposed as `proxima_tensor::align::host_page_size()`; the Metal value is used when the `metal` feature is on and a test asserts the two agree on this host. A test builds `-p proxima-model-interop --features std` (no metal) to prove the arena obtains a page size on a non-Metal build.

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs
```
2. `edit: proxima-tensor/src/align.rs:56-58,69 — add host_page_size() via libc::sysconf(_SC_PAGESIZE); first production caller`
3. `edit: proxima-model-interop/src/generate.rs:621-654 — LayerCache's three Vec<f32> become three AlignedBuffers sized kv_capacity_tokens × row_elements, registered once through 5.1's register_host_span; append writes into the reserved region, base pointer never moves`
4. `edit: generate.rs:1364-1389 — named_blocks still hands &arena[..cached_len*row], element count still equals the declared Symbolic(1) extent, neither validator touched`
4a. `edit: proxima-model-interop/src/generate.rs:1804-1860 — forward_node_values's OWN LayerCache::new() and its own named_blocks assembly are re-typed to the same AlignedBuffer arena, in this same commit [round-4 synth S18]`
5. `edit: omega/src/metal.rs:350-362 — mark_resident gains the KV names so 23e2e5e's non-resident routing no longer applies`
6. `edit: proxima-model-interop-runtime.toml + new build.rs — kv_capacity_tokens as a build-time key with the G8 byte-formula assertion; context_length never sizes an allocation; feature metal-kv-resident, default-off, forwarded`
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_KV_CAPACITY_TOKENS=1000000 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build -p proxima-model-interop --release --features metal-kv-resident \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.2-trap-build.log; echo "EXIT=${PIPESTATUS[0]}"
```
8.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build -p proxima-model-interop --release --features std \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.2-no-metal-build.log; echo "EXIT=${PIPESTATUS[0]}"
```
9. BENCH cell, feature OFF then ON, interleaved 3×:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.2-bench-<arm>-<round>.log; echo "EXIT=${PIPESTATUS[0]}"
```
10.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/runs/5.2-gate.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1: a build with `PROXIMA_KV_CAPACITY_TOKENS=1000000` **fails** with the byte arithmetic in the panic — a successful build is RED, that is the 34 GB trap re-opened.
- N2: `nocopy_reuses >= 3 × layers × S` per run; `copying_uploads` for KV nodes == **0**; `BLOCK_COPIED_BYTES` for KV == 0 on every steady step (0.2's counter).
- N3: `kv_cache_upload_bytes` stops growing (+262,144 B/token → 0 after step 1).
- N4: `element_count(shapes.of(kv_node)) == block_element_count(block)` asserted on the real program with **no validator edit**.
- N5: `generated_text` and `2651`/`"known"` unchanged.
- N6: the no-metal build test; the page-size agreement test; pointer stability across appends; the qwen3.5 `DenseAttention`/`Ssm` cache states (`generate.rs:687`, `:738`) untouched with their `unreachable!` at `:1386-1388` intact; `forward_node_values` (`generate.rs:1804-1860`) runs with the feature on and reads through the same re-typed arena [round-4 synth S18].
- N7: the `&[f32]` accessor `named_blocks` needs from `AlignedBuffer` exists on main or the card adds it as a `&[f32]` view with a test; its name is recorded on the row [round-4 judge J5].
- N==0 is RED.

##### predict (milli → bench)
`block_upload_ms` **2.00 → ≤0.5** (R13 records 0.4 on step 2, the weights-only step, so the "otherwise" 1.7–3.5 is the KV term) ⇒ `step_wall_ms` **[53.5, 57.5]**; `gpu_exec_ms` **unchanged** within CoV — this card moves no GPU work, and if `gpu_exec` moves, something else changed.

##### kill
- `gpu_exec_ms` outside its CoV band; `device_allocated_bytes` peak above G8 clause 3a/3b [round-4 synth S1]; RSS slope > 1 MB/step; `generated_text` drift.
- `block_upload_ms` falls but wall does not, beyond both CoV bands ⇒ record it inside the noise band and stop (R2's "~3.3%" is MEMORY with no counterpart term in R13 and may not be claimed).

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1]
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
       (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
- what this card allocates: the arena is allocated up front (512 × 262,144 = 134,217,728 B), so prefill RSS rises ~134 MB and **the MG-2/MG-3 RSS ceiling becomes 540 MB for this card and every later one**, stated here once with its arithmetic. The card **prints the computed byte total before allocating**. Both slopes must go to 0 on the KV term — a resident cache that still grows per token has not become resident, and that is RED **on the slope**, not on the timing.

##### rollback
Feature default-off; `git revert` restores `Vec::new()`. The build-time byte assertion is **kept regardless** — it is a correctness guard and §15 forbids reverting a repair to restore a number.

##### blast
`generate.rs` KV path (`:621-654`, `:1364-1389`, `:1804-1860` — `forward_node_values`'s own `LayerCache::new()`/`named_blocks` site [round-4 synth S18]), a new `build.rs` on proxima-model-interop, `proxima-tensor/src/align.rs` (+`host_page_size`, first production caller), `metal.rs` `mark_resident` name set.

##### observe
`block_upload_ms`, `BLOCK_COPIED_BYTES`/`_NOCOPY_BOUND`/`_OFFSET_BOUND` (0.2), `nocopy_reuses`, `nocopy_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, `phys_footprint_bytes`.

##### reprove
The trap-build command + the welded BENCH cell in both arms, interleaved 3×.

##### row
```
## ROW <NEXT> -- the KV cache stops round-tripping through the host: one registered span per layer at capacity, addressed by offset, and the validators never moved because the block is still cached_len elements
**Card:** 5.2. **Worktree/branch/commit:** proxima-wt-risc06/risc/5-kv-residency/<sha at report time>. **Feature:** metal-kv-resident, default-off.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1]; PREFILL_CAP 4_654_467_728 B / STEADY_CAP 4_505_367_728 B at kv_capacity_tokens=512 (DERIVED); RSS ceiling 540 MB (400 + 134,217,728 B arena), arithmetic stated here [round-4 fix F10] [round-4 synth S1].
**Predict (one rung ahead, written before running):** block_upload_ms 2.00 -> <=0.5; step_wall_ms 67.92 -> [53.5, 57.5]; gpu_exec_ms unchanged. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| feature OFF | BENCH | 3 rounds | <...> | <...>/<...> |
| feature ON | BENCH | 3 rounds | <...> | <...>/<...> |
**Gates:** omega <ran_count/passed_count>; proxima-model-interop <N/N> incl. no-metal build and page-size agreement test.
**Parity:** generated_text and 2651/"known" unchanged, both arms.
**Census:** nocopy_reuses, copying_uploads (KV==0), BLOCK_COPIED_BYTES (KV==0 steady), kv_cache_upload_bytes slope.
**Home-turf arm:** none — this card moves no GPU work, only orchestration.
**Principles engaged and what each changed:** §1 (extend, don't add a peer — KV arena through 5.1's registered-span primitive); G8 (the 34 GB trap closed at build time). **Abandoned:** VIII.6 (parallel declared_capacity slice + new QuantizedBlock field), VIII.7 (bucketing before residency).
**Re-prove:** the trap-build command + the welded BENCH cell in both arms, interleaved 3x.
```

##### report skeleton
```
CARD 5.2 — <STATUS>
ran: <each command above, EXIT>
N: N1..N6 per expect
numbers: block_upload_ms, step_wall_ms, gpu_exec_ms both arms; kv_cache_upload_bytes slope; RSS/device_allocated_bytes vs G8 clause 3a/3b [round-4 synth S1]
predict vs observed: block_upload_ms <=0.5, step_wall_ms [53.5,57.5] vs <observed>
files: generate.rs:621-654,1364-1389; proxima-tensor/src/align.rs; metal.rs:350-362; new build.rs  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

### 5.3.6 PHASE 6 — plan stability
*Worktree `proxima-wt-risc07` / `risc/6-plan-stable`, base `risc/5-kv-residency`.*

#### CARD 6.1 — Bucket the KV leaf extent; the tail mask spelled with the ops that exist `[S2 4.1, B3 P6.1, crit RS-1, RS-4, MS-1, MS-2, b, c, HC-1]`

tier: worker
depends_on: [5.2, 0.8]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07`          branch: `risc/6-plan-stable`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `kv-capacity-bucket`       default: off

##### opens
- `proxima-tensor/src/spec.rs:6216-6245` (the three KV leaves at `[Symbolic(1), kv_heads, pairs|head_dim]`)
- `spec.rs:823-845` `causal_mask` (two `Iota{Symbolic(0)}` + `Greater` + `scalar_constant(NEG_INFINITY)`)
- `spec.rs:2604-2615` (`Select(is_future, neg_infinity, score_new_scaled)` — the consumption pattern to mirror argument-for-argument)
- `spec.rs:2303-2319` (the doc naming the premise, corrected in this commit), `:2336-2865`, sole caller `:6282`
- `proxima-tensor/src/op.rs:60-78` — 17 bodies, no `Less`, no `GreaterEqual` (verified this session)
- `generate.rs:1304-1309` `build_position_inputs` (keeps the TRUE `cached_len`; `start_position` used only at `:813` for cos/sin, extent = symbol 0 — bucketing symbol 1 does not touch RoPE or `is_future`, R17)
- `generate.rs:1313-1318` (`Vec::with_capacity(… + 3 + …)`), `:1332-1334`, `:1393` (`symbols`), `:1364-1389`
- `metal.rs:1111-1118` `block_node_ids` (program order), `:984-990` `InputCountMismatch`
- `generate.rs:1838-1860` — `forward_node_values`'s OWN `named_blocks` assembly (a second site that pushes `ids`/`eps`/`rope_cos`/`rope_sin` and the 32 KV names from an `empty_cache` and constructs `LayerCache::new()`); the leaf is pushed here too, or `InputCountMismatch` (`metal.rs:984-990`) fires on this public entry the moment the leaf lands; a test calls `forward_node_values` with the feature on [round-4 fix F12]

##### the design — the mask, with the arithmetic the closed set actually permits [crit MS-1, RS-4]
There is **no `GreaterEqual`**, so the tail mask is, with every literal naming every field [round-4 synth S19]:
```
kv_valid_len : Op::Input { dtype: DType::Float32, shape: vec![], name: Some("kv_valid_len".into()) }      # the TRUE cached_len, ONE leaf for the whole program
cache_slot   : Op::Iota { dtype: DType::Float32, extent: Extent::Symbolic(1) }      # over the BUCKETED cache axis, op.rs:229 [round-4 fix F12]
is_valid     : Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Greater,
                 operands: vec![(kv_valid_len, broadcast("->t")), (cache_slot, projection("t->t"))],
                 name: Some("kv_is_valid".into()) }   # 1.0 where slot < cached_len
score_masked : Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Select,
                 operands: vec![(is_valid, ..), (score_cached_scaled, ..), (neg_infinity, ..)],
                 name: None }
```
`kv_valid_len` and `cache_slot` depend only on the bucketed axis and the shared scalar, so they are built once for the program, not per layer; the per-layer cost is one `Select`, expected to fuse into `score_cached_scaled`'s existing `ComposedBody`. **Zero new `Op`, `BoundOpKind`, `ScalarOp` or `IndexMap` variants** — mechanically proved by 0.9's four exhaustive matches, which now include `ScalarOp` [crit SD-3].

**The scalar leaf is wired, not assumed** [crit MS-1, MS-2]: an `Op::Input` is a block node, and both drivers hard-fail on count mismatch. In this same commit `build_position_inputs` gains a fifth output field owning the backing storage; `named_blocks.push(("kv_valid_len", …))` lands beside `"eps"` at BOTH assembly sites — `generate.rs:1332` and `forward_node_values` at `:1858` [round-4 synth S19]. The `+3` `Vec::with_capacity` hint at `generate.rs:1316` is **LEFT ALONE** — it already under-counts (the code pushes FOUR fixed blocks: `ids` `:1322`, `eps` `:1332`, `rope_cos` `:1333`, `rope_sin` `:1334`), it has no observable, and no card asserts it either way (supersedes F12's `+5`) [round-4 fix F12] [round-4 synth S19]. The observable is `named_blocks.len() == block_node_ids(program).len()`, asserted on the real program at BOTH assembly sites [round-4 synth S19] — because the leaf is layer-invariant (there are no 32 per-layer index leaves; there are no index leaves at all in this design [conflict 3]).

**`found == expected` holds at every stage** [conflict 2, crit RS-1, d]: after bucketing, the leaf's symbol-1 extent **is** `bucket`, and `named_blocks` hands `&arena[..bucket*row]` — a `bucket`-sized slice of 5.2's `kv_capacity_tokens`-sized arena, with `bucket <= kv_capacity_tokens` asserted. So `element_count == block_element_count`, **neither validator is edited, and the arena exists one card earlier so there is no code-level cycle.** Rows in `[cached_len, bucket)` were never written and are masked to `-inf` by the `Select`, so they contribute `exp(-inf) = 0` exactly — **provided the arena is zero-filled**, which is not assumed: 5.2's N7 asserts every byte of the arena beyond `cached_len × row` reads as zero after allocation and after every append (a NaN or Inf in an unwritten V row would survive `0 × x` as NaN), and `AlignedBuffer`'s allocation path is read to confirm it zero-fills (`align.rs:69` region; if it does not, 5.2 adds the explicit fill and states the cost) [round-3 j9].

**The rollback fork is decided here** [crit RB-3]: `KV_BUCKET_TOKENS` is a build-time const from `[kv] bucket_tokens` beside 5.2's `capacity_tokens`, so feature-off rollback requires a rebuild — stated on the card, not left to the row. 256 ports the incumbent's own `n_kv` padding (R8/M6′); 6.3 sweeps it.

**The baselines this card invalidates, re-captured here** [crit HC-1]: adding a leaf renumbers `NodeId`s, so 3.1's golden emitted source and 2.1's fingerprint vectors are **re-captured for the feature-ON tree in this same commit** and both hashes recorded; the feature-OFF hashes must equal Phase 3's recorded values, which is what makes "off is main" a hash rather than a claim.

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs
```
2.
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 -b risc/6-plan-stable risc/5-kv-residency
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
```
3. `edit: proxima-tensor/Cargo.toml — declare kv-capacity-bucket feature, forwarded from omega and proxima-model-interop (verified passthrough pattern)`
4. `edit: proxima-tensor/src/spec.rs:6216-6245 — the tail-mask construction (kv_valid_len, cache_slot, is_valid, score_masked) as designed above, under the feature`
5. `edit: generate.rs:1393 — let bucket = (cached_len + new_count).div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS; let symbols = [new_count as u64, bucket as u64];`
6. `edit: generate.rs:1304-1318 — build_position_inputs gains a fifth output field owning kv_valid_len's storage; named_blocks capacity +3 -> +4`
6a. `edit: generate.rs:1838-1860 — forward_node_values's OWN named_blocks assembly (a second site that pushes ids/eps/rope_cos/rope_sin and the 32 KV names from an empty_cache and constructs LayerCache::new()); the leaf is pushed here too, or InputCountMismatch (metal.rs:984-990) fires on this public entry the moment the leaf lands; a test calls forward_node_values with the feature on [round-4 fix F12]`
7.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo nextest run -p proxima-tensor --release \
  --features kv-capacity-bucket,std --run-ignored all -E 'test(cpu_mask_zero_ulp)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.1-cpu-mask-test.log; echo "EXIT=${PIPESTATUS[0]}"
```
8.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.1-oracle.log; echo "EXIT=${PIPESTATUS[0]}"
```
9.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.1-gate-omega.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/proxima-tensor-gate.sh \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.1-gate-tensor.log; echo "EXIT=${PIPESTATUS[0]}"
```
10.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.1-bench.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1: the CPU mask test: for `bucket ∈ {8, 32, 256}` × `cached_len` spanning a boundary in each = **9 cases**, attention output **bit-identical** to unbucketed (0-ULP, not approximate).
- N2: `2651`/`"known"` and `generated_text` unchanged.
- N3: `named_blocks.len() == block_node_ids(program).len()`; `InputCountMismatch` never fires.
- N4: `op_count` rises by **+1, +2, +33 or +34** (supersedes F12's `{1,2,34}`), and the row names which and why — `Op::Input` is never a bound op (`BoundOpKind` has no `Input`, `proxima-tensor/src/bind.rs:221-264`): `+1` both `Greater` and `Select` fuse into the existing `ComposedBody` — the BEST outcome, not a kill; `+2` `Iota` + an unfused `Greater`; `+33` `Iota` + 32 unfused `Select`s; `+34` neither fused [round-4 fix F12] [round-4 synth S19].
- N5: the route census shows **no new route**.
- N6: feature-OFF golden and fingerprint hashes equal Phase 3's.
- N==0 is RED.

##### predict (nano → micro)
`op_count == OPS_AFTER_PRUNE + 1`, i.e. both elementwise nodes fuse [round-4 synth S19]. `+2` or `+33` localises which fusion failed; `+34` refutes both and is a bind-level composition question, not a graph question, and that is the finding.

##### kill
- Any ULP difference on the CPU mask test, or token drift at any bucket size ⇒ the mask is wrong; revert immediately (§14).
- `op_count` delta ∉ {1, 2, 33, 34} ⇒ the graph changed in a way this card did not design [round-4 fix F12] [round-4 synth S19].

##### memory gate
- gate: MG-3, RSS ceiling 540 MB (5.2)
- what this card allocates: `kv_cache_upload_bytes` becomes constant at `bucket × 262,144 / capacity`-shaped rather than linear — flat is the point, and the row headlines the trade. Device cap uses `kv_capacity_tokens`, never `bucket`. **KILL** on breach.

##### rollback
Feature off **plus a rebuild** (the bucket is a build-time const); the two-arm `#[cfg]` pairing in 6.2 keeps both worlds green.

##### blast
`spec.rs` cached-layer builder and the three KV leaves under the feature only; `proxima-tensor-runtime.toml` +1 key; `generate.rs` (`build_position_inputs`, BOTH `named_blocks` assemblies — `:1332` and `forward_node_values`'s `:1858` — `symbols`) [round-4 synth S19]; `generate.rs:1838-1860` (`forward_node_values`'s own `named_blocks` assembly, the second site the leaf must land in) [round-4 fix F12]. `append_qwen35_*` builders are **not** touched and a test asserts it.

##### observe
`op_count` and its delta [round-4 synth S19], the 9 parity cases, the oracle, the two hash sets, `kv_cache_upload_bytes`.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p proxima-tensor --all-features -E 'test(kv_tail_mask)'` [round-4 synth S19] + the ORACLE cell + the BENCH cell, all welded.

##### row
```
## ROW <NEXT> -- the KV extent buckets to a capacity and the tail masks with Greater + Select, because there is no Less and no GreaterEqual
**Card:** 6.1. **Worktree/branch/commit:** proxima-wt-risc07/risc/6-plan-stable/<sha at report time>. **Feature:** kv-capacity-bucket, default-off.
**Allocation budget (hot/setup/cold):** RSS ceiling 540 MB (5.2's arithmetic); kv_cache_upload_bytes flat at bucket-shaped, not linear (MG-3).
**Predict (one rung ahead, written before running):** op_count == OPS_AFTER_PRUNE + 1, both elementwise nodes fuse (a +2 or +33 result localises which fusion failed; +34 refutes both) [round-4 synth S19]. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| feature ON | CPU mask test | 9 cases | n/a (0-ULP) | <...>/<...> |
| feature ON | ORACLE | 1 | n/a | <...>/<...> |
| feature ON | BENCH | <...> | <...> | <...>/<...> |
**Gates:** omega <ran_count/passed_count>; proxima-tensor <N/N> incl. 9-case mask test.
**Parity:** 9-case bit-identical (0-ULP) vs unbucketed; 2651/"known" oracle unchanged.
**Census:** route census shows no new route; op_count delta in {1, 2, 33, 34}, the row names which and why [round-4 synth S19].
**Home-turf arm:** none — this card produces graph records, not a comparison cell.
**Principles engaged and what each changed:** conflict 2 (residency before bucketing — found==expected holds at every stage); crit RS-4/SD-3 (zero new Op/BoundOpKind/ScalarOp/IndexMap variants, mechanically proved). **Abandoned:** VIII.7 (bucketing before residency, the 4.1<->5.3 cycle).
**Re-prove:** the mask test + the ORACLE cell + the BENCH cell, all welded.
```

##### report skeleton
```
CARD 6.1 — <STATUS>
ran: <each command above, EXIT>
N: N1..N6 per expect
numbers: op_count delta (in {1,2,33,34}); 9-case ULP diffs (all 0); kv_cache_upload_bytes shape; feature-OFF hash equality [round-4 synth S19]
predict vs observed: op_count == OPS_AFTER_PRUNE+1 vs <observed> [round-4 synth S19]
files: proxima-tensor/src/spec.rs:6216-6245; generate.rs:1304-1318,1393; proxima-tensor-runtime.toml  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

#### CARD 6.2 — Invert the `plan_hits == 0` assertion, `#[cfg]`-paired, as a formula `[S2 4.1 assertion half, B3 P6.2, crit k]`

tier: worker
depends_on: [6.1]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07`          branch: `risc/6-plan-stable`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `kv-capacity-bucket` (the `#[cfg]`-paired assertion pair)       default: off

##### opens
- `proxima-model-interop/src/bind.rs:2994-2998` (the doc stating the finding), `proxima-model-interop/src/bind.rs:3051`, `proxima-model-interop/src/bind.rs:3053-3055`, `proxima-model-interop/src/bind.rs:3057-3059` [round-4 synth S35]
- `generate.rs:966` (the key), `:973` `self.plans.clear()` — the cache holds exactly one entry, so a hit requires the **immediately preceding** key

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs
```
2. `edit: proxima-model-interop/src/bind.rs — BOTH assertions become #[cfg(not(feature="kv-capacity-bucket"))]: the plan_hits == 0 assertion at :3052-3055 AND the plan_misses == forward_calls_taken assertion at :3056-3059; inverting only the first leaves the second red, so both are edited in the same commit [round-4 synth S20]`
3. `edit: bind.rs — a new PAIR is added under #[cfg(feature="kv-capacity-bucket")], computing expected_misses = 1 + #{consecutive key changes} over forward_calls_taken from generated.0.len() + usize::from(generated.2), then assert_eq!(plan_misses, expected_misses), assert_eq!(plan_hits, F - expected_misses), assert!(plan_hits > 0, "a bucketed extent that never hits is the null result, not a pass"). No literal token count anywhere [round-2 j6].`
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.2-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.2-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"
```
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && grep -oE 'symbols=\([0-9]+,[0-9]+\)' /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.2-bench-on.log \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.2-symbols-dump.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- Feature **off** → the original assertions hold unchanged (`default` behaviour untouched).
- Feature **on** → `plan_hits > 0` (`plan_hits == 0` is RED) and both counts equal the formula exactly, with the key sequence taken from 0.8's `symbols` dump, never assumed.
- N==0 is RED.

##### predict (milli → bench)
`prepare_ms` **1.97 → ≤0.2** on hit steps. **`op_setup_ms` is unchanged at 3.90** — a plan hit alone does not remove per-op buffer and uniform allocation (R11 M6′′); that is 6.5's target. The honest `step_wall` reduction from this card alone is **~1.7 ms**, and reporting the smaller number here is the point.

##### kill
- Feature-off behaviour changes at all.
- `plan_hits` stays 0 with the feature on ⇒ the key is still moving; dump `symbols` and name the other varying symbol.
- `plan_hits` rises while `prepare_ms` does not fall beyond CoV ⇒ the cost was never in `plan_named`; re-instrument before 6.5.

##### memory gate
- gate: MG-3, plus `plan_cache_len <= 1` every step
- what this card allocates: a bucketed key that fills the map is the `ff749a0` leak re-opened — KILL if `plan_cache_len` rises above 1.

##### rollback
Feature off.

##### blast
One test module, `#[cfg]`-paired so both worlds are asserted.

##### observe
`plan_hits`, `plan_misses`, `plan_cache_len`, `prepare_ms`, `op_setup_ms`, the `symbols` dump, `UNIFORM_CACHE_LEN` (a bucketed key makes uniforms token-invariant, so the cache should stop growing — an observable here even though 6.5 owns the lever) [round-4 synth S20].

##### reprove
The welded BENCH cell in both arms.

##### row
```
## ROW <NEXT> -- the harness asserted plan_hits==0; now it asserts the formula, both ways, and the formula is consecutive-key because the cache clears on miss
**Card:** 6.2. **Worktree/branch/commit:** proxima-wt-risc07/risc/6-plan-stable/<sha at report time>. **Feature:** kv-capacity-bucket, #[cfg]-paired assertion.
**Allocation budget (hot/setup/cold):** plan_cache_len <= 1 every step, both arms (MG-3).
**Predict (one rung ahead, written before running):** prepare_ms 1.97 -> <=0.2 on hit steps; op_setup_ms unchanged at 3.90. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| feature OFF | BENCH | <...> | <...> | <...>/<...> |
| feature ON | BENCH | <...> | <...> | <...>/<...> |
**Gates:** proxima-model-interop <N/N> incl. both #[cfg]-paired assertion tests.
**Parity:** n/a — assertion-inversion card, not a correctness change.
**Census:** plan_hits, plan_misses, plan_cache_len against the formula; symbols dump (0.8); UNIFORM_CACHE_LEN [round-4 synth S20].
**Home-turf arm:** none — this card measures orchestration only.
**Principles engaged and what each changed:** G5 (the N contract as a formula, never a literal token count); round-2 j6 (no card assumes the prompt length). **Abandoned:** none.
**Re-prove:** the welded BENCH cell in both arms.
```

##### report skeleton
```
CARD 6.2 — <STATUS>
ran: <each command above, EXIT>
N: feature-off assertion pair unchanged; feature-on formula assertions pass; symbols dump non-empty
numbers: prepare_ms, op_setup_ms both arms; plan_hits/plan_misses/plan_cache_len both arms
predict vs observed: prepare_ms <=0.2 on hit steps vs <observed>
files: proxima-model-interop/src/bind.rs:3051-3059 (test module only)  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

#### CARD 6.3 — The bucketing trade cell: orchestration saved against padding added `[B3 P6.3, conflict 2]`

tier: hands
depends_on: [6.2]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07`          branch: `risc/6-plan-stable`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `kv-capacity-bucket`, swept via `OMEGA_KV_BUCKET_TOKENS`       default: off

##### opens
- R13's per-family table: `kv_cache.v` 1.559, `k_odd` 0.783, `k_even` 0.774 ms, **Σ 3.116 ms** at the R13 context

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs
```
2. Per bucket value `∈ {8, 32, 64, 256}`, each value prebuilt into its own target dir (`/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target/<bucket>`) before any measurement, then interleaved with the feature-off control, 3 runs each, both rungs (MILLI, BENCH) [round-4 synth S21]:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_KV_BUCKET_TOKENS=<value> PROXIMA_METAL_OP_PROFILE_STEP=3 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.3-milli-<value>-run<N>.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 OMEGA_KV_BUCKET_TOKENS=<value> \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.3-bench-<value>-run<N>.log; echo "EXIT=${PIPESTATUS[0]}"
```
3. Feature-off control, same rungs, same interleave slot:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.3-milli-control-run<N>.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.3-bench-control-run<N>.log; echo "EXIT=${PIPESTATUS[0]}"
```
Both MILLI cells above are **5-token cells**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` and the env profile step above are inert on that rung — milli budget 5, bench budget 8, the two rungs are never read as the same cell [round-4 judge J4].

##### expect
- 5 configurations (4 bucket values + control) × 3 runs × 2 rungs = **30 cells; N==0 is RED**.
- Each cell carries the three `kv_cache.*` family times, `prepare_ms`, `op_setup_ms`, `plan_hits`, `step_wall_ms`, `gpu_exec_ms`, `UNIFORM_CACHE_LEN` [round-4 synth S21].

##### predict (milli → bench)
The cached-range reduce now runs over `bucket` slots instead of `cached_len`, so the three `kv_cache.*` families inflate by `bucket / mean(cached_len)`. At the R13 context (mean cached_len ≈ 34.5, observed not assumed) bucket 256 gives ×7.4 ⇒ a net LOSS of **+18.6 ms batched-equivalent** (DERIVED) [round-4 synth S21]; bucket 32/64 gives ×1.1–1.9 ⇒ **δ_b ∈ [0.3, 2.8] ms** against 1.7 ms saved. δ_b is this card's MEASURED output, carried to §IV's board at both endpoints [0.3, 2.8], never capped [round-4 synth S21]. **The pre-registered claim is that the optimum sits near `bucket ≈ prompt_tokens + PROXIMA_MAX_TOKENS`** — exactly why the incumbent, running at real context lengths, pads to 256 and we cannot at this budget.

##### kill (the decisive one)
If `Δ(kv_cache.* gpu_ms) > Δ(prepare + op_setup ms)` at **every** bucket value, **bucketing is dead as a landing route**, the fork resolves to 6.4, and the loss is recorded without softening.

##### memory gate
- gate: MG-3 at each bucket
- what this card allocates: the device cap uses `kv_capacity_tokens`, not `bucket` — no additional allocation beyond 5.2's arena.

##### rollback
Feature off.

##### blast
None beyond 6.1/6.2.

##### observe
The three `kv_cache.*` families, `prepare_ms`, `op_setup_ms`, `plan_hits`, wall, gpu, `UNIFORM_CACHE_LEN`, per-bucket [round-4 synth S21].

##### reprove
The sweep (all commands above).

##### row
```
## ROW <NEXT> -- bucketing the KV extent: the orchestration saving against the padding cost, by bucket size, with the optimum measured
**Card:** 6.3. **Worktree/branch/commit:** proxima-wt-risc07/risc/6-plan-stable/<sha at report time>. **Feature:** kv-capacity-bucket, swept via OMEGA_KV_BUCKET_TOKENS in {8,32,64,256}.
**Allocation budget (hot/setup/cold):** device cap uses kv_capacity_tokens, not bucket (MG-3, no new allocation beyond 5.2's arena).
**Predict (one rung ahead, written before running):** bucket 256 net LOSS +18.6 ms batched-equivalent (DERIVED); bucket 32/64 delta_b in [0.3, 2.8] ms vs 1.7 ms saved, carried to the board at both endpoints, never capped; optimum near prompt_tokens + PROXIMA_MAX_TOKENS. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S21]
| arm | cell | n | CoV | load before/after |
| control (feature off) | MILLI+BENCH | 3 runs | <...> | <...>/<...> |
| bucket=8 | MILLI+BENCH | 3 runs | <...> | <...>/<...> |
| bucket=32 | MILLI+BENCH | 3 runs | <...> | <...>/<...> |
| bucket=64 | MILLI+BENCH | 3 runs | <...> | <...>/<...> |
| bucket=256 | MILLI+BENCH | 3 runs | <...> | <...>/<...> |
**Gates:** none run in this card — measurement only, gates already recorded by 6.1/6.2.
**Parity:** n/a — carried from 6.1/6.2, not re-run here.
**Census:** kv_cache.v/k_odd/k_even family ms per bucket; prepare_ms, op_setup_ms, plan_hits, UNIFORM_CACHE_LEN per bucket. [round-4 synth S21]
**Home-turf arm:** none — this card measures our own bucket sweep, not an incumbent comparison.
**Principles engaged and what each changed:** conflict 2 (residency before bucketing, measured trade). **Abandoned:** none new here; VIII.18 (BoundOp.extents symbolic up front) is UN-PARKED if this card's decisive kill fires.
**Re-prove:** the sweep.
```

##### report skeleton
```
CARD 6.3 — <STATUS>
ran: <each command above, EXIT>
N: 30 cells (5 configs x 3 runs x 2 rungs)
numbers: kv_cache.v/k_odd/k_even ms, prepare_ms, op_setup_ms, plan_hits, step_wall_ms, gpu_exec_ms per configuration
predict vs observed: net loss at 256 / net win at 32-64 vs <observed>; optimum bucket vs prediction
files: none edited — measurement card  diff --stat: n/a
row: see above
reprove: see above
open:
```

---

#### CARD 6.4 — *(contingent on 6.3's kill)* The shape-invariant plan `[B3 P6.4, S2 abandoned-9, R18]`

**Contingency, verbatim:** *tier: worker · depends_on: 6.3 **and only if 6.3 killed bucketing at every bucket size**.*

tier: worker
depends_on: [6.3] (only if 6.3's decisive kill fired at every bucket value)
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14`          branch: `risc/6-plan-stable-invariant`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: the same feature flag that gates 6.2's `#[cfg]` pair (`kv-capacity-bucket`) — source states "behind the same feature flag" without naming a distinct one; carried as-is       default: off

##### opens
- `omega/src/msl.rs:2207-2218` — the `Uniforms` struct already carries `output_total`, `reduction_total`, `output_extents[]`, `reduction_extents[]`, `operand_base[]`, `operand_strides[][]`, `out_base`, `out_strides[]` — every one already a per-dispatch uniform (R18)
- `metal.rs:2070` `upload_uniforms`, `:2069` `UNIFORM_BUFFER_REUSES` (a live reuse path at `:2075`)
- `msl.rs:1517-1560` `grid_threads`, `:731` `kernel_cache_key`, `:797` `kernel_dispatch_shape`
- `proxima-tensor/src/bind.rs:200-215` (`BoundOp.extents: Vec<u64>` — the one baked thing)

##### the design, stated now so the fork is decidable, built only if reached
Split `Plan` into a **shape-invariant** part (kernel source, pipelines, routes, retirement, packed operands, resident marking) and a **per-token** part (uniform bytes, grid dims, output buffer sizes). The evidence it is cheaper than it looks: `pipeline_lookup` is already **0.04 ms** (R13), so pipelines already survive `cached_len` changes and the MSL source is already extent-independent; what is rebuilt per token is bind + retirement + packed-operand resolution, none of which depends on the *value* of `cached_len`. KV-dependent intermediates are allocated at `kv_capacity_tokens` while **the grid is dispatched at the true extent** — so there is no padding cost at all and **no tail mask is needed**, which is the whole reason this fork can beat bucketing's best cell.

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/runs
```
2.
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14 -b risc/6-plan-stable-invariant risc/5-kv-residency
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/target
```
3. `edit: omega — split Plan into a shape-invariant part (kernel source, pipelines, routes, retirement, packed operands, resident marking) and a per-token part (uniform bytes, grid dims, output buffer sizes); KV-dependent intermediates allocate at kv_capacity_tokens, grid dispatches at the true extent`
4. The 6.2 command pair, on this branch, behind the same feature flag:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/runs/6.4-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-capacity-bucket \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc14/runs/6.4-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- `plan_hits == F − 1` (every step after prefill).
- `op_count` unchanged (the mask nodes 6.1 added are removed with the bucket).
- `generated_text` unchanged.
- `UNIFORM_CACHE_LEN` stops growing (the bytes become token-invariant) [round-4 synth S22].
- N==0 is RED.

##### predict (milli → bench)
`prepare_ms` → ≤0.2 with **zero** GPU inflation, i.e. strictly better than bucketing's best cell; `gpu_exec_ms` unchanged within CoV.

##### kill
- `gpu_exec_ms` rises at all — it must not, the grid is unchanged.
- If the split cannot be made without `BoundOp.extents` becoming symbolic (`Vec<Extent>`), stop: that touches `grid_threads`, `kernel_cache_key` and `kernel_dispatch_shape` (R11 M6′ calls it "a bigger change") and the card parks with the un-park condition "a real-context (≥1024-token) decode cell exists".

##### memory gate
- gate: MG-3
- what this card allocates: capacity-sized intermediates add `kv_capacity_tokens × kv_heads × head_dim × 4 B` per KV-dependent intermediate — the count and the total are printed before running, and the total is inside G8 clause 3b (STEADY_CAP) [round-4 synth S1] or the card does not run.

##### rollback
Feature off.

##### blast
`Plan` construction in omega — the widest omega-internal change in the plan, which is why it is contingent and paid for only if the cheap route is proven dead.

##### observe
`plan_hits`, `prepare_ms`, `gpu_exec_ms`, `op_count`, `UNIFORM_BUFFER_REUSES`.

##### reprove
The 6.2 command pair, on this branch.

##### row
```
## ROW <NEXT> -- the plan splits shape-invariant from per-token: the extents were already uniforms
**Card:** 6.4 (contingent — reached only if 6.3 killed bucketing at every bucket value). **Worktree/branch/commit:** proxima-wt-risc14/risc/6-plan-stable-invariant/<sha at report time>. **Feature:** kv-capacity-bucket (the same flag 6.2 gates), default-off.
**Allocation budget (hot/setup/cold):** kv_capacity_tokens x kv_heads x head_dim x 4 B per KV-dependent intermediate, printed before running, inside G8 clause 3b (STEADY_CAP) [round-4 synth S1] (MG-3).
**Predict (one rung ahead, written before running):** prepare_ms -> <=0.2 with zero GPU inflation; gpu_exec_ms unchanged within CoV. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| feature OFF | BENCH | <...> | <...> | <...>/<...> |
| feature ON (invariant plan) | BENCH | <...> | <...> | <...>/<...> |
**Gates:** the 6.2 command pair, this branch, <N/N>.
**Parity:** generated_text unchanged; op_count unchanged (mask nodes removed with the bucket).
**Census:** plan_hits == F-1 every step after prefill; UNIFORM_BUFFER_REUSES.
**Home-turf arm:** none — this card measures our own split against 6.1-6.3's bucketed cell, not the incumbent.
**Principles engaged and what each changed:** VIII.18 (BoundOp.extents symbolic up front — un-parked by 6.3's decisive kill). **Abandoned:** VIII.18 is the un-parked design this card builds; the un-park condition is 6.3's kill firing at every bucket size.
**Re-prove:** the 6.2 command pair, on this branch.
```

##### report skeleton
```
CARD 6.4 — <STATUS>
ran: <each command above, EXIT> (report "not reached" if 6.3 did not kill at every bucket size)
N: plan_hits==F-1, op_count unchanged, generated_text unchanged
numbers: prepare_ms, gpu_exec_ms both arms; UNIFORM_BUFFER_REUSES
predict vs observed: prepare_ms <=0.2 with zero GPU inflation vs <observed>
files: omega Plan construction (shape-invariant/per-token split)  diff --stat: <...>
row: see above
reprove: see above
open: <"card not reached — 6.3 did not kill bucketing at every value" if applicable>
```

---

#### CARD 6.5 — The device output/uniform arena: whole-buffer sharing only `[S2 4.2, crit RS-3, RB-3, h, HC-2, conflict 5]`

tier: worker
depends_on: [6.3] (or 6.4 if reached)
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07`          branch: `risc/6-plan-stable`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: `metal-plan-stable-buffers`       default: off

##### opens
- `omega/src/metal.rs:2179-2252` `encode_op` — `:2193-2194` `kernel_cache_key` + `kernel_dispatch_shape` per op per token even on a pipeline hit, `:2210` `allocate_buffer`, `:2211` `upload_uniforms`, `:2249` `device_buffers.insert(bound.node, (output, 0))`
- `metal.rs:2068-2078` `UNIFORM_BUFFER_REUSES` already exists with a live reuse path at `:2075` — **read its current hit rate BEFORE assuming uniforms are the cost** [Q11]
- `metal.rs:541-543` the retire loop `device_buffers.remove(retired)` [crit RB-3]
- `metal.rs:2355-2378` `finish`'s readback with the invariant spelled out at `:2360-2364`: "an output node's buffer is always freshly allocated by `encode_op` at offset 0 — only a weight INPUT can carry a nonzero offset … so reading from the buffer's own start is always correct here" (verified verbatim this session) [crit RS-3]
- `metal.rs:1128-1147` `bound_op_retirement`'s `if !outputs.contains(&node)` (verified) — the constraint that pins outputs, not merely a test oracle [crit h]
- `metal.rs:1132-1137` — the `last_use` walk inserts both `*source` and `lookup.indices`, the operand-set edge 9.1/9.2 change [round-4 synth S23]
- `generate.rs:1393-1400` (the KV roots are program **outputs** today — 97 effective outputs) [crit HC-2]
- 0.2's `UNIFORM_CACHE_LEN` reading — read BEFORE assuming the content cache needs a per-position replacement [round-4 synth S23]
- `omega/src/metal.rs:2074` (`UNIFORM_BUFFERS.get`) and `:2092` (`UNIFORM_BUFFERS.insert`) — on main the map has exactly these two operations and no recency field, so the bound is a data-structure change on the upload path [round-4 judge J2]

##### the two soundness constraints, stated as constraints and not as tests
1. **Whole-buffer sharing only.** The arena is a free list of whole `MetalBuffer`s by size class, reused across positions whose live ranges (from `bound_op_retirement`) do not overlap. **It never sub-allocates within one buffer**, because the moment it does, `device_buffers.insert(node, (buffer, 0))` at `:2249` is a lie and `finish` reads another op's data from the buffer's start, against the invariant at `:2360-2364` [crit RS-3, h].
2. **Outputs are pinned by construction.** `bound_op_retirement` excludes `effective_outputs`, so an output's buffer is never returned to the free list — that is what makes readback sound, and it is written on the card as the constraint the partition rests on.
3. **The retire loop is NOT made a no-op** [crit RB-3]. `device_buffers.remove(retired)` still fires, so operand lookups keep walking the **live** set instead of ~1196 entries; the arena keeps the `MetalBuffer` alive for reuse behind the map. Strict O(1) per op in steady state: index the free list by size class, no allocation, no hashing on the dispatch path.

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs
```
2.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && grep -oE 'UNIFORM_BUFFER_REUSES=[0-9]+' /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.2-bench-on.log /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.3-bench-*.log \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.5-pre-existing-reuse-rate.log; echo "EXIT=${PIPESTATUS[0]}"
```
3. `edit: COMMIT 1 of two — omega/src/metal.rs — the UNIFORM_BUFFERS bound: a capacity from a new [spans] uniform_cache_entries key via emit_sizing_consts, LRU eviction over the existing get (:2074) / insert (:2092) pair (no recency field exists today, so this is a data-structure change on the upload path); this is a LEAK REPAIR riding inside a performance card and gets its own N, test and rollback line — it is kept regardless of whether commit 2 lands [round-4 judge J2]`
3a. `edit: COMMIT 2 of two — omega/src/metal.rs — hang a BufferArena off the cached Plan, built once when the plan is built; encode_op binds rather than allocates. Uniforms are NOT written in place through UNIFORM_BUFFERS and NOT left to the content cache: UNIFORM_BUFFERS (metal.rs:2055-2078) is a content-keyed dedup cache, BTreeMap<Vec<u8>, MetalBuffer> keyed by the uniform bytes and shared across ops with identical bytes, so writing through it in place would corrupt every co-keyed op. This card instead adds a plan-owned Vec<MetalBuffer> of one uniform buffer per plan position (PlanUniforms), allocated once with the plan, written by encode_op at its own index, bypassing upload_uniforms entirely on the plan path [round-4 fix F13] [round-4 synth S23]. Recover 0.7's all-tracked.patch metal-buffer-pool hunk as REFERENCE ONLY and rewrite against the stable Plan. Feature metal-plan-stable-buffers, default-off, forwarded.`
3b. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo nextest run -p omega --release --features metal,cpu,instrument --run-ignored all -E 'test(uniform_cache_bounded)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.5-uniform-cache-bound.log; echo "EXIT=${PIPESTATUS[0]}"` — commit 1's own test: inserts `capacity + 1` distinct uniform blobs and asserts `UNIFORM_CACHE_LEN == capacity` and the evicted entry's buffer is released [round-4 judge J2]
4.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.5-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"
```
5.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,metal-plan-stable-buffers \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.5-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"
```
(repeat steps 4-5 interleaved ON/OFF, 3× total)
6.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --release \
  --features metal,cpu,instrument,metal-plan-stable-buffers --run-ignored all -E 'test(pooled_equals_unpooled) + test(no_rebind_before_last_consumer) + test(output_never_in_free_list) + test(extent_change_forces_realloc) + test(uniform_contents_change_buffer_stable) + test(live_buffer_count_bounded) + test(distinct_uniform_bytes_get_distinct_plan_buffers)' \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.5-arena-tests.log; echo "EXIT=${PIPESTATUS[0]}"
```

##### expect
- N1: `OUTPUT_BUFFER_ALLOCATIONS == op_count` on the first step and **0 on every steady step**; a nonzero steady value is RED and names the position that reallocated.
- N2: `PLAN_UNIFORM_WRITES == op_count` per steady step and `UNIFORM_BUFFER_REUSES` FALLS to ~0 on the plan path — reported against its pre-card rate from 0.2, and stated as the expected direction, not as a regression: the cache's own content-key makes a rise to `op_count` impossible once uniforms bypass it on the plan path (supersedes F13) [round-4 fix F13] [round-4 synth S23].
- N3: `plan_hits` still satisfies 6.2's formula (this card is a no-op without plan stability); `UNIFORM_CACHE_LEN` is bounded by the new `[spans] uniform_cache_entries` capacity and stops growing (clause 5 asserted, not observed) [round-4 synth S23].
- N4: `generated_text` unchanged.
- N5: ≥7 tests — pooled == unpooled on the real forward; a pooled buffer never rebound before its last consumer (asserted against `bound_op_retirement`'s own order on the real program); a program output's buffer never in the free list (the readback-soundness test); an extent change forces a documented realloc; uniform contents change per step while the buffer does not; the live buffer count is bounded in steady state; two ops with identical uniform bytes get DISTINCT plan-owned buffers (the anti-aliasing test the content cache would otherwise hide) [round-4 synth S23].
- N6: commit 1's own test — inserting `capacity + 1` distinct uniform blobs asserts `UNIFORM_CACHE_LEN == capacity` and the evicted entry's buffer is released [round-4 judge J2].
- N7: `upload_uniforms` ns/call unchanged within the 0.5 band for `op_setup_ms` (the bound is a data-structure change on the upload path, not a performance change) [round-4 judge J2].
- N==0 is RED.

##### predict (milli → bench)
`op_setup_ms` **3.90 → [0.4, 0.8]** (what remains is `kernel_cache_key` + `pipeline_for` + bind; `pipeline_lookup` is already 0.04) ⇒ `step_wall_ms` improves by **[3.1, 3.5] ms** from wherever 6.3 left it, landing **[49.2, 53.3] + δ_b** (predict band per synth4 §IV) [round-4 synth S23].

##### kill
- `OUTPUT_BUFFER_ALLOCATIONS` nonzero in steady state ⇒ the plan is not stable; the work item goes **back to 6.3/6.4**, not forward.
- `op_setup_ms` falls but wall does not, beyond both CoV bands ⇒ the orchestration slice overlaps GPU execution — the same shape as R12's dispatch-count null; record it and **stop Phase 6** rather than continuing.

##### memory gate — the highest-risk memory card in the plan
- gate: MG-3
- what this card allocates: the arena holds intermediates alive for the plan's lifetime. **The card's FIRST action, before allocating, is to compute and print `Σ over resolved ops of product(extents) × dtype_bytes`** and compare it against ARENA_TRANSIENT_CAP = 172_812_125 B (DERIVED); if the naive arena exceeds it, it **must** be liveness-partitioned (whole buffers only) and the row records the arena's peak bytes and reuse factor. **`ARENA_PEAK_BYTES` above the term rolls the card back regardless of the `op_setup` win. KILL.** [round-4 fix F11] [round-4 synth S1]

##### re-derivation hook [crit HC-2]
The KV roots are outputs today, so the partition is computed against a 97-output set. **9.2 shrinks that set to ~1**, which changes every liveness range — the re-derivation hook names the `lookup.indices` edge at `metal.rs:1132-1137` (`bound_op_retirement`'s `last_use` walk inserts both `*source` and `lookup.indices`) [round-4 synth S23] — 9.2 therefore re-runs this card's peak assertion and its soundness tests in its own commit, and that obligation is written on both cards.

##### rollback
Feature default-off (`get_or_allocate` is `allocate_buffer` with the feature off); rollback reverts **commit 2 only** (the arena and plan-owned uniforms) — commit 1 (the `UNIFORM_BUFFERS` bound) is **kept regardless** — it is a leak repair and §15 forbids reverting a repair to restore a number [round-4 synth S23] [round-4 judge J2].

##### blast
`omega/src/metal.rs` `encode_op` (two call sites), `Plan` (+2 fields: the arena and `PlanUniforms`), the arena and uniform vector in `plan_named`, the `UNIFORM_BUFFERS` capacity [round-4 synth S23]. No IR, no graph, no emitter, no other backend.

##### observe
`OP_SETUP_CALLS`/`_TICKS`, `UNIFORM_BUFFER_REUSES` before and after, `PLAN_UNIFORM_WRITES` (**NEW**, `metal.rs:2211` region) [round-4 synth S23], `UNIFORM_CACHE_LEN` (0.2) [round-4 synth S23], `upload_uniforms` ns/call (commit 1's own N7) [round-4 judge J2], `OUTPUT_BUFFER_ALLOCATIONS` (**NEW**, `metal.rs:2210`), `ARENA_PEAK_BYTES` (**NEW**, the arena constructor), `device_allocated_bytes`, `phys_footprint_bytes`.

##### reprove
The welded BENCH cell ON/OFF interleaved 3×; the row's claim is N1 plus the arena peak.

##### row
```
## ROW <NEXT> -- 1196 device buffers and 1196 uniform uploads per token become zero: whole buffers only, outputs pinned by retirement, and the retire loop still fires
**Card:** 6.5. **Worktree/branch/commit:** proxima-wt-risc07/risc/6-plan-stable/<sha at report time>. **Feature:** metal-plan-stable-buffers, default-off.
**Allocation budget (hot/setup/cold):** Sigma over resolved ops of product(extents) x dtype_bytes, printed before allocating, compared against ARENA_TRANSIENT_CAP = 172_812_125 B (DERIVED) (MG-3, highest-risk card). [round-4 fix F11] [round-4 synth S1]
**Predict (one rung ahead, written before running):** op_setup_ms 3.90 -> [0.4, 0.8]; step_wall_ms improves [3.1, 3.5] ms from 6.3's landing point, to [49.2, 53.3] + delta_b. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S23]
| arm | cell | n | CoV | load before/after |
| feature OFF | BENCH | 3 rounds | <...> | <...>/<...> |
| feature ON | BENCH | 3 rounds | <...> | <...>/<...> |
**Gates:** omega <N/N> incl. 7 arena soundness tests + commit 1's uniform_cache_bounded test. [round-4 judge J2]
**Parity:** pooled == unpooled on the real forward; generated_text unchanged.
**Census:** OUTPUT_BUFFER_ALLOCATIONS (0 steady), PLAN_UNIFORM_WRITES == op_count steady, UNIFORM_BUFFER_REUSES falls to ~0 on the plan path, UNIFORM_CACHE_LEN bounded (== capacity + evicted-buffer-released), upload_uniforms ns/call unchanged, ARENA_PEAK_BYTES vs ARENA_TRANSIENT_CAP = 172_812_125 B. [round-4 fix F11] [round-4 synth S1] [round-4 synth S23] [round-4 judge J2]
**Home-turf arm:** none — this card measures orchestration only.
**Principles engaged and what each changed:** crit RS-3 (readback invariant, whole-buffer sharing only); crit RB-3 (retire loop not made a no-op); crit HC-2 (re-derivation hook for 9.2). **Abandoned:** VIII.13 (threading op_setup/the encode loop — this card removes the work instead of distributing it).
**Re-prove:** the welded BENCH cell ON/OFF interleaved 3x.
```

##### report skeleton
```
CARD 6.5 — <STATUS>
ran: <each command above, EXIT>
N: N1..N7 per expect; 7 arena soundness tests + commit 1's uniform_cache_bounded test [round-4 synth S23] [round-4 judge J2]
numbers: op_setup_ms, step_wall_ms both arms; PLAN_UNIFORM_WRITES vs op_count; UNIFORM_BUFFER_REUSES pre/post; UNIFORM_CACHE_LEN bounded; ARENA_PEAK_BYTES vs ARENA_TRANSIENT_CAP = 172_812_125 B [round-4 fix F11] [round-4 synth S1] [round-4 synth S23]
predict vs observed: op_setup_ms [0.4,0.8] vs <observed>; step_wall_ms [49.2,53.3]+delta_b vs <observed> [round-4 synth S23]
files: omega/src/metal.rs:2179-2252,541-543  diff --stat: <...>
row: see above
reprove: see above
open:
```

---

#### CARD 6.6 — Orchestration re-seal `[S2 3.3 analogue]`

tier: hands
depends_on: [6.5]
worktree: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07`          branch: `risc/6-plan-stable`          base: `risc/5-kv-residency`
target_dir: `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`        lock: `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`
feature: all landed Phase 4-6 features enabled together (the Q4_K body winner, `metal-kv-resident`, `kv-capacity-bucket` at 6.3's measured optimum, `metal-plan-stable-buffers`)       default: n/a — re-seal card, no new default set here

##### opens
- (none new — this card re-seals with `scripts/gpu-cell.sh`, defined at 0.4, and both incumbent arms, present since Phase 0)

##### commands
1.
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07
TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs
```
2. Loadout, before:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg
```
3.
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target std,metal,instrument 5 \
  2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/runs/6.6-reseal.log; echo "EXIT=${PIPESTATUS[0]}"
```
4. Loadout, after:
```
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg
```

##### expect
- 5 runs × S rows; both incumbent arms present; every board cell filled or explicitly `FEATURE GAP` with its reason.
- `greedy_pick_ms` reported explicitly (the ~1.6 ms residual gets a counter); pre-registered ≥1.0 ms ⇒ §VIII.27's un-park condition fires and the residual is named; if near zero, the residual is elsewhere and the row says where [round-4 synth S24].
- N==0 is RED; a blank cell is RED.

##### predict (milli → bench)
`step_wall_ms` **[49.2, 53.3] + δ_b**, `gpu_exec_ms` **[44.9, 48.6] + δ_b**, ratio **2.83–3.20x** against 0.5's chosen incumbent arm, plus the fraction-of-ceiling against 0.6's `traffic_gbs` column, named [round-4 synth S24]. Every term is carried from the measured deltas of 4.2, 5.2, 6.3 and 6.5 — nothing is re-derived from theory.

##### kill
`step_wall_ms` does not fall below **58.1** (5.2's own band top) beyond both CoV bands ⇒ the GPU-side and orchestration wins are cancelling somewhere; name **which counter did not move** before Phase 7 is scheduled [round-4 synth S24].

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1], at the landed `kv_capacity_tokens`
- what this card allocates: a breach is a board-level NEGATIVE and demotes the offending feature.

##### rollback
Demote features one line each.

##### blast
Docs.

##### observe
All seven phase counters, `greedy_pick_ms`, the route census, `gpu_device_ms`, both memory slopes, CoV per arm, loadout. [round-4 synth S24]

##### reprove
The seal command (`scripts/gpu-cell.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target std,metal,instrument 5`).

##### row
```
## ROW <NEXT> -- the board after the body, the residency and the plan: what is left is the non-matmul bucket and the graph
**Card:** 6.6. **Worktree/branch/commit:** proxima-wt-risc07/risc/6-plan-stable/<sha at report time>. **Feature:** landed Q4_K body + metal-kv-resident + kv-capacity-bucket (6.3 optimum) + metal-plan-stable-buffers, all ON together.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1] at the landed kv_capacity_tokens; a breach demotes the offending feature.
**Predict (one rung ahead, written before running):** step_wall_ms [49.2, 53.3] + delta_b; gpu_exec_ms [44.9, 48.6] + delta_b; ratio 2.83-3.20x vs 0.5's chosen incumbent, plus fraction-of-ceiling against 0.6's traffic_gbs; greedy_pick_ms >= 1.0 ms pre-registered. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S24]
| arm | cell | n | CoV | load before/after |
| ours, all landed features | gpu-cell.sh seal | 5 runs | <...> | <...>/<...> |
| llama -fa 0 | gpu-cell.sh seal | 5 runs | <...> | <...>/<...> |
| llama -fa 1 | gpu-cell.sh seal | 5 runs | <...> | <...>/<...> |
**Gates:** none run directly — this card composes 4.1-6.5's already-gated features; no conditional (this card carries no FEATURE GAP for the incumbent arms, both present since Phase 0).
**Parity:** generated_text carried from constituent cards, not re-run here.
**Census:** all seven phase counters, greedy_pick_ms, route census, gpu_device_ms, both memory slopes. [round-4 synth S24]
**Home-turf arm:** llama.cpp-Metal -fa 0 and -fa 1, both present; frequency-weighted read stated on the row.
**Principles engaged and what each changed:** §IV band ladder (6.3, 6.5's bands composed here); G6 (arms interleaved, loadout before/after). **Abandoned:** none new — composes prior cards' abandonments.
**Re-prove:** the seal command.
```

##### report skeleton
```
CARD 6.6 — <STATUS>
ran: <each command above, EXIT>
N: 5 runs x S rows, both incumbent arms present, all board cells filled or FEATURE GAP
numbers: step_wall_ms, gpu_exec_ms, ratio vs incumbent, fraction of 0.6's traffic_gbs ceiling
predict vs observed: step_wall_ms [49.2,53.3]+delta_b, gpu_exec_ms [44.9,48.6]+delta_b vs <observed>; greedy_pick_ms vs 1.0 ms [round-4 synth S24]
files: none edited — re-seal card  diff --stat: n/a
row: see above
reprove: see above
open:
```

---

cards: 11 | commands: 31 | citations: 63
### 5.3.7 PHASE 7 — the sizing config and the non-matmul 16.6 ms
*Worktree `proxima-wt-risc08`, branch `risc/7-geometry`, base `risc/6-plan-stable`.*

#### CARD 7.1 — Every geometry constant into `omega-runtime.toml` `[S2 1.3, B3 P7.1 half]`

tier: worker
depends_on: [3.4]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08          branch: risc/7-geometry          base: risc/6-plan-stable
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — build-time consts via `omega-runtime.toml`, not a cargo feature   default: n/a

##### opens
- `omega/src/msl.rs:1017` — `const PACKED_ROWS_PER_GROUP: usize = 4`, a verified §12 policy-const violation — MOVES
- `omega/src/msl.rs:1029-1030` — `TILE_DIM = 8`, `#[cfg(feature = "metal-tiled-gemm")]`, doc: "fixed 8x8 by the MSL type itself" — STAYS BARE, a named §12 exception [round-4 synth S25]
- `omega/src/msl.rs:1046` — `TILED_GEMM_NSG = 4`, doc: "a value other than 4 would need a different kernel body" — STAYS BARE, a named §12 exception [round-4 synth S25]
- `omega/src/msl.rs:2516` — `let lanes_per_block = 8;`, a **local**, not overridable at all today — MOVES
- `omega/src/msl.rs:2527` — the packed-row-blocked loop step `SIMD_WIDTH/lanes_per_block` — MOVES
- `omega/src/msl.rs:3172` / `:3190` / `:3193` — bare `SIMD_WIDTH` literals pinning the cooperative width — MOVES
- `omega/build.rs:16` / `:35` / `:43` / `:59` / `:79` / `:85` / `:105` — the compliant surface: `require_nonzero`, `resolve_int` + `rerun-if-env-changed`, `emit_sizing_consts`
- `omega/src/sized.rs:45` — `SIMD_WIDTH` stays a source const, documented as a hardware-family fact, never a policy knob

##### the rule
Only POLICY consts move: `[packed_row_block] rows_per_group, lanes_per_block` and `[cooperative_reduce] max_threads, vec_width`, all emitted UNCONDITIONALLY [round-4 synth S25]. `SIMD_WIDTH` is the one geometry quantity that stays a **source** const in `sized.rs`, because it names a hardware-family fact, not a tunable policy. `TILE_DIM` (`msl.rs:1029-1030`, `#[cfg(feature = "metal-tiled-gemm")]`, doc: "fixed 8x8 by the MSL type itself") and `TILED_GEMM_NSG` (`msl.rs:1046`, doc: "a value other than 4 would need a different kernel body") stay bare as three named §12 exceptions with `SIMD_WIDTH`, their own docs quoted on the row (supersedes F14) [round-4 fix F14] [round-4 synth S25].

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 -b risc/7-geometry risc/6-plan-stable`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target`
3. edit: `omega-runtime.toml` — add ONLY `[packed_row_block] rows_per_group, lanes_per_block` and `[cooperative_reduce] max_threads, vec_width` (the last two consumed by 7.2), each key documented with both values, ALL emitted UNCONDITIONALLY. `TILE_DIM`/`TILED_GEMM_NSG` do NOT get toml keys — `TILE_DIM` (`msl.rs:1030`) is `#[cfg(feature = "metal-tiled-gemm")]` (`:1029`), doc: "fixed 8x8 by the MSL type itself"; `TILED_GEMM_NSG` (`:1046`), doc: "a value other than 4 would need a different kernel body"; both stay bare source consts, named §12 exceptions beside `SIMD_WIDTH` (supersedes F14: routing them through a feature-conditional toml key would still red the alloc-tier clippy arm on an unreferenced const when the feature is off) [round-4 fix F14] [round-4 synth S25]
4. edit: `omega/build.rs:16,35,43,59,79,85,105` — route each of the FOUR policy keys through `resolve_int` + a cross-axis validator + `emit_sizing_consts`: `lanes_per_block` must divide `SIMD_WIDTH`, `rows_per_group` nonzero (a `div_ceil` denominator at `:1552`) [round-4 synth S25]
5. edit: `omega/src/msl.rs:1017,2516,2527,3172,3190-3193` — replace the FOUR bare policy consts/local with the emitted sizing consts (NOT `:1030`/`:1046`, which stay bare exceptions); add each key's measurement record on the consuming const's doc in `sized.rs`, per that file's convention [round-4 synth S25]
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- grep -rnE '^ *(pub )?const [A-Z_]+ *: *(usize|u64|u32) *= *[0-9]' omega/src/ 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.1-const-grep.log; echo "EXIT=${PIPESTATUS[0]}"`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features metal,cpu,instrument -E 'test(sizing_config)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.1-sizing-tests.log; echo "EXIT=${PIPESTATUS[0]}"`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=99999999 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build -p omega --features metal 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.1-negative-value.log; echo "EXIT=${PIPESTATUS[0]}"`
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.1-gate.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
- N1: `grep -rnE '^ *(pub )?const [A-Z_]+ *: *(usize|u64|u32) *= *[0-9]' omega/src/ | grep -v sized.rs` returns **only** GGUF wire-format constants (`Q4K_BLOCK_BYTES=144` at `:294`, `Q4K_BLOCK_ELEMENTS=256` at `:299`, and the Q5K/Q6K/Q8_0/Q4_0/F16/BF16 siblings) PLUS the two named exceptions `TILE_DIM` (`:1029-1030`) and `TILED_GEMM_NSG` (`:1046`), the grep result listing them explicitly with their docs quoted [round-4 synth S25]; each gaining a one-line doc saying it is a wire-format fact or a named exception; any OTHER surviving policy const is RED
- N2: ≥8 tests — source == TOML per key, an env override per key, and the override **visibly changes the emitted MSL** against 3.1's golden (proving `rerun-if-env-changed` works and a cached build did not silently ignore it)
- N3: an invalid value fails at **build time** with the validator's message
- N4: the six-step gate green including [4/6]-arm-1's `--lib --no-default-features --features alloc` clippy [round-4 synth S25]
- N==0 is RED.

##### predict
(nano -> micro) behaviour-neutral at default values: 3.1's golden hashes unchanged for every route and `gpu_exec_ms` moves < 0.7% (inside R13's CoV).

##### kill
- moving a const changes emitted MSL at the **default** value ⇒ an off-by-one; the golden catches it and the card stops
- a value that cannot become a build-time const without becoming a runtime read stays a source const with a one-line why at the site, recorded as a **named exception** on the row, never silently

##### memory gate
- gate: MG-1 — compile-time only.
- what this card allocates: nothing at runtime; all four moves are compile-time consts, value-identical by construction [round-4 synth S25].

##### rollback
`git revert` — the `build.rs` + toml + const-site hunks together; values identical by construction.

##### blast
`omega/build.rs`, `omega-runtime.toml`, `omega/src/sized.rs`, `msl.rs` policy-const sites (four moved) plus the two named-exception `msl.rs` const sites (two, not three — `SIMD_WIDTH`'s own site is `sized.rs`, not `msl.rs`) [round-4 synth S25].

##### observe
the N1 grep (a source-level assertion runnable in the gate); `OUT_DIR/omega_sized.rs`; 3.1's golden hashes.

##### reprove
the grep + the two override commands (steps 6-7 above), welded.

##### row
```
## ROW <NEXT> -- every GPU geometry constant traces to omega-runtime.toml with its cross-axis validator; SIMD_WIDTH stays a hardware fact and the row says why

**Card:** 7.1. **Worktree/branch/commit:** proxima-wt-risc08/risc/7-geometry/<sha at report time>. **Feature:** none — build-time consts, default n/a.
**Allocation budget (hot/setup/cold):** none — compile-time only, MG-1.
**Predict (one rung ahead, written before running):** golden hashes unchanged for every route; `gpu_exec_ms` moves < 0.7% (inside R13's CoV). **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| default values | golden-hash + gpu_exec_ms | <n> | <CoV> | <before>/<after> |
**Gates:** omega <N run/N passed> (features: metal,cpu,instrument); the alloc-tier build is not run by this card.
**Parity:** n/a — no kernel body changed, value-identical by construction.
**Census:** the N1 grep count (target: only wire-format consts plus the two named exceptions TILE_DIM/TILED_GEMM_NSG, listed explicitly with their docs; 0 other policy consts) [round-4 synth S25]; 3.1's golden hashes per route per backend.
**Home-turf arm:** none — this card produces no timed decode cell.
**Principles engaged and what each changed:** §12 (no bare policy const outside `sized.rs`'s wire-format exception). **Abandoned:** none.
**Re-prove:** the grep + the two override commands, welded.
```

##### report skeleton
```
CARD 7.1 — <STATUS>
ran: <each command above, EXIT>
N: N1 (const grep) / N2 (>=8 sizing tests) / N3 (negative-value build)
numbers: <grep hit count table> <golden hash table per key>
predict vs observed: <one line; if miss: category + work item>
files: omega/src/msl.rs:1017,2516,2527,3172,3190-3193 (moved) and :1029-1030,1046 (named exceptions, unchanged); omega/build.rs:16,35,43,59,79,85,105; omega/src/sized.rs:45; omega-runtime.toml   diff --stat: <...> [round-4 synth S25]
row: <see above>
reprove: the grep + the two override commands, welded
open: <anything the card could not do, by name>
```

---

#### CARD 7.2 — The cooperative reduce stops being 32 lanes wide for every size `[S2 3.1, B3 P7.1]`

tier: worker
depends_on: [7.1, 6.6]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08          branch: risc/7-geometry          base: risc/6-plan-stable
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: metal-wide-reduce, forwarded from `proxima-tensor` through `omega`   default: off

##### opens
- `omega/src/msl.rs:1517-1560` `grid_threads` — the cooperative arm is `output_total * SIMD_WIDTH`, i.e. 32 threads per output element for **every** reduction length
- `omega/src/msl.rs:3188-3196` — `output_index = gid/SIMD_WIDTH`, `lane = gid % SIMD_WIDTH` — the pin
- `omega/src/msl.rs:824` `reduce_is_cooperative`
- `omega/src/sized.rs:45`
- incumbent `ggml-metal.m:3797-3804` — nth doubles from 32 to `min(ne00/4, maxTotalThreadsPerThreadgroup)` (R8)
- incumbent `ggml-metal.metal:1679-1721` — float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`; a 4096-wide row gets 1024 threads (R8)
- `omega/src/metal.rs:1402-1426` `pipeline_for` — the device's own `maxTotalThreadsPerThreadgroup`
- 0.7's `gpudisp-tracked.patch` — recovered as reference only

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs` (same worktree as 7.1; no new worktree block)
2. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-milli-pre.log; echo "EXIT=${PIPESTATUS[0]}"` — first, **re-measure the mechanism**: the current `Route::ReduceCooperative` time on 6.6's tree, before any body change [crit m]. This is a **5-token cell**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` is inert on this rung — milli budget 5, bench budget 8, the two rungs are never read as the same cell [round-4 judge J4]
3. edit: `omega/src/msl.rs:1517-1560,3164-3196` — a two-level tree: threads-per-output = `min(next_pow2(reduction_len / vec_width), COOPERATIVE_REDUCE_MAX_THREADS)`, floored at `SIMD_WIDTH`, clamped against the pipeline's own `maxTotalThreadsPerThreadgroup`, never against a constant; float4 loads when the extent is a multiple of 4; both consts from 7.1's `[cooperative_reduce]`; behind `#[cfg(feature="metal-wide-reduce")]`
4. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p proxima-tensor --features std,metal-wide-reduce -E "test(cpu_oracle_reduce_extents)" 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-cpu-oracle.log; echo "EXIT=${PIPESTATUS[0]}"` — CPU-oracle parity at reduction extents `{31, 32, 33, 127, 128, 4096}`, 6 cases
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features metal,cpu,metal-wide-reduce -E "test(metal_parity)" 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-metal-parity.log; echo "EXIT=${PIPESTATUS[0]}"`
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=32 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo build -p omega --features metal,metal-wide-reduce 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-golden-32.log; echo "EXIT=${PIPESTATUS[0]}"`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=1024 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo build -p omega --features metal,metal-wide-reduce 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-golden-1024.log; echo "EXIT=${PIPESTATUS[0]}"` — the override-changes-golden-MSL test, 32 vs 1024
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,metal-wide-reduce \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/runs/7.2-milli-on.log; echo "EXIT=${PIPESTATUS[0]}"` — ON/OFF interleaved MILLI pair (this ON, step 2 was OFF)
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
- N1: the `reduce-cooperative` op count must not move across arms — only the time; a moved count is RED (the route changed, not the geometry)
- N2: CPU-oracle parity across reduction extents `{31, 32, 33, 127, 128, 4096}` — 6 cases, proving the tree is correct at non-multiples of the width
- N3: `metal_parity` runs its full case set with the feature on, count recorded
- N4: the `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS` override visibly changes emitted MSL at 32 vs 1024 against 3.1's golden
- N==0 is RED.

##### predict
(milli -> bench) the cooperative bucket enters the band ladder as **9.113 ÷ 1.073 = 8.49 ms batched-equivalent** [round-4 synth S26]; the bucket falls **≥10%** from the figure this card just re-measured (step 2), with **−20%** as the target (R3/M4, **MEMORY**, flagged; floor −10%; band **[6.8, 7.6]** against the batched-equivalent 8.49) [round-4 synth S26] ⇒ `gpu_exec_ms` **[43.2, 47.8] + δ_b** and `step_wall_ms` **[47.5, 52.5] + δ_b** [round-4 synth S26].

##### kill
- `metal_parity` or `backend_parity` regresses ⇒ a wider tree changes float summation order and §14 binds on the oracle, not on speed
- the bucket does not fall ≥10% beyond the measured CoV ⇒ R7's −20% did not survive the rebase; record the negative and do not promote
- the device clamp makes the width identical to 32 for our shapes ⇒ the lever is dead; record it
- one measured loss on the real graph retires this lever; it does not get a second attempt [round-4 synth S26]

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1]:
  (1) phys_footprint slope over steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 2_097_152 B/step (the 262_144 term went to 0 after 5.2 and stays there)
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_654_467_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_505_367_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: 540 MB (5.2 onward: 400 + the 512×262,144 = 134,217,728 B arena).
- what this card allocates: threadgroup memory only, on-chip, does not touch `device_allocated_bytes`; the per-threadgroup bytes (`max_threads/32 × 4`) go on the row; any `device_allocated_bytes` increase is a NEGATIVE.

##### rollback
`[cooperative_reduce] max_threads = 32` — the old behaviour exactly, no source revert.

##### blast
the cooperative body + `grid_threads`'s cooperative arm only — tiled-GEMM and packed-row-block `return` before it (`msl.rs:3164-3187`) and are untouched.

##### observe
`Route::ReduceCooperative` count and tick share, `op_profile_bucket … gpu_ms` and `gpu_ns_per_op` (R13: 9.113 ms / 23,670 ns), `gpu_exec_ticks`, the +7.3% inflation quoted.

##### reprove
the ON/OFF interleaved MILLI pair (steps 2 and 8), welded.

##### row
```
## ROW <NEXT> -- SIMD_WIDTH is a lane count, not a thread budget: the width comes from the sizing config and the clamp from the device

**Card:** 7.2. **Worktree/branch/commit:** proxima-wt-risc08/risc/7-geometry/<sha at report time>. **Feature:** metal-wide-reduce, default-off.
**Allocation budget (hot/setup/cold):** threadgroup memory only (`max_threads/32 × 4` bytes on-chip); device allocation must not increase.
**Predict (one rung ahead, written before running):** cooperative bucket 9.113 -> 8.49 ms batched-equivalent (÷1.073); falls >=10% from the re-measured baseline into band [6.8, 7.6], target -20% (MEMORY, floor -10%) => gpu_exec_ms [43.2, 47.8] + δ_b, step_wall_ms [47.5, 52.5] + δ_b. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S26]
| arm | cell | n | CoV | load before/after |
| OFF (re-measured baseline) | MILLI | <n> | <CoV> | <before>/<after> |
| ON (feature=metal-wide-reduce) | MILLI | <n> | <CoV> | <before>/<after> |
**Gates:** omega <N run/N passed> (features: metal,cpu,instrument,metal-wide-reduce); proxima-tensor <N/N> (cpu_oracle_reduce_extents, 6 cases).
**Parity:** metal_parity full case set, feature on, count recorded; CPU-oracle 6-extent set at {31,32,33,127,128,4096}.
**Census:** `Route::ReduceCooperative` op count (unchanged across arms), tick share, `gpu_ns_per_op` vs R13's 23,670 ns.
**Home-turf arm:** none — this card runs the MILLI rung only, not an incumbent-comparable cell.
**Principles engaged and what each changed:** §12 (geometry from `omega-runtime.toml`, clamp from the device, never a constant). **Abandoned:** none.
**Re-prove:** the ON/OFF interleaved MILLI pair, welded.
```

##### report skeleton
```
CARD 7.2 — <STATUS>
ran: <each command above, EXIT>
N: N1 (op count stable) / N2 (6 CPU-oracle extents) / N3 (metal_parity count) / N4 (golden 32 vs 1024)
numbers: <cooperative bucket ms OFF vs ON, per-extent parity table, golden hash pair>
predict vs observed: <one line; if miss: category + work item>
files: omega/src/msl.rs:1517-1560,3164-3196   diff --stat: <...>
row: <see above>
reprove: the ON/OFF interleaved MILLI pair, welded
open: <anything the card could not do, by name>
```

---

### 5.3.8 PHASE 8 — one emitter core, scoped honestly
*Worktree `proxima-wt-risc09`, branch `risc/8-emitter-core`, base `risc/3-route-value`. Scope-down stated up front: the full three-backend reorganisation of ~78 near-duplicates has **no measured payoff** and the widest blast radius in the plan. This phase lands the classification, closes CUDA's two coverage holes, and proves the core's shape on **one** kind with byte-identical emission; continuing to the remaining kinds is gated on that proof and on a pre-registered line-count delta. S2's "≤5 unclassifiable functions" threshold had no evidence behind it [round-2 j4] and is not used. **Phase 8 is SCHEDULED AFTER 11.2** — 11.2's `depends_on` drops 8.3, so the 12,553-line reorganisation lands after the final board, entangled with no perf number [round-4 synth S27].*

#### CARD 8.1 — The `Dialect` classification: which of the ~26 functions are TEXT and which are STRUCTURE `[S2 7.1, B3 II, crit MS-5, g, round-2 j4]`

tier: judge
depends_on: [3.4]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09          branch: risc/8-emitter-core          base: risc/3-route-value
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — docs only, no source change   default: n/a

##### the two binary questions for `trait Dialect` (answered here, before 8.3 types a line) [round-3 j9]
*Can an existing primitive express it?* Written out: the three emitters are three copies of one function body differing only in seven text fragments (`scalar_op_expr`, `preamble`, `kernel_signature`, `entry_name`, `simd_reduce_intrinsic`, `threadgroup_barrier`, `atomic_fold`). A generic `fn render<D: Dialect>(op: &BoundOp, route: Route) -> String` over a unit struct per backend expresses it with the existing `BoundOp` and `Route`; a pipe does not, because emission is a pure function of one op, not a stream. *What can a caller do that it could not before?* Before: `msl::emit(op)`, `wgsl::emit(op)`, `cuda::emit(op)` — three entry points whose structural halves can drift (R5: ~78 near-duplicates; CUDA rejects two kinds). After: `render::<Metal>(op, route)`; the caller can prove the three backends share one structure by hashing the core once — 8.3's byte-identity gate is the capability. The trait is consumed through a generic parameter only (§20: no `Box<dyn Dialect>`); the three dialects are a closed compile-time set.

##### opens
- R5's census re-verified: `msl.rs` 4712 + `wgsl.rs` 1929 + `cuda.rs` 1838 + `metal.rs` 2385 + `wgpu_driver.rs` 872 + `backend.rs` 615 + `sized.rs` 45 + `error.rs` 84 + `lib.rs` 73 = **12,553 lines**
- exactly **3 types + 2 fns** shared: `Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots` (`wgsl.rs:105`, `cuda.rs:66`)
- `omega/src/msl.rs:673-697`, `:4656`
- `omega/src/cuda.rs:146-183`
- `omega/src/wgpu_driver.rs` (872 lines) — named because `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a wgpu break lands in the gate for every later card
- `omega/src/msl.rs:1690`, `omega/src/wgsl.rs:538`, `omega/src/cuda.rs:464` — `fn entry_name(resolved: &BoundOp) -> String`, signature unchanged, no `Route` parameter [round-4 synth S27]

##### the design
The deliverable is the **seven text methods**, enumerated here and not elided [crit MS-5, g]: `trait Dialect`, consumed through a generic parameter (§20: no `Box<dyn>`; the three backends are a closed compile-time set and each dialect monomorphises into a hot string builder):
1. `scalar_op_expr(ScalarOp, &[&str]) -> String`
2. `preamble() -> &'static str`
3. `kernel_signature(&Bindings) -> String`
4. `entry_name(&BoundOp) -> String` — signature unchanged, no `Route` parameter (supersedes F15's "8.1 adds the parameter"); the `Route` must NOT enter the emitted name, because `kernel_cache_key` (`msl.rs:730-736`) starts from `entry_name`'s output and 3.1's N2 pinned it byte-stable [round-4 fix F15] [round-4 synth S27]
5. `simd_reduce_intrinsic(ScalarOp) -> &'static str`
6. `threadgroup_barrier() -> &'static str`
7. `atomic_fold(ScalarOp) -> Option<&'static str>` — `None` is a legitimate answer producing `Route::Declined(NoDialectIntrinsic)`.

Everything structural stays in **one generic core**: `validate`, `reduction_dims`, `bindings`, `push_body_steps`, `operand_read`, the four `render_*`, and — the classification crit-g said was missing — `grid_threads` (`msl.rs:1517-1560`) is arithmetic and `reduce_is_cooperative` (`:824`) is a predicate, so **both go to the core, not to a dialect.**

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 -b risc/8-emitter-core risc/3-route-value`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target`
3. write: `docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md` — one row per one of the ~26 functions × 3 backends, each classified TEXT (which of the seven methods) / STRUCTURE (core) / DELETE (duplicate) / BEHAVIOUR (neither), EVERY ROW CARRYING ITS MEASURED LINE COUNT, plus the commit order; **no source change**. For `Elementwise`: the STRUCTURE total on main is **423 lines** (msl 148 + wgsl 137 + cuda 138) and the TEXT total is **408 lines** — the numbers 8.3's continuation gate measures against, not invented before the measurement [round-4 synth S27].
4. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.1-row-count.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
N = 26 × 3 = **78 rows**, every one classified with no blanks AND every one carrying its measured line count [round-4 synth S27]; the seven signatures written out; the `(Backend, Route)` exhaustiveness plan stated; the **BEHAVIOUR** column counted and each member named; the STRUCTURE column's total for `Elementwise` — 423 lines (msl 148 + wgsl 137 + cuda 138) — is the number 8.3's continuation gate measures against [round-4 synth S27]. N==0 is RED; a blank classification is RED.

##### predict
none — this card produces a design record.

##### kill
the BEHAVIOUR count is nonzero ⇒ the one-core claim is **scoped to the classified subset in this row, with the members named**, before 8.3 starts. No numeric threshold is asserted, because none has evidence.

##### memory gate
- gate: MG-1 — docs only, no build, no process runs the model.
- what this card allocates: nothing.

##### rollback
docs revert.

##### blast
`docs/bench-campaigns/`; no source.

##### observe
the 78-row table with measured line counts; the per-method fan-in; the BEHAVIOUR list; the STRUCTURE total per kind [round-4 synth S27].

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md | wc -l`

##### row
```
## ROW <NEXT> -- 78 near-duplicates classified before one line moves: seven text methods, one structural core, and the functions that are neither

**Card:** 8.1. **Worktree/branch/commit:** proxima-wt-risc09/risc/8-emitter-core/<sha at report time>. **Feature:** none, default n/a.
**Allocation budget (hot/setup/cold):** none — MG-1, docs only.
**Predict (one rung ahead, written before running):** none — this card produces a design record. **Observed:** <...>. **Miss category + work item:** n/a.
| arm | cell | n | CoV | load before/after |
| n/a | n/a | n/a | n/a | n/a |
**Gates:** none — no build, no process.
**Parity:** n/a.
**Census:** 78 rows (26 functions × 3 backends), each carrying its measured line count; BEHAVIOUR-column count and named members; Elementwise STRUCTURE total 423 (msl 148 + wgsl 137 + cuda 138), TEXT total 408. [round-4 synth S27]
**Home-turf arm:** none — no timed cell.
**Principles engaged and what each changed:** §20 (no `Box<dyn>`, closed backend set, generic monomorphisation). **Abandoned:** none.
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md | wc -l`.
```

##### report skeleton
```
CARD 8.1 — <STATUS>
ran: <each command above, EXIT>
N: 78-row count / blank-classification count / BEHAVIOUR-column count
numbers: <78-row table summary>
predict vs observed: none — design record
files: docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md (new)   diff --stat: <...>
row: <see above>
reprove: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md | wc -l`
open: <anything the card could not do, by name>
```

---

#### CARD 8.2 — CUDA covers `Iota` and `Constant`, in two commits `[S2 7.2, crit B-4]`

tier: worker
depends_on: [8.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09          branch: risc/8-emitter-core          base: risc/3-route-value
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — CUDA is an existing backend feature, not a new one   default: n/a

##### opens
- `omega/src/cuda.rs:146-183` `emit_cuda` + `CudaUnsupportedOpKind`
- reference `omega/src/msl.rs:2044-2103` `render_iota`/`render_constant`
- `omega/src/error.rs`

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs` (same worktree as 8.1; no new worktree block)
2. edit (commit 1): `omega/src/cuda.rs:146-183` — implement `Iota` and `Constant` emission, referencing `msl.rs:2044-2103`'s `render_iota`/`render_constant`; **do not delete `CudaUnsupportedOpKind` in this commit** — land the implementation first, green, so rollback is one small revert rather than an unwind through downstream exhaustive matches
3. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --features cuda,cpu 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.2-cuda-impl.log; echo "EXIT=${PIPESTATUS[0]}"`
4. edit (commit 2): `omega/src/cuda.rs`, `omega/src/error.rs` — remove the now-unreachable `CudaUnsupportedOpKind` variant; update 3.3's coverage-matrix pre-registration in this same commit
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- grep -c CudaUnsupportedOpKind omega/src/cuda.rs 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.2-variant-grep.log; echo "EXIT=${PIPESTATUS[0]}"`
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.2-coverage-matrix.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
- ≥2 emit tests per kind (`Iota`, `Constant`)
- `cargo nextest run -p omega --features cuda,cpu` runs a **non-zero** count (`cuda` is not in `default` — the N==0 trap)
- `grep -c CudaUnsupportedOpKind omega/src/cuda.rs == 0` after the second commit
- 3.3's matrix moves from CUDA 2/4 to **4/4** and the test asserts the new pre-registration
- N==0 is RED.

##### predict
(nano -> micro) the 15-cell matrix (3 backends × 4 `BoundOpKind` + `Keep::Scan`) is **15/15** either `Ok` or an explicitly named `Route::Declined(reason)`; a silent absence is RED.

##### kill
CUDA cannot be compiled on this host (no toolchain) ⇒ emission is still testable **as text**, which is the point of a sans-IO emitter (§11); assert the emitted CUDA parses structurally and do **not** claim it runs.

##### memory gate
- gate: MG-1 — emission only, no device.
- what this card allocates: nothing.

##### rollback
revert the second commit (restores the variant), then the first.

##### blast
`omega/src/cuda.rs`, `omega/src/error.rs`.

##### observe
the coverage matrix; per-backend route census.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`

##### row
```
## ROW <NEXT> -- one RISC means every backend covers every kind: the 15-cell matrix and the deleted rejection

**Card:** 8.2. **Worktree/branch/commit:** proxima-wt-risc09/risc/8-emitter-core/<sha at report time>. **Feature:** none, default n/a.
**Allocation budget (hot/setup/cold):** none — MG-1, emission only, no device.
**Predict (one rung ahead, written before running):** the 15-cell matrix is 15/15 either Ok or a named Route::Declined(reason); a silent absence is RED. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| n/a — emission only, no device timing | n/a | n/a | n/a | n/a |
**Gates:** omega <N run/N passed> (features: cuda,cpu); omega <N run/N passed> (features: all, backend_route_coverage).
**Parity:** n/a — emission text only; CUDA is not run, only compiled/emitted as text.
**Census:** 15-cell coverage matrix (3 backends x 4 BoundOpKind + Keep::Scan); CUDA moves 2/4 -> 4/4.
**Home-turf arm:** none — no timed cell.
**Principles engaged and what each changed:** §4 (every backend covers every kind, or names the hole); §11 (sans-IO emission is testable as text without a toolchain). **Abandoned:** none.
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`.
```

##### report skeleton
```
CARD 8.2 — <STATUS>
ran: <each command above, EXIT>
N: emit-tests-per-kind (>=2 each) / CudaUnsupportedOpKind grep count (0) / 15-cell matrix (15/15)
numbers: <matrix table, CUDA 2/4 -> 4/4>
predict vs observed: <one line; if miss: category + work item>
files: omega/src/cuda.rs:146-183; omega/src/error.rs   diff --stat: <...>
row: <see above>
reprove: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`
open: <anything the card could not do, by name>
```

---

#### CARD 8.3 — The core, one kind (`Elementwise`), with the continuation gate `[S2 7.3, B3 —, round-2 j4]`

tier: worker
depends_on: [8.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09          branch: risc/8-emitter-core          base: risc/3-route-value
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — **not feature-gated**; a gate on a refactor means two emitters, 3.1's golden is the firewall   default: n/a

##### opens
- 8.1's rows for `render_elementwise`, `scalar_op_expr`, `operand_read`, `push_body_steps`, `bindings`, `preamble`, `kernel_signature`, `entry_name`
- `omega/src/msl.rs:2104-2177` and its wgsl/cuda twins
- `omega/src/msl.rs:4656` `emit_is_deterministic_byte_equal`
- `entry_name(resolved: &BoundOp) -> String` (`msl.rs:1690`, `wgsl.rs:538`, `cuda.rs:464`) takes no `Route` today and 8.1's method 4 keeps that signature unchanged (supersedes F15's "8.1 adds the parameter"); N1 requires that the route never enters the emitted name [round-4 fix F15] [round-4 synth S27]

##### the ruling
The continuation gate is decided in this row, not deferred: the remaining kinds (`Reduce` incl. every route, `Iota`/`Constant`, `Keep::Scan`) are scheduled **only if** N1 holds byte-for-byte **and** 8.1's `dialect-map.md` pre-registers, per kind, the line count of the STRUCTURE rows it classifies (for `Elementwise` on main: `bindings` + `push_body_steps` + `operand_read` + `render_elementwise` = 148 msl + 137 wgsl + 138 cuda = 423 lines, READ this session; TEXT rows stay 3× by design), and the continuation gate is: **(a) N1 byte-identity holds AND (b) the number of STRUCTURE rows from 8.1's table now deleted from all three backends equals 8.1's count for this kind, with the measured `wc -l` delta recorded as a fact beside it, not as the threshold itself** (supersedes F15's "within 10%") [round-4 fix F15] [round-4 synth S27]. A pure relocation into the core module without deleting duplicates falls short and is the finding. The earlier `≥ 600` was unreachable by ~180 lines — the maximal fall is 423 and it measures relocation, not duplication removed [round-4 fix F15] [round-4 synth S27]; otherwise the phase closes here with the classification and the coverage landed, and the row records the scope with its number.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs` (same worktree as 8.1/8.2; no new worktree block)
2. edit: `omega/src/msl.rs:2104-2177` and its wgsl/cuda twins — move the structural half of `Elementwise` emission to the new core module per 8.1's classification; implement the touched `Dialect` methods three times (MSL, WGSL, CUDA); delete the duplicates. One commit, green.
3. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.3-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
4. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 900 -- cargo build -p omega --no-default-features --features alloc 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.3-alloc-tier.log; echo "EXIT=${PIPESTATUS[0]}"` — states which modules it built
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run -p omega --all-features -E 'test(golden_source)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.3-golden-source.log; echo "EXIT=${PIPESTATUS[0]}"`
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && wc -l omega/src/msl.rs omega/src/wgsl.rs omega/src/cuda.rs 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/runs/8.3-line-counts.log`

##### expect
- N1: **byte-identical emitted source** for every `Elementwise` op in the real program, before and after, per backend — one byte of drift is RED
- N2: the gate's `ran_count` **≥** prior
- N3: the alloc-tier build (`--no-default-features --features alloc`) compiles the core and all three dialects and **states which modules it built** (§3's N==0 warning)
- N4: the deleted-STRUCTURE count equals 8.1's row count for this kind (423 lines: msl 148 + wgsl 137 + cuda 138), with the measured `wc -l` delta recorded beside it as a fact [round-4 synth S27]
- N==0 is RED.

##### predict
(nano -> micro) the measured `wc -l omega/src/{msl,wgsl,cuda}.rs` fall lands in **[300, 423]**, bounded above by 8.1's own measurement — the earlier `≥600` was unreachable by ~180 lines and measured relocation, not duplication removed (supersedes F15's "within 10%") [round-4 fix F15] [round-4 synth S27]; `gpu_exec_ms` unchanged within R13's 0.7% CoV; per-op `gpu_ns` unchanged within 1% (the more sensitive of the two tests).

##### kill
N1 fails ⇒ the refactor changed emission; bisect the dialect method that moved and either restore its text or record the change as its own row with its own parity evidence. **Never accept "the new text is equivalent" without the byte comparison.**

##### memory gate
- gate: MG-1 — compile-time; runtime allocation identical by N1's construction.
- what this card allocates: nothing — a text-emission refactor, no new runtime allocation.

##### rollback
`git revert` one commit. **Not feature-gated** — a gate on a refactor means two emitters; 3.1's golden is the firewall.

##### blast
the `Elementwise` paths in all three emitters + the new core module. Deliberately sequenced **after** every perf card so no number is entangled with a 12,553-line reorganisation.

##### observe
line counts, golden hash per route per backend, per-op `gpu_ns`.

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run -p omega --all-features -E 'test(golden_source)'` + the welded gate.

##### row
```
## ROW <NEXT> -- kind one of four through the core: the elementwise bytes did not move, here is the hash, and here is the delta that decides whether the rest follows

**Card:** 8.3. **Worktree/branch/commit:** proxima-wt-risc09/risc/8-emitter-core/<sha at report time>. **Feature:** none — not feature-gated, default n/a.
**Allocation budget (hot/setup/cold):** none — MG-1, compile-time only, runtime allocation identical by construction.
**Predict (one rung ahead, written before running):** `wc -l` fall lands in [300, 423], bounded above by 8.1's own measurement; gpu_exec_ms unchanged within 0.7% CoV; per-op gpu_ns unchanged within 1%. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S27]
| arm | cell | n | CoV | load before/after |
| before core move | golden-hash + gpu_ns | <n> | <CoV> | <before>/<after> |
| after core move | golden-hash + gpu_ns | <n> | <CoV> | <before>/<after> |
**Gates:** omega <N run/N passed> (all-features); alloc-tier build EXIT <..> (modules built: <...>).
**Parity:** byte-identical emitted source per Elementwise op per backend, before vs after (N1).
**Census:** deleted-STRUCTURE count vs 8.1's 423-line count for Elementwise (msl 148 + wgsl 137 + cuda 138); measured `wc -l` delta recorded as a fact beside it, not a threshold [round-4 synth S27].
**Home-turf arm:** none — this card's claim is emission-text identity, not a timed cell.
**Principles engaged and what each changed:** §11 (sans-IO emitter core, generic over a closed `Dialect` set); round-2 j4 (no unevidenced threshold — the STRUCTURE-row count is pre-registered per kind and measured, not assumed) [round-4 fix F15] [round-4 synth S27]. **Abandoned:** the full three-backend reorganisation as a mandatory phase [§VIII.19]; the "within 10%"/"≥600" continuation gates [round-4 fix F15] [round-4 synth S27].
**Re-prove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run -p omega --all-features -E 'test(golden_source)'` + the welded gate.
```

##### report skeleton
```
CARD 8.3 — <STATUS>
ran: <each command above, EXIT>
N: N1 (byte-identical emission) / N2 (gate ran_count >= prior) / N3 (alloc-tier build, modules built) / N4 (deleted-STRUCTURE count == 8.1's count) [round-4 synth S27]
numbers: <line-count delta table> <per-op gpu_ns before/after>
predict vs observed: <one line; if miss: category + work item>
files: omega/src/msl.rs:2104-2177 + wgsl/cuda twins; new core module   diff --stat: <...>
row: <see above>
reprove: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run -p omega --all-features -E 'test(golden_source)'` + the welded gate
open: continuation to the remaining kinds (Reduce incl. every route, Iota/Constant, Keep::Scan) — gated on this card's N1 and N4 (deleted-STRUCTURE count == 8.1's count, `wc -l` fall in [300,423]) [round-4 fix F15] [round-4 synth S27], not scheduled by this card
```

---

### 5.3.9 PHASE 9 — write placement and the op count
*Worktree `proxima-wt-risc10`, branch `risc/9-write-placement`, base `risc/7-geometry`. Scheduled last on purpose: R12's own control (1194 → 616 dispatches moved wall 51.571 → 51.535) says this lever moves the least. **It is here for the RISC, not for the milliseconds, and the plan says so.***

#### CARD 9.1 — A nonzero static offset on an Affine write map; `Computed`/scatter untouched `[B3 P8.1, crit a, OB-3, conflict 3]` [round-4 synth S28]

tier: worker
depends_on: [7.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10          branch: risc/9-write-placement          base: risc/7-geometry
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — the checker is unconditional, not feature-gated   default: n/a

##### opens
- `proxima-tensor/src/shape.rs:183-201`, read verbatim: `if reduce.out_map.is_data_dependent() { … return scatter_output_shape(here, &reduce.out_map, &iter_extents); }`, so `project_output_shape` is reached only at `:205` in the `Keep::Reduce` arm below that branch, and its own body opens `out_map.affine()` at `:474` — a `Computed` out_map **never reaches** `project_output_shape` [round-4 synth S28]
- `proxima-tensor/src/shape.rs:469-485` `project_output_shape`: `[term] if term.coeff == 1 => Ok(iter_extents[term.axis])`, which today **ignores `axis.offset` entirely** — a nonzero offset on an affine write map is silently accepted at inference and produces an out-of-range `Layout.base`, the latent hazard this card closes [round-4 synth S28]
- `proxima-tensor/src/shape.rs:441-467` `bounds_check`, whose first two lines are `let mut max_index = i64::from(axis.offset); let mut min_index = i64::from(axis.offset);` — the read-side fold this card mirrors on the write side
- `proxima-tensor/src/shape.rs:495` `scatter_output_shape` — explicitly OUT OF SCOPE, untouched
- `proxima-tensor/src/bind.rs:962` — for `IndexMap::Affine(pattern)` the bound op already takes `out_layout = layout_of(pattern, shapes.of(node))`, and `layout_of` (`:1594-1606`) ALREADY does `base += i64::from(axis.offset) * stride`, so the write offset already lands in `out_layout.base` [round-4 synth S28]
- `proxima-tensor/src/bind.rs:1011` `build_scatter_out_layout` — untouched; `:95-98` `Layout`
- `proxima-tensor/src/map.rs:105-131`, whose repurposing of `offset` to carry a destination extent is scoped to `Computed` at `gathered_dim` — an Affine write offset does not collide with it
- `proxima-tensor/src/map.rs:175-201` `IndexMap::scatter`, `:209` `scatter_extent`, `:238` `as_gather_from_output` — ALL untouched
- `proxima-autograd/src/adjoint.rs:835` `let out_map_as_operand = IndexMap::Affine(reduce.out_map.affine().clone());` — the affine adjoint reuses the write map as a **read** map, so the offset flows through `bounds_check`/`layout_of` symmetrically and the adjoint of a placed write is a sliced read BY CONSTRUCTION, with no autograd edit needed [round-4 synth S28]
- `proxima-autograd/src/adjoint.rs:806` `is_data_dependent()` → `ScatterOutputUnsupported`; `proxima-autograd/src/error.rs:73-88` — the guard the REJECT/data-dependent path must still fire exactly as today [round-4 judge J1]
- `omega/src/msl.rs:933`, `omega/src/wgsl.rs:364`, `omega/src/cuda.rs:241`, `omega/src/error.rs:53`
- `proxima-tensor/src/cpu.rs:6911` `run_reduce_scatter` — untouched, stays for the `Computed`/data-dependent path
- `scripts/proxima-autograd-gate.sh` [round-4 fix F16]

##### the rule
The mechanism is restricted to a static nonzero `axis.offset` on an AFFINE write map [round-4 synth S28]. `Computed`/scatter/`scatter_output_shape`/`build_scatter_out_layout`/`IndexMap::scatter`/`as_gather_from_output` are ALL untouched — a `Computed` out_map never reaches `project_output_shape` (`shape.rs:183-201` short-circuits data-dependent maps to `scatter_output_shape` at `:200`; `project_output_shape` is called only at `:205` and opens `out_map.affine()` at `:474`). `project_output_shape` accepts `[term] if term.coeff == 1` with a nonzero `axis.offset` and returns `iter_extents[axis] + offset` — the destination axis is `offset` wider than the iteration space, the write-side mirror of a read-side slice. `layout_of` (`proxima-tensor/src/bind.rs:1594-1606`, reached from `proxima-tensor/src/bind.rs:962` for Affine) ALREADY folds `offset * stride` into `out_layout.base`. Injectivity is then arithmetic, not a prover: two producers writing one destination with `coeff == 1` and static offsets `o1, o2` and iteration extents `e1, e2` are disjoint iff `[o1, o1+e1)` and `[o2, o2+e2)` do not overlap — an interval-overlap check at bind over the producers of one destination node, not a prover [round-4 synth S28]. No `Computed` out_map, no indices walk, no GPU scatter emitter, no atomics question, no `ScatterMayCollide` decline on the hot path.

##### the reconciliation with round 2 (conflict 3)
Round 2 ruled "`shape.rs` untouched is the proof", citing `map.rs:118-124`. Read verbatim, that passage rejects a **`Reduce`-wide destination-extent field** — where a scatter's static output extent lives — not a write offset on an Affine map. Accepting `coeff == 1` plus a nonzero `axis.offset` on an Affine map is a different change on a different code path, with three guardrails: the offset is a static `i32` in the `AxisIndex` and therefore loop-invariant by type; the inferred output extent adds it so bounds stay checkable, asserting `offset + iter_extent <= destination_extent`; and every existing `Reduce` with `offset == 0` must produce a byte-identical fingerprint and byte-identical emitted source (2.1's vectors + 3.1's golden) [round-4 synth S28].

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 -b risc/9-write-placement risc/7-geometry`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target`
3. edit: `proxima-tensor/src/shape.rs:469-485` — extend `project_output_shape`'s affine path to accept `[term] if term.coeff == 1` with a nonzero `axis.offset`, returning `iter_extents[axis] + offset`, asserting `offset + iter_extent <= destination_extent`; `scatter_output_shape` and the `Computed` destination-extent convention stay untouched [round-4 synth S28]
4. edit: `proxima-tensor/src/bind.rs` — add the disjointness check inside `bind`: for every destination node with multiple Affine-write producers at `coeff == 1`, compare `[o1, o1+e1)` against `[o2, o2+e2)` by interval overlap; REJECT on overlap with a named error; delete the backward-walk prover and the `Computed`-rewrite text entirely [round-4 synth S28]
5. edit: `proxima-autograd` (new test) — the autograd round-trip test: differentiate a placed `Reduce` and assert the adjoint reads the gradient at `base + offset` through `adjoint.rs:835`'s existing `out_map_as_operand` path with NO autograd edit (an edit needed is a finding that grows the card) [round-4 synth S28]; plus J1's REJECT-side test: a placed `Reduce` with a `Computed` (data-dependent) out_map is differentiated and the adjoint's existing guard fires (`adjoint.rs:806` `is_data_dependent()` → `ScatterOutputUnsupported`, `error.rs:73-88`) exactly as it does today, proving the affine extension changed nothing on the data-dependent path; and a case where two producers overlap and bind's interval check rejects BEFORE autograd sees the graph [round-4 judge J1]
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p proxima-tensor --all-features -E 'test(write_offset)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.1-write-offset.log; echo "EXIT=${PIPESTATUS[0]}"` [round-4 synth S28]
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/proxima-tensor-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.1-tensor-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
7a. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/proxima-autograd-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.1-autograd-gate.log; echo "EXIT=${PIPESTATUS[0]}"` — with `ran_count`/`passed_count` recorded; either 0 is RED [round-4 fix F16] (kept: F16's autograd gate command)
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.1-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.1-oracle.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
- a case table with **both** directions — **ACCEPT:** offset 0 (byte-identical to today); offset `k>0` with `coeff == 1`; two producers at disjoint `[o, o+e)` ranges; a producer whose range exactly abuts another's. **REJECT:** overlapping ranges (a named error, not silent last-writer-wins); `coeff != 1` (the v1 restriction narrows, it does not vanish); a negative offset; an offset whose `o + e` exceeds the declared destination extent [round-4 synth S28]
- **8 cases minimum; N==0 is RED, and a table with only ACCEPT cases is RED** — the fallback is the load-bearing half
- every existing `Reduce`'s fingerprint and golden byte-identical
- the six parity suites green
- **the autograd round-trip:** differentiating a placed `Reduce` reads the gradient at `base + offset` through `adjoint.rs:835`'s existing `out_map_as_operand` path with NO autograd edit [round-4 synth S28]; PLUS J1's two cases — a `Computed` (data-dependent) out_map still trips `adjoint.rs:806`'s existing guard unchanged, and an overlapping-producer graph is rejected by bind's interval check BEFORE autograd ever sees it [round-4 judge J1]; `scripts/proxima-autograd-gate.sh` green with `ran_count`/`passed_count`, either 0 is RED
- N==0 is RED.

##### predict
(nano -> micro) on ACCEPT the emitted MSL contains **no gather-index read** for that operand and `u.out_base` carries the offset — asserted on the emitted text of a **synthetic** fixture, never on the real concatenated codec region (R16's undecidability applies there). The CPU reduce path's throughput is unchanged within ±1% (the offset folds into a base computed once, not into the inner loop) [round-4 synth S28].

##### kill
- any REJECT case the checker ACCEPTs — **a false ACCEPT is two producers racing on one buffer**, a correctness defect that kills the card outright regardless of the perf story [round-4 synth S28]
- any fingerprint or golden drift on an existing `offset == 0` `Reduce` ⇒ the extension is not behaviour-preserving; stop
- `proxima-autograd-gate.sh` red ⇒ the adjoint is not what `:835` implies; stop and derive it before proceeding [round-4 synth S28]

##### memory gate
- gate: MG-1 for the checker (steps 6-8, no process runs the model). MG-3 for the oracle run (step 9), all five clauses [round-4 synth S1]:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 2_097_152 B/step (the 262_144 term is 0 after 5.2)
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_654_467_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_505_367_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: 540 MB (5.2 onward: 400 + the 512×262,144 = 134,217,728 B arena).
- what this card allocates: nothing new — a `project_output_shape` extension and a bind-time interval check, zero runtime allocation of their own; the oracle run allocates the standard decode budget only.

##### rollback
`git revert`; but 9.2 and 9.3 depend on this card and restructure `spec.rs`/`generate.rs` under their features — reverting 9.1 after either has landed means reverting them first, in reverse topological order (§VII); this card is the one point in the plan where a revert is a three-card unwind, stated here rather than as a one-liner (kept: F16's rollback ordering) [round-4 fix F16].

##### blast
`proxima-tensor/src/shape.rs` + `bind.rs`, **cross-crate**: `proxima-autograd` is a consumer of `Reduce::out_map` semantics (`adjoint.rs:835`, `:1033`; `error.rs:73-88`) and `omega/Cargo.toml:196` depends on it, so `scripts/proxima-autograd-gate.sh` runs on this card [round-4 synth S28]. Cross-backend: the CPU path must re-test, since a placed write now runs as an ordinary strided store. `Computed`/scatter/`scatter_output_shape`/`build_scatter_out_layout`/`IndexMap::scatter`/`as_gather_from_output` are ALL untouched [round-4 synth S28].

##### observe
the ACCEPT/REJECT table; the disjointness rejections on the real program; the fingerprint vectors; the goldens; `proxima-tensor-gate.sh`/`proxima-autograd-gate.sh`/`omega-gate.sh`'s counts [round-4 synth S28].

##### reprove
`cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p proxima-tensor --all-features -E 'test(write_offset)'` [round-4 synth S28] + steps 7, 7a, 8, 9, welded.

##### row
```
## ROW <NEXT> -- the write offset was already folded into out_layout.base by layout_of; only project_output_shape ignored it, and only for Affine maps — the scatter path was never on this road

**Card:** 9.1. **Worktree/branch/commit:** proxima-wt-risc10/risc/9-write-placement/<sha at report time>. **Feature:** none, default n/a.
**Allocation budget (hot/setup/cold):** MG-1 for the checker (none); MG-3 for the oracle run (standard decode budget, all five clauses [round-4 synth S1], 540 MB RSS ceiling, PREFILL_CAP 4,654,467,728 B / STEADY_CAP 4,505,367,728 B at kv_capacity_tokens=512). [round-4 fix F11] [round-4 synth S1]
**Predict (one rung ahead, written before running):** on ACCEPT the emitted MSL contains no gather-index read for that operand and `u.out_base` carries the offset, asserted on a synthetic fixture; CPU reduce throughput unchanged within ±1%. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S28]
| arm | cell | n | CoV | load before/after |
| ORACLE (single forward pass) | ORACLE | <n> | <CoV> | <before>/<after> |
**Gates:** proxima-tensor <N run/N passed> (test(write_offset)); proxima-tensor-gate.sh <ran/passed>; proxima-autograd-gate.sh <ran_count/passed_count>; omega-gate.sh <ran_count/passed_count>. [round-4 synth S28]
**Parity:** six parity suites green; fingerprint + golden byte-identity for every existing offset==0 Reduce.
**Census:** ACCEPT/REJECT case table (8 minimum, both directions); disjointness rejections on the real program.
**Home-turf arm:** none — this card runs the ORACLE cell only, not an incumbent-comparable cell.
**Principles engaged and what each changed:** §1 (expression already exists; no new type); §6 (write placement via existing out_map/out_layout.base). **Abandoned:** the backward-walk `Computed`-chain prover and the `Computed`-to-Affine rewrite [round-4 synth S28]; a GPU scatter emitter with name-convention injectivity [§VIII.3]; an atomics-based GPU scatter [§VIII.4]; "shape.rs untouched is the proof" [§VIII.5].
**Re-prove:** the `test(write_offset)` filter + steps 7, 7a, 8, 9, welded. [round-4 synth S28]
```

##### report skeleton
```
CARD 9.1 — <STATUS>
ran: <each command above, EXIT>
N: 8-case ACCEPT/REJECT table / fingerprint+golden identity / six parity suites / autograd round-trip + J1's two cases
numbers: <case table> <fingerprint diff, expect none> <golden hash diff, expect none>
predict vs observed: <one line; if miss: category + work item>
files: proxima-tensor/src/shape.rs:469-485; proxima-tensor/src/bind.rs   diff --stat: <...>
row: <see above>
reprove: test(write_offset) + steps 7, 7a, 8, 9, welded
open: <anything the card could not do, by name>
```

---

#### CARD 9.2 — The per-token write base, and the KV write moves into the graph `[B3 P8.2 first half, S2 5.3 graph half, crit HC-2, MS-3]`

tier: worker
depends_on: [9.1, 6.5]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10          branch: risc/9-write-placement          base: risc/7-geometry
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: kv-scatter-write, in `proxima-tensor`, forwarded   default: off

##### opens
- `omega/src/msl.rs:2216` (`long out_base;` inside `struct Uniforms`) and `:2361` (`long out_offset = u.out_base;`) — **the write offset is ALREADY a per-dispatch uniform, not baked MSL text** (R18, verified in R15's citation set)
- `omega/src/metal.rs:2070` `upload_uniforms`, `:2075` the reuse path
- `proxima-tensor/src/bind.rs:200-215` (`Layout.base` is a baked `i64` on the `BoundOp` — the one place a per-token value cannot live today)
- `proxima-model-interop/src/generate.rs:1393-1400` (the KV roots are program **outputs** today — 97 effective outputs), `:1559` `cached_len += new_count`
- `omega/src/metal.rs:1128-1147` `bound_op_retirement`

##### the one thing this plan may mint, bounded here [crit MS-3]
The KV write's base is `cached_len × row`, which changes every token; a baked `Layout.base` would defeat plan reuse and re-open D4. **The per-token base must not enter the content-keyed uniform cache.** `UNIFORM_BUFFERS` (`metal.rs:2055-2078`) is keyed by the uniform BYTES; a base that changes every token changes the key, misses the cache, allocates a fresh buffer and inserts a new entry every token — unbounded host-heap growth at ~100 B/entry that MG-3's 1 MB/step slope would not catch. So `dynamic_bases: Vec<(position, i64)>` is patched into **6.5's plan-owned per-position uniform buffer (`PlanUniforms`)**, never through `upload_uniforms`/`UNIFORM_BUFFERS` (supersedes F17's separate `dynamic_bases` MetalBuffer) [round-4 fix F17] [round-4 synth S29]; the `+ dynamic_bases[slot]` kernel term from F17 is dropped — the plan-owned uniform carries `out_base` directly, so no added term appears in the emitted MSL for placed writes [round-4 synth S29]. **no new `Op`, no new `BoundOpKind`, no re-bind.** The alternative (making `BoundOp.extents`/`Layout.base` symbolic) is parked with its blast radius named (`grid_threads`, `kernel_cache_key`, `kernel_dispatch_shape`).

##### the output-set change, re-derived here and not left to Phase 6 [crit HC-2]
Turning the K/V writes into in-place placements shrinks `effective_outputs` from ~97 to ~1, so every liveness range from `bound_op_retirement` changes. **This card re-runs 6.5's two soundness tests and its `ARENA_PEAK_BYTES` assertion in its own commit**, and the row carries the before/after output count and the new arena peak. The operand set changes shape as well as the output set: a folded write's `indices` node loses its only consumer (`bound_op_retirement` at `metal.rs:1132-1137` keys `last_use` on `lookup.indices`), so the partition is re-run on the new operand graph, not only on the new output set [round-4 fix F17].

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs` (same worktree as 9.1; no new worktree block)
2. edit: `proxima-tensor/src/spec.rs` — the KV component writes become placed `Reduce`s with `out_layout.base` from `dynamic_bases`, behind `#[cfg(feature="kv-scatter-write")]`
3. edit: `omega/src/metal.rs` — `Plan` gains the `dynamic_bases: Vec<(position, i64)>` field; the base is patched directly into 6.5's `PlanUniforms` buffer at its own position, bypassing `upload_uniforms`/`UNIFORM_BUFFERS` entirely [round-4 synth S29]
4. edit: `proxima-model-interop/src/generate.rs:1393-1400` — the KV roots leave the effective-output set
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-scatter-write \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.2-oracle.log; echo "EXIT=${PIPESTATUS[0]}"` — CPU oracle first
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.2-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo test -p proxima-tensor --all-features 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.2-tensor-gate.log; echo "EXIT=${PIPESTATUS[0]}"`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,kv-scatter-write \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.2-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"` — feature ON
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.2-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"` — feature OFF, interleaved against step 8
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
- N1: `generated_text` and `2651`/`"known"` unchanged
- N2: the effective-output count falls from 97 to ~1, **printed**, and 6.5's arena peak re-asserted against ARENA_TRANSIENT_CAP = 172_812_125 B [round-4 synth S1]
- N3: a scatter write and a read of the same buffer within one command buffer produce the written data — we have one encoder and one command buffer (`metal.rs:449-568`) and the incumbent has no barrier API at `n_cb=1` (R8), so intra-encoder ordering is what is relied on, and the test asserts it
- N4: `plan_hits` still satisfies 6.2's formula (a per-token base must not re-key the plan) — a drop is RED
- N5: `kv_cache_upload_bytes == 0` on every steady step
- N6: `UNIFORM_CACHE_LEN` does not grow (replaces F17's N7 gauge wording — the base is patched directly into the plan-owned `PlanUniforms` buffer, never through the content-keyed `UNIFORM_BUFFERS` cache, so the cache length must stay flat; growth here is the cache-miss failure and a KILL) [round-4 fix F17] [round-4 synth S29]
- N7 the ON tree's 2.1 fingerprint vectors and 3.1 goldens are re-captured in this commit (kept) — the output-set collapse renumbers `NodeId`s far more than 6.1's one leaf did; HC-1's fix applies here too [round-4 fix F17] [round-4 synth S29]
- N==0 is RED.

##### the two binary questions for `dynamic_bases` (answered by writing the code, not by arguing)
*Can an existing primitive express it?* The expression is `plan.dynamic_bases[k] = (position, cached_len * row)` written into the bytes `upload_uniforms` (`metal.rs:2070`) already writes — a `Vec` on the existing `Plan`, no new type, no new `Op`, no new `BoundOpKind`. *What can a caller do that it could not before?* Before: `Layout.base` is baked at bind (`proxima-tensor/src/bind.rs:200-215`), so a per-token write base forces a re-bind and a plan miss. After: `plan.set_dynamic_base(position, base)` moves the write without touching the plan key, so `plan_hits` keeps 6.2's formula (N4). Both lines are written on the row; if N4 fails, the answer to the second question was "nothing" and the field is deleted.

##### predict
(milli -> bench) `gpu_exec_ms` and `step_wall_ms` unchanged within CoV — this card moves the *write*, not the work; the payoff is 9.3's op count. `UNIFORM_BUFFER_REUSES` unchanged between the ON and OFF arms — stated as unchanged ONLY because the plan path no longer consults that cache at all, not because the write is reusing a cached buffer [round-4 synth S29].

##### kill
- the write-then-read within one encoder returns stale data ⇒ the write moves to its own dispatch ordered before the read; if that also fails, the card dies and the row records the ordering finding
- `plan_hits` falls ⇒ the dynamic base leaked into the plan key; stop
- `UNIFORM_CACHE_LEN` grows ⇒ the patch is going through the content cache instead of the plan-owned `PlanUniforms` buffer; stop [round-4 synth S29]
- any `generated_text` drift ⇒ revert (§14)

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1] (per G8), **plus the re-asserted `ARENA_PEAK_BYTES`**:
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 2_097_152 B/step (the 262_144 term is 0 after 5.2)
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_654_467_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_505_367_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: 540 MB (5.2 onward: 400 + the 512×262,144 = 134,217,728 B arena). Both KV slopes must stay at 0.
- what this card allocates: `ARENA_PEAK_BYTES` is re-derived against ARENA_TRANSIENT_CAP = 172_812_125 B (DERIVED) because the output set shrinking 97 -> ~1 changes 6.5's liveness partition; a peak above the term is a NEGATIVE. [round-4 fix F11] [round-4 synth S1]

##### rollback
feature off; `git revert` restores the host-side append path and the 97-output set.

##### blast
`spec.rs` cached-layer builder, `omega/src/metal.rs` (`Plan` +1 field, `upload_uniforms` patch), `generate.rs` roots.

##### observe
effective-output count, `ARENA_PEAK_BYTES`, `plan_hits`, `kv_cache_upload_bytes`, `UNIFORM_BUFFER_REUSES`, `UNIFORM_CACHE_LEN`, the oracle. [round-4 synth S29]

##### reprove
the ORACLE + BENCH cells, welded, in both arms (steps 5, 8, 9).

##### row
```
## ROW <NEXT> -- the KV write moves into the graph at out_layout.base, the output set collapses from 97 to one, and the arena's peak is re-derived on the graph that actually exists

**Card:** 9.2. **Worktree/branch/commit:** proxima-wt-risc10/risc/9-write-placement/<sha at report time>. **Feature:** kv-scatter-write, default-off.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1]; ARENA_PEAK_BYTES re-asserted against ARENA_TRANSIENT_CAP = 172_812_125 B (DERIVED); 540 MB RSS ceiling. [round-4 fix F11] [round-4 synth S1]
**Predict (one rung ahead, written before running):** gpu_exec_ms and step_wall_ms unchanged within CoV; UNIFORM_BUFFER_REUSES unchanged only because the plan path no longer consults that cache. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S29]
| arm | cell | n | CoV | load before/after |
| OFF (feature off) | BENCH | <n> | <CoV> | <before>/<after> |
| ON (kv-scatter-write) | BENCH | <n> | <CoV> | <before>/<after> |
**Gates:** omega <N run/N passed> (omega-gate.sh); proxima-tensor <N/N> (all-features); proxima-model-interop ORACLE <N/N>.
**Parity:** generated_text and 2651/"known" unchanged; intra-encoder write-then-read ordering test.
**Census:** effective-output count (97 -> ~1), plan_hits vs 6.2's formula, kv_cache_upload_bytes (== 0 steady), UNIFORM_BUFFER_REUSES, UNIFORM_CACHE_LEN (bounded, not growing). [round-4 synth S29]
**Home-turf arm:** none — this card's payoff is measured against itself (ON vs OFF), not the incumbent.
**Principles engaged and what each changed:** §1 (extend the existing uniform mechanism, mint nothing but a plan-level Vec); §6 (write placement via out_layout.base). **Abandoned:** none in this card; obliges a re-derivation of 6.5's partition [crit HC-2].
**Re-prove:** the ORACLE + BENCH cells, welded, in both arms.
```

##### report skeleton
```
CARD 9.2 — <STATUS>
ran: <each command above, EXIT>
N: N1 (generated_text unchanged) / N2 (output count 97->~1, arena peak) / N3 (intra-encoder ordering) / N4 (plan_hits formula) / N5 (kv_cache_upload_bytes==0)
numbers: <effective-output count before/after> <ARENA_PEAK_BYTES before/after> <plan_hits/plan_misses> <BENCH ON vs OFF table>
predict vs observed: <one line; if miss: category + work item>
files: proxima-tensor/src/spec.rs; omega/src/metal.rs; proxima-model-interop/src/generate.rs:1393-1400   diff --stat: <...>
row: <see above>
reprove: the ORACLE + BENCH cells, welded, in both arms
open: <anything the card could not do, by name>
```

---

#### CARD 9.3 — Single-range attention; the op-count cell, with wall in the criterion `[S2 6.1+6.2, B3 P8.2, crit b, H-4]`

tier: worker
depends_on: [9.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10          branch: risc/9-write-placement          base: risc/7-geometry
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: attention-single-range, default-off   default: off

##### opens
- `proxima-tensor/src/spec.rs:2303-2319` (the doc naming the IR constraint: "`Reduce::out_map` must stay a pure projection … so nothing upstream of a reduce can splice two tensors into one axis" — **rewritten in this commit to record how the constraint was satisfied**)
- `proxima-tensor/src/spec.rs:2596-2720` (the two-range combine), `:2616-2617`, `:2610`
- fan-out sites the row must scope: `proxima-tensor/src/spec.rs:2867-2891` (Qwen3.5 dense attention), `:3455` (MoE), **`:6465`** (the split-half RoPE path, documented as **NOT** going through this function — so "the even/odd RoPE split collapses in the same move" does **not** cover it and the row says which checkpoints it does cover) [crit H-4]
- `proxima-tensor/src/spec.rs:6282` the sole caller
- R3/M11 (single-range proven 488/488, 1196 → 939 BoundOps)
- the incumbent's 23 real ops/layer (`llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`; views/reshape/permute are no-ops, `ggml-metal.m:1835-1847`; ~740 dispatches/token, R8)

##### the mechanism
With 9.2 landed the new token's K/V are **already in the cache buffer before the attention reduce runs**, so there is only ever ONE source range and the two-range combine is dead code: one `Reduce` over `[0, bucket)` with 6.1's tail mask and the existing causal mask.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10`; `TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs` (same worktree as 9.1/9.2; no new worktree block)
2. recover: `merge-tracked.patch` (0.7; 1 file +1181/−82 on `spec.rs`) as **reference only** — main moved `spec.rs +8735/−2836` since (`0c3bd4f`), so expect a full conflict and **rewrite rather than merge**
3. edit: `proxima-tensor/src/spec.rs:2596-2720,2303-2319` — collapse the two-range combine to one `Reduce` over `[0, bucket)`; the two-range body stays under `#[cfg(not(feature="attention-single-range"))]` so both arms bisect green; rewrite the `:2303-2319` doc to record how the pure-projection constraint was satisfied; behind `#[cfg(feature="attention-single-range")]`
4. edit: `proxima-tensor/src/spec.rs` — add the ops/layer census by grouping `NodeId` ranges to `layer_roots`
5. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,attention-single-range \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.3-oracle-on.log; echo "EXIT=${PIPESTATUS[0]}"` — CPU oracle first, ON
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.3-oracle-off.log; echo "EXIT=${PIPESTATUS[0]}"` — CPU oracle, OFF
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,attention-single-range \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.3-bench-on.log; echo "EXIT=${PIPESTATUS[0]}"` — BENCH ON, round 1
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.3-bench-off.log; echo "EXIT=${PIPESTATUS[0]}"` — BENCH OFF, round 1 (interleave ON/OFF 3x total, repeating steps 7-8 twice more)
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo nextest run -p proxima-tensor --all-features -E 'test(qwen35_hybrid_parity)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/runs/9.3-qwen35-parity.log; echo "EXIT=${PIPESTATUS[0]}"` — Qwen3.5's hybrid path re-parity-tested, not assumed
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
- N1: CPU parity **488/488** (R3 M11's own count; lower is RED) and single-range == two-range to **1e-6** with the **token bit-identical** (summation order genuinely changes; the tolerance is stated and justified)
- N2: real ops/layer **≤ 23**; derived total `32×23 + get_rows + rms_norm + mul + output mul_mat = 740`, so **`op_count <= 780`** (a 5% allowance for our `Iota`/`Constant` control nodes, R13: 39 ops / 0.169 ms); `op_count > 780` fails brief item 8
- N3: `generated_text` and `2651`/`"known"` unchanged
- N4: the route census shows the elementwise count collapsing toward the incumbent's shape
- N5: Qwen3.5's hybrid path **re-parity-tested, not assumed**
- N6 the ON tree's 2.1 fingerprint vectors and 3.1 goldens re-captured in this commit — deleting the two-range combine (`spec.rs:2596-2720`) renumbers every downstream `NodeId` [round-4 fix F18]
- N==0 is RED.

##### predict
(milli -> bench) ops/layer **37 → ≤23**; `encode_dispatch_calls → ≤780`; **`step_wall_ms` moves by less than 1 ms and `gpu_exec_ms` may rise slightly** (one wide reduce over `bucket` slots instead of two narrower ones). **The value of this card is the RISC — the dispatch count, `Concat` still nonexistent, no new `Op` — not the milliseconds. Predicting a win here would be the dishonest move** (R12: 1194 → 616 moved wall 0.036 ms and moved GPU **up** 13.5%).

##### kill
written against BOTH the control and the noise floor [crit b]: the success shape is **dispatches down AND `gpu_exec_ms` not risen beyond 2× the measured CoV (>1.4%) AND `step_wall_ms` not risen beyond both CoV bands**. **Wall is in the criterion**, not only counts. If wall does not move, that is the **second independent** measurement saying the graph is not the mass; record the two-agreeing-results conclusion, land the graph for the RISC and maintenance, and **not as a perf row**. 1.1's adjudication re-opens only if ≤23 ops/layer proves unreachable.

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1] (per G8):
  (1) phys_footprint slope steps 3..S <= 1_000_000 B/step
  (2) device_allocated_bytes slope <= 2_097_152 B/step (the 262_144 term is 0 after 5.2)
  (3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_654_467_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144; at kv_capacity_tokens=512 => 4_505_367_728 B (DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) plan_cache_len <= 1
  (5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  RSS ceiling: 540 MB (5.2 onward: 400 + the 512×262,144 = 134,217,728 B arena).
- what this card allocates: fewer nodes ⇒ fewer arena buffers ⇒ `device_allocated_bytes` must **decrease** vs 9.2; an increase is a NEGATIVE.

##### rollback
feature off; the two-range body stays under `#[cfg(not(...))]`. Highest rebase-conflict surface in the plan (`spec.rs`); rebase against main early and often.

##### blast
`proxima-tensor/src/spec.rs` only — Mistral, Qwen3, Qwen3.5 hybrid and the MoE counterpart; **the split-half RoPE path at `:6465` is explicitly out of scope and the row says so.** Zero emitter change, zero driver change — the duplication was a graph defect upstream of any backend.

##### observe
ops/layer census, `encode_dispatch_calls`, `op_profile_bucket kind=elementwise op_count` (R13: 547 / 7.350 ms), `gpu_exec_ticks`, `step_wall_ms`, the per-route split, `device_allocated_bytes`.

##### reprove
the ORACLE + BENCH cells ON/OFF interleaved 3×, welded (steps 5-8, repeated), + the census assertion.

##### row
```
## ROW <NEXT> -- attention was duplicated because the graph could not write in place: with affine placement the two ranges become one, at or under the incumbent's 23 ops per layer

**Card:** 9.3. **Worktree/branch/commit:** proxima-wt-risc10/risc/9-write-placement/<sha at report time>. **Feature:** attention-single-range, default-off.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1]; device_allocated_bytes must decrease vs 9.2; 540 MB RSS ceiling.
**Predict (one rung ahead, written before running):** ops/layer 37 -> <=23; encode_dispatch_calls -> <=780; step_wall_ms moves < 1 ms, gpu_exec_ms may rise slightly. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| OFF (feature off) | BENCH | <n> | <CoV> | <before>/<after> |
| ON (attention-single-range) | BENCH | <n> | <CoV> | <before>/<after> |
**Gates:** proxima-tensor <N run/N passed> (qwen35_hybrid_parity); proxima-model-interop ORACLE <N/N> ON and OFF.
**Parity:** CPU parity 488/488 (R3 M11); single-range == two-range to 1e-6, token bit-identical; Qwen3.5 hybrid re-parity-tested.
**Census:** ops/layer (target <=23), encode_dispatch_calls (target <=780), elementwise route count collapse toward incumbent shape.
**Home-turf arm:** none — this card's kill is the control (R12's 1194->616 refutation), not an incumbent cell.
**Principles engaged and what each changed:** §6 (write placement collapses the two-range combine); §14 (oracle binds regardless of speed). **Abandoned:** BoundOpKind::CachedAttention [§VIII.1] re-opens only if <=23 ops/layer proves unreachable.
**Re-prove:** the ORACLE + BENCH cells ON/OFF interleaved 3x, welded, + the census assertion.
```
```
## ROW <NEXT> -- the control that says a dispatch collapse is not automatically a win, restated on our tree with wall in the criterion

**Card:** 9.3. **Worktree/branch/commit:** proxima-wt-risc10/risc/9-write-placement/<sha at report time>. **Feature:** attention-single-range, default-off.
**Allocation budget (hot/setup/cold):** MG-3 all five clauses [round-4 synth S1]; device_allocated_bytes must decrease vs 9.2; 540 MB RSS ceiling.
**Predict (one rung ahead, written before running):** the dispatch collapse alone does not predict a wall win — R12's control (1194->616 moved wall 0.036 ms, gpu_exec UP 13.5%) is the standing analogue. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| OFF (feature off) | BENCH | <n> | <CoV> | <before>/<after> |
| ON (attention-single-range) | BENCH | <n> | <CoV> | <before>/<after> |
**Gates:** same as the row above — proxima-tensor and proxima-model-interop gates shared between both titles on this card.
**Parity:** same as the row above.
**Census:** gpu_exec_ms delta vs 2x measured CoV (>1.4% threshold); step_wall_ms delta vs both CoV bands.
**Home-turf arm:** none.
**Principles engaged and what each changed:** §14 (wall is in the kill criterion, not only dispatch counts). **Abandoned:** none.
**Re-prove:** the ORACLE + BENCH cells ON/OFF interleaved 3x, welded, + the census assertion.
```

##### report skeleton
```
CARD 9.3 — <STATUS>
ran: <each command above, EXIT>
N: N1 (488/488 CPU parity, 1e-6, bit-identical) / N2 (ops/layer <=23, op_count <=780) / N3 (generated_text unchanged) / N4 (route census collapse) / N5 (Qwen3.5 re-parity)
numbers: <ops/layer before/after> <encode_dispatch_calls before/after> <BENCH ON vs OFF table, step_wall_ms and gpu_exec_ms> <route census>
predict vs observed: <one line; if miss: category + work item, decomposed into inconsistency vs understanding-gap>
files: proxima-tensor/src/spec.rs:2303-2319,2596-2720,2867-2891,3455,6282   diff --stat: <...>
row: <see above, two ROW blocks>
reprove: the ORACLE + BENCH cells ON/OFF interleaved 3x, welded, + the census assertion
open: <anything the card could not do, by name>
```

---

cards: 8 | commands: 37 | citations: 40
### 5.3.10 PHASE 10 — the sweeps and the cells that do not exist
*Sweeps run **before** the final board so nothing changes the default tree after the board is written [crit O-5]. Worktrees `proxima-wt-risc11` / `risc/10-sweeps` (10.1, 10.2, base `risc/9-write-placement`) and `proxima-wt-risc12` / `risc/10-cross-runtime` (10.3, 10.4, base `risc/0-measure-truth`).*

#### CARD 10.1 — The geometry sweeps, re-run against the graph that now exists `[S2 10.1+10.3, B3 —]`

tier: hands
depends_on: [9.3, 7.1, 7.2]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11          branch: risc/10-sweeps          base: risc/9-write-placement
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — two build-time sizing-config env overrides ([packed_row_block], [cooperative_reduce])      default: n/a

##### opens
- `msl.rs:1550-1552` — the packed arm's `div_ceil(PACKED_ROWS_PER_GROUP) * SIMD_WIDTH` simdgroup count
- 7.1's `[packed_row_block]` and `[cooperative_reduce]` sizing-config sections
- R13's family table (`attn_q` 5.172 ms at 58.5 GB/s, `attn_output` 78.6, `attn_v`/`attn_k` 50-53 vs `ffn_*` 97-109)

##### the ruling
R3/M5 name a MEMORY curve (52 → 147 GB/s from 256 → 8001 simdgroups). **The anchor here is R13's MEASURED family table, not the MEMORY curve.** The reduction lengths changed when attention collapsed (9.3), so 7.2's tuning was against a graph that no longer exists. A config sweep is cheaper than a new kernel and may retire the lever (§1 applied to geometry).

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 -b risc/10-sweeps risc/9-write-placement && mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target`
3. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
4. rows-per-group sweep: each value first prebuilt into its own target dir (`/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target/rows-<value>`) before any measurement [round-4 synth S30], then interleaved round-robin, never blocked by value, `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP ∈ {1,2,4,8}` × 3 runs = 12 cells, MILLI rung:
   ```
   for rep in 1 2 3; do for rows in 1 2 4 8; do
     cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
       OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=$rows PROXIMA_METAL_OP_PROFILE_STEP=3 \
       bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- \
       cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
       2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.1-rows${rows}-rep${rep}.log; echo "EXIT=${PIPESTATUS[0]}"
   done; done
   ```
5. cooperative-reduce width sweep: each value first prebuilt into its own target dir (`/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target/width-<value>`) before any measurement [round-4 synth S30], then interleaved round-robin, `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS ∈ {32,64,128,256,512,1024}` × 3 runs = 18 cells, MILLI rung:
   ```
   for rep in 1 2 3; do for width in 32 64 128 256 512 1024; do
     cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
       OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=$width PROXIMA_METAL_OP_PROFILE_STEP=3 \
       bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- \
       cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
       2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.1-width${width}-rep${rep}.log; echo "EXIT=${PIPESTATUS[0]}"
   done; done
   ```
6. Both MILLI sweeps above are **5-token cells**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` above is inert on that rung — milli budget 5, the two rungs are never read as the same cell [round-4 judge J4]
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. (G9, this card builds omega under 10 distinct config values) `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.1-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
N1 = 12 rows-per-group cells, each nextest invocation reports `N run / N passed`
N2 = 18 cooperative-reduce cells, each nextest invocation reports `N run / N passed`
N3 = 30 total cells; per-family GB/s use 0.2's per-op-codec bytes [round-4 synth S30]; `attn_*` and `reduce-cooperative` op counts **constant across arms** — a moved count is RED (the route changed, not the geometry); monotonic-then-flat expected on the width axis, a non-monotonic curve is the finding and gets its own row
N4 = `scripts/omega-gate.sh` `ran_count`/`passed_count` (step [3/6]/[6/6])
N==0 is RED

##### predict (nano -> micro)
`rows_per_group = 8` halves the simdgroup count and is **10-30% slower**; a value below 4 is faster on the low-row families by **≥5%**. The width optimum is `min(reduction_len/4, 1024)` per the incumbent's own rule (R8) and `reduce-cooperative` lands **≤60%** of its post-9.3 value.

##### kill
- No `rows_per_group` value beats 4 by more than the measured CoV ⇒ **do not build split-K**; row the negative with all 12 numbers.
- No width value beats 32 by more than 2× CoV ⇒ the lever is dead on the new graph; record the negative with all 18 numbers and demote 7.2's feature permanently.
- `nsg=2` regrouping is a four-time negative (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260) and is a different mechanism from split-K; re-proposing it is a discipline failure, not an experiment.

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1], any one failing is a KILL:
  (1) `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  (2) `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak `device_allocated_bytes` <= `PREFILL_CAP` = 4_305_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_654_467_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak `device_allocated_bytes` over steps 3..S <= `STEADY_CAP` = 4_163_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_505_367_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) `plan_cache_len` <= 1 on every step
  (5) `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  plus the RSS ceiling that applies to this card: **540 MB from 5.2 onward** (400 MB + the 512×262,144 = 134,217,728 B arena); build-time config only, `device_allocated_bytes` unchanged in slope and absolute — any increase is a NEGATIVE.
- what this card allocates, in bytes, from synth3: none — build-time config only, no new allocation site.

##### rollback
Two TOML integers.

##### blast
Build config only.

##### observe
Per-route and per-family GB/s; `gpu_exec_ticks`; `gpu_ns_per_op`.

##### reprove
The two sweeps at the landed values.

##### row
```
## ROW <NEXT> -- the simdgroup-starvation hypothesis, tested with a config knob before a kernel
**Card:** 10.1. **Worktree/branch/commit:** proxima-wt-risc11/risc/10-sweeps/<sha at report time>. **Feature:** none, build-time config.
**Allocation budget (hot/setup/cold):** none — build-time config only, no new allocation site.
**Predict (one rung ahead, written before running):** rows_per_group=8 is 10-30% slower; a value below 4 is faster on low-row families by >=5%. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| rows_per_group=1 | MILLI per-op family GB/s | 3 | <...> | <...>/<...> |
| rows_per_group=2 | MILLI per-op family GB/s | 3 | <...> | <...>/<...> |
| rows_per_group=4 | MILLI per-op family GB/s | 3 | <...> | <...>/<...> |
| rows_per_group=8 | MILLI per-op family GB/s | 3 | <...> | <...>/<...> |
**Gates:** omega <N run/N passed> (features: metal); alloc-tier check of omega EXIT <..>.
**Parity:** n/a — config sweep, no body change.
**Census:** attn_*/reduce-cooperative op counts, constant-across-arms check, with N.
**Home-turf arm:** none — geometry sweep against our own kernel only, no incumbent cell run this card.
**Principles engaged and what each changed:** §1 config sweep before kernel. **Abandoned:** none.
**Re-prove:** the rows-per-group sweep loop, command 4.
```
```
## ROW <NEXT> -- the cooperative-reduce width, re-swept against the collapsed graph
**Card:** 10.1. **Worktree/branch/commit:** proxima-wt-risc11/risc/10-sweeps/<sha at report time>. **Feature:** none, build-time config.
**Allocation budget (hot/setup/cold):** none — build-time config only, no new allocation site.
**Predict (one rung ahead, written before running):** width optimum near min(reduction_len/4, 1024); reduce-cooperative lands <=60% of its post-9.3 value. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| width=32 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
| width=64 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
| width=128 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
| width=256 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
| width=512 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
| width=1024 | MILLI reduce-cooperative GB/s | 3 | <...> | <...>/<...> |
**Gates:** omega <N run/N passed> (features: metal); alloc-tier check of omega EXIT <..>.
**Parity:** n/a — config sweep, no body change.
**Census:** attn_*/reduce-cooperative op counts, constant-across-arms check, with N.
**Home-turf arm:** none — geometry sweep against our own kernel only, no incumbent cell run this card.
**Principles engaged and what each changed:** §1 config sweep before kernel. **Abandoned:** demote 7.2's feature permanently, if the kill fires.
**Re-prove:** the width sweep loop, command 5.
```

##### report skeleton
```
CARD 10.1 — <STATUS>
ran: env, worktree add, loadout x2, rows-per-group sweep x12, width sweep x18, omega-gate.sh, EXIT each
N: N1 (12 rows-per-group), N2 (18 width), N3 (30 total cells), N4 (omega-gate ran/passed)
numbers: rows_per_group in {1,2,4,8} vs GB/s x CoV; width in {32,64,128,256,512,1024} vs GB/s x CoV
predict vs observed:
files: opens list above  diff --stat: <build config only>
row: see above (two ROWs)
reprove: the two sweeps at the landed values
open:
```

---

#### CARD 10.2 — Split-K for the starving low-row shapes — with a delete-the-card entry gate `[S2 10.2, crit SD-3, h]`

tier: worker
depends_on: [10.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11          branch: risc/10-sweeps          base: risc/9-write-placement
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: metal-packed-split-k      default: off

##### opens
- `msl.rs:1550-1552`, `:2452-2530`
- 0.7's `splitk-tracked.patch` and `lat-tracked.patch` (R7: `perf/q4k-split-k`, 5 files +352/-48 — the only cards that open them)
- 7.1's `[packed_row_block]`

##### the design / entry gate — this card is DELETED if it does not fire
After 9.3 and 10.1, the route census + family table must **still** show `attn_q`/`attn_k`/`attn_v`/`attn_output` achieving **under 70%** of the `ffn_*` families' GB/s on the same body. If 4.3's winning body or 10.1's knob already closed it, **the card is deleted and the row records why, with the numbers** — a lever no longer needed is a negative worth writing down, not silent scope.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs`
2. entry-gate check (no rebuild): `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.2-entry-gate.log; echo "EXIT=${PIPESTATUS[0]}"` — compute `attn_*` GB/s vs `ffn_*` GB/s from the printed family table; **if >= 70%, stop here, this card is DELETED, write the row with the numbers, do not proceed to step 3.** This and the split-K sweep below are **5-token cells**: `profiles_one_real_decode_step_by_per_op_gpu_time` hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8` above is inert on this rung — milli budget 5, the two rungs are never read as the same cell [round-4 judge J4]
3. edit: `msl.rs:2452-2530` (`push_packed_row_blocked_body`) — a K-axis split into `split_k` partitions with a cheap combine, gated behind `#[cfg(feature = "metal-packed-split-k")]`, `split_k` traced to `[packed_row_block].split_k` (§12)
4. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 1800 -- cargo build --release --features metal,metal-packed-split-k 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.2-build.log; echo "EXIT=${PIPESTATUS[0]}"`
5. split-K sweep: each value first prebuilt into its own target dir (`/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target/splitk-<value>`) before any measurement [round-4 synth S30], then interleaved round-robin, `OMEGA_PACKED_ROW_BLOCK_SPLIT_K ∈ {1,2,4,8}` × 3 runs = 12 cells:
   ```
   for rep in 1 2 3; do for splitk in 1 2 4 8; do
     cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
       OMEGA_PACKED_ROW_BLOCK_SPLIT_K=$splitk PROXIMA_METAL_OP_PROFILE_STEP=3 \
       bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- \
       cargo nextest run --release -p proxima-model-interop --features metal,instrument,metal-packed-split-k \
  --run-ignored all --no-capture -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' \
       2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.2-splitk${splitk}-rep${rep}.log; echo "EXIT=${PIPESTATUS[0]}"
   done; done
   ```
6. parity: `cpu::evaluate` on real `blk.0.attn_q.weight` at every `split_k` value, tolerance 1e-6, and the generated token bit-identical, and the same fixture run 100× byte-identical: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument,metal-packed-split-k \
  --run-ignored all --no-capture -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.2-parity.log; echo "EXIT=${PIPESTATUS[0]}"`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/runs/10.2-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
N1 = entry-gate check, 1 run, decides DELETE vs proceed
N2 = 12 split-K sweep cells (4 values × 3 reps), `attn_*` op counts unchanged (a split that changes the op count changed the route — RED)
N3 = parity vs `cpu::evaluate` at 1e-6 at every split value, generated token bit-identical, same fixture run 100× byte-identical
N4 = `scripts/omega-gate.sh` `ran_count`/`passed_count`
N==0 is RED

##### predict (nano -> micro)
On `metal_vs_cpu.rs`'s `matvec_batch1_f32` Mistral arm, `split_k = 4` raises achieved GB/s at the 4096-row shape from ~58 toward the `ffn` families' ~97, i.e. **[85, 105] GB/s**.

##### kill
- The combine's cost exceeds the split's win (the elementwise count rises and the net is flat) ⇒ dead lever, negative row, all four numbers recorded.
- Any non-determinism across the 100 runs.

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1]:
  (1) `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  (2) `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak `device_allocated_bytes` <= `PREFILL_CAP` = 4_305_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_654_467_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak `device_allocated_bytes` over steps 3..S <= `STEADY_CAP` = 4_163_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_505_367_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) `plan_cache_len` <= 1 on every step
  (5) `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  plus the RSS ceiling: **540 MB from 5.2 onward** (400 MB + 134,217,728 B arena).
- what this card allocates, in bytes, from synth3: partials are `split_k` extra intermediates per matvec through 6.5's arena — the peak grows by `split_k × attn_output_bytes`, **computed and printed before enabling**; a peak above ARENA_TRANSIENT_CAP (172_812_125 B) is a NEGATIVE and rolls back. [round-4 fix F11] [round-4 synth S1]

##### rollback
Default-off feature.

##### blast
`msl.rs` packed body + `grid_threads`'s packed arm, behind a feature. No graph, no driver change.

##### observe
The four `attn_*` families' GB/s, using 0.2's per-op-codec bytes [round-4 synth S30], `ARENA_PEAK_BYTES`, the route census.

##### reprove
The sweep at the landed `split_k`.

##### row
```
## ROW <NEXT> -- split-K for the starving low-row attention shapes — or: the census says the body already closed it, and here is the number that deleted this card
**Card:** 10.2. **Worktree/branch/commit:** proxima-wt-risc11/risc/10-sweeps/<sha at report time>. **Feature:** metal-packed-split-k, default-off.
**Allocation budget (hot/setup/cold):** split_k x attn_output_bytes extra intermediates through 6.5's arena, computed and printed before enabling; ceiling = ARENA_TRANSIENT_CAP = 172_812_125 B. [round-4 fix F11] [round-4 synth S1]
**Predict (one rung ahead, written before running):** split_k=4 raises 4096-row attn shape GB/s to [85, 105]. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| entry-gate check | attn_* vs ffn_* GB/s ratio | 1 | n/a | <...>/<...> |
| split_k=1 | matvec_batch1_f32 GB/s | 3 | <...> | <...>/<...> |
| split_k=2 | matvec_batch1_f32 GB/s | 3 | <...> | <...>/<...> |
| split_k=4 | matvec_batch1_f32 GB/s | 3 | <...> | <...>/<...> |
| split_k=8 | matvec_batch1_f32 GB/s | 3 | <...> | <...>/<...> |
**Gates:** omega <N run/N passed> (features: metal,metal-packed-split-k); alloc-tier check of omega EXIT <..>.
**Parity:** real blk.0.attn_q.weight, relative-to-batch-peak 1e-6 vs f32 oracle at every split_k; generated token bit-identical; 100x byte-identical.
**Census:** attn_* op counts unchanged across split_k values, with N.
**Home-turf arm:** none — this card measures our own kernel against our own cpu::evaluate oracle, not llama.cpp.
**Principles engaged and what each changed:** §1 the entry gate as a delete-the-card mechanism. **Abandoned:** none, or the card itself if the entry gate does not fire.
**Re-prove:** the split-K sweep at the landed split_k, command 5.
```

##### report skeleton
```
CARD 10.2 — <STATUS: DONE | DELETED | NEGATIVE | VOID | STOPPED>
ran: env, entry-gate check, [if proceeding: edit, build, split-K sweep x12, parity, omega-gate.sh], EXIT each
N: N1 (entry-gate), N2 (12 split-K cells), N3 (parity + 100x determinism), N4 (omega-gate ran/passed)
numbers: attn_* vs ffn_* GB/s ratio at entry; split_k in {1,2,4,8} vs GB/s x CoV
predict vs observed:
files: msl.rs:2452-2530 (if not deleted)  diff --stat: <...>
row: see above
reprove: the split-K sweep at the landed split_k
open:
```

---

#### CARD 10.3 — torch-MPS, honest scope `[S2 8.1, B3 P9.1]`

tier: worker
depends_on: [0.6]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12          branch: risc/10-cross-runtime          base: risc/0-measure-truth
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — `--device` flag added to an existing script, default `cpu`      default: n/a

##### opens
- `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` — verified: only `--threads` and `--runs`, **no device flag**
- `model.py` (the model is **mnist.onnx**: 3× Conv+Relu, BatchNorm, Flatten, Gemm+Relu, Gemm, BatchNorm, LogSoftmax — not a transformer, not Q4_K)
- `omega/benches/metal_vs_cpu.rs` (the **only** GPU bench outside decode, registered `omega/Cargo.toml:207-210`, doc says **UNRUN**, R9) — this card does not run the omega `metal_vs_cpu` bench for the first time; it runs it in FULL mode for the first time. CI already smoke-runs it with `-- --quick` on `macos-latest` (`.github/workflows/proxima-tensor.yml:191-205`, the `omega-compare-bench` job) and `omega-gate.sh [2/6]` builds it, so the bench compiles and runs; what does not exist is a RECORDED cell (the bench's own doc still says UNRUN, R9 — a stale doc, fixed by this card) [round-4 fix F19]
- `omega/tests/training_step_parity.rs:400-607` (GPU train step, untimed)
- torch 2.13.0 with MPS verified present (R0)

##### the honest scope, on the row before any number
Torch has no Q4_K kernel and no GGUF loader: **a torch-MPS arm on the openchat decode does not exist and cannot be built without changing what is compared.** Two comparable surfaces exist and both are labelled: (a) **mnist batch-1 f32**, the surface the repo already models — a cold-path arm by the frequency bands, `design-favors: incumbent`; (b) the **matvec shapes** (`[1,4096]×[4096,4096]`, `[1,4096]×[4096,14336]`, `[1,14336]×[14336,4096]`) as a **roofline companion** — "what a tuned framework achieves at these shapes on this silicon", **not** a decode competitor.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 -b risc/10-cross-runtime risc/0-measure-truth && mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target`
3. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
4. edit: `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` — add `--device {cpu,mps}` argument, default `cpu` so the existing arm stays byte-identical, and call `torch.mps.synchronize()` **before** each timer stop when `device == "mps"` (the MPS analogue of 0.6's readback rule)
5. torch arms, interleaved A/B/A/B, 5 runs each, mnist batch-1: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- python3.12 proxima-onnx/scripts/torch_reference/inference_bench.py --threads 1 --runs 200 --device cpu 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.3-torch-cpu.log; echo "EXIT=${PIPESTATUS[0]}"` interleaved with `--device mps` into `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.3-torch-mps.log`
6. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo bench --release --features metal --bench metal_vs_cpu 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.3-omega-bench.log; echo "EXIT=${PIPESTATUS[0]}"`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.3-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
N1 = 3 torch arms (cpu existing, cpu explicit, mps) × 5 runs = 15, reporting p50/p95/p99/mean/CoV
N2 = 4 omega bench arms (`gemm_square_f32` 512/1024/2048, `matvec_batch1_f32`) × 5 runs = 20 — this is the first CoV-bearing, mutex-serialised, non-`--quick` LOCAL cell for `metal_vs_cpu`, and it cross-checks against the CI `--quick` baseline (`.github/workflows/proxima-tensor.yml:191-205`); a full-mode number that disagrees with the quick smoke beyond CoV is named, not silently kept [round-4 synth S31]
N3 = the harness asserts `torch.backends.mps.is_available()` **and** `next(model.parameters()).device.type == "mps"` or the arm silently ran on CPU and exits 0
N4 = `scripts/omega-gate.sh` `ran_count`/`passed_count` — a `required-features`-gated bench never invoked compiles to nothing and may not even compile
N==0 is RED

##### predict (micro -> milli)
**torch-MPS is SLOWER than torch-CPU at mnist batch 1** (a ~14-node graph where per-op MPS dispatch dominates); `matvec_batch1_f32` at Mistral shapes lands under **25%** of 0.6's measured `traffic_gbs` ceiling (the low-simdgroup starvation R3/M5 names, anchored on 0.6's MEASURED ceiling and not on the MEMORY curve). Both directions are the result; the loss is reported first (§19).

##### kill
- MPS silently falls back ⇒ the arm is void; report it as a gap in torch's own harness, never as our win.
- the full-mode run's numbers disagree with the `--quick` CI run's by more than the measured CoV ⇒ `--quick` is not a smoke of the same cell; record both [round-4 fix F19].

##### memory gate
- gate: MG-2 at the **540 MB ceiling** (5.2, 400 MB + 134,217,728 B arena): peak task RSS <= 540 MB (R13 prefill 310-357 MB was the 400 MB-era figure); `current_allocated_size()` returns to its pre-probe value ±2 MB; the row records peak RSS via `/usr/bin/time -l` and `torch.mps.current_allocated_memory()`, because an arm that swaps invalidates every interleaved cell sharing the box.
- what this card allocates, in bytes, from synth3: none stated — a python torch process and an existing unrun omega bench, no new Rust allocation site.

##### rollback
Revert the flag and the timed arm; the venv is gitignored by 0.9.

##### blast
One python file + one existing unrun bench; zero Rust hot path.

##### observe
p50/p95/p99/mean/CoV per arm; the device assertion; peak RSS; GB/s against 0.6's named denominator column.

##### reprove
The three welded commands.

##### row
```
## ROW <NEXT> -- the first torch-MPS cell: mnist batch-1 and our matvec shapes, and what neither tells us about Q4_K decode
**Card:** 10.3. **Worktree/branch/commit:** proxima-wt-risc12/risc/10-cross-runtime/<sha at report time>. **Feature:** none, --device flag default cpu.
**Allocation budget (hot/setup/cold):** none stated — python torch process + existing unrun omega bench, no new Rust allocation site.
**Predict (one rung ahead, written before running):** torch-MPS slower than torch-CPU at mnist batch 1; matvec_batch1_f32 under 25% of 0.6's traffic_gbs ceiling. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| torch-cpu (existing) | mnist batch-1 p50/p95/p99/mean | 5 | <...> | <...>/<...> |
| torch-mps | mnist batch-1 p50/p95/p99/mean | 5 | <...> | <...>/<...> |
| omega gemm_square_f32 512/1024/2048 | Metal GB/s | 5 each | <...> | <...>/<...> |
| omega matvec_batch1_f32 (Mistral shapes) | Metal GB/s | 5 | <...> | <...>/<...> |
**Gates:** omega <N run/N passed> (features: metal); alloc-tier check of omega EXIT <..>.
**Parity:** n/a — a comparability card, not a fidelity card.
**Census:** torch device assertion (is_available + device.type==mps), pass/fail, with N.
**Home-turf arm:** none — no llama.cpp-Metal cell this card; the comparison is torch-CPU vs torch-MPS and our own omega bench, not the decode incumbent.
**Principles engaged and what each changed:** §19 the loss is reported first. **Abandoned:** none.
**Re-prove:** the three welded commands, step 5-6.
```

##### report skeleton
```
CARD 10.3 — <STATUS>
ran: env, worktree add, loadout, edit, torch arms x15, omega bench x20, loadout, omega-gate.sh, EXIT each
N: N1 (15 torch-arm runs), N2 (20 omega-bench runs), N3 (device assertion pass/fail), N4 (omega-gate ran/passed)
numbers: torch-cpu vs torch-mps p50/p95/p99/mean/CoV; omega bench GB/s per arm
predict vs observed:
files: proxima-onnx/scripts/torch_reference/inference_bench.py:29-32  diff --stat: <...>
row: see above
reprove: the three welded commands
open:
```

---

#### CARD 10.4 — ORT-CoreML, honest scope, and the one unbounded build `[S2 8.2, B3 P9.2, crit SD-2, k]`

tier: worker
depends_on: [0.9]        scheduled terminal within Phase 10
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12          branch: risc/10-cross-runtime          base: risc/0-measure-truth
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — `--provider` flag added to an existing script, default `cpu`      default: n/a

##### opens
- repo-root `scripts/onnx_reference/bench.py:96` — verified: `providers=["CPUExecutionProvider"]` hardcoded [round-4 synth S32]
- `export_model.py` (the model is **BAAI/bge-small-en-v1.5**, f32)
- the fidelity fields `cosine_similar` / `cosine_dissimilar_a/b` at `:82-85`
- `run.sh` (pinned venv, `ONNX_REF_PYTHON=python3.12`); `onnxruntime` is **not installed** (R0); an ORT source checkout exists at `~/repos/others/onnxruntime`

##### the honest scope
ORT has no Q4_K GGUF path and the CoreML EP has no int4 matvec: **there is no ORT arm on the openchat decode.** The comparable surface is BGE-small f32 embedding, which the repo already exports and benches on CPU. Frequency band: the **80% case for the BGE product**, near-zero for the decode product — a loss here gates the embedding claim, not the decode claim, and the row says which.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs` (worktree already created by 10.3 on this branch — no worktree-add block)
2. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
3. edit: repo-root `scripts/onnx_reference/bench.py:96` — add `--provider {cpu,coreml}` argument, default `cpu` (preserving today's arm byte-for-byte) into the `InferenceSession` call [round-4 synth S32]
4. **OWNER-AUTHORIZED ACTION**: install `onnxruntime` into a **dedicated** venv, not `torch_reference/venv` [round-4 synth S32]: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && python3.12 -m venv .venv-ort && .venv-ort/bin/pip install onnxruntime` (the wheel path is tried first, no mutex — this is not a build/probe/bench/decode/gate command)
5. if no wheel, the capped source build, mutex-held with `--wait 14400`, no other card scheduled while it runs [round-4 synth S32], scheduled terminal: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 14400 -- /usr/bin/time -l ~/repos/others/onnxruntime/build.sh --config Release --use_coreml --build_wheel --parallel 4 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.4-ort-build.log; echo "EXIT=${PIPESTATUS[0]}"`
6. measurement runs, welded, 2 providers × 5 runs: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- .venv-ort/bin/python3.12 scripts/onnx_reference/bench.py --provider cpu 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.4-ort-cpu.log; echo "EXIT=${PIPESTATUS[0]}"` interleaved with `--provider coreml` into `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc12/runs/10.4-ort-coreml.log`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc12 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`

##### expect
N1 = `get_available_providers()` contains `CoreMLExecutionProvider`
N2 = 2 providers × 5 runs with the fidelity fields per provider **or** one explicit `FEATURE GAP: CoreMLExecutionProvider unavailable/rejected the graph` row carrying the ORT error text
N3 = assert `session.get_providers()[0] == "CoreMLExecutionProvider"` **and** the **partition node count** — a 1-node CoreML partition beside 200 CPU nodes is a CPU cell wearing a CoreML label
**A missing arm is RED; a documented gap is green.**

##### predict (micro -> milli)
The CoreML EP takes a **partial** partition (>1 partition, <100% of nodes) and ms/sentence lands within **2×** of the CPU EP with the fidelity fields unchanged (fp32 path).

##### kill
- Fidelity drift (cosine similar/dissimilar move) ⇒ CoreML chose fp16; that is a **different arm** and must be labelled as one (§14).
- Wheel unavailable **and** the source build fails ⇒ a documented **feature gap**, never omitted (§19: an omitted loss is a verdict).

##### memory gate
- gate: MG-2; the build's peak RSS recorded via `/usr/bin/time -l`; **KILL** if the build's peak forces swap, and the box is re-sealed before the next measuring card; RSS ceiling for the measurement runs: **540 MB from 5.2 onward** (400 MB + 134,217,728 B arena) — the build step is exempt from this ceiling by its own `--parallel 4` cap and peak-RSS recording, not by the 540 MB bound.
- what this card allocates, in bytes, from synth3: none stated for the measurement runs; the build's peak RSS is recorded, not bounded by a synth3-given number — zero Rust allocation site either way.

##### rollback
Revert the flag; remove the venv (gitignored).

##### blast
Two python files; zero Rust.

##### observe
ms/sentence, CoV, provider partition counts (`sess_options.log_severity_level=0`), fidelity fields, peak RSS.

##### reprove
The two welded measurement commands.

##### row
```
## ROW <NEXT> -- the first ORT-CoreML cell: BGE-small, partitioned, and what "GPU" means when half the graph runs on CPU
**Card:** 10.4. **Worktree/branch/commit:** proxima-wt-risc12/risc/10-cross-runtime/<sha at report time>. **Feature:** none, --provider flag default cpu.
**Allocation budget (hot/setup/cold):** none stated for measurement runs; build peak RSS recorded, not bounded; zero Rust allocation site.
**Predict (one rung ahead, written before running):** CoreML EP takes a partial partition, ms/sentence within 2x of CPU EP, fidelity unchanged. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| ORT-CPU | BGE-small ms/sentence + fidelity | 5 | <...> | <...>/<...> |
| ORT-CoreML | BGE-small ms/sentence + fidelity | 5 | <...> | <...>/<...> |
**Gates:** n/a — zero Rust touched this card, no omega-gate.sh run.
**Parity:** cosine_similar / cosine_dissimilar_a/b fidelity fields, CPU vs CoreML, fp32 path.
**Census:** provider partition node counts (session.get_providers()[0], node-count split), with N.
**Home-turf arm:** none — this is the embedding lane's own CPU-vs-CoreML comparison, not the decode incumbent.
**Principles engaged and what each changed:** §14 an omitted loss is a verdict — documented gap is green. **Abandoned:** none.
**Re-prove:** the two welded measurement commands, step 6.
```

##### report skeleton
```
CARD 10.4 — <STATUS>
ran: env, loadout, edit, venv install, [build if no wheel], measurement runs x10, loadout, EXIT each
N: N1 (provider availability), N2 (10 measurement runs or 1 documented gap), N3 (provider[0]+partition assertion)
numbers: ORT-CPU vs ORT-CoreML ms/sentence, CoV, fidelity fields, partition counts
predict vs observed:
files: repo-root scripts/onnx_reference/bench.py:96  diff --stat: <...> [round-4 synth S32]
row: see above
reprove: the two welded measurement commands
open:
```

---

### 5.3.11 PHASE 11 — the log, the roofline, and the board
*Worktree `proxima-wt-risc13` / `risc/11-board`, base `risc/10-sweeps`.*

#### CARD 11.1 — Discipline rows, rooflines, ai_docs closure `[S2 9.1, B3 P10.1 half]`

tier: hands
depends_on: [0.9, 1.1, 3.4, 0.6, 10.1]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13          branch: risc/11-board          base: risc/10-sweeps
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none      default: n/a

##### opens
- `discipline.md:18736` (ROW 233 is main's last, its row format is the one this card's new rows follow) [round-4 synth S33]
- `rooflines.md:396-479`, `:411` (GPU ceiling = DEBT), `:751` (summary row), `:766-773` (the doc's own note that the GPU ratio "is not a gap-to-machine at all" — now answerable)
- `ai_docs/AGENT.md`
- ai_docs is FOUR JSONL files (index, examples-index, task-routes, invariants) [round-4 synth S33]

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs`
2. `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 -b risc/11-board risc/10-sweeps && mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target`
3. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1` — read the current last row before renumbering
4. renumber every `<NEXT>` placeholder sequentially from that grep, in land order: `edit: proxima-tensor/docs/discipline.md — replace each literal "## ROW <NEXT>" heading with the next sequential integer above the tail-grep value, in the order the cards landed`
5. `edit: rooflines.md:411 — replace the DEBT cell with 0.6's measured ceiling in both denominators; edit: rooflines.md:751 — update the summary row; edit: rooflines.md:766-773 — answer the doc's own note now that a real ceiling exists`
6. `edit: ai_docs/AGENT.md's invariants section — attach at least one evidence pointer to each of 0.9's five invariants and add two more: proxima.gpu.dispatch_count_is_not_the_denominator (evidence: R12's 1194->616 and 9.3's second test), proxima.gpu.memory_is_a_kill_criterion (evidence: G8's formula and both slopes per steady step)`
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && grep -c '^## ROW <NEXT>' proxima-tensor/docs/discipline.md` (must be 0)
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1` (last row number must be monotonically greater than 233)
9. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && jq -c . ai_docs/task-routes.jsonl > /dev/null; echo "EXIT=$?"` and same for `ai_docs/invariants.jsonl`, `ai_docs/index.jsonl` and `ai_docs/examples-index.jsonl` [round-4 synth S33]
10. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && bash ai_docs/query.sh gpu-decode-perf`
11. (d) the CI matrix: every feature and cfg this plan landed is present in `.github/workflows/proxima-tensor.yml`'s job list, checked by grep per feature name; a feature with no CI job is RED (memory: the gate-glob is not the CI job-set) [round-4 fix F20]
12. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && grep -c "3.54x\|17.470\|228.9" proxima-tensor/docs/discipline.md` — run BOTH before landing (expect 0, main's log does not know the 2026-09-02 session happened) and after landing (expect >= 3) [round-4 synth S33]
13. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && grep -c '[^/]bind\.rs:' docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md` — must be 0 (G0: every `bind.rs` cite carries its crate path) [round-4 judge J3]

##### expect
N1 = `grep -c '^## ROW <NEXT>' discipline.md == 0` after landing
N2 = the last row number monotonically greater than 233
N3 = `jq -c .` parses all four JSONL files (a malformed line is RED) [round-4 synth S33]
N4 = `bash ai_docs/query.sh gpu-decode-perf` returns >= 1 row
N5 = every landed row has zero blank cells across the 16-gate table; every negative row carries its number
N6 = `grep -c "3.54x\|17.470\|228.9" proxima-tensor/docs/discipline.md` goes from 0 before landing to >= 3 after (main's log does not know the 2026-09-02 session happened) [round-4 synth S33]
N7 = `grep -c '[^/]bind\.rs:' docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md == 0` (G0: every `bind.rs` cite carries its crate path) [round-4 judge J3]
N==0 on any file is RED

##### predict
none — a protocol and a records card.

##### kill
- Two branches carrying the same literal row number reach main ⇒ the protocol was bypassed; renumber before the next land.
- Malformed JSONL or an empty query ⇒ fix the record shape, never bypass the index (AGENT.md is explicit).

##### memory gate
- gate: MG-1 — build/lint only, no process runs the model; build exit 0; no new heap-holding `static`/`thread_local` without a bound stated at the site; docs and JSONL edits carry no process-memory surface. RSS ceiling: n/a — no process runs the model this card.
- what this card allocates, in bytes, from synth3: none — docs only.

##### rollback
`git revert`; docs only.

##### blast
`discipline.md`, `rooflines.md`, four JSONL files [round-4 synth S33]. Zero source.

##### observe
Row monotonicity; record counts per file; query hit count; the `3.54x\|17.470\|228.9` grep count before/after [round-4 synth S33].

##### reprove
The grep + the four `jq` commands [round-4 synth S33] + `bash ai_docs/query.sh gpu-decode-perf` + `grep -c '[^/]bind\.rs:' docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md` (must be 0) [round-4 judge J3].

##### row
```
## ROW <NEXT> -- main's log learns the GPU session happened, the roofline DEBT is paid, and ai_docs carries the lane's invariants with evidence
**Card:** 11.1. **Worktree/branch/commit:** proxima-wt-risc13/risc/11-board/<sha at report time>. **Feature:** none.
**Allocation budget (hot/setup/cold):** none — docs only.
**Predict (one rung ahead, written before running):** none — a protocol and a records card. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
| n/a | docs/JSONL edit, no timed cell | n/a | n/a | n/a |
**Gates:** n/a — zero source touched, no omega-gate.sh run.
**Parity:** n/a.
**Census:** row monotonicity (last row > 233), record counts across four JSONL files, ai_docs query hit count, the 3.54x/17.470/228.9 grep count before/after (0 -> >=3), with N. [round-4 synth S33]
**Home-turf arm:** none — no llama.cpp-Metal cell this card.
**Principles engaged and what each changed:** §16 re-provability; ai_docs/AGENT.md's own rule to ADD records, not bypass. **Abandoned:** none.
**Re-prove:** the grep + the four jq commands + bash ai_docs/query.sh gpu-decode-perf. [round-4 synth S33]
```
Row format follows ROW 233's [round-4 synth S33].

##### report skeleton
```
CARD 11.1 — <STATUS>
ran: env, worktree add, renumber grep, doc edits x3, verification grep/jq/query, EXIT each
N: N1 (leftover <NEXT> count == 0), N2 (last row > 233), N3 (jq parse x4 files), N4 (query hit count >= 1), N5 (blank-cell/negative-row count), N6 (3.54x/17.470/228.9 grep 0 -> >=3), N7 (bare bind.rs cite grep in plan.md == 0) [round-4 synth S33] [round-4 judge J3]
numbers: last discipline.md row number; JSONL record counts; ai_docs query hit count
predict vs observed: predict is none — a records card
files: discipline.md, rooflines.md:411,751,766-773, ai_docs/AGENT.md, four JSONL files  diff --stat: <...> [round-4 synth S33]
row: see above
reprove: the grep + the four jq commands + bash ai_docs/query.sh gpu-decode-perf + the bare-bind.rs-cite grep in plan.md [round-4 judge J3]
open:
```

---

#### CARD 11.2 — The final board, sealed AFTER every sweep `[S2 9.2, B3 P10.1, crit O-5]`

tier: hands
depends_on: [11.1, 9.3, 10.1, 10.2, 10.3, 10.4, 6.6] (drops 8.3 — Phase 8 runs after the board) [round-4 synth S27] [round-4 synth S34]
worktree: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13          branch: risc/11-board          base: risc/10-sweeps
target_dir: /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target        lock: /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
feature: none — composed sweep of every already-landed feature at its landed value      default: n/a

##### opens
- none — synth3 gives no file:line citations for 11.2; the card composes numbers already produced by cards 9.3, 8.3, 10.1, 10.2, 10.3, 10.4, 6.6, and the discipline/roofline updates from 11.1.

##### commands
1. `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13; TD=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target; LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock; mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs` (worktree already created by 11.1 on this branch — no worktree-add block)
2. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
3. ours, 5 rounds interleaved, BENCH rung, every landed feature at its landed value: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- cargo nextest run --release -p proxima-model-interop --features metal,instrument \
  --run-ignored all --no-capture -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs/11.2-ours-round${n}.log; echo "EXIT=${PIPESTATUS[0]}"` repeated for round in 1..5, interleaved A/B/A/B/A with both incumbent arms below
4. llama.cpp-Metal `-fa 0`, 5 rounds interleaved with step 3: `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99 -fa 0 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs/11.2-llama-fa0-round${n}.log; echo "EXIT=${PIPESTATUS[0]}"`
5. llama.cpp-Metal `-fa 1`, 5 rounds interleaved: same command with `-fa 1` into `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs/11.2-llama-fa1-round${n}.log`
6. torch-MPS and ORT-CoreML cells at 10.3/10.4's landed arms, folded into the same interleaved round set (or their documented gap rows carried forward unchanged)
7. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && pgrep -fl 'cargo|rustc|llama|criterion|nextest'; sysctl -n vm.loadavg`
8. `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock --wait 5400 -- bash scripts/omega-gate.sh 2>&1 | tee /Users/brianbruggeman/repos/slot-0/proxima-wt-risc13/runs/11.2-omega-gate.log; echo "EXIT=${PIPESTATUS[0]}"`

##### expect
N1 = 5 rounds × ours BENCH cell, both incumbent arms (`-fa 0`, `-fa 1`) interleaved
N2 = torch-MPS and ORT-CoreML cells present or their documented gap carried forward
N3 = every board cell filled: ours (`step_wall_ms`, `gpu_exec_ms` [named host-ticks] / `gpu_device_ms` [named GPU-timestamp], `greedy_pick_ms`, CoV, n) [round-4 synth S34], llama `-fa 0`, llama `-fa 1`, torch-MPS, ORT-CoreML (or its documented gap), the roofline fraction **naming 0.6's denominator column**, both memory slopes with both caps, `design-favors` and a frequency band per cell, and a provenance tag MEASURED / DERIVED / CHOSEN per number [round-4 synth S34]. **A blank cell is RED.**
N4 = `scripts/omega-gate.sh` `ran_count`/`passed_count`
N==0 is RED

##### predict (bench) — the plan's single composed prediction, made once, here
`step_wall_ms` **[46.8, 57.3]** and `gpu_exec_ms` **[43.5, 50.6]**, ratio **2.67-3.27x** against 0.5's chosen incumbent arm, with δ_b at its 6.3-measured endpoints [round-4 synth S34]. **Derivation: §IV's band ladder, every term carried from a MEASURED card delta, none re-derived from theory. This is a composition prediction and therefore the weakest number in this document** — each component was measured alone and their sum is DERIVED. Noted without being treated as confirmation: the parallel lane reached 2.95x by a different route (R12).

##### kill
- `step_wall_ms` **> 59.0** ⇒ R13's decomposition is wrong somewhere and the row names **which bucket did not move, by counter**, before any further card is scheduled.
- If the composed number is worse than the best single-card number, **the cards interact** and the interaction is the next work item, decomposed into inconsistency vs understanding-gap.

##### memory gate
- gate: MG-3, all five clauses [round-4 synth S1], at the landed `kv_capacity_tokens`:
  (1) `phys_footprint` slope over steps 3..S <= 1_000_000 B/step
  (2) `device_allocated_bytes` slope <= 262_144 + 2_097_152 B/step
  (3a) PREFILL peak `device_allocated_bytes` <= `PREFILL_CAP` = 4_305_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_654_467_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (3b) STEADY peak `device_allocated_bytes` over steps 3..S <= `STEADY_CAP` = 4_163_000_000 * 1.05 + `kv_capacity_tokens`*262_144 (at `kv_capacity_tokens`=512 => 4_505_367_728 B, DERIVED) [round-4 fix F10] [round-4 fix F11] [round-4 synth S1]
  (4) `plan_cache_len` <= 1 on every step
  (5) `UNIFORM_CACHE_LEN` does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main) [round-4 synth S1]
  plus the RSS ceiling: **540 MB from 5.2 onward** (400 MB + 134,217,728 B arena). A breach is a board-level NEGATIVE and demotes the offending feature **before** the board is written.
- what this card allocates, in bytes, from synth3: none new — the underlying features' allocations are already accounted on their own cards; this card composes them.

##### rollback
n/a (measurement); the underlying features demote one line each. A wrong row is corrected in place with a dated note, **never silently deleted**.

##### blast
Docs.

##### observe
Every counter in the board, `greedy_pick_ms`, the route census, the fraction-of-ceiling, both memory slopes, CoV per arm, loadout. [round-4 synth S34]

##### reprove
The seal command.

##### row
```
## ROW <NEXT> -- the GPU board, composed and re-sealed after the sweeps: <ratio>x against -fa 0 and <ratio>x against -fa 1
**Card:** 11.2. **Worktree/branch/commit:** proxima-wt-risc13/risc/11-board/<sha at report time>. **Feature:** none, composed sweep of every landed feature at its landed value.
**Allocation budget (hot/setup/cold):** none new — composed from the already-accounted allocations of every landed feature.
**Predict (one rung ahead, written before running):** step_wall_ms [46.8, 57.3], gpu_exec_ms [43.5, 50.6], ratio 2.67-3.27x against 0.5's chosen incumbent arm, delta_b at its 6.3-measured endpoints — DERIVED, the weakest number in this document. **Observed:** <...>. **Miss category + work item:** <...|none>. [round-4 synth S34]
| arm | cell | n | CoV | load before/after |
| ours (every feature landed) | step_wall_ms / gpu_exec_ms (host-ticks) / gpu_device_ms (GPU-timestamp) / greedy_pick_ms | 5 | <...> | <...>/<...> |
| llama.cpp-Metal -fa 0 | ms/token | 5 | <...> | <...>/<...> |
| llama.cpp-Metal -fa 1 | ms/token | 5 | <...> | <...>/<...> |
| torch-MPS | 10.3's landed cell | <...> | <...> | <...>/<...> |
| ORT-CoreML | 10.4's landed cell or documented gap | <...> | <...> | <...>/<...> |
**Gates:** omega <N run/N passed> (features: metal); alloc-tier check of omega EXIT <..>.
**Parity:** generated_text == "Here is a simple Python function that returns"; oracle token 2651/"known" (G10).
**Census:** route census, memory slopes (both clauses), fraction-of-ceiling against 0.6's denominator, with N.
**Home-turf arm:** llama.cpp-Metal at both -fa 0 and -fa 1, both on this row.
**Principles engaged and what each changed:** §IV the band ladder, every card's own delta. **Abandoned:** §VIII items 1-19 as applicable to what did not land.
**Re-prove:** the seal command, steps 3-6.
```

##### report skeleton
```
CARD 11.2 — <STATUS>
ran: env, loadout, ours BENCH x5, llama -fa0 x5, llama -fa1 x5, torch-MPS/ORT-CoreML cells, loadout, omega-gate.sh, EXIT each
N: N1 (5-round ours+incumbent interleave), N2 (cross-runtime cells present or documented gap), N3 (board cell blank count == 0), N4 (omega-gate ran/passed)
numbers: step_wall_ms, gpu_exec_ms (host-ticks), gpu_device_ms (GPU-timestamp), greedy_pick_ms, CoV, n per arm; ratio vs -fa 0 and -fa 1; roofline fraction; both memory slopes [round-4 synth S34]
predict vs observed: [46.8, 57.3] / [43.5, 50.6] / 2.67-3.27x vs <observed> [round-4 synth S34]
files: docs only (discipline.md, rooflines.md board table)  diff --stat: <...>
row: see above
reprove: the seal command
open:
```

---

cards: 6 | commands: 47 | citations: 20

### 5.4.1 Dependency graph

**Prose, agreeing with the picture.** Phase 0 gates everything: nothing is measured before the mutex exists (0.1), the byte counters tell the truth in **both** loops and the uniform cache is read for the first time (0.2), the batched device window exists (0.3), the cell script exists and emits **per-phase CoV** (0.4), the anchor is re-sealed with the second incumbent arm and the per-phase bands 3.2/6.2/6.5 need (0.5), the roofline debt is paid (0.6), the uncommitted work is inside git (0.7), the harness asserts **both** its assertions (0.8), and the four closed sets are compile-error-to-change (0.9). **0.5 is an ancestor of every card whose kill quotes a CoV band, per-phase included; 0.5 and 0.6 are ancestors of 6.6 and 11.2.** Phase 1 forks off 0.7: the adjudication (1.1) releases two independent recovery strands (1.2, 1.3) feeding only Phase 4; **1.3 publishes `OPS_AFTER_PRUNE`**. Phase 2 hangs off 0.9 and 1.3 and closes brief item 1 **early**: 2.1 RED → 2.2 fix (three commits, four omega targets migrated) → 2.3 re-anchor. Phase 3 requires 2.3. **3.4 is where the census, the fingerprint and the census-sum all become gates.** Phase 4 requires 3.4 and both recovery strands. Phase 5 requires 4.3. **Phase 6 requires 5.2** — the capacity arena exists before anything buckets the leaf extent, so `found == expected` holds at every step. Within Phase 6, 6.1 → 6.2 → 6.3, with 6.4 reached **only** if 6.3's decisive kill fires; 6.5 requires plan stability from 6.3 or 6.4; 6.6 re-seals and names the `greedy_pick` residual. Phase 7 needs 3.4 for the config card and 6.6 for the width card. Phase 9 requires 7.2, and **9.2 re-derives 6.5's arena partition** because it collapses the output set from ~97 to ~1 **and removes the `lookup.indices` edge from `last_use`**; 9.2 and 9.3 both re-capture goldens and fingerprints for their ON arms. Phase 10's sweeps require 9.3 and precede the board; 10.3/10.4 hang off Phase 0 alone. **Phase 8 hangs off 3.4 but is SCHEDULED AFTER 11.2** — it is off every perf path, produces no measured payoff by its own words, and every command in it takes the single mutex. Phase 11 requires the terminal card of every landing phase except Phase 8.

```
0.1 mutex ─ 0.2 byte counters + uniform cache ─ 0.3 device window ─┬─ 0.4 gpu-cell.sh (per-phase CoV) ─ 0.5 re-seal + -fa 1 + bands ─┬─ 0.8 both assertions
                                                                   └─ 0.6 roofline ───────────────────────────────────────────────┴─ 0.9 cardinality+ai_docs
0.1 ─ 0.7 quarantine ─ 1.1 adjudicate ─┬─ 1.2 mask-fma ──────────┐
                                        └─ 1.3 picks (OPS_AFTER_PRUNE) ─┐
0.9 + 1.3 ─ 2.1 fingerprint RED ─ 2.2 bind takes the set (3 commits) ─ 2.3 re-anchor
                                        └─ 3.1 Route ─ 3.2 census ─ 3.3 backends ─ 3.4 gates ─┤
                                                                                              ├─ 4.1 selector ─ 4.2 bake-off ─ 4.3 land
3.4 ─ 7.1 geometry config (policy only) ──────────────────────────────────────────────────────┤        │
                                                                                    5.1 spans ─┴─ 5.2 KV arena (+ forward_node_values)
                                                                                              6.1 bucket+mask (+2nd assembly site)
                                                                                              6.2 invert BOTH assertions
                                                                                              6.3 trade cell (measures δ_b) ─┬─(kill)─ 6.4 invariant plan
                                                                                                                             └─ 6.5 arena + plan-owned uniforms ─ 6.6 re-seal
                                                                                    7.1 + 6.6 ─ 7.2 wide reduce
                                                                                              9.1 affine write offset (+autograd gate) ─ 9.2 dynamic base ─ 9.3 single-range
                                                                                              10.1 sweeps ─ 10.2 split-K (entry-gated, deletable)
0.6 ─ 10.3 torch-MPS ;  0.9 ─ 10.4 ORT-CoreML (terminal, exclusive on the mutex)
0.9 + 1.1 + 3.4 + 0.6 + 10.1 ─ 11.1 docs ─ 11.2 FINAL BOARD ← 9.3, 10.1, 10.2, 10.3, 10.4, 6.6
                             11.2 ─ 8.1 dialect map ─ 8.2 CUDA kinds ─ 8.3 core:Elementwise   (maintenance, after the board)
```

**The true longest chain**: `0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.9 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 3.4 → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 → 6.1 → 6.2 → 6.3 → 6.5 → 6.6 → 7.2 → 9.1 → 9.2 → 9.3 → 10.1 → 10.2 → 11.2` = **30 cards**. Every edge is a real `depends_on`. **The caveat the path length hides:** G3 serialises every measuring card on one mutex, so the schedule length is the count of measuring cards (**33 of 44**), not the path length. Moving Phase 8 behind the board removes three mutex-taking cards from the pre-board schedule. There is no parallel phase; non-measuring cards (0.7, 1.1, 8.1, 11.1) may proceed only while no build runs on the box, and 10.4 excludes every other card for its duration.

---

### 5.4.2 Rollback map

| card | rollback | main's default affected | firewall |
|---|---|---|---|
| 0.1 | `git revert`; `rm` the lock | no | the two-process contention test, exit 75 |
| 0.2 | `git revert` | instrument only | the partition check **stated as vacuous-by-construction**; `BLOCK_COPIED_BYTES` 8–10 MB/token; the per-op-codec family table |
| 0.3, 0.4 | `git revert` | instrument / scripts only | `gpu_device_ms <= gpu_exec_ms`; the per-phase CoV bands |
| 0.5 | none (measurement) | no | a memory breach on unmodified main is RED and stops the plan — **at the corrected caps, which R13's own prefill peak no longer breaches** |
| 0.6 | `git revert` | example + one doc | `READBACK_BYTES` delta == 4N; two denominators; saturation |
| 0.7 | `git revert` — **patches and untracked contents remain in git history** | docs only | per-worktree HEAD; ten apply-checks; ten `--stat` vs R7 |
| 0.8, 0.9 | `git revert` | test-only / doc+tests+JSONL | **both** assertions still fire on an injected hit; **four** exhaustive matches fail to compile on any variant change |
| 1.1 | a ruling; reversal requires new evidence in its own row | no | the 42-commit accounting; `physical.rs` +576 / `bind.rs` +666 / `discipline.md` +938 |
| 1.2 | `worktree remove --force` + `branch -D` | no | oracle + parity; nothing depends on it until 4.1 |
| 1.3 | per-commit `git revert` | yes (generic passes land in `default`) | zero `discipline.md` hunks per pick; `gpu_exec` move < 0.2 ms; `device_allocated_bytes` must not rise |
| 2.1 | `git revert` | test + one pure fn + two `#[cfg]` accessors | **the RED is the expected state**; the field list is fixed here |
| 2.2 | **revert commit 3, then 2 (restores `metal.rs:1013`), then 1 (restores `lib.rs:244`'s re-export AND the four omega target call sites, which moved together in commit 1)** | yes — `bind`'s signature, three crates, two omega examples, one omega test | 2.1 returns to its **documented** RED; golden byte-identity; six parity suites; all three gates |
| 2.3, 6.6, 11.2 | n/a (measurement); features demote one line each + a rebuild | yes | the memory gate is a board-level kill |
| 3.1 | `git revert` — multi-commit unwind once 8.3 lands; revert 8.3 first | yes (`route::of` is not gated) | `kernel_cache_key` byte-stability **and entry-name stability**, plus golden emitted source |
| 3.2, 3.3, 3.4 | `git revert` | instrument + emit entry points / gates | the sum identity; the two greps at zero; the budget vs `max(5%, 2×CoV)` |
| 4.1 | revert the `build.rs`+toml hunks + CI matrix | no | both values pass the **full six-step** gate including both clippy arms |
| 4.2, 4.3 | **flip one toml line AND rebuild omega + downstream** — stated as a rebuild, not a flip | the default body | parity gate first; route-count pin; batched primary; terminal tie-break; loser kept selectable |
| 5.1 | `git revert` | yes (slot 0 is the checkpoint) | `mapping_offset_uploads` unchanged at 291 |
| 5.2 | feature off + rebuild; `git revert` | no while gated | **`found == expected` by construction**; **both** assembly sites re-typed; the pre-allocation print; both KV slopes → 0. **The build-time byte assertion is NOT rolled back** (§15) |
| 6.1, 6.2 | feature off **plus a rebuild** | no while gated | `#[cfg]`-paired **pairs** of assertions keep both arms green; feature-OFF hash identity; 0-ULP CPU parity; `op_count` delta in {1,2,33,34} |
| 6.3 | none (measurement) | no | **the recorded loss, if it lost, is not softened**; δ_b is carried, not capped |
| 6.4 | feature off + rebuild | no | reached only if 6.3's decisive kill fired |
| 6.5 | feature default-off + rebuild; `git revert` — **except the `UNIFORM_BUFFERS` capacity bound, which is a leak repair and stays** | no while gated | whole-buffer sharing; outputs pinned; the retire loop still fires; **plan-owned uniforms distinct from the content cache**; the peak printed before allocating |
| 7.1 | revert build.rs + toml + const sites together | yes, value-identical by construction | per-key equality; env-override-changes-MSL; **`[4/6]`-arm-1 alloc clippy green**, the arm a feature-gated key would have broken |
| 7.2 | **one toml integer (`max_threads = 32`) + a rebuild** | no while gated | `metal_parity`/`backend_parity`; the op count must not move |
| 8.1 | docs revert | no | the 78-row classification **with measured line counts** is the gate on 8.3 |
| 8.2 | revert the variant-deletion commit, then the implementation | yes | two commits, not one |
| 8.3 | `git revert` one commit | yes, byte-identical by construction | 3.1's golden; the continuation gate is a deleted-duplicate count, not a `wc -l` threshold |
| 9.1 | `git revert` — **and, once 9.2 has landed, only after unwinding `dynamic_bases` from `Plan` and restoring the 97-root output set** | yes — `shape.rs`+`bind.rs`, cross-backend, **plus proxima-autograd as a consumer** | the REJECT half of the case table incl. overlapping ranges; fingerprint + golden byte-identity for every `offset == 0` `Reduce`; **`proxima-autograd-gate.sh` green** |
| 9.2 | feature off + rebuild | no while gated | intra-encoder ordering test; `plan_hits` must not fall; `UNIFORM_CACHE_LEN` must not grow; the arena peak re-asserted |
| 9.3 | feature off + rebuild; the two-range body stays under `#[cfg(not(...))]` | no while gated | wall is in the kill; split-half RoPE explicitly out of scope; Qwen3.5 re-parity-tested; both hash sets re-captured |
| 10.1, 10.2 | one/two toml integers + a rebuild; default-off feature | build config only | negatives rowed with all cells; 100× determinism for split-K |
| 10.3, 10.4 | revert the flag / the timed arm; venvs gitignored | no Rust hot path | device and provider assertions; partition counts; fidelity fields; **the CI `--quick` baseline as the cross-check** |
| 11.1 | docs revert; **a wrong row is corrected in place with a dated note, never deleted** | no | row monotonicity; `jq` parses four files |

**Ordering rule.** Rolling back a card requires rolling back everything downstream of it in §VI first, in reverse topological order. Two exceptions are designed in: 4.2/4.3 (a config value plus a rebuild, so the graph stays intact) and 7.2/10.1 (toml integers plus a rebuild). **The 6.1↔5.2 pair has an explicit order**: 6.1 may be turned off while 5.2 stays on (the block reverts to a `cached_len`-sized slice of the same arena and `found == expected` still holds), but **5.2 may not be turned off while 6.1 is on**. **The 9.1↔9.2 pair has an explicit order**: 9.2 must be off before 9.1 is reverted. Every landing commit is a green bisect point; primitives land before callers (2.2 before 3.1; 3.1 before 8.3; 7.1 before 7.2 and 10.1; 5.2 before 6.1; 6.3/6.4 before 6.5; 9.1 before 9.2 before 9.3). **No commit lands without owner authorization, and no worktree is created without it either.**

---

### 5.4.3 Abandoned designs (each traced to the constraint that ruled it out)

1. **`BoundOpKind::CachedAttention`** — a fifth bound kind carrying an eight-input fused online-softmax macro-op with a post-bind structural matcher and `physical.rs` (+576). *Ruled out by* AGENTS.md's invariant against arbitrary rules for specific instances against a closed 4-variant set; §1's binary question (the expression exists at `proxima-tensor/src/spec.rs:2596-2720`); its own measurement (R12: 51.535 ON vs 51.571 OFF with `gpu_exec` **+4.7 ms worse**); the matcher's per-token cost (ROW 247: `prepare` 150.7 → 11.6 ms/token); and the coverage burden it would place on CUDA/WGSL, which already fail coverage. *What survives:* `prune_dead`, the consumer index, the paired Q4_K body, ROW 263 as 3.2's third witness. *Re-open condition:* 9.3 measuring ≤23 ops/layer unreachable.

2. **`Op::Concat` / `Op::Pad` / `Op::Tile` / `PlacedBuffer` / `write_placement`** — zero hits on main, none added. *Ruled out by* §1 + §6: `Layout.base` exists on every bound `Reduce` and is already emitted as `long out_base` and consumed at five sites, and `layout_of` already folds a write-map offset into it.

3. **A `Computed`-out_map injectivity prover walking the indices chain backwards, extending `project_output_shape`.** *Ruled out by* `proxima-tensor/src/shape.rs:183-201`, read verbatim: a data-dependent out_map short-circuits to `scatter_output_shape` at `:200`, so `project_output_shape` (`:205`, body `out_map.affine()` at `:474`) **is never reached for the case that rule governed**. *What changed:* 9.1 accepts a static nonzero `axis.offset` on an **Affine** write map, where injectivity is an interval-overlap check rather than a prover, and where the offset already folds into `out_layout.base`.

4. **A GPU scatter emitter, with injectivity by a leaf-name convention (`"*.write_row"`).** *Ruled out by* `grep -rn write_row` finding nothing enforceable and, decisively, by a host-fed `Op::Input` indices leaf being **unprovable at bind in principle** — the census would record the name-check's answer rather than the property, the same defect `classify_kind` embodies. *What changed:* the KV write never becomes a scatter at all.

5. **An atomics-based GPU scatter.** *Ruled out by* §21 and by `proxima-tensor/src/map.rs:110-131`'s own reasoning: the CPU needs no atomics only because its loop is sequential. The one case that matters is affine and needs none.

6. **Repurposing `AxisIndex.offset` on a `Computed` write map.** *Ruled out by* `map.rs:110-131` + `map.rs:175-197`: for a scatter, `base`'s entry at `gathered_dim` **already carries the destination extent**, and `bind::build_scatter_out_layout` skips that axis for exactly that reason. 9.1 stays on `Affine`, where `map.rs` itself calls the offset "otherwise-always-`0`" — no collision.

7. **A parallel `&[Option<u64>] declared_capacity` slice threaded into both validators, and a new `QuantizedBlock` field.** *Ruled out by* three findings: the predicate degenerates to "accept any length"; the slice must be built in **node** order while `named_blocks` is built in **name** order; and its unit was never stated against `AlignedBuffer`'s contract. *What changed:* residency-before-bucketing makes the handed block always exactly the declared extent, so neither validator is edited.

8. **Bucketing before residency.** *Ruled out by* the code-level cycle: setting `symbols[1]` to the bucket while `LayerCache` still hands a `cached_len`-sized `Vec` returns `InputSizeMismatch` on the first token on both backends.

9. **Two mutually-exclusive cargo features for the two Q4_K bodies.** *Ruled out by* `scripts/omega-gate.sh` read in full: `compile_error!`-guarded exclusivity breaks `[2/6]`, `[3/6]`, `[4/6]`-arm-2, `[5/6]`-arm-2 **and** `[6/6]` — **five of six steps**, not one. *What changed:* a build-time profile axis (`[q4k] body` → `rustc-cfg`), with the loser kept selectable.

10. **Adding `omega_q4k_body` as an axis on `proxima-build`'s `Profile`.** *Ruled out by* `proxima-build/src/profile.rs:50` and `src/lib.rs:75-96`: the axes are a fixed **workspace-runtime** table (`proxima_alloc`, `proxima_std`, `proxima_executor`, …) that every consumer declares; a single crate's kernel-body choice does not belong there, and omega does not depend on the crate. *What changed:* 4.1 reuses `proxima-build`'s exact directive form in `omega/build.rs` beside `emit_sizing_consts`, and the row names `proxima-build` and this reason so the second mechanism is not unexplained.

11. **Moving `packed_operands_of` into proxima-tensor.** *Ruled out by* types: it returns `PackedOperands` = `BTreeMap<NodeId, PackedCodec>`, both **omega** types (`omega/src/msl.rs:656`, `:591`), which proxima-tensor cannot name; and it is unnecessary, since `correct_packed_matmul_layouts` already takes a bare `&BTreeSet<NodeId>` (`proxima-tensor/src/bind.rs:1648`). *What changed:* 2.2 is a signature change on `bind`, not a relocation.

12. **A `Mutex<BTreeMap>` route census mirroring `WIDTH_TILE_DECLINE`.** *Ruled out by* §21 and arithmetic: 1196 lock/unlock + BTreeMap lookups per token sit **inside** the slice Phase 6 measures, guarded only by a device-adjacent window. *What changed:* a per-plan route table plus a fixed-size atomic array, with a measured budget against a measured band.

13. **`route as usize` over a data-carrying `Declined(reason)`, and a hot counter slot for `Declined`.** *Ruled out by* E0605 and by a declined op never dispatching (its slot would be structurally 0, making the sum identity vacuous). *What changed:* a unit-only `Route` with `fn slot(&self)`, `[Counter; 8]` with eight explicit non-`Copy` initializers, declines cold-only.

14. **A single `u64` plan fingerprint.** *Ruled out by* three requirements needing node identities. *What changed:* a per-op `Vec<u64>` plus `pub` accessors.

15. **Writing per-position uniforms in place through `UNIFORM_BUFFERS`.** *Ruled out by* `omega/src/metal.rs:2056-2065` read verbatim: the map is keyed by uniform **bytes** and shared **across ops** ("two ops with identical uniform bytes want identical contents by definition"), so an in-place write corrupts every co-keyed op, and a per-token value **misses and inserts** rather than reusing. *What changed:* 6.5 adds a plan-owned uniform buffer per position that bypasses the cache, bounds the cache, and predicts `UNIFORM_BUFFER_REUSES` **falling**; 9.2 patches the dynamic base into that plan-owned buffer.

16. **A synthetic 40 MiB "activations+uniforms" term in the device cap.** *Ruled out by* arithmetic: 22.6 MB × 1.85 = 41.81 MB ≠ 41,943,040 B (= 40.00 MiB); the multiplier was the constant's quotient rounded, the endpoint choice was unstated, and the resulting cap of 4,182,360,064 **fired on unmodified main**, whose prefill peak R13 measured at 4.299–4.305 GB. *What changed:* G8's caps are the MEASURED prefill and steady peaks with a labelled CHOSEN headroom, split into 3a/3b so prefill is inside the peak clause.

17. **Landing the second-rewrite fix late, behind the emitter reorganisation.** *Ruled out by* brief item 1 staying known-false for thirty cards while every downstream claim rested on it, and by blast radius being cheapest when nothing else is in flight. *What changed:* it is Phase 2.

18. **Threading `op_setup` / the encode loop.** *Ruled out by* §21, R3/M9 (non-`Send` `MTLBuffer`) and R4 ("thread count explains zero of the gap"). *What changed:* 6.5 removes the work instead of distributing it.

19. **Headlining the dispatch-count reduction as the spine.** *Ruled out by* R12's control (1194 → 616 moved wall 0.07% and moved GPU **up** 13.5%) plus R13's per-op table. *What changed:* Phase 9 is scheduled after Phase 7, for the RISC.

20. **A kernel-fusion engine as the route to parity.** *Ruled out by* R8: `grep -rln fuse ggml/src` is empty at `b25346221`.

21. **A spec-sheet GPU bandwidth figure to close the roofline debt.** *Ruled out by* §18 and by `rooflines.md:411` refusing it once already.

22. **A second marker string to fix the classifier mislabel.** *Ruled out by* "find where information is destroyed": another substring is more of the mechanism that caused two proven relabels.

23. **`wc -l` as the emitter-core continuation gate.** *Ruled out by* measurement: the STRUCTURE functions for `Elementwise` across all three backends total **423 lines** against a ≥600 threshold, so the gate could never fire; and `wc -l` measures relocation, not duplication removed. *What changed:* 8.1 measures the line counts, and 8.3's gate is byte-identity plus a deleted-duplicate count against that measurement.

24. **Claiming `omega/benches/metal_vs_cpu.rs` is unrun.** *Ruled out by* `.github/workflows/proxima-tensor.yml:191-205`, a dedicated `omega-compare-bench` job whose comment says it exists so the bench "can never again sit registered-but-unexecuted". *What changed:* 10.3 produces the first **CoV-bearing, mutex-serialised, non-`--quick` local** cell and compares it to the CI baseline.

25. **Asserting the `+3` → `+4` `Vec::with_capacity` literal at `generate.rs:1316`.** *Ruled out by* reading the site: four fixed blocks are already pushed against a `+3` hint, so it under-counts today and would still under-count after; a capacity hint has no functional consequence and no observable. *What changed:* 6.1 asserts `named_blocks.len() == block_node_ids(program).len()` at **both** assembly sites instead.

26. **`BoundOp.extents` symbolic up front (`Vec<Extent>`).** *Parked, not deleted*, by blast radius: `grid_threads`, `kernel_cache_key` and `kernel_dispatch_shape` all read extents. *Claim it gates:* a 100% plan-hit rate at any context length with zero padding cost. *Un-park condition:* 6.3 killing bucketing at every bucket size (→ 6.4), or 9.2's dynamic-base mechanism proving insufficient.

27. **On-device argmax, to overlap CPU and GPU around `greedy_pick`.** *Parked, and named here because the previous round neither built it nor abandoned it* [crit SD-D]. R3/M7 measured the mechanism: `greedy_pick`'s argmax depends on `waitUntilCompleted`, a **true data dependency**, so threading cannot help and the fix is to move argmax on-device. R13 leaves **~1.6 ms/token** unexplained (`wall − gpu_exec` = 11.0 against Σ phases ≈ 9.4), and a live counter exists at `proxima-model-interop/src/generate.rs:1640-1670`. *Why parked rather than built:* moving argmax on-device is a new reduce route and a new readback contract, and at 1.6 ms it is smaller than every card in Phases 4–7. *Un-park condition:* 6.6 measuring `greedy_pick_ms` **≥ 1.0 ms** — the number the ladder has been carrying as a constant now gets read, and if it is that large it becomes the next work item after 11.2.

28. **The full three-backend emitter reorganisation as a mandatory, pre-board phase.** *Ruled out by* the absence of any measured payoff, the widest blast radius in the plan, and the mutex: three cards' worth of gate runs ahead of the board buy nothing measurable. *What changed:* Phase 8 lands the classification, closes CUDA's coverage holes, proves the core on one kind with byte-identical emission, and is **scheduled after 11.2**, which no longer depends on it.

---

### 5.4.4 Open questions, each resolved by measurement (never by asking)

| # | question | card | the number that answers it | pre-registered answer | what a miss means |
|---|---|---|---|---|---|
| Q1 | Are the profiler's bytes wrong in a third place, and does the fix reach the path that produces every family number? | 0.2 | the partition check in **both** loops; the 8-family table recomputed per-op-codec | `BLOCK_COPIED_BYTES` 8–10 MB/token, not 4.15 GB | >5% family disagreement ⇒ the derivation is wrong and **no GB/s row may be written anywhere** |
| Q2 | Is `gpu_exec` real kernel time or partly wakeup latency? | 0.3 | `gpu_exec_ms − gpu_device_ms` | ≤ 1.0 ms/token | a larger gap relocates mass to orchestration and **rewrites §I.1** |
| Q3 | **Is the uniform buffer cache growing every token on main?** | 0.2, 6.5 | `UNIFORM_CACHE_LEN` per step, unmodified | **grows by ≈ `op_count`/token**, because every `Uniforms` carries `reduction_total` | flat ⇒ D6 is wrong and 6.5/9.2 are re-scoped in their own rows |
| Q4 | Does the box move R13's means or only its CoV — and does memory move? | 0.5 | wall, gpu, **per-phase CoV**, both memory slopes | means reproduce, CoV tightens | a memory breach on unmodified main is RED and stops the plan |
| Q5 | Is `-fa 1` a stronger incumbent, making 3.88x an understatement? | 0.5 | ms/token `-fa 0` vs `-fa 1`, interleaved | **`-fa 1` is faster**; every later ratio re-bases | `-fa 1` slower ⇒ `-fa 0` is their design point and R13's ratio stands |
| Q6 | Is 228.9 GB/s the machine's ceiling or the incumbent's achieved rate? | 0.6 | copy-arm `traffic_gbs` at saturation, device window, 21 runs | **exceeds 228.9** | below 228.9 ⇒ the reduce probe never measured bandwidth; 228.9 is the ceiling by default |
| Q7 | Is there really ONE bound plan across the executors? | 2.1, 2.2, 3.4 | the two per-op fingerprint vectors and the differing node set | differs by **exactly** `packed_operands`; green after 2.2 | a node outside that set ⇒ a **third** rewrite, outranking every performance card |
| Q8 | Was `classify_kind` lying about the route distribution, and what does the census cost? | 3.2, 3.4 | census counts vs 225/385/547/37/2; the sum vs `ENCODE_DISPATCH_CALLS`; `encode_dispatch_ms` ON−OFF | exact match; Δ ≤ `max(0.0235 ms, 2×CoV)` | a mismatch ⇒ **every R13 bucket is restated against routes** |
| Q9 | Are mask-fma and pair-dot the same mechanism at the same speed? | 4.2 | batched `gpu_exec_ms` vs pooled CoV with the route count pinned; then per-op family ms deflated; then parity; then emitted bytes | one body wins at both shapes | a shape split makes the body a per-route selection — a new card, not a tie-break |
| Q10 | Do the 2026-09-02 numbers survive a rebase onto nine commits? | 1.2, 1.3, 4.2, 7.2 | re-earned against 0.5's cell, never carried | mask-fma ≥20% at ffn; wide-reduce ≥10% | R7's numbers did not survive; record the negatives |
| Q11 | Does KV residency move `gpu_exec` at all? | 5.2 | `gpu_exec_ms` before/after, CoV band | unchanged; only `block_upload` moves 2.0 → ≤0.5 | a `gpu_exec` move means the card changed GPU work |
| Q12 | Does the tail mask fuse into the existing `ComposedBody`? | 6.1 | `op_count` delta ∈ {1,2,33,34} | **+1** (both fuse; the leaf is not a bound op) | +2/+33 localise which fusion failed; +34 refutes both |
| Q13 | Does bucketing pay at this budget, and where is the optimum? | 6.3 | `Δ(kv_cache.* gpu_ms)` vs `Δ(prepare + op_setup)` at {8,32,64,256} | net loss at 256, net win at 32–64; **δ_b measured and carried to the board** | a loss at **every** bucket ⇒ bucketing is dead and **6.4 is unparked** |
| Q14 | Does removing 1196 allocations remove the 3.9 ms, and what does the arena cost in bytes? | 6.5 | `op_setup_ms` vs 3.90, `encode_dispatch_ms` vs 0.47, wall, `ARENA_PEAK_BYTES` vs **172,812,125** | op_setup → [0.4,0.8]; peak inside the cap | op_setup falls but wall does not ⇒ orchestration overlaps GPU; **stop Phase 6** |
| Q15 | Is the affine write offset sound, and is its adjoint already right? | 9.1 | the REJECT half of the case table incl. overlapping ranges; the autograd round-trip; `proxima-autograd-gate.sh` | every REJECT rejected; the adjoint needs no autograd edit (`adjoint.rs:835` reuses the write map as a read map) | a false ACCEPT is two producers racing — a correctness defect that kills the card. An autograd edit being needed is a finding that grows the card |
| Q16 | Does the per-token write base defeat plan reuse or leak uniforms? | 9.2 | `plan_hits` against 6.2's formula **and `UNIFORM_CACHE_LEN`** | both unchanged | `plan_hits` drop ⇒ the base leaked into the key; cache growth ⇒ the patch went through `upload_uniforms` |
| Q17 | Does halving the dispatch count buy wall on **our** stack? | 9.3 | `encode_dispatch_calls`, `gpu_exec_ms` and **`step_wall_ms` jointly** | **< 1 ms change**, matching R12's refutation | a large win contradicts R12's control and **both** cells are re-run before either is believed |
| Q18 | Is the elementwise bucket concentrated or uniform? | 3.4 | the top-5 nodes' share of the 6.85 ms batched-equivalent bucket | ≥50% | uniform ⇒ no single-node lever; the only lever is fewer nodes |
| Q19 | Is simdgroup starvation fixable with a config knob before a kernel? | 10.1, 10.2 | the 12-cell `rows_per_group` sweep; then 10.2's entry gate and its 12-cell `split_k` sweep | knob helps below 4; the gate may **delete** 10.2 | no knob beats 4 ⇒ do not build split-K |
| Q20 | Was the reduce width tuned for a graph that no longer exists? | 10.1 | the 18-row width sweep on the post-9.3 graph | optimum at `min(reduction_len/4, 1024)` | no value beats 32 ⇒ demote 7.2's feature permanently |
| Q21 | Do torch-MPS and ORT-CoreML beat us on any lane this repo measures? | 10.3, 10.4 | p50/p95/p99 per provider with fidelity fields and partition counts | MPS slower at batch 1; CoreML takes a **partial** partition | MPS faster ⇒ a real incumbent at that shape; one CoreML partition ⇒ a genuine whole-graph GPU arm for embedding |
| Q22 | **How large is the 1.6 ms residual, and is it `greedy_pick`?** | 6.6, 11.2 | `greedy_pick_ms` against `wall − gpu_exec − Σphases` | **≥ 1.0 ms** ⇒ §VIII.27 un-parks as the next work item | near zero ⇒ the residual is elsewhere and the row names where |
| Q23 | Can 8.3's continuation gate fire at all? | 8.1, 8.3 | 8.1's measured STRUCTURE line total (423 across three backends) vs the deleted-duplicate count | the gate is the count, not a threshold | a `wc -l`-only gate is a metric on relocation and is not used |
| Q24 | Does the composed stack equal the sum of its measured parts? | 11.2 | composed wall/gpu vs §IV's DERIVED sum at δ_b's endpoints | within CoV of the sum | worse than the best single card ⇒ **the cards interact**, and that is the next work item |
| Q25 | Did the memory rule hold on every card that ran a process? | every measuring card | MG-3's five clauses per steady step against G8's byte formula | held | any breach is a NEGATIVE that rolls the card back **even when it wins on time** |

---

### 5.4.5 Conflict resolutions

1. **The mutex, given `flock` is absent.** **B4/B3 win, mechanism unchanged from round 3 and re-verified.** `flock(1)` does not exist (`command -v flock` → exit 1) and `fcntl.flock` is present, so `scripts/gpu-measure-lock.sh` is card one. It lands in git, is re-provable under §16, mutates no host state, and carries a `--wait` bound exiting 75. The weld covers all three gate scripts.

2. **Residency-before-bucketing vs bucketing-first.** **Residency first.** Setting `symbols[1] = bucket` while `LayerCache` still hands a `cached_len`-sized `Vec` returns `InputSizeMismatch` on the first token on both backends, and the card that pads the buffer *depended on* the card that breaks. With 5.2 first, `element_count == block_element_count` holds at every stage and **neither validator is ever edited**. Round 4 adds the site both prior rounds missed: **`forward_node_values` (`proxima-model-interop/src/generate.rs:1804-1860`) has its own `LayerCache::new()` and its own `named_blocks` assembly**, so 5.2 re-types it and 6.1 pushes the new leaf there too, or `InputCountMismatch` fires on a public entry [crit RS-E].

3. **Placement: which `shape.rs` line, and which `IndexMap` variant.** **B4's framing wins on the destination (`Layout.base`, `long out_base`), and round 4 corrects the path.** Both prior plans routed the rule through a `Computed{indices}` out_map while citing `project_output_shape`. `shape.rs:183-201`, read verbatim this pass, short-circuits every data-dependent out_map to `scatter_output_shape` at `:200` — **`project_output_shape` is reached only for `Affine` maps**, so the cited line never governs the cited case [crit B4, RS-B]. The resolution is the simpler and more provable one: **a static nonzero `axis.offset` on an Affine write map**, accepted in `project_output_shape` as `iter_extents[axis] + offset`, mirroring `bounds_check`'s existing read-side fold at `:441-467`, already folded into `out_layout.base` by `layout_of` (`bind.rs:1594-1606`, reached from `proxima-tensor/src/bind.rs:962`), with injectivity as an interval-overlap check. This also disposes of RS-C: `map.rs:110-131`'s repurposing of `offset` to carry a destination extent is scoped to `Computed` at `gathered_dim` and calls the Affine case "otherwise-always-`0`", so there is no collision. And it disposes of RS-A's sharpest edge: `proxima-autograd/src/adjoint.rs:835` reuses the write map **as a read map**, so the adjoint of a placed write is a sliced read by construction — but the crate is still a consumer, `omega/Cargo.toml:196` depends on it, `scripts/proxima-autograd-gate.sh` exists, and 9.1 now runs it.

4. **Two Q4_K features vs a build-time profile axis.** **The profile axis wins, and B4's reading of the gate is adopted verbatim**: `compile_error!`-guarded exclusivity breaks **five of six** gate steps (`[2/6]`, `[3/6]`, `[4/6]`-arm-2, `[5/6]`-arm-2, `[6/6]`), not one [B4 adoption C-C]. Round 4 adds two things: the card **names `proxima-build`'s existing `emit_cfg_directives` and states why the axis is not added to the workspace `Profile`** [crit MS-D], and 4.2 **builds each arm once into its own target dir before measuring**, because a `rustc-cfg` arm switch is a rebuild of omega and everything downstream, and nine rebuilds interleaved between measurements is not an interleaved cell [crit RB-C].

5. **The two arenas.** **Both, because they are different objects — and round 4 adds a third correction.** The registered host span (5.1/5.2) is the **KV** arena. The device arena (6.5) is the **output/uniform** arena. Constraints: whole-buffer sharing only (sub-allocating makes `device_buffers.insert(node, (buffer, 0))` a lie against the readback invariant at `metal.rs:2360-2364`); outputs pinned by `bound_op_retirement`'s `!outputs.contains`; the retire loop not made a no-op. **The new one: `UNIFORM_BUFFERS` is a content-keyed dedup cache shared across ops, not a per-position buffer** (`metal.rs:2056-2065`, doc read verbatim), so 6.5 adds plan-owned uniform buffers that bypass it, bounds the cache, and predicts `UNIFORM_BUFFER_REUSES` **falling** rather than rising to `op_count`; 9.2 patches the dynamic base into the plan-owned buffer rather than through a cache whose key it would change every token [crit HC-A, B5, B6]. 9.2 also re-derives 6.5's partition, because the output set collapses 97 → ~1 **and** the `lookup.indices` edge leaves `last_use` (`metal.rs:1132-1137`) [crit HC-B].

6. **The board prediction's arithmetic.** **§IV is re-derived a second time.** Round 3's ladder was internally consistent but mixed measurement modes: it subtracted **per-op-mode** bucket figures, which R13 itself brands **+7.3% inflated**, from a **batched** `gpu_exec_ms`. Every bucket is now deflated by 1.073 first (44.450 → 41.43; 9.113 → 8.49), which moves 4.3 from `[44.0, 48.0]` to `[44.9, 48.6]` and the board to `[46.8, 57.3]` wall / `[43.5, 50.6]` gpu = **2.67–3.27x**. And `δ_b` is **carried at 6.3's own pre-registered endpoints `[0.3, 2.8]` rather than capped at ≤1.0**, which is what made the previous ladder's own successor contradict it [crit O-B]. The board's kill (59.0) still sits outside the band the cards' terms produce (57.3), so it cannot false-fire. Two terms are labelled **cross-tree** and are not called "MEASURED card deltas" until 4.2 and 1.3 re-earn them.

7. **The number of worktrees and cards.** **One worktree per phase-branch, cards strictly sequential: 13 worktrees, 44 cards** — unchanged, and re-verified collision-free (68 worktrees, none matching `risc`; `git branch --list 'risc/*'` = 0). Round 4 adds that **worktree creation is owner-authorized** [B4 adoption] and that repo-wide greps must be crate-scoped because a nested checkout lives inside the repo [B4 adoption].

8. **Where the second-rewrite fix lands, and what it actually moves.** **Early (Phase 2), and as a signature change rather than a relocation.** B4's C5.1 and round 3's 2.2 agreed on the finding and both got the mechanism wrong in the same way: `packed_operands_of` cannot descend, because its return type is an omega type over an omega enum, and it does not need to, because `correct_packed_matmul_layouts` already takes `&BTreeSet<NodeId>` [crit MS-A, MS-B, B2]. Round 4 also fixes the assertion: `grep -rn correct_packed_matmul_layouts` returns **22 hits in 9 files** on main, so "exactly 1 hit" is untrue of any reachable state; the reachable assertions are zero code hits outside proxima-tensor, no `pub fn`, and no re-export [crit B1]. And the blast list gains the four **omega target** call sites (`omega/tests/attn_multi_axis_tiled_gemm_parity.rs:223,312`; `omega/examples/{attention_tiled_gemm_probe.rs:121, real_forward_packed_probe.rs:124}`) that `[2/6] --all-targets --all-features` compiles, which is the gate the card ends with [crit RS-D].

9. **The census index mapping, the counter array, and `Declined`.** **Unchanged and re-verified:** `Route` is unit-only with an explicit `fn slot(&self)`; the hot array is `[Counter; 8]` with eight named const initializers (`Counter` holds an `AtomicU64` and is not `Copy`); `Declined` gets no hot slot, because a declined op never dispatches and a structurally-zero slot makes the sum identity vacuous. Round 4 adds that **`entry_name`'s signature stays `(&BoundOp)`** and the route must not enter the emitted name, because `kernel_cache_key` (`msl.rs:730-736`) starts from it and 3.1's N2 pins it byte-stable — 8.1's method list said `entry_name(&BoundOp, Route)` and nothing named the consequence [crit residual].

10. **How the harness is asserted, and by whom.** **B4's C-B wins outright.** There are **two** assertions (`proxima-model-interop/src/bind.rs:3052-3055` and `:3056-3059`), not one; R13 named only the first, round 3 inverted only the first, and inverting only the first leaves the second red under `kv-capacity-bucket`. 6.2 now `#[cfg]`-pairs **both**. B4's second half is adopted too: the test is `#[cfg(feature="metal")]` **and** `#[ignore]` and lives in proxima-model-interop, so **no gate runs it** — G9 states this and every card that touches it re-proves it by name with `--run-ignored all` [B4 adoption C-B, D7].

11. **Which numbers the memory gate is built from.** **Neither prior round's, because both were unsourced or self-breaking.** Round 3's cap contained an invented 40 MiB term whose stated derivation does not produce it, and the resulting cap fired on unmodified main's own measured prefill peak [crit B14, SD-A, B7]. B4's flat ceiling (`4.40e9`) clears R13's 4.299–4.305 GB prefill peak on main (4.40e9 > 4.305e9 — an earlier draft of this sentence had the inequality backwards [round-4 j11]) but has no KV term, so B4's own 2048-token arena (536,870,912 B) pushes 4.163e9 + 0.537e9 = 4.70e9 past it; a cap without a capacity term self-kills the card that builds the arena. G8 now uses **R13's MEASURED prefill peak and steady peak, split into clauses 3a and 3b, with a labelled CHOSEN 1.05 headroom**, plus `ARENA_TRANSIENT_CAP = 172,812,125 B` derived from those two measured numbers for 6.5 and 10.2. B4's `capacity % 8 == 0` observation and its `require_multiple_of_eight` pointer are adopted [B4 adoption].

12. **What Phase 8 costs and when it runs.** **It runs after the board.** It sits at depth 4 off 3.4, produces no measured payoff by its own words, and every command in it takes the single mutex — so scheduling it before 11.2 lengthens the serialized measuring schedule for nothing. 11.2's `depends_on` drops 8.3 [crit O-C]. And its continuation gate stops being an invented `wc -l` threshold that 8.1's own measurement shows to be unreachable by ~180 lines [crit B3, OB-C, SD-C].

---

### Critical Files for Implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` — the three byte defects `:694-700` / **`:465-468` (fires before the path match, so it counts *offered* bytes)** / the second upload loop `:663-690`; the batched host-tick window `:545-555` vs the device window `:733-734`; **`:2056-2078` the content-keyed, unbounded `UNIFORM_BUFFERS` + `UNIFORM_BUFFER_REUSES` `:2069`**; validators `:984-1000`; `bind` `:1003` then `correct_packed_matmul_layouts` `:1013`; `packed_operands_of` `:375-390`; private `Prepared` `:859-864`; `classify_kind` `:785-826` + call sites `:709-710`; `plan_named` `:578-585`; `encode_op` `:2179-2252` with `allocate_buffer` `:2210`, `upload_uniforms` `:2211`, `ENCODE_DISPATCH_CALLS` `:2243`, `insert` `:2249`; the retire loop `:541-543`; **`bound_op_retirement` `:1128-1147` and its `last_use` walk `:1132-1137`**; the readback invariant `:2355-2378` stated at `:2360-2364`; `READBACK_BYTES` `:1496`/`:2373` with `snapshot_and_reset` `:1583`; `register_checkpoint_mapping` `:1744-1815`; `NOCOPY_BUFFERS` `:1848-1892`; `mark_resident` `:350-362`; `page_size` `:1606`
- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs` — `emit` `:673-697`; **`kernel_cache_key` `:730-736` (`pub(crate)`, cfg-gated) and `entry_name` `:1690` (no `Route` parameter)**; `grid_threads` `:1517-1560` (cooperative arm `:1554`); `ScatterNotSupported` `:933`; `Uniforms` `:2207-2218` with `long out_base` `:2216`, consumed `:2361`; `push_packed_row_blocked_body` `:2452-2530` (`lanes_per_block` local `:2516`, step `:2527`); `push_cooperative_reduce_body` `:3140-3196` (`:3190`/`:3194` the 32-lane pin); **`PACKED_ROWS_PER_GROUP` `:1017` (policy) vs `TILE_DIM` `:1029-1030` and `TILED_GEMM_NSG` `:1046`, whose own docs declare them hardware facts**; codec block constants `:294-556`; the delimiter-free unpack concatenation `:1978-1982`; `PackedCodec` `:591` / `PackedOperands` `:656`; `emit_is_deterministic_byte_equal` `:4656`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/shape.rs` — **`infer_reduce`'s `is_data_dependent()` short-circuit `:183-201`, which routes every `Computed` out_map to `scatter_output_shape` `:495` and leaves `project_output_shape` `:469-485` reachable only for `Affine` maps** — the correction 9.1 rests on; `bounds_check` `:441-467`, whose first two lines fold `axis.offset` on the read side
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs` — the plan-cache key `:966` and `plans.clear()` `:973`; `build_position_inputs` `:799-827`, called `:1304-1309`; the **first** `named_blocks` assembly `:1313-1334` and the **second, in `forward_node_values` `:1804-1860`**; the KV loop `:1364-1389`; `symbols` `:1393`; the 97 roots `:1393-1400`; `LayerCache` `:621-654`; `cached_len +=` `:1559`; `phys_footprint_bytes` `:248`; `greedy_pick_started/ticks` `:1640-1670`; `token_breakdown` `:1657` / `token_breakdown_metal` `:1723-1764`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/bind.rs` — **not `proxima-tensor/src/bind.rs`**: the three harnesses `:3002` (BENCH), `:3084` (MILLI), the ORACLE; `#[ignore]` + `#[cfg(feature="metal")]` `:3001`; `PROXIMA_MAX_TOKENS` `:2719`; `forward_calls_taken` `:3051`; **both** assertions `:3052-3055` and `:3056-3059`; the greedy oracle `:2797-2803`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/bind.rs` — `BoundOp` `:200-215`; `BoundOpKind` `:221-264` (**four variants, no `Input`**); `Layout` `:95-98`; the Affine `out_layout = layout_of(...)` selection `:955-998`; `layout_of` `:1594-1606` (`base += offset * stride`); `build_scatter_out_layout` `:1011`; `correct_packed_matmul_layouts(&mut [BoundOp], &BTreeSet<NodeId>)` `:1648`; `pub fn bind` `:1718`

---


## 6. ai_docs records (card 0.9 appends these, card 11.1 attaches evidence; JSONL is the source of truth, this is the projection)

`ai_docs/index.jsonl` (kind 3 = Concept, 5 = Decision, 7 = Failure):
```json
{"id":"proxima.gpu.one_risc_plan","kind":5,"summary":"GPU parity through one RISC: the 2026-09-03 plan, evidence register, and Luna task cards for proxima-tensor -> omega -> Metal/wgpu/CUDA.","path":"docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md","read_when":["gpu","metal","omega","tensor-decode","kv-cache","q4k","emitter","route-census"],"source_paths":["docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md","proxima-tensor/docs/discipline.md","proxima-tensor/docs/rooflines.md","omega/src/msl.rs","omega/src/metal.rs","proxima-model-interop/src/generate.rs","proxima-tensor/src/spec.rs"],"relations":[{"idx":7,"target":"proxima.AGENTS.hot_path"},{"idx":7,"target":"proxima.invariants.performance"}]}
{"id":"proxima.gpu.incumbent_ggml_metal_b25346221","kind":3,"summary":"llama.cpp-Metal at checkout b25346221 is the deployed incumbent for quantized LLM decode: 23 real ops/layer, ~740 dispatches/token, no fusion, KV written in place by ggml_cpy into a view at a byte offset of a persistent device tensor, flash-attn off by default.","path":"docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md","read_when":["gpu","metal","incumbent","llama.cpp"],"source_paths":["scripts/llama_reference/run.sh"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
```

`ai_docs/task-routes.jsonl`:
```json
{"task":"gpu-lane","purpose":"Any change to omega emitters/drivers, the tensor decode graph, KV cache handling, or a GPU bench cell.","must_read":["ai_docs/AGENT.md","ai_docs/invariants.jsonl","docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md"],"then_read_if_relevant":["proxima-tensor/docs/rooflines.md","proxima-tensor/docs/rewrite-algebra.md","omega/src/backend.rs"],"queries":["jq -c 'select(any(.applies_to[]?; . == \"gpu\" or . == \"omega\" or . == \"kv-cache\"))' ai_docs/invariants.jsonl"],"done_when":["the card's row is filled with N and CoV","the route census reports every reduce's route and every decline reason with N>0","step_wall_ms on the real checkpoint is re-sealed interleaved against llama.cpp-Metal"]}
```

`ai_docs/invariants.jsonl` (each rule is what a reviewer greps for):
```json
{"id":"proxima.gpu.no_fifth_bound_kind","kind":5,"summary":"BoundOpKind stays at four variants (Elementwise, Reduce, Iota, Constant); fused shapes are produced by rewrite laws over these, never by a per-model macro-op variant.","rule":"A diff adding a BoundOpKind or Op variant to express a model-specific cluster is rejected; express it as Reduce/Elementwise with placement (out_layout.base) and, later, a rewrite-engine law. The 2026-09-03 adjudication of BoundOpKind::CachedAttention is the worked example.","applies_to":["gpu","omega","tensor","bind","new-component"],"evidence_required":["grep -n 'pub enum BoundOpKind' -A 45 proxima-tensor/src/bind.rs shows exactly four variants","the route census reports the cluster as ordinary reduces"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
{"id":"proxima.gpu.route_is_a_value","kind":5,"summary":"The kernel route a Reduce takes (Serial | CooperativeGeneric | PackedRowBlock | TiledGemm) is decided once as an enum value before any emitter runs and censused (NodeId, reason); no instrument may classify by substring of emitted source.","rule":"classify_kind reads the route value. Every decline carries a RouteDecline reason. The census asserts N>0 per route present in the real decode graph.","applies_to":["gpu","omega","emitter","instrument"],"evidence_required":["omega route census output for the real openchat decode step lists per-route counts summing to the reduce count (610 on main 4be2f3a)"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
{"id":"proxima.gpu.geometry_from_sizing_config","kind":5,"summary":"Every GPU kernel geometry constant (rows per group, lanes per block, cooperative reduce width, KV bucket, buffer-pool sizes) traces to omega-runtime.toml through build.rs; SIMD_WIDTH alone is a hardware fact.","rule":"No bare geometry const in omega/src/msl.rs, wgsl.rs, cuda.rs. A new tunable adds a toml key, a build.rs emit with rerun-if-env-changed, and a doc line on the generated const.","applies_to":["gpu","omega","no-magic-numbers"],"evidence_required":["grep -n 'const [A-Z_]* *: *u\\(64\\|32\\|size\\) *= *[0-9]' omega/src/msl.rs returns only codec block-size facts"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
{"id":"proxima.gpu.plan_stable_across_tokens","kind":5,"summary":"A decode step's plan key must not change every token: cached_len is bucketed to a build-time KV capacity so the plan cache hits on steady tokens.","rule":"plan_hits on steady decode tokens must be > 0 within every KV bucket; plan_misses per generated token must be <= 1/KV_BUCKET amortized. A change that makes cached_len a plan symbol again is rejected.","applies_to":["gpu","kv-cache","interop","decode"],"evidence_required":["token_breakdown_metal lines show plan_hits rising on steady steps"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
{"id":"proxima.gpu.kv_written_in_place","kind":5,"summary":"K/V for a new token are written by the graph into a persistent device buffer at base = cached_len * row_stride; the cache never round-trips through the host per token.","rule":"kv_cache_upload_bytes on steady tokens is 0; the K/V reduce's out_layout.base equals the position offset; the cache Input leaf aliases the same named persistent buffer.","applies_to":["gpu","kv-cache","placement","interop"],"evidence_required":["instrument counters: kv_cache_upload_bytes == 0 and copying_uploads == 0 for kv_cache.* blocks on steps >= 1"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
{"id":"proxima.gpu.dead_levers_2026_09","kind":7,"summary":"Measured negatives on the Metal decode lane, not to be re-proposed: two-simdgroup Q4_K geometry (4x), encoder churn, per-dispatch fixed cost as the gap, incumbent -t threads, blanket rematerialization, float4 accumulation in the Q4_K helper, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode.","rule":"A card proposing any of these must cite a NEW mechanism that the prior measurement did not cover, with the prior row number.","applies_to":["gpu","omega","q4k"],"evidence_required":["proxima-tensor/docs/discipline.md rows cited in docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md section 3.8"],"relations":[{"idx":7,"target":"proxima.gpu.one_risc_plan"}]}
```

## 7. Discipline-log row templates (numbers assigned at land time; main's last row is 233)

### 7.1 ROW 234 — the 2026-09-03 seal (sealed this session; card 0.5 replicates; card 11.1 lands the row)
```
## ROW 234 -- GPU decode re-sealed on 4be2f3a, interleaved against llama.cpp-Metal b25346221

**Box:** M1 Max 64 GiB; `uptime` before/after each cell: <...>; concurrent cargo/rustc: <...>.
**Arms (A B A B A B):** A = the llama-bench invocation in §5 card 0.5; B = the decode harness
invocation in §5 card 0.5 (both verbatim there).

| arm | ms/token | n | CoV | GB/s (3.9996 GB/token) | GMAC/s (7,110,402,048) |
| llama.cpp-Metal | | 3x5 | | | |
| ours step_wall_ms (steps 1-7) | | 3x7 | | | |
| ours gpu_exec_ms | | 3x7 | | | |
Per-phase (ours, mean of 3): prepare | block_upload | emit | pipeline_lookup | op_setup | encode_dispatch | readback | plan_hits/misses | op_count.
Per-op profile (step 3, diagnostic; Σgpu_ns vs batched gpu_exec_ms agreement %): bucket table + family table.
Degenerate control: constant/iota ns per op = <...> (the dispatch floor).
**Read:** what moved vs ROW 221/193 and why, phase by phase; generated_text byte-identity across cells.
**Re-prove:** the two commands in card 0.5; raw logs at the path the card names.
```

### 7.2 ROW 235 — the reconciliation record (cards 0.7, 1.1, 1.2, 1.3)
```
## ROW 235 -- lane reconciliation: what was uncommitted, what the parallel branch measured, what landed

Table of every diff in §3.7 with: branch, base, files, measured claim, disposition (landed as
ROW n | negative recorded | rejected by §8.x | superseded by card y). The parallel branch's
ROWs 234-267 are cited by their branch numbers with one line each; its negatives are copied
into §3.8. The `BoundOpKind::CachedAttention` adjudication (§8.1) is quoted in full.
```

### 7.3 Per-card row (every landed card)
```
## ROW <n> -- <card title>

**Card:** <id>. **Worktree/branch/commit:** <...>. **Feature:** <name>, default-off.
**Allocation budget (hot/setup/cold):** <...>.
**Predict (one rung ahead, written before running):** <...>. **Observed:** <...>. **Miss category + work item:** <...|none>.
| arm | cell | n | CoV | load before/after |
**Gates:** omega <N run/N passed> (features: <...>); proxima-tensor <N/N>; proxima-model-interop <N/N>; clippy <crates>; alloc-tier check of omega EXIT <..> (modules built: msl, wgsl, cuda).
**Parity:** <fixture>, relative-to-batch-peak <...> vs f32 oracle.
**Census:** route counts / decline reasons / phase counters relevant to this card, with N.
**Home-turf arm:** llama.cpp-Metal <ms/token> same cell; frequency-weighted read: <...>.
**Principles engaged and what each changed:** <§n: ...>. **Abandoned:** <...>.
**Re-prove:** <command>.
```
## 8. Forks adjudicated (tier: judge — decided here so no card has to)

Each fork is stated with the two shapes written out, the constraint that decides it, and what
the losing shape would have cost. A card that finds the decision wrong stops and reports the
line that contradicts it; it does not pick the other branch on its own.

### 8.1 `BoundOpKind::CachedAttention` (parallel branch) vs the four kinds

**Shape A (the branch):** a fifth bound kind carrying the eight attention operands inline; a
post-bind structural matcher (`cached_attention_candidates`, `is_exact_causal_mask`,
`removable_attention_dependencies`) recognises the two-range online-softmax cluster; a bespoke
CPU arm (`cpu.rs:19141-19190` on the branch) and a bespoke MSL kernel
(`render_cached_attention`) execute it. Measured: 1194 -> 616 dispatches, wall unchanged
(51.535 vs 51.571), GPU +13% (39.841 vs 35.117) (§3.7, ROW 262 vs 267). `prepare` rose to
150.7 ms with the matcher, 11.6 ms after indexing (ROW 247/248), and is still paid every token.

**Shape B (this plan):** no new kind. The duplication is deleted at its source (§4.1 items 3-4:
one contiguous K/V range in a persistent, in-place-written cache), after which attention is
`Reduce(scores) -> [softmax as Reduce(max) + Elementwise + Reduce(sum) + Elementwise] ->
Reduce(V·p)` — five ops of the existing kinds; the incumbent runs the same five (two matmuls,
one fused softmax kernel that our rewrite engine's Law 2 softmax instance would reach later).

**Decision: B. A is not merged.** Constraints: workspace AGENTS.md "do not add arbitrary
rules/code for specific instances"; the central-claim lint ("everything is Elementwise or
Reduce" — enumerate what is not: `CachedAttention` is not, and its justification paragraph in
`failure-cached-attention-matcher.md` is the finding); guiding-principles §1 (can the existing
primitive express it? — yes, once placement exists, five ops do); the measurement itself
(halving dispatches moved wall 0%: the macro-op removed COUNT, not WORK, and added GPU time).
What A would cost if kept: every new model shape needs a new matcher and a new kernel pair;
the CPU and Metal arms of the macro-op are two more hand-written kernels to keep at parity;
the matcher is a per-token CPU cost as long as P4 stands.

**What IS taken from the branch:** `prune_dead`/`dead_resolved_nodes` (216d925, generic,
RISC-conformant); the paired-nibble Q4_K body as a candidate in §8.2; the 12 measured
negatives (into §3.8); the exact-prompt llama-cli incumbent protocol (ROW 250); the
`omega/tests/q4k_real_checkpoint_parity.rs` fixture; the classifier fix ONLY as evidence for
§4.1 item 5 (the fix adds a second substring; the plan replaces substrings with the route
enum). Card 1.3 cherry-picks exactly these, one commit each (`--no-commit`, `discipline.md` hunk dropped), and records the adjudication as
a `kind=5` ai_docs record and a log row so the branch's author sees why.

### 8.2 Which Q4_K body: `metal-q4k-mask-fma` (uncommitted, 2b95210) vs `q4k_pair_dot` (branch)

Both re-derive the incumbent's `kernel_mul_mv_q4_K_f32_impl` inner loop: mask without shift,
fold 1/16 and 1/256 into the sub-block scale, branch-free scale/min. mask-fma: -36% on ffn,
-17.2% gpu_exec, parity 2.46e-6 vs 2.58e-6 baseline (MEMORY, N=1 then stacked N=3).
pair_dot: family -29%, parity 3.1e-6 (READ, N=3), plus the lane->byte remap that kills the
2x redundant load (which mask-fma explicitly skipped; the uncommitted `metal-q4k-single-fetch`
does the same remap separately, perf unmeasured).

**Decision (round 0), amended by §8.13 after round 3:** ONE body lands. The two existing
re-derivations are recovered onto today's main as separate commits (mask-fma by card 1.2,
pair-dot by card 1.3), made mutually exclusive by the build-time `[q4k] body` profile axis
(card 4.1, not two cargo features — `omega-gate.sh` runs `--all-features`), and baked off in
one interleaved cell with batched `gpu_exec_ms` as the primary metric and a seven-rung
terminal tie-break (card 4.2); the loser stays selectable and is a recorded row (card 4.3).
A third body transcribed line-for-line from `ggml-metal.metal:5086-5193` is NOT an arm in
this plan: it would be a third re-derivation of a mechanism that already exists twice (§1),
and it re-opens only if 4.2's kill fires (neither body clears −10% on the packed-row-blocked
bucket beyond both CoV bands), at which point the port lands with an attribution header per
§10. Constraint: guiding-principles §14 (parity vs `cpu::evaluate` on the real tensor is the
first rung of the tie-break, ≤ 1e-4 or the body is out at any speed) and §1.

### 8.3 `cached_len`: capacity bucket + mask vs runtime-uniform extent

**Bucket (chosen):** the cache leaf's extent is `Static(C)` for the plan, `C = ceil(cached_len
/ KV_BUCKET) * KV_BUCKET`; positions `>= cached_len` are masked with the bodies that exist —
there is no `GreaterEqual` and no `Less` in the 17-body `ScalarOp` (`op.rs:60-78`) — as
`is_valid = Greater(kv_valid_len, cache_slot)` (1.0 where `slot < cached_len`, with
`kv_valid_len` one rank-0 `Op::Input` leaf for the whole program and `cache_slot` an
`Iota{Symbolic(1)}`) then `Select(is_valid, score, -inf)`, mirroring `causal_mask`'s
consumption at `spec.rs:2604-2615` argument-for-argument. Plan key `(new_count, C)`. Cost: the
score reduce and softmax run over `C` instead of `cached_len` (≤ `KV_BUCKET-1` wasted columns
per head); one `Select` per layer, expected to fuse into the existing `ComposedBody` (card 6.1
pre-registers `op_count` +2, and +34 as the finding that it did not fuse).
**Uniform extent (rejected for now):** `BoundOp.extents` become runtime uniforms for the cache
axis; zero waste; but it touches `bind`, both drivers' uniform packing, `grid_threads`, and
every `extents`-reading kernel body — blast radius the whole emitter. Constraint: §1 reuse
first (the mask composition is already in the algebra), and the incumbent's own choice
(padding, `ggml-metal.m:2508` shows softmax nth sizing to 256 granularity). The uniform
form is revisited only if card 6.3's trade cell measures the padding cost above the
orchestration saving at EVERY bucket size — then card 6.4 (the shape-invariant plan, whose
`Uniforms` already carry every extent per dispatch, `msl.rs:2207-2218`) is unparked.

### 8.4 Placement: relax the strict `found != expected` check vs typed output placement

**Relax to `>=` (rejected):** the check at `metal.rs:991-1000` (and its CPU twin
`cpu.rs:346-356`) iterates EVERY named block, weights included; a relaxation there silently
accepts an over-sized WEIGHT as a prefix read — a correctness hole (§14). **A declared
over-allocation field on `QuantizedBlock` (rejected):** `QuantizedBlock` is proxima-tensor
public surface (`cpu.rs:3084`); a new field fails the relocation question because, once
`cached_len` is bucketed (§8.3), the KV leaf's extent IS the capacity `C` and the buffer is
exactly `C` elements — `found == expected` holds with no relaxation and no field. **Chosen:**
bucketing (§8.3) plus the driver alias. The cache buffer is one `AlignedBuffer` of `C` rows
per tensor (`align.rs:69`, first production caller); its `(pointer, byte_length)` is stable so
`NOCOPY_BUFFERS` (`metal.rs:1848`) hits from token 2; the K/V-producing reduce's output is
bound to that same buffer at `out_layout.base = cached_len * row_stride` by inserting
`(persistent, offset)` into `device_buffers` (`metal.rs:2249` — the map is already
`BTreeMap<NodeId, (MetalBuffer, usize)>`, so the alias is an insert, not a type) instead of
`allocate_buffer` (`:2210`); the cache `Input` leaf aliases the same buffer by name
(`mark_resident`, `metal.rs:350-362`). Constraint: §1 (extend `AlignedBuffer`; write the call
site both ways — identical lines), §4 (placement is data: name + capacity), §11, §14.

### 8.5 Wide cooperative reduce: cap from a generated const, not a device query

`SIMD_WIDTH` "cannot be runtime config at any tier" (`sized.rs:36-45`). The reduce width
becomes `[cooperative_reduce] max_width` in `omega-runtime.toml` (default 1024 = Apple's
`maxTotalThreadsPerThreadgroup` on every device this crate targets), with the two-level
`simd_sum -> threadgroup -> simd_sum` tree emitted from the generated const; the driver
asserts `pipeline.maxTotalThreadsPerThreadgroup >= max_width` once at pipeline creation and
fails loudly otherwise. Constraint: §12 (no magic number) and the sized.rs doc's own rule.

### 8.6 Row numbers and the log

Main ends at ROW 233. The three unlanded "ROW 234"s and the branch's 234-267 are placeholders.
Assignment is made in this document (§7) at land time in landing order: ROW 234 = today's
seal, 235 = the reconciliation record, 236+ = one per landed card. The branch's rows that are
NOT landed are copied into an appendix row (ROW 235) as negatives with their original branch
numbers cited, so nothing measured is lost.

### 8.8 The second rewrite engine: `correct_packed_matmul_layouts` belongs in `bind`

Found by the round-2 critique and verified: after `bind` returns, the Metal driver rewrites
every packed Q4_K/Q5_K/Q6_K weight operand's `Layout` in place
(`omega/src/metal.rs:1003-1013`, function defined at `proxima-tensor/src/bind.rs:1618-1647`,
whose own doc says "`layout_of` has no way to get this right on its own"). The CPU path
(`cpu.rs:358`) does not apply it because its quantized kernels never read packed bytes through
`layout_of`. So the bound plan Metal executes differs from the plan CPU executes, on main,
today — brief item 1 ("one bound plan, identical for every backend") is false before this plan
starts. **Decision:** `bind` takes the packed-operand set (it already exists as
`PackedOperands` in omega and as the `packed_operands` map the driver passes to the fix) and
`layout_of` emits the physical layout for a packed operand at bind time; the driver-side call is
deleted; the one-bound-plan test (§5, card for item 1) captures the plan AFTER each driver's
prepare, per backend, and is pre-registered to FAIL on main because of `metal.rs:1013` — a test
that cannot fail on main is not a test of the claim. Constraint: §1 (the rewrite exists once,
in the one place that binds), the central-claim lint ("everything is one bound plan" —
enumerate what is not: this).

### 8.9 The route census must be lock-free and per plan, not per dispatch

The shipped census pattern `WIDTH_TILE_DECLINE` is a `Mutex<BTreeMap>` (`instrument.rs:842`,
`.lock()` at `:857`); the proposed record site's neighbour `ENCODE_DISPATCH_CALLS`
(`metal.rs:2243`) is an atomic `Counter`. A mutex taken 1196 times per token inside `encode_op`
would sit uncosted in the exact `op_setup`/`encode_dispatch` slice Phase 4 measures. **Decision:**
the route is a property of the plan, not of the dispatch — `prepare` computes `Vec<KernelRoute>`
once per plan beside `resolved`, the per-dispatch record is an atomic per-route counter array
(`[Counter; ROUTES]`, no lock, no map), and the census-sum gate compares the array to
`ENCODE_DISPATCH_CALLS`. Constraint: §21 (lock-free first; a lock is a missing owner — the owner
is the `Plan`), and the measurement rule that an instrument may not live uncosted inside the
slice it measures (an OFF arm is measured once, card 3.2, with `ENCODE_DISPATCH_TICKS` as the guard).

### 8.10 The measurer mutex: `flock(1)` does not exist on this Mac

Verified: `which flock` finds nothing and `/opt/homebrew/bin/flock` is absent. Every measuring
command in the plan is serialized through a mutex, so the mutex is card one, not boilerplate:
`brew install flock` (the discoteq port) or, if unavailable, a repo-local
`scripts/gpu-measure-lock.sh` that takes the lock with python's `fcntl.flock` and `execvp`s the
command. Every later card welds the lock the same way; `bash scripts/omega-gate.sh` runs under it
too because its `--all-features` steps run the Metal test suites.

### 8.11 Residency before bucketing (the 4.1/5.3 cycle)

Bucketing the KV leaf's extent to a capacity while the host still hands a `cached_len`-sized
buffer trips the strict `found != expected` check on both backends at the first token; the card
that pads the buffer would depend on the card that buckets. **Decision:** the KV cache becomes
device-resident FIRST as a page-aligned capacity arena registered as a host span (generalising
`checkpoint_mapping_offset`, `metal.rs:1786-1815`, to N spans), and the block handed to omega
stays the `cached_len`-sized prefix slice of that arena — so `found == expected` holds with no
validator edit and the no-copy cache hits on a stable pointer. Bucketing lands second, behind its
own default-off feature, with a trade cell (bucket ∈ {8, 32, 256} vs feature-off) whose kill is
`Δ(cached-range GPU work) > Δ(prepare + op_setup)` at every bucket; if that fires, the fallback is
the shape-invariant plan (extents are already per-dispatch uniforms, `msl.rs:2207-2218`;
`pipeline_lookup` is 0.04 ms, so pipelines already survive `cached_len` changes) rather than a
padded reduce. Constraint: §15 (no relaxed check), §4 (capacity is config), and the owner's
memory rule (the arena size is a build-time key asserted in `build.rs`, never `context_length`).

### 8.12 Placement, settled (round 4): a static offset on an AFFINE write map; scatter never enters

Round 2 ruled "scatter only, `shape.rs` untouched" by reading `map.rs:118-124` as rejecting a
write offset; it rejects a `Reduce`-WIDE destination field, which is a different thing. Round 3
proposed proving injectivity by walking a `Computed` index chain back to `Iota(coeff 1) +
loop-invariant scalar` and folding it into `out_layout.base`. Round 4's critique refuted the path
that rule ran on: `infer_reduce` (`proxima-tensor/src/shape.rs:183-201`) short-circuits every
data-dependent out_map to `scatter_output_shape` (`:200`, `:495`), so `project_output_shape`
(`:469-485`, body `out_map.affine()` at `:474`) is reached ONLY for an `Affine` map — a `Computed`
out_map never arrives at the line the rule extended — and the `Computed` form's `offset` slot at
`gathered_dim` already carries the scatter's destination extent (`map.rs:175-197`, `shape.rs:168-181`),
so it cannot also carry a write base. **Decision:** the KV write is an `Affine` write map with a
static nonzero `axis.offset`. `project_output_shape` accepts `[term] if term.coeff == 1` with a
nonzero offset and returns `iter_extents[axis] + offset`; `layout_of`
(`proxima-tensor/src/bind.rs:1594-1606`, reached from `:962` for the Affine arm) ALREADY folds
`offset × stride` into `out_layout.base`, and the emitters already consume it as `u.out_base`
(`omega/src/msl.rs:2216`, `:2361`). Injectivity is arithmetic, not a prover: two producers of one
destination with `coeff == 1` are disjoint iff `[o1, o1+e1)` and `[o2, o2+e2)` do not overlap, checked
at bind, with overlap a named error. The adjoint is right by construction: `differentiate_reduce`
reuses the write map as a read map (`proxima-autograd/src/adjoint.rs:835`), so a placed write's
gradient is a sliced read — and because `proxima-autograd` consumes `Reduce::out_map` semantics,
card 9.1 runs `scripts/proxima-autograd-gate.sh` and an adjoint round-trip test. `Computed`,
`scatter_output_shape`, `build_scatter_out_layout`, `IndexMap::scatter`, `as_gather_from_output`
and the three `ScatterNotSupported` sites are all untouched. §8.4's "typed placement by name" text
is superseded by this. No GPU scatter emitter, no atomics, no new Op, no new IndexMap variant.

### 8.13 One Q4_K body: a build-time profile axis, not two features

Two mutually-exclusive cargo features guarded by `compile_error!` would turn FIVE of
`scripts/omega-gate.sh`'s six steps red — [2/6] `--all-targets --all-features`, [3/6] nextest
`--all-features`, [4/6]'s second clippy arm, [5/6]'s second rustdoc arm and [6/6] `cargo test --doc
--all-features` — not one (the script read in full in round 4). **Decision:** `[q4k] body = "main"
| "mask_fma" | "pair_dot"` in `omega-runtime.toml` — three values, because the body on `4be2f3a`
today stays selectable as the CONTROL arm of the bake-off — emitted by `omega/build.rs` as a
`rustc-cfg` in the same directive form `proxima-build` uses (`proxima-build/src/lib.rs:205-234`:
`cargo:rustc-check-cfg=cfg(omega_q4k_body, values(...))` + `cargo:rustc-cfg=omega_q4k_body="…"` +
`cargo:rerun-if-env-changed=OMEGA_Q4K_BODY`) beside `emit_sizing_consts`. The axis is NOT added to
the workspace `Profile`: its axes are a fixed workspace-runtime table (`proxima-build/src/profile.rs:50`)
and omega does not depend on that crate; the row names this so the second mechanism is explained.
Selecting a value is a REBUILD, so the bake-off builds each arm once into its own target dir and
interleaves the three prebuilt test binaries — nine rebuilds between measurements would not be an
interleaved cell. The tie-break (card 4.2) is terminal: parity ≤ 1e-4 vs `cpu::evaluate` on the real
`blk.0.attn_q.weight` or out; the `ReduceRowBlockedPacked` route count pinned equal across arms or
the comparison is void; batched `gpu_exec_ms` beyond the pooled CoV; then summed per-family per-op
ms (deflated by the +7.3% per-op inflation); then lower parity error; then total emitted MSL bytes
(`emit_is_deterministic_byte_equal`, `msl.rs:4656`); then `pair_dot`, the body that is already a git
commit. The loser is a recorded negative row and stays selectable, not a deletion.

### 8.7 One lane owner

Two agents worked this lane on two trees on two consecutive days and rewrote one kernel body
twice. From this document on: one worktree per card, cards serialized on the box (protocol
0.3 rule 4), and the parallel branch is frozen after card 1.3's cherry-picks. The owner
communicates this to the other agent; this plan records it as the constraint.

## 9. Tournament trail (plan-rigor; the ordering in §5 is what survived it)

Shape: round 0 produced an incumbent Plan A; each later round ran a fresh critique of the
incumbent and a blind alternative in parallel, a synthesis of the three, then three fresh judges
who ranked the incumbent, the alternative and the synthesis under neutral labels in randomized
order. Borda 2/1/0. Every author, critic, synthesizer and judge ran on Opus; each verified
citations against main `4be2f3a` read-only before ranking. Intermediate artifacts (plans,
critiques, tallies, the evidence ledger R0-R18, the three fix lists that patched the cards) are
committed beside this file in `tournament/`; the raw seal logs in `baseline-2026-09-03/`.

| round | candidates | judge rankings | Borda | winner | decisive findings |
|---|---|---|---|---|---|
| 0 | Plan A (incumbent) | — | — | A | first complete card plan; 26 worktrees; ordered census after body swap |
| 1 | A, blind B, synthesis_AB | [S,B,A] ×3 | S 6 / B 3 / A 0 | synthesis_AB | A and B independently converged on the same spine (seal → census → Q4_K body → wide reduce → plan-stable cached_len → buffer pool → placement → single-range → emitter core → incumbent cells). Critique of A: the harness asserts `plan_hits == 0` (`proxima-model-interop/src/bind.rs:3053`); the cache-tail mask needs a symbol-1 Iota and a `cached_len` leaf; `operand_bytes` reports the 4.14 GB mapping; three branch names already checked out; MEMORY numbers used as delta anchors; the census must record per dispatch. B found: `layout_of` already folds a write offset into `Layout.base`; `u.out_base` already emitted; scatter implemented on CPU but rejected by all three GPU emitters; `sealed-pass.sh` hardcodes a worktree path. |
| 2 | synthesis_AB (incumbent), blind B2, synthesis_2 | [S2,B2,S1] ×3 | S2 6 / B2 3 / S1 0 | synthesis_2 | Critique of the incumbent found the second rewrite engine (`metal.rs:1013`, Metal-only — brief item 1 false on main), the proposed census copying a `Mutex<BTreeMap>` 1196 times per token into the slice it measures, an unsound `declared == found` predicate, the scalar leaf not wired through `named_blocks`, memory as a kill on one card of eighteen. B2 brought the `flock` measurer queue, the byte-formula memory gate (4,140,417,024 + capacity×262,144 + 40 MB), scatter-only placement leaving `shape.rs` untouched, and a device arena that prints its peak before allocating. Judges' residuals on S2: injectivity by leaf name, 4.1 as a pre-registered loss on the critical path, `0.6`'s `$wt` inside single-quoted `sh -c`, a literal `plan_misses=8`, an unbounded onnxruntime build. |
| 3 | synthesis_2 (incumbent), blind B3, synthesis_3 | [S3,B3,S2] ×3 | S3 6 / B3 3 / S2 0 | synthesis_3 | Critique of S2 found: `flock(1)` absent on this Mac (every weld would have failed on card one); `ScalarOp::GreaterEqual` does not exist (the tail mask must be `Greater(cached_len_leaf, iota)` + `Select`); `route as usize` on a data-carrying variant is E0605; `Prepared` private so the fingerprint test cannot reach it; a code-level cycle between bucketing and the KV buffer (`found != expected` on token 1); the op-timed path's own upload loop untouched; leaf-name injectivity unprovable for a host-fed `Op::Input`. B3 brought residency-before-bucketing, the build-time `[q4k] body` profile axis (two exclusive features would red `omega-gate.sh --all-features`), the repo-local lock shim with `--wait`/exit 75, structural injectivity folding into `out_layout.base`, and the whole-buffer-only arena constraint from the readback invariant at `metal.rs:2360-2364`. S3 merged both and re-derived every band from its predecessor's delta. Judges' residuals on S3 are listed in §9.1. |
| 4 | synthesis_3 (incumbent), blind B4, synthesis_4 | [S4,B4,S3] ×3 | S4 6 / B4 3 / S3 0 | synthesis_4 (cap round; NOT converged) | Critique of S3 verified 14 first-execution breaks on main: `PackedOperands`/`PackedCodec` are omega types (`msl.rs:656`, `:591`) so nothing descends into proxima-tensor and the `grep == 1` assertion is unreachable (22 hits in 9 files); a `Computed` out_map never reaches `project_output_shape` (`shape.rs:183-201` short-circuits to `scatter_output_shape`) and the `Computed` offset slot already holds the destination extent; `UNIFORM_BUFFERS` is a content-keyed unbounded map (`metal.rs:2056-2078`), so in-place writes corrupt co-keyed ops and a per-token base inserts forever; the memory cap fired on unmodified main at prefill (4.30e9 measured vs 4.18e9 cap) and its 40 MiB term had no derivation; 8.3's 600-line gate was unreachable (423 max); `TILE_DIM` is feature-gated and self-documents as a hardware fact; `metal_vs_cpu` is run by CI. B4 brought the two-`bind.rs` correction, both harness assertions, the six-step gate, `capacity % 8`. S4 resolved every break: affine-only write offset with interval-overlap injectivity and an autograd round-trip, plan-owned uniforms plus a bounded cache, caps from MEASURED peaks with a labelled CHOSEN headroom, per-op inflation deflated in the ladder, Phase 8 after the board. |

### 9.1 Round 3, round 4 and convergence

Round 3 was unanimous (three first-place votes for synthesis_3), but the incumbent lost, so the
"same candidate wins two consecutive rounds" test is not met. Round 4 is the cap round. The
residuals the round-3 judges named on synthesis_3, each carried into round 4's critique and
fixed in §5 regardless of round 4's outcome:

- card 9.1 opens `map.rs:238` (`IndexMap::as_gather_from_output`) but never scopes the autograd
  adjoint path in blast, expect or kill;
- card 9.2 asserts `UNIFORM_BUFFER_REUSES` unchanged with no expect line and no ON/OFF arm;
- card 8.3's `>= 600` line-count continuation gate is an evidence-free threshold;
- on-device argmax is neither a card nor an abandoned design;
- `$WT`/`$TD`/`$LOCK` are shell variables and shell state does not persist between a hands
  model's tool calls — every command must carry absolute paths;
- card 0.7's capture loop is bash-specific (`read -r -d ''`) on a zsh host;
- card 6.1 relies on the KV arena being zero-initialised without a citation or a test;
- 5.1, 8.1 and 9.2 add surface without both binary questions written in-line;
- 10.4 holds the mutex through a C++ build and `--wait 5400` makes waiting cards exit 75;
- the MILLI rung hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3105`) and sets
  `PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`), so `PROXIMA_MAX_TOKENS` does not govern
  it and only the `F`-from-`generated.0.len()` contract does;
- 33 of 44 cards measure on one mutex, so the schedule length is the measuring-card count;
- the board band carries `δ_b` as a free symbol until card 6.3 measures it.

Round 4 (the cap round) was unanimous for synthesis_4 (three first-place votes, 6-3-0 over blind
B4 and the incumbent synthesis_3). The incumbent lost again, so the convergence test ("the same
candidate wins two consecutive rounds") is NOT met at the four-round cap, and per the plan-rigor
cap rule this document emits synthesis_4 as the final plan with the flag **NO CONVERGENCE**. The
trajectory is the reason the flag is not alarming: every one of the four syntheses beat its
incumbent 6-3-0, every critique found first-execution breaks that the next synthesis closed, and
the residual set shrank from execution-blocking (rounds 1-3: an absent `flock`, an absent
`ScalarOp::GreaterEqual`, an E0605 cast, a private `Prepared`, a bucket↔buffer cycle, an omega
type named from proxima-tensor, a memory cap that fired on unmodified main) to scheduling,
provenance and one under-quoted API (round 4). Per-axis scores on synthesis_4 across the three
judges: risk 9/8/9, ordering 9/9/9, rollback 8/8/9, missing steps 8/8/8, hidden coupling 9/8/9,
observability 9/9/9, scope discipline 8/7/8.

Round-4 residuals, and where each landed:
- `$WT`/`$TD`/`$LOCK` shell variables in commands — closed in §5 already: every command line
  carries the absolute path (round-3 j9).
- exit 75 had no retry policy — G3 now states one (re-issue after a 300 s pause, up to six times,
  waits logged; the seventh is a STOP).
- D7 over-generalised the harness gating — the ORACLE test (`proxima-model-interop/src/bind.rs:2764`)
  carries no `metal` cfg; G4 now states each test's cfg.
- 9.1's REJECT half must drive `proxima-autograd/src/adjoint.rs:806`'s data-dependent guard and
  `error.rs:73-88` — added as expect cases.
- the MILLI rung is a 5-token cell (`proxima-model-interop/src/bind.rs:3103`) with inert env vars — every milli row is
  labelled "milli budget 5, bench budget 8".
- the `UNIFORM_BUFFERS` bound is a leak repair riding inside 6.5 — it is commit 1 of 6.5 with a
  capacity+1 eviction test and is kept on rollback.
- nothing gated G0's bare-`bind.rs` rule — card 11.1 greps this document for bare cites.
- 5.4 §X.11 asserted B4's 4.40e9 ceiling sat below R13's prefill peak — false (4.40e9 > 4.305e9);
  corrected.
- 5.2 relied on an `AlignedBuffer` → `&[f32]` accessor it never quoted — opens/expect added.
- the board `[46.8, 57.3]` is a seven-term composition carrying two cross-tree anchors and δ_b's
  pre-registered endpoints; it stays labelled the weakest number in the document.
- 33 of 44 cards serialise on one mutex and there is no parallel lane; stated, not fixed.
- brief item 3 (one emitter core) is the least-closed of the eight: one kind lands, after the
  board, behind a measured gate.

