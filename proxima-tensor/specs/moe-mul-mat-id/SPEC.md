# moe-mul-mat-id

status: audited
owner: brian
created: 2026-09-19

## problem

On qwen35moe (qwen3.6:35b-a3b) and gemma4 (26b-a4b) GPU decode, collapsing the
~1,211 per-token routed-expert `reduce-packed-row-blocked` dispatches into one
batched indexed matmul (llama.cpp `ggml_mul_mat_id` shape) behind a default-off
flag lowers France 32-token mean ms/token toward the Ollama/llama.cpp incumbent
without changing the generated tokens.

Mechanism this rests on (measured, wf_fecbb835-f14 Baseline + earlier reads):
the routed-expert product is ~1,211 tiny latency-bound dispatches/token, each a
0.59 MB slab matvec at ~33 GB/s against a 381 GB/s ceiling — one dispatch per
expert. Baseline ratio proxima:llama ≈ 2.4x (load-sensitive absolute ms; the op
counts are the stable signal).

## refutation condition

If the batched kernel lands correct (flag-on token stream == default path) but,
on France 32-tok GPU decode, TTNT does NOT drop below the flag-off baseline
beyond CoV noise — OR the `reduce-packed-row-blocked` op count does not collapse
— then batching the expert product is not the lever (the term is the combine/
gather, or elsewhere), and this component is the wrong build. This is precisely
the prior failure: the naive horizontal merge measured 74.9 vs 57.7 ms because
the grouped form added per-round slicing + cooperative reduces. Separately, if the
kernel is correct and faster for qwen35moe but breaks or fails to batch gemma4
26b-a4b, it is a qwen-specific hack, not the model-generic path the reckoning
requires — that too refutes the component.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | flag `metal-moe-mul-mat-id` exists, default-off; main's default decode is byte-unchanged | yes |
| R2 | both `--features metal` (flag off) and `--features metal,metal-moe-mul-mat-id` (flag on) compile | yes |
| R3 | flag-on real-checkpoint France answers Paris AND emits the same greedy token stream as flag-off | yes |
| R4 | flag-on collapses per-token `reduce-packed-row-blocked` op count sharply (batching landed) | yes |
| R5 | flag-on France 32-tok TTNT is measured vs flag-off and vs llama.cpp, CoV-tracked, delta recorded (win or negative) | yes |
| R6 | no new type in a proxima library crate (primitives/core/tensor public surface); kernel/plumbing confined to omega/msl + tensor bind | yes |
| R7 | the batched kernel is model-generic: gemma4 26b-a4b (a second MoE architecture) flag-on decodes correctly, matches its own flag-off token stream, and collapses its packed-row op count — proving the reckoning's "any model" claim, not a qwen-specific hack | yes |

## architecture

The Op graph is unchanged (algebra is already expert-generic; the defect is
dispatch granularity, not the algebra). The change is a Metal kernel + a bind
admission gated by the compile-time flag.

- **Kernel (omega/msl):** one batched indexed matmul dispatch per expert weight
  matrix per MoE round. The k selected experts index the gathered weight rows; k
  is the grid-z dimension; k-independent products + one combine reduce. Contract:
  inputs = (gathered/indexed expert weights, activation, routing selection);
  output = bit-identical to the current per-expert `reduce-packed-row-blocked`
  path. Exact MSL signature pinned by the design phase (workflow wf_fecbb835-f14).
- **Bind admission (proxima-tensor):** the routed-expert product nodes are
  admitted to the batched kernel only when `metal-moe-mul-mat-id` is enabled;
  otherwise the existing per-expert classification stands. The admission emits a
  NEW `BoundOpKind::RoundBatchedReduce` variant (carrying the reduce fields + a
  non-optional `round_count: u32`), NOT an `Option` field on the existing
  `Reduce` — so every backend's exhaustive match on `BoundOpKind` must handle it
  or explicitly decline; a catch-all consumer cannot silently run round 0 only.
  Until the Metal kernel lands, every backend (msl, cpu, cuda, wgsl, wgpu)
  declines the variant with a typed error (no silent-wrong path).
- **Flag:** `metal-moe-mul-mat-id` in the omega + proxima-tensor `[features]`,
  NOT in `default`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| where the batching lives | Metal kernel + bind admission | a spec/builder change would rewrite the algebra, which is already correct; the defect is granularity |
| round structure | k-independent products + one combine | grouping the existing per-round product shape re-adds the slicing/reduces that lost 74.9 vs 57.7 (prior 7 slices) |
| new library type? | none (no new public type) | the batched product is an omega/msl edge concern; the pipe question is answered by writing the kernel call, not a `MergedExpertProduct` type (guiding-principles §1, rust.md pipe rule) |
| batched-op encoding | a new `BoundOpKind::RoundBatchedReduce` variant, NOT an `Option<u32>` field on `Reduce` | ~~round_count field~~ an additive Option field rides through every non-Metal backend's `..`-match and silently runs round 0 only — no error, wrong numbers (compliance critical ab0590177, and the "information destroyed at a match boundary" footgun). A variant forces the exhaustive-match compile error every backend then resolves by handling or explicitly declining — the exact safety MoeTopK/GatedDeltaNet rely on |
| firewall | default-off compile-time flag | main's default decode must stay byte-identical until the e2e bench shows a win (disciplined-component gate 1) |

## acceptance criteria

Every AC command runs with this setup sourced once (binds the vars the auditor
flagged as unbound, and builds the two named binaries `/tmp/ggf.off` / `/tmp/ggf.on`):

```sh
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-moe-fusion; cd "$WT"
BASE=/Users/brianbruggeman/repos/slot-0/proxima     # clean local main 5ec9c8698
BLOB=/Users/brianbruggeman/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d
Q="What is the capital of France?"
OLL=/Applications/Ollama.app/Contents/Resources/ollama
reclaim(){ for m in $($OLL ps|awk 'NR>1{print $1}'); do $OLL stop "$m"; done; }
pkops(){ grep 'reduce-packed-row-blocked' "$1"|grep -oE 'op_count=[0-9]+'|grep -oE '[0-9]+'|head -1; }
gtext(){ grep generated_text "$1"; }
GEMMA_TAG=batiai/gemma4-26b:latest   # the tag actually installed on this box (NOT gemma4:26b-a4b)
gemma_blob(){ $OLL show --modelfile "$GEMMA_TAG" 2>/dev/null|grep -oE '/[^ ]*blobs/sha256-[a-f0-9]+'|head -1; }
CARGO_TARGET_DIR=$WT/target cargo build --release --example gguf_generate                                   && cp $WT/target/release/examples/gguf_generate /tmp/ggf.off
CARGO_TARGET_DIR=$WT/target cargo build --release --example gguf_generate --features omega/metal-moe-mul-mat-id && cp $WT/target/release/examples/gguf_generate /tmp/ggf.on
```

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 (flag, default-off) | `grep -c metal-moe-mul-mat-id proxima-tensor/Cargo.toml omega/Cargo.toml; awk '/^\[features\]/{f=1} f&&/^default *=/{print}' omega/Cargo.toml \| grep -c metal-moe-mul-mat-id` | each Cargo.toml ≥1; `0` under `default` |
| AC2 | R2 | `CARGO_TARGET_DIR=$WT/target cargo build --release --example gguf_generate 2>&1\|grep -c 'error\['; CARGO_TARGET_DIR=$WT/target cargo build --release --example gguf_generate --features omega/metal-moe-mul-mat-id 2>&1\|grep -c 'error\['` | `0` and `0` |
| AC3 | R1 (byte-unchanged default) | `reclaim; /tmp/ggf.off "$BLOB" "$Q" 16 gpu >/tmp/off16.txt 2>&1; CARGO_TARGET_DIR=$BASE/target cargo build --release --manifest-path $BASE/Cargo.toml --example gguf_generate -q && cp $BASE/target/release/examples/gguf_generate /tmp/ggf.base; reclaim; /tmp/ggf.base "$BLOB" "$Q" 16 gpu >/tmp/base16.txt 2>&1; grep -c Paris /tmp/off16.txt; diff <(gtext /tmp/off16.txt) <(gtext /tmp/base16.txt)\|grep -c '^[<>]'` | flag-off Paris `1`; `0` differing lines vs clean 5ec9c8698 |
| AC4 | R3 (qwen parity) | `reclaim; /tmp/ggf.off "$BLOB" "$Q" 16 gpu >/tmp/off16.txt 2>&1; reclaim; /tmp/ggf.on "$BLOB" "$Q" 16 gpu >/tmp/on16.txt 2>&1; test -s /tmp/off16.txt && test -s /tmp/on16.txt && echo files_ok; grep -c Paris /tmp/on16.txt; diff <(gtext /tmp/off16.txt) <(gtext /tmp/on16.txt)\|grep -c '^[<>]'` | `files_ok`; flag-on Paris `1`; `0` differing lines vs flag-off (self-contained: regenerates both) |
| AC5 | R4 (qwen op-collapse) | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.off "$BLOB" "$Q" 32 gpu >/tmp/offp.txt 2>&1; reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.on "$BLOB" "$Q" 32 gpu >/tmp/onp.txt 2>&1; echo off=$(pkops /tmp/offp.txt) on=$(pkops /tmp/onp.txt)` | `off` in [1000,1300]; `on` ≤ 200 |
| AC6 | R5 (bench) | `for f in off on; do echo -n "$f "; { for i in 1 2 3; do reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.$f "$BLOB" "$Q" 32 gpu 2>&1\|grep -oE 'mean[^0-9]*[0-9.]+'\|grep -oE '[0-9.]+'\|head -1; done; }\|awk '{s+=$1;ss+=$1*$1;n++} END{m=s/n;printf "mean=%.2f cov=%.1f%%\n",m,100*sqrt(ss/n-m*m)/m}'; done` | 2 lines: `off mean=<x> cov=<y>%` and `on mean=<x> cov=<y>%` (command computes both; delta = off-mean − on-mean; a run with cov≥5% is flagged no-signal) |
| AC7 | R6 (no new lib type) | `git -C $WT diff 5ec9c8698 -- proxima-primitives proxima-core proxima-tensor/src\|grep -E '^\+'\|grep -cE 'pub (struct\|enum\|trait) '` | `0` |
| AC8 | R6 (confinement) | `git -C $WT diff 5ec9c8698 --name-only\|grep -vcE '^(omega/\|proxima-tensor/)'` | `0` (all changed files under omega/ or proxima-tensor/) |
| AC9 | R7 (gemma parity) | `G=$(gemma_blob); if [ -z "$G" ]; then echo GEMMA_BLOB_MISSING; else reclaim; /tmp/ggf.off "$G" "$Q" 16 gpu >/tmp/goff.txt 2>&1; reclaim; /tmp/ggf.on "$G" "$Q" 16 gpu >/tmp/gon.txt 2>&1; test -s /tmp/goff.txt && test -s /tmp/gon.txt && echo files_ok; grep -c Paris /tmp/gon.txt; diff <(gtext /tmp/goff.txt) <(gtext /tmp/gon.txt)\|grep -c '^[<>]'; fi` | if `$G` empty prints only `GEMMA_BLOB_MISSING` (hard stop); else `files_ok`, gemma flag-on Paris `1`, `0` differing lines vs gemma flag-off |
| AC10 | R7 (gemma op-collapse) | `G=$(gemma_blob); if [ -z "$G" ]; then echo GEMMA_BLOB_MISSING; else reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.off "$G" "$Q" 32 gpu >/tmp/goffp.txt 2>&1; reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.on "$G" "$Q" 32 gpu >/tmp/gonp.txt 2>&1; echo goff=$(pkops /tmp/goffp.txt) gon=$(pkops /tmp/gonp.txt); fi` | if `$G` empty prints only `GEMMA_BLOB_MISSING`; else `gon` ≤ 0.3 × `goff` |

## out of scope

- The 551 `reduce-cooperative` bucket (GatedDeltaNet + attention fusion) — a separate component.
- Prefill / TTFT.
- Any change to the Op graph, spec builders, or the algebra.
- Pushing to origin — local main only (identity phase).

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| batched form re-adds per-round slicing/reduces → net loss (the prior trap) | medium | another 74.9-vs-57.7 dead end | I5 invariant; Build validates the shape before implementing, STOPs rather than build the losing form; Bench measures |
| non-resident mapping reads as zeros under Ollama pressure → false correctness fail | high on this box | a real parity pass looks broken | reclaim Ollama (`ollama stop`) before every GPU run |
| CoV noise hides a small delta | medium | a phantom win/loss | 3 runs, CoV-tracked, "no signal" below the floor |

## context

- Root cause + per-kind breakdown: memory `project_qwen35moe_dispatch_count_bound` (ROW 542), `feedback_finish_gate_ollama_residency_contention`.
- Incumbent: llama.cpp `ggml_mul_mat_id` (batched indexed expert matmul), measured via Ollama `qwen3.6:35b-a3b`.
- Prior attempts (do not repeat the grouped shape): ROW 537-545 grouped-expert product, closed as net loss.
- Discipline log: `proxima-tensor/specs/moe-mul-mat-id/discipline.md` (main-thread contract, lands with this spec).
- Base: local main `5ec9c8698`; worktree `proxima-wt-moe-fusion`; workflow `wf_fecbb835-f14`.
