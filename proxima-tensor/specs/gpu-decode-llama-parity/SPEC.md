# gpu-decode-llama-parity

status: audited
owner: brian
created: 2026-09-19

## problem

proxima qwen35moe (qwen3.6:35b-a3b) GPU decode runs ~2.4x slower per token than
llama.cpp/Ollama on the same GGUF (54-77 ms vs 31 ms) because each token is
shattered into ~2,704 tiny latency-bound Metal dispatches, so collapsing the
three dominant dispatch-kind buckets — packed-row 1,211, cooperative 551,
elementwise 503 — into batched/fused kernels moves decode TTNT toward the
incumbent without changing generated tokens.

Measured basis (wf_fecbb835-f14 Baseline + ROW 542 kind ablation, all default-off
today): per-token gpu_exec splits packed-row 25.1 ms / cooperative 19.8 ms /
elementwise 12.1 ms; the byte floor is ~5.5 ms (under target), and pruning dead
dispatches moved gpu_exec 0 ms — so the term is per-dispatch latency × count, not
bandwidth and not raw count. This is the disciplined-component incumbent frame:
llama.cpp on the same model is the named rival.

## refutation condition

If the three buckets' op counts each collapse as designed (packed-row ≤200,
cooperative and elementwise sharply down) but France 32-tok decode TTNT does NOT
move toward llama.cpp — stays ≥2x — then decode is not dispatch-latency-bound and
this whole batching/fusion thesis is wrong (the term is bandwidth, residency, or
something unmeasured), and the campaign is retracted, not iterated.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | the routed-expert packed-row bucket (~1,211 ops/token) collapses via a batched indexed matmul — component `moe-mul-mat-id` | yes |
| R2 | the cooperative-reduce bucket (~551 ops: GatedDeltaNet mixer + attention) collapses via fused Metal kernels — component `gdn-attention-fusion` (not yet spec'd) | yes |
| R3 | the elementwise bucket (~503 ops: rmsnorm/silu/softplus/conv chains) collapses via stage fusion — component `elementwise-stage-fusion` (not yet spec'd) | yes |
| R4 | with all three flags on, France 32-tok decode TTNT is ≤ 1.1 × llama.cpp mean ms/token on BOTH qwen35moe and gemma4 (batiai/gemma4-26b), token stream unchanged vs all-off | yes |
| R5 | every component is a default-off compile-time flag; main's default decode stays byte-identical until the e2e win is measured | yes |

## architecture

Three independent disciplined components, each its own `specs/<slug>` (spec-first
per component), each a bind-admission + Metal kernel behind a default-off flag.
This parent tracks the measured decode TTNT as each component's flag lands, and
owns the only cross-component claim (R4: do the three together reach parity).

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| decompose by dispatch KIND | packed-row / cooperative / elementwise, one component each | that is the axis the ROW 542 in-buffer ablation actually measured; a component per kind has a clean, independently-measurable op-count target |
| Metal-only first | fuse the Metal decode path; non-Metal backends explicitly decline each flag | the incumbent (llama.cpp) is a Metal path on this box; cpu/cuda/wgsl parity is a separate effort |
| default-off per component | each flag is its own firewall | R4 (the e2e win) is unproven until all three land; nothing flips main's default until then |
| separate specs, not one mega-spec | one `specs/<slug>` per component | each component is one commit-series with its own AC gate; a mega-spec cannot be audited or sliced cleanly |

## acceptance criteria

Every AC runs with this setup sourced once:

```sh
WT=/Users/brianbruggeman/repos/slot-0/proxima          # or the active worktree
BLOB=/Users/brianbruggeman/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d
Q="What is the capital of France?"
OLL=/Applications/Ollama.app/Contents/Resources/ollama
reclaim(){ for m in $($OLL ps|awk 'NR>1{print $1}'); do $OLL stop "$m"; done; }
pk(){ grep 'reduce-packed-row-blocked' "$1"|grep -oE 'op_count=[0-9]+'|grep -oE '[0-9]+'|head -1; }
co(){ grep 'reduce-cooperative' "$1"|grep -oE 'op_count=[0-9]+'|grep -oE '[0-9]+'|head -1; }
el(){ grep -w 'elementwise' "$1"|grep -oE 'op_count=[0-9]+'|grep -oE '[0-9]+'|head -1; }
mean(){ grep -oE 'mean[^0-9]*[0-9.]+' "$1"|grep -oE '[0-9.]+'|head -1; }
# ALL="--features omega/metal-moe-mul-mat-id,omega/metal-gdn-attention-fusion,omega/metal-elementwise-stage-fusion" (as each lands)
# build /tmp/ggf.alloff (no flags) and /tmp/ggf.allon ($ALL) the same way moe-mul-mat-id/SPEC.md does.
```

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu >/tmp/on.txt 2>&1; pk /tmp/on.txt` | packed-row op_count ≤ 200 (from ≈1211) |
| AC2 | R2 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu >/tmp/on.txt 2>&1; co /tmp/on.txt` | cooperative op_count ≤ 150 (from ≈551) |
| AC3 | R3 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu >/tmp/on.txt 2>&1; el /tmp/on.txt` | elementwise op_count ≤ 250 (from ≈503) |
| AC4 | R4 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu >/tmp/on.txt 2>&1; reclaim; /tmp/ggf.alloff "$BLOB" "$Q" 32 gpu >/tmp/off.txt 2>&1; LT=$($OLL run qwen3.6:35b-a3b --verbose "$Q" 2>&1|grep -oE 'eval rate:[^0-9]*[0-9.]+'|grep -oE '[0-9.]+'|head -1); echo on=$(mean /tmp/on.txt) llama_ms=$(awk "BEGIN{print 1000/$LT}"); diff <(grep generated_text /tmp/on.txt) <(grep generated_text /tmp/off.txt)|grep -c '^[<>]'` | printed `on` ≤ 1.1 × printed `llama_ms`; `0` differing lines vs all-off |
| AC5 | R4 (gemma) | `G=$($OLL show --modelfile batiai/gemma4-26b:latest|grep -oE '/[^ ]*blobs/sha256-[a-f0-9]+'|head -1); if [ -z "$G" ]; then echo GEMMA_BLOB_MISSING; else reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$G" "$Q" 32 gpu >/tmp/gon.txt 2>&1; reclaim; /tmp/ggf.alloff "$G" "$Q" 32 gpu >/tmp/goff.txt 2>&1; LT=$($OLL run batiai/gemma4-26b:latest --verbose "$Q" 2>&1|grep -oE 'eval rate:[^0-9]*[0-9.]+'|grep -oE '[0-9.]+'|head -1); echo gon=$(mean /tmp/gon.txt) gllama_ms=$(awk "BEGIN{print 1000/$LT}"); diff <(grep generated_text /tmp/gon.txt) <(grep generated_text /tmp/goff.txt)|grep -c '^[<>]'; fi` | non-empty `$G` (else GEMMA_BLOB_MISSING); printed `gon` ≤ 1.1 × `gllama_ms`; `0` differing lines |
| AC6 | R5 | `for f in metal-moe-mul-mat-id metal-gdn-attention-fusion metal-elementwise-stage-fusion; do awk '/^\[features\]/{x=1} x&&/^default *=/{print}' omega/Cargo.toml | grep -c "$f"; done; reclaim; /tmp/ggf.alloff "$BLOB" "$Q" 16 gpu 2>&1 | grep -c Paris` | `0` `0` `0` (no flag in default); default France Paris `1` |

## out of scope

- Prefill / TTFT (the 8,492-dispatch prefill path is a separate campaign).
- Non-Metal backends (cpu/cuda/wgsl decline every flag; their parity is separate).
- Any change to the Op graph, spec builders, or the algebra.
- Pushing to origin — local main only (identity phase).

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| op counts collapse but TTNT does not move (refutation fires) | medium | the whole thesis is wrong | that IS the refutation condition — retract, do not iterate; report the trace |
| a component wins alone but the three interact (fusion boundaries collide) | medium | e2e < sum of parts | R4 measures all-on together, not per-component sums |
| memory residency blocks the 24GB GPU runs | high on this box | gates cannot complete | reclaim Ollama before every GPU run (see moe-mul-mat-id risks) |
| the batched forms re-add slicing/reduce overhead (the prior 74.9-vs-57.7 trap) | medium per component | net loss dressed as a win | each component's own spec carries the anti-trap invariant + measures the delta |

## context

- Root cause + per-kind trace: memory `project_qwen35moe_dispatch_count_bound` (ROW 542), `feedback_finish_gate_ollama_residency_contention`.
- Component 1 spec: `proxima-tensor/specs/moe-mul-mat-id/` (scaffold landed 85ac70da5; kernel in flight).
- Components 2 and 3: not yet spec'd — start each with `/spec-first` after component 1's e2e number is in.
- Incumbent: llama.cpp via Ollama `qwen3.6:35b-a3b` / `batiai/gemma4-26b:latest`.
