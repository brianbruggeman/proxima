# Round-4 synthesis deltas to apply to the expanded cards (cards-all.canon.md)

Source of truth for every item: `scratchpad/synth4.md` (the round-4 synthesis). Where an item says
"per synth4 card X", open that card in synth4.md (`grep -n '^### X ' synth4.md`) and carry its text
into the expanded card in the named field, verbatim in meaning. Mark every changed or added
sentence with the literal tag `[round-4 synth <id>]`. Apply each item only to the named card(s).
Do not remove any `[round-4 fix F..]` tag; where an item supersedes an earlier fix, replace that
fix's sentence and keep both tags on the new sentence.

## S1 — memory cap, every card (supersedes F10/F11's numbers)
In EVERY card's `memory gate` block (and its row/report text) that lists MG-3 clauses: replace the
clause `(3) peak device_allocated_bytes ... 45_165_952 ...` (and any `4_319_800_704` / `45,165,952`
/ `4,319,800,704` / prefill-cap sentence added by F10/F11) with synth4 G8's clauses verbatim:
```
(3a) PREFILL peak device_allocated_bytes <= PREFILL_CAP = 4_305_000_000 * 1.05 + kv_capacity_tokens * 262_144
     (kv=0 => 4_520_250_000; kv=512 => 4_654_467_728; MEASURED R13 prefill peak, CHOSEN 1.05 headroom, DERIVED totals)
(3b) STEADY peak device_allocated_bytes over steps 3..S <= STEADY_CAP = 4_163_000_000 * 1.05 + kv_capacity_tokens * 262_144
     (kv=0 => 4_371_150_000; kv=512 => 4_505_367_728)
(4) plan_cache_len <= 1 on every step
(5) UNIFORM_CACHE_LEN does not grow after step 3 (D6: the content-keyed uniform cache is unbounded on main)
```
and say "all five clauses" wherever the text says "all four". Wherever a card asserts
`ARENA_PEAK_BYTES` against the activation term (6.5, 9.2, 10.2, Q14 text), the cap is
`ARENA_TRANSIENT_CAP = (4_305_000_000 - 4_140_417_024) * 1.05 = 172_812_125 B` (DERIVED).

## S2 — card 0.2 (per synth4 0.2)
Title becomes "Three byte counters in two loops, the uniform cache read for the first time".
`BLOCK_UPLOAD_BYTES` is RETIRED and renamed `BLOCK_OFFERED_BYTES` (not kept as "their sum"); N1 is
stated as a partition check true by construction; N2 the load-bearing number is
`BLOCK_COPIED_BYTES` 8-10 MB/token, `BLOCK_OFFSET_BOUND_BYTES` ≈ 4.14 GB, COPIED > 100 MB/token is
RED; per-operand codec from the GGUF header (not one codec per family); add `UNIFORM_CACHE_LEN`
gauge (read at `omega/src/metal.rs:2064`) printed per step, with the pre-registered prediction
"grows by ≈ op_count per token because every `Uniforms` carries `reduction_total`, a function of
`cached_len`; if flat, D6 is wrong and 6.5/9.2 re-scope". Add opens `omega/src/metal.rs:2056-2078`.
(Supersedes F1's wording for N1; keep F1's per-codec sentence.)

## S3 — cards 0.4 and 0.5 (per synth4 0.4, 0.5)
0.4: the script emits mean and CoV PER PHASE COUNTER (prepare, emit, block_upload, op_setup,
pipeline_lookup, encode_dispatch, readback) plus `greedy_pick_ms` (`generate.rs:1640-1670`);
asserts clause 3a at kv=0 (4,520,250,000) and 3b (4,371,150,000); `sealed-pass.sh`'s
`MACS_PER_TOKEN`/`WEIGHT_BYTES_PER_TOKEN_GB` named as magic numbers that belong in a config.
0.5: expect adds "a CoV band recorded for each phase counter and greedy_pick — the bands 3.2, 6.2
and 6.5 set kills outside of"; observe adds `UNIFORM_CACHE_LEN`, `greedy_pick_ms`. (Supersedes F2.)

## S4 — card 0.6 (per synth4 0.6)
Expect: the window-cleanliness proof is `READBACK_BYTES.snapshot_and_reset()` delta == exactly 4N
per run AND `gpu_device_ms < host_window_ms − readback_ms` on every run; the ceiling is taken at
simdgroup saturation (sweep the simdgroup count; R3/M5 52 → 147 GB/s from 256 → 8001). Opens add
`omega/src/metal.rs:1496` (`READBACK_BYTES`), `:1583` (`snapshot_and_reset`), `:2369-2373`; the Op
literal names all four fields with `name: Some("membw_copy")`. (Supersedes F3's readback sentence.)

## S5 — card 0.7
Add capture of `git -C $W diff --stat > $R/$wt-stat.txt` inside quarantine.sh and an expect line:
each `*-stat.txt` reconciled against R7's table (`proxima-wt-all` = 13 files +3859/−197, 6
untracked); a mismatch means the tree moved since the ledger.

## S6 — card 0.8 (per synth4 0.8)
There are TWO assertions: `plan_hits == 0` at `proxima-model-interop/src/bind.rs:3052-3055` and
`plan_misses == forward_calls_taken` at `:3056-3059`; the card restates BOTH as G5's formula and
adds two negative-path tests (an injected hit trips the first; an injected extra miss trips the
second). Every `bind.rs` cite in this card carries the crate path.

## S7 — card 0.9 (per synth4 0.9)
ai_docs is FOUR JSONL files (index, task-routes, invariants, plus the fourth the card lists on
opening `ai_docs/`), 95 lines total on main; 8 records not 7; a sixth invariant "the uniform buffer
cache is content-keyed and must not grow per token" with `evidence_required` = `UNIFORM_CACHE_LEN`
flat over steady steps; `jq -c .` parses all four files.

## S8 — card 1.1
Expect adds B4's diff-stat pins: `git diff --stat main..perf/cached-attention-streaming` shows
`physical.rs` +576, `bind.rs` +666, `discipline.md` +938; `git show 216d925 --stat` names
`prune_dead`/`dead_resolved_nodes`; a mismatch means the branch moved.

## S9 — card 1.3
Commands add `bash scripts/proxima-tensor-gate.sh` (welded) after the omega gate, per G9.

## S10 — card 2.2 (per synth4 2.2; supersedes F5's single-commit rollback)
Three commits, each green: (1) `bind` gains `packed: &BTreeSet<NodeId>`,
`correct_packed_matmul_layouts` becomes private, the `lib.rs:244` re-export is removed and the four
omega target call sites (`omega/tests/attn_multi_axis_tiled_gemm_parity.rs:223, :312`,
`omega/examples/attention_tiled_gemm_probe.rs:121`, `omega/examples/real_forward_packed_probe.rs:124`)
are migrated in the SAME commit; (2) `omega/src/metal.rs:1013` deleted and `cpu.rs:358` passes its
set; (3) 2.1's `EXPECTED RED` marker removed. Rollback: revert 3, then 2, then 1, in that order.
Expect N2: `grep -rn correct_packed_matmul_layouts --include='*.rs' proxima-tensor omega
proxima-model-interop | grep -v 'proxima-tensor/src/bind.rs'` returns 0 (crate-scoped, per G2).
Add `bash scripts/proxima-tensor-gate.sh` to commands.

## S11 — card 2.3
Expect: every phase counter within its 0.5 band, not only wall/gpu/op_count.

## S12 — card 3.1 (per synth4 3.1)
Opens/expect add: `kernel_cache_key` is `pub(crate)` and `#[cfg(any(test, all(feature = "metal",
target_os = "macos")))]` (`omega/src/msl.rs:730-736`), so N2 runs under a macOS metal build while
N4 runs under the alloc-tier build — two build configurations, stated; `entry_name(&BoundOp)`
(`msl.rs:1690`) keeps its signature and the `Route` must never enter the emitted name because
`kernel_cache_key` starts from it; add an `emit_ms` ±5% control (R13 0.81 ms).

## S13 — card 3.2
Greps scoped to crate directories (`proxima-tensor omega proxima-model-interop`), never `.` (G2).

## S14 — card 3.4
The elementwise bucket is quoted as 7.350 per-op = 6.85 batched-equivalent (÷1.073); the gate
step labels follow the script's `[n/6]` numbering.

## S15 — card 4.1 (per synth4 4.1; supersedes F8)
Do NOT add `proxima-build` as a build-dependency and do NOT add the axis to the workspace
`Profile` (its axes are a fixed workspace-runtime table, `proxima-build/src/profile.rs:50`,
`src/lib.rs:75-96`, and omega does not depend on it); instead `omega/build.rs` emits the SAME
directive form `proxima-build` uses (`cargo:rustc-check-cfg=cfg(omega_q4k_body, values(...))` +
`cargo:rustc-cfg=omega_q4k_body="..."` + `cargo:rerun-if-env-changed=OMEGA_Q4K_BODY`) beside
`emit_sizing_consts`, and the row names `proxima-build/src/lib.rs:205-234` and this reason so the
second mechanism is not unexplained. Exclusive cargo features would break FIVE of six gate steps
([2/6], [3/6], [4/6]-arm-2, [5/6]-arm-2, [6/6]), not one. The feature/cfg enters the CI matrix
(G13).

## S16 — card 4.2 / 4.3
4.2: add a `control` arm (`OMEGA_Q4K_BODY=main`) — already present as C; cells ≥ 20; the packed
bucket enters bands as 44.450 ÷ 1.073 = 41.43 batched-equivalent; predict band per synth4 §IV:
`gpu_exec_ms` [44.9, 48.6], `step_wall_ms` [55.9, 59.6]. 4.3: same deflated band; the three
target dirs are reclaimed after the row.

## S17 — card 5.1
`[spans] slots` is emitted UNCONDITIONALLY (it feeds the default upload path), with the
`omega/build.rs:99-104` convention (feature-gated consts only under `CARGO_FEATURE_*`) explained.

## S18 — card 5.2 (per synth4 5.2)
Opens/commands/blast add `proxima-model-interop/src/generate.rs:1804-1860` — `forward_node_values`
has its OWN `LayerCache::new()` and its own `named_blocks` assembly; the arena re-types it too, and
N6 asserts `forward_node_values` runs with the feature on. Memory caps per S1.

## S19 — card 6.1 (per synth4 6.1; supersedes F12's `+5` and `{1,2,34}`)
The `+3` `Vec::with_capacity` hint is LEFT ALONE (it already under-counts, it has no observable);
the observable is `named_blocks.len() == block_node_ids(program).len()` at BOTH assembly sites
(`generate.rs:1332` and `forward_node_values` `:1858`). `op_count` delta ∈ {1, 2, 33, 34} with each
value's meaning (+1 both fuse — the best outcome; +2 Iota + unfused Greater; +33 Iota + 32 unfused
Selects; +34 neither); predict +1. The mask literals name every field:
`Op::Input { dtype: DType::Float32, shape: vec![], name: Some("kv_valid_len".into()) }`,
`Op::Iota { dtype: DType::Float32, extent: Extent::Symbolic(1) }`, and the two `Op::Elementwise`
with `dtype`, `body`, `operands: vec![...]`, `name`. Reprove filter `test(kv_tail_mask)`.

## S20 — card 6.2 (per synth4 6.2)
BOTH assertions (`:3052-3055` and `:3056-3059`) become `#[cfg(not(feature="kv-capacity-bucket"))]`
and a new PAIR is added under the feature; inverting only the first leaves the second red. Observe
adds `UNIFORM_CACHE_LEN` (a bucketed key makes uniforms token-invariant, so the cache should stop
growing — an observable here even though 6.5 owns the lever).

## S21 — card 6.3
δ_b is this card's MEASURED output, carried to §IV's board at both endpoints [0.3, 2.8], never
capped; each bucket value is prebuilt into its own target dir (`$TD/<bucket>`) then run
interleaved with the control; cells carry `UNIFORM_CACHE_LEN`; the 256-bucket loss is +18.6 ms
batched-equivalent (DERIVED).

## S22 — card 6.4
Expect adds "`UNIFORM_CACHE_LEN` stops growing (the bytes become token-invariant)".

## S23 — card 6.5 (per synth4 6.5; supersedes F13)
Uniforms are NOT written in place through `UNIFORM_BUFFERS` and NOT left to the content cache:
the card adds a plan-owned `Vec<MetalBuffer>` of one uniform buffer per plan position
(`PlanUniforms`), allocated once with the plan, written by `encode_op` at its own index, bypassing
`upload_uniforms` on the plan path; the content cache stays for the non-plan path and is BOUNDED
in this same commit (`[spans] uniform_cache_entries`, LRU) so clause 5 is assertable. N2 becomes
`PLAN_UNIFORM_WRITES == op_count` per steady step and `UNIFORM_BUFFER_REUSES` FALLS to ~0 on the
plan path (the expected direction); N3 `UNIFORM_CACHE_LEN` bounded; N6 gains "two ops with
identical uniform bytes get DISTINCT plan-owned buffers"; `ARENA_PEAK_BYTES` asserted against
`ARENA_TRANSIENT_CAP = 172_812_125 B`; predict band per synth4 §IV `step_wall_ms` [49.2, 53.3] +
δ_b; the re-derivation hook names the `lookup.indices` edge at `metal.rs:1132-1137`; rollback keeps
the cache bound (a leak repair, §15). Observe adds `PLAN_UNIFORM_WRITES` (NEW), `UNIFORM_CACHE_LEN`.

## S24 — card 6.6
Expect/predict/observe add `greedy_pick_ms` (the 1.6 ms residual gets a counter; pre-registered
≥ 1.0 ms ⇒ §VIII.27's un-park condition); predict band `step_wall_ms` [49.2, 53.3] + δ_b,
`gpu_exec_ms` [44.9, 48.6] + δ_b, ratio 2.83–3.20x; kill threshold 58.1 (5.2's band top).

## S25 — card 7.1 (per synth4 7.1; supersedes F14)
Only POLICY consts move: `[packed_row_block] rows_per_group, lanes_per_block` and
`[cooperative_reduce] max_threads, vec_width`, all emitted unconditionally. `TILE_DIM`
(`msl.rs:1029-1030`, `#[cfg(feature = "metal-tiled-gemm")]`, doc: "fixed 8x8 by the MSL type
itself") and `TILED_GEMM_NSG` (`:1046`, doc: "a value other than 4 would need a different kernel
body") stay bare as three named §12 exceptions with `SIMD_WIDTH`, their own docs quoted on the row.
N1's grep result lists them explicitly; N4 adds "the six-step gate green including [4/6]-arm-1's
`--lib --no-default-features --features alloc` clippy". Blast: `msl.rs` const sites (two, not
three).

## S26 — card 7.2
The cooperative bucket enters bands as 9.113 ÷ 1.073 = 8.49 batched-equivalent; band [6.8, 7.6];
predict `gpu_exec_ms` [43.2, 47.8] + δ_b, `step_wall_ms` [47.5, 52.5] + δ_b; kill adds "one measured
loss on the real graph retires this lever; it does not get a second attempt".

## S27 — cards 8.1 / 8.3 (per synth4 8.1, 8.3; supersedes F15's "8.1 adds the parameter" and "within 10%")
Phase 8 is SCHEDULED AFTER 11.2 (11.2's `depends_on` drops 8.3; the phase note says so). 8.1:
every classification row carries its MEASURED line count; the STRUCTURE total for `Elementwise`
on main is 423 (msl 148 + wgsl 137 + cuda 138) and the TEXT total 408; `entry_name(&BoundOp)`
keeps its signature (no `Route` parameter) and the row states the route must not enter the emitted
name. 8.3: the continuation gate is (a) byte-identity AND (b) the number of STRUCTURE rows from
8.1's table now deleted from all three backends equals 8.1's count for this kind, with the measured
`wc -l` delta recorded as a fact beside it; predict the fall lands in [300, 423]; the earlier `≥600`
was unreachable by ~180 lines and measured relocation, not duplication removed.

## S28 — card 9.1 (per synth4 9.1; supersedes F16's Computed→Affine rewrite mechanism)
The mechanism is restricted to a static nonzero `axis.offset` on an AFFINE write map: a `Computed`
out_map never reaches `project_output_shape` (`shape.rs:183-201` short-circuits data-dependent
maps to `scatter_output_shape` at `:200`; `project_output_shape` is called only at `:205` and opens
`out_map.affine()` at `:474`), so `Computed`/scatter/`scatter_output_shape`/`build_scatter_out_layout`
/`IndexMap::scatter`/`as_gather_from_output` are ALL untouched; `project_output_shape` accepts
`[term] if term.coeff == 1` with a nonzero `axis.offset` and returns `iter_extents[axis] + offset`;
`layout_of` (`bind.rs:1594-1606`, reached from `bind.rs:962` for Affine) ALREADY folds
`offset * stride` into `out_layout.base`; injectivity is an interval-overlap check at bind over the
producers of one destination node (`[o1, o1+e1)` vs `[o2, o2+e2)`), not a prover. Delete the
backward-walk prover text and the Computed-rewrite text. Expect table — ACCEPT: offset 0
(byte-identical); offset k>0 with coeff 1; two producers at disjoint ranges; abutting ranges.
REJECT: overlapping ranges (named error); `coeff != 1`; negative offset; `o + e` beyond the declared
destination extent. Add the autograd round-trip test: differentiate a placed `Reduce` and assert
the adjoint reads the gradient at `base + offset` through `adjoint.rs:835`'s existing
`out_map_as_operand` path with NO autograd edit (an edit needed is a finding that grows the card).
Keep F16's autograd gate command and rollback ordering. Reprove filter `test(write_offset)`. Title
becomes "A nonzero static offset on an Affine write map; `Computed`/scatter untouched".

## S29 — card 9.2 (per synth4 9.2; supersedes F17's separate `dynamic_bases` MetalBuffer)
`dynamic_bases: Vec<(position, i64)>` is patched into 6.5's plan-owned per-position uniform
buffer (`PlanUniforms`), never through `upload_uniforms`/`UNIFORM_BUFFERS`; N6 `UNIFORM_CACHE_LEN`
does not grow (replace F17's N7 gauge wording with this); N7 goldens + fingerprints re-captured
for the ON arm (keep); predict states `UNIFORM_BUFFER_REUSES` unchanged ONLY because the plan path
no longer consults that cache; the operand-set edge (`lookup.indices` at `metal.rs:1132-1137`) is
named in the re-derivation hook; the `+ dynamic_bases[slot]` kernel term from F17 is dropped (the
plan-owned uniform carries `out_base` directly).

## S30 — cards 10.1 / 10.2
Each sweep value is prebuilt into its own target dir before any measurement; 10.2's memory kill
uses `ARENA_TRANSIENT_CAP`; family GB/s use 0.2's per-op-codec bytes.

## S31 — card 10.3
Add: the card produces the first CoV-bearing, mutex-serialised, non-`--quick` local cell and
cross-checks the CI `--quick` baseline (`.github/workflows/proxima-tensor.yml:191-205`).

## S32 — card 10.4
Installing `onnxruntime` into the dedicated venv is an owner-authorized action (named on the card);
the mutex is held with `--wait 14400` and no other card is scheduled while it runs; the path is
repo-root `scripts/onnx_reference/bench.py:96`.

## S33 — card 11.1
ai_docs is four JSONL files; the grep `grep -c "3.54x\|17.470\|228.9" proxima-tensor/docs/discipline.md`
goes from 0 before landing to ≥ 3 after (main's log does not know the 2026-09-02 session
happened); the row format follows ROW 233's.

## S34 — card 11.2 (per synth4 11.2)
`depends_on` drops 8.3 (Phase 8 runs after the board); every number carries a provenance tag
MEASURED / DERIVED / CHOSEN; `gpu_exec_ms` named host-ticks and `gpu_device_ms` named
GPU-timestamp; `greedy_pick_ms` on the board; predict `step_wall_ms` [46.8, 57.3], `gpu_exec_ms`
[43.5, 50.6], ratio 2.67–3.27x with δ_b at its 6.3-measured endpoints; kill > 59.0 unchanged.

## S35 — every card: crate-qualified `bind.rs` (G0)
Every bare `bind.rs:` cite in the cards becomes `proxima-tensor/src/bind.rs:` for lines ≤ 1720
(`:95-98`, `:200-215`, `:221-264`, `:962`, `:1011`, `:1594-1606`, `:1618-1707`, `:1648`, `:1718`)
and `proxima-model-interop/src/bind.rs:` for the harness lines (`:2719`, `:2765`, `:2797-2803`,
`:2994-2998`, `:2999-3003`, `:3046`, `:3051`, `:3052-3059`, `:3084`, `:3103`, `:3122`). A cite you
cannot classify is left as is and listed in your report.
