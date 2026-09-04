# Round-4 fixes to apply to the expanded cards (cards-all.canon.md)

Every item below is a bounded text change. Apply each in place; do not restructure cards; do not
touch cards not named. Every new number is tagged. Mark each applied change with the literal tag
`[round-4 fix <id>]` at the end of the sentence you changed or added, so the edits can be counted.

## F1 — card 0.2 (byte counters)
- expect N1: replace "and the steady-token sum equals R13's **4,147,777,096**" with: "and the sum
  is reported under a NEW name, `BLOCK_DECLARED_BYTES` (it equals the old counter's 4,147,777,096
  on main by construction — that number is the artefact, not a target); `BLOCK_UPLOAD_BYTES` is
  redefined as `BLOCK_COPIED_BYTES` alone, so the headline number changes from 4.147 GB/token to
  the copied bytes (R13: `copying_uploads=4`, the KV's 8-10 MB) [round-4 fix F1]".
- expect N3: after "within **1%** of R13's shape-derived column" add: "— where the column is
  re-derived per tensor from the codec the GGUF header declares for THAT tensor (Q4_K_M
  checkpoints carry Q6_K for some `ffn_down`/`attn_v` layers, which is why R13's `ffn_down`
  34.00 MB differs from `ffn_up`'s 33.05 MB at identical element counts; a uniform
  `rows*k*0.5625` would put the two 2.9% apart, inside the 1%-5% band this card assigns no action
  to). The per-tensor codec is read, never assumed [round-4 fix F1]".

## F2 — card 0.5 (re-seal)
- expect: add "N5 per-phase CoV across the 5 runs for EVERY phase counter (`prepare_ms`,
  `emit_ms`, `block_upload_ms`, `op_setup_ms`, `pipeline_lookup_ms`, `encode_dispatch_ms`,
  `readback_ms`) and for `greedy_pick_ticks` (`generate.rs:1640-1670`) — these bands are what
  3.2's and 6.5's per-phase kills are set outside of; a phase with no band cannot carry a kill
  [round-4 fix F2]".
- observe: add "`greedy_pick_ticks` — the ~1.6 ms residual R13 attributes to sampling and cache
  append gets a counter, so §VIII.20's un-park condition is measurable [round-4 fix F2]".

## F3 — card 0.6 (roofline)
- In the `edit:` line, replace the Op literal with the real field set:
  "`Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: vec![(src,
  <the affine identity IndexMap, built the way causal_mask builds its maps at spec.rs:823-845>)],
  name: None }` (`op.rs:191-196` — `dtype` and `name` are required fields, `operands` is a
  `Vec`) [round-4 fix F3]".
- expect: replace "`readback_bytes == 0` inside the timed window (the assertion that proves the
  window is clean)" with "the timed window is `GPUEndTime − GPUStartTime` of the command buffer,
  which by construction ends before `finish` runs; `READBACK_BYTES` (`metal.rs:2369-2373`) is a
  monotonic counter, so the assertion is a DELTA: `READBACK_BYTES_after − READBACK_BYTES_before ==
  4N` per run (the copy's whole output was read back, outside the window), and `readback_ticks`
  are not inside `[GPUStartTime, GPUEndTime]` [round-4 fix F3]".

## F4 — card 0.9 (cardinality, ai_docs)
- commands: add "(e) open `.github/workflows/proxima-tensor.yml:176-205` (the `omega-gate.sh`
  job and the `omega-compare-bench` job that runs `cargo bench -p omega --features metal --bench
  metal_vs_cpu -- --quick` on `macos-latest`) and record in `row-protocol.md` the rule: every
  feature this plan declares (`metal-kv-resident`, `kv-capacity-bucket`,
  `metal-plan-stable-buffers`, `kv-scatter-write`, `attention-single-range`,
  `metal-wide-reduce`, `metal-packed-split-k`) and the `omega_q4k_body` cfg is added to the CI
  matrix by the card that lands it — the CI job-set is not the gate-glob (memory:
  hand-picked gates miss CI's config) [round-4 fix F4]".

## F5 — card 2.2 (bind owns the packed layout) — REWRITE the commands and expect
- Replace move (1) "`packed_operands_of` descends into proxima-tensor …" with: "(1) NOTHING
  moves down. `PackedOperands` is `BTreeMap<NodeId, PackedCodec>` at `omega/src/msl.rs:656` and
  `PackedCodec` is an omega enum at `:591`, so proxima-tensor cannot name the return type of
  `packed_operands_of`; and it does not need to — `correct_packed_matmul_layouts` already takes
  `&BTreeSet<NodeId>` (`bind.rs:1648`), which `metal.rs:1013` builds as
  `packed_operands.keys().copied().collect()`. `bind` gains exactly that parameter, `packed:
  &BTreeSet<NodeId>`, and the CPU driver (`cpu.rs:358`) passes the set it derives from its own
  `QuantizedBlock`s [round-4 fix F5]".
- Move (2): after "the public function **removed**, not deprecated (§15)" add: "— together with
  the re-export at `proxima-tensor/src/lib.rs:244` and its three omega TARGET call sites
  (`omega/examples/attention_tiled_gemm_probe.rs:121`,
  `omega/examples/real_forward_packed_probe.rs:124`,
  `omega/tests/attn_multi_axis_tiled_gemm_parity.rs:223, :312`), which `omega-gate.sh [2/6]
  --all-targets --all-features` builds, plus the doc references at
  `proxima-model-interop/src/bind.rs:719, :729` and `hf_bind.rs:315`; all in the same commit as
  the signature change, because a commit that deletes the function and leaves the examples is
  not a green bisect point [round-4 fix F5]".
- expect N2: replace the "exactly 1 hit" clause with: "`grep -rn correct_packed_matmul_layouts
  --include='*.rs' . | grep -v 'proxima-tensor/src/bind.rs'` returns **0** — main returns 22
  hits in 9 files today; hits inside `bind.rs` (the private definition, its doc and its own
  tests such as `:2267`) are expected and are not counted [round-4 fix F5]".
- rollback: replace with "one `git revert` of the single commit (signature + deletion + the three
  targets + the re-export are one logical change: 'bind owns the packed layout'); 2.1 returns to
  its documented RED [round-4 fix F5]".

## F6 — card 3.2 (census)
- kill, first bullet: replace "the budget is exceeded" with "the budget is exceeded — where the
  budget is `max(0.0235 ms, 2 × CoV(encode_dispatch_ms) as 0.5 measured it)`, so the kill sits
  outside a band that exists (G6) [round-4 fix F6]".

## F7 — card 3.3 (backends)
- expect: replace "3 backends × 4 `BoundOpKind` (+ `Keep::Scan`) = **15 cells**" with "3
  backends × 5 emission shapes (the 4 `BoundOpKind`s plus `Reduce` under `Keep::Scan`, which is a
  field of `Reduce`, not a fifth kind — `BoundOpKind` stays 4) = **15 cells** [round-4 fix F7]".

## F8 — card 4.1 (build-time selector)
- commands: replace "and `emit_profile_cfg()` in `build.rs` emitting
  `cargo:rustc-check-cfg=…` + `cargo:rustc-cfg=…` + `cargo:rerun-if-env-changed=OMEGA_Q4K_BODY`"
  with "and, instead of hand-rolling the directives, `proxima-build`'s existing
  `emit_cfg_check_directives()` / `emit_cfg_directives(&Resolved)`
  (`proxima-build/src/lib.rs:205-234`, the workspace's own §8 profile input) — `omega/build.rs`
  gains `proxima-build` as a build-dependency (`cargo add --build proxima-build -p omega`, never a
  manual Cargo.toml edit); the three values `main`, `mask_fma`, `pair_dot` are the profile's
  domain [round-4 fix F8]".

## F9 — card 4.2 (bake-off)
- commands: after "`OMEGA_Q4K_BODY` selects the arm; features are identical across arms." add:
  "**Selecting the arm is a rebuild** (the value is a `rustc-cfg`), so the arms are NOT rebuilt
  between measurements: build the three test binaries once, one per value, with `--no-run`
  (`cargo test --release -p proxima-model-interop --features metal,instrument --lib --no-run`
  under `OMEGA_Q4K_BODY=<value>` and a distinct `CARGO_TARGET_DIR` per value:
  `<worktree>/target-main`, `<worktree>/target-mask_fma`, `<worktree>/target-pair_dot`), record
  the three binary paths, then interleave the prebuilt binaries A B C A B C with `--exact
  --nocapture --ignored bind::real_openchat_file::<test>` (the technique of `discipline.md:9904`);
  the mutex is held per run, never across a build [round-4 fix F9]".
- memory gate: add "three target dirs ≈ 3× one release build on disk; recorded, not gated".

## F10 — card 5.2 (KV arena) — memory cap arithmetic
- Everywhere this card states `DEVICE_CAP_BYTES … + 41_943_040` replace `41_943_040` with
  `45_165_952` and add once: "the activation term is `2 × (R13 steady maximum 4.163e9 −
  4,140,417,024 mapping) = 2 × 22,582,976 = 45,165,952 B` (DERIVED; the factor 2 is the most
  6.5's arena may grow live intermediates before it must liveness-partition; the earlier
  41,943,040 was 40 MiB with no derivation) [round-4 fix F10]".
- Replace the DERIVED total "4_316_577_792" with "4_319_800_704" wherever it appears in this
  card (= 4,140,417,024 + 512×262,144 + 45,165,952).
- Add the prefill clause: "clause (3) is a STEADY-STATE cap over steps 3..S like clauses (1) and
  (2); prefill (step 0) carries its own cap `PREFILL_CAP_BYTES = 4,305,000,000 (R13's measured
  prefill peak, 3 significant digits) + kv_capacity_tokens × 262,144 + 45,165,952` = 4,484,383,680
  B at 512 (DERIVED) — on main today prefill peaks at 4.299-4.305e9 and would breach a cap that
  did not distinguish it [round-4 fix F10]".

## F11 — every other card whose memory gate quotes `41_943_040` or `4_316_577_792`
- Replace `41_943_040` → `45_165_952` and `4_316_577_792` → `4_319_800_704` in the memory-gate
  clause lines of every card (0.x through 11.x), and append to the FIRST such clause in card 0.2
  the sentence "clause (3) is steady-state (steps 3..S); prefill has its own cap, see 5.2
  [round-4 fix F11]". Do the same replacement in the row/report text of those cards.

## F12 — card 6.1 (bucket + mask)
- Replace the `Op::Iota { extent: Extent::Symbolic(1) }` literal with `Op::Iota { dtype:
  DType::Float32, extent: Extent::Symbolic(1) }` (`op.rs:229`) [round-4 fix F12].
- Replace the `+ 3 → + 4` sentences with: "the `Vec::with_capacity(… + 3 + …)` hint at
  `generate.rs:1316` already under-counts — the code pushes FOUR fixed blocks (`ids` `:1322`,
  `eps` `:1332`, `rope_cos` `:1333`, `rope_sin` `:1334`); it becomes `+ 5` with the new leaf. It
  is a capacity hint with no observable; the observable is `named_blocks.len() ==
  block_node_ids(program).len()`, asserted on the real program [round-4 fix F12]".
- opens/commands/blast: add "`generate.rs:1838-1860` — `forward_node_values`'s OWN
  `named_blocks` assembly (a second site that pushes `ids`/`eps`/`rope_cos`/`rope_sin` and the
  32 KV names from an `empty_cache` and constructs `LayerCache::new()`); the leaf is pushed here
  too, or `InputCountMismatch` (`metal.rs:984-990`) fires on this public entry the moment the
  leaf lands; a test calls `forward_node_values` with the feature on [round-4 fix F12]".
- expect N4 and the kill: replace "rises by **exactly 2**" / "delta ∉ {2, 34}" with "rises by
  **+1 or +2** — `Op::Input` is never a bound op (`BoundOpKind` has no `Input`, `bind.rs:221-264`),
  the `Iota` is one bound op, and the `Greater` may fuse into the `Select` which may fuse into the
  score body; +1 is the fully-fused case and is the BETTER outcome, not a kill; delta ∉ {1, 2, 34}
  is the kill [round-4 fix F12]".

## F13 — card 6.5 (device arena)
- Replace "one uniform buffer per position written in place (the `:2069-2078` mechanism already
  proves the shape is legal)" with: "**uniforms are NOT written in place and NOT per position.**
  `UNIFORM_BUFFERS` (`metal.rs:2055-2078`) is a content-keyed dedup cache,
  `BTreeMap<Vec<u8>, MetalBuffer>` keyed by the uniform bytes and shared across ops with identical
  bytes; writing through a shared buffer would corrupt every co-keyed op. Because a stable plan's
  uniform bytes are a pure function of the bound op, the cache already hits once the plan is
  stable — so Q13 is answered FIRST: on 6.3's tree, `UNIFORM_BUFFER_REUSES` per steady step is
  read and expected to equal `op_count` with NO change from this card. This card scopes to
  OUTPUT buffers only [round-4 fix F13]".
- expect N2: replace with "`UNIFORM_BUFFER_REUSES == op_count` per steady step on the PRE-card
  tree (6.3's) and unchanged by this card; if it is below `op_count` pre-card, the reason is
  named (which ops' bytes differ step to step) before any arena work [round-4 fix F13]".
- blast: remove "uniform" from the arena's scope; observe: keep `UNIFORM_BUFFER_REUSES` as a
  before/after invariant.

## F14 — card 7.1 (geometry config)
- Replace "`[packed_row_block] rows_per_group, lanes_per_block, tile_dim`" with
  "`[packed_row_block] rows_per_group, lanes_per_block`; `[tiled_gemm] nsg, tile_dim` — `TILE_DIM`
  (`msl.rs:1030`) is `#[cfg(feature = "metal-tiled-gemm")]` (`:1029`) and belongs to the tiled
  path; its generated const is emitted ONLY when Cargo reports that feature active, exactly as
  `omega/build.rs:99-104` documents for the existing `[tiled_gemm]` key, or the alloc-tier clippy
  arm (`omega-gate.sh:56`, `-D warnings`) goes red on an unreferenced const [round-4 fix F14]".

## F15 — card 8.3 (emitter core, one kind)
- Replace every "≥ 600" / "≥600" continuation and predict clause with: "8.1's `dialect-map.md`
  pre-registers, per kind, the line count of the STRUCTURE rows it classifies (for `Elementwise`
  on main: `bindings` + `push_body_steps` + `operand_read` + `render_elementwise` = 148 msl + 137
  wgsl + 138 cuda = 423 lines, READ this session; TEXT rows stay 3× by design), and the
  continuation gate is: N1 byte-identity holds AND the measured `wc -l` fall is within 10% of
  8.1's pre-registered STRUCTURE count for this kind (a pure relocation into the core module
  without deleting duplicates falls short and is the finding). The earlier `≥ 600` was
  unreachable — the maximal fall is 423 [round-4 fix F15]".
- Add to opens: "`entry_name(resolved: &BoundOp) -> String` (`msl.rs:1690`, `wgsl.rs:538`,
  `cuda.rs:464`) takes no `Route` today; 8.1's method 4 adds the parameter, and N1 requires that
  the route never enters the emitted name [round-4 fix F15]".

## F16 — card 9.1 (injectivity) — the mechanism, restated
- Replace the sentence "Accepting `coeff == 1` **plus a nonzero `axis.offset`** in
  `project_output_shape` is a **different** change …" (in the reconciliation paragraph) and the
  `edit: proxima-tensor/src/shape.rs:469-485` command with the following mechanism: "A
  `Computed` out_map never reaches `project_output_shape`: `infer_reduce` (`shape.rs:186`)
  branches on `out_map.is_data_dependent()` at `:200` into `scatter_output_shape` (`:495`), and
  `project_output_shape` is called only at `:205` for an `Affine` map (its body opens
  `out_map.affine()`, `:474`). And the `Computed` form's `offset` slot at `gathered_dim` already
  carries the scatter's DESTINATION EXTENT (`map.rs:175-197`, documented at `shape.rs:168-181`),
  so it cannot also carry a write base. Therefore the ACCEPT path is a REWRITE at bind:
  `Computed{indices…}` whose chain proves `index(i) = coeff·i + base` is replaced by the
  equivalent `Affine` map (`coeff` in the term, `base` in `axis.offset`), and ONLY the affine
  path of `project_output_shape` (`:469-485`) is extended to accept `coeff == 1` with a
  loop-invariant nonzero offset. The REJECT path leaves the `Computed` map, its
  destination-extent convention and `scatter_output_shape` untouched. The destination extent of
  a placed affine write is the DECLARED destination leaf's extent (for the KV write, the
  symbol-1 leaf); bind asserts `offset + iter_extent <= destination_extent`, which is the
  bounds check `bounds_check` (`:441-467`) performs on the read side [round-4 fix F16]".
- opens: add "`proxima-tensor/src/shape.rs:186-205` (`infer_reduce`, the branch), `:495`
  (`scatter_output_shape`), `:168-181` (the destination-extent-in-offset convention);
  `proxima-autograd/src/adjoint.rs:784` (`differentiate_reduce`), `:1033` (the
  `as_gather_from_output` call), `proxima-autograd/src/error.rs:73-88`
  (`ScatterOutputUnsupported`); `scripts/proxima-autograd-gate.sh` [round-4 fix F16]".
- commands: add, after the omega gate: "`cd <worktree> && CARGO_TARGET_DIR=<worktree>/target
  bash scripts/gpu-measure-lock.sh <lock> --wait 5400 -- bash scripts/proxima-autograd-gate.sh
  2>&1 | tee <worktree>/runs/9.1-autograd-gate.log; echo "EXIT=${PIPESTATUS[0]}"` — with
  `ran_count`/`passed_count` recorded; either 0 is RED [round-4 fix F16]" (use this card's real
  absolute worktree path and lock path).
- rollback: replace with "`git revert`; but 9.2 and 9.3 depend on this card and restructure
  `spec.rs`/`generate.rs` under their features — reverting 9.1 after either has landed means
  reverting them first, in reverse topological order (§VII); this card is the one point in the
  plan where a revert is a three-card unwind, stated here rather than as a one-liner
  [round-4 fix F16]".

## F17 — card 9.2 (per-token write base)
- Replace the "the one thing this plan may mint" paragraph's mechanism sentence ("patched into
  the uniform bytes at `upload_uniforms`") with: "**the per-token base must not enter the
  content-keyed uniform cache.** `UNIFORM_BUFFERS` (`metal.rs:2055-2078`) is keyed by the uniform
  BYTES; a base that changes every token changes the key, misses the cache, allocates a fresh
  buffer and inserts a new entry every token — unbounded host-heap growth at ~100 B/entry that
  MG-3's 1 MB/step slope would not catch. So `dynamic_bases` is a SEPARATE, plan-owned
  `MetalBuffer` of one `i64` per placed op, bound at its own argument index, written in place per
  token (single owner, never inserted into `UNIFORM_BUFFERS`), and the placed-write kernel reads
  `out_offset = u.out_base + dynamic_bases[slot]` — one added term in the emitted MSL for placed
  writes only [round-4 fix F17]".
- expect: add "N7 `UNIFORM_BUFFERS.len()` (a NEW instrument-gated gauge) is constant across steady
  steps in the ON arm — the map must not grow per token; a growing map is the cache-miss failure
  and a KILL [round-4 fix F17]".
- Add "N8 the ON tree's 2.1 fingerprint vectors and 3.1 goldens are re-captured in this commit
  (the output-set collapse renumbers `NodeId`s far more than 6.1's one leaf did; HC-1's fix
  applies here too) [round-4 fix F17]".
- Add to the re-derivation paragraph: "the operand set changes shape as well as the output set:
  a folded write's `indices` node loses its only consumer (`bound_op_retirement` at
  `metal.rs:1132-1137` keys `last_use` on `lookup.indices`), so the partition is re-run on the
  new operand graph, not only on the new output set [round-4 fix F17]".

## F18 — card 9.3 (single-range)
- expect: add "N6 the ON tree's 2.1 fingerprint vectors and 3.1 goldens re-captured in this
  commit — deleting the two-range combine (`spec.rs:2596-2720`) renumbers every downstream
  `NodeId` [round-4 fix F18]".

## F19 — card 10.3 (torch-MPS)
- Replace "run the omega `metal_vs_cpu` bench **for the first time**" with "run the omega
  `metal_vs_cpu` bench in FULL mode for the first time — CI already smoke-runs it with
  `-- --quick` on `macos-latest` (`.github/workflows/proxima-tensor.yml:191-205`, the
  `omega-compare-bench` job) and `omega-gate.sh [2/6]` builds it, so the bench compiles and runs;
  what does not exist is a RECORDED cell (the bench's own doc still says UNRUN, R9 — a stale doc,
  fixed by this card) [round-4 fix F19]".
- kill: replace "The omega bench does not build under `--features metal` ⇒ that is the finding
  and fixing the registration is the card" with "the full-mode run's numbers disagree with the
  `--quick` CI run's by more than the measured CoV ⇒ `--quick` is not a smoke of the same cell;
  record both [round-4 fix F19]".

## F20 — card 11.1 (docs)
- commands: add "(d) the CI matrix: every feature and cfg this plan landed is present in
  `.github/workflows/proxima-tensor.yml`'s job list, checked by grep per feature name; a feature
  with no CI job is RED (memory: the gate-glob is not the CI job-set) [round-4 fix F20]".
