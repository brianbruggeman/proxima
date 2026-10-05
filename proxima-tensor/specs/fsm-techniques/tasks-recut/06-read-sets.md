# slice 6 cards (re-cut): the read hook

anchors read at main 4b4be6cf (full sha 4b4be6cf78bcd536fecdd4e4620a9769b9b7364c), except every anchor marked `at a7c08c4c`, which was
re-read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0). Read each with
`git show main:<path>`; the working tree is not the source. Paths are relative to the proxima repo root.

Governing spec: `proxima-tensor/specs/pipeline-as-data/SPEC.md` (stage H12 "read", stage H7 "specialize") and
`sketches/08-rectified-sparse-attention.md` (hook gaps for the read stage and the fusion matcher). Rules: CARDS.md applies to
every card. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_<n>` and removes it when done. Logs go under
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards06/`.

## what this file builds

The hook, not the techniques. The read stage decides which cache rows one decode step attends. After this file:

- the two-range cached attention builder takes a read function. For every attention layer it hands the function the layer's
  grouped rotated query planes and the cached-side mask, and takes back a mask (and optionally a per-row weight). Dense is the
  identity function, so the dense program is node for node what it is today;
- one library function, `read_skip_operand`, is the generic leaf: a per-layer per-row skip flag the host fills each step, which
  the config-selected `ReadSpec::Operand` binds;
- the decode loop binds that leaf from a pure read rule (a function of layer index and cached row count) the caller sets on the
  loaded model; under `ReadSpec::Dense` nothing is declared, bound or counted;
- the cached-attention fusion keeps the fused path under the skip operand and the kernels skip flagged rows before the key read.

A technique appears only as a proof that the hook expresses it, in a test or an example:
block top-n read with per-dimension key bounds (cards 6.5 and 6.6) and importance-weighted sampled read (cards 6.3, 6.7, 6.21, 6.22, 6.8).
None of that selection or sampling code lives in a library crate.

## prerequisites outside this file (an executor whose premise is false stops and reports)

- needs: the slice 3 re-cut card that adds `top_fraction_mask` to `proxima-tensor/src/spec/primitives.rs` (absent on main:
  `git grep -n top_fraction_mask main` prints nothing). Expected signature
  `pub fn top_fraction_mask(program: &mut Vec<Op>, scores: NodeId, keep_count: NodeId, keep_rows: Option<NodeId>) -> Result<NodeId, TensorError>`
  with `keep_count` a rank-0 input and the tie rule "the lower index wins". Cards 6.5 and 6.6 use it. Its id is FT3.2 or the id the
  slice 3 id map gives.
- needs: the slice 2 re-cut card that adds `ServingConfig.attention.read: ReadSpec` (absent from Rust on main: `git grep -n ReadSpec main -- '*.rs'`
  prints nothing; the only 3 hits without the `*.rs` scope are prose in `proxima-tensor/specs/pipeline-as-data/sketches/08-rectified-sparse-attention.md`, lines 15, 188 and 256). This file assumes `ReadSpec` is a `Copy`, `PartialEq`, `Debug` enum whose `Default` is `Dense` and which has one
  more unit variant, `Operand` (the decode step binds the read operand from the model's read rule). If the slice 2 re-cut names
  these differently, cards 6.9 to 6.14, 6.19 and 6.20 stop and report; the fix is a rename in this file, never an improvisation.
  `ServingConfig` is `Copy` (`proxima-model-interop/src/serving.rs::ServingConfig (~line 719)`), so `ReadSpec` holds no list and no
  float; the read rule is a function on the model, not a config field.
- no tap card is needed: cards 6.21 and 6.22 tap one layer's rotated key planes, value rows and grouped rotated query planes with this file's own
  read hook (card 6.3: the hook receives the grouped rotated query planes, and the builder's returned cache roots are the rotated key
  halves and value) and `LoadedModel::forward_node_values_on_backend`, which evaluates the loaded model's own program. No card in
  `tasks-recut` taps those planes (the layer residual tap of the cache blending file is a different node), and 6.21 does not depend
  on one.

## decided in this file (and why)

- The hook is a function argument, not a config enum in the tensor crate. The read decision needs the layer's own query, which
  exists only inside the graph (sketch 08, hook gap on the data-dependent visibility term), so a leaf the host fills cannot express
  every read rule. A function argument can: it may declare leaves, build any ScalarOp chain, and return the mask. The leaf-only form
  (`read_skip_operand`) is one instance of that function, and it is what `ReadSpec::Operand` selects.
- Read function signature, positional (no new type, the call site is the same either way):
  `FnMut(&mut Vec<Op>, u32, Extent, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>` with arguments
  `(program, layer, key_extent, query_even, query_odd, cached_mask)`. `key_extent` is the extent of the layer's cache rows
  (`Extent::Symbolic(1)` for a full layer, `SLIDING_KV_SYMBOL` for a ring layer). `query_even` and `query_odd` have axes `sugi`
  (query row, kv head, group member, pair index). `cached_mask` is rank 2 over `st` (1.0 = skipped, the convention of
  `causal_mask_cached_windowed`). The returned mask is rank 2 over `st`. The optional weight is rank 1 over cache rows `t` and
  multiplies the cached exponentials before both softmax sums.
- The mask is shared by every kv head and group member of a layer (`"st->stug"` at the existing call site). A per-head read is out
  of reach of this hook and is not built.
- The hook applies to every attention layer, including a layer that shares its donor's K and V. It is called once per layer.
- Block read applies to decode steps of one query row in the proof; prefill and a multi-row verify read densely.
- Granite MoE: not a card here. A read hook on a single-range cached engine is an engine change (the single-range cached block is
  never masked; `proxima-tensor/src/spec/descriptor.rs` doc, sketch 08 hook gap on the visibility term). The hook is proven on
  gemma4 only: E2B (dense) and 26B (MoE). No qwen checkpoint appears in any card.

## old to new id map (the previous cut of slice 6 is tasks/06-read-sets.md)

| old | verdict | new |
|---|---|---|
| FT6.1 | dropped | none: `block_read_keep_count` is block read selection arithmetic, a technique (see "dropped") |
| FT6.2 | dropped | none: `block_read_row_count` is the same (see "dropped"); FT6.12 checks the counter against the worked selection with test-local arithmetic |
| FT6.3 block summary score | recut | test-local reference and graph score in FT6.5 |
| FT6.4 reference selection | recut | test-local reference in FT6.5 |
| FT6.5 attended row iterator | recut | the row set is derived in the FT6.5 tests |
| FT6.6 in-graph block score | recut | test-local in FT6.5 |
| FT6.7 in-graph block mask | recut | test-local in FT6.5 |
| FT6.8 row expansion and the four worked tests | recut | FT6.5 (same four test names) |
| FT6.9 thread a block read through `build_forward` | recut | FT6.3 (hook) and FT6.4 (skip operand) |
| FT6.10 read kind in the decode plan key | keep, re-derived | FT6.10 (resident plan identity) |
| FT6.11 fill block read inputs | recut | FT6.11 (generic bind of the read rule), proof in FT6.6 |
| FT6.12 key rows read counter | keep | FT6.12 |
| FT6.13 full keep oracle | keep, retargeted, one card per checkpoint | FT6.13 (gemma4 E2B) and FT6.19 (gemma4 26B) |
| FT6.14 dense oracle | keep, retargeted, one card per checkpoint | FT6.14 (gemma4 E2B) and FT6.20 (gemma4 26B) |
| FT6.15 sampled budget | recut | test-local in FT6.8 |
| FT6.16 sampled mask | recut | test-local in FT6.8 |
| FT6.17 capture fixture | keep, corrected | FT6.7, FT6.21, FT6.22 |
| FT6.18 sampled error bound | keep, rewritten | FT6.8 |
| FT6.19 sampled graph | recut | weight output of the hook in FT6.3 |
| FT6.20 sampled inputs | recut | generic bind in FT6.11; quantile and draws test-local in FT6.8 |
| FT6.21 fusion recognizes block mask | recut | FT6.16 (generic read skip operand) |
| FT6.22 kernel skips rows | keep | FT6.17 |
| FT6.23 row-tiled kernel on Metal | keep | FT6.18 |
| new | | FT6.9 (the read operand program capability on the architecture trait), FT6.15 (the streaming routine skips flagged cached rows), FT6.16 (the CPU interpreter honors the operand, in the same commit as the fusion) |

## dropped

- Library `block_summary_score`, `block_read_select`, `block_read_rows`, `block_read_scores`, `block_read_block_mask`,
  `block_read_row_mask`, `sampled_read_budget`, `sampled_read_mask`, `ReadGraph::Block`, `ReadGraph::Sampled`, the per-layer
  `block_key_*` leaves and `read_inputs.rs::fill_block_read_inputs`: each is a technique, not a hook; the logic reappears as
  test-local code in FT6.5, FT6.6 and FT6.8.
- `proxima_core::read_decision::{block_read_keep_count, block_read_row_count}` and the file `proxima-core/src/read_decision.rs` (the old
  FT6.1 and FT6.2): the top-block count and the attended row count are block read selection arithmetic, which the scope statement above
  keeps out of every library crate, and neither had a non-test caller on the branch. The worked selection (4 sealed blocks of 16, keep
  count 2, one local block, tail 5, 53 attended rows) appears as  arithmetic in the FT6.12 test.
- `sampled_z` (Acklam normal quantile) and the SplitMix64 uniform draws in the library: technique code; test-local in FT6.8.
- The qwen2, qwen3 and openchat checkpoints from the two oracles: owner rule (gemma4 for dense or MoE, granite for MoE, no qwen).
  The oracle counts go from 4 to 2.
- The `DecodePlanKey` fifth element and `read_plan_tag`: the read kind is a property of the runtime, and `PlanIdentity`
  (`generate/resident_plans.rs`) is the existing place a runtime-wide property enters plan reuse. FT6.10 uses it.
- Layer 2 as the captured layer: on gemma4 E2B layers 0 to 3 are sliding (window 512) and layer 4 is the first full-attention layer
  (`proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/swa_layers.txt`); a sampled read over 4096 rows on a sliding layer
  reads at most 512 of them. FT6.22 captures layer 4.

## shared worked values (slice 0 holds the derivations)

Block scoring (head dimension 2, 4 sealed blocks, per-dimension key min and max), pooled query q = (2, -1):

| block | min | max | score = sum_j max(q_j*max_j, q_j*min_j) |
|---|---|---|---|
| 0 | (0, 0) | (1, 1) | max(2,0) + max(-1,0) = 2 |
| 1 | (-2,-1) | (0, 3) | max(0,-4) + max(-3,1) = 1 |
| 2 | (1,-3) | (2,-1) | max(4,2) + max(1,3) = 7 |
| 3 | (-1,-1) | (1, 1) | max(2,-2) + max(-1,1) = 3 |

Block selection: block size b = 16, 4 sealed blocks (rows 0..63) plus an unsealed tail of 5 rows (rows 64..68, 69 rows),
keep_ratio 0.5 (`keep_ratio_milli = 500`), min_blocks 1, local_blocks 1, block scores `[6, 9, 4, 1]`.
`nonlocal = 3`, `n = min(3, max(1, ceil(1.5) = 2)) = 2`. Top 2 among blocks 0..2: block 1 (9), block 0 (6). Local: block 3
(score 1, attended anyway). Attended blocks {0, 1, 3}; block 2 (rows 32..47) is skipped; tail rows 64..68 all attended.
Attended rows = rows 0..31 and 48..68 = 3*16 + 5 = 53 of 69.

Row count (per layer, per decode step s): `len_s = P + s + 1`, `sealed_end_s = seal_target(len_s, b, H)`, `M_s = sealed_end_s / b`,
`rows_s = (n_s + local_s) * b + (len_s - sealed_end_s)`. Instance b = 64, H = 64, n_min = 16, local 1, layers 36, P = 4096, T = 2:
`len = 4097, 4098`, `sealed_end = 4032`, `M = 63`; keep 0.1 -> `n = 16`, rows 1153 and 1154, total 36 * 2307 = 83052;
keep 0.9 -> `n = 56`, rows 3713 and 3714, total 36 * 7427 = 267372.

Sampled budget: base-rate sample weights `[0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 0.9]`, `z = 1.959964`, `epsilon = 0.05`:
`mu = 1.0`, `variance = 0.12 / 7 = 0.0171429`, `n_float = z^2 * variance / (epsilon^2 * mu^2) = 26.341`, `n = ceil = 27`.

## cards

### 6.3 a read hook on the two-range cached attention builder

- id: FT6.3
- needs: none
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::append_lfm2_two_range_cached_attention (~line 431 at a7c08c4c)`: the grouped rotated query planes `q_even_grouped` and `q_odd_grouped` (~line 546), `score_cached_masked` (~line 617, where `is_future_cached` is consumed with `"st->stug"`), and `weights_cached` (~line 728, the `Exponential` that `sum_cached` (~line 747) and the attended product (~line 782) both read);
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::lfm2_two_range_cached_forward_program_with_experts_and_head_repeats (~line 1300 at a7c08c4c)`: the builder that owns the layer loop (`cache_bound` ~line 1519, the one call to the attention function ~line 1641); `lfm2_two_range_cached_forward_program_with_experts (~line 1249)` only forwards to it with `head_repeats = 1`;
  - `proxima-tensor/src/spec/descriptor.rs::build_forward (~line 387 at a7c08c4c)`: the `CacheStrategy::TwoRange` arm (~lines 392-410) calls the builder above with `descriptor.head_repeats`; no other arm does;
  - `proxima-tensor/src/spec/tests.rs`: `mod head_repeats (~line 17466)` with `descriptor`, `built`, `digest` and the `TWO_RANGE_PRE_CHANGE_*` constants; `mod gemma4_synthetic_parity (~line 11901)` with `wave`, `norm_wave`, `layer_weights`, the constants, and `two_range_cached_gemma4_shared_kv_layer_matches_cacheless_oracle (~line 14404)` for a 3-layer schedule (sliding, full donor, shared).
- change:
  1. `proxima-tensor/src/spec/lfm2_single_range_cached.rs`: `append_lfm2_two_range_cached_attention` gains a last parameter
     `read: &mut impl FnMut(&mut Vec<Op>, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>` (arguments `(program, query_even, query_odd, cached_mask)`).
     Right after `q_even_grouped` and `q_odd_grouped` exist, `let (is_future_cached, weight) = read(program, q_even_grouped, q_odd_grouped, is_future_cached)?;`
     and the rest of the function uses the returned mask. When `weight` is `Some(node)` (rank 1 over `t`): `weights_cached = Multiply(weights_cached, node "t->stug")` built
     with `elementwise`, immediately after `weights_cached` is formed and before `sum_cached` and the attended product read it. When `None`, no node is added.
  2. same file: `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` gains a last parameter
     `read: &mut impl FnMut(&mut Vec<Op>, u32, Extent, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>`
     (arguments `(program, layer, key_extent, query_even, query_odd, cached_mask)`; `key_extent` is the layer's `cache_bound`, the extent its `kv_cache.{layer}.*` leaves use). Per layer it passes the attention function
     `&mut |program, query_even, query_odd, cached_mask| read(program, layer, cache_bound, query_even, query_odd, cached_mask)`.
     `lfm2_two_range_cached_forward_program_with_experts` keeps its exact signature and body, and passes `&mut |_, _, _, _, _, cached_mask| Ok((cached_mask, None))` as the new last argument of its one call (`head_repeats` stays 1).
     Doc on the new parameter's function: the read hook; points to `causal_mask_cached_windowed` for the mask convention (1.0 = skipped).
     Premise check: `git grep -n "lfm2_two_range_cached_forward_program_with_experts(" -- '*.rs'` prints 12 lines (11 call sites and the definition), and `git grep -n "lfm2_two_range_cached_forward_program_with_experts_and_head_repeats(" -- '*.rs'` prints 3 lines (the definition, the forwarder above and `descriptor.rs`). The 11 call sites (`omega/tests/gemma4_rows_support/mod.rs`, `proxima-model-interop/tests/arch_data_baseline.rs`, 9 in `proxima-tensor/src/spec/tests.rs`) are untouched. A different count means stop and report.
  3. `proxima-tensor/src/spec/descriptor.rs`: add `pub fn build_forward_with_read(descriptor: &ModelDescriptor, last_row_only: bool, read: &mut impl FnMut(&mut Vec<Op>, u32, Extent, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>) -> Result<BuildForwardProgram, TensorError>`.
     It matches `descriptor.cache_strategy`: the `TwoRange` arm body moves here unchanged except that `read` is the new last argument of `..._and_head_repeats` (`descriptor.head_repeats` is still passed); the two other strategies return
     `Err(TensorError::UnsupportedInBuilder { builder: "build_forward_with_read", feature: "read hook on a cache strategy other than two-range" })` (the variant and its two fields exist; `descriptor.rs (~line 477)` builds one).
     `build_forward` keeps its signature and its other arms; its `TwoRange` arm becomes `build_forward_with_read(descriptor, last_row_only, &mut |_, _, _, _, _, cached_mask| Ok((cached_mask, None)))`.
- test: add in `proxima-tensor/src/spec/tests.rs`:
  - in `mod head_repeats`:
    - `read_hook_identity_equals_the_pre_change_two_range_graph`: `build_forward_with_read(&descriptor(CacheStrategy::TwoRange, 1), false, &mut |_, _, _, _, _, cached_mask| Ok((cached_mask, None)))` returns a program of `TWO_RANGE_PRE_CHANGE_NODES` (268) ops whose `digest` is `TWO_RANGE_PRE_CHANGE_DIGEST` (`0x0b9f2bd7af6a7a91`) and an empty seventh element;
    - `read_hook_keeps_the_descriptor_head_repeats`: with `descriptor(CacheStrategy::TwoRange, 3)` the same call returns the program and the seventh element (2 duplicate roots) that `built(CacheStrategy::TwoRange, 3)` returns;
  - in `gemma4_synthetic_parity`, two helpers `pub(super) fn three_layer_read_schedule() -> Vec<LayerSchedule>` (the schedule of the shared-KV test above) and `pub(super) fn two_layer_read_schedule() -> Vec<LayerSchedule>` (the full donor config of that schedule twice, so no sliding distance mask depends on the cache length), and one helper
    `pub(super) fn read_hook_decode(schedule: &[LayerSchedule], total_rows: usize, kept_rows: &[usize], extra_inputs: &[(String, Vec<f32>)], extra_outputs: &RefCell<Vec<NodeId>>, read: &mut impl FnMut(&mut Vec<Op>, u32, Extent, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>) -> (Vec<Vec<f32>>, Vec<Op>, NodeId, Vec<u64>)`.
    The same module's private consts `KV_HEADS` and `PAIRS` become `pub(super) const` so card 6.6 can import them (a visibility edit only, no value changes).
    The helper builds the program with `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` (the arguments the shared-KV test passes to `lfm2_two_range_cached_forward_program_with_experts`, then `1` and `read`; `sliding_kv_ring` false), feeds one new token (ids `[1]`), the rope rows of position `total_rows`, `cached_len = kept_rows.len()`, `extra_inputs`, and for every cache-owning layer `kv_cache.{layer}.k_even`, `.k_odd`, `.v` assembled from the rows `kept_rows` of `wave("source.k_even", total_rows * KV_HEADS * PAIRS)` (and the odd and value analogues).
    It evaluates with `crate::cpu::evaluate_named` over `[logits]` followed by the nodes in `extra_outputs` (read after the build, so a read function may record nodes there) exactly as the existing two-step decode test does for its second step, and returns `(outputs in that order, program, logits root, symbols [1, kept_rows.len()])`. The tests below take the logits as `outputs[0]`:
    - `read_hook_is_called_once_per_attention_layer`: on the 3-layer schedule a function that pushes `(layer, key_extent == Extent::Symbolic(1))` into a captured `Vec` and returns the mask unchanged is called 3 times, with `[(0, true), (1, true), (2, true)]`;
    - `read_hook_weight_of_three_equals_the_row_three_times`: on `two_layer_read_schedule()`, `total_rows = 6`, `kept_rows = [0,1,2,3,4,5]`: a function that declares a rank-1 leaf `read_weight.{layer}` over `t` (`input_leaf` with the layer's `key_extent`) and returns `(cached_mask, Some(leaf))`, fed `read_weight.0` and `read_weight.1` as `[1,1,3,1,1,1]` through `extra_inputs`, gives logits within `1e-5` of the identity function over `kept_rows = [0,1,2,2,2,3,4,5]` (row 2 present three times).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_3 cargo nextest run -p proxima-tensor -E 'test(/read_hook_/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`; `cargo check -p proxima-model-interop --features std,metal` (every `build_forward` caller still compiles unchanged); `cargo check -p omega --features metal`
- stage: `proxima-tensor/src/spec/lfm2_single_range_cached.rs`, `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): add a read hook to two-range cached attention`
- done when: the expect line printed, clippy and checks clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change any existing caller of `lfm2_two_range_cached_forward_program_with_experts`; add an enum or struct; declare a leaf under the identity function; touch the single-range or cacheless builders.
- gpu: none

### 6.4 `read_skip_operand`, the generic per-row skip leaf

- id: FT6.4
- needs: FT6.3
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::causal_mask_cached_windowed (~line 328 at 4b4be6cf)`: the mask convention (1.0 = skipped) and the `Maximum` that ORs two masks (`is_invalid`);
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` (FT6.3): the builder this one is passed to, as its last argument;
  - `proxima-tensor/src/spec/primitives.rs::input_leaf` and `::elementwise (~line 518)`: how a named leaf is declared and how `"t->st"` broadcasts a rank-1 leaf over the mask;
  - `proxima-tensor/src/spec/tests.rs::read_hook_decode` (FT6.3): the helper the tests use; its `extra_inputs` argument feeds the `read_skip.{layer}` leaves and its second element is the program.
- change:
  1. `proxima-tensor/src/spec/lfm2_single_range_cached.rs`: add
     `pub fn read_skip_operand(program: &mut Vec<Op>, layer: u32, key_extent: Extent, cached_mask: NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>`:
     declares `input_leaf(program, DType::Float32, vec![key_extent], &format!("read_skip.{layer}"))` and returns `(Maximum(cached_mask "st->st", leaf "t->st"), None)`.
     Doc: the generic read hook instance whose rows the host fills each step (1.0 skips a cached row, 0.0 keeps it); pass it as the read function of `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats`
     as `&mut |program, layer, key_extent, _, _, cached_mask| read_skip_operand(program, layer, key_extent, cached_mask)`; the dense default declares nothing.
- test: add in `proxima-tensor/src/spec/tests.rs`, inside `gemma4_synthetic_parity`, on `two_layer_read_schedule()` (FT6.3), `total_rows = 40` (every run goes through `read_hook_decode`; the identity function is `|_, _, _, _, _, cached_mask| Ok((cached_mask, None))`):
  - `read_skip_operand_all_visible_equals_dense`: with `kept_rows = 0..40` and `read_skip.0` and `read_skip.1` fed 40 zeros through `extra_inputs`, logits equal the identity-function logits within `1e-6`, and the program (the helper's second element) holds exactly 2 leaves whose name starts with `read_skip.` (names `read_skip.0`, `read_skip.1`);
  - `read_skip_operand_skipped_rows_equal_deleted_rows`: with `kept_rows = 0..40` and both leaves fed 1.0 on rows `8..24` and 0.0 elsewhere, logits equal the identity-function logits over `kept_rows = (0..8).chain(24..40)` within `1e-5`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_4 cargo nextest run -p proxima-tensor -E 'test(/read_skip_operand_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/lfm2_single_range_cached.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): add the read skip operand for cached attention`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add an `Op` or `ScalarOp` variant; add a weight leaf (the weight output of the hook is for caller-supplied functions); change `causal_mask_cached_windowed`.
- gpu: none

### 6.5 proof: a block top-n selection written against the read hook's graph conventions

- id: FT6.5
- needs: FT3.2
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/primitives.rs::elementwise (~line 518)` and `::reduce (~line 541)` (operand notation `"sugi->bsugi"`) and `::gather_computed (~line 790)` with `::embedding_lookup (~line 758)` (the computed-index gather pattern `table[ids[s], ..]`);
  - `proxima-tensor/src/spec/primitives.rs::top_fraction_mask` (FT3.2): signature in the prerequisites; `keep_count` is a rank-0 input the host fills; `keep_rows` is unioned in;
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::append_lfm2_two_range_cached_attention`: the layout of the grouped query planes (axes `sugi`) the read hook receives (FT6.3);
  - this file's shared worked values (block scoring and block selection).
- change:
  1. `proxima-tensor/src/spec/tests.rs`: test-local helpers in a new `mod block_read_proof` (not exported, not in any library file), about 40 lines of graph in total:
     - `fn block_scores_in_graph(program, query_even, query_odd, key_min_even, key_min_odd, key_max_even, key_max_odd) -> Result<NodeId, TensorError>`: `query_*` have axes `sugi`, `key_*` have axes `bui` (block, kv head, pair).
       Per plane `high = Multiply(query "sugi->bsugi", key_max "bui->bsugi")`, `low = Multiply(query "sugi->bsugi", key_min "bui->bsugi")`, `best = Maximum(high, low)`, `sum = Reduce Add Zero (best "bsugi->bsugi", "b->bsugi")`; the result is `Add(even_sum, odd_sum)`, rank 1 over `b`;
     - `fn block_mask_in_graph(program, scores, keep_count, block_local) -> Result<NodeId, TensorError>`: `neg_infinity = scalar_constant(program, f32::NEG_INFINITY)`; `non_local = Select(block_local "b->b", neg_infinity "->b", scores "b->b")` (via `elementwise` with `ScalarOp::Select`);
       return `top_fraction_mask(program, non_local, keep_count, Some(block_local))`. Local blocks rank last, so the top `keep_count` are all non-local; the union adds the local blocks;
     - `fn row_skip_in_graph(program, selected_blocks, row_block, tail_rows) -> Result<NodeId, TensorError>`: `selected_blocks` is rank 1 over blocks (1.0 = selected), `row_block` an `Int32` leaf over cache rows (index values are exact in f32, so the test feeds them as f32), `tail_rows` a rank-1 `Float32` leaf over cache rows.
       `by_row = gather_computed(program, selected_blocks, row_block, map::projection(1, &[0]), IndexPattern { iter_rank: 1, axes: vec![AxisIndex::default()] }, 0, DType::Float32)`: the `embedding_lookup` pattern (`IndexMap::Computed` with `gathered_dim: 0`, the gathered source axis left `AxisIndex::default()`) with the feature axis removed, so the iteration space is the cache row alone and the source has one axis;
       `attended = Maximum(by_row, tail_rows)`; return `Greater(scalar_constant(1.0), attended)` (1.0 where the row is not attended, the mask convention). Premise: shape inference accepts a rank-1 source for `gather_computed` (its doc on main, `primitives.rs (~lines 785-790 at a7c08c4c)`, describes a rank-three source, and `embedding_lookup` is the rank-two case); if it rejects this graph the executor stops and reports, it does not pick another pattern;
     - `fn block_scores_host(query: &[f32], key_min: &[f32], key_max: &[f32]) -> f32` (`sum_j max(q_j * kmax_j, q_j * kmin_j)`) and
       `fn select_blocks_host(scores: &[f32], local_blocks: usize, keep_count: usize) -> Vec<bool>` (mark the last `local_blocks.min(len)` blocks, then `keep_count` times mark the highest-scored unmarked non-local block, ties to the lower index with `f32::total_cmp`): the reference the graph is checked against.
- test: add in `proxima-tensor/src/spec/tests.rs` (`block_read_` names; the substring `block_read_selection_worked_` appears only in the four tests below):
  - `block_read_scores_worked_four_blocks`: leaves `query_even = [2.0]`, `query_odd = [-1.0]` of shape `[1, 1, 1, 1]`; key leaves of shape `[4, 1, 1]`: `key_min_even = [0, -2, 1, -1]`, `key_min_odd = [0, -1, -3, -1]`, `key_max_even = [1, 0, 2, 1]`, `key_max_odd = [1, 3, -1, 1]`;
    `crate::cpu::evaluate_named` gives length 4 equal to `[2.0, 1.0, 7.0, 3.0]`, and `block_scores_host` agrees block by block (compared with `==`, exact in f32);
  - four tests sharing `fn evaluate_worked_row_skip() -> Vec<f32>` (scores `[6, 9, 4, 1]`, `block_local = [0, 0, 0, 1]`, `keep_count = 2`, 69 cache rows, `row_block[t] = min(t / 16, 3)`, `tail_rows[t] = 1.0` for `t >= 64`, each asserting the length 69 first):
    `block_read_selection_worked_top` (rows 0..32 are 0.0), `block_read_selection_worked_local` (rows 48..64 are 0.0), `block_read_selection_worked_tail` (rows 64..69 are 0.0),
    `block_read_selection_worked_nothing_else` (the rows with value 1.0 are exactly 32..48 and the 0.0 rows `.eq((0..32).chain(48..69))`, count 53);
  - `block_read_selection_matches_the_host_reference`: each case is `(scores, local_blocks, requested_keep)` over 4 blocks. The test derives, with plain integer arithmetic in the test body, `local_count = local_blocks.min(4)`, `non_local_count = 4 - local_count` and `graph_keep = requested_keep.min(non_local_count)`; `block_local` marks the last `local_count` blocks with 1.0; the graph receives `graph_keep` as `keep_count` (never the requested value), and `select_blocks_host` receives `graph_keep` too. The expected block mask (1.0 = selected, pinned here) of each case, with a failure message naming the tuple:
    - `([6,9,4,1], 1, 2)`: `graph_keep = 2`, `[1,1,0,1]`;
    - `([6,9,4,1], 1, 1)`: `graph_keep = 1`, `[0,1,0,1]`;
    - `([1,2,3,0], 1, 1)` (local wins over a low score): `graph_keep = 1`, `[0,0,1,1]`;
    - `([5,5,5,0], 1, 2)` (ties to the lower index): `graph_keep = 2`, `[1,1,0,1]`;
    - `([6,9,4,1], 9, 2)` (all blocks local): `local_count = 4`, `graph_keep = 0`, `[1,1,1,1]`;
    - `([6,9,4,1], 1, 9)` (requested keep above the non-local count): `graph_keep = 3`, `[1,1,1,1]`;
    the in-graph block mask equals both the pinned vector and `select_blocks_host`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_5 cargo nextest run -p proxima-tensor -E 'test(/block_read_scores_worked_four_blocks|block_read_selection_worked_|block_read_selection_matches_the_host_reference/)'`
- expect: `6 passed` (1 + 4 + 1)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/tests.rs`
- commit: `test(tensor): select top and local blocks in graph by key bounds`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: put any of these helpers in a non-test file; add an `Op` or `ScalarOp` variant; reuse the `block_read_selection_worked_` substring in any other test.
- gpu: none

### 6.6 proof: a block read expressed through the read hook, no library code

- id: FT6.6
- needs: FT6.5, FT6.4
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/tests.rs::block_read_proof` (FT6.5): `block_scores_in_graph`, `block_mask_in_graph`, `row_skip_in_graph`, `select_blocks_host`;
  - `proxima-tensor/src/spec/tests.rs::gemma4_synthetic_parity::{read_hook_decode, two_layer_read_schedule, KV_HEADS, PAIRS}` (FT6.3; the card makes the two consts `pub(super)` and imports all four into `mod block_read_proof` with `use super::gemma4_synthetic_parity::{read_hook_decode, two_layer_read_schedule, KV_HEADS, PAIRS};`): `extra_inputs` feeds this card's leaves and `extra_outputs` is the `record` cell below;
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::read_skip_operand` (FT6.4): the leaf-only form this card's function generalizes;
  - `proxima-tensor/src/cpu/arena.rs::evaluate_named (~line 310 at a7c08c4c)`: it takes a slice of output nodes, which is how `read_hook_decode` reads the logits and the recorded mask node in one evaluation.
- change:
  1. `proxima-tensor/src/spec/tests.rs`, inside `mod block_read_proof` (FT6.5, which is a sibling of `gemma4_synthetic_parity`; the imports are in the read-first list): add `fn block_read_function(blocks: usize, record: &RefCell<Vec<NodeId>>) -> impl FnMut(&mut Vec<Op>, u32, Extent, NodeId, NodeId, NodeId) -> Result<(NodeId, Option<NodeId>), TensorError>`
     (about 15 lines): per layer it declares leaves `block_key_min_even.{layer}`, `block_key_min_odd.{layer}`, `block_key_max_even.{layer}`, `block_key_max_odd.{layer}` (shape `[Static(blocks), Static(KV_HEADS), Static(PAIRS)]`) and, once per call, rank-0 `block_keep_count` and rank-1 `block_local` (extent `Static(blocks)`), `row_block` and `row_tail` (extent `key_extent`),
     builds `block_scores_in_graph`, `block_mask_in_graph`, `row_skip_in_graph`, ORs the row skip into `cached_mask` with `Maximum "st->st"` and `"t->st"`, pushes the returned mask node onto `record`, and returns `(mask, None)`.
     A leaf declared twice for the same name is an error: the function declares `block_keep_count`, `block_local`, `row_block`, `row_tail` only when the layer index is the first it sees.
  2. in the same module, a host fold `fn fold_block_bounds(rows: &[f32], width: usize, block_rows: usize) -> (Vec<f32>, Vec<f32>)` (per-block per-dimension min and max) used to feed the key bound leaves from the same cache data the program reads.
- test: add in `proxima-tensor/src/spec/tests.rs`, inside `mod block_read_proof`, on `two_layer_read_schedule()` with `total_rows = 69`, block size 16 (4 sealed blocks and a 5 row tail), real cache data from `wave`:
  - `block_read_through_the_hook_at_full_keep_equals_dense`: `block_keep_count = 3` (every non-local block, `local_blocks = 1`), logits within `1e-5` of the identity function over all 69 rows, and the recorded mask of layer 0 holds 0 skipped rows;
  - `block_read_through_the_hook_equals_deleting_the_skipped_rows`: `block_keep_count = 2`; pass `record` as `extra_outputs` so the evaluated outputs are the logits followed by the recorded mask nodes (layer 0's first); the mask row 0 holds exactly 16 ones (derived from the setup: 4 sealed blocks of 16 rows, `block_keep_count = 2` and 1 local block leave 3 non-local blocks, so exactly one non-local block of 16 rows is skipped, and the 5 tail rows are always attended; the 16 ones are one whole block, rows `block * 16..(block + 1) * 16` for a `block` in `0..3`, and the other 53 of the 69 cached rows are 0.0; this is also the control that the read skipped something) and exactly the rows it marks 1.0 are removed from `kept_rows` for the identity-function run; the two logits agree within `1e-5`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_6 cargo nextest run -p proxima-tensor -E 'test(/block_read_through_the_hook_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/tests.rs`
- commit: `test(tensor): drive a block read through the read hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit a non-test file; call `read_skip_operand` from this function; add a library function that selects blocks.
- gpu: none

### 6.7 the capture example reads its flags and builds the fixed-length prompt

- id: FT6.7
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/long_context_niah.rs::run (~line 390 at a7c08c4c)`: the memory map, `parse_complete`, `vocab_from_metadata` and `encode_with_bos_eos` pattern this example copies;
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 667 at a7c08c4c)`: how `wants_bos` and `wants_eos` are read off the vocab;
  - `proxima-model-interop/Cargo.toml (~lines 430-447 at a7c08c4c)`: the `[[example]]` entries `long_context_niah`, `compare_local` and `rerank_local`, each with only `name` and `required-features`; no entry on main carries `test = true` (`git grep -n -E '^\s*test\s*=\s*true' main -- '*/Cargo.toml'` prints nothing), so this card adds that key itself;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt`: the checkpoint's architecture name (`gemma4`).
- change:
  1. `proxima-model-interop/Cargo.toml`: an `[[example]]` entry `kv_capture` (`path = "examples/kv_capture.rs"`, `required-features = ["std", "metal"]`, `test = true`); `test = true` is new to this manifest and is what makes `nextest --example kv_capture` run the example's `#[cfg(test)] mod tests`.
  2. new file `proxima-model-interop/examples/kv_capture.rs` (about 70 lines of non-test code):
     - `#[derive(Debug, thiserror::Error)] enum CaptureError { Usage(String), Io { path: String, source: std::io::Error }, Model(String), PromptLength { expected: usize, found: usize } }`, each message lowercase and naming its subject (the missing flag, the path, the two counts); `Model` carries the text of a parse or vocabulary error;
     - `struct Flags { model: PathBuf, corpus: PathBuf }` and `fn parse_flags(arguments: impl Iterator<Item = String>) -> Result<Flags, CaptureError>`: `--model <gguf>` and `--corpus <text file>`; a missing flag or an unknown flag is `CaptureError::Usage` naming it;
     - `fn build_prompt(corpus: &str, vocab: &Vocab, wants_bos: bool, wants_eos: bool) -> Result<String, CaptureError>`: `body_ids = encode_with_bos_eos(<the first 40000 chars of corpus>, vocab, false, false)`; keep the first `4096 - usize::from(wants_bos) - usize::from(wants_eos)` of them; `prompt = proxima_tokenizer::decode(<those ids>, vocab)`; if `encode_with_bos_eos(&prompt, vocab, wants_bos, wants_eos)` is not exactly 4096 ids, return `CaptureError::PromptLength { expected: 4096, found }`;
     - `fn main() -> Result<(), CaptureError>`: parse the flags, read the corpus (`Io` on failure), memory-map the checkpoint (`Io`), `parse_complete` and `vocab_from_metadata` (`Model`), compute `wants_bos` and `wants_eos` as `llama_parity` does, call `build_prompt`, and print exactly one line, `prompt_ids=4096` (the found count, formatted).
     Every item is used by `main` or by the tests: no item exists only for a later card. Nothing in the file, its messages or its output carries a slice number, a card id or an FT id. No model weights are loaded: the checkpoint is mapped and parsed only.
- test: add in the example's `#[cfg(test)] mod tests`:
  - `kv_capture_flags_name_the_missing_flag`: `parse_flags` over `["--corpus", "war_and_peace.txt"]` is `Err(CaptureError::Usage(message))` and `message` contains `--model`;
  - `kv_capture_flags_read_both_paths`: `parse_flags` over `["--model", "gemma4-e2b.gguf", "--corpus", "war_and_peace.txt"]` gives `model == PathBuf::from("gemma4-e2b.gguf")` and `corpus == PathBuf::from("war_and_peace.txt")`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_7 cargo run -p proxima-model-interop --example kv_capture --features std,metal -- --model "$GEMMA4" --corpus proxima-model-interop/examples/data/war_and_peace.txt` (with `$GEMMA4` from `proxima-tensor/specs/long-context/env.sh`), then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_7 cargo nextest run -p proxima-model-interop --example kv_capture --features std,metal -E 'test(/kv_capture_flags_/)'`
- expect: the run prints exactly 1 line, `prompt_ids=4096` (derived from the card's own length check, not yet observed); then `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/examples/kv_capture.rs`, `proxima-model-interop/Cargo.toml`
- commit: `test(interop): add a capture example that builds its prompt`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; load model weights; add a flag other than the two named (the output directory flag arrives with the card that writes the file); run another model process at the same time.
- gpu: none (a memory map and a parse, no weight load)

### 6.21 the capture example taps layer 4's key, value and query nodes

- id: FT6.21
- needs: FT6.7, FT6.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-tensor/src/spec/descriptor.rs::build_forward_with_read` (FT6.3) and `::BuildForwardProgram (~line 341 at a7c08c4c)`: the tuple `(program, logits, cache_roots, moe_sites, layer_residuals, _, duplicate_head_roots)`; `cache_roots` holds one `CachedLayerRoots` per cache-owning layer (its rotated key halves and its value root; read its field names from its definition);
  - `proxima-model-interop/src/gemma4/bind.rs::descriptor_from_gguf (~line 658 at a7c08c4c)`: `family_profile` over `general.architecture`, then `gemma4_descriptor_from_gguf(&parsed, true, &profile)` (`LoadedModel::load` uses the sliding ring layout, hence `true`); `::rebuild_layer_roots (~line 667)` shows how `cache_roots` pair with `ProjectedK` layers in schedule order;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/swa_layers.txt` and `gguf_kv.txt`: layers 0 to 3 are sliding, layer 4 is the first full-attention layer; `gemma4.attention.key_length = 512`, `head_count = 8`, `head_count_kv = 1`, `shared_kv_layers = 20`;
  - `proxima-model-interop/examples/kv_capture.rs::main` (FT6.7): where the call goes.
- change:
  1. `proxima-model-interop/examples/kv_capture.rs`:
     - `fn projected_k_layers_before(key_sources: &[KeySourceKind], layer: usize) -> usize`: the count of `KeySourceKind::ProjectedK` among `key_sources[..layer]` (the layer's index into `cache_roots`);
     - `fn tap_layer_four(parsed: &ParsedGguf) -> Result<(Vec<Op>, [NodeId; 5]), CaptureError>`: `family_profile("gemma4")` (an error is `CaptureError::Model`), `gemma4_descriptor_from_gguf(parsed, true, &profile)`; layer `4` of the schedule must have `mask_window == None`, `key_source_kind == KeySourceKind::ProjectedK` and `score_scale == AttentionScoreScale::Unscaled`, else `CaptureError::Model` naming the layer and the field; then `build_forward_with_read(&descriptor, true, &mut tap)` where `tap` pushes `(query_even, query_odd)` into a local `Vec<(NodeId, NodeId)>` when its `layer` argument is 4 and returns `(cached_mask, None)` unchanged; exactly one query pair must have been recorded, else `CaptureError::Model`; the returned array is the rotated key halves and the value root of `cache_roots[projected_k_layers_before(<the schedule's key sources>, 4)]` followed by the recorded query pair;
     - `main` calls `tap_layer_four` after the prompt step and prints a second line, `layer=4 cache_root=4 tapped_nodes=5` (derived: layers 0 to 3 each own a key, so the index is 4; not yet observed).
- test: add `projected_k_layers_before_counts_the_owning_layers` in the example's `mod tests`: for the key sources of the real E2B schedule shape, 15 `ProjectedK` entries then 20 `SharedFromLayer(14)` entries (`shared_kv_layers = 20` of 35 layers), `projected_k_layers_before(&sources, 4) == 4`, `projected_k_layers_before(&sources, 15) == 15` and `projected_k_layers_before(&sources, 30) == 15`.
- validate: the run command of FT6.7 with `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_21`, then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_21 cargo nextest run -p proxima-model-interop --example kv_capture --features std,metal -E 'test(/kv_capture_flags_|projected_k_layers_before_/)'`
- expect: the run prints exactly 2 lines, `prompt_ids=4096` and `layer=4 cache_root=4 tapped_nodes=5`; then `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/examples/kv_capture.rs`
- commit: `test(interop): tap layer 4 attention nodes in the capture example`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; load model weights; write a file; put a slice number, a card id or an FT id in the example, its messages or its output. If a layer 4 field differs from the three required values, the executor stops and reports.
- gpu: none (a memory map, a parse and a graph build, no weight load)

### 6.22 the capture example writes layer 4's rows from one gemma4 E2B forward pass (real-data input)

- id: FT6.22
- needs: FT6.21
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/decode.rs::LoadedModel::forward_node_values_on_backend (~line 6975 at a7c08c4c)`: evaluates requested `NodeId`s of the loaded model's own program over a whole prompt from an empty cache, so a node id of an identically built program reads the loaded model's value; `LoadedModel::node_kind` (`generate/pregather.rs`) is the check that the ids belong to that program;
  - `proxima-model-interop/examples/kv_capture.rs::tap_layer_four` (FT6.21): returns the program and the five nodes;
  - `proxima-model-interop/examples/long_context_niah.rs::run (~line 390 at a7c08c4c)`: how an example loads the model from the mapping;
  - the layout in this card's change list (the file is the contract the sampled-read test reads).
- change:
  1. `proxima-model-interop/examples/kv_capture.rs`:
     - `parse_flags` also reads `--out <dir>` (default `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/kv_captures`) into a new `Flags.out: PathBuf`; add `CaptureError::NodeMismatch { node: u32 }`;
     - `fn write_capture(sink: &mut impl Write, header: [u32; 5], planes: [&[f32]; 5]) -> Result<u64, CaptureError>`: writes the five header words little-endian, then the five planes in order, each f32 little-endian, and returns the byte count written (a write failure is `CaptureError::Io`);
     - in `main`, after the tap: `model = LoadedModel::load(&parsed, &mapping)`; for each of the five nodes `model.node_kind(node)` must equal the built program's `program[node.0 as usize].kind()`, else `CaptureError::NodeMismatch { node }`; then `model.forward_node_values_on_backend(&prompt, &nodes, GPU_LAYERS_ALL)` (one model load, one forward pass); the planes are K even and K odd (each `4096 * 256` f32), V (`4096 * 512`), then the grouped rotated Q even and Q odd (each `2000 * 8 * 256`), the queries being the rows at prefill positions `2096..=4095` (query `j` sits at position `2096 + j` and attends rows `0..=2096 + j`); the header is `[4096, 2000, 256, 8, 1.0f32.to_bits()]` (`1.0f32.to_bits() = 1065353216`, the unscaled attention score);
     - write `<out>/gemma4-e2b-layer4.f32` through `write_capture` (create the directory first; `Io` on failure) and print `rows=4096+2000` and `bytes=<the returned count>`.
     The file is real data generated by the model and is not committed (about 50 MB). The sampled-read test reads it from that durable path, and only under its own feature (FT6.8), so a clean checkout never needs it.
- test: add `kv_capture_writes_the_header_then_the_planes_little_endian` in the example's `mod tests`: `write_capture` into a `Vec<u8>` with header `[3, 2, 4, 2, 1065353216]` and the five planes `0.0..12.0` (12 values), `12.0..24.0` (12), `24.0..48.0` (24), `48.0..64.0` (16) and `64.0..80.0` (16), which are the 80 f32 values `0.0, 1.0, .. 79.0` in order, returns `340`; the buffer is 340 bytes: bytes `0..20` are the five header words as `u32::to_le_bytes`, bytes `20..340` are the 80 values as `f32::to_le_bytes` in order; a sink whose every write fails gives `Err(CaptureError::Io { .. })`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_22 cargo run -p proxima-model-interop --example kv_capture --features std,metal -- --model "$GEMMA4" --corpus proxima-model-interop/examples/data/war_and_peace.txt` (with `$GEMMA4` from `proxima-tensor/specs/long-context/env.sh`), then `stat -f %z /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/kv_captures/gemma4-e2b-layer4.f32`, then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_22 cargo nextest run -p proxima-model-interop --example kv_capture --features std,metal -E 'test(/kv_capture_|projected_k_layers_before_/)'`
- expect: the run prints exactly 4 lines, `prompt_ids=4096`, `layer=4 cache_root=4 tapped_nodes=5`, `rows=4096+2000` and `bytes=49545236` (derived: `20 + 4 * 12386304`, not yet observed); `stat` prints `49545236`; then `4 passed` (2 flag tests, the tap count test and the writer test)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/examples/kv_capture.rs`
- commit: `test(interop): write layer 4 attention rows from a gemma4 forward pass`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; write the data by hand; commit the data file; run another model process at the same time; put a slice number, a card id or an FT id in the example, its messages or its output.
- gpu: one run (a model load), waiting for a quiet box (CARDS.md machine safety).

### 6.8 proof: importance-weighted sampled read, error bound on real captured rows

- id: FT6.8
- needs: FT6.22, FT6.3
- budget: 20 min
- crate(s): proxima-tensor (features: default; `real-capture-tests` for the second validate command)
- read first:
  - `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/kv_captures/gemma4-e2b-layer4.f32` (FT6.22): header (`[4096, 2000, 256, 8, 1065353216]` as five little-endian `u32`) and plane order (not in git);
  - `proxima-tensor/src/spec/tests.rs::read_hook_weight_of_three_equals_the_row_three_times` (FT6.3): the proof that the hook's weight output multiplies a row's exponential, so the estimator below is what a read function returning a skip mask and a weight computes;
  - `proxima-tensor/tests/` for an existing integration test file to copy the file header from, and `proxima-tensor/Cargo.toml (~line 42)` for the feature style;
  - this file's shared worked values (sampled budget).
- change:
  1. `proxima-tensor/Cargo.toml`: add the feature `real-capture-tests = []` (default off). Reason, stated in one line above it: the test that reads the captured rows needs a 50 MB file that is not committed, so a clean checkout must neither compile nor run it.
  2. new file `proxima-tensor/tests/sampled_read_error_bound.rs`, test-local code only (the sampler is about 40 lines; nothing is added to `src/`). Nothing in the file, its names or its messages carries a slice number, a card id or an FT id. Layout:
     - outside any cfg: `fn sampled_read_budget(base_weights: &[f64], epsilon: f64, max_budget: usize) -> usize` (`n = min(max_budget, ceil(z^2 * variance / (epsilon^2 * mean^2)))`, `z = 1.959964`, the sample variance with divisor `len - 1`, and at least 1) and the test `sampled_read_budget_worked_base_sample`;
     - one module `#[cfg(feature = "real-capture-tests")] mod real_rows { .. }` holding everything below, so a build without the feature leaves no unused item;
     - in that module, reading the capture from the path in the environment variable `PROXIMA_READ_CAPTURE`, or from `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/kv_captures/gemma4-e2b-layer4.f32` when it is unset; when the file is missing it panics (never skips) with exactly the message `capture file <path> is missing; create it with: cargo run -p proxima-model-interop --example kv_capture --features std,metal -- --model <gemma4 e2b gguf> --corpus proxima-model-interop/examples/data/war_and_peace.txt` (`<path>` is the path it tried).
     For every query row `j` (position `p = 2096 + j`, prefix length `p + 1`) and every query head, with score `q . k * scale` over the even and odd planes:
     exact attention in f64 over the prefix; sampled read: fixed rows are sink rows `0..4`, local rows `prefix - 64..prefix`, and the rows of the 8 highest-scored 64-row sealed blocks (block score `sum_d max(q_d * kmax_d, q_d * kmin_d)` over the block's per-dimension bounds, ties to the lower block);
     candidates are the remaining prefix rows (count `C`); a base sample draws each candidate with probability `0.05` from a SplitMix64 generator seeded `(j, head)`; the base sample's weights `exp(score - m)` with `m` the maximum fixed score give
     `n = sampled_read_budget(<those weights>, 0.05, 1024)`; the draw probability is `p_draw = min(1, n / C)`; each candidate is read when its uniform (SplitMix64 seeded `(j, head, 1)`) is below `p_draw`, with weight `1 / p_draw`, fixed rows weight 1;
     the estimator is `sum(weight * exp(score - m) * value) / sum(weight * exp(score - m))`; the relative error is `||sampled - exact|| / ||exact||` over the value vector.
- test: add in that file:
  - `sampled_read_budget_worked_base_sample` (always compiled): the budget function on the 8 worked base weights with `epsilon = 0.05` is exactly `27`, with `epsilon = 0.5` it is `1`, with `epsilon = 0.001` and `max_budget = 64` it is `64`;
  - `sampled_read_error_bound_real_layer_rows` (inside `real_rows`, so only under `real-capture-tests`): over the 2000 rows times 8 heads = 16000 (row, head) pairs, at least 15200 (a fraction `1 - delta` with `delta = 0.05`) have relative error `<= 0.05`; the message prints the achieved count. A control inside the test: the same pipeline with the budget forced to 1 gives fewer than 15200 passing pairs (otherwise the measurement is not measuring the sampler).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_8 cargo nextest run -p proxima-tensor -E 'test(/sampled_read_/)'`; then, on a machine that holds the capture file (the one that ran FT6.22), `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_8 cargo nextest run -p proxima-tensor --features real-capture-tests -E 'test(/sampled_read_/)'`
- expect: `1 passed` for the first command (the budget test; a clean checkout is green); `2 passed` for the second (the budget test and the real-rows test). If the second command cannot find the file it prints the panic message above and the executor stops and reports; it does not skip.
- also green: `cargo clippy -p proxima-tensor --all-targets`; `cargo clippy -p proxima-tensor --features real-capture-tests --all-targets`
- stage: `proxima-tensor/tests/sampled_read_error_bound.rs`, `proxima-tensor/Cargo.toml`
- commit: `test(tensor): bound sampled read error on real captured rows`
- done when: the expect lines printed, clippy clean in both feature sets, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; lower the 15200 threshold or raise 0.05 (a miss is reported with the printed count, not widened); add a quantile, a budget or a sampler function to a library crate; read the capture outside the `real-capture-tests` module; add a default feature.
- gpu: none

### 6.9 a read operand program capability on the architecture trait

- id: FT6.9
- needs: FT6.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/architecture.rs::Architecture::speculative_verify_program_with_kv_layout (~line 363 at 4b4be6cf)`: the default-`None` capability method this card copies the shape of, and `gemma4/bind.rs::Gemma4Arch::speculative_verify_program_with_kv_layout (~line 852)`;
  - `proxima-model-interop/src/gemma4/bind.rs::bind_gemma4_with_last_row_only (~line 663)`: where `build_forward` is called (~line 679) and the bound program is assembled; its 4 callers are in the same file (~lines 774, 808, 823, 858);
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::read_skip_operand` (FT6.4) and `descriptor.rs::build_forward_with_read` (FT6.3).
- change:
  1. `proxima-model-interop/src/architecture.rs`: add the trait method `read_operand_program_with_kv_layout<'file>(&self, parsed: &ParsedGguf, file_bytes: &'file [u8], layout: KvLayout) -> Result<Option<BoundProgram<'file>>, InteropError>` with default `Ok(None)`. Doc: the program whose attention layers declare the read skip operand; `None` for a family without one.
  2. `proxima-model-interop/src/gemma4/bind.rs`: `bind_gemma4_with_last_row_only` gains a trailing generic parameter `read: &mut impl FnMut(..)` (the read function type of FT6.3) and calls `build_forward_with_read`; its 4 callers pass `&mut |_, _, _, _, _, cached_mask| Ok((cached_mask, None))`.
     `Gemma4Arch` overrides the trait method with `Ok(Some(..))` from `bind_gemma4_with_last_row_only(parsed, file_bytes, true, layout, &mut |program, layer, key_extent, _, _, cached_mask| read_skip_operand(program, layer, key_extent, cached_mask))`.
- test: add `read_operand_program_declares_one_skip_leaf_per_layer_gemma4_e2b` in `proxima-model-interop/tests/arch_data_baseline.rs` (the real E2B checkpoint; `GEMMA4_E2B` and `Checkpoint::open` exist): resolve the architecture through `ArchitectureRegistry::with_builtin().resolve(..)`, call the new method with `KvLayout::SlidingRing`, assert `Some`, and assert the count of program leaves named `read_skip.<n>` equals the literal `35`, the `gemma4.block_count` of the checkpoint (`proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt` line 17; every one of the 35 layers is an attention layer, so the hook is called 35 times; the literal does not come from the descriptor that builds the program). Derived by reading, not yet observed: if the printed count differs, stop and report it; do not edit the integer to match;
  the dense program is unchanged: `arch_data_digest_gemma4_e2b` still passes in the same run.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_9 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/read_operand_program_declares_one_skip_leaf_per_layer_gemma4_e2b|arch_data_digest_gemma4_e2b/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/architecture.rs`, `proxima-model-interop/src/gemma4/bind.rs`, `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `feat(interop): add a read operand program capability to architectures`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the dense program or the verify program; override the method for any family but gemma4; add a field to `LoadedModel` (the next cards).
- gpu: none (a checkpoint is mapped and parsed, no Metal run)

### 6.10 the read spec joins the resident plan identity

- id: FT6.10
- needs: FT2.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/resident_plans.rs::PlanIdentity (~line 48 at 4b4be6cf)` and `::PlanIdentity::of (~line 79)`: the exhaustive destructure of `ServingConfig` and the identity two generations must share to reuse a plan;
  - `proxima-model-interop/src/generate/residency_caches.rs::BackendRuntime::evaluate_with_placements (~line 2305)`: the decode plan key is `(new_count, kv_bound_extent, outputs, epilogue_sources)`, so a read program is told apart from the dense one in one runtime only by its output node ids (the read program declares leaves before the logits root, so they differ; nothing states it);
  - `proxima-model-interop/src/serving.rs::AttentionConfig` (the slice 2 re-cut card FT2.1): `read: ReadSpec`; the destructure in `PlanIdentity::of` lists `attention` after FT2.1 (as `attention: _` or a nested pattern).
- change:
  1. `proxima-model-interop/src/generate/resident_plans.rs`: `PlanIdentity` gains `read: ReadSpec` (the struct already derives `PartialEq`; `ReadSpec` is `Copy` and `PartialEq`), doc: which cache rows a decode step reads; a plan built with the read operand declares leaves a dense plan does not. `PlanIdentity::of` binds `attention.read` and stores it; the rest of the destructure is unchanged.
- test: `generate/resident_plans.rs` has no tests module on main (its only `#[cfg(test)]` items are the helper fns `resident_len` and `entry_count`, ~lines 224 and 236, and no test names `PlanIdentity`). Create one at the end of the file: `#[cfg(test)] #[allow(clippy::unwrap_used, clippy::expect_used)] mod tests { use super::*; use crate::serving::AttentionConfig; use crate::serving_grammar::ReadSpec; .. }` (the allow is the workspace rule for a new test module; nothing else in the file is touched). Add `plan_identity_separates_read_specs` inside it: `PlanIdentity::of(&ServingConfig::default()) == PlanIdentity::of(&dense_config)` where `dense_config` sets `attention.read = ReadSpec::Dense` explicitly, and `PlanIdentity::of(&dense_config) != PlanIdentity::of(&operand_config)` with `attention.read = ReadSpec::Operand`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_10 cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/plan_identity_separates_read_specs/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/resident_plans.rs`
- commit: `feat(interop): key resident decode plans by the read spec`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add an element to `DecodePlanKey`; put a float in the identity; change the clear-on-miss behaviour of `resolve_cached_plan`.
- gpu: none

### 6.25 the model carries a read operand program and a read rule

- id: FT6.25
- needs: FT6.9, FT6.10
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/pregather.rs (~line 2685 at 4b4be6cf)`: where `LoadedModel::load_with_kv_layout` resolves the verify program, and `generate/load_model.rs::LoadedModel.speculative_verify_program (~line 962)`: the field shape the new program field copies;
  - `proxima-model-interop/src/generate/decode.rs (~lines 3645 and 3769-3800 at 4b4be6cf)`: how the single-row decode step chooses `self.speculative_verify_program`, which is where this card's refusal is called;
  - `proxima-model-interop/src/error.rs::InteropError::UnsupportedServingConfig (~line 322)`: the existing typed error for a serving configuration the model cannot honor;
  - `proxima-model-interop/src/generate/chunked_prefill_tests.rs::{gemma4_checkpoint, dense_checkpoint, config}` (~lines 94, 150 and 186 at 4b4be6cf): the synthetic gemma4 checkpoint (which overrides the read operand program of FT6.9) and the synthetic dense checkpoint (which does not), and the config helper; the file's existing `LoadedModel::load` call shows how a fixture becomes a model.
- change:
  1. `proxima-model-interop/src/generate/load_model.rs`: `LoadedModel` gains `pub(super) read_operand_program: Option<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>)>` (the verify program's shape), `read_rule: Option<Box<dyn Fn(usize, usize, &mut [f32]) + Send + Sync + 'file>>`
     (a `Box<dyn Fn>` because the set of read rules is open and user written; the one legitimate dyn in the change) and `pub fn set_read_rule(&mut self, rule: impl Fn(usize, usize, &mut [f32]) + Send + Sync + 'file)`.
     Doc on the setter: the pure read function of the decode step; arguments are the layer index, the number of rows that layer's cache hands the program this step (every cached row for a full layer, the most recent `min(cached rows, window)` for a ring layer), and a slice of that length the rule marks with 1.0 for every row to skip.
     `LoadedModel` is built by four exhaustive struct literals with no `..` base (`proxima-model-interop/src/generate/pregather.rs` at ~lines 2741, 2878 and 3010, each ending `speculative_verify_program`, `checkpoint_mapping`; and `proxima-model-interop/src/generate/tests_all.rs::memory_fit_gate_tests::model_with` at ~line 2597), so every one of the four gets both new fields in this commit.
  2. `proxima-model-interop/src/generate/pregather.rs`: `load_with_kv_layout` (literal at ~line 2741) resolves `read_operand_program_with_kv_layout` (FT6.9) beside the verify program, stores it in `read_operand_program` (`None` for a family that does not override the method) and sets `read_rule: None`; the other two literals (~lines 2878 and 3010) set `read_operand_program: None, read_rule: None`.
     `proxima-model-interop/src/generate/tests_all.rs`: the `model_with` literal (~line 2597) sets `read_operand_program: None, read_rule: None`.
  3. `proxima-model-interop/src/generate/decode.rs`: add `read_operand_parts(&self) -> Result<(&(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), &(dyn Fn(usize, usize, &mut [f32]) + Send + Sync)), InteropError>` (`pub(super)`, on `LoadedModel`, about 15 lines): with `self.read_rule == None` it is `Err(InteropError::UnsupportedServingConfig("attention.read = operand needs a read rule; call set_read_rule".to_owned()))`; with `self.read_operand_program == None` it is `Err(InteropError::UnsupportedServingConfig("attention.read = operand needs a read operand program; this architecture has none".to_owned()))` (the rule is checked first); otherwise it returns the two references (`Option::as_ref` and `Option::as_deref`, no unwrap).
     At the single-row decode step, when `serving.attention.read == ReadSpec::Operand`, run the statement `self.read_operand_parts()?;` once before the step loop, discarding the returned references (the statement exists only for its refusal; no local is bound, so nothing is unused and neither rustc nor clippy warns; FT6.11 turns this statement into `let (program, rule) = self.read_operand_parts()?;`); under `ReadSpec::Dense` nothing is called. This card does not switch the program and does not bind anything: the program and the rule are carried and refused when absent, and FT6.11 makes them act.
- test: all three go in `proxima-model-interop/src/generate/chunked_prefill_tests.rs` as flat `#[test]` functions that call `read_operand_parts` on a model loaded from the file's fixtures (the file already carries the workspace allow for test expects; `generate` descendants reach `decode`'s `pub(super)` items through `LoadedModel`; if the method does not resolve, stop and report). `tests_all.rs` gets only the one-line field additions to its `model_with` literal, because that literal is exhaustive.
  - `operand_read_without_a_rule_names_what_is_missing`: the gemma4 fixture, no rule set: `read_operand_parts()` is `Err(InteropError::UnsupportedServingConfig(message))` and the message contains `read rule`;
  - `operand_read_on_a_family_without_the_program_names_what_is_missing`: the dense fixture with `set_read_rule(|_, _, _| {})`: it is `Err(UnsupportedServingConfig(message))` and the message contains `read operand program`;
  - `operand_read_with_a_rule_returns_the_program_and_the_rule`: the gemma4 fixture with `set_read_rule(|_, _, flags| flags.fill(1.0))`: `read_operand_parts()` is `Ok`, the returned program's root node id equals the one `read_operand_program_with_kv_layout` reports for the same fixture, and calling the returned rule on a 4 element slice of zeros leaves four ones (the stored closure is the one that was set).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_25 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/operand_read_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/load_model.rs`, `proxima-model-interop/src/generate/pregather.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/tests_all.rs`, `proxima-model-interop/src/generate/chunked_prefill_tests.rs`
- commit: `feat(interop): carry a read operand program and rule on the model`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: switch the decode program; bind or fill any input; write anything under `ReadSpec::Dense`; add an error variant; add a mutex to the model.
- gpu: none (a synthetic fixture on the CPU backend; no checkpoint file)

### 6.11 the decode step binds the read rule's output

- id: FT6.11
- needs: FT6.9, FT6.10, FT6.25, FT2.17, FT2.21
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `proxima-model-interop/src/serving_settings/refusal.rs::ServingRefusal::ReadHookNotBound` and `serving_settings/refusals.rs::{check_read, refusals_for, tests}` (FT2.34, FT2.17, FT2.21): the settings refusal that says no read hook is bound, which this card's decode arm makes false;
  - `proxima-model-interop/src/generate/decode.rs (~lines 3645 and 3769-3800 at 4b4be6cf)`: how the decode loop chooses `self.speculative_verify_program` for a step, `(~line 3828)` the `build_position_inputs` call whose named inputs this card extends, and `LoadedModel::read_operand_parts` (FT6.25), called at the step before the loop;
  - `proxima-model-interop/src/error.rs::InteropError::UnsupportedServingConfig (~line 322)`: the existing typed error for a serving configuration the model cannot honor;
  - `proxima-model-interop/src/generate/kv_ring.rs::KvRing::{live_rows, bound_extent}` (~lines 68 and 75 at 4b4be6cf) and `::sliding_ring_geometry (~line 255)`: a ring layer hands the program `min(cached_len, window)` rows and its leaves are bound by `min(kv_bound_extent, window)`, which is the per-layer count and bound this card must pass; `KvRing` is reached from `generate` descendants as `super::KvRing` (a private `use` in `generate/mod.rs`, ~line 260);
  - `proxima-model-interop/src/generate/residency_caches.rs::cache_leaf_bound_slot (~line 902)`: the slot of a declared leaf's first extent, which is how a layer's leaf is told to be ring bound (`proxima_tensor::spec::SLIDING_KV_SYMBOL`, `architecture::symbols::SLIDING_KV_BOUND`) or full bound (`symbols::KV_BOUND`), and `proxima-tensor/src/spec/lfm2_single_range_cached.rs (~lines 1519-1523)`: `cache_bound` is `Extent::Symbolic(SLIDING_KV_SYMBOL)` for a layer with a mask window when the ring layout is on, else `Extent::Symbolic(1)`; the read leaf takes that same extent from `key_extent` (FT6.4);
  - `proxima-model-interop/src/generate/chunked_prefill_tests.rs::{gemma4_checkpoint, config, prefill}` (~lines 94, 186 and 205 at 4b4be6cf): the synthetic gemma4 checkpoint with 3 layers, sliding pattern `[true, true, false]` and a sliding window of 8 (layers 0 and 1 are ring layers, layer 2 is a full layer), which is the model this card's decode test runs; `prefill` shows the `run_decode_loop_observed_seeded` call to copy (its 12 arguments) and the `LogitsSink::Collect` use.
- change:
  1. `proxima-model-interop/src/generate/decode.rs`: add three `pub(super)` functions (about 45 lines in all):
     - `read_skip_extent(slot: u16, ring: Option<&KvRing>, cached_len: usize, kv_bound_extent: usize) -> Result<(usize, usize), InteropError>`, returning `(cached_rows, bound_rows)` for one layer's `read_skip.{layer}` leaf, by the symbol slot that bounds the leaf. Slot `symbols::SLIDING_KV_BOUND` with `Some(ring)` gives `(ring.live_rows(cached_len), ring.bound_extent(kv_bound_extent))`, that is `(min(cached_len, window), min(kv_bound_extent, window))`; with `None` it is `Err(InteropError::UnsupportedServingConfig("a sliding read skip leaf needs a sliding ring; this model has none".to_owned()))`. Slot `symbols::KV_BOUND` gives `(cached_len, kv_bound_extent)`. Any other slot is `Err(UnsupportedServingConfig(format!("read skip leaf is bound by unknown slot {slot}")))`. When `cached_rows > bound_rows` it is `Err(UnsupportedServingConfig(format!("read skip leaf holds {bound_rows} rows but the layer hands {cached_rows}")))`. A ring layer and a full layer therefore get their own row count and their own leaf length; one number never serves both. A shared-key layer takes the slot its own leaf declares, and every ring layer shares the one window (`sliding_ring_geometry`'s doc: a program binds one sliding slot);
     - `read_skip_layer_slots(program: &[Op]) -> Vec<(usize, u16)>`: for every `Op::Input` named `read_skip.<layer>`, in program order, `(layer, slot)` where `slot` is `cache_leaf_bound_slot(program, name)`; a leaf whose first extent is static is left out (it has no bound slot, and the bind then names it as unbound);
     - `fill_read_skip(rule: &(dyn Fn(usize, usize, &mut [f32]) + Send + Sync), layer_slots: &[(usize, u16)], ring: Option<&KvRing>, cached_len: usize, kv_bound_extent: usize, scratch: &mut Vec<f32>) -> Result<(), InteropError>`: it clears `scratch`, and for each `(layer, slot)` in order gets `(cached_rows, bound_rows)` from `read_skip_extent(slot, ring, cached_len, kv_bound_extent)?`, resizes `scratch` by `bound_rows` zeros (the layer's segment, starting at the sum of the earlier layers' `bound_rows`), then calls the rule with `(layer, cached_rows, &mut segment[..cached_rows])`, so the rows from `cached_rows` to `bound_rows` stay 0.0. `clear` then `resize` within the existing capacity allocates nothing at the same total length. The rule is a reference because `read_operand_parts` (FT6.25) has already refused a missing rule.
  2. At the single-row decode step, when `serving.attention.read == ReadSpec::Operand`: use the program from the `read_operand_parts` result (FT6.25) instead of the dense program; compute `read_skip_layer_slots` of that program once per call, before the step loop; at each step call `fill_read_skip(rule, &layer_slots, sliding_ring_geometry(&layer_caches).as_ref(), cached_len, kv_bound_extent, &mut read_skip_scratch)`, then add the named input `read_skip.{layer}` for every listed layer as that layer's segment of `read_skip_scratch` (the segment's start is re-derived with `read_skip_extent`, the same function the fill used).
     The statement `self.read_operand_parts()?;` that FT6.25 placed before the step loop becomes `let (program, rule) = self.read_operand_parts()?;`, and `program` and `rule` are the values this step uses.
     Under `ReadSpec::Dense` nothing changes: no program switch, no input, no scratch. Prefill and the verify call sites are untouched.
  3. Settings stop refusing the read the decode step now serves (the FT2.34 amendment in `02-serving-settings.md`, same commit). `proxima-model-interop/src/serving_settings/refusal.rs`: delete the `ReadHookNotBound` variant and its `field_path` arm. `proxima-model-interop/src/serving_settings/refusals.rs`: delete `check_read` and its call in `refusals()` (the other `check_*` calls stay), delete the test `serving_settings_refuses_read_hook_not_bound`, and rewrite the two `serving_settings_refuses_read_needs_` tests (FT2.17) to drop the leading `ReadHookNotBound` from every expected list: `refusals_for(&hybrid)` with `Operand` is `vec![ReadNeedsAttentionLayers { layer: 5 }]`, `refusals_for(&two_range)` is empty, `Operand` with `dense` is `vec![ReadNeedsTwoRangeCache]`, `Operand` with `two_range` is empty, `Operand` with the hybrid single-range descriptor is `vec![ReadNeedsAttentionLayers { layer: 5 }, ReadNeedsTwoRangeCache]`, and the `Dense` rows stay empty. In `serving_settings_refusal_field_paths` (FT2.21) delete table row 14 (`ReadHookNotBound`), so `rows.len() == 13` and the model-free `validate()` rows are eleven (rows 2 and 3 are still excepted); rows 2 and 3 no longer find their variant behind a leading `ReadHookNotBound`, so their `refusals_for` list holds exactly that variant. These three files are the only places `ReadHookNotBound` appears; `git grep -n ReadHookNotBound -- proxima-model-interop` prints nothing afterwards.
- test: all five tests go in `proxima-model-interop/src/generate/chunked_prefill_tests.rs` as flat `#[test]` functions (the file already carries the synthetic gemma4 fixture and the workspace allow for test expects; its `use super::{..}` list gains `KvRing`, `fill_read_skip` and `read_skip_extent`, and it imports `AttentionConfig` from `crate::serving`, `ReadSpec` from `crate::serving_grammar`, `InteropError` from `crate::error` and `symbols` from `crate::architecture`; `generate` descendants reach `decode`'s `pub(super)` items through `super`; if a path does not resolve, stop and report). The model struct files carry no edit in this card: `load_model.rs`, `pregather.rs` and `tests_all.rs` were FT6.25's.
  - `fill_read_skip_follows_the_rule_and_zeroes_the_padding`: `layer_slots = [(0, KV_BOUND), (1, KV_BOUND)]`, no ring, `cached_len = 40`, `kv_bound_extent = 64`, a rule that marks rows `8..24` for layer 0 and nothing for layer 1: `scratch.len() == 128`; layer 0's segment (`scratch[0..64]`) has 16 ones at `8..24` and 48 zeros; layer 1's segment (`scratch[64..128]`) is all zeros; rows `40..64` of layer 0's segment are all zeros;
  - `fill_read_skip_sizes_a_ring_layer_by_the_window`: `layer_slots = [(0, SLIDING_KV_BOUND), (1, KV_BOUND)]`, `ring = Some(&KvRing::new(32, 0, 2, 3, 0))` (window 32, no slack; 2 and 3 are arbitrary row widths), `cached_len = 40`, `kv_bound_extent = 64`, a rule that pushes `(layer, cached_rows, flags.len())` into a `Mutex<Vec<_>>`: the pushed list is exactly `[(0, 32, 32), (1, 40, 40)]` and `scratch.len() == 32 + 64 == 96` (the ring layer is 32 rows wide because `min(64, 32)`, the full layer 64);
  - `fill_read_skip_reuses_the_scratch_across_steps`: two calls with `cached_len` 40 then 41 at the same `kv_bound_extent` and layers leave `scratch.capacity()` unchanged after the first call;
  - `read_skip_extent_names_what_it_cannot_size`: `read_skip_extent(KV_BOUND, None, 40, 64) == Ok((40, 64))`, `read_skip_extent(SLIDING_KV_BOUND, None, 40, 64)` is an `UnsupportedServingConfig` whose message contains `sliding ring`, `read_skip_extent(7, None, 40, 64)` is one whose message contains `slot 7`, and `read_skip_extent(KV_BOUND, None, 65, 64)` is one whose message contains `holds 64 rows`;
  - `read_rule_runs_per_layer_with_the_ring_row_count_on_two_range_gemma4` (the test that exercises the read operand program switch and the input bind; the card's own model test, on the 3 layer synthetic fixture, CPU backend, no checkpoint file): a new helper `fn decode_logits(model: &LoadedModel<'_>, prompt: &str, max_tokens: usize, serving_config: &ServingConfig) -> Vec<f32>` copies `prefill`'s `run_decode_loop_observed_seeded` call with `max_tokens` in place of the literal `1` and returns the last collected logits row. The test: `let bytes = gemma4_checkpoint()`, parse it, `let mut model = LoadedModel::load(..)`; `recorded: Arc<Mutex<Vec<(usize, usize)>>>` (the file's `Mutex`); `model.set_read_rule({ let recorded = Arc::clone(&recorded); move |layer, cached_rows, _flags| recorded.lock().push((layer, cached_rows)) })`; `prompt = prompt_of(20, '3')`; `max_tokens = 3` (step 0 is the prefill and makes no rule call; steps 1 and 2 are single-row decode steps).
    1. dense: `decode_logits` under `config(0)` leaves `recorded` empty (nothing is declared, bound or called under the dense read); keep those logits as `dense`;
    2. operand: `decode_logits` under `ServingConfig { attention: AttentionConfig { read: ReadSpec::Operand }, ..config(0) }` records exactly `[(0, 8), (1, 8), (2, 20), (0, 8), (1, 8), (2, 21)]`, and its logits equal `dense` within `1e-5` by `worst_relative_error` (the rule marks nothing). Derived from the fixture, not yet observed: layers 0 and 1 are ring layers with window 8, so each hands `min(cached, 8) = 8` rows at both steps, and layer 2 is a full layer handing all 20 then 21 cached rows, where the 20 is the prompt's id count (one byte token per digit); if the recorded list differs, stop and report it, do not edit the integers to match;
    3. control: `model.set_read_rule(|_, cached_rows, flags| flags[..cached_rows - 1].fill(1.0))` (skip every cached row but the newest) and the operand run's logits differ from `dense` by more than `1e-5` (otherwise the flags never reached the program).
  - the settings tests of change item 3 are edits to existing tests in `refusals.rs` `tests` (no new test): `serving_settings_refuses_read_needs_attention_layers` and `serving_settings_refuses_read_needs_two_range_cache` with the lists above, and `serving_settings_refusal_field_paths` with 13 rows; `serving_settings_refuses_read_hook_not_bound` no longer exists.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_11 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/fill_read_skip_|read_skip_extent_|read_rule_runs_per_layer_|serving_settings_refuses_read_needs_|serving_settings_refusal_field_paths/)'`
- expect: `8 passed` (5 decode tests + 2 read-needs tests + 1 field-path test)
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration,metal --all-targets`; the validate command of FT2.21 with `refuses_read_hook_not_bound|` removed from its filter prints `20 passed`; `git grep -n ReadHookNotBound -- proxima-model-interop` prints nothing
- stage: `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/chunked_prefill_tests.rs`, `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): bind the read rule output at each decode step`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: write anything under `ReadSpec::Dense`; compute a selection in interop (the rule is the caller's); touch the prefill or verify call sites; add a mutex to the model; add an error variant; pass one row count or one leaf length for every layer; edit `load_model.rs` or `pregather.rs`; keep `ReadHookNotBound`, `check_read` or the hook-not-bound test; edit any `serving_settings` file other than `refusal.rs` and `refusals.rs`.
- gpu: none (the synthetic fixture on the CPU backend; the real-checkpoint check is FT6.23)

### 6.26 the model keeps a key row counter

- id: FT6.26
- needs: FT6.25
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel (~line 803 at 4b4be6cf)`: where a lock-free counter field sits, and the four exhaustive struct literals (`pregather.rs` at ~lines 2741, 2878 and 3010; `tests_all.rs::memory_fit_gate_tests::model_with` at ~line 2597), each of which ends with the two read fields FT6.25 added;
  - `proxima-model-interop/src/generate/chunked_prefill_tests.rs::gemma4_checkpoint (~line 94 at 4b4be6cf)`: the synthetic checkpoint the test loads.
- change:
  1. `proxima-model-interop/src/generate/load_model.rs`: add two `pub(super) AtomicU64` fields, `kv_rows_read` and `kv_decode_steps`, to `LoadedModel` and `pub fn take_kv_read_stats(&self) -> (u64, u64)` returning `(kv_rows_read, kv_decode_steps)` and resetting both with `swap(0, Ordering::Relaxed)`.
     Doc: rows are summed per query row over the layers that carry a read operand; a dense step carries none and adds nothing, so `(0, 0)` under `ReadSpec::Dense`.
  2. Both fields are initialised `AtomicU64::new(0)` in all four exhaustive `LoadedModel` literals (no `..` base): `proxima-model-interop/src/generate/pregather.rs` at ~lines 2741, 2878 and 3010, and `proxima-model-interop/src/generate/tests_all.rs::memory_fit_gate_tests::model_with` at ~line 2597.
     Nothing writes the counters in this commit; FT6.12 writes them from the decode step.
- test: in `proxima-model-interop/src/generate/chunked_prefill_tests.rs` as a flat `#[test]` (`tests_all.rs` gets only the one-line field additions to its `model_with` literal):
  - `take_kv_read_stats_returns_the_counters_and_resets_both`: load the synthetic gemma4 fixture; a fresh model gives `take_kv_read_stats() == (0, 0)`; after `model.kv_rows_read.fetch_add(79, Ordering::Relaxed)` and `model.kv_decode_steps.fetch_add(2, Ordering::Relaxed)` it gives `(79, 2)`, and the call after that gives `(0, 0)` (the swap reset both). The file's `use` list gains `std::sync::atomic::Ordering`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_26 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/take_kv_read_stats_returns_the_counters_and_resets_both/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/load_model.rs`, `proxima-model-interop/src/generate/pregather.rs`, `proxima-model-interop/src/generate/tests_all.rs`, `proxima-model-interop/src/generate/chunked_prefill_tests.rs`
- commit: `feat(interop): add a key row read counter to the model`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: use a mutex; touch `decode.rs`; write the counters from the decode step (FT6.12).
- gpu: none (a synthetic fixture on the CPU backend; no checkpoint file)

### 6.12 count the key rows each decode step reads

- id: FT6.12
- needs: FT6.11, FT6.26
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel::{kv_rows_read, kv_decode_steps, take_kv_read_stats}` (FT6.26): the counter pair this card writes and the reader the test uses;
  - `proxima-model-interop/src/generate/decode.rs::fill_read_skip` and `::read_skip_extent` (FT6.11) and the single-row decode step after its call: the skip rows the step just bound.
- change:
  1. `proxima-model-interop/src/generate/decode.rs`: add `pub(super) fn rows_read_by_layer(cached_rows: usize, skip: &[f32]) -> u64` = `cached_rows + 1` (the cached rows plus the step's own new row) minus the number of non-zero values in `skip[..cached_rows]`; after each single-row step under `ReadSpec::Operand` add the sum of `rows_read_by_layer(cached_rows, segment)` over the listed layers to `kv_rows_read` and add 1 to `kv_decode_steps` (both with `fetch_add(.., Ordering::Relaxed)`); `cached_rows` and `segment` are each layer's own row count and scratch segment as `read_skip_extent` (FT6.11) gives them, so a ring layer counts at most its window and a full layer all its cached rows.
- test: all three go in `proxima-model-interop/src/generate/chunked_prefill_tests.rs` as flat `#[test]` functions (as in FT6.11; its `use super::{..}` list gains `rows_read_by_layer`):
    - `rows_read_by_layer_counts_the_unskipped_rows`: `cached_rows = 40`, skip rows `8..24` gives `41 - 16 = 25`; no skip gives `41`; a skip flag past `cached_rows` (in the padding) is not counted;
    - `rows_read_by_layer_agrees_with_the_worked_block_selection`: a skip vector built from the worked block selection (4 sealed blocks of 16, tail 5, block 2 skipped, so 69 cached rows with rows 32..48 flagged) gives `rows_read_by_layer(69, &skip) - 1 == (2 + 1) * 16 + 5` (53: two selected blocks, one local block, the 5 tail rows);
    - `take_kv_read_stats_sums_the_unskipped_rows_and_resets_on_two_range_gemma4` (exercises the per-step counter writes on the 3 layer synthetic fixture of FT6.11, CPU backend; it reuses FT6.11's `decode_logits`, `prompt_of(20, '3')`, `max_tokens = 3` and the `Operand` config): after `model.set_read_rule(|_, _, _| {})` and one `decode_logits` call, `model.take_kv_read_stats() == (79, 2)` and a second `take_kv_read_stats()` is `(0, 0)` (the swap reset both). Derived from the fixture, not yet observed: 2 single-row decode steps; per step a ring layer (layers 0 and 1, window 8) reads `8 + 1 = 9` rows and the full layer 2 reads `20 + 1` rows at the first step and `21 + 1` at the second, so the rows are `2 * (9 + 9) + (21 + 22) = 36 + 43 = 79`; if the printed pair differs, stop and report it. Then `model.set_read_rule(|_, cached_rows, flags| flags[0] = if cached_rows > 0 { 1.0 } else { 0.0 })` and a second `decode_logits` call gives `(73, 2)` (6 skipped rows: one per layer per step); a dense `decode_logits` call afterwards gives `(0, 0)`;
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_12 cargo nextest run -p proxima-model-interop --features std -E 'test(/rows_read_by_layer_|take_kv_read_stats_sums_the_unskipped_rows_and_resets_on_two_range_gemma4/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/chunked_prefill_tests.rs`
- commit: `feat(interop): count the key rows each decode step reads`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: use a mutex; allocate in the step; touch the prefill path; edit `load_model.rs`, `pregather.rs` or `tests_all.rs`.
- gpu: none (the synthetic fixture on the CPU backend; the real-checkpoint check is FT6.24)

### 6.13 an all-visible read keeps llama's ids on gemma4 E2B (oracle)

- id: FT6.13
- needs: FT6.12, FT2.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 667 at 4b4be6cf)`, `::llama_cases (~line 616)` and the `llama_parity_gemma4_e2b` test (~line 725): the comparison of the first generated ids against the vendored llama ids, per prompt;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` (vendored once, never re-queried);
  - the slice 2 re-cut `ServingConfig.attention.read` field (FT2.1) and `LoadedModel::set_read_rule` (FT6.25, bound by FT6.11);
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel::take_kv_read_stats` (FT6.26, written by FT6.12).
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: refactor `llama_parity(checkpoint)` into `llama_parity_with(checkpoint: &Checkpoint, config: &ServingConfig, prepare: impl FnOnce(&mut LoadedModel))` (the model is made `mut`, `prepare` runs after load) and make `llama_parity` call it with its current config and a no-op closure. `llama_parity_with` returns `ParityRun { generated_lens: Vec<usize>, compared_lens: Vec<usize>, stats: (u64, u64) }` (one entry per prompt, in file order: the number of ids the model generated, `compared_len` as `llama_parity` computes it, and the model's `take_kv_read_stats()` read once after the last prompt); it still asserts, inside, that the compared ids equal the vendored llama ids.
  2. add the test `read_all_visible_oracle_gemma4_e2b`: config `ServingConfig { prompt_cache: PromptCacheConfig::off(), attention: <read = ReadSpec::Operand>, ..ServingConfig::default() }`, `prepare` sets the rule `|_, _, _| {}` (marks nothing, so every row is visible).
- test: `read_all_visible_oracle_gemma4_e2b` asserts, on the returned `run`: `run.compared_lens == vec![3, 32, 1]` (the 3 prompts of `gemma4_e2b/llama_ids.json` hold 3, 32 and 1 generated ids, read from that file; 36 ids compared in all; a run that generates nothing gives zeros here and fails, so this is the count that separates a run from no run), `run.generated_lens == vec![32, 32, 32]` and `run.stats == (129_115, 93)` (35 is `gemma4.block_count`, `gemma4_e2b/gguf_kv.txt` line 17). The ids equal the vendored llama ids over those 36 positions through the comparison inside `llama_parity_with`.
  Derived from code, not yet observed. The decode loop (`decode_until_stop_or_budget`, `residency_caches.rs` ~line 3448) stops only on `vocab.eos_token_id()` (id 1, "eos id only", `pipeline-as-data/SPEC.md` row H15), not on llama's end-of-generation id 106, and `llama_parity_with` passes `LLAMA_GENERATED_TOKENS` (32) as the budget, so each of the 3 prompts generates 32 ids unless id 1 appears; `compared_lens` is `[3, 32, 1]` only because `compared_len` takes the `.min()` against the 3, 32 and 1 vendored llama ids. The prompt id counts 6, 8 and 57 are read from `llama_ids.json`, not from the run. With `n = 32` ids the decode steps per prompt are `n - 1 = 31` (the first id comes from the prefill) and step `k` has `p + k` cached rows plus its own new row with no row skipped, so the rows per layer are `sum over k in 0..31 of (p + k + 1) = 31 * (p + 1) + 465`: `682 + 744 + 2263 = 3689`, times 35 layers is `129_115` rows, and `3 * 31 = 93` steps. If `generated_lens` prints other than `[32, 32, 32]` (id 1 was generated) or the stats differ, stop and report the printed values; do not edit the literals to match.
  Ring layers: 28 of the 35 layers are sliding (`gemma4_e2b/swa_layers.txt`, 28 lines with `is_swa = 1`; `gguf_kv.txt` line 29 gives `gemma4.attention.sliding_window = 512`), and the model loads with the ring layout, so each of those hands the rule `min(cached rows, 512)` rows (FT6.11); the deepest step has `57 + 30 + 1 = 88` rows, below 512, so the ring never truncates and every layer counts `p + k + 1` rows as the sum above assumes. If a prompt or budget ever exceeded 512 rows, the sliding layers would count 512 and the stats would differ.
  A checkpoint whose dense parity already fails is a finding to report, not to skip.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_13 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_all_visible_oracle_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check an all-visible read on gemma4 e2b ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; weaken the comparator; add a qwen or openchat case.
- gpu: one run (`-j 1`, one checkpoint load), waiting for a quiet box (CARDS.md machine safety: check `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"`).

### 6.14 an explicit dense read keeps llama's ids on gemma4 E2B (oracle)

- id: FT6.14
- needs: FT6.13
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity_with` (FT6.13);
  - `proxima-model-interop/tests/arch_data_baseline.rs::arch_data_digest_gemma4_e2b (~line 362 at 4b4be6cf)`: the op graph digest that the slice exit command runs (dense adds no graph nodes; it is not part of this card's run).
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `read_dense_oracle_gemma4_e2b`: the config sets `attention.read = ReadSpec::Dense` explicitly, no rule is set, and `llama_parity_with` returns the `ParityRun` of FT6.13.
- test: `read_dense_oracle_gemma4_e2b` asserts, on the returned `run`: `run.compared_lens == vec![3, 32, 1]` (36 ids compared against the vendored llama ids by the comparison inside `llama_parity_with`; this count, not the stats, is what separates a run from no run) and `run.stats == (0, 0)` (the dense step carries no read operand and counts nothing; `(0, 0)` alone is also what no run produces, which is why the compared count is asserted beside it).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_14 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_dense_oracle_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check explicit dense read on gemma4 e2b ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; add a read operand test here.
- gpu: one run (`-j 1`, one checkpoint load), waiting for a quiet box.

### 6.15 the streaming cached attention routine skips flagged cached rows

- id: FT6.15
- needs: none
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/physical.rs::stream_cached_attention_split_gqa (~line 325 at a7c08c4c)`: the buffer-length checks before the loops, and the key-row loop (`for key_row in 0..key_rows`) whose band test `continue`s past a row, so no key or value of that row is read; `CachedAttentionScore (~line 308)` is the `score` argument;
  - `proxima-tensor/src/cpu/run_node.rs::run_cached_attention (~line 775 at a7c08c4c)` and `proxima-tensor/benches/bench_cached_attention.rs (~lines 129 and 185)`: the callers of the routine; they stay byte for byte as they are;
  - `proxima-tensor/src/physical.rs` tests (~line 679 on): `streams_cached_and_new_ranges_without_materialized_weights` and its neighbours, as the style for a test over `stream_cached_attention_split_gqa` (the module is `mod tests`).
- change:
  1. `proxima-tensor/src/physical.rs`: the body of `stream_cached_attention_split_gqa` moves into `pub fn stream_cached_attention_split_gqa_skipping(queries: [&[f32]; 2], keys: [[&[f32]; 2]; 2], values: [&[f32]; 2], output: &mut [f32], extents: AttentionExtents, rotary: CachedAttentionRotary<'_>, score: (CachedAttentionScore, Option<&[f32]>)) -> bool` (`#[must_use]`). The last argument is a pair so the routine keeps seven parameters (the clippy argument limit); the second element is the per-cached-row skip flags (1.0 skips a row, 0.0 reads it), and the first is exactly what the old `score` argument was.
     - Right after the existing length checks: when the flags are `Some` and their length is not `extents.cached_key_rows as usize`, return `false`;
     - at the top of the key-row loop: `if range_index == 0 && flags.is_some_and(|flags| flags[key_row] != 0.0) { continue; }` (a skipped cached row adds nothing to the running maximum, the running sum or the output; the new range is never skipped; positions of the other rows are unchanged, so the band test is unaffected);
     - `stream_cached_attention_split_gqa` keeps its exact signature and `#[must_use]`, and its body is the one call `stream_cached_attention_split_gqa_skipping(queries, keys, values, output, extents, rotary, (score, None))`.
     Doc on the new function: points to `stream_cached_attention_split_gqa` for the layout; says a flagged cached row costs no key or value read; says the flags are caller supplied per-cached-row skip flags (1.0 skips a row, 0.0 reads it), one per cached key row.
- test: add in `physical.rs`'s `mod tests`, on one fixture shared by the tests (`kv_heads = 1`, `query_groups = 1`, `head_dim = 2`, `rotary_dim = 2` so `pair_dim = 1`, `query_rows = 1`, `cached_key_rows = 5`, `new_key_rows = 1`, `scale = 1.0`, bands `[CausalBand { lower_inclusive: i64::MIN, upper_inclusive: i64::MAX }, CausalBand { lower_inclusive: i64::MIN, upper_inclusive: 0 }]`, no pass plane; query even `[0.5]`, query odd `[-0.25]`; cached key even `[0.2, -0.4, 0.9, 0.1, 0.6]`, cached key odd `[0.3, 0.5, -0.2, 0.7, -0.1]`; new key even `[0.4]`, new key odd `[0.2]`; cached values (rows of 2) `[1.0, 0.0, 0.0, 1.0, 2.0, -1.0, -1.0, 3.0, 0.5, 0.5]`; new value `[1.5, -0.5]`):
  - `cached_row_skip_equals_deleted_rows`: flags `[0.0, 1.0, 0.0, 1.0, 0.0]` give an output exactly equal (`==`, elementwise) to the output of `stream_cached_attention_split_gqa` over the same fixture with rows 1 and 3 deleted from every cached buffer and `cached_key_rows = 3`; control: flags all `0.0` give an output that differs from the skipping output by more than `1e-3` in some element;
  - `cached_row_skip_without_flags_equals_the_plain_routine`: `stream_cached_attention_split_gqa_skipping(.., (score, None))` and the plain routine give exactly equal outputs, and so do flags all `0.0`;
  - `cached_row_skip_refuses_flags_of_the_wrong_length`: flags of length 4 return `false`, and flags of length 6 return `false`, for `cached_key_rows = 5`; flags of length 5 return `true`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_15 cargo nextest run -p proxima-tensor -E 'test(/cached_row_skip_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`; `cargo check -p proxima-tensor --benches`
- stage: `proxima-tensor/src/physical.rs`
- commit: `feat(tensor): skip flagged cached rows in streaming attention`
- done when: the expect line printed, clippy and check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `CachedAttentionScore`, `CachedAttentionRotary` or `AttentionExtents` (every literal of them stays as it is); edit `run_node.rs` or the bench; skip a row of the new range.
- gpu: none

### 6.16 the cached-attention fusion keeps the fused path under a read skip operand, and the CPU interpreter honors it

- id: FT6.16
- needs: FT6.4, FT6.15
- budget: 20 min
- crate(s): proxima-tensor (features: read-mask-fusion, new)
- read first:
  - `proxima-tensor/src/bind/dead_code_cached_attention.rs::cached_padding_mask_lower_bound (~line 577 at 4b4be6cf)` and `::unwrap_cached_padding_select (~line 553)`: the gemma template that recognizes `Select(predicate, -inf, inner)` with `predicate` either the padding `Greater` or `Maximum(padding, too_old)`; and the by-name leaf precedent for `cached_len` (`named_cached_len`);
  - `proxima-tensor/src/bind/types_layout_boundop.rs::BoundOpKind::CachedAttention (~line 190)` doc (operands count 8, 9, 11 or 12, "no tenth operand because the two shapes are never both live", so 10 and 13 are free for the read operand) and `BoundOp::operands (~line 572)`;
  - `proxima-tensor/src/cpu/run_node.rs::run_cached_attention (~line 570 at a7c08c4c)`: `expected_lengths` (`&[11, 12]` or `&[8, 9]`), `cached_len_index`, the pass plane start, and the call of the streaming routine (~line 775), which FT6.15 left on the plain routine; and `physical.rs::stream_cached_attention_split_gqa_skipping` (FT6.15), the routine this card calls;
  - `proxima-tensor/src/spec/tests.rs::read_hook_decode` (FT6.3): returns the program, its logits root and its symbols, which the tests below bind with `crate::bind::bind` as `bind/tests.rs::cached_attention_rewrite_replaces_the_bound_attention_subgraph (~line 432 at a7c08c4c)` does, counting `BoundOpKind::CachedAttention` ops.
- change:
  1. `proxima-tensor/Cargo.toml`: add `read-mask-fusion = ["metal-fuse-attn-decode"]` (default off; the feature style of `metal-fuse-attn-decode = ["cached-attention-streaming"]`, `~line 42`).
  2. `proxima-tensor/src/bind/dead_code_cached_attention.rs`: under the feature, when the mask predicate is `Maximum(P, skip)` with `P` a predicate the existing recognizer accepts, `skip` an `Op::Input` whose name starts with `read_skip.` (found by name, the `cached_len` precedent), and the fused op already carries its `cached_len` operand, the fused op is kept and the `skip` leaf is appended as the LAST operand (after `cached_len` and, for partial rotary, after the pass planes), built the way the `cached_len` operand is built. Without a `cached_len` operand the match fails and the op stays unfused, because 8 plus the skip would read as 9 operands, the `cached_len` count.
  3. `proxima-tensor/src/bind/types_layout_boundop.rs`: add `pub fn read_rows(&self) -> Option<&(NodeId, Layout, Option<Lookup>)>` on `BoundOpKind` (on the kind, so the omega form classifier, which sees only the kind, can call it; compiled whether or not the feature is on, because it only reads the operand count): for a `CachedAttention` with `cached_key_rows != 0` and `operands.len()` equal to 10 (when `rotary_dim == head_dim`) or 13 (when `rotary_dim < head_dim`) it is the last operand, else `None`; update the operand count doc of the variant to say 8, 9, 10, 11, 12 or 13 and why 10 and 13 are the read operand. No field is added to the variant, so no literal of `BoundOpKind::CachedAttention` changes.
  4. `proxima-tensor/src/cpu/run_node.rs::run_cached_attention` (not feature-gated; with no read operand every line it runs is what it is today): `let read_operand = resolved.kind.read_rows();` and `let base_operand_count = operands.len() - usize::from(read_operand.is_some());`; `expected_lengths.contains(&base_operand_count)` and `cached_len_index` use `base_operand_count` where they used `operands.len()`. When `read_operand` is `Some((node, layout, _))`: the layout must be zero-based and contiguous (the check the eight sources get, else `NotLowerable` with a reason naming the read operand), the flags are `buffer_of(buffers, *node)?` sliced to `..live_cached_key_rows_usize` (`out_of_range` when shorter), and the streaming call becomes `stream_cached_attention_split_gqa_skipping(.., (score, Some(flags)))`; with `None` it is `(score, None)`. The pass plane start is unchanged (the read operand is last).
- test: add in `proxima-tensor/src/spec/tests.rs`, inside `gemma4_synthetic_parity`, on `two_layer_read_schedule()` (FT6.3) with `total_rows = 40` (the program from `read_hook_decode` with `read_skip_operand` (FT6.4) as the read function and `kept_rows = 0..40`, bound with `crate::bind::bind(&program, &crate::shape::infer(&program, &symbols).expect("the program infers"), &[logits_root], NumericPolicy::bit_exact())`):
  - `read_skip_keeps_cached_attention_fused` (`#[cfg(feature = "read-mask-fusion")]`): exactly 2 bound ops are `BoundOpKind::CachedAttention` and `kind.read_rows().is_some()` for both;
  - `fused_read_skip_equals_the_deleted_rows_dense_program` (`#[cfg(feature = "read-mask-fusion")]`): `skipped` is `read_hook_decode` over `kept_rows = 0..40` with `read_skip_operand` as the read function, `extra_inputs` holding `read_skip.0` and `read_skip.1` (each 1.0 on rows `8..24` and 0.0 elsewhere) and an empty `extra_outputs`; `deleted` is `read_hook_decode` over `kept_rows = (0..8).chain(24..40)` with the identity function and no extra inputs; `skipped.0[0]` and `deleted.0[0]` agree within `1e-5` (the fused interpreter executes the skip).
    Control against a vacuous pass: with `shapes = crate::shape::infer(&skipped.1, &skipped.3).expect("the program infers")`, `crate::bind::bind(&skipped.1, &shapes, &[skipped.2], NumericPolicy::bit_exact())` holds exactly 2 `BoundOpKind::CachedAttention` ops and both have `kind.read_rows().is_some()`, and `crate::bind::bind_with_fusion(&skipped.1, &shapes, &[skipped.2], false, NumericPolicy::bit_exact())` (the signature on main is `bind_with_fusion(program, shapes, outputs, fuse_cached_attention, numeric_policy)`) holds 0 `CachedAttention` ops (the recognizer is what produces them).
    The skip moved the answer: `skipped.0[0]` differs from `read_hook_decode(.., kept_rows = 0..40, no extra inputs, the identity function).0[0]` by more than `1e-4` in some element;
  - `read_skip_without_the_feature_leaves_cached_attention_unfused` (`#[cfg(not(feature = "read-mask-fusion"))]`, run with `metal-fuse-attn-decode`): the same program as the first test binds with 0 `CachedAttention` ops (the control: the recognizer is what keeps the fusion).
  The two `read_skip_operand_` tests of FT6.4 also run under the feature in the first validate command: with the interpreter honoring the operand, they are green through the fused path.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_16 cargo nextest run -p proxima-tensor --features read-mask-fusion -E 'test(/read_skip_keeps_cached_attention_fused|fused_read_skip_equals_the_deleted_rows_dense_program|read_skip_operand_/)'`; then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_16 cargo nextest run -p proxima-tensor --features metal-fuse-attn-decode -E 'test(/read_skip_without_the_feature_leaves_cached_attention_unfused/)'`; then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_16 cargo nextest run -p proxima-tensor -E 'test(/read_skip_operand_/)'`
- expect: `4 passed` (1 + 1 + 2), then `1 passed`, then `2 passed`
- also green: `cargo clippy -p proxima-tensor --features read-mask-fusion --all-targets`; `cargo clippy -p proxima-tensor --all-targets`; `cargo check -p omega --features metal`
- stage: `proxima-tensor/Cargo.toml`, `proxima-tensor/src/bind/dead_code_cached_attention.rs`, `proxima-tensor/src/bind/types_layout_boundop.rs`, `proxima-tensor/src/cpu/run_node.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): keep cached attention fused under a read skip operand`
- done when: the expect lines printed, clippy and checks clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `omega/`; change the dense or windowed recognition; add a field to `BoundOpKind::CachedAttention`; change the interpreter's text on a path without a read operand; edit `physical.rs`.
- gpu: none

### 6.17 the cached-attention kernel skips flagged rows before the key read

- id: FT6.17
- needs: FT6.16
- budget: 20 min
- crate(s): omega (features: metal, read-mask-fusion)
- read first:
  - `omega/src/msl/cached_attention_render.rs::cached_attention_scalar_score_body (~line 33 at a7c08c4c)` and `::render_cached_attention (~line 51)`: the per-key loop body shared by the three sequential walks (its first line declares `cached`; the next lines `continue` on the band), the `pass_base_index`, `out_index_offset` and `out_buffer_index` lines (~lines 155-186) that fix the buffer indices, and the kernel signature `format!` (~line 234);
  - `omega/src/msl/signature_tokens_prelude.rs::cached_attention_form (~line 1728 at a7c08c4c)` and `::entry_name (~line 291)`: the one classifier every renderer, grid, packer and name matches on; it reads the operand count (`base_operand_len + 1` means the ninth operand), so an op with one more trailing operand is classified `Static` today; the `row_tiled_form` and `decode_split_form` attempts follow;
  - `omega/Cargo.toml (~line 29 and ~line 50)`: passthrough feature style (`metal-fuse-attn-decode = ["proxima-tensor/metal-fuse-attn-decode"]`, `moe-topk-fusion = ["proxima-tensor/moe-topk-fusion"]`);
  - `omega/src/msl/attn_split_tests.rs` and `omega/src/msl/attn_golden_tests.rs::attention_op`: the device-free op builder (`attention_op(operand_count, groups, head_dim, cached_key_rows, new_key_rows, cached_lower_inclusive)`), the classifier test (`the_form_classifier_reads_the_operand_count_and_the_row_discriminator`) and the byte-identity gate (`golden_identity::attention_sources_match_the_recorded_main_goldens`, which compares 5 kernel texts, 11 with `metal-attn-split-decode`, against `omega/tests/fixtures/attn_split_decode/main_*.msl`).
- change:
  1. `omega/Cargo.toml`: `read-mask-fusion = ["metal-fuse-attn-decode", "proxima-tensor/read-mask-fusion"]`.
  2. `omega/src/msl/signature_tokens_prelude.rs`:
     - `cached_attention_form`: `let read_operands = usize::from(kind.read_rows().is_some());` and the count test becomes `operands.len() != base_operand_len + 1 + read_operands`; right after the `cached_key_rows == 0` branch and before the `row_tiled_form` attempt, `if read_operands == 1 { return Some(CachedAttentionForm::TwoRangeCachedBound); }`. Doc: the row-tiled and decode-split kernels do not read the skip operand, so an op that carries one is never given them; the plain two-range kernel does read it;
     - `entry_name`: the `CachedAttention` arm appends `_rr` to the name it builds when `resolved.kind.read_rows().is_some()`.
  3. `omega/src/msl/cached_attention_render.rs`: `cached_attention_scalar_score_body(pass_present: bool)` becomes `cached_attention_scalar_score_body(pass_present: bool, read_present: bool)`; when `read_present` it inserts the line `        if (cached && read_skip[key] != 0.0f) { continue; }\n` immediately after the line that declares `cached` and before the `relative` lines, so a skipped row costs no K or V memory read; its three call sites pass `read_present`. In `render_cached_attention`: `let read_present = resolved.kind.read_rows().is_some();`, `let read_skip_index = pass_base_index + out_index_offset;`, `out_buffer_index` becomes `read_skip_index + usize::from(read_present)`, and the kernel signature gains `{read_param}` between `{pass_param}` and `, device {element_type}* out`, with `read_param = format!(", device const {element_type}* read_skip [[buffer({read_skip_index})]]")` when `read_present` and an empty string otherwise. The kernel text and the buffer indices are unchanged when `read_present` is false.
- test: add in `omega/src/msl/attn_split_tests.rs` (`#[cfg(feature = "read-mask-fusion")]`):
  - `read_rows_kernel_skips_before_reading_keys`: `emit(&attention_op(10, 8, 512, 2048, 1, i64::MIN), &PackedOperands::new(), NumericPolicy::bit_exact())` (unqualified, through the module's `use super::*`, as `attn_golden_tests.rs` calls it) has a source that contains `read_skip[key]` exactly once, where `read_skip[key]` appears before the first `in2[kbase`, contains `* read_skip [[buffer(9)]]`, `* out [[buffer(10)]]` and `constant Uniforms& u [[buffer(11)]]`, and an entry that ends with `_cb_rr`; the source of `attention_op(9, 8, 512, 2048, 1, i64::MIN)` under the same policy contains no `read_skip` and an entry that does not end with `_rr`;
  - `a_read_operand_op_is_classified_as_the_two_range_cached_bound_form`: `cached_attention_form` of `attention_op(10, 8, 512, 2048, 1, i64::MIN).kind` under `llama_relaxed()` is `Some(CachedAttentionForm::TwoRangeCachedBound)`, and so is the partial-rotary op: build it as `let mut partial_rotary = attention_op(13, 8, 512, 2048, 1, i64::MIN);`, then `let BoundOpKind::CachedAttention { rotary_dim, .. } = &mut partial_rotary.kind else { unreachable!("attention_op always builds a CachedAttention kind") }; *rotary_dim = 256;` (the same mutation `a_twelve_operand_single_range_op_is_named_as_the_dynamic_form` in this file uses; 13 operands are 8 base, 3 pass planes, the ninth and the read operand), and `cached_attention_form(&partial_rotary.kind, ..)` under `llama_relaxed()` is also `Some(CachedAttentionForm::TwoRangeCachedBound)`;
  - `a_read_operand_op_is_never_given_the_decode_split_form` (`#[cfg(all(feature = "read-mask-fusion", feature = "metal-attn-split-decode"))]`): under `llama_relaxed()` `cached_attention_form` of `attention_op(9, 8, 512, 2048, 1, i64::MIN).kind` is `Some(CachedAttentionForm::TwoRangeDecodeSplit { .. })` (the control: that op is decode-split without the operand) and of `attention_op(10, 8, 512, 2048, 1, i64::MIN).kind` is `Some(CachedAttentionForm::TwoRangeCachedBound)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_17 cargo nextest run -p omega --features metal,read-mask-fusion -E 'test(/read_rows_kernel_skips_before_reading_keys|a_read_operand_op_is_classified_as_the_two_range_cached_bound_form|attention_sources_match_the_recorded_main_goldens/)'`; then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_17 cargo nextest run -p omega --features metal,read-mask-fusion,metal-attn-split-decode -E 'test(/a_read_operand_op_is_never_given_the_decode_split_form|attention_sources_match_the_recorded_main_goldens/)'`; then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_17 cargo check -p omega --features metal > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards06/ft_6_17_check.log 2>&1`, followed by `grep -c -E '^(warning|error)' /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards06/ft_6_17_check.log` and `grep -c -E '^ +Finished ' /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards06/ft_6_17_check.log`
- expect: `3 passed` for the first command (the golden gate holds the kernel text of every op without the operand byte for byte); `2 passed` for the second; for the third, the first grep prints `0` (the feature off build prints no warning and no error line) and the second prints `1` (the one `Finished` line, so a build that ran is told apart from a build that never started, which also prints `0` for the first grep)
- also green: `cargo clippy -p omega --features metal,read-mask-fusion --all-targets`
- stage: `omega/Cargo.toml`, `omega/src/msl/cached_attention_render.rs`, `omega/src/msl/signature_tokens_prelude.rs`, `omega/src/msl/attn_split_tests.rs`
- commit: `feat(omega): skip flagged key rows in cached attention`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the kernel text or the form of an op without the operand; edit the row-tiled kernel (next card); edit the decode-split kernel.
- gpu: none

### 6.18 the row-tiled kernel skips flagged rows, checked on Metal

- id: FT6.18
- needs: FT6.17
- budget: 20 min
- crate(s): omega (features: metal, read-mask-fusion, metal-attn-split-rows)
- read first:
  - `omega/src/msl/cached_attention_row_tiled.rs::render_cached_attention_row_tiled (~line 31 at a7c08c4c)` and its `ROW_TILED_KERNEL` template: the fixed buffer indices (`out` at `[[buffer(9)]]`, `u` at `[[buffer(10)]]`) and the cached-key mask line `if (key < slice_end && (key - live - query_row) >= cached_lower)` in the score normalization loop (the keys of a block are read by `simdgroup_load` before that line, so this form skips by masking the score to `-INFINITY`, which gives the flagged row weight 0; it does not save the key read);
  - `omega/src/msl/signature_tokens_prelude.rs::cached_attention_form` (FT6.17): the guard to move, `row_tiled_form (~line 1886)` for the shape it requires (`query_rows == new_key_rows` in `[mma_min_query_rows, max_query_rows]` = `[2, 64]` from `omega/omega-runtime.toml`, full rotary, `query_groups` a multiple of 8, `cached_key_rows` a multiple of 8);
  - `omega/tests/cached_attention_row_tiled_parity.rs` and `omega/tests/gemma4_rows_support/mod.rs::fixture (~line 161 at a7c08c4c)`: how a test binds the gemma4-shaped two-layer program, checks the entry names, and compares `omega::execute_plan_named` (Metal) with `evaluate_quantized_named_with_scratch` (CPU) at relative tolerance `1e-4`;
  - `omega/src/msl/cached_attention_render.rs` (FT6.17): the operand and buffer index shift to copy.
- change:
  1. `omega/src/msl/signature_tokens_prelude.rs::cached_attention_form`: the guard FT6.17 added moves to between the `row_tiled_form` attempt and the `decode_split_form` attempt, so a read op may take the row-tiled form and still never takes the decode-split form (which does not read the operand; the plain two-range kernel serves it).
  2. `omega/src/msl/cached_attention_row_tiled.rs`: the `ROW_TILED_KERNEL` template gains the tokens `@READ_PARAM@`, `@OUT_BUFFER@`, `@UNIFORMS_BUFFER@` and `@READ_GUARD@`: `@READ_PARAM@` sits between `in8`'s declaration and `out`; the substitution is `, device const float* read_skip [[buffer(9)]]`, `10` and `11` and ` && read_skip[key] == 0.0f` when `resolved.kind.read_rows().is_some()`, and an empty string, `9`, `10` and an empty string otherwise (the text without the operand is byte for byte what it is today); `@READ_GUARD@` is appended inside the parentheses of the cached-key mask line above.
  3. `omega/tests/gemma4_rows_support/mod.rs`: extract the tail of `fixture` (shape inference and the named input generation) into `fn assemble(program: Vec<Op>, logits: NodeId, rows: usize, cached_len: usize) -> Fixture` that `fixture` calls, and add `pub fn fixture_with_read_skip(rows: usize, cached_len: usize, skipped: core::ops::Range<usize>) -> Fixture`: the same program built with `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` (the arguments of `fixture`, then `1` and `&mut |program, layer, key_extent, _, _, cached_mask| read_skip_operand(program, layer, key_extent, cached_mask)`), assembled the same way, with every named input whose name starts with `read_skip.` replaced by 1.0 for the rows in `skipped` and 0.0 elsewhere.
  4. new file `omega/tests/read_skip_attention.rs` (`#![cfg(all(target_os = "macos", feature = "metal", feature = "read-mask-fusion", feature = "metal-attn-split-rows"))]`, the module layout of `cached_attention_row_tiled_parity.rs`).
- test: `read_skip_attention_skips_rows_on_metal` in `omega/tests/read_skip_attention.rs`: for `rows` in `[1, 2]` over 40 live cached rows (`fixture_with_read_skip(rows, 40, 8..24)`, bound at the capacity bucket of 64): bind with `production_numeric_policy()`; both attention ops have `kind.read_rows().is_some()`; their entry names end with `_cb_rr` at `rows = 1` (the plain kernel) and `_rt_rr` at `rows = 2` (the row-tiled kernel); then assert (4 assertions in all, 2 per `rows`):
  - the Metal root (`omega::plan_named` and `execute_plan_named`) equals the CPU interpreter's root within relative `1e-4`;
  - the control: the Metal root of `fixture_with_read_skip(rows, 40, 0..0)` (no row skipped) differs from the skipped Metal root by more than relative `1e-3` (the skip moved the answer).
  The CPU interpreter's skip equals deleting the rows (FT6.16), so Metal equal to the CPU is Metal equal to deletion.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_18 cargo nextest run -p omega --features metal,read-mask-fusion,metal-attn-split-rows -j 1 -E 'test(/read_skip_attention_skips_rows_on_metal/)'`
- expect: `1 passed`
- also green: `cargo clippy -p omega --features metal,read-mask-fusion,metal-attn-split-rows --all-targets`; `cargo nextest run -p omega --features metal,read-mask-fusion,metal-attn-split-rows -E 'test(/attention_sources_match_the_recorded_main_goldens/)'` prints `1 passed`
- stage: `omega/src/msl/cached_attention_row_tiled.rs`, `omega/src/msl/signature_tokens_prelude.rs`, `omega/tests/gemma4_rows_support/mod.rs`, `omega/tests/read_skip_attention.rs`
- commit: `feat(omega): skip flagged key rows in the row-tiled attention kernel`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit the decode-split kernel (an op with a read operand is classified as the plain form instead, FT6.17; a measured reason is needed before touching it); change the text of the row-tiled kernel for an op without the operand.
- gpu: one run (`-j 1`), waiting for a quiet box.

### 6.19 an all-visible read keeps llama's ids on gemma4 26B (oracle)

- id: FT6.19
- needs: FT6.13
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity_with` and `::read_all_visible_oracle_gemma4_e2b` (FT6.13): the test this card copies with the 26B checkpoint;
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity_gemma4_26b (~line 730 at 4b4be6cf)` and `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/llama_ids.json` (vendored once, never re-queried).
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test `read_all_visible_oracle_gemma4_26b`: the same config and `prepare` rule as the E2B test, `GEMMA4_26B` as the checkpoint.
- test: `read_all_visible_oracle_gemma4_26b` asserts, on the returned `run`: `run.compared_lens == vec![9, 12, 12]` (the 3 prompts of `gemma4_26b/llama_ids.json` hold 9, 12 and 12 generated ids, read from that file; 33 ids compared in all, through the comparison inside `llama_parity_with`), `run.generated_lens == vec![32, 32, 32]` and `run.stats == (103_230, 93)` (30 is `gemma4.block_count`, `gemma4_26b/gguf_kv.txt` line 21). Derived from code, not yet observed: the decode loop does not stop at llama's end-of-generation id 106 (it stops only on id 1) and the budget is 32 (`LLAMA_GENERATED_TOKENS`), so each prompt generates 32 ids unless id 1 appears; `compared_lens` is `[9, 12, 12]` only because of the `.min()` against the vendored llama ids. The prompt id counts 18, 23 and 22 are read from `llama_ids.json`. Rows per layer are `31 * (p + 1) + 465` per prompt: `1054 + 1209 + 1178 = 3441`, times 30 layers is `103_230` rows, and `3 * 31 = 93` steps. If `generated_lens` prints other than `[32, 32, 32]` or the stats differ, stop and report the printed values; do not edit the literals to match. Ring layers: 25 of the 30 layers are sliding (`gemma4_26b/swa_layers.txt`, 25 lines with `is_swa = 1`) with `gemma4.attention.sliding_window = 1024` (`gguf_kv.txt` line 35), so each hands the rule `min(cached rows, 1024)` rows (FT6.11); the deepest step has `23 + 30 + 1 = 54` rows, below 1024, so every layer counts `p + k + 1` rows as the sum above assumes. A checkpoint whose dense parity already fails is a finding to report, not to skip.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_19 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_all_visible_oracle_gemma4_26b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check an all-visible read on gemma4 26b ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; weaken the comparator; add a qwen or openchat case.
- gpu: one run (`-j 1`, one checkpoint load), waiting for a quiet box.

### 6.20 an explicit dense read keeps llama's ids on gemma4 26B (oracle)

- id: FT6.20
- needs: FT6.14, FT6.19
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::read_dense_oracle_gemma4_e2b` (FT6.14): the test this card copies with the 26B checkpoint;
  - `proxima-model-interop/tests/arch_data_baseline.rs::arch_data_digest_gemma4_26b (~line 351 at 4b4be6cf)`: the 26B op graph digest (13314 ops ending at NodeId(13313)) that the slice exit command runs; it is not part of this card's run.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test `read_dense_oracle_gemma4_26b`: `attention.read = ReadSpec::Dense` set explicitly, no rule, `GEMMA4_26B` as the checkpoint.
- test: `read_dense_oracle_gemma4_26b` asserts, on the returned `run`: `run.compared_lens == vec![9, 12, 12]` (33 ids compared against the vendored llama ids by the comparison inside `llama_parity_with`; this count separates a run from no run) and `run.stats == (0, 0)` (the dense step counts nothing; `(0, 0)` alone is also what no run produces).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_20 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_dense_oracle_gemma4_26b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check explicit dense read on gemma4 26b ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; add a read operand test here.
- gpu: one run (`-j 1`, one checkpoint load), waiting for a quiet box.

### 6.23 the decode step calls the read rule once per layer on gemma4 E2B

- id: FT6.23
- needs: FT6.11
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 667 at a7c08c4c)` and `::llama_cases (~line 616)`: how a test loads the real E2B checkpoint and generates from vendored prompt ids;
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel::set_read_rule` (FT6.25) and `proxima-model-interop/src/generate/decode.rs::fill_read_skip` (FT6.11): the setter and the fill this test drives through the real program switch.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test `read_rule_runs_once_per_layer_each_decode_step_gemma4_e2b`.
- test:
  - add in `proxima-model-interop/tests/arch_data_baseline.rs` (exercises `set_read_rule`, the resolved read operand program, the program switch and the input bind on the real gemma4 E2B checkpoint, 35 attention layers) `read_rule_runs_once_per_layer_each_decode_step_gemma4_e2b`:
    open `GEMMA4_E2B`, `LoadedModel::load`, `let mut model`; atomics `calls` (total rule calls), `lowest_rows` (initially `usize::MAX`, `fetch_min` of `cached_rows`), `highest_rows` (`fetch_max` of `cached_rows`) and `deepest_layer` (`fetch_max` of `layer`) shared with `model.set_read_rule(move |layer, cached_rows, _flags| { .. })` through `Arc`s (the rule marks nothing);
    `prompt_ids` is `llama_cases(&GEMMA4_E2B)[0].prompt_ids`; `operand = ServingConfig { prompt_cache: PromptCacheConfig::off(), speculative: SpeculativeConfig::none(), attention: AttentionConfig { read: ReadSpec::Operand }, ..ServingConfig::default() }`; `model.generate_from_ids(&prompt_ids, 4, &operand, &mut |_event| ControlFlow::Continue(()))`.
    Assert exactly `calls == 105`, `lowest_rows == 6`, `highest_rows == 8` and `deepest_layer == 34`.
    Derived, not yet observed: the vendored prompt `llama_cases(&GEMMA4_E2B)[0]` is `"The capital of France is"` with 6 prompt ids (`llama_ids.json`: `[2, 818, 5279, 529, 7001, 563]`); `decode_until_stop_or_budget` (`generate/residency_caches.rs`) runs one forward per step for `max_tokens = 4` steps (the eos id is 1 per `gguf_kv.txt`, and the oracle's three ids end in 106, which is not eos); step 0 is the 6-row prefill (no rule call), steps 1 to 3 are single-row decode steps with 6, 7 and 8 cached rows, each calling the rule once for each of the 35 layers: `3 * 35 = 105`. If the printed values differ, stop and report them; do not edit the integers to match.
    Ring layers: 28 of the 35 layers are sliding with `gemma4.attention.sliding_window = 512` (`gguf_kv.txt` line 29), and each hands the rule `min(cached rows, 512)` rows (FT6.11), which equals the cached rows at 6, 7 and 8 rows, so one count `6..=8` holds for all 35 layers on this prompt only because 8 < 512.
    Control: a second `generate_from_ids` with `ServingConfig { attention: AttentionConfig { read: ReadSpec::Dense }, ..operand }` leaves `calls` unchanged (nothing is declared, bound or called under the dense read).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_23 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/read_rule_runs_once_per_layer_each_decode_step_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check the read rule runs once per layer each step`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; weaken the asserted integers; add a second checkpoint.
- gpu: one run (`-j 1`): one model load of the E2B checkpoint on the CPU backend, waiting for a quiet box (CARDS.md machine safety: check `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"`).

### 6.24 the key row counter sums the unskipped rows on gemma4 E2B

- id: FT6.24
- needs: FT6.12, FT6.23
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::read_rule_runs_once_per_layer_each_decode_step_gemma4_e2b` (FT6.23): the model test this card copies the setup of;
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel::take_kv_read_stats` (FT6.26, written by FT6.12).
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test `take_kv_read_stats_sums_the_unskipped_rows_and_resets_gemma4_e2b`.
- test:
  - add in `proxima-model-interop/tests/arch_data_baseline.rs` `take_kv_read_stats_sums_the_unskipped_rows_and_resets_gemma4_e2b` (the setup of the FT6.23 model test, with a rule that marks the first `cached_rows / 2` rows and itself adds `cached_rows + 1 - cached_rows / 2` to an `AtomicU64` `expected_rows` and 1 to `calls`): after `generate_from_ids(&prompt_ids, 4, &operand, ..)`, `calls == 105`, `model.take_kv_read_stats() == (expected_rows, 3)` and `expected_rows == 490` (derived, not yet observed, from the FT6.23 count: 3 single-row steps with 6, 7 and 8 cached rows, per layer `(7 - 3) + (8 - 3) + (9 - 4) = 14`, times 35 layers), and a second `model.take_kv_read_stats()` is `(0, 0)` (the swap reset both counters).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_6_24 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/take_kv_read_stats_sums_the_unskipped_rows_and_resets_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): check the key row counter on gemma4 e2b`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`; weaken the asserted integers; add a second checkpoint.
- gpu: one run (`-j 1`): one model load of the E2B checkpoint on the CPU backend, waiting for a quiet box.

## spec drift

Each item: where the spec or the previous cut says one thing and these cards do another, with the reason.

1. The previous cut put block read, sampled read and their inputs into the library (`ReadGraph::Block`, `ReadGraph::Sampled`, `block_read_*`, `sampled_read_*`, `fill_block_read_inputs`). These cards put none of it there. Reason: owner direction (hooks, not techniques); the techniques are proofs in tests.
2. The hook is a function argument to the builder, not a `ReadSpec` field read by the tensor crate. Reason: the selection needs the layer's own query, which exists only inside the graph; a host filled leaf cannot express it. `ReadSpec::Operand` selects the leaf-only instance.
3. `ReadSpec` has two variants here (`Dense`, `Operand`), not `Dense | Block | Sampled`. Reason: `ServingConfig` is `Copy` and holds no list; the technique parameters are not config.
4. The decode plan key keeps its 4 elements; the read spec enters `PlanIdentity`. Reason: the plan cache is per runtime, the runtime holds one read spec, and `PlanIdentity` is the existing home of runtime-wide properties. Plans for the dense and the read program are told apart in one runtime only by output node ids, which differ because the read program declares leaves before the logits root; this is stated in the FT6.10 read-first, and a test that two programs collide would need a program pair that does not exist.
5. The captured layer is 4, not 2 (the first full-attention layer of gemma4 E2B). The captured queries attend causally (query `j` reads rows `0..=2096 + j`), where the previous cut softmaxed every query over all 4096 rows. The fixture is not committed (about 50 MB) and lives at a durable path, as the llama-parity checkpoints do. Because a clean checkout does not hold it, the one test that reads it sits behind the default-off `real-capture-tests` feature of `proxima-tensor` (FT6.8); without the feature the committed tree compiles and runs no test that needs the file, and with it a missing file is a loud panic, never a skip. The capture example is split into three cards (FT6.7, FT6.21, FT6.22) so each stays inside the size rule and no item exists only for a later card; the ids FT6.21 and FT6.22 are appended after FT6.20 so no earlier id moves.
6. The error bound counts (row, head) pairs (16000), at `1 - delta` of them (15200), where the previous cut counted 1900 of 2000 rows. Reason: the estimator runs per head; `delta` is the owner's stated failure probability.
7. The cached-attention fusion carries the read operand as the last operand, found by operand count, not as a new `BoundOpKind::CachedAttention` field. Reason: the field would change every literal of the variant (omega, examples, tests) in one commit; the variant's doc already uses operand count as the discriminator for the cached length and the pass planes.
8. The CPU interpreter honors the operand (FT6.16, in the same commit as the fusion that produces it, over the skipping streaming routine of FT6.15), which the previous cut never built: without it a fused op under the feature would silently read dense on the CPU path, and `evaluate_named` (which binds with fusion on) would return `NotLowerable` for an operand count of 10 or 13, so every commit that turned the feature on without the interpreter was red under the feature.
9. The previous cut's oracle count was 4 (gemma4 E2B, openchat, qwen2, qwen3); it is 2 here (gemma4 E2B dense, gemma4 26B MoE). Granite has no read card (see "decided in this file").
10. Anchors at 4b4be6cf: `top_fraction_mask`, `ServingConfig.attention` and the slice 0 vendored fixtures for the new checkpoints are not on main. Cards 6.5, 6.6, 6.10, 6.11, 6.13, 6.14, 6.19, 6.20, 6.23 and 6.24 stop when their prerequisite is absent.
11. The old FT6.1 and FT6.2 (`block_read_keep_count`, `block_read_row_count` in `proxima-core/src/read_decision.rs`) are dropped, not kept. Reason: they are block read selection arithmetic, which this file's scope keeps out of every library crate, and no non-test code called them. `tasks-recut/15-niah-read-arms.md` names both functions and the ids FT6.1 and FT6.2 (its prerequisites and its cards that compute the expected row counts); that file needs the same arithmetic as test-local code, and this file does not edit it.
12. The read operand is read through `BoundOpKind::read_rows` (on the kind, not on `BoundOp`), and omega's one cached-attention form classifier subtracts it from the operand count (FT6.17). Reason: the classifier sees only the kind, and without the subtraction an op with the operand would be classified as the eight operand static form. An op with a read operand is never given the decode-split form, and is given the row-tiled form only from FT6.18 on, because those kernels read dense until then: the guard in the classifier keeps every intermediate commit correct. The tensor-side commits are green under `read-mask-fusion` from FT6.16 on: FT6.15 adds no producer of the operand, and FT6.16 adds the producer together with the interpreter that executes it.
13. FT6.21 and FT6.22 tap the layer with the FT6.3 hook and `LoadedModel::forward_node_values_on_backend` instead of waiting on a layer tap card, because no card in `tasks-recut` taps rotated key, value or query planes (the cache blending file's tap is a residual).
14. The oracle cards FT6.13 and FT6.14 were one card each for two checkpoints; FT6.19 and FT6.20 carry the 26B checkpoint so each card loads one checkpoint. The digest checks of the two gemma4 checkpoints moved out of FT6.14 into the slice exit.
15. FT6.11 and FT6.12 were each one card that added two or more items another card consumes (FT6.11: `set_read_rule` and `read_skip_extent`; FT6.12: `take_kv_read_stats` and the counter pair) and staged 5 files, over the size rule with no admitted exception. Each is split in two, and the new ids are appended after FT6.24 so no earlier id moves, while the new cards sit before the card they precede in this file:
    - FT6.25 (before FT6.11): the `LoadedModel` fields `read_operand_program` and `read_rule`, the pub `set_read_rule`, the load-time resolve, the four literal edits, and `read_operand_parts` in `decode.rs`, which the decode step calls under `ReadSpec::Operand` and which refuses a missing rule or program with a typed `UnsupportedServingConfig`. The refusal is what reads the two fields, so neither is dead code in that commit; the program is not switched and nothing is bound there. Under `ReadSpec::Operand` and until FT6.11 lands, a model that has both runs the dense program and ignores the rule; settings refuse `operand` through `ServingRefusal::ReadHookNotBound` (FT2.34) over exactly that span.
    - FT6.11: `decode.rs` only (`read_skip_extent`, `read_skip_layer_slots`, `fill_read_skip`, the program switch and the bind) and its tests. It adds one item another card consumes (`read_skip_extent`, by FT6.12). `fill_read_skip` takes the rule by reference because `read_operand_parts` already refused a missing rule, so the no-rule test of the earlier cut is `operand_read_without_a_rule_names_what_is_missing` in FT6.25. FT6.11 is the card that makes `set_read_rule` act, so the FT2.34 amendment in `02-serving-settings.md` (the `ReadHookNotBound` deletion) stays with FT6.11.
    - FT6.26 (before FT6.12, needs FT6.25 because both edit the same four literals): the `AtomicU64` pair `kv_rows_read` and `kv_decode_steps`, the pub `take_kv_read_stats`, the four literal edits. Its test sets the counters directly and proves the swap reset; nothing writes them until FT6.12.
    - FT6.12: `decode.rs` only (`rows_read_by_layer` and the per-step counter writes) and its tests; it adds no item another card consumes.
    The model-loading tests moved to FT6.23 and FT6.24 as before; `chunked_prefill_tests.rs` keeps the synthetic 3 layer gemma4 tests (2 ring layers, 1 full layer, CPU backend, no checkpoint) so `set_read_rule`, the program switch and `take_kv_read_stats` are each exercised in the card that adds them. `tasks-recut/15-niah-read-arms.md` names FT6.12 as the adder of `take_kv_read_stats` and FT6.11 as the adder of `set_read_rule`; that file needs the card that first makes each one act (FT6.11 and FT6.12, both of which transitively need FT6.25 and FT6.26), and this file does not edit it.
16. The read rule's row count and the `read_skip.{layer}` leaf length are per layer (FT6.11), not one `cached_rows` and one `bound_rows` for every layer. Reason: on main a ring layer's leaves are bound by the sliding symbol (`lfm2_single_range_cached.rs` ~lines 1519-1523) and the ring hands `min(cached, window)` rows under `min(kv_bound, window)` (`kv_ring.rs::KvRing::{live_rows, bound_extent}`), while a full layer keeps the full count and bound; gemma4 E2B loads with the ring layout (28 sliding layers of 35), so one count would have mis-sized every sliding layer once the cache outgrew the window. The rule's second argument is therefore the rows that layer hands the program; `tasks-recut/15-niah-read-arms.md` and `16-technique-configs.md` call `set_read_rule` with the same signature and are not edited by this file; a rule that needs the absolute cached length of a ring layer is not expressible through it.

## slice exit

- Commands, in order, each at the stated count: `cargo nextest run -p proxima-tensor -E 'test(/read_hook_|read_skip_operand_/)'` prints `6 passed` (4 + 2); `cargo nextest run -p proxima-tensor -E 'test(/block_read_selection_worked_/)'` prints `4 passed`; `cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/operand_read_/)'` prints `3 passed` (FT6.25); `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/fill_read_skip_|read_skip_extent_|read_rule_runs_per_layer_|serving_settings_refuses_read_needs_|serving_settings_refusal_field_paths/)'` prints `8 passed` (FT6.11); `cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/take_kv_read_stats_returns_the_counters_and_resets_both/)'` prints `1 passed` (FT6.26); `cargo nextest run -p proxima-model-interop --features std -E 'test(/rows_read_by_layer_|take_kv_read_stats_sums_the_unskipped_rows_and_resets_on_two_range_gemma4/)'` prints `3 passed`; `cargo nextest run -p proxima-tensor -E 'test(/block_read_through_the_hook_/)'` prints `2 passed`; `cargo nextest run -p proxima-tensor -E 'test(/cached_row_skip_/)'` prints `3 passed`; `cargo nextest run -p proxima-tensor --features read-mask-fusion -E 'test(/read_skip_keeps_cached_attention_fused|fused_read_skip_equals_the_deleted_rows_dense_program|read_skip_operand_/)'` prints `4 passed`; `cargo nextest run -p proxima-tensor -E 'test(/sampled_read_budget_worked_base_sample/)'` prints `1 passed`; on the machine that holds the capture file, `cargo nextest run -p proxima-tensor --features real-capture-tests -E 'test(/sampled_read_error_bound_real_layer_rows/)'` prints `1 passed` (the same name without the feature runs 0 tests, which is a red count, not a pass); `cargo nextest run -p proxima-model-interop --features std -j 1 -E 'test(/read_rule_runs_once_per_layer_each_decode_step_gemma4_e2b|take_kv_read_stats_sums_the_unskipped_rows_and_resets_gemma4_e2b/)'` prints `2 passed`; `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_all_visible_oracle_/)'` prints `2 passed` (1 + 1, one card per checkpoint); `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/read_dense_oracle_/)'` prints `2 passed` (1 + 1); `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/arch_data_digest_gemma4/)'` prints `2 passed`.
- Feature-off builds: `cargo check -p proxima-tensor` and `cargo check -p omega --features metal` are clean (this file touches no code in `proxima-core`).
- Size: no card in this file claims an exemption from the size rule. FT6.25 is the widest at 3 non-test source files (`load_model.rs`, `pregather.rs`, `decode.rs`: the model fields, the load-time resolve, the refusal that reads them) plus two test files (`chunked_prefill_tests.rs`, and `tests_all.rs`, which gets only the one-line field additions to its `model_with` literal, an exhaustive struct literal, so the edit cannot move to another card); FT6.26 stages 2 non-test source files (`load_model.rs`, `pregather.rs`) and the same two test files; FT6.12 stages 1 non-test source file (`decode.rs`) and `chunked_prefill_tests.rs`; FT6.11 stages 3 non-test source files (`decode.rs`, `refusal.rs`, `refusals.rs`: the program switch and the settings refusal it makes false) and `chunked_prefill_tests.rs`. Each adds at most one item another card consumes. Their real-checkpoint tests are FT6.23 and FT6.24, one test file each. FT6.11's first production caller of the rule fill lands in the same commit because `generate` is a private module. Every card runs at most one model load or GPU run: FT6.23 and FT6.24 one CPU load of the E2B checkpoint each, FT6.22 one Metal forward pass (FT6.7 and FT6.21 only map and parse the checkpoint, no weight load), FT6.13, FT6.14, FT6.19 and FT6.20 one checkpoint each (the oracle cards were one card for two checkpoints and are now one card per checkpoint), FT6.18 one Metal run. An executor that overruns the budget stops and reports; it does not split a card.
- Decide-later items with owners: a per-head read (the mask is shared by every head of a layer); a read function that needs the layer's value rows; a read hook on a single-range cached engine (granite MoE and every non-gemma4 family); an in-graph selection kernel for the rank-count selection (speed only, owned by the specialize stage). The sampled arm's top-k uses block summaries, not an approximate index; that choice is plausible, not measured, and the real-row error bound in FT6.8 is its test.
- Unverified premises the cards stop on rather than guess: a rank-1 source for `gather_computed` (FT6.5), the signature of `top_fraction_mask` (FT6.5), the shape of the `ReadSpec` and `AttentionConfig` fields (FT6.10 to FT6.14, FT6.19, FT6.20), that the example's own program is node for node the loaded model's program (FT6.22 checks the kind of every tapped node and stops on a mismatch) and that the gemma4 prompt cut at a token boundary re-tokenizes to the same 4096 ids (FT6.7 checks the count and stops on a mismatch), that layer 4 of the E2B schedule is unwindowed, projects its own key and is unscaled (FT6.21 stops on a mismatch), that a two-layer all-full two-range program with a `cached_len` input binds to two `cached_attention` ops under `NumericPolicy::bit_exact()` at one new row with only `read-mask-fusion` on (the FT6.16 counts), and that the Metal host binds the trailing read operand from the op's read sources the way it binds the other operands (FT6.17, FT6.18). `crate::cpu::evaluate_named` binds with the fusion features on: `proxima-tensor/src/cpu/arena.rs::plan_trace_named` documents it as running the same admission pipeline and calls `bind::bind` with `NumericPolicy::bit_exact()`.
