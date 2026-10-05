# slice 9 (re-cut): hooks that let selective recompute be written as a pipe (cards FT9.3 - FT9.11; FT9.1 and FT9.2 withdrawn)

anchors read at main a7c08c4c (`git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`); line numbers are hints, re-locate by symbol.

Spec: `proxima-tensor/specs/pipeline-as-data/SPEC.md` (hooks H9 assemble and H12 read; stages N2 tap and edit and N9 position; the cross-cutting cache-key rule), `sketches/09-cacheblend.md` (gap list GAP-1 to GAP-6), `research.md` (CC-1). CARDS.md template and rules are binding.

Owner direction this re-cut obeys: the hooks are built so the technique can be written against them; the technique is never library code. A selective-recompute technique therefore appears here only as tests: a test of at most about 40 lines driving the hooks (card FT9.9) and the worked selection example over a library primitive (card FT9.8). The per-check row count and the mask-in-force rule are the technique's own schedule, so no library function carries them.

Test models (owner, 2026-10-04): gemma4 E2B (ollama `gemma4:e2b-it-qat`, blob `sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`, constant `GEMMA4_E2B` in `proxima-model-interop/tests/arch_data_baseline.rs`). No qwen model appears in any card, fixture, oracle or validation. Oracles are recorded once; tests replay the recorded ids and never query Ollama or llama-server. The attention cache is untouched by the MoE feed-forward, so no granite card exists in this slice.

## id map (old card to new card)

| old | new | what happened |
|---|---|---|
| FT9.1 `blend_keep_count` | (dropped) | see "dropped" |
| FT9.2 `blend_mask_index` | (dropped) | see "dropped" |
| FT9.3 `blend_deviation` | FT9.3 | recut: the hook is the per-layer residual tap on the two-range builder; deviation moved into the proof test |
| FT9.4 `blend_select_kv` | FT9.4 | recut: the hook is a generic key/value row edit input on the two-range builder; the selection policy moved into the proof test |
| FT9.5 `blend_used_mask` | FT9.9 | recut: three ops, no hook; folded into the proof test |
| FT9.6 worked 3-chunk selection | FT9.8 | kept, re-anchored; the keep counts are literals of the worked example, no library function computes them |
| FT9.7 blend leaves in the layer loop | FT9.3, FT9.4 | recut: only the generic tap and edit leaf survive; deviation, top fraction and used mask moved into the proof test |
| FT9.8 override through attention | FT9.4 | recut: generic row edit input |
| FT9.9 `build_forward_blended` | (dropped) | see "dropped" |
| FT9.10 `LoadedModel::blend_program` | (dropped) | see "dropped" |
| FT9.11 plan key blend flag | FT9.7 | recut: the producing configuration goes in the prompt-cache key (cross-cutting cache-key rule); the decode plan key needs no change (see FT9.7 "why no plan key card") |
| FT9.12 blend prefill stretch | FT9.5, FT9.6, FT9.9 | recut: sparse-position attention read (FT9.5), descriptor configuration (FT9.6), technique as a test (FT9.9); the host halves are listed under "not built here" |
| FT9.13 blend step of assembly | (see "not built here") | recut target needs an attachment point that does not exist; named below |
| FT9.14 oracle ratio one | FT9.10 | kept, retargeted to gemma4 E2B only |
| FT9.15 oracle ratio zero | FT9.11 | kept, retargeted to gemma4 E2B only |

## dropped

- FT9.1 `blend_keep_count` and FT9.2 `blend_mask_index` (`proxima-core/src/serving_state/blend.rs`): the per-check keep ratio and the mask in force at a layer are the technique's own selection schedule, named after it. Their only consumer was the test of card FT9.8, which is no first production caller (the standard stated under "not built here"). The same arithmetic is two lines in a technique's own test, which FT9.8 and FT9.9 already are, so nothing in the library replaces them and `proxima-core` is untouched by this slice.
- FT9.9 `build_forward_blended`: a technique-named parallel entry beside `build_forward` (`proxima-tensor/src/spec/descriptor.rs::build_forward`, ~line 387 at a7c08c4c). Call site with and without it differs only by two slices, which the descriptor fields of FT9.6 carry as configuration.
- FT9.10 `LoadedModel::blend_program`: per-technique lowering gated on a descriptor field `LoadedModel` does not hold (its own premise check fails on main), and its test loads a qwen2 checkpoint, which is banned. Superseded by FT9.3 to FT9.6.

## not built here (exact parts, and why)

Each part below needs a first production caller before a commit adding it is coherent (CARDS.md coherence rule: a `pub(super)` or private item used only under `#[cfg(test)]` trips `dead_code` in the non-test clippy build). The only possible caller is the technique's own pipe, which this slice keeps in tests. They are not skipped silently; each is listed so the owner can grant a coherence exception or choose an attachment point.

1. Sketch GAP-2 host halves: a RoPE table built from a position list (the loop `for offset in 0..new_count` in `proxima-model-interop/src/generate/residency_caches.rs::build_position_inputs`, ~line 1206, computes `start_position + offset`), a cache write at a position list (`proxima-model-interop/src/generate/kv_ring.rs::LayerCache::append_at`, ~line 117, takes one start position), and the decode loop binding per-request row-aligned inputs (`proxima-model-interop/src/generate/decode.rs::push_step_named_blocks`, ~line 1680). The graph half of GAP-2 is built (FT9.5).
2. Sketch GAP-1, a request-given list of source entries: `lift_chunks` (`proxima-model-interop/src/generate/chunk_shift.rs`, ~line 459) already takes the source `entry` as a parameter, so lifting from a second entry needs no library change; the missing piece is the attachment point that lets a pipe call it, which is item 3.
3. The attachment point itself: a request-scoped step that runs after the configured assemble steps. `run_decode_loop_through_cache` (`proxima-model-interop/src/generate/prompt_cache.rs`, ~line 1202) is reached from `generate_from_ids` and its siblings, so a step supplied by a caller needs either a new public parameter on those entry points or state on `LoadedModel`. That is a public API choice, not a card.
4. Sketch GAP-6 (seam trim between planning and lifting): a pure function over `Vec<ChunkRun>`; it needs item 3 to run between plan and lift.
5. An engine-level intermediate-ratio run on gemma4 E2B: needs items 1 and 3 plus a bind variant beside `bind_gemma4_with_last_row_only` (`proxima-model-interop/src/gemma4/bind.rs`, ~line 662). Intermediate ratios are proved on the synthetic gemma4-shaped fixture instead (FT9.9); the endpoints through the real engine are FT9.10 and FT9.11.

Sketch GAP-4 (chunk size against ring capacity) needs no card: the refusal already exists in code (`chunk_shift.rs::ring_rows_live`, ~line 160 refuses a run whose ring rows are gone), and the bound `chunk_tokens + 1 <= window + slack` is a property of the technique's own configuration, which is not library code.

## findings that changed the cut (read at a7c08c4c)

- The two-range builder already returns each cache-owning layer's rotated key halves and value as `cache_roots` (`proxima-tensor/src/spec/lfm2_single_range_cached.rs::lfm2_two_range_cached_forward_program_with_experts_and_head_repeats`, ~line 1300, which `lfm2_two_range_cached_forward_program_with_experts` (~line 1249) forwards to with a head repeat count of 1; `proxima-tensor/src/spec/mistral_forward_cached.rs::CachedLayerRoots`, ~line 298). What it does not return is each layer's residual: `proxima-tensor/src/spec/descriptor.rs::BuildForwardProgram` documents the residual vector as empty under the two-range strategy, and `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b.digest` line 9 records `bind.residual_roots=0`. So the missing readout is the residual tap (FT9.3), not the key/value readout.
- `causal_mask_cached_windowed` is a private function in `lfm2_single_range_cached.rs` (~line 328), not in `primitives.rs`. Its mask is query-independent: a cache row is excluded when its index is at or past `cached_len`.
- On the 3-row fixture below, a first-layer key/value row depends only on its own token, so the first layer shows zero deviation; the deviation worked example in FT9.9 is therefore taken at the second layer and the edit applies at the third.
- `ServingConfig<'model>` (`proxima-model-interop/src/serving.rs`, ~line 720) is `Copy` and `CacheKey::of` (`proxima-model-interop/src/generate/prompt_cache_key.rs`, ~line 82) destructures it with no `..`. A list-valued stage reaches the key as a `u64` digest (FT9.7), because the key is `Copy`.

Common to every card (not repeated):
- target dir `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_<n>`, removed when done; logs under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/9.<n>/`;
- `proxima-tensor` cards: also green = `cargo clippy -p proxima-tensor --all-targets` clean; `proxima-model-interop` cards: `cargo clippy -p proxima-model-interop --features std --all-targets` clean;
- gpu: none unless the card says otherwise; no comments in code unless the why is a gotcha;
- everything an executor writes into the tree (code, doc comments, test names, error text, commit subjects) is plain English about behaviour: no slice, stage, AC, R, W, FT, DAD or card ids and no spec pointers. This card file's own prose keeps ids;
- every card is one commit that stages only its `stage:` list, with exactly its `commit:` subject, no attribution trailer;
- never run `cargo fmt` or rustfmt on a src file; never `git checkout`, `stash`, `reset` or `worktree`.

## prerequisites (each verified against main a7c08c4c; each card names the ones it needs)

- `FT3.2` = `proxima-tensor/src/spec/primitives.rs::top_fraction_mask(program: &mut Vec<Op>, scores: NodeId, keep_count: NodeId, keep_rows: Option<NodeId>) -> Result<NodeId, TensorError>` (`keep_count` a rank-0 f32 `Input`; `scores` and `keep_rows` rank-1 f32 of extent `Symbolic(0)`; rank(i) = #{j: s[j] > s[i]} + #{j < i: s[j] == s[i]}, selected iff rank < keep_count, then OR keep_rows). Absent on main (`git grep -n "fn top_fraction_mask" main` prints nothing). Premise check: `git grep -n "pub fn top_fraction_mask" -- proxima-tensor/src` prints 1 line; else stop.
- `FT0.16` = `proxima-tensor/specs/fsm-techniques/worked-examples.md` section `## per-layer recompute selection`. Premise check: `grep -c -F 'RESULT per-layer recompute selection: check=[1,3,5,6,8,11] layer2=[3,6,11] layer3=[6,11] counts=[6,3,2] per_chunk_check=[2,2,2]' proxima-tensor/specs/fsm-techniques/worked-examples.md` prints 1; else stop.
- `FT2` = the serving configuration holds a configured assemble list. Absent on main (`git grep -n -i assemble main -- proxima-model-interop/src/serving.rs` prints nothing). The cards read it as `ServingConfig`'s `prefill.assemble` field holding a slice of `AssembleStep` values (the step enum has a `Prefix` variant and a `Shift` variant), with a settings grammar that builds the list from `[[prefill.assemble]]` tables. Premise check: `git grep -n "enum AssembleStep" -- proxima-model-interop/src` prints 1 line and `git grep -n "Shift" -- proxima-model-interop/src/serving_grammar.rs` prints at least 1 line; else stop and report the real names.
- `FT8` = the prefill assembly fold exists. Absent on main (`git grep -n "fn assemble_request" main` prints nothing). Premise check: `git grep -n "fn assemble_request" -- proxima-model-interop/src` prints 1 line; else stop.
- `FT0.4` = `proxima-tensor/specs/fsm-techniques/env.sh`, which exports `$LOGS`, `$LLAMA` and `$GEMMA4_E2B`. Absent on main (`git ls-tree main proxima-tensor/specs/fsm-techniques/` prints nothing). Premise check: `test -f proxima-tensor/specs/fsm-techniques/env.sh && echo 1` prints `1`; else stop.
- `FT0.50` and `FT0.51` = `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json`: FT0.50 writes it with `request1` and `request2` records, FT0.51 adds the `request2_plain` record (absent on main: the directory holds `gguf_kv.txt`, `llama_ids.json`, `swa_layers.txt` only). Only the gemma4_e2b file is read here.

## design decisions the cards encode (each with its one-line why)

1. No row-count or mask-in-force function exists in the library. Why: both are the technique's own schedule, and a library item whose only caller is the technique's test has no coherent first production caller.
2. The tap and the edit are builder inputs on the two-range builder, off by default; the default program is node-for-node the program built today. Why: the default must reproduce today's op graph byte for byte, which the seven recorded op-graph digests (`proxima-model-interop/tests/fixtures/llama-parity/*.digest`) check.
3. An edit leaf set is `kv_edit.{layer}.take` plus `kv_edit.{layer}.k_even`, `.k_odd`, `.v`. Where `take` is 1.0 the supplied rows replace this step's computed rows for that layer, elsewhere the computed rows stay; what attention reads is what the cache stores. Why: one generic input expresses "load these rows", "overwrite these rows" and "keep these rows" with no technique name.
4. A step's attention read can take explicit query positions and a per-cache-row dead flag as inputs instead of `iota + cached_len`; off by default. Why: a step that recomputes scattered positions inside a loaded cache cannot be described by a contiguous start and count.
5. The prompt-cache key carries the digest of the assemble list that produced an entry's rows, `0` for rows no assemble step touched. A request admits an entry when every other key field matches and the entry's digest is `0` or equals the request's own. Why: the cross-cutting rule (a cached row's key binds its producing configuration), expressed in the existing key instead of a new provenance type; a fresh entry stays reusable by everyone, a shifted or edited entry only by a request running the same list.

## cards

### 9.3 the two-range builder exposes each layer's residual

- id: FT9.3
- needs: none
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs::lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` (~line 1300 at a7c08c4c, the function that holds the layer loop and takes the `head_repeats` parameter), `lfm2_two_range_cached_forward_program_with_experts` (~line 1249, a wrapper that forwards to it with `1` and has no loop of its own) and `TwoRangeForwardProgram` (~line 1241): the line `x = append_lfm2_layer_ffn(` (~line 1682) and `cache_roots.push(layer_roots)` (~line 1709), the final `Ok((program, logits, cache_roots, MoeSites(moe_sites), duplicate_head_roots))` (~line 1792);
  - `proxima-tensor/src/spec/descriptor.rs::build_forward` (~line 387), its `CacheStrategy::TwoRange` arm (~line 394, which calls `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats`), and `BuildForwardProgram`'s doc (~lines 332-341), which says the residual vector is empty under the two-range strategy;
  - `proxima-tensor/src/spec/tests.rs::two_range_cached_gemma4_matches_prefill_oracle_with_decode_loop_realistic_zero_padding` (~line 13465): the fixture and the per-layer references `layer0_next`, `layer1_next` this card's test reuses;
  - `proxima-model-interop/src/gemma4/bind.rs::bind_gemma4_with_last_row_only` (~line 662): it destructures `build_forward`'s fifth element as `_layer_residuals` (~line 733) and must stay unchanged, so the recorded `bind.residual_roots=0` digest line does not move.
- change:
  1. `lfm2_single_range_cached.rs`: rename `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` to `lfm2_two_range_cached_forward_program_with_layer_taps`, same parameters in the same order (including the last, `head_repeats: u32`), return type `Result<(TwoRangeForwardProgram, Vec<NodeId>), TensorError>`. Before the layer loop add `let mut layer_residuals: Vec<NodeId> = Vec::with_capacity(block_count as usize);`; directly after `x = append_lfm2_layer_ffn(..)?;` add `layer_residuals.push(x);`; the final expression becomes `Ok(((program, logits, cache_roots, MoeSites(moe_sites), duplicate_head_roots), layer_residuals))`. Move the existing doc block to the new name and add one sentence: the second element is each layer's output node in layer order.
  2. Same file: add back `lfm2_two_range_cached_forward_program_with_experts_and_head_repeats` with its exact old signature and `#[allow(clippy::too_many_arguments)]`, whose body forwards every argument to the new function and ends `.map(|(built, _layer_residuals)| built)`; doc: one line saying it is the new function without the residual nodes. `lfm2_two_range_cached_forward_program_with_experts` already forwards to it and is not edited. None of the old function's callers is edited: its 11 call sites (`git grep -n "lfm2_two_range_cached_forward_program_with_experts(" -- .` prints 12 lines at a7c08c4c: the definition, `omega/tests/gemma4_rows_support/mod.rs` 1, `proxima-model-interop/tests/arch_data_baseline.rs` 1, `proxima-tensor/src/spec/tests.rs` 9) keep compiling, and the one caller of `_and_head_repeats` (`descriptor.rs`) moves in change 3.
  3. `descriptor.rs`: the `CacheStrategy::TwoRange` arm of `build_forward` calls `lfm2_two_range_cached_forward_program_with_layer_taps` (same arguments, `descriptor.head_repeats` last), destructures `((program, logits, cache_roots, moe_sites, duplicate_head_roots), layer_residuals)` and returns `layer_residuals` as the fifth element instead of `Vec::new()`. Edit `BuildForwardProgram`'s doc and `build_forward`'s doc: the residual vector is empty only under the cacheless strategy.
- test: add `two_range_layer_taps_match_the_per_layer_reference` in `proxima-tensor/src/spec/tests.rs`: copy the whole body of the named oracle test above (same fixture, schedule, weights and zero-padded cache leaves), then build `tapped` with `lfm2_two_range_cached_forward_program_with_layer_taps` (the same arguments plus `head_repeats` `1` last) and `base` with `lfm2_two_range_cached_forward_program_with_experts`. Assert, in this order:
  1. the two programs are equal (`assert_eq!` on the `Vec<Op>`), so the default reproduces the old program;
  2. `layer_residuals.len() == 2`;
  3. evaluating `[layer_residuals[0], layer_residuals[1]]` through `crate::cpu::evaluate_named(&program, &[SEQ as u64, SEQ as u64], &named, ..)`, each value vector has length `SEQ * EMBEDDING` (24);
  4. `max_abs_diff(layer 0 values, flatten(&layer0_next)) < TOLERANCE` and `max_abs_diff(layer 1 values, flatten(&layer1_next)) < TOLERANCE`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_3 cargo nextest run -p proxima-tensor -E 'test(/two_range_layer_taps_/)'`
- expect: `1 passed`
- also green: `cargo nextest run -p proxima-tensor -E 'test(/two_range/) | test(/build_forward/)'` prints the same pass count as before the edit plus 1 (write the before count in the commit body); clippy: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/lfm2_single_range_cached.rs`, `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): expose per-layer residuals on the two-range builder`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: edit a caller of the old function; edit `gemma4/bind.rs`; add an `Op` or `ScalarOp` variant
- gpu: none

### 9.4 a two-range step can overwrite chosen key and value rows per layer

- id: FT9.4
- needs: FT9.3
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `lfm2_single_range_cached.rs::append_lfm2_two_range_cached_attention` (~line 431): after `let v_new = if value_norm { .. }` (~line 537) and before the `fused_rope_pair(program, q, 'h', ..)` call, the nodes `rotated_k_new_even`, `rotated_k_new_odd`, `v_new` are final; the local block scores them with `(rotated_k_new_even, "wui->swugi")` (~line 636); its end `Ok((post_mixer, (rotated_k_new_even, rotated_k_new_odd, v_new)))` (~line 852); its one caller (~line 1600);
  - the `Select` use `(is_future_local, "sw->swug"), (neg_infinity_local, "->swug"), (score_new_scaled, "swug->swug")` (~line 700): operand order is (condition, value when true, value when false);
  - `lfm2_two_range_cached_forward_program_with_layer_taps` (card 9.3, the renamed `_and_head_repeats` function): the layer loop, `shared_source`, `stored_kv`, `cache_roots`;
  - `proxima-tensor/src/error.rs::TensorError::UnsupportedInBuilder` (~line 389): `{ builder: &'static str, feature: &'static str }`.
- change:
  1. `append_lfm2_two_range_cached_attention` gains a last parameter `kv_edit: Option<(NodeId, CachedLayerRoots)>` (doc: the take mask and the replacement rows). Right after `v_new` is final: `let (rotated_k_new_even, rotated_k_new_odd, v_new) = match kv_edit { Some((take, rows)) => (select_rows(program, take, rows.0, rotated_k_new_even, "wui")?, select_rows(program, take, rows.1, rotated_k_new_odd, "wui")?, select_rows(program, take, rows.2, v_new, "wud")?), None => (rotated_k_new_even, rotated_k_new_odd, v_new) };`. The private helper `select_rows(program: &mut Vec<Op>, take: NodeId, replacement: NodeId, computed: NodeId, axes: &str) -> Result<NodeId, TensorError>` binds `let take_notation = format!("w->{axes}"); let rows_notation = format!("{axes}->{axes}");` and returns `elementwise(program, DType::Float32, ScalarOp::Select, &[(take, take_notation.as_str()), (replacement, rows_notation.as_str()), (computed, rows_notation.as_str())])` (the final parameter of `elementwise` is `&[(NodeId, &str)]`, so the owned `String`s are bound first and borrowed). The returned roots are the selected nodes.
  2. `lfm2_two_range_cached_forward_program_with_layer_taps` gains a last parameter `kv_edit_layers: &[u32]`. For each layer in the set that owns its cache (`shared_source.is_none()`), before its attention call declare four leaves with `input_leaf` (f32): `kv_edit.{layer}.take` shape `[Extent::Symbolic(0)]`, `kv_edit.{layer}.k_even` and `kv_edit.{layer}.k_odd` shape `[Extent::Symbolic(0), Extent::Static(kv_heads), Extent::Static(pairs)]`, `kv_edit.{layer}.v` shape `[Extent::Symbolic(0), Extent::Static(kv_heads), Extent::Static(head_dim)]`, and pass `Some((take, (k_even, k_odd, v)))`; every other layer passes `None`. A layer in the set that is not below `block_count` or that is shared-KV returns `TensorError::UnsupportedInBuilder { builder: "lfm2_two_range_cached_forward_program_with_layer_taps", feature: "kv edit layer owns no kv cache" }`.
  3. The `_and_head_repeats` wrapper (card 9.3) passes `&[]` after its existing arguments (so the old-name `_with_experts` function keeps building the program it builds today). `descriptor.rs`'s two-range arm passes `&[]`. The one existing test call of `lfm2_two_range_cached_forward_program_with_layer_taps` (card 9.3's `two_range_layer_taps_match_the_per_layer_reference`, the `tapped` build) gets `&[]` appended as its last argument, so the test build compiles at this commit (`git grep -n "lfm2_two_range_cached_forward_program_with_layer_taps(" -- proxima-tensor/src/spec/tests.rs` lists every call to check).
- test: add `two_range_kv_edit_replaces_selected_rows` in `proxima-tensor/src/spec/tests.rs`: copy the fixture from card 9.3's test. Build `base` with `lfm2_two_range_cached_forward_program_with_experts` and `edited` with `kv_edit_layers = &[1]`. Assert, in this order:
  1. the program built with `&[]` equals `base` (`Vec<Op>`);
  2. `Op::Input` names of `edited` include `kv_edit.1.take`, `kv_edit.1.k_even`, `kv_edit.1.k_odd`, `kv_edit.1.v` and none starts with `kv_edit.0`;
  3. the count of `Op::Elementwise { body: ScalarOp::Select, .. }` in `edited` minus the count in `base` is exactly 3;
  4. with `take = [0.0, 0.0, 0.0]` and all-zero edit rows, the logits equal `base`'s (`max_abs_diff < TOLERANCE`);
  5. with `take = [1.0, 1.0, 1.0]` and the edit rows set to the values `base`'s layer-1 `cache_roots` evaluate to, the logits equal `base`'s (`< TOLERANCE`);
  6. with `take = [1.0, 1.0, 1.0]` and those rows plus 1.0 added to every value, `max_abs_diff > 100.0 * TOLERANCE` (the control that must fail);
  7. `kv_edit_layers = &[5]` returns `Err(TensorError::UnsupportedInBuilder { .. })` (`matches!`).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_4 cargo nextest run -p proxima-tensor -E 'test(/two_range_kv_edit_/)'`
- expect: `1 passed`
- also green: `cargo nextest run -p proxima-tensor -E 'test(/two_range/)'` prints the same count as card 9.3's after count plus 1; clippy: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/lfm2_single_range_cached.rs`, `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): let a step overwrite chosen key value rows per layer`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: change what a layer outside the edit set builds; add an `Op` or `ScalarOp` variant; edit the single-range builders
- gpu: none

### 9.5 the two-range cached read can take explicit query positions

- id: FT9.5
- needs: FT9.4
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `lfm2_single_range_cached.rs::causal_mask_cached_windowed` (~line 328): the exact node creation order (query iota, key iota, `query_absolute`, `cached_len_row`, `one`, `cached_len_ceiling_row`, `is_padding`, `neg_infinity`, then `distance`, `window_ceiling`, `too_old`, `is_invalid` when windowed), which the existing path must keep;
  - `lfm2_two_range_cached_forward_program_with_layer_taps`: `let mut cached_masks` and its dedup loop (~line 1376), `sliding_cached_len` (~line 1367);
  - sketch `proxima-tensor/specs/pipeline-as-data/sketches/09-cacheblend.md` GAP-2: positions as an input list.
- change:
  1. `lfm2_single_range_cached.rs`: add `fn causal_mask_cached_at_positions(program: &mut Vec<Op>, positions: NodeId, cache_dead: NodeId, key_extent: Extent, window: Option<u32>) -> Result<(NodeId, NodeId), TensorError>`. Body: `key_index = Iota{Float32, key_extent}`; `one = scalar_constant(1.0)`; `ceiling = Subtract(positions "s->s", one "->s")`; `is_future = Greater(key_index "t->st", ceiling "s->st")`; `is_padding = Maximum(is_future "st->st", cache_dead "t->st")`; `neg_infinity = scalar_constant(f32::NEG_INFINITY)`; without a window return `(is_padding, neg_infinity)`; with a window add `distance = Subtract(positions "s->st", key_index "t->st")`, `window_ceiling = scalar_constant(window - 1)`, `too_old = Greater(distance "st->st", window_ceiling "->st")` and return `(Maximum(is_padding, too_old), neg_infinity)`. `positions` is a rank-1 f32 leaf with the absolute position of each new row; `cache_dead` is a rank-1 f32 leaf over the cache rows, 1.0 where a row must not be read.
  2. `lfm2_two_range_cached_forward_program_with_layer_taps` gains a last parameter `sparse_positions: bool`. When true: declare once, before the cached-mask loop, `positions` (`[Extent::Symbolic(0)]`) and `cache_dead` (`[Extent::Symbolic(1)]`) with `input_leaf` (f32), and build every cached mask with `causal_mask_cached_at_positions(.., Extent::Symbolic(1), window)`. When `sparse_positions && sliding_kv_ring` return `TensorError::UnsupportedInBuilder { builder: "lfm2_two_range_cached_forward_program_with_layer_taps", feature: "sparse positions with a sliding ring" }` (a ring's slot index is not an absolute position). When false nothing changes: the built program stays node-for-node equal.
  3. The `_and_head_repeats` wrapper (card 9.3) and `descriptor.rs`'s two-range arm pass `false`. Every existing test call of `lfm2_two_range_cached_forward_program_with_layer_taps` in `spec/tests.rs` gets `false` appended as its new last argument, after the `&[]` or `&[1]` that card 9.4 left there: card 9.3's `tapped` build in `two_range_layer_taps_match_the_per_layer_reference` and both builds (`&[]` and `&[1]`, plus the `&[5]` error build) in card 9.4's `two_range_kv_edit_replaces_selected_rows`. `git grep -n "lfm2_two_range_cached_forward_program_with_layer_taps(" -- proxima-tensor/src/spec/tests.rs` lists every call to check, so the test build compiles at this commit.
- test: add `two_range_sparse_positions_read_the_cache_by_position` in `spec/tests.rs`: copy the fixture from card 9.3's test (`sliding_kv_ring = false`). Assert, in this order:
  1. the program with `sparse_positions = false` equals the old-name program (`Vec<Op>`);
  2. the `Op::Input` names of the `true` program include `positions` and `cache_dead`, and its `Op::Input` count is the `false` program's plus 2;
  3. evaluating the `true` program with `positions = [0.0, 1.0, 2.0]` and `cache_dead = [1.0, 1.0, 1.0]` gives logits equal to the old-name program's (`max_abs_diff < TOLERANCE`), since every zero-padded cache row is dead;
  4. with `cache_dead = [0.0, 0.0, 0.0]` the logits differ from the old-name program's by more than `TOLERANCE` (the control: admitting the zero rows changes the softmax);
  5. `sliding_kv_ring = true` with `sparse_positions = true` is `Err(TensorError::UnsupportedInBuilder { .. })`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_5 cargo nextest run -p proxima-tensor -E 'test(/two_range_sparse_positions_/)'`
- expect: `1 passed`
- also green: `cargo nextest run -p proxima-tensor -E 'test(/two_range/)'` prints card 9.4's count plus 1; clippy: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/lfm2_single_range_cached.rs`, `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): read cache rows by explicit query positions`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: edit `causal_mask_cached_windowed`; add an `Op` or `ScalarOp` variant
- gpu: none

### 9.6 the model descriptor carries the edit layers and the sparse-position switch

- id: FT9.6
- needs: FT9.5
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/descriptor.rs::ModelDescriptor` (~line 54; `sliding_kv_ring` ~line 99 is the precedent for a flag only the two-range arm reads) and the constructor `mistral_descriptor_from_shape` (~line 247, literal at ~line 288);
  - `proxima-tensor/src/spec/gguf_descriptor.rs`: the `Ok(ModelDescriptor { .. })` literal (~line 133);
  - `proxima-tensor/src/spec/tests.rs::build_forward_two_range_matches_direct_builder_call` (~line 14773): the descriptor literal (~line 14888) and the full evaluation fixture; and the helper `fn descriptor(cache_strategy: CacheStrategy, head_repeats: u32) -> ModelDescriptor` (~line 17482), whose second literal (~line 17504) also needs the two new fields;
  - `descriptor.rs::build_forward`'s `CacheStrategy::Cacheless` and `CacheStrategy::SingleRange` arms.
- premise check (stop and report if it fails): `git grep -n "ModelDescriptor {" -- . | grep -v "///"` prints exactly 8 lines at a7c08c4c: `descriptor.rs` ~54 (the struct), ~262 (the return type of `mistral_descriptor_from_shape`) and ~288 (its literal); `gguf_descriptor.rs` ~133 (literal); `spec/tests.rs` ~14888 (literal), ~17482 (the signature `fn descriptor(cache_strategy: CacheStrategy, head_repeats: u32) -> ModelDescriptor {`, which needs no edit) and ~17504 (literal); `proxima-model-interop/src/gemma4/bind.rs` ~725 (a struct-update `ModelDescriptor { head_repeats: .., ..descriptor }` that names only the fields it changes and needs no edit). Four literals need the new fields: `descriptor.rs`, `gguf_descriptor.rs` and the two in `spec/tests.rs`; a further literal that lists every field means this card also edits it.
- change:
  1. `descriptor.rs::ModelDescriptor` gains `pub kv_edit_layers: Vec<u32>` (doc: layers whose key and value rows a step may overwrite; empty is the ordinary program; two-range only, like `sliding_kv_ring`) and `pub sparse_positions: bool` (doc: the step reads its cache by explicit query positions; two-range without a sliding ring only). `mistral_descriptor_from_shape` sets `Vec::new()` and `false`.
  2. `build_forward`'s two-range arm passes `&descriptor.kv_edit_layers` and `descriptor.sparse_positions` (replacing the `&[]` and `false` of cards 9.4 and 9.5). The `Cacheless` and `SingleRange` arms return `TensorError::UnsupportedInBuilder { builder: "build_forward", feature: "kv edit layers or sparse positions outside the two-range strategy" }` when `!descriptor.kv_edit_layers.is_empty() || descriptor.sparse_positions`.
  3. `gguf_descriptor.rs` sets `kv_edit_layers: Vec::new()` and `sparse_positions: false`. both literals in `spec/tests.rs` (the one in `build_forward_two_range_matches_direct_builder_call` and the one in `fn descriptor`) do the same, so the `proxima-tensor` test build compiles.
- test: add in `spec/tests.rs`, next to the named existing test:
  - `build_forward_passes_kv_edit_layers_to_the_two_range_builder`: copy the descriptor literal of the named test with `kv_edit_layers: vec![1]`; `build_forward(&descriptor, false)` succeeds and the program's `Op::Input` names include `kv_edit.1.take`; with `kv_edit_layers: Vec::new()` the program equals the program of the old-name builder call over the same schedule (`Vec<Op>`);
  - `build_forward_rejects_edit_layers_on_other_strategies`: a single-range descriptor from `mistral_descriptor_from_shape(..)` (copy the arguments of the call at ~line 15108) with `kv_edit_layers` set to `vec![0]` matches `Err(TensorError::UnsupportedInBuilder { builder: "build_forward", .. })`; the same descriptor with `sparse_positions` true does too.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_6 cargo nextest run -p proxima-tensor -E 'test(/build_forward_passes_kv_edit_layers_/) | test(/build_forward_rejects_edit_layers_/)'`
- expect: `2 passed`
- also green: `cargo nextest run -p proxima-tensor -E 'test(/build_forward/)'` prints the same count as before the edit plus 2; `cargo check -p proxima-model-interop --features std` passes (the only interop use is the struct-update at `gemma4/bind.rs` ~725, which names only `head_repeats`, so the new fields come from `..descriptor`); clippy: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/gguf_descriptor.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): put kv edit and sparse positions in the descriptor`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: touch `proxima-model-interop`; give `LoadedModel` a descriptor field
- gpu: none

### 9.7 a cached entry is reusable only by a request running the assemble list that produced its rows

- id: FT9.7
- needs: FT2, FT8
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey` (~line 32) and `CacheKey::of` (~line 82): the struct is `Copy, PartialEq` and `of` destructures `ServingConfig` with no `..`; after the assemble slice the destructure names the assemble field `_` with a reason, and this card binds it;
  - `proxima-model-interop/src/generate/prompt_cache.rs`: `CacheEntry::new` and `CacheEntry::empty` (~line 385, both `const fn`), and the three comparisons `entry.key == *key` in `best_candidate` (~line 715), `miss_reason` (~line 728) and `bloom_candidates` (~line 877);
  - `proxima-model-interop/src/generate/chunk_shift.rs::prefill_through_runs` (~line 538): `entry.apply_moved(lifted)?` is where an entry gets rows from another position;
  - `proxima-model-interop/src/serving.rs` (~lines 509-513): the documented hazard that an entry built from a shift "carries those rows into later requests".
- premise checks (stop and report the real names if any fails): the prerequisites FT2 and FT8 checks; `git grep -n "entry.key == \*key" -- proxima-model-interop/src/generate/prompt_cache.rs` prints 3 lines.
- why no plan key card: the decode plan key (`proxima-model-interop/src/generate/resident_plans.rs::DecodePlanKey`, ~line 40, `(usize, usize, Vec<NodeId>, bool)`) is built for a program that is fixed at load (`plan_life`, `resident_plans.rs` header); in this slice the edit leaves live in the descriptor (card 9.6), so a model's program does not change per request and the key already separates plans by the shape and outputs they read. Its second premise, that `resolve_cached_plan` (`residency_caches.rs`, ~line 2755) clears the map on a miss, is true on main and is irrelevant to this design.
- change:
  1. `prompt_cache_key.rs`: `CacheKey` gains `pub(super) assemble: u64` (doc: the digest of the assemble list whose steps wrote rows into the cached state; `0` when no assemble step did). Add `pub(super) fn assemble_digest_of(config: &ServingConfig) -> u64`: `0` when `config.prefill.assemble.is_empty()` (the default `PrefillConfig` holds `&[]`), otherwise an FNV-1a 64-bit fold over the bytes of `format!("{list:?}")`, with a result of `0` mapped to `1`. `CacheKey::of` sets `assemble: assemble_digest_of(config)` (the `prefill: _,` destructure line and its comment stay: `config` is still the whole struct). A key built for a non-empty list now differs from the default key, so the test `cache_key_ignores_the_assemble_list` that the FT2 slice added to `prompt_cache_key.rs` states the opposite of this card; it is replaced in place (see test, first bullet). Add `pub(super) fn admits(&self, entry: &CacheKey) -> bool { Self { assemble: 0, ..*entry } == Self { assemble: 0, ..*self } && (entry.assemble == 0 || entry.assemble == self.assemble) }` (doc: `self` is the request's key).
  2. `prompt_cache.rs`: `CacheEntry::new` stores `key: CacheKey { assemble: 0, ..key }` (a new entry holds no assembled rows yet). The three comparisons become `key.admits(&entry.key)`.
  3. `chunk_shift.rs::prefill_through_runs` (~line 538): its body ends in the tail expression `self.prefill_through_stops(ids, entry, &remaining, &remaining, widths, config, serving_config, runtime, forced_draft_width, &mut |_position| ControlFlow::Continue(()))` that returns the entry. Bind that call's result and fix the key after it, so the digest is written on the entry the function returns: replace the tail expression with `entry = self.prefill_through_stops(<the same arguments>)?;` followed by `if !moved.is_empty() { entry.key.assemble = assemble_digest_of(serving_config); }` followed by the new tail `Ok(entry)` (`entry` is already `mut`; `moved` is still alive, it is only borrowed by the loop). Import `assemble_digest_of` at the top of the file.
- test:
  - replace `cache_key_ignores_the_assemble_list` in the existing `#[cfg(test)] mod tests` of `prompt_cache_key.rs` (created by the FT2 slice; add no second `mod tests`) with `cache_key_binds_the_assemble_list`, same setup (`steps = [AssembleStep::Prefix, AssembleStep::Shift]`, `config = ServingConfig { prefill: PrefillConfig { assemble: &steps }, ..ServingConfig::default() }`, keys built with `CacheKey::of(&config, false, RopeScaling::None, 0, 0)`): the key's `assemble` equals `assemble_digest_of(&config)` and is not `0`; the key is not equal to `CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)`; two keys built from the same `config` are equal. The old test is deleted, so the test count of the module does not change from this replacement;
  - add `cache_key_admission_table` in the same existing `mod tests`: with `base = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)` (the helper the `prompt_cache.rs` tests use as `base_key`), `blended = CacheKey { assemble: 7, ..base }` and `other = CacheKey { assemble: 9, ..base }`: `base.admits(&base)`; `blended.admits(&base)` (a fresh entry is admitted by any list); `blended.admits(&blended)`; `!base.admits(&blended)` (a default request refuses rows another list wrote); `!other.admits(&blended)`; `!blended.admits(&CacheKey { gpu_route: true, ..base })` (any other field still has to match);
  - add `assemble_digest_distinguishes_lists` in the same module, with `digest_of(steps: &[AssembleStep]) -> u64 = assemble_digest_of(&ServingConfig { prefill: PrefillConfig { assemble: steps }, ..ServingConfig::default() })` as a local test helper: `digest_of(&[]) == 0`; `digest_of(&[Prefix])`, `digest_of(&[Prefix, Shift])` and `digest_of(&[Prefix, Shift, Shift])` are each nonzero and pairwise different; `digest_of(&[Prefix, Shift]) == digest_of(&[Prefix, Shift])`;
  - add `shifted_entry_is_refused_by_a_default_request` in the existing `mod tests` of `prompt_cache.rs`, using that module's helpers: `let mut entry = state_with_ids(&[2, 105, 2364, 107]); entry.key.assemble = 7;` (set after construction, because `CacheEntry::new` stores `0`); `let mut cache = PromptCache::new(); cache.store(entry, &enabled_config());`; prompt `[2, 105, 2364, 107, 9259]`, widths `shared_widths()`, `min_similarity_milli = ANY_OVERLAP`, `lift = None`. First call `cache.take_best_shifting(&prompt, &base_key(), &shared_widths(), ANY_OVERLAP, None)` returns `(None, report)` with `report.miss == Some(MissReason::ConfigMismatch)` and `cache.stored_bytes() > 0` (the entry stays). Second call with the key `CacheKey { assemble: 7, ..base_key() }` and the same other arguments returns `(Some(entry), report)` with `report.miss == None`, `report.lcp == 4` and `entry.state.ids == [2, 105, 2364, 107]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_7 cargo nextest run -p proxima-model-interop --features std -E 'test(/cache_key_binds_the_assemble_list/) | test(/cache_key_admission_table/) | test(/assemble_digest_distinguishes_lists/) | test(/shifted_entry_is_refused_by_a_default_request/)'`
- expect: `4 passed`
- also green: `cargo nextest run -p proxima-model-interop --features std -E 'test(/prompt_cache/)'` prints the same pass count as before the edit plus 3 (the replaced test keeps its slot; write the before count in the commit body); `git grep -c "cache_key_ignores_the_assemble_list" -- proxima-model-interop/src` prints nothing (the old test is gone); clippy: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache_key.rs`, `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/chunk_shift.rs`
- commit: `fix(interop): refuse entries built by another assemble list`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: add a provenance type or a field to `CacheEntry`; change the signature of `take_best`, `take_best_shifting` or `prompt_cache_lookup`
- gpu: none

### 9.8 the worked 3-chunk selection test

- id: FT9.8
- needs: FT3.2, FT0.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `worked-examples.md` section `## per-layer recompute selection` (FT0.16): the numbers below are copied from it; if they differ, stop;
  - `proxima-tensor/src/spec/primitives.rs::top_fraction_mask` (FT3.2);
  - `proxima-model-interop/tests/arch_data_baseline.rs` header (~lines 14-36): the `#![cfg(feature = "std")]` and import style of an integration test;
  - `proxima-model-interop/Cargo.toml` (`[dev-dependencies]`, ~line 348): `proxima-tensor` with `config`, `std`, `test-support` is already a dev-dependency, so the test may `use proxima_tensor::...` with no Cargo edit.
- premise checks (stop and report if any fails): FT3.2 and FT0.16 checks above; `grep -n '^proxima-tensor' proxima-model-interop/Cargo.toml` prints at least one line.
- change: `proxima-model-interop/tests/cacheblend.rs` (new), `#![cfg(feature = "std")]`, `#![allow(clippy::unwrap_used, clippy::expect_used)]`. No production code is changed: the card is the test, which drives the top-fraction selection primitive over the worked example. The keep counts are the literals of the worked example; no library function computes them.
- test: add `blend_selection_worked_three_chunks`. Rows M = 12 (3 chunks of 4: chunk c holds rows 4c..4c+3); keep counts `[6, 3, 2]` for check layers `[1, 2, 3]` (half of the 12 loaded rows, then half again, rounded up, as the worked example computes). Inputs (f32, length 12):
  - `deviation_1` = `[0.05, 0.90, 0.10, 0.40, 0.02, 0.30, 0.75, 0.08, 0.60, 0.15, 0.04, 0.50]`;
  - `deviation_2` (rows outside the first selection are 0.0) = `[0.0, 0.50, 0.0, 0.80, 0.0, 0.20, 0.70, 0.0, 0.10, 0.0, 0.0, 0.60]`;
  - `deviation_3` = `[0.0, 0.0, 0.0, 0.30, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.50]`;
  - `keep` = 1.0 at row 11 only (the last row), else 0.0.
  Build one program: `input_leaf` for `deviation_1..3` (shape `[Extent::Symbolic(0)]`), `keep` (same shape), and `keep_count_1..3` (rank 0, `Vec::new()` shape); `mask_j = top_fraction_mask(&mut program, deviation_j, keep_count_j, Some(keep))`. Bind `keep_count_1`, `keep_count_2`, `keep_count_3` to the literals 6.0, 3.0, 2.0. Evaluate with `proxima_tensor::cpu::evaluate_named(&program, &[12_u64], &named, &[mask_1, mask_2, mask_3])`. Assertions, each after asserting the mask length is 12:
  - indices equal to 1.0 in `mask_1` are exactly `[1, 3, 5, 6, 8, 11]`;
  - in `mask_2` exactly `[3, 6, 11]`; in `mask_3` exactly `[6, 11]`;
  - the per-chunk counts of `mask_1` (rows 0..4, 4..8, 8..12) are `[2, 2, 2]`;
  - the counts of 1.0 in `mask_1`, `mask_2`, `mask_3` are `[6, 3, 2]` (the bound keep counts, each already holding the kept last row).
  Imports: `proxima_tensor::spec::{input_leaf, top_fraction_mask}`, `proxima_tensor::{DType, op::Extent}`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_8 cargo nextest run -p proxima-model-interop --features std -E 'test(/blend_selection_worked_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cacheblend.rs`
- commit: `test(interop): check the three chunk blend selection by hand`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: name any other test `blend_selection_worked_`; edit `worked-examples.md`; load a model
- gpu: none

### 9.9 the technique as a test pipe: selective recompute driven through the tap and the edit hook

- id: FT9.9
- needs: FT9.5
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/tests.rs::two_range_cached_gemma4_shared_kv_layer_matches_cacheless_oracle` (~line 14404): a three-layer fixture (sliding layer, full layer, third layer that shares key and value from layer 1) with all three layers' weights; this card changes the third layer to own its cache;
  - the test added by card 9.4 (`two_range_kv_edit_replaces_selected_rows`): the `kv_edit` leaf names and how rows are evaluated and fed back;
  - `proxima-tensor/src/spec/tests.rs::rope_table` (~line 12012) and `rope_table_partial` (~line 12041): the cos/sin tables for an arbitrary position list;
  - sketch `09-cacheblend.md` section 5 (the worked example): chunks prefilled alone, rows reused at new positions, a fraction recomputed by key/value deviation.
- change: none outside the test. This card is the proof that the hooks express the technique: the pipe below is test code, at most about 40 lines, and no library item is added.
- design of the test (worked by hand, because the numbers decide what the assertions are): prompt ids `[1, 3, 2]` (rows 0, 1, 2); chunk A is row 0 prefilled alone at position 0, chunk B is rows 1 and 2 prefilled alone at positions 1 and 2 (same ids as the fixture, rope tables built for positions `[1, 2]`, which stands for the shift the lifted rows have already had applied). Layers: 0 sliding (window `SWA_WINDOW = 2`), 1 full, 2 full, all owning their cache. The first layer's key/value row depends only on its own token, so the first-layer deviation is zero. At layer 1 the sliding layer 0 has already mixed rows: row 0 attends itself only (same alone and in context), row 1 attends rows 0 and 1 in context but only itself alone, row 2 attends rows 1 and 2 in context and in chunk B (the window is 2), so chunk B's row 2 is unchanged. So the layer-1 deviation is zero for rows 0 and 2 and positive for row 1, and the check layer is 1 and the edit layer is 2.
- test: add `blend_through_the_hooks_selects_by_deviation_and_matches_endpoints` in `proxima-tensor/src/spec/tests.rs`: copy the three-layer test's fixture, change layer 2's schedule entry to the config of layer 1 (`value_source_kind: SharedWithKey`, `key_source_kind: ProjectedK`, `mask_window: None`), add the leaves `blk.2.attn_k.weight` (`layer2.wk`), `blk.2.attn_k_norm.weight` (`layer2.k_norm`) and the zero `kv_cache.2.k_even`, `kv_cache.2.k_odd`, `kv_cache.2.v`. Build the program with `lfm2_two_range_cached_forward_program_with_layer_taps(.., head_repeats = 1, kv_edit_layers = &[2], sparse_positions = false)` (the parameter order card 9.4 and 9.5 left it in: the original parameters, `head_repeats`, `kv_edit_layers`, `sparse_positions`). A local closure `run(ids, positions, take, loaded)` evaluates logits, the three layers' `cache_roots` values and the layer residuals with symbols `[ids.len() as u64, ids.len() as u64]`, feeding `kv_edit.2.*` from `loaded`. The pipe, in order:
  1. `plain` = `run([1, 3, 2], [0, 1, 2], take = [0.0; 3], loaded = zeros)`: the full recompute; keep its logits and the layer-1 and layer-2 `cache_roots` values (fresh rows);
  2. `alone_a` = `run([1], [0], ..)` and `alone_b` = `run([3, 2], [1, 2], ..)`; `loaded` for layers 1 and 2 are the rows of `alone_a` followed by the rows of `alone_b`;
  3. `deviation[row]` = the sum over the layer-1 key even, key odd and value values of `|loaded - fresh|` for that row (plain loops, about 6 lines);
  4. for each `ratio_milli` in `[0, 333, 1000]`: `keep = (3 * ratio_milli).div_ceil(1000)`; the recompute set is the `keep` rows of largest deviation (ties to the lower index) plus row 2; `take[row] = 1.0` for rows outside the set; `logits = run([1, 3, 2], [0, 1, 2], take, loaded layer 2)`.
  Assertions, in this order:
  - `deviation[0] < TOLERANCE`, `deviation[2] < TOLERANCE`, `deviation[1] > TOLERANCE` (the hand-worked pattern above);
  - the recompute sets for the three ratios are exactly `{2}`, `{1, 2}`, `{0, 1, 2}`;
  - `ratio_milli = 1000` and `333`: `max_abs_diff(logits, plain logits) < TOLERANCE` (row 0's loaded rows equal its fresh rows, so keeping them changes nothing);
  - `ratio_milli = 0`: `max_abs_diff(logits, plain logits) > TOLERANCE` (row 1's layer-2 rows are stale and the last row reads them);
  - the program built with `kv_edit_layers = &[]` and `sparse_positions = false` equals the old-name program over the same schedule (`Vec<Op>`), repeating the default-reproduces check on this three-layer schedule.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_9 cargo nextest run -p proxima-tensor -E 'test(/blend_through_the_hooks_/)'`
- expect: `1 passed`
- also green: clippy: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/tests.rs`
- commit: `test(tensor): drive blend through the edit and tap hooks`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: add library code; add a function named after the technique outside a test; if an inequality above fails, stop and report the printed values instead of changing the threshold
- gpu: none

### 9.10 oracle: prefix-only assembly on gemma4 E2B equals llama's plain ids

- id: FT9.10
- needs: FT8, FT2, FT0.50, FT0.51, FT9.7, FT9.8
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json` (FT0.50, FT0.51): `[0].request1.prompt_ids`, `[0].request2.prompt_ids`, `[0].request2.generated_ids` (llama-server with `--cache-reuse 64`), `[0].request2_plain.generated_ids` (the same prompt with no chunk reuse). Premise check: `jq -e '.[0] | (.request2_plain.generated_ids | length) == 32' <file>` prints `true`; if the file or key is absent, stop and report;
  - `proxima-model-interop/tests/arch_data_baseline.rs`: `Checkpoint` and `GEMMA4_E2B` (~lines 49-59, `resolved_path` and `open` ~lines 99-122), `llama_parity` (~line 667) and `first_divergence` (~line 650): the loader pattern, copied into `tests/cacheblend.rs` because they are private to that test binary. `LoadedModel::load(&parsed, file_bytes)` with `parsed = proxima_gguf::parse_complete(file_bytes)`;
  - `proxima-model-interop/src/generate/decode.rs::LoadedModel::generate_from_ids` (~line 2277): `(&self, prompt_ids: &[u32], max_tokens: usize, serving_config: &ServingConfig, on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>) -> Result<(Vec<u32>, String, bool), InteropError>`;
  - `proxima-model-interop/src/generate/prompt_cache.rs::CacheReport` (~line 161): `path`, `shifted_tokens`; `LoadedModel::last_prompt_cache_report` (~line 1017) returns `Option<CacheReport>`.
- premise checks (stop and report if any fails): FT9.8 left `proxima-model-interop/tests/cacheblend.rs` in the tree (`git ls-files proxima-model-interop/tests/cacheblend.rs` prints that path); `git grep -n "pub struct PrefillConfig" -- proxima-model-interop/src/serving.rs` and `git grep -n "pub enum AssembleStep" -- proxima-model-interop/src/serving_grammar.rs` print one line each (FT2); `proxima_model_interop` exports `AssembleStep`, `PrefillConfig`, `PromptCacheConfig`, `ServingConfig`, `CachePath` and `LoadedModel` from `lib.rs`.
- change: `proxima-model-interop/tests/cacheblend.rs` (created by card 9.8). Add at the end of the file one module that holds everything this card and card 9.11 add, gated so an unfiltered `--features std` run never loads a model: `#[cfg(feature = "metal")] mod gemma4_e2b { .. }`, with its own `use` lines inside the module (`core::ops::ControlFlow`, `std::fs::File`, `std::path::Path`, `proxima_gguf::parse_complete`, `proxima_model_interop::{AssembleStep, CachePath, LoadedModel, PrefillConfig, PromptCacheConfig, ServingConfig}`). Inside it:
  - `const E2B_ENV: &str = "PROXIMA_ARCH_GEMMA4_E2B_GGUF";` and `const E2B_PATH: &str` set to the `path` literal of `GEMMA4_E2B` in `arch_data_baseline.rs` (~line 59, copy it byte for byte); `fn open_e2b() -> memmap2::Mmap`: the path is `std::env::var(E2B_ENV)` or `E2B_PATH`; assert the file exists with a message naming the path and `E2B_ENV`; open it and `unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only")` (the same body as `Checkpoint::open`);
  - `fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize>`: the body of the one in `arch_data_baseline.rs` (~line 650);
  - `fn ids_of(value: &serde_json::Value) -> Vec<u32>`: the body of the one in `arch_data_baseline.rs` (~line 649);
  - `fn gemma4_e2b_cache_reuse_record() -> serde_json::Value`: reads `Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json")` with `std::fs::read_to_string` (failing with the path in the message), parses it as `Vec<serde_json::Value>`, asserts it is not empty, and returns element 0;
  - `fn two_requests(assemble: &[AssembleStep]) -> (Vec<u32>, CachePath, usize)`: opens `open_e2b()`, parses and loads the model (`parse_complete(&mapping)`, `LoadedModel::load(&parsed, &mapping)`, each failure a panic naming the step), builds `let config = ServingConfig { prompt_cache: PromptCacheConfig { byte_budget: 1_073_741_824, cache_reuse_min: 64, ..PromptCacheConfig::standard() }, prefill: PrefillConfig { assemble }, ..ServingConfig::default() };` (the same values as the TOML `[prompt_cache] byte_budget = 1073741824, cache_reuse_min = 64`, written as the struct the loader lowers to, so no loader call is needed), reads `record = gemma4_e2b_cache_reuse_record()`, then calls `model.generate_from_ids(&ids_of(&record["request1"]["prompt_ids"]), 1, &config, &mut |_event| ControlFlow::Continue(()))` (result `.expect("request 1 stores the entry")`), then `model.generate_from_ids(&ids_of(&record["request2"]["prompt_ids"]), 32, &config, &mut |_event| ControlFlow::Continue(()))` and keeps the first tuple element as `generated`; then `let report = model.last_prompt_cache_report().expect("a cached request records its report");` and returns `(generated, report.path, report.shifted_tokens)`.
- test: inside `mod gemma4_e2b`, add `#[test] fn assemble_prefix_only_matches_llama_plain_ids_gemma4_e2b()`: `let (generated, path, shifted_tokens) = two_requests(&[AssembleStep::Prefix]);` (a list with no shift step, so chunk moves are off even though `cache_reuse_min` is set); assert `first_divergence(&ids_of(&gemma4_e2b_cache_reuse_record()["request2_plain"]["generated_ids"]), &generated) == None`; assert `path != CachePath::Shift` and `shifted_tokens == 0`.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing (wait while it prints a match); then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_10 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/assemble_prefix_only_matches_llama_plain_ids_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/cacheblend.rs`
- commit: `test(interop): prefix only assembly matches llama plain ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: query llama-server or Ollama; load any checkpoint other than gemma4 E2B; change the recorded ids
- gpu: one run, waiting for a quiet box (the peer-gate check in CARDS.md machine safety)

### 9.11 oracle: prefix then shift assembly on gemma4 E2B equals llama's cache-reuse ids

- id: FT9.11
- needs: FT9.10
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first: card 9.10 (same record, and the `mod gemma4_e2b` helpers `two_requests`, `first_divergence`, `ids_of`, `gemma4_e2b_cache_reuse_record`); `proxima-model-interop/src/generate/chunk_shift.rs` header (~lines 21-35): moved rows keep the keys they had under the older context, which llama-server's `--cache-reuse` does too, so the recorded `request2.generated_ids` is the oracle.
- change: `proxima-model-interop/tests/cacheblend.rs`, inside the `#[cfg(feature = "metal")] mod gemma4_e2b` that card 9.10 created: add the test below, reusing `two_requests` of card 9.10 with the assemble list `[AssembleStep::Prefix, AssembleStep::Shift]` (the configuration of card 9.10 plus a shift step after the prefix step).
- test: add `#[test] fn assemble_prefix_then_shift_matches_llama_cache_reuse_ids_gemma4_e2b()`: `let (generated, path, shifted_tokens) = two_requests(&[AssembleStep::Prefix, AssembleStep::Shift]);` assert `first_divergence(&ids_of(&gemma4_e2b_cache_reuse_record()["request2"]["generated_ids"]), &generated) == None`; assert `path == CachePath::Shift` and `shifted_tokens >= 64`. If the ids diverge, stop and report the first divergent index and the sliding-layer window (`gemma4_e2b/swa_layers.txt`): sliding layers are where moved rows older than the window are absent. Do not change the expected ids.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing; then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_9_11 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/assemble_prefix_then_shift_matches_llama_cache_reuse_ids_gemma4_e2b/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/cacheblend.rs`
- commit: `test(interop): prefix then shift assembly matches llama reuse ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message
- do not: run card 9.10's test here; query llama-server or Ollama
- gpu: one run, waiting for a quiet box

## spec drift

1. Old cards 9.1 and 9.2 put the per-check row count and the mask-in-force rule in `proxima-core/src/serving_state/blend.rs`. Withdrawn: they are the technique's own schedule and their only consumer was a test (see "dropped"). This slice touches no `proxima-core` file.
2. The sketch's GAP-3 text ("no hidden state at a layer boundary on the two-range engine") is about the residual only; the key and value rows are already returned. Card 9.3 is cut to the residual accordingly.
3. The sketch's GAP-5 proposed a provenance enum on `CacheEntry`. The existing key already is "what an entry was built under" (`prompt_cache_key.rs` header), so card 9.7 extends it with a digest and adds no type. The cost: a request running a non-default assemble list can no longer reuse entries that a default-list request left holding shifted rows; the written reason is the documented hazard at `serving.rs` ~lines 509-513.
4. SPEC R12a said the recompute ratio ranks the deviation "in the top recompute_ratio of rows"; the ratio is over loaded rows and the selection is the integer keep count, written as a literal in the worked example of card 9.8 and computed in the technique's own test (card 9.9: `(3 * ratio_milli).div_ceil(1000)`).
5. Card 9.9's second-layer deviation pattern follows from the sliding window of 2 in the fixture; with a different window the zero and positive rows change and the card's assertions must be re-derived, not loosened.

## slice exit

From the checkout holding main, after the FT0.4 premise check passes and `source proxima-tensor/specs/fsm-techniques/env.sh`:
1. `cargo nextest run -p proxima-tensor -E 'test(/two_range_layer_taps_/) | test(/two_range_kv_edit_/) | test(/two_range_sparse_positions_/) | test(/build_forward_passes_kv_edit_layers_/) | test(/build_forward_rejects_edit_layers_/) | test(/blend_through_the_hooks_/)'`: `6 passed`.
2. `cargo nextest run -p proxima-model-interop --features std -E 'test(/cache_key_binds_the_assemble_list/) | test(/cache_key_admission_table/) | test(/assemble_digest_distinguishes_lists/) | test(/shifted_entry_is_refused_by_a_default_request/) | test(/blend_selection_worked_/)'`: `5 passed`.
3. One command at a time, `-j 1`, after the quiet-box check (`ps -axo comm` peer check first):
   - `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/assemble_prefix_only_matches_llama_plain_ids_gemma4_e2b/)'`: `1 passed`;
   - `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/assemble_prefix_then_shift_matches_llama_cache_reuse_ids_gemma4_e2b/)'`: `1 passed`;
   - `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^arch_data_digest_/)'`: the recorded digest count unchanged from before slice 9 (cards 9.3 to 9.6 must not move any op-graph digest, including the `bind.residual_roots=0` line of the gemma4 digests). Print the count before the first card and compare; a count of zero is red.
   A different count is reported with the failing test names and first divergent indices, not fixed here.
4. `git grep -nP '\b(struct|enum|trait)\s+\w*(CacheBlend|Hkvd|HKVD|Blend)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src | wc -l`: `0` (no type named after the technique).
5. `git grep -niP '\b(fn|const)\s+\w*(blend|hkvd)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src ':!*/tests.rs' | wc -l`: `0` (no function or constant named after the technique outside a test file), and `git diff --stat main -- proxima-core | wc -l`: `0` (this slice leaves `proxima-core` untouched).
Record each printed line in TASKS.md slice 9's `Done:` field.
