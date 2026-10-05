# slice 3 cards (re-cut): top-fraction row selection

anchors read at main d53f78f3 (full sha d53f78f38be9007ac3d29fe135d1a437f6cce837). Read each with
`git show main:<path>`; the working tree is not the source. Line numbers are hints only: re-locate
every anchor by its symbol before editing.

## what this slice is, and what it is not

Hook served: the specialize hook (lowering thresholds in `omega-runtime.toml` choose the kernel). No
technique is built here: no read rule, no recompute rule, no policy for how many rows to keep. The cards add:

1. one expression in the five-op algebra, the rank-count top-fraction mask. Its caller supplies the number of
   rows to keep as a rank-0 input, the way `causal_mask_merged` takes its cached length. It sits in
   `spec/primitives.rs` beside the other mask builders;
2. the specialization of that expression: a fused bound kind, its CPU reference, its bind-time matcher and its
   Metal kernel. The fused kind selects the same rows as the expression, so a lowering that picks it changes
   speed and nothing else;
3. the threshold, in `omega-runtime.toml`, that chooses between the plain expression and the kernel.

Techniques consume the expression from their own hooks in later slices (block top-n at the read hook, the
recompute choice at the assemble hook), as tests or examples. They reach the kernel with no further wiring,
because the Metal prepare path binds through `bind_with_top_fraction` once the card that lands the kernel is in.
Interop plans through `omega::plan_named` (`proxima-model-interop/src/generate/residency_caches.rs`), whose Metal path
binds in `omega/src/metal/prepare_uniforms_pack.rs::prepare`; the Metal test card observes the kernel run through
`plan_named`. The two interop calls to `bind_with_fusion` (`proxima-model-interop/src/generate/decode.rs`,
`run_attn_fuse_parity_probe` and the unfused-chain walk) are diagnostic probes, not the production bind.

The proof through the hook is the Metal card and the kernel-count card. They build the expression the way a
technique does and plan it through the production entry, and the threshold in `omega-runtime.toml` alone decides
whether the kernel runs. Control, observed in the scratch export described below: the Metal test file with
`OMEGA_SELECTION_TOP_FRACTION_MIN_ROWS=100000` set at build time printed `5 tests run: 2 passed, 3 failed`, the
three failures being the assertions that the selection kernel ran. A model-level proof (gemma4 dense or MoE,
granite MoE) belongs to the slices that consume the expression; this slice loads no model, so the test-model rule
has nothing to attach to here and every test uses synthetic scores.

Rules: CARDS.md applies to every card. Every card uses
`CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_<n>` and removes it when done.

## build-check recipe (cards that run `cargo check`)

A build check is counted, not eyeballed. Each card that runs one carries these two lines, then the table of
checks. With `LOG` set to the card's log directory (`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft3_<n>`,
created with `mkdir -p`):

```
count_build() { grep -E '"reason":"(compiler-artifact|build-finished)"' "$1" | grep -o -E "\"kind\":\[\"lib\"\],\"crate_types\":\[\"lib\"\],\"name\":\"$2\"|\"success\":(true|false)" | sort | uniq -c; }
cargo check <args> --message-format json > "$LOG/<label>.json"; count_build "$LOG/<label>.json" <crate_name>
```

`count_build` prints two lines: how many library targets of `<crate_name>` finished (the library, and its unit
test build when `--all-targets` is given), and `1 "success":true`. A check that built nothing prints neither;
a check that hit a compile error prints `1 "success":false`.

## triage result for this file

Every card was re-derived against main d53f78f3. The changes of the cards were applied in order to a scratch export
of main (`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft3_repair/export`, outside the working tree),
and every test, check and clippy command below was run there at the counts the cards state; the Metal test file
ran on the local GPU with no peer process present.

old-to-new id map:

| old id | now | what happened |
|---|---|---|
| FT3.1 | none | dropped: the keep-count rule is a policy a technique supplies, not a library function. Tests state their own count |
| FT3.2 | FT3.2 | kept; the tests state the count (a quarter of the rows, rounded up) instead of calling the dropped rule |
| FT3.3 | FT3.3 | kept |
| FT3.4 | FT3.4 | recut: the variant is no longer behind a feature, so every arm in proxima-tensor, omega and one interop test lands in the same commit; the CPU runner moved out |
| FT3.4 (runner half) | FT3.5 | new card, split from the old FT3.4 |
| FT3.5 | FT3.6 and FT3.7 | recut: the recognizer (FT3.6) and the bind entry point (FT3.7), split because the file was 211 lines |
| FT3.6 | none | dropped as a card: its arms moved into FT3.4, its Metal routing moved into FT3.9 |
| FT3.7 | FT3.8 | recut: 135 lines down to 118 |
| FT3.8 | FT3.9 | recut: now also routes the Metal prepare path through the fusion, in the commit that has a kernel to run |
| FT3.9 | FT3.10 | kept |

## premises that main falsified (fixed in the cards below)

1. `BoundOpKind` has no cfg-gated variant on main: `RoundBatchedReduce` (`types_layout_boundop.rs` ~line 425) and
   `MoeTopK` (~line 308) are both ungated, and omega's own feature comments say its renderers are always compiled
   while a feature gates only the bind-time matcher (`omega/Cargo.toml`, the `gated-delta-net-fusion` comment). So
   `TopFractionSelect` is ungated too. A gated variant would leave a workspace build that unifies the feature
   without omega's arms red, and an ungated one forces an arm in every exhaustive match in the same commit. That
   fan-out is FT3.4. The feature `top-fraction-fusion` gates only the matcher (FT3.6, FT3.7) and the Metal routing
   (FT3.9).
2. The exhaustive matches over `BoundOpKind` are larger than the old list, and part of them compile only under
   non-default features. `cargo check -p proxima-tensor --all-features --all-targets`,
   `cargo check -p omega --all-targets --features instrument,cuda,wgpu-backend` and
   `cargo check -p proxima-model-interop --features std --all-targets` print every one as an E0004. proxima-tensor
   needs arms in `types_layout_boundop.rs` (six), `refresh.rs`, `quantized_eval.rs`, `run_node.rs`, `typed_eval.rs`
   (three), `identity_copy_alias.rs` (feature `identity-copy-alias`), `repeat_nodes.rs` (two, feature `instrument`),
   `arena.rs` (feature `instrument`), `examples/scaling.rs`, `bind/tests.rs` and `spec/tests.rs`. omega needs arms in
   `identity.rs`, `msl/signature_tokens_prelude.rs` (two), `msl/emit_and_classify.rs`,
   `metal/prepare_uniforms_pack.rs` (two), `metal/dispatch_timed_and_classify.rs` (feature `instrument`), `cuda.rs`
   (three), `wgsl.rs` (two) and `wgpu_driver.rs`. proxima-model-interop needs one in
   `tests/real_qwen35moe_registry_probe.rs`, which compiles only under interop's `std` feature (an earlier cut
   missed it because interop's default features are empty). `omega/src/msl/kernel_types_identity.rs` and
   `metal/arena_encode_dispatch_finish.rs` need none. `spec` (where `top_fraction_mask` lives) is behind the
   `config` feature, so the matcher's feature pulls in `config`.
3. `omega/src/sized.rs` gets its constants through `include!` of the generated file, which makes them public
   already. There is no per-name re-export to add, and the file has no test module yet.
4. The matcher cannot use `moe_binary_elementwise`, `moe_reduce_operand`, `moe_consumers`: they are gated on
   `moe-topk-fusion`. With `reduce-epilogue-fusion` on (omega's default Metal build), the rank reduce absorbs the
   final compare as an epilogue, so the rank-0 count input is an epilogue operand: layouts must be read through
   `BoundOp::all_read_sources`, not `operands`.
5. The omega Metal entry (`plan_named`) always binds with `SELECTION_TOP_FRACTION_MIN_ROWS` (256), so a 12-row
   worked input can never reach the kernel through it. The Metal tests use the same twelve scores tiled to 255 and
   256 rows, with the selected set derived by hand.
6. The unsupported-backend arms cannot all use one error variant. `EmitError::UnsupportedOpKind` exists only under
   `wgpu-backend` and `EmitError::CudaUnsupportedOpKind` only under `cuda` (`omega/src/error.rs`), so only the wgsl
   and cuda arms use them. The Metal arm (compiled by default) returns `EmitError::EpilogueNotSupported`, the variant
   the neighbouring no-renderer arm for `RoundBatchedReduce` returns.
7. `bind_with_top_fraction` takes `fuse_cached_attention`, because the Metal path passes its own flag
   (`prepare_uniforms_pack.rs` ~line 155), not a constant.
8. Zero and NaN: `-0.0 == 0.0` in the expression's `Equal`, so the reference and the kernel canonicalise zero before
   ordering. NaN scores are outside the contract (the expression ranks a NaN row 0): stated in FT3.4's doc.
9. A union expression (`keep_rows` given) also contains the mask-only expression as its prefix, so a recognizer that
   tries every node as a root finds two overlapping matches. `top_fraction_candidates` (FT3.6) drops a candidate whose
   node another candidate absorbs, so the union expression yields one candidate.
10. The kernel needs no special branch for a keep count of zero: the radix passes then pick the all-ones key, nothing
    compares above it, and no tie is taken, so every row writes 0 (or its `keep_rows` flag). FT3.9 checks it on the GPU.
11. The previous file capped test names so that a substring count held. That cap is dropped; each card states its own
    count and names its tests.

## worked value (every card below uses these numbers)

Scores, M = 12, indices 0..11:
`[0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0]`.

Count to keep: a quarter of the rows, rounded up: `K = ceil(M / 4)`, in Rust `rows.div_ceil(4)`. No `keep_rows`
unless a card says so.

Rank of row i: `rank(i) = #{j : s[j] > s[i]} + #{j < i : s[j] == s[i]}`. Row i is selected iff
`rank(i) < K`. Ties are broken toward the lower index, and exactly K rows are always selected.

| case | M | scores used | K | selected set |
|---|---|---|---|---|
| at | 12 | all 12 | `ceil(12/4)` = 3 | {1, 7, 11} |
| below | 11 | first 11 (drop index 11) | `ceil(11/4)` = 3 | {1, 3, 7} |

Hand check, `at`: 5.0 (index 11) has rank 0; 4.0 (index 7) rank 1; 3.0 (index 1) has two greater, rank 2; 3.0 (index 3)
has two greater plus one equal-lower, rank 3, not below 3. So {11, 7, 1}. Hand check, `below`: 4.0 (index 7) rank 0; 3.0
(index 1) rank 1; 3.0 (index 3) one greater plus one equal-lower, rank 2, selected; 2.5 (index 5) three greater, rank 3.
So {7, 1, 3}.

`keep_rows` union case (used once): `keep_rows` is 1.0 at index 0 and 0.0 elsewhere; case `at` gives {0, 1, 7, 11}.

Tiled worked value (Metal cards only): the same twelve scores repeated, row r holds `scores[r % 12]`; `rows = 255` and
`rows = 256`, K = 64 for both (ceil of 63.75 and exactly 64). Hand derivation: 5.0 sits at rows with `r % 12 == 11`
(21 of them below 255 and 256), 4.0 at `r % 12 == 7` (21), together 42; 22 more are needed from the 3.0 rows
(`r % 12` of 1 or 3), lowest index first: copies 0 through 10, rows 1, 3, 13, 15, ... 121, 123. So the selected set is
every row with `r % 12` of 11 or 7, plus every row with `r % 12` of 1 or 3 and `r / 12 <= 10`. The tie test uses 3000
rows (K = 750) and 4097 rows (K = 1025) against a sort-based reference.

The row-count threshold used by the Metal and kernel-count cards is the production default, 256 (FT3.3):
255 rows stays on the plain expression, 256 rows lowers to the kernel.

## cards

### 3.2 `top_fraction_mask`, the rank-count expression, evaluated on CPU against the worked value

- id: FT3.2
- needs: none
- budget: 20 min
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/spec/primitives.rs::elementwise (~line 518)` and `::reduce (~line 541)`: operand notation `"j->ij"` (operand axes to iteration axes) and the out-map of `reduce`;
  - `proxima-tensor/src/spec/primitives.rs::causal_mask_merged (~line 1354)` and `::causal_mask_merged_windowed (~line 1406)`: an `Iota` over `Extent::Symbolic(0)` and a rank-0 `Op::Input` read through `"->s"`; the function this one is placed after;
  - `proxima-tensor/src/op.rs::ScalarOp (~line 76)`: the closed set; `Greater`, `Equal`, `Multiply`, `Add`, `Maximum` all exist, and nothing is added;
  - `proxima-tensor/src/spec/tests.rs::causal_mask_merged_windowed_matches_the_hand_worked_three_by_five_table (~line 11772)`: the `crate::cpu::evaluate_named` call shape and `.get(node)` result.
- change:
  1. `proxima-tensor/src/spec/primitives.rs`: directly after `causal_mask_merged_windowed`, add exactly:
     ```rust
     /// Marks the `keep_count` highest `scores` rows with 1.0 and every other row with 0.0, in the
     /// five-op algebra: a row's rank is the number of rows scoring higher plus the number of
     /// lower-indexed rows scoring equal, and a row is kept when `keep_count > rank`. Equal scores
     /// therefore keep the lower index and exactly `keep_count` rows are kept. `keep_rows`, when
     /// given, is a rank-1 node holding 1.0 on rows kept regardless of rank. `scores` is rank-1
     /// over `Extent::Symbolic(0)`; `keep_count` is a rank-0 input the caller fills with how many
     /// rows to keep. Scores must be finite.
     ///
     /// The nodes are appended in a fixed order (the bind-time fusion recognises the expression by
     /// rebuilding it with this function and comparing), and the cost is O(rows squared), which is
     /// why `omega` lowers it to a selection kernel at `selection.top_fraction_min_rows`.
     pub fn top_fraction_mask(
         program: &mut Vec<Op>,
         scores: NodeId,
         keep_count: NodeId,
         keep_rows: Option<NodeId>,
     ) -> Result<NodeId, TensorError> {
         let iota = op::append(
             program,
             Op::Iota {
                 dtype: DType::Float32,
                 extent: Extent::Symbolic(0),
             },
         );
         let greater = elementwise(
             program,
             DType::Float32,
             ScalarOp::Greater,
             &[(scores, "j->ij"), (scores, "i->ij")],
         )?;
         let equal = elementwise(
             program,
             DType::Float32,
             ScalarOp::Equal,
             &[(scores, "j->ij"), (scores, "i->ij")],
         )?;
         let lower = elementwise(
             program,
             DType::Float32,
             ScalarOp::Greater,
             &[(iota, "i->ij"), (iota, "j->ij")],
         )?;
         let tie = elementwise(
             program,
             DType::Float32,
             ScalarOp::Multiply,
             &[(equal, "ij->ij"), (lower, "ij->ij")],
         )?;
         let hit = elementwise(
             program,
             DType::Float32,
             ScalarOp::Add,
             &[(greater, "ij->ij"), (tie, "ij->ij")],
         )?;
         let rank = reduce(
             program,
             DType::Float32,
             ScalarOp::Add,
             ReduceInit::Zero,
             hit,
             "ij->ij",
             "i->ij",
         )?;
         let selected = elementwise(
             program,
             DType::Float32,
             ScalarOp::Greater,
             &[(keep_count, "->i"), (rank, "i->i")],
         )?;
         match keep_rows {
             Some(keep) => elementwise(
                 program,
                 DType::Float32,
                 ScalarOp::Maximum,
                 &[(selected, "i->i"), (keep, "i->i")],
             ),
             None => Ok(selected),
         }
     }
     ```
     Appended nodes, in order: iota, greater, equal, lower, tie, hit, rank, selected, and the union node when `keep_rows` is given. No `Op` or `ScalarOp` variant is added. `spec/mod.rs` already glob re-exports `primitives`, so no `mod.rs` edit.
  2. `proxima-tensor/src/spec/tests.rs`: at the end of the file add the helpers and three tests below. `input_leaf`, `DType`, `Extent` are already in scope through `use super::*`.
     ```rust
     const TWELVE_SCORES: [f32; 12] = [0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0];

     fn evaluate_twelve_scores(rows: usize, keep_rows_data: Option<&[f32]>) -> Vec<f32> {
         let mut program = Vec::new();
         let scores = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "scores");
         let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
         let keep_rows = keep_rows_data
             .map(|_| input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "keep_rows"));
         let mask = top_fraction_mask(&mut program, scores, keep_count, keep_rows)
             .expect("the top-fraction mask lowers");
         let keep = [rows.div_ceil(4) as f32];
         let mut inputs: Vec<(&str, &[f32])> =
             vec![("scores", &TWELVE_SCORES[..rows]), ("keep_count", &keep)];
         if let Some(data) = keep_rows_data {
             inputs.push(("keep_rows", data));
         }
         let evaluated = crate::cpu::evaluate_named(&program, &[rows as u64], &inputs, &[mask])
             .expect("the top-fraction mask evaluates");
         let (values, _shape) = evaluated.get(mask).expect("the mask node was requested");
         values.to_vec()
     }

     fn selected_rows(mask: &[f32]) -> Vec<usize> {
         mask.iter()
             .enumerate()
             .filter(|(_, value)| **value == 1.0)
             .map(|(row, _)| row)
             .collect()
     }
     ```
- test: add in `proxima-tensor/src/spec/tests.rs` (each asserts the mask length first, because a vacuous mask proves nothing):
  - `top_fraction_cpu_worked_below`: `evaluate_twelve_scores(11, None)` has length 11 and `selected_rows` is exactly `vec![1, 3, 7]`;
  - `top_fraction_cpu_worked_at`: `evaluate_twelve_scores(12, None)` has length 12 and `selected_rows` is exactly `vec![1, 7, 11]`;
  - `rank_select_cpu_keep_rows_union`: `keep` is twelve zeros with `keep[0] = 1.0`; `evaluate_twelve_scores(12, Some(&keep))` has length 12 and `selected_rows` is exactly `vec![0, 1, 7, 11]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_2 cargo nextest run -p proxima-tensor -E 'test(/top_fraction_cpu_worked_/) + test(/rank_select_cpu_keep_rows_union/)'`
- expect: `3 passed`, with `top_fraction_cpu_worked_below`, `top_fraction_cpu_worked_at` and `rank_select_cpu_keep_rows_union` named in the output
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/primitives.rs`, `proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): add rank-count top-fraction mask expression`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `omega/`, `op.rs`, `map.rs`; add a `ScalarOp` or `Op` variant; add a function that computes the keep count; reorder the appended nodes (FT3.6 recognises this exact order).
- gpu: none

### 3.3 the threshold key `selection.top_fraction_min_rows` in omega-runtime.toml

- id: FT3.3
- needs: none
- budget: 20 min
- crate(s): omega (features: default)
- read first:
  - `omega/omega-runtime.toml::[attention_rows]`: how a section is documented (each key carries a comment on its meaning and the cost of raising or lowering it);
  - `omega/build.rs::emit_sizing_consts (~line 219)`: the `resolve_int` plus `require_nonzero` plus `out.push_str(&format!("pub const NAME: u64 = {value};\n"))` shape; the override env var is `OMEGA_<SECTION>_<KEY>` uppercased, so this key's is `OMEGA_SELECTION_TOP_FRACTION_MIN_ROWS`, and `resolve_int` already emits the `cargo:rerun-if-env-changed` line;
  - `omega/src/sized.rs`: the `include!` at ~line 100 (generated constants become public through it) and the module doc bullet list at the top.
- change:
  1. `omega/omega-runtime.toml`: immediately before the `[cached_attention]` section add exactly:
     ```toml
     [selection]
     # smallest row count at which the rank-count top-fraction expression lowers to one selection
     # kernel instead of the O(M^2) plain expression. below it the plain expression runs.
     # 256 is unmeasured (plausible, not proven): at 256 rows the plain expression is 65,536
     # compare-adds. raising it keeps the plain expression longer; lowering it launches the
     # kernel for smaller inputs. must be nonzero. override via OMEGA_SELECTION_TOP_FRACTION_MIN_ROWS.
     top_fraction_min_rows = 256
     ```
  2. `omega/build.rs`: in `emit_sizing_consts`, after the `LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES` block and before `let out_dir`, add:
     ```rust
     let selection_top_fraction_min_rows = require_nonzero(
         "selection.top_fraction_min_rows",
         resolve_int(&root, "selection", "top_fraction_min_rows"),
     );
     out.push_str(&format!(
         "pub const SELECTION_TOP_FRACTION_MIN_ROWS: u64 = {selection_top_fraction_min_rows};\n"
     ));
     ```
  3. `omega/src/sized.rs`: add one bullet to the module doc list, after the `ATTENTION_ROWS_*` bullet: "`SELECTION_TOP_FRACTION_MIN_ROWS` (always compiled) -- the row count at which the rank-count top-fraction expression lowers to one selection kernel; see `omega-runtime.toml`'s `[selection]`." At the end of the file add:
     ```rust
     #[cfg(test)]
     mod tests {
         use super::*;

         #[test]
         fn selection_min_rows_is_the_toml_value() {
             assert_eq!(SELECTION_TOP_FRACTION_MIN_ROWS, 256);
         }
     }
     ```
- test: `selection_min_rows_is_the_toml_value` in `omega/src/sized.rs` asserting `SELECTION_TOP_FRACTION_MIN_ROWS == 256`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_3 cargo nextest run -p omega -E 'test(/selection_min_rows_is_the_toml_value/)'`; then `grep -l "OMEGA_SELECTION_TOP_FRACTION_MIN_ROWS" /private/tmp/cargo_target_ft3_3/debug/build/omega-*/output | wc -l` (the build script's `rerun-if-env-changed` line; the fresh target directory holds exactly one omega build output)
- expect: `1 passed`; the grep pipeline prints `1`
- also green: `cargo clippy -p omega --all-targets`
- stage: `omega/omega-runtime.toml`, `omega/build.rs`, `omega/src/sized.rs`
- commit: `feat(omega): add selection row threshold sizing key`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a runtime reader for the key (FT3.9 is its first reader); touch `proxima-tensor/`.
- gpu: none

### 3.4 `BoundOpKind::TopFractionSelect` and an arm in every exhaustive match

- id: FT3.4
- needs: none
- budget: 20 min. About 85 non-test lines across 17 source files and 4 test files; the admitted file-count exception for this card applies: a new `BoundOpKind` variant forces one copied arm in every exhaustive match, in every crate, in the same commit (see premise 1 and 2). The variant has no feature gate, so a workspace build that unifies any feature set compiles.
- crate(s): proxima-tensor, omega, proxima-model-interop (tests only) (features: default; compile-checks below also use `identity-copy-alias`, `instrument`, `cuda`, `wgpu-backend` and `std` through `--all-features` or the listed flags)
- read first:
  - `proxima-tensor/src/bind/types_layout_boundop.rs::BoundOpKind::MoeTopK (~line 308)` and the six `impl` matches that list it (`name`, `operands`, `all_read_sources`, `element_body`, `split_axis`, `rebase_chunk`): the precedent for a fused kind and every match that must learn the new one;
  - `proxima-tensor/src/cpu/typed_eval.rs::run_typed_program (~line 862)`: the `MoeTopK` arm that returns `TensorError::NotLowerable`, the pattern the new arms copy;
  - `omega/src/msl/signature_tokens_prelude.rs::grid_threads (~line 3)` and `::entry_name (~line 291)`: how `MoeTopK` yields its thread count (`expert_count`, one threadgroup) and `omega_moe_topk_e{expert_count}_k{top_k}`;
  - `omega/src/cuda.rs::emit_cuda_with_policy (~line 221)`: the `CudaUnsupportedOpKind` arm for `MoeTopK`.
- change: add every arm below. Each arm is its own arm or one more alternative on the line after the `MoeTopK` alternative (a new alternative needs no new body); the compiler lists the same set as E0004 errors under the flags in the build table, so add arms until none remain.
  1. `proxima-tensor/src/bind/types_layout_boundop.rs`: add the variant directly after `MoeTopK`:
     ```rust
     /// The bound form of `spec::top_fraction_mask`: a 0/1 f32 mask over `rows` scores, 1.0 on the
     /// `keep_count` highest (lower index wins a tie). `operands` is `[scores, keep_count]`, plus
     /// `keep_rows` last when `has_keep_rows`. Scores must be finite; `-0.0` and `0.0` compare equal.
     /// Built only by the bind-time fusion that collapses the expression; nothing else produces it.
     TopFractionSelect {
         operands: BoundOperands,
         rows: u64,
         has_keep_rows: bool,
     },
     ```
     and one arm in each of the six matches:
     - `name`: `BoundOpKind::TopFractionSelect { .. } => "top_fraction_select",`
     - `operands`: `BoundOpKind::TopFractionSelect { operands, .. } => operands,`
     - `all_read_sources` (the inner `let epilogue = match`): `BoundOpKind::TopFractionSelect { .. } => &[],`
     - `element_body`: `BoundOpKind::TopFractionSelect { .. } => &EMPTY_BODY,`
     - `split_axis`: `BoundOpKind::TopFractionSelect { .. } => None,`
     - `rebase_chunk`: `kind @ BoundOpKind::TopFractionSelect { .. } => kind.clone(),`
  2. `proxima-tensor/src/bind/refresh.rs::refresh_one`: add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the group that returns `RefreshRefusal::MatcherFusedKind`.
  3. `proxima-tensor/src/cpu/quantized_eval.rs::materialize_quantized_weights_read_by_non_primary_operands`: add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the group that yields `&[]`.
  4. `proxima-tensor/src/cpu/typed_eval.rs`: three matches (`run_typed_program` ~line 862, and the two inside `run_widened_program`, ~lines 1039 and 1104). After each `MoeTopK` arm add an arm of the same shape with `BoundOpKind::TopFractionSelect { .. }`, `node: node.node` and the reason `"top-fraction select binding is not wired into the typed executor"` in `run_typed_program`, `"top-fraction select binding is not wired into the widened executor"` in the two widened ones.
  5. `proxima-tensor/src/bind/identity_copy_alias.rs::apply_identity_copy_alias`: add `| BoundOpKind::TopFractionSelect { operands, .. }` after `| BoundOpKind::MoeTopK { operands, .. }` in the group whose body calls `rewrite_read_sources(operands, &alias_of)`.
  6. `proxima-tensor/src/bind/repeat_nodes.rs` (feature `instrument`): in `max_node_id`, add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::Reduce { .. }` in the group whose body is `{}`; in `duplicate_bound_op`, add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the group that ends `=> return None`.
  7. `proxima-tensor/src/cpu/arena.rs::arena_node_kind_label` (feature `instrument`): add `BoundOpKind::TopFractionSelect { .. } => "top_fraction_select",`.
  8. `proxima-tensor/src/cpu/run_node.rs::run_node_into_with_round_sink (~line 130)`: directly after the `BoundOpKind::MoeTopK { .. } => run_moe_topk(..)` arm add
     ```rust
     BoundOpKind::TopFractionSelect { .. } => Err(TensorError::NotLowerable {
         node: resolved.node,
         reason: "top-fraction select has no cpu runner",
     }),
     ```
     Nothing builds this kind yet, so the arm is unreachable; it states the truth about the interpreter at this commit.
  9. `proxima-tensor/examples/scaling.rs::reduce_output_len`: add `| proxima_tensor::BoundOpKind::TopFractionSelect { .. }` after `| proxima_tensor::BoundOpKind::MoeTopK { .. }`.
  10. `proxima-tensor/src/bind/tests.rs`: (a) in the `match &mut bound.kind` whose last arm is `=> continue` (~line 1786), add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }`; (b) the test below.
  11. `proxima-tensor/src/spec/tests.rs` (~line 6695): after the `CachedSoftmaxWeights` arm add `crate::bind::BoundOpKind::TopFractionSelect { .. } => { panic!("this Mistral cached-forward program never binds a TopFractionSelect op") }`.
  12. `omega/src/msl/signature_tokens_prelude.rs`: after `use super::*;` add `pub(super) const SELECTION_THREADGROUP_WIDTH: u64 = 1024;` (Metal's largest threadgroup; one threadgroup runs the whole selection). In `grid_threads` after the `MoeTopK` arm add `BoundOpKind::TopFractionSelect { .. } => SELECTION_THREADGROUP_WIDTH,`. In `entry_name` after the `MoeTopK` arm add:
      ```rust
      BoundOpKind::TopFractionSelect {
          rows,
          has_keep_rows,
          ..
      } => format!(
          "omega_top_fraction_select_r{rows}_k{}",
          u8::from(*has_keep_rows)
      ),
      ```
  13. `omega/src/msl/emit_and_classify.rs::emit_inner (~line 89)`: after the `MoeTopK` arm add
      ```rust
      BoundOpKind::TopFractionSelect { .. } => Err(EmitError::EpilogueNotSupported {
          node: resolved.node,
          reason: "top_fraction_select has no metal renderer",
      }),
      ```
      Nothing builds the kind for Metal yet (the Metal prepare path is routed in a later card, together with the kernel), so no program reaches this arm.
  14. `omega/src/identity.rs::kernel_identity (~line 737)`: after the `MoeTopK` arm add `BoundOpKind::TopFractionSelect { rows, has_keep_rows, .. } => format!("{prefix}_top_fraction_select_r{rows}_k{}", u8::from(*has_keep_rows)),`.
  15. `omega/src/metal/prepare_uniforms_pack.rs`: in `pack_uniforms_byte_len` after the `MoeTopK` arm add `BoundOpKind::TopFractionSelect { .. } => WORD,`; in `pack_uniforms_into` after the `MoeTopK` arm add `BoundOpKind::TopFractionSelect { .. } => { pack_leaf_uniforms(bound, scratch); Ok(()) }`. Rows are baked into the kernel, so the uniform blob is the same one-word leaf the `MoeTopK` arm packs.
  16. `omega/src/metal/dispatch_timed_and_classify.rs::classify_kind` (feature `instrument`): add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the group returning `bound.kind.name()`.
  17. `omega/src/cuda.rs`: in `emit_cuda_with_policy`'s match, after the `MoeTopK` arm add `BoundOpKind::TopFractionSelect { .. } => { return Err(EmitError::CudaUnsupportedOpKind { node: resolved.node, kind: "top_fraction_select" }); }`; in `pack_cuda_uniforms` and in `grid_threads`, add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the existing group (the first returns `CudaUnsupportedOpKind { .. kind: resolved.kind.name() }`, the second yields `extents_product()`).
  18. `omega/src/wgsl.rs`: in `emit_wgsl_with_policy`'s match add `BoundOpKind::TopFractionSelect { .. } => { return Err(EmitError::UnsupportedOpKind { node: resolved.node, kind: "top_fraction_select" }); }` after the `MoeTopK` arm; in `grid_threads` add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }`.
  19. `omega/src/wgpu_driver.rs::pack_uniforms`: add `| BoundOpKind::TopFractionSelect { .. }` after `| BoundOpKind::MoeTopK { .. }` in the group that returns `Ok(pack_leaf_uniforms(bound))`.
  20. `omega/src/msl/tests.rs`: the helper and test below.
  21. `proxima-model-interop/tests/real_qwen35moe_registry_probe.rs`: in the `match &reader.kind` after the `MoeTopK` arm add `proxima_tensor::bind::BoundOpKind::TopFractionSelect { .. } => "TopFractionSelect",`.
- test: add at the end of `proxima-tensor/src/bind/tests.rs`:
  ```rust
  #[test]
  fn top_fraction_select_reports_its_name_and_reads_every_operand() {
      let select = BoundOp {
          node: NodeId(9),
          dtype: DType::Float32,
          extents: alloc::vec![12],
          kind: BoundOpKind::TopFractionSelect {
              operands: alloc::vec![
                  (
                      NodeId(0),
                      Layout {
                          base: 0,
                          strides: SmallVec::from_slice(&[1]),
                      },
                      None,
                  ),
                  (
                      NodeId(1),
                      Layout {
                          base: 0,
                          strides: SmallVec::new(),
                      },
                      None,
                  ),
              ],
              rows: 12,
              has_keep_rows: false,
          },
      };

      assert_eq!(select.kind.name(), "top_fraction_select");
      assert_eq!(select.operands().len(), 2);
      assert_eq!(select.all_read_sources().count(), 2);
  }
  ```
  and at the end of `omega/src/msl/tests.rs` (`Layout`, `BoundOp`, `NodeId`, `DType`, `NumericPolicy`, `entry_name` and `grid_threads` come from the file's imports and `use super::*`):
  ```rust
  fn top_fraction_select_op(rows: u64, has_keep_rows: bool) -> BoundOp {
      let vector = || Layout {
          base: 0,
          strides: vec![1_i64].into(),
      };
      let scalar = || Layout {
          base: 0,
          strides: Vec::<i64>::new().into(),
      };
      let mut operands = vec![(NodeId(0), vector(), None), (NodeId(1), scalar(), None)];
      if has_keep_rows {
          operands.push((NodeId(2), vector(), None));
      }
      BoundOp {
          node: NodeId(9),
          dtype: DType::Float32,
          extents: vec![rows],
          kind: BoundOpKind::TopFractionSelect {
              operands,
              rows,
              has_keep_rows,
          },
      }
  }

  #[test]
  fn rank_select_entry_and_grid_follow_rows_and_keep_flag() {
      let policy = NumericPolicy::default();
      let plain = top_fraction_select_op(12, false);
      let unioned = top_fraction_select_op(12, true);
      assert_eq!(entry_name(&plain, policy), "omega_top_fraction_select_r12_k0");
      assert_eq!(entry_name(&unioned, policy), "omega_top_fraction_select_r12_k1");
      assert_eq!(grid_threads(&plain, &[], policy, false).expect("grid resolves"), 1024);
  }
  ```
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_4 cargo nextest run -p proxima-tensor -p omega -E 'test(/top_fraction_select_reports_its_name_and_reads_every_operand/) + test(/rank_select_entry_and_grid_follow_rows_and_keep_flag/)'`
- expect: `2 passed` (`top_fraction_select_reports_its_name_and_reads_every_operand` and `rank_select_entry_and_grid_follow_rows_and_keep_flag` named); and the build table below, each row printing the two lines `count_build` prints with the stated library count and `1 "success":true`. Run `mkdir -p "$LOG"` first and define `count_build` once (recipe at the top of this file).

  | label | command | crate_name | lib targets |
  |---|---|---|---|
  | tensor_default | `cargo check -p proxima-tensor --all-targets` | proxima_tensor | 2 |
  | tensor_all | `cargo check -p proxima-tensor --all-features --all-targets` | proxima_tensor | 2 |
  | omega_default | `cargo check -p omega --all-targets` | omega | 2 |
  | omega_ext | `cargo check -p omega --all-targets --features instrument,cuda,wgpu-backend` | omega | 2 |
  | omega_linux | `cargo check -p omega --no-default-features --features metal-core --target x86_64-unknown-linux-gnu` | omega | 1 |
  | interop_std | `cargo check -p proxima-model-interop --features std --all-targets` | proxima_model_interop | 2 |
- also green: `cargo clippy -p proxima-tensor --all-targets`, `cargo clippy -p proxima-tensor --all-features --all-targets`, `cargo clippy -p omega --all-targets`
- stage: `proxima-tensor/src/bind/types_layout_boundop.rs`, `proxima-tensor/src/bind/refresh.rs`, `proxima-tensor/src/bind/repeat_nodes.rs`, `proxima-tensor/src/bind/identity_copy_alias.rs`, `proxima-tensor/src/bind/tests.rs`, `proxima-tensor/src/cpu/arena.rs`, `proxima-tensor/src/cpu/quantized_eval.rs`, `proxima-tensor/src/cpu/typed_eval.rs`, `proxima-tensor/src/cpu/run_node.rs`, `proxima-tensor/src/spec/tests.rs`, `proxima-tensor/examples/scaling.rs`, `omega/src/identity.rs`, `omega/src/cuda.rs`, `omega/src/wgsl.rs`, `omega/src/wgpu_driver.rs`, `omega/src/metal/prepare_uniforms_pack.rs`, `omega/src/metal/dispatch_timed_and_classify.rs`, `omega/src/msl/signature_tokens_prelude.rs`, `omega/src/msl/emit_and_classify.rs`, `omega/src/msl/tests.rs`, `proxima-model-interop/tests/real_qwen35moe_registry_probe.rs`
- commit: `feat(tensor): add top-fraction select bound op across all backends`
- done when: the expect line printed, the six build rows printed their counts, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a Cargo feature; add a matcher or a runner; use `unreachable!`, `todo!` or `panic!` in any non-test arm; edit a file not in the stage list.
- gpu: none

### 3.5 CPU runner for the select kind

- id: FT3.5
- needs: FT3.4
- budget: 20 min (about 60 non-test lines)
- crate(s): proxima-tensor (features: default)
- read first:
  - `proxima-tensor/src/cpu/run_node.rs::run_node_into_with_round_sink (~line 130)` and `::run_moe_topk (~line 1258)`: the CPU dispatcher and a fused-kind runner;
  - `proxima-tensor/src/cpu/quantized_eval.rs::buffer_of (~line 2284)`: how a runner reads an operand buffer (reached through `use super::*`);
  - `proxima-tensor/src/cpu/run_node.rs::run_node (~line 4)`: the `#[cfg(test)]` helper that runs one bound op against `Vec<f32>` buffers.
- change:
  1. `proxima-tensor/src/cpu/run_node.rs`: replace the arm FT3.4 added (`BoundOpKind::TopFractionSelect { .. } => Err(TensorError::NotLowerable { .. })`) with `BoundOpKind::TopFractionSelect { .. } => run_top_fraction_select(resolved, buffers, output),`. At the end of the file (before any test module) add:
     ```rust
     pub(super) fn run_top_fraction_select<B: Deref<Target = [f32]> + Sync>(
         resolved: &BoundOp,
         buffers: &[Option<B>],
         output: &mut [f32],
     ) -> Result<(), TensorError> {
         let BoundOpKind::TopFractionSelect {
             operands,
             rows,
             has_keep_rows,
         } = &resolved.kind
         else {
             return Err(TensorError::NotLowerable {
                 node: resolved.node,
                 reason: "top-fraction select runner received another bound operation",
             });
         };
         let mut sources = operands.iter().map(|(node, _, _)| buffer_of(buffers, *node));
         let scores = sources.next().ok_or(TensorError::Empty)??;
         let keep_count = sources.next().ok_or(TensorError::Empty)??;
         let keep_rows = if *has_keep_rows {
             Some(sources.next().ok_or(TensorError::Empty)??)
         } else {
             None
         };
         if scores.len() != *rows as usize || output.len() != scores.len() {
             return Err(TensorError::NotLowerable {
                 node: resolved.node,
                 reason: "top-fraction select operand length differs from its row count",
             });
         }
         let count = keep_count.first().copied().unwrap_or(0.0) as usize;
         rank_select_mask(scores, count, keep_rows, output);
         Ok(())
     }

     fn rank_select_mask(
         scores: &[f32],
         keep_count: usize,
         keep_rows: Option<&[f32]>,
         output: &mut [f32],
     ) {
         let canonical = |value: f32| if value == 0.0 { 0.0 } else { value };
         let mut order: Vec<usize> = (0..scores.len()).collect();
         order.sort_by(|left, right| {
             canonical(scores[*right])
                 .total_cmp(&canonical(scores[*left]))
                 .then(left.cmp(right))
         });
         output.fill(0.0);
         for index in order.into_iter().take(keep_count.min(scores.len())) {
             output[index] = 1.0;
         }
         if let Some(keep) = keep_rows {
             for (slot, flag) in output.iter_mut().zip(keep) {
                 *slot = slot.max(*flag);
             }
         }
     }
     ```
     This is the CPU oracle for the kernel: it allocates and is not a hot path.
- test: add at the end of `proxima-tensor/src/cpu/run_node.rs` (a test module with no `expect` or `unwrap`, so no lint allow is needed):
  ```rust
  #[cfg(test)]
  mod rank_select_tests {
      use crate::Layout;
      use smallvec::SmallVec;

      use super::*;

      const TWELVE_SCORES: [f32; 12] = [0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0];

      fn selected(scores: &[f32], keep_count: usize, keep_rows: Option<&[f32]>) -> Vec<usize> {
          let mut output = vec![0.0_f32; scores.len()];
          rank_select_mask(scores, keep_count, keep_rows, &mut output);
          output
              .iter()
              .enumerate()
              .filter(|(_, value)| **value == 1.0)
              .map(|(index, _)| index)
              .collect()
      }

      #[test]
      fn rank_select_reference_twelve_scores() {
          let mut keep_first = [0.0_f32; 12];
          keep_first[0] = 1.0;
          assert_eq!(selected(&TWELVE_SCORES[..11], 3, None), vec![1, 3, 7]);
          assert_eq!(selected(&TWELVE_SCORES, 3, None), vec![1, 7, 11]);
          assert_eq!(selected(&TWELVE_SCORES, 3, Some(&keep_first)), vec![0, 1, 7, 11]);
          assert_eq!(selected(&TWELVE_SCORES, 0, None), Vec::<usize>::new());
          assert_eq!(selected(&TWELVE_SCORES, 99, None), (0..12).collect::<Vec<_>>());
          assert_eq!(selected(&[1.0, -0.0, 0.0, -1.0], 2, None), vec![0, 1]);
      }

      #[test]
      fn rank_select_runner_reads_scores_then_keep_count() {
          let select = BoundOp {
              node: NodeId(9),
              dtype: DType::Float32,
              extents: vec![12],
              kind: BoundOpKind::TopFractionSelect {
                  operands: vec![
                      (NodeId(0), Layout { base: 0, strides: SmallVec::from_slice(&[1]) }, None),
                      (NodeId(1), Layout { base: 0, strides: SmallVec::new() }, None),
                  ],
                  rows: 12,
                  has_keep_rows: false,
              },
          };
          let mut buffers: Vec<Option<Vec<f32>>> = vec![None; 10];
          buffers[0] = Some(TWELVE_SCORES.to_vec());
          buffers[1] = Some(vec![3.0]);
          let Ok(mask) = run_node(&select, &buffers) else {
              panic!("the top-fraction select runs");
          };
          assert_eq!(mask.len(), 12, "a vacuous mask proves nothing");
          let chosen: Vec<usize> = (0..12).filter(|row| mask[*row] == 1.0).collect();
          assert_eq!(chosen, vec![1, 7, 11]);
      }
  }
  ```
  The last assertion of the first test is the zero rule: `-0.0` and `0.0` tie, so the lower index (1) wins. The second test goes through the interpreter's dispatcher, so the arm and the runner are both exercised.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_5 cargo nextest run -p proxima-tensor -E 'test(/rank_select_reference_twelve_scores/) + test(/rank_select_runner_reads_scores_then_keep_count/)'`
- expect: `2 passed`, with `rank_select_reference_twelve_scores` and `rank_select_runner_reads_scores_then_keep_count` named in the output
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/cpu/run_node.rs`
- commit: `feat(tensor): run top-fraction select on the cpu interpreter`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `omega/`; add a matcher; use `unreachable!`, `todo!` or `panic!` in non-test code.
- gpu: none

### 3.6 find collapsible top-fraction masks at bind time

- id: FT3.6
- needs: FT3.2, FT3.4
- budget: 20 min (about 105 non-test lines in the new file)
- crate(s): proxima-tensor (features: top-fraction-fusion, new; the second validate run adds reduce-epilogue-fusion and cached-attention-streaming)
- read first:
  - `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs::moe_topk_candidates (~line 1064)`: the candidate finder this card follows (find, check for leaks, read layouts from a bind that requested the operands);
  - `proxima-tensor/src/op.rs::Op::dependencies (~line 322)`: operand ids in the order the op addresses them, which the recognizer reads;
  - `proxima-tensor/src/bind/tests.rs::run_resolved (~line 7)`: how a test binds and reads results (used by the next card);
  - `proxima-tensor/src/spec/primitives.rs::top_fraction_mask`: the eight-node shape (nine with `keep_rows`) the recognizer rebuilds and compares against.
- change:
  1. `proxima-tensor/Cargo.toml`: in `[features]`, directly after `moe-topk-fusion = []`, add (a features edit, not a dependency edit):
     ```toml
     # Bind-time collapse of the rank-count top-fraction mask (`spec::top_fraction_mask`) into one
     # `BoundOpKind::TopFractionSelect` when the score row count reaches a caller-supplied threshold.
     # The matcher rebuilds the expression through `spec::top_fraction_mask`, which lives behind
     # `config`. Default-off until the selection kernel has a measured A/B against the plain expression.
     top-fraction-fusion = ["config"]
     ```
  2. new file `proxima-tensor/src/bind/top_fraction_fusion.rs`, exactly:
     ```rust
     use super::*;

     const EXPRESSION_LEN: usize = 8;

     fn rebuilt_expression(
         program: &[Op],
         root: usize,
         keep_rows: Option<NodeId>,
     ) -> Option<(usize, NodeId, NodeId)> {
         let selected = root.checked_sub(usize::from(keep_rows.is_some()))?;
         let first = selected.checked_sub(EXPRESSION_LEN - 1)?;
         let keep_count = *program.get(selected)?.dependencies().first()?;
         let scores = *program.get(first + 1)?.dependencies().first()?;
         let mut reference = alloc::vec![
             Op::Constant {
                 dtype: DType::Float32,
                 shape: Vec::new(),
                 value: 0.0,
             };
             first
         ];
         crate::spec::top_fraction_mask(&mut reference, scores, keep_count, keep_rows).ok()?;
         (reference.get(first..)? == program.get(first..=root)?).then_some((first, scores, keep_count))
     }

     fn expression_leaks(program: &[Op], inside: &[usize], absorbed: &BTreeSet<NodeId>) -> bool {
         program.iter().enumerate().any(|(position, operation)| {
             !inside.contains(&position)
                 && operation
                     .dependencies()
                     .iter()
                     .any(|dependency| absorbed.contains(dependency))
         })
     }

     fn operand_layout(resolved: &[BoundOp], wanted: NodeId) -> Option<Layout> {
         resolved
             .iter()
             .flat_map(BoundOp::all_read_sources)
             .find(|(node, _, _)| *node == wanted)
             .filter(|(_, layout, lookup)| {
                 lookup.is_none() && layout.strides.iter().all(|stride| *stride >= 0)
             })
             .map(|(_, layout, _)| layout.clone())
     }

     /// Every `spec::top_fraction_mask` expression in `program` whose score row count is at least
     /// `min_rows`, as the `BoundOpKind::TopFractionSelect` that could replace it and the node ids
     /// it would absorb. Skipped: an expression with an absorbed node that is a requested output or
     /// is read outside it, and one whose operands `resolved` does not read as plain layouts, so
     /// `resolved` must be bound with every operand requested as an output.
     pub fn top_fraction_candidates(
         program: &[Op],
         shapes: &Shapes,
         resolved: &[BoundOp],
         effective_outputs: &[NodeId],
         min_rows: u64,
     ) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
         let mut candidates = Vec::new();
         for (root, operation) in program.iter().enumerate() {
             let union_keep = match operation {
                 Op::Elementwise {
                     body: ScalarOp::Maximum,
                     operands,
                     ..
                 } => operands.get(1).map(|(node, _)| *node),
                 _ => None,
             };
             let Some((first, scores, keep_count)) = rebuilt_expression(program, root, union_keep)
             else {
                 continue;
             };
             let rows = shapes.of(scores).first().copied().unwrap_or(0);
             let absorbed: BTreeSet<NodeId> = (first..root)
                 .map(|position| NodeId(position as u32))
                 .collect();
             let inside: Vec<usize> = (first..=root).collect();
             let sources = [Some(scores), Some(keep_count), union_keep];
             let operands = sources
                 .into_iter()
                 .flatten()
                 .map(|node| operand_layout(resolved, node).map(|layout| (node, layout, None)))
                 .collect::<Option<Vec<_>>>();
             let Some(operands) = operands.filter(|_| {
                 shapes.of(scores).len() == 1
                     && rows >= min_rows
                     && !expression_leaks(program, &inside, &absorbed)
                     && !absorbed.iter().any(|node| effective_outputs.contains(node))
             }) else {
                 continue;
             };
             let kind = BoundOpKind::TopFractionSelect {
                 operands,
                 rows,
                 has_keep_rows: union_keep.is_some(),
             };
             let fused = BoundOp {
                 node: NodeId(root as u32),
                 dtype: DType::Float32,
                 extents: alloc::vec![rows],
                 kind,
             };
             candidates.push((fused, absorbed));
         }
         let covered: BTreeSet<NodeId> = candidates
             .iter()
             .flat_map(|(_, absorbed)| absorbed.iter().copied())
             .collect();
         candidates.retain(|(fused, _)| !covered.contains(&fused.node));
         candidates
     }
     ```
     Why the recognizer rebuilds and compares instead of walking operands: it then accepts exactly what `top_fraction_mask` emits, including every index map, and nothing near it. It reads only two operand ids from the program (the keep count and the scores) and the union operand; the comparison checks the rest.
  3. `proxima-tensor/src/bind/mod.rs`: after `mod refresh;` add `#[cfg(feature = "top-fraction-fusion")] mod top_fraction_fusion;`; after `pub use refresh::{..};` add `#[cfg(feature = "top-fraction-fusion")] pub use top_fraction_fusion::top_fraction_candidates;` (each attribute on its own line).
- test: add at the end of `proxima-tensor/src/bind/tests.rs`:
  ```rust
  #[cfg(feature = "top-fraction-fusion")]
  mod top_fraction_fusion_tests {
      use super::*;
      use crate::spec::{input_leaf, top_fraction_mask};

      const TWELVE_ROWS: u64 = 12;

      struct Fixture {
          program: Vec<Op>,
          scores: NodeId,
          keep_count: NodeId,
          keep_rows: Option<NodeId>,
          mask: NodeId,
      }

      fn fixture(with_keep_rows: bool) -> Fixture {
          let mut program = Vec::new();
          let scores = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "scores");
          let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
          let keep_rows = with_keep_rows.then(|| {
              input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "keep_rows")
          });
          let mask = top_fraction_mask(&mut program, scores, keep_count, keep_rows).expect("mask lowers");
          Fixture { program, scores, keep_count, keep_rows, mask }
      }

      fn candidates_at(
          fixture: &Fixture,
          rows: u64,
          extra_outputs: &[NodeId],
      ) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
          let shapes = shape::infer(&fixture.program, &[rows]).expect("program infers");
          let mut outputs = alloc::vec![fixture.mask, fixture.scores, fixture.keep_count];
          outputs.extend(fixture.keep_rows);
          let resolved = bind_plain(&fixture.program, &shapes, &outputs, NumericPolicy::bit_exact())
              .expect("plain binds");
          let requested: Vec<NodeId> = [fixture.mask].into_iter().chain(extra_outputs.iter().copied()).collect();
          top_fraction_candidates(&fixture.program, &shapes, &resolved, &requested, TWELVE_ROWS)
      }

      #[test]
      fn rank_select_candidate_fires_at_threshold() {
          let fixture = fixture(false);
          let candidates = candidates_at(&fixture, 12, &[]);
          assert_eq!(candidates.len(), 1);
          let (fused, absorbed) = &candidates[0];
          assert_eq!(fused.node, fixture.mask);
          assert_eq!(fused.kind.name(), "top_fraction_select");
          assert_eq!(absorbed.len(), 7, "the iota through the rank nodes");
      }

      #[test]
      fn rank_select_candidate_carries_keep_rows_operand() {
          let fixture = fixture(true);
          let candidates = candidates_at(&fixture, 12, &[]);
          assert_eq!(candidates.len(), 1, "the mask-only prefix is covered by the union candidate");
          let (fused, absorbed) = &candidates[0];
          let BoundOpKind::TopFractionSelect { operands, rows, has_keep_rows } = &fused.kind else {
              panic!("the candidate must be a top-fraction select, got {}", fused.kind.name());
          };
          assert_eq!((*rows, *has_keep_rows, operands.len()), (12, true, 3));
          let sources: Vec<NodeId> = operands.iter().map(|(node, _, _)| *node).collect();
          assert_eq!(sources, [fixture.scores, fixture.keep_count, fixture.keep_rows.expect("union fixture")]);
          assert_eq!(absorbed.len(), 8, "the iota through the selected nodes");
      }

      #[test]
      fn rank_select_candidate_declines_below_threshold() {
          let fixture = fixture(false);
          assert!(candidates_at(&fixture, 11, &[]).is_empty());
      }

      #[test]
      fn rank_select_candidate_declines_when_an_intermediate_is_requested() {
          let fixture = fixture(false);
          let rank = NodeId(fixture.mask.0 - 1);
          assert!(candidates_at(&fixture, 12, &[rank]).is_empty());
      }
  }
  ```
  Node counts: the program is the scores input (node 0), the keep-count input (node 1), then the expression from node 2; without `keep_rows` the mask is node 9 and the absorbed set is nodes 2 to 8 (7 nodes); with `keep_rows` (node 2, expression from node 3) the union node is the root and the absorbed set is the 8 nodes before it.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_6 cargo nextest run -p proxima-tensor --features top-fraction-fusion -E 'test(/rank_select_candidate_/)'`, then the same command with `--features top-fraction-fusion,reduce-epilogue-fusion,cached-attention-streaming` (the second run's filter is `test(/rank_select_candidate_/)` too)
- expect: `4 passed` from each of the two runs, with `rank_select_candidate_fires_at_threshold`, `rank_select_candidate_carries_keep_rows_operand`, `rank_select_candidate_declines_below_threshold` and `rank_select_candidate_declines_when_an_intermediate_is_requested` named; and the build table below (define `count_build` once, recipe at the top of this file; `mkdir -p "$LOG"` first), each row printing its library count and `1 "success":true`:

  | label | command | crate_name | lib targets |
  |---|---|---|---|
  | tensor_feature | `cargo check -p proxima-tensor --features top-fraction-fusion --all-targets` | proxima_tensor | 2 |
  | tensor_all | `cargo check -p proxima-tensor --all-features --all-targets` | proxima_tensor | 2 |
  | tensor_default | `cargo check -p proxima-tensor --all-targets` | proxima_tensor | 2 |
  | tensor_noalloc | `cargo check -p proxima-tensor --no-default-features --features alloc` | proxima_tensor | 1 |
- also green: `cargo clippy -p proxima-tensor --features top-fraction-fusion --all-targets`; `cargo clippy -p proxima-tensor --all-targets` (feature off)
- stage: `proxima-tensor/Cargo.toml`, `proxima-tensor/src/bind/top_fraction_fusion.rs`, `proxima-tensor/src/bind/mod.rs`, `proxima-tensor/src/bind/tests.rs`
- commit: `feat(tensor): find collapsible top-fraction masks at bind time`
- done when: both expect lines printed, the four build rows printed their counts, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `bind_with_fusion`'s signature or body; touch `omega/`; reuse `moe_*` helpers (they are gated on another feature); add `bind_with_top_fraction` (the next card).
- gpu: none

### 3.7 `bind_with_top_fraction`, the bind entry point

- id: FT3.7
- needs: FT3.5, FT3.6
- budget: 20 min (about 72 non-test lines)
- crate(s): proxima-tensor (features: top-fraction-fusion; the second validate run adds reduce-epilogue-fusion and cached-attention-streaming)
- read first:
  - `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs::apply_moe_topk_fusion (~line 1135)`: the two-pass rewrite (find candidates, widen the planning outputs to the operand sources, rebind with `bind_plain`, find again, rewrite) this card copies;
  - `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs::bind_with_fusion (~line 62)`: its signature stays unchanged;
  - `proxima-tensor/src/bind/tests.rs::run_resolved (~line 7)` and `::fused_moe_topk_matches_unfused_routing_over_random_scores_and_exact_ties (~line 6346)`: how a test binds fused and unfused and runs both on the CPU interpreter;
  - `proxima-tensor/src/bind/top_fraction_fusion.rs::top_fraction_candidates` (FT3.6): the finder this card calls twice.
- change:
  1. `proxima-tensor/src/bind/top_fraction_fusion.rs`: append exactly:
     ```rust
     fn apply_top_fraction_fusion(
         built: Vec<BoundOp>,
         program: &[Op],
         shapes: &Shapes,
         outputs: &[NodeId],
         min_rows: u64,
     ) -> Result<Vec<BoundOp>, TensorError> {
         let initial = top_fraction_candidates(program, shapes, &built, outputs, min_rows);
         if initial.is_empty() {
             return Ok(built);
         }
         let mut planning_outputs = outputs.to_vec();
         if planning_outputs.is_empty() {
             let root = program
                 .len()
                 .checked_sub(1)
                 .map(|position| NodeId(position as u32))
                 .ok_or(TensorError::Empty)?;
             planning_outputs.push(root);
         }
         for (fused, _) in &initial {
             for (source, _, _) in fused.operands() {
                 if !planning_outputs.contains(source) {
                     planning_outputs.push(*source);
                 }
             }
         }
         let rebuilt = bind_plain(program, shapes, &planning_outputs, NumericPolicy::bit_exact())?;
         let candidates = top_fraction_candidates(program, shapes, &rebuilt, outputs, min_rows);
         if candidates.is_empty() {
             return Ok(built);
         }
         let fused_by_node = candidates
             .iter()
             .map(|(fused, _)| (fused.node, fused))
             .collect::<BTreeMap<_, _>>();
         let absorbed = candidates
             .iter()
             .flat_map(|(_, absorbed)| absorbed.iter().copied())
             .collect::<BTreeSet<_>>();
         let mut rewritten = Vec::with_capacity(rebuilt.len());
         for bound in rebuilt {
             if let Some(fused) = fused_by_node.get(&bound.node) {
                 rewritten.push((*fused).clone());
             } else if !absorbed.contains(&bound.node) {
                 rewritten.push(bound);
             }
         }
         Ok(rewritten)
     }

     /// [`bind_with_fusion`], then collapses every `spec::top_fraction_mask` expression whose score
     /// row count is at least `min_rows` into one `BoundOpKind::TopFractionSelect`. The threshold is
     /// an argument because it lives in `omega`'s sizing file, which this crate cannot see. Expressions
     /// below the threshold, and expressions whose intermediate nodes are requested outputs or are read
     /// outside the expression, are left exactly as `bind_with_fusion` binds them.
     pub fn bind_with_top_fraction(
         program: &[Op],
         shapes: &Shapes,
         outputs: &[NodeId],
         fuse_cached_attention: bool,
         numeric_policy: NumericPolicy,
         min_rows: u64,
     ) -> Result<Vec<BoundOp>, TensorError> {
         let built = bind_with_fusion(
             program,
             shapes,
             outputs,
             fuse_cached_attention,
             numeric_policy,
         )?;
         apply_top_fraction_fusion(built, program, shapes, outputs, min_rows)
     }
     ```
  2. `proxima-tensor/src/bind/mod.rs`: change the line FT3.6 added to `pub use top_fraction_fusion::{bind_with_top_fraction, top_fraction_candidates};` (its attribute line stays). `proxima-tensor/src/lib.rs`: after the `pub use bind::{RepeatNodeRefusal, apply_repeat_nodes};` pair add the two attribute lines `#[cfg(feature = "top-fraction-fusion")]` and `#[cfg(any(feature = "std", feature = "alloc"))]` and then `pub use bind::bind_with_top_fraction;`.
- test: in `proxima-tensor/src/bind/tests.rs`, inside the module `top_fraction_fusion_tests` FT3.6 added, replace its closing brace with the helpers and four tests below (the module keeps FT3.6's `TWELVE_ROWS`, `Fixture`, `fixture` and `candidates_at`):
  ```rust
      const TWELVE_SCORES: [f32; 12] = [0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0];

      fn kinds(bound: &[BoundOp]) -> Vec<&'static str> {
          bound.iter().map(|bound_op| bound_op.kind.name()).collect()
      }

      fn fused_count(bound: &[BoundOp]) -> usize {
          bound
              .iter()
              .filter(|bound_op| bound_op.kind.name() == "top_fraction_select")
              .count()
      }

      fn bind_both(fixture: &Fixture, rows: usize) -> (Vec<BoundOp>, Vec<BoundOp>) {
          let shapes = shape::infer(&fixture.program, &[rows as u64]).expect("program infers");
          let outputs = [fixture.mask];
          let plain = bind_with_fusion(&fixture.program, &shapes, &outputs, true, NumericPolicy::bit_exact())
              .expect("plain binds");
          let fused = bind_with_top_fraction(
              &fixture.program, &shapes, &outputs, true, NumericPolicy::bit_exact(), TWELVE_ROWS,
          )
          .expect("fused binds");
          (plain, fused)
      }

      fn selected_rows(fixture: &Fixture, bound: &[BoundOp], rows: usize, keep_first: bool) -> Vec<usize> {
          let keep = rows.div_ceil(4) as f32;
          let mut inputs = alloc::vec![
              (fixture.scores, TWELVE_SCORES[..rows].to_vec()),
              (fixture.keep_count, alloc::vec![keep]),
          ];
          if let Some(node) = fixture.keep_rows {
              let mut flags = alloc::vec![0.0_f32; rows];
              flags[0] = f32::from(u8::from(keep_first));
              inputs.push((node, flags));
          }
          let buffers = run_resolved(fixture.program.len(), bound, inputs);
          let mask = buffers[fixture.mask.0 as usize].as_ref().expect("mask resolves");
          assert_eq!(mask.len(), rows, "a vacuous mask proves nothing");
          mask.iter().enumerate().filter(|(_, value)| **value == 1.0).map(|(row, _)| row).collect()
      }

      #[test]
      fn rank_select_fusion_fires_at_threshold() {
          let fixture = fixture(false);
          let (plain, fused) = bind_both(&fixture, 12);
          assert_eq!(fused_count(&fused), 1, "kinds {:?}", kinds(&fused));
          assert!(fused.len() < plain.len(), "fused {:?} plain {:?}", kinds(&fused), kinds(&plain));
      }

      #[test]
      fn rank_select_fusion_declines_below_threshold() {
          let fixture = fixture(false);
          let (plain, fused) = bind_both(&fixture, 11);
          assert_eq!(fused_count(&fused), 0);
          assert_eq!(kinds(&fused), kinds(&plain));
      }

      #[test]
      fn rank_select_fusion_declines_when_an_intermediate_is_requested() {
          let fixture = fixture(false);
          let shapes = shape::infer(&fixture.program, &[12]).expect("program infers");
          let rank = NodeId(fixture.mask.0 - 1);
          let bound = bind_with_top_fraction(
              &fixture.program,
              &shapes,
              &[fixture.mask, rank],
              true,
              NumericPolicy::bit_exact(),
              TWELVE_ROWS,
          )
          .expect("binds");
          assert_eq!(fused_count(&bound), 0, "kinds {:?}", kinds(&bound));
      }

      #[test]
      fn rank_select_fused_equals_plain_twelve_scores() {
          let plain_fixture = fixture(false);
          let union_fixture = fixture(true);
          for (rows, expected) in [(11_usize, alloc::vec![1, 3, 7]), (12, alloc::vec![1, 7, 11])] {
              let (plain, fused) = bind_both(&plain_fixture, rows);
              assert_eq!(selected_rows(&plain_fixture, &plain, rows, false), expected, "plain rows {rows}");
              assert_eq!(selected_rows(&plain_fixture, &fused, rows, false), expected, "fused rows {rows}");
          }
          let (plain, fused) = bind_both(&union_fixture, 12);
          assert_eq!(selected_rows(&union_fixture, &plain, 12, true), alloc::vec![0, 1, 7, 11]);
          assert_eq!(selected_rows(&union_fixture, &fused, 12, true), alloc::vec![0, 1, 7, 11]);
          assert_eq!(fused_count(&fused), 1);
      }
  }
  ```
  Measured at this commit: with the plain bind, rows 12 binds 3 ops and the fused bind 1 op (feature `reduce-epilogue-fusion` off); the assertions are written as relations so they hold with the epilogue feature on as well.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_7 cargo nextest run -p proxima-tensor --features top-fraction-fusion -E 'test(/rank_select_fus/)'`, then the same command with `--features top-fraction-fusion,reduce-epilogue-fusion,cached-attention-streaming`
- expect: `4 passed` from each of the two runs, with `rank_select_fusion_fires_at_threshold`, `rank_select_fusion_declines_below_threshold`, `rank_select_fusion_declines_when_an_intermediate_is_requested` and `rank_select_fused_equals_plain_twelve_scores` named
- also green: `cargo clippy -p proxima-tensor --features top-fraction-fusion --all-targets`; `cargo clippy -p proxima-tensor --all-targets` (feature off); `cargo clippy -p proxima-tensor --all-features --all-targets`
- stage: `proxima-tensor/src/bind/top_fraction_fusion.rs`, `proxima-tensor/src/bind/mod.rs`, `proxima-tensor/src/lib.rs`, `proxima-tensor/src/bind/tests.rs`
- commit: `feat(tensor): bind top-fraction masks above a row threshold as one op`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `bind_with_fusion`'s signature or body; touch `omega/`; reuse `moe_*` helpers (they are gated on another feature).
- gpu: none

### 3.8 selection kernel: `selection_render.rs`

- id: FT3.8
- needs: FT3.4
- budget: 20 min (118 non-test lines in the new file)
- crate(s): omega (features: default; the module and every item it adds are always compiled, like every other renderer)
- read first:
  - `omega/src/msl/signature_tokens_prelude.rs::render_moe_topk (~line 1259)`: a single-threadgroup selection kernel; the signature and buffer-slot layout this card follows (operands from buffer 0, then the output, then `Uniforms`);
  - `omega/src/msl/emit_and_classify.rs::emit_inner (~line 89)`: the arm FT3.4 left returning `EpilogueNotSupported`;
  - `omega/src/msl/mod.rs` (~lines 119 and 157): where `mod` lines and `use` lines for the render files sit;
  - `omega/src/msl/emit_and_classify.rs::bindings (~line 2112)`: confirms the order `[operands..., output, uniforms]`.
- change:
  1. new file `omega/src/msl/selection_render.rs`, exactly:
     ```rust
     use super::*;

     const RADIX_BITS: u32 = 8;

     pub(super) fn render_top_fraction_select(
         resolved: &BoundOp,
         entry: &str,
     ) -> Result<String, EmitError> {
         let BoundOpKind::TopFractionSelect {
             operands,
             rows,
             has_keep_rows,
         } = &resolved.kind
         else {
             return Err(EmitError::RenderKindMismatch {
                 node: resolved.node,
                 expected: "top_fraction_select",
                 found: resolved.kind.name(),
             });
         };
         let output_buffer = operands.len();
         let uniforms_buffer = output_buffer + 1;
         let chunk = rows.div_ceil(SELECTION_THREADGROUP_WIDTH);

         let mut source = String::new();
         preamble(&mut source, false);
         source.push_str("struct Uniforms { long unused; };\n\n");
         source.push_str("static inline uint order_key(float value) {\n");
         source.push_str("    value = (value == 0.0f) ? 0.0f : value;\n");
         source.push_str("    const uint bits = as_type<uint>(value);\n");
         source.push_str("    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);\n}\n\n");
         source.push_str(&format!(
             "kernel void {entry}(device const float* scores [[buffer(0)]], device const float* keep_count [[buffer(1)]],\n"
         ));
         if *has_keep_rows {
             source.push_str("    device const float* keep_rows [[buffer(2)]],\n");
         }
         source.push_str(&format!(
             "    device float* out [[buffer({output_buffer})]], constant Uniforms& u [[buffer({uniforms_buffer})]],\n\
              \tuint tid [[thread_position_in_threadgroup]],\n\
              \tuint sg_id [[simdgroup_index_in_threadgroup]],\n\
              \tuint sg_lane [[thread_index_in_simdgroup]]) {{\n"
         ));
         source.push_str(&format!(
             "    constexpr uint ROWS = {rows}u; constexpr uint THREADS = {SELECTION_THREADGROUP_WIDTH}u; constexpr uint CHUNK = {chunk}u;\n\
              \t(void)u;\n\
              \tconst uint keep = min(uint(keep_count[0]), ROWS);\n"
         ));
         source.push_str(
             "    threadgroup atomic_uint histogram[256];\n\
              \tthreadgroup uint partials[32];\n\
              \tthreadgroup uint shared_prefix;\n\
              \tthreadgroup uint shared_remaining;\n\
              \tif (tid == 0u) { shared_prefix = 0u; shared_remaining = keep; }\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n",
         );
         for pass in 0..u32::BITS / RADIX_BITS {
             source.push_str(&radix_pass(pass));
         }
         source.push_str(&tie_walk(*has_keep_rows));
         source.push_str("}\n");
         Ok(source)
     }

     fn radix_pass(pass: u32) -> String {
         let shift = u32::BITS - RADIX_BITS * (pass + 1);
         let high = shift + RADIX_BITS;
         format!(
             "    // pass {pass}\n\
              \tif (tid < 256u) {{ atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed); }}\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
              \tfor (uint row = tid; row < ROWS; row += THREADS) {{\n\
              \t\tconst uint key = order_key(scores[row]);\n\
              \t\tif ((ulong(key) >> {high}u) == (ulong(shared_prefix) >> {high}u)) {{ atomic_fetch_add_explicit(&histogram[(key >> {shift}u) & 0xFFu], 1u, memory_order_relaxed); }}\n\
              \t}}\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
              \tif (tid == 0u) {{\n\
              \t\tuint need = shared_remaining;\n\
              \t\tuint digit = 255u;\n\
              \t\twhile (need > atomic_load_explicit(&histogram[digit], memory_order_relaxed)) {{\n\
              \t\t\tneed -= atomic_load_explicit(&histogram[digit], memory_order_relaxed);\n\
              \t\t\tdigit -= 1u;\n\
              \t\t}}\n\
              \t\tshared_prefix |= digit << {shift}u;\n\
              \t\tshared_remaining = need;\n\
              \t}}\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n"
         )
     }

     fn tie_walk(has_keep_rows: bool) -> String {
         let store = if has_keep_rows {
             "out[row] = max(selected ? 1.0f : 0.0f, keep_rows[row]);"
         } else {
             "out[row] = selected ? 1.0f : 0.0f;"
         };
         format!(
             "    const uint threshold = shared_prefix;\n\
              \tconst uint ties_to_take = shared_remaining;\n\
              \tconst uint start = min(tid * CHUNK, ROWS);\n\
              \tconst uint end = min(start + CHUNK, ROWS);\n\
              \tuint equal_count = 0u;\n\
              \tfor (uint row = start; row < end; row++) {{ if (order_key(scores[row]) == threshold) {{ equal_count++; }} }}\n\
              \tconst uint within = simd_prefix_exclusive_sum(equal_count);\n\
              \tconst uint simd_total = simd_sum(equal_count);\n\
              \tif (sg_lane == 0u) {{ partials[sg_id] = simd_total; }}\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
              \tif (tid == 0u) {{\n\
              \t\tuint running = 0u;\n\
              \t\tfor (uint index = 0u; index < 32u; index++) {{ const uint total = partials[index]; partials[index] = running; running += total; }}\n\
              \t}}\n\
              \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
              \tuint seen = 0u;\n\
              \tconst uint ties_before = partials[sg_id] + within;\n\
              \tfor (uint row = start; row < end; row++) {{\n\
              \t\tconst uint key = order_key(scores[row]);\n\
              \t\tconst bool tied = key == threshold;\n\
              \t\tconst bool selected = key > threshold || (tied && ties_before + seen < ties_to_take);\n\
              \t\tif (tied) {{ seen++; }}\n\
              \t\t{store}\n\
              \t}}\n"
         )
     }
     ```
     Kernel contract, for reading the code: one threadgroup of 1024 threads; `order_key` maps a finite float to an unsigned key that orders like the float (larger float, larger key; zero is canonicalised so `-0.0` and `0.0` tie); four passes of an 8-bit radix select fix the key of the K-th largest row into `shared_prefix`, leaving `shared_remaining` = how many rows equal to that key must still be taken (the prefix test widens both sides to 64 bits so the first pass, with nothing fixed yet, compares zero with zero); each thread then owns a contiguous chunk of rows, counts its keys equal to the threshold, a simdgroup prefix plus a 32-entry partials array turns that into "ties in earlier chunks", and a row equal to the threshold is taken only while that running tie count is below `shared_remaining` (lowest index wins). A keep count of zero needs no branch: the passes then settle on the all-ones key, no key exceeds it and no tie is taken, so only the `keep_rows` flags are written (premise 10). Scratch is threadgroup memory declared in the kernel; there is no per-dispatch allocation.
  2. `omega/src/msl/mod.rs`: after `mod cached_softmax_weights_render;` add `mod selection_render;`; after `use cached_softmax_weights_render::render_cached_softmax_weights;` add `use selection_render::render_top_fraction_select;`.
  3. `omega/src/msl/emit_and_classify.rs::emit_inner`: replace the arm FT3.4 added (`BoundOpKind::TopFractionSelect { .. } => Err(EmitError::EpilogueNotSupported { .. })`) with `BoundOpKind::TopFractionSelect { .. } => render_top_fraction_select(resolved, &entry),`.
- test: append to `omega/src/msl/tests.rs` (`top_fraction_select_op` is the helper FT3.4 added):
  ```rust
  #[test]
  fn rank_select_render_emits_kernel_entry_bindings_and_grid() {
      let policy = NumericPolicy::default();
      let kernel = emit(&top_fraction_select_op(12, false), &PackedOperands::new(), policy)
          .expect("selection kernel emits");
      assert_eq!(kernel.entry, "omega_top_fraction_select_r12_k0");
      assert_eq!(kernel.bindings.len(), 4);
      assert_eq!(kernel.grid.threads, 1024);
  }

  #[test]
  fn rank_select_render_declares_four_passes() {
      let policy = NumericPolicy::default();
      let kernel = emit(&top_fraction_select_op(12, false), &PackedOperands::new(), policy)
          .expect("selection kernel emits");
      assert!(kernel.source.contains("constexpr uint ROWS = 12u"));
      assert_eq!(kernel.source.matches("simd_prefix_exclusive_sum").count(), 1);
      assert_eq!(
          kernel.source.matches("atomic_fetch_add_explicit(&histogram").count(),
          4
      );
      assert!(!kernel.source.contains("keep_rows"));
      let unioned = emit(&top_fraction_select_op(12, true), &PackedOperands::new(), policy)
          .expect("union kernel emits");
      assert!(unioned.source.contains("device const float* keep_rows [[buffer(2)]]"));
      assert!(unioned.source.contains("device float* out [[buffer(3)]]"));
  }
  ```
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_8 cargo nextest run -p omega -E 'test(/rank_select_render_/)'`
- expect: `2 passed` (`rank_select_render_emits_kernel_entry_bindings_and_grid` and `rank_select_render_declares_four_passes` named)
- also green: `cargo clippy -p omega --all-targets`
- stage: `omega/src/msl/selection_render.rs`, `omega/src/msl/mod.rs`, `omega/src/msl/emit_and_classify.rs`, `omega/src/msl/tests.rs`
- commit: `feat(omega): render radix-select kernel for top-fraction select`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: compile the MSL on a device in this card (the next card runs it); change any other render file; route the Metal prepare path (the next card).
- gpu: none

### 3.9 Metal lowers the mask to the selection kernel at the threshold

- id: FT3.9
- needs: FT3.2, FT3.3, FT3.7, FT3.8
- budget: 20 min (about 20 non-test lines: a feature line, two imports and one bind call; the rest is the test file)
- crate(s): omega (features: metal, instrument, top-fraction-fusion)
- read first:
  - `omega/src/metal/prepare_uniforms_pack.rs` (the `bind_with_fusion(` call, ~line 155): the production bind every plan goes through;
  - `omega/src/metal/mod.rs` (the `use proxima_tensor::{ .. }` list, ~line 223);
  - `omega/tests/moe_topk_metal_tie_parity.rs`: how a test plans and executes a program on Metal (`omega::plan_named`, `omega::execute_plan_named`) and compares with the CPU evaluator;
  - `omega/tests/reduction_literal_ab.rs` (~line 424): the one place `omega::pipeline_cache_keys()` is read to prove which kernels ran (feature `instrument`; the cache is per thread, nextest runs each test in its own process).
- change:
  1. `omega/Cargo.toml`: after `moe-topk-fusion = [..]` add (a features edit, not a dependency edit):
     ```toml
     # Default-off passthrough to `proxima-tensor/top-fraction-fusion`: the Metal prepare path binds with
     # `bind_with_top_fraction`, so a rank-count top-fraction mask at or above
     # `selection.top_fraction_min_rows` lowers to the selection kernel instead of the plain expression.
     top-fraction-fusion = ["proxima-tensor/top-fraction-fusion"]
     ```
  2. `omega/src/metal/mod.rs`: in the `use proxima_tensor::{ .. }` list remove `bind_with_fusion`, and add directly after the list, each attribute on its own line:
     ```rust
     #[cfg(not(feature = "top-fraction-fusion"))]
     use proxima_tensor::bind_with_fusion;
     #[cfg(feature = "top-fraction-fusion")]
     use proxima_tensor::bind_with_top_fraction;
     ```
     (this keeps both builds free of an unused import).
  3. `omega/src/metal/prepare_uniforms_pack.rs` (~line 155): put `#[cfg(not(feature = "top-fraction-fusion"))]` on the existing `let mut resolved = bind_with_fusion(..)?;` and add after it
     ```rust
     #[cfg(feature = "top-fraction-fusion")]
     let mut resolved = bind_with_top_fraction(
         program,
         &shapes,
         &effective_outputs,
         fuse_cached_attention,
         numeric_policy,
         crate::sized::SELECTION_TOP_FRACTION_MIN_ROWS,
     )?;
     ```
  4. new file `omega/tests/top_fraction_select_metal.rs`, exactly:
     ```rust
     #![cfg(all(
         target_os = "macos",
         feature = "metal",
         feature = "instrument",
         feature = "top-fraction-fusion"
     ))]
     #![allow(clippy::unwrap_used, clippy::expect_used)]

     use proxima_tensor::spec::{input_leaf, top_fraction_mask};
     use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

     const TWELVE_SCORES: [f32; 12] = [0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0];

     struct Fixture {
         program: Vec<Op>,
         mask: NodeId,
     }

     fn fixture(with_keep_rows: bool) -> Fixture {
         let mut program = Vec::new();
         let scores = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "scores");
         let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
         let keep_rows = with_keep_rows
             .then(|| input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "keep_rows"));
         let mask = top_fraction_mask(&mut program, scores, keep_count, keep_rows).expect("mask lowers");
         Fixture { program, mask }
     }

     fn tiled_scores(rows: usize) -> Vec<f32> {
         (0..rows).map(|row| TWELVE_SCORES[row % 12]).collect()
     }

     fn quarter_of_rows(rows: usize) -> usize {
         rows.div_ceil(4)
     }

     fn hand_derived_set(rows: usize) -> Vec<usize> {
         (0..rows)
             .filter(|row| match row % 12 {
                 11 | 7 => true,
                 1 | 3 => row / 12 <= 10,
                 _ => false,
             })
             .collect()
     }

     fn selected(values: &[f32]) -> Vec<usize> {
         values.iter().enumerate().filter(|(_, value)| **value == 1.0).map(|(row, _)| row).collect()
     }

     fn run_metal(rows: usize, keep_count: usize, keep_rows_data: Option<&[f32]>) -> Vec<f32> {
         let fixture = fixture(keep_rows_data.is_some());
         let scores = tiled_scores(rows);
         let keep = [keep_count as f32];
         let mut named = vec![
             ("scores", QuantizedBlock::Float32(&scores)),
             ("keep_count", QuantizedBlock::Float32(&keep)),
         ];
         if let Some(data) = keep_rows_data {
             named.push(("keep_rows", QuantizedBlock::Float32(data)));
         }
         let symbols = [rows as u64];
         let plan = omega::plan_named(&fixture.program, &symbols, &named, &[fixture.mask], NumericPolicy::default())
             .expect("metal plan builds");
         let evaluated = omega::execute_plan_named(&plan, &named).expect("metal evaluates");
         let (values, _shape) = evaluated.get(fixture.mask).expect("mask requested");
         values.to_vec()
     }

     fn run_cpu_plain(rows: usize, keep_count: usize, keep_rows_data: Option<&[f32]>) -> Vec<f32> {
         let fixture = fixture(keep_rows_data.is_some());
         let scores = tiled_scores(rows);
         let keep = [keep_count as f32];
         let mut inputs: Vec<(&str, &[f32])> = vec![("scores", &scores), ("keep_count", &keep)];
         if let Some(data) = keep_rows_data {
             inputs.push(("keep_rows", data));
         }
         let evaluated = proxima_tensor::cpu::evaluate_named(&fixture.program, &[rows as u64], &inputs, &[fixture.mask])
             .expect("cpu evaluates");
         let (values, _shape) = evaluated.get(fixture.mask).expect("mask requested");
         values.to_vec()
     }

     fn ranked_reference_set(rows: usize) -> Vec<usize> {
         let scores = tiled_scores(rows);
         let mut order: Vec<usize> = (0..rows).collect();
         order.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]).then(left.cmp(right)));
         let mut chosen: Vec<usize> = order.into_iter().take(quarter_of_rows(rows)).collect();
         chosen.sort_unstable();
         chosen
     }
     ```
- test: add in `omega/tests/top_fraction_select_metal.rs`:
  ```rust
  #[test]
  fn selection_kernel_matches_hand_derived_set_at_threshold() {
      let rows = 256;
      let metal = run_metal(rows, quarter_of_rows(rows), None);
      assert_eq!(metal.len(), rows);
      assert_eq!(selected(&metal), hand_derived_set(rows));
      assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), None));
      let keys = omega::pipeline_cache_keys();
      assert!(
          keys.iter().any(|key| key.contains("top_fraction_select_r256_k0")),
          "the selection kernel must be what ran, got {keys:?}"
      );
  }

  #[test]
  fn plain_expression_matches_hand_derived_set_below_threshold() {
      let rows = 255;
      let metal = run_metal(rows, quarter_of_rows(rows), None);
      assert_eq!(metal.len(), rows);
      assert_eq!(selected(&metal), hand_derived_set(rows));
      assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), None));
      let keys = omega::pipeline_cache_keys();
      assert!(
          keys.iter().all(|key| !key.contains("top_fraction_select")),
          "below the threshold only the plain expression may run, got {keys:?}"
      );
  }

  #[test]
  fn selection_kernel_unions_keep_rows() {
      let rows = 256;
      let mut keep = vec![0.0_f32; rows];
      keep[0] = 1.0;
      keep[255] = 1.0;
      let metal = run_metal(rows, quarter_of_rows(rows), Some(&keep));
      let mut expected = hand_derived_set(rows);
      expected.extend([0, 255]);
      expected.sort_unstable();
      assert_eq!(selected(&metal), expected);
      assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), Some(&keep)));
      assert!(
          omega::pipeline_cache_keys().iter().any(|key| key.contains("top_fraction_select_r256_k1")),
          "the selection kernel must be what ran with the keep flags operand"
      );
  }

  #[test]
  fn selection_kernel_breaks_ties_by_index_across_thread_chunks() {
      for rows in [3000_usize, 4097] {
          let metal = run_metal(rows, quarter_of_rows(rows), None);
          assert_eq!(metal.len(), rows);
          assert_eq!(selected(&metal), ranked_reference_set(rows), "rows {rows}");
          let expected_key = format!("top_fraction_select_r{rows}_k0");
          assert!(
              omega::pipeline_cache_keys().iter().any(|key| key.contains(&expected_key)),
              "the selection kernel must be what ran at {rows} rows"
          );
      }
  }

  #[test]
  fn selection_kernel_keeps_only_flagged_rows_when_the_count_is_zero() {
      let rows = 256;
      let mut keep = vec![0.0_f32; rows];
      keep[5] = 1.0;
      let metal = run_metal(rows, 0, Some(&keep));
      assert_eq!(metal.len(), rows);
      assert_eq!(selected(&metal), vec![5]);
      assert_eq!(metal, run_cpu_plain(rows, 0, Some(&keep)));
      assert!(selected(&run_metal(rows, 0, None)).is_empty());
      let keys = omega::pipeline_cache_keys();
      for expected_key in ["top_fraction_select_r256_k0", "top_fraction_select_r256_k1"] {
          assert!(
              keys.iter().any(|key| key.contains(expected_key)),
              "the selection kernel must be what ran, missing {expected_key} in {keys:?}"
          );
      }
  }
  ```
  The first test is the hand-derived check on the kernel and compares the kernel with the plain expression run on the CPU; the second is the same set on the plain expression running on Metal, one row below the threshold; the third checks the union; the fourth uses 3000 and 4097 rows so each thread owns more than one row (the per-thread tie-count path), against a sort-based reference; the fifth runs a keep count of zero with and without flags.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_9 cargo nextest run -p omega --features metal,instrument,top-fraction-fusion -j 1 --test top_fraction_select_metal`
- expect: `5 passed` (`selection_kernel_matches_hand_derived_set_at_threshold`, `plain_expression_matches_hand_derived_set_below_threshold`, `selection_kernel_unions_keep_rows`, `selection_kernel_breaks_ties_by_index_across_thread_chunks` and `selection_kernel_keeps_only_flagged_rows_when_the_count_is_zero` named); and the build table below (define `count_build` once, recipe at the top of this file; `mkdir -p "$LOG"` first), each row printing its library count and `1 "success":true`:

  | label | command | crate_name | lib targets |
  |---|---|---|---|
  | omega_default | `cargo check -p omega --all-targets` | omega | 2 |
  | omega_feature | `cargo check -p omega --features top-fraction-fusion --all-targets` | omega | 2 |
  | omega_ext_feature | `cargo check -p omega --all-targets --features instrument,cuda,wgpu-backend,top-fraction-fusion` | omega | 2 |
  | omega_linux_off | `cargo check -p omega --no-default-features --features metal-core --target x86_64-unknown-linux-gnu` | omega | 1 |
  | omega_linux_on | `cargo check -p omega --no-default-features --features metal-core,top-fraction-fusion --target x86_64-unknown-linux-gnu` | omega | 1 |
- also green: `cargo clippy -p omega --features metal,instrument,top-fraction-fusion --all-targets`; `cargo clippy -p omega --all-targets` (feature off)
- stage: `omega/Cargo.toml`, `omega/src/metal/mod.rs`, `omega/src/metal/prepare_uniforms_pack.rs`, `omega/tests/top_fraction_select_metal.rs`
- commit: `feat(omega): lower top-fraction masks to the selection kernel on metal`
- done when: the expect line printed, the five build rows printed their counts, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a model or checkpoint to the test; run it concurrently with any other Metal or model process; edit any renderer.
- gpu: one run, waiting for a quiet box (CARDS.md machine safety: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` must print nothing first). The test loads no model; it plans and runs one small program five times.

### 3.10 kernel-count tests at the configured threshold (CPU only)

- id: FT3.10
- needs: FT3.2, FT3.3, FT3.7, FT3.8
- budget: 20 min
- crate(s): omega (features: top-fraction-fusion)
- read first:
  - `omega/tests/lowering_census.rs` (head): how a test binds a program and emits each `BoundOp` through the CPU-side emit with no device, then inspects what came out;
  - `omega/src/msl/kernel_types_identity.rs::Kernel (~line 184)`: `entry`, `bindings`, `grid`;
  - `omega/src/sized.rs::SELECTION_TOP_FRACTION_MIN_ROWS` (FT3.3);
  - `proxima-tensor/src/bind/top_fraction_fusion.rs::bind_with_top_fraction` (FT3.7).
- change:
  1. new file `omega/tests/top_fraction_kernel_count.rs`, exactly:
     ```rust
     #![cfg(feature = "top-fraction-fusion")]
     #![allow(clippy::unwrap_used, clippy::expect_used)]

     use omega::sized::SELECTION_TOP_FRACTION_MIN_ROWS;
     use omega::{Binding, PackedOperands, emit};
     use proxima_tensor::spec::{input_leaf, top_fraction_mask};
     use proxima_tensor::{
         BoundOp, DType, Extent, NumericPolicy, bind_with_fusion, bind_with_top_fraction, infer,
     };

     fn bound_lists(rows: u64) -> (Vec<BoundOp>, Vec<BoundOp>) {
         let mut program = Vec::new();
         let scores = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "scores");
         let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
         let mask = top_fraction_mask(&mut program, scores, keep_count, None).expect("mask lowers");
         let shapes = infer(&program, &[rows]).expect("program infers");
         let policy = NumericPolicy::default();
         let plain = bind_with_fusion(&program, &shapes, &[mask], true, policy).expect("plain binds");
         let fused = bind_with_top_fraction(
             &program,
             &shapes,
             &[mask],
             true,
             policy,
             SELECTION_TOP_FRACTION_MIN_ROWS,
         )
         .expect("fused binds");
         (plain, fused)
     }

     fn selection_kernels(bound: &[BoundOp]) -> Vec<omega::Kernel> {
         bound
             .iter()
             .map(|bound_op| {
                 emit(bound_op, &PackedOperands::new(), NumericPolicy::default()).expect("bound op emits")
             })
             .filter(|kernel| kernel.entry.starts_with("omega_top_fraction_select_"))
             .collect()
     }
     ```
- test: add in `omega/tests/top_fraction_kernel_count.rs`:
  ```rust
  #[test]
  fn no_selection_kernel_below_the_configured_row_threshold() {
      let (plain, fused) = bound_lists(SELECTION_TOP_FRACTION_MIN_ROWS - 1);
      assert_eq!(selection_kernels(&fused).len(), 0);
      let kinds = |bound: &[BoundOp]| {
          bound
              .iter()
              .map(|bound_op| bound_op.kind.name())
              .collect::<Vec<_>>()
      };
      assert_eq!(kinds(&fused), kinds(&plain));
  }

  #[test]
  fn one_selection_kernel_at_the_configured_row_threshold() {
      let rows = SELECTION_TOP_FRACTION_MIN_ROWS;
      let (plain, fused) = bound_lists(rows);
      let kernels = selection_kernels(&fused);
      assert_eq!(kernels.len(), 1);
      assert_eq!(kernels[0].entry, format!("omega_top_fraction_select_r{rows}_k0"));
      assert_eq!(kernels[0].grid.threads, 1024);
      assert_eq!(kernels[0].bindings.len(), 4);
      assert!(matches!(kernels[0].bindings[2], Binding::Output(_)));
      assert!(fused.len() < plain.len());
  }
  ```
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft3_10 cargo nextest run -p omega --features top-fraction-fusion --test top_fraction_kernel_count`
- expect: `2 passed` (`no_selection_kernel_below_the_configured_row_threshold` and `one_selection_kernel_at_the_configured_row_threshold` named)
- also green: `cargo clippy -p omega --features top-fraction-fusion --all-targets`
- stage: `omega/tests/top_fraction_kernel_count.rs`
- commit: `test(omega): count selection kernels around the row threshold`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the threshold default; edit anything under `src/`.
- gpu: none (emission only; no device is opened)

## spec drift

Each item: where the older spec text says one thing and the cards do another, with the reason.

1. The spec names the expression `top_fraction_mask(program, scores, fraction, min_rows, keep_rows)` and, in its "Top fraction (slice 3)" override, a host function `top_fraction_keep_count(fraction_milli, min_keep, rows)`. The cards use `top_fraction_mask(program, scores, keep_count, keep_rows)` and no keep-count function. Reasons: the algebra has no ceiling and the row count is symbolic, so the caller computes K per call and feeds it as a rank-0 input (the `cached_len` precedent); and how many rows to keep is the policy of the technique that consumes the selection (a read rule, a recompute rule), so it is not library code. Slices that consume the expression define their own count rule where they use it.
2. `min_rows` is overloaded in the spec: the key `selection.top_fraction_min_rows` is the lowering threshold; the old signature's `min_rows` is the minimum rows kept. The cards keep the key and drop the other, with the keep-count function.
3. The spec places a "radix-select recognizer" in `omega/src/msl/selection_render.rs` and "classify at the threshold" in `emit_and_classify.rs`. A recognizer over a multi-node subgraph cannot live in a renderer: the repo's precedent (`match_moe_topk`) is a matcher in `proxima-tensor/src/bind/` producing a new `BoundOpKind`. The cards add `BoundOpKind::TopFractionSelect`, a recognizer in `bind/top_fraction_fusion.rs` and `bind_with_top_fraction(.., min_rows)`; `selection_render.rs` holds only the kernel.
4. The threshold lives in omega's TOML and the matcher in proxima-tensor, which cannot read omega constants, so it is passed as an argument. `bind_with_fusion` keeps its signature. The production bind is omega's Metal prepare path, which FT3.9 routes through `bind_with_top_fraction`, so a decode step planned by `omega::plan_named` reaches the kernel with no interop change; the two interop calls to `bind_with_fusion` are diagnostic probes and stay.
5. CPU tests for the expression are in `proxima-tensor/src/spec/tests.rs`, beside every other `spec` primitive test; the fused-kind reference function and its tests are in `cpu/run_node.rs`.
6. Tie rule: the spec's expression says "rank is a Reduce Add over Greater" with no tie rule, which selects more than K rows on ties and differs from any kernel. The cards add the lower-index-wins term (`Equal` times `Greater(i, j)`), and the worked value has a tie at the boundary. Zero and NaN: see premise 8.
7. The spec's boundary pair is "M = threshold minus one and M = threshold" using the production default. The expression and fusion cards use the literal threshold 12 because the worked value has 12 scores; the Metal and kernel-count cards use the production default 256 with the tiled worked value (255 and 256 rows). The default 256 is unmeasured (status: plausible).
8. The spec's variant is "cfg-gated". The variant is ungated, following the repo's `MoeTopK` and `GatedDeltaNet` precedent, because a gated variant leaves a feature-unified workspace build red between the card that adds the variant and the card that adds omega's arms; the feature gates the matcher and the Metal routing instead (premise 1).

## slice exit

- Commands, in order, each at the stated count (all on one checkout with every card landed):
  1. `cargo nextest run -p proxima-tensor --features top-fraction-fusion -E 'test(/top_fraction_cpu_worked_/) + test(/rank_select_/) + test(/top_fraction_select_reports_its_name_and_reads_every_operand/)'` prints `14 passed` (2 expression worked, 1 union expression, 1 CPU reference, 1 CPU runner, 4 recognizer, 4 fusion, 1 kind introspection).
  2. `cargo nextest run -p omega --features top-fraction-fusion -E 'test(/rank_select_/) + test(/selection_min_rows_is_the_toml_value/)'` prints `4 passed` (1 entry and grid, 2 render, 1 threshold).
  3. `cargo nextest run -p omega --features top-fraction-fusion --test top_fraction_kernel_count` prints `2 passed`.
  4. `cargo nextest run -p omega --features metal,instrument,top-fraction-fusion -j 1 --test top_fraction_select_metal` prints `5 passed` (one GPU run, quiet box).
- Builds, counted with the recipe at the top of this file; every row prints its library count and `1 "success":true`:

  | label | command | crate_name | lib targets |
  |---|---|---|---|
  | tensor_default | `cargo check -p proxima-tensor --all-targets` | proxima_tensor | 2 |
  | tensor_feature | `cargo check -p proxima-tensor --features top-fraction-fusion --all-targets` | proxima_tensor | 2 |
  | tensor_all | `cargo check -p proxima-tensor --all-features --all-targets` | proxima_tensor | 2 |
  | tensor_noalloc | `cargo check -p proxima-tensor --no-default-features --features alloc` | proxima_tensor | 1 |
  | omega_default | `cargo check -p omega --all-targets` | omega | 2 |
  | omega_feature | `cargo check -p omega --features top-fraction-fusion --all-targets` | omega | 2 |
  | omega_ext_feature | `cargo check -p omega --all-targets --features instrument,cuda,wgpu-backend,top-fraction-fusion` | omega | 2 |
  | omega_linux_off | `cargo check -p omega --no-default-features --features metal-core --target x86_64-unknown-linux-gnu` | omega | 1 |
  | omega_linux_on | `cargo check -p omega --no-default-features --features metal-core,top-fraction-fusion --target x86_64-unknown-linux-gnu` | omega | 1 |
  | interop_std | `cargo check -p proxima-model-interop --features std --all-targets` | proxima_model_interop | 2 |

  The two omega all-targets rows with the feature on also build `omega/tests/top_fraction_select_metal.rs` and `omega/tests/top_fraction_kernel_count.rs`; their library counts stay 2 because the count names the library target only.
- Not cuttable (and why): FT3.4 touches 21 files. Adding an enum variant forces every exhaustive `match` over `BoundOpKind` in every crate to gain an arm in the same commit (rustc), and the arms are one-line copies of the neighbouring `MoeTopK` arm or typed refusals; the variant is ungated so that no feature set of a workspace build can turn it on under a crate that lacks the arms. The list is in the card, and the exception for it is the one CARDS.md already admits for this card.
- Transitional state: between FT3.4 and FT3.5 the CPU interpreter refuses the new kind with a typed `NotLowerable` (nothing builds the kind: the recognizer arrives in FT3.6); between FT3.4 and FT3.8 `emit_inner` refuses it with `EpilogueNotSupported` (nothing routes a program to it until FT3.9). Neither window has a commit where a program binds the kind and then fails at run time: the matcher is behind a default-off feature, and the Metal routing lands in the commit that has the kernel.
- Decide-later items with owners: the default 256 is plausible, not measured; the measurement belongs to the slice that first builds the expression at long context, which can change the TOML value without touching code. NaN scores are outside the contract and the kernel, reference and expression disagree on them; a caller that can produce NaN must clean scores before this selection.
