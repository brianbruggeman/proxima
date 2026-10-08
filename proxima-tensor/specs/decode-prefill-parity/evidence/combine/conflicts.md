# combine: integration conflicts (git am --3way, order r9 r3 r2 r4 r6 r5 r7)

Tip after integration: 5ae7d822 (48 commits on top of 7d693c1a; r6 0004 skipped). Base before integration: 7d693c1a. origin/main: ab69ec03.

## clean (no conflict, no manual edit)
r9 0001-0009 (9), r3 0001 (auto-merged omega/src/msl/elementwise_reduce_core.rs, tests.rs via 3-way fallback), r3 0002-0005,
r2 0001-0004, 0006-0013 (r2 0005 conflicted, below), r4 0001 (auto-merge decode_arms.rs), 0002, 0003, 0006, 0008,
r6 0001-0003, 0006, 0007, r5 0001-0002, r7 0001-0002, 0004, 0005.

## conflicts and resolutions (both intents kept)

1. r2 0005 "run every selected expert of a projection in one grouped gemm" vs r3 (all described codecs on tiled/grouped gemm)
   - omega/src/msl/expert_grouped_gemm.rs push_grouped_stage_pointers: r3 side kept `let token_axis = geometry.token_axis`,
     r2 side kept Q8_0 `block_bytes/block_elements` locals. Both are dead after the merge (r2 moved the token coordinate to
     `geometry.token_axes` / push_token_coordinates, r3 moved the weight decode to `decode.call`). Both lines deleted; omega compiles.
   - omega/src/msl/emit_and_classify.rs classify_tiled_gemm: kept r3's codec admission via the tiled_decode table and
     CodecSwitchedOff, plus r2's `let mut gathered` (route_flat is set later). r2's `NotQ4K` gate not kept (r3 removed that
     rejection).
   - classify_expert_gather: dropped r2's `codec != Codec::Q8_0 -> GatheredCodecNotAdmitted` (the variant is gone from the
     merged enum; r3's intent is every described codec is admitted for the grouped path), kept r2's
     `ExpertGather { slot, route_flat: true }` and `expert_route_stride`.
   - omega/src/msl/tests.rs (r3 test): `ExpertGather { slot: 0 }` -> `ExpertGather { slot: 0, route_flat: true }` (field added by r2).
     Folded into the r2 0005 commit so the commit builds.
2. r4 0004 "bind a multi-output op's extra outputs from the plan arena" vs r2 (MoeTopK at any token count, stacked outputs)
   - `git am` could not build a fake ancestor (r4's blobs are not in this repo); fetched r4/tree objects temporarily and used
     `git apply --3way`. Conflict in omega/src/metal/arena_encode_dispatch_finish.rs encode_op: kept r4's single loop over
     `extra_output_nodes(bound)`; replaced r4's hard-coded `(node, 1)` MoeTopK counts with
     `bound.kind.moe_topk_extra_outputs(token_count)` (r2's sizes: token_count per route/weight, token_count*top_k per stacked).
3. r4 0005 "ignore compile-only nextest runs in the decode_arms peer gate" vs main's decode_arms (no ollama_is_a_peer parameter)
   - proxima-model-interop/examples/decode_arms.rs running_peers: kept the no-parameter signature and the IGNORE_OLLAMA test,
     added r4's `--no-run` filter and first-token program match.
4. r4 0007 "resolve fold and recurrent-state buffers once per plan step" vs the merged extra_output_nodes
   - r4 made extra_output_nodes an allocation-free iterator; r2 made MoeTopK counts come from a Vec-returning method.
     Resolution: added `BoundOpKind::moe_topk_extra_outputs_iter` in proxima-tensor/src/bind/types_layout_boundop.rs
     (`moe_topk_extra_outputs` now collects it, same nodes/counts/order), and extra_output_nodes chains it with the softmax
     array. Folded into the r4 0007 commit (e97ea0b8).
5. r6 0003 "record the r6 norm fold diagnosis" vs r9 0004/0009 spec sections: both appended to
   proxima-tensor/specs/decode-prefill-parity/SPEC.md at the same position. Kept both sections (r9 section, closing fence added,
   then r6 section).
6. r6 0004 "bind a multi-output op's extra outputs from the plan arena": SKIPPED. Same source hunks as r4 0004
   (diff of the two patch files differs only in the subject count and r4's extra omega/tests/step_buffer_allocations.rs, which r4
   0004 already applied). Recorded as already applied.
7. r6 0005 "twin-output elementwise kind": `git am` same missing-blob failure; used `git apply --3way` after a temporary fetch of
   r6/tree. Conflict in arena_encode_dispatch_finish.rs extra_output_nodes: r6 added an `ElementwiseTwin` arm to the Vec-returning
   form; ported it to the iterator form (`twin` Option chained last).
8. r7 0003 "default prefill chunk to llama.cpp's 512" vs r5 0002: both added constants after DEFAULT_GPU_LAYERS in
   proxima-model-interop/src/serving.rs and imports in serving_settings.rs. Kept DEFAULT_RESIDENT_PREFILL_PLAN_BYTES, DEFAULT_BATCH_SIZE
   and DEFAULT_UBATCH_SIZE.

Temporary refs refs/remotes/r4tree/main and refs/remotes/r6tree/main were deleted after use.

9. Process note: `git add omega proxima-tensor` during r4 0007 staged the owner's untracked decode-as-data/ and evidence/slice3/ into
   e97ea0b8. The 16 commits e97ea0b8..b336fa12 were rebuilt with git commit-tree (same author/committer/date/message, trees minus those
   9 files), main moved with update-ref from b336fa12 to 5ae7d822, then `git reset -q` (mixed). The untracked trees are byte-identical to
   the copy in combine/others_untracked_backup (diff -r clean). Old tip b336fa12 recorded in combine/rewrite_newtip.txt. Nothing was pushed.
   Commit ids in the list above (e97ea0b8 onward) are the OLD ids; new tip is 5ae7d822.

## fixes and reverts made at the integrated tip (each its own commit)

Every item below was found by a gate run named in gates.md; base ab69ec03 was run on the same command where a failure needed attribution
(source export in combine/src_base, build dir /private/tmp/cargo_target_base).

- b18a6537 REVERT of r2 0013 "make stacked moe experts the metal default" (9aa0833c). With `moe-stacked-experts` in the interop `metal`
  set, `external_expert_paging` (2 tests) failed with `NotLowerable { node: 132, "quantized matmul batch shape does not evenly divide by
  its packed weight rows" }`; both tests pass at base. Cause read in `proxima-tensor/src/cpu/run_reduce_scan.rs:436-439`: the CPU quantized
  reduce requires `activation.len() == leading_total * k`, and a stacked gate/up projection reads one activation row for `selected`
  gathered experts (stride 0 on the selected axis), so the check fails. Not a one-line local fix (the per-position loop at :588 indexes
  `activation[position * k..]`). The stacked strategy stays available behind its own feature; only the default-on membership was reverted.
  Consequence: the granite digest is unchanged from base (the 792-op drop seen before the revert was this default).
- df9e38a1 `dead_resolved_nodes` treated an op as dead when its primary `.node` was unread, so a fused top-k whose consumers read only its
  stacked route/weight outputs was dropped and read as zeros; the gemma4 synthetic parity tests then ran on NaN logits and passed or failed
  on `f32::max` dropping NaN (6868398c makes the diff reject non-finite values). Found with `--features moe-stacked-experts`.
- 8eca5477 r9's `256*l+d@256` view set the multi-term flag on the whole per-layer-input activation chain, so
  `composed_packed_product_activation` read the activation as the packed side and `blk.0.proj` fused into a 3-operand body that the CPU
  packed reduce rejects ("packed reduce admits only W.a with a materialized a"); the flag now passes only through operands that fuse
  (`retires.contains` + identity map + held). base passes `gemma4_e2b_tiled_gemm_defaults_vs_all_off_full_logit_vector_diff`, the integrated
  tip before this commit did not.
- 5cf74898 r3's tiled f16 projection tests compared to an f32 CPU oracle at 1e-3 on outputs near 1800 (one f32 ulp is 1.2e-4); the
  measured error against an f64 dot is 0.0043 for the tiled kernel and 0.0009 for the CPU f32 evaluator. The tiled arm is now held to the
  f32 accumulation bound n * 2^-24 * sum|a*w|; a control test shows the bound rejects a +10.0 corruption.
- b8b036b1 r4's warm round-batched allocation test counted the non-placements executor, which allocates every output per call by design;
  it now runs `execute_plan_named_with_placements`, the executor the serving path uses.
- 0780fb6c a bind census `match` lacked the `ElementwiseTwin` arm (r6) under the workspace feature set.
- a2edd376 `gemma4_attention_chain_census` pinned 1661; the tip prints 1559 = 1661 - 34 * 3, and the kind histogram (reduce 903->869,
  elementwise 488->454, constant 262->228) shows the 34 folded per-layer-input norms.
- 5b63d61c perf(omega): extra outputs of a multi-output op (twin, top-k, softmax weights) took dedicated arena slots for the plan's
  lifetime. With `PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION=1` the E2B 971-token peak footprint measured 680.6 MB against 844.9 MB with the
  twin pass (combine/bench/runD); extras now take and return slots through the same free list as primary outputs.
- digest fixture commit was swept into 5b63d61c by a staged-but-uncommitted add (the first commit attempt failed the 72-char subject check); split with commit-tree into 04c67331 (fixture) and 85f1f23e (perf); tree at tip unchanged (git diff --stat empty).
