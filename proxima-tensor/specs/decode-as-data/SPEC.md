---
status: admitted
---

# decode as data

## problem

Problem: on main 0ac67c63 proxima runs one multi-row decode technique (an n-gram chain
verify, gemma4 only), and this spec makes chain, tree, lookahead and bidirectional-block
decoding into serving-config data on one lowered program, measured as llama.cpp f1ea20621
per-row id parity on 3 checkpoints x 3 prompts (AC3, AC5) with zero technique-named types
(AC9).

Owner, 2026-10-04: "I want to explore these: speculative decoding, mtp, medusa/eagle,
lookahead decoding, and diffusion in proxima. what I want is for proxima to structurally
enable these within its programmable fsm before lowering into a precise execution path."

Owner, same thread: "we're really just moving around different forms of multiplexing and
not realistically creating a streaming. it's a kappa architecture."

Today each technique is planned as its own mechanism. The four audited sub-specs each
add an `Architecture` impl or a capability method:

- `speculative-draft-mtp` R2 and R9 (`gemma4-assistant`);
- `speculative-draft-eagle3` R1 (`Eagle3Arch`);
- `speculative-draft-dflash` R2 (`DflashArch`);
- `speculative-draft-dspark` (extends `DflashDraftProgram`).

`architecture-as-data` R2 (admitted 2026-10-04) forbids every one of those. Medusa,
lookahead and diffusion have no spec at all.

## the frame: one log, one pass shape, many multiplexings

The committed token sequence is an append-only log, and the KV cache is a materialized fold
over it. Prefill replays the log. Decode appends one entry. Every technique on the list
puts more than one *uncommitted* row into a single pass over the log head, then commits
some of them. The techniques differ in only four places, and each place is a pipe:

| place | form | in, out |
|---|---|---|
| propose | transform | (history, last logits, taps) to candidate rows |
| shape | transform | candidate rows to (`mask_pos[s]`, `seq_member[s, g]`, rope positions) |
| accept | transform | (candidate rows, logits per row) to accepted row indices, or "refine" |
| commit | sink | accepted row indices to cache (gather; prefix truncate is the degenerate case) |

| technique | propose | shape | accept | commit |
|---|---|---|---|---|
| n-gram / prompt lookup (live) | n-gram drafter | chain of k+1 | sample-and-match | prefix |
| draft model | second descriptor, own FSM instance | chain | sample-and-match | prefix |
| MTP | extra readout roots on the target program (nextn block) | chain | sample-and-match | prefix |
| Medusa | k head readouts, cartesian top-k | tree | sample-and-match (lossless) or typical (lossy, config) | path gather |
| EAGLE / EAGLE3 | draft descriptor fed by target hidden-state taps | tree | sample-and-match | path gather |
| lookahead (Jacobi) | previous pass's window logits + n-gram pool | llama's W+G+1 sequence layout | n-gram verify | path gather |
| DFlash / DSpark | masked block on a draft descriptor | block | sample-and-match on the target | prefix |
| diffusion LM (LLaDA, Dream) | masked block, the target itself | block, bidirectional | confidence remask, T refine steps | whole block after the final step |

This is kappa, not lambda: no separate batch program and speed program. It is the same
graph with a symbolic row axis. The only part that is not append-only is the uncommitted
tail, the staging window that `accept` rewrites. Streaming exists at the token-output edge.
Inside a pass it is row multiplexing. The one place genuine pipelining enters is running
propose(t+1) concurrently with verify(t), as two pipe stages. That is listed under out of
scope.

## context

- FSM: `proxima-model-interop/src/serving_fsm.rs`.
- Live decode loop: `proxima-model-interop/src/generate/decode.rs`.
- Position inputs and KV cache: `proxima-model-interop/src/generate/residency_caches.rs`.
- Lowering: `proxima-tensor/src/spec/descriptor.rs` (`build_forward` :538).
- Masks:
  - `proxima-tensor/src/spec/primitives.rs`;
  - `proxima-tensor/src/spec/lfm2_single_range_cached.rs`.
- Sibling specs:
  - `proxima-tensor/specs/architecture-as-data/`;
  - `proxima-tensor/specs/speculative-decode-llama-parity/`;
  - `proxima-tensor/specs/speculative-draft-{mtp,eagle3,dflash,dspark}/`.
- Oracle source: llama.cpp f1ea20621 at `/Users/brianbruggeman/repos/others/llama.cpp`.
- Fixture home: `proxima-model-interop/tests/fixtures/llama-parity/`, the
  `architecture-as-data` slice 0 home.

## evidence (read 2026-10-04)

| # | artifact | what it shows |
|---|---|---|
| E1 | `proxima-model-interop/src/serving_fsm.rs:48-74` | `ServingState` has Prefill, Decode, Verify, Accept, Rollback and Done. `accept` (:186-237) scores a chain prefix only, by `take_while` over the draft. |
| E2 | `serving_fsm.rs:36-38` | `#![allow(dead_code)]`, not driven by generate. Wiring it is `architecture-as-data` slice 10b. |
| E3 | `proxima-model-interop/src/generate/residency_caches.rs:1206-1250` | RoPE cos/sin are host-built per row: `position = start_position + offset`. Positions are already data; only the host generator assumes contiguity. |
| E4 | `proxima-tensor/src/spec/lfm2_single_range_cached.rs:416-424, 1360` | The TwoRange engine scores the new block with the block-local, iota-based `causal_mask_windowed`. The cached block uses `causal_mask_cached_windowed` (:328), which handles padding and window. |
| E5 | `proxima-tensor/src/spec/primitives.rs` `causal_mask_merged` | Causality is `key_index > iota(s) + cached_len`. The iota is the only graph node that encodes "rows are contiguous and causal". |
| E6 | `residency_caches.rs:155` `LayerCache::truncate` | Commit today is prefix truncation. Ring layers rewind only by `cached_len`. |
| E7 | `proxima-tensor/src/spec/descriptor.rs:538` `build_forward(descriptor, last_row_only)` | All-rows logits is a per-call bool. There is no readout list (no hidden-state taps, no extra heads) on main. |
| E8 | llama.cpp f1ea20621 `examples/lookahead/lookahead.cpp:123-126, 156-158, 210`; `src/llama-kv-cache.cpp:1592, 1651, 1666-1670` | The incumbent expresses lookahead visibility as per-token sequence-id sets: seq 0 is the input, seqs 1..W are Jacobi rows, seqs W+1..W+G are n-grams. A key cell is visible iff it holds the query's first seq id and `pos_key <= pos_query`. |
| E9 | llama.cpp f1ea20621 `examples/diffusion/diffusion.cpp:122`, `diffusion.h:7-12` | Diffusion turns causality off (`llama_set_causal_attn(ctx, false)`). It has 5 remask algorithms; Dream and LLaDA are archs (`src/llama-arch.h:135-138`). |
| E10 | llama.cpp f1ea20621 `examples/` | `lookahead`, `diffusion`, `speculative`, `lookup` exist. No Medusa example, so Medusa has no llama oracle. |

## the one graph change: visibility as data (E4, E5, E8)

    visible(q, k) = in_seq(q, k) AND mask_pos[k] <= mask_pos[q]
    in_seq(q, k)  = sum over g of seq_primary[q, g] * seq_member[k, g] > 0

This is llama.cpp's own rule, transcribed:
- the key cell must hold the query's *first* sequence id (`src/llama-kv-cache.cpp:1592`,
  `seq_id = ubatch->seq_id[i][0]`, and `:1651`, `cells.seq_has(j, seq_id)`);
- then, under causal attention, the cell is masked when `p0 > p1` (`:1666-1670`).

It is "share the query's primary sequence", not "share any sequence". The looser rule
over-admits lookahead keys.

The inputs are `Op::Input` leaves over the new-block axis:
- `mask_pos: [s]`;
- `seq_member: [s, G]`, 0/1 membership for each key row;
- `seq_primary: [s, G]`, a one-hot of each query row's first sequence id.

The formula uses only `Elementwise {Multiply, Greater}` and a `Reduce` sum. No new `Op` or
`ScalarOp` variant.

For trees the rule is exact:
- give every node one leaf below it as its primary sequence;
- a key passes `in_seq` iff it lies on the root-to-primary-leaf path;
- with `mask_pos = depth`, that admits exactly the node's ancestors and itself. Siblings
  never share a path. It replaces the block-local iota mask at E4 :1360,
and `mask_pos` replaces `iota + cached_len` in the cached-side window distance.

| configuration | `seq_member` | `mask_pos` |
|---|---|---|
| causal chain (today) | one column, all ones | 0..s |
| tree (Medusa, EAGLE) | one column per leaf; a node is a member of every leaf below it | depth |
| lookahead | llama's W+G+1 layout verbatim (E8) | llama's pos |
| bidirectional block (diffusion, DFlash) | one column, all ones | constant |

RoPE keeps its own per-row positions (E3). `mask_pos` is not the rope position. For
diffusion they differ, which is why they are separate inputs.

Graph identity: under `architecture-as-data`'s `cache_mask` decision, the causal
configuration stays on the existing iota algebra (byte-identical, AC0 there). Visibility is
the alternate arm, selected by config. This spec never perturbs the incumbent graph.

## the FSM change (E1)

The FSM needs no new state, only two transition changes:

1. `accept` returns accepted *row indices*, not a prefix count. Chain acceptance is the
   case where the indices are `0..n`.
2. `Verify -> Verify` (refine) is legal when the configured accept pipe returns refine.
   That is the case for Jacobi and diffusion, and the commit is skipped.

Reachability comes from config, per `architecture-as-data` R2.
- A tree or block shape validates only when every descriptor layer is attention-rewindable.
- Recurrent layers (GDN, SSM) admit chain shapes only, with snapshot restore. This is
  parity spec R3b.
- That is a `Validate` rule, never a code branch.

## lowering: finite shapes, planned once

A fixed serving config admits a finite set of evaluation shapes:

- prefill M (symbolic);
- decode 1;
- verify N, where N = k+1, the tree node count, W+G+1, or the block length.

Each shape is lowered once, with `Extent::Static(N)` for N, to a plan with precise kernels.
The row-tiled cached attention from f3c7c2f5 is the verify kernel. The FSM selects the
plan per transition. Plans are not rebuilt per step.

## beyond decode: the whole request is one log (owner, 2026-10-04)

Owner: "consider lambda not just in the decode. outside through the entire set of steps -
tokenization, prefill, etc." / "tokenize, embed, prefill, kv cache - ... decode is in
there... maybe there's more".

The invariant applies to every stage. A stage is kappa when the bulk work is the
incremental path replayed with more rows. It is lambda when two implementations are
selected by row count, phase, flag, env or feature, and the two must then be reconciled.

| stage | incumbent (llama.cpp f1ea20621) | proxima on main 0ac67c63 |
|---|---|---|
| prefill and decode entry | one `llama_context::decode(batch)` (`src/llama-context.cpp:1710`). The batch is split into ubatches (:1868), output rows are chosen per token (`n_outputs_all`, :1772), and the graph is reused when the shape matches (:1410). | Three forks, each re-read at the line in `generate/decode.rs`:<br>(a) Prefill is either one M-row evaluation or M sequential steps. Env vars select it: `PROXIMA_PREFILL_ONE_EVALUATION` / `PROXIMA_PREFILL_SEQUENTIAL` (:1301-1305), read at :3110.<br>(b) A separate `run_decode_loop_placed_kv` loop runs under feature `metal-output-placement` + `is_metal()` + `single_range` (:3059-3064). Otherwise the two-range loop runs.<br>(c) The pre-gather arm is selected by config plus `architecture.ffn_routing()` (:3121). |
| prefill shape | one ubatch loop; ubatch split chosen by memory type (`src/llama-kv-cache.cpp:713`) | Four shapes in one loop, chosen by flags and row count (`generate/decode.rs:3715-3740`):<br>- `one_evaluation_prefill`<br>- `chunked_prefill` (`ubatch_size`)<br>- `split_prefill` (one row at a time when `single_position_step`)<br>- whole range |
| multi-row recurrent step | gated delta net is lowered by shape under one graph op; see the next table | `bind_symbols` rejects `new_count != 1` when `single_position_step` is set (`architecture.rs:84-86`), so M>1 needs a second program pinned `Extent::Static(M)`.<br>That program's real-checkpoint oracle measures text divergence, with the first GDN `block_output` off by about 5x, cause not yet found (`generate/decode.rs:3436-3448`). |
| GDN scan kernel | two algorithms: recurrent for 1 row, chunked for more (`delta-net-base.cpp:433-446`) | One scan, `run_gdn_prefill_scan` (`proxima-tensor/src/gdn.rs:78`), with `positions = n`, so decode is n=1. This is more uniform than the incumbent. The fork is one level up, in `single_position_step` (previous row). |
| GDN prefill backend knob | none | `gdn_prefill_backend` is `Cpu` or `Mlx` (`serving.rs:918`). Selecting `Mlx` without feature `mlx-gdn` is refused (:1237).<br>`git grep 'GdnPrefillBackend::Mlx'` finds only the example parser, one cache-key test case and that guard, so no executor reads it.<br>It is a control-plane knob for a lowering choice, and it still partitions plan and cache keys. |
| detokenize | Per-token `common_token_to_piece` with a `validate_utf8` hold-back. The bulk path is unused by the stream (`server-common.cpp:1582`, "TODO: reuse llama_detokenize"). | Two implementations, sharing `drain_lossy_utf8`:<br>- bulk `decode` (`proxima-tokenizer/src/pipe.rs:198-214`) filters control ids, then applies `unigram::unescape` for SentencePiece or `replace_space_markers` for char-level;<br>- streaming `decode_streamed_piece` (`generate/residency_caches.rs:3395-3409`) applies `replace_space_markers` for unigram OR char-level. It adds a hand-written leading-space trim that "Mirrors" bulk (:3458-3467) and filters control ids again (:3453).<br>No test asserts that the concatenated stream equals bulk. |
| input embedding | one `ggml_get_rows` (`src/llama-graph.cpp:2396`) | One `embedding_lookup` op for M or 1 rows (`proxima-tensor/src/spec/primitives.rs:758`). The gemma4 PLE preamble is plain graph ops. Weight storage forks by codec (packed gather vs dequantized f32, `bind.rs:1220-1262`); that is a codec fork, not a row fork. |
| kernels by row count | mv/mm at `ne11 > 8`, flash-attn vec at `< 20` rows | Same class, one op each:<br>- `CachedAttentionForm` decode-split / row-tiled / single-range (`omega/src/msl/signature_tokens_prelude.rs:1566,1735-1761`);<br>- tiled GEMM at `TILED_GEMM_MIN_TOKENS` (`emit_and_classify.rs:2856`, `omega/omega-runtime.toml`);<br>- cooperative reduce at `COOPERATIVE_REDUCE_MIN_LEN` (`emit_and_classify.rs:1596-1618`).<br>These are lowering specializations. The attention and GEMM forms each have CPU-reference parity tests; no dedicated parity test for cooperative reduce was found. |
| cache engine | one KV memory per arch | three `CacheStrategy` engines selected by code (`descriptor.rs:15`) |
| KV materialization | one KV store | Three stores:<br>- host `LayerCacheState` rows<br>- ring rows<br>- device KV on the placed path<br>The placed path returns an empty `PrefixState` (`generate/decode.rs:3078-3088`), so it cannot feed or reuse the prompt cache. |
| prompt reuse | the same KV store: common prefix plus `seq_rm` (`server-context.cpp:3196-3198`). A separate RAM store holds only evicted slots (:258-296). | A separate host store (`PromptCache` of `PrefixState`). Taking an entry moves it out, and the request owns it (`generate/prompt_cache.rs:24-26`). Bypassed when a seed, override or non-discard sink is present (:1212-1218). |
| rewind | `llama_memory_seq_rm` | `rewind_to` (`prompt_cache.rs:302`) wraps `LayerCache::truncate` (`residency_caches.rs:155`) for attention. Recurrent state cannot rewind and is restored from checkpoints instead (`ring_checkpoint.rs`). |
| expert residency | none (mmap) | Several selectors: the pre-gather arm, `monolithic_all_low` and `monolithic_high_mmap` (`serving.rs:914-953`), with pre-gather disabled under one-evaluation prefill |
| stop / EOG, sampling | one path | one path: `decode_until_stop_or_budget` (`residency_caches.rs:3433`) and `select_decoded_token`, shared by decode and verify |
| tokenize | bulk only (`src/llama-vocab.cpp:4496`) | Bulk. The seeded path tokenizes the suffix with no BOS, while a fresh prompt gets BOS (`generate/decode.rs:2959-2967`). |
| chat template | `common_chat_templates_init` from the GGUF template (`tools/server/server-context.cpp:1436`) | only in `examples/gguf_generate.rs:178-195`, hard-coded ChatML for qwen35 |

### two kinds of split, and only one is lambda

The incumbent is kappa at the control plane and specialized at the lowering. Its forks are
all shape-keyed choices of how to execute one op:

| op | llama.cpp f1ea20621 selector | equality |
|---|---|---|
| gated delta net | `n_seq_tokens == 1`: autoregressive or fused-ar; `> 1`: chunked (CS=64) or fused-ch (`src/models/delta-net-base.cpp:433-446`) | equal in exact arithmetic, not bitwise |
| matmul (Metal) | mat-mv-ext for `ne11` 2..8; mul_mm for `ne11 > 8 && ne00 >= 64` (`ggml-metal-common.cpp:38-41`); MoE mm for `ne21 >= 32` (:48) | same op |
| flash attention (Metal) | vec kernel for `ne01 < 20` query rows (`ggml-metal-ops.cpp:2949-2956`) | same op |
| output-row gather | always emitted, even when every row is output, "to keep topology constant" (`src/llama-graph.cpp:2488-2496`) | none needed |

The other stages show no fork:
- token input embedding: one `ggml_get_rows` (`src/llama-graph.cpp:2396`);
- tokenize: always bulk (`src/llama-vocab.cpp:4496`);
- live prompt reuse: the same KV store, via common prefix plus `seq_rm`
  (`tools/server/server-context.cpp:3196-3198`).

The incumbent does have two control-plane splits:
- detokenize is per-token `common_token_to_piece` with a `validate_utf8` hold-back
  (`server-context.cpp:1825-1829`), beside a bulk `llama_detokenize`
  (`src/llama-vocab.cpp:4517`) that the stream does not use. The comment at
  `server-common.cpp:1582` says "TODO: reuse llama_detokenize";
- a second, RAM-resident prompt store for evicted slots (`server_prompt_cache`,
  `server-context.cpp:258-296, 1340`).

So the rule this spec adopts:
- a **lowering specialization** (one op in the graph, a kernel chosen by shape at plan
  time) is the "precise execution path". It is admissible and belongs below the FSM;
- a **lambda** is a control-plane fork: two loops, two state writers, two stores, or an
  env/feature switch above the op graph. It is removed;
- a specialization that is equal only in exact arithmetic, such as chunked vs recurrent
  GDN, needs a stated tolerance and a test that holds it.

## requirements

- R1. One visibility primitive: `mask_pos` plus `seq_member` inputs. Causal, tree,
  lookahead and bidirectional are expressed as data, with zero new `Op` variants.
- R2. The causal configuration keeps the op graph byte-identical to the incumbent.
- R3. `accept` returns row indices, and commit gathers rows. Prefix truncation is the
  degenerate case and keeps today's `LayerCache::truncate` behaviour.
- R4. `Verify -> Verify` refine is reachable only when the accept pipe is iterative.
- R5. `build_forward` takes a readout list: logits rows, hidden-state taps at named layers,
  extra head roots. MTP heads, Medusa heads and EAGLE taps are readouts, not
  architectures. This supersedes MTP R2 and R9, EAGLE3 R1 and R2, and DFlash R2 and R3 as
  mechanisms; their oracles stand.
- R6. A draft model is a second `ModelDescriptor` driving its own `ServingState`, with no
  per-family type.
- R7. Lossless techniques give output byte-identical to non-speculative decode for any
  `SamplingConfig` (parity spec R1, extended to tree and lookahead shapes). Medusa has no
  llama oracle (E10), so this losslessness is its only correctness bar.
- R8. Lookahead token ids equal llama.cpp f1ea20621 `llama-lookahead` (E8).
- R9. Each verify shape is planned once per serving config, and a decode step builds no
  plan.
- R10. Diffusion token ids equal llama.cpp f1ea20621 `llama-diffusion-cli` (E9) on a LLaDA
  GGUF.
- R11. The MTP, EAGLE3, DFlash and DSpark sub-specs name no `Architecture` impl or
  per-family type. They are restated as readouts plus a draft descriptor; their oracle
  requirements are theirs and unchanged.
- R13. Prefill is one path: the same program evaluated over M rows.
  - Ubatch chunking is a host replay of that program with smaller M, which is the
    incumbent's shape.
  - `single_position_step`, the separately built static-width program, `split_prefill`,
    and the env switches `PROXIMA_PREFILL_ONE_EVALUATION` / `PROXIMA_PREFILL_SEQUENTIAL` /
    `PROXIMA_BOUNDARIES_PREFILL` are removed.
  - This requires root-causing the 5x first-GDN-layer divergence (`generate/decode.rs:3436-3448`).
- R14. Every decode path, including device-placed KV, returns a real `PrefixState`. Device
  KV is a placement of the one state, not a second store, so the prompt cache works on
  every path.
- R15. Detokenize is one implementation. Bulk `decode` is the concatenation of the
  streamed pieces, and the SentencePiece trim and control filtering live in exactly one
  place.
- R16. `gdn_prefill_backend` leaves `ServingConfig`. Choosing an executor for an op is a
  lowering and placement choice below the graph, and does not key plans or caches above
  it.
- R17. A continuation is tokenized by the same function as a fresh prompt, with BOS as a
  log event at position 0: `add_bos = (cached_len == 0)`.
- R18. Every lowering specialization that is selected by row count or length has a parity
  test against the CPU reference on both sides of its threshold.
- R12. The tree-oracle fixtures and the parity comparator are themselves validated. The
  harness reproduces independently vendored llama-server ids on a chain-shaped tree, and
  the comparator rejects a perturbed id.

## refutation condition

This is the wrong thing to build if either is observed:

- a technique on the list cannot be expressed by the four pipes plus `mask_pos` and
  `seq_member` without a family-keyed branch;
- tree-verify logits for a chain-shaped tree are not bit-identical to chain-verify logits on
  the same rows. Masked entries contribute `exp(-inf) = 0` exactly, and the reduction order
  is unchanged, so any bit difference is a defect, not noise;
- per-row greedy ids from a tree batch disagree with llama.cpp's for the same seq-id layout,
  on any row of the AC3 fixtures.

## acceptance criteria (counts; run from the checkout holding main)

This is a list, not a table, so the shell pipes stay literal. Each AC states its kind, the
requirements it discharges, its command, its expected count and its control at 0ac67c63.

The checkpoints are gemma4 E2B, openchat and qwen3, the `architecture-as-data` AC2
targets. Each one runs 3 vendored prompts.

- **AC1, consistency (R1, R2).**
  - Command: `cargo nextest run -p proxima-tensor -E 'test(/visibility_causal_config_graph_identity_/)'`.
  - Expected: 7 passed. Under the causal config, the op graph is byte-identical to the
    incumbent digest for each of the 7 `architecture-as-data` checkpoints.
  - Proof that it discriminates: the same test with the visibility arm forced on gives 7
    failed.
  - Control: absent.
- **AC2, consistency (R1).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/visibility_chain_tree_equal_/)'`.
  - Expected: 3 passed. A chain-shaped tree's logits are bit-identical to chain verify, one
    test per checkpoint.
  - Control: absent.
- **AC3, oracle (R1).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/tree_verify_llama_parity_/)'`.
  - Expected: 9 passed, 3 checkpoints x 3 prompts. The per-row greedy ids of a two-branch
    depth-3 tree equal llama.cpp f1ea20621's for the same tree, encoded as llama seq ids:
    - a node carries the seq ids of the leaves below it, with primary = its first leaf;
    - `pos = n_past + depth`.
  - The ids come from slice 0's fixture harness, under the same harness precedent as
    `speculative-draft-eagle3` R9.
  - Control: absent.
- **AC4, oracle (R3, R7).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/lossless_shape_/)'`.
  - Expected: 9 passed. With speculation on in each of {chain, tree, lookahead}, the ids on
    3 checkpoints equal the llama-server greedy ids from llama.cpp f1ea20621 vendored under
    `proxima-model-interop/tests/fixtures/llama-parity/`. They are compared through llama's
    first EOG id, per the `architecture-as-data` D1 policy.
  - The expected value is llama's output, not proxima's. A lossless shape must reproduce
    the non-speculative incumbent's greedy ids.
  - Control: absent.
- **AC5, oracle (R8).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/lookahead_llama_parity_/)'`.
  - Expected: 9 passed. Ids equal `llama-lookahead` at W=15, N=5, G=15
    (`examples/lookahead/lookahead.cpp:52-54`), on 3 checkpoints x 3 prompts.
  - Control: absent.
- **AC6, oracle (R10, R4).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/diffusion_llama_parity_/)'`.
  - Expected: 3 passed. One LLaDA-8B-Instruct GGUF's ids equal `llama-diffusion-cli` ids on
    3 prompts, under algorithm 4 (confidence-based, the default at `diffusion.h:40`). Steps
    and block length are recorded in the fixture.
  - Control: absent.
- **AC7, consistency (R5).**
  - Command: `cargo nextest run -p proxima-tensor -E 'test(/readout_/)'`.
  - Expected: 2 passed:
    - `readout_layer_taps_`: 3 tap roots at gemma4 E2B layers {2, 13, 25}, each of shape
      `[s, embedding]`;
    - `readout_logits_root_unchanged_`: a logits-only readout list reproduces the
      incumbent digest.
  - Control: absent.
- **AC8, consistency (R9).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/verify_plan_count_/)'`.
  - Expected: 1 passed. Over 64 decode steps with config chain k=4, the plan is built
    exactly 3 times: prefill, decode-1 and verify-5.
  - Control: absent.
- **AC9, consistency (R4, R6).**
  - Command: `git grep -nP '\b(trait|impl|struct|enum)\b[^;{]*(Medusa|Eagle|Mtp|Dflash|Lookahead|Diffusion)' -- proxima-model-interop/src proxima-tensor/src | wc -l`.
  - Expected: 0.
  - Control: 0, measured 2026-10-04.
  - Proof that it discriminates: the same pattern with `Architecture` in place of the
    alternation prints 17 at 0ac67c63 (measured).
- **AC11, oracle-control (R12).**
  - Command: `cargo nextest run -p proxima-model-interop --features std -E 'test(/oracle_harness_control_/)'`.
  - Expected: 5 passed:
    - 3 `oracle_harness_control_chain_`: on a one-leaf tree, the slice-0 harness's ids equal
      the llama-server ids `architecture-as-data` slice 0 already vendored, one per
      checkpoint. That is a different llama entry point, so the harness is checked against
      an independent path;
    - `oracle_harness_control_self_`: the comparator fed llama's AC3 fixture ids on both
      sides passes;
    - `oracle_harness_control_perturbed_`: the comparator fed one flipped id returns a
      mismatch, and the test asserts that rejection.
  - Control: absent.
- **AC10, consistency (R11).**
  - Command: `git grep -nE 'Architecture|Eagle3Arch|DflashArch|DflashDraftProgram' -- proxima-tensor/specs/speculative-draft-mtp/SPEC.md proxima-tensor/specs/speculative-draft-eagle3/SPEC.md proxima-tensor/specs/speculative-draft-dflash/SPEC.md proxima-tensor/specs/speculative-draft-dspark/SPEC.md | wc -l`.
  - Expected: 0. The sub-specs name no `Architecture` impl or per-family type; llama's
    `general.architecture` strings (for example `gemma4-assistant`) appear in them only as
    GGUF-key values read at load, which is data and not a type, so the pattern matches
    type names and not those strings.
  - Control: the same pattern over the same four files as they were at 0ac67c63, before they
    were restated, `git grep -nE 'Architecture|Eagle3Arch|DflashArch|DflashDraftProgram' 0ac67c63 -- <the same four paths> | wc -l`,
    prints 25 (measured 2026-10-12). The previous pattern, which also matched the string
    `gemma4-assistant`, prints 41 over the four files at 0ac67c63 by the same form of
    command (measured 2026-10-12).

- **AC12, oracle (R13).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/prefill_one_path_llama_parity_/)'`.
  - Expected: 9 passed. On 3 checkpoints, prefill at ubatch sizes {1, 7, whole prompt} gives
    generated ids equal to the vendored llama-server ids.
  - qwen35 and qwen35moe join when `architecture-as-data` O1 restores their oracle.
  - Control: absent.
- **AC13, consistency (R13).**
  - Command: `git grep -nE 'PROXIMA_PREFILL_ONE_EVALUATION|PROXIMA_PREFILL_SEQUENTIAL|PROXIMA_BOUNDARIES_PREFILL|single_position_step|split_prefill' -- proxima-model-interop/src proxima-tensor/src omega/src ':!*tests*' | wc -l`.
  - Expected: 0.
  - Control: 62, measured 2026-10-04.
- **AC14, consistency (R14).**
  - Command: `cargo nextest run -p proxima-model-interop --features std,metal,metal-output-placement -E 'test(/placed_kv_prefix_state_/)'`.
  - Expected: 2 passed:
    - the placed path returns a `PrefixState` whose `ids` equal prompt plus generated ids;
    - resuming from it gives ids equal to the vendored llama-server ids for openchat.
  - Control: absent. Today the placed path returns empty `ids` (`generate/decode.rs:3078-3088`).
- **AC15, oracle (R15).**
  - Command: `cargo nextest run -p proxima-model-interop --features std -E 'test(/detokenize_one_path_/)'`.
  - Expected: 3 passed, one per checkpoint vocab. The concatenated streamed pieces equal bulk
    `decode`, and both equal llama-server's `content` string, vendored at slice 9 beside
    each `llama_ids.json`.
  - Control: absent.
- **AC16, consistency (R15).**
  - Command: `git grep -nE 'unigram_leading_space_trimmed|fn decode_streamed_piece' -- proxima-model-interop/src ':!*tests*' | wc -l`.
  - Expected: 0.
  - Control: 4, measured 2026-10-04.
- **AC19, consistency (R16).**
  - Command: `git grep -nE 'gdn_prefill_backend|GdnPrefillBackend' -- proxima-model-interop/src omega/src ':!*tests*' | wc -l`.
  - Expected: 0.
  - Control: 26, measured 2026-10-04.
- **AC17, oracle (R17).**
  - Command: `cargo nextest run -p proxima-model-interop --features std -E 'test(/continuation_tokenize_/)'`.
  - Expected: 3 passed. On 3 checkpoints, a prompt split at a word boundary and resumed
    from its prefix gives prompt ids equal to llama's `prompt_ids` for the whole prompt.
  - Control: absent.
- **AC18, consistency (R18).**
  - Command: `cargo nextest run -p proxima-omega --features metal,metal-tiled-gemm -E 'test(/threshold_both_sides_/)'`.
  - Expected: 3 passed, one each for cached attention form, tiled GEMM and cooperative
    reduce. Each runs at threshold-1 and at threshold against the CPU reference.
  - Control: absent.

## dependencies

- `architecture-as-data` slice 6 (generic verify program), slice 7 (`cache_mask`) and
  slice 10b (FSM live). This spec's slices 1-3 stack on 7, and slice 4 onward on 10b.
- The parity spec's R19 verify-width cost curve decides which shapes pay. It is a
  measurement, not a gate on building the structure.

## out of scope

- Pipelined proposer and verifier (draft t+1 while verify t). That is genuine streaming.
  It is a pipe composition over this spec's stages and is specced separately once R19
  exists.
- New kernels. Row-tiled cached attention already covers verify N.
