# speculative-draft-eagle3

status: draft
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)

## problem

Parent spec `speculative-decode-llama-parity` R11b requires an audited sub-spec for `draft-eagle3`
before draft correctness can be claimed; today proxima has no path to it -- no family profile is
selected by a `general.architecture` value of `eagle3` (profiles are embedded TOML files under
`proxima-model-interop/src/profiles/`, none named for it), gemma4's lowered program carries no
per-layer residual roots (its profile selects `schedule_source = "sliding_pattern"`,
`proxima-model-interop/src/profiles/gemma4.toml:6`, whose descriptor is `CacheMask::Padded`,
`proxima-tensor/src/spec/gguf_descriptor.rs:170`; `build_forward`'s Padded arm returns
`layer_residuals: Vec::new()` and `hidden: None`, `proxima-tensor/src/spec/descriptor.rs:958-959`, so
`BoundProgram::residual_roots` and `BoundProgram::hidden_root` are empty for it,
`proxima-model-interop/src/lowering.rs:141` and `:137`), and no EAGLE3 draft GGUF
for any proxima-shipped target family exists on this machine, so for a caller of `ServingConfig`'s
speculative section (parent R9) `draft-eagle3` is unreachable on every model proxima ships -- this
sub-spec makes it configuration data on the `decode-as-data` frame (hidden-state tap readouts on the
target, `decode-as-data` R5, plus a second `ModelDescriptor` driving its own `ServingState`,
`decode-as-data` R6), with drafted tokens equal to llama.cpp `f1ea20621`'s
`common_speculative_impl_draft_eagle3` (`common/speculative.cpp:455-920`) on the same target and draft
GGUFs for the gemma4 target family, the first target (see decisions: llama.cpp's EAGLE3 sidecar list, `docs/speculative.md:37-51`, also names Llama3.1/3.3, Qwen3 and gpt-oss pairs), and adds no technique-named type (`decode-as-data` AC9).

## refutation condition

Written before evidence: if, on the target and draft pair named in slice 1, running the ported
`draft-eagle3` propose pipe over the 50+ prompt corpus (`proxima-model-interop/examples/data/speculative_corpus.jsonl`)
produces a mean of 0 accepted draft tokens per verify step while llama.cpp's own
`llama-speculative-simple --spec-type draft-eagle3` on the identical pair and prompts produces a
nonzero mean, the tap-and-descriptor design (R2, R3, R4) is wrong and must be redesigned, not
tuned -- the failure is structural, not a threshold to adjust.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a draft descriptor is built from GGUF-key reads, not a per-family type: `eagle3.target_layers` (exactly 3 `i32` entries), `eagle3.target_hidden_size`, `eagle3.attention.layer_norm_rms_epsilon`, optional `eagle3.norm_before_residual`/`eagle3.norm_before_fc`, and tensors `fc.weight`, optional `d2t`, optional `enc.output_norm.weight`, `blk.0.{attn_norm,attn_norm_2,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_gate,ffn_down,ffn_up}.weight`, optional `blk.0.rope_freqs.weight`, `output_norm.weight`, optional `output.weight`, optional `token_embd.weight` | yes |
| R2 | the target's lowered program exposes an all-positions hidden-state tap readout (`decode-as-data` R5) at caller-chosen layer indices (0-indexed blocks plus one post-final-norm-input pseudo-index), one gathered row per new position per requested layer; the 3 indices come from R1's `eagle3.target_layers` | yes |
| R3 | the draft encoder graph (3-layer feature fuse -> optional RMSNorm -> `fc` -> g_embd) is expressed with `Op::Input`/`Elementwise`/`Reduce` only and equals `src/models/eagle3.cpp:124-146` on llama.cpp's own tensors | yes |
| R4 | the draft decoder step graph (RMSNorm(tok_embd) concat RMSNorm(g_embd) -> attention -> FFN -> residual -> output_norm -> lm_head -> optional `d2t` scatter) is expressed with `Op::Input`/`Elementwise`/`Reduce` only and equals `src/models/eagle3.cpp:151-326` on llama.cpp's own tensors | yes |
| R5 | the propose pipe's deferred-boundary bookkeeping (cross-ubatch bridge, seed step, top-k=10 autoregressive loop with `p_min` early stop and `n_min`/`n_max` bounds) equals `common_speculative_impl_draft_eagle3::process`/`draft` (`common/speculative.cpp:593-859`) | yes |
| R6 | the accept stage rewinds the deferred boundary to the accepted row exactly as `common/speculative.cpp:861-876` | yes |
| R7 | for an attention-only target the draft's state stash is empty and restoring it is a no-op, equal to the `stash_bytes = 0` llama.cpp `f1ea20621` records from `get_state` (`common/speculative.cpp:880-919`, which stashes the boundary only when the target is recurrent or hybrid); the recurrent-target stash content is out of scope (see out of scope) | yes |
| R8 | `ServingConfig`'s speculative section (parent R9) accepts the llama type name `draft-eagle3` (the existing `SpeculativeTypeName::DraftEagle3`, `proxima-model-interop/src/speculative_settings.rs:42`) as data that selects the propose pipe carrying the target tap handle (R2) and the loaded draft descriptor (R1), dispatched like the existing draft-model support (`proxima-model-interop/src/generate/draft_model.rs:174-183`), not like the token-id-only n-gram drafters | yes |
| R9 | pairing compatibility (`target_layer_ids_n == 3`, every id `<=` the target's `n_layer`, both contexts present) is checked with a typed error in proxima, where llama.cpp uses three different mechanisms: both contexts present is `GGML_ASSERT(ctx_tgt && ctx_dft ...)` (`common/speculative.cpp:495`, abort), `target_layer_ids_n != 3` is a thrown `std::runtime_error` (`:502-505`), and an id above the target's `n_layer` is `GGML_ABORT` in the extraction loop (`:541-549`; an id equal to `n_layer` is accepted there and taps the post-final-norm input via `llama_set_embeddings_nextn`, so the bound is `<=`); proxima does not go through `common_speculative_are_compatible`'s vocab/bos/eos gate (`common/speculative.cpp:67-130`, which EAGLE3's own constructor never calls) | yes |
| R10 | the ported `draft-eagle3` achieves a nonzero mean of accepted draft tokens per verify step across the 50+ prompt corpus -- the measurable form of this sub-spec's refutation condition | yes |
| R11 | the vendored fixtures are produced by a C++ harness linking llama.cpp `f1ea20621` and calling the same public API EAGLE3's own implementation calls (`llama_set_embeddings_layer_inp`, `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn(_ith)`, `llama_model_target_layer_ids(_n)`, `src/llama-ext.h:96-128`), never re-derived from reading the C++; each fixture file records the upstream commit in its header | yes |
| R12 | each oracle comparator is itself validated: the four comparators (ids, `hidden_tap` float rows, `encoder` float rows, `decoder_draft` float rows) each pass when fed llama's own fixture on both sides, and each return a mismatch when fed one perturbed value (one flipped id; for a float comparator, one element of one row moved by more than the `1e-4` absolute tolerance AC4-AC6 compare under) | yes |
| R13 | for the same target, draft, prompt and seed, proxima's generated token ids with `draft-eagle3` enabled equal the ids llama.cpp `f1ea20621`'s `llama-speculative-simple` generates with `--spec-type draft-eagle3 --model-draft`, for 3 seeds, and every vendored oracle run itself drafted (recorded `n_drafted` >= 1) | yes |

## architecture

Nothing here is a new type. EAGLE3 is a hidden-state tap readout list on the target descriptor plus a
second descriptor (the draft) with its own `ServingState`, driven through the four pipes of
`decode-as-data` (propose, shape, accept, commit). Its row in that spec's table is "draft descriptor
fed by target hidden-state taps | tree | sample-and-match | path gather"; this sub-spec uses the
chain shape (parity with llama.cpp's linear draft), and trees are `decode-as-data`'s concern.

- **parse (R1)**: no proxima-gguf change. `MetadataArray::I32(Vec<i32>)` (`proxima-gguf/src/value.rs`)
  already carries an arbitrary-length signed-int array, and `find_tensor`
  (`proxima-model-interop/src/bind.rs:298-304`) looks up any tensor name with no allowlist. The gap is
  that nothing reads the `eagle3.*` keys or looks up the `fc`/`d2t`/`enc.output_norm`/`attn_norm_2`
  tensors. A family profile TOML for the `eagle3` `general.architecture` value, beside the existing
  profiles, supplies the layer shape; the keys feed the `ModelDescriptor`
  (`proxima-tensor/src/spec/descriptor.rs:130-296`) the way the other profiles' keys do. The family name
  is a key value read at load, never a Rust type.
- **target tap (R2)**: `BoundProgram` carries `hidden_root` and `residual_roots`
  (`proxima-model-interop/src/lowering.rs:137` and `:141`), copied from the `ForwardProgram` at
  `lowering.rs:419-420`. The Cached/Bounded engine fills one residual root per layer
  (`proxima-tensor/src/spec/attention_forward.rs:3014`, pushed before the final `gather_last_row`
  at `:3025`), but gemma4 does not lower through it: the sliding-pattern header
  (`proxima-model-interop/src/lowering.rs:277-284`) builds a `CacheMask::Padded` descriptor
  (`proxima-tensor/src/spec/gguf_descriptor.rs:170`), and `build_forward`'s Padded arm returns
  `layer_residuals: Vec::new()` and `hidden: None` (`proxima-tensor/src/spec/descriptor.rs:958-959`).
  Three things are missing before EAGLE3 can tap gemma4: the Padded engine
  (`proxima-tensor/src/spec/scheduled_cached.rs:1136-1853`) must expose per-layer residual nodes the
  way the Bounded engine does, the caller must select 3 specific indices (llama's
  `target_layer_ids`, not every layer), and the gather must cover every new position. All three are the
  readout list of `decode-as-data` R5 (slice 3 there), consumed here; this sub-spec adds no gather of
  its own.
- **encoder/decoder graphs (R3/R4)** need no new `Op` variant. `Op::Input`
  (`proxima-tensor/src/op.rs:201-205`) is a named leaf where data enters, which is what llama.cpp's
  `common_batch::set_embd` provides (an externally supplied float row bound to a decode-step input
  instead of an embedding-table lookup). The feature concat is `Elementwise` into a wider
  `Input`-shaped buffer; `fc` and the decoder's attention/FFN projections are `Reduce` contractions
  with the single decoder layer (`GGML_ASSERT(n_layer == 1)`, `src/models/eagle3.cpp:156`) and the concatenated-input width `n_embd_attn_input = 2 * n_embd` (`src/models/eagle3.cpp:42`, applied to the `wq`/`wk`/`wv` shapes at `:88-90`); the concat call itself is `:220`.
- **draft loop (R5/R6/R7)** is the propose pipe configured by the `draft-eagle3` value (R8), with
  accept as sample-and-match and commit as prefix truncate. No closed-enum variant is added: the
  existing `SpeculativeType::DraftEagle3` (`proxima-model-interop/src/serving.rs:208`) is the config
  vocabulary, and the work it selects is a composition of the pipes above.
- **fixture generator (R11)** is a C++ harness in the same shape as the parent's n-gram generator
  (`proxima-tokenizer/tests/fixtures/llama-ngram/`), compiled against llama.cpp `f1ea20621`, driving a
  real gemma4 target context and the real EAGLE3 draft context through the same public API, dumping
  the 3 tapped hidden-state rows per position, the encoder's g_embd rows, and the drafted token
  sequences with their accept-rewind positions to JSON.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| target family | gemma4 first | proxima profiles exist for llama, qwen3, qwen3moe, qwen2 and mistral (`proxima-model-interop/src/profiles/{llama,qwen3,qwen3moe,qwen2,mistral}.toml`) and llama.cpp lists Llama3.1/3.3 and Qwen3 sidecars (`docs/speculative.md:37-38,41-48`), so gemma4 is not the only possible target. The reason it goes first: R2's tap diagnosis in this spec is gemma4-specific (its profile alone selects `sliding_pattern`, so it alone lowers through the Padded arm that returns no residual roots), the parity example and the n-gram fixtures already run on gemma4 E2B, and the sidecars llama.cpp names for it are `RedHatAI/gemma-4-31B-it-speculator.eagle3` and `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` (`docs/speculative.md:39-40`). No sidecar for a gemma4 E2B-sized target is listed anywhere in `docs/speculative.md:37-51`; the fixtures therefore use the 26B-A4B pair (slice 1), larger than the workspace's designated small test models (gemma4 E2B, granite 1b), and no E2B pairing is claimed. A llama or qwen3 first target is untried: whether their default-schedule programs yield the 3 taps is unmeasured |
| hidden-state tap shape | the `decode-as-data` readout list, all-positions, indices from `eagle3.target_layers` | a bespoke tap type would duplicate the readout list for no new capability (principle 1) and is a technique-named type (`decode-as-data` AC9) |
| decoder input | `Op::Input` bound at decode time, same as token embeddings | a new external-embedding `Op` variant was considered and rejected: `Op::Input` is already an untyped named leaf; g_embd needs a new binding at runtime, not new algebra |
| oracle | llama.cpp `f1ea20621`'s own `common_speculative_impl_draft_eagle3`, via a compiled fixture generator and `llama-speculative-simple`, on the pair from slice 1 | re-deriving expected g_embd/logit rows from reading the C++ is principle 14's "from memory" failure; no stock tool dumps `llama_get_embeddings_layer_inp` outside this implementation |
| compat check | EAGLE3-specific (`target_layer_ids_n == 3`, ids in range) | the vocab/bos/eos gate is never called from EAGLE3's own constructor (`common/speculative.cpp:486-561`, unlike draft-simple's constructor, which calls it and throws on mismatch at `:242-249`); copying it would diverge from the incumbent |
| KV/rollback ownership | parent R3a/R3b | EAGLE3 uses the same prefix-truncate commit as every other drafter; this sub-spec owns only its deferred-boundary rewind (R6), additional per-drafter state, not a KV mechanism |

## acceptance criteria

`$PAIR_TGT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/gemma4-target.gguf`, the gemma4 target GGUF slice 1 converts; slice 1 places the file (or a symlink) at exactly this path
`$PAIR_DFT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/gemma4-eagle3.gguf`, the EAGLE3 draft GGUF slice 1 converts, at exactly this path
`$LLAMA_SPEC_SIMPLE` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple`, an absolute binary path built from llama.cpp `f1ea20621` by slice 0b (the prebuilt `build/bin` in the checkout predates the source and does not register `--spec-type`); once built it registers `--spec-type` and `--model-draft` (`common/arg.cpp:4238,4246`) and prints `n_drafted = N` (`examples/speculative-simple/speculative-simple.cpp:355`)

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R13 (prerequisite) | `ls -1 "$PAIR_TGT" "$PAIR_DFT" 2>/dev/null \| wc -l` | 2 |
| AC2 | R11 | `for name in hidden_tap encoder decoder_draft; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/$name.json; done` | three lines printed, each >= 50 |
| AC3 | R1 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_descriptor_reads_target_layers_and_tensors/)' 2>&1 \| grep -cE 'target_layers = \[[0-9]+, [0-9]+, [0-9]+\] target_hidden_size = [0-9]+'` | 1 (the test ASSERTS that `target_layers` and `target_hidden_size` equal the `target_layers` and `target_hidden_size` header fields of `hidden_tap.json` and panics otherwise, then prints `target_layers = [L0, L1, L2] target_hidden_size = N`; the count of that printed line is 1 only on a passing run; oracle: the values llama.cpp `f1ea20621` returns through `llama_model_target_layer_ids(_n)` for `$PAIR_DFT`, recorded by the slice-2 harness) |
| AC4 | R2 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/gemma4_hidden_tap_matches_llama_fixture/)' 2>&1 \| grep -cE 'positions = [0-9]+ layers = 3 max_abs_diff = '` | 1 (the test ASSERTS `positions` >= 50, `layers` == 3 and `max_abs_diff` < 1e-4, panicking otherwise, and prints `positions = N layers = 3 max_abs_diff = X` only after those assertions pass, so the count is 0 on a failing run and on a zero-position loop; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/hidden_tap.json`, the rows llama.cpp `f1ea20621` returns from `llama_get_embeddings_layer_inp`) |
| AC5 | R3 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_encoder_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+ max_abs_diff = '` | 1 (the test ASSERTS `cases` >= 50 and `max_abs_diff` < 1e-4, panicking otherwise, and prints `cases = N max_abs_diff = X` only after those assertions pass; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/encoder.json`, g_embd rows from llama.cpp `f1ea20621`) |
| AC6 | R4 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_decoder_step_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+ max_abs_diff = '` | 1 (the test ASSERTS `cases` >= 50 and `max_abs_diff` < 1e-4, panicking otherwise, and prints `cases = N max_abs_diff = X` only after those assertions pass; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json`, step logits from llama.cpp `f1ea20621`) |
| AC7 | R5 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_draft_loop_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+ p_min_early_stops = [0-9]+ n_max_stops = [0-9]+'` | 1 (the test ASSERTS `cases` >= 50, `p_min_early_stops` >= 1 and `n_max_stops` >= 1, panicking otherwise, and prints `cases = N p_min_early_stops = P n_max_stops = M` only after those assertions pass; oracle: the drafted sequences in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` recorded from llama.cpp `f1ea20621`'s `common_speculative_impl_draft_eagle3`) |
| AC8 | R6 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_accept_rewinds_boundary/)' 2>&1 \| grep -cE 'cases = [0-9]+'` | 1 (the test ASSERTS `cases` >= 50, panicking otherwise, and prints `cases = N` only after that assertion passes; oracle: the accept-rewind positions recorded in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` from llama.cpp `f1ea20621`) |
| AC9 | R7 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_state_stash_empty_for_attention_target/)' 2>&1 \| grep -cE 'cases = [0-9]+'` | 1 (the test ASSERTS that proxima's stash length equals the `stash_bytes` (0) recorded per case in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` by llama.cpp `f1ea20621`'s `get_state`, and that `cases` >= 50, panicking otherwise, and prints `cases = N` only after both assertions pass) |
| AC10 | R8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_selects_draft_eagle3/)'` | 2 passed (the string `draft-eagle3` parses to `SpeculativeType::DraftEagle3`, the same type llama.cpp `f1ea20621` maps that string to at `common/speculative.cpp:33-45`; a config selecting it with no draft model path returns the typed error, matching llama.cpp's `GGML_ASSERT(ctx_tgt && ctx_dft ...)` requirement at `common/speculative.cpp:495`) |
| AC11 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_pairing_rejects_wrong_layer_count/)'` | 2 passed (a draft with `target_layer_ids_n != 3` and a draft with an id above the target's `n_layer` each return the typed error); oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json`, the accept/reject outcome llama.cpp `f1ea20621` records for each of the same two synthetic drafts (written by `proxima-gguf`'s writer), which proxima's outcome must equal |
| AC12 | R10 | `cargo run --release -p proxima-model-interop --features std --example speculative_acceptance_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model "$PAIR_DFT" "$PAIR_TGT" > acceptance_eagle3.txt; grep -cE 'mean_accepted_per_step=([1-9][0-9]*\.[0-9]+\|0\.[0-9]*[1-9])' acceptance_eagle3.txt` | 1 (the `summary` line prints a nonzero `mean_accepted_per_step=`; a printed `0.000000` matches 0 lines and invokes the refutation clause once AC18 shows the oracle's mean is nonzero) |
| AC18 | R10 | `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 -p "Repeat exactly five times: the quick brown fox jumps over the lazy dog." -n 40 2>&1 \| grep -cE 'n_accept +=  *[1-9]'` | 1 (the oracle itself accepts at least one drafted token on the same pair, which makes AC12's zero decidable against the refutation condition) |
| AC17 | R9 | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json` | 2 |
| AC13 | R12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_oracle_control_/)'` | 8 passed: for each of the four comparators (ids, hidden_tap, encoder, decoder_draft) one `eagle3_oracle_control_self_<comparator>` test (llama's own fixture ids or float rows on both sides pass) and one `eagle3_oracle_control_perturbed_<comparator>` test (one flipped id, or one float element moved beyond `1e-4`, returns a mismatch, asserted) |
| AC14 | R13 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_llama_parity_seed_/)' 2>&1 \| grep -cE 'ids_compared = [0-9]+'` | 3 (one printed line per seed 1, 2, 3; each test ASSERTS `ids_compared` >= 8 and id equality, panicking otherwise, and prints `ids_compared = N` only after those assertions pass; oracle: ids vendored from `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 --seed <seed> -n 40` (llama.cpp `f1ea20621`)) |
| AC15 | R13 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/ids.json` | 3 (every vendored oracle run drafted, so an oracle that drafted nothing cannot back AC14) |

## out of scope

- Llama3, Qwen3 and gpt-oss EAGLE3 pairs -- llama and qwen3 profiles exist (`proxima-model-interop/src/profiles/llama.toml`, `qwen3.toml`, `qwen3moe.toml`) but this sub-spec ports and fixtures only the gemma4 pairing (see decisions); gpt-oss has no profile (`FAMILY_PROFILES`, `proxima-model-interop/src/profiles/mod.rs:25-40`, lists none, and `grep -rniE 'gpt.?oss' proxima-model-interop/src proxima-gguf/src` prints nothing)
- the `norm_before_residual`/`norm_before_fc` RedHat/eagle3.1 variants beyond reading the two optional GGUF bools and branching the two known cases (`src/models/eagle3.cpp:24-29`, `:131-134`, `:213-217`) -- no third variant is named upstream
- backend-offloaded top-k sampling (`params.draft.backend_sampling`, `common/speculative.cpp:526-538`) -- CPU-side top-k=10 sampling only; offload is a performance path, and llama.cpp itself falls back to CPU on offload failure (`:531-535`)
- the recurrent-target stash content (the non-empty branch of `get_state`/`set_state`, `common/speculative.cpp:880-919`) -- legitimately deferred: it needs an EAGLE3 sidecar for a recurrent or hybrid target, and none is named upstream (`docs/speculative.md:37-51` lists the named pairs: llama3.1/3.3, gemma4, qwen3, gpt-oss), so no oracle run can exist; the dependency that unblocks it is such a pair, and R7 covers the attention-only branch now
- tree-shaped EAGLE3 drafts -- the tree shape and path-gather commit belong to `decode-as-data`; this sub-spec matches llama.cpp's linear draft
- DFlash / DSpark (`common/speculative.cpp:923-1329`) -- parent R11c/R11d, separate sub-specs
- the readout list itself (`decode-as-data` R5, slice 3) -- owned by that spec; this sub-spec consumes it

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| RedHatAI's gemma-4-31B-it target is too large to run repeatedly in the dev loop (unlike the 2B E2B fixture) | high | slow iteration; possibly the 26B-A4B MoE pairing instead, which the gemma4 MoE path must also support | slice 1 records actual download size and picks the smaller of the two named pairs first |
| gemma4's per-layer residual shape does not line up with llama.cpp's `n_embd` per-layer convention (a distinct pre/post-norm residual stream) | medium | R2's tap reads the wrong tensor and produces a plausible-but-wrong g_embd that still runs | AC4 compares to llama's rows at 1e-4, not just "compiles and runs" |
| the HF checkpoint format of RedHatAI's speculator matches neither documented format (SpecForge `LlamaForCausalLMEagle3` / vLLM `Eagle3LlamaForCausalLM`) | low | slice 1's conversion fails before any proxima code is touched | `convert_hf_to_gguf.py` is llama.cpp's own converter and declares both formats supported (`docs/speculative.md:23-26`) |

## context

- proxima: `proxima-tensor/src/spec/descriptor.rs:130-296` (`ModelDescriptor`), `proxima-tensor/src/op.rs:201-205` (`Op::Input`), `proxima-model-interop/src/lowering.rs:118-189` (`BoundProgram`; `hidden_root` at `:137`, `residual_roots` at `:141`), `proxima-model-interop/src/profiles/gemma4.toml`, `proxima-model-interop/src/bind.rs:298-304` (`find_tensor`), `proxima-model-interop/src/generate/draft_model.rs:174-183` (`load_draft`), `proxima-model-interop/src/speculative_settings.rs:36-59` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205-217` (`SpeculativeType`), `proxima-gguf/src/value.rs` (`MetadataArray`), `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-model-interop/examples/speculative_acceptance_corpus.rs`, `proxima-model-interop/examples/data/speculative_corpus.jsonl`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp:455-920` (`common_speculative_impl_draft_eagle3`: ctor `:486-561`, `process` `:593-736`, `draft` `:738-859`, `accept` `:861-876`, `get_state`/`set_state` `:885-919`), `:67-130` (`common_speculative_are_compatible`, not called by EAGLE3), `src/llama-ext.h:96-128`, `src/llama-model.h:709` (`target_layer_ids`), `src/llama-model.cpp:3387-3394`, `src/llama-arch.cpp:145` (`LLM_ARCH_EAGLE3 = "eagle3"`), `:365` (`LLM_KV_TARGET_LAYERS`), `src/models/eagle3.cpp:1-338` (`load_arch_hparams` `:3-36`, `load_arch_tensors` `:38-101`, encoder `:124-146`, decoder `:151-326`), `common/arg.cpp:4238` (`--spec-draft-model`), `:4246` (`--spec-type`), `:3094-3100` (`--eagle3` download flag), `examples/speculative-simple/speculative-simple.cpp:355`, `docs/speculative.md:16-53` (named pairs, conversion command)
- no local EAGLE3 draft GGUF exists on this machine as of 2026-09-28 (`~/.ollama/models/` and a home-directory search for `*eagle3*` found none) -- slice 1 obtains one
