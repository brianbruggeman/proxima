# speculative-draft-eagle3

status: draft
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)

## problem

Parent spec `speculative-decode-llama-parity` R11b requires an audited sub-spec for `draft-eagle3`
before draft correctness can be claimed; today proxima has no path to it -- no family profile is
selected by a `general.architecture` value of `eagle3` (profiles are embedded TOML files under
`proxima-model-interop/src/profiles/`, none named for it), gemma4's lowered program exposes no
intermediate-layer hidden states at any position (`BoundProgram::hidden_root` and `residual_roots`
are last-row-only consumers, `proxima-model-interop/src/lowering.rs:137,141`), and no EAGLE3 draft GGUF
for any proxima-shipped target family exists on this machine, so for a caller of `ServingConfig`'s
speculative section (parent R9) `draft-eagle3` is unreachable on every model proxima ships -- this
sub-spec makes it configuration data on the `decode-as-data` frame (hidden-state tap readouts on the
target, `decode-as-data` R5, plus a second `ModelDescriptor` driving its own `ServingState`,
`decode-as-data` R6), with drafted tokens equal to llama.cpp `f1ea20621`'s
`common_speculative_impl_draft_eagle3` (`common/speculative.cpp:455-920`) on the same target and draft
GGUFs for the gemma4 target family, the one family both proxima and llama.cpp's own EAGLE3 sidecar
list (`docs/speculative.md:39-40`) support, and adds no technique-named type (`decode-as-data` AC9).

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
| R8 | `ServingConfig`'s speculative section (parent R9) accepts the llama type name `draft-eagle3` (the existing `SpeculativeTypeName::DraftEagle3`, `proxima-model-interop/src/speculative_settings.rs:36`) as data that selects the propose pipe carrying the target tap handle (R2) and the loaded draft descriptor (R1), dispatched like the existing draft-model support (`proxima-model-interop/src/generate/draft_model.rs:174`), not like the token-id-only n-gram drafters | yes |
| R9 | pairing compatibility (`target_layer_ids_n == 3`, every id `<=` the target's `n_layer`, both contexts present) is checked per `common/speculative.cpp:495-505` and `:542-547` with a typed error, not through `common_speculative_are_compatible`'s vocab/bos/eos gate (`common/speculative.cpp:67-99`, which EAGLE3's own constructor never calls) | yes |
| R10 | the ported `draft-eagle3` achieves a nonzero mean of accepted draft tokens per verify step across the 50+ prompt corpus -- the measurable form of this sub-spec's refutation condition | yes |
| R11 | the vendored fixtures are produced by a C++ harness linking llama.cpp `f1ea20621` and calling the same public API EAGLE3's own implementation calls (`llama_set_embeddings_layer_inp`, `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn(_ith)`, `llama_model_target_layer_ids(_n)`, `src/llama-ext.h:96-128`), never re-derived from reading the C++; each fixture file records the upstream commit in its header | yes |
| R12 | the oracle comparator is itself validated: fed llama's own fixture on both sides it passes, and fed one flipped id it returns a mismatch | yes |
| R13 | for the same target, draft, prompt and seed, proxima's generated token ids with `draft-eagle3` enabled equal the ids llama.cpp `f1ea20621`'s `llama-speculative-simple` generates with `--spec-type draft-eagle3 --model-draft`, for 3 seeds, and every vendored oracle run itself drafted (recorded `n_drafted` >= 1) | yes |

## architecture

Nothing here is a new type. EAGLE3 is a hidden-state tap readout list on the target descriptor plus a
second descriptor (the draft) with its own `ServingState`, driven through the four pipes of
`decode-as-data` (propose, shape, accept, commit). Its row in that spec's table is "draft descriptor
fed by target hidden-state taps | tree | sample-and-match | path gather"; this sub-spec uses the
chain shape (parity with llama.cpp's linear draft), and trees are `decode-as-data`'s concern.

- **parse (R1)**: no proxima-gguf change. `MetadataArray::I32(Vec<i32>)` (`proxima-gguf/src/value.rs`)
  already carries an arbitrary-length signed-int array, and `find_tensor`
  (`proxima-model-interop/src/bind.rs:298`) looks up any tensor name with no allowlist. The gap is
  that nothing reads the `eagle3.*` keys or looks up the `fc`/`d2t`/`enc.output_norm`/`attn_norm_2`
  tensors. A family profile TOML for the `eagle3` `general.architecture` value, beside the existing
  profiles, supplies the layer shape; the keys feed the `ModelDescriptor`
  (`proxima-tensor/src/spec/descriptor.rs:130`) the way the other profiles' keys do. The family name
  is a key value read at load, never a Rust type.
- **target tap (R2)**: `BoundProgram` carries `hidden_root` and `residual_roots`
  (`proxima-model-interop/src/lowering.rs:137,141`), populated only by the dense family and consumed
  last-row-only. Three things are missing before EAGLE3 can tap gemma4: the gemma4 lowering must
  expose per-layer taps at all, the caller must select 3 specific indices (llama's
  `target_layer_ids`, not every layer), and the gather must be all-positions. All three are the
  readout list of `decode-as-data` R5 (slice 3 there), consumed here; this sub-spec adds no gather of
  its own.
- **encoder/decoder graphs (R3/R4)** need no new `Op` variant. `Op::Input`
  (`proxima-tensor/src/op.rs:201`) is a named leaf where data enters, which is what llama.cpp's
  `common_batch::set_embd` provides (an externally supplied float row bound to a decode-step input
  instead of an embedding-table lookup). The feature concat is `Elementwise` into a wider
  `Input`-shaped buffer; `fc` and the decoder's attention/FFN projections are `Reduce` contractions
  with `n_layer == 1` and the concatenated-input width `src/models/eagle3.cpp:220-221` names.
- **draft loop (R5/R6/R7)** is the propose pipe configured by the `draft-eagle3` value (R8), with
  accept as sample-and-match and commit as prefix truncate. No closed-enum variant is added: the
  existing `SpeculativeType::DraftEagle3` (`proxima-model-interop/src/serving.rs:205`) is the config
  vocabulary, and the work it selects is a composition of the pipes above.
- **fixture generator (R11)** is a C++ harness in the same shape as the parent's n-gram generator
  (`proxima-tokenizer/tests/fixtures/llama-ngram/`), compiled against llama.cpp `f1ea20621`, driving a
  real gemma4 target context and the real EAGLE3 draft context through the same public API, dumping
  the 3 tapped hidden-state rows per position, the encoder's g_embd rows, and the drafted token
  sequences with their accept-rewind positions to JSON.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| target family | gemma4 only | it is the only family both proxima ships and llama.cpp's supported-EAGLE3-draft list names (`docs/speculative.md:39-40`: `RedHatAI/gemma-4-31B-it-speculator.eagle3`, `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`); a Qwen3/Llama3 pair would need a second target profile not yet written |
| hidden-state tap shape | the `decode-as-data` readout list, all-positions, indices from `eagle3.target_layers` | a bespoke tap type would duplicate the readout list for no new capability (principle 1) and is a technique-named type (`decode-as-data` AC9) |
| decoder input | `Op::Input` bound at decode time, same as token embeddings | a new external-embedding `Op` variant was considered and rejected: `Op::Input` is already an untyped named leaf; g_embd needs a new binding at runtime, not new algebra |
| oracle | llama.cpp `f1ea20621`'s own `common_speculative_impl_draft_eagle3`, via a compiled fixture generator and `llama-speculative-simple`, on the pair from slice 1 | re-deriving expected g_embd/logit rows from reading the C++ is principle 14's "from memory" failure; no stock tool dumps `llama_get_embeddings_layer_inp` outside this implementation |
| compat check | EAGLE3-specific (`target_layer_ids_n == 3`, ids in range) | the vocab/bos/eos gate is never called from EAGLE3's own constructor (`common/speculative.cpp:486-561`, unlike draft-simple's constructor at `:242-248`); copying it would diverge from the incumbent |
| KV/rollback ownership | parent R3a/R3b | EAGLE3 uses the same prefix-truncate commit as every other drafter; this sub-spec owns only its deferred-boundary rewind (R6), additional per-drafter state, not a KV mechanism |

## acceptance criteria

`$PAIR_TGT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/gemma4-target.gguf`, the gemma4 target GGUF slice 1 converts; slice 1 places the file (or a symlink) at exactly this path
`$PAIR_DFT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/gemma4-eagle3.gguf`, the EAGLE3 draft GGUF slice 1 converts, at exactly this path
`$LLAMA_SPEC_SIMPLE` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple`, an absolute binary path built from llama.cpp `f1ea20621` by slice 0b (the prebuilt `build/bin` in the checkout predates the source and does not register `--spec-type`); once built it registers `--spec-type` and `--model-draft` (`common/arg.cpp:4238,4246`) and prints `n_drafted = N` (`examples/speculative-simple/speculative-simple.cpp:355`)

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R13 (prerequisite) | `ls -1 "$PAIR_TGT" "$PAIR_DFT" 2>/dev/null \| wc -l` | 2 |
| AC2 | R11 | `for name in hidden_tap encoder decoder_draft; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/$name.json; done` | three lines printed, each >= 50 |
| AC3 | R1 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_descriptor_reads_target_layers_and_tensors/)'` | 1 passed; prints `target_layers = [L0, L1, L2] target_hidden_size = N` and asserts both equal the `target_layers` and `target_hidden_size` header fields of `hidden_tap.json` (oracle: the values llama.cpp `f1ea20621` returns through `llama_model_target_layer_ids(_n)` for `$PAIR_DFT`, recorded by the slice-2 harness) |
| AC4 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/gemma4_hidden_tap_matches_llama_fixture/)'` | 1 passed; prints `positions = N layers = 3 max_abs_diff < 1e-4` with N >= 50; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/hidden_tap.json`, the rows llama.cpp `f1ea20621` returns from `llama_get_embeddings_layer_inp` |
| AC5 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_encoder_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/encoder.json`, g_embd rows from llama.cpp `f1ea20621` |
| AC6 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_decoder_step_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50; oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json`, step logits from llama.cpp `f1ea20621` |
| AC7 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_draft_loop_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50, `p_min_early_stops` >= 1, `n_max_stops` >= 1; oracle: the drafted sequences in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` recorded from llama.cpp `f1ea20621`'s `common_speculative_impl_draft_eagle3` |
| AC8 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_accept_rewinds_boundary/)'` | 1 passed; prints `cases = N` with N >= 50; oracle: the accept-rewind positions recorded in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` from llama.cpp `f1ea20621` |
| AC9 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_state_stash_empty_for_attention_target/)'` | 1 passed; asserts proxima's stash length equals `stash_bytes` (0) recorded per case in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/decoder_draft.json` by llama.cpp `f1ea20621`'s `get_state`; prints `cases = N` with N >= 50 |
| AC10 | R8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_selects_draft_eagle3/)'` | 2 passed (the string `draft-eagle3` parses to `SpeculativeType::DraftEagle3`, the same type llama.cpp `f1ea20621` maps that string to at `common/speculative.cpp:33-45`; a config selecting it with no draft model path returns the typed error, matching llama.cpp's `GGML_ASSERT(ctx_tgt && ctx_dft ...)` requirement at `common/speculative.cpp:495`) |
| AC11 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_pairing_rejects_wrong_layer_count/)'` | 2 passed (a draft with `target_layer_ids_n != 3` and a draft with an id above the target's `n_layer` each return the typed error); oracle: `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json`, the accept/reject outcome llama.cpp `f1ea20621` records for each of the same two synthetic drafts (written by `proxima-gguf`'s writer), which proxima's outcome must equal |
| AC12 | R10 | `cargo run --release -p proxima-model-interop --features std --example speculative_acceptance_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model "$PAIR_DFT" "$PAIR_TGT" > acceptance_eagle3.txt; grep -cE 'mean_accepted_per_step=([1-9][0-9]*\.[0-9]+\|0\.[0-9]*[1-9])' acceptance_eagle3.txt` | 1 (the `summary` line prints a nonzero `mean_accepted_per_step=`; a printed `0.000000` matches 0 lines and invokes the refutation clause once AC18 shows the oracle's mean is nonzero) |
| AC18 | R10 | `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 -p "Repeat exactly five times: the quick brown fox jumps over the lazy dog." -n 40 2>&1 \| grep -cE 'n_accept +=  *[1-9]'` | 1 (the oracle itself accepts at least one drafted token on the same pair, which makes AC12's zero decidable against the refutation condition) |
| AC17 | R9 | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json` | 2 |
| AC13 | R12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_oracle_control_/)'` | 2 passed: `eagle3_oracle_control_self_` (llama's fixture ids on both sides pass) and `eagle3_oracle_control_perturbed_` (one flipped id returns a mismatch, asserted) |
| AC14 | R13 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_llama_parity_seed_/)'` | 3 passed (seeds 1, 2, 3); each prints `ids_compared = N` with N >= 8; oracle: ids vendored from `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 --seed <seed> -n 40` (llama.cpp `f1ea20621`) |
| AC15 | R13 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/ids.json` | 3 (every vendored oracle run drafted, so an oracle that drafted nothing cannot back AC14) |

## out of scope

- Qwen3/Llama3/gpt-oss EAGLE3 pairs -- proxima has no target profile for those families; gemma4 is the only pairing available today (see decisions)
- the `norm_before_residual`/`norm_before_fc` RedHat/eagle3.1 variants beyond reading the two optional GGUF bools and branching the two known cases (`src/models/eagle3.cpp:24-29`, `:131-134`, `:213-217`) -- no third variant is named upstream
- backend-offloaded top-k sampling (`params.draft.backend_sampling`, `common/speculative.cpp:526-538`) -- CPU-side top-k=10 sampling only; offload is a performance path, and llama.cpp itself falls back to CPU on offload failure (`:531-535`)
- the recurrent-target stash content (the non-empty branch of `get_state`/`set_state`, `common/speculative.cpp:880-919`) -- legitimately deferred: it needs an EAGLE3 sidecar for a recurrent or hybrid target, and none is named upstream (`docs/speculative.md:37-42` lists the named pairs: llama3.1/3.3, gemma4, qwen3), so no oracle run can exist; the dependency that unblocks it is such a pair, and R7 covers the attention-only branch now
- tree-shaped EAGLE3 drafts -- the tree shape and path-gather commit belong to `decode-as-data`; this sub-spec matches llama.cpp's linear draft
- DFlash / DSpark (`common/speculative.cpp:923+`) -- parent R11c/R11d, separate sub-specs
- the readout list itself (`decode-as-data` R5, slice 3) -- owned by that spec; this sub-spec consumes it

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| RedHatAI's gemma-4-31B-it target is too large to run repeatedly in the dev loop (unlike the 2B E2B fixture) | high | slow iteration; possibly the 26B-A4B MoE pairing instead, which the gemma4 MoE path must also support | slice 1 records actual download size and picks the smaller of the two named pairs first |
| gemma4's per-layer residual shape does not line up with llama.cpp's `n_embd` per-layer convention (a distinct pre/post-norm residual stream) | medium | R2's tap reads the wrong tensor and produces a plausible-but-wrong g_embd that still runs | AC4 compares to llama's rows at 1e-4, not just "compiles and runs" |
| the HF checkpoint format of RedHatAI's speculator matches neither documented format (SpecForge `LlamaForCausalLMEagle3` / vLLM `Eagle3LlamaForCausalLM`) | low | slice 1's conversion fails before any proxima code is touched | `convert_hf_to_gguf.py` is llama.cpp's own converter and declares both formats supported (`docs/speculative.md:24-25`) |

## context

- proxima: `proxima-tensor/src/spec/descriptor.rs:130` (`ModelDescriptor`), `proxima-tensor/src/op.rs:201` (`Op::Input`), `proxima-model-interop/src/lowering.rs:118-141` (`BoundProgram`, `hidden_root`, `residual_roots`), `proxima-model-interop/src/profiles/gemma4.toml`, `proxima-model-interop/src/bind.rs:298` (`find_tensor`), `proxima-model-interop/src/generate/draft_model.rs:174` (`load_draft`), `proxima-model-interop/src/speculative_settings.rs:36` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205` (`SpeculativeType`), `proxima-gguf/src/value.rs` (`MetadataArray`), `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-model-interop/examples/speculative_acceptance_corpus.rs`, `proxima-model-interop/examples/data/speculative_corpus.jsonl`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp:455-920` (`common_speculative_impl_draft_eagle3`: ctor `:486-561`, `process` `:593-736`, `draft` `:738-859`, `accept` `:861-876`, `get_state`/`set_state` `:885-919`), `:67-99` (`common_speculative_are_compatible`, not called by EAGLE3), `src/llama-ext.h:96-128`, `src/llama-model.h:709` (`target_layer_ids`), `src/llama-model.cpp:3387-3394`, `src/llama-arch.cpp:145` (`LLM_ARCH_EAGLE3 = "eagle3"`), `:365` (`LLM_KV_TARGET_LAYERS`), `src/models/eagle3.cpp:1-338` (`load_arch_hparams` `:3-36`, `load_arch_tensors` `:38-100`, encoder `:124-146`, decoder `:151-326`), `common/arg.cpp:4238` (`--spec-draft-model`), `:4246` (`--spec-type`), `:3094-3100` (`--eagle3` download flag), `examples/speculative-simple/speculative-simple.cpp:355`, `docs/speculative.md:16-53` (named pairs, conversion command)
- no local EAGLE3 draft GGUF exists on this machine as of 2026-09-28 (`~/.ollama/models/` and a home-directory search for `*eagle3*` found none) -- slice 1 obtains one
