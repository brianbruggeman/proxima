# speculative-draft-eagle3 -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$PAIR_TGT` / `$PAIR_DFT` as defined in SPEC.md acceptance criteria.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | obtain the named target+draft pair: download `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` (smaller of the two named pairs, per risks table) + its matching gemma-4-26B-A4B-it target, convert both with `convert_hf_to_gguf.py --target-model-dir` per `docs/speculative.md:23-32`; record both paths | AC1 | `ls -1 "$PAIR_TGT" "$PAIR_DFT" 2>/dev/null \| wc -l` | `2` | [ ] | record exact HF repo ids + converted file paths + file sizes in this note once run |
| 2 | fixture generator: C++ harness linking llama.cpp `f1ea20621`, calling `llama_set_embeddings_layer_inp`/`llama_set_embeddings_nextn`/`llama_get_embeddings_nextn(_ith)` directly plus the real `common_speculative_impl_draft_eagle3` end-to-end on real prompts against the pair from slice 1; dumps 3 JSON fixtures (hidden taps, encoder g_embd rows, full draft-loop token sequences) to `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/`, upstream commit in each header | AC2 | `for f in proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/*.json; do jq '.cases \| length' "$f"; done` | 3 files, each ≥ 50 | [ ] | |
| 3 | `Eagle3Arch` metadata + tensor parse only (R1); no forward program yet | AC3 | AC3 | 1 passed | [ ] | |
| 4 | gemma4 all-positions hidden-state tap at caller-chosen layer indices (R2) | AC4 | AC4 | 1 passed, `max_abs_diff < 1e-4` | [ ] | |
| 5 | eagle3 encoder graph: 3-layer fuse -> optional RMSNorm -> `fc` -> g_embd (R3, half) | AC5 (encoder half) | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_encoder_matches_llama_fixture/)'` | 1 passed, cases ≥ 50 | [ ] | |
| 6 | eagle3 decoder graph: single autoregressive step, concat(embd_norm, g_norm) -> attn -> ffn -> output_norm -> lm_head -> optional `d2t` (R3, other half) | AC5 (decoder half) | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_decoder_step_matches_llama_fixture/)'` | 1 passed, cases ≥ 50 | [ ] | |
| 7 | full `draft()` loop: deferred-boundary cross-ubatch bridge, seed step, top-k=10 autoregressive sampling, `p_min` early-stop, `n_min`/`n_max` bounds (R4) | AC6 | AC6 | 1 passed, cases ≥ 50, both stop conditions ≥ 1 | [ ] | |
| 8 | `accept()` boundary rewind + recurrent/hybrid-only `get_state`/`set_state` stash (R5) | AC7 | AC7 | 2 passed | [ ] | |
| 9 | `Drafter::DraftEagle3(..)` variant + eagle3-specific pairing compatibility check (R6, R7) | AC8 | AC8 | 2 passed | [ ] | |
| 10 | wire `--drafter draft-eagle3` into `examples/speculative_decode_parity.rs`; end-to-end parity run | AC9 | AC9 | identical = true, steps ≥ 1, exit 0 | [ ] | |
| 11 | `speculative_acceptance_corpus` example gains `--drafter`/`--draft-model` flags; refutation measurement recorded | AC10 | AC10 | `mean_accepted_per_step` > 0.0 | [ ] | if this is 0, stop and redesign R2 per the refutation condition -- do not proceed to promote this spec's status |
| 12 | promote `status: draft` -> `status: audited` after `spec-auditor` admits this file | AC15 | `grep -c '^status: audited' proxima-tensor/specs/speculative-draft-eagle3/SPEC.md` | `1` | [ ] | run `spec-auditor` first; fix any refusal before flipping status |

## resume

Last landed slice: none
Next action: slice 1 -- resolve and record the exact HF repo ids for the gemma-4-26B-A4B-it target + its eagle3 speculator, download, convert with `convert_hf_to_gguf.py --target-model-dir`
Open question, if any: whether the 26B-A4B (MoE) pairing exercises gemma4's MoE forward path in a way that complicates R2's hidden-state tap (the 31B dense pairing may be the simpler first target despite its size) -- resolve during slice 1 by checking actual download sizes and whether proxima's gemma4 MoE bind already produces per-layer residual output shapes compatible with R2

## struck

-
