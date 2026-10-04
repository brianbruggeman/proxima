# architecture as data: slices

Each slice is one commit, green at every commit, and lands by `git merge --ff-only` in the
checkout that holds main. Commands run from that checkout.

| # | slice | AC | validation | expected | done |
|---|---|---|---|---|---|
| 0 | At 9dd9deef, capture the incumbent op-graph digests and bound names/bytes for 7 checkpoints. Vendor into `proxima-model-interop/tests/fixtures/llama-parity/`, from llama.cpp f1ea20621: greedy ids per checkpoint, `gguf_kv.txt` per checkpoint, and `swa_layers.txt` for gemma4 E2B and 26B. Add the `arch_data_digest_` and `llama_parity_` tests, plus `generic_binder_` asserting a non-empty incumbent capture. Run AC8 (20 passed at 9dd9deef, measured by the spec auditor). | AC0, AC3 capture, AC6, AC4 control, AC8 | AC0; AC6; AC3; AC4; AC8 | 7 passed; 7 passed; 7 passed; 1214; 20 passed | [x] 2026-10-04: AC0 7/7, AC3 7/7, AC4 1214, AC8 20/20; AC6 4/7 (gemma4 26b divergence D2, qwen35+qwen35moe no oracle O1) |
| 1 | SWA RoPE `(base, dim)` from GGUF metadata into the descriptor rope table; delete the literal `1.0e4, 256` | AC1, AC0, AC6 | AC1, then AC0, then AC6 | 2; 7; 7 passed | [x] 2026-10-04: AC1 2/2, AC0 7/7, AC6 subset 4/4, clippy 0, lib 397/397 |
| 2 | `ExpertResidency` dims from `block_count`/`expert_count`; error labels from the descriptor's family string | AC0, AC6 | AC0, AC6 | 7; 7 passed | [x] 2026-10-04: AC0 7/7, AC6 subset 4/4, clippy 0, lib 400/400; reconcile reuses preallocated buffers (test asserts stable ptr+capacity at 40x256); "qwen35moe" literals 61->22 |
| 3 | one production descriptor builder; delete test-only `gemma4_descriptor`/`mistral_descriptor` | AC0 | AC0 | 7 passed | [x] 2026-10-04: isolated staged-tree gate: clippy 0, alloc check ok, tensor 703/703, digest+real-dims 9/9, interop lib 400/400 |
| 4 | family profile TOML (activation, scales, norm shift, rope pairing, value_norm); delete `force_split_half_rope`; GGUF and HF read one profile | AC0, AC6 | AC0, AC6 | 7; 7 passed | [ ] |
| 4b | the descriptor is the whole pre-lowering program: serde (no_std+alloc) over every field build_forward reads (layer schedule, mixers, FFN, norms, embedding\/head, cache layout and mask, verify shape, step-input tables); lowering is a pure function of (config, weight directory) | AC9, AC11, AC0 | AC9, then AC11, then AC0 | 7 passed; exit 0 and 0; 9 passed | [ ] |
| 4c | conflaguration at the std boundary: `Settings` + `Validate` with layers profile -> GGUF -> env -> explicit TOML, plus a fluent builder that round-trips (was 10a) | AC9, AC10 | AC9, then AC10 | 7 passed; 2 passed | [ ] |
| 5 | windows from `attention.sliding_window` for every layer kind | AC7, AC0 | AC7, AC0 | 2; 7 passed | [ ] |
| 6 | generic verify (`last_row_only=false`) for any rewindable family | AC2 | AC2 | 3 passed | [ ] |
| 7 | one cache engine with descriptor `cache_mask` | AC0 | AC0 | 7 passed | [ ] |
| 8 | bind from lowered `Op::Input` leaves; delete per-arch name tables; generic fused gate_up split | AC3, AC6 | AC3, AC6 | 7; 7 passed | [ ] |
| 9 | `LayerKind::Gdn`; qwen35, then qwen35moe as descriptors; lfm2 described by config, reaching the same FSM | AC0, AC6 | AC0, AC6 (lfm2 adds an 8th parity case) | 7 passed; 8 passed | [ ] |
| 10a | (moved to 4c) | - | - | - | [-] |
| 10b | wire `ServingState` as the live serving loop, driven by the descriptor config; delete the `run_decode_loop_from_ids` closure | AC5 (FSM half), AC6, AC2 | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/serving_fsm_drives_/)'`, then AC6, then AC2 | 2 passed; 8 passed; 3 passed | [ ] |
| 10c | delete the `Architecture` trait, its 4 impls and `ArchitectureRegistry` (family profile lookup is a config layer keyed by the GGUF string); generic names for `qwen35moe_*` config and `Qwen35*` runtime types | AC5 | AC5 | 0; 9 passed | [ ] |
| 11 | tokenizer: drop the `"gemma4"` arm, read `tokenizer.ggml.pre`; presence-based `task.rs` | AC8, AC4, AC6 | AC8, then AC4, then AC6 | 20 passed; 0; 8 passed | [ ] |
| 12 | docs sweep | AC4 | AC4 | 0 | [ ] |
