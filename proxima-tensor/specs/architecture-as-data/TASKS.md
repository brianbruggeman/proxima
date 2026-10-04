# architecture as data: slices

Each slice is one commit, green at every commit, and lands by `git merge --ff-only` in the
checkout that holds main. Commands run from that checkout.

| # | slice | AC | validation | expected | done |
|---|---|---|---|---|---|
| 0 | At 9dd9deef, capture the incumbent op-graph digests and bound names/bytes for 7 checkpoints. Vendor into `proxima-model-interop/tests/fixtures/llama-parity/`, from llama.cpp f1ea20621: greedy ids per checkpoint, `gguf_kv.txt` per checkpoint, and `swa_layers.txt` for gemma4 E2B and 26B. Add the `arch_data_digest_` and `llama_parity_` tests, plus `generic_binder_` asserting a non-empty incumbent capture. Run AC8 (20 passed at 9dd9deef, measured by the spec auditor). | AC0, AC3 capture, AC6, AC4 control, AC8 | AC0; AC6; AC3; AC4; AC8 | 7 passed; 7 passed; 7 passed; 1214; 20 passed | [ ] |
| 1 | SWA RoPE `(base, dim)` from GGUF metadata into the descriptor rope table; delete the literal `1.0e4, 256` | AC1, AC0, AC6 | AC1, then AC0, then AC6 | 2; 7; 7 passed | [ ] |
| 2 | `ExpertResidency` dims from `block_count`/`expert_count`; error labels from `Architecture::name()` | AC0, AC6 | AC0, AC6 | 7; 7 passed | [ ] |
| 3 | one production descriptor builder; delete test-only `gemma4_descriptor`/`mistral_descriptor` | AC0 | AC0 | 7 passed | [ ] |
| 4 | family profile TOML (activation, scales, norm shift, rope pairing, value_norm); delete `force_split_half_rope`; GGUF and HF read one profile | AC0, AC6 | AC0, AC6 | 7; 7 passed | [ ] |
| 5 | windows from `attention.sliding_window` for every layer kind | AC7, AC0 | AC7, AC0 | 2; 7 passed | [ ] |
| 6 | generic verify (`last_row_only=false`) for any rewindable family | AC2 | AC2 | 3 passed | [ ] |
| 7 | one cache engine with descriptor `cache_mask` | AC0 | AC0 | 7 passed | [ ] |
| 8 | bind from lowered `Op::Input` leaves; delete per-arch name tables; generic fused gate_up split | AC3, AC6 | AC3, AC6 | 7; 7 passed | [ ] |
| 9 | `LayerKind::Gdn`; qwen35, then qwen35moe as descriptors; lfm2 registered through the generic path | AC0, AC6 | AC0, AC6 (lfm2 adds an 8th parity case) | 7 passed; 8 passed | [ ] |
| 10 | one generic `Architecture`; generic names for `qwen35moe_*` config and `Qwen35*` runtime types | AC5, AC6 | AC5, AC6 | 1; 8 passed | [ ] |
| 11 | tokenizer: drop the `"gemma4"` arm, read `tokenizer.ggml.pre`; presence-based `task.rs` | AC8, AC4, AC6 | AC8, then AC4, then AC6 | 20 passed; 0; 8 passed | [ ] |
| 12 | docs sweep | AC4 | AC4 | 0 | [ ] |
