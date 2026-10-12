# decode as data: slices

Each slice is one commit, green at every commit. Slices 1-3 stack on `architecture-as-data`
slice 7. Slices 4 onward stack on its slice 10b. Commands are spelled out in SPEC.md's
acceptance criteria.

| # | slice | AC | validation | expected | done |
|---|---|---|---|---|---|
| 0 | Vendor llama.cpp f1ea20621 fixtures into `proxima-model-interop/tests/fixtures/llama-parity/`:<br>- tree per-row ids (harness)<br>- `llama-lookahead` ids<br>- `llama-diffusion-cli` ids for LLaDA-8B-Instruct<br>Then add the AC3, AC5, AC6 and AC11 tests. | AC11, AC3, AC5, AC6 | AC11 first, then AC3, AC5 and AC6, each run once and expected to fail | 5 passed; then 9 failed, 9 failed, 3 failed. The harness is validated, and the feature is absent. | [ ] |
| 1 | `mask_pos`, `seq_member` and `seq_primary` input leaves; a visibility arm beside the iota arm, selected by `cache_mask` | AC1, AC2 | AC1, then AC2 | 7 passed; 3 passed | [ ] |
| 2 | Host shape pipes for chain, tree, lookahead layout and block. RoPE comes from a positions slice. | AC3 | AC3 | 9 passed | [ ] |
| 3 | `build_forward` readout list: logits rows, layer taps, extra head roots | AC7 | AC7 | 2 passed | [ ] |
| 4 | FSM `accept` returns row indices, commit gathers rows, and `Verify -> Verify` refine | AC4 | AC4 | 9 passed | [ ] |
| 5 | Lookahead proposer and accept as pipes | AC5 | AC5 | 9 passed | [ ] |
| 6 | Diffusion: block shape, remask accept (llama's 5 algorithms), refine loop | AC6 | AC6 | 3 passed | [ ] |
| 7 | Per-config static plan cache, one plan per shape | AC8 | AC8 | 1 passed | [ ] |
| 9 | Detokenize as one implementation (bulk = fold over streamed pieces); vendor llama `content` strings | AC15, AC16 | AC15, then AC16 | 3 passed; 0 | [ ] |
| 10 | Continuation tokenize with `add_bos = (cached_len == 0)` | AC17 | AC17 | 3 passed | [ ] |
| 11 | Remove `gdn_prefill_backend` from `ServingConfig` and from plan and cache keys | AC19 | AC19 | 0 | [ ] |
| 12 | Root-cause the 5x first-GDN-layer M-row divergence (`/root-cause-loop`); then one prefill path; delete `single_position_step`, `split_prefill` and the env switches | AC12, AC13 | AC12, then AC13 | 9 passed; 0 | [ ] |
| 13 | Placed KV returns a real `PrefixState` | AC14 | AC14 | 2 passed | [ ] |
| 14 | Both-sides threshold parity for the three lowering specializations | AC18 | AC18 | 3 passed | [ ] |
| 8 | Restate the MTP, EAGLE3, DFlash and DSpark mechanisms as readouts plus a draft descriptor, keeping their oracles | AC10, AC9 | AC10, then AC9 | 0; 0 | [ ] |
