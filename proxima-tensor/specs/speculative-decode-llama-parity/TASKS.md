# speculative-decode-llama-parity -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$EX` and `$E2B` as defined in SPEC.md acceptance criteria.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | sample-and-match verify via one shared per-row selection fn; remove plain-argmax gate; parity example runs greedy + sampled blocks and `--seed-mismatch-control` (includes the std-build fix and verify-step counting already on main uncommitted) | AC1, AC2 | `$EX` then `$EX --seed-mismatch-control` | per AC1, AC2 | [x] | AC1: no-flag run, both blocks identical=true, steps 4/2, exit 0. AC2: `--seed-mismatch-control` (seed 42 vs 4242), sampled identical=false at first_divergence=0, greedy unchanged identical=true steps=4, exit 0 |
| 2 | fixture generator: C++ harness compiled against llama.cpp `common/ngram-*.cpp` emitting JSON cases from gemma4-tokenized real text; vendored at `proxima-tokenizer/tests/fixtures/llama-ngram/` with upstream commit in each header | AC5-AC9 (inputs) | `for f in proxima-tokenizer/tests/fixtures/llama-ngram/fixtures/ngram_*.json; do jq '.cases \| length' $f; done` | 5 files, each ≥ 200 | [x] | 4346/4395/4546/2081/4800 cases; non-empty 559/391/365/91/137; k vs k4v drafts differ in 24 cases; ngram_mod occupancy_resets=1 low_accept_resets=1; 2.4M; commit f1ea20621. streams 6-11 are constructed from real sentences to force k4v's tie guard (`ngram-map.cpp:495-499`); ngram_mod's low-accept reset uses a constructed stream (streams[5]) |
| 3 | `draft/ngram_simple.rs`; delete `draft_ngram_lookup` and repoint decode loop | AC5, AC11 | AC5 then AC11 | 1 passed, cases ≥ 200; 0 matches | [x] | tokenizer 115 passed; fixture cases=4346 non_empty=559; AC11 grep 0 matches; clippy/doctest/model-interop filter all green; found+fixed a real decode.rs bug (`token_history` already held `next_ids[0]` when drafting was called, corrupting every pattern -- speculation silently drafted nothing end-to-end on the real gemma4-E2B checkpoint until fixed); `speculative_decode_parity` (no flag, and `--seed-mismatch-control`) both pass, `speculative_verify_steps` > 0 on greedy AND sampled; `ngram_simple_draft` takes a caller-owned `&mut Vec<u32>` output buffer, zero per-call allocation, per mid-slice owner directive |
| 4 | `draft/ngram_map.rs` key-only | AC6 | AC6 | 1 passed | [ ] | |
| 5 | `draft/ngram_map.rs` k4v | AC7 | AC7 | 1 passed | [ ] | |
| 6 | `draft/ngram_mod.rs` | AC8 | AC8 | 1 passed, resets ≥ 1 each | [ ] | |
| 7 | `draft/ngram_cache.rs` + llama cache file loader | AC9 | AC9 | 2 passed | [ ] | |
| 8 | ServingConfig speculative section + builder/loader parity; delete env var | AC12, AC13 | AC12 then AC13 | 1 passed; 0 matches | [ ] | |
| 9 | `Drafter` enum driven by the decode loop (begin/draft/accept); `--drafter` flag | AC10 | AC10 loop | 5 runs identical | [ ] | |
| 10 | telemetry draft_n / draft_n_accepted; `--telemetry-file` flag | AC16 | AC16 | counts equal | [ ] | |
| 11 | recurrent snapshot rollback (attention truncate already covered by AC4a, 3 existing tests) | AC4a, AC4b | AC4a then AC4b | ≥ 3 passed; ≥ 2 passed | [ ] | read upstream test-recurrent-state-rollback first |
| 12 | enumerate `Architecture` impls; add `verify_program_bound_for_every_architecture` test (fails until slice 13+ complete; lands with the last arch) | AC3 | `git grep -c 'impl Architecture for' proxima-model-interop/src` | count recorded in note | [ ] | one follow-on slice per arch, rows added here when counted |
| 13 | draft-simple second model + vocab compatibility typed error | AC14 | AC14 | identical; 1 passed | [ ] | |
| 14 | sub-spec draft-mtp, audited | AC15 | `grep -c '^status: audited' proxima-tensor/specs/speculative-draft-mtp/SPEC.md` | 1 | [ ] | |
| 15 | sub-spec draft-eagle3, audited | AC15 | same, eagle3 | 1 | [ ] | |
| 16 | sub-spec draft-dflash, audited | AC15 | same, dflash | 1 | [ ] | |
| 17 | sub-spec draft-dspark, audited | AC15 | same, dspark | 1 | [ ] | |
| 18 | corpus (≥ 50 prompts: chat, code, RAG) + `speculative_acceptance_corpus` example | AC17 | AC17 | prompts ≥ 50; 5 rows | [ ] | refutation check |

## resume

Last landed slice: 3
Next action: slice 4 -- `draft/ngram_map.rs` key-only (AC6)
Open question, if any: none

## struck

- slice 0 (stopgap gate + std fix + probe counting) folded into slice 1: it discharged no AC on its own
