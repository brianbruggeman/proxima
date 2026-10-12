# packet: model identity, real-model checkpoints and llama evidence

**Route:** Luna for artifact inventory, runner plumbing and recount tests;
Sol reviews attribution and model/backend pairing. Keep implementation runner
changes in the existing example/harness surface and do not create a production
Ouro-specific path.

**Owned paths:** `proxima-model-interop/examples/unified_execution_lab.rs`,
`proxima-model-interop/examples/unified_execution_recount.rs`,
`proxima-tensor/specs/ouro-mainline/unified-programmable/models.json`,
`results/models/`, `results/checkpoints/`, `results/llama/`, and the EVIDENCE /
DISCIPLINE / JSONL records. No edits to model execution sources in this packet.

**Observed local identities:** `models.json` records the exact digest of every
GGUF. Ouro is ByteDance/Ouro-1.4B revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16 SHA-256
`d3cdf276005d768c5def9cfb6679eb5600a424c5cd23b45157e65204708ae478`.
Qwen3-1.7B is `Qwen3ForCausalLM`, 28 layers, hidden width 2048; its source
revision is unrecorded and identity is the local config plus GGUF digest; local BF16
GGUF SHA-256 is
`ef2864916b951ebd2f072c759d11a442ba31acc3bcb7e5db5ca1d6fcf78b93cb` and
Q8_0 SHA-256 is
`75f92ec08d79252ca0a5ec6044c222864e0d9fe2bb5492e47462032f4cc51322`.
Granite's exact local artifact is `ibm-granite/granite-3.1-1b-a400m-instruct`,
GGUF SHA-256 `cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`,
`granitemoe`, 32 experts / 8 selected per token. Gemma is the local Gemma 4 E2B
QAT GGUF, SHA-256
`3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`, named
`2B_dequant_qat_it_hf` in the existing manifest. The source manifest is
`proxima-tensor/specs/decode-performance-lab/manifest.json`; its captured runs
are in `results-ngram-1x3/`. Do not replace either exact checkpoint with the
partial Gemma 4 26B-A4B HuggingFace metadata cache.

**Workloads:** Ouro recurrent dense with frozen French/ball inputs; Qwen3-1.7B
dense with tokenizer-derived literal prompts; Granite 3.1 1B-A400M MoE and
Gemma 4 E2B QAT with the real completion prefixes in the existing manifest;
and an additional longer three-stream mixed-arrival workload with exact literal
multi-turn messages, one cold/evicted stream and staggered prefill/decode. Save
tokenizer outputs once, compare Proxima and llama.cpp token arrays before any
timed pair, and reject a mismatch. Keep the same arrays throughout each pair.

**Record boundary:** one cohort run is one manifest workload invocation. Each
stream within it has its own `request_id`, continuation state, output and
per-token rows; all are joined by `cohort_id`. Thus 96 cohort runs expand to 192
stream-request records because each mixed workload cohort contains three
streams. The `single` and `mixed_three_stream` manifest entries are the two
workload summary keys.

**Record schema:** source/model digest and tokenizer; backend/device/driver;
precision and numeric permissions; concurrency/arrival trace/context/output cap;
requested and effective policy/plan; every request id, complete generated IDs,
decoded text and EOS; per-token and TTFT/total response duration; CPU/RSS and
allocation/resource rows; actual dispatch/kernel/source digest, copies/transfers,
plan and residency/cache decisions; all errors/drops; sample count and N-1 CoV%.
Retain failed runs and changed output trajectories in both directions.

**AC2 full-field recount:** implement `unified_execution_recount` options
`--policy-results` and `--policy-fixture` exactly as AC2 spells them. Assemble
`results/executor/policy-consumption.jsonl` from actual executor and serving
owner events/results, not from `field_owners`. Accept exactly 26 distinct
`execution_result` rows with all six required fields non-null. Require the
fixture's exact path set, actual effective values for `consumed`, named error
variants/fields for typed rejections, and source pointers that resolve to the
retained consumer/rejection events. Missing, duplicate, extra, owner_mapping
and unaccounted rows each have expected count 0. Run the independent literal
26-row schema control through the same row checker and retain its 26 accepted
rows, zero mismatches, and exactly one rejection after changing the first
row's status to `owner_mapping`. Mark literal controls separately; never count
their source pointers or values as actual execution evidence. Retain recount
and control output under `results/recount/policy-consumption/`.

**Invariants:** llama.cpp is performance oracle; paired settings must match
artifact, input IDs, precision, context, output cap, concurrency, arrival trace
and instrumentation. Existing Proxima is the before source condition. Separate
semantic vectors from performance. Host `GPU_EXEC_TICKS` is not GPU busy time.
Metrics lead to raw request/token payloads and source mechanism; do not declare
causality from aggregates. No model quality statement from short synthetic
prefixes.

**Validation:** AC2 and AC11-AC13. Before the shared runner is executable, retain
owner tests and existing dense and MoE compatibility checkpoints. Once executable, capture
dense and MoE real-model controls before the next source integration checkpoint.
The final AC12/AC13 matrix captures all four identities and both workload cells:
96 cohort runs and 192 per-stream request rows per runtime. Recount every
request/token row, policy result, and duration/CoV;
mutate one input digest/sample to prove the comparer fails. Preserve per-run
manifest, output IDs, errors and source hashes.

**Negative controls:** one changed input ID, one precision mismatch, one
concurrency mismatch and one altered raw timing row must each invalidate a
paired comparison; the independent oracle vector itself must pass the same
comparator; one-coordinate corruption is rejected exactly once.

**Artifacts:** use unique checkpoint directories and never overwrite prior
conditions: `results/checkpoints/checkpoint-N/`,
`results/llama/paired-N/`, `results/models/manifest-audit.jsonl`, and
`results/recount/`. Preserve the full command lines and output status.

**Isolation:** runner and recount work use `/private/tmp/proxima-unified-e2e`
on branch `codex/unified-e2e`, with build output
`/private/tmp/proxima-target-unified-e2e`. Model execution uses the shared
local artifacts and one-at-a-time hardware serialization; root integrates
runner changes into the dirty mainline checkout after reviewing each diff.
