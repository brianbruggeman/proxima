# serve-flint-through-omega

status: admitted
owner: brian
created: 2026-11-07

## problem

RAGD's in-process `ProximaProvider` must generate text from the flint checkpoint
(`general.architecture = flint`, 49 blocks, 256 experts / 6 used / 2 shared,
2 leading dense blocks, sliding-window attention with per-head sinks) through the generic
proxima graph and omega backend, with zero family-name branches in Rust, reproducing the
llama.cpp greedy id sequence and serving through `POST /v1/chat/send` with ollama stopped.

## refutation condition

A working flint load or generation that requires a conditional on the string `flint`
outside `proxima-model-interop/src/profiles/` and the test fixtures refutes the data-only
premise: the spec must be replaced by an engine-redesign spec rather than patched.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Real flint fixtures (52 metadata entries, 779-tensor directory, llama.cpp greedy ids at temp 0, llama.cpp tokenizer ids for one sample text) are committed under `proxima-model-interop/tests/fixtures/flint/`; a fixture test asserts `architecture=flint`, `block_count=49`, `expert_count=256`, `expert_used_count=6`, `expert_shared_count=2`, `leading_dense_block_count=2`, `attention.sliding_window=513`, `tensor_count=779`. | yes |
| R2 | The generic sliding-pattern header reader builds a descriptor for a checkpoint that omits `attention.key_length_swa`, `attention.value_length_swa`, and `rope.dimension_count`; the descriptor reports key/value length 128 and rope dimension 128 on both sliding and full layers (literal values from the flint header), explicit swa keys still win, and the gemma4 fixture reports its own literal values. | yes |
| R3 | The generic reader reads `leading_dense_block_count=2`, `expert_shared_count=2`, `expert_shared_feed_forward_length=1024`, and `expert_gating_func` (generic value table) from the flint header, and records sink presence from a `blk.0.attn_sinks.weight` leaf; absent keys keep pre-change defaults. | yes |
| R4 | `{arch}.attention.rope_pattern` selects rope per layer from data; the flint descriptor's per-layer selection equals the committed 49-entry array, and a checkpoint without the key is unchanged. | yes |
| R5 | The scheduled engine lowers `FfnCombination::RoutedWithSharedExpert` (replacing the refusal at `proxima-tensor/src/spec/attention_forward.rs:1301`) through the generic shared-expert primitives, selected by profile data, and the resulting FFN output equals a hand-worked literal reference for one synthetic layer; dense-only and routed-only programs are unchanged. | yes |
| R6 | Attention sinks are a generic capability: a descriptor knob (default off), a `blk.{layer}.attn_sinks.weight` program leaf, a CPU evaluator equal to a hand-worked softmax-with-sink literal, and omega MSL emission equal to that same literal on Metal. | yes |
| R7 | Loading the real flint GGUF binds all 779 tensors with zero `UnknownTensor`, zero unbound leaves, and zero unclaimed tensors. | yes |
| R8 | Flint's non-header values live only in `proxima-model-interop/src/profiles/flint.toml`; no source file outside `profiles/` and test fixtures contains the literal `flint`. | yes |
| R9 | `tokenizer.ggml.pre = "flint"` resolves through the existing pre-type table; encoding the fixture sample text equals the llama.cpp ids and decoding round-trips. | yes |
| R10 | The committed llama.cpp greedy ids pass the same id comparator the parity test uses (the oracle control), and `LoadedModel::load` + generate on the flint GGUF reproduces those ids on CPU and on Metal with neither arm skipped. | yes |
| R11 | RAGD's `ProximaProvider` serves flint end to end: `GET /v1/llm` reports three roles, all `provider: proxima`; `POST /v1/chat/send` returns at least one part with route tag `llm-only`, with ollama stopped. | yes |
| R12 | RAGD builds against the proxima revision that carries R10 and keeps its provider-resolution tests green. | yes |

## architecture

The work is data and generic-engine capability, not a family module.

- **Header story.** `general.architecture` keys the family profile
  (`proxima-model-interop/src/profiles/mod.rs`). Flint's header (49 blocks, sliding-window
  pattern array, per-layer rope pattern, routed FFN with shared experts, leading dense
  blocks, per-head sinks) is read by the existing `ScheduleSource::SlidingPattern` reader
  extended with optional keys. Every read is keyed on `{arch}.…` metadata already in the
  header; nothing branches on the family name.
- **FFN story.** The descriptor already carries `leading_dense_block_count` and
  `expert_shared_feed_forward`; the scheduled engine already selects dense vs routed per
  layer. Only `FfnCombination::RoutedWithSharedExpert` is unimplemented in that engine —
  the shared-expert lowering exists in the hybrid engine and moves into the scheduled
  engine as a generic combination, selected by the profile's `combination` value.
- **Sink story.** A sink is one learned scalar per query head added to the attention
  denominator (and thus the normalizer) before the value mix. It becomes a descriptor flag
  plus a leaf the engine binds; the CPU evaluator and omega MSL renderer consume the same
  descriptor flag, so no renderer names flint. Both validate against one hand-worked
  literal, not against each other.
- **Tokenizer story.** `tokenizer.ggml.pre` is a name table lookup in
  `proxima-tokenizer/src/gguf.rs`; flint is a table entry, not a code path.
- **Serving story.** `ragd_llm::providers::proxima::ProximaProvider` already loads a GGUF
  and calls `LoadedModel::generate_with_serving_config` with omega dispatch. RAGD needs
  only a proxima pin that carries the engine support and the daemon configured with
  `provider = proxima`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| where flint-specific values live | `proxima-model-interop/src/profiles/flint.toml` (data) | a Rust module per family is the layout the profiles migration deleted (`sliding_pattern/program.rs` header records the deleted bespoke graph) |
| sink representation | descriptor flag + leaf consumed by the generic attention builder | a per-renderer branch would put the family name in omega and in every backend |
| oracle | committed llama.cpp ids, re-checked by an oracle-control test, plus one hand-worked literal per engine change | comparing the new path to itself (CPU vs Metal, or "unchanged") cannot detect a shared error |
| leading dense blocks | reuse `ModelDescriptor::leading_dense_block_count` | a new schedule source duplicates the existing selector |
| shared experts | implement `RoutedWithSharedExpert` in the scheduled engine | routing flint through the hybrid engine invents recurrence flint does not have |
| fixtures | committed real captures, not synthetic tensors | principle 9: a test with fake bytes tests plumbing, not the contract |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_fixture_header_)'` | 1 passed, 0 failed |
| AC2 | R2 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(sliding_pattern_header_)'` | 2 passed, 0 failed |
| AC3 | R3 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_reader_leading_dense_and_shared_)'` | 1 passed, 0 failed |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_reader_rope_pattern_)'` | 1 passed, 0 failed |
| AC5 | R5 | `cargo nextest run -p proxima-tensor --lib -E 'test(routed_with_shared_expert_literal_)'` | 1 passed, 0 failed |
| AC6 | R6 | `cargo nextest run -p proxima-tensor --lib -E 'test(attention_sink_descriptor_) or test(attention_sink_program_)'` | 2 passed, 0 failed |
| AC7 | R6 | `cargo nextest run -p proxima-tensor --lib -E 'test(attention_sink_cpu_literal_)'` | 1 passed, 0 failed |
| AC8 | R6 | `cargo nextest run -p proxima-omega --lib -E 'test(attention_sink_metal_parity_literal_)'` | 1 passed, 0 failed |
| AC9 | R7 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_bind_all_tensors_)'` | 1 passed, 0 failed |
| AC10 | R8 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_profile_is_data_)'` | 1 passed, 0 failed |
| AC11 | R9 | `cargo nextest run -p proxima-tokenizer --lib --features gguf -E 'test(flint_pre_)'` | 1 passed, 0 failed |
| AC12 | R10 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_oracle_control_)'` | 1 passed, 0 failed |
| AC13 | R10 | `cargo nextest run -p proxima-model-interop --lib --features std,metal -E 'test(flint_greedy_parity_)'` | 2 passed, 0 failed |
| AC14 | R12 | `bash -c 'CARGO_TARGET_DIR=/private/tmp/flint-ragd-target cargo build --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_daemon_memory >/dev/null && ls /private/tmp/flint-ragd-target/debug/ragd \| wc -l'` | `1` |
| AC15 | R11 | `bash -c '! curl -sf http://127.0.0.1:11434/api/tags >/dev/null && curl -sS http://127.0.0.1:9091/v1/llm \| jq -r "if all(.[]; .config.provider==\"proxima\") then length else \"mixed\" end"'` | 3 |
| AC16 | R11 | `bash -c '! curl -sf http://127.0.0.1:11434/api/tags >/dev/null && curl -sS -X POST http://127.0.0.1:9091/v1/chat/send -H "Content-Type: application/json" -d "{\"message\":\"Reply with the single word: flint.\"}" \| jq -r ".data.turn \| \"\(.parts\|length) \(.route.tag)\""'` | `1 llm-only` |
| AC17 | R9 | `cargo nextest run -p proxima-tokenizer --lib --features gguf -E 'test(control_flint_tokenizer_)'` | 1 passed, 0 failed |
| AC18 | R5 | `cargo nextest run -p proxima-tensor --lib -E 'test(control_routed_with_shared_expert_)'` | 1 passed, 0 failed |
| AC19 | R6 | `cargo nextest run -p proxima-tensor --lib -E 'test(control_attention_sink_cpu_)'` | 1 passed, 0 failed |
| AC20 | R6 | `cargo nextest run -p proxima-omega --lib -E 'test(control_attention_sink_metal_parity_)'` | 1 passed, 0 failed |
| AC21 | R1 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_fixture_control_)'` | 1 passed, 0 failed |
| AC22 | R12 | `cargo nextest run --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_llm -E 'test(providers::proxima::tests)'` | 9 passed, 1 skipped, 0 failed |
| AC23 | R2 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(control_rejects_mutated_sliding_pattern_header)'` | 1 passed, 0 failed |
| AC24 | R3 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(control_flint_reader_leading_dense_)'` | 1 passed, 0 failed |
| AC25 | R4 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(control_rejects_mutated_rope_pattern)'` | 1 passed, 0 failed |
| AC26 | R6 | `cargo nextest run -p proxima-tensor --lib -E 'test(control_rejects_planted_sink_node)'` | 1 passed, 0 failed |
| AC27 | R7 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(control_rejects_missing_tensor)'` | 1 passed, 0 failed |
| AC28 | R8 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(control_finds_planted_literal)'` | 1 passed, 0 failed |
| AC29 | R11 | `cargo nextest run --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_daemon_memory -E 'test(control_proxima_serving_roles_)'` | 1 passed, 0 failed |
| AC30 | R11 | `cargo nextest run --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_daemon_memory -E 'test(control_proxima_serving_text_)'` | 1 passed, 0 failed |

AC1–AC30 are prospective: each slice adds its test and runs the command; the count
becomes recovered acceptance only after that run. Every correctness AC has a control
that feeds its reference through the same check: AC21 for AC1, AC23 for AC2, AC24 for
AC3, AC25 for AC4, AC18 for AC5, AC26 for AC6, AC19 for AC7, AC20 for AC8, AC27 for
AC9, AC28 for AC10, AC17 for AC11, AC29 for AC15, AC30 for AC16. AC12 is the control
for AC13 by construction; AC22 is a regression suite whose assertions are their own
checks. AC15/AC16 require the daemon and the flint GGUF; their commands assert ollama
is stopped as part of the command, not as prose.
## out of scope

- Any flint-specific Rust module, renderer, or matcher.
- CUDA or WGPU arms for the sink term (only CPU and Metal are in this spec).
- Model conversion, quantization, or tokenizer training.
- Changing the ollama provider; it remains available for other models.
- RAGD source changes beyond the proxima pin and daemon configuration.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| `rope_pattern` semantics differ from the sliding pattern | medium | wrong per-layer rope → wrong ids | AC4/AC11 compare literals from the header, then AC13 ids |
| sink term differs from llama.cpp's formulation | medium | silent numeric divergence | AC7/AC8 one hand-worked literal; AC13 end-to-end ids |
| shared-expert lowering reorders the residual | medium | parity failure | AC5 hand-worked literal, then AC13 |
| sink kernel is expensive on Metal | low | slower decode | measure after parity; no perf claim in this spec |
| the pinned proxima in ragd predates the engine | high | rebuild required | AC14 pins the rev that carries AC13 |

## context

Entry order for a fresh session (absolute paths):

1. `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/serve-flint-through-omega/SPEC.md`
   and `TASKS.md`.
2. `/Users/brianbruggeman/.claude/agents/spec-auditor.md` — ADMIT before code.
3. Per slice: implement, run the slice command, then `/Users/brianbruggeman/.agents/skills/validate-slice/SKILL.md`.
4. Card close: `/Users/brianbruggeman/.agents/skills/coherent-cards/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/coherent-commit/SKILL.md`, then
   `/Users/brianbruggeman/.agents/skills/postmortem/SKILL.md`.
5. Design and measurement: `/Users/brianbruggeman/.agents/skills/guiding-principles/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/algorithm-development/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/bench-metrics/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/instrument/SKILL.md`.
6. Domain agents: `/Users/brianbruggeman/.claude/agents/proxima-architect.md`,
   `proxima-critic.md`, `proxima-test-writer.md`, `card-worker-lean.md`, `postmortem.md`.

Source pointers (absolute):

- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/profiles/mod.rs`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/sliding_pattern/hparams.rs`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec/attention_forward.rs:1301`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec/descriptor.rs:148`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate/pregather.rs:2746`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tokenizer/src/gguf.rs:17-62`
- `/Users/brianbruggeman/repos/slot-0/ragd-wt-final/core.ragd.ai/libs/ragd_llm/src/providers/proxima.rs:257`

Work tracker: PM items `PROXIMAOMEGA-4..15` and `RAGD-80/81` under the parent
"Serve flint through omega" (`01a1b8b0-de83-77b1-ae92-28f4eb8f23ea`), each carrying the
`flint` facet; read them back with `GET /work-with/flint/true`.
