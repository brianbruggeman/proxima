# packet: shared programmable policy and generated floor

**Route:** Luna for bounded settings/build/default work. Sol reviews field
ownership and precedence before landing because this extends a public config
consumed across Omega and model serving.

**Owned paths:** `omega/src/config.rs`, `omega/src/config_tests.rs`,
`omega/omega-runtime.toml`, `omega/build.rs`, `omega/src/sized.rs`,
`omega/Cargo.toml` only via `cargo add`, `docs/configuration.md`, and the
`ai_docs` route/invariant records. Do not edit serving, decode, planner or driver
paths in this packet.

**Existing surface:** extend `omega::config::RuntimeConfig`, its
`Settings`/`Validate`/serde derivations, `RuntimeConfigBuilder`, and the
`[sized]` generated-default bridge. Keep existing public fields source
compatible. Keep runtime resolution at a cold boundary; no environment lookup
per request or inner-loop. `ServingConfig::omega_runtime` is the downstream
consumer; do not create a second settings type.

**Contract:** defaults from generated constants, then file values, then
`OMEGA_*`, then explicit fluent values/serving overrides. Validation separates
resource ceilings and numeric permissions from workload objectives and
strategy requests. Requested mandatory-but-unavailable capabilities fail with
the existing typed config/planning error surface. Automatic candidate reports
name actual choice and exclusions. Every new field participates in serde and
cache/effective-policy identity where it changes output/layout/backend.

**Worked inputs:** literal config requests 3 active sequences, 2 in-flight
submissions, 4 CPU workers and 64 resident bytes. File says 2 workers, env says
3, fluent says 4; resolved worker budget is exactly 4. A resource request of
64 bytes is accepted for a 64-byte plan; 63 rejects before allocator count
changes. Numeric reassociation remains false while a decode-kernel strategy is
selected. CPU-only host plus `required_driver=cuda` is a typed error with zero
CPU fallback dispatches.

**Invariants:** no new `RuntimeConfig` competitor; conflag defaults are seeded
from generated sizing constants; `rerun-if-env-changed` covers every consulted
override; no `std` config in alloc-only builds; config builder and deserialized
fixture produce the same struct; no policy identity omits effective numeric,
layout or backend fields.

**Dependency order:** after the unified SPEC/TASKS admission. Land before any
serving/driver consumer. Root serializes edits to shared `omega/src/config.rs`
and `omega/src/backend.rs` and runs real-model dense and MoE generation after
the combined implementation checkpoint, before the next source integration.
Use existing mainline compatibility commands before the unified runner is
executable; use its dense/MoE controls thereafter and the full final matrix.

**Validation:** AC1 and AC12 in `SPEC.md`. Also run the existing Omega config
test selection and `cargo check -p proxima-omega --no-default-features --features
alloc --message-format=json`. Record exact counts and changed source lines.

**Negative controls:** mutate one override source so the resolved worker count
is 3, not 4; request a zero capacity; deny required numeric permission; require
unavailable CUDA on a host without CUDA. Each must be rejected or differ at
the exact named owner. Run the independent config fixture itself through the
same comparison function and require zero mismatches.

**Artifacts:** `results/config/` for raw config/source/effective-plan JSONL,
command logs, comparator controls and generated-default capture; preserve every
failure and source digest.

**Isolation:** after admission, own worktree is
`/private/tmp/proxima-unified-config` on branch `codex/unified-config`, with
`CARGO_TARGET_DIR=/private/tmp/proxima-target-unified-config`. Return a patch and
exact command output; root applies reviewed hunks to the preserved dirty tree.
