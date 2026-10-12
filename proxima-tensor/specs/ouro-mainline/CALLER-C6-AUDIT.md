# caller C6 admission audit

## Re-audit after configured-pair recount guard

**ADMIT.** The new setup guard verifies the two configured arms differ in the
recorded tile build gate while retaining the same default threshold, Omega
runtime configuration, and complete serving settings. The duplicated-off
control is rejected through that same guard.

## Opened artifacts

- `CALLER-SPEC.md:29` requires gate off/on with identical requests/numeric
  policy, both trajectories, and full diagnostic vectors.
- `CALLER-SPEC.md:96-103,117-150` registers CAC6, exact paired commands, per-arm
  counts, recount command, source provenance requirements, timing cells,
  trajectory pairs, full coordinate comparisons and comparator controls.
- `TASKS.md:12` registers the configured planner/tile gate slice and its exact
  count expectations.
- `performance.rs:105-110` writes observed model/checkpoint digests, Omega
  config, settings, `build_dense_float_storage` and `build_tiled_min_tokens`.
- `performance.rs:122-129` writes the complete initial-prefix logits vector;
  C6 explicitly scopes this to the baseline terminal/generation root, while
  C2 separately tests LoopCD references and hidden roots.
- `performance.rs:133-210` writes 12 requests, 2 summaries, and one completion
  per invocation, including actual IDs, callback events, latencies, wall time,
  resource snapshots and errors.
- `cached.rs:16-157` validates each input's completion count, distinct cases,
  one warmup/diagnostic/summary per case, five unique measured repetitions,
  valid token/latency arrays and per-request provenance.
- `cached.rs:158-180,196-306` validates paired model/checkpoint digests, IDs,
  prefixes/budgets; calculates timing statistics independently; emits every
  generated-ID pair and every changed diagnostic coordinate with signed delta;
  and records equality plus one-coordinate corruption controls.

## Earlier refusal and closure evidence

The previous refusal was based on absent gate provenance checks. The CAC6
commands at `CALLER-SPEC.md:117-128` explicitly set gate values 0 and 1 and share
the same feature, config path, request budget and sample count. The prior
`configured_pair` guard now closes the missing recount condition:

- `examples/omega_programmable_recount/cached.rs:207-220` requires exactly one
  `cached_setup` in each arm.
- `cached.rs:223-226` requires gate false/true and threshold 160 in both arms.
- `cached.rs:227-232` requires non-null and equal Omega config, equal complete
  serving settings, and `gpu` / `metal` / `configured` selection. Since the
  serialized Omega config is compared as a whole, its numeric policy and
  resource limits must match; `settings` is also compared as a whole.
- `cached.rs:323-340` applies the guard to the actual pair, proves a duplicated
  off arm is rejected, and retains that negative-control result alongside the
  vector comparator's identical and one-coordinate-corruption controls.
- `CALLER-SPEC.md:139-152` registers the recount command and exact payload counts;
  `TASKS.md:12` binds those counts and the same-policy/default-threshold
  condition to the slice.

The guard reads actual `cached_setup` rows emitted by
`examples/ouro_loopcd_lab/performance.rs:105-110`, which include gate,
threshold, Omega config and debug-formatted serving settings. Timed request and
diagnostic payload checks remain independent in `cached.rs:196-306`.

## Audit boundary

This is specification admission based on the current source and assertions. The
configured off/on GPU cohorts and native recount still need their recorded
execution evidence before the slice has runtime evidence.
