# Work packet: consume fusion and numerical policy at binding

## Why this is a dependency

`RuntimeConfig::allow_fusion` describes algebraic fusion. The existing
`bind_with_fusion` boolean only controls cached attention. Passing the config
boolean into that parameter would leave chain, reduction-epilogue, GDN and MoE
fusion enabled. CPU evaluation also binds with `NumericPolicy::bit_exact()`
inside `quantized_eval.rs`, independently of Omega's planner argument.

## Owned source and exact change

The execution worker integrates this dependency before its backend callers.
Use the existing binder and evaluator; do not add an alternate interpreter.

- `proxima-tensor/src/bind/builder_compose_window.rs`: extend
  `BoundOpBuilder` with a fusion permission defaulting to `true` in `new`, and
  a consuming `with_fusion(bool)` setter. A disabled builder must materialize
  each held elementwise node immediately after inserting it, using the existing
  `materialize_if_held` path. This leaves no held producer to compose into a
  later operation. Disable masked-window reduction elimination in this mode.
  Preserve node positions, computed-index materialization, reachability skips,
  and existing numeric identity permission checks.
- `proxima-tensor/src/bind/cached_attention_epilogue_liveness.rs`: factor the
  body of `bind_plain` into a shared private helper accepting that permission.
  The existing function supplies `true`; the new unfused entry supplies `false`.
  Do not duplicate the reachability/liveness traversal.
- `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs` and existing reexports:
  expose `bind_unfused(program, shapes, outputs, numeric_policy)`, which calls
  that shared builder path and does not run any graph-fusion passes. Existing
  `bind`, `bind_with_fusion`, and the capability-specific entry retain their
  current behavior. The configured caller chooses the fused or unfused entry
  explicitly. Numeric permission is necessary for a bit-changing rewrite;
  `allow_fusion=true` alone never grants it.
- `proxima-tensor/src/cpu/quantized_eval.rs` and existing CPU reexports:
  factor the current evaluator body so a configured caller can pass numeric
  policy and fusion permission along with exact-activation mode. Legacy
  evaluator functions forward their current defaults. Use an existing options
  type if one is found; otherwise add a small options record only to replace
  the growing boolean argument bundle, documenting those three independent
  meanings. Do not add a callback or trait-object evaluator.
- Backend prepare functions must call the same policy-sensitive binding seam.
  Their operation-capability restrictions remain in force: `allow_fusion`
  does not authorize a fused operation that a driver cannot lower.

## Eight literal assertion cases

Use prefix `programmable_binding_` and run exactly these eight tests. Tests
must inspect `BoundOp` contents, not only final output or number of operations.

1. `programmable_binding_unfused_chain`: input `[2,3]`, add literal `[1,1]`,
   multiply literal `[4,4]`. Expected unfused graph has separate Add and
   Multiply nodes; expected output is `[12,16]`.
2. `programmable_binding_fused_chain`: same graph; enabled mode may compose
   producer and consumer. Assert the composed Add/Multiply steps, their operand
   references and requested output ID; expected output remains `[12,16]`.
3. `programmable_binding_unfused_projection`: `[1,2]` multiplied by weight
   rows `[1,2]`, `[3,4]`, `[5,6]`, then summed, yields `[5,11,17]`.
   Assert standalone product and reduce in disabled mode.
4. `programmable_binding_requested_intermediate`: request the Add and
   Multiply outputs from case 1. Expect both IDs and payloads `[3,4]` and
   `[12,16]` in both modes, without renumbering.
5. `programmable_binding_unreachable_branch`: add an unused malformed-shape
   independent tail only where existing inference permits it; otherwise a
   well-formed unused branch. Assert its ID is absent from bound execution
   while all requested IDs remain. Do not depend on invalid-program acceptance.
6. `programmable_binding_signed_zero_permission`: use the existing x+0
   identity fixture. With signed-zero permission denied, retain the operation
   and compare output bits with the hand-derived positive-zero result.
7. `programmable_binding_oracle_self_control`: pass the literal `[12,16]`
   through the same coordinate comparator used in cases 1/2; expect zero
   mismatches among two coordinates.
8. `programmable_binding_oracle_corrupt_control`: pass `[13,16]` through that
   comparator against `[12,16]`; expect exactly one mismatch at coordinate 0.

Command from the isolated repo root:

```text
cargo nextest run -p proxima-tensor --lib -E 'test(programmable_binding_)'
```

Expected count: 8 tests, with the literal payload assertions above. Capture the
actual runner summary and inspect each new assertion. An empty filter is not
evidence. Run the existing binder suite separately to catch legacy regressions.

## Dependency and failure handling

Configured planning may not be exposed before this seam is consumed by its
CPU and GPU prepare paths. Do not use process environment mutation to implement
per-plan behavior. A no-fusion graph can use substantially more storage; its
resource estimate must be computed from that actual bound graph. If a backend
cannot render a resulting unfused form, return its typed lowering failure;
never turn fusion back on silently.
