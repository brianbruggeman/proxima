# Shared graph spec audit

VERDICT: ADMIT
gates: G1 pass, G2 pass, G3 pass, G4 pass, G5 pass, G6 pass, G7 pass
counts: R 6 / discharged 6 | AC 4 / orphaned 0 | slices 1 / unvalidated 0 | paths 6 / dangling 0

## Counts and checks

- Requirements G1-G6 each appear in at least one acceptance criterion. GAC1/GAC2/GAC3/GAC4 discharge 6/2/2/6 requirements; no acceptance criterion is orphaned.
- GAC1 expects exactly 10 named recurrent graph cases. GAC2 expects exactly 2 existing graph-comparison cases. GAC3 names exactly 1 library artifact and 0 compiler error diagnostics. GAC4 names 0 compiler/clippy errors.
- Scoped TASKS slice 1 supplies the GAC1 command plus GAC2-GAC4 and a numeric expected set; it is validated by those commands. Downstream interop/lab/measurement rows are outside this standalone graph audit.
- The revised problem statement at `SHARED-GRAPH-SPEC.md:9-14` binds the fixture to 2 physical layers, 2 iterations, 4 logical cache owners, 2 physical weight sets, 3 frozen ordinary graph/root payloads, and the absolute-plus-relative 0.000001 irrational-output tolerance.
- Ordinary-graph preservation now compares against three immutable pre-edit Op/root payloads, with a deliberately mutated graph/root negative control (`SHARED-GRAPH-SPEC.md:56-62,95-97`; `WORK-PACKET.md:145`). This avoids treating the compatibility wrapper over the new body as an independent oracle.
- The hand-derived epsilon-1 full-load fixture is consistent with RMSNorm's mean-square convention: `[3,4]` first scales by `sqrt(2/27)`; the next normalization multiplies by `sqrt(27/52)`, giving cumulative `sqrt(2/52)` (`WORK-PACKET.md:137`; RMSNorm divides the sum of squares by width at `proxima-tensor/src/spec/tests.rs:12009-12010`).
- Context paths: 6 checked, 0 dangling. The `spec/` paths resolve under `proxima-tensor/src/`; loopcd-panel references resolve under `proxima-tensor/specs/`.
