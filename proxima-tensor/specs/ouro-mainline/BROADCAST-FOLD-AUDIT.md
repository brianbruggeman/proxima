# broadcast fold spec audit

**REFUSE — no source edits until the acceptance contract closes the gaps below.**

Audited files: `BROADCAST-FOLD-SPEC.md` (190 lines), `BROADCAST-FOLD-TASKS.md` (18 lines), and `BROADCAST-FOLD-C2-COMMANDS.md` (36 lines), against the current numeric policy, classifier, renderers, device tests, and saved replay artifacts.

## Inventory

- 6 requirements: R1–R6, each with an isolation statement.
- 8 acceptance criteria: AC1–AC8. AC1–AC7 have explicit commands or command sections; AC5 delegates exact runs to a present command file. AC8 has its command.
- 3 implementation slices, each linked to acceptance criteria and count-bearing expected output.
- 23 source files in the source hash command: 8 interop, 5 example, 2 tensor graph, 1 backend, 3 Metal lowering/classification, 2 test, 1 numeric policy, and 1 diagnostic file.
- No acceptance criterion relies on `exit 0` as its only expected result; each names counts or assertions.
- The C2 command file has 8 candidate invocations and 8 comparison invocations with fixed France/Ball prefixes and reference paths. It names 26 vectors, 901,120 coordinates, 12 scalars, 8 selections, and the existing `.003` bounds.

## Source evidence opened

- `proxima-tensor/src/numeric.rs:1–90` defines bit-exact policy as all permissions false; `NumericRewrite::TreeReduce` is the reassociation permission.
- `omega/src/msl/emit_and_classify.rs:1636–1695` makes the cooperative-policy gate and shows the broadcast-epilogue bypass in `reduce_is_cooperative_dispatch`.
- `omega/src/msl/elementwise_reduce_core.rs:518–616` contains the current serial seed, ordered `r` loop, input reads, body steps, combine, and output tail proposed for extraction.
- `omega/src/msl/tiled_gemm_cooperative_scan.rs:1792–2278` contains specialized early returns and cooperative fold; `:2278–2455` contains the cooperative tail and parallel broadcast writes.
- `omega/tests/reduce_epilogue_fusion_parity.rs:1–220` is the current three-case device target and its Metal feature gate.
- `results/weight-owner/adaptive-strength-diagnostic-findings.md` and its referenced r2 JSON/MSL/dispatch metadata retain the output-set-dependent Metal fold source and literal payloads.

## Findings that prevent admission

1. **The source immutability acceptance is not executable.** SPEC AC4 (line 132) requires 23 source hashes unchanged across the cached run; AC8 (lines 155–158) compares the strict live panel with immutable `mainline-methods-live-r3`. The command block at lines 163–165 prints one 23-file hash set but gives no before/after paths, comparison command, or raw manifest log. `BROADCAST-FOLD-TASKS.md` slice 2 likewise promises 23 hashes. Add explicit native commands to preserve a pre-run snapshot, compare the post-run snapshot, and assert 23 matching entries for each timed checkpoint. The current single `shasum` command cannot establish unchanged sources across either run.

2. **The strict live comparison is named but not checked.** AC8 expects signed changes against `mainline-methods-live-r3`, but its command only runs the new `--performance` panel. Add a concrete native comparison/recount command and expected record/payload count so the baseline is actually joined to the new records. Retaining both runs without a keyed comparison does not discharge the stated comparison.

3. **The all-target lint clause has no command.** AC7 (line 135) invokes `cargo clippy ... --lib`, then its expected text additionally requires “all-target compile/lint runs” with command-line-only `expect`/`unwrap` exceptions and zero other diagnostics. The `--lib` command cannot establish that clause. Add a separate all-target command with the exact test-only lint overrides and an expected diagnostic count, or remove the unsupported all-target clause while retaining the strict library lint command.

4. **R3’s specialized-path preservation lacks a named assertion.** R3 requires packed/tiled/expert emission to remain unchanged (line 28), and the architecture places the new branch after specialized early returns (lines 47–67). AC1 names only ordered-structure, permitted-tree, and serial-helper tests (lines 129, 137–141); AC2 exercises the broadcast epilogue fixture, not packed/tiled/expert specializations. Add an explicit assertion/control for specialized dispatch emission, or name and run existing cases that exercise the relevant packed/tiled/expert branches with an exact count.

The implementation outline otherwise reuses the existing serial loop, keeps the cooperative broadcast write tail, and does not add a new policy/config/API or device resource. The worked Float32 cancellation and positive rows have hand-checkable fold outcomes. These observations do not waive the acceptance gaps above.

## re-audit: revised checkpoint command surface

**ADMIT.** The revised `BROADCAST-FOLD-SPEC.md`, `BROADCAST-FOLD-TASKS.md`, `BROADCAST-FOLD-C2-COMMANDS.md`, and `BROADCAST-FOLD-CHECKPOINT-COMMANDS.md` close the four original gaps.

- Source freeze: `BROADCAST-FOLD-CHECKPOINT-COMMANDS.md` contains before/after `shasum` snapshots for both live and cached runs, a `cmp` for each pair, a cross-run `cmp`, and jq cardinality checks requiring 23 rows. The listed rows match the 23 files in the source manifest.
- Live-r3 comparison: its keyed `sample_id` jq join checks 48 unique requests, equal contexts and vector fields/cardinalities, and writes all signed vector/selected-ID/wall-time changes. The following recount checks 48 requests, 168 vectors, 5,996,544 coordinates, 8 timing cells, 40 samples and 8 warmups; independent timing summaries require five raw durations per arm/cell.
- Lint: SPEC AC7 retains strict production `--lib` lint; the AC1 command block now has exact `--all-targets` clippy flags with only the stated command-line `expect_used` and `unwrap_used` exceptions. Its result is count-bearing: zero other diagnostics.
- Specialized preservation: AC1 runs three named existing tests: Q8 packed row emission, Q4 tiled multiply/staging/bounds, and mixed expert descriptor codec/hoisting, with exactly three selected cases. The source assertions exercise the branches that R3 requires preserved.

Final inventory is 6 requirements, 8 acceptance criteria, 3 slices, 23 source-manifest entries, 3 new emitter cases, 3 specialized existing cases, 5 device cases, 2 saved-vector replay cases with 19 nodes each and 4 controls, 12 relaxed cached requests, 48 strict live requests, 8 C2 candidates plus 8 reference comparisons, and 1 alloc library artifact. Each criterion names a command and concrete assertion or count; none is exit-code-only.

The source plan matches existing interfaces: `NumericPolicy::bit_exact` and `NumericRewrite::TreeReduce` in `proxima-tensor/src/numeric.rs`; the bypass in `omega/src/msl/emit_and_classify.rs:1670–1695`; the reusable serial loop in `omega/src/msl/elementwise_reduce_core.rs:518–616`; cooperative body/tail in `omega/src/msl/tiled_gemm_cooperative_scan.rs:1792–2455`; and the actual broadcast-write helper at `:2431`. The admitted structure keeps tiled, dense, packed, and expert early returns ahead of the generic strict broadcast fold, then uses the existing broadcast write tail. No API, numeric-policy field, config field, device buffer, or lock is proposed.
