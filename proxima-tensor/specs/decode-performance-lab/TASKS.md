# decode-performance-lab -- slices

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | inventory, literal corpus and method reference configs | AC1,AC4 | `cargo run --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example decode_performance_lab -- --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --inventory --check-references` | models=2, prompts=2, planned_runs=32, methods=5, reference_configs=3, reference_model_type_checks=3 | [x] | observed counts in labvalidation/decode-lab-ngram3-inventory-controls-final.log |
| 2 | runner, parser and independent literal controls | AC2,AC5 | `cargo test --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example decode_performance_lab`; compiled runner `--check-fixtures` | tests=8, failures=0, errors=0, incumbent_fixtures=2, parser_mismatches=0 | [x] | actual assertions and counts in labvalidation/proxima-decode-lab-ngram3-debug-test.log and inventory controls log |
| 3 | isolated release binary and recorded pilot campaign | AC3 | compiled release runner with `--output /private/tmp/decode-lab-campaign-ngram1x3-20261002`, archived verbatim in results-ngram-1x3 | attempted_runs=32, pair_records=16 | [x] | 32 captured records; 16 matching-payload pairs; four Gemma repetition ON arms recorded drafts; every raw arm retained |

## resume

Executed slices: 1, 2 and 3; no commits authorized
Evidence: results/ preserves the default campaign; results-ngram-3x3/ preserves the negative 3/3 campaign; results-ngram-1x3/ preserves the 1/3 campaign; labvalidation/ preserves build and control output.
Interpretation limit: only Gemma's synthetic repeating-color prefix recorded positive draft counters. Matching OFF/ON payloads are not an independent model-quality oracle. Granite binding fidelity, MTP, LoopCD and diffusion inference remain unmeasured by this setup.

## struck

- none
