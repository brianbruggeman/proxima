# Omega programmable adaptation log

## Design constraints applied

The existing RuntimeConfig and Plan are extended. A second runtime, IR or
backend registry was rejected because existing Op/binding/Plan already express
the requested program. Configuration becomes consumed data at the existing
planning boundary, not another facade around environment variables.

No-alloc pressure keeps the existing alloc-tier binder/emitter free of runtime
I/O and clocks; runtime selection and config loading remain std-only. The plan
owns execution state; a caller owns selection evidence. No shared settings
lock or process-wide measurement cache is introduced.

Resource accounting names payload ownership rather than claiming to bound
opaque driver/compiler memory. Budget checks cover actual physical buffer
capacities and replacement overlap. Unsupported workspace accounting is an
explicit error. Allocation-free execution is unmeasured and is not asserted.

## Experiment register

| ID | Condition and source | Observation/status |
|---|---|---|
| P0 | Initial executable packets and root SPEC/TASKS | First audit refused combined tier/gate requirement, incomplete validation commands, and missing positive comparator controls; retained at `/private/tmp/omega-programmable-spec-audit.txt`. No Rust edits authorized by that draft. |
| P1 | Corrected spec and packets | Audit artifact `/private/tmp/omega-programmable-spec-audit-v2.txt` records G1-G7 ADMIT with 11 requirements, 14 ACs, 14 validation slices, 14 resolved context paths. This is spec admission, not implementation evidence. |
| P2 | `omega/omega-runtime.toml:61` and dense classifier minimum | The current minimum tiled token extent is 160. The frozen Ouro prefixes have fewer positions, so enabling BF16 storage alone would still decline their projections. Kernel-selection effect remains unmeasured. |
| P3 | Activation-first literal Metal fixture; `results/validation/omega-root-ac6.txt` | Tiled identity assertion failed. Classifier fixed weight at operand zero and tested the column extent as tokens. Role inference now uses exclusive output ownership. Both operand orders and decode-size refusal are asserted; captured repeat run is `omega-root-ac6-captured.txt`. |
| P4 | Existing packed BF16 projection with fusion both enabled and disabled | CPU decoder dispatch excluded F16/BF16 despite existing GGUF decoders; materialized multiply also needed native packed layout correction. Existing decoders and evaluator were reused. Binding assertions record `[17,39,23,53]` in both modes; `omega-execution-s2-bf16-tests-r3.txt` records eight executions. |
| P5 | Checkpoint query projection; same policy and threshold, compile gate toggled | All three recorded tiled means exceed scalar means; full means, CoV and source conditions are in `EVIDENCE.md`. Experimental gate stays default-off; no production threshold reduction is made. Raw samples, both error signs and auxiliary capture are retained. |
| P6 | wgpu allocation ownership review | Queue writes hide upload staging. Configured bounded execution uses explicit staging retained through synchronized completion; the physical reservation includes those buffers. This corrects the reservation model rather than dropping the resource constraint. |
| P7 | Configured paper hidden-readout control | The first enabled run repeated the first hidden logit in all channels. Source selection depended on operand roles omitted by the cache identity. The existing key now includes both roles. Same-codec reversed-order fixtures exercise BF16 and Float32; the enabled rerun records distinct expected coordinates in `results/ouro-configured/records.jsonl:2`. No tolerance was changed. |
| P8 | Final projection cohort, source/key repair included | The primary enabled cohort has high CoV at rows 1 and 24; the disabled cohort has high CoV at rows 1 and 5. Two additional complete paired cohorts retain variability without discarding samples. Each cohort has a separate output directory; all means and CoV appear in `EVIDENCE.md`. |

## Preregistered lowering comparison

Keep the production minimum unchanged. For the experimental micro and Ouro
builds, use `OMEGA_TILED_GEMM_MIN_TOKENS=1` in both compile-off and compile-on
arms, and vary only `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0/1`. Both arms
receive the same explicit numeric permission config and program/input bytes.
Record these build inputs with kernel identities. A third unmodified-default
record remains the Proxima before-change baseline. Different numerical
permissions are separate conditions and cannot be hidden inside a kernel
speed comparison.

Hypothesis: allowing native BF16/F16 reads through existing float tiles changes
the projection dispatch from serial/cooperative scalar work to matrix tiles for
eligible shapes. Refutation is an emitted/selected scalar path, different
payload outside the admitted numerical contract, or a synchronized wall-time
loss on the measured workload. Retain losses including small-row overhead.

Measurement artifacts and limits are indexed in `EVIDENCE.md`; changes to the
default production gate require evidence beyond the forced-small-row cohort.
