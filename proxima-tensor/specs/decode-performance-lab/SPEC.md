# decode-performance-lab

status: pilot evidence captured
owner: brian
created: 2026-10-02

## problem

Proxima needs a repeatable local experiment setup for Granite 1B-A400M and Gemma 4 E2B that records actual generated payloads and measured costs for baseline and configured speculative decoding, while accounting explicitly for LoopCD and Google's released performance-oriented models.

## refutation condition

If a failed load, CPU fallback, empty output, or configured-but-unused accelerator can appear as a measured speed improvement, this setup fails its purpose.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Inventory the existing Granite and Gemma checkpoint paths and model identities; retain primary-source URLs for LoopCD, Gemma MTP, QAT and DiffusionGemma. | yes |
| R2 | Execute an explicit manifest of two models, two literal text-completion prompts, four alternating OFF/ON pairs per prompt, and a bounded output budget using the existing gguf_generate CLI. | yes |
| R3 | Store each process's command, environment controls, raw stdout/stderr, generated IDs/text, token events, elapsed costs, CPU time and peak RSS; detect errors even when the child exits successfully. | yes |
| R4 | Compare payloads per pair and retain both slower and faster timing observations without claiming correctness or gains; exclude warmup and invalid runs from derived summaries. Report positive latency and throughput distributions with sample count, mean, sample standard deviation, range, percentiles, and CoV percent; undefined CoV remains null. | yes |
| R5 | Reject empty/malformed manifests, missing checkpoints, zero-token output, CPU fallback, and success-exit failure text using independent literal fixtures and explicit counts. | yes |
| R6 | Record execution eligibility separately for baseline, ngram speculation, LoopCD, Gemma MTP and diffusion; provision public reference configs for the latter methods without presenting an unavailable runtime as measured. | yes |

## architecture

`examples/decode_performance_lab.rs` reads a JSON manifest containing literal prompts with the existing root `serde` and `serde_json` dependencies, invokes the existing CLI with `std::process::Command`, writes one JSONL record plus raw stdout, stderr, and `/usr/bin/time -l` output per attempted arm, and derives a per-pair payload comparison and grouped distributions. CoV percent is `100 * sample_standard_deviation / mean`, undefined for fewer than two samples or a nonpositive mean. The generation CLI uses the existing speculative statistics API to expose drafted and accepted token counts. The runtime and model math are unchanged by this setup. Campaign execution is sequential within one process; concurrent campaign exclusion is not provided. Build artifacts use `/private/tmp/proxima-decode-lab-target`.

All prompts are literal completion prefixes, passed unchanged to both arms. Cross-model results carry independent prompt-token counts and are not ranked against each other. Each OFF/ON pair changes only `PROXIMA_SPECULATIVE_TYPES` (`none`/`ngram-simple`); both arms receive the same explicit ngram-simple parameters (`size_n=1`, `size_m=3`) so short repeated token continuations can be drafted within the 24-token cap. Deterministic greedy sampling and repetition penalties are fixed explicitly. Granite's missing verification capability is recorded as an eligibility limitation, even if its ON arm emits output.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| measurement surface | reuse gguf_generate token events and existing instrumentation | a second inference implementation would measure different behavior |
| framing | literal completion corpus | the existing speculative_bench always applies Gemma chat tokens and cannot establish Granite behavior |
| inference methods | explicit eligibility records | enabling an unsupported name does not execute its algorithm |
| controls | independent captured-output fixtures | self-agreement of two model runs is only a payload comparison, never an external correctness oracle |

## acceptance criteria

Run from the Proxima repository root.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1,R2,R6 | `cargo run --example decode_performance_lab -- --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --inventory` | models=2, prompts=2, planned_runs=32, methods=5 |
| AC2 | R3,R4,R5 | `cargo test --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example decode_performance_lab` | tests=8, failures=0, errors=0; independent literal-fixture parsing, fallback and failure controls included; sample [10,20,30] has mean=20, sample standard deviation=10, CoV percent=50; warmup and invalid pairs excluded |
| AC3 | R2,R3,R4 | compiled release runner with `--manifest proxima-tensor/specs/decode-performance-lab/manifest.json --binary /private/tmp/proxima-decode-lab-target/release/examples/gguf_generate --output /private/tmp/decode-lab-new-campaign`, archived to a new results directory | attempted_runs=32, pair_records=16, every attempt has raw logs and a result status; failed attempts are retained and cannot enter a timing comparison; earlier campaigns remain archived |
| AC4 | R1,R6 | `cargo run --no-default-features --target-dir /private/tmp/proxima-decode-lab-target --example decode_performance_lab -- --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --check-references` | reference_configs=3, reference_model_type_checks=3; local runtime eligibility=5 methods; generated checkpoint architecture/name checks occur in AC3 and AC5 |
| AC5 | R3,R5 | `cargo run --example decode_performance_lab -- --manifest proxima-tensor/specs/decode-performance-lab/manifest.json --check-fixtures` | incumbent_fixtures=2, parser_mismatches=0; captured gguf_generate raw output is checked against separately transcribed literal IDs, text and failure status in fixtures/expected.json, not against another parser invocation |

## out of scope

- Changing model architectures, mathematical guidance rules, GPU kernels or deployment defaults in this experiment-setup task.
- Claiming a LoopCD, MTP or diffusion speedup without executing that method.
- Public publication, commits, paid cloud jobs and unloading other users' model processes.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| Granite generic binding may reject its checkpoint or fail to express its architecture | unknown until execution | attempted runs cannot become valid model evidence | keep errors and exact payloads, expose eligibility; repair bounded setup failures |
| fresh process pipeline compilation hides warm decode costs | expected | end-to-end timing differs from cached serving | label each arm process-cold, report TTFT and later token deltas separately |
| host contention | unknown until execution | unstable timings | capture host load and GPU process inventory, retain CoV and no speed verdict |
| GPU sandbox rejects device access | possible | fallback falsely looks like GPU | reject fallback and rerun with execution approval when necessary |
| new methods have unavailable Proxima bind/decode capability | confirmed for inspected MTP and LoopCD paths | config acquisition alone does not run inference | mark unmeasured eligibility explicitly |

## context

- `ai_docs/research-loopcd.jsonl:2`
- `proxima-model-interop/src/architecture.rs:317`
- `proxima-model-interop/src/generate/decode.rs:3578`
- `proxima-model-interop/examples/speculative_bench.rs:1699`
- `examples/gguf_generate.rs:787`
- `proxima-tensor/specs/speculative-draft-mtp/SPEC.md:46`
