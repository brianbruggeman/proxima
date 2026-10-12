# ouro-mainline -- slices

Dependency order and delegated ownership. The three standalone contracts have
independent admission records; checked status below requires their actual runs.

Test counts name the selected cases. Preserve nextest's excluded-by-filter count
separately: a selection executing every named case does not mean the entire
crate suite ran. No selected case may be ignored or omitted.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Generalize existing forward/attention data and helpers against frozen incumbent graphs | R1,R2,R4,R5 | `cargo nextest run -p proxima-tensor --features std --lib -E 'test(recurrent_shared_)' --no-capture` plus GAC2-GAC4 in SHARED-GRAPH-SPEC.md | Exactly 10 named cases, including three frozen ordinary graph payloads; exactly 2 legacy cases; 1 alloc library artifact; 0 compiler/clippy errors | [x] | `results/weight-owner/ouro-owner-shared20-r2.txt` includes all 10 recurrent, 2 legacy and 8 programmable cases; 679 excluded. Alloc artifact in owner tensor-floor JSONL; shared clippy-r4 records zero diagnostics. |
| 2 | Bind Ouro through the existing registry and serving path | R3,R4,R6 | `cargo nextest run -p proxima-model-interop --features std --lib -E 'test(recurrent_mainline_)' --no-capture` plus IAC2-IAC4 | 8 CPU cases, 3 Metal cases, 1 no-default artifact, 0 compiler/clippy errors | [x] | `results/weight-owner/ouro-owner-tests-r5.txt` includes all 11 recurrent cases, 3 owner cases and 1 task-output case; 553 excluded. No-default artifact and final library/all-target lint logs retained in the same directory. |
| 3 | Replace experimental execution with mainline calls and forward Omega config | R5,R6,R7 | `cargo nextest run --example ouro_loopcd_lab --features proxima-tensor -E 'test(mainline_caller_)' --no-capture` plus CAC2,CAC5 | 4 caller cases, 3 planner cases, 1 example artifact | [x] | `results/weight-owner/ouro-owner-lab-caller4-r3.txt` executes 4; `ouro-owner-planner3.txt` executes 3 with 565 excluded. Strict lab clippy-r2 records zero diagnostics. Independent C2 references and comparison payloads retained. |
| 4 | Run paired mainline/llama cached measurements | R8 | CAC3/CAC4 exact commands in CALLER-SPEC | 12 cached request records per runtime; complete IDs/timings, resource records, CoV | [x] | Raw payloads in results/proxima-cached-r2 and results/llama-cached-r2; exact commands/failure/source hashes in results/cached-checkpoint/COMMANDS.md |
| 5 | Exercise shared configured planner and dense tile gate end to end | C5,C6 | CAC6 gate-0/gate-1 and native cached recount commands in CALLER-SPEC | Each arm 12 requests/2 diagnostics/2 summaries; recount 36 request rows/6 timing cells/4 diagnostic comparisons/24 trajectory pairs/1 comparator control | [x] | Executed payloads and independent recount in results/cached-recount; ubatch32 below-floor condition retained in EVIDENCE.md |
| 6 | Re-run configured tile gate with native-matched 256-row microbatches and step-zero dispatch capture | C5-C7 | CAC7 paired `--cached-performance` commands, fixed cached-run oracle, and `--cached-matched-batch` recount in CALLER-SPEC | Each arm 12 requests/2 diagnostics/2 summaries, 12 profile bracket pairs and zero drops; recount 36 request rows/6 timing cells/4 diagnostic comparisons/24 trajectory pairs/1 comparator control; each arm has an actual dense multi-token dispatch at step zero joined to its MSL source | [x] | Executed paired payloads, request-bracketed node116 source audit and strict results/cached-matched-batch-recount-r5; rejected unequal-batch control retained. See CALLER-C7-AUDIT.md |
| 7 | Share the existing immutable checkpoint owner across HF forward descriptors | R3,R5,R6,C2 | WEIGHT-OWNER-SPEC.md native selection below, then caller/planner/floor/clippy commands and real model commands in CALLER-SPEC | 3 new owner cases plus 8 CPU and 3 Metal recurrent cases = 14; shared tensor 20; caller 4; planner 3; alloc library artifact 1/errors 0; strict clippy diagnostics 0; live four-arm forward records 48; independent method comparisons 8; fresh configured cached checkpoint 12 requests/2 diagnostics/2 summaries/12 bracket pairs/0 drops | [x] | Raw validation in results/weight-owner; live48 in mainline-methods-live-r3; all eight C2 comparison directories retained. Cached owner r1 and post-lint r2 each retain all12 requests and matching profile pairs. Concurrent-drop control CPU-only; Metal execution and cleanup on owning thread. Adaptive host/device scalar diagnostic has its own contract and is not treated as bit equality by this count. |
| 8 | Preserve ordered generic broadcast folds when numeric policy denies TreeReduce | R7 plus BROADCAST-FOLD-SPEC R1-R6 | Exact emitter/device/replay/C2/end-to-end commands in BROADCAST-FOLD-SPEC and BROADCAST-FOLD-C2-COMMANDS | Counted criteria and source checkpoints in BROADCAST-FOLD-TASKS | [x] | Executed 3 emitter/3 specialization/5 device cases, 1 alloc artifact, zero final strict lint diagnostics, 2 saved replays/4 controls, 48 strict live requests and 12 relaxed cached requests. Fresh C2 retains 8 cells/26 vectors/901120 coordinates/12 scalars/8 selections and wrong-prefix rejection. Actual emitted ordered fold and both signed numerical directions are recorded in results/broadcast-fold-final-audit.md; timing/CoV and strict-policy increases remain in EVIDENCE.md. |

Shared-owner native case command (the config disables the repository's shell wrapper):

```text
cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-model-interop --features 'std metal metal-output-placement' --lib -E 'test(shared_forward_weights_) | test(recurrent_mainline_) | test(recurrent_metal_)' --no-capture --target-dir /private/tmp/proxima-ouro-mainline-target
```

The 14-case slice retains literal full vectors, pointer/codec payloads, typed
descriptor rejection, survivor and final-owner drop observations. The separate
48-record live-arm runner and 12-request GGUF checkpoint are mandatory after
the owner source change; isolated per-method processes do not replace the live
ownership control. Exact commands and source brackets belong in
`results/cached-checkpoint/COMMANDS.md`.

## shared-owner lint commands

The shared-owner slice's lint checks are the following two commands; each must
emit zero diagnostics. The library command retains all production lints; the
all-target command permits only the owner's test `expect`/`unwrap` convention.

```text
cargo clippy -p proxima-model-interop --features 'std metal metal-output-placement' --lib --target-dir /private/tmp/proxima-ouro-mainline-target -- -D warnings
cargo clippy -p proxima-model-interop --features 'std metal metal-output-placement' --all-targets --target-dir /private/tmp/proxima-ouro-mainline-target -- -D warnings -A clippy::expect_used -A clippy::unwrap_used
```

## checkpoint record

The recount repair's two literal cases and strict C7 replay are recorded in
`results/cached-checkpoint/ouro-mainline-recount-tests-r5.txt` and
`results/cached-matched-batch-recount-r5`. The four-mode shared-owner caller
emitted 48 warmup/sample requests in `results/mainline-methods-live-r3`.
Eight independent C2 comparisons retain all signed differences in
`results/c2-compare-owner-*-r1`; the register and audit are under
`results/weight-owner`. Cached owner r1 retains twelve requests with matching
profile brackets. The final lab helper lint repairs have their own fresh r2
checkpoint, with all twelve profile pairs and complete token/EOS trajectories;
the r2 source manifests compare identically before and after the run. It does
not replace a frozen timing arm.

Owner direction: every implementation checkpoint runs real-model end-to-end
generation. Build/unit completion alone does not close a checkpoint. The
checkpoint records retain actual tokens, latency payloads and CoV for each
executed condition.

No commit made. Default, C6 and C7 mainline/llama end-to-end cohorts are retained
as separate conditions. HF custom-descriptor admission and immutable ownership
are exercised by the real-model C2 and four-arm records. None of these cohorts
measures multi-stream scheduling.

## struck

- A separate Ouro runtime, attention implementation or cache owner is rejected by the owner's reuse requirement.
