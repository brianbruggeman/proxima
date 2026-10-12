# caller C7 preregistration audit

**ADMIT the corrected capture condition as an additive matched-work checkpoint.**
This admission covers the fixed 256/256/320 workload, symmetric instrumentation,
and the use of all-node capture followed by payload-based dense-projection
selection. It does not attest to completed off/on runtime records.

## opened sources

- `CALLER-SPEC.md:20-30,100-108,140-224` defines CAC7, its counts, matched
  workload, corrected `PROXIMA_CAPTURE_NODES=all` commands, and dispatch/source
  join requirement.
- `TASKS.md:13` registers CAC7 separately from the earlier 32-row condition.
- `proxima-model-interop/src/serving.rs:821-835,1239-1240` defines production
  microbatch width and its default.
- `proxima-model-interop/src/generate/decode.rs:3707,3733-3735` selects and
  forms production prefill chunks.
- `examples/ouro_loopcd_lab/performance.rs:92-122,142-203,296-376` records
  batch/ubatch/context, emits per-request bracket markers, and records drain
  counts outside the timed region.
- `examples/ouro_llama_oracle/main.cpp:1022-1024,1055-1056` sets and records
  the cached native context and batch sizes.
- `omega/src/msl/emit_and_classify.rs:2499-2526` shows that the old
  `packed-multi-token` selector uses a packed-codec whitelist and declines
  BF16/Float16. The initial claim that `omega/instrument` was absent is
  retracted; the root dependency already enabled that feature.
- `omega/src/metal/arena_encode_dispatch_finish.rs:817-895,920-925,1012-1084`
  writes `bound_op` and `dispatch_capture` records with node, step, grid and
  source digest. `omega/src/metal/pipeline_buffers_upload.rs:193-230` writes
  exact source bytes under the digest-derived filename.
- `examples/omega_programmable_recount/cached.rs:338-376` checks setup values
  and per-request profile drop/timer fields. It does not itself join the raw
  GPU event markers or dispatch/source records; that join is a separate native
  evidence audit below.

## mechanical requirements

Each production arm has two cases, one warmup and five measured requests per
case: 12 requests, two diagnostics and two summaries. The two production arms
and matched llama run produce 36 request records, six timing cells, four full
signed diagnostic comparisons, 24 trajectory pairs and one equality/corruption
control. Each production arm must have 12 begin/end bracket pairs keyed by
`(case_id, repetition, warmup)`, zero recorder drops and a dense multi-token
projection dispatch at step zero inside a bracket. `gpu-events.jsonl` markers
are joined by parsing their nested `fields` JSON string; warmup rows match
`cached_warmup`, while measured marker repetitions 1–5 map to sample row
repetitions 0–4. The request row's `generated_count` retains actual emitted
work, including early EOS.

The corrected capture selector is `all`, because the instrumentation helper's
packed-only classifier excludes the target float-storage codec. Select the
dense projection from the captured `bound_op` extents and operands, then join
the same node and timestamp neighborhood to `dispatch_capture` and its
`pipeline_create` record. A valid source join has equal full SHA-256 in the
dispatch and pipeline records, and `shasum -a 256` of
`pipeline_<first-16-sha-hex>.metal` must reproduce that digest. Confirm the
dispatch timestamp falls between the matching begin/end markers. Record the
entry, node, step, grid, extents, full digest, and source path for both arms.

The corrected setup uses production `batch_size=256`, `ubatch_size=256`, and
`ContextLength::Within(320)`. Native `--cached-run` uses 320 context for the
64-token generation budget and records its 256/256 batch widths. Both arms use
identical filters, recorder and capture settings. The instrumentation-on timing
pair is compared only within CAC7, not against CAC3/CAC6 or as an uninstrumented
native/proxima ratio. The recount's batch/profile checks do not replace the raw
event join above.

## observed partial payload

The off-r3 repeated-ball256 warmup has these records in
`results/proxima-matched-batch-off-r3/gpu-events.jsonl`:

- begin marker: timestamp `1791061651466258417`, case `repeated-ball256`,
  repetition `0`, warmup `true`;
- `bound_op`: timestamp `1791061651610593250`, node `116`, kind
  `keep::reduce fold`, extents `[256,2048,5632]`, BF16 weight operand;
- `dispatch_capture`: timestamp `1791061651610595708`, step `0`, node `116`,
  entry `omega_reduce_r3_o2_n2_multiply_add_zero`, grid threads `369098752`,
  full MSL SHA-256
  `a07d5d715b8c0579f2bf1ce6ce631acfb76769bdcd29eec383c14256ff55b22d`;
- end marker: timestamp `1791061689470822625`, same case/repetition/warmup.

Native `shasum -a 256` of
`results/cac7-pipeline-off-r3/pipeline_a07d5d715b8c0579.metal` returned the
same full digest. That dispatch/source join is inside the timed
`repeated-ball256` warmup bracket. The completed off-r3 `cached.jsonl` contains
12 requests, two diagnostics and two summaries. The 24 marker rows form 12
exact begin/end groups by `(case_id,repetition,warmup)`, with zero malformed
groups. Each request row has `dropped_before=0`, `dropped_after=0`, and
`outside_timer=true`. The request records retain 24 emitted tokens for each
`ball24` request and 64 for each `repeated-ball256` request, including warmups.
The long-prefix warmup's step-zero node 116 dispatch falls between its marker
timestamps above; a separate node 116 diagnostic dispatch occurs outside the
request bracket and does not qualify. A native `rg -n 'simdgroup_'` search of
the gate-off source printed no lines; its captured source is the generic
`omega_reduce_r3_o2_n2_multiply_add_zero` path.

The on-r3 repeated-ball256 warmup has these records in
`results/proxima-matched-batch-on-r3/gpu-events.jsonl`:

- begin marker: timestamp `1791062034565923958`, case `repeated-ball256`,
  repetition `0`, warmup `true`;
- `bound_op`: timestamp `1791062034710262125`, node `116`, kind
  `keep::reduce fold`, extents `[256,2048,5632]`, activation node `111`
  in Float32 and weight node `112` in BFloat16;
- `dispatch_capture`: timestamp `1791062034710263958`, step `0`, node `116`,
  entry `omega_reduce_r3_o2_n2_multiply_add_zero`, grid threads `90112`,
  full MSL SHA-256
  `74eada85477b2aa77cd1e2bac1562a15b9fe6d72bd6c94b829ce1f9b3513ade8`;
- end marker: timestamp `1791062050815914708`, same case/repetition/warmup.

Native `shasum -a 256` of
`results/cac7-pipeline-on-r3/pipeline_74eada85477b2aa7.metal` returned the
same full digest. `rg -n 'simdgroup_'` found `simdgroup_float8x8`,
`simdgroup_load`, and `simdgroup_multiply_accumulate` in this source. The
off/on source manifests `results/cached-checkpoint/source-cac7-before.sha256`,
`source-cac7-between.sha256`, and `source-cac7-after.sha256` compare byte for
byte equal.

Both r3 production `cached.jsonl` files contain 12 request rows (10 measured,
two warmups), two diagnostics, two summaries, and one completion row. Each
arm's `gpu-events.jsonl` has 24 markers forming 12 exact begin/end groups by
`(case_id,repetition,warmup)`. Each request has `dropped_before=0`,
`dropped_after=0`, and `outside_timer=true`; emitted counts are 24 for
`ball24` and 64 for `repeated-ball256`, including warmups. The per-record join
extract is retained in `results/C7-PROFILE-AUDIT.jsonl`.

The matched native runner log is
`results/cached-checkpoint/ouro-mainline-cac7-llama-r3.txt`; its run records
use context 320, batch 256, and ubatch 256. The independent recount command
compiled but exited while checking the off arm with
`summary sample_standard_deviation_ms differs`; its log is
`results/cached-checkpoint/ouro-mainline-cac7-recount-r3.txt`. The partial
`results/cached-matched-batch-recount-r3/timing-recount.jsonl` holds 12 off-arm
`request_payload_recount` rows and no completed paired timing row. No paired
timing comparison is recorded here.

The first-sample-centered Welford and serde `float_roundtrip` repair was
checked with two named native tests. The log
`results/cached-checkpoint/ouro-mainline-recount-tests-r5.txt` records
`recount_centered_variance` and `recount_float_roundtrip` passing. The former
prints the C7 wall samples, mean `37987.5544664` and sample SD
`26.50438946723304`, alongside the `[1,2,3,4,5]` variance fixture and a
deliberately corrupted summary rejected by the comparator. The latter prints
the exact recovered values `37978.240083000004` and `37947.256541999996`.

The fresh r5 recount log is
`results/cached-checkpoint/ouro-mainline-cac7-recount-r5.txt`; its output has
36 `request_payload_recount` rows and six timing cells. Each timing cell keeps
five measured first-token and wall durations with source path and line; emitted
counts are 24 for `ball24` and 64 for `repeated-ball256`. The four full signed
vector records each compare 49,152 coordinates. Their payload lengths match
the difference counts; all values are finite, coordinates are ordered, and
each signed delta equals `actual-reference`. Proxima off/on records have zero
differences for `ball24` and 48,757 for `repeated-ball256`; llama/off records
have 49,152 differences each. The 24 trajectory rows cover both cases,
repetitions 0–4, warmup and five samples; each proxima pair preserves equal ID
arrays and EOS flags. The one comparator control records no differences for
identical input, rejects duplicated gate-off inputs, and contains the
one-coordinate corruption (`[1,2]` versus `[1,3]`).

The wrong-batch negative uses the earlier CAC6 files under
`results/proxima-configured-off/cached.jsonl`,
`results/proxima-configured-on/cached.jsonl`, and
`results/llama-cached-r2/llama.jsonl` with a fresh output directory. It exits
1 with `matched batch or instrumentation condition differs`; its partial
36-request/six-cell recount remains at
`results/cached-matched-batch-recount-r5-c6-negative/timing-recount.jsonl`.

## exact native commands for the completed audit

```text
jq -s '[.[] | select(.body == "gpu profile bracket marker recorded" and (.tags.marker == "ouro_gpu_profile_begin" or .tags.marker == "ouro_gpu_profile_end")) | {marker:.tags.marker,fields:(.tags.fields|fromjson)}] as $markers | ($markers | group_by([.fields.case_id,.fields.repetition,.fields.warmup])) as $groups | {marker_rows:($markers|length),groups:($groups|length),malformed:([$groups[] | select(length != 2 or ([.[].marker] | sort) != ["ouro_gpu_profile_begin","ouro_gpu_profile_end"])]|length)}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/gpu-events.jsonl
jq -s '[.[] | select(.body == "gpu profile bracket marker recorded" and (.tags.marker == "ouro_gpu_profile_begin" or .tags.marker == "ouro_gpu_profile_end")) | {marker:.tags.marker,fields:(.tags.fields|fromjson)}] as $markers | ($markers | group_by([.fields.case_id,.fields.repetition,.fields.warmup])) as $groups | {marker_rows:($markers|length),groups:($groups|length),malformed:([$groups[] | select(length != 2 or ([.[].marker] | sort) != ["ouro_gpu_profile_begin","ouro_gpu_profile_end"])]|length)}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/gpu-events.jsonl
jq -s '[.[] | select(.kind == "cached_sample" or .kind == "cached_warmup") | {kind,case_id,repetition,generated_count,drops:[.gpu_profile.dropped_before,.gpu_profile.dropped_after],outside_timer:.gpu_profile.outside_timer}]' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/cached.jsonl
jq -s '[.[] | select(.kind == "cached_sample" or .kind == "cached_warmup") | {kind,case_id,repetition,generated_count,drops:[.gpu_profile.dropped_before,.gpu_profile.dropped_after],outside_timer:.gpu_profile.outside_timer}]' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/cached.jsonl
jq -c 'select(.body == "gpu profile bracket marker recorded" and (.tags.marker == "ouro_gpu_profile_begin" or .tags.marker == "ouro_gpu_profile_end")) | {ts_ns,marker:.tags.marker,fields:(.tags.fields|fromjson)}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/gpu-events.jsonl
jq -c 'select(.kind == "cached_warmup" or .kind == "cached_sample") | {kind,case_id,repetition,generated_count,max_new_tokens,gpu_profile}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/cached.jsonl
jq -c 'select(.body == "bound_op" and .tags.extents == "[256, 2048, 5632]") | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/gpu-events.jsonl
jq -c 'select(.body == "dispatch_capture" and .tags.step == 0 and .tags.node == 116) | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/gpu-events.jsonl
jq -c 'select(.body == "pipeline_create" and .tags.msl_sha256 == "a07d5d715b8c0579f2bf1ce6ce631acfb76769bdcd29eec383c14256ff55b22d") | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/gpu-events.jsonl
jq -c 'select(.body == "bound_op" and .tags.node == 116 and .tags.extents == "[256, 2048, 5632]") | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/gpu-events.jsonl
jq -c 'select(.body == "dispatch_capture" and .tags.step == 0 and .tags.node == 116) | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/gpu-events.jsonl
jq -c 'select(.body == "pipeline_create" and .tags.msl_sha256 == "74eada85477b2aa77cd1e2bac1562a15b9fe6d72bd6c94b829ce1f9b3513ade8") | {ts_ns,tags}' proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/gpu-events.jsonl
shasum -a 256 proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-off-r3/pipeline_a07d5d715b8c0579.metal
shasum -a 256 proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-on-r3/pipeline_74eada85477b2aa7.metal
rg -n 'simdgroup_' proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-off-r3/pipeline_a07d5d715b8c0579.metal
rg -n 'simdgroup_' proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-on-r3/pipeline_74eada85477b2aa7.metal
cmp proxima-tensor/specs/ouro-mainline/results/cached-checkpoint/source-cac7-before.sha256 proxima-tensor/specs/ouro-mainline/results/cached-checkpoint/source-cac7-between.sha256
cmp proxima-tensor/specs/ouro-mainline/results/cached-checkpoint/source-cac7-between.sha256 proxima-tensor/specs/ouro-mainline/results/cached-checkpoint/source-cac7-after.sha256
env CARGO_TARGET_DIR=/private/tmp/proxima-ouro-mainline-recount-r5-target cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run --example omega_programmable_recount -E 'test(recount_)' --no-capture
env CARGO_TARGET_DIR=/private/tmp/proxima-ouro-mainline-recount-r5-target cargo run --example omega_programmable_recount -- --cached-matched-batch proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/cached.jsonl proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/cached.jsonl proxima-tensor/specs/ouro-mainline/results/llama-matched-batch-r3/llama.jsonl proxima-tensor/specs/ouro-mainline/results/cached-matched-batch-recount-r5
env CARGO_TARGET_DIR=/private/tmp/proxima-ouro-mainline-recount-r5-target cargo run --example omega_programmable_recount -- --cached-matched-batch proxima-tensor/specs/ouro-mainline/results/proxima-configured-off/cached.jsonl proxima-tensor/specs/ouro-mainline/results/proxima-configured-on/cached.jsonl proxima-tensor/specs/ouro-mainline/results/llama-cached-r2/llama.jsonl proxima-tensor/specs/ouro-mainline/results/cached-matched-batch-recount-r5-c6-negative
```
