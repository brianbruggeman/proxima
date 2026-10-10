# Card 20: test Metal trace labels for cached-attention replay intervals

## Question

Does Metal System Trace attach command-buffer or encoder labels to replay intervals, allowing the GPU intervals from Card 19 to be associated with their Legacy or Groups8 dispatch source?

## Experiment

The test-only `CapturedDispatch::time_gpu_ns` path was first changed to set the existing Omega command-buffer label to `node + entry + extents`. A second serial replay added the same label to the compute encoder. Both were warmup-controlled Groups8-versus-F16-MMA/F32-KV-Legacy replays at two prompt lengths under `Metal System Trace`. Both label changes were removed after exporting the traces because the template did not expose either label.

```sh
nice -n 20 xcrun xctrace record --template 'Metal System Trace' \
  --output proxima-tensor/specs/granite-attention-variant-performance/evidence/card20-groups8-encoder-labeled.trace \
  --env PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups8 \
  --env PROXIMA_TEST_TIMEOUT_MS=900000 \
  --env PROXIMA_CAPTURE_LIVE=1 \
  --env PROXIMA_CAPTURE_NODES=all \
  --env PROXIMA_CAPTURE_STEPS=0 \
  --env PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card20-groups8-encoder-labeled-report.json \
  --launch -- /tmp/card20-granite-tests \
  perf_granite_simdgroup_count_warmup_control_two_shapes_against_f16_legacy --nocapture

python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py \
  --expected-shapes 2 --selected-arm simdgroups8 \
  proxima-tensor/specs/granite-attention-variant-performance/evidence/card20-groups8-encoder-labeled-report.json

python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py \
  --expected-shapes 2 --selected-arm simdgroups8 --negative-controls \
  proxima-tensor/specs/granite-attention-variant-performance/evidence/card20-groups8-encoder-labeled-report.json
```

## Observations

- Both traced processes exited 0. Each report passed the two-shape positive check and rejected all 13 negative controls; each has 20 measured pairs per arm per shape and `round_differences_positive=40 negative=0 zero=0`.
- At actual 256 and 972 tokens, both reports record equal output bytes and generated IDs `[322]` between Legacy and Groups8. The output hashes match the Card 18/19 outputs.
- The command-buffer-only trace has 116 target-process GPU intervals; the command-buffer-plus-encoder trace also has 116. Every interval in both exports has an empty Metal object/event label. The Metal Application settings still show `Counter Set: (null)` and `Shader Timeline: Disabled`.
- This trace cannot associate its GPU interval rows with the emitted cached-attention entries. It adds no mechanism attribution and remains separate from Card 18's three-process timing record.

Report SHA256 values: command-buffer-only `a30922b06bfff880a0bbe2cd61cd90c31f5e7f3d2a834a44b841222234c1e518`; command-buffer-plus-encoder `00954c4051d59dee270d85cb318be86fc7290b52ad8a5559124ce55cbc6b052d`. Both reports are in `evidence/`. Exported GPU intervals are `evidence/card20-groups8-labeled-gpu-intervals.xml` (SHA256 `06c97c29286413abb8b9f6b24e56f4c1f573ad1c39267734c25e2f9a2fd5f055`) and `evidence/card20-groups8-encoder-labeled-gpu-intervals.xml` (SHA256 `a21ef5ecacf5d63e07b84544736a4da28c4fe37127544446172d9d9cbbef400e`). The raw trace bundles remain on this Mac at `/private/tmp/granite-attention-traces/card20/`; the two trace TOCs are committed beside the interval exports.

## Boundary

The temporary labels did not change the trace payload available from this Instruments template and were removed from Omega. Groups8 timing remains unexplained. The next schedule experiment must keep arm identity in its replay report and use matched GPU timestamps plus emitted MSL; system-trace interval attribution is unavailable from this host/template.
