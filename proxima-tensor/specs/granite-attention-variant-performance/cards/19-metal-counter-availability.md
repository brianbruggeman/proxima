# Card 19: inspect Metal counter availability during Granite replay

## Question

Can this Mac expose GPU execution or occupancy counters that explain the Groups8 timing distribution from Card 18? This is an instrumentation check around the existing F16-MMA/F32-KV Legacy versus configured Groups8 replay; no kernel field changes.

## Commands

The successful capture used the copied integration binary so dyld did not start it from Cargo's flat `deps` directory:

```sh
nice -n 20 xcrun xctrace record \
  --template 'Metal System Trace' \
  --output proxima-tensor/specs/granite-attention-variant-performance/evidence/card19-groups8-metal-system-selected.trace \
  --env PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups8 \
  --env PROXIMA_TEST_TIMEOUT_MS=900000 \
  --env PROXIMA_CAPTURE_LIVE=1 \
  --env PROXIMA_CAPTURE_NODES=all \
  --env PROXIMA_CAPTURE_STEPS=0 \
  --env PROXIMA_GRANITE_AB_REPORT=/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card19-groups8-report.json \
  --launch -- /tmp/card18-granite-attention-tests \
  perf_granite_simdgroup_count_warmup_control_two_shapes_against_f16_legacy --nocapture

python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py \
  --expected-shapes 2 --selected-arm simdgroups8 \
  proxima-tensor/specs/granite-attention-variant-performance/evidence/card19-groups8-report.json

python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py \
  --expected-shapes 2 --selected-arm simdgroups8 --negative-controls \
  proxima-tensor/specs/granite-attention-variant-performance/evidence/card19-groups8-report.json

swift -e 'import Metal; let device = MTLCreateSystemDefaultDevice()!; print(device.name); for set in device.counterSets ?? [] { print(set.name); for counter in set.counters { print("  \\(counter.name): \\(counter.description)") } }'
```

Two diagnostic trace attempts are retained separately: `card19-groups8-metal-system.trace` omitted the required count override, and `card19-legacy-metal-system.trace` explicitly selected Legacy even though this test accepts only Groups2/Groups8. The latter process exited 101 and produced no replay report; neither trace is used as timing evidence.

## Observations

- The Groups8 trace process exited 0 on an Apple M1 Max running macOS 15.8. The report checker printed `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0`; the negative-control check printed `negative_controls=13 rejected=13`.
- The trace report retains output equality at both actual prompt lengths (256 and 972), generated ID `[322]`, and the same output hashes as both arms: `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` at 256 and `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` at 972. It contains two warmup pairs and 20 measured pairs per arm per shape.
- In this traced replay, selected-minus-Legacy paired signs were 19 positive/1 negative at 256 and 20/0 at 972. p50 GPU timestamps (Legacy → Groups8) were 182,041.7 → 184,416.7 ns and 1,085,583.4 → 1,377,125.0 ns. These are this one traced process, not a replacement for Card 18's three-process record.
- The captured dispatch records show Legacy/Groups8 threadgroup widths 64/256, total grid threads 16,384/65,536 at 256 tokens and 62,464/249,856 at 972 tokens. Pipeline tuples `(max_threads, static_threadgroup_bytes, exec_width)` are `(448,9600,32)` and `(640,9600,32)`.
- The trace's Metal Application settings report `Counter Set: (null)` and `Shader Timeline: Disabled`. Its exported GPU-interval table has 117 rows for the replay process; each is a generic `Compute` interval with an empty label. The 116 application command-buffer submissions likewise do not identify the cached-attention entry, so these records cannot attribute intervals to either arm.
- Direct Metal API enumeration printed only counter set `timestamp`, containing `GPUTimestamp: A timestamp in nanoseconds on the GPU.` No occupancy, utilization, or shader execution counter is exposed through this API on this host. Thus this instrumentation route adds no kernel mechanism evidence; the timing cause remains unexplained.
- Direct Metal API enumeration used the command shown above. The host returned only the `timestamp` set with `GPUTimestamp`; this records the available API counters on this M1 Max, not a statement about other Apple GPUs.

Raw report: `evidence/card19-groups8-report.json` (SHA256 `3d3ef5ce9c0e1c25496eefed84f120cca9572bc714b48dcd59ee7fde77487e03`). The full selected and diagnostic trace bundles remain on this Mac under `/private/tmp/granite-attention-traces/card19/`; they are not in Git because their internal Instruments stores contain the repository hook's blocked-marker pattern. The selected trace's reviewable exports are committed as `evidence/card19-groups8-trace-toc.xml` (SHA256 `e36a0de17f988072bc6d78e1303c0a3852e93e1b0cde8a6694ca6e808971b00f`), `evidence/card19-groups8-gpu-intervals.xml` (SHA256 `480388a2517c5b3eb81b6cb2bfdd256ebf46ab93c4e568602657fd1b3771d79c`), and `evidence/card19-groups8-command-buffer-submissions.xml` (SHA256 `a57124f09da52021419bf9c0b82f9178f9d36b7bfd2f1c55e5c627299cc5f5dd`).

## Boundary

This experiment does not explain the Groups8 slowdown. Metal exposes timestamp sampling but no additional hardware counter set on this Mac, and the system trace's generic interval labels do not identify the cached-attention dispatch. No kernel or dispatch default changed; report any timing behavior without a traced mechanism as unexplained.
