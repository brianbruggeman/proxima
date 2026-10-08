# ANE lane cards: NPU performance on this machine (2026-10-08)

This box has one NPU, the Apple Neural Engine, reached only through CoreML.
Hexagon, Intel NPU and XDNA lanes are catalogued in
`docs/research/npu-lanes-census-2026-10-08.md` and cannot run here; they are
not in this set. Every card is a measurement: a prediction written before it
runs, welded commands, an asserted count, a kill line, and a row template. No
card changes library code. A card that wins becomes a spec slice afterwards,
with its own three parity checks (tests, token parity, wall clock and RAM).

The card contract and the hands rules are those of
`docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md` section 0, with two
changes that reflect today's workspace rules: no new worktrees or branches
(each card works in an export tree under `.long_ctx_backups/ane/<card>/tree`,
produced by `git archive` from a named main commit, with its own target dir),
and the GPU lock is the directory `.long_ctx_backups/gpu.lock`.

## Harness, one Rust example, no scripts

`proxima-model-interop/examples/ane_lanes.rs`, behind a default-off cargo
feature `coreml` on `proxima-model-interop`. CoreML is driven through the
`objc2-core-ml` bindings (added with `cargo add`); the model file is written
from Rust as the CoreML `Model.proto` through `prost` (one `innerProduct`
or `batchedMatmul` layer, or an MIL program for the attention card), then
compiled at runtime with `MLModel.compileModelAtURL` and loaded with the
requested `MLComputeUnits`. Timing is wall clock around `prediction(from:)`.
Which unit actually ran a layer comes from `MLComputePlan`; a card whose
"ANE" arm the plan reports on CPU or GPU is VOID, not a number.

Every timed cell: box quiet first (`ioreg` Device Utilization near 0 on three
settled samples, no `cargo`/`rustc`, `uptime` recorded before and after into
the cell's log), arms interleaved per iteration, CoV reported over all runs,
peak RSS and footprint from `/usr/bin/time -l`, CPU% as user+sys over wall.
Ollama never runs. Models: granite moe 1b (the blob path `decode_arms` uses)
and gemma4 E2B only.

Incumbent numbers (Metal, this box, main `0688ae5b`), from
`proxima-tensor/specs/decode-prefill-parity/evidence/attr3/`:

| shape | Metal, own-cb | source |
| --- | --- | --- |
| granite decode Q8_0 expert matvec, 1024x512, 1 token | 19.9 us/op | `rank/granite_decode.md` |
| granite decode attention partial + merge, kv 1024 | 50.3 + 6.5 us/op | same |
| granite decode router f32 1024x32 | 10.4 us/op | same |
| granite prefill 1000 tokens, stacked expert GEMM | 2268 to 2411 us/op | `rank/granite_prefill.md` |
| granite prefill attention partial | 1149 us/op | same |
| Metal per-dispatch floor (24-op head chunk) | 10.9 us/op | `timeline/` |
| granite decode step wall | 6.468 ms, 24 layers | `timeline/` |

Published ANE floors, for the predictions only (private API, not shippable;
`docs/research/npu-lanes-census-2026-10-08.md` [8]): 0.095 ms dispatch,
2.3 ms IOSurface round trip per dispatch. CoreML's public path has no
published number; card 1.1 produces the first one for this box.

## Scope (owner, 2026-10-08): the router, not the model and not the experts

The ANE holds the routing part only. Router weight per layer is hidden x
experts, so every MoE's full router set fits the ANE with room to spare.
Every row below is read from the model's published `config.json`
(`docs/research/moe-router-sizes-2026-10-08.md` carries the URLs and the
open items: Scout read from a mirror, gemma 4 and Qwen3.6 all-layers-MoE
inferred from the absence of a sparse-step key; router bias vectors excluded):

| model | moe layers | hidden x experts | router params | fp16 |
| --- | --- | --- | --- | --- |
| granite 3.1 1b MoE (local) | 24 | 1024 x 32 | 0.79M | 1.6 MB |
| Mixtral 8x7B | 32 | 4096 x 8 | 1.0M | 2.1 MB |
| gpt-oss-20b | 24 | 2880 x 32 | 2.2M | 4.4 MB |
| Llama 4 Scout | 48 | 5120 x 16 | 3.9M | 7.9 MB |
| gemma 4 26B-A4B | 30 | 2816 x 128 | 10.8M | 21.6 MB |
| Qwen3-30B-A3B | 48 | 2048 x 128 | 12.6M | 25.2 MB |
| gpt-oss-120b | 36 | 2880 x 128 | 13.3M | 26.5 MB |
| Qwen3.6-35B-A3B (local, 22.3 GB blob) | 40 | 2048 x 256 | 21.0M | 41.9 MB |
| Qwen3-235B-A22B | 94 | 4096 x 128 | 49.3M | 98.6 MB |
| DeepSeek-V3 | 58 | 7168 x 256 | 106M | 213 MB |
| Kimi K2 | 60 | 7168 x 384 | 165M | 330 MB |

So the ANE's size limit never binds for routers. What binds is the dispatch
count: the exact router needs the layer's hidden state, which only exists on
the GPU after the layer below, so an exact router on the ANE is one round trip
per layer per token: layers x (ANE floor from 1.1 + 2 handoffs from 1.2). At a
0.3 ms floor and 24 layers that is 7 ms per token, more than granite's whole
6.5 ms step. The shape that fits the floor is the owner's second idea: the
ANE runs a route PREDICTOR once per token for every layer at once (SiDA's
signal: the token embedding; or ProMoE's: an early layer's input), and the GPU
keeps the exact top-k. The prediction is used ahead of time: to prefetch paged
experts, to warm them, or to pre-dispatch. That is one ANE dispatch per token
regardless of depth, and the model size it enables is the paged ceiling
(experts on disk, hot set resident), not the ANE's.

Owner's ruling, 2026-10-08: the exact router on the ANE is out unless it can
run on a warm path, off the per-layer critical path; nothing in this set
pursues it. The predictor line is the campaign. Card 1.1's floor is still
measured first because every predictor cost is a multiple of it.

Primary line, in order: 0.1 -> 1.1 -> 5.1 (hit rate, CPU) -> 5.3 (predictor
on the ANE, one dispatch per token) -> 5.4 (prefetch gain on paged experts).
Secondary, only if the primary line pays: 1.2, 1.3, 2.x, 3.1, 4.x (the
attention front and experts on the ANE). 5.2 is needed only by 2.1.

## Sequence

0.1 -> 1.1 -> 5.1 -> 5.3 -> 5.4. Then, optionally, 1.2 -> 1.3 -> 2.1 and 2.2
-> 3.1 -> 4.1 -> 4.2 (contingent on 4.1). 5.1 and 5.2 are CPU-only, need no
ANE, and run whenever the box is not benching. 5.1 needs the captured routes
from an instrumented run; 0.1 makes that capture.

---

#### CARD 0.1: toolchain, box census, and the route capture

tier: worker
depends_on: []
export: `cd /Users/brianbruggeman/repos/slot-0/proxima-windows && git archive <main sha> | tar -x -C /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/ane/0.1/tree`
target_dir: /private/tmp/cargo_target_ane_01
lock: /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/gpu.lock (held only for the capture run)
opens: `proxima-model-interop/Cargo.toml` (one `[[example]]`, one feature `coreml`, `cargo add --optional objc2-core-ml prost`); new `proxima-model-interop/examples/ane_lanes.rs`
commands:
1. `cd <tree> && cargo add -p proxima-model-interop --optional objc2-core-ml && cargo add -p proxima-model-interop --optional prost && cargo add -p proxima-model-interop --build --optional prost-build`
2. `cd <tree> && cargo check -p proxima-model-interop --features std,metal,coreml --example ane_lanes > runs/0.1-check.log 2>&1; echo EXIT=$?`
3. `cd <tree> && cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- census > runs/0.1-census.log 2>&1; echo EXIT=$?`
4. the route capture, lock held: `cd <tree> && cargo run --release -p proxima-model-interop --features std,metal,instrument --example ane_lanes -- capture-routes --model <granite blob> --prompt proxima-tensor/specs/decode-prefill-parity/evidence/round2fix/bench/prompt1k.txt --new 128 --out runs/0.1-routes.bin > runs/0.1-capture.log 2>&1; echo EXIT=$?` (reads the top-k extra outputs the capture now records per dispatch and the layer inputs that feed each router)
expect:
- N1 = chip line (`sysctl -n machdep.cpu.brand_string`) and the ANE presence line, 2 lines
- N2 = compile + 1 prediction of a 1024->512 fp16 innerProduct succeeds on each of `cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine`, `all`, with the unit `MLComputePlan` reports for the layer: 4 lines
- N3 = routes captured: 1128 tokens x 24 layers = 27,072 (expert ids, gate weights, and the 1024-wide router input per route)
predict: all four units accept the model; `cpuAndNeuralEngine` reports the Neural Engine for the innerProduct; the capture is 27,072 routes and about 111 MB (27,072 x 1024 x 4 bytes) plus ids
kill: `objc2-core-ml` does not expose `MLComputePlan` or `compileModelAtURL` (then the card reports the missing symbol and the harness falls back to an `MLModel` load of a pre-compiled `.mlmodelc`, labelled); the proto is rejected by the compiler (error text verbatim)
memory gate: MG-2 for steps 2 and 3 (peak RSS under 400 MB); MG-3 clauses 1 to 3 for step 4 (the capture run is a decode run)
rollback: `rm -rf` the export and the target dir; nothing on main
blast: `proxima-model-interop/Cargo.toml` and one example; zero library code
observe: the `MLComputePlan` device per layer; the capture's route count
row: `ROW <n>: ANE toolchain and the route capture: <chip>, 4/4 units, 27,072 routes`
reprove: commands 3 and 4

#### CARD 1.1: the ANE dispatch floor on the public path

tier: hands
depends_on: [0.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock, held for the timed run
opens: none (harness subcommand `floor` from 0.1)
commands:
1. quiet check: `cd <tree> && pgrep -fl 'cargo|rustc|decode_arms|nextest'; uptime; for i in 1 2 3; do ioreg -r -d 1 -c IOAccelerator | grep 'Device Utilization'; done > runs/1.1-box-before.txt 2>&1`
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- floor --in 1024 --out 512 --iters 1000 --units cpuOnly,cpuAndGPU,cpuAndNeuralEngine > runs/1.1.log 2>&1; echo EXIT=$?` (arms interleaved per iteration: one prediction per unit, round robin)
3. `uptime >> runs/1.1-box-after.txt`
expect:
- N1 = 1000 timed predictions per unit, 3 units: 3000 lines
- N2 = per unit: p50, p99, CoV over the 1000; the `MLComputePlan` device for the layer (the ANE arm must say Neural Engine or the cell is VOID)
- N3 = peak RSS, footprint, CPU% for the process
predict: the ANE p50 lands between 0.3 and 3 ms per prediction (bracketed by the private-API 0.095 ms dispatch and its 2.3 ms IOSurface round trip); the CPU unit lands under 0.1 ms for this shape
kill: ANE p50 above 5 ms (the lane is whole-layer-only or nothing; 2.1 and 2.2 still run, 3.1 is skipped); CoV above 10% on the ANE arm (cell VOID, re-run after a quieter box)
memory gate: MG-2
rollback: none, measurement only
blast: none
observe: the ratio ANE p50 / Metal per-op floor (10.9 us); it sets the granule for every later card and the row prints it
row: `ROW <n>: ANE dispatch floor: cpu <p50> ms, gpu <p50> ms, ane <p50> ms (CoV <x>%), <ratio>x the Metal per-op floor`
reprove: command 2

#### CARD 1.2: handoff cost between a Metal buffer and a CoreML input, per token

tier: hands
depends_on: [1.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: none (harness subcommand `handoff`)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- handoff --bytes 2048,16384,131072 --iters 1000 --modes copy,iosurface > runs/1.2.log 2>&1; echo EXIT=$?` (a Metal buffer of the given size, written by a Metal kernel, read by a CoreML prediction, result read back into a Metal buffer; `copy` goes through host memory, `iosurface` wraps an IOSurface-backed `MLMultiArray` and a Metal texture/buffer on the same surface)
expect:
- N1 = 1000 round trips x 3 sizes x 2 modes: 6000 lines
- N2 = per (size, mode): p50 each direction and CoV; for `iosurface`, a printed pointer-equality assertion proving zero copies, or the exact API that refused
- N3 = peak RSS, footprint, CPU%
predict: `copy` of 2 KB under 20 us each way; `iosurface` zero-copy proven, with a sync cost under 50 us; 128 KB copy under 100 us
kill: `iosurface` cannot share with Metal on this OS (card reports the refusal; later cards use `copy` and say so)
memory gate: MG-2
rollback: none
blast: none
observe: the two-way handoff per token against the Metal attention it would replace per layer (50.3 + 6.5 us at granite decode)
row: `ROW <n>: handoff: copy <us> / iosurface <us> per 2 KB round trip; zero-copy <proven|refused>`
reprove: command 2

#### CARD 1.3: the graph-size ceiling and the swap cost between compiled graphs

The ANE takes one compiled graph at a time, and the community limit for a graph
on Apple silicon is about 1 GB of weights (ANEMLL chunks its models at that
size; NPUMoE reports graphs over about 1.2 GB falling to CPU on M2 Ultra; Apple
publishes no number). Anything larger than one graph runs as several graphs
swapped per token, so the swap cost, not the ceiling, decides how big a model
the ANE can serve.

tier: hands
depends_on: [1.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: none (harness subcommand `graphs`)
commands:
1. quiet check as 1.1
2. ceiling: `cd <tree> && cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- graphs --ceiling --sizes 256,512,768,1024,1536,2048 --form fp16 > runs/1.3-ceiling.log 2>&1; echo EXIT=$?` (one innerProduct chain per size in MB of fp16 weights; compile, load, one prediction; record the unit the plan reports and the load time; the first size the plan sends to CPU or that fails to compile is the ceiling)
3. swap: `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- graphs --swap --size <largest ANE-accepted size from step 2> --graphs 2,4,8 --iters 200 > runs/1.3-swap.log 2>&1; echo EXIT=$?` (N graphs of that size loaded at once, predictions round-robin across them, one per iteration; versus the same graph predicted repeatedly)
expect:
- N1 = 6 sizes with (unit, compile ms, load ms, first-prediction ms, steady p50)
- N2 = per graph count: steady p50 per prediction when alternating, CoV, peak footprint; the single-graph p50 beside it
- N3 = memory and CPU%
predict: the ceiling lands between 1024 and 1536 MB fp16; alternating between 2 graphs costs under 2x the single-graph p50; at 8 graphs of the ceiling size (8 GB resident) the per-prediction cost rises past 5x (weights re-streamed into the ANE per swap)
kill: the ceiling is under 512 MB (then only the front, never experts, is an ANE candidate on this box); or the 2-graph swap exceeds 5x (then a model is one graph or nothing)
memory gate: MG-2, with the ceiling raised to the resident graphs' size plus 400 MB, stated per cell
rollback: none
blast: none
observe: ceiling MB; swap cost per graph count; both decide which rows of the size table below are reachable
row: `ROW <n>: ANE graph ceiling <MB> fp16; swap p50 x2 <ms> x4 <ms> x8 <ms> vs single <ms>`
reprove: commands 2 and 3

What the ceiling holds, derived from model configs (fp16 bytes = 2 x params;
int8 and 4-bit palettized shrink the weights, compute stays fp16 on M1; W8A8
compute is documented only for A17 Pro and M4, so int8 here is a weight form):

| piece | params | fp16 | int8 | 4-bit |
| --- | --- | --- | --- | --- |
| granite 1b, whole model | 1.3B | 2.6 GB | 1.3 GB | 0.65 GB |
| granite 1b, front (attention + router + norms, 24 layers) | 40M | 80 MB | 40 MB | 20 MB |
| granite 1b, one expert (1024x512x3) | 1.5M | 3 MB | 1.5 MB | 0.75 MB |
| gpt-oss-20b front (24 layers, q/o 2880x4096, k/v 2880x512) | ~640M | 1.3 GB | 0.64 GB | 0.32 GB |
| gpt-oss-20b, one expert (2880x2880x3) | ~25M | 50 MB | 25 MB | 12 MB |
| qwen3-30B-A3B-shaped front (48 layers, q/o 2048x4096, k/v 2048x512) | ~0.9B | 1.8 GB | 0.9 GB | 0.45 GB |
| qwen3-30B-A3B-shaped expert (2048x768x3) | 4.7M | 9.4 MB | 4.7 MB | 2.4 MB |

So at a 1 GB ceiling: the whole granite model fits in one graph at 4-bit and in
two at int8; the gpt-oss-20b front fits at int8; the Qwen-shaped front needs
4-bit or two graphs; experts fit at about 20 per graph for gpt-oss-20b and
about 100 per graph for the Qwen shape. The gpt-oss and Qwen rows use configs
from memory and are unverified until the GGUF headers are read; the granite row
is from the attr3 census shapes.

#### CARD 2.1: one real expert FFN on the ANE against Metal, by token count and weight form

tier: hands
depends_on: [1.1, 5.2]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: none (harness subcommands `expert` and the Metal arm via the existing `norm_variant_ab` per-dispatch timing)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- expert --model <granite blob> --layer 0 --projection gate --tokens 1,8,32,128,512 --forms <the forms 5.2 admitted> --iters 200 > runs/2.1.log 2>&1; echo EXIT=$?` (real layer-0 gate weights from the GGUF; inputs are the captured router inputs from 0.1; the Metal arm is the same op captured and replayed per dispatch, interleaved with the ANE prediction per iteration)
expect:
- N1 = 5 token counts x F forms x 200 iters on the ANE, plus 5 x 200 Metal replays
- N2 = per (tokens, form): ANE us/op, Metal us/op, ratio, CoV each; output max-abs and cosine against the Metal Q8_0 result on the same inputs (a row with no parity number is RED)
- N3 = peak RSS, footprint, CPU%
predict: at 1 token the ANE is slower than Metal's 19.9 us/op by the 1.1 floor; the crossover where the ANE fp16 op is at or under Metal's per-op time is at or above 128 tokens; palettized forms are not faster than fp16 at these shapes
kill: no crossover below 512 tokens (the expert lane is prefill-only or nothing; 4.1 still runs for the front, with experts on Metal)
memory gate: MG-2
rollback: none
blast: none
observe: the crossover token count per weight form; the parity columns
row: `ROW <n>: expert on ANE: crossover at <tokens> tokens (<form>); 1-token <x>x Metal; parity max-abs <e> cosine <c>`
reprove: command 2

#### CARD 2.2: the attention front of one layer on the ANE

tier: worker
depends_on: [1.1, 1.2]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `attention` (an MIL program: q/k/v/o projections, GQA 16/8 heads of 64, RMSNorm, rope, SDPA over a static KV of 1024)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- attention --model <granite blob> --layer 0 --kv-len 1024 --tokens 1,32,512 --iters 200 > runs/2.2.log 2>&1; echo EXIT=$?` (real layer-0 weights; a real KV from the 0.1 capture; the Metal arm is the captured attention chain replayed per dispatch, interleaved)
expect:
- N1 = 3 token counts x 200 iters ANE, plus 3 x 200 Metal
- N2 = per token count: ANE us/layer, Metal us/layer (projections + partial + merge + norms + rope, summed from the replay), ratio, CoV; parity of the layer's attention output against Metal (max-abs, cosine); the `MLComputePlan` device per MIL op (an op the plan sends to CPU is listed by name)
- N3 = memory and CPU% as above
predict: at 1 token the ANE front runs in 0.2 to 0.5 ms against Metal's about 0.11 ms per layer; at 512 tokens the ANE is within 2x of Metal's prefill attention (1149 us/op) only if SDPA stays on the ANE; rope or RMSNorm fall to CPU in the plan
kill: SDPA is placed on CPU by the plan at every shape (then the front on the ANE is projections-only and the card says so); parity drift beyond fp16 rounding (max-abs above 1e-2 on normalized outputs)
memory gate: MG-2
rollback: none
blast: none
observe: one printed line per token count: ANE front + 2 x handoff (1.2) against Metal front; that is the stage-split arithmetic per layer
row: `ROW <n>: ANE attention front: 1 token <us> vs Metal <us>; +2 handoffs <us>; plan: <ops on ANE>/<ops on CPU>`
reprove: command 2

#### CARD 3.1: ANE and Metal in flight at the same time

tier: hands
depends_on: [2.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: none (harness subcommand `concurrent`)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- concurrent --iters 500 > runs/3.1.log 2>&1; echo EXIT=$?` (three arms interleaved per iteration: ANE expert alone, Metal Q8_0 GEMM alone at 128 tokens, both issued back to back and waited together; the GPU utilization sampler runs alongside)
expect:
- N1 = 3 arms x 500 iters
- N2 = per arm: p50, CoV; the slowdown of each side under concurrency; the GPU sampler's utilization during each arm
- N3 = memory and CPU%
predict: each side slows by under 15% when concurrent (HeteroLLM saw concurrent bandwidth rise on Snapdragon; Apple has no published number, so this is a guess)
kill: CoreML serializes behind the Metal queue (both arms sum, no overlap); then the card re-runs the concurrent arm from a second process and labels it
memory gate: MG-2
rollback: none
blast: none
observe: overlap fraction = 1 - (both / (ane + metal))
row: `ROW <n>: concurrency: ane <p50> alone / <p50> concurrent, metal <p50> alone / <p50> concurrent, overlap <f>`
reprove: command 2

#### CARD 4.1: whole-layer granule: front on the ANE, experts on Metal, one granite layer

tier: worker
depends_on: [2.2, 1.2]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `layer` (composes 2.2's MIL front with the existing Metal expert path for layer 0, through the 1.2 handoff mode that proved cheapest)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- layer --model <granite blob> --layer 0 --tokens 1 --iters 500 > runs/4.1.log 2>&1; echo EXIT=$?` (interleaved with the all-Metal layer-0 step replay from `norm_variant_ab` on the 0.1 capture)
expect:
- N1 = 500 iters x 2 arms
- N2 = per arm p50 and CoV; the hybrid arm split into ANE / handoff in / handoff out / Metal experts, summing to its p50 within 5%; parity of the layer output against all-Metal (max-abs, cosine)
- N3 = memory and CPU%
predict: hybrid per layer within 1.5x of all-Metal (0.27 ms per layer from the timeline: 6.468 ms / 24)
kill: hybrid above 2x all-Metal at 1 token; or parity drift beyond fp16 rounding
memory gate: MG-2
rollback: none
blast: none
observe: the split line; the implied whole-step number (x 24 layers) beside the measured 6.468 ms
row: `ROW <n>: hybrid layer: <ms> vs all-Metal <ms>; split ane <ms> / handoff <ms> / experts <ms>; implied step <ms> vs 6.468`
reprove: command 2

#### CARD 4.2: the split model end to end (contingent: only if 4.1 is within 1.5x)

tier: worker
depends_on: [4.1, 3.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: a new arm `ane_front` in `decode_arms` / `decode_gbps_baseline` behind the `coreml` feature; no library change (the front runs through the harness's own composition)
commands:
1. quiet check as 1.1
2. the `decode_arms` protocol exactly as the round-two and round-three benches: 3 processes x (1 warm-up + 7 timed), arms `metal` (base) / `ane_front` / `control` (byte copy of the base), prompts `prompt1k.txt` and `prompt_short_hippo.txt`, granite only, interleaved; every field the owner's bench rule names: whole-generation wall, TTFT, ms/token, CPU time and CPU%, peak RSS and footprint, GPU bytes, GPU busy fraction, CoV and the bound line per timed column, text hash
expect:
- N1 = 21 runs per arm per prompt; text hash equal to the all-Metal arm on all generations
- N2 = the full bench table, llama's recorded row beside it (`evidence/slice0/ac1`, wall marked derived)
predict: decode ms/token within 10% of all-Metal with lower CPU% and lower GPU busy fraction; prefill worse (static shapes force chunk padding); RSS up by the compiled model's size
kill: a text-hash mismatch (a parity failure, not a perf result); decode worse than all-Metal by more than the bound
memory gate: MG-3, all five clauses
rollback: the arm is behind the feature; nothing on main
blast: two examples
observe: the bench table
row: the `decode_arms` row shape from the round-two result, plus the ANE columns
reprove: command 2

#### CARD 5.1: route-predictor hit rate on the captured routes (CPU only, no ANE)

tier: worker
depends_on: [0.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: none (CPU only; still never while a bench runs)
opens: harness subcommand `predict-route`
commands:
1. `cd <tree> && cargo run --release -p proxima-model-interop --features std --example ane_lanes -- predict-route --routes runs/0.1-routes.bin --schemes prev1,prev2,embedding,layer-input-mlp --holdout 0.2 > runs/5.1.log 2>&1; echo EXIT=$?` (prev1/prev2: a table keyed on the previous one or two layers' selected sets, EdgeMoE's signal; embedding: a linear probe from the token embedding, SiDA's signal; layer-input-mlp: a 2-layer MLP on the layer input, ProMoE's signal; all fit on 80% of the captured tokens and scored on the held-out 20%, per layer)
expect:
- N1 = held-out routes scored per scheme: 0.2 x 27,072 = 5,414 per scheme, per layer counts printed
- N2 = per scheme and per layer: top-8 set hit rate, top-1 hit rate, fraction of tokens with all 8 predicted; the break-even line
- N4 = per scheme: CPU time per token for the prediction over all layers (p50 over the held-out tokens, single performance core), beside the ANE floor from 1.1; this is the CPU-predictor fallback's cost
predict: prev2 reaches 80 to 90% top-8 on granite; layer-input-mlp 85 to 95% (ProMoE reports 84.7%, SiDA 91.7 to 99.0%, EdgeMoE one example at 87.1%, all on other models, so these are guesses)
kill: every scheme under 60% top-8 (the prefetch idea does not pay on this model; the card says so and 4.x proceed without it)
memory gate: MG-2
rollback: none
blast: none
observe: break-even hit rate = ANE floor (1.1) / (8 experts x 3 projections x 19.9 us); printed beside each scheme's rate
row: `ROW <n>: route prediction: prev2 <%>, embedding <%>, layer-input-mlp <%> top-8 on 5,414 held-out routes; break-even <%>`
reprove: command 1

#### CARD 5.2: Q8_0 expert to CoreML weight-form fidelity (CPU only, no ANE)

tier: worker
depends_on: [0.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: none
opens: harness subcommand `weight-form`
commands:
1. `cd <tree> && cargo run --release -p proxima-model-interop --features std --example ane_lanes -- weight-form --model <granite blob> --layer 0 --forms fp16,palettized8,palettized4,linear8 --tokens 128 > runs/5.2.log 2>&1; echo EXIT=$?` (dequantize the real Q8_0 expert, re-encode in each CoreML form in Rust, run both through the CPU reference matmul in proxima-tensor on 128 captured router inputs, compare)
expect:
- N1 = 4 forms x 3 projections x 128 tokens compared: 1,536 rows
- N2 = per form: max-abs, cosine, bytes per expert against the GGUF block size (Q8_0: 34 bytes per 32 weights)
predict: fp16 and palettized8 within 1e-3 max-abs of Q8_0; palettized4 past 1e-2 because the palette is per-tensor or per-channel, not per-32-block; linear8 per-channel within 5e-3
kill: none; a form that fails fidelity is excluded from 2.1's arms, and the card lists the admitted forms
memory gate: MG-2
rollback: none
blast: none
observe: the admitted-forms list consumed by 2.1
row: `ROW <n>: weight forms: fp16 <e>, pal8 <e>, pal4 <e>, lin8 <e> max-abs; admitted: <list>`
reprove: command 1

#### CARD 5.3: the route predictor on the ANE, one dispatch per token for every layer

tier: worker
depends_on: [1.1, 5.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `predictor-ane` (takes the scheme 5.1 scored best, exports its weights as one CoreML graph whose input is the predictor's signal for one token and whose output is a [layers x experts] score matrix; one prediction per token)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- predictor-ane --routes runs/0.1-routes.bin --scheme <best from 5.1> --iters 1000 --units cpuOnly,cpuAndNeuralEngine > runs/5.3.log 2>&1; echo EXIT=$?` (interleaved per iteration across units; each prediction scored against the captured exact top-k for that token, all layers)
expect:
- N1 = 1000 predictions per unit, 2 units; each scored over 24 layers (24,000 layer-routes per unit)
- N2 = per unit: p50 and CoV per token; top-8 hit rate per layer, equal to 5.1's CPU number within 0.5 points (a drop means the fp16 export changed the predictor, and the card says so); the `MLComputePlan` device
- N3 = memory and CPU%
predict: ANE p50 for the whole-depth prediction is within 1.5x of the 1.1 floor (the predictor is tiny; the floor dominates); the hit rate matches 5.1
kill: ANE p50 above the Metal decode step it would run alongside (6.468 ms for granite); or hit rate drops more than 2 points from 5.1
memory gate: MG-2
rollback: none
blast: none
observe: predicted-expert set per token, available how many ms before the GPU's exact top-k for layer L (the lead time per layer, printed as a table over layers from the attr3 timeline)
row: `ROW <n>: predictor on ANE: <p50> ms per token (CoV <x>%), top-8 <%> vs CPU <%>, lead over GPU exact route at layer 12: <ms>`
reprove: command 2

#### CARD 5.4: what the prediction buys on paged experts (contingent on 5.3)

tier: worker
depends_on: [5.3]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: an arm in the existing external expert paging path (`proxima-model-interop/tests/external_expert_paging.rs` names the mechanism) that accepts a predicted expert set per token and issues the page-ins ahead of the layer; no library change beyond a hook the harness drives, behind the `coreml` feature
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- prefetch --model <granite blob> --resident-experts <fraction> --prompt <prompt1k.txt> --new 128 --arms none,oracle,ane-predicted --iters 3 > runs/5.4.log 2>&1; echo EXIT=$?` (granite with only a fraction of experts resident, the rest paged from disk, to stand in for a model larger than memory; `none` pages on demand, `oracle` prefetches the exact route one layer ahead, `ane-predicted` prefetches 5.3's set; interleaved; text hash must match across arms)
expect:
- N1 = 3 arms x 3 runs x 128 tokens; text hash equal across arms
- N2 = per arm: ms/token p50 and CoV, page-in count and bytes per token, miss count per token, GPU busy fraction, peak RSS
- N3 = `ane-predicted` miss count against `oracle` miss count: the gap is the predictor's cost in misses
predict: `oracle` recovers most of the on-demand penalty; `ane-predicted` lands between, closer to oracle when 5.1's hit rate is above 85%
kill: `ane-predicted` is no faster than `none` within the bound (the prediction does not pay at this hit rate and miss cost; the row records it as a negative)
memory gate: MG-3, clauses 1 to 3, with the resident-expert fraction stated
rollback: the arm is behind the feature
blast: the paging path's hook and the harness
observe: misses per token by arm; the derived ceiling: at this miss cost, the largest model the disk and the hit rate support at a target ms/token
row: `ROW <n>: prefetch: none <ms> / oracle <ms> / ane <ms> per token; misses <n>/<n>/<n>; implied ceiling <B params> at <ms/token>`
reprove: command 2

## Where the ANE's shape fits (owner, 2026-10-08: "hardware that is completely unused")

The ANE is a throughput engine for static fp16 graphs with a launch cost in
the 0.1 to 3 ms range; decode is a latency-bound chain of tiny dependent ops.
That mismatch is structural, so anything placed on the decode critical path
loses to the launch floor. The uses that fit are the ones OFF the critical
path, where the ANE's compute is free and its launch cost is hidden:

| use | why it fits | card |
| --- | --- | --- |
| route predictor, one dispatch per token | off path; drives prefetch for paged experts | 5.3, 5.4 |
| speculative draft source | the drafter runs while the GPU verifies the previous batch; k draft tokens per launch amortize the floor; the GPU is the bottleneck and the ANE is idle compute | 6.1 |
| prefill chunks at 512 tokens | static shape, big matmuls; the launch is amortized over the chunk; frees the GPU for a concurrent decode stream | 6.2 |
| embeddings and reranking | whole-graph fp16 inference, batch-friendly; the earlier ORT-CoreML card (gpu-one-risc 10.4) is the precedent | not in this set |

5.1 also times the CPU predictor per token, because if the ANE floor is too
high the CPU is the fallback: ProMoE runs a 2M-parameter MLP per layer on the
CPU at about 200 us with 84.7% accuracy, and the CPU is idle during decode here
(the bench's new CPU% column shows how idle).

#### CARD 6.1: the speculative draft head on the ANE

The repo's speculative decoding already has draft sources (the default
`ngram-simple`, and the draft specs under `proxima-tensor/specs/speculative-draft-*`).
A one-layer draft head (EAGLE/MTP shape) is a few tens of MB and runs k steps
per target step; on the ANE it runs while the GPU verifies.

tier: worker
depends_on: [1.1, 0.1]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `draft-ane`; whichever draft head the speculative path has implemented (read `proxima-model-interop/src/generate/` for the draft source trait and pick the smallest head with weights available for gemma4 E2B or granite; if only `ngram-simple` is implemented, this card measures the head's forward on the ANE in isolation and says so)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- draft-ane --model <blob> --head <name> --k 4,8 --iters 500 --units cpuOnly,cpuAndNeuralEngine > runs/6.1-head.log 2>&1; echo EXIT=$?` (the head alone: ms per k-token draft on each unit, interleaved; Metal's draft time for the same head from the existing speculative bench beside it)
3. if the head is wired as a draft source: the `decode_arms` protocol with arms `metal-draft` / `ane-draft` / `control`, granite and E2B, both prompts, every bench field; acceptance rate and draft tokens per step recorded per run
expect:
- N1 = 500 drafts x 2 k x 2 units; N2 = per (k, unit) p50 and CoV, and Metal's; N3 = for step 3, 21 runs per arm per prompt with text hashes equal across arms (speculative decoding is lossless under greedy; a hash change is a bug)
predict: a k=8 draft on the ANE costs under 2x the 1.1 floor (the head is tiny, the floor dominates); with the drafter off the GPU, ms/token improves by the GPU's draft share from the speculative bench, bounded by the acceptance rate
kill: ANE draft p50 exceeds the target's verify step (then the drafter cannot hide behind it); hash mismatch
memory gate: MG-3 for step 3
rollback: behind the feature
blast: harness; the draft-source wiring if step 3 runs
observe: GPU busy fraction with the drafter on the ANE vs on Metal; draft tokens per second per unit
row: `ROW <n>: ANE draft k=8 <ms> vs Metal <ms>; e2e <ms/token> vs <ms/token>, acceptance <%>, hashes equal`
reprove: commands 2 and 3

#### CARD 6.2: a prefill chunk on the ANE, and what it frees

tier: worker
depends_on: [1.3, 2.2]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `prefill-ane` (one full granite layer, attention plus the dense part of the FFN path, as a static 512-token MIL graph; experts stay on Metal)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- prefill-ane --model <granite blob> --layer 0 --chunk 512 --iters 100 > runs/6.2.log 2>&1; echo EXIT=$?` (ANE layer front at 512 tokens vs Metal's, interleaved; then both with a concurrent Metal decode stream running, to measure what the ANE frees)
expect: N1 = 100 iters x 2 arms x 2 conditions; N2 = p50 and CoV per cell; the decode stream's ms/token with and without the ANE prefill alongside; parity of the layer output
predict: the ANE front at 512 tokens is within 2x of Metal's (Metal prefill attention is 1149 us/op); the concurrent decode stream slows less with prefill on the ANE than with prefill on Metal
kill: the plan sends SDPA to CPU at 512 tokens (card 2.2's kill), or the ANE front is above 3x Metal
memory gate: MG-2
rollback: none
blast: none
observe: decode ms/token under concurrent prefill, both placements
row: `ROW <n>: ANE prefill front 512: <ms> vs Metal <ms>; concurrent decode <ms/token> vs <ms/token>`
reprove: command 2

#### CARD 6.3: pipelining across devices (owner, 2026-10-08: "what about pipelining?")

Within one sequence a token passes every layer in order, so a two-device
pipeline cannot shorten one token's path; it can only overlap token t of
stream A on the ANE with token t of stream B on the GPU. Pipelining is a
throughput lever that needs at least two sequences in flight (or speculative
batches), and it does not move the single-stream ms/token scoreboard. A stage
is a CONTIGUOUS block of whole layers, experts included, so the handoff is one
per token per boundary rather than one per layer. What a 1 GB graph holds as
whole layers (per-layer weights = attention + all experts; configs as in the
router table):

| model | per-layer weights fp16 | int8 | 4-bit | layers per 1 GB graph (int8 / 4-bit) |
| --- | --- | --- | --- | --- |
| granite 3.1 1b MoE | 110 MB | 55 MB | 28 MB | 18 / 36 (the whole model at 4-bit) |
| gemma 4 26B-A4B | 1.5 GB | 760 MB | 380 MB | 1 / 2 |
| Qwen3.6-35B-A3B | 1.6 GB | 810 MB | 400 MB | 1 / 2 |

So a pipeline stage on the ANE holds a real block of layers only for a
granite-sized model; for the local 26B and 35B models the ANE stage would be
one or two layers, and the boundary floor is paid for almost nothing.

tier: worker
depends_on: [1.3, 2.2]
export: as 0.1
target_dir: /private/tmp/cargo_target_ane_01
lock: gpu.lock
opens: harness subcommand `pipeline` (granite layers 0..K as one static MIL graph at 1 token, int8 or 4-bit weights, experts included with top-k inside the graph; layers K..24 on Metal; two independent sequences interleaved so the ANE stage of one overlaps the Metal stage of the other)
commands:
1. quiet check as 1.1
2. `cd <tree> && /usr/bin/time -l cargo run --release -p proxima-model-interop --features std,metal,coreml --example ane_lanes -- pipeline --model <granite blob> --ane-layers 8,12,16 --streams 1,2,4 --new 128 --iters 3 > runs/6.3.log 2>&1; echo EXIT=$?` (arms: all-Metal at the same stream counts, interleaved; text hashes per stream must match all-Metal)
expect:
- N1 = 3 splits x 3 stream counts x 3 runs x 128 tokens, plus the all-Metal arms; hashes equal
- N2 = per cell: single-stream ms/token, aggregate tokens per second, GPU busy fraction, ANE time per token, handoff per token, peak RSS and footprint, CPU%, CoV
predict: single-stream ms/token is worse than all-Metal at every split (one ANE launch plus a handoff per token, on the path); aggregate tokens per second at 2 streams exceeds all-Metal's 2-stream figure only if the ANE stage time is under the Metal stage time for the other stream; at 4 streams the GPU is the bottleneck either way
kill: the MoE top-k inside a static MIL graph is refused by the compiler (then the ANE stage is attention-only and the card says so, and the per-layer boundary returns); or aggregate tokens per second never exceeds all-Metal at any split

Kill pre-triggered by sources, 2026-10-08 (`docs/research/coreml-topk-ane-2026-10-08.md`):
MIL has a `topk` op, but on the M1 the ANE code generator rejects top-k, sort
and dynamic slice (reverse-engineered ANE paper, arXiv 2606.22283, sec 4.4);
gather on the M1 is valid only at batch 1, depth 1, a three-element index
channel (sec 4.2); scatter has no path on any family M1 through M5 (table
4.1). NPUMoE keeps routing, gather and scatter on the CPU and compiles the
experts as a static grouped dense FFN with fixed per-layer capacity tiers,
gathering routed tokens into expert slices before each invocation. So on this
box a whole-layer MoE stage on the ANE does not exist as a single graph. What
remains for 6.3 is NPUMoE's shape: attention on the ANE, experts as
capacity-tiered static groups on the ANE with the gather on the host, routing
on the CPU or GPU, and the boundary paid per group per layer. That is a
different and heavier card; it runs only if 1.1's floor and 2.1's crossover
make it worth writing. Card 5.3 is unaffected: the predictor emits a
[layers x experts] score matrix and the top-k over 24 x 32 scores runs on
the CPU in microseconds.
memory gate: MG-3, clauses 1 to 3
rollback: none
blast: harness
observe: the overlap: ANE stage time vs Metal stage time per stream, and the sum vs the measured step
row: `ROW <n>: pipeline K=<n>: 1-stream <ms/token> vs Metal <ms/token>; 2-stream aggregate <tok/s> vs <tok/s>; 4-stream <tok/s> vs <tok/s>`
reprove: command 2

## The knowledge side

Embedding and reranking are whole-graph fp16 inference over fixed token
windows, batched at ingest, with no per-layer handoff: the ANE's shape, and
the strongest fit in this set. Those lanes belong to the consumers of omega,
not to this repo; each consumer carries its own campaign that depends on cards
0.1 and 1.1 here.

---
## Rows (filled as cards run; the discipline-log numbering continues main's)

| card | status | predict | observed | miss category + work item |
| --- | --- | --- | --- | --- |
| 0.1 | | all 4 units; ANE for innerProduct; 27,072 routes | | |
| 1.1 | | ANE p50 0.3 to 3 ms | | |
| 1.2 | | copy < 20 us; iosurface zero-copy | | |
| 1.3 | | ceiling 1024 to 1536 MB; 2-graph swap < 2x | | |
| 5.3 | | ANE predictor within 1.5x of the floor; hit rate = 5.1 | | |
| 5.4 | | ane-predicted between none and oracle | | |
| 6.1 | | k=8 draft < 2x floor; e2e gains the GPU's draft share | | |
| 6.2 | | front at 512 within 2x; concurrent decode slows less | | |
| 6.3 | pre-killed by sources (no top-k, gather, scatter on the M1 ANE) | | | NPUMoE-shaped rewrite only if 1.1 and 2.1 warrant |
| 2.1 | | crossover >= 128 tokens | | |
| 2.2 | | 1 token 0.2 to 0.5 ms; SDPA on ANE | | |
| 3.1 | | < 15% mutual slowdown | | |
| 4.1 | | hybrid within 1.5x | | |
| 4.2 | | decode within 10%, lower CPU% | | |
| 5.1 | | prev2 80 to 90% | | |
| 5.2 | | pal4 fails, others pass | | |
