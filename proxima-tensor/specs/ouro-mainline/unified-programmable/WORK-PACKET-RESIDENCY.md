# packet: shared residency, bounded plans and persistent state

**Route:** GPT-6 Sol for contract/source changes. This crosses mmap lifetime,
Metal no-copy ownership, graph cuts, model binding, KV state and completion
fences. GPT-6 Luna may do a mechanically bounded source census or native build
after the interface is frozen. Opus is unavailable in the active model catalog;
the paper plan used GPT-6 Astra for this foundational design.

**Owned paths:** begin with `proxima-model-interop/src/mapping_residency.rs`,
`proxima-model-interop/src/generate/pregather.rs`,
`proxima-model-interop/src/generate/model_weights.rs`,
`proxima-model-interop/src/bind.rs`,
`proxima-model-interop/src/generate/residency_caches.rs`,
`proxima-model-interop/src/expert_sidecar.rs`, `proxima-model-interop/src/expert_slab.rs`,
`proxima-model-interop/src/generate/decode.rs`, `omega/src/metal/`, and the
existing graph partition/Plan owners. Keep changes within existing owners;
do not create a second model executor, whole alternate weight store, or
Ouro-specific pipeline. Root reviews and integrates isolated changes.

**Required evidence opened before edits:**

- `results/checkpoints/e2e-checkpoint-9/RUN-STATUS.json`
- `results/checkpoints/e2e-checkpoint-9/vmmap-summary-pid1142.txt`
- `results/checkpoints/e2e-checkpoint-9/vmmap-pid1142.txt`
- `results/checkpoints/e2e-checkpoint-9/proxima/plan_sources.jsonl`
- `results/checkpoints/e2e-checkpoint-9/proxima/resources.jsonl`
- `results/checkpoints/e2e-checkpoint-9/MEMORY-REGION-INVESTIGATION.md`
- `results/executor/RESIDENCY-500MB-SOURCE-EVIDENCE.md`
- `results/executor/RESIDENCY-500MB-LEDGER.jsonl`

The checkpoint9 run is a retained partial source-capture condition. It exceeded
the requested physical-footprint ceiling; Granite and llama did not execute.
Keep `RSS`, physical footprint, mapped resident pages and Metal device-wide
allocation as distinct measures. Do not infer address ownership from matching
aggregate sizes. The captured malloc/graphics joins remain unresolved.

**Source-traced mechanisms to preserve:** shared load calls whole-slice
`prove_resident` before mapping registration; a no-copy Metal mapping borrows
the mmap and must survive its final submitted command. Existing per-layer
checkpoint discard and selected expert-range discard have production callers;
they do not prove lock retention, completion ordering or reclamation. Dense
binding falls back to owned F32 conversion for BF16 embedding tables. The first
Qwen plan contains output-head intermediate node3192 with shape
`[1, 2048, 151936]` F32 and a `keep::reduce fold` consumer; its retained arena
slots and live peak are different values. Existing layer-window graph execution
is Qwen35-specific and is an adaptation seam, not proof of general dense or
Granite segmentation.

**Execution contract:** one reservation owner accounts the overlapping
process-backed live set across baseline/runtime, checkpoint page unions,
converted/staged weights, device slots, Plan caches, KV/recurrent state,
cross-cut values, logits/readback, compiler/source capture, transitions and
release-after-fence overlap. Each record carries owner, pointer/range, backing
identity, alias, bytes, generation, phase, lease, source location and completion
fence. Unknown ownership remains charged and unresolved. Admission happens
before allocation and rejects over-budget work with a typed result; it cannot
silently reduce context, token budget, precision, concurrency or routed experts.

**Design constraints:**

- Keep mapping/window and GPU view leases valid through the final completion
  fence; discard advice is not a release event.
- Preserve original checkpoint bytes/codecs and numeric/reduction semantics.
- Bound both Plan live peak and retained output slots; do not retain all layer
  plans at once.
- Reuse exact existing graph partition, plan, expert-range, KV, serving and
  backend owners. No whole-model copy or parallel executor.
- Group MoE only after actual router decisions. Every selected expert and route
  weight remains represented in the per-token output trace.
- Avoid new knobs that have no consumer. Hard limits are not strategy
  preferences.
- Follow lock-free ownership and no_std/alloc boundaries where those owners
  compile. Build-time defaults remain the restricted tier source.

**Dependency order:** (1) immutable checkpoint9+runner and raw allocation
ledger; (2) lifecycle admission/reservation and event records; (3) finite mapped
weight windows with fence leases; (4) dense embedding/output and bounded graph
segments/arena slots; (5) exact KV and routed expert windowing; (6) cache,
compile/capture and old/new generation transition closure. These substeps form
one memory integration checkpoint because mapping, tensor intermediates, KV
and retained slots overlap in the same process budget.

**Checkpoint gate:** every integrated source checkpoint must run exact Qwen3
dense and Granite MoE real-model paths plus paired llama.cpp inputs through the
shared execution route, under a maximum process physical footprint of
500,000,000 bytes from startup through teardown. Preserve complete generated
IDs/text/EOS, prompts, per-token timings, model digests, dispatches, Metal
source/hash, routes, KV/cut records, allocation/fence events, CPU/RSS/physical
footprint, errors, all five raw samples and N-1 CoV. A cap rejection or stopped
model run remains an incomplete checkpoint; continue this same memory boundary.
Use a new immutable checkpoint path and preserve losses and failed rows.

**Required controls and literal sources:**

- Exact-fit and one-byte-over budget: `fixtures.json#/memory_admission`,
  expected admission true, one typed rejection, and zero allocations/dispatches
  after rejection.
- Failed pin, missing window, stale generation, release before fence and
  overlapping-page double-charge: `memory_admission.literal_ledger.lifecycle_controls`,
  one rejected record each.
- Model/policy/context/position/logical-visit/allocation-generation mutations:
  `fixtures.json#/residency_strategy`, seven identity rejections.
- Missing dependency/cross-cut hazard: `fixtures.json#/dependency_dag`, one
  hazard violation (AC2).
- Corrupted token/embedding or output coordinate: `fixtures.json#/decode_strategy`,
  one comparator rejection. Dense BF16 row bytes must also be checked against
  `fixtures.json#/weight_conversion` with conversion denied producing zero
  converted bytes.
- Omitted/reordered KV state and cancellation isolation:
  `fixtures.json#/packed_decode`, `#/reset_isolation` and `#/cancellation`,
  exact expected IDs, positions, lengths, slots and output counts.
- Wrong selected expert/route: `fixtures.json#/moe_grouping`, one changed-route
  rejection; the model checkpoint retains actual Granite route IDs.

Each control uses the production checker and literal oracle, retains its raw
row, and has the exact rejection count stated above.

**Rollback:** preserve the full checkpoint9 source condition and payloads. Keep
each owner adaptation as a focused diff. If model outputs, KV/routes, lifecycle
or cap evidence refutes a subchange, restore that source condition and rerun its
controls; do not restore an unrestricted whole-file-prefault path for a capped
run. Do not commit.

**Artifacts:**
`results/checkpoints/e2e-checkpoint-N/`,
`results/serving/raw/`, `results/executor/`,
`results/recount/`, plus exact diffs and command output. Every ledger row points
to its raw owner event/address record. Record unknown ranges explicitly.

**Abandoned designs:** deleting whole-file residency proof and advising pages
away after each token would permit GPU reads from missing no-copy pages and
would leave the over-cap whole Plan. Fixing only cache/scheduler controls would
leave whole-file prefault, dense owned conversion and retained Plan output
slots. Saving every segmented Plan would keep all its arenas alive. Attribution
of the equal-sized MALLOC_LARGE and graphics ranges by size alone is not
admissible.
