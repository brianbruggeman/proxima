# Design task: the Metal decode path as pipes, sans-IO FSMs, RISC algebra, generic

Owner directive (verbatim, binding): "pipe shaped, fsm x sansio + fsm x orchestration over pipes
and also I want you to make sure that we are using our risc architecture and algebra. it should
be _generic_". Also binding: "less work + same output = keep" and "llama is not the floor" (the
floor is bytes / measured device ceiling; below it, fewer bytes).

Repo: /Users/brianbruggeman/repos/slot-0/proxima at main (3b9735e or later). Read
/Users/brianbruggeman/repos/slot-0/AGENTS.md and proxima/AGENTS.md first (pipe algebra: four forms
transform/source/sink/observe named by In/Out, one Pipe trait, the two binary questions, no
blanket impls, no newtype to host an impl, no_std tiers, sizing config, telemetry).

## State of the path (measured, main 3b9735e)
openchat-3.5 Q4_K_S decode on M1 Max: 33.0 ms/token (llama.cpp 17.45 on the same box; device
ceiling being measured). One command buffer per token, 616 dispatches (224 packed-row matvecs,
32 fused CachedAttention, ~360 elementwise/cooperative), 419 dataflow barriers, 1 readback.
Default features: output placement, wide cooperative reduce, kv bucket 32, Q5_K pair-dot,
plan-stable buffer arena + per-position uniforms, fused cached attention (coop K/V loads),
concurrent dispatch with HazardTracker.

## Audit findings the design must answer (file:line on main 3b9735e)
1. `BoundOpKind::CachedAttention` (`proxima-tensor/src/bind.rs:225-251`) is a 10-field macro-op
   carrying attention semantics (query_rows, kv_heads, query_groups, head_dim, scale, band
   bounds) — not Op/ScalarOp/IndexMap structure. Every other BoundOpKind is algebra. A program
   that is the same attention laid out differently silently falls back to the 7-op chain.
2. `render_cached_attention` (`omega/src/msl.rs:2578`) ignores operand `Layout.strides`; the
   matcher compensates with eight literal stride tuples (`bind.rs:2452-2474`) and rank gates
   (`bind.rs:2417-2427`, head_dim = pairs*2 hardcodes the even/odd RoPE split).
3. Single-range fused kernel runs 2t loop iterations for t of work: `bind.rs:2504-2506` sets
   cached_key_rows = new_key_rows = key_shape[0], `cached_lower = i64::MAX`, kernel loop
   `msl.rs:2586` skips the first half. Operands 4/5/7 duplicate 0/1/3 to fit a fixed 8/9-operand
   signature; `operands.len() == 8 | 9` is a runtime state discriminator read in four files
   (`bind.rs:237`, `cpu.rs:4847`, `msl.rs:2030`, `msl.rs:2543`); band sentinels i64::MIN/MAX.
4. Nine executor entry points = one driver × {named, placed, timed} (`omega/src/metal.rs:519,
   535, 978, 1177, 1192, 1210, 1409, 1484, 1509, 1613`); the block-upload loop is copy-pasted
   (`metal.rs:1039-1060` vs `1530-1553`). None is a Pipe. The driver (command buffer → encode
   → commit → wait → readback → sample) is a bespoke function ×4.
5. Hidden state machines (struct+methods or loop-mutated sets, none an enum): HazardTracker
   (`metal.rs:839-887`), the encode loop (`1076-1133`), BufferArena construction (`3585-3612`),
   OUTPUT_BUFFER_POOL lifecycle (`2551`), UNIFORM_BUFFERS LRU (`3260-3325`), Plan two-phase
   init via `mark_resident` (`373`), the one-entry plan cache cleared on miss ×3
   (`generate.rs:1417, 1325, 1373`), the decode step closure (`generate.rs:2278-2545`),
   `Counter::snapshot_and_reset` "exactly once per step" protocol (`metal.rs:2719`).
6. `plan()` performs device IO (`metal.rs:490-500`: device_and_queue + arena + uniform
   buffers) — planning is not sans-IO; the arena is built for all four executors and used by
   two. No builder/config surface on the path.
7. Eight `thread_local! RefCell` globals (`metal.rs:266, 273, 2551, 2939, 3038, 3162, 3260,
   3266`) + `register_checkpoint_mapping` side channel; a Plan is not a self-contained value.
8. Per-token allocation storm in the decode closure (`generate.rs:2301-2306` rebuilds
   named_blocks every step; `2323, 2361-2364, 2400-2406, 2504`; `metal.rs:1032-1040,
   1095-1100, 1073`); no allocation-counter test.
9. `fuse_cached_attention: bool` (`bind.rs:2635-2640`) collapses "which fused kinds can this
   backend render" to a bool; bind_plain and both matchers run twice per plan (`2641/2678`).
10. Generic: production decode builds programs from hand-written model functions
    (`spec.rs:898..7184` append_mistral_*/qwen35_*; `generate.rs:624 SingleRangeProgram`,
    `731 Qwen35SsmShape`) while `proxima-tensor/specs/mistral_layer.toml` + a test
    (`spec.rs:10317`) prove the data path exists and production does not use it.
    `CachedLayerRoots = (NodeId, NodeId, NodeId)` (`spec.rs:2333`) positional; 23-parameter
    layer builders with `#[allow(too_many_arguments)]`. `cached_len` carried as Float32.
    `KV_BUCKET_TOKENS` (Metal cache-key policy) lives in the IR crate `proxima-tensor/src/sized.rs`.
11. Barrier policy: `HazardTracker::reset` clears both sets on any barrier and the barrier is
    `MTLBarrierScope::Buffers` (all buffers); the schedule is a static property of the plan
    recomputed per token. `memoryBarrierWithResources` exists.
12. `classify_kind` (`metal.rs:1631-1692`) substring-greps generated MSL to recover the
    emitter's own routing decision; profiler groups by &str.
13. Algebra blockers already verified: `unify_iteration_space` (shape.rs:225-229) resolves an
    axis extent only from a single-term coeff-1 offset-0 operand axis; `Multiply` is arity 2
    (op.rs:95-103); `CachedLayerRoots` consumed at 15+ sites — RoPE even/odd fusion and the
    Identity copies are blocked by these.

## What the design must deliver (concrete Rust signatures, a principle cited per decision)
A. Attention in the RISC algebra: how a softmax-weighted banded reduction over a key axis is
   expressed with Op/ScalarOp/IndexMap (+ what minimal algebra extension, if any, is REQUIRED —
   e.g. a tuple-valued/online reduce monoid, an axis-band IndexMap, or wider arity — and the
   proof that nothing smaller works), such that the fused kernel is a fusion RULE over that
   structure (any program with the structure fuses; strides come from Layout; no operand-count
   discriminator; no model name), and the CPU evaluator and every emitter implement the same
   rule. Include the migration from `BoundOpKind::CachedAttention`.
B. The decode step as FSM × orchestration over pipes: the sans-IO step state machine (enum
   states, transitions consuming old → new), the pipe composition of the driver
   (which form each stage is, by In/Out), where the single Metal edge lives, how the
   HazardTracker/arena/uniforms/plan cache become sans-IO transforms computed at plan time
   (the barrier schedule as a static plan property), how `plan()` becomes pure, how the eight
   thread-locals become owned state, and the allocation budget per token (target: zero on plan
   hits) with the counter test that proves it. Collapse the nine executors to one driver.
C. Generic model programs: production decode built from spec data (the TOML path), the layer
   builder's 23 NodeIds as a typed struct, `CachedLayerRoots` typed, `cached_len` as an integer
   leaf, KV bucket policy moved to the driver crate, the fused-kind capability as a set not a
   bool. Name what a new architecture needs to add: data only, or which Rust.
D. Bytes — THE PRIMARY SECTION. Owner target (2026-09-04): "realistically, I'd love to 5x
   llama" = 3.5 ms/token on this box. The M1 Max spec is 400 GB/s, so streaming all 4.169 GB per
   token has a floor of 10.4 ms: 5x llama is unreachable by kernel work — it requires ≤ 1.4 GB
   moved per generated token (≤ 1.0 GB at a realistic 300 GB/s ceiling). Every lever is a
   bytes-per-token lever and must be expressed in the RISC algebra as a program, not a special
   case: multi-token per pass (speculative/draft verification: a pass with new_count = k
   amortizes the weight stream k ways — the `s` axis exists; what does the algebra need for
   draft+verify as ONE program?), dynamic row elision / contextual sparsity of the FFN and
   projection matvecs (the csr sparse machinery: see memory pointers
   project_sparse_matmul_equivalence, project_sparse_dynamic_elision_probe ("generic
   DISPATCH-BOUND"), project_csr_structured_dense — an IndexMap::Computed gather over a
   selected row set IS the algebra form; what selects the rows, and is the selector itself a
   pipe?), lower-bit weights (Q4 → Q3/Q2/ternary) as codecs the program already abstracts,
   output.weight (107 MB) top-k/tied tricks, KV bytes. For each lever: bytes/token before → after,
   the algebra it needs (existing / extension with the two binary questions answered), the
   quality gate (text identical is NOT available for lossy levers — name the metric: exact-match
   rate vs the full model on a held-out prompt set, and the kill criterion), and the ORDER
   (the product of the levers must reach ≤ 1.4 GB/token; show the arithmetic).
E. Ordering + tripwires: a stepwise plan (cards) with the gate per card (parity ≤1e-4 vs cpu on
   the real program, 100× byte-identical, text identical, ms/token and gpu_exec_ms not worse
   beyond CoV, allocation counter), what each constraint (no_std tier, lock-free, reuse-first,
   pipe question) CHANGED in the design, and at least one design you abandoned because of it.

Output: a stepwise plan with signatures; ≤ 1500 lines; every claim about existing code cites
file:line you opened; no adjectives, no verdict words.
