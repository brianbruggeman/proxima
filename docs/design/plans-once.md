# plans once

A model's execution plan is built once per structural class, at load. A request
or a step changes only runtime values. Nothing is re-bound, re-fused, re-emitted,
refit, patched, evicted or cached per step.

## why the current design cannot do that

The program is already symbolic. `Extent::Symbolic(slot)`
(`proxima-tensor/src/op.rs:45`) names slot 0 = new rows, slot 1 = cached-key
extent, slot 2 = sliding-window rows (`SLIDING_KV_SYMBOL`, `op.rs:56`).

The symbols die at the first step. `infer(program, symbols)`
(`proxima-tensor/src/shape.rs:707`) returns `Shapes { extents: Vec<Vec<u64>> }`
(`shape.rs:46`): concrete numbers. Everything after it is concrete: bind, fusion,
the bound ops' extents and strides, the emitted uniforms, the pool schedule.

So a plan describes exactly one `(rows, kv)`. Every step that moves either number
runs `CudaPlan::refit_symbols` (`omega/src/cuda_driver.rs:2041`): re-infer, then
patch cached attention (`:2176`), else `refresh_bound_resolved`
(`proxima-tensor/src/bind/refresh.rs`), else a full `bind_resolved` + emit with
prior-kernel reuse (`:2121`), plus a pool rebuild (`:2147`, `:2218`). The plan key
carries `kv_bound_extent` (`proxima-model-interop/src/generate/cuda_resident_plans.rs:27`),
so decode refits every token. The resident plan store, width classes, eviction,
take/put and drain exist to keep refits rare.

Every plan defect found on 2026-11-02 lives in that compensating machinery:

- kernel text reused across a length change with a stale baked extent
- the plan rebuilt per request
- refit refused across the 1-row and kv=0 boundaries
- the logits plan evicting the middle-chunk plan
- a stale `append_cached_rows` after flush and re-upload

## the design

1. **Symbolic shapes.** Each dimension is an affine form
   `constant + sum(coefficient * symbol[slot])`. Inference keeps the form. The
   concrete `Shapes` becomes evaluation of the forms at a symbol vector, so every
   existing caller keeps working while it migrates.
2. **Bind once per class.** The binder takes a class: the facts every value of
   that class satisfies. A fusion or lowering decision that depends on a symbol's
   value is either true for the whole class (keep it) or not (it is a class
   boundary). Known boundaries: rows == 1 vs rows >= 2 (the elementwise body
   recomposes differently, `RecomposedBodyMismatch`); the output set (with or
   without logits). kv == 0 vs kv >= 1 is a boundary only if the attention kernel
   cannot take cached length 0 at runtime; the static kernel is the cached kernel
   at length 0, so the target is one class.
3. **Kernels take symbols at runtime.** Kernel text is already extent-agnostic
   (extents and strides come from uniforms, `omega/src/cuda.rs` text reuse fix of
   2026-11-02). A uniform field whose value is an affine form is computed on the
   device from one per-step symbol buffer (a few `u64`), not repacked on the host.
   The host uploads that buffer once per step and nothing else.
4. **Memory at class capacity.** Buffers and the pool schedule are sized for the
   class's maximum (prefill chunk width, KV capacity), once. Growth is a
   capacity event at the request edge, never a per-step rebuild.
5. **What is deleted.** The refit, refresh, patch, full-rebind and prior-kernel
   reuse paths; the resident plan store and its classes, eviction, take/put and
   drain; per-step uniform repack. What stays: one plan per class, held by the
   loaded model.

## worked example (qwen3-0.6b, layer 0)

The K projection: weights `[1024, 1024]`, activations `[rows, 1024]`, output
`[rows, 1024]`. Output extent 0 = `0 + 1*s0`. Prefill chunk 512, then decode.

| step | s0 | s1 | today | plans once |
|---|---|---|---|---|
| prefill chunk 1 | 512 | 0 | build or take plan (512, 0) | prefill class; upload `[512, 0]` |
| prefill chunk 2 | 512 | 512 | refit (512, 512): patch attention | same plan; upload `[512, 512]` |
| last chunk | 512 | 1024 | plan with logits outputs | prefill+logits class; upload `[512, 1024]` |
| decode 1 | 1 | 1536 | take or build decode plan, refit | decode class; upload `[1, 1536]` |
| decode 2 | 1 | 1537 | refit: patch 28 attention ops, repack | same plan; upload `[1, 1537]` |

The matmul kernel computes `token_total = 0 + 1*s0` from the symbol buffer.
Cached attention computes key rows `0 + 1*s1`. No host work changes between rows
of the table except the upload.

## proof obligations

- Affine closure: every dimension of every op in real programs is affine in the
  symbols. Measured by symbolic inference over the qwen3, Ternary-Bonsai-8B,
  gemma4-shaped and qwen35-shaped synthetic programs. Any non-affine dimension is
  listed with its op.
- Class soundness: for each class, binding at two symbol vectors inside the class
  yields identical bound ops after symbol substitution. Tested on every class.
- Parity: ids and logits equal today's path for prefill, decode, chunked prefill
  over a warm cache, and speculative verify, on every backend.
- No per-step host work: a step's host work is one symbol upload plus launches.
  Measured by host timing per step before and after.

## slices

1. Symbolic shapes in `proxima-tensor`, with concrete evaluation as the
   compatibility path; the affine-closure measurement over real programs.
2. Class-parameterized bind; the class-soundness test; the class set found from
   data, not assumed.
3. Device-side symbol evaluation in uniforms (CUDA, MSL, WGSL from the shared
   lowering in `omega/src/lowering.rs`).
4. Plans built at load per class, held by the loaded model; delete the
   compensating machinery listed above.
