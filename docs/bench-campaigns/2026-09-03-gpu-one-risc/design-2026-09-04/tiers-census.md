# Tier builds and module census (main f3c4e98, read-only gate run 2026-09-04)

| crate | --no-default-features | --features alloc | --features std |
|---|---|---|---|
| omega | EXIT 101 (`msl.rs:2546` `.to_string()` on `&str` without alloc import, 11 errors) | EXIT 0: compiles error, msl, sized (3 of 8 modules; backend/cuda/metal/wgpu cfg'd out) | EXIT 101 (`proxima_tensor::{Evaluated, QuantizedBlock}` unresolved: `std` does not enable proxima-tensor's std/cpu; non-exhaustive `&mut Plan` match) |
| proxima-tensor | 0 | 0 (align bind error live map op partition shape sized + convert dtype physical) | 0 |
| proxima-gguf | 0 | 0 (all but config, edge) | 0 |
| proxima-tokenizer | 0 | 0 (all but config, gguf, hf) | 0 |
| proxima-model-interop | 0 (bind capability dtype error hf_config serving transform) | NO `alloc` FEATURE EXISTS | 0 |

All five declare `#![cfg_attr(not(feature = "std"), no_std)]`. Every alloc build compiled its crate
(rustc invocation count 1 each; none N==0). No `async_trait`/`Pin<Box` anywhere.

Census: `omega/src/metal.rs` 4796 lines, 12 std refs, 87 alloc refs, 46 `dyn` sites (all
`ProtocolObject<dyn MTL*>` FFI), 6 `thread_local!` (252, 2477, 2933, 3007, 3150, 3246); `msl.rs`
6576 lines, 0 dyn, 0 thread_local, ungated (alloc tier); `wgsl.rs` 1948, `cuda.rs` 1857, both 0 dyn.
`proxima-tensor/src/cpu.rs` 27,829 lines cfg(std), 53 std refs; `spec.rs` 14,354 lines cfg(config);
`instrument.rs` one `std::thread_local!` (1593). omega public surface: 67 unique items, no `pub trait`.

# Lowering-choice audit (scratchpad/lowering-audit.md): 43 decision points — 16 by STRUCTURE,
27 by NAME (14 cargo feature flags, 4 substrings of generated source, 2 env-var/tensor-name,
rest Option/arity/bool). Top findings: (1) `kernel_cache_key` (msl.rs:930-981) omits the reduce
extents while `cooperative_reduce_width` bakes the extent-derived width into the source
(msl.rs:4390-4406, 4540) and `pipeline_for` (metal.rs:2374-2400) serves cache hits without
re-emitting → latent wrong-kernel-served (READ, not reproduced); (2) `classify_kind` greps MSL;
(3) 14 feature flags select kernel bodies, incompatible pairs hand-guarded (msl.rs:3289-3315,
4310-4319, 4446-4459); (4) CachedAttention CISC, wgsl/cuda cannot lower it; (5) the fusion
template misses silently (no rejection enum, unlike packed-row/tiled-gemm); (6)
`PACKED_ROW_BLOCK_SIMDGROUPS` only consumer is unreachable (msl.rs:4322) — ROW 234's sweep
measured a constant that never reached a dispatch; (7) `encode_op` allocates per op per step
(kernel_cache_key Strings, Vecs; metal.rs:3772-3773, msl.rs:860-1003); (8) no shared lowering
layer: 28 function names duplicated across msl/wgsl/cuda, `reduce_is_cooperative` ×3 with
diverged signatures; (9) 8 execute entry points; (10) `OMEGA_BACKEND` env with silent fallback
(backend.rs:128-136); (11) `mark_resident` discards its names (backend.rs:400); (12)
`element_type == "float"` string gate admits Int32/Bool (msl.rs:3274-3277); (13) `EmitError`
kinds as `&'static str`; (14) error.rs doc contradicts wgsl.rs:207-208; (15) `Backend` 7 variants,
3 executable, cuda text never executed; (16) arity literal `quantized.len() != 2` (msl.rs:1404).
