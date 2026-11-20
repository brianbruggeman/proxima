# PROXNUM-598 design: one generic numeric interchange for proxima-numeric

## Re-scope, 2026-11-20

Owner: "why use f64 when we know that bf6 probably is 99% as efficient". Every format of 32 bits or
narrower (bf16, f16, fp8, fp6 E3M2/E2M3, fp4 E2M1) is exactly representable in f32, so the f32 host
pivot loses nothing for them.

- **Deferred:** the wide u128 kernel, the sealed `Scalar` trait and the binary64 slices (slice 1,
  the f64 half of slice 2, slice 4). The design below stays as the record of that option.
- **Promoted:** `Numerics` as conflaguration config (slice 5, item 3), the `DenseMmaInputLayout`
  fallback to `FloatLayout::from_name` (slice 5, item 2), per-backend capability rows, and the 586
  config slices.

Read on main `f24198af6`. Every `file:line` below is `git show main:<path>`; counts are `git grep` on non-comment lines and include tests and examples. Attribution of a count to a symbol was checked by hand only where a line is cited.

## 0. Shape chosen, and the one contested decision

**Shape: a sealed `Scalar` trait generic at the host boundary (candidate a), backed by ONE exact layout-defined wide kernel that is internal and never appears in a signature (candidate b demoted to an implementation detail).** The f32 hot kernels are untouched and become the `f32` instantiation. `FloatLayout` stays the only description of a number; a host float type is only the typed slot a codec reads or writes.

**The contested decision: do we pivot (one wide type everywhere) or monomorphise (generic over the host scalar)?** Pivoting loses on two readings of the source:

- The f32 kernels are hand-specialised on the f32 bit pattern: magic-add rounding (`float_layout_codec.rs:488-519`), embedding widen (`:685-707`), straight-line selection (`:543-563`), a native-f16 route (`:416-427`). A comment at `:948` records that forty inlined loop bodies decoded binary16 16x slower than the same loop alone. Routing f32 through a 128-bit unpack/pack pivot throws that work away.
- An f64 pivot is not wide enough for the layouts `FloatLayout` already names: `INT64`, `UINT64`, `INT128`, `UINT128` (`float_layout.rs:172-175`), and binary128 is expressible by the struct.

Monomorphising makes f32 the existing code and f64 the new code, with the 128-bit exact kernel as the shared definition of rounding for every layout.

Designs abandoned, and why:

1. **f64 pivot.** Loses INT64/INT128 and doubles scratch bandwidth on the f32 KV-cache paths.
2. **`impl<T: num_traits::Float> Scalar for T`.** A blanket impl over an open foreign set; AGENTS.md "No blanket impls" rules it out. The trait is sealed and has exactly two impls.
3. **`enum HostSlice<'a> { F32(&[f32]), F64(&[f64]) }`.** A runtime tag and a branch per call to recover what the type system already knows.
4. **Parallel `encode_f64`/`decode_f64` entry points.** Companion functions compensating for one non-generic signature; the defect is the signature.
5. **`Op::Constant.value: f64`.** Still a cap, one rung up (see slice 7).

## 1. Inventory

### 1a. proxima-numeric public API typed on a concrete float

Zero public `f64` anywhere in the crate (`git grep` on `src` finds f64 only in two tests, a doc, and `two_level.rs:174`). Every item below is `f32`.

| # | API | file:line |
|---|---|---|
| 1 | `float_layout_codec::encode_counting(f32, FloatLayout, &mut u64) -> u32` | float_layout_codec.rs:335 |
| 2 | `float_layout_codec::encode_with(f32, layout, Rounding, Overflow, &mut u64) -> u32` | :352 |
| 3 | `float_layout_codec::encode(f32, FloatLayout) -> u32` | :575 |
| 4 | `float_layout_codec::decode(u32, FloatLayout) -> f32` | :621 |
| 5 | `layout_supported` 32-bit / 23-mantissa / 8-exponent / 63-int domain (`MAX_EXPONENT_BITS`, `MAX_MANTISSA_BITS`) | :24, :27, :36-55 |
| 6 | `encode_body`, `decode_body<Slot>`, `encode_slice`, `decode_slice` (pub(crate), the kernels) | :790, :842, :921, :940 |
| 7 | `decode::decode_scalar -> f32` | decode.rs:28 |
| 8 | `decode::encode_scalar(f32, ..)` | decode.rs:53 |
| 9 | `decode::decode_block(.., &mut [f32])` | decode.rs:94 |
| 10 | `decode::decode_blocks(.., &mut [f32])` | decode.rs:135 |
| 11 | `encode::encode_blocks(.., &[f32], ..)` | encode.rs:620 |
| 12 | `encode::reach -> f32` | encode.rs:537 |
| 13 | `convert::narrow_to_format(f32, ..) -> f32` | convert.rs:51 |
| 14 | `convert::narrow_to_layout(f32, ..) -> f32` | convert.rs:69 |
| 15 | `FormatCodec::encode(&[f32], ..)` | format_codec.rs:217 |
| 16 | `FormatCodec::decode(.., &mut [MaybeUninit<f32>])` | format_codec.rs:279 |
| 17 | `EncodeFn`, `DecodeFn`, `BlockEncodeFn`, `BlockDecodeFn` fn-pointer aliases | block_codec.rs:36, 39, 44, 49 |
| 18 | `Numeric::{new, one, decode, values, scalar}` over `&[f32]` | value.rs:78, 101, 118, 136, 148 |
| 19 | `Numerics::decode_weights(.., &mut [f32])` | numerics.rs:51 |
| 20 | `PackedRows::{from_values, extend, overwrite, decode_range, decode_all}` | packed_rows.rs:76, 152, 172, 301, 317 |
| 21 | `Philox::fill_normal(&mut [f32])`, `Philox::normal -> Vec<f32>` | rng.rs:123, 135 |
| 22 | `FloatLayout::{max_normal, min_normal, min_subnormal} -> f32` (doc says "binary32 projection") | float_layout.rs:358, 376, 383 |
| 23 | `LevelFormat::level -> Option<f32>` | format.rs:86 |
| 24 | `range_fit::{nearest_even(f32), fit_range(.., &[f32; N])}` | range_fit.rs:20, 49 |
| 25 | `RangeFit.rmin / rdelta: f32` (pub fields, serde) | encode_rule.rs:73, 75 |
| 26 | `fused::{mul_add, min_term, scaled_sub}` x3 cfg variants | fused.rs:41-106 |
| 27 | `two_level::encode_two_level(.., &[f32], ..)` (pub(crate)) | two_level.rs:49 |

Items 22-27 are not codec pivots. The block formats' arithmetic is f32 because the incumbents (llama.cpp `quantize_row_*_ref`, ggml `make_qkx2_quants`) define it in f32 (`encode_rule.rs:66-69`, `block_codec.rs:17-21`). AGENTS.md and principle 14 make the incumbent the oracle there, so they stay f32 and become an explicit "compute layout" (section 3).

Already generic, no change needed: `DType` is `Bool | Layout(FloatLayout)` (`dtype.rs:9-12`) with `DType::FLOAT64` (`dtype.rs:70`); `NumericFormat`, `FloatLayout::F64` (`float_layout.rs:176`) and the `float64` name (`:453`) exist; `NumericPolicy` is a conflaguration `Settings` (`numeric.rs:23`).

### 1b. Direct callers of the f32-typed API (grep, non-comment lines, tests and examples included)

| crate | lines | files | where |
|---|---|---|---|
| proxima-tensor | 11 | 4 | cpu/quantized_eval.rs:3145-3146; cpu/run_block_quantize.rs:93,118,119,120,129; cpu/tests.rs:15768,15772; convert.rs (`max_normal`) |
| omega | 38 | 14 | msl/layout_render.rs:719,783; metal/execute_and_hazards.rs:1398,1432,1467,2399-2400; 9 in msl/layout_render_tests.rs; 5 integration tests |
| proxima-model-interop | 65 | 16 | generate/residency_caches.rs (25); device_kv.rs (7); kv_ring.rs (7); prefix_state_file.rs (4); chunk_shift.rs (4); ring_checkpoint.rs (3); convert.rs:108,111; cold_artifact.rs:221,728; tests/flux2_klein.rs:64 |
| proxima-autograd | 0 into proxima-numeric's f32 API | - | its own f32 surface is in 1c |
| **total** | **114** | **34** | proxima-gguf is outside the ask but is the heaviest user (FormatCodec 35, decode_blocks 11, encode_blocks 7 lines) and moves in the same slices |

Most numeric imports in these crates are metadata, not f32: `DType` (tensor 85, omega 204, interop 54, autograd 37 occurrences) and `NumericPolicy` (omega 206). Those are already layout-generic. The f32 pivot is concentrated at host buffers.

The 20 that matter most:

1. omega/src/config.rs:327-334, the 32-bit cap (`layout_width > 32 || exponent_bits > 8 || mantissa_bits > 23`).
2. omega/src/msl/layout_render.rs:719,783, `FormatCodec::for_format` for emit.
3. omega/src/metal/execute_and_hazards.rs:1398,1432,1467, `&FormatCodec` parameters on the upload/readback path.
4. proxima-model-interop/src/generate/residency_caches.rs: `decode_range` at 866-868, 1552, 1672; `decode_all` at 707, 2044. This is the KV hot path.
5. generate/device_kv.rs:356-361, `decode_range` for K-even, K-odd, V.
6. generate/kv_ring.rs:247,384, `decode_range` and `overwrite`.
7. generate/chunk_shift.rs:272-273, `decode_all`.
8. generate/prefix_state_file.rs:88, `decode_all` on the persisted state.
9. generate/ring_checkpoint.rs (3 lines).
10. proxima-model-interop/src/convert.rs:108,111, a local `encode_blocks(&[f32], layout)`.
11. proxima-model-interop/src/cold_artifact.rs:221,728, `max_normal`.
12. proxima-model-interop/tests/flux2_klein.rs:64, `Philox::normal` (S1; compared to the sd.cpp reference with a tolerance, not bits).
13. proxima-tensor/src/cpu/quantized_eval.rs:3145-3146, bf16 round trip `decode(encode(v, BF16), BF16)`.
14. proxima-tensor/src/cpu/run_block_quantize.rs:119-120, `scale_of` through `layout.scale.layout`.
15. run_block_quantize.rs:93,118,129, `reach -> f32`.
16. omega/tests/word_load_run_decode_parity.rs:61,65, `encode_blocks`/`decode_blocks`.
17. omega/tests/unnamed_scalar_render.rs:61, `FormatCodec::scalar` ("an unnamed scalar of at most 32 bits").
18. omega/src/msl/layout_render_tests.rs:489, `decode_scalar`.
19. omega/tests/support/grouped_gemm_oracle.rs:14, `encode`/`decode`.
20. omega/src/msl/tests.rs:47, `float_layout_codec::decode`.

### 1c. proxima-autograd numeric surface (training path)

`src` has no `pub fn` that takes an f32 into proxima-numeric. Its graphs are already `DType`-parametric (`relu(&mut program, dtype, x, rank)`, `activation.rs:38`). The f32 caps are:

- **Op literal:** `Op::Constant { dtype, shape, value: f32 }` (`proxima-tensor/src/op.rs:279-282`), built by `expr::constant(.., value: f32)` (`expr.rs:82`) and `broadcast_anchor(.., value: f32)` (`expr.rs:111`). `cpu/float64_island.rs` says in a comment that "an f64 constant's literal is an f32". A FLOAT64 graph has f32-precision constants today.
- **Hyperparameters:** `AdamConfig` and siblings `learning_rate/beta1/beta2/epsilon/decay/weight_decay: f32` (`optimizer.rs:73-81, 387, 488-494, 662-670`).
- **Host buffers:** `train::State = Vec<(String, Vec<f32>)>` (`train.rs:49`); `train_step`-family take `&[(&str, &[f32])]` and return `(f32, State)` (`train.rs:73-74, 114-115, 177-178`).
- **Low precision:** `low_precision::weight_view(&[f32], WeightFormat) -> Vec<f32>` (`low_precision.rs:20`).

CPU evaluation already runs FLOAT64 nodes in f64 (`proxima-tensor/src/cpu/float64_island.rs`, `Cow<[f64]>` wide buffers), so only the literal and the host boundary cap precision, not the compute.

## 2. Candidate shapes

### (a) A sealed scalar trait, codecs generic over it

```rust
// proxima-numeric/src/scalar.rs  (no_std, no alloc, no libm in the trait bounds)
mod sealed { pub trait Sealed {} impl Sealed for f32 {} impl Sealed for f64 {} }

pub trait Scalar: sealed::Sealed + Copy + PartialEq + core::fmt::Debug + 'static {
    /// the layout this host type is: BF32 for f32, F64 for f64
    const LAYOUT: FloatLayout;
    const BYTES: usize;
    fn from_le_slice(bytes: &[u8]) -> Self;
    fn write_le_slice(self, out: &mut [u8]);
    /// the scalar's own route: f32 -> today's kernels, f64 -> `wide`
    fn encode_run(layout: &FloatLayout, values: &[Self], out: &mut [u8], rounding: Rounding, overflow: Overflow, nan_count: &mut u64) -> Result<(), SliceFault>;
    fn decode_run<Slot>(layout: &FloatLayout, data: &[u8], out: &mut [Slot], write: impl Fn(&mut Slot, Self) + Copy) -> Result<(), SliceFault>;
    /// the Box-Muller transform of one Philox block, in this scalar's own precision
    fn from_philox(words: [u32; 4]) -> Self;
}

pub fn encode<S: Scalar>(value: S, layout: FloatLayout) -> u128;
pub fn decode<S: Scalar>(code: u128, layout: FloatLayout) -> S;
pub fn encode_with<S: Scalar>(value: S, layout: FloatLayout, rounding: Rounding, overflow: Overflow, nan_count: &mut u64) -> u128;
pub fn decode_scalar<S: Scalar>(format: &NumericFormat, bytes: &[u8]) -> Result<S, NumericError>;
pub fn encode_scalar<S: Scalar>(value: S, format: &NumericFormat, bytes: &mut [u8]) -> Result<usize, NumericError>;
pub fn narrow_to_layout<S: Scalar>(value: S, layout: FloatLayout, rounding: Rounding, overflow: Overflow) -> Result<S, NumericError>;
impl FormatCodec {
    pub fn encode<S: Scalar>(&self, values: &[S], out: &mut [u8]) -> Result<usize, NumericError>;
    pub fn decode<S: Scalar>(&self, data: &[u8], out: &mut [MaybeUninit<S>]) -> Result<usize, NumericError>;
}
```

`FormatCodec` stays one non-generic `Copy` value, so `Route` (`format_codec.rs:46-57`) does not double. `encode<S>` calls `S::encode_run`; the f32 impl is `#[inline(always)]` over today's `encode_body`. A `Numeric<'a, S: Scalar = f32>` default parameter keeps `Numeric<'a>` in type position compiling.

- **f32 hot-path cost:** by construction the same code, because the f32 hook is the existing function body and the generic wrapper is `inline(always)`. That is DERIVED from reading, not measured. The gate in slice 1 is a byte-identical disassembly of the f32 `encode_body`/`decode_body` symbols plus `benches/bench_kv_cache_codec.rs` with CoV under 5% on 3-5 runs.
- **Weakness of (a) alone:** the f64 hook would have to re-derive each hand-tuned kernel for u64 bit patterns, or fall to a slow generic path with nothing defining its rounding. It also cannot express layouts wider than 64 bits.
- **no_std/no-alloc:** `Scalar` has no allocation or libm in its bounds; `from_philox` uses `libm::{sqrtf, logf, sinf}` / `{sqrt, log, sin}`, already a dependency (`Cargo.toml`, `rng.rs:165`).
- **Omega IR / backends:** `DType` is already `Layout(FloatLayout)`, so the IR needs no new type; the trait only types the host slot.
- **Autograd:** generic `State<S>` and `train_step<S>`; hyperparameters stay config values (see section 3).
- **Config:** none by itself; `S` is a compile-time type.

### (b) A single wide pivot

```rust
// internal to float_layout_codec/wide.rs, never in a public signature
enum Unpacked {
    Zero { negative: bool },
    Finite { negative: bool, exponent: i32, significand: u128 }, // msb-normalised, exact for <= 113 bits
    Infinity { negative: bool },
    Nan { negative: bool, payload: u128 },
}
fn unpack(code: u128, layout: &FloatLayout) -> Unpacked;                                    // exact
fn pack(value: Unpacked, layout: &FloatLayout, rounding: Rounding, overflow: Overflow) -> u128; // the layout defines rounding
```

Each layout's rounding is defined by its `Specials`, `NanPolicy`, `bias_override` and `SignEncoding`, the data `FloatLayout` already carries (`float_layout.rs:8-110`). Every (source, destination) pair up to 128 bits is `unpack` then `pack`. As the public interchange it fails the f32 hot path: every f32 row would pay unpack and pack instead of the magic-add loop, and a 16-lane vectorisable loop becomes a branchy u128 one. Cost is ASSUMED, not measured. As an internal definition it is exactly right.

- **no_std/no-alloc:** `u128` is `core`; integer-only; no libm. A bare-metal build can compile it out with the sized cap below.
- **Omega IR:** `DType` carries the layout; `pack`/`unpack` run on whatever layout the dtype names.
- **Config:** nothing new; the layout is the config.

### (c) The recommendation: (a) on the surface, (b) underneath, plus byte-level routing

- Public signatures are `<S: Scalar>`. `S = f32` routes to the existing f32 kernels for every layout of at most 32 bits, as today.
- `S = f64`, and any layout wider than 32 bits for any `S`, routes to `wide::{unpack, pack}`.
- The same `wide` kernel is the oracle the f32 kernels are cross-checked against in tests: the f32 field-by-field reference tests (`float_layout_codec.rs:1086-1127`) gain a second, independent reference.
- `layout_supported` stops being "at most 32 bits" and becomes "structurally valid and at most `sized::MAX_LAYOUT_BITS`".

This is the combination that gives (a)'s zero-cost f32 and (b)'s exactness for f64, INT64/INT128 and binary128, with no new public type except the sealed trait.

## 3. Cross-cutting answers

**Block formats.** `FormatCodec` reports its arithmetic layout: add `FormatCodec::compute(&self) -> FloatLayout`, `BF32` for every block and level format (incumbent-defined in f32), the layout's own for scalar routes.

- Decoding a block format into `f64` is an exact widening of the incumbent's f32 value: allowed.
- Encoding `f64` into a block format would narrow to the compute layout. That silently destroys precision, which is the defect on the card. It returns `NumericError::LossyInput { from: S::LAYOUT, compute }` instead.
- The caller who wants the narrowing says so with the existing named seam, `Numeric::to(target, Rounding, Overflow)` (`value.rs:175`; `convert.rs:1-10` says every narrowing names its rounding and overflow).

This is where a `bool`/silent cap is replaced by a typed reason. It is also the honest limit of the design: Q4_K from f64 input is f32-arithmetic by the incumbent, and a block layout whose `scale.layout` is `F64` still gets its scale through `wide`.

**Hazard: unsuffixed float literals.** Making `encode(1.0, F16)` generic changes the inferred type of a bare `{float}` literal from `f32` to the integer-fallback `f64`. The call compiles and double-rounds differently (`0.1` narrowed from f64 versus from f32). Known sites include `numerics.rs:76` and `proxima-tensor/src/cpu/tests.rs:15768`. Slice 2 enables `clippy::default_numeric_fallback` (deny) on the four affected crates for the duration of the change and suffixes every flagged site `_f32`.

**no_std tiers (principle 3, 8, 12).**

- tier 1 (no_std + alloc): all of `Scalar`, `wide`, `FormatCodec`, `Numeric`. `PackedRows::decode_all<S>(&self, out: &mut Vec<S>)` stays behind `feature = "alloc"` as today (`lib.rs`).
- tier 3 (no alloc): same minus `PackedRows`; no function allocates, all kernels write into caller slices.
- A new `proxima-numeric/build.rs` plus `proxima-numeric.toml` (the proxima-telemetry `sized` pattern from the conflag skill, option B) emits `sized::MAX_LAYOUT_BITS` (default 128; a Cortex-M profile sets 32 and the u128 kernel folds out) and `sized::RNG_CHUNK` (default 64, the stack scratch for packed noise). Env override `PROXIMA_NUMERIC_LAYOUTS_MAX_WIDTH_BITS`. The magic `max: 32` at `float_layout_codec.rs:40` becomes this constant, and the runtime default is seeded from it, never retyped.

**Conflaguration surface: which keys choose compute and storage layouts.** `Numerics` (`numerics.rs:17-28`) already has the right fields (`weights`, `activation`, `accumulator`, `epilogue`, `policy`) and has no external caller: `git grep` finds it only inside proxima-numeric, so it is the dormant carrier.

- Give it `#[cfg_attr(feature = "config", derive(Settings, Serialize, Deserialize))]` with prefix `NUMERICS`, matching `NumericPolicy` (`numeric.rs:23-26`). `NumericFormat` already serialises by name or by structure (`format.rs:1109-1190`, `dtype.rs:144-188`).
- Nest it in omega's `RuntimeConfig` beside `numeric` (`config.rs:94`): `OMEGA_NUMERICS_WEIGHTS=q4_k`, `OMEGA_NUMERICS_ACTIVATION=float16`, `OMEGA_NUMERICS_ACCUMULATOR=float64`, or `[numerics] accumulator = "float64"`, or a structural `{ exponent_bits = 11, mantissa_bits = 52, ... }` for a flavor with no name. Defaults are `Numerics::REFERENCE` (BF32 everywhere), which is today's behaviour.
- `dense_mma_input_layout` stops accepting only `auto|f32|bf16` (`config.rs:254-261`): `FromEnvStr` falls back to `FloatLayout::from_name`, keeping `f32` as an alias.
- The host scalar is derived, not configured: the caller does one setup-time dispatch `if <f32 as Scalar>::admits(&numerics.accumulator_layout()) { run::<f32>() } else { run::<f64>() }`, with `const fn admits(layout) -> bool` on the trait. Choosing f64 is a config value; both instantiations are compiled in; hot loops stay monomorphic. This is config-as-composition (principle 4) without a runtime branch in a loop.
- `rng = "philox"` is already a key (`rng.rs:79-93`); the noise precision follows the destination node's dtype, not a second key.

**Reaching omega's backends: lift the cap per backend capability.**

- The cap at `config.rs:327-334` is a structural check wearing a capability's clothes. A layout's validity is the codec's question (`layout_supported`); whether a backend executes it is the backend's.
- Backend capability already exists as a value, `Executes { forms, limits, activation }` (`proxima-tensor/src/bind/executes.rs:~150-225`), carried in plan identity, with `DeviceLimits { memory }` (`:100-125`) and `ActivationLimits` as the same Copy shape. It has no dtype axis; each backend refuses late with `EmitError::UnsupportedDType` (`cuda.rs:1356`, `wgsl.rs:428-433`, `metal/pipeline_buffers_upload.rs:1359`, `msl/signature_tokens_prelude.rs:320`).
- Extend `DeviceLimits` with `pub compute: FloatLayout`, "the widest layout this backend's renderer executes" (the doc of `Form::Int8DotReduce` sets the precedent that a grant is what the renderer executes, not what the silicon can do). `intersect` takes the fieldwise minimum; `Refusal` is `#[non_exhaustive]`, so add `Refusal::Scalar { dtype }`; `Executes::admit(&BoundOp)` checks `op.dtype`. No new type: it reuses `FloatLayout` and the existing value.
- Initial rows, as data: CPU `F64` (the f64 island exists); Metal `BF32` (the MSL emitter has `half` and `float` only, `signature_tokens_prelude.rs:310-318`; no 64-bit float is ASSUMED to be absent from MSL); CUDA `BF32` until its emitter grows `double` (`cuda.rs:201` says Float32/Float16 only), then one row flips; WGSL `BF32` (`wgsl.rs:177`). A new backend is a table row, not a recompile.
- omega config validation then calls the codec's structural check, so `float64` passes `RuntimeConfig::validate` and is refused at plan time, by name, on a backend whose row says no. The second branch at `config.rs:336-343` (MMA input limited to BF32/BF16) is a real lowerer capability and stays until the MMA lowerer widens.

**Autograd.**

- Op literal: `Op::Constant { dtype, shape, code: u128 }` where `code` is the dtype's own stored code, built by `Op::constant<S: Scalar>(dtype, shape, value: S) -> Result<Op, NumericError>` through the codec with RNE and `Overflow::Format`. That deletes the f32 cap and the hand-written `Hash` caveat (`op.rs:~285`, "`f32` has no `Hash`"). `expr::constant` takes `S: Scalar`.
- Hyperparameters stay config values (`f64` in `AdamConfig`, since a config file is text and a hyperparameter is a host-boundary value), encoded once into the graph's dtype at graph construction. Spelled as an explicit host boundary, as `feedback-omega-front-generic-numerics` allows.
- `State<S = f32>` and `train_step<S: Scalar>` carry the host buffers; `weight_view<S>`. An autograd step in a FLOAT64 dtype then has f64 constants, f64 hyperparameters, f64 state, and the f64 island already computes it.
- A new op or kernel must work where autograd can use it: `Scalar` has no forward-only method.

## 4. Recommendation and principle citations

**Adopt (c).** One `Scalar` trait, sealed, two impls; the codec entry points, `FormatCodec`, `Numeric`, `PackedRows`, `Numerics` and `Philox` generic over it; one internal `wide` kernel that defines every layout's rounding; capability and config as data.

- **P1 / AGENTS "Before adding a type":** one primitive the consumer can be generic over, not `_f64` twins. Reuses `FloatLayout` (as the capability), `DeviceLimits`, `Numerics`, `NumericFormat`'s existing serde; mints no new config mechanism.
- **Pipe gate (written, not reasoned):** a codec is a synchronous, allocation-free function over caller slices, below the async `Pipe` atom (`In -> Future<Result<Out>>`). `Pipe::call` returns a future; wrapping a 16-lane decode loop in one is the cost the hot-path rules forbid. The pipe question is answered "no" honestly because the unit is sans-IO leaf compute, not because a helper would be nicer. The existing `proxima-tensor/src/convert.rs` `Convert<From, To>` pipe stays and gains `f64` pairs by delegating to this codec, so there is still one place per pair where the arithmetic is written.
- **Second gate, for `Scalar`: what can a caller do that they could not before?** Before: `encode_scalar(x as f32, &F64, ..)`, which stores an f32-rounded value in a binary64 slot. After: `encode_scalar(x, &F64, ..)`, which round-trips bit-exactly. The two lines differ in observable bits, so it is not a relocation. It is also not an erasure wrapper: it is a sealed closed set that names which host types are the typed slot of a layout.
- **Non-pipe exceptions, each justified:** `Scalar` and `wide` (sync leaf compute, sans-IO P11); `FormatCodec` function pointers (existing, `format_codec.rs:80-110`, no `dyn`, no `Box`); `Executes` (a value so it can be intersected and hashed into plan identity, `executes.rs` doc).
- **P3 / sealed trait, no blanket impl (AGENTS "No blanket impls", P20):** two impls; deleting either does not force a host wrapper.
- **P4:** `Numerics` as `Settings` + the fluent `with_*`; a parity fixture builds `Numerics` from env, TOML and builder and asserts equality (slice 5).
- **P8 / P12:** `MAX_LAYOUT_BITS`, `RNG_CHUNK` from `build.rs`; no magic `32`.
- **P11:** `Unpacked` is a discriminated enum, each variant owns only its legal data; all transitions are pure functions; `#[must_use]` on results; no `Box`, no `dyn`.
- **P14:** the incumbent is the oracle: f32 kernels untouched; `half::f16::from_f64`/`bf16::from_f64`, ml_dtypes/OCP vectors and the existing sd.cpp noise as independent references; block formats stay f32-arithmetic with a typed refusal rather than a divergent f64 fit.
- **P15:** `layout_supported`'s 63-bit integer limit, the `Op::Constant` f32 literal, `Numerics` having no caller, and the 32-bit omega cap are all in scope; none is deferred.
- **P17:** every slice below ends in a test that reads as documentation.
- **Hot-path (AGENTS):** no allocation added; a 100k-iteration allocation-counter test over f32 and f64 `FormatCodec::decode` pins zero.

## 5. Migration plan: slices, each one green commit

**Slice 1 - `wide` kernel and `Scalar` trait, no public signature changes (additive).**
Adds `scalar.rs`, `float_layout_codec/wide.rs`, `build.rs` + `proxima-numeric.toml` (`MAX_LAYOUT_BITS`, `RNG_CHUNK`); `layout_supported` consults the constant and accepts up to 128 bits and 15 exponent bits.

- *f32 and narrower bit-identical by construction:* no existing function body changes. Fixtures that must stay green unmodified: `proxima-numeric/tests/scalar_codec_parity.rs` (`exhaustive_decode_of_every_f16_code_matches_half_bit_for_bit`, `..._bf16_...`, `..._bf8_...onnxruntime...`, `random_f32_bit_patterns_encode_like_the_oracles_with_nan_and_snan_present`, `random_f32_bit_patterns_roundtrip_through_decode_like_the_oracles`, `an_exact_halfway_f32_rounds_to_even_for_each_narrow_format`, `per_element_encode_of_a_negative_into_a_sign_less_layout_stays_inside_the_layout_width`); `proxima-numeric/tests/format_codec_windows.rs`; `proxima-gguf/tests/float_layout_oracles.rs`; `float_layout_codec.rs` unit tests at `:1048` (`named_instantiation_matches_the_runtime_layout_path_bit_for_bit`), `:1086-1127`, `:1163-1194`, `:1278`, `:1289`, `:1306`; `format_codec.rs:405` (`resolved_codec_matches_encode_blocks_bit_for_bit_for_every_named_scalar`).
- *New:* `wide_kernel_agrees_with_the_f32_kernels_on_every_stride_of_f32_for_every_named_layout` (the reference the existing `FLOAT_LAYOUTS` list at `:1072` drives).
- *Bench gate:* `benches/bench_kv_cache_codec.rs` decode:f32 and encode cells before and after, CoV below 5%, plus the disassembly of the f32 `encode_body`/`decode_body` instantiations compared. Nothing is claimed until that artifact exists.
- *Fixture edit that is deliberate:* `format_codec.rs:674` pins `layout_constants(&codec) == ["BF32","F16"]`; `SOURCE_LAYOUT` becomes `<f32 as Scalar>::LAYOUT` and the list shrinks to `["F16"]`, with the one-source-layout rule moved to `scalar.rs`.

**Slice 2 - generic scalar entry points (`encode`, `decode`, `encode_with`, `encode_counting`, `decode_scalar`, `encode_scalar`, `narrow_to_*`, `decode_block`, `encode_blocks`, `decode_blocks`).** Code type becomes `u128`.

- *f32 bit-identity:* the same fixtures, now through the generic signature with `S = f32`; `default_numeric_fallback` deny on tensor, omega, model-interop, autograd, gguf lists every bare-literal site, each suffixed `_f32`. A test `an_unsuffixed_literal_does_not_change_the_narrowed_bits` pins one case (`0.1` into F16) against `half::f16::from_f32`.
- *Callers moved in this commit:* the 114 lines of section 1b that name these functions (tensor 11, omega's `decode_scalar`/`encode_blocks`/`decode_blocks` sites, model-interop `convert.rs`).

**Slice 3 - `FormatCodec`, `Numeric<'a, S = f32>`, `PackedRows`, `Numerics::decode_weights<S>`, `FormatCodec::compute`.** Adds `NumericError::LossyInput`.

- *f32 bit-identity:* `format_codec_windows.rs`, the `format_codec.rs` tests, and the interop residency/KV tests that read `PackedRows` (`residency_caches.rs`, `kv_ring.rs`, `device_kv.rs`, `prefix_state_file.rs`) unchanged.
- *New:* `encoding_f64_into_a_block_format_is_refused_with_the_compute_layout_named`.
- *New:* an allocation-counter test over 100k `FormatCodec::decode` calls for `f32` and `f64`.

**Slice 4 - binary64 round-trips bit-exactly (the card's acceptance).** New `proxima-numeric/tests/binary64_round_trip.rs`:

- `every_f64_bit_pattern_in_a_two_million_sample_round_trips_through_the_f64_layout_bit_for_bit` (random bits, subnormals, +-0, +-inf, quiet and signalling NaN payloads): `encode::<f64>(v, F64) == v.to_bits()` and `decode::<f64>` inverts, per element and through `FormatCodec::scalar(&F64)` slice route.
- `f64_into_f16_bf16_e4m3fn_e5m2_matches_the_independent_oracles` (`half::{f16,bf16}::from_f64`, the OCP/ml_dtypes vectors already used at `scalar_codec_parity.rs` and `float_layout_oracles.rs`, tie and overflow census as at `scalar_codec_parity.rs:402-413`).
- `f32_widened_into_f64_is_exact_and_f64_narrowed_to_f32_matches_the_cpu_cast`.
- `int64_and_int128_layouts_round_trip_every_value_they_hold` (the layouts at `float_layout.rs:172-175`).
- `a_binary128_description_round_trips_through_serde_and_the_wide_kernel`.

**Slice 5 - omega: lift the cap per backend.**

1. `DeviceLimits.compute: FloatLayout`, `Refusal::Scalar`, `Executes::admit` checks dtype; rows per section 3.
2. `omega/src/config.rs:327-334` calls the codec's structural check; `DenseMmaInputLayout::from_env_str` accepts any `FloatLayout::from_name`.
3. `Numerics` as `Settings`, nested in `RuntimeConfig`.
4. *Tests:* `config_tests.rs` gains `a_float64_accumulator_validates_and_a_metal_plan_refuses_it_by_name` and the builder/env/TOML parity fixture for `Numerics`; the existing explicit-layout refusal at `config_tests.rs:217-221` (E3M2 into the MMA input) stays green because that branch is the lowerer's. CPU: a FLOAT64 program with an f64 literal round-trips an `Input` -> `Constant` -> output without f32 loss (needs slice 7 for the literal; until then the test uses an f64 `Input`).
5. *f32 bit-identity:* `omega/tests/word_load_run_decode_parity.rs`, `omega/tests/unnamed_scalar_render.rs`, `msl/layout_render_tests.rs` (named, run against Metal where present).

**Slice 6 - Philox S1 becomes generic (PROXNUM-597).**
`Philox::fill_normal<S: Scalar>(&mut self, out: &mut [S])`, `normal<S>(&mut self, count) -> Vec<S>`, and `fill_normal_packed(&mut self, layout: &FloatLayout, out: &mut [u8]) -> Result<(), RngError>` that draws in the narrowest `Scalar` that admits `layout` and encodes in `sized::RNG_CHUNK` chunks on the stack, so no allocation.

- *Prerequisite, first commit of this slice:* the existing rng tests (`rng.rs:172-209`) are properties only and `flux2_klein.rs:55-70` compares to sd.cpp with a tolerance, so no bit is pinned. Record a golden of `Philox::new(42).normal(256)` f32 bits from main before touching `rng.rs`: `f32_noise_stays_bit_identical_to_the_recorded_main_stream`. Without it "f32 bit-identical" is unprovable.
- *f32 stream:* `from_philox` for `f32` is the existing `box_muller(words[0] as f32, words[1] as f32)` (`rng.rs:127,162-166`), same constants, same libm calls.
- *f64 stream:* a new definition, no incumbent (sd.cpp is f32), so P14 does not bind and it is recorded as new semantics. Element `i` uses the whole block: `u1` from words 0 and 2 as a 53-bit uniform, `angle` from words 1 and 3. It remains a pure function of `(seed, offset, i)`; `element_value_does_not_depend_on_draw_length` is parameterised over both scalars.
- *Tests:* `f64_noise_is_standard_normal_and_independent_of_draw_length`; `packed_noise_into_bf16_equals_f32_noise_encoded_to_bf16`; `fill_normal_packed_refuses_a_layout_no_host_scalar_holds_by_name`.
- *Info destroyed today:* `RngKind` carries no output layout. It stays that way on purpose: the destination dtype is the node's, not a second config key.

**Slice 7 - IR literal and autograd.**
`Op::Constant { dtype, shape, code: u128 }`, `Op::constant<S>`, `expr::constant<S>`, `State<S>`, `train_step<S>`, `AdamConfig` hyperparameters `f64`, then removal of the "literal is an f32" comment and cast in `float64_island.rs`.

- *f32 bit-identity:* every existing program that builds a FLOAT32 `Constant` hashes and evaluates identically. The structural fingerprint that "derives from the rendered form" (`dtype.rs:26-28`, "load-bearing for every recorded graph hash") is checked: a graph-hash golden over the existing recorded graphs must be unchanged, or the slice is wrong. `op.rs` `Hash` is now derived, and a test asserts the bit-pattern hashing of `-0.0` versus `0.0` is preserved.
- *New:* `an_f64_adam_step_with_epsilon_1e_minus_12_keeps_the_epsilon_exactly` against a hand-computed f64 reference.

**Order:** 1, 2, 3, 4 are proxima-numeric alone and mergeable on their own. 5 depends on 2 and 3. 6 depends on 1 and 2. 7 depends on 2. Each slice is one logical change, one commit, with the fixtures named above run as its gate (`cargo val`, not nextest, per owner); the doctest class is run separately because nextest skips doctests (AGENTS "Testing").

## 6. Not proven, and what would change the design

- **f32 zero-cost** is DERIVED from "the f32 hook is the existing body", not MEASURED. If the disassembly diff in slice 1 shows a different instruction stream for the f32 `decode_body` instantiation, the hook must be restructured so the generic wrapper is a thin shim over a non-generic function (the shape `format_codec.rs:306-337` already uses with `#[inline(never)]`).
- **u128 `wide` speed** on 32-bit targets is ASSUMED acceptable because it sits on the cold/f64 path only. A bench arm with an f64 incumbent (`half::f16::from_f64` for f64 to f16; `f64::to_bits` for identity) belongs in `bench_kv_cache_codec.rs` before anything calls `wide` hot.
- **Metal and CUDA having no 64-bit float** in their current emitters is read from `cuda.rs:201` and `signature_tokens_prelude.rs:310-318`; hardware or driver support for a wider type is not checked here and is exactly what flipping a `DeviceLimits.compute` row would assert.
- **Out of the ask but affected:** proxima-gguf (35 `FormatCodec`, 11 `decode_blocks`, 7 `encode_blocks` lines) and the `Convert<From, To>` pipe pairs in `proxima-tensor/src/convert.rs`.
