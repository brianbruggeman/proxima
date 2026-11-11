//! Host-side KV cache store / load / narrow: the per-type conversions `omega/src/metal/execute_and_hazards.rs`
//! used at `HEAD` (OLD) against the one generic path now in the working tree (NEW), which composes
//! `proxima_numeric::encode::encode_blocks` and `proxima_numeric::decode::decode_blocks`.
//!
//! Both paths are host memory ops over `storageModeShared` bytes, so the arms run on `Vec<u8>` slices
//! with the same unsafe casts the omega functions use. The OLD arms are copied from
//! `git show HEAD:omega/src/metal/execute_and_hazards.rs` (`write_placed_buffer_f32` 1348,
//! `write_placed_buffer_f32_as_{f16,bf16,bf8}` 1366/1406/1450, `read_placed_buffer_{f16,bf16,bf8}_as_f32`
//! 1387/1428/1466, `read_placed_buffer_f32` 1318, `narrow_placed_buffer_f32_to_*` 1486/1523/1567); the BF8
//! conversion is `proxima-tensor/src/bfloat8.rs` (`encode_f32_bits` / `decode_f32_bits`), copied verbatim
//! because `proxima-tensor` depends on this crate. `store` and `narrow` are the same encode loop, so one
//! `encode` arm covers both; `decode` is the load and allocates its returned `Vec` in both paths, as the
//! real functions do.
//!
//! Before any timing the bench asserts OLD bytes == NEW bytes (encode) and OLD bits == NEW bits (decode)
//! on the bench inputs plus 2^20 random f32 bit patterns, and prints allocations per call per arm.
//!
//! Re-prove: `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-numeric --bench bench_kv_cache_codec -- --save-baseline kv-codec-r1`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::ops::Range;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use core::mem::MaybeUninit;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use proxima_gguf::quant::QuantError;
use proxima_gguf::quant::{pq2_0, q1_0, q2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, q8_k};
use proxima_numeric::{BlockLayout, Field, FormatCodec, NumericFormat};
use proxima_numeric::decode::decode_blocks;
use proxima_numeric::encode::encode_blocks;
use proxima_numeric::float_layout_codec::encode as encode_field;

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const SIZES: [usize; 4] = [256, 512, 262_144, 2_097_152];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scalar {
    Float32,
    Float16,
    BFloat16,
    BFloat8,
}

impl Scalar {
    const ALL: [Self; 4] = [Self::Float32, Self::Float16, Self::BFloat16, Self::BFloat8];

    fn name(self) -> &'static str {
        match self {
            Self::Float32 => "f32",
            Self::Float16 => "f16",
            Self::BFloat16 => "bf16",
            Self::BFloat8 => "bf8",
        }
    }

    fn format(self) -> NewPath {
        NewPath::new(match self {
            Self::Float32 => NumericFormat::Float32,
            Self::Float16 => NumericFormat::Float16,
            Self::BFloat16 => NumericFormat::BFloat16,
            Self::BFloat8 => NumericFormat::BFloat8,
        })
    }

    fn width(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::BFloat8 => 1,
        }
    }
}

// ---- OLD path, copied from HEAD ----------------------------------------------------------------

/// `write_placed_buffer_f32` / `narrow_*` / `write_*_as_*` bodies at HEAD, over host bytes.
#[inline(never)]
fn old_encode(scalar: Scalar, values: &[f32], destination_bytes: &mut [u8]) {
    let pointer = destination_bytes.as_mut_ptr();
    match scalar {
        Scalar::Float32 => unsafe {
            core::ptr::copy_nonoverlapping(values.as_ptr(), pointer.cast::<f32>(), values.len());
        },
        Scalar::Float16 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer.cast::<f16>(), values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = f16::from_f32(*value);
            }
        }
        Scalar::BFloat16 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer.cast::<bf16>(), values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = bf16::from_f32(*value);
            }
        }
        Scalar::BFloat8 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer, values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = BFloat8::from_f32(*value).to_bits();
            }
        }
    }
}

/// `read_placed_buffer_f32` / `read_placed_buffer_{f16,bf16,bf8}_as_f32` bodies at HEAD.
#[inline(never)]
fn old_decode(scalar: Scalar, source_bytes: &[u8], element_count: usize) -> Vec<f32> {
    let pointer = source_bytes.as_ptr();
    match scalar {
        Scalar::Float32 => unsafe { core::slice::from_raw_parts(pointer.cast::<f32>(), element_count) }.to_vec(),
        Scalar::Float16 => unsafe { core::slice::from_raw_parts(pointer.cast::<f16>(), element_count) }
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        Scalar::BFloat16 => unsafe { core::slice::from_raw_parts(pointer.cast::<bf16>(), element_count) }
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        Scalar::BFloat8 => unsafe { core::slice::from_raw_parts(pointer, element_count) }
            .iter()
            .map(|value| BFloat8::from_bits(*value).to_f32())
            .collect(),
    }
}

// ---- NEW path, as omega's write/read/narrow_placed_buffer call it ------------------------------

/// A scalar format resolved once, as `DeviceKvLayer` holds its cache format and codec.
struct NewPath {
    format: NumericFormat,
    codec: FormatCodec,
}

impl NewPath {
    fn new(format: NumericFormat) -> Self {
        let codec = FormatCodec::for_format(&format).expect("a scalar format resolves");
        Self { format, codec }
    }
}

impl core::ops::Deref for NewPath {
    type Target = NumericFormat;

    fn deref(&self) -> &NumericFormat {
        &self.format
    }
}

/// `#[inline(never)]`, like `old_encode`: both arms cross one call boundary, as production does across
/// crates. The window is the whole buffer, the way omega's `write_placed_buffer` passes the rest of
/// the allocation; the codec sizes the packed range from `values` and checks it once.
#[inline(never)]
fn new_encode(values: &[f32], buffer: &mut [u8], path: &NewPath) {
    path.codec.encode(values, buffer).unwrap();
}

/// See [`new_encode`]; the window is the whole buffer, as `read_placed_buffer` passes it.
#[inline(never)]
fn new_decode(buffer: &[u8], element_count: usize, path: &NewPath) -> Vec<f32> {
    let codec = &path.codec;
    let mut decoded: Vec<f32> = Vec::with_capacity(element_count);
    codec.decode(buffer,&mut decoded.spare_capacity_mut()[..element_count]).unwrap();
    // SAFETY: `decode` initialised exactly `element_count` slots, the whole `spare_capacity_mut` prefix.
    unsafe { decoded.set_len(element_count) };
    decoded
}

// ---- BFloat8, verbatim from proxima-tensor/src/bfloat8.rs --------------------------------------

#[derive(Clone, Copy)]
struct BFloat8(u8);

impl BFloat8 {
    const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    const fn to_bits(self) -> u8 {
        self.0
    }

    fn from_f32(value: f32) -> Self {
        Self(encode_f32_bits(value.to_bits()))
    }

    fn to_f32(self) -> f32 {
        f32::from_bits(decode_f32_bits(self.0))
    }
}

fn round_shift_to_even(significand: u32, shift: u32) -> u32 {
    if shift == 0 {
        return significand;
    }
    if shift > 24 {
        return 0;
    }

    let quotient = significand >> shift;
    let remainder = significand & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

fn encode_f32_bits(source: u32) -> u8 {
    let sign = ((source >> 24) & 0x80) as u8;
    let exponent = (source >> 23) & 0xff;
    let mantissa = source & 0x7f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 { sign | 0x7c } else { sign | 0x7e };
    }
    if exponent == 0 && mantissa == 0 {
        return sign;
    }

    let (significand, binary_power, leading_exponent) = if exponent == 0 {
        (mantissa, -149, (u32::BITS - mantissa.leading_zeros() - 1) as i32 - 149)
    } else {
        (0x80_0000 | mantissa, exponent as i32 - 150, exponent as i32 - 127)
    };

    if leading_exponent < -14 {
        let shift = -(binary_power + 16) as u32;
        let units = round_shift_to_even(significand, shift);
        return sign | units.min(4) as u8;
    }

    let mut rounded_significand = round_shift_to_even(significand, 21);
    let mut rounded_exponent = leading_exponent;
    if rounded_significand == 8 {
        rounded_exponent += 1;
        rounded_significand = 4;
    }
    if rounded_exponent > 15 {
        return sign | 0x7c;
    }

    sign | (((rounded_exponent + 15) as u8) << 2) | (rounded_significand - 4) as u8
}

fn decode_f32_bits(source: u8) -> u32 {
    let sign = u32::from(source & 0x80) << 24;
    let exponent = (source >> 2) & 0x1f;
    let mantissa = u32::from(source & 0x03);

    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let leading_bit = u32::from(mantissa >= 2);
            let f32_exponent = leading_bit + 111;
            let f32_mantissa = (mantissa - (1 << leading_bit)) << (23 - leading_bit);
            sign | (f32_exponent << 23) | f32_mantissa
        }
        31 => sign | 0x7f80_0000 | (mantissa << 21),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 21),
    }
}

// ---- inputs ------------------------------------------------------------------------------------

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Gaussian-ish K values (sum of twelve uniforms, sigma 1) with two outlier channels per 64-wide head
/// scaled x24, the shape of a real attention key row (a few massive-activation channels).
fn kv_like_values(count: usize) -> Vec<f32> {
    let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
    (0..count)
        .map(|index| {
            let gaussian: f32 = (0..12).map(|_| rng.unit()).sum::<f32>() - 6.0;
            if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian }
        })
        .collect()
}

fn random_bit_patterns(count: usize) -> Vec<f32> {
    let mut rng = Xorshift(0xdead_beef_cafe_f00d);
    (0..count).map(|_| f32::from_bits((rng.next() >> 16) as u32)).collect()
}

/// Backing store aligned to 8 so the f16/f32 casts the OLD path makes are valid.
fn aligned_bytes(length: usize) -> Vec<u8> {
    let words = vec![0u64; length.div_ceil(8)];
    let mut bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    bytes.truncate(length);
    assert_eq!(bytes.as_ptr() as usize % 4, 0, "byte buffer must be 4-aligned for the f32 casts");
    bytes
}

// ---- pre-timing checks -------------------------------------------------------------------------

fn count_allocations(call: impl FnOnce()) -> (u64, u64) {
    let allocations_before = ALLOCATIONS.load(Ordering::SeqCst);
    let bytes_before = ALLOCATED_BYTES.load(Ordering::SeqCst);
    call();
    (
        ALLOCATIONS.load(Ordering::SeqCst) - allocations_before,
        ALLOCATED_BYTES.load(Ordering::SeqCst) - bytes_before,
    )
}

const F32_QUIET_BIT: u32 = 0x0040_0000;

fn is_signaling_nan(value: f32) -> bool {
    value.is_nan() && value.to_bits() & F32_QUIET_BIT == 0
}

/// The mismatch counters below only count NaNs among mismatches, so zero mismatches prints 0/0 whether or not
/// NaNs were fed. This census counts NaN and sNaN inputs themselves; a set that must cover NaN reports a failure when it does not.
fn input_census(label: &str, inputs: &[f32], must_cover_nan: bool) -> Vec<String> {
    let nan_inputs = inputs.iter().filter(|value| value.is_nan()).count();
    let snan_inputs = inputs.iter().filter(|value| is_signaling_nan(**value)).count();
    println!("census {label}: inputs={} nan_inputs={nan_inputs} snan_inputs={snan_inputs}", inputs.len());
    let mut failures = Vec::new();
    if must_cover_nan && nan_inputs == 0 {
        failures.push(format!("{label} fed no NaN inputs; the parity oracle is blind to NaN handling"));
    }
    if must_cover_nan && snan_inputs == 0 {
        failures.push(format!("{label} fed no sNaN inputs; the parity oracle is blind to sNaN quieting"));
    }
    failures
}

/// OLD's bf8 decode keeps NaN payloads; the external incumbent (onnxruntime `float8.h:454-462`) widens
/// every NaN code to the canonical quiet NaN with its sign, and NEW follows it, so the parity oracle is
/// OLD with its NaNs canonicalised. The timing arms run OLD untouched.
fn parity_oracle_decode(scalar: Scalar, source_bytes: &[u8], element_count: usize) -> Vec<f32> {
    let mut decoded = old_decode(scalar, source_bytes, element_count);
    if scalar == Scalar::BFloat8 {
        for value in decoded.iter_mut().filter(|value| value.is_nan()) {
            *value = f32::from_bits(value.to_bits() & 0x8000_0000 | 0x7fc0_0000);
        }
    }
    decoded
}

fn bit_exactness(label: &str, inputs: &[f32], must_cover_nan: bool) -> Vec<String> {
    let mut failures = input_census(label, inputs, must_cover_nan);
    for scalar in Scalar::ALL {
        let format = scalar.format();
        let width = scalar.width();
        let mut old_bytes = aligned_bytes(inputs.len() * width);
        let mut new_bytes = aligned_bytes(inputs.len() * width);
        old_encode(scalar, inputs, &mut old_bytes);
        new_encode(inputs, &mut new_bytes, &format);
        let mismatched_elements = old_bytes
            .chunks_exact(width)
            .zip(new_bytes.chunks_exact(width))
            .filter(|(old, new)| old != new)
            .count();
        let old_decoded = parity_oracle_decode(scalar, &old_bytes, inputs.len());
        let new_decoded = new_decode(&old_bytes, inputs.len(), &format);
        let mismatched_decodes = old_decoded
            .iter()
            .zip(&new_decoded)
            .filter(|(old, new)| old.to_bits() != new.to_bits())
            .count();
        let nan_inputs_in_mismatch = old_bytes
            .chunks_exact(width)
            .zip(new_bytes.chunks_exact(width))
            .zip(inputs)
            .filter(|((old, new), input)| old != new && input.is_nan())
            .count();
        let first_mismatches: Vec<String> = old_bytes
            .chunks_exact(width)
            .zip(new_bytes.chunks_exact(width))
            .zip(inputs)
            .filter(|((old, new), _)| old != new)
            .take(3)
            .map(|((old, new), input)| format!("in={:#010x} old={old:02x?} new={new:02x?}", input.to_bits()))
            .collect();
        let nan_decodes_in_mismatch = old_decoded
            .iter()
            .zip(&new_decoded)
            .filter(|(old, new)| old.to_bits() != new.to_bits() && old.is_nan())
            .count();
        let first_decode_mismatches: Vec<String> = old_decoded
            .iter()
            .zip(&new_decoded)
            .filter(|(old, new)| old.to_bits() != new.to_bits())
            .take(3)
            .map(|(old, new)| format!("old={:#010x} new={:#010x}", old.to_bits(), new.to_bits()))
            .collect();
        println!(
            "  mismatch detail {label} {}: encode nan_inputs={nan_inputs_in_mismatch}/{mismatched_elements} first={first_mismatches:?} | decode old_is_nan={nan_decodes_in_mismatch}/{mismatched_decodes} first={first_decode_mismatches:?}",
            scalar.name(),
        );
        println!(
            "bitexact {label} {}: encode compared={} mismatched={} | decode compared={} mismatched={}",
            scalar.name(),
            inputs.len(),
            mismatched_elements,
            inputs.len(),
            mismatched_decodes,
        );
        if mismatched_elements + mismatched_decodes != 0 {
            failures.push(format!(
                "{label} {} disagrees with OLD: {mismatched_elements} encodes, {mismatched_decodes} decodes",
                scalar.name()
            ));
        }
    }
    failures
}

/// Every code of each narrow scalar decoded by OLD and NEW, compared bit for bit with NaN payloads
/// included: 65536 f16, 65536 bf16, 256 bf8. Reports a failure per format with any mismatch.
fn exhaustive_decode_parity() -> Vec<String> {
    let mut failures = Vec::new();
    for (scalar, code_count) in [(Scalar::Float16, 1usize << 16), (Scalar::BFloat16, 1 << 16), (Scalar::BFloat8, 1 << 8)] {
        let width = scalar.width();
        let mut bytes = aligned_bytes(code_count * width);
        for (code, slot) in bytes.chunks_exact_mut(width).enumerate() {
            slot.copy_from_slice(&(code as u32).to_le_bytes()[..width]);
        }
        let old = parity_oracle_decode(scalar, &bytes, code_count);
        let new = new_decode(&bytes, code_count, &scalar.format());
        let mismatched: Vec<usize> = (0..code_count).filter(|&code| old[code].to_bits() != new[code].to_bits()).collect();
        let nan_codes = old.iter().filter(|value| value.is_nan()).count();
        println!(
            "exhaustive decode {}: codes={code_count} nan_codes={nan_codes} mismatched={} first={:?}",
            scalar.name(),
            mismatched.len(),
            mismatched.iter().take(3).map(|&code| format!("code={code:#06x} old={:#010x} new={:#010x}", old[code].to_bits(), new[code].to_bits())).collect::<Vec<_>>(),
        );
        if !mismatched.is_empty() {
            failures.push(format!("exhaustive decode of {} disagrees with OLD on {} codes", scalar.name(), mismatched.len()));
        }
    }
    failures
}

fn report_allocations(values: &[f32]) {
    let element_count = 512;
    let inputs = &values[..element_count];
    for scalar in Scalar::ALL {
        let format = scalar.format();
        let mut buffer = aligned_bytes(element_count * 4);
        let (old_encode_allocs, _) = count_allocations(|| old_encode(scalar, inputs, &mut buffer));
        let (new_encode_allocs, _) = count_allocations(|| new_encode(inputs, &mut buffer, &format));
        let (old_decode_allocs, old_decode_bytes) = count_allocations(|| drop(old_decode(scalar, &buffer, element_count)));
        let (new_decode_allocs, new_decode_bytes) = count_allocations(|| drop(new_decode(&buffer, element_count, &format)));
        println!(
            "allocs {}: encode old={old_encode_allocs} new={new_encode_allocs} | decode old={old_decode_allocs} ({old_decode_bytes} B) new={new_decode_allocs} ({new_decode_bytes} B)",
            scalar.name(),
        );
    }
}

// ---- timing ------------------------------------------------------------------------------------

fn bench_encode(criterion: &mut Criterion, values: &[f32]) {
    let mut group = criterion.benchmark_group("encode");
    for &count in &SIZES {
        let inputs = &values[..count];
        group.throughput(Throughput::Elements(count as u64));
        for scalar in Scalar::ALL {
            let format = scalar.format();
            let mut buffer = aligned_bytes(count * scalar.width());
            group.bench_function(BenchmarkId::new(format!("{}/old", scalar.name()), count), |bencher| {
                bencher.iter(|| old_encode(scalar, black_box(inputs), black_box(&mut buffer)));
            });
            group.bench_function(BenchmarkId::new(format!("{}/new", scalar.name()), count), |bencher| {
                bencher.iter(|| new_encode(black_box(inputs), black_box(&mut buffer), &format));
            });
        }
    }
    group.finish();
}

fn bench_decode(criterion: &mut Criterion, values: &[f32]) {
    let mut group = criterion.benchmark_group("decode");
    for &count in &SIZES {
        let inputs = &values[..count];
        group.throughput(Throughput::Elements(count as u64));
        for scalar in Scalar::ALL {
            let format = scalar.format();
            let mut buffer = aligned_bytes(count * scalar.width());
            old_encode(scalar, inputs, &mut buffer);
            group.bench_function(BenchmarkId::new(format!("{}/old", scalar.name()), count), |bencher| {
                bencher.iter(|| old_decode(scalar, black_box(&buffer), count));
            });
            group.bench_function(BenchmarkId::new(format!("{}/new", scalar.name()), count), |bencher| {
                bencher.iter(|| new_decode(black_box(&buffer), count, &format));
            });
        }
    }
    group.finish();
}

// ---- block formats: FormatCodec (NEW) against proxima-gguf's per-format routines (incumbent) ------

type IncumbentEncode = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;
type IncumbentDecode = fn(&[u8], &mut [f32]) -> Result<(), QuantError>;

/// Where a block format's incumbent decode comes from: a `proxima-gguf` routine, or, where that crate has
/// none (`mxfp4`, `tq2_0` and `fp6_e3m2` only decode through the runtime walker or a per-block API), the
/// runtime interpreter `decode_blocks` that NEW replaces. The label is printed with every cell.
#[derive(Clone, Copy)]
enum Decoder {
    Gguf(IncumbentDecode),
    Legacy,
}

/// Where a block format's timing incumbent for encode comes from: a `proxima-gguf` routine, or the runtime
/// walker `encode_blocks` that NEW replaces (`proxima-gguf` has no `tq2_0` quantize outside its test-only
/// llama.cpp transcription). The label is printed with every cell.
#[derive(Clone, Copy)]
enum Encoder {
    Gguf(IncumbentEncode),
    Walker,
}

#[derive(Clone, Copy)]
struct BlockCase {
    name: &'static str,
    layout: BlockLayout,
    encoder: Option<Encoder>,
    decoder: Decoder,
}

fn block_cases() -> Vec<BlockCase> {
    let case = |name, layout, encoder, decoder| BlockCase { name, layout, encoder, decoder };
    let gguf = |routine: IncumbentEncode| Some(Encoder::Gguf(routine));
    vec![
        case("q4_0", BlockLayout::Q4_0, gguf(q4_0::quantize), Decoder::Gguf(q4_0::dequantize)),
        case("q4_1", BlockLayout::Q4_1, None, Decoder::Gguf(q4_1::dequantize)),
        case("q5_0", BlockLayout::Q5_0, gguf(q5_0::quantize), Decoder::Gguf(q5_0::dequantize)),
        case("q5_1", BlockLayout::Q5_1, None, Decoder::Gguf(q5_1::dequantize)),
        case("q8_0", BlockLayout::Q8_0, gguf(q8_0::quantize), Decoder::Gguf(q8_0::dequantize)),
        case("q8_1", BlockLayout::Q8_1, None, Decoder::Gguf(q8_1::dequantize)),
        case("q8_k", BlockLayout::Q8_K, None, Decoder::Gguf(q8_k::dequantize)),
        case("q1_0", BlockLayout::Q1_0, gguf(q1_0::quantize), Decoder::Gguf(q1_0::dequantize)),
        case("q2_0", BlockLayout::Q2_0, gguf(q2_0::quantize), Decoder::Gguf(q2_0::dequantize)),
        case("pq2_0", BlockLayout::PQ2_0, gguf(pq2_0::quantize), Decoder::Gguf(pq2_0::dequantize)),
        case("tq2_0", BlockLayout::TQ2_0, Some(Encoder::Walker), Decoder::Legacy),
        case("mxfp4", BlockLayout::MXFP4, None, Decoder::Legacy),
        case("fp6_e3m2", BlockLayout::FP6_E3M2, None, Decoder::Legacy),
    ]
}

impl BlockCase {
    fn codec(&self) -> FormatCodec {
        FormatCodec::block(&self.layout).expect("a simple-family layout resolves")
    }

    fn elements(&self) -> usize {
        usize::from(self.layout.elements)
    }

    fn block_bytes(&self) -> usize {
        usize::from(self.layout.block_bytes)
    }

    fn packed_len(&self, count: usize) -> usize {
        count / self.elements() * self.block_bytes()
    }

    fn encoder_label(&self) -> &'static str {
        match self.encoder {
            Some(Encoder::Gguf(_)) => "gguf",
            Some(Encoder::Walker) => "walker",
            None => "none",
        }
    }

    fn decoder_label(&self) -> &'static str {
        match self.decoder {
            Decoder::Gguf(_) => "gguf",
            Decoder::Legacy => "legacy-walker",
        }
    }
}

fn run_incumbent_encode(case: &BlockCase, values: &[f32], out: &mut [u8]) {
    match case.encoder.expect("a case with an incumbent encode") {
        Encoder::Gguf(routine) => routine(values, out).unwrap(),
        Encoder::Walker => {
            encode_blocks(&NumericFormat::Block(case.layout), values, out).unwrap();
        }
    }
}

#[inline(never)]
fn incumbent_block_encode(case: &BlockCase, values: &[f32], out: &mut [u8]) {
    run_incumbent_encode(case, values, out);
}

/// [`incumbent_block_encode`] compiled a second time, the A/A control.
#[inline(never)]
fn incumbent_block_encode_copy(case: &BlockCase, values: &[f32], out: &mut [u8]) {
    black_box(());
    run_incumbent_encode(case, values, out);
}

#[inline(never)]
fn incumbent_block_decode(case: &BlockCase, data: &[u8], out: &mut [f32]) {
    match case.decoder {
        Decoder::Gguf(routine) => routine(data, out).unwrap(),
        Decoder::Legacy => {
            decode_blocks(&NumericFormat::Block(case.layout), data, out).unwrap();
        }
    }
}

#[inline(never)]
fn incumbent_block_decode_copy(case: &BlockCase, data: &[u8], out: &mut [f32]) {
    black_box(());
    match case.decoder {
        Decoder::Gguf(routine) => routine(data, out).unwrap(),
        Decoder::Legacy => {
            decode_blocks(&NumericFormat::Block(case.layout), data, out).unwrap();
        }
    }
}

/// NEW across the same call boundary as the incumbent; the codec was resolved once, before timing.
#[inline(never)]
fn new_block_encode(codec: &FormatCodec, values: &[f32], out: &mut [u8]) {
    codec.encode(values, out).unwrap();
}

#[inline(never)]
fn new_block_decode(codec: &FormatCodec, data: &[u8], out: &mut [f32]) {
    // SAFETY: `MaybeUninit<f32>` has the layout of `f32` and every slot of `out` is initialised already.
    let slots = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<MaybeUninit<f32>>(), out.len()) };
    codec.decode(data, slots).unwrap();
}

fn degenerate_block_encode(codec: &FormatCodec, values: &[f32], out: &mut [u8]) {
    new_block_encode(codec, values, out);
    for byte in out.iter_mut() {
        *byte = black_box(*byte).wrapping_add(1);
    }
}

fn degenerate_block_decode(codec: &FormatCodec, data: &[u8], out: &mut [f32]) {
    new_block_decode(codec, data, out);
    for value in out.iter_mut() {
        *value = black_box(*value);
    }
}

/// One block per row: all zero, a NaN, +inf, -inf, a signalling NaN, -0.0, NaN beside +inf, an exact
/// +/- tie, one repeated value and `f32::MAX`.
fn special_block_values(elements: usize, base: &[f32]) -> Vec<f32> {
    let mut rows = vec![0.0f32; elements];
    for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fa0_0001), -0.0] {
        let mut block = base[..elements].to_vec();
        block[elements / 3] = special;
        rows.extend(block);
    }
    let mut both = base[..elements].to_vec();
    both[1] = f32::NAN;
    both[2] = f32::INFINITY;
    rows.extend(both);
    rows.extend((0..elements).map(|index| if index % 2 == 0 { 4.0 } else { -4.0 }));
    rows.extend(vec![0.75f32; elements]);
    rows.extend(vec![f32::MAX; elements]);
    rows
}

fn random_bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xorshift(seed);
    (0..count).map(|_| (rng.next() >> 24) as u8).collect()
}

fn is_f16_nan(low: u8, high: u8) -> bool {
    u16::from_le_bytes([low, high]) & 0x7fff > 0x7c00
}

/// Random block bytes for `case`, untouched, with the counts of blocks whose f16 scale is NaN and whose
/// f16 min is NaN (a NaN scale and a NaN min in one block add two NaNs, so the payload that survives
/// is the operand order's); both counts are returned and printed.
fn random_blocks(case: &BlockCase, count: usize, seed: u64) -> (Vec<u8>, NanBlocks) {
    let bytes = random_bytes(case.packed_len(count), seed);
    let nan = nan_census(case, &bytes);
    (bytes, nan)
}

#[derive(Default)]
struct NanBlocks {
    scale: usize,
    min: usize,
    both: usize,
}

fn field_bytes(field: Field) -> Range<usize> {
    let start = usize::from(field.byte_offset);
    start..start + usize::from(field.layout.width()).div_ceil(8)
}

/// Whether a scale or min field holds a NaN: an E8M0 byte of 0xff, an f16 or an f32 NaN.
fn field_is_nan(block: &[u8], field: Field) -> bool {
    match block[field_bytes(field)] {
        [exponent] => exponent == 0xff,
        [low, high] => is_f16_nan(low, high),
        [first, second, third, fourth] => f32::from_le_bytes([first, second, third, fourth]).is_nan(),
        _ => false,
    }
}

fn nan_census(case: &BlockCase, bytes: &[u8]) -> NanBlocks {
    let mut nan = NanBlocks::default();
    for block in bytes.chunks(case.block_bytes()) {
        let scale_nan = field_is_nan(block, case.layout.scale);
        let min_nan = case.layout.min.is_some_and(|field| field_is_nan(block, field));
        nan.scale += usize::from(scale_nan);
        nan.min += usize::from(min_nan);
        nan.both += usize::from(scale_nan && min_nan);
    }
    nan
}

/// For a format the walker refuses (a stored min is not an absolute-maximum scale, or bytes beyond the
/// scale and codes): each block's scale field is its peak over 8 and its min field its smallest value, both
/// from the kv values and encoded in the field's own layout.
fn write_derived_fields(case: &BlockCase, inputs: &[f32], blocks: &mut [u8]) {
    let fields = [(Some(case.layout.scale), 0), (case.layout.min, 1)];
    for (block, values) in blocks.chunks_mut(case.block_bytes()).zip(inputs.chunks(case.elements())) {
        let peak = values.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        let lowest = values.iter().copied().fold(f32::MAX, f32::min);
        for (field, role) in fields {
            let Some(field) = field else { continue };
            let value = if role == 0 { peak / 8.0 } else { lowest };
            let range = field_bytes(field);
            let width = range.len();
            block[range].copy_from_slice(&encode_field(value, field.layout).to_le_bytes()[..width]);
        }
    }
}

/// The blocks a decode is timed on, and their NaN census. The template returns early on a block whose
/// scale or min is NaN, so raw random bytes would time a mix of early returns and full decodes. A format
/// with an incumbent encode is timed on the blocks it encodes from the kv values; a decode-only format
/// takes random code bytes with the scale and min fields of the blocks the walker encodes from the same
/// values, so every scale and min is finite and the codes are still arbitrary. Parity keeps raw bytes.
fn timing_blocks(case: &BlockCase, values: &[f32], count: usize) -> (Vec<u8>, NanBlocks) {
    let inputs = &values[..count];
    let mut encoded = vec![0u8; case.packed_len(count)];
    let bytes = if case.encoder.is_some() {
        run_incumbent_encode(case, inputs, &mut encoded);
        encoded
    } else {
        if encode_blocks(&NumericFormat::Block(case.layout), inputs, &mut encoded).is_err() {
            write_derived_fields(case, inputs, &mut encoded);
        }
        let mut bytes = random_bytes(encoded.len(), 0xb10c);
        let fields = [Some(case.layout.scale), case.layout.min];
        for (block, source) in bytes.chunks_mut(case.block_bytes()).zip(encoded.chunks(case.block_bytes())) {
            for field in fields.into_iter().flatten() {
                block[field_bytes(field)].copy_from_slice(&source[field_bytes(field)]);
            }
        }
        bytes
    };
    let census = nan_census(case, &bytes);
    (bytes, census)
}

fn mismatched_blocks(left: &[u8], right: &[u8], block_bytes: usize) -> (usize, usize) {
    let blocks = left.chunks(block_bytes).zip(right.chunks(block_bytes)).filter(|(a, b)| a != b).count();
    let bytes = left.iter().zip(right).filter(|(a, b)| a != b).count();
    (blocks, bytes)
}

/// NEW against the incumbents, bit for bit, on 2,097,152 KV-like values, the special blocks and random block
/// bytes; one line per format with the mismatch counts. One failure per format with any mismatch.
fn block_bit_exactness(values: &[f32]) -> Vec<String> {
    let mut failures = Vec::new();
    for case in block_cases() {
        let codec = case.codec();
        let count = values.len() / case.elements() * case.elements();
        let kv = &values[..count];
        let special = special_block_values(case.elements(), values);
        failures.extend(input_census(&format!("{} special", case.name), &special, true));
        let (mut encode_blocks_bad, mut encode_bytes_bad, mut encode_compared) = (0, 0, 0);
        if case.encoder.is_some() {
            // the walker refuses non-finite input (NonFinite), so only a gguf incumbent sees the special blocks
            let walker = matches!(case.encoder, Some(Encoder::Walker));
            for input in [kv, if walker { &[] } else { &special[..] }] {
                let mut expected = vec![0u8; case.packed_len(input.len())];
                let mut got = vec![0u8; expected.len()];
                run_incumbent_encode(&case, input, &mut expected);
                codec.encode(input, &mut got).unwrap();
                let (blocks, bytes) = mismatched_blocks(&expected, &got, case.block_bytes());
                encode_blocks_bad += blocks;
                encode_bytes_bad += bytes;
                encode_compared += input.len() / case.elements();
            }
        }
        if let Some(Encoder::Gguf(incumbent)) = case.encoder {
            let mut expected = vec![0u8; case.packed_len(count)];
            let mut legacy = vec![0u8; expected.len()];
            incumbent(kv, &mut expected).unwrap();
            let outcome = encode_blocks(&NumericFormat::Block(case.layout), kv, &mut legacy);
            match outcome {
                Ok(_) => {
                    let (blocks, bytes) = mismatched_blocks(&expected, &legacy, case.block_bytes());
                    println!(
                        "encode_blocks vs gguf {}: compared_blocks={} differing_blocks={blocks} differing_bytes={bytes}",
                        case.name,
                        count / case.elements(),
                    );
                }
                Err(error) => println!("encode_blocks vs gguf {}: error={error}", case.name),
            }
        }
        let (random_packed, nan_blocks) = random_blocks(&case, count, 0x5eed_0001 ^ case.block_bytes() as u64);
        println!(
            "  decode {}: random blocks={} nan_scale_blocks={} nan_min_blocks={} nan_scale_and_min_blocks={}",
            case.name,
            random_packed.len() / case.block_bytes(),
            nan_blocks.scale,
            nan_blocks.min,
            nan_blocks.both,
        );
        let mut packed_sets = vec![random_packed];
        if case.encoder.is_some() {
            let mut packed = vec![0u8; case.packed_len(count)];
            run_incumbent_encode(&case, kv, &mut packed);
            packed_sets.push(packed);
        }
        let (mut decode_bad, mut decode_compared) = (0usize, 0usize);
        for packed in &packed_sets {
            let elements = packed.len() / case.block_bytes() * case.elements();
            let mut expected = vec![0.0f32; elements];
            let mut got = vec![0.0f32; elements];
            incumbent_block_decode(&case, packed, &mut expected);
            new_block_decode(&codec, packed, &mut got);
            let first = expected.iter().zip(&got).position(|(a, b)| a.to_bits() != b.to_bits());
            if let Some(index) = first {
                let block = index / case.elements();
                println!(
                    "  decode mismatch {}: index={index} incumbent={:#010x} new={:#010x} block_bytes={:02x?}",
                    case.name,
                    expected[index].to_bits(),
                    got[index].to_bits(),
                    &packed[block * case.block_bytes()..block * case.block_bytes() + 4],
                );
            }
            decode_bad += expected.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            decode_compared += elements;
        }
        let encode_oracle = match (case.encoder, codec.can_encode()) {
            (Some(_), _) => case.encoder_label(),
            (None, true) => "none(no incumbent quantize; new encode instantiated, unchecked)",
            (None, false) => "none(no incumbent quantize; new encode absent)",
        };
        println!(
            "blockexact {}: encode oracle={encode_oracle} compared_blocks={encode_compared} mismatched_blocks={encode_blocks_bad} mismatched_bytes={encode_bytes_bad} | decode oracle={} compared_values={decode_compared} mismatched_values={decode_bad}",
            case.name,
            case.decoder_label(),
        );
        // the walker is a timing incumbent, not an oracle: proxima-gguf's block_codec_parity checks tq2_0
        // encode against the llama.cpp transcription on these same 2,097,152 values, so a walker
        // disagreement is printed above and does not gate
        let gating_encode_bad = if matches!(case.encoder, Some(Encoder::Walker)) { 0 } else { encode_blocks_bad };
        if gating_encode_bad + decode_bad != 0 {
            failures.push(format!(
                "block {} disagrees with its incumbent: {encode_blocks_bad} encode blocks, {decode_bad} decode values",
                case.name
            ));
        }
    }
    failures
}

fn bench_blocks(criterion: &mut Criterion, values: &[f32]) {
    for case in block_cases() {
        let codec = case.codec();
        let mut group = criterion.benchmark_group(format!("block_{}", case.name));
        for &count in &SIZES {
            let inputs = &values[..count];
            group.throughput(Throughput::Elements(count as u64));
            let packed = timing_blocks(&case, values, count).0;
            if case.encoder.is_some() {
                let mut buffer = vec![0u8; packed.len()];
                group.bench_function(BenchmarkId::new("encode/incumbent", count), |bencher| {
                    bencher.iter(|| incumbent_block_encode(&case, black_box(inputs), black_box(&mut buffer)));
                });
                group.bench_function(BenchmarkId::new("encode/new", count), |bencher| {
                    bencher.iter(|| new_block_encode(&codec, black_box(inputs), black_box(&mut buffer)));
                });
            }
            let mut out = vec![0.0f32; count];
            group.bench_function(BenchmarkId::new("decode/incumbent", count), |bencher| {
                bencher.iter(|| incumbent_block_decode(&case, black_box(&packed), black_box(&mut out)));
            });
            group.bench_function(BenchmarkId::new("decode/new", count), |bencher| {
                bencher.iter(|| new_block_decode(&codec, black_box(&packed), black_box(&mut out)));
            });
        }
        group.finish();
    }
}

struct BlockCell {
    direction: &'static str,
    name: &'static str,
    label: &'static str,
    count: usize,
    versus_incumbent: Comparison,
    incumbent_versus_copy: Comparison,
    degenerate: Comparison,
    degenerate_versus_new: Comparison,
    timing_blocks: usize,
    timing_nan: NanBlocks,
}

fn run_block_encode_cell(case: &BlockCase, values: &[f32], count: usize) -> BlockCell {
    let codec = case.codec();
    let inputs = &values[..count];
    let size = case.packed_len(count);
    let (mut incumbent_buffer, mut copy_buffer, mut new_buffer, mut degenerate_buffer) =
        (vec![0u8; size], vec![0u8; size], vec![0u8; size], vec![0u8; size]);
    let versus_incumbent = compare(
        || incumbent_block_encode(case, black_box(inputs), black_box(&mut incumbent_buffer)),
        || new_block_encode(&codec, black_box(inputs), black_box(&mut new_buffer)),
    );
    let incumbent_versus_copy = compare(
        || incumbent_block_encode(case, black_box(inputs), black_box(&mut incumbent_buffer)),
        || incumbent_block_encode_copy(case, black_box(inputs), black_box(&mut copy_buffer)),
    );
    let degenerate = compare(
        || incumbent_block_encode(case, black_box(inputs), black_box(&mut incumbent_buffer)),
        || degenerate_block_encode(&codec, black_box(inputs), black_box(&mut degenerate_buffer)),
    );
    let degenerate_versus_new = compare(
        || new_block_encode(&codec, black_box(inputs), black_box(&mut new_buffer)),
        || degenerate_block_encode(&codec, black_box(inputs), black_box(&mut degenerate_buffer)),
    );
    BlockCell {
        direction: "encode",
        name: case.name,
        label: case.encoder_label(),
        count,
        versus_incumbent,
        incumbent_versus_copy,
        degenerate,
        degenerate_versus_new,
        timing_blocks: 0,
        timing_nan: NanBlocks::default(),
    }
}

fn run_block_decode_cell(case: &BlockCase, values: &[f32], count: usize) -> BlockCell {
    let codec = case.codec();
    let (packed, timing_nan) = timing_blocks(case, values, count);
    let (mut incumbent_out, mut copy_out, mut new_out, mut degenerate_out) =
        (vec![0.0f32; count], vec![0.0f32; count], vec![0.0f32; count], vec![0.0f32; count]);
    let versus_incumbent = compare(
        || incumbent_block_decode(case, black_box(&packed), black_box(&mut incumbent_out)),
        || new_block_decode(&codec, black_box(&packed), black_box(&mut new_out)),
    );
    let incumbent_versus_copy = compare(
        || incumbent_block_decode(case, black_box(&packed), black_box(&mut incumbent_out)),
        || incumbent_block_decode_copy(case, black_box(&packed), black_box(&mut copy_out)),
    );
    let degenerate = compare(
        || incumbent_block_decode(case, black_box(&packed), black_box(&mut incumbent_out)),
        || degenerate_block_decode(&codec, black_box(&packed), black_box(&mut degenerate_out)),
    );
    let degenerate_versus_new = compare(
        || new_block_decode(&codec, black_box(&packed), black_box(&mut new_out)),
        || degenerate_block_decode(&codec, black_box(&packed), black_box(&mut degenerate_out)),
    );
    BlockCell {
        direction: "decode",
        name: case.name,
        label: case.decoder_label(),
        count,
        versus_incumbent,
        incumbent_versus_copy,
        degenerate,
        degenerate_versus_new,
        timing_blocks: packed.len() / case.block_bytes(),
        timing_nan,
    }
}

/// `KV_BENCH_CELLS=decode:q4_1,encode:q4_0,decode:bf16:256` limits the block and scalar cells to those
/// `direction:name` pairs (every size) or `direction:name:count` triples (one size); unset runs every
/// cell. A scalar is named `f32`, `f16`, `bf16` or `bf8`.
fn cell_filter() -> Option<Vec<String>> {
    let text = std::env::var("KV_BENCH_CELLS").ok()?;
    Some(text.split(',').map(str::trim).filter(|entry| !entry.is_empty()).map(str::to_string).collect())
}

fn cell_selected(filter: &Option<Vec<String>>, direction: &str, name: &str, count: usize) -> bool {
    filter.as_ref().is_none_or(|entries| {
        entries.iter().any(|entry| *entry == format!("{direction}:{name}") || *entry == format!("{direction}:{name}:{count}"))
    })
}

fn block_cells(values: &[f32]) -> Vec<BlockCell> {
    let filter = cell_filter();
    let mut cells = Vec::new();
    for case in block_cases() {
        for &count in &SIZES {
            if case.encoder.is_some() && cell_selected(&filter, "encode", case.name, count) {
                cells.push(run_block_encode_cell(&case, values, count));
            }
            if cell_selected(&filter, "decode", case.name, count) {
                cells.push(run_block_decode_cell(&case, values, count));
            }
        }
    }
    cells
}

// ---- interleaved harness: `KV_BENCH_INTERLEAVED=1` ---------------------------------------------
//
// Per cell, OLD and NEW batches alternate inside one process, the order flipping every round, so box
// load drifts into both arms equally. The statistic is the median of per-round NEW/OLD ratios with a
// bootstrap 95% interval, beside the min/min ratio. An A/A comparison (OLD against a separately
// compiled copy of OLD) bounds what the harness can resolve; a degenerate comparison (NEW plus a
// deliberate extra pass over its output) must come out FAIL.

/// Rounds per comparison; `KV_BENCH_ROUNDS` shortens a run and is printed with every summary.
fn rounds() -> usize {
    std::env::var("KV_BENCH_ROUNDS").ok().and_then(|text| text.parse().ok()).unwrap_or(200)
}
const MIN_BATCH_NANOS: f64 = 1_000_000.0;
const BOOTSTRAP_RESAMPLES: usize = 2000;
const DELTA: f64 = 0.02;
const LOAD_LIMIT: f64 = 2.0;
const OTHER_CPU_LIMIT: f64 = 20.0;

/// `old_encode` compiled a second time; the empty `black_box` keeps LLVM from merging the two.
#[inline(never)]
fn old_encode_copy(scalar: Scalar, values: &[f32], destination_bytes: &mut [u8]) {
    black_box(());
    let pointer = destination_bytes.as_mut_ptr();
    match scalar {
        Scalar::Float32 => unsafe {
            core::ptr::copy_nonoverlapping(values.as_ptr(), pointer.cast::<f32>(), values.len());
        },
        Scalar::Float16 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer.cast::<f16>(), values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = f16::from_f32(*value);
            }
        }
        Scalar::BFloat16 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer.cast::<bf16>(), values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = bf16::from_f32(*value);
            }
        }
        Scalar::BFloat8 => {
            let destination = unsafe { core::slice::from_raw_parts_mut(pointer, values.len()) };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = BFloat8::from_f32(*value).to_bits();
            }
        }
    }
}

/// `old_decode` compiled a second time; see [`old_encode_copy`].
#[inline(never)]
fn old_decode_copy(scalar: Scalar, source_bytes: &[u8], element_count: usize) -> Vec<f32> {
    black_box(());
    let pointer = source_bytes.as_ptr();
    match scalar {
        Scalar::Float32 => unsafe { core::slice::from_raw_parts(pointer.cast::<f32>(), element_count) }.to_vec(),
        Scalar::Float16 => unsafe { core::slice::from_raw_parts(pointer.cast::<f16>(), element_count) }
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        Scalar::BFloat16 => unsafe { core::slice::from_raw_parts(pointer.cast::<bf16>(), element_count) }
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        Scalar::BFloat8 => unsafe { core::slice::from_raw_parts(pointer, element_count) }
            .iter()
            .map(|value| BFloat8::from_bits(*value).to_f32())
            .collect(),
    }
}

/// NEW encode plus one deliberate read-modify-write pass over the output.
fn degenerate_encode(values: &[f32], buffer: &mut [u8], path: &NewPath) {
    new_encode(values, buffer, path);
    for byte in buffer.iter_mut() {
        *byte = black_box(*byte).wrapping_add(1);
    }
}

/// NEW decode plus one deliberate read-modify-write pass over the decoded values.
fn degenerate_decode(buffer: &[u8], element_count: usize, path: &NewPath) -> Vec<f32> {
    let mut decoded = new_decode(buffer, element_count, path);
    for value in decoded.iter_mut() {
        *value = black_box(*value);
    }
    decoded
}

struct Comparison {
    median: f64,
    low: f64,
    high: f64,
    min_over_min: f64,
    first_nanos_per_call: f64,
    second_nanos_per_call: f64,
    calls: usize,
    shortest_batch_nanos: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Pass,
    Fail,
    Undecided,
    UndecidedBox,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Undecided => "UNDECIDED",
            Self::UndecidedBox => "UNDECIDED(box)",
        }
    }
}

impl Comparison {
    fn half_width(&self) -> f64 {
        (self.high - self.low) / 2.0
    }

    fn is_slower(&self) -> bool {
        self.low > 1.0
    }

    fn contains_one_within_delta(&self) -> bool {
        self.low <= 1.0 && self.high >= 1.0 && self.half_width() <= DELTA
    }
}

/// PASS iff the box is quiet, the A/A control resolves 1 within `DELTA`, the degenerate control is
/// slower than NEW, NEW's median ratio is at most 1 and the interval's upper bound is at most 1 plus
/// this cell's own A/A half-width. FAIL iff the interval's lower bound is above 1. Anything else is
/// UNDECIDED, and UNDECIDED(box) when the box was loaded.
fn cell_rule(quiet: bool, versus: &Comparison, a_a: &Comparison, degenerate_versus_new: &Comparison) -> Rule {
    if versus.is_slower() {
        return Rule::Fail;
    }
    let resolved = a_a.contains_one_within_delta() && degenerate_versus_new.is_slower();
    let not_slower = versus.median <= 1.0 && versus.high <= 1.0 + a_a.half_width();
    match (quiet, resolved && not_slower) {
        (false, _) => Rule::UndecidedBox,
        (true, true) => Rule::Pass,
        (true, false) => Rule::Undecided,
    }
}

fn batch_nanos(arm: &mut impl FnMut(), calls: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..calls {
        arm();
    }
    start.elapsed().as_nanos() as f64
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn bootstrap_interval(ratios: &[f64]) -> (f64, f64) {
    let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
    let mut resample = vec![0.0f64; ratios.len()];
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        for slot in resample.iter_mut() {
            *slot = ratios[(rng.next() % ratios.len() as u64) as usize];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    (medians[BOOTSTRAP_RESAMPLES * 25 / 1000], medians[BOOTSTRAP_RESAMPLES * 975 / 1000 - 1])
}

/// Calls per batch so the faster arm's batch lasts at least `MIN_BATCH_NANOS`.
fn calls_per_batch(first: &mut impl FnMut(), second: &mut impl FnMut()) -> usize {
    let mut calls = 1usize;
    loop {
        let shortest = batch_nanos(first, calls).min(batch_nanos(second, calls)).max(1.0);
        if shortest >= MIN_BATCH_NANOS * 1.05 {
            return calls;
        }
        let scale = (MIN_BATCH_NANOS * 1.2 / shortest).min(1000.0);
        calls = ((calls as f64 * scale).ceil() as usize).max(calls + 1);
    }
}

/// `ROUNDS` rounds of one `first` batch and one `second` batch, the order alternating per round;
/// ratios are `second / first`.
fn compare(mut first: impl FnMut(), mut second: impl FnMut()) -> Comparison {
    let calls = calls_per_batch(&mut first, &mut second);
    for _ in 0..3 {
        batch_nanos(&mut first, calls);
        batch_nanos(&mut second, calls);
    }
    let round_count = rounds();
    let mut first_times = Vec::with_capacity(round_count);
    let mut second_times = Vec::with_capacity(round_count);
    for round in 0..round_count {
        if round % 2 == 0 {
            first_times.push(batch_nanos(&mut first, calls));
            second_times.push(batch_nanos(&mut second, calls));
        } else {
            second_times.push(batch_nanos(&mut second, calls));
            first_times.push(batch_nanos(&mut first, calls));
        }
    }
    let mut ratios: Vec<f64> = second_times.iter().zip(&first_times).map(|(second, first)| second / first).collect();
    let (low, high) = bootstrap_interval(&ratios);
    let minimum = |times: &[f64]| times.iter().copied().fold(f64::MAX, f64::min);
    let mut first_sorted = first_times.clone();
    let mut second_sorted = second_times.clone();
    Comparison {
        median: median(&mut ratios),
        low,
        high,
        min_over_min: minimum(&second_times) / minimum(&first_times),
        first_nanos_per_call: median(&mut first_sorted) / calls as f64,
        second_nanos_per_call: median(&mut second_sorted) / calls as f64,
        calls,
        shortest_batch_nanos: minimum(&first_times).min(minimum(&second_times)),
    }
}

struct BoxGate {
    load_one_minute: f64,
    busiest_other_cpu: f64,
    uptime: String,
    processes: Vec<String>,
}

impl BoxGate {
    fn quiet(&self) -> bool {
        self.load_one_minute < LOAD_LIMIT && self.busiest_other_cpu <= OTHER_CPU_LIMIT
    }
}

fn command_stdout(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

/// `uptime` and `ps -Ao pid,%cpu,comm -r` read now; the busiest process other than this one decides
/// whether the box was quiet.
fn box_gate() -> BoxGate {
    let uptime = command_stdout("uptime", &[]).trim().to_string();
    let load_one_minute = uptime
        .split("load average")
        .nth(1)
        .and_then(|tail| tail.split(':').nth(1))
        .and_then(|numbers| numbers.split_whitespace().next())
        .and_then(|first| first.trim_end_matches(',').parse().ok())
        .unwrap_or(f64::MAX);
    let listing = command_stdout("ps", &["-Ao", "pid,%cpu,comm", "-r"]);
    let processes: Vec<String> = listing.lines().take(6).map(str::to_string).collect();
    let own_pid = std::process::id().to_string();
    let busiest_other_cpu = processes
        .iter()
        .skip(1)
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?;
            let cpu: f64 = fields.next()?.parse().ok()?;
            (pid != own_pid).then_some(cpu)
        })
        .fold(0.0, f64::max);
    BoxGate { load_one_minute, busiest_other_cpu, uptime, processes }
}

fn print_box_gate(label: &str, gate: &BoxGate) {
    println!("boxgate {label}: quiet={} load1={} busiest_other_cpu={}", gate.quiet(), gate.load_one_minute, gate.busiest_other_cpu);
    println!("boxgate {label}: uptime: {}", gate.uptime);
    for line in &gate.processes {
        println!("boxgate {label}: ps: {line}");
    }
}

struct Cell {
    direction: &'static str,
    scalar: Scalar,
    count: usize,
    versus_old: Comparison,
    old_versus_copy: Comparison,
    degenerate: Comparison,
    degenerate_versus_new: Comparison,
}

fn run_encode_cell(scalar: Scalar, values: &[f32], count: usize) -> Cell {
    let path = scalar.format();
    let inputs = &values[..count];
    let size = count * scalar.width();
    let (mut old_buffer, mut copy_buffer, mut new_buffer, mut degenerate_buffer) =
        (aligned_bytes(size), aligned_bytes(size), aligned_bytes(size), aligned_bytes(size));
    let versus_old = compare(
        || old_encode(scalar, black_box(inputs), black_box(&mut old_buffer)),
        || new_encode(black_box(inputs), black_box(&mut new_buffer), &path),
    );
    let old_versus_copy = compare(
        || old_encode(scalar, black_box(inputs), black_box(&mut old_buffer)),
        || old_encode_copy(scalar, black_box(inputs), black_box(&mut copy_buffer)),
    );
    let degenerate = compare(
        || old_encode(scalar, black_box(inputs), black_box(&mut old_buffer)),
        || degenerate_encode(black_box(inputs), black_box(&mut degenerate_buffer), &path),
    );
    let degenerate_versus_new = compare(
        || new_encode(black_box(inputs), black_box(&mut new_buffer), &path),
        || degenerate_encode(black_box(inputs), black_box(&mut degenerate_buffer), &path),
    );
    Cell { direction: "encode", scalar, count, versus_old, old_versus_copy, degenerate, degenerate_versus_new }
}

fn run_decode_cell(scalar: Scalar, values: &[f32], count: usize) -> Cell {
    let path = scalar.format();
    let mut packed = aligned_bytes(count * scalar.width());
    old_encode(scalar, &values[..count], &mut packed);
    let versus_old = compare(
        || drop(black_box(old_decode(scalar, black_box(&packed), count))),
        || drop(black_box(new_decode(black_box(&packed), count, &path))),
    );
    let old_versus_copy = compare(
        || drop(black_box(old_decode(scalar, black_box(&packed), count))),
        || drop(black_box(old_decode_copy(scalar, black_box(&packed), count))),
    );
    let degenerate = compare(
        || drop(black_box(old_decode(scalar, black_box(&packed), count))),
        || drop(black_box(degenerate_decode(black_box(&packed), count, &path))),
    );
    let degenerate_versus_new = compare(
        || drop(black_box(new_decode(black_box(&packed), count, &path))),
        || drop(black_box(degenerate_decode(black_box(&packed), count, &path))),
    );
    Cell { direction: "decode", scalar, count, versus_old, old_versus_copy, degenerate, degenerate_versus_new }
}

fn print_interval(label: &str, comparison: &Comparison) {
    println!(
        "    {label}: old={:.1}ns new={:.1}ns calls={} median={:.4} ci95=[{:.4},{:.4}] min/min={:.4} half_width={:.4} slower={}",
        comparison.first_nanos_per_call,
        comparison.second_nanos_per_call,
        comparison.calls,
        comparison.median,
        comparison.low,
        comparison.high,
        comparison.min_over_min,
        comparison.half_width(),
        comparison.is_slower(),
    );
}

/// What one run of one comparison family (the 32 scalar cells, or the block cells) counted.
#[derive(Default)]
struct Tally {
    cells: usize,
    passed: usize,
    failed: usize,
    undecided: usize,
    undecided_box: usize,
    a_a_outside_gate: usize,
    control_vs_old_not_failed: usize,
    control_vs_new_not_failed: usize,
    batches_under_1ms: usize,
    trailing: Vec<String>,
}

impl Tally {
    fn record(&mut self, quiet: bool, name: &str, versus: &Comparison, a_a: &Comparison, degenerate: &Comparison, degenerate_versus_new: &Comparison) -> Rule {
        let rule = cell_rule(quiet, versus, a_a, degenerate_versus_new);
        self.cells += 1;
        match rule {
            Rule::Pass => self.passed += 1,
            Rule::Fail => self.failed += 1,
            Rule::Undecided => self.undecided += 1,
            Rule::UndecidedBox => self.undecided_box += 1,
        }
        if versus.median > 1.0 {
            self.trailing.push(format!("{name} median={:.4}", versus.median));
        }
        self.a_a_outside_gate += usize::from(!a_a.contains_one_within_delta());
        self.control_vs_old_not_failed += usize::from(!degenerate.is_slower());
        self.control_vs_new_not_failed += usize::from(!degenerate_versus_new.is_slower());
        self.batches_under_1ms += [versus, a_a, degenerate, degenerate_versus_new]
            .iter()
            .filter(|comparison| comparison.shortest_batch_nanos < MIN_BATCH_NANOS)
            .count();
        rule
    }

    fn print(&self, label: &str, quiet: bool) {
        println!("{label}trailing median>1: {} {:?}", self.trailing.len(), self.trailing);
        println!(
            "{label}summary cells={} PASS/FAIL/UNDECIDED/UNDECIDED(box)={}/{}/{}/{} quiet_box={quiet} a_a_cells_outside_gate={} control_vs_old_cells_not_FAIL={} control_vs_new_cells_not_FAIL={} batches_under_1ms={} rounds={} delta={DELTA}",
            self.cells,
            self.passed,
            self.failed,
            self.undecided,
            self.undecided_box,
            self.a_a_outside_gate,
            self.control_vs_old_not_failed,
            self.control_vs_new_not_failed,
            self.batches_under_1ms,
            rounds(),
        );
    }

    fn not_passed(&self) -> usize {
        self.cells - self.passed
    }
}

fn print_block_cells(cells: &[BlockCell], quiet: bool) -> Tally {
    let mut tally = Tally::default();
    for cell in cells {
        println!("blockcell {} {} {} incumbent={}:", cell.direction, cell.name, cell.count, cell.label);
        print_interval("new vs inc ", &cell.versus_incumbent);
        print_interval("a/a inc/inc", &cell.incumbent_versus_copy);
        print_interval("control/inc", &cell.degenerate);
        print_interval("control/new", &cell.degenerate_versus_new);
        let name = format!("{} {} {}", cell.direction, cell.name, cell.count);
        let rule = tally.record(quiet, &name, &cell.versus_incumbent, &cell.incumbent_versus_copy, &cell.degenerate, &cell.degenerate_versus_new);
        println!("    status={}", rule.name());
    }
    tally.print("block", quiet);
    tally
}

fn print_scalar_cells(cells: &[Cell], quiet: bool) -> Tally {
    let mut tally = Tally::default();
    for cell in cells {
        println!("cell {} {} {}:", cell.direction, cell.scalar.name(), cell.count);
        print_interval("new vs old ", &cell.versus_old);
        print_interval("a/a old/old", &cell.old_versus_copy);
        print_interval("control/old", &cell.degenerate);
        print_interval("control/new", &cell.degenerate_versus_new);
        let name = format!("{} {} {}", cell.direction, cell.scalar.name(), cell.count);
        let rule = tally.record(quiet, &name, &cell.versus_old, &cell.old_versus_copy, &cell.degenerate, &cell.degenerate_versus_new);
        println!("    status={}", rule.name());
    }
    tally.print("", quiet);
    tally
}

/// 4 scalars x encode and decode x the 4 sizes.
const SCALAR_CELLS: usize = 32;

/// (7 instantiated encodes with a timing incumbent: q4_0, q5_0, q8_0, q1_0, q2_0 and pq2_0 against
/// `proxima-gguf`, tq2_0 against the walker; plus 13 table decodes) x the 4 sizes.
const BLOCK_CELLS: usize = (7 + 13) * 4;

/// Sums the NaN census of every decode cell's timing input; the returned line is a failure when any
/// timed block has a NaN scale or min, since the template returns early on those.
fn timing_census(cells: &[BlockCell]) -> Option<String> {
    let (blocks, scale, min) = cells.iter().fold((0, 0, 0), |(blocks, scale, min), cell| {
        (blocks + cell.timing_blocks, scale + cell.timing_nan.scale, min + cell.timing_nan.min)
    });
    println!("census timing_inputs: blocks={blocks} nan_scale_blocks={scale} nan_min_blocks={min}");
    (blocks == 0 || scale + min != 0).then(|| format!("timing inputs: {blocks} blocks, {scale} nan scale, {min} nan min (want blocks > 0 and 0 nan)"))
}

/// Runs every cell, then returns the failures: one line per cell not PASS and one for a cell count
/// that is not the expected one. A run is a pass only when this is empty.
fn interleaved(values: &[f32]) -> Vec<String> {
    let before = box_gate();
    print_box_gate("before", &before);
    let filtered = cell_filter().is_some();
    let blocks_only = std::env::var_os("KV_BENCH_BLOCKS_ONLY").is_some();
    let filter = cell_filter();
    let mut cells = Vec::new();
    let scalars: &[Scalar] = if blocks_only { &[] } else { &Scalar::ALL };
    for &scalar in scalars {
        for &count in &SIZES {
            if cell_selected(&filter, "encode", scalar.name(), count) {
                cells.push(run_encode_cell(scalar, values, count));
            }
            if cell_selected(&filter, "decode", scalar.name(), count) {
                cells.push(run_decode_cell(scalar, values, count));
            }
        }
    }
    let block_cells = block_cells(values);
    let after = box_gate();
    print_box_gate("after", &after);
    let quiet = before.quiet() && after.quiet();
    let block_tally = print_block_cells(&block_cells, quiet);
    let scalar_tally = print_scalar_cells(&cells, quiet);
    let mut failures = Vec::new();
    let expected_scalar_cells = if blocks_only { 0 } else { SCALAR_CELLS };
    if !filtered && cells.len() != expected_scalar_cells {
        failures.push(format!("scalar cells {} != expected {expected_scalar_cells}", cells.len()));
    }
    println!("blockcells={} expected={BLOCK_CELLS}", block_cells.len());
    if filtered {
        println!(
            "AC6 UNCHECKED: KV_BENCH_CELLS selected {} of {BLOCK_CELLS} block cells and {} of {expected_scalar_cells} scalar cells",
            block_cells.len(),
            cells.len(),
        );
    } else if block_cells.len() != BLOCK_CELLS {
        failures.push(format!("block cells {} != expected {BLOCK_CELLS}", block_cells.len()));
    }
    failures.extend(timing_census(&block_cells));
    if blocks_only {
        println!("AC6 UNCHECKED: KV_BENCH_BLOCKS_ONLY skipped the {SCALAR_CELLS} scalar cells");
    }
    for (family, tally) in [("scalar", &scalar_tally), ("block", &block_tally)] {
        if tally.not_passed() > 0 {
            failures.push(format!(
                "{family} cells not PASS: {} of {} (FAIL {}, UNDECIDED {}, UNDECIDED(box) {})",
                tally.not_passed(),
                tally.cells,
                tally.failed,
                tally.undecided,
                tally.undecided_box,
            ));
        }
    }
    failures
}

/// The scalar and block parity checks, all run, each reporting on its own line; the returned list is
/// every failure. Run before any timing in every mode.
fn parity_checks(values: &[f32]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut report = |name: &str, found: Vec<String>| {
        println!("paritycheck {name}: failures={}", found.len());
        failures.extend(found);
    };
    report("exhaustive_decode", exhaustive_decode_parity());
    report("block_bit_exactness", block_bit_exactness(values));
    report("kv_inputs_2097152", bit_exactness("kv_inputs_2097152", values, false));
    report("random_bits_1048576", bit_exactness("random_bits_1048576", &random_bit_patterns(1 << 20), true));
    failures
}

fn exit_with(failures: &[String]) {
    if failures.is_empty() {
        return;
    }
    println!("FAILURES {}:", failures.len());
    for failure in failures {
        println!("  {failure}");
    }
    std::process::exit(1);
}

fn main() {
    let values = kv_like_values(*SIZES.last().unwrap());
    let parity_failures = parity_checks(&values);
    if std::env::var_os("KV_BENCH_INTERLEAVED").is_some() {
        exit_with(&parity_failures);
        exit_with(&interleaved(&values));
        return;
    }
    exit_with(&parity_failures);
    report_allocations(&values);
    let measurement_seconds = std::env::var("KV_BENCH_SECS").ok().and_then(|text| text.parse().ok()).unwrap_or(2);
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(measurement_seconds));
    bench_encode(&mut criterion, &values);
    bench_decode(&mut criterion, &values);
    bench_blocks(&mut criterion, &values);
    criterion.final_summary();
}
