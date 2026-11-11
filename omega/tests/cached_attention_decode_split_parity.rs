//! Metal-vs-CPU parity for the split-KV decode attention form
//! (`CachedAttentionForm::TwoRangeDecodeSplit`: `omega/src/msl/
//! cached_attention_decode_split.rs` plus the interleaved merge in
//! `cached_attention_render.rs`). Internal consistency only: the CPU evaluator
//! is this repo's own oracle, never llama.cpp's -- the llama.cpp greedy-id gate
//! is `proxima-model-interop/tests/gemma4_attn_split_decode_oracle.rs`.
//!
//! The program is the qwen3 two-range cached decode builder at a gemma-like
//! GQA shape (8 query heads on one kv head) at head_dim 64 and at the two gemma4-E2B
//! heads (256 sliding, 512 global), swept over cached lengths that land on every regime: a single split (direct output), two
//! splits, a handful, and the full 32. Each cell asserts the bound op really
//! takes the split form (entry name `_ds`) before comparing, so a pass cannot
//! be two copies of the one-dispatch kernel agreeing with each other.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-decode",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::{bf16, f16};
use proxima_numeric::{FormatCodec, NumericFormat};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::qk_norm_cached_forward_program;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BFloat8, BoundOpKind, NodeId, Op, bind, infer};

mod support;
use support::{as_named_blocks, evaluate_gpu_oracle, production_numeric_policy};

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 512;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
/// The narrow fixture head (a lane owns one float4 of an eight-float4 plane) and the
/// two gemma4-E2B heads: sliding 256 and global 512.
const HEAD_DIMS: [u32; 3] = [64, 256, 512];
const LAYERS: u32 = 2;

/// Cached lengths: capacity 32 (one split, direct output), 34 (two), 200
/// (seven), 1100 (thirty-two, two chunks do not engage below 2048).
const CACHED_LENGTHS: [u64; 4] = [31, 33, 199, 1099];

struct Fixture {
    program: Vec<Op>,
    symbols: Vec<u64>,
    roots: Vec<NodeId>,
    named: Vec<(String, Vec<f32>)>,
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn gqa_decode_fixture(cached_len: u64, head_dim: u32) -> Fixture {
    let (program, logits_root, cache_roots) = qk_norm_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        head_dim,
        LAYERS,
    )
    .expect("the gemma-like gqa decode program builds");
    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.extend([*even, *odd, *value]);
    }
    let symbols = vec![1, cached_len];
    let shapes = infer(&program, &symbols).expect("the decode program infers");

    let mut named = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let count: usize = shapes
            .of(NodeId(position as u32))
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let data = match name.as_str() {
            "ids" => vec![3.0f32; count],
            "eps" => vec![1e-5f32; count],
            "cached_len" => vec![cached_len as f32],
            _ => random_vec(position as u64 + 1, count),
        };
        named.push((name, data));
    }
    Fixture {
        program,
        symbols,
        roots,
        named,
    }
}

fn is_kv_cache(name: &str) -> bool {
    name.starts_with("kv_cache.")
}

fn round_to_half(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect()
}

/// The cache leaves as the device holds them: binary16, little-endian.
fn half_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect()
}

fn round_to_bf16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn round_to_bf8(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| BFloat8::from_f32(*value).to_f32())
        .collect()
}

fn bf8_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| BFloat8::from_f32(*value).to_bits())
        .collect()
}

/// `half_kv` runs the cache leaves as `Codec::Float16` blocks on Metal and as
/// their binary16-rounded f32 values on the CPU, so both executors attend over
/// exactly the same numbers and the comparison isolates the kernel's half read.
fn parity_cell(cached_len: u64, head_dim: u32, half_kv: bool) -> f32 {
    let policy = production_numeric_policy();
    let mut fixture = gqa_decode_fixture(cached_len, head_dim);
    let half_blocks: Vec<(String, Vec<u8>)> = if half_kv {
        fixture
            .named
            .iter_mut()
            .filter(|(name, _)| is_kv_cache(name))
            .map(|(name, data)| {
                *data = round_to_half(data);
                (name.clone(), half_bytes(data))
            })
            .collect()
    } else {
        Vec::new()
    };
    assert_eq!(
        half_blocks.len(),
        if half_kv { 3 * LAYERS as usize } else { 0 },
        "every layer contributes a K even, K odd and V cache leaf"
    );
    let output_roots = [fixture.roots[0]];
    let named = as_named_blocks(&fixture.named);

    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved =
        bind(&fixture.program, &shapes, &output_roots, policy).expect("the decode program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert!(
        !attention.is_empty(),
        "head_dim {head_dim} cached_len {cached_len}: zero CachedAttention ops would compare nothing"
    );
    for bound in &attention {
        let mut packed = omega::PackedOperands::new();
        if half_kv {
            for index in [2, 3, 6] {
                packed.insert(bound.operands()[index].0, omega::Codec::Float16);
            }
        }
        let kernel = omega::emit(bound, &packed, policy).expect("the bound attention op emits");
        assert_eq!(
            kernel.source.contains("device const half* in2"),
            half_kv,
            "head_dim {head_dim} cached_len {cached_len}: the cached K plane binds as half exactly when the cache is half"
        );
        assert!(
            kernel.entry.ends_with("_ds"),
            "head_dim {head_dim} cached_len {cached_len}: the op must take the decode split form, got {}",
            kernel.entry
        );
    }

    let cpu = evaluate_gpu_oracle(&fixture.program, &fixture.symbols, &named, &output_roots);
    let metal_named: Vec<(&str, QuantizedBlock<'_>)> = named
        .iter()
        .map(
            |(name, block)| match half_blocks.iter().find(|(half_name, _)| half_name == name) {
                Some((_, bytes)) => (
                    *name,
                    QuantizedBlock::Packed {
                        codec: omega::Codec::Float16,
                        bytes: bytes.as_slice(),
                    },
                ),
                None => (*name, *block),
            },
        )
        .collect();
    let plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &metal_named,
        &output_roots,
        policy,
    )
    .expect("metal plans the decode program");
    let metal =
        omega::execute_plan_named(&plan, &metal_named).expect("metal runs the decode program");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        expected.len(),
        "head_dim {head_dim} cached_len {cached_len}"
    );
    assert!(
        !expected.is_empty(),
        "head_dim {head_dim} cached_len {cached_len}: an empty root compares nothing"
    );
    assert_all_finite(expected, &format!("head_dim {head_dim} cached_len {cached_len} cpu root"));
    assert_all_finite(actual, &format!("head_dim {head_dim} cached_len {cached_len} metal root"));
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "decode_split parity: head_dim={head_dim} cached_len={cached_len} half_kv={half_kv} attention_ops={} max_diff={max_diff} relative={relative}",
        attention.len()
    );
    relative
}

#[test]
fn the_decode_split_kernels_hold_parity_with_the_cpu_evaluator_across_split_counts() {
    let mut cells = 0_usize;
    for head_dim in HEAD_DIMS {
        for cached_len in CACHED_LENGTHS {
            let relative = parity_cell(cached_len, head_dim, false);
            assert!(
                relative < 1e-4,
                "head_dim {head_dim} cached_len {cached_len}: metal disagrees with cpu on the decode split root: relative={relative}"
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 12, "3 head dims x 4 cached lengths");
}

#[test]
fn the_decode_split_reads_a_float16_cache_as_the_cpu_reads_the_same_rounded_values() {
    let mut cells = 0_usize;
    for head_dim in HEAD_DIMS {
        for cached_len in CACHED_LENGTHS {
            let relative = parity_cell(cached_len, head_dim, true);
            assert!(
                relative < 1e-4,
                "head_dim {head_dim} cached_len {cached_len}: the half-read kernel disagrees with cpu on identical binary16 values: relative={relative}"
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 12, "3 head dims x 4 cached lengths");
}

#[test]
fn card_06_bf16_decode_matches_f32_decode_output_bits() {
    let policy = production_numeric_policy();
    let mut fixture = gqa_decode_fixture(31, 64);
    let exact_values = [0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
    let bf16_blocks: Vec<(String, Vec<u8>)> = fixture
        .named
        .iter_mut()
        .filter(|(name, _)| is_kv_cache(name))
        .map(|(name, values)| {
            for (index, value) in values.iter_mut().enumerate() {
                *value = exact_values[index % exact_values.len()];
            }
            assert_eq!(
                *values,
                round_to_bf16(values),
                "fixture values are exact BF16 values"
            );
            (name.clone(), bf16_bytes(values))
        })
        .collect();
    assert_eq!(bf16_blocks.len(), 3 * LAYERS as usize);

    let output_roots = [fixture.roots[0]];
    let f32_named = as_named_blocks(&fixture.named);
    let bf16_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, values)| {
            match bf16_blocks
                .iter()
                .find(|(cache_name, _)| cache_name == name)
            {
                Some((_, bytes)) => (
                    name.as_str(),
                    QuantizedBlock::Packed {
                        codec: omega::Codec::BFloat16,
                        bytes,
                    },
                ),
                None => (name.as_str(), QuantizedBlock::Float32(values)),
            }
        })
        .collect();
    let f32_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &f32_named,
        &output_roots,
        policy,
    )
    .expect("F32 cache dispatch plans");
    let bf16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &bf16_named,
        &output_roots,
        policy,
    )
    .expect("BF16 cache dispatch plans");

    let f32_output = finite_root_bits(
        omega::execute_plan_named(&f32_plan, &f32_named)
            .expect("F32 cache decode executes")
            .root(),
        "F32 cache",
    );
    let bf16_output = finite_root_bits(
        omega::execute_plan_named(&bf16_plan, &bf16_named)
            .expect("BF16 cache decode executes")
            .root(),
        "BF16 cache",
    );
    assert_eq!(
        bf16_output, f32_output,
        "the cache decode has the same f32 inputs and operations"
    );
}

#[test]
fn card_06_bf16_decode_selects_source_and_declines_mixed_cache_codecs() {
    let policy = production_numeric_policy();
    let fixture = gqa_decode_fixture(31, 64);
    let output_roots = [fixture.roots[0]];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved =
        bind(&fixture.program, &shapes, &output_roots, policy).expect("the decode program binds");
    let attention = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the fixture contains a cached attention operation");
    let bf16 = [2, 3, 6]
        .into_iter()
        .map(|index| (attention.operands()[index].0, omega::Codec::BFloat16))
        .collect::<omega::PackedOperands>();
    let emitted = omega::emit(attention, &bf16, policy).expect("BF16 decode source emits");
    assert!(
        emitted.entry.ends_with("_ds"),
        "selected source is decode split"
    );
    assert!(emitted.source.contains("device const ushort* in2"));
    assert!(emitted.source.contains("omega_bf16x4_to_float4"));
    assert!(emitted.source.contains("float4(kr4_new[index])"));

    let mixed = [
        (attention.operands()[2].0, omega::Codec::BFloat16),
        (attention.operands()[3].0, omega::Codec::BFloat16),
        (attention.operands()[6].0, omega::Codec::Float16),
    ]
    .into_iter()
    .collect::<omega::PackedOperands>();
    let error = omega::emit(attention, &mixed, policy)
        .expect_err("mixed BF16 and F16 cache operands must decline");
    assert!(matches!(
        error,
        omega::EmitError::CachedAttentionKvCodecNotSupported { .. }
    ));
}

#[test]
fn card_09_bf8_decode_matches_golden_vectors_and_declines_mixed_cache_codecs() {
    let inputs = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3fa0_0000,
        0x3fc0_0000,
        0x3f90_0000,
        0x3fb0_0000,
        0x3880_0000,
        0x3780_0000,
        0x4760_0000,
        0x4770_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
    ]
    .map(f32::from_bits);
    let encoded = [
        0x00, 0x80, 0x3c, 0xbc, 0x3d, 0x3e, 0x3c, 0x3e, 0x04, 0x01, 0x7b, 0x7c, 0x7c, 0xfc, 0x7e,
        0xfe,
    ];
    let decoded = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3fa0_0000,
        0x3fc0_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x3880_0000,
        0x3780_0000,
        0x4760_0000,
        0x7f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
    ];
    assert_eq!(
        inputs.map(|value| BFloat8::from_f32(value).to_bits()),
        encoded
    );
    assert_eq!(
        encoded.map(|value| BFloat8::from_bits(value).to_f32().to_bits()),
        decoded
    );

    let policy = production_numeric_policy();
    let fixture = gqa_decode_fixture(31, 64);
    let output_roots = [fixture.roots[0]];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved =
        bind(&fixture.program, &shapes, &output_roots, policy).expect("the decode program binds");
    let attention = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the fixture contains a cached attention operation");
    let bf8 = [2, 3, 6]
        .into_iter()
        .map(|index| (attention.operands()[index].0, omega::Codec::BFloat8))
        .collect::<omega::PackedOperands>();
    let emitted = omega::emit(attention, &bf8, policy).expect("BF8 decode source emits");
    assert!(
        emitted.entry.ends_with("_ds"),
        "selected source is decode split"
    );
    assert!(emitted.source.contains("device const uchar* in2"));
    assert!(emitted.source.contains("device const uchar4* kr4_cached"));
    assert!(emitted.source.contains("omega_bf8x4_to_float4"));
    assert!(emitted.source.contains("ldexp((float)mantissa, -16)"));

    let mixed = [
        (attention.operands()[2].0, omega::Codec::BFloat8),
        (attention.operands()[3].0, omega::Codec::BFloat8),
        (attention.operands()[6].0, omega::Codec::Float16),
    ]
    .into_iter()
    .collect::<omega::PackedOperands>();
    let error = omega::emit(attention, &mixed, policy)
        .expect_err("mixed BF8 and F16 cache operands must decline");
    assert!(matches!(
        error,
        omega::EmitError::CachedAttentionKvCodecNotSupported { .. }
    ));
}

#[test]
fn card_09_bf8_decode_matches_f32_cache_output_bits() {
    let policy = production_numeric_policy();
    let mut fixture = gqa_decode_fixture(31, 64);
    let exact_values = [0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
    let bf8_blocks: Vec<(String, Vec<u8>)> = fixture
        .named
        .iter_mut()
        .filter(|(name, _)| is_kv_cache(name))
        .map(|(name, values)| {
            for (index, value) in values.iter_mut().enumerate() {
                *value = exact_values[index % exact_values.len()];
            }
            assert_eq!(
                *values,
                round_to_bf8(values),
                "fixture values are exact BF8 values"
            );
            (name.clone(), bf8_bytes(values))
        })
        .collect();
    assert_eq!(bf8_blocks.len(), 3 * LAYERS as usize);

    let output_roots = [fixture.roots[0]];
    let f32_named = as_named_blocks(&fixture.named);
    let bf8_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(
            |(name, values)| match bf8_blocks.iter().find(|(cache_name, _)| cache_name == name) {
                Some((_, bytes)) => (
                    name.as_str(),
                    QuantizedBlock::Packed {
                        codec: omega::Codec::BFloat8,
                        bytes,
                    },
                ),
                None => (name.as_str(), QuantizedBlock::Float32(values)),
            },
        )
        .collect();
    let f32_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &f32_named,
        &output_roots,
        policy,
    )
    .expect("F32 cache dispatch plans");
    let bf8_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &bf8_named,
        &output_roots,
        policy,
    )
    .expect("BF8 cache dispatch plans");
    assert_ne!(
        f32_plan
            .kernel_keys()
            .expect("F32 plan exposes kernel identities"),
        bf8_plan
            .kernel_keys()
            .expect("BF8 plan exposes kernel identities"),
        "the BF8 cache reader has a distinct pipeline identity"
    );

    let bf8_output = finite_root_bits(
        omega::execute_plan_named(&bf8_plan, &bf8_named)
            .expect("BF8 cache decode executes")
            .root(),
        "BF8 cache",
    );
    let bf8_output_repeat = finite_root_bits(
        omega::execute_plan_named(&bf8_plan, &bf8_named)
            .expect("repeated BF8 cache decode executes")
            .root(),
        "repeated BF8 cache",
    );
    assert_eq!(
        bf8_output_repeat, bf8_output,
        "BF8 dispatch repeats its output bytes"
    );
    let f32_output = finite_root_bits(
        omega::execute_plan_named(&f32_plan, &f32_named)
            .expect("F32 cache decode executes")
            .root(),
        "F32 cache",
    );
    assert_eq!(
        bf8_output, f32_output,
        "BF8 cache values widen into the same f32 decode"
    );
}

fn is_key_leaf(name: &str) -> bool {
    name.ends_with(".k_even") || name.ends_with(".k_odd")
}

/// `values` packed by the host codec and decoded back by it: the bytes the device reads and the
/// f32 values the CPU evaluator attends over.
fn block_round_trip(format: &NumericFormat, values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let codec = FormatCodec::for_format(format).expect("the format resolves a host codec");
    let mut packed = vec![0u8; codec.packed_len(values.len()).expect("a whole number of blocks")];
    codec.encode(values, &mut packed).expect("the host encodes the cache rows");
    let mut slots = vec![std::mem::MaybeUninit::<f32>::uninit(); values.len()];
    codec.decode(&packed, &mut slots).expect("the host decodes the cache rows");
    // decode returned Ok, which initialises every slot of its output
    let decoded = slots.iter().map(|slot| unsafe { slot.assume_init() }).collect();
    (packed, decoded)
}

/// The largest `|decoded - original|` as a fraction of its 32-element block's largest magnitude:
/// a Q4_0 element is rounded to a step of `max / 8` and the extreme opposite the block maximum
/// clamps a full step short, so no element is off by more than `1/8` of the block maximum.
fn worst_error_over_block_max(original: &[f32], decoded: &[f32]) -> f32 {
    original
        .chunks(32)
        .zip(decoded.chunks(32))
        .flat_map(|(block, rebuilt)| {
            let block_max = block.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
            block
                .iter()
                .zip(rebuilt)
                .map(move |(want, got)| (want - got).abs() / block_max.max(f32::MIN_POSITIVE))
        })
        .fold(0.0f32, f32::max)
}

/// f32::max drops a NaN operand, so a kernel that writes NaN everywhere would score zero error:
/// every comparison first requires finite values on both sides.
fn assert_all_finite(values: &[f32], what: &str) {
    let non_finite = values.iter().filter(|value| !value.is_finite()).count();
    assert_eq!(non_finite, 0, "{non_finite}/{} {what} values are not finite", values.len());
}

/// The bit patterns of a root for an equality check: bit equality passes when both sides are the
/// same NaN, so a root must be non-empty and finite before its bits are compared.
fn finite_root_bits(root: &[f32], what: &str) -> Vec<u32> {
    assert!(!root.is_empty(), "an empty {what} root compares nothing");
    assert_all_finite(root, what);
    root.iter().map(|value| value.to_bits()).collect()
}

fn max_relative(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    assert!(!expected.is_empty(), "an empty root compares nothing");
    assert_all_finite(expected, "expected");
    assert_all_finite(actual, "actual");
    let peak = expected.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
    let worst = expected.iter().zip(actual).fold(0.0f32, |worst, (want, got)| worst.max((want - got).abs()));
    worst / peak.max(f32::MIN_POSITIVE)
}

/// The CPU side attends over the host-decoded K and half-rounded V, so Metal's block read and the
/// oracle see identical numbers; the bound is the 1e-4 the binary16-read test holds for the same
/// situation (identical values in, only the kernel's f32 accumulation order differs).
const BLOCK_K_CPU_BOUND: f32 = 1e-4;

struct BlockKeyCell {
    metal_vs_cpu: f32,
    metal_vs_host_decode: f32,
    metal_vs_f16_cache: f32,
    step_fraction: f32,
}

/// Runs the decode program with K cached as `format` blocks and V as binary16 on Metal, against
/// (a) the CPU evaluator over the host-decoded K and half-rounded V, which isolates the kernel's
/// block read, and (b) Metal over the same un-quantized inputs cached as binary16, which measures
/// what the K quantization costs the output.
fn block_key_cell(format: NumericFormat, cached_len: u64, head_dim: u32) -> BlockKeyCell {
    let policy = production_numeric_policy();
    let fixture = gqa_decode_fixture(cached_len, head_dim);
    let output_roots = [fixture.roots[0]];

    let mut host_named = fixture.named.clone();
    let mut block_named: Vec<(String, omega::Codec, Vec<u8>)> = Vec::new();
    let mut step_fraction = 0.0f32;
    for (name, data) in host_named.iter_mut().filter(|(name, _)| is_kv_cache(name)) {
        if is_key_leaf(name) {
            let (packed, decoded) = block_round_trip(&format, data);
            assert_all_finite(&decoded, "host-decoded K");
            step_fraction = step_fraction.max(worst_error_over_block_max(data, &decoded));
            *data = decoded;
            block_named.push((name.clone(), format, packed));
        } else {
            *data = round_to_half(data);
            block_named.push((name.clone(), omega::Codec::Float16, half_bytes(data)));
        }
    }
    assert_eq!(block_named.len(), 3 * LAYERS as usize, "a K even, K odd and V leaf per layer");

    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved = bind(&fixture.program, &shapes, &output_roots, policy).expect("the decode program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert!(!attention.is_empty(), "zero CachedAttention ops would compare nothing");
    for bound in &attention {
        let mut packed = omega::PackedOperands::new();
        for index in [2, 3] {
            packed.insert(bound.operands()[index].0, format);
        }
        packed.insert(bound.operands()[6].0, omega::Codec::Float16);
        let kernel = omega::emit(bound, &packed, policy).expect("a block K beside a binary16 V emits");
        assert!(kernel.entry.ends_with("_ds"), "the op must take the decode split form, got {}", kernel.entry);
        assert!(kernel.source.contains("device const uchar* in2"), "the cached K plane binds as raw blocks");
        assert!(kernel.source.contains("omega_key_block_"), "the K read goes through the rendered layout helper");
    }

    let host_blocks = as_named_blocks(&host_named);
    let cpu = evaluate_gpu_oracle(&fixture.program, &fixture.symbols, &host_blocks, &output_roots);

    let block_blocks: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, data)| match block_named.iter().find(|(leaf, _, _)| leaf == name) {
            Some((_, codec, bytes)) => (name.as_str(), QuantizedBlock::Packed { codec: *codec, bytes }),
            None => (name.as_str(), QuantizedBlock::Float32(data)),
        })
        .collect();
    let block_plan = omega::plan_named(&fixture.program, &fixture.symbols, &block_blocks, &output_roots, policy)
        .expect("metal plans the block K decode program");
    let block_run = omega::execute_plan_named(&block_plan, &block_blocks).expect("metal runs the block K decode program");

    let half_leaves: Vec<(String, Vec<u8>)> = fixture
        .named
        .iter()
        .filter(|(name, _)| is_kv_cache(name))
        .map(|(name, data)| (name.clone(), half_bytes(data)))
        .collect();
    let half_blocks: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, data)| match half_leaves.iter().find(|(leaf, _)| leaf == name) {
            Some((_, bytes)) => (name.as_str(), QuantizedBlock::Packed { codec: omega::Codec::Float16, bytes }),
            None => (name.as_str(), QuantizedBlock::Float32(data)),
        })
        .collect();
    let half_plan = omega::plan_named(&fixture.program, &fixture.symbols, &half_blocks, &output_roots, policy)
        .expect("metal plans the binary16 decode program");
    let half_run = omega::execute_plan_named(&half_plan, &half_blocks).expect("metal runs the binary16 decode program");

    let plain_plan = omega::plan_named(&fixture.program, &fixture.symbols, &host_blocks, &output_roots, policy)
        .expect("metal plans the decoded plain f32 program");
    let plain_run = omega::execute_plan_named(&plain_plan, &host_blocks).expect("metal runs the decoded plain f32 program");

    let cell = BlockKeyCell {
        metal_vs_cpu: max_relative(cpu.root(), block_run.root()),
        metal_vs_host_decode: max_relative(plain_run.root(), block_run.root()),
        metal_vs_f16_cache: max_relative(half_run.root(), block_run.root()),
        step_fraction,
    };
    eprintln!(
        "block_key parity: head_dim={head_dim} cached_len={cached_len} metal_vs_cpu={} metal_vs_host_decode={} metal_vs_f16_cache={} worst_key_error_over_block_max={}",
        cell.metal_vs_cpu, cell.metal_vs_host_decode, cell.metal_vs_f16_cache, cell.step_fraction
    );
    cell
}

#[test]
fn cached_attention_decode_split_reads_a_q4_0_k_cache_as_the_host_codec_decodes_it() {
    let mut cells = 0_usize;
    for head_dim in HEAD_DIMS {
        for cached_len in CACHED_LENGTHS {
            let cell = block_key_cell(NumericFormat::Q4_0, cached_len, head_dim);
            assert!(
                cell.metal_vs_host_decode < 1e-4,
                "head_dim {head_dim} cached_len {cached_len}: metal over the block K disagrees with metal over the same host-decoded values as plain f32: {}",
                cell.metal_vs_host_decode
            );
            assert!(
                cell.metal_vs_cpu < BLOCK_K_CPU_BOUND,
                "head_dim {head_dim} cached_len {cached_len}: metal over the block K disagrees with the f32 cpu oracle over the host-decoded K and half-rounded V: {}",
                cell.metal_vs_cpu
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 12, "3 head dims x 4 cached lengths");
}

/// Q4_0 rounds each element to a step of `block max / 8`, so no K element moves by more than
/// one step of its block maximum: the block's largest value maps to code 0 exactly and the
/// opposite extreme clamps to code 15, a full step short (asserted on the data below). The output
/// is compared at that same `1/8` relative bound: attention is smooth in K, and the fixture's K
/// is uniform in [-1, 1).
#[test]
fn cached_attention_q4_0_k_cache_moves_the_output_by_no_more_than_the_q4_0_step_of_the_inputs() {
    let half_step = 1.0f32 / 8.0; // one full Q4_0 step as a fraction of the block maximum
    let mut cells = 0_usize;
    for head_dim in HEAD_DIMS {
        for cached_len in CACHED_LENGTHS {
            let cell = block_key_cell(NumericFormat::Q4_0, cached_len, head_dim);
            assert!(
                // the stored scale is binary16, so the step carries up to 8 * 2^-11 of relative slack
                cell.step_fraction <= half_step * (1.0 + 1.0 / 256.0),
                "head_dim {head_dim}: a K element moved by {} of its block max, past one Q4_0 step",
                cell.step_fraction
            );
            assert!(
                cell.metal_vs_f16_cache <= half_step,
                "head_dim {head_dim} cached_len {cached_len}: Q4_0 K moved the output by {} against the binary16 cache, past {half_step}",
                cell.metal_vs_f16_cache
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 12, "3 head dims x 4 cached lengths");
}

#[test]
fn cached_attention_decode_split_reads_a_q8_0_k_cache_through_the_same_rendered_layout_load() {
    let cell = block_key_cell(NumericFormat::Q8_0, 199, 256);

    assert!(cell.metal_vs_host_decode < 1e-4, "{}", cell.metal_vs_host_decode);
    assert!(cell.metal_vs_cpu < BLOCK_K_CPU_BOUND, "{}", cell.metal_vs_cpu);
}

#[cfg(all(feature = "instrument", feature = "metal-output-placement"))]
fn attention_gpu_ns(timings: &[omega::metal::OpGpuTiming]) -> (u64, usize) {
    timings
        .iter()
        .filter(|timing| timing.kind.contains("attention"))
        .fold((0, 0), |(total, ops), timing| (total + timing.gpu_ns, ops + 1))
}

/// The K and V cache leaves of the fixture with K stored as `key_format` (binary16 when `None`) and V as binary16.
#[cfg(all(feature = "instrument", feature = "metal-output-placement"))]
fn timed_cache_leaves(fixture: &Fixture, key_format: Option<NumericFormat>) -> Vec<(String, omega::Codec, Vec<u8>)> {
    fixture
        .named
        .iter()
        .filter(|(name, _)| is_kv_cache(name))
        .map(|(name, data)| match key_format {
            Some(format) if is_key_leaf(name) => (name.clone(), format, block_round_trip(&format, data).0),
            _ => (name.clone(), omega::Codec::Float16, half_bytes(data)),
        })
        .collect()
}

#[cfg(all(feature = "instrument", feature = "metal-output-placement"))]
fn timed_blocks<'a>(
    fixture: &'a Fixture,
    leaves: &'a [(String, omega::Codec, Vec<u8>)],
) -> Vec<(&'a str, QuantizedBlock<'a>)> {
    fixture
        .named
        .iter()
        .map(|(name, data)| match leaves.iter().find(|(leaf, _, _)| leaf == name) {
            Some((_, codec, bytes)) => (name.as_str(), QuantizedBlock::Packed { codec: *codec, bytes }),
            None => (name.as_str(), QuantizedBlock::Float32(data)),
        })
        .collect()
}

/// Per-step GPU time of the decode-split attention ops with the K cache as binary16, Q4_0 and Q8_0,
/// interleaved so box load lands on every arm alike. Reports min / median / max of the sample.
#[cfg(all(feature = "instrument", feature = "metal-output-placement"))]
#[test]
fn cached_attention_decode_split_times_a_binary16_k_cache_against_a_q4_0_k_cache() {
    const SAMPLES: usize = 25;
    let policy = production_numeric_policy();
    let (head_dim, cached_len) = (256, 1099);
    let fixture = gqa_decode_fixture(cached_len, head_dim);
    let output_roots = [fixture.roots[0]];
    let arms = [
        ("k=binary16 v=binary16", None),
        ("k=q4_0 v=binary16", Some(NumericFormat::Q4_0)),
        ("k=q8_0 v=binary16", Some(NumericFormat::Q8_0)),
    ];
    let leaves: Vec<_> = arms.iter().map(|(_, format)| timed_cache_leaves(&fixture, *format)).collect();
    let blocks: Vec<_> = leaves.iter().map(|arm_leaves| timed_blocks(&fixture, arm_leaves)).collect();
    let plans: Vec<_> = blocks
        .iter()
        .zip(&arms)
        .map(|(arm_blocks, (label, _))| {
            omega::plan_named(&fixture.program, &fixture.symbols, arm_blocks, &output_roots, policy)
                .unwrap_or_else(|error| panic!("metal plans the {label} decode program: {error:?}"))
        })
        .collect();

    let mut samples: Vec<Vec<u64>> = arms.iter().map(|_| Vec::new()).collect();
    for _ in 0..SAMPLES {
        for (arm, plan) in plans.iter().enumerate() {
            let (_, timings) = omega::execute_plan_named_with_placements_op_timed(plan, &blocks[arm], &[], &[])
                .expect("the plan times");
            let (nanos, ops) = attention_gpu_ns(&timings);
            assert_eq!(ops, LAYERS as usize, "one decode-split attention op per layer");
            samples[arm].push(nanos);
        }
    }
    for ((label, _), sample) in arms.iter().zip(&mut samples) {
        sample.sort_unstable();
        eprintln!(
            "decode_split attention gpu_ns head_dim={head_dim} cached_len={cached_len} layers={LAYERS} {label}: min={} median={} max={} samples={}",
            sample[0],
            sample[sample.len() / 2],
            sample[sample.len() - 1],
            sample.len()
        );
        assert!(sample[0] > 0, "the {label} arm must report attention time");
    }
    let plane_elements = head_dim as usize / 2;
    eprintln!(
        "k bytes per row per kv head: binary16={} q4_0={} q8_0={}",
        plane_elements * 2,
        NumericFormat::Q4_0.packed_len(plane_elements).expect("whole blocks"),
        NumericFormat::Q8_0.packed_len(plane_elements).expect("whole blocks"),
    );
}

#[test]
fn cached_attention_declines_a_block_v_cache_instead_of_reading_it_as_f32() {
    let policy = production_numeric_policy();
    let fixture = gqa_decode_fixture(199, 64);
    let output_roots = [fixture.roots[0]];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved = bind(&fixture.program, &shapes, &output_roots, policy).expect("the decode program binds");
    let attention = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the fixture contains a cached attention operation");
    let mut mixed_with_block_value = omega::PackedOperands::new();
    for index in [2, 3, 6] {
        mixed_with_block_value.insert(attention.operands()[index].0, NumericFormat::Q4_0);
    }

    let error = omega::emit(attention, &mixed_with_block_value, policy)
        .expect_err("a block V cache has no read path");

    assert!(matches!(error, omega::EmitError::CachedAttentionKvCodecNotSupported { .. }), "{error:?}");
}
