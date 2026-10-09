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
use proxima_tensor::cpu::{QuantizedBlock, evaluate_quantized_named_with_scratch};
use proxima_tensor::spec::qk_norm_cached_forward_program;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BoundOpKind, NodeId, Op, bind, infer};

mod support;
use support::{as_named_blocks, production_numeric_policy};

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
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
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

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &fixture.program,
        &fixture.symbols,
        &named,
        &output_roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the decode program");
    let metal_named: Vec<(&str, QuantizedBlock<'_>)> = named
        .iter()
        .map(|(name, block)| {
            match half_blocks.iter().find(|(half_name, _)| half_name == name) {
                Some((_, bytes)) => (
                    *name,
                    QuantizedBlock::Packed {
                        codec: omega::Codec::Float16,
                        bytes: bytes.as_slice(),
                    },
                ),
                None => (*name, *block),
            }
        })
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
    assert_eq!(actual.len(), expected.len(), "head_dim {head_dim} cached_len {cached_len}");
    assert!(
        !expected.is_empty(),
        "head_dim {head_dim} cached_len {cached_len}: an empty root compares nothing"
    );
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
            assert_eq!(*values, round_to_bf16(values), "fixture values are exact BF16 values");
            (name.clone(), bf16_bytes(values))
        })
        .collect();
    assert_eq!(bf16_blocks.len(), 3 * LAYERS as usize);

    let output_roots = [fixture.roots[0]];
    let f32_named = as_named_blocks(&fixture.named);
    let bf16_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, values)| match bf16_blocks.iter().find(|(cache_name, _)| cache_name == name) {
            Some((_, bytes)) => (
                name.as_str(),
                QuantizedBlock::Packed {
                    codec: omega::Codec::BFloat16,
                    bytes,
                },
            ),
            None => (
                name.as_str(),
                QuantizedBlock::Float32(values),
            ),
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

    let f32_output = omega::execute_plan_named(&f32_plan, &f32_named)
        .expect("F32 cache decode executes")
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let bf16_output = omega::execute_plan_named(&bf16_plan, &bf16_named)
        .expect("BF16 cache decode executes")
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bf16_output, f32_output, "the cache decode has the same f32 inputs and operations");
}

#[test]
fn card_06_bf16_decode_selects_source_and_declines_mixed_cache_codecs() {
    let policy = production_numeric_policy();
    let fixture = gqa_decode_fixture(31, 64);
    let output_roots = [fixture.roots[0]];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved = bind(&fixture.program, &shapes, &output_roots, policy)
        .expect("the decode program binds");
    let attention = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the fixture contains a cached attention operation");
    let bf16 = [2, 3, 6]
        .into_iter()
        .map(|index| (attention.operands()[index].0, omega::Codec::BFloat16))
        .collect::<omega::PackedOperands>();
    let emitted = omega::emit(attention, &bf16, policy).expect("BF16 decode source emits");
    assert!(emitted.entry.ends_with("_ds"), "selected source is decode split");
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
    assert!(matches!(error, omega::EmitError::CachedAttentionKvCodecNotSupported { .. }));
}
