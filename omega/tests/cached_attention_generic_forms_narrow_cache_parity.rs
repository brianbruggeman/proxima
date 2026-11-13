//! A narrow (binary16 / bfloat16) K/V cache read by the three generic cached attention forms
//! (`CachedAttentionForm::Static`, `SingleRangeDynamic`, `TwoRangeCachedBound`) at head dims the
//! decode split cannot take (it needs a multiple of 32). The packed cache is read through the one
//! rendered scalar load (`omega/src/msl/layout_render.rs` `scalar_cache_read`), so Metal over the
//! packed bytes must equal Metal over the same host-decoded values as plain f32 bit for bit.
//! Each cell asserts the entry is not a decode split (`_ds`), so a pass cannot be the split form.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::mem::MaybeUninit;

use proxima_numeric::{FormatCodec, NumericFormat};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::{
    DuplicateHeadPosition, gqa_cached_forward_program, gqa_single_range_cached_forward_program,
    qk_norm_cached_forward_program,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BoundOpKind, NodeId, Op, bind, infer};

mod support;
use support::production_numeric_policy;

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 64;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;
const LAYERS: u32 = 2;

#[derive(Clone, Copy, Debug)]
enum Builder {
    Plain,
    QkNorm,
    SingleRange,
}

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

fn fixture(builder: Builder, cached_len: u64, head_dim: u32) -> Fixture {
    let (program, logits, cache_roots, symbols) = match builder {
        Builder::Plain => {
            let (program, logits, caches) =
                gqa_cached_forward_program(VOCAB, EMBEDDING, FEED_FORWARD, QUERY_HEADS, KV_HEADS, head_dim, LAYERS)
                    .expect("the plain cached program builds");
            (program, logits, caches, vec![1, cached_len])
        }
        Builder::QkNorm => {
            let (program, logits, caches) =
                qk_norm_cached_forward_program(VOCAB, EMBEDDING, FEED_FORWARD, QUERY_HEADS, KV_HEADS, head_dim, LAYERS)
                    .expect("the qk-norm cached program builds");
            (program, logits, caches, vec![1, cached_len])
        }
        Builder::SingleRange => {
            let (program, logits, caches, _) = gqa_single_range_cached_forward_program(
                VOCAB,
                EMBEDDING,
                FEED_FORWARD,
                QUERY_HEADS,
                KV_HEADS,
                head_dim,
                LAYERS,
                false,
                DuplicateHeadPosition::None,
                false,
            )
            .expect("the single-range program builds");
            (program, logits, caches, vec![1, cached_len + 1])
        }
    };
    let mut roots = vec![logits];
    for (even, odd, value) in &cache_roots {
        roots.extend([*even, *odd, *value]);
    }
    let shapes = infer(&program, &symbols).expect("the program infers");
    let mut named = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let count: usize = shapes.of(NodeId(position as u32)).iter().map(|extent| *extent as usize).product();
        let name = name.clone().expect("every block input in this program is named");
        let data = match name.as_str() {
            "ids" => vec![3.0f32; count],
            "eps" => vec![1e-5f32; count],
            "cached_len" => vec![cached_len as f32],
            _ => random_vec(position as u64 + 1, count),
        };
        named.push((name, data));
    }
    Fixture { program, symbols, roots, named }
}

fn is_kv_cache(name: &str) -> bool {
    name.starts_with("kv_cache.")
}

/// `values` packed by the host codec and decoded back by it.
fn round_trip(format: &NumericFormat, values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let codec = FormatCodec::for_format(format).expect("the format resolves a host codec");
    let mut packed = vec![0u8; codec.packed_len(values.len()).expect("a whole number of codes")];
    codec.encode(values, &mut packed).expect("the host encodes the cache rows");
    let mut slots = vec![MaybeUninit::<f32>::uninit(); values.len()];
    codec.decode(&packed, &mut slots).expect("the host decodes the cache rows");
    // decode returned Ok, which initialises every slot of its output
    let decoded = slots.iter().map(|slot| unsafe { slot.assume_init() }).collect();
    (packed, decoded)
}

fn assert_all_finite(values: &[f32], what: &str) {
    let non_finite = values.iter().filter(|value| !value.is_finite()).count();
    assert_eq!(non_finite, 0, "{non_finite}/{} {what} values are not finite", values.len());
}

fn max_diff(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    assert!(!expected.is_empty(), "an empty root compares nothing");
    assert_all_finite(expected, "plain-cache");
    assert_all_finite(actual, "packed-cache");
    expected.iter().zip(actual).fold(0.0f32, |worst, (want, got)| worst.max((want - got).abs()))
}

/// Metal over the packed cache against Metal over its host-decoded plain f32 twin; returns the max
/// absolute difference and the entry name of every cached attention kernel the cell emitted.
fn cell(builder: Builder, format: NumericFormat, cached_len: u64, head_dim: u32) -> (f32, Vec<String>) {
    let policy = production_numeric_policy();
    let fixture = fixture(builder, cached_len, head_dim);
    let output_roots = [fixture.roots[0]];

    let mut decoded_named = fixture.named.clone();
    let mut packed_named: Vec<(String, Vec<u8>)> = Vec::new();
    for (name, data) in decoded_named.iter_mut().filter(|(name, _)| is_kv_cache(name)) {
        let (packed, decoded) = round_trip(&format, data);
        *data = decoded;
        packed_named.push((name.clone(), packed));
    }
    assert_eq!(packed_named.len(), 3 * LAYERS as usize, "a K even, K odd and V leaf per layer");

    let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
    let resolved = bind(&fixture.program, &shapes, &output_roots, policy).expect("the program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert!(!attention.is_empty(), "zero CachedAttention ops would compare nothing");
    let mut entries = Vec::new();
    for bound in &attention {
        let mut packed = omega::PackedOperands::new();
        for index in [2, 3, 6] {
            packed.insert(bound.operands()[index].0, format);
        }
        let kernel = omega::emit(bound, &packed, policy).expect("a narrow scalar cache emits in every form");
        assert!(!kernel.entry.ends_with("_ds"), "the cell must not pass through the decode split: {}", kernel.entry);
        assert!(kernel.source.contains("omega_kv_scalar_"), "the narrow cache is read through the rendered load");
        entries.push(kernel.entry.clone());
    }

    let plain_blocks: Vec<(&str, QuantizedBlock<'_>)> = decoded_named
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
        .collect();
    let narrow_blocks: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, data)| match packed_named.iter().find(|(leaf, _)| leaf == name) {
            Some((_, bytes)) => (name.as_str(), QuantizedBlock::Packed { codec: format, bytes }),
            None => (name.as_str(), QuantizedBlock::Float32(data)),
        })
        .collect();
    let plain_plan = omega::plan_named(&fixture.program, &fixture.symbols, &plain_blocks, &output_roots, policy)
        .expect("metal plans the decoded plain f32 program");
    let plain_run = omega::execute_plan_named(&plain_plan, &plain_blocks).expect("metal runs the plain program");
    let narrow_plan = omega::plan_named(&fixture.program, &fixture.symbols, &narrow_blocks, &output_roots, policy)
        .expect("metal plans the narrow-cache program");
    let narrow_run = omega::execute_plan_named(&narrow_plan, &narrow_blocks).expect("metal runs the narrow-cache program");

    let diff = max_diff(plain_run.root(), narrow_run.root());
    eprintln!("generic narrow parity: {builder:?} {format:?} head_dim={head_dim} cached_len={cached_len} max_diff={diff} entries={entries:?}");
    (diff, entries)
}

#[test]
fn cached_attention_generic_forms_read_a_whole_byte_scalar_cache_bit_exactly() {
    let formats = [NumericFormat::Float16, NumericFormat::BFloat16];
    let mut cells = 0_usize;
    let mut forms = BTreeSet::new();
    for builder in [Builder::Plain, Builder::QkNorm] {
        for format in formats {
            for head_dim in [2_u32, 8, 16] {
                for cached_len in [0_u64, 3, 33] {
                    let (diff, entries) = cell(builder, format, cached_len, head_dim);
                    assert_eq!(
                        diff, 0.0,
                        "{builder:?} {format:?} head_dim {head_dim} cached_len {cached_len}: the packed read differs from the host-decoded plain read"
                    );
                    forms.extend(entries.iter().map(|entry| entry.ends_with("_cb")));
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 2 * 2 * 3 * 3, "2 builders x 2 formats x 3 head dims x 3 cached lengths");
    assert_eq!(forms.len(), 2, "the cells must reach the static form (no _cb) and the bound two-range form (_cb), saw {forms:?}");
}

/// The single-range form binds one merged cache node as both the cached and the new operands
/// (`in2 == in4`, `in3 == in5`, `in6 == in7`), so packing that node packs the new-row operands
/// too, which the narrow read does not cover: the emitter refuses rather than reading codes as
/// floats.
#[test]
fn a_single_range_cache_shared_with_the_new_rows_is_refused_when_packed() {
    let policy = production_numeric_policy();
    let fixture = fixture(Builder::SingleRange, 3, 8);
    let output_roots = [fixture.roots[0]];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
    let resolved = bind(&fixture.program, &shapes, &output_roots, policy).expect("the program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert!(!attention.is_empty(), "zero CachedAttention ops would compare nothing");
    for bound in attention {
        let mut packed = omega::PackedOperands::new();
        for index in [2, 3, 6] {
            packed.insert(bound.operands()[index].0, NumericFormat::Float16);
        }
        let error = omega::emit(bound, &packed, policy).expect_err("a packed node also bound as the new rows is refused");
        assert!(matches!(error, omega::EmitError::CachedAttentionKvCodecNotSupported { .. }), "{error:?}");
    }
}
