//! Lowering census: for every named production matmul weight family in the
//! real openchat single-range decode/prefill program
//! (`proxima_tensor::spec::mistral_single_range_cached_forward_program`),
//! at the real `(in_dim, out_dim)` pair that weight is declared at
//! (`row_376_cached_attention_batched.rs`'s own decode dims: `QUERY_HEADS`
//! 32, `KV_HEADS` 8, `HEAD_DIM` 128, `EMBEDDING = QUERY_HEADS * HEAD_DIM` =
//! 4096, `FEED_FORWARD` 256, `VOCAB` 64), builds a standalone
//! `Add`-reduce-over-`Multiply` matmul `BoundOp` (the same construction
//! `omega/src/msl.rs`'s own `matmul_op`/`matmul_op_with_reduce` unit-test
//! fixtures use) at token count M in `{1, 8, 31, 256}`, emits it through the
//! real [`omega::msl::emit`] path (CPU-side, no device), and asserts the
//! emitted source carries the branch marker the (codec, M-class) cell in
//! `lowering-coverage.md`'s section 2 table says it must: the
//! packed-row-blocked `plain_product` pair-dot body at M=1, the multi-row
//! body's `feature_first` marker at M>1. A shape sliding from one to the
//! other fails here, not months later in a per-token op-profile
//! investigation.
//!
//! A standalone matmul per weight family, not the full fused graph: binding
//! the real `mistral_single_range_cached_forward_program` directly hits two
//! structurally different lowerings this census does not model yet
//! (`attn_k`/`attn_v` fuse into the KV-cache write's fold; `attn_output`
//! resolves to a bare `Elementwise`, found empirically via `diagnose_packed_row_block`
//! and `BoundOpKind::name()` against the real bound program). This file
//! instead pins the one variable the (codec, M-class) table is actually
//! about -- token count M -- on a standalone matmul per weight family, at
//! that family's real `(in_dim, out_dim)`.
//!
//! Codec assignment mirrors ROW 389's own per-op family split of the real
//! openchat checkpoint (`proxima-tensor/docs/discipline.md:27137`): `Q4_K`
//! for every matmul weight by default, `Q5_K` for `ffn_down` (the family
//! ROW 389 itself calls out as partially `Q5_K`), `Q6_K` for `output.weight`
//! alone -- this fixture has no GGUF to load codecs from, so the assignment
//! is asserted structurally instead of read from a checkpoint, same as
//! every other synthetic-shape fixture in this directory.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use omega::msl::{PackedCodec, PackedOperands, emit};
use proxima_tensor::{
    BoundOp, DType, Extent, IndexMap, Keep, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp,
    append, bind, infer, map,
};

// REAL openchat decode dims (`row_376_cached_attention_batched.rs`'s own
// constants).
const QUERY_HEADS: u32 = 32;
const HEAD_DIM: u32 = 128;
const EMBEDDING: u32 = QUERY_HEADS * HEAD_DIM;
const FEED_FORWARD: u32 = 256;
const VOCAB: u32 = 64;

/// One named production matmul weight family: its real `(in_dim, out_dim)`
/// pair and the codec ROW 389's own per-op family split assigns it.
struct WeightFamily {
    name: &'static str,
    in_dim: u32,
    out_dim: u32,
    codec: PackedCodec,
}

const WEIGHT_FAMILIES: &[WeightFamily] = &[
    WeightFamily {
        name: "attn_q.weight",
        in_dim: EMBEDDING,
        out_dim: EMBEDDING,
        codec: PackedCodec::Q4K,
    },
    WeightFamily {
        name: "ffn_gate.weight",
        in_dim: EMBEDDING,
        out_dim: FEED_FORWARD,
        codec: PackedCodec::Q4K,
    },
    WeightFamily {
        name: "ffn_up.weight",
        in_dim: EMBEDDING,
        out_dim: FEED_FORWARD,
        codec: PackedCodec::Q4K,
    },
    WeightFamily {
        name: "ffn_down.weight",
        in_dim: FEED_FORWARD,
        out_dim: EMBEDDING,
        codec: PackedCodec::Q5K,
    },
    WeightFamily {
        name: "output.weight",
        in_dim: EMBEDDING,
        out_dim: VOCAB,
        codec: PackedCodec::Q6K,
    },
];

/// `metal`'s own feature list turns `metal-q4k-ggml-port` ON by default
/// (`Cargo.toml`'s `metal = [.., "metal-q4k-ggml-port"]`, found empirically
/// via this same census -- see `PACKED_ROW_BODY_MARKERS`'s own doc), and
/// `use_ggml_port` takes PRIORITY over the named pair-dot arm for every
/// codec it covers (`Q4_K`/`Q5_K`/`Q6_K`) whenever `plain_product` holds --
/// which every M=1 cell here does. Only `Q3_K` (not one of the three the
/// ggml port covers) still emits its own `q3k_pair_dot(blk`; `WEIGHT_FAMILIES`
/// never assigns it, so this only needs the two ggml-port markers.
fn codec_marker(codec: PackedCodec) -> &'static str {
    match codec {
        PackedCodec::Q3K => "q3k_pair_dot(blk",
        PackedCodec::Q4K | PackedCodec::Q5K => "acc1_0",
        PackedCodec::Q6K => "sums0",
        PackedCodec::Q2K
        | PackedCodec::Q8_0
        | PackedCodec::Q4_0
        | PackedCodec::Float16
        | PackedCodec::BFloat16 => {
            unreachable!("WEIGHT_FAMILIES never assigns a non-K-quant codec")
        }
    }
}

/// The multi-row body's own `feature_first` local -- unique to
/// `push_packed_row_multi_row_body`'s emitted text (verified: the token
/// never appears anywhere else `omega/src/msl.rs` renders), so its presence
/// alone identifies the `'M'` branch the way a pair-dot marker identifies
/// `'B'`.
const MULTI_ROW_MARKER: &str = "feature_first";

/// `m` activation rows (token count) x `k`-wide `Add`-reduce over a plain
/// `weight * activation` body, `n`-wide output -- unlike `omega/src/msl.rs`'s
/// own `matmul_op` (whose declared output-axis order puts the WEIGHT's own
/// axis first, `[m, n]` with weight varying along `m`), this declares
/// output axis order `[token, feature]` = `[m, n]` with the WEIGHT
/// independent of `m` (shape `[n, k]`, no `m` dimension at all) and the
/// activation independent of `n` (shape `[m, k]`) -- the one axis order
/// [`split_token_feature_axes`] (`omega/src/msl.rs`) requires to ever
/// classify a token axis at all (found empirically: `msl.rs`'s own
/// `matmul_op` convention puts the weight-varying axis FIRST, which fails
/// that function's own `reassembled == output_axes` check and silently
/// defaults every one of its own row-blocked tests to the single-row path
/// regardless of `m`/`n` -- true for every codec at every `(m, n)` those
/// tests use, never previously exercising `packed_row_block_token_total > 1`
/// at all). Returns the weight's own `NodeId` alongside the bound op since
/// operand order after fusion is not guaranteed to match program order.
fn matmul_op(m: u32, k: u32, n: u32) -> (BoundOp, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    let bound = bind(&program, &shapes, &[], NumericPolicy::default())
        .expect("matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound op emitted");
    (bound, weight)
}

/// Emits every weight family's real-shape matmul at token count `m` and
/// asserts its (codec, M-class) branch marker, printing the full census
/// (family name, codec, expected vs found marker) on the first failure so a
/// regression names itself instead of needing a follow-up per-op
/// investigation.
fn assert_census_cell(m: u32) {
    let mut census = String::new();
    let mut failures = Vec::new();
    for family in WEIGHT_FAMILIES {
        let (bound, weight_node) = matmul_op(m, family.in_dim, family.out_dim);
        let mut packed_operands: PackedOperands = BTreeMap::new();
        packed_operands.insert(weight_node, family.codec);

        let kernel = emit(&bound, &packed_operands, NumericPolicy::default())
            .unwrap_or_else(|error| panic!("{}: emit failed: {error:?}", family.name));
        let expected_marker = if m == 1 {
            codec_marker(family.codec)
        } else {
            MULTI_ROW_MARKER
        };
        let found = kernel.source.contains(expected_marker);
        census.push_str(&format!(
            "  {name} codec={codec:?} expected_marker={expected_marker:?} found={found}\n",
            name = family.name,
            codec = family.codec,
        ));
        if !found {
            failures.push(family.name);
        }
    }

    assert!(
        failures.is_empty(),
        "M={m}: {failures:?} did not carry their expected branch marker\ncensus:\n{census}"
    );
}

#[test]
fn decode_m1_every_weight_family_takes_the_packed_row_blocked_pair_dot_body() {
    assert_census_cell(1);
}

#[test]
fn prefill_m8_every_weight_family_takes_the_multi_row_body() {
    assert_census_cell(8);
}

#[test]
fn prefill_m31_every_weight_family_takes_the_multi_row_body() {
    assert_census_cell(31);
}

#[test]
fn long_prompt_m256_every_weight_family_takes_the_multi_row_body() {
    assert_census_cell(256);
}
