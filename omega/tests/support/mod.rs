//! The real cached-forward graph fixture, shared by `metal_real_forward.rs`
//! (CPU vs the raw Metal driver), `backend_parity.rs` (CPU vs Metal through
//! `omega::backend`'s wrapper), `wgpu_parity.rs`, and
//! `cached_attention_coop_load_parity.rs` (the single-range fused kind) —
//! lifted out of the first so the SAME program, roots and named block data
//! feed every gate rather than a copy per binary that can drift on which
//! named block gets which random seed.

// fixture construction is hand-built to succeed; an expect failure here IS
// the fixture being broken, same convention as every other `omega/tests/*.rs` file.
#![allow(clippy::unwrap_used, clippy::expect_used)]
// each `omega/tests/*.rs` file is its own separate binary crate that `mod
// support`s this file wholesale, and no single binary calls every builder
// here (e.g. `cached_attention_coop_load_parity.rs` only ever needs the
// single-range builder, never the two-range one `metal_real_forward.rs`
// exercises) -- `-D dead-code` is workspace-wide (`Cargo.toml`'s `[lints]`),
// so whichever builder a given binary does not reach reads as genuinely
// dead FROM THAT BINARY's isolated compilation, even though a sibling
// binary calls it. A per-binary `#[cfg(test)]`-style split would only
// re-create the very duplication this shared module exists to avoid.
#![allow(dead_code)]

use proxima_tensor::spec::{
    DuplicateHeadPosition, mistral_cached_forward_program,
    mistral_single_range_cached_forward_program, qwen3_cached_forward_program,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, NumericPolicy, Op, QuantizedBlock, block_node_ids, infer};

/// The numeric policy production actually runs under --
/// `proxima-model-interop/src/serving.rs:255`'s `ServingConfig::default()`
/// sets `numeric_policy: NumericPolicy::llama_relaxed()`, which resolves to
/// `MTLMathMode::Relaxed` (`omega::metal`'s `numeric_policy_as_metal_math_mode`).
/// Every bare timing/cost harness in `omega/tests` that reports a number
/// meant to be comparable to production (or to another arm that already
/// runs under production's policy) must bind/plan/emit under THIS value,
/// never `NumericPolicy::default()` (bit-exact `Safe`) -- ROW 374 added the
/// `NumericPolicy` parameter and every bare arm silently kept `default()`,
/// so every post-374 per-shape number was measured at the wrong math mode.
#[must_use]
pub fn production_numeric_policy() -> NumericPolicy {
    NumericPolicy::llama_relaxed()
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// [`real_forward_fixture`]'s return shape: the program, its decode-step
/// symbols, every root a caller should request as output, and deterministic
/// `(name, data)` pairs for every named block input.
pub type RealForwardFixture = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// A small (2-layer, 64-wide) instance of the real cached-forward graph:
/// production op set and graph shape, scaled down only so it runs in a
/// test. Returns the program, the decode-step symbols (one step, no cached
/// history), every root a caller should request as output (the logits, plus
/// every layer's KV-cache write), and deterministic f32 data for every named
/// block input -- sized from the graph's own inferred shapes, so this needs
/// no checkpoint on disk.
pub fn real_forward_fixture() -> RealForwardFixture {
    real_forward_fixture_with_cached_len(0)
}

/// [`real_forward_fixture`], but with symbol 1 (`cached_len`) set to
/// `cached_len` instead of always zero -- exercises the online-softmax
/// combine's cached-block reduce over a genuinely non-empty range, the one
/// shape a single-step (`cached_len == 0`) fixture can never reach since
/// every fold over the `t` axis degenerates to its `ReduceInit` identity
/// there. The KV-cache named blocks are sized from the same inferred
/// shapes, so a non-zero `cached_len` needs no checkpoint either.
pub fn real_forward_fixture_with_cached_len(cached_len: u64) -> RealForwardFixture {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 64;
    const FEED_FORWARD: u32 = 128;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 16;
    const LAYERS: u32 = 2;

    let (program, logits_root, cache_roots) = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the real forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![1u64, cached_len];
    let shapes = infer(&program, &symbols).expect("the real forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        // an empty block is legitimate here: a KV-cache input is genuinely
        // zero-length at `cached_len == 0`, and padding it to one element is
        // an invented value the shape check correctly rejects.
        let data = if name == "ids" {
            // a token id, not a weight: must be an in-range integer
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            // the REAL cached length, not a random fill -- `bind::
            // cached_attention_candidates` reads this named leaf as the
            // fused op's own ninth, runtime `cached_key_rows` operand
            // (`BoundOpKind::CachedAttention`'s own doc), so a random value
            // here would corrupt only the GPU-fused path's mask/bound while
            // the CPU evaluator (which folds directly from the program's
            // own shapes, never this scalar) stayed correct -- exactly the
            // silent CPU/Metal divergence `fused_cached_attention_root_
            // agrees_between_cpu_and_metal` exists to catch.
            vec![cached_len as f32]
        } else {
            random_vec(position as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}

/// [`real_forward_fixture_with_cached_len`]'s GQA-plus-QK-norm counterpart:
/// the real Qwen3-1.7B shape's two distinguishing features (`query_heads !=
/// kv_heads`, already present in the Mistral fixture; split-half RoPE plus
/// per-head `q_norm`/`k_norm`, which that fixture does NOT exercise) --
/// built from [`qwen3_cached_forward_program`] instead of
/// [`mistral_cached_forward_program`], same 2-layer/64-wide/GQA=2 shape.
/// `new_count` (symbol 0, hardcoded to `1` in the Mistral fixture) is a
/// caller parameter here so a query_rows > 1 (multi-row prefill/resume)
/// step can be reproduced against the same cache, not only single-token
/// decode.
pub fn qwen3_gqa_qk_norm_forward_fixture(new_count: u64, cached_len: u64) -> RealForwardFixture {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 64;
    const FEED_FORWARD: u32 = 128;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 16;
    const LAYERS: u32 = 2;

    let (program, logits_root, cache_roots) = qwen3_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the qwen3 gqa+qk_norm forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![new_count, cached_len];
    let shapes = infer(&program, &symbols).expect("the qwen3 gqa+qk_norm forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let data = if name == "ids" {
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![cached_len as f32]
        } else {
            random_vec(position as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}

/// The single-range counterpart of [`real_forward_fixture_with_cached_len`]:
/// same real op set and shape family (2-layer, 64-wide GQA), built from
/// [`mistral_single_range_cached_forward_program`] instead, whose
/// `causal_mask_merged` band `bind`'s `cached_attention_single_range_
/// candidates` pattern-matches into the NINE-operand dynamic-`cached_len`
/// [`proxima_tensor::BoundOpKind::CachedAttention`] (`omega/src/msl.rs`'s
/// `render_cached_attention` doc) rather than the eight-operand two-range
/// kind [`real_forward_fixture_with_cached_len`] exercises.
///
/// `padding` reproduces `kv-capacity-bucket`'s KV-extent rounding directly:
/// every `kv_cache.*` named block gets `cached_len + new_count` real random
/// rows followed by `padding` zero-filled rows (`sequence = cached_len +
/// new_count + padding` total, fed as symbol 1 — the buffer's own shape),
/// while the `cached_len` named scalar input carries the true band bound
/// unchanged — exactly the shape a real KV-capacity bucket one step ahead of
/// the true cache length looks like at runtime, and the divergence this is
/// built to catch: a kernel that mis-sizes its cooperative-load stride
/// against the padded shape instead of the real band would read past
/// `merged_len` into the zero-filled tail.
pub fn real_single_range_forward_fixture_with_padding(
    cached_len: u64,
    new_count: u64,
    padding: u64,
) -> RealForwardFixture {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 64;
    const FEED_FORWARD: u32 = 128;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 16;
    const LAYERS: u32 = 2;

    let (program, logits_root, cache_roots, _) = mistral_single_range_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
        false,
        DuplicateHeadPosition::None,
        false,
    )
    .expect("the single-range forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let merged_len = cached_len + new_count;
    let sequence = merged_len + padding;
    let symbols = vec![new_count, sequence];
    let shapes = infer(&program, &symbols).expect("the single-range forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for node in block_node_ids(&program) {
        let Op::Input { name, .. } = &program[node.0 as usize] else {
            unreachable!("block_node_ids only ever returns Op::Input nodes")
        };
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let data = if name == "ids" {
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![cached_len as f32]
        } else if name.starts_with("kv_cache.") {
            let per_row = count / sequence as usize;
            let mut data = random_vec(node.0 as u64 + 1, merged_len as usize * per_row);
            data.resize(count, 0.0);
            data
        } else {
            random_vec(node.0 as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}

/// Borrows [`real_forward_fixture`]'s owned `(name, data)` pairs into the
/// `&[(&str, QuantizedBlock<'_>)]` shape both evaluators bind against.
pub fn as_named_blocks(owned: &[(String, Vec<f32>)]) -> Vec<(&str, QuantizedBlock<'_>)> {
    owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect()
}

/// qwen35's real partial-rotary dense-attention chain
/// (`proxima_tensor::spec::append_qwen35_dense_attention_only_with_taps`,
/// `kv_heads` 2, `group` 8 -> 16 query heads, `attn_head_dim` 256,
/// `rotary_dim` 64 -> 192-wide pass plane), ported from
/// `proxima_tensor::bind`'s own private `qwen35_partial_rotary_attention_
/// fixture` test fixture into this crate's named-block shape so a Metal run
/// (`omega::plan_named`/`execute_plan_named`) can be compared against the
/// CPU reference the same way [`real_single_range_forward_fixture_with_padding`]
/// already does for the full-rotary case. `cached_extent` is the compiled
/// KV-capacity-bucket width; `cached_len` is the REAL runtime length fed as
/// the fused op's own ninth operand, always `<= cached_extent`.
#[cfg(feature = "cached-attention-streaming")]
#[allow(clippy::too_many_lines)]
pub fn qwen35_partial_rotary_forward_fixture(
    new_count: u64,
    cached_extent: u64,
    cached_len: u64,
) -> RealForwardFixture {
    use proxima_tensor::spec::{
        append_qwen35_dense_attention_only_with_taps, causal_mask, input_leaf, scalar_constant,
    };
    use proxima_tensor::{DType, Extent};

    const KV_HEADS: u32 = 2;
    const GROUP: u32 = 8;
    const ATTN_HEAD_DIM: u32 = 256;
    const ROTARY_DIM: u32 = 64;
    const PASS_DIM: u32 = ATTN_HEAD_DIM - ROTARY_DIM;
    const PAIR_DIM: u32 = ROTARY_DIM / 2;
    // `proxima_tensor::bind`'s own private fixture uses embedding width 1 --
    // valid on CPU, but `attn_norm`'s RMSNorm reduce then folds over a
    // length of 1, below `omega::sized::COOPERATIVE_REDUCE_MIN_LEN`, and a
    // non-cooperative reduce has no Metal renderer for a broadcast epilogue
    // (`msl.rs`'s own `EpilogueNotSupported` gate) -- unrelated to the
    // partial-rotary plane this file exists to test, so a wider embedding
    // (still degenerate relative to a real model, but past the cooperative
    // threshold) sidesteps it without touching that renderer.
    const EMBEDDING: u32 = 64;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
        "x",
    );
    let inv_dim = scalar_constant(&mut program, 1.0 / EMBEDDING as f32);
    let eps = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (ATTN_HEAD_DIM as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / ATTN_HEAD_DIM as f32);
    let rotary_shape = vec![Extent::Symbolic(0), Extent::Static(PAIR_DIM)];
    let cos_new = input_leaf(&mut program, DType::Float32, rotary_shape.clone(), "cos");
    let sin_new = input_leaf(&mut program, DType::Float32, rotary_shape, "sin");
    let group_ones = proxima_tensor::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
    let cached_len_node = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING)],
        "attn_norm_weight",
    );
    let norm_shape = vec![Extent::Static(ATTN_HEAD_DIM)];
    let q_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape.clone(), "q_norm_weight");
    let k_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape, "k_norm_weight");
    let wq_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EMBEDDING),
            Extent::Static(KV_HEADS * GROUP),
            Extent::Static(2 * ATTN_HEAD_DIM),
        ],
        "wq_gate",
    );
    let wk_wv_shape = vec![
        Extent::Static(EMBEDDING),
        Extent::Static(KV_HEADS),
        Extent::Static(ATTN_HEAD_DIM),
    ];
    let wk = input_leaf(&mut program, DType::Float32, wk_wv_shape.clone(), "wk");
    let wv = input_leaf(&mut program, DType::Float32, wk_wv_shape, "wv");
    let wo = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(KV_HEADS),
            Extent::Static(GROUP),
            Extent::Static(ATTN_HEAD_DIM),
            Extent::Static(EMBEDDING),
        ],
        "wo",
    );
    let cache_rotary_shape = vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS),
        Extent::Static(PAIR_DIM),
    ];
    let cache_pass_shape = vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS),
        Extent::Static(PASS_DIM),
    ];
    let cache_v_shape = vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS),
        Extent::Static(ATTN_HEAD_DIM),
    ];
    let k_first_cache = input_leaf(&mut program, DType::Float32, cache_rotary_shape.clone(), "k_first_cache");
    let k_second_cache = input_leaf(&mut program, DType::Float32, cache_rotary_shape, "k_second_cache");
    let k_pass_cache = input_leaf(&mut program, DType::Float32, cache_pass_shape, "k_pass_cache");
    let v_cache = input_leaf(&mut program, DType::Float32, cache_v_shape, "v_cache");

    let (residual1, taps) = append_qwen35_dense_attention_only_with_taps(
        &mut program,
        x,
        inv_dim,
        eps,
        ones,
        inv_sqrt_attn_head_dim,
        inv_attn_head_dim,
        cos_new,
        sin_new,
        group_ones,
        is_future,
        cached_len_node,
        GROUP,
        ROTARY_DIM,
        ATTN_HEAD_DIM,
        attn_norm_weight,
        q_norm_weight,
        k_norm_weight,
        wq_gate,
        wk,
        wv,
        wo,
        k_first_cache,
        k_second_cache,
        k_pass_cache,
        v_cache,
    )
    .expect("qwen35 partial-rotary dense attention fixture lowers");

    let symbols = vec![new_count, cached_extent];
    let shapes = infer(&program, &symbols).expect("qwen35 partial-rotary fixture infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes.of(node).iter().map(|extent| *extent as usize).product();
        let name = name.clone().expect("every input in this fixture is named");
        let data = if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![cached_len as f32]
        } else {
            random_vec(position as u64 + 1, count.max(1))
        };
        named.push((name, data));
    }

    // `taps.attended` (the fusion's own anchor node) has exactly one reader
    // (the per-head gate multiply), and the three cache-write taps have
    // exactly one reader each too -- every one of them must be pinned as its
    // own materialization boundary, the same way a real KV-cache write root
    // already is, or the matcher never gets a standalone `BoundOp` to fuse
    // (`proxima_tensor::bind`'s own `qwen35_partial_rotary_attention_
    // fixture` doc, ported verbatim).
    let roots = vec![
        residual1,
        taps.attended,
        taps.rotated_k_new_first,
        taps.rotated_k_new_second,
        taps.k_pass,
        taps.v_new,
    ];
    (program, symbols, roots, named)
}
