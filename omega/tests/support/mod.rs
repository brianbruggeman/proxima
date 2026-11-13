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
    DuplicateHeadPosition, gqa_cached_forward_program, gqa_single_range_cached_forward_program,
    qk_norm_cached_forward_program,
};
use proxima_tensor::config::TensorExecutionConfig;
use proxima_tensor::cpu::{BufferPools, Evaluated, evaluate_quantized_named_with_config};
use proxima_tensor::test_support::Lcg;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

use proxima_primitives::pipe::Pipe;
use proxima_tensor::bind::ReadyBatch;
use proxima_tensor::{BoundOp, BoundOpKind, CachedAttentionArithmetic, Interpreter, NodeId, Op, QuantizedBlock, block_node_ids, infer};
use proxima_numeric::{NumericPolicy};

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

    let (program, logits_root, cache_roots) = gqa_cached_forward_program(
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
/// built from [`qk_norm_cached_forward_program`] instead of
/// [`gqa_cached_forward_program`], same 2-layer/64-wide/GQA=2 shape.
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

    let (program, logits_root, cache_roots) = qk_norm_cached_forward_program(
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
/// [`gqa_single_range_cached_forward_program`] instead, whose
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

    let (program, logits_root, cache_roots, _) = gqa_single_range_cached_forward_program(
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

/// The CPU oracle for a GPU parity gate: the strict `NumericPolicy::bit_exact()`
/// reference with flash narrowing granted and nothing else.
/// Without the grant the CPU binds `CachedAttentionArithmetic::LlamaFlash`
/// (llama.cpp's f16 query, key, value and accumulator); the GPU kernels run
/// `OnlineF32`, so the two would differ by f16 rounding in the fused attention
/// op and the gate would measure that, not the kernel. The oracle takes no
/// caller policy: contraction and reassociation are permissions the GPU plan
/// exercises, so granting them to the reference would hide the very
/// difference the gate exists to bound.
pub fn evaluate_gpu_oracle(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Evaluated {
    let config = TensorExecutionConfig::builder()
        .numeric(gpu_oracle_numeric_policy())
        .build();
    let mut free_buffers = BufferPools::default();
    let mut validated = None;
    evaluate_quantized_named_with_config(
        program,
        symbols,
        named,
        outputs,
        (&mut free_buffers, &mut validated),
        None,
        &config,
    )
    .expect("the cpu oracle evaluates the program")
}

/// qwen35's real partial-rotary dense-attention chain
/// (`proxima_tensor::spec::append_gated_attention_only_with_taps`,
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
        append_gated_attention_only_with_taps, causal_mask, input_leaf, count_constant, scalar_constant,
    };
    use proxima_tensor::{Extent};
    use proxima_numeric::{DType};

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
        DType::FLOAT32,
        vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
        "x",
    );
    let inv_dim = count_constant(&mut program, EMBEDDING);
    let eps = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Symbolic(0)],
        "eps",
    );
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (ATTN_HEAD_DIM as f32).sqrt());
    let inv_attn_head_dim = count_constant(&mut program, ATTN_HEAD_DIM);
    let rotary_shape = vec![Extent::Symbolic(0), Extent::Static(PAIR_DIM)];
    let cos_new = input_leaf(&mut program, DType::FLOAT32, rotary_shape.clone(), "cos");
    let sin_new = input_leaf(&mut program, DType::FLOAT32, rotary_shape, "sin");
    let group_ones = proxima_tensor::append(
        &mut program,
        Op::Constant {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
    let cached_len_node = input_leaf(&mut program, DType::FLOAT32, Vec::new(), "cached_len");

    let attn_norm_weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(EMBEDDING)],
        "attn_norm_weight",
    );
    let norm_shape = vec![Extent::Static(ATTN_HEAD_DIM)];
    let q_norm_weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        norm_shape.clone(),
        "q_norm_weight",
    );
    let k_norm_weight = input_leaf(&mut program, DType::FLOAT32, norm_shape, "k_norm_weight");
    let wq_gate = input_leaf(
        &mut program,
        DType::FLOAT32,
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
    let wk = input_leaf(&mut program, DType::FLOAT32, wk_wv_shape.clone(), "wk");
    let wv = input_leaf(&mut program, DType::FLOAT32, wk_wv_shape, "wv");
    let wo = input_leaf(
        &mut program,
        DType::FLOAT32,
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
    let k_first_cache = input_leaf(
        &mut program,
        DType::FLOAT32,
        cache_rotary_shape.clone(),
        "k_first_cache",
    );
    let k_second_cache = input_leaf(
        &mut program,
        DType::FLOAT32,
        cache_rotary_shape,
        "k_second_cache",
    );
    let k_pass_cache = input_leaf(
        &mut program,
        DType::FLOAT32,
        cache_pass_shape,
        "k_pass_cache",
    );
    let v_cache = input_leaf(&mut program, DType::FLOAT32, cache_v_shape, "v_cache");

    let (residual1, taps) = append_gated_attention_only_with_taps(
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
        None,
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
        None,
    )
    .expect("qwen35 partial-rotary dense attention fixture lowers");

    let symbols = vec![new_count, cached_extent];
    let shapes = infer(&program, &symbols).expect("qwen35 partial-rotary fixture infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
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

/// The qk-norm forward bound at a capacity bucket of `capacity` cache rows
/// while the live `cached_len` is smaller, so the cache blocks carry
/// `capacity - cached_len` pad rows set to `pad_value`. Returns the program,
/// both the padded run (symbol 1 = `capacity`) and the live
/// reference run (symbol 1 = `cached_len`, cache blocks truncated to their
/// first `cached_len` rows). Rows are row-major, so the reference rows are
/// exactly the padded run's live prefix. The named `cached_len` scalar is the
/// true `cached_len` in both runs.
pub fn qwen3_gqa_qk_norm_padded_cache_fixture(
    cached_len: u64,
    capacity: u64,
    pad_value: f32,
) -> (RealForwardFixture, RealForwardFixture) {
    let (program, padded_symbols, roots, padded_source) =
        qwen3_gqa_qk_norm_forward_fixture(1, capacity);
    let (_, reference_symbols, _, _) = qwen3_gqa_qk_norm_forward_fixture(1, cached_len);
    let mut padded = padded_source.clone();
    let mut reference = padded_source;
    for ((name, padded_data), (_, reference_data)) in padded.iter_mut().zip(reference.iter_mut()) {
        if name == "cached_len" {
            *padded_data = vec![cached_len as f32];
            *reference_data = vec![cached_len as f32];
        } else if name.starts_with("kv_cache.") || name.ends_with("_cache") {
            let row_stride = padded_data.len() / capacity as usize;
            let live = cached_len as usize * row_stride;
            padded_data[live..]
                .iter_mut()
                .for_each(|value| *value = pad_value);
            reference_data.truncate(live);
        }
    }
    let padded_run = (program.clone(), padded_symbols, roots.clone(), padded);
    let reference_run = (program, reference_symbols, roots, reference);
    (reference_run, padded_run)
}

/// The query rows a sampled attention parity check compares: the first and
/// last eight, every 64th, and the rows either side of each point where the
/// sliding window starts excluding a key (the oldest cached key, and the
/// first new key). Each query row's attention reads only its own query and
/// the shared K/V, so a row computed alone equals that row of the full run.
pub fn sampled_attention_rows(
    query_rows: u64,
    cached_len: u64,
    cached_lower_inclusive: i64,
) -> Vec<u64> {
    let total = query_rows as i64;
    let mut rows: Vec<i64> = (0..8).chain(total - 8..total).chain((0..total).step_by(64)).collect();
    if cached_lower_inclusive != i64::MIN {
        for boundary in [-cached_lower_inclusive, -(cached_len as i64) - cached_lower_inclusive] {
            rows.extend((boundary - 1)..=(boundary + 1));
        }
    }
    rows.retain(|row| (0..total).contains(row));
    rows.sort_unstable();
    rows.dedup();
    rows.into_iter().map(|row| row as u64).collect()
}

/// Every node the sampled check reads from the device run: each fused
/// attention op's output and its operand tensors.
pub fn attention_probe_roots(resolved: &[BoundOp], logits: NodeId) -> Vec<NodeId> {
    let mut roots = vec![logits];
    for bound in resolved {
        if matches!(bound.kind, BoundOpKind::CachedAttention { .. }) {
            roots.push(bound.node);
            roots.extend(bound.operands().iter().map(|(node, _, _)| *node));
        }
    }
    roots.sort_unstable();
    roots.dedup();
    roots
}

/// Rows checked and the largest `|cpu - device|` over the largest `|cpu|`,
/// for one fused attention op whose operands and output came from `device`.
pub struct SampledAttention {
    pub rows_checked: usize,
    pub relative: f32,
}

/// Runs the CPU attention kernel for the sampled query rows only, at full key
/// width, over the operand tensors the device run produced, and compares those
/// rows with the device output. A contiguous run of rows starting at `first`
/// is a shorter query with every causal band shifted by `first`, which the
/// kernel evaluates exactly.
pub fn sampled_attention_against_device(
    program_len: usize,
    bound: &BoundOp,
    device: &Evaluated,
    cpu_operand_overrides: &[(NodeId, Vec<f32>)],
) -> SampledAttention {
    let BoundOpKind::CachedAttention {
        operands,
        query_rows,
        cached_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        cached_lower_inclusive,
        ..
    } = &bound.kind
    else {
        panic!("sampled attention parity needs a fused cached attention op");
    };
    assert!(*cached_key_rows != 0, "the shifted band needs the two-range form");
    let query_operands: Vec<usize> = if rotary_dim < head_dim {
        vec![0, 1, if has_cached_len(bound) { 9 } else { 8 }]
    } else {
        vec![0, 1]
    };
    let query_nodes: Vec<NodeId> = query_operands.iter().map(|index| operands[*index].0).collect();
    let (device_out, _) = device.get(bound.node).expect("device returned the attention output");
    let row_width = (*kv_heads * *query_groups * *head_dim) as usize;
    let cached_len = live_cached_len(bound, device);
    let sampled = sampled_attention_rows(*query_rows, cached_len, *cached_lower_inclusive);

    let mut expected = Vec::new();
    let mut actual = Vec::new();
    let mut start = 0;
    while start < sampled.len() {
        let mut end = start + 1;
        while end < sampled.len() && sampled[end] == sampled[end - 1] + 1 {
            end += 1;
        }
        let (first, count) = (sampled[start], (end - start) as u64);
        let run = cpu_attention_rows(
            program_len,
            bound,
            (device, cpu_operand_overrides),
            &query_nodes,
            (first, count),
        );
        expected.extend_from_slice(&run);
        actual.extend_from_slice(&device_out[first as usize * row_width..(first + count) as usize * row_width]);
        start = end;
    }
    let magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let difference = expected
        .iter()
        .zip(&actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    SampledAttention {
        rows_checked: sampled.len(),
        relative: difference / magnitude.max(f32::MIN_POSITIVE),
    }
}

fn has_cached_len(bound: &BoundOp) -> bool {
    let BoundOpKind::CachedAttention {
        operands,
        rotary_dim,
        head_dim,
        attention_sink,
        ..
    } = &bound.kind
    else {
        unreachable!("checked by the caller");
    };
    let before_sink = operands.len() - usize::from(*attention_sink);
    if rotary_dim < head_dim { before_sink == 12 } else { before_sink == 9 }
}

fn live_cached_len(bound: &BoundOp, device: &Evaluated) -> u64 {
    let BoundOpKind::CachedAttention { operands, cached_key_rows, .. } = &bound.kind else {
        unreachable!("checked by the caller");
    };
    if !has_cached_len(bound) {
        return *cached_key_rows;
    }
    let (value, _) = device
        .get(operands[8].0)
        .expect("device returned the cached_len operand");
    value[0] as u64
}

fn cpu_attention_rows(
    program_len: usize,
    bound: &BoundOp,
    (device, overrides): (&Evaluated, &[(NodeId, Vec<f32>)]),
    query_nodes: &[NodeId],
    (first, count): (u64, u64),
) -> Vec<f32> {
    let mut shifted = bound.clone();
    let BoundOpKind::CachedAttention {
        query_rows,
        cached_lower_inclusive,
        new_upper_inclusive,
        arithmetic,
        ..
    } = &mut shifted.kind
    else {
        unreachable!("checked by the caller");
    };
    // the f32 online softmax is the oracle the device kernels are held to; the
    // llama f16 flash mode the fused op may carry is a different arithmetic
    *arithmetic = CachedAttentionArithmetic::OnlineF32;
    let full_rows = *query_rows;
    *query_rows = count;
    *cached_lower_inclusive = cached_lower_inclusive.saturating_add(first as i64);
    *new_upper_inclusive = new_upper_inclusive.saturating_add(first as i64);
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program_len];
    for (node, _, _) in shifted.operands() {
        let (device_data, _) = device.get(*node).expect("device returned the attention operand");
        let data = overrides
            .iter()
            .find(|(candidate, _)| candidate == node)
            .map_or(device_data, |(_, replacement)| replacement.as_slice());
        let sliced = if query_nodes.contains(node) {
            let per_row = data.len() / full_rows as usize;
            data[first as usize * per_row..(first + count) as usize * per_row].to_vec()
        } else {
            data.to_vec()
        };
        buffers[node.0 as usize] = Some(sliced);
    }
    let interpreter = Interpreter::new(&mut buffers);
    let batch: ReadyBatch = [shifted.clone()].into_iter().collect();
    let mut context = Context::from_waker(Waker::noop());
    match pin!(interpreter.call(batch)).as_mut().poll(&mut context) {
        Poll::Ready(result) => result.expect("the sampled cpu attention computes"),
        Poll::Pending => unreachable!("cpu pipes never yield"),
    }
    buffers[shifted.node.0 as usize]
        .clone()
        .expect("the sampled cpu attention wrote its output")
}

/// The math mode of [`evaluate_gpu_oracle`], for binding the ops a sampled CPU
/// attention check runs.
#[must_use]
pub fn gpu_oracle_numeric_policy() -> NumericPolicy {
    NumericPolicy::bit_exact().with_flash_narrowing(true)
}
