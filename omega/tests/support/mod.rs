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
    DuplicateHeadPosition, mistral_cached_forward_program, mistral_single_range_cached_forward_program,
    qwen3_cached_forward_program,
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
