//! Metal emit gate for PROXNUM-262's end-aligned single query
//! (`kv-epic/PROXNUM-262-row-liveness.md`, worked examples 1 and 2): the
//! prefill's two-range cached attention rewritten to ONE query
//! (`query_rows = 1`) over `new_key_rows = N` with both bounds shifted by
//! N - 1. The CPU half of the contract is
//! `proxima-tensor` `cpu::tests::one_end_aligned_query_equals_the_last_row_of_the_full_prefill_attention`.
//! No Metal builder yields a one-query op over N new keys, so this gate asks
//! the renderer whether it accepts the shape.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-rows",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::qk_norm_cached_forward_program;
use proxima_tensor::{BoundOp, BoundOpKind, infer};

mod support;
use support::production_numeric_policy;

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 512;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const CACHED_LEN: u64 = 31;

fn prefill_attention_op(rows: u64, head_dim: u32) -> BoundOp {
    let (program, logits, _caches) = qk_norm_cached_forward_program(
        VOCAB, EMBEDDING, FEED_FORWARD, QUERY_HEADS, KV_HEADS, head_dim, 1,
    )
    .expect("the gemma-like gqa prefill program builds");
    let symbols = vec![rows, CACHED_LEN];
    let shapes = infer(&program, &symbols).expect("the prefill program infers");
    proxima_tensor::bind(&program, &shapes, &[logits], production_numeric_policy())
        .expect("the prefill program binds")
        .into_iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the prefill program fuses one cached attention op")
}

fn end_aligned(mut bound: BoundOp, lower: i64) -> BoundOp {
    let BoundOpKind::CachedAttention {
        query_rows,
        new_key_rows,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &mut bound.kind
    else {
        unreachable!("prefill_attention_op returns a cached attention op");
    };
    let shift = *new_key_rows as i64 - 1;
    *query_rows = 1;
    *cached_lower_inclusive = lower.saturating_add(shift);
    *new_upper_inclusive = shift;
    bound.extents[0] = 1;
    bound
}

#[proxima::test]
#[case::n4_sliding_head_dim_256(4, -511, 256)]
#[case::n4_global_head_dim_512(4, i64::MIN, 512)]
#[case::n600_sliding_head_dim_256(600, -511, 256)]
#[case::n600_global_head_dim_512(600, i64::MIN, 512)]
async fn the_metal_renderer_emits_a_one_end_aligned_query_over_n_new_keys(
    #[case] rows: u64,
    #[case] lower: i64,
    #[case] head_dim: u32,
) {
    let full = prefill_attention_op(rows, head_dim);
    let BoundOpKind::CachedAttention { query_rows, new_key_rows, .. } = &full.kind else {
        unreachable!("prefill_attention_op returns a cached attention op");
    };
    assert_eq!((*query_rows, *new_key_rows), (rows, rows), "the unshifted op is the N-query prefill");
    let one = end_aligned(full, lower);
    let kernel = omega::emit(&one, &omega::PackedOperands::new(), production_numeric_policy())
        .expect("the renderer must accept query_rows = 1 over new_key_rows = N");
    eprintln!("end-aligned rows {rows} head_dim {head_dim}: entry {}", kernel.entry);
}
