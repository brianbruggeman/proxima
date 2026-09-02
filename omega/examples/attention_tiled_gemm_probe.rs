//! Step 1's own ask (ROW 114): which condition rejects each REAL attention
//! matmul from the tiled-GEMM `simdgroup_matrix` path
//! ([`omega::msl::classify_tiled_gemm`]), read off the real bound graph a
//! decode step actually builds -- not a synthetic fixture, not an inferred
//! guess from the source.
//!
//! Sibling to `real_forward_packed_probe.rs` (same program, same shapes,
//! same bind/`correct_packed_matmul_layouts` sequence), extended two ways:
//!
//! 1. `real_forward_packed_probe` only visits reduces with a PACKED operand
//!    (the row-blocked gate's own precondition). This probe ALSO visits the
//!    two-operand attention reduces that have NO packed operand at all --
//!    `score_product` (`Q @ K`) and `value_product` (`softmax_weights @ V`)
//!    -- because those are real candidates the task asks about even though
//!    they can never carry a weight tile to reuse.
//! 2. For every candidate it runs BOTH `diagnose_packed_row_block` (the
//!    prerequisite gate) and `diagnose_tiled_gemm_block` (the additional
//!    narrowing), so the table shows exactly which of the two rejected a
//!    given op, never a guess about which layer failed.
//!
//! `score_product`/`value_product` have no [`Op::Input`] name to key a
//! family off (the fused reduce's own two operands are both computed
//! nodes), so they are identified structurally, over the REAL graph
//! `mistral_cached_forward_program` returns as `Vec<Op>`: a fused two-operand
//! `Add`-reduce with neither operand tagged as a packed weight is either
//! `rmsnorm`'s sum-of-squares (both operand slots are the SAME `NodeId` --
//! `x * x`) or an attention product (the two operand slots are DISTINCT
//! nodes). Among the attention products, `value_product` is the one where at
//! least one operand's IMMEDIATE producer in `program` is an
//! `Op::Elementwise { body: ScalarOp::Exponential, .. }` (the softmax
//! numerator); `score_product` is everything else in that bucket.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use proxima_tensor::spec::mistral_cached_forward_program;
use proxima_tensor::{
    BoundOpKind, Keep, NodeId, Op, ScalarOp, bind, correct_packed_matmul_layouts, infer,
};

#[cfg(feature = "instrument")]
struct RejectionShape {
    count: usize,
    extents: Vec<u64>,
    output_axes: Vec<u16>,
    reduce_dims: Vec<u16>,
}

fn main() {
    // openchat-3.5-1210 / Mistral-7B -- identical architecture parameters to
    // `real_forward_packed_probe.rs` so the two tables describe the same
    // real forward.
    const VOCAB: u32 = 32000;
    const EMBEDDING: u32 = 4096;
    const FEED_FORWARD: u32 = 14336;
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 128;
    const BLOCKS: u32 = 32;

    let (program, logits_root, cache_roots) = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        BLOCKS,
    )
    .expect("the real forward's own architecture parameters build a program");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    // one decode step: one new position, no cached history -- the steady
    // decode shape `proxima-tensor/docs/discipline.md` ROW 82 measured, same
    // shape ROW 113's own FFN probe used.
    let symbols = [1u64, 0u64];
    let shapes = infer(&program, &symbols).expect("shapes resolve for a concrete decode step");

    let mut matmul_weight_names: BTreeSet<String> = BTreeSet::new();
    for layer in 0..BLOCKS {
        for suffix in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            matmul_weight_names.insert(format!("blk.{layer}.{suffix}.weight"));
        }
    }
    matmul_weight_names.insert("output.weight".to_string());

    let q4k_operands: BTreeSet<NodeId> = program
        .iter()
        .enumerate()
        .filter_map(|(position, op)| {
            let name = op.name()?;
            matmul_weight_names
                .contains(name)
                .then_some(NodeId(position as u32))
        })
        .collect();

    #[cfg(feature = "instrument")]
    let packed_operands: omega::PackedOperands = q4k_operands
        .iter()
        .map(|node| (*node, omega::PackedCodec::Q4K))
        .collect();

    // MIRRORS `metal::prepare` exactly, same as `real_forward_packed_probe.rs`.
    let mut bound = bind(&program, &shapes, &roots).expect("the real forward binds");
    correct_packed_matmul_layouts(&mut bound, &q4k_operands);

    #[cfg(feature = "instrument")]
    let mut rejection_table: std::collections::BTreeMap<
        (String, String, String),
        RejectionShape,
    > = std::collections::BTreeMap::new();
    let mut attention_matmuls_seen = 0usize;

    for op in &bound {
        let BoundOpKind::Reduce {
            reduce_op,
            init,
            keep: Keep::Reduce,
            output_axes,
            ..
        } = &op.kind
        else {
            continue;
        };
        let operands = op.operands();
        if operands.len() != 2 {
            // rmsnorm's normalizer (`sum_cached`) and softmax's own sum/max
            // are single-operand reduces -- not a matmul shape at all, and
            // not part of this table (see this file's own module doc).
            continue;
        }

        let (node_a, _, _) = operands[0];
        let (node_b, _, _) = operands[1];
        let packed_a = q4k_operands.contains(&node_a);
        let packed_b = q4k_operands.contains(&node_b);

        let label = if packed_a || packed_b {
            let weight_node = if packed_a { node_a } else { node_b };
            let weight_name = program[weight_node.0 as usize]
                .name()
                .expect("a q4k_operands node is always an Op::Input with a name");
            let family = strip_layer_index(weight_name);
            if !family.contains("attn_") {
                // ffn_gate/ffn_up/ffn_down/output.weight already have their
                // own table, ROW 113 -- this probe's ask is attention only.
                continue;
            }
            family
        } else if node_a == node_b {
            // rmsnorm's `x * x` sum-of-squares -- same node in both operand
            // slots, no weight, not an attention product at all.
            continue;
        } else {
            let is_value_product = [node_a, node_b].into_iter().any(|node| {
                matches!(
                    &program[node.0 as usize],
                    Op::Elementwise {
                        body: ScalarOp::Exponential,
                        ..
                    }
                )
            });
            if is_value_product {
                "value_product (softmax_weights @ V)".to_string()
            } else {
                "score_product (Q @ K)".to_string()
            }
        };

        attention_matmuls_seen += 1;

        #[cfg(feature = "instrument")]
        {
            let quantized: Vec<Option<omega::PackedCodec>> = operands
                .iter()
                .map(|(node, _, _)| packed_operands.get(node).copied())
                .collect();
            let packed_reason = match omega::msl::diagnose_packed_row_block(op, &quantized) {
                Ok(()) => "PASS (row-blocked)".to_string(),
                Err(rejection) => format!("packed_row_block: {rejection:?}"),
            };
            let tiled_reason = match omega::msl::diagnose_tiled_gemm_block(
                op,
                &quantized,
                *reduce_op,
                *init,
                output_axes,
            ) {
                Ok(()) => "PASS (tiled-gemm)".to_string(),
                Err(rejection) => format!("tiled_gemm: {rejection:?}"),
            };
            let reduce_dims: Vec<u16> = (0..op.extents.len() as u16)
                .filter(|dim| !output_axes.contains(dim))
                .collect();
            let entry = rejection_table
                .entry((label, packed_reason, tiled_reason))
                .or_insert_with(|| RejectionShape {
                    count: 0,
                    extents: op.extents.clone(),
                    output_axes: output_axes.to_vec(),
                    reduce_dims,
                });
            entry.count += 1;
        }
        #[cfg(not(feature = "instrument"))]
        {
            let _ = (label, reduce_op, init, output_axes);
        }
    }

    println!(
        "attention matmuls (Q/K/V/O projections + score/value products) seen: {attention_matmuls_seen}"
    );
    assert_ne!(
        attention_matmuls_seen, 0,
        "degenerate probe: no attention matmul ever visited"
    );

    #[cfg(feature = "instrument")]
    {
        println!(
            "\n=== attention tiled-gemm rejection table: op family, extents, output_axes, reduce_dims, row-blocked gate, tiled-gemm gate ==="
        );
        for ((label, packed_reason, tiled_reason), shape) in &rejection_table {
            println!(
                "op={label:?} count={} extents={:?} output_axes={:?} reduce_dims={:?} \
                 {packed_reason} | {tiled_reason}",
                shape.count, shape.extents, shape.output_axes, shape.reduce_dims,
            );
        }
    }
}

/// `blk.7.attn_output.weight` -> `attn_output.weight` -> `attn_output`:
/// drops exactly one `.`-delimited numeric segment (the layer index) and the
/// trailing `.weight`, same shape as
/// `real_forward_packed_probe.rs`'s own `strip_layer_index` (duplicated here
/// for the same reason that file gives: standalone example binary, four
/// lines, not worth a shared dependency for).
fn strip_layer_index(name: &str) -> String {
    name.split('.')
        .filter(|segment| {
            segment.parse::<u32>().is_err() && *segment != "blk" && *segment != "weight"
        })
        .collect::<Vec<&str>>()
        .join(".")
}
