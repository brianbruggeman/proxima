//! Does the real 7B forward's Q4_K matmuls actually take the row-blocked
//! packed kernel [`omega::examples::q4k_matvec_probe`]'s 143.5 GB/s
//! (`proxima-tensor/docs/discipline.md` ROW 77) measured, or the generic
//! scalar `q4k_element` path?
//!
//! `real_forward_emit_probe.rs` answers "can the emitter emit every op" with
//! `q4k_operands` EMPTY -- every operand looks like `Float32` to `emit`, so
//! it never exercises `msl::packed_row_block` at all. This probe marks every
//! matmul-weight `Op::Input` the real checkpoint binds as `Q4_K` packed
//! (`bind_matmul_weight`'s own set: `attn_q`/`attn_k`/`attn_v`/`attn_output`/
//! `ffn_gate`/`ffn_up`/`ffn_down` per layer, plus `output.weight` -- never
//! the norms or `token_embd.weight`, which `bind_dense` binds `Float32`
//! regardless of on-disk codec) and reports, per reduce op with a packed
//! operand, whether the emitted kernel source is the row-blocked
//! `q4k_run8`/four-row-fold body or the generic per-element `q4k_element`
//! scalar read -- the literal text a real forward's own kernel carries, not
//! an inference from the source of `msl::packed_row_block` (private to that
//! module, so this probe cannot call it directly; the emitted SOURCE is the
//! same decision made visible through the one public seam, [`omega::emit`]).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use proxima_tensor::spec::mistral_cached_forward_program;
use proxima_tensor::{BoundOpKind, Keep, NodeId, bind, correct_packed_matmul_layouts, infer};

fn main() {
    // openchat-3.5-1210 / Mistral-7B
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
    // decode shape `proxima-tensor/docs/discipline.md` ROW 82 measured.
    let symbols = [1u64, 0u64];
    let shapes = infer(&program, &symbols).expect("shapes resolve for a concrete decode step");

    // every matmul-weight name `proxima-model-interop::bind::bind_matmul_weight`
    // packs, never the norms/embedding `bind_dense` keeps Float32.
    let mut matmul_weight_names: BTreeSet<String> = BTreeSet::new();
    for layer in 0..BLOCKS {
        for suffix in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
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
    println!(
        "matmul weight names={} resolved to q4k_operands={}",
        matmul_weight_names.len(),
        q4k_operands.len()
    );

    // MIRRORS `metal::prepare` exactly (`omega/src/metal.rs`'s own
    // `bind` -> `correct_packed_matmul_layouts` sequence): `bind` alone
    // assumes every operand is row-major in its DECLARED axis order, which
    // is false for a packed Q4_K weight's native `[out, in]` bytes, so the
    // real driver ALWAYS runs this post-pass before `emit`. Calling plain
    // `bind` without it here would diagnose a stride nothing in production
    // ever emits.
    let mut bound = bind(&program, &shapes, &roots).expect("the real forward binds");
    correct_packed_matmul_layouts(&mut bound, &q4k_operands);

    let mut reduce_with_packed_operand = 0usize;
    let mut row_blocked = 0usize;
    let mut generic_scalar = 0usize;
    let mut first_generic_source: Option<String> = None;
    let mut first_row_blocked_source: Option<String> = None;

    for op in &bound {
        let is_reduce_keep_reduce = matches!(op.kind, BoundOpKind::Reduce { keep: Keep::Reduce, .. });
        let has_packed_operand = op.operands().iter().any(|(node, _, _)| q4k_operands.contains(node));
        if !(is_reduce_keep_reduce && has_packed_operand) {
            continue;
        }
        reduce_with_packed_operand += 1;
        if reduce_with_packed_operand <= 6 {
            let packed_operand_count =
                op.operands().iter().filter(|(node, _, _)| q4k_operands.contains(node)).count();
            println!(
                "diag node={:?} extents={:?} operand_count={} packed_operand_count={} kind={:?}",
                op.node,
                op.extents,
                op.operands().len(),
                packed_operand_count,
                op.kind
            );
        }

        let kernel = omega::emit(op, &q4k_operands).expect("the real forward's own bound ops emit");
        // `q4k_run8`/`q4k_element`'s own FUNCTION DEFINITIONS are always
        // part of the emitted prelude regardless of which path a given
        // reduce body takes, so the marker has to be the CALL SITE, not the
        // bare name -- `q4k_run8(blk` is that function's one real call
        // (`push_packed_row_blocked_body`'s own `"q4k_run8(blk, slot + ..."`
        // format string); `q4k_element(in` is the generic body's per-element
        // read (`push_cooperative_reduce_body`'s own `"q4k_element(in{index}
        // + ..."` format string).
        let took_row_blocked = kernel.source.contains("q4k_run8(blk");
        let took_generic_scalar = kernel.source.contains("q4k_element(in");
        if took_row_blocked {
            row_blocked += 1;
            if first_row_blocked_source.is_none() {
                first_row_blocked_source = Some(kernel.source.clone());
            }
        }
        if took_generic_scalar {
            generic_scalar += 1;
            if first_generic_source.is_none() {
                first_generic_source = Some(kernel.source.clone());
            }
        }
        assert!(
            took_row_blocked != took_generic_scalar,
            "node {:?} kernel source should contain exactly one of q4k_run8/q4k_element, source:\n{}",
            op.node,
            kernel.source
        );
    }

    println!(
        "reduce ops with a packed operand: {reduce_with_packed_operand} (row_blocked={row_blocked} generic_scalar={generic_scalar})"
    );

    if let Some(source) = &first_generic_source {
        println!("\n=== first GENERIC SCALAR kernel body (q4k_element path) ===");
        println!("{source}");
    }
    if let Some(source) = &first_row_blocked_source {
        println!("\n=== first ROW-BLOCKED kernel body (q4k_run8 path) ===");
        println!("{source}");
    }

    assert_ne!(reduce_with_packed_operand, 0, "degenerate probe: no reduce op ever saw a packed operand");
}
