//! Rootcause round 9, wiring gate 2: the SAME real gemma4-E2B decode-step
//! bind as `attn_node_dump.rs` (copy of its own binding prologue), but
//! with `fuse_cached_attention: true` so node 166's own attention chain
//! resolves to ONE `BoundOpKind::CachedAttention` op instead of the 15
//! unfused per-node kernels -- dumps that op's emitted MSL text
//! (`omega::emit`) so a byte-parity investigation can diff it against
//! `R9/../r3/fused_production.metal`'s own dumped entry without
//! re-deriving the bind.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;

use memmap2::Mmap;
use omega::PackedOperands;
use proxima_gguf::parse_complete;
use proxima_model_interop::{Architecture, GEMMA4, bind_symbols};
use proxima_tensor::bind::BoundOpKind;
use proxima_tensor::{NumericPolicy, bind_with_fusion, infer, prune_dead};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const OUTPUT_PATH: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/rootcause/r9/wiring/gemma_fused_op.metal";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

fn main() {
    let file = std::fs::File::open(REAL_GEMMA4_E2B_GGUF_PATH).expect("open the real gemma4-E2B checkpoint");
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");

    let bound_program = GEMMA4
        .bind(&parsed, bytes)
        .expect("bind the real gemma4-E2B checkpoint's production decode program");

    let outputs = alloc_outputs(&bound_program);

    let symbols = bind_symbols(NEW_COUNT, KV_BUCKET_EXTENT, &[], bound_program.single_position_step)
        .expect("bind_symbols: gemma4 declares no extra symbolic step_inputs");
    let shapes =
        infer(&bound_program.program, &symbols).expect("shape inference over the real program");

    let numeric_policy = NumericPolicy::llama_relaxed();

    // fuse_cached_attention: true -- the FUSED production shape, one
    // BoundOpKind::CachedAttention per attention layer instead of the 15
    // unfused per-node kernels `attn_node_dump.rs`'s own `false` shape
    // resolves.
    let bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, true, numeric_policy)
        .expect("bind_with_fusion (production, fused)");
    let bound_ops = prune_dead(bound_ops, &outputs);

    let cached_attention_ops: Vec<_> = bound_ops
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    println!("attn_fused_op_dump: cached_attention_op_count={}", cached_attention_ops.len());

    let first = cached_attention_ops
        .first()
        .expect("the fused gemma4 production plan resolves at least one CachedAttention op");

    let packed_operands = PackedOperands::new();
    let kernel = omega::emit(first, &packed_operands, numeric_policy)
        .unwrap_or_else(|error| panic!("emit node={} failed: {error}", first.node.0));

    println!("attn_fused_op_dump: node={} entry={}", first.node.0, kernel.entry);
    fs::create_dir_all(std::path::Path::new(OUTPUT_PATH).parent().expect("has parent"))
        .expect("create output dir");
    fs::write(OUTPUT_PATH, &kernel.source).unwrap_or_else(|error| panic!("write {OUTPUT_PATH}: {error}"));

    // With `metal-fuse-attn-decode` ON, the recognizer now ALWAYS sets
    // `two_pass: true` for an eligible gemma op (Part A's own fix -- a
    // gemma decode candidate is never accepted with `two_pass: false`
    // anymore, by design). `r3/fused_production.metal` was captured before
    // `two_pass` existed at all, so there is no live recognizer path left
    // that reproduces its ONLINE-kernel text for this exact op shape.
    // Force the field back to `false` on a clone of the SAME resolved op
    // (same query_groups/head_dim/cached_key_rows/scale/window -- only the
    // dispatch selector bit differs) to exercise the unmodified fallthrough
    // branch `render_cached_attention` always ran before this session's
    // `if *two_pass` insertion.
    let mut online_kernel_op = (**first).clone();
    if let BoundOpKind::CachedAttention { two_pass, .. } = &mut online_kernel_op.kind {
        *two_pass = false;
    }
    let online_kernel = omega::emit(&online_kernel_op, &packed_operands, numeric_policy)
        .unwrap_or_else(|error| panic!("emit (two_pass forced false) node={} failed: {error}", online_kernel_op.node.0));
    println!("attn_fused_op_dump: online_kernel entry={}", online_kernel.entry);
    let online_kernel_path = format!("{OUTPUT_PATH}.online_kernel_fallback.metal");
    fs::write(&online_kernel_path, &online_kernel.source)
        .unwrap_or_else(|error| panic!("write {online_kernel_path}: {error}"));
}

fn alloc_outputs(bound_program: &proxima_model_interop::BoundProgram<'_>) -> Vec<proxima_tensor::NodeId> {
    let mut outputs = Vec::with_capacity(1 + bound_program.layer_roots.len() * 3);
    outputs.push(bound_program.logits_root);
    for roots_for_layer in &bound_program.layer_roots {
        match roots_for_layer {
            proxima_tensor::spec::Qwen35LayerRoots::Attention((even, odd, value)) => {
                outputs.push(*even);
                outputs.push(*odd);
                outputs.push(*value);
            }
            proxima_tensor::spec::Qwen35LayerRoots::SharedFromLayer(_) => {}
            proxima_tensor::spec::Qwen35LayerRoots::DenseAttention(_)
            | proxima_tensor::spec::Qwen35LayerRoots::Ssm { .. } => {
                panic!("gemma4 E2B's own layer schedule is Attention/SharedFromLayer only")
            }
        }
    }
    outputs
}
