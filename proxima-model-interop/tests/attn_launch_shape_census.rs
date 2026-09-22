//! MEASUREMENT-ONLY launch-shape census, not part of the landing gate: emits
//! one CSV row per `BoundOp` in the production gemma4-E2B decode program,
//! for both `fuse_cached_attention` settings, via [`omega::emit`]'s own
//! [`omega::msl::Kernel::grid`] -- the same bind + emit path
//! `gemma4_attention_chain_census.rs` already exercises, widened here from
//! "chain ops at 3 census layers" to "every BoundOp, both arms" because this
//! file's only job is the CSV, not the chain classification.
//!
//! This is a launch-LAYOUT census (grid_threads, threadgroup_width as
//! `omega::emit` computes them from the bound shapes) -- it runs no device,
//! encodes no command buffer, and measures no occupancy.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::Write;

use memmap2::Mmap;
use omega::PackedOperands;
use proxima_gguf::parse_complete;
use proxima_model_interop::{Architecture, GEMMA4, bind_symbols};
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, bind_with_fusion, infer};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const OFF_CSV_PATH: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/measure/saturation/launch_shapes_off.csv";
const ON_CSV_PATH: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/measure/saturation/launch_shapes_on.csv";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

fn production_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = Vec::with_capacity(1 + layer_roots.len() * 3);
    outputs.push(logits_root);
    for roots_for_layer in layer_roots {
        if let Qwen35LayerRoots::Attention((even, odd, value)) = roots_for_layer {
            outputs.push(*even);
            outputs.push(*odd);
            outputs.push(*value);
        }
    }
    outputs
}

fn write_launch_shapes(path: &str, label: &str, program_outputs: &(Vec<proxima_tensor::bind::BoundOp>,)) {
    let mut csv = File::create(path).unwrap_or_else(|error| panic!("create {path}: {error}"));
    writeln!(csv, "index,node,kind,entry,grid_threads,threadgroup_width")
        .expect("write csv header");
    let packed_operands = PackedOperands::new();
    let numeric_policy = NumericPolicy::llama_relaxed();
    let mut emitted = 0usize;
    let mut errored = 0usize;
    for (index, bound) in program_outputs.0.iter().enumerate() {
        match omega::emit(bound, &packed_operands, numeric_policy) {
            Ok(kernel) => {
                emitted += 1;
                writeln!(
                    csv,
                    "{index},{},{},{},{},{:?}",
                    bound.node.0,
                    bound.kind.name(),
                    kernel.entry,
                    kernel.grid.threads,
                    kernel.grid.threadgroup_width
                )
                .expect("write csv row");
            }
            Err(error) => {
                errored += 1;
                writeln!(
                    csv,
                    "{index},{},{},emit_error,0,{error:?}",
                    bound.node.0,
                    bound.kind.name()
                )
                .expect("write csv error row");
            }
        }
    }
    println!(
        "attn_launch_shape_census[{label}]: total_bound_ops={} emitted={emitted} errored={errored} csv={path}",
        program_outputs.0.len()
    );
}

#[proxima::test]
async fn attn_launch_shape_census() {
    let Ok(file) = File::open(REAL_GEMMA4_E2B_GGUF_PATH) else {
        eprintln!("skipping: real gemma4-E2B blob not found at {REAL_GEMMA4_E2B_GGUF_PATH}");
        return;
    };
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");

    let bound_program = GEMMA4
        .bind(&parsed, bytes)
        .expect("bind the real gemma4-E2B checkpoint's production decode program");

    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);

    let symbols = bind_symbols(NEW_COUNT, KV_BUCKET_EXTENT, &[], bound_program.single_position_step)
        .expect("bind_symbols: gemma4 declares no extra symbolic step_inputs");
    let shapes =
        infer(&bound_program.program, &symbols).expect("shape inference over the real program");

    let numeric_policy = NumericPolicy::llama_relaxed();
    println!(
        "attn_launch_shape_census: debug outputs_len={} program_len={} block_count={} single_position_step={}",
        outputs.len(),
        bound_program.program.len(),
        bound_program.architecture.block_count,
        bound_program.single_position_step
    );

    let unfused_bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, false, numeric_policy)
        .expect("bind_with_fusion unfused (OFF)");
    let fused_bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, true, numeric_policy)
        .expect("bind_with_fusion fused (ON)");
    let cached_attention_count = fused_bound_ops
        .iter()
        .filter(|bound| matches!(bound.kind, proxima_tensor::bind::BoundOpKind::CachedAttention { .. }))
        .count();
    println!(
        "attn_launch_shape_census: unfused_len={} fused_len={} fused_cached_attention_count={cached_attention_count}",
        unfused_bound_ops.len(),
        fused_bound_ops.len()
    );

    write_launch_shapes(OFF_CSV_PATH, "off_unfused", &(unfused_bound_ops,));
    write_launch_shapes(ON_CSV_PATH, "on_fused", &(fused_bound_ops,));
}
