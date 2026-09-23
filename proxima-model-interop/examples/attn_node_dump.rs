//! Dumps, for a fixed list of production-plan `NodeId`s in gemma4-E2B's
//! decode-step attention chain, the exact MSL [`omega::Kernel`] `emit`
//! produces plus the packed `Uniforms` bytes [`omega::metal::pack_uniforms_for`]
//! would upload for that same `BoundOp` -- so a byte-parity investigation
//! can diff kernel source/uniforms per node without re-deriving the bind.
//!
//! Copy of `gemma4_attention_chain_census.rs`'s own binding prologue
//! (open the real blob, `GEMMA4.bind`, `bind_symbols`/`infer`, then
//! `bind_with_fusion(.., false, ..)` -- the unfused PRODUCTION shape) plus
//! `prune_dead`, matching `prepare.rs`'s own driver-facing plan exactly.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;

use memmap2::Mmap;
use omega::PackedOperands;
use proxima_gguf::parse_complete;
use proxima_model_interop::{Architecture, GEMMA4, bind_symbols};
use proxima_tensor::bind::BoundOp;
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, bind_with_fusion, infer, prune_dead};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const DEFAULT_OUTPUT_DIR: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/rootcause/r9/nodes";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

/// 14 absorbed nodes' ids minus the layer's own attended node id --
/// `S/census/absorbed_nodes.txt`'s per-layer `ABSORBED` blocks (0, 1, 2, 4,
/// 10 checked directly) show every gemma4 layer shares this exact relative
/// layout, so `PROXIMA_ATTN_LAYER`'s attended node alone regenerates the
/// full 15-node target list for any layer.
const ATTN_ABSORBED_NODE_OFFSETS: [i32; 14] =
    [-32, -31, -27, -20, -24, -17, -15, -14, -12, -9, -10, -8, -4, -2];

fn production_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = Vec::with_capacity(1 + layer_roots.len() * 3);
    outputs.push(logits_root);
    for roots_for_layer in layer_roots {
        match roots_for_layer {
            Qwen35LayerRoots::Attention((even, odd, value)) => {
                outputs.push(*even);
                outputs.push(*odd);
                outputs.push(*value);
            }
            Qwen35LayerRoots::SharedFromLayer(_) => {}
            Qwen35LayerRoots::DenseAttention(_) | Qwen35LayerRoots::Ssm { .. } => {
                panic!("gemma4 E2B's own layer schedule is Attention/SharedFromLayer only")
            }
        }
    }
    outputs
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut rendered = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        rendered.push_str(&format!("{byte:02x}"));
    }
    rendered
}

fn dump_node(bound: &BoundOp, numeric_policy: NumericPolicy, output_dir: &str, manifest: &mut File) {
    let node_id = bound.node.0;
    let packed_operands = PackedOperands::new();
    let kernel = omega::emit(bound, &packed_operands, numeric_policy)
        .unwrap_or_else(|error| panic!("emit node={node_id} failed: {error}"));

    let metal_path = format!("{output_dir}/{node_id}.metal");
    fs::write(&metal_path, &kernel.source).unwrap_or_else(|error| panic!("write {metal_path}: {error}"));

    let grid_path = format!("{output_dir}/{node_id}.grid.txt");
    let mut grid_file =
        File::create(&grid_path).unwrap_or_else(|error| panic!("create {grid_path}: {error}"));
    writeln!(grid_file, "entry={}", kernel.entry).expect("write entry");
    writeln!(grid_file, "bindings={:?}", kernel.bindings).expect("write bindings");
    writeln!(grid_file, "grid={:?}", kernel.grid).expect("write grid");
    writeln!(grid_file, "kind_name={}", bound.kind.name()).expect("write kind_name");
    writeln!(grid_file, "extents={:?}", bound.extents).expect("write extents");
    writeln!(grid_file, "operands={:?}", bound.operands()).expect("write operands");
    let all_read_sources: Vec<_> = bound.all_read_sources().collect();
    writeln!(grid_file, "all_read_sources={all_read_sources:?}").expect("write all_read_sources");

    let uniforms = omega::metal::pack_uniforms_for(bound, numeric_policy)
        .unwrap_or_else(|error| panic!("pack_uniforms_for node={node_id} failed: {error}"));
    let uniforms_path = format!("{output_dir}/{node_id}.uniforms.hex");
    fs::write(&uniforms_path, hex_encode(&uniforms))
        .unwrap_or_else(|error| panic!("write {uniforms_path}: {error}"));

    let manifest_line = format!(
        "node={node_id} kind={} entry={} grid={}/{:?} uniforms_bytes={}",
        bound.kind.name(),
        kernel.entry,
        kernel.grid.threads,
        kernel.grid.threadgroup_width,
        uniforms.len()
    );
    println!("{manifest_line}");
    writeln!(manifest, "{manifest_line}").expect("write manifest line");
}

fn main() {
    let output_dir = std::env::var("PROXIMA_ATTN_NODES_DIR").unwrap_or_else(|_| DEFAULT_OUTPUT_DIR.to_string());
    let layer_index: usize = std::env::var("PROXIMA_ATTN_LAYER")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0);
    fs::create_dir_all(&output_dir).unwrap_or_else(|error| panic!("create {output_dir}: {error}"));

    let file = File::open(REAL_GEMMA4_E2B_GGUF_PATH).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");

    let bound_program = GEMMA4
        .bind(&parsed, bytes)
        .expect("bind the real gemma4-E2B checkpoint's production decode program");

    println!(
        "attn_node_dump: logits_root={} program_len={}",
        bound_program.logits_root.0,
        bound_program.program.len()
    );
    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);

    let kv_bucket_extent: usize = std::env::var("PROXIMA_ATTN_KV_BUCKET_EXTENT")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(KV_BUCKET_EXTENT);
    let symbols = bind_symbols(NEW_COUNT, kv_bucket_extent, &[], bound_program.single_position_step)
        .expect("bind_symbols: gemma4 declares no extra symbolic step_inputs");
    let shapes =
        infer(&bound_program.program, &symbols).expect("shape inference over the real program");

    let numeric_policy = NumericPolicy::llama_relaxed();

    // PRODUCTION shape: `fuse_cached_attention: false`, then `prune_dead`
    // over the SAME requested outputs -- the exact plan `prepare.rs`'s own
    // driver builds (`prepare_uniforms_pack.rs:132-167`).
    let bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, false, numeric_policy)
        .expect("bind_with_fusion (production, unfused)");
    let bound_ops = prune_dead(bound_ops, &outputs);
    println!("attn_node_dump: production_bound_op_count={}", bound_ops.len());

    let by_node: BTreeMap<u32, &BoundOp> = bound_ops.iter().map(|bound| (bound.node.0, bound)).collect();

    // `bound_ops` above is the UNFUSED bind (production shape) -- a
    // `CachedAttention` op never appears as that `BoundOpKind` there, it is
    // already decomposed into the elementwise/reduce chain `dump_node` below
    // reads. A second, FUSED-only bind over the same outputs (never dumped,
    // only inspected) names the per-layer `CachedAttention` node ids in
    // program order, the same discovery `run_attn_fuse_parity_probe`
    // (`decode.rs`) uses for its own `candidates`.
    let fused_bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, true, numeric_policy)
        .expect("bind_with_fusion (fused, for CachedAttention node-id discovery only)");
    let candidates: Vec<u32> = fused_bound_ops
        .iter()
        .filter(|bound| matches!(bound.kind, proxima_tensor::BoundOpKind::CachedAttention { .. }))
        .map(|bound| bound.node.0)
        .collect();
    let extra_node_ids: Vec<u32> = std::env::var("PROXIMA_ATTN_EXTRA_NODES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|entry| entry.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default();
    // this build's recognizer declines the fusion for the real checkpoint's
    // mask shape (0 candidates); extra-node dumps still work off the
    // unfused bind, so only require a candidate when no extras were given.
    let target_node_ids: Vec<u32> = match candidates.get(layer_index) {
        Some(&attended_node) => ATTN_ABSORBED_NODE_OFFSETS
            .iter()
            .map(|offset| (attended_node as i32 + offset) as u32)
            .chain(core::iter::once(attended_node))
            .chain(extra_node_ids.clone())
            .collect(),
        None if !extra_node_ids.is_empty() => extra_node_ids.clone(),
        None => panic!("layer {layer_index} out of range: only {} CachedAttention candidates", candidates.len()),
    };
    println!("attn_node_dump: layer={layer_index} candidates={candidates:?} output_dir={output_dir}");

    let manifest_path = format!("{output_dir}/manifest.txt");
    let mut manifest =
        File::create(&manifest_path).unwrap_or_else(|error| panic!("create {manifest_path}: {error}"));

    for node_id in target_node_ids {
        match by_node.get(&node_id) {
            Some(bound) => dump_node(bound, numeric_policy, &output_dir, &mut manifest),
            None => {
                let missing_line = format!("node={node_id} kind=MISSING entry= grid= uniforms_bytes=0");
                println!("{missing_line}");
                writeln!(manifest, "{missing_line}").expect("write manifest missing line");
            }
        }
    }

    // consumer census for the extra nodes: who reads them, and are they a
    // requested output (both gate whether a fold is byte-safe).
    for extra_node in &extra_node_ids {
        let is_output = outputs.iter().any(|node| node.0 == *extra_node);
        let consumers: Vec<(u32, &str, proxima_tensor::Layout, Option<()>)> = bound_ops
            .iter()
            .flat_map(|bound| {
                bound
                    .all_read_sources()
                    .filter(|(node, _, _)| node.0 == *extra_node)
                    .map(move |(_, layout, lookup)| {
                        (bound.node.0, bound.kind.name(), layout.clone(), lookup.as_ref().map(|_| ()))
                    })
            })
            .collect();
        println!(
            "attn_node_dump: consumer_census node={extra_node} is_requested_output={is_output} reader_count={} readers={consumers:?}",
            consumers.len()
        );
    }
}
