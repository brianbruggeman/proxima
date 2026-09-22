//! Candidate B (owner-authorized partial fusion, bounded experiment,
//! standalone only -- see
//! `/private/tmp/attn_fusion_final_disposition.md` and
//! `R9/PROGRESS.md` "Candidate B"): compares the production 14-node
//! attention-chain closure (dumped by `attn_node_dump`, R9/nodes*) run as
//! individual dispatches ("OFF") against a partial-fusion chain that keeps
//! the QK folds (139/142/146/149) and the cached-key AV fold (162) as
//! separate production dispatches and fuses the softmax/normalization glue
//! (151/152/154/156/157/158/164) plus node 166 (as 162's epilogue) into two
//! kernels ("B"), all reading the SAME captured leaf inputs
//! (`R9/vectors*/<node>.bits`), one command buffer per chain, GPU-timed via
//! `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime`.
//!
//! STATUS 2026-09-21: OFF chain implemented and gated (byte-exact against
//! the captured node 166 output, all four contexts this slice reached).
//! The B chain's fused kernels (`K_softmax`, `162`-with-166-epilogue) are
//! NOT implemented yet -- see R9/PROGRESS.md "Candidate B" for the derived
//! per-stage source (read from the real dumped bodies) and the concrete
//! next step. This binary reports OFF-only numbers until that lands; the
//! `--chain off` default keeps this runnable meanwhile.
//!
//! # Run
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-attn-fuse \
//!     cargo run -p omega --release --features metal --example attn_partial_fusion_chain \
//!     -- --nodes-dir <R9>/nodes --vectors-dir <R9>/vectors_relaxed --math-mode relaxed \
//!        --label "layer0_c32"
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_partial_fusion_chain requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ChainOutputs = HashMap<u32, MetalBuffer>;

// `Kernel`/`PackedOperands`/`emit` renamed on import: this file's own
// hand-parsed `GridSpec`/`Binding` (read from `<node>.grid.txt` text,
// `struct GridSpec`/`enum Binding` above) share a name with `omega`'s own
// types of the same purpose -- the library path below never constructs
// `omega::Binding`/`omega::GridSpec` by name, only reads fields off the
// `Kernel` value `omega_emit` returns, so no alias is needed for those two.
use omega::{PackedOperands, emit as omega_emit};
use proxima_tensor::{BoundOp, BoundOpKind, DType, Layout, NodeId, NumericPolicy};

// the OFF chain's dispatch order is read from `<nodes_dir>/manifest.txt`
// (`attn_node_dump`'s emission order = the production plan's node order
// within the attention chain, role-stable across layers even though node
// ids shift by the per-layer stride -- see `attn_node_gate::target_nodes`,
// omega/examples/attn_node_gate.rs:50): each node's Input(NodeId(..))
// bindings resolve either to a captured leaf (`<vectors_dir>/<id>.bits`)
// or to an earlier chain node's own dispatch output (chained, not
// re-read), so the OFF chain measures true dependent-dispatch GPU time.
// The LAST manifest line is the attended node (166's role) this chain's
// gate compares against the captured reference.
fn chain_node_order(nodes_dir: &Path) -> Vec<u32> {
    let manifest = std::fs::read_to_string(nodes_dir.join("manifest.txt"))
        .unwrap_or_else(|error| panic!("read {}/manifest.txt: {error}", nodes_dir.display()));
    manifest
        .lines()
        .filter_map(|line| line.strip_prefix("node="))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(|id| id.parse::<u32>().unwrap_or_else(|error| panic!("manifest node id `{id}`: {error}")))
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum Binding {
    Input(u32),
    Output(u32),
    Uniforms,
}

struct GridSpec {
    threads: usize,
    threadgroup_width: Option<usize>,
}

struct NodeSpec {
    entry: String,
    bindings: Vec<Binding>,
    grid: GridSpec,
}

fn parse_bindings(line: &str) -> Vec<Binding> {
    let start = line.find('[').expect("bindings= line carries a [...] list") + 1;
    let end = line.rfind(']').expect("bindings= line's list is closed");
    line[start..end]
        .split(", ")
        .map(|token| {
            let token = token.trim();
            if let Some(inner) =
                token.strip_prefix("Input(NodeId(").and_then(|rest| rest.strip_suffix("))"))
            {
                Binding::Input(inner.parse::<u32>().expect("Input NodeId is numeric"))
            } else if let Some(inner) =
                token.strip_prefix("Output(NodeId(").and_then(|rest| rest.strip_suffix("))"))
            {
                Binding::Output(inner.parse::<u32>().expect("Output NodeId is numeric"))
            } else if token == "Uniforms" {
                Binding::Uniforms
            } else {
                panic!("unrecognized binding token `{token}`");
            }
        })
        .collect()
}

fn parse_grid(line: &str) -> GridSpec {
    let threads_start = line.find("threads: ").expect("grid= line carries threads:") + "threads: ".len();
    let threads_end =
        line[threads_start..].find(',').expect("threads field is comma-terminated") + threads_start;
    let threads = line[threads_start..threads_end].trim().parse::<usize>().expect("threads value is numeric");

    let width_start = line.find("threadgroup_width: ").expect("grid= line carries threadgroup_width:")
        + "threadgroup_width: ".len();
    let width_end =
        line[width_start..].find(", depth").expect("threadgroup_width field precedes , depth") + width_start;
    let width_field = line[width_start..width_end].trim();
    let threadgroup_width = if width_field == "None" {
        None
    } else {
        let inner = width_field
            .strip_prefix("Some(")
            .and_then(|rest| rest.strip_suffix(')'))
            .unwrap_or_else(|| panic!("unrecognized threadgroup_width field `{width_field}`"));
        Some(inner.parse::<usize>().expect("threadgroup_width inner value is numeric"))
    };
    GridSpec { threads, threadgroup_width }
}

fn read_node_spec(nodes_dir: &Path, id: u32) -> NodeSpec {
    let grid_text = std::fs::read_to_string(nodes_dir.join(format!("{id}.grid.txt")))
        .unwrap_or_else(|error| panic!("read nodes/{id}.grid.txt: {error}"));
    let entry_line = grid_text
        .lines()
        .find(|line| line.starts_with("entry="))
        .unwrap_or_else(|| panic!("nodes/{id}.grid.txt missing entry= line"));
    let bindings_line = grid_text
        .lines()
        .find(|line| line.starts_with("bindings="))
        .unwrap_or_else(|| panic!("nodes/{id}.grid.txt missing bindings= line"));
    let grid_line = grid_text
        .lines()
        .find(|line| line.starts_with("grid="))
        .unwrap_or_else(|| panic!("nodes/{id}.grid.txt missing grid= line"));
    NodeSpec {
        entry: entry_line.trim_start_matches("entry=").trim().to_string(),
        bindings: parse_bindings(bindings_line),
        grid: parse_grid(grid_line),
    }
}

fn parse_bits_file(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let stripped = line
                .trim()
                .strip_prefix("0x")
                .unwrap_or_else(|| panic!("{} line `{line}` missing 0x prefix", path.display()));
            u32::from_str_radix(stripped, 16)
                .unwrap_or_else(|error| panic!("{} line `{line}` parses as hex u32: {error}", path.display()))
        })
        .collect()
}

fn parse_uniforms_hex(path: &Path) -> Vec<u8> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let hex = text.trim();
    assert!(hex.len().is_multiple_of(2), "{} carries an even number of hex digits", path.display());
    (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .unwrap_or_else(|error| panic!("{} byte at {index} parses as hex: {error}", path.display()))
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|value| f32::from_bits(*value)).collect()
}

fn shared_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let padded_len = bytes.len().max(4);
    let mut padded = bytes.to_vec();
    padded.resize(padded_len, 0);
    let pointer = unsafe { NonNull::new_unchecked(padded.as_ptr() as *mut c_void) };
    unsafe { device.newBufferWithBytes_length_options(pointer, padded.len(), MTLResourceOptions::StorageModeShared) }
        .expect("device allocates a fresh shared buffer copied from real fixture bytes")
}

fn zeroed_buffer(device: &ProtocolObject<dyn MTLDevice>, elements: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    shared_buffer(device, &f32_bytes(&vec![0.0_f32; elements]))
}

fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
    let pointer = buffer.contents().as_ptr().cast::<f32>();
    unsafe { core::slice::from_raw_parts(pointer, count) }.to_vec()
}

fn compile(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
    math_mode: MTLMathMode,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(math_mode);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .unwrap_or_else(|error| panic!("compiles {entry}: {}", error.localizedDescription()));
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .unwrap_or_else(|| panic!("kernel entry `{entry}` missing from its own compiled library"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("creates the pipeline for {entry}: {}", error.localizedDescription()))
}

// one compiled node ready to encode: pipeline, its own uniforms buffer, and
// its grid -- built once, reused across every timing iteration so the
// timed region is encode+dispatch+wait only, never `newLibraryWithSource`.
struct CompiledNode {
    id: u32,
    entry: String,
    bindings: Vec<Binding>,
    grid: GridSpec,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    elements: usize,
}

fn compile_node(
    device: &ProtocolObject<dyn MTLDevice>,
    nodes_dir: &Path,
    vectors_dir: &Path,
    id: u32,
    math_mode: MTLMathMode,
) -> CompiledNode {
    let spec = read_node_spec(nodes_dir, id);
    let source = std::fs::read_to_string(nodes_dir.join(format!("{id}.metal")))
        .unwrap_or_else(|error| panic!("read nodes/{id}.metal: {error}"));
    let uniforms_bytes = parse_uniforms_hex(&nodes_dir.join(format!("{id}.uniforms.hex")));
    let elements = parse_bits_file(&vectors_dir.join(format!("{id}.bits"))).len();
    let pipeline = compile(device, &source, &spec.entry, math_mode);
    let uniforms = shared_buffer(device, &uniforms_bytes);
    CompiledNode { id, entry: spec.entry, bindings: spec.bindings, grid: spec.grid, pipeline, uniforms, elements }
}

// resolves a node's Input bindings against either a captured leaf
// (`<vectors_dir>/<node>.bits`, read once and cached) or an earlier chain
// node's own output buffer (`chain_outputs`), so the intermediate values
// flowing between dispatches are the CHAIN's own GPU-produced bytes, never
// the captured reference (that would hide a real fusion bug behind the
// oracle's bit pattern).
fn encode_chain(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    device: &ProtocolObject<dyn MTLDevice>,
    vectors_dir: &Path,
    compiled: &[CompiledNode],
    leaf_cache: &mut ChainOutputs,
) -> (Retained<ProtocolObject<dyn MTLCommandBuffer>>, ChainOutputs) {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let mut chain_outputs: ChainOutputs = HashMap::new();

    for node in compiled {
        let output_id = node
            .bindings
            .iter()
            .find_map(|binding| match binding {
                Binding::Output(id) => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| panic!("node {} carries no Output binding", node.id));
        let output_buffer = zeroed_buffer(device, node.elements);

        let encoder = command_buffer
            .computeCommandEncoder()
            .unwrap_or_else(|| panic!("compute encoder for node {} ({})", node.id, node.entry));
        encoder.setComputePipelineState(&node.pipeline);
        for (index, binding) in node.bindings.iter().enumerate() {
            let buffer: &ProtocolObject<dyn MTLBuffer> = match binding {
                Binding::Output(_) => &output_buffer,
                Binding::Uniforms => &node.uniforms,
                Binding::Input(source_id) => {
                    if let Some(buffer) = chain_outputs.get(source_id) {
                        buffer
                    } else {
                        leaf_cache.entry(*source_id).or_insert_with(|| {
                            let bits = parse_bits_file(&vectors_dir.join(format!("{source_id}.bits")));
                            shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
                        })
                    }
                }
            };
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        let max_threadgroup = node.pipeline.maxTotalThreadsPerThreadgroup();
        let threadgroup_width = match node.grid.threadgroup_width {
            Some(width) => width.min(max_threadgroup).max(1),
            None => node.grid.threads.min(max_threadgroup).max(1),
        };
        let grid_size = MTLSize { width: node.grid.threads, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
        encoder.endEncoding();

        chain_outputs.insert(output_id, output_buffer);
    }
    (command_buffer, chain_outputs)
}

struct TimingResult {
    median_us: f64,
    p90_us: f64,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
    sorted[index.min(sorted.len().saturating_sub(1))]
}

fn time_chain(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    device: &ProtocolObject<dyn MTLDevice>,
    vectors_dir: &Path,
    compiled: &[CompiledNode],
    warmup: usize,
    iterations: usize,
) -> (TimingResult, HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>) {
    let mut leaf_cache: HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>> = HashMap::new();
    let mut last_outputs = None;

    for _ in 0..warmup {
        let (command_buffer, outputs) = encode_chain(queue, device, vectors_dir, compiled, &mut leaf_cache);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        last_outputs = Some(outputs);
    }

    let mut samples_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (command_buffer, outputs) = encode_chain(queue, device, vectors_dir, compiled, &mut leaf_cache);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let start = command_buffer.GPUStartTime();
        let end = command_buffer.GPUEndTime();
        samples_us.push((end - start) * 1_000_000.0);
        last_outputs = Some(outputs);
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).expect("GPU duration samples are finite"));
    let result = TimingResult { median_us: percentile(&samples_us, 0.5), p90_us: percentile(&samples_us, 0.9) };
    (result, last_outputs.expect("at least one iteration ran (warmup or timed)"))
}

fn gate_final(
    label: &str,
    vectors_dir: &Path,
    outputs: &HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>,
    final_node: u32,
    elements: usize,
) -> bool {
    let reference = parse_bits_file(&vectors_dir.join(format!("{final_node}.bits")));
    let buffer = outputs
        .get(&final_node)
        .unwrap_or_else(|| panic!("{label} chain produced no node {final_node} output"));
    let produced = read_f32_buffer(buffer, elements);
    let mut first_diff = None;
    for (index, (produced_value, reference_bits)) in produced.iter().zip(reference.iter()).enumerate() {
        let produced_bits = produced_value.to_bits();
        if produced_bits != *reference_bits {
            first_diff = Some((index, *reference_bits, produced_bits));
            break;
        }
    }
    match first_diff {
        None => {
            println!("chainB_gate chain={label} elements={elements} match=true");
            true
        }
        Some((index, expected, produced)) => {
            println!(
                "chainB_gate chain={label} elements={elements} match=false first_diff=({index}, 0x{expected:08x}, 0x{produced:08x})"
            );
            false
        }
    }
}

// prints `softmax_gate ctx=<n> role=<154|157|158|164> elements=<e>
// match=true|first_diff=(idx, 0x.., 0x..)` for one role's produced buffer
// against its captured reference (`gate_final`'s own shape, generalized
// off a fixed `final_node` to an arbitrary role/id pair since this
// function checks four roles per context, not one).
fn gate_softmax_role(
    ctx_label: &str,
    role: &str,
    vectors_dir: &Path,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    node_id: u32,
    elements: usize,
) -> bool {
    let reference = parse_bits_file(&vectors_dir.join(format!("{node_id}.bits")));
    let produced = read_f32_buffer(buffer, elements);
    let mut first_diff = None;
    for (index, (produced_value, reference_bits)) in produced.iter().zip(reference.iter()).enumerate() {
        let produced_bits = produced_value.to_bits();
        if produced_bits != *reference_bits {
            first_diff = Some((index, *reference_bits, produced_bits));
            break;
        }
    }
    match first_diff {
        None => {
            println!("softmax_gate ctx={ctx_label} role={role} elements={elements} match=true");
            true
        }
        Some((index, expected, produced)) => {
            println!(
                "softmax_gate ctx={ctx_label} role={role} elements={elements} match=false first_diff=({index}, 0x{expected:08x}, 0x{produced:08x})"
            );
            false
        }
    }
}

// coordinator's own integration item (`omega::msl::render_cached_softmax_
// weights`, `omega/src/msl/cached_softmax_weights_render.rs`): a
// RE-TRANSCRIPTION of this file's own gated `k_softmax_fused_source`,
// tested only for determinism upstream (`cached_softmax_weights_render_is_
// deterministic_and_width_dependent`) -- this is that renderer's fixture
// gate, dispatching `omega::emit`'s own compiled `Kernel` in complete
// isolation (no chain, captured leaves bound directly) against the SAME
// captured 154/157/158/164 bits this file's own `build_b_chain_setup`
// path gates. `BoundOp` shape (operands' `Layout`, `cached_key_rows`,
// `attention_rows`, `head_dim`) is read from the REAL dumped grid.txt
// files at `order`'s own role positions, matching `build_b_chain_setup`'s
// `library_softmax` construction exactly so the two paths can never drift
// apart on shape.
fn run_softmax_library_gate(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    nodes_dir: &Path,
    vectors_dir: &Path,
    order: &[u32],
    math_mode: MTLMathMode,
    ctx_label: &str,
) -> bool {
    let elements = |id: u32| parse_bits_file(&vectors_dir.join(format!("{id}.bits"))).len();
    let attention_rows = elements(order[6]);
    let cached_key_rows = elements(order[8]) / attention_rows.max(1);
    let head_dim = elements(order[13]) / attention_rows.max(1);
    let new_value_id = read_node_spec(nodes_dir, order[13])
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::Input(id) if *id != order[10] => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("164-role node {} carries no new-key-V leaf Input", order[13]));

    let bound_op = BoundOp {
        node: NodeId(order[8]),
        dtype: DType::Float32,
        extents: vec![cached_key_rows as u64, attention_rows as u64],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands: vec![
                (NodeId(order[4]), Layout { base: 0, strides: vec![attention_rows as i64, 1].into() }, None),
                (NodeId(order[5]), Layout { base: 0, strides: vec![1i64].into() }, None),
                (NodeId(new_value_id), Layout { base: 0, strides: vec![0i64, 1].into() }, None),
            ],
            cached_weight_sum: NodeId(order[9]),
            new_weight_sum: NodeId(order[11]),
            new_attended: NodeId(order[13]),
            cached_key_rows: cached_key_rows as u64,
            new_key_rows: 1,
            query_rows: attention_rows as u64,
            attention_rows: attention_rows as u64,
            head_dim: head_dim as u64,
        },
    };
    let kernel = omega_emit(&bound_op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
        .unwrap_or_else(|error| panic!("omega::emit(CachedSoftmaxWeights) failed: {error:?}"));
    if let Ok(dump_dir) = std::env::var("SOFTMAX_DUMP_DIR") {
        let _ = std::fs::write(format!("{dump_dir}/{ctx_label}_dispatched.metal"), &kernel.source);
    }
    let library = compile_library(device, &kernel.source, math_mode);
    let pipeline = pipeline_from_library(device, &library, &kernel.entry);

    let leaf = |id: u32| -> MetalBuffer {
        let bits = parse_bits_file(&vectors_dir.join(format!("{id}.bits")));
        shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
    };
    let cached_scores = leaf(order[4]);
    let new_scores = leaf(order[5]);
    let new_value = leaf(new_value_id);
    let uniforms = shared_buffer(device, &(kernel.grid.threads as i64).to_le_bytes());

    let out154 = zeroed_buffer(device, elements(order[8]));
    let out157 = zeroed_buffer(device, elements(order[9]));
    let out158 = zeroed_buffer(device, elements(order[11]));
    let out164 = zeroed_buffer(device, elements(order[13]));

    let command_buffer = queue.commandBuffer().expect("command buffer");
    dispatch_encoder(
        &command_buffer,
        &pipeline,
        &[&cached_scores, &new_scores, &new_value, &out154, &uniforms, &out157, &out158, &out164],
        kernel.grid.threads as usize,
        kernel.grid.threadgroup_width.unwrap_or(kernel.grid.threads) as usize,
    );
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let checks = [
        ("154", &out154, order[8], elements(order[8])),
        ("157", &out157, order[9], elements(order[9])),
        ("158", &out158, order[11], elements(order[11])),
        ("164", &out164, order[13], elements(order[13])),
    ];
    let mut all_match = true;
    for (role, buffer, node_id, count) in checks {
        all_match &= gate_softmax_role(ctx_label, role, vectors_dir, buffer, node_id, count);
    }
    all_match
}

// ---------------------------------------------------------------------
// Candidate B fused chain: 134/135/139/146/142/149/162 stay production
// dispatches (verbatim, unmodified); 151/152/154/156/157/158/164 fuse into
// ONE kernel (`k_softmax_fused`); 166's math is appended as an epilogue on
// 162's own store site (one text splice, `k_av_epilogue`). Every statement
// below is transcribed from the real dumped bodies (R9/nodes/{151,152,154,
// 156,157,158,162,164,166}.metal, read in full above) -- see
// R9/PROGRESS.md "Candidate B" for the per-stage native-gid derivation this
// mirrors. Roles are read positionally from `chain_node_order()` (index 6
// = 151-role, 7 = 152-role, 8 = 154-role, 9 = 157-role, 10 = 156-role,
// 11 = 158-role, 13 = 164-role), not hardcoded node ids, so this runs
// unchanged against any layer's dump.

// declares each stage's OWN `Uniforms` struct shape verbatim from its
// dump (they differ by operand/epilogue count -- one struct definition
// cannot serve all seven), renamed to avoid collision in one translation
// unit.
const FUSED_STRUCTS: &str = r"
#include <metal_stdlib>
using namespace metal;

struct U151 { long output_total; long reduction_total; long output_extents[3]; long reduction_extents[1]; long operand_base[1]; long operand_strides[1][4]; long out_base; long out_strides[4]; };
struct U152 { long output_total; long reduction_total; long output_extents[3]; long reduction_extents[1]; long operand_base[1]; long operand_strides[1][4]; long out_base; long out_strides[4]; long epilogue_operand_base[1]; long epilogue_operand_strides[1][3]; };
struct U154 { long total_elements; long extents[4]; long operand_base[2]; long operand_strides[2][4]; };
struct U156 { long total_elements; long extents[4]; long operand_base[2]; long operand_strides[2][4]; };
struct U157 { long output_total; long reduction_total; long output_extents[3]; long reduction_extents[1]; long operand_base[1]; long operand_strides[1][4]; long out_base; long out_strides[4]; };
struct U158 { long output_total; long reduction_total; long output_extents[3]; long reduction_extents[1]; long operand_base[1]; long operand_strides[1][4]; long out_base; long out_strides[4]; };
struct U164 { long output_total; long reduction_total; long output_extents[4]; long reduction_extents[1]; long operand_base[2]; long operand_strides[2][5]; long out_base; long out_strides[5]; };
";

const STAGE_151: &str = r"
    // -- 151 (R9/nodes/151.metal verbatim, gid = tg) --
    if (local == 0u) {
        long gid = (long)tg;
        long full_coord[4]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0;
        long output_coord[3];
        long remaining = gid;
        output_coord[2] = remaining % u151.output_extents[2]; remaining /= u151.output_extents[2];
        output_coord[1] = remaining % u151.output_extents[1]; remaining /= u151.output_extents[1];
        output_coord[0] = remaining % u151.output_extents[0]; remaining /= u151.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        float accumulator = -INFINITY;
        bool seeded = true;
        for (long r = 0; r < u151.reduction_total; r++) {
            long reduction_coord0 = r % u151.reduction_extents[0];
            full_coord[1] = reduction_coord0;
            long off0 = u151.operand_base[0];
            off0 += full_coord[0] * u151.operand_strides[0][0];
            off0 += full_coord[1] * u151.operand_strides[0][1];
            off0 += full_coord[2] * u151.operand_strides[0][2];
            off0 += full_coord[3] * u151.operand_strides[0][3];
            float value = in149[off0];
            accumulator = seeded ? max(accumulator, value) : value;
            seeded = true;
        }
        long out_offset = u151.out_base;
        out_offset += full_coord[0] * u151.out_strides[0];
        out_offset += full_coord[1] * u151.out_strides[1];
        out_offset += full_coord[2] * u151.out_strides[2];
        out_offset += full_coord[3] * u151.out_strides[3];
        out151[out_offset] = accumulator;
    }
    threadgroup_barrier(mem_flags::mem_device);
";

const STAGE_156: &str = r"
    // -- 156 (R9/nodes/156.metal verbatim, gid = tg) --
    if (local == 0u) {
        long gid = (long)tg;
        uint coord0; uint coord2; uint coord3;
        uint remaining = (uint)gid;
        coord3 = remaining % (uint)u156.extents[3]; remaining /= (uint)u156.extents[3];
        coord2 = remaining % (uint)u156.extents[2]; remaining /= (uint)u156.extents[2];
        remaining /= (uint)u156.extents[1];
        coord0 = remaining % (uint)u156.extents[0]; remaining /= (uint)u156.extents[0];
        long off0 = u156.operand_base[0] + gid;
        long off1 = u156.operand_base[1];
        off1 += (long)coord0 * u156.operand_strides[1][0];
        off1 += (long)coord2 * u156.operand_strides[1][2];
        off1 += (long)coord3 * u156.operand_strides[1][3];
        float step0 = (in149[off0] - out152[off1]);
        out156[gid] = exp(step0);
    }
    threadgroup_barrier(mem_flags::mem_device);
";

const STAGE_158: &str = r"
    // -- 158 (R9/nodes/158.metal verbatim, gid = tg) --
    if (local == 0u) {
        long gid = (long)tg;
        long full_coord[4]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0;
        long output_coord[3];
        long remaining = gid;
        output_coord[2] = remaining % u158.output_extents[2]; remaining /= u158.output_extents[2];
        output_coord[1] = remaining % u158.output_extents[1]; remaining /= u158.output_extents[1];
        output_coord[0] = remaining % u158.output_extents[0]; remaining /= u158.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        float accumulator = 0.0f;
        bool seeded = true;
        for (long r = 0; r < u158.reduction_total; r++) {
            long reduction_coord0 = r % u158.reduction_extents[0];
            full_coord[1] = reduction_coord0;
            long off0 = u158.operand_base[0];
            off0 += full_coord[0] * u158.operand_strides[0][0];
            off0 += full_coord[1] * u158.operand_strides[0][1];
            off0 += full_coord[2] * u158.operand_strides[0][2];
            off0 += full_coord[3] * u158.operand_strides[0][3];
            float value = out156[off0];
            accumulator = seeded ? (accumulator + value) : value;
            seeded = true;
        }
        long out_offset = u158.out_base;
        out_offset += full_coord[0] * u158.out_strides[0];
        out_offset += full_coord[1] * u158.out_strides[1];
        out_offset += full_coord[2] * u158.out_strides[2];
        out_offset += full_coord[3] * u158.out_strides[3];
        out158[out_offset] = accumulator;
    }
    threadgroup_barrier(mem_flags::mem_device);
";

// stage 154 participation bound generalizes from the hardcoded `< 32u`
// (only valid when key_count == 32) to `< {key_count}u` -- the native gid
// formula (`local*8+tg`, verified against R9/nodes/154.metal's own
// coordinate decomposition) is unchanged; only how many `local` values
// carry real keys changes with capacity.
fn stage154(key_count: u64) -> String {
    format!(
        r"
    // -- 154 (R9/nodes/154.metal verbatim, native gid = local*8+tg) --
    if (local < {key_count}u) {{
        long gid = (long)local * 8 + (long)tg;
        uint coord0; uint coord2; uint coord3;
        uint remaining = (uint)gid;
        coord3 = remaining % (uint)u154.extents[3]; remaining /= (uint)u154.extents[3];
        coord2 = remaining % (uint)u154.extents[2]; remaining /= (uint)u154.extents[2];
        remaining /= (uint)u154.extents[1];
        coord0 = remaining % (uint)u154.extents[0]; remaining /= (uint)u154.extents[0];
        long off0 = u154.operand_base[0] + gid;
        long off1 = u154.operand_base[1];
        off1 += (long)coord0 * u154.operand_strides[1][0];
        off1 += (long)coord2 * u154.operand_strides[1][2];
        off1 += (long)coord3 * u154.operand_strides[1][3];
        float step0 = (in142[off0] - out152[off1]);
        out154[gid] = exp(step0);
    }}
",
        key_count = key_count,
    )
}

// stage 164 gains a `local < {dim_count}u` guard it did not need before:
// previously the fused kernel's threadgroup was always exactly
// `dim_count` wide, so every thread was a real (group, dim) pair. Once
// the threadgroup widens to also cover `key_count`/`key_width` (capacity
// 512, where key_count > dim_count for layer 0), threads beyond
// `dim_count` are spectators here and must not compute a bogus
// (group, dim) pair or write outside the real output range.
fn stage164(dim_count: u64) -> String {
    format!(
        r"
    // -- 164 (R9/nodes/164.metal verbatim, native gid = tg*{dim_count}+local) --
    if (local < {dim_count}u) {{
        long gid = (long)tg * {dim_count} + (long)local;
        long full_coord[5]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0; full_coord[4]=0;
        long output_coord[4];
        long remaining = gid;
        output_coord[3] = remaining % u164.output_extents[3]; remaining /= u164.output_extents[3];
        output_coord[2] = remaining % u164.output_extents[2]; remaining /= u164.output_extents[2];
        output_coord[1] = remaining % u164.output_extents[1]; remaining /= u164.output_extents[1];
        output_coord[0] = remaining % u164.output_extents[0]; remaining /= u164.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        full_coord[4] = output_coord[3];
        float accumulator = 0.0f;
        bool seeded = true;
        for (long r = 0; r < u164.reduction_total; r++) {{
            long reduction_coord0 = r % u164.reduction_extents[0];
            full_coord[1] = reduction_coord0;
            long off0 = u164.operand_base[0];
            off0 += full_coord[0] * u164.operand_strides[0][0];
            off0 += full_coord[1] * u164.operand_strides[0][1];
            off0 += full_coord[2] * u164.operand_strides[0][2];
            off0 += full_coord[3] * u164.operand_strides[0][3];
            off0 += full_coord[4] * u164.operand_strides[0][4];
            long off1 = u164.operand_base[1];
            off1 += full_coord[0] * u164.operand_strides[1][0];
            off1 += full_coord[1] * u164.operand_strides[1][1];
            off1 += full_coord[2] * u164.operand_strides[1][2];
            off1 += full_coord[3] * u164.operand_strides[1][3];
            off1 += full_coord[4] * u164.operand_strides[1][4];
            float step0 = (in127[off0] * out156[off1]);
            accumulator = seeded ? (accumulator + step0) : step0;
            seeded = true;
        }}
        long out_offset = u164.out_base;
        out_offset += full_coord[0] * u164.out_strides[0];
        out_offset += full_coord[1] * u164.out_strides[1];
        out_offset += full_coord[2] * u164.out_strides[2];
        out_offset += full_coord[3] * u164.out_strides[3];
        out_offset += full_coord[4] * u164.out_strides[4];
        out164[out_offset] = accumulator;
    }}
",
        dim_count = dim_count,
    )
}

// stage 152/157's cooperative reduce: at `key_width == 32` (a single
// physical simdgroup) transcribed exactly as R9/nodes/{152,157}.metal show
// at capacity 32 -- `simd_{max,sum}(accumulator)` directly, no
// threadgroup memory, no internal barrier. At `key_width > 32` (multiple
// simdgroups cooperating) transcribed exactly as the REAL capacity-512
// dump shows (R9/candidate_b/ctx3_l0_c512/nodes/{152,157}.metal, read
// verbatim, not re-derived): `threadgroup float partials[key_width/32]`,
// each simdgroup's lane 0 stores its own `simd_{max,sum}` there, ONE
// `threadgroup_barrier(mem_flags::mem_threadgroup)`, then lane 0 folds
// `partials[]` serially. That internal barrier sits OUTSIDE the
// `local < key_width` participation guard so every thread in the fused
// kernel's wider threadgroup (sized for 154/164's own bounds, not just
// this stage's `key_width`) reaches it uniformly; threads outside the
// guard never touch `partials[]`.
struct CooperativeStageSpec<'a> {
    node: &'a str,
    reduce_fn: &'a str,
    seed: &'a str,
    accumulate_line: &'a str,
    epilogue: &'a str,
    read_buffer: &'a str,
    uniforms: &'a str,
    key_width: u64,
}

fn cooperative_stage(spec: CooperativeStageSpec) -> String {
    let CooperativeStageSpec { node, reduce_fn, seed, accumulate_line, epilogue, read_buffer, uniforms, key_width } =
        spec;
    if key_width == 32 {
        format!(
            r"
    // -- {node} (R9/nodes/{node}.metal verbatim, gid = tg*32+local, native) --
    if (local < 32u) {{
        long gid = (long)tg * 32 + (long)local;
        long output_index = gid / 32;
        uint lane = (uint)(gid % 32);
        long full_coord[4]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0;
        long output_coord[3];
        long remaining = output_index;
        output_coord[2] = remaining % {uniforms}.output_extents[2]; remaining /= {uniforms}.output_extents[2];
        output_coord[1] = remaining % {uniforms}.output_extents[1]; remaining /= {uniforms}.output_extents[1];
        output_coord[0] = remaining % {uniforms}.output_extents[0]; remaining /= {uniforms}.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        float accumulator = {seed};
        bool seeded = true;
        long stride0 = {uniforms}.operand_strides[0][1];
        long off0 = {uniforms}.operand_base[0];
        off0 += full_coord[0] * {uniforms}.operand_strides[0][0];
        off0 += full_coord[2] * {uniforms}.operand_strides[0][2];
        off0 += full_coord[3] * {uniforms}.operand_strides[0][3];
        off0 += (long)lane * stride0;
        int walk0 = (int)off0;
        int advance0 = (int)(stride0 * 32);
        for (int r = (int)lane; r < (int){uniforms}.reduction_total; r += 32) {{
            float value = {read_buffer}[walk0];
            {accumulate_line}
            walk0 += advance0;
        }}
        float reduced = {reduce_fn}(accumulator);
        if (lane == 0u) {{
{epilogue}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_device);
",
            node = node,
            uniforms = uniforms,
            seed = seed,
            read_buffer = read_buffer,
            accumulate_line = accumulate_line,
            reduce_fn = reduce_fn,
            epilogue = epilogue,
        )
    } else {
        let simdgroups = key_width / 32;
        format!(
            r"
    // -- {node} (R9/candidate_b/ctx3_l0_c512/nodes/{node}.metal verbatim wide combine, gid = tg*{key_width}+local, native) --
    threadgroup float partials_{node}[{simdgroups}];
    if (local < {key_width}u) {{
        long gid = (long)tg * {key_width} + (long)local;
        long output_index = gid / {key_width};
        uint lane = (uint)(gid % {key_width});
        long full_coord[4]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0;
        long output_coord[3];
        long remaining = output_index;
        output_coord[2] = remaining % {uniforms}.output_extents[2]; remaining /= {uniforms}.output_extents[2];
        output_coord[1] = remaining % {uniforms}.output_extents[1]; remaining /= {uniforms}.output_extents[1];
        output_coord[0] = remaining % {uniforms}.output_extents[0]; remaining /= {uniforms}.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        float accumulator = {seed};
        bool seeded = true;
        long stride0 = {uniforms}.operand_strides[0][1];
        long off0 = {uniforms}.operand_base[0];
        off0 += full_coord[0] * {uniforms}.operand_strides[0][0];
        off0 += full_coord[2] * {uniforms}.operand_strides[0][2];
        off0 += full_coord[3] * {uniforms}.operand_strides[0][3];
        off0 += (long)lane * stride0;
        int walk0 = (int)off0;
        int advance0 = (int)(stride0 * {key_width});
        for (int r = (int)lane; r < (int){uniforms}.reduction_total; r += {key_width}) {{
            float value = {read_buffer}[walk0];
            {accumulate_line}
            walk0 += advance0;
        }}
        float partial = {reduce_fn}(accumulator);
        if (lane % 32u == 0u) {{ partials_{node}[lane / 32u] = partial; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local < {key_width}u) {{
        long output_index = ((long)tg * {key_width} + (long)local) / {key_width};
        uint lane = (uint)local;
        long full_coord[4]; full_coord[0]=0; full_coord[1]=0; full_coord[2]=0; full_coord[3]=0;
        long output_coord[3];
        long remaining = output_index;
        output_coord[2] = remaining % {uniforms}.output_extents[2]; remaining /= {uniforms}.output_extents[2];
        output_coord[1] = remaining % {uniforms}.output_extents[1]; remaining /= {uniforms}.output_extents[1];
        output_coord[0] = remaining % {uniforms}.output_extents[0]; remaining /= {uniforms}.output_extents[0];
        full_coord[0] = output_coord[0];
        full_coord[2] = output_coord[1];
        full_coord[3] = output_coord[2];
        if (lane == 0u) {{
            float reduced = partials_{node}[0];
            for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{
                reduced = {fold_expr};
            }}
{epilogue}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_device);
",
            node = node,
            uniforms = uniforms,
            seed = seed,
            read_buffer = read_buffer,
            accumulate_line = accumulate_line,
            reduce_fn = reduce_fn,
            key_width = key_width,
            simdgroups = simdgroups,
            fold_expr = fold_expr_for(node, reduce_fn),
            epilogue = epilogue,
        )
    }
}

fn fold_expr_for(node: &str, reduce_fn: &str) -> String {
    let node_var = format!("partials_{node}[fold_index]");
    if reduce_fn == "simd_max" {
        format!("max(reduced, {node_var})")
    } else {
        format!("(reduced + {node_var})")
    }
}

// `dim_count`/`key_count`/`key_width` are baked as literals (matching how
// production bakes its own shape constants at emit time, e.g. the
// `advance0 = stride0*32` literal seen in every dumped body above) -- one
// query group's threadgroup is sized `max(dim_count, key_count,
// key_width)` lanes wide so it can host 152/157's cooperative reduce
// (lanes < key_width), 154/156's per-key work (lanes < key_count), and
// 164's per-dim AV product (lanes < dim_count) in the same threadgroup.
fn k_softmax_fused_source(dim_count: u64, key_count: u64, key_width: u64) -> String {
    let mut body = String::new();
    body.push_str(
        r"
kernel void k_softmax_fused(
    device const float* in142 [[buffer(0)]],
    device const float* in149 [[buffer(1)]],
    device const float* in127 [[buffer(2)]],
    device float* out151 [[buffer(3)]],
    device float* out152 [[buffer(4)]],
    device float* out154 [[buffer(5)]],
    device float* out156 [[buffer(6)]],
    device float* out157 [[buffer(7)]],
    device float* out158 [[buffer(8)]],
    device float* out164 [[buffer(9)]],
    constant U151& u151 [[buffer(10)]],
    constant U152& u152 [[buffer(11)]],
    constant U154& u154 [[buffer(12)]],
    constant U156& u156 [[buffer(13)]],
    constant U157& u157 [[buffer(14)]],
    constant U158& u158 [[buffer(15)]],
    constant U164& u164 [[buffer(16)]],
    uint local [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
",
    );
    body.push_str(STAGE_151);
    body.push_str(&cooperative_stage(CooperativeStageSpec {
        node: "152",
        reduce_fn: "simd_max",
        seed: "-INFINITY",
        accumulate_line: "accumulator = seeded ? max(accumulator, value) : value;\n            seeded = true;",
        epilogue: "            long out_offset = u152.out_base;\n            out_offset += full_coord[0] * u152.out_strides[0];\n            out_offset += full_coord[1] * u152.out_strides[1];\n            out_offset += full_coord[2] * u152.out_strides[2];\n            out_offset += full_coord[3] * u152.out_strides[3];\n            long epi_off0 = u152.epilogue_operand_base[0];\n            epi_off0 += output_coord[0] * u152.epilogue_operand_strides[0][0];\n            epi_off0 += output_coord[1] * u152.epilogue_operand_strides[0][1];\n            epi_off0 += output_coord[2] * u152.epilogue_operand_strides[0][2];\n            float epi0_value = out151[epi_off0];\n            float epi_step1 = reduced;\n            float epi_step2 = max(epi_step1, epi0_value);\n            out152[out_offset] = epi_step2;",
        read_buffer: "in142",
        uniforms: "u152",
        key_width,
    }));
    body.push_str(&stage154(key_count));
    body.push_str(STAGE_156);
    body.push_str(&cooperative_stage(CooperativeStageSpec {
        node: "157",
        reduce_fn: "simd_sum",
        seed: "0.0f",
        accumulate_line: "accumulator = seeded ? (accumulator + value) : value;\n            seeded = true;",
        epilogue: "            long out_offset = u157.out_base;\n            out_offset += full_coord[0] * u157.out_strides[0];\n            out_offset += full_coord[1] * u157.out_strides[1];\n            out_offset += full_coord[2] * u157.out_strides[2];\n            out_offset += full_coord[3] * u157.out_strides[3];\n            out157[out_offset] = reduced;",
        read_buffer: "out154",
        uniforms: "u157",
        key_width,
    }));
    body.push_str(STAGE_158);
    body.push_str(&stage164(dim_count));
    body.push_str("}\n");
    body
}

// splices 166's math (R9/nodes/166.metal, its actual statement order:
// `step0=(157+158); step1=1/step0; step2=(162+164); step3=step1*step2`,
// NOT a paraphrase) onto 162's own unmodified body at its `if (lane==0u)`
// store site; 157/158/164 read as plain flat arrays since 164 and 162
// share node 166's identical (group,dim) output linearization (162's own
// `out_offset` equals that same flat index) and 157/158 are 8-element
// per-group scalars indexed by `out_offset / {dim_count}`.
fn splice_162_epilogue(node162_source: &str, dim_count: u64) -> String {
    let needle = "device float* out [[buffer(2)]],\n    constant Uniforms& u [[buffer(3)]],\n    uint gid [[thread_position_in_grid]])";
    let replacement = "device float* out [[buffer(2)]],\n    constant Uniforms& u [[buffer(3)]],\n    device const float* in157 [[buffer(4)]],\n    device const float* in158 [[buffer(5)]],\n    device const float* in164 [[buffer(6)]],\n    uint gid [[thread_position_in_grid]])";
    let with_signature = node162_source.replacen(needle, replacement, 1);
    assert!(with_signature != node162_source, "162.metal signature did not match the expected splice point");

    let store_needle = "out[out_offset] = reduced;";
    let store_replacement = format!(
        "long epi_group = out_offset / {dim_count};\n        float epi_step0 = (in157[epi_group] + in158[epi_group]);\n        float epi_step1 = (1.0f / epi_step0);\n        float epi_step2 = (reduced + in164[out_offset]);\n        float epi_step3 = (epi_step1 * epi_step2);\n        out[out_offset] = epi_step3;"
    );
    let spliced = with_signature.replacen(store_needle, &store_replacement, 1);
    assert!(spliced != with_signature, "162.metal store site did not match the expected splice point");
    spliced
}

struct BChainOutcome {
    dispatches: usize,
    timing: TimingResult,
    gate: bool,
    intermediate_gates: Vec<(u32, bool)>,
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    math_mode: MTLMathMode,
) -> Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(math_mode);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .unwrap_or_else(|error| panic!("compiles fused library: {}", error.localizedDescription()))
}

fn pipeline_from_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    entry: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .unwrap_or_else(|| panic!("kernel entry `{entry}` missing from the fused library"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("creates the pipeline for {entry}: {}", error.localizedDescription()))
}

fn dispatch_encoder(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    grid_threads: usize,
    threadgroup_width: usize,
) {
    let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
    encoder.setComputePipelineState(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let width = threadgroup_width.min(max_threadgroup).max(1);
    let grid_size = MTLSize { width: grid_threads, height: 1, depth: 1 };
    let threadgroup = MTLSize { width, height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
    encoder.endEncoding();
}

#[allow(clippy::too_many_lines)]
// everything about the B chain that is invariant across iterations
// (compiled pipelines, uniform buffers, resolved leaf ids, grid shapes) --
// built once, reused by both `run_b_chain` (standalone timing) and
// `run_interleaved` (clock-state-robust timing) so the two report
// identical kernels, not two independently-drifted copies.
// The final combine stage splices 166's math onto 162's own unmodified
// body as an epilogue (`splice_162_epilogue`) -- node 162-role is a plain
// reduce and 166-role a separate elementwise combine in the real bound
// plan at every bucket this harness gates against (R9/PROGRESS.md,
// "Candidate B, contexts 3/4, RE-CAPTURED AGAINST HEAD BYTES": a prior
// slice's capacity-512 capture showed 162 fused away by a bind change that
// did not survive -- reverted, so that shape no longer occurs here).
struct FinalStage {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    grid: GridSpec,
    in82_id: u32,
}

// `omega::msl::render_cached_softmax_weights`'s own compiled dispatch: one
// kernel call replaces this file's staged `k_softmax_fused` entirely (151/
// 152/156 are register-only inside the library kernel, never separate
// buffers) and produces the SAME four outputs (154/157/158/164) the spliced
// 162+166 final stage already reads -- so swapping this in for
// `k_softmax_pipeline` needs no change downstream (`encode_b_chain`'s
// final-stage dispatch is untouched either way). Gated by `--softmax-from-
// library 1` (coordinator's own integration item; harness's `k_softmax_
// fused_source` stays the default so a regression here never touches the
// already-gated path).
struct LibrarySoftmaxKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    grid_threads: usize,
    threadgroup_width: usize,
}

struct BChainSetup {
    compiled_kept: Vec<CompiledNode>,
    k_softmax_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    library_softmax: Option<LibrarySoftmaxKernel>,
    final_stage: FinalStage,
    u151: Retained<ProtocolObject<dyn MTLBuffer>>,
    u152: Retained<ProtocolObject<dyn MTLBuffer>>,
    u154: Retained<ProtocolObject<dyn MTLBuffer>>,
    u156: Retained<ProtocolObject<dyn MTLBuffer>>,
    u157: Retained<ProtocolObject<dyn MTLBuffer>>,
    u158: Retained<ProtocolObject<dyn MTLBuffer>>,
    u164: Retained<ProtocolObject<dyn MTLBuffer>>,
    e151: usize,
    e152: usize,
    e154: usize,
    e156: usize,
    e157: usize,
    e158: usize,
    e164: usize,
    threadgroup_width: usize,
    grid_threads: usize,
    id142: u32,
    id149: u32,
    id151: u32,
    id152: u32,
    id154: u32,
    id156: u32,
    id157: u32,
    id158: u32,
    id164: u32,
    in127_id: u32,
    final_node: u32,
    final_elements: usize,
    dispatches: usize,
}

fn build_b_chain_setup(
    device: &ProtocolObject<dyn MTLDevice>,
    nodes_dir: &Path,
    vectors_dir: &Path,
    order: &[u32],
    math_mode: MTLMathMode,
    use_library_softmax: bool,
) -> BChainSetup {
    // production-kept nodes (unchanged dispatches): 134,135,139,146,142,149
    // by role position in `order` (see chain_node_order's doc comment).
    // 162-role is NOT dispatched here -- it is superseded by
    // `epilogue_pipeline` below (162's own body with 166's epilogue
    // spliced on); dispatching the unmodified 162 as well would be wasted,
    // discarded GPU work inflating both dispatch count and B's timing.
    let kept_roles = [order[0], order[1], order[2], order[3], order[4], order[5]];
    let final_node = order[14];
    let dim_count = parse_bits_file(&vectors_dir.join(format!("{}.bits", order[13]))).len()
        / parse_bits_file(&vectors_dir.join(format!("{}.bits", order[11]))).len().max(1);

    let compiled_kept: Vec<CompiledNode> =
        kept_roles.iter().map(|&id| compile_node(device, nodes_dir, vectors_dir, id, math_mode)).collect();

    // `key_width` (152/157's cooperative reduce width) is read from the
    // REAL dumped grid.txt -- `wide_cooperative_reduce_width` evaluated at
    // this bucket's actual `reduction_total` (32 at bucket 32, 128 at
    // bucket 512, omega/src/msl/tiled_gemm_cooperative_scan.rs:620) --
    // never recomputed independently, so this kernel can never drift onto
    // a width the dumped bytes it is gated against did not use.
    let key_width = read_node_spec(nodes_dir, order[7])
        .grid
        .threadgroup_width
        .unwrap_or_else(|| panic!("152-role node {} carries no cooperative threadgroup_width", order[7]))
        as u64;

    let node162_source = std::fs::read_to_string(nodes_dir.join(format!("{}.metal", order[12])))
        .unwrap_or_else(|error| panic!("read nodes/{}.metal: {error}", order[12]));
    let spliced_162_source = splice_162_epilogue(&node162_source, dim_count as u64);
    let epilogue_library = compile_library(device, &spliced_162_source, math_mode);
    let epilogue_entry = read_node_spec(nodes_dir, order[12]).entry;
    let pipeline = pipeline_from_library(device, &epilogue_library, &epilogue_entry);
    let uniforms = shared_buffer(device, &parse_uniforms_hex(&nodes_dir.join(format!("{}.uniforms.hex", order[12]))));
    let grid = read_node_spec(nodes_dir, order[12]).grid;
    let in82_id = read_node_spec(nodes_dir, order[12])
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::Input(id) if *id != order[8] => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("162-role node carries no cached-V leaf Input"));
    let final_stage = FinalStage { pipeline, uniforms, grid, in82_id };

    let uniform_buffer = |id: u32| shared_buffer(device, &parse_uniforms_hex(&nodes_dir.join(format!("{id}.uniforms.hex"))));
    let u151 = uniform_buffer(order[6]);
    let u152 = uniform_buffer(order[7]);
    let u154 = uniform_buffer(order[8]);
    let u156 = uniform_buffer(order[10]);
    let u157 = uniform_buffer(order[9]);
    let u158 = uniform_buffer(order[11]);
    let u164 = uniform_buffer(order[13]);

    let elements = |id: u32| parse_bits_file(&vectors_dir.join(format!("{id}.bits"))).len();
    let (e151, e152, e154, e156, e157, e158, e164) = (
        elements(order[6]),
        elements(order[7]),
        elements(order[8]),
        elements(order[10]),
        elements(order[9]),
        elements(order[11]),
        elements(order[13]),
    );
    let group_count = e151;
    let key_count = e154 / group_count.max(1);
    let threadgroup_width = dim_count.max(key_count).max(key_width as usize);
    let grid_threads = group_count * threadgroup_width;

    let fused_source =
        format!("{FUSED_STRUCTS}\n{}", k_softmax_fused_source(dim_count as u64, key_count as u64, key_width));
    let fused_library = compile_library(device, &fused_source, math_mode);
    let k_softmax_pipeline = pipeline_from_library(device, &fused_library, "k_softmax_fused");

    let in127_id = read_node_spec(nodes_dir, order[13])
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::Input(id) if *id != order[10] => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("164-role node {} carries no new-key-V leaf Input", order[13]));

    // `cached_scores`/`new_scores`/`new_value` layouts here are read from
    // the REAL dumped grid.txt files (152/151/164's own `operands` entry
    // for 142/149/127, R9/PROGRESS.md coordinator item's own instruction),
    // not assumed: KEY-major/ROW-minor for cached_scores (`stride_key ==
    // attention_rows`, `stride_row == 1`), ROW-only for new_scores
    // (`stride_row == 1`), and -- verified identical across all four
    // gated contexts, not just this bucket -- new_value's `stride_row == 0`
    // (every attention row reads the SAME physical value vector; this tiny
    // fixture model's `kv_heads == 1`, `attention_rows == query_groups`),
    // `stride_dim == 1`.
    let library_softmax = use_library_softmax.then(|| {
        let bound_op = BoundOp {
            node: NodeId(order[8]),
            dtype: DType::Float32,
            extents: vec![key_count as u64, group_count as u64],
            kind: BoundOpKind::CachedSoftmaxWeights {
                operands: vec![
                    (NodeId(order[4]), Layout { base: 0, strides: vec![group_count as i64, 1].into() }, None),
                    (NodeId(order[5]), Layout { base: 0, strides: vec![1i64].into() }, None),
                    (NodeId(in127_id), Layout { base: 0, strides: vec![0i64, 1].into() }, None),
                ],
                cached_weight_sum: NodeId(order[9]),
                new_weight_sum: NodeId(order[11]),
                new_attended: NodeId(order[13]),
                cached_key_rows: key_count as u64,
                new_key_rows: 1,
                query_rows: group_count as u64,
                attention_rows: group_count as u64,
                head_dim: dim_count as u64,
            },
        };
        let kernel = omega_emit(&bound_op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
            .unwrap_or_else(|error| panic!("omega::emit(CachedSoftmaxWeights) failed: {error:?}"));
        let library = compile_library(device, &kernel.source, math_mode);
        let pipeline = pipeline_from_library(device, &library, &kernel.entry);
        let uniforms = shared_buffer(device, &(kernel.grid.threads as i64).to_le_bytes());
        LibrarySoftmaxKernel {
            pipeline,
            uniforms,
            grid_threads: kernel.grid.threads as usize,
            threadgroup_width: kernel.grid.threadgroup_width.unwrap_or(kernel.grid.threads) as usize,
        }
    });

    let dispatches = compiled_kept.len() + 2;

    BChainSetup {
        compiled_kept,
        k_softmax_pipeline,
        library_softmax,
        final_stage,
        u151,
        u152,
        u154,
        u156,
        u157,
        u158,
        u164,
        e151,
        e152,
        e154,
        e156,
        e157,
        e158,
        e164,
        threadgroup_width,
        grid_threads,
        id142: order[4],
        id149: order[5],
        id151: order[6],
        id152: order[7],
        id154: order[8],
        id156: order[10],
        id157: order[9],
        id158: order[11],
        id164: order[13],
        in127_id,
        final_node,
        final_elements: elements(final_node),
        dispatches,
    }
}

struct BChainEncodeResult {
    final_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    intermediates: Vec<(u32, MetalBuffer, usize)>,
}

// one B-chain iteration: encodes onto `command_buffer` (caller commits and
// waits) using `setup`'s precompiled pipelines/uniforms, reading leaves
// from `leaf_cache` (shared across iterations, populated on first use).
fn encode_b_chain(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    vectors_dir: &Path,
    setup: &BChainSetup,
    leaf_cache: &mut HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>,
) -> BChainEncodeResult {
    let mut chain_outputs: HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>> = HashMap::new();
    for node in &setup.compiled_kept {
        let output_id = node
            .bindings
            .iter()
            .find_map(|binding| match binding {
                Binding::Output(id) => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| panic!("node {} carries no Output binding", node.id));
        let output_buffer = zeroed_buffer(device, node.elements);
        let encoder = command_buffer
            .computeCommandEncoder()
            .unwrap_or_else(|| panic!("compute encoder for node {} ({})", node.id, node.entry));
        encoder.setComputePipelineState(&node.pipeline);
        for (index, binding) in node.bindings.iter().enumerate() {
            let buffer: &ProtocolObject<dyn MTLBuffer> = match binding {
                Binding::Output(_) => &output_buffer,
                Binding::Uniforms => &node.uniforms,
                Binding::Input(source_id) => {
                    if let Some(buffer) = chain_outputs.get(source_id) {
                        buffer
                    } else {
                        leaf_cache.entry(*source_id).or_insert_with(|| {
                            let bits = parse_bits_file(&vectors_dir.join(format!("{source_id}.bits")));
                            shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
                        })
                    }
                }
            };
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        let max_threadgroup = node.pipeline.maxTotalThreadsPerThreadgroup();
        let node_threadgroup_width = match node.grid.threadgroup_width {
            Some(width) => width.min(max_threadgroup).max(1),
            None => node.grid.threads.min(max_threadgroup).max(1),
        };
        let grid_size = MTLSize { width: node.grid.threads, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: node_threadgroup_width, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
        encoder.endEncoding();
        chain_outputs.insert(output_id, output_buffer);
    }

    let in142 = chain_outputs.get(&setup.id142).expect("142-role dispatched above");
    let in149 = chain_outputs.get(&setup.id149).expect("149-role dispatched above");
    let in127_buffer = leaf_cache.entry(setup.in127_id).or_insert_with(|| {
        let bits = parse_bits_file(&vectors_dir.join(format!("{}.bits", setup.in127_id)));
        shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
    });

    let out151 = zeroed_buffer(device, setup.e151);
    let out152 = zeroed_buffer(device, setup.e152);
    let out154 = zeroed_buffer(device, setup.e154);
    let out156 = zeroed_buffer(device, setup.e156);
    let out157 = zeroed_buffer(device, setup.e157);
    let out158 = zeroed_buffer(device, setup.e158);
    let out164 = zeroed_buffer(device, setup.e164);

    match &setup.library_softmax {
        Some(library) => {
            dispatch_encoder(
                command_buffer,
                &library.pipeline,
                &[in142, in149, in127_buffer, &out154, &library.uniforms, &out157, &out158, &out164],
                library.grid_threads,
                library.threadgroup_width,
            );
        }
        None => {
            dispatch_encoder(
                command_buffer,
                &setup.k_softmax_pipeline,
                &[
                    in142, in149, in127_buffer, &out151, &out152, &out154, &out156, &out157, &out158, &out164,
                    &setup.u151, &setup.u152, &setup.u154, &setup.u156, &setup.u157, &setup.u158, &setup.u164,
                ],
                setup.grid_threads,
                setup.threadgroup_width,
            );
        }
    }

    let in82_buffer = leaf_cache.entry(setup.final_stage.in82_id).or_insert_with(|| {
        let bits = parse_bits_file(&vectors_dir.join(format!("{}.bits", setup.final_stage.in82_id)));
        shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
    });
    let out_final = zeroed_buffer(device, setup.final_elements);
    dispatch_encoder(
        command_buffer,
        &setup.final_stage.pipeline,
        &[in82_buffer, &out154, &out_final, &setup.final_stage.uniforms, &out157, &out158, &out164],
        setup.final_stage.grid.threads,
        setup.final_stage.grid.threadgroup_width.unwrap_or(setup.final_stage.grid.threads),
    );

    BChainEncodeResult {
        final_buffer: out_final,
        intermediates: vec![
            (setup.id151, out151, setup.e151),
            (setup.id152, out152, setup.e152),
            (setup.id154, out154, setup.e154),
            (setup.id156, out156, setup.e156),
            (setup.id157, out157, setup.e157),
            (setup.id158, out158, setup.e158),
            (setup.id164, out164, setup.e164),
        ],
    }
}

struct BChainRunConfig<'a> {
    nodes_dir: &'a Path,
    vectors_dir: &'a Path,
    order: &'a [u32],
    math_mode: MTLMathMode,
    warmup: usize,
    iterations: usize,
    use_library_softmax: bool,
}

fn run_b_chain(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    config: BChainRunConfig,
) -> BChainOutcome {
    let setup = build_b_chain_setup(
        device,
        config.nodes_dir,
        config.vectors_dir,
        config.order,
        config.math_mode,
        config.use_library_softmax,
    );
    time_and_gate_b_chain(device, queue, config.vectors_dir, &setup, config.warmup, config.iterations)
}

fn time_and_gate_b_chain(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    vectors_dir: &Path,
    setup: &BChainSetup,
    warmup: usize,
    iterations: usize,
) -> BChainOutcome {
    let mut leaf_cache: HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>> = HashMap::new();
    let mut last_result: Option<BChainEncodeResult> = None;

    let mut samples_us = Vec::with_capacity(iterations);
    for iteration in 0..(warmup + iterations) {
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let result = encode_b_chain(device, &command_buffer, vectors_dir, setup, &mut leaf_cache);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if iteration >= warmup {
            let start = command_buffer.GPUStartTime();
            let end = command_buffer.GPUEndTime();
            samples_us.push((end - start) * 1_000_000.0);
        }
        if iteration == warmup + iterations - 1 {
            last_result = Some(result);
        }
    }

    samples_us.sort_by(|a, b| a.partial_cmp(b).expect("GPU duration samples are finite"));
    let timing = TimingResult { median_us: percentile(&samples_us, 0.5), p90_us: percentile(&samples_us, 0.9) };

    let result = last_result.expect("at least one iteration ran");
    let mut intermediate_gates = Vec::new();
    for (id, buffer, count) in &result.intermediates {
        let reference = parse_bits_file(&vectors_dir.join(format!("{id}.bits")));
        let produced = read_f32_buffer(buffer, *count);
        let matches = produced.iter().zip(reference.iter()).all(|(value, bits)| value.to_bits() == *bits);
        intermediate_gates.push((*id, matches));
    }

    let reference = parse_bits_file(&vectors_dir.join(format!("{}.bits", setup.final_node)));
    let produced = read_f32_buffer(&result.final_buffer, setup.final_elements);
    let mut first_diff = None;
    for (index, (value, bits)) in produced.iter().zip(reference.iter()).enumerate() {
        if value.to_bits() != *bits {
            first_diff = Some((index, *bits, value.to_bits()));
            break;
        }
    }
    let gate = match first_diff {
        None => {
            println!("chainB_gate chain=B elements={} match=true", setup.final_elements);
            true
        }
        Some((index, expected, produced)) => {
            println!(
                "chainB_gate chain=B elements={} match=false first_diff=({index}, 0x{expected:08x}, 0x{produced:08x})",
                setup.final_elements
            );
            false
        }
    };
    for (id, matches) in &intermediate_gates {
        println!("chainB_gate chain=B intermediate={id} match={matches}");
    }

    BChainOutcome { dispatches: setup.dispatches, timing, gate, intermediate_gates }
}

struct InterleavedResult {
    off_median_us: f64,
    off_p90_us: f64,
    b_median_us: f64,
    b_p90_us: f64,
    ratio_median: f64,
    ratio_p90: f64,
    fraction_b_faster: f64,
}

// clock-state-robust timing: per iteration, encode+commit+wait ONE chain,
// then the OTHER (alternating which goes first each iteration so neither
// arm systematically inherits the other's post-dispatch clock/thermal
// state), recording each command buffer's own GPUEndTime-GPUStartTime.
// Kernels are untouched -- this only changes measurement order, reusing
// the exact same `compiled_off`/`BChainSetup` pipelines `run()` already
// builds and gates.
fn run_interleaved(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    compiled_off: &[CompiledNode],
    b_setup: &BChainSetup,
    vectors_dir: &Path,
    warmup: usize,
    iterations: usize,
) -> InterleavedResult {
    let mut off_leaf_cache: HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>> = HashMap::new();
    let mut b_leaf_cache: HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>> = HashMap::new();

    let time_off_once = |off_leaf_cache: &mut HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>| -> f64 {
        let (command_buffer, _outputs) = encode_chain(queue, device, vectors_dir, compiled_off, off_leaf_cache);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1_000_000.0
    };
    let time_b_once = |b_leaf_cache: &mut HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>| -> f64 {
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let _result = encode_b_chain(device, &command_buffer, vectors_dir, b_setup, b_leaf_cache);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1_000_000.0
    };

    for iteration in 0..warmup {
        if iteration % 2 == 0 {
            time_off_once(&mut off_leaf_cache);
            time_b_once(&mut b_leaf_cache);
        } else {
            time_b_once(&mut b_leaf_cache);
            time_off_once(&mut off_leaf_cache);
        }
    }

    let mut off_samples = Vec::with_capacity(iterations);
    let mut b_samples = Vec::with_capacity(iterations);
    let mut ratios = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let (off_us, b_us) = if iteration % 2 == 0 {
            let off_us = time_off_once(&mut off_leaf_cache);
            let b_us = time_b_once(&mut b_leaf_cache);
            (off_us, b_us)
        } else {
            let b_us = time_b_once(&mut b_leaf_cache);
            let off_us = time_off_once(&mut off_leaf_cache);
            (off_us, b_us)
        };
        off_samples.push(off_us);
        b_samples.push(b_us);
        ratios.push(b_us / off_us);
    }

    off_samples.sort_by(|a, b| a.partial_cmp(b).expect("GPU duration samples are finite"));
    b_samples.sort_by(|a, b| a.partial_cmp(b).expect("GPU duration samples are finite"));
    let mut sorted_ratios = ratios.clone();
    sorted_ratios.sort_by(|a, b| a.partial_cmp(b).expect("ratio samples are finite"));
    let fraction_b_faster = ratios.iter().filter(|&&ratio| ratio < 1.0).count() as f64 / ratios.len() as f64;

    InterleavedResult {
        off_median_us: percentile(&off_samples, 0.5),
        off_p90_us: percentile(&off_samples, 0.9),
        b_median_us: percentile(&b_samples, 0.5),
        b_p90_us: percentile(&b_samples, 0.9),
        ratio_median: percentile(&sorted_ratios, 0.5),
        ratio_p90: percentile(&sorted_ratios, 0.9),
        fraction_b_faster,
    }
}

fn cli_flag(name: &str, default: String) -> String {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == name {
            return args.next().unwrap_or_else(|| panic!("{name} requires a value"));
        }
        if let Some(value) = flag.strip_prefix(&format!("{name}=")) {
            return value.to_string();
        }
    }
    default
}

fn r9_root() -> PathBuf {
    PathBuf::from(
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/rootcause/r9",
    )
}

pub fn run() {
    let root = r9_root();
    let nodes_dir = PathBuf::from(cli_flag("--nodes-dir", root.join("nodes").display().to_string()));
    let vectors_dir = PathBuf::from(cli_flag("--vectors-dir", root.join("vectors_relaxed").display().to_string()));
    let math_mode_flag = cli_flag("--math-mode", "relaxed".to_string());
    let label = cli_flag("--label", "layer0_c32".to_string());
    let math_mode = match math_mode_flag.as_str() {
        "safe" => MTLMathMode::Safe,
        "relaxed" => MTLMathMode::Relaxed,
        other => panic!("unrecognized --math-mode `{other}`, expected safe|relaxed"),
    };
    let use_library_softmax = cli_flag("--softmax-from-library", "0".to_string()) == "1";

    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    let order = chain_node_order(&nodes_dir);
    let final_node = *order.last().expect("manifest.txt carries at least one node");
    let compiled_off: Vec<CompiledNode> =
        order.iter().map(|&id| compile_node(&device, &nodes_dir, &vectors_dir, id, math_mode)).collect();
    let final_elements = compiled_off.last().expect("order is non-empty, checked above").elements;

    if use_library_softmax {
        run_softmax_library_gate(&device, &queue, &nodes_dir, &vectors_dir, &order, math_mode, &label);
    }

    let (off_timing, off_outputs) = time_chain(&queue, &device, &vectors_dir, &compiled_off, 20, 200);
    let off_gate = gate_final("OFF", &vectors_dir, &off_outputs, final_node, final_elements);

    let b_outcome = run_b_chain(
        &device,
        &queue,
        BChainRunConfig {
            nodes_dir: &nodes_dir,
            vectors_dir: &vectors_dir,
            order: &order,
            math_mode,
            warmup: 20,
            iterations: 200,
            use_library_softmax,
        },
    );
    let ratio = b_outcome.timing.median_us / off_timing.median_us;
    let all_intermediates_match = b_outcome.intermediate_gates.iter().all(|(_, matches)| *matches);

    println!(
        "chainB_report label={label} math_mode={math_mode_flag} dispatches_off={} off_median_us={:.2} off_p90_us={:.2} off_gate={} dispatches_b={} b_median_us={:.2} b_p90_us={:.2} b_gate={} b_intermediates_match={} ratio={:.4}",
        compiled_off.len(),
        off_timing.median_us,
        off_timing.p90_us,
        off_gate,
        b_outcome.dispatches,
        b_outcome.timing.median_us,
        b_outcome.timing.p90_us,
        b_outcome.gate,
        all_intermediates_match,
        ratio,
    );

    if cli_flag("--interleave", "0".to_string()) == "1" {
        let interleaved_setup =
            build_b_chain_setup(&device, &nodes_dir, &vectors_dir, &order, math_mode, use_library_softmax);
        let interleaved =
            run_interleaved(&device, &queue, &compiled_off, &interleaved_setup, &vectors_dir, 100, 300);
        println!(
            "chainB_interleaved label={label} math_mode={math_mode_flag} off_median_us={:.2} off_p90_us={:.2} b_median_us={:.2} b_p90_us={:.2} ratio_median={:.4} ratio_p90={:.4} fraction_b_faster={:.4}",
            interleaved.off_median_us,
            interleaved.off_p90_us,
            interleaved.b_median_us,
            interleaved.b_p90_us,
            interleaved.ratio_median,
            interleaved.ratio_p90,
            interleaved.fraction_b_faster,
        );
    }
}

}
