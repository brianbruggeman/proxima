//! Rootcause round 9: byte-parity gate for all 15 attention-chain target
//! nodes, dispatched STANDALONE (own compile, own buffers, own dispatch)
//! from the exact MSL/grid/uniforms omega emitted for the real production
//! plan (`R9/nodes/<id>.{metal,grid.txt,uniforms.hex}`, dumped by
//! `attn_node_dump`, see `R9/PROGRESS.md`), fed real captured input tensors
//! (`R9/vectors/<node>.bits`) and compared element-for-element against the
//! real captured output tensor for that node. No golden math is
//! re-derived here -- this only proves the SAME kernel source, SAME
//! uniforms, and SAME dispatch shape reproduce the SAME output bytes when
//! run in isolation, which is what `resident_nocopy_cache::dispatch`
//! (the one production dispatch path, `omega/src/metal/resident_nocopy_cache.rs:1310-1334`)
//! does for every node in the real plan.
//!
//! # Run
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-attn-fuse \
//!     cargo run -p omega --release --features metal --example attn_node_gate
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_node_gate requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {

use core::ffi::c_void;
use core::ptr::NonNull;
use std::path::{Path, PathBuf};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

/// The gate's own target list is read from `<nodes_dir>/manifest.txt`
/// (`node=<id> ...` lines, one per `attn_node_dump` output) instead of a
/// fixed layer-0 array -- `--nodes-dir` (default `<root>/nodes`) lets a
/// caller regate any layer's dump (e.g. `<root>/nodes_l4`) without a second
/// hardcoded node-id list drifting from the first.
fn target_nodes(nodes_dir: &Path) -> Vec<u32> {
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

fn r9_root() -> PathBuf {
    std::env::var("ATTN_R9_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/rootcause/r9",
            )
        })
}

fn parse_bindings(line: &str) -> Vec<Binding> {
    let start = line.find('[').expect("bindings= line carries a [...] list") + 1;
    let end = line.rfind(']').expect("bindings= line's list is closed");
    line[start..end]
        .split(", ")
        .map(|token| {
            let token = token.trim();
            if let Some(inner) = token
                .strip_prefix("Input(NodeId(")
                .and_then(|rest| rest.strip_suffix("))"))
            {
                Binding::Input(inner.parse::<u32>().expect("Input NodeId is numeric"))
            } else if let Some(inner) = token
                .strip_prefix("Output(NodeId(")
                .and_then(|rest| rest.strip_suffix("))"))
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
    let threads_end = line[threads_start..]
        .find(',')
        .expect("threads field is comma-terminated")
        + threads_start;
    let threads = line[threads_start..threads_end]
        .trim()
        .parse::<usize>()
        .expect("threads value is numeric");

    let width_start = line.find("threadgroup_width: ").expect("grid= line carries threadgroup_width:")
        + "threadgroup_width: ".len();
    let width_end = line[width_start..]
        .find(", depth")
        .expect("threadgroup_width field precedes , depth")
        + width_start;
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
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let stripped = line.trim().strip_prefix("0x").unwrap_or_else(|| {
                panic!("{} line `{line}` missing 0x prefix", path.display())
            });
            u32::from_str_radix(stripped, 16)
                .unwrap_or_else(|error| panic!("{} line `{line}` parses as hex u32: {error}", path.display()))
        })
        .collect()
}

fn parse_uniforms_hex(path: &Path) -> Vec<u8> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let hex = text.trim();
    assert!(hex.len() % 2 == 0, "{} carries an even number of hex digits", path.display());
    (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .unwrap_or_else(|error| panic!("{} byte at {index} parses as hex: {error}", path.display()))
        })
        .collect()
}

// mirrors `omega::metal::pipeline_buffers_upload::MathMode`
// (`omega/src/metal/pipeline_buffers_upload.rs:31-48`) narrowed to the two
// values `R9/vectors` and this gate care about: `Safe` is what the parity
// probe's `build_plan` (`decode.rs` ~181-190) selects when
// `PROXIMA_ATTN_MATH_MODE` is unset -- the mode the captured
// `R9/vectors/<node>.bits` reference bytes were produced under -- and
// `Relaxed` is the production default the earlier run in this file
// mistakenly compiled with, which is a byte-parity cross-mode comparison,
// not a transcription bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateMathMode {
    Safe,
    Relaxed,
}

impl GateMathMode {
    fn as_mtl(self) -> MTLMathMode {
        match self {
            GateMathMode::Safe => MTLMathMode::Safe,
            GateMathMode::Relaxed => MTLMathMode::Relaxed,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            GateMathMode::Safe => "safe",
            GateMathMode::Relaxed => "relaxed",
        }
    }

    fn parse(value: &str) -> GateMathMode {
        match value {
            "safe" => GateMathMode::Safe,
            "relaxed" => GateMathMode::Relaxed,
            other => panic!("unrecognized --math-mode value `{other}`, expected safe|relaxed"),
        }
    }

}

fn cli_math_mode() -> GateMathMode {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--math-mode" {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("--math-mode requires a value (safe|relaxed)"));
            return GateMathMode::parse(&value);
        }
        if let Some(value) = flag.strip_prefix("--math-mode=") {
            return GateMathMode::parse(value);
        }
    }
    GateMathMode::Safe
}

// `<root>/vectors` (the safe-native capture) by default; a caller runs
// against a different capture -- e.g. `<root>/vectors_relaxed`, the
// Relaxed-native capture -- with `--vectors-dir <path>`.
fn cli_vectors_dir(root: &Path) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--vectors-dir" {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("--vectors-dir requires a path"));
            return PathBuf::from(value);
        }
        if let Some(value) = flag.strip_prefix("--vectors-dir=") {
            return PathBuf::from(value);
        }
    }
    root.join("vectors")
}

// `<root>/nodes` (layer 0's dump) by default; `--nodes-dir <path>` regates a
// different `attn_node_dump` output directory (e.g. `<root>/nodes_l4`).
fn cli_nodes_dir(root: &Path) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--nodes-dir" {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("--nodes-dir requires a path"));
            return PathBuf::from(value);
        }
        if let Some(value) = flag.strip_prefix("--nodes-dir=") {
            return PathBuf::from(value);
        }
    }
    root.join("nodes")
}

fn compile(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
    math_mode: GateMathMode,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(math_mode.as_mtl());
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

fn shared_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let padded_len = bytes.len().max(4);
    let mut padded = bytes.to_vec();
    padded.resize(padded_len, 0);
    let pointer = unsafe { NonNull::new_unchecked(padded.as_ptr() as *mut c_void) };
    unsafe { device.newBufferWithBytes_length_options(pointer, padded.len(), MTLResourceOptions::StorageModeShared) }
        .expect("device allocates a fresh shared buffer copied from real fixture bytes")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|value| f32::from_bits(*value)).collect()
}

fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
    let pointer = buffer.contents().as_ptr().cast::<f32>();
    unsafe { core::slice::from_raw_parts(pointer, count) }.to_vec()
}

// mirrors `omega::metal::resident_nocopy_cache::dispatch`
// (`omega/src/metal/resident_nocopy_cache.rs:1310-1334`) exactly: honors
// an explicit `threadgroup_width` as a correctness requirement rather than
// folding it into a generic occupancy pick, and falls back to
// `min(threads, max_threadgroup)` only when the plan left it unset.
fn dispatch(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    grid: &GridSpec,
) {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
    encoder.setComputePipelineState(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_width = match grid.threadgroup_width {
        Some(width) => width.min(max_threadgroup).max(1),
        None => grid.threads.min(max_threadgroup).max(1),
    };
    let grid_size = MTLSize { width: grid.threads, height: 1, depth: 1 };
    let threadgroup = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
}

struct GateOutcome {
    node: u32,
    entry: String,
    elements: usize,
    first_diff: Option<(usize, u32, u32)>,
    diff_count: usize,
}

fn run_node_gate(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    nodes_dir: &Path,
    vectors_dir: &Path,
    id: u32,
    math_mode: GateMathMode,
) -> GateOutcome {
    let spec = read_node_spec(nodes_dir, id);
    let source = std::fs::read_to_string(nodes_dir.join(format!("{id}.metal")))
        .unwrap_or_else(|error| panic!("read nodes/{id}.metal: {error}"));
    let uniforms_bytes = parse_uniforms_hex(&nodes_dir.join(format!("{id}.uniforms.hex")));
    let reference_bits = parse_bits_file(&vectors_dir.join(format!("{id}.bits")));
    let elements = reference_bits.len();

    let output_binding = spec
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::Output(node) => Some(*node),
            _ => None,
        })
        .unwrap_or_else(|| panic!("node {id} bindings carry no Output entry"));
    assert!(output_binding == id, "node {id} Output binding id mismatches target id {output_binding}");

    let owned_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = spec
        .bindings
        .iter()
        .map(|binding| match binding {
            Binding::Input(node) => {
                let bits = parse_bits_file(&vectors_dir.join(format!("{node}.bits")));
                shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
            }
            Binding::Output(_) => shared_buffer(device, &f32_bytes(&vec![0.0_f32; elements])),
            Binding::Uniforms => shared_buffer(device, &uniforms_bytes),
        })
        .collect();
    let buffer_refs: Vec<&ProtocolObject<dyn MTLBuffer>> = owned_buffers.iter().map(|buffer| &**buffer).collect();

    let pipeline = compile(device, &source, &spec.entry, math_mode);
    dispatch(queue, &pipeline, &buffer_refs, &spec.grid);

    let output_index = spec
        .bindings
        .iter()
        .position(|binding| matches!(binding, Binding::Output(_)))
        .expect("output binding position located above");
    let produced = read_f32_buffer(&owned_buffers[output_index], elements);

    let mut first_diff = None;
    let mut diff_count = 0usize;
    for (index, (produced_value, reference_bits_value)) in produced.iter().zip(reference_bits.iter()).enumerate() {
        let produced_bits = produced_value.to_bits();
        if produced_bits != *reference_bits_value {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some((index, *reference_bits_value, produced_bits));
            }
        }
    }

    GateOutcome { node: id, entry: spec.entry, elements, first_diff, diff_count }
}

// output log filename within `<root>`, default `node_gates.log`; a caller
// runs multiple mode/vectors-dir combinations against the SAME 15 target
// nodes with `--out <name>` to keep each combination's record separate
// (e.g. `node_gates_relaxed.log`, `node_gates_safe_vs_relaxed_capture.log`).
fn cli_out_log_name() -> String {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--out" {
            return args.next().unwrap_or_else(|| panic!("--out requires a filename"));
        }
        if let Some(value) = flag.strip_prefix("--out=") {
            return value.to_string();
        }
    }
    "node_gates.log".to_string()
}

fn gate_pass(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    nodes_dir: &Path,
    vectors_dir: &Path,
    math_mode: GateMathMode,
) -> (Vec<String>, usize) {
    let targets = target_nodes(nodes_dir);
    let mut log_lines = Vec::new();
    let mut passed = 0usize;
    for &id in targets.iter() {
        let outcome = run_node_gate(device, queue, nodes_dir, vectors_dir, id, math_mode);
        let mode_field = math_mode.as_str();
        let line = match outcome.first_diff {
            None => {
                passed += 1;
                format!(
                    "node_gate node={} entry={} elements={} math_mode={mode_field} match=true",
                    outcome.node, outcome.entry, outcome.elements,
                )
            }
            Some((index, expected, produced)) => format!(
                "node_gate node={} entry={} elements={} math_mode={mode_field} match=false first_diff=({index}, 0x{expected:08x}, 0x{produced:08x}) diff_count={}",
                outcome.node, outcome.entry, outcome.elements, outcome.diff_count,
            ),
        };
        println!("attn_node_gate {line}");
        log_lines.push(line);
    }
    let summary = format!("node_gates passed={passed}/{} math_mode={}", targets.len(), math_mode.as_str());
    println!("attn_node_gate {summary}");
    log_lines.push(summary);
    (log_lines, passed)
}

pub fn run() {
    let root = r9_root();
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    // `--math-mode safe|relaxed` (default `safe`), `--vectors-dir <path>`
    // (default `<root>/vectors`), `--nodes-dir <path>` (default
    // `<root>/nodes`), `--out <name>` (default `node_gates.log`) -- ONE gate
    // pass per invocation over `<nodes-dir>/manifest.txt`'s own target list,
    // so a caller re-runs it against any `attn_node_dump` output directory
    // (any layer) under either capture directory and either compile mode
    // without this binary hard-coding which combination that is.
    let math_mode = cli_math_mode();
    let nodes_dir = cli_nodes_dir(&root);
    let vectors_dir = cli_vectors_dir(&root);
    let out_log_name = cli_out_log_name();

    let (log_lines, _passed) = gate_pass(&device, &queue, &nodes_dir, &vectors_dir, math_mode);
    let log_path = root.join(&out_log_name);
    std::fs::write(&log_path, log_lines.join("\n") + "\n")
        .unwrap_or_else(|error| panic!("write {}: {error}", log_path.display()));

    let progress_path = root.join("PROGRESS.md");
    let mut progress_entry = String::new();
    progress_entry.push_str(&format!(
        "\n## 2026-09-21 attn_node_gate run (math_mode={}, vectors_dir={}, -> {})\n\n",
        math_mode.as_str(),
        vectors_dir.display(),
        out_log_name,
    ));
    for line in &log_lines {
        progress_entry.push_str(line);
        progress_entry.push('\n');
    }
    let mut existing = std::fs::read_to_string(&progress_path).unwrap_or_default();
    existing.push_str(&progress_entry);
    std::fs::write(&progress_path, existing)
        .unwrap_or_else(|error| panic!("append {}: {error}", progress_path.display()));
}

}
