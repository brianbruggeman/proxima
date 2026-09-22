//! Rootcause round 9, staged fusion: ONE Metal kernel dispatched as ONE
//! threadgroup of 8 simdgroups (256 threads, simdgroup `g` = query group
//! `g`), each of the production nodes' per-thread bodies transcribed
//! VERBATIM (`R9/nodes/<id>.metal`'s own loop text, same variable names,
//! same `seeded ? (accumulator + value) : value` form, same `simd_sum`
//! call, same epilogue statement order) into its own
//! `__attribute__((noinline))` device function -- ONLY the address
//! computations are replaced by the staged layout's direct
//! key/group/lane indices (production used `Uniforms`-driven
//! `full_coord`/stride decode over an 8192-thread dispatch; here the
//! caller already knows which (key, group) a given simdgroup call is
//! for). Node-to-node data that the production plan round-tripped
//! through separate device dispatches now lives in `threadgroup` memory
//! within the one dispatch, with a `threadgroup_barrier` between stages;
//! each stage's FINAL (post-`simd_sum`/post-epilogue) value is also
//! stored once to its own dedicated device output buffer, which is both
//! this file's diagnostic store and the value the host gates against
//! `R9/vectors_relaxed/<id>.bits`.
//!
//! Stages this slice: 134/135 (identity copies of leaf 130/133), 139
//! (odd-dot fold, 256), 142 (even-dot fold + epilogue add/select, 256),
//! 146 (new-key odd fold, 8), 149 (new-key even fold + epilogue, 8).
//! Compiled under `MTLMathMode::Relaxed` -- the production compile mode
//! that reproduced all 15 target nodes standalone (`R9/node_gates_relaxed.log`).
//!
//! # Run
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-attn-fuse \
//!     cargo run -p omega --release --features metal --example attn_staged_replay
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_staged_replay requires --features metal on macOS");
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
use proxima_tensor::{BoundOp, BoundOpKind, DType, NodeId};

// The kernel TEXT (previously a second, independently-transcribed copy
// here) now comes ENTIRELY from `omega::msl::render_cached_attention_two_pass`,
// called directly (not through `omega::emit`, so a `diag` run reaches the
// `diag: bool` parameter `omega::emit`'s own call site never sets). `run()`
// below reads the LIBRARY kernel's own `device float* scratch` intermediate
// values back (`omega::msl::two_pass_scratch_layout`) instead of dispatching
// a second, hand-transcribed kernel -- ONE kernel text, never two diverging
// transcriptions.

fn r9_root() -> PathBuf {
    std::env::var("ATTN_R9_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/rootcause/r9",
            )
        })
}

fn parse_bits_file(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
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

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|value| f32::from_bits(*value)).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn shared_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let padded_len = bytes.len().max(4);
    let mut padded = bytes.to_vec();
    padded.resize(padded_len, 0);
    let pointer = unsafe { NonNull::new_unchecked(padded.as_ptr() as *mut c_void) };
    unsafe { device.newBufferWithBytes_length_options(pointer, padded.len(), MTLResourceOptions::StorageModeShared) }
        .expect("device allocates a fresh shared buffer copied from real fixture bytes")
}

fn leaf_buffer(device: &ProtocolObject<dyn MTLDevice>, vectors_dir: &Path, node: u32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bits = parse_bits_file(&vectors_dir.join(format!("{node}.bits")));
    shared_buffer(device, &f32_bytes(&bits_to_f32(&bits)))
}

fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
    let pointer = buffer.contents().as_ptr().cast::<f32>();
    unsafe { core::slice::from_raw_parts(pointer, count) }.to_vec()
}

fn cli_math_mode() -> MTLMathMode {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--math-mode" {
            let value = args.next().unwrap_or_else(|| panic!("--math-mode requires a value (safe|relaxed)"));
            return match value.as_str() {
                "safe" => MTLMathMode::Safe,
                "relaxed" => MTLMathMode::Relaxed,
                other => panic!("unrecognized --math-mode value `{other}`, expected safe|relaxed"),
            };
        }
    }
    MTLMathMode::Relaxed
}

fn cli_vectors_dir(root: &Path) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--vectors-dir" {
            let value = args.next().unwrap_or_else(|| panic!("--vectors-dir requires a path"));
            return PathBuf::from(value);
        }
    }
    root.join("vectors_relaxed")
}

// `--layer <n>` or `PROXIMA_ATTN_LAYER` (default 0) -- informational only
// (log filename suffix, gates the layer-0-only production9 pass below); the
// actual per-layer node ids come from `--nodes-dir`/`--vectors-dir`, not
// from this index.
fn cli_layer_index() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--layer" {
            let value = args.next().unwrap_or_else(|| panic!("--layer requires a value"));
            return value.parse().unwrap_or_else(|error| panic!("--layer value `{value}`: {error}"));
        }
    }
    std::env::var("PROXIMA_ATTN_LAYER")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0)
}

// `--window <n|none>` (default `511`, layer 0's own sliding-window shape) --
// the `BoundOpKind::CachedAttention` field `render_cached_attention_two_pass`
// derives its in-kernel mask from is a WINDOW size, not `cached_lower_inclusive`
// directly (`window = 1 - cached_lower_inclusive`, `render_cached_attention_two_pass`'s
// own doc); `none` selects a global layer (`cached_lower_inclusive = i64::MIN`).
fn cli_window() -> Option<u64> {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--window" {
            let value = args.next().unwrap_or_else(|| panic!("--window requires a value (a number or `none`)"));
            return if value == "none" {
                None
            } else {
                Some(value.parse().unwrap_or_else(|error| panic!("--window value `{value}`: {error}")))
            };
        }
    }
    Some(511)
}

// `--cached-len <f32>` (default `6.0`, the step-1 fixture value both R9/vectors
// and R9/vectors_l4 were captured at) -- the live cache row count `cached_len_buf`
// carries (`cached_attention_two_pass.rs`'s own kernel body reads `cached_len_buf[0]`).
fn cli_cached_len() -> f32 {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--cached-len" {
            let value = args.next().unwrap_or_else(|| panic!("--cached-len requires a value"));
            return value.parse().unwrap_or_else(|error| panic!("--cached-len value `{value}`: {error}"));
        }
    }
    6.0
}

// `--nodes-dir <path>` (default `<root>/nodes`, the layer-0 dump) -- a
// caller regates a different `attn_node_dump` output directory, e.g.
// `<root>/nodes_l4`.
fn cli_nodes_dir(root: &Path) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--nodes-dir" {
            let value = args.next().unwrap_or_else(|| panic!("--nodes-dir requires a path"));
            return PathBuf::from(value);
        }
    }
    root.join("nodes")
}

/// One layer's worth of node ids this file needs, derived entirely from
/// `nodes_dir`'s own `attn_node_dump` output -- nothing here is a
/// per-layer literal. `attended` is the dump's own last target
/// (`attn_node_dump.rs`'s `target_node_ids` appends it last); the 14
/// absorbed-node roles and the 8 far operand leaves are each a FIXED delta
/// from `attended` (verified identical across `S/census/absorbed_nodes.txt`
/// layers 0, 1, 2, 4, 10 -- see `decode.rs`'s `ATTN_ABSORBED_NODE_OFFSETS`).
/// No mask-leaf fields: the library kernel derives its mask in-kernel from
/// `cached_len` alone (`render_cached_attention_two_pass`'s own doc), so a
/// caller dispatching that kernel directly never reads a captured mask
/// buffer the way the deleted local diagnostic kernel once did.
struct LayerFixture {
    attended: u32,
    head_dim: u32,
    role_134: u32,
    role_135: u32,
    role_139: u32,
    role_142: u32,
    role_146: u32,
    role_149: u32,
    role_151: u32,
    role_152: u32,
    role_154: u32,
    role_156: u32,
    role_157: u32,
    role_158: u32,
    role_162: u32,
    role_164: u32,
    q_even: u32,
    q_odd: u32,
    k_even_cache: u32,
    k_odd_cache: u32,
    new_k_even: u32,
    new_k_odd: u32,
    v_cache: u32,
    v_new: u32,
}

impl LayerFixture {
    /// Maps [`omega::msl::ScratchRegion::role`]'s layer-0-numbering tag to
    /// THIS layer's real node id -- `166` is a sentinel for the attended
    /// node itself (never a `ScratchRegion` role, since node 166 is `out`,
    /// not `scratch`), included here so callers can use one lookup for both.
    fn node_for_role(&self, role: u32) -> u32 {
        match role {
            134 => self.role_134,
            135 => self.role_135,
            139 => self.role_139,
            142 => self.role_142,
            146 => self.role_146,
            149 => self.role_149,
            151 => self.role_151,
            152 => self.role_152,
            154 => self.role_154,
            156 => self.role_156,
            157 => self.role_157,
            158 => self.role_158,
            162 => self.role_162,
            164 => self.role_164,
            166 => self.attended,
            other => panic!("LayerFixture carries no role={other}"),
        }
    }
}

fn derive_layer_fixture(nodes_dir: &Path) -> LayerFixture {
    let manifest = std::fs::read_to_string(nodes_dir.join("manifest.txt"))
        .unwrap_or_else(|error| panic!("read {}/manifest.txt: {error}", nodes_dir.display()));
    let target_ids: Vec<u32> = manifest
        .lines()
        .filter_map(|line| line.strip_prefix("node="))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(|id| id.parse::<u32>().unwrap_or_else(|error| panic!("manifest node id `{id}`: {error}")))
        .collect();
    let attended = *target_ids.last().expect("manifest carries at least the attended node");

    let attended_grid = std::fs::read_to_string(nodes_dir.join(format!("{attended}.grid.txt")))
        .unwrap_or_else(|error| panic!("read nodes/{attended}.grid.txt: {error}"));
    let extents_line = attended_grid
        .lines()
        .find(|line| line.starts_with("extents="))
        .unwrap_or_else(|| panic!("nodes/{attended}.grid.txt missing extents= line"));
    let head_dim: u32 = extents_line
        .trim_start_matches("extents=")
        .trim_matches(|c| c == '[' || c == ']')
        .rsplit(", ")
        .next()
        .expect("extents carries at least one dim")
        .parse()
        .expect("last extents dim is head_dim");

    let at = attended as i64;

    LayerFixture {
        attended,
        head_dim,
        role_134: (at - 32) as u32,
        role_135: (at - 31) as u32,
        role_139: (at - 27) as u32,
        role_142: (at - 24) as u32,
        role_146: (at - 20) as u32,
        role_149: (at - 17) as u32,
        role_151: (at - 15) as u32,
        role_152: (at - 14) as u32,
        role_154: (at - 12) as u32,
        role_156: (at - 10) as u32,
        role_157: (at - 9) as u32,
        role_158: (at - 8) as u32,
        role_162: (at - 4) as u32,
        role_164: (at - 2) as u32,
        q_even: (at - 36) as u32,
        q_odd: (at - 33) as u32,
        k_even_cache: (at - 86) as u32,
        k_odd_cache: (at - 85) as u32,
        new_k_even: (at - 51) as u32,
        new_k_odd: (at - 48) as u32,
        v_cache: (at - 84) as u32,
        v_new: (at - 39) as u32,
    }
}

fn compile(device: &ProtocolObject<dyn MTLDevice>, source: &str, entry: &str, math_mode: MTLMathMode) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
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

fn node_gate(reference: &[u32], produced: &[f32], id: u32, entry: &str, math_mode_name: &str) -> String {
    let elements = reference.len();
    assert!(produced.len() == elements, "node {id} produced {} elements, reference carries {elements}", produced.len());
    let mut first_diff = None;
    let mut diff_count = 0usize;
    for (index, (produced_value, reference_bits_value)) in produced.iter().zip(reference.iter()).enumerate() {
        let produced_bits = produced_value.to_bits();
        if produced_bits != *reference_bits_value {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some((index, *reference_bits_value, produced_bits));
            }
        }
    }
    match first_diff {
        None => format!("node_gate node={id} entry={entry} elements={elements} math_mode={math_mode_name} match=true"),
        Some((index, expected, produced_bits)) => format!(
            "node_gate node={id} entry={entry} elements={elements} math_mode={math_mode_name} match=false first_diff=({index}, 0x{expected:08x}, 0x{produced_bits:08x}) diff_count={diff_count}",
        ),
    }
}

pub fn run() {
    let root = r9_root();
    let layer_index = cli_layer_index();
    let nodes_dir = cli_nodes_dir(&root);
    let vectors_dir = cli_vectors_dir(&root);
    let math_mode = cli_math_mode();
    let window = cli_window();
    let cached_len = cli_cached_len();
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    let fixture = derive_layer_fixture(&nodes_dir);
    let head_dim = u64::from(fixture.head_dim);
    let qg_head_dim = 8 * head_dim;
    let cached_cap_qg = parse_bits_file(&vectors_dir.join(format!("{}.bits", fixture.role_139))).len() as u64;
    let cached_capacity = cached_cap_qg / 8;
    let math_mode_name = match math_mode {
        MTLMathMode::Safe => "safe",
        MTLMathMode::Relaxed => "relaxed",
        _ => "fast",
    };
    println!(
        "attn_staged_replay layer={layer_index} attended={} head_dim={head_dim} cached_capacity={cached_capacity} window={window:?} cached_len={cached_len} nodes_dir={} vectors_dir={}",
        fixture.attended, nodes_dir.display(), vectors_dir.display(),
    );

    let leaf80 = leaf_buffer(&device, &vectors_dir, fixture.k_even_cache);
    let leaf81 = leaf_buffer(&device, &vectors_dir, fixture.k_odd_cache);
    let leaf82 = leaf_buffer(&device, &vectors_dir, fixture.v_cache);
    let leaf127 = leaf_buffer(&device, &vectors_dir, fixture.v_new);
    let leaf130 = leaf_buffer(&device, &vectors_dir, fixture.q_even);
    let leaf133 = leaf_buffer(&device, &vectors_dir, fixture.q_odd);
    let leaf115 = leaf_buffer(&device, &vectors_dir, fixture.new_k_even);
    let leaf118 = leaf_buffer(&device, &vectors_dir, fixture.new_k_odd);

    let cached_lower_inclusive = match window {
        Some(window_value) => 1 - (window_value as i64),
        None => i64::MIN,
    };
    let cached_len_buffer = shared_buffer(&device, &f32_bytes(&[cached_len]));
    let bound_op = BoundOp {
        node: NodeId(fixture.attended),
        dtype: DType::Float32,
        extents: vec![1, 1, 8, head_dim],
        kind: BoundOpKind::CachedAttention {
            operands: Vec::new(),
            query_rows: 1,
            cached_key_rows: cached_capacity,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 8,
            head_dim,
            rotary_dim: head_dim,
            scale: 1.0,
            cached_lower_inclusive,
            new_upper_inclusive: 0,
            two_pass: true,
        },
    };

    // base (diag=false) dispatch -- the SAME kernel text and buffer
    // signature production wiring uses (`render_cached_attention`'s own
    // `if *two_pass` branch always passes `diag=false`).
    let base_entry = "attn_staged_base_tp";
    let base_source = omega::msl::render_cached_attention_two_pass(&bound_op, base_entry, false)
        .unwrap_or_else(|error| panic!("emit diag=false two_pass kernel for node={}: {error}", bound_op.node.0));
    println!("attn_staged_replay production9 entry={base_entry}");
    let kernel_dump_name = if layer_index == 0 { "two_pass_layer0.metal".to_string() } else { format!("two_pass_layer{layer_index}.metal") };
    let kernel_dump_path = root.join("wiring").join(&kernel_dump_name);
    std::fs::create_dir_all(kernel_dump_path.parent().expect("has parent"))
        .unwrap_or_else(|error| panic!("create {}: {error}", kernel_dump_path.parent().expect("has parent").display()));
    std::fs::write(&kernel_dump_path, &base_source)
        .unwrap_or_else(|error| panic!("write {}: {error}", kernel_dump_path.display()));
    let base_pipeline = compile(&device, &base_source, base_entry, math_mode);

    let scratch_elements = omega::msl::two_pass_scratch_elements(8, head_dim, cached_capacity, false);
    let base_scratch = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; scratch_elements as usize]));
    let base_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; qg_head_dim as usize]));
    let uniforms = shared_buffer(&device, &(qg_head_dim as i64).to_le_bytes());

    let base_command_buffer = queue.commandBuffer().expect("command buffer");
    let base_encoder = base_command_buffer.computeCommandEncoder().expect("compute encoder");
    base_encoder.setComputePipelineState(&base_pipeline);
    let base_buffers: [&ProtocolObject<dyn MTLBuffer>; 12] = [
        &leaf130, &leaf133, &leaf80, &leaf81, &leaf115, &leaf118, &leaf82, &leaf127, &cached_len_buffer,
        &base_scratch, &base_out, &uniforms,
    ];
    for (index, buffer) in base_buffers.iter().enumerate() {
        unsafe { base_encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    // dispatch width is `two_pass_physical_threadgroup_width(QUERY_GROUPS,
    // ..)` -- the SAME function `grid_threads`'s and
    // `tiled_gemm_threadgroup_width`'s own `two_pass` arms call
    // (`signature_tokens_prelude.rs`, `tiled_gemm_cooperative_scan.rs`), so
    // this standalone harness can never dispatch a threadgroup wider than
    // root cause 3's own wave-split kernel text assumes.
    let dispatch_width = omega::msl::two_pass_physical_threadgroup_width(8, head_dim as u64, cached_capacity as u64) as usize;
    let grid = MTLSize { width: dispatch_width, height: 1, depth: 1 };
    let threadgroup = MTLSize { width: dispatch_width, height: 1, depth: 1 };
    base_encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    base_encoder.endEncoding();
    base_command_buffer.commit();
    base_command_buffer.waitUntilCompleted();

    let produced_base_out = read_f32_buffer(&base_out, qg_head_dim as usize);
    let final_reference = parse_bits_file(&vectors_dir.join(format!("{}.bits", fixture.attended)));

    let mut log_lines = Vec::new();

    // Route step 3, item 1: gate the seven base scratch regions -- reading
    // the LIBRARY kernel's own `device float* scratch` intermediate values
    // back, at the SAME offsets `staged_two_pass_source` computed its own
    // `SCRATCH_*_OFFSET` constexprs from (`two_pass_scratch_layout`) -- not
    // a second kernel, not a second transcription.
    let base_layout = omega::msl::two_pass_scratch_layout(8, head_dim, cached_capacity, false);
    let produced_scratch = read_f32_buffer(&base_scratch, scratch_elements as usize);
    for region in &base_layout {
        let node_id = fixture.node_for_role(region.role);
        let reference = parse_bits_file(&vectors_dir.join(format!("{node_id}.bits")));
        let region_values =
            &produced_scratch[region.offset_elements as usize..(region.offset_elements + region.len_elements) as usize];
        let line = node_gate(&reference, region_values, node_id, "library_scratch_readback", math_mode_name);
        println!("attn_staged_replay {line}");
        log_lines.push(line);
    }

    // Route step 3, item 2: diag=true re-dispatch -- proves the diagnostic
    // tail is store-only (attended output bits must equal the diag=false
    // run) BEFORE any diag value is trusted, then gates the seven scalar
    // regions the same way as the base seven above.
    let diag_scratch_elements = omega::msl::two_pass_scratch_elements(8, head_dim, cached_capacity, true);
    let diag_entry = "attn_staged_diag_tp";
    let diag_source = omega::msl::render_cached_attention_two_pass(&bound_op, diag_entry, true)
        .unwrap_or_else(|error| panic!("emit diag=true two_pass kernel: {error}"));
    let diag_pipeline = compile(&device, &diag_source, diag_entry, math_mode);
    let diag_scratch = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; diag_scratch_elements as usize]));
    let diag_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; qg_head_dim as usize]));
    let diag_command_buffer = queue.commandBuffer().expect("command buffer");
    let diag_encoder = diag_command_buffer.computeCommandEncoder().expect("compute encoder");
    diag_encoder.setComputePipelineState(&diag_pipeline);
    let diag_buffers: [&ProtocolObject<dyn MTLBuffer>; 12] = [
        &leaf130, &leaf133, &leaf80, &leaf81, &leaf115, &leaf118, &leaf82, &leaf127, &cached_len_buffer,
        &diag_scratch, &diag_out, &uniforms,
    ];
    for (index, buffer) in diag_buffers.iter().enumerate() {
        unsafe { diag_encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    diag_encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    diag_encoder.endEncoding();
    diag_command_buffer.commit();
    diag_command_buffer.waitUntilCompleted();
    let produced_diag_out = read_f32_buffer(&diag_out, qg_head_dim as usize);
    let diag_vs_base_mismatches = produced_diag_out
        .iter()
        .zip(produced_base_out.iter())
        .filter(|(diag_value, base_value)| diag_value.to_bits() != base_value.to_bits())
        .count();
    println!(
        "attn_staged_replay diag_vs_base math_mode={math_mode_name} elements={qg_head_dim} mismatches={diag_vs_base_mismatches} base_element0=0x{:08x} diag_element0=0x{:08x}",
        produced_base_out[0].to_bits(), produced_diag_out[0].to_bits(),
    );
    if diag_vs_base_mismatches == 0 {
        let diag_scratch_values = read_f32_buffer(&diag_scratch, diag_scratch_elements as usize);
        let diag_layout = omega::msl::two_pass_scratch_layout(8, head_dim, cached_capacity, true);
        let diag_roles = [146u32, 149, 151, 152, 156, 157, 158];
        for region in diag_layout.iter().filter(|region| diag_roles.contains(&region.role)) {
            let node_id = fixture.node_for_role(region.role);
            let reference = parse_bits_file(&vectors_dir.join(format!("{node_id}.bits")));
            let region_values = &diag_scratch_values
                [region.offset_elements as usize..(region.offset_elements + region.len_elements) as usize];
            let line = node_gate(&reference, region_values, node_id, "library_scratch_readback_diag", math_mode_name);
            println!("attn_staged_replay {line}");
            log_lines.push(line);
        }
    }

    let final_line = node_gate(&final_reference, &produced_base_out, fixture.attended, "render_cached_attention_two_pass", math_mode_name)
        .replacen("node_gate", "staged_final", 1);
    println!("attn_staged_replay {final_line}");
    log_lines.push(final_line.clone());

    let log_name = if layer_index == 0 { "stage_gates.log".to_string() } else { format!("stage_gates_l{layer_index}.log") };
    let log_path = root.join(&log_name);
    std::fs::write(&log_path, log_lines.join("\n") + "\n")
        .unwrap_or_else(|error| panic!("write {}: {error}", log_path.display()));

    let progress_path = root.join("PROGRESS.md");
    let mut progress_entry = String::new();
    progress_entry.push_str(&format!(
        "\n## 2026-09-21 attn_staged_replay stage gates (library-only, layer={layer_index}, attended={}, head_dim={head_dim}, math_mode={math_mode_name}, vectors_dir={})\n\n",
        fixture.attended, vectors_dir.display(),
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
