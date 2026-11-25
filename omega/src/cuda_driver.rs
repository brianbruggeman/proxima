//! Minimal real CUDA driver boundary over the [`crate::cuda`] emitter.
//!
//! The emitter is deliberately usable without NVIDIA libraries. This module
//! is the separate runtime half: cudarc initializes the CUDA Driver API,
//! NVRTC compiles CUDA C into PTX, and a CUDA stream launches the resulting
//! kernel. The smoke method is intentionally tiny and numerical, so a host
//! with a CUDA device can prove initialization, compilation, argument
//! marshalling, launch, synchronization, and device-to-host readback before
//! the full graph binding layer is connected.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::ffi::CString;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use conflaguration::Settings;
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, DriverError,
};
use crate::gpu_runtime::{CudarcRuntime, DeviceElement, GpuRuntime, LaunchArguments, LaunchShape};
use crate::encode_render::{EncodeRows, RenderedEncode, row_bytes};
use cudarc::nvrtc::sys as nvrtc_sys;
use cudarc::nvrtc::result as nvrtc_result;
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use proxima_core::kv_carry::{
    Appended, Extent, KvCarry, KvCarryError, KvCommand, KvEvent, RowPair, StateCarryError, StateCommand, StateEvent,
    StateSwap,
};
use proxima_primitives::sync::blocking::Mutex;

use crate::config::{ResourceConstraints, RuntimeConfig};
use crate::cuda::target::Target;
use crate::cuda::{CudaGridSpec, CudaKernel, EmitDevice, cuda_cached_attention_class, executes as cuda_executes};
use crate::cuda_kernel_cache::{
    CacheKey, decode_binary_entry, decode_entry, encode_binary_entry, encode_entry, header_digest,
};
use crate::kernel_cache_config::KernelCacheConfig;
use crate::resident_identity::{ResidentIdentity, ResidentLedger, ResidentRefusal, owner_generation};
use crate::cuda_ptx_store::PtxStore;
use crate::bind_plan::{BindConfig, bind_plan, finish_bound};
use crate::derived_quantize::{ActivationRefusal, validate_storage};
use proxima_numeric::NumericFormat;
use crate::lowering::{
    Binding, Codec, OperandDtypes, PackedOperands, gather_count, program_operand_dtypes,
};
use proxima_tensor::bind::{
    RefreshRefusal, changed_nodes, kv_read_only_by_fused_attention, refit_cached_attention_at,
    RowTemplate, SymbolsTemplate, apply_row_template, apply_symbols_template, refresh_bound_ops,
    refresh_cached_attention, row_template, rows_enumerable, symbols_shapes_template,
    touched_positions, touches_changed,
};
use proxima_tensor::{BoundOp, BoundOpKind, Evaluated, Executes, NodeId, Op, QuantizedBlock, Shapes, block_node_ids, infer, resolve_named_blocks, underive_block_quantize};
use proxima_numeric::{DType, NumericPolicy, NumericRewrite, admit};

/// Errors from CUDA initialization, NVRTC compilation, or execution.
#[derive(Debug, thiserror::Error)]
pub enum CudaDriverError {
    #[error("CUDA cannot enforce `{field}`: {reason}")]
    UnsupportedConstraint {
        field: &'static str,
        reason: &'static str,
    },
    #[error("CUDA needs {requested} bytes for `{field}`, exceeding {limit}")]
    ResourceLimit {
        field: &'static str,
        requested: u64,
        limit: u64,
    },
    #[error("CUDA overflowed payload accounting")]
    ResourceOverflow,
    #[error("input contract changed for {node:?}: {reason}")]
    InputContractChanged { node: NodeId, reason: &'static str },
    #[error("CUDA driver: {0}")]
    Driver(String),
    #[error("CUDA NVRTC: {0}")]
    Nvrtc(String),
    #[error("CUDA header `{header}` not found; tried: {tried}")]
    CudaHeaderNotFound { header: String, tried: String },
    #[error("CUDA header `{path}` could not be read: {error}")]
    CudaHeaderUnreadable { path: String, error: String },
    #[error("CUDA smoke input is too large for the launch ABI: {0} elements")]
    InputTooLarge(usize),
    #[error("CUDA launch of {threads} threads needs more blocks than the launch index holds")]
    GridTooLarge { threads: u64 },
    #[error("CUDA refused to launch an idle grid: an op with no live output dispatches nothing")]
    IdleLaunch,
    #[error("CUDA kernel binding contract is not inputs, output, uniforms")]
    BindingContract,
    #[error("CUDA graph buffer for node {0:?} is missing")]
    MissingBuffer(NodeId),
    #[error("CUDA resident cache: {0}")]
    ResidentRefused(#[from] ResidentRefusal),
    #[error("CUDA gather index out of range in node {node:?}, slot {slot}: index={index}")]
    GatherIndexOutOfRange {
        node: NodeId,
        slot: usize,
        index: i64,
    },
    #[error("CUDA graph path rejected node {node:?}: {reason}")]
    UnsupportedGraph { node: NodeId, reason: &'static str },
    #[error("CUDA graph path rejected op kinds (node {node:?} is the first): {kinds}")]
    UnsupportedOpKinds { node: NodeId, kinds: String },
    #[error("CUDA graph node {node:?} has unsupported output dtype {dtype:?}")]
    UnsupportedDtype { node: NodeId, dtype: DType },
    #[error("CUDA graph node {node:?} could not be emitted: {error}")]
    Emit { node: NodeId, error: String },
    #[error("CUDA graph node {node:?} failed during execution: {error}")]
    Execute { node: NodeId, error: String },
    #[error(
        "CUDA appended rows overflow cache node {node:?}: {valid_elements} valid + {new_elements} new elements exceed {capacity_elements} of capacity"
    )]
    AppendOverflow {
        node: NodeId,
        valid_elements: usize,
        new_elements: usize,
        capacity_elements: usize,
    },
    #[error("CUDA kv carry state machine: {0}")]
    KvCarry(String),
    #[error("CUDA cannot store activations in the configured format: {0}")]
    ActivationStorage(#[from] Box<ActivationRefusal>),
}

impl From<rocm::HipDriverError> for CudaDriverError {
    fn from(error: rocm::HipDriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

impl From<StateCarryError<NodeId>> for CudaDriverError {
    fn from(error: StateCarryError<NodeId>) -> Self {
        Self::KvCarry(error.to_string())
    }
}

impl From<KvCarryError<NodeId>> for CudaDriverError {
    fn from(error: KvCarryError<NodeId>) -> Self {
        match error {
            KvCarryError::Overflow { input, valid_elements, capacity_elements } => Self::AppendOverflow {
                node: input,
                valid_elements: valid_elements as usize,
                new_elements: 0,
                capacity_elements: capacity_elements as usize,
            },
            other => Self::KvCarry(other.to_string()),
        }
    }
}

impl From<DriverError> for CudaDriverError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

// the shared name-keyed identity rule (`crate::resident_identity`) so a fresh plan or arena finds the weight another plan uploaded and a recycled host address is refused.
type ResidentBuffers<R = CudarcRuntime> = ResidentLedger<CudaGraphBuffer<R>>;

fn buffer_elements<R: GpuRuntime>(buffer: &CudaGraphBuffer<R>) -> u64 {
    match buffer {
        CudaGraphBuffer::F32(values) => R::len(values) as u64,
        CudaGraphBuffer::Bytes(values) => R::len(values.as_ref()) as u64,
    }
}

fn cuda_float_bytes(extents: &[u64]) -> Result<u64, CudaDriverError> {
    extents.iter().try_fold(4_u64, |bytes, extent| {
        bytes
            .checked_mul(*extent)
            .ok_or(CudaDriverError::ResourceOverflow)
    })
}

fn cuda_input_contract(
    block: &QuantizedBlock<'_>,
) -> Result<(Option<Codec>, bool, usize), CudaDriverError> {
    let (codec, integer) = match block {
        QuantizedBlock::Float32(_) => (None, false),
        QuantizedBlock::Int32(_) => (None, true),
        QuantizedBlock::Packed { codec, .. } => (Some(*codec), false),
        QuantizedBlock::PackedStrided { codec, .. } => (Some(*codec), false),
    };
    Ok((
        codec,
        integer,
        block
            .element_count()
            .map_err(|_| CudaDriverError::InputContractChanged {
                node: NodeId(0),
                reason: "malformed packed input",
            })?,
    ))
}

#[derive(Debug)]
pub struct CompileCounters {
    compilations: AtomicU64,
    nvrtc_micros: AtomicU64,
    load_micros: AtomicU64,
    disk_cache_hits: AtomicU64,
    disk_read_micros: AtomicU64,
    module_load_micros: AtomicU64,
    function_load_micros: AtomicU64,
    build_micros: AtomicU64,
    key_micros: AtomicU64,
    cubin_arch: OnceLock<&'static str>,
    header_digests: Mutex<BTreeMap<PathBuf, [u8; 32]>>,
    disk_store: PtxStore,
}

impl CompileCounters {
    fn new(disk_store: PtxStore) -> Self {
        Self {
            compilations: AtomicU64::new(0),
            nvrtc_micros: AtomicU64::new(0),
            load_micros: AtomicU64::new(0),
            disk_cache_hits: AtomicU64::new(0),
            disk_read_micros: AtomicU64::new(0),
            module_load_micros: AtomicU64::new(0),
            function_load_micros: AtomicU64::new(0),
            build_micros: AtomicU64::new(0),
            key_micros: AtomicU64::new(0),
            cubin_arch: OnceLock::new(),
            header_digests: Mutex::new(BTreeMap::new()),
            disk_store,
        }
    }

    fn add(counter: &AtomicU64, started: Instant) {
        counter.fetch_add(elapsed_micros(started), Ordering::Relaxed);
    }

    fn lookup_micros(&self) -> [u64; 5] {
        [
            self.build_micros.load(Ordering::Relaxed),
            self.key_micros.load(Ordering::Relaxed),
            self.disk_read_micros.load(Ordering::Relaxed),
            self.module_load_micros.load(Ordering::Relaxed),
            self.function_load_micros.load(Ordering::Relaxed),
        ]
    }
}

/// How a launch completes: waited on here, or left on the stream with an optional window of a
/// fault array the caller reads after its own synchronization.
enum Completion<'fault, R: GpuRuntime> {
    Synchronize,
    Defer(Option<(&'fault R::Buffer<u32>, usize, usize)>),
}

/// Carried state parked by an evicted plan, shared by every plan of the driver.
type ParkedState<R> = Arc<Mutex<BTreeMap<NodeId, (NodeId, CudaGraphBuffer<R>)>>>;
/// Appended-row caches parked by an evicted plan.
type ParkedAppends<R> = Arc<Mutex<BTreeMap<NodeId, (u64, CudaGraphBuffer<R>)>>>;

/// One CUDA context and stream owned by a caller's inference runtime.
#[derive(Debug, Clone)]
pub struct CudaDriver<R: GpuRuntime = CudarcRuntime> {
    runtime: R,
    resident_buffers: Arc<Mutex<ResidentBuffers<R>>>,
    packed_upload_bytes: Arc<AtomicU64>,
    graph_uploads: Arc<AtomicU64>,
    graph_launches: Arc<AtomicU64>,
    // carried recurrent state whose plan was evicted: keyed by state input node, valued by its output node and device buffer, so the replacement plan adopts it without a host round trip.
    parked_state: ParkedState<R>,
    // appended-row caches whose plan was evicted: keyed by cache input node, valued by its valid row count and device buffer.
    parked_appends: ParkedAppends<R>,
    // device bytes already in use when this driver was created, so other contexts and the display are not charged to it.
    baseline_used_bytes: u64,
}

/// The grant every `CudaPlan` binds under; the operand-dtype table reads the same one, so the
/// table names the dtypes the bind produced. HIP binds through its own driver in `rocm`.
fn plan_grant() -> Executes {
    cuda_executes(&Target::CUDA)
}

fn multiprocessor_count(context: &CudaContext) -> Result<u32, CudaDriverError> {
    let count = context.attribute(cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    Ok(u32::try_from(count).unwrap_or(0))
}

/// Device identity and successful graph upload/launch submissions from one CUDA driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceSnapshot {
    pub ordinal: usize,
    pub name: String,
    pub graph_uploads: u64,
    pub graph_launches: u64,
}

#[derive(Debug)]
pub(crate) struct CachedCudaModule {
    pub(crate) source: String,
    pub(crate) module: Arc<CudaModule>,
    pub(crate) functions: BTreeMap<String, CudaFunction>,
}

/// Persistent f32 storage for a graph's named inputs and intermediate nodes.
/// The map owns allocations across dispatches; a shape change is the only
/// event that replaces a node's device buffer.
#[derive(Debug)]
pub struct CudaF32Arena<R: GpuRuntime = CudarcRuntime> {
    buffers: BTreeMap<NodeId, CudaGraphBuffer<R>>,
    route_compaction: BTreeMap<NodeId, R::Buffer<u32>>,
    uniforms: BTreeMap<NodeId, R::Buffer<u8>>,
    uniform_host: BTreeMap<NodeId, Vec<u8>>,
    resident_sources: BTreeMap<NodeId, ResidentIdentity>,
    pool: BTreeMap<usize, Vec<R::Buffer<f32>>>,
    allocations: u64,
    launch_profile: LaunchProfile,
}

impl<R: GpuRuntime> Default for CudaF32Arena<R> {
    fn default() -> Self {
        Self {
            buffers: BTreeMap::new(),
            route_compaction: BTreeMap::new(),
            uniforms: BTreeMap::new(),
            uniform_host: BTreeMap::new(),
            resident_sources: BTreeMap::new(),
            pool: BTreeMap::new(),
            allocations: 0,
            launch_profile: LaunchProfile::default(),
        }
    }
}

/// Which intermediate buffers return to the arena pool after each launch.
/// `release[position]` lists the nodes whose last reader is the kernel at
/// `position`; `pooled` is every node that takes its buffer from the pool.
#[derive(Debug, Default)]
struct ReleasePlan {
    release: Vec<Vec<NodeId>>,
    pooled: BTreeSet<NodeId>,
}

impl ReleasePlan {
    fn build(resolved: &[BoundOp], kernels: &[CudaKernel], outputs: &[NodeId]) -> Self {
        let extra = kernels
            .iter()
            .flat_map(|kernel| kernel.extra_outputs.iter().map(|(node, _)| *node));
        let pinned: BTreeSet<NodeId> = outputs.iter().copied().chain(extra).collect();
        let mut produced: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut last_read: BTreeMap<NodeId, usize> = BTreeMap::new();
        for (position, (bound, kernel)) in resolved.iter().zip(kernels).enumerate() {
            let reads = kernel_read_nodes(kernel);
            reads.for_each(|node| {
                last_read.insert(node, position);
            });
            produced.insert(bound.node, position);
        }
        let mut plan = Self {
            release: vec![Vec::new(); resolved.len()],
            pooled: BTreeSet::new(),
        };
        for (bound, kernel) in resolved.iter().zip(kernels) {
            let poolable = !pinned.contains(&bound.node)
                && !matches!(bound.kind, BoundOpKind::Constant { .. })
                && kernel.extra_outputs.is_empty();
            if poolable {
                let position = produced[&bound.node];
                let last = last_read.get(&bound.node).copied().unwrap_or(position);
                plan.release[last.max(position)].push(bound.node);
                plan.pooled.insert(bound.node);
            }
        }
        plan
    }

    fn pool_shape(&self, produced: &[(NodeId, usize)]) -> BTreeMap<usize, usize> {
        let mut free: BTreeMap<usize, usize> = BTreeMap::new();
        let mut created: BTreeMap<usize, usize> = BTreeMap::new();
        let lengths: BTreeMap<NodeId, usize> = produced.iter().copied().collect();
        for (position, (node, length)) in produced.iter().enumerate() {
            if self.pooled.contains(node) {
                let available = free.entry(*length).or_default();
                match available.checked_sub(1) {
                    Some(remaining) => *available = remaining,
                    None => *created.entry(*length).or_default() += 1,
                }
            }
            for released in &self.release[position] {
                let length = lengths.get(released).copied().unwrap_or_default();
                *free.entry(length).or_default() += 1;
            }
        }
        created
    }
}

/// The release schedule and pool size classes of one plan, built once and
/// rebuilt only when a refit changes a kernel's output length.
#[derive(Debug, Default)]
struct PoolSchedule {
    plan: ReleasePlan,
    peak: BTreeMap<usize, usize>,
}

/// The `f32` words the device buffer for `bound` holds. A `BlockQuantize` output is `UINT8`
/// bytes (`BoundOp::output_len`) that this driver stores packed into `f32` words, which
/// `cuda::block_quantize::servable_layout` guarantees by requiring word-sized blocks.
fn device_words(bound: &BoundOp) -> usize {
    match &bound.kind {
        BoundOpKind::BlockQuantize { .. } => bound.output_len() / core::mem::size_of::<f32>(),
        _ => bound.output_len(),
    }
}

impl PoolSchedule {
    fn build(resolved: &[BoundOp], kernels: &[CudaKernel], outputs: &[NodeId]) -> Self {
        let plan = ReleasePlan::build(resolved, kernels, outputs);
        let produced: Vec<(NodeId, usize)> = resolved
            .iter()
            .map(|bound| (bound.node, device_words(bound)))
            .collect();
        let peak = plan.pool_shape(&produced);
        Self { plan, peak }
    }

    fn relength(&mut self, resolved: &[BoundOp]) {
        let produced: Vec<(NodeId, usize)> = resolved
            .iter()
            .map(|bound| (bound.node, device_words(bound)))
            .collect();
        self.peak = self.plan.pool_shape(&produced);
    }
}

fn kernel_read_nodes(kernel: &CudaKernel) -> impl Iterator<Item = NodeId> + '_ {
    std::iter::once(kernel)
        .chain(kernel.prologue.iter())
        .flat_map(|launch| launch.bindings.iter())
        .filter_map(|binding| match binding {
            Binding::Input(node) | Binding::Indices(node) => Some(*node),
            _ => None,
        })
}

impl<R: GpuRuntime> CudaF32Arena<R>
where CudaDriverError: From<R::Error> + From<R::BuildError>
{
    fn seat_pooled(&mut self, node: NodeId, length: usize) {
        if self.buffers.contains_key(&node) {
            return;
        }
        if let Some(buffer) = self.pool.get_mut(&length).and_then(Vec::pop) {
            self.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        }
    }

    fn release_to_pool(&mut self, nodes: &[NodeId]) {
        for node in nodes {
            if let Some(CudaGraphBuffer::F32(buffer)) = self.buffers.remove(node) {
                self.pool.entry(R::len(&buffer)).or_default().push(buffer);
            }
        }
    }

    fn pool_bytes(&self) -> u64 {
        self.pool
            .iter()
            .map(|(length, buffers)| (*length as u64) * 4 * buffers.len() as u64)
            .sum()
    }
}

#[derive(Debug, Default)]
struct LaunchProfile {
    enabled: bool,
    phase_nanos: [u64; LAUNCH_PHASES],
    output_allocs: u64,
    extra_allocs: u64,
    uniform_uploads: u64,
    function_lookups: u64,
}

/// The fewest rows the prefill kernels share one form for: below it cached attention runs untiled.
const ROW_CLASS_MIN: u64 = 16;

#[derive(Default)]
struct FollowMicros {
    grid: u64,
    uniform: u64,
    attention: u64,
}

fn learning_rows(rows: u64) -> Option<[u64; 2]> {
    let mut others = [rows / 2, rows / 4, rows * 2]
        .into_iter()
        .filter(|candidate| *candidate >= ROW_CLASS_MIN && *candidate != rows);
    Some([others.next()?, others.next()?])
}

const LAUNCH_PHASES: usize = 6;
const LAUNCH_PHASE_NAMES: [&str; LAUNCH_PHASES] = [
    "upload_uniform",
    "extra_outputs",
    "output_buffer",
    "fault_buffer",
    "launch_builder_args",
    "launch_call",
];

fn report_launch_profile(profile: &LaunchProfile, kind_rows: &BTreeMap<&str, (u64, u64)>) {
    let mut kinds: Vec<_> = kind_rows.iter().collect();
    kinds.sort_by_key(|(_, (_, nanos))| std::cmp::Reverse(*nanos));
    for (kind, (count, nanos)) in kinds.into_iter().take(8) {
        eprintln!(
            "cuda_host_timing: submit_kind={kind} launches={count} total_us={} per_launch_us={:.2}",
            nanos / 1_000,
            *nanos as f64 / 1_000.0 / *count as f64
        );
    }
    for (name, nanos) in LAUNCH_PHASE_NAMES.iter().zip(profile.phase_nanos) {
        eprintln!(
            "cuda_host_timing: launch_phase={name} total_us={}",
            nanos / 1_000
        );
    }
    eprintln!(
        "cuda_host_timing: launch_counts output_allocs={} extra_allocs={} uniform_uploads={} function_lookups={}",
        profile.output_allocs,
        profile.extra_allocs,
        profile.uniform_uploads,
        profile.function_lookups
    );
}

impl LaunchProfile {
    fn mark(&mut self, phase: usize, since: &mut Option<Instant>) {
        if let Some(start) = since {
            let now = Instant::now();
            let nanos = now.duration_since(*start).as_nanos();
            self.phase_nanos[phase] += u64::try_from(nanos).unwrap_or(u64::MAX);
            *start = now;
        }
    }
}

#[derive(Debug, Clone)]
enum CudaGraphBuffer<R: GpuRuntime = CudarcRuntime> {
    F32(R::Buffer<f32>),
    // shared with the driver's resident cache, so a clone is a refcount bump rather than a device copy.
    Bytes(Arc<R::Buffer<u8>>),
}

/// The event pair bracketing one launch of a timed execute, with the operand bytes read at launch.
// the fields are read only by the `instrument`-gated `op_timing`; the serving path never builds one
#[cfg_attr(not(feature = "instrument"), allow(dead_code))]
struct LaunchEvents<R: GpuRuntime> {
    position: usize,
    start: R::Event,
    end: R::Event,
    operand_bytes: u64,
}

fn push_buffer<'a, 'b, R: GpuRuntime + 'a>(args: &'b mut R::Launch<'a>, buffer: &'a CudaGraphBuffer<R>) -> &'b mut R::Launch<'a> {
    match buffer {
        CudaGraphBuffer::F32(input) => args.push(input),
        CudaGraphBuffer::Bytes(input) => args.push(input.as_ref()),
    }
}

/// A reusable CUDA graph for the currently connected dense/reduction CUDA ABI.
/// Planning resolves shapes, fusion, dead nodes, and kernel structure once;
/// execution refreshes only named inputs and reuses the device arena.
#[derive(Debug)]
pub struct DevicePlan<R: GpuRuntime = CudarcRuntime> {
    constraints: Option<ResourceConstraints>,
    block_contract: Vec<(Option<Codec>, bool, usize)>,
    driver: CudaDriver<R>,
    program: Vec<Op>,
    shapes: Shapes,
    resolved: Vec<BoundOp>,
    kernels: Vec<CudaKernel>,
    uniforms: Vec<Vec<u8>>,
    block_nodes: Vec<NodeId>,
    outputs: Box<[NodeId]>,
    // boxed: the arena is three maps (72 bytes) that only move when the plan is built, keeping CudaPlan inside clippy's large_enum_variant gap against the cpu plan
    arena: Box<CudaF32Arena<R>>,
    numeric_policy: NumericPolicy,
    activation_storage: Option<NumericFormat>,
    packed_operands: PackedOperands,
    exact_activations: bool,
    // resident_nodes is sorted; boxed slices keep CudaPlan under clippy's large_enum_variant gap against the cpu plan
    resident_nodes: Box<[NodeId]>,
    // cumulative per-plan totals; read by the decode path's per-step telemetry, never printed per dispatch.
    refit_calls: u64,
    patch_memo: Option<(Vec<NodeId>, Vec<usize>)>,
    // the row count the bound ops were last bound at, and the proof that every integer in them is affine in it: `None` until tried, `Some(None)` once refused
    bound_rows: u64,
    // boxed: keeps CudaPlan inside clippy's large_enum_variant gap against the cpu plan
    rows_template: Box<Option<Option<RowTemplate>>>,
    // the extents of every node as a line in the row count, valid while the other symbols equal the first element
    shapes_template: Box<Option<SymbolsTemplate>>,
    // one resolved device function per kernel, so a launch skips the module-cache text comparison
    functions: Vec<Option<R::Function>>,
    emit_calls: u64,
    emit_reused: u64,
    pack_calls: u64,
    repack_calls: u64,
    carry: Box<StateCarry>,
    pool: Box<PoolSchedule>,
}

/// The plan the CUDA executor builds, on the cudarc runtime. External callers name this alias.
pub type CudaPlan = DevicePlan<CudarcRuntime>;

/// The plan the same executor builds on the HIP runtime.
pub type HipPlan = DevicePlan<rocm::HipRuntime>;

/// Device-carried state between calls. Both machines are sans-IO: [`StateSwap`] decides which recurrent
/// state buffers swap and park, [`KvCarry`] every cache-input row append, growth, truncation and handover;
/// `host_bytes` counts every carried byte that crossed PCIe in either direction.
#[derive(Debug, Default)]
struct StateCarry {
    state: StateSwap<NodeId>,
    host_bytes: u64,
    rows: KvCarry<NodeId>,
    cached_rows: u64,
    fresh_cached_rows: Option<u64>,
    encodes: BTreeMap<NodeId, KvEncode>,
}

/// The format a carried cache input stores and the encode rendered for it: every appended row goes through it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct KvEncode {
    format: NumericFormat,
    rendered: RenderedEncode,
}

fn kv_encode(target: Target, format: &NumericFormat) -> Option<KvEncode> {
    let dialect = crate::cuda_encode::CudaEncode { target };
    let rendered = crate::encode_render::format_encode_kernel(&dialect, format, "kv_carry_encode").ok()?;
    Some(KvEncode { format: *format, rendered })
}

/// Numeric and activation policy for one plan, plus the runtime config when the
/// plan is built from one. Not a `RuntimeConfig`: the config-less entry points
/// plan with fusion enabled, which `RuntimeConfig::default()` (fusion off) would
/// silently change.
struct PlanPolicy<'config> {
    /// A descriptor's block activation format (`activation_storage`); `None` keeps the default pair.
    activation_storage: Option<NumericFormat>,
    numeric_policy: NumericPolicy,
    exact_activations: bool,
    config: Option<&'config RuntimeConfig>,
}

impl<R: GpuRuntime> CudaF32Arena<R>
where CudaDriverError: From<R::Error> + From<R::BuildError>
{
    /// Number of device allocations performed by this arena.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }
}

impl CudaDriver<CudarcRuntime> {
    /// Initializes the CUDA primary context for `ordinal`.
    ///
    /// This is the first real device boundary: on a machine without a
    /// visible NVIDIA device it returns a typed `Driver` error rather than
    /// pretending that emitted CUDA source is executable.
    pub fn new(ordinal: usize) -> Result<Self, CudaDriverError> {
        let created = Instant::now();
        prefetch_nvrtc_version();
        // the dynamic loader otherwise panics before it can return a driver error.
        if !unsafe { cuda_sys::is_culib_present() } {
            return Err(CudaDriverError::Driver(
                "cuda driver library unavailable".into(),
            ));
        }
        let library_us = elapsed_micros(created);
        let step = Instant::now();
        let context = CudaContext::new(ordinal)?;
        let context_us = elapsed_micros(step);
        let step = Instant::now();
        let device = EmitDevice { multiprocessors: multiprocessor_count(&context)? };
        let multiprocessors_us = elapsed_micros(step);
        // every launch runs on this one default stream, so per-buffer read/write events order nothing and cost two cuEventCreate per buffer
        unsafe { context.disable_event_tracking() };
        let stream = context.default_stream();
        let step = Instant::now();
        let (free, total) = context.mem_get_info()?;
        let mem_info_us = elapsed_micros(step);
        let baseline_used_bytes = total.saturating_sub(free) as u64;
        let step = Instant::now();
        let compile_counters = Arc::new(CompileCounters::new(kernel_disk_store()));
        let store_us = elapsed_micros(step);
        proxima_telemetry::debug!(
            library_us = library_us,
            context_us = context_us,
            multiprocessors_us = multiprocessors_us,
            mem_info_us = mem_info_us,
            store_us = store_us,
            total_us = elapsed_micros(created),
            "cuda driver created"
        );
        Ok(Self {
            runtime: CudarcRuntime::new(context, stream, device, compile_counters),
            resident_buffers: Arc::new(Mutex::new(ResidentLedger::default())),
            packed_upload_bytes: Arc::new(AtomicU64::new(0)),
            graph_uploads: Arc::new(AtomicU64::new(0)),
            graph_launches: Arc::new(AtomicU64::new(0)),
            parked_state: Arc::default(),
            parked_appends: Arc::default(),
            baseline_used_bytes,
        })
    }

    /// Reads the selected CUDA device and the driver's graph activity counters.
    pub fn device_snapshot(&self) -> Result<CudaDeviceSnapshot, CudaDriverError> {
        Ok(CudaDeviceSnapshot {
            ordinal: self.runtime.context.ordinal(),
            name: self.runtime.context.name()?,
            graph_uploads: self.graph_uploads.load(Ordering::Relaxed),
            graph_launches: self.graph_launches.load(Ordering::Relaxed),
        })
    }

    /// Loads a precompiled PTX module and returns one of its entry points; see [`CudarcRuntime::precompiled_function`].
    ///
    /// # Errors
    /// The module failed to load or the entry is missing.
    pub fn precompiled_function(&self, module_key: &str, entry: &str, ptx_source: &str) -> Result<CudaFunction, CudaDriverError> {
        self.runtime.precompiled_function(module_key, entry, ptx_source)
    }

    /// Runs the same numerical smoke through a hand-authored PTX module.
    ///
    /// The test intentionally uses the existing launch ABI (input pointer,
    /// output pointer, uniform pointer), proving that the driver path works
    /// independently of runtime CUDA-C/NVRTC availability.
    pub fn smoke_add_one_precompiled(&self, input: &[f32]) -> Result<Vec<f32>, CudaDriverError> {
        let n =
            u32::try_from(input.len()).map_err(|_| CudaDriverError::InputTooLarge(input.len()))?;
        let function =
            self.precompiled_function("smoke_add_one", "proxima_add_one", PRECOMPILED_ADD_ONE_PTX)?;
        let device_input = self.runtime.upload(input)?;
        let mut device_output = self.runtime.alloc_zeros::<f32>(input.len())?;
        let device_uniforms = self.runtime.upload::<u8>(&n.to_ne_bytes())?;
        let mut args = self.runtime.launch_arguments(&function);
        args.push(&device_input);
        args.push_mut(&mut device_output);
        args.push(&device_uniforms);
        unsafe {
            self.runtime.launch(args, LaunchShape {
                grid_dim: ((input.len() as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.runtime.synchronize()?;
        Ok(self.runtime.download(&device_output)?)
    }

}

impl<R: GpuRuntime> CudaDriver<R>
where CudaDriverError: From<R::Error> + From<R::BuildError>
{
    /// Number of NVRTC/module compilations performed by this driver. A
    /// steady-state plan should stop increasing this after its first dispatch
    /// of each distinct kernel identity.
    #[must_use]
    pub fn kernel_compilations(&self) -> u64 {
        self.runtime.compile_counters().compilations.load(Ordering::Relaxed)
    }

    /// Kernels whose PTX was loaded from the on-disk cache instead of NVRTC.
    /// A warm run with unchanged keys stops increasing `kernel_compilations`
    /// and grows this instead.
    #[must_use]
    pub fn disk_cache_hits(&self) -> u64 {
        self.runtime.compile_counters()
            .disk_cache_hits
            .load(Ordering::Relaxed)
    }

    /// Cumulative wall-clock microseconds spent in NVRTC source-to-PTX
    /// compilation across this driver's `kernel_compilations`.
    #[must_use]
    pub fn nvrtc_micros(&self) -> u64 {
        self.runtime.compile_counters().nvrtc_micros.load(Ordering::Relaxed)
    }

    /// Cumulative wall-clock microseconds spent in `cuModuleLoadData` (PTX to
    /// device module, including driver JIT) across this driver's
    /// `kernel_compilations`.
    #[must_use]
    pub fn load_micros(&self) -> u64 {
        self.runtime.compile_counters().load_micros.load(Ordering::Relaxed)
    }

    /// Returns the CUDA driver's current `(free_bytes, total_bytes)` view.
    /// This is the authoritative device-memory measurement used by a tier
    /// fit check; host RSS and checkpoint file size are only proxies.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaDriverError> {
        Ok(self.runtime.memory_info()?)
    }

    /// Device bytes in use now beyond what was in use when this driver was created, `None` when the driver cannot report.
    #[must_use]
    pub fn allocated_bytes(&self) -> Option<u64> {
        let (free, total) = self.memory_info().ok()?;
        Some((total.saturating_sub(free) as u64).saturating_sub(self.baseline_used_bytes))
    }

    /// Removes device copies for resident input nodes after their host source changes.
    pub fn invalidate_resident_buffers(&self, nodes: &[NodeId]) -> Result<(), CudaDriverError> {
        let mut buffers = self.resident_buffers.lock();
        if !nodes.is_empty() {
            // graph-local node ids cannot identify entries keyed by host source identity.
            buffers.clear();
        }
        Ok(())
    }

    /// Creates an empty persistent buffer arena owned by this driver.
    #[must_use]
    pub fn new_arena(&self) -> CudaF32Arena<R> {
        CudaF32Arena::default()
    }

    /// Host-to-device bytes of packed weights uploaded by this driver. A
    /// weight marked resident adds to it once per loaded model, never again.
    #[must_use]
    pub fn packed_upload_bytes(&self) -> u64 {
        self.packed_upload_bytes.load(Ordering::Relaxed)
    }

    /// Evicts the device buffers cached for these resident weight names, so a
    /// dropped model frees its device memory and its address range cannot
    /// alias a later mapping.
    pub fn release_resident_names<'name>(&self, names: impl IntoIterator<Item = &'name str>) {
        self.resident_buffers.lock().release_names(names);
    }

    /// Evicts every device buffer cached for a host block starting inside one
    /// of these host address ranges: the same eviction as
    /// [`Self::release_resident_names`], addressed by host range for an owner
    /// that frees bytes without knowing names. The shared identity rule
    /// ([`ResidentLedger`]) already refuses a recycled address, so this is
    /// the reclaim of device memory, not the only guard.
    pub fn release_resident_ranges(&self, ranges: &[Range<usize>]) {
        self.resident_buffers.lock().release_ranges(ranges);
    }

    /// Number of host blocks currently holding a cached resident device buffer.
    #[must_use]
    pub fn resident_buffer_count(&self) -> usize {
        self.resident_buffers.lock().len()
    }

    /// Encodes `rows` of the device binary32 `source` into `destination` at `rows.byte_offset`
    /// with the kernel `rendered` was rendered as for `format`: Metal's encode dispatch contract.
    /// Bytes outside the packed range are untouched. The caller owns synchronizing the stream.
    ///
    /// # Errors
    /// [`CudaDriverError::BindingContract`] when the rows are not whole blocks of `format`, are not
    /// whole threads, do not fit `source` or `destination`, or `Row` is not one host row element.
    pub fn encode_rows<Row: DeviceElement>(
        &self,
        rendered: &RenderedEncode,
        format: &NumericFormat,
        source: &R::Buffer<Row>,
        destination: &mut R::Buffer<u8>,
        rows: EncodeRows,
    ) -> Result<(), CudaDriverError> {
        let length = R::len(destination);
        let mut window = R::view_mut(destination, 0..length);
        self.encode_rows_into(rendered, format, source, &mut window, rows)
    }

    /// [`Self::encode_rows`] over a window of device bytes, so a cache whose storage is not a byte slice is encoded through the same dispatch.
    ///
    /// # Errors
    /// As [`Self::encode_rows`].
    pub fn encode_rows_into<Row: DeviceElement>(
        &self,
        rendered: &RenderedEncode,
        format: &NumericFormat,
        source: &R::Buffer<Row>,
        destination: &mut R::ViewMut<'_, u8>,
        rows: EncodeRows,
    ) -> Result<(), CudaDriverError> {
        if row_bytes() != Some(core::mem::size_of::<Row>()) {
            return Err(CudaDriverError::BindingContract);
        }
        let range = rows.byte_range(format).ok_or(CudaDriverError::BindingContract)?;
        let threads = rows.threads(rendered.elements_per_thread);
        if threads * rendered.elements_per_thread != rows.element_count || rows.element_count > R::len(source) || range.end > R::view_mut_len(destination) {
            return Err(CudaDriverError::BindingContract);
        }
        let count = u32::try_from(threads).map_err(|_| CudaDriverError::InputTooLarge(threads))?;
        let kernel = encode_kernel(rendered, threads)?;
        let function = self.runtime.build_function(&kernel)?;
        let mut window = R::view_mut_range(destination, range);
        let mut args = self.runtime.launch_arguments(&function);
        args.push(source);
        args.push_view_mut(&mut window);
        args.push_value(&count);
        let config = LaunchShape {
            grid_dim: (launch_blocks(&kernel.grid)?, 1, 1),
            block_dim: (kernel.grid.block_width_or_default() as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { self.runtime.launch(args, config) }?;
        Ok(())
    }

    /// Encodes `count` host row values from `source` into `cache` at the packed position of element `offset`, the cache's storage read as bytes.
    fn encode_into_cache<Row: DeviceElement>(
        &self,
        encode: &KvEncode,
        source: &R::Buffer<Row>,
        cache: &mut R::Buffer<Row>,
        offset: usize,
        count: usize,
    ) -> Result<(), CudaDriverError> {
        let byte_offset = encode.format.packed_len(offset).ok_or(CudaDriverError::BindingContract)?;
        let mut window = R::bytes_mut(cache).ok_or(CudaDriverError::BindingContract)?;
        self.encode_rows_into(&encode.rendered, &encode.format, source, &mut window, EncodeRows { element_count: count, byte_offset })
    }

}

impl CudarcRuntime {
    pub(crate) fn function_for(&self, kernel: &CudaKernel) -> Result<CudaFunction, CudaDriverError> {
        self.cached_function(&kernel.entry, &kernel.entry, &kernel.source, || {
            self.build_kernel_module(kernel)
        })
    }

    // nvrtc has the same panic-on-missing-library loader contract; a disk hit skips nvrtc and counts no compilation.
    fn build_kernel_module(&self, kernel: &CudaKernel) -> Result<Arc<CudaModule>, CudaDriverError> {
        let started = Instant::now();
        let module = self.build_kernel_module_untimed(kernel);
        CompileCounters::add(&self.compile_counters.build_micros, started);
        module
    }

    fn build_kernel_module_untimed(&self, kernel: &CudaKernel) -> Result<Arc<CudaModule>, CudaDriverError> {
        let header_dir = cuda_header_dir(&kernel.source)?;
        let options = nvrtc_options(header_dir.as_deref());
        let header = self.header_stand_in(header_dir.as_deref())?;
        if let Some(module) = self.cubin_module(kernel, header_dir.as_deref(), header.as_ref()) {
            return Ok(module);
        }
        let key = self.kernel_cache_key(&kernel.source, header.as_ref().map(|digest| digest.as_slice()), &options)?;
        let read_started = Instant::now();
        let cached = cached_ptx(&self.compile_counters.disk_store, &key);
        CompileCounters::add(&self.compile_counters.disk_read_micros, read_started);
        if let Some(text) = cached {
            self.compile_counters
                .disk_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            proxima_telemetry::debug!(entry = %kernel.entry, "kernel loaded from disk cache");
            return self.timed_load_module(Ptx::from_src(text));
        }
        require_nvrtc()?;
        let nvrtc_started = Instant::now();
        let ptx = compile_ptx_with_opts(&kernel.source, options)
            .map_err(|error| CudaDriverError::Nvrtc(error.to_string()))?;
        let nvrtc_us = elapsed_micros(nvrtc_started);
        store_ptx(&self.compile_counters.disk_store, &key, &ptx);
        let load_started = Instant::now();
        let module = self.load_module(ptx)?;
        let load_us = elapsed_micros(load_started);
        self.record_compile(&kernel.entry, kernel.source.len(), nvrtc_us, load_us);
        Ok(module)
    }

    fn header_stand_in(&self, header_dir: Option<&Path>) -> Result<Option<[u8; 32]>, CudaDriverError> {
        let Some(dir) = header_dir else {
            return Ok(None);
        };
        let mut digests = self.compile_counters.header_digests.lock();
        if let Some(digest) = digests.get(dir) {
            return Ok(Some(*digest));
        }
        let digest = header_digest(&cuda_header_bytes(header_dir)?.unwrap_or_default());
        digests.insert(dir.to_path_buf(), digest);
        Ok(Some(digest))
    }

    fn timed_load_module(&self, image: Ptx) -> Result<Arc<CudaModule>, CudaDriverError> {
        let started = Instant::now();
        let module = self.load_module(image);
        CompileCounters::add(&self.compile_counters.module_load_micros, started);
        Ok(module?)
    }

    fn cubin_arch(&self) -> Result<&'static str, CudaDriverError> {
        if let Some(arch) = self.compile_counters.cubin_arch.get() {
            return Ok(arch);
        }
        let (major, minor) = self.context.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        Ok(self.compile_counters.cubin_arch.get_or_init(|| arch))
    }

    // a cubin loads without the driver JIT; any failure here falls back to the ptx path.
    fn cubin_module(
        &self,
        kernel: &CudaKernel,
        header_dir: Option<&Path>,
        header: Option<&[u8; 32]>,
    ) -> Option<Arc<CudaModule>> {
        if std::env::var_os("PROXIMA_CUDA_PTX_ONLY").is_some() {
            return None;
        }
        let arch = self.cubin_arch().ok()?;
        let mut options = nvrtc_options(header_dir);
        options.arch = Some(arch);
        let key = self.kernel_cache_key(&kernel.source, header.map(|digest| digest.as_slice()), &options).ok()?;
        let image = self
            .cached_cubin(&key)
            .or_else(|| self.compiled_cubin(kernel, &key, header_dir, arch))?;
        match self.timed_load_module(Ptx::from_binary(image)) {
            Ok(module) => Some(module),
            Err(error) => {
                proxima_telemetry::debug!(entry = %kernel.entry, error = %error, "cubin load refused; using ptx");
                None
            }
        }
    }

    fn cached_cubin(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let started = Instant::now();
        let entry = self.compile_counters.disk_store.get(&key.to_hex()).ok().flatten();
        CompileCounters::add(&self.compile_counters.disk_read_micros, started);
        let image = decode_binary_entry(key, &entry?).ok()?.to_vec();
        self.compile_counters
            .disk_cache_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(image)
    }

    fn compiled_cubin(
        &self,
        kernel: &CudaKernel,
        key: &CacheKey,
        header_dir: Option<&Path>,
        arch: &str,
    ) -> Option<Vec<u8>> {
        require_nvrtc().ok()?;
        let started = Instant::now();
        let image = match compile_cubin(&kernel.source, header_dir, arch) {
            Ok(image) => image,
            Err(error) => {
                proxima_telemetry::debug!(entry = %kernel.entry, error = %error, "cubin compile refused; using ptx");
                return None;
            }
        };
        let nvrtc_us = elapsed_micros(started);
        if let Err(error) = self
            .compile_counters
            .disk_store
            .put(&key.to_hex(), &encode_binary_entry(key, &image))
        {
            proxima_telemetry::warn!(entry = %kernel.entry, error = %error, "cubin cache write failed; kernel stays uncached");
        }
        self.record_compile(&kernel.entry, kernel.source.len(), nvrtc_us, 0);
        Some(image)
    }

    fn kernel_cache_key(
        &self,
        source: &str,
        header: Option<&[u8]>,
        options: &CompileOptions,
    ) -> Result<CacheKey, CudaDriverError> {
        let started = Instant::now();
        let nvrtc = nvrtc_version()?;
        let capability = self.context.compute_capability()?;
        let key = CacheKey::new(source, header, nvrtc, options, capability);
        CompileCounters::add(&self.compile_counters.key_micros, started);
        Ok(key)
    }

    fn record_compile(&self, entry: &str, source_bytes: usize, nvrtc_us: u64, load_us: u64) {
        self.compile_counters
            .compilations
            .fetch_add(1, Ordering::Relaxed);
        self.compile_counters
            .nvrtc_micros
            .fetch_add(nvrtc_us, Ordering::Relaxed);
        self.compile_counters
            .load_micros
            .fetch_add(load_us, Ordering::Relaxed);
        crate::backend::record_pipeline_compile((nvrtc_us + load_us) * 1_000);
        proxima_telemetry::debug!(
            entry = %entry,
            source_bytes = source_bytes,
            nvrtc_us = nvrtc_us,
            load_us = load_us,
            "kernel compiled"
        );
    }

    /// Looks a function up by `(cache_key, source)`, so distinct sources that
    /// share one entry name coexist instead of evicting each other, and keeps
    /// the loaded handle so a repeat launch does no module or function load.
    fn cached_function(
        &self,
        cache_key: &str,
        entry: &str,
        source: &str,
        build_module: impl FnOnce() -> Result<Arc<CudaModule>, CudaDriverError>,
    ) -> Result<CudaFunction, CudaDriverError> {
        let cached = self
            .modules
            .lock()
            .get(cache_key)
            .and_then(|slot| slot.iter().find(|cached| cached.source == source))
            .map(|cached| (cached.module.clone(), cached.functions.get(entry).cloned()));
        let module = match cached {
            Some((_, Some(function))) => return Ok(function),
            Some((module, None)) => module,
            None => build_module()?,
        };
        let load_started = Instant::now();
        let function = self.load_function(&module, entry);
        CompileCounters::add(&self.compile_counters.function_load_micros, load_started);
        let function = function?;
        proxima_telemetry::debug!(
            entry = %entry,
            registers = function.num_regs().unwrap_or(-1),
            shared_bytes = function.shared_size_bytes().unwrap_or(-1),
            local_bytes = function.local_size_bytes().unwrap_or(-1),
            max_threads = function.max_threads_per_block().unwrap_or(-1),
            "kernel function loaded; occupancy follows from these resource counts"
        );
        self.store_function(cache_key, source, entry, module, function.clone());
        Ok(function)
    }

    fn store_function(
        &self,
        cache_key: &str,
        source: &str,
        entry: &str,
        module: Arc<CudaModule>,
        function: CudaFunction,
    ) {
        let mut modules = self.modules.lock();
        let slot = modules.entry(cache_key.to_owned()).or_default();
        match slot.iter_mut().find(|cached| cached.source == source) {
            Some(cached) => {
                cached.functions.insert(entry.to_owned(), function);
            }
            None => slot.push(CachedCudaModule {
                source: source.to_owned(),
                module,
                functions: BTreeMap::from([(entry.to_owned(), function)]),
            }),
        }
    }

    /// Loads a precompiled PTX module and returns one of its entry points.
    ///
    /// This is the deployment path for hosts that have the CUDA driver but do
    /// not ship NVRTC.  Graph kernels may use it once their PTX artifact has
    /// been produced by the build/release toolchain; the runtime does not
    /// need a CUDA toolkit in that configuration.
    pub(crate) fn precompiled_function(
        &self,
        module_key: &str,
        entry: &str,
        ptx_source: &str,
    ) -> Result<CudaFunction, CudaDriverError> {
        let cache_key = format!("ptx:{module_key}");
        self.cached_function(&cache_key, entry, ptx_source, || {
            let module = self.load_module(CudarcRuntime::image_from_source(ptx_source))?;
            self.compile_counters
                .compilations
                .fetch_add(1, Ordering::Relaxed);
            Ok(module)
        })
    }

}

impl<R: GpuRuntime> CudaDriver<R>
where CudaDriverError: From<R::Error> + From<R::BuildError>
{
    /// Resolves a dense/reduction graph into a persistent CUDA plan.
    pub fn plan(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        numeric_policy: NumericPolicy,
        named: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<DevicePlan<R>, CudaDriverError> {
        let policy = PlanPolicy {
            activation_storage: None,
            numeric_policy,
            exact_activations: true,
            config: None,
        };
        self.plan_with_options(program, symbols, outputs, named, policy)
    }

    /// [`Self::plan`] storing activations in the block format a descriptor names through
    /// `ModelDescriptor::activation_storage`.
    ///
    /// # Errors
    /// [`CudaDriverError::ActivationStorage`] when the producer or the readers lack a render piece
    /// for the format; the refusal names each piece and the file:line that found it missing.
    pub fn plan_with_activation_storage(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        numeric_policy: NumericPolicy,
        activation_storage: Option<NumericFormat>,
        named: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<DevicePlan<R>, CudaDriverError> {
        let policy = PlanPolicy {
            activation_storage,
            numeric_policy,
            exact_activations: true,
            config: None,
        };
        self.plan_with_options(program, symbols, outputs, named, policy)
    }

    /// Retains the configured activation policy on the reusable CUDA plan.
    pub fn plan_with_activation_policy(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        numeric_policy: NumericPolicy,
        exact_activations: bool,
        named: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<DevicePlan<R>, CudaDriverError> {
        let policy = PlanPolicy {
            activation_storage: None,
            numeric_policy,
            exact_activations,
            config: None,
        };
        self.plan_with_options(program, symbols, outputs, named, policy)
    }

    pub fn plan_with_config(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        named: &[(&str, QuantizedBlock<'_>)],
        config: &RuntimeConfig,
    ) -> Result<DevicePlan<R>, CudaDriverError> {
        let policy = PlanPolicy {
            activation_storage: None,
            numeric_policy: config.numeric,
            exact_activations: config.exact_activations,
            config: Some(config),
        };
        self.plan_with_options(program, symbols, outputs, named, policy)
    }

    fn plan_with_options(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        named: &[(&str, QuantizedBlock<'_>)],
        policy: PlanPolicy<'_>,
    ) -> Result<DevicePlan<R>, CudaDriverError> {
        let PlanPolicy {
            activation_storage,
            numeric_policy,
            exact_activations,
            config,
        } = policy;
        validate_storage(activation_storage)?;
        let plan_started = Instant::now();
        let blocks = resolve_named_blocks(program, named).map_err(|_| {
            CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "named input resolution failed",
            }
        })?;
        let block_nodes = block_node_ids(program);
        let packed_operands: PackedOperands = block_nodes
            .iter()
            .zip(blocks.iter())
            .filter_map(|(node, block)| packed_codec(block).map(|codec| (*node, codec)))
            .collect();
        let shapes = infer(program, symbols).map_err(|_| CudaDriverError::UnsupportedGraph {
            node: NodeId(0),
            reason: "shape inference failed",
        })?;
        let root = program
            .len()
            .checked_sub(1)
            .map(|last| NodeId(last as u32))
            .ok_or(CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "empty program",
            })?;
        let outputs: Vec<NodeId> = if outputs.is_empty() {
            vec![root]
        } else {
            outputs.to_vec()
        };
        let outputs = outputs.as_slice();
        if config.is_some() {
            for (node, block) in block_nodes.iter().zip(&blocks) {
                let expected = shapes.of(*node).iter().try_fold(1_usize, |count, extent| {
                    count
                        .checked_mul(
                            usize::try_from(*extent)
                                .map_err(|_| CudaDriverError::ResourceOverflow)?,
                        )
                        .ok_or(CudaDriverError::ResourceOverflow)
                })?;
                if cuda_input_contract(block)?.2 != expected {
                    return Err(CudaDriverError::InputContractChanged {
                        node: *node,
                        reason: "element count does not match the planned shape",
                    });
                }
            }
        }
        let unfused = config.is_some_and(|value| !value.allow_fusion);
        let resolve_us = elapsed_micros(plan_started);
        let bind_started = Instant::now();
        let (resolved, packed_operands, mut micros) = bind_resolved(
            program,
            &shapes,
            outputs,
            &packed_operands,
            numeric_policy,
            unfused,
            activation_storage,
            &self.runtime.executes(),
        )?;
        let operand_dtypes = program_operand_dtypes(
            program,
            &resolved,
            &packed_operands,
            numeric_policy,
            &self.runtime.executes(),
        );
        let bind_us = elapsed_micros(bind_started);
        let emit_started = Instant::now();
        let (kernels, uniforms) = emit_and_pack(
            self.runtime.emit_target(),
            &resolved,
            None,
            &packed_operands,
            numeric_policy,
            self.runtime.emit_device(),
            &operand_dtypes,
            &mut micros,
        )?;
        let emit_us = elapsed_micros(emit_started);
        let assemble_started = Instant::now();
        let pool = Box::new(PoolSchedule::build(&resolved, &kernels, outputs));
        let mut plan = DevicePlan {
            pool,
            constraints: config.map(|value| value.constraints.clone()),
            block_contract: blocks
                .iter()
                .map(cuda_input_contract)
                .collect::<Result<Vec<_>, _>>()?,
            driver: self.clone(),
            program: program.to_vec(),
            shapes,
            resolved,
            functions: vec![None; kernels.len()],
            kernels,
            uniforms,
            block_nodes,
            outputs: outputs.into(),
            arena: Box::new(self.new_arena()),
            numeric_policy,
            activation_storage,
            packed_operands,
            exact_activations,
            resident_nodes: Box::default(),
            refit_calls: 0,
            patch_memo: None,
            bound_rows: symbols.first().copied().unwrap_or(0),
            rows_template: Box::default(),
            shapes_template: Box::default(),
            emit_calls: 0,
            emit_reused: 0,
            pack_calls: 0,
            repack_calls: 0,
            carry: Box::default(),
        };
        if config.is_some() {
            plan.validate_resources(&blocks, None)?;
            for kernel in &plan.kernels {
                self.runtime.build_function(kernel)?;
            }
        } else {
            plan.learn_rows_at_build(symbols, numeric_policy);
        }
        proxima_telemetry::debug!(
            resolve_us = resolve_us,
            bind_us = bind_us,
            emit_us = emit_us,
            assemble_us = elapsed_micros(assemble_started),
            total_us = elapsed_micros(plan_started),
            node_count = program.len(),
            kernel_count = plan.kernels.len(),
            "cuda plan built"
        );
        Ok(plan)
    }

    /// Uploads or refreshes one named f32 input, retaining its allocation for
    /// later dispatches when the element count is unchanged.
    pub fn upload_f32(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        values: &[f32],
    ) -> Result<(), CudaDriverError> {
        crate::backend::record_block_copied(core::mem::size_of_val(values) as u64);
        if let Some(CudaGraphBuffer::F32(buffer)) = arena.buffers.get_mut(&node)
            && R::len(buffer) == values.len()
        {
            self.runtime.upload_into(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let buffer = self.runtime.upload(values)?;
        self.graph_uploads.fetch_add(1, Ordering::Relaxed);
        arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        arena.allocations += 1;
        Ok(())
    }

    fn upload_bytes(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        values: &[u8],
    ) -> Result<(), CudaDriverError> {
        self.packed_upload_bytes
            .fetch_add(values.len() as u64, Ordering::Relaxed);
        crate::backend::record_block_copied(values.len() as u64);
        // a buffer shared with the resident cache is never overwritten in place
        if let Some(CudaGraphBuffer::Bytes(shared)) = arena.buffers.get_mut(&node)
            && R::len(shared.as_ref()) == values.len()
            && let Some(buffer) = Arc::get_mut(shared)
        {
            self.runtime.upload_into(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let buffer = self.runtime.upload(values)?;
        self.graph_uploads.fetch_add(1, Ordering::Relaxed);
        arena
            .buffers
            .insert(node, CudaGraphBuffer::Bytes(Arc::new(buffer)));
        arena.allocations += 1;
        Ok(())
    }

    /// Uploads a strided expert stack as the contiguous `[expert, ...]` buffer
    /// the kernels expect: one host-to-device copy per expert run into its
    /// slot of a fresh device allocation, so no host-side gather buffer exists.
    fn upload_runs(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        block: &QuantizedBlock<'_>,
    ) -> Result<(), CudaDriverError> {
        let (Some(runs), Some(total)) = (block.packed_runs(), block.packed_len()) else {
            return Err(CudaDriverError::UnsupportedGraph {
                node,
                reason: "block has no packed runs",
            });
        };
        self.packed_upload_bytes
            .fetch_add(total as u64, Ordering::Relaxed);
        crate::backend::record_block_copied(total as u64);
        let mut buffer = self.runtime.alloc_zeros::<u8>(total)?;
        let mut offset = 0;
        for run in runs {
            self.runtime.upload_at(run, &mut buffer, offset)?;
            offset += run.len();
        }
        self.graph_uploads.fetch_add(1, Ordering::Relaxed);
        arena
            .buffers
            .insert(node, CudaGraphBuffer::Bytes(Arc::new(buffer)));
        arena.allocations += 1;
        Ok(())
    }

    /// Uploads one node's launch metadata while retaining its device
    /// allocation across evaluations.  Uniforms are small, but allocating a
    /// fresh device buffer for every node on every token turns the persistent
    /// graph into an allocation-heavy path and prevents safe graph capture.
    fn upload_uniform(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        values: &[u8],
    ) -> Result<(), CudaDriverError> {
        if let Some(buffer) = arena.uniforms.get_mut(&node)
            && R::len(buffer) == values.len()
        {
            if arena.uniform_host.get(&node).is_some_and(|held| held == values) {
                return Ok(());
            }
            self.runtime.upload_into(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            arena.launch_profile.uniform_uploads += 1;
            arena.uniform_host.insert(node, values.to_vec());
            return Ok(());
        }
        let buffer = self.runtime.upload(values)?;
        self.graph_uploads.fetch_add(1, Ordering::Relaxed);
        arena.uniform_host.insert(node, values.to_vec());
        arena.uniforms.insert(node, buffer);
        arena.allocations += 1;
        Ok(())
    }

    /// Integer inputs reach the kernels as f32 -- the emitter maps every
    /// integer dtype to `float` -- so both input kinds share this upload.
    fn upload_float_input(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        values: &[f32],
    ) -> Result<(), CudaDriverError> {
        if values.is_empty() {
            arena.allocations += 1;
            arena.buffers.insert(
                node,
                CudaGraphBuffer::F32(self.runtime.alloc_zeros::<f32>(0)?),
            );
            return Ok(());
        }
        self.upload_f32(arena, node, values)
    }

    /// Materializes a plan-constant directly into the persistent arena. A
    /// constant has no data dependency and never needs a CUDA launch; keeping
    /// its device buffer alive also means repeated token evaluations do not
    /// re-upload it.
    fn materialize_constant(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        value: f32,
        output_len: usize,
    ) -> Result<(), CudaDriverError> {
        if matches!(
            arena.buffers.get(&node),
            Some(CudaGraphBuffer::F32(buffer)) if R::len(buffer) == output_len
        ) {
            return Ok(());
        }
        let buffer = if output_len == 0 {
            self.runtime.alloc_zeros::<f32>(0)?
        } else {
            let values = vec![value; output_len];
            let buffer = self.runtime.upload(values.as_slice())?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            buffer
        };
        arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        arena.allocations += 1;
        Ok(())
    }

    /// Launches a minimal device computation through the complete runtime
    /// path and returns its result. This is a smoke oracle for the future
    /// graph argument packer, not a model inference shortcut.
    pub fn smoke_add_one(&self, input: &[f32]) -> Result<Vec<f32>, CudaDriverError> {
        let n =
            i32::try_from(input.len()).map_err(|_| CudaDriverError::InputTooLarge(input.len()))?;
        let kernel = add_one_kernel(NodeId(0), NodeId(1), input.len())?;
        self.launch_f32(&kernel, &[input], &n.to_ne_bytes())
    }

    /// Runs two add-one dispatches through one persistent arena. The returned
    /// allocation count is an observable reuse oracle: five allocations are
    /// expected (input, two distinct output nodes, and one retained uniform
    /// buffer per output node), not one new allocation per dispatch plus a
    /// host round trip.
    pub fn smoke_add_one_persistent(
        &self,
        input: &[f32],
    ) -> Result<(Vec<f32>, u64), CudaDriverError> {
        let n =
            i32::try_from(input.len()).map_err(|_| CudaDriverError::InputTooLarge(input.len()))?;
        let mut arena = self.new_arena();
        arena.buffers.insert(
            NodeId(0),
            CudaGraphBuffer::F32(self.runtime.upload(input)?),
        );
        arena.allocations = 1;
        let first = add_one_kernel(NodeId(0), NodeId(1), input.len())?;
        self.launch_f32_persistent(&mut arena, &first, &n.to_ne_bytes())?;
        let second = add_one_kernel(NodeId(1), NodeId(2), input.len())?;
        self.launch_f32_persistent(&mut arena, &second, &n.to_ne_bytes())?;
        let output = self.read_f32(&arena, NodeId(2))?;
        Ok((output, arena.allocations()))
    }

    /// Launches an emitted f32 kernel whose bindings contain only input
    /// buffers, one output buffer, and a uniform byte buffer. Binding order
    /// comes from [`CudaKernel::bindings`], never from source-text parsing.
    pub fn launch_f32(
        &self,
        kernel: &CudaKernel,
        inputs: &[&[f32]],
        uniforms: &[u8],
    ) -> Result<Vec<f32>, CudaDriverError> {
        if kernel.grid.is_idle() {
            return Err(CudaDriverError::IdleLaunch);
        }
        let input_count = kernel
            .bindings
            .iter()
            .take_while(|binding| matches!(binding, Binding::Input(_)))
            .count();
        if input_count != inputs.len()
            || !matches!(kernel.bindings.get(input_count), Some(Binding::Output(_)))
            || !matches!(
                kernel.bindings.get(input_count + 1),
                Some(Binding::Uniforms)
            )
        {
            return Err(CudaDriverError::BindingContract);
        }
        let output_len = usize::try_from(kernel.grid.threads())
            .map_err(|_| CudaDriverError::InputTooLarge(usize::MAX))?;
        let function = self.runtime.build_function(kernel)?;
        let device_inputs = inputs
            .iter()
            .map(|input| self.runtime.upload(input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut device_output = self.runtime.alloc_zeros::<f32>(output_len)?;
        let device_uniforms = self.runtime.upload(uniforms)?;
        let config = LaunchShape {
            grid_dim: (launch_blocks(&kernel.grid)?, 1, 1),
            block_dim: (kernel.grid.block_width_or_default() as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = self.runtime.launch_arguments(&function);
        for input in &device_inputs {
            args.push(input);
        }
        args.push_mut(&mut device_output);
        args.push(&device_uniforms);
        unsafe { self.runtime.launch(args, config) }?;
        self.runtime.synchronize()?;
        Ok(self.runtime.download(&device_output)?)
    }

    /// Dispatches a kernel against persistent arena buffers. Inputs and the
    /// output are selected from `CudaKernel::bindings`; no host tensor copy
    /// occurs for an intermediate node already resident in the arena.
    pub fn launch_f32_persistent(
        &self,
        arena: &mut CudaF32Arena<R>,
        kernel: &CudaKernel,
        uniforms: &[u8],
    ) -> Result<(), CudaDriverError> {
        if kernel.grid.is_idle() {
            return Err(CudaDriverError::IdleLaunch);
        }
        self.launch_f32_persistent_sized(arena, kernel, uniforms, kernel.grid.threads() as usize)
    }

    /// Persistent launch with an output allocation sized independently from
    /// the dispatch thread count (reductions dispatch over output*reduction
    /// elements but materialize only the output shape).
    pub fn launch_f32_persistent_sized(
        &self,
        arena: &mut CudaF32Arena<R>,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
    ) -> Result<(), CudaDriverError> {
        self.launch_f32_persistent_sized_with_sync(
            arena, kernel, uniforms, output_len, Completion::Synchronize, None,
        )
    }

    /// Enqueues a kernel on the driver's stream without synchronizing.
    ///
    /// CUDA stream order already preserves the graph's dependency order. The
    /// graph plan uses this form for ordinary kernels and performs one stream
    /// synchronization after the whole evaluation, avoiding a host/device
    /// round trip after every node. A kernel with a fault binding must be
    /// given `deferred_fault`: an `(array, offset, slots)` window of a device fault array that the
    /// caller reads back once after the final synchronization. Without it the
    /// diagnostic would never be observed, so the launch is rejected.
    pub fn launch_f32_persistent_sized_async(
        &self,
        arena: &mut CudaF32Arena<R>,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        deferred_fault: Option<(&R::Buffer<u32>, usize, usize)>,
        function: Option<&R::Function>,
    ) -> Result<(), CudaDriverError> {
        let faults = kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Fault));
        if faults && deferred_fault.is_none() {
            return Err(CudaDriverError::BindingContract);
        }
        self.launch_f32_persistent_sized_with_sync(
            arena,
            kernel,
            uniforms,
            output_len,
            Completion::Defer(deferred_fault),
            function,
        )
    }

    /// Launches each of `kernel`'s prologue kernels on the stream ahead of it, sharing its
    /// uniforms. A prologue's output lives in the arena under its own node and is sized by the
    /// kernel's extra output of that node, so it is allocated once and read by the launch after.
    fn launch_prologue(
        &self,
        arena: &mut CudaF32Arena<R>,
        kernel: &CudaKernel,
        uniforms: &[u8],
    ) -> Result<(), CudaDriverError> {
        if kernel.route_compaction_words > 0 {
            return Ok(());
        }
        for prologue in &kernel.prologue {
            let Some(Binding::Output(node)) = prologue
                .bindings
                .iter()
                .find(|binding| matches!(binding, Binding::Output(_)))
            else {
                return Err(CudaDriverError::BindingContract);
            };
            let output_len = kernel
                .extra_outputs
                .iter()
                .find_map(|(extra, length)| (extra == node).then_some(*length))
                .ok_or(CudaDriverError::BindingContract)?;
            self.launch_f32_persistent_sized_with_sync(
                arena,
                prologue,
                uniforms,
                output_len,
                Completion::Defer(None),
                None,
            )?;
        }
        Ok(())
    }

    fn launch_f32_persistent_sized_with_sync(
        &self,
        arena: &mut CudaF32Arena<R>,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        completion: Completion<'_, R>,
        function: Option<&R::Function>,
    ) -> Result<(), CudaDriverError> {
        let (synchronize, deferred_fault) = match completion {
            Completion::Synchronize => (true, None),
            Completion::Defer(deferred_fault) => (false, deferred_fault),
        };
        self.launch_prologue(arena, kernel, uniforms)?;
        let Some((output_index, Binding::Output(output_node))) = kernel
            .bindings
            .iter()
            .enumerate()
            .find(|(_, binding)| matches!(binding, Binding::Output(_)))
        else {
            return Err(CudaDriverError::BindingContract);
        };
        let output_node = *output_node;
        let Some(uniform_index) = kernel
            .bindings
            .iter()
            .position(|binding| matches!(binding, Binding::Uniforms))
        else {
            return Err(CudaDriverError::BindingContract);
        };
        if uniform_index <= output_index {
            return Err(CudaDriverError::BindingContract);
        }
        let input_nodes = || {
            kernel.bindings.iter().filter_map(|binding| match binding {
                Binding::Input(node) | Binding::Indices(node) => Some(*node),
                Binding::Output(_)
                | Binding::Uniforms
                | Binding::Fault
                | Binding::ExpertPayloads(_)
                | Binding::ExpertDescriptors(_)
                | Binding::Scratch => None,
            })
        };
        if let Some(node) = input_nodes().find(|node| !arena.buffers.contains_key(node)) {
            return Err(CudaDriverError::MissingBuffer(node));
        }
        let fault_count = kernel
            .bindings
            .iter()
            .filter(|binding| matches!(binding, Binding::Fault))
            .count();
        if fault_count > 1 {
            return Err(CudaDriverError::BindingContract);
        }
        let mut lap = arena.launch_profile.enabled.then(Instant::now);
        let looked_up;
        let function = match function {
            Some(function) => function,
            None => {
                arena.launch_profile.function_lookups += 1;
                looked_up = self.runtime.build_function(kernel)?;
                &looked_up
            }
        };
        self.upload_uniform(arena, output_node, uniforms)?;
        arena.launch_profile.mark(0, &mut lap);
        let mut extra_buffers = Vec::with_capacity(kernel.extra_outputs.len());
        for (node, length) in &kernel.extra_outputs {
            match self.take_extra_output(arena, *node, *length) {
                Ok(buffer) => extra_buffers.push((*node, buffer)),
                Err(error) => {
                    Self::restore_extra_outputs(arena, extra_buffers);
                    return Err(error);
                }
            }
        }
        arena.launch_profile.mark(1, &mut lap);
        let device_uniforms = arena
            .uniforms
            .get(&output_node)
            .ok_or(CudaDriverError::BindingContract)?;
        let mut device_output = match arena.buffers.remove(&output_node) {
            Some(CudaGraphBuffer::F32(buffer)) => buffer,
            Some(CudaGraphBuffer::Bytes(_)) => return Err(CudaDriverError::BindingContract),
            None => {
                arena.allocations += 1;
                arena.launch_profile.output_allocs += 1;
                self.runtime.alloc_zeros::<f32>(output_len)?
            }
        };
        if R::len(&device_output) != output_len {
            let replacement = match self.runtime.alloc_zeros::<f32>(output_len) {
                Ok(buffer) => buffer,
                Err(error) => {
                    arena
                        .buffers
                        .insert(output_node, CudaGraphBuffer::F32(device_output));
                    return Err(error.into());
                }
            };
            device_output = replacement;
            arena.allocations += 1;
            arena.launch_profile.output_allocs += 1;
        }
        arena.launch_profile.mark(2, &mut lap);
        if input_nodes().any(|node| !arena.buffers.contains_key(&node)) {
            arena
                .buffers
                .insert(output_node, CudaGraphBuffer::F32(device_output));
            return Err(CudaDriverError::BindingContract);
        }
        let config = LaunchShape {
            grid_dim: (launch_blocks(&kernel.grid)?, 1, 1),
            block_dim: (kernel.grid.block_width_or_default() as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather_slots = kernel
            .bindings
            .iter()
            .filter(|binding| matches!(binding, Binding::Indices(_)))
            .count();
        let mut fault_buffer = (fault_count > 0 && deferred_fault.is_none())
            .then(|| self.runtime.alloc_zeros::<u32>(gather_slots))
            .transpose()?;
        let fault_window =
            deferred_fault.map(|(array, offset, slots)| R::view(array, offset..offset + slots));
        arena.launch_profile.mark(3, &mut lap);
        let mut route_buffer = self.take_route_compaction(&mut arena.route_compaction, output_node, kernel.route_compaction_words)?;
        let prepass_result = match route_buffer.as_mut() {
            Some(compaction) => {
                let owned_view = fault_buffer.as_ref().map(|buffer| R::view(buffer, 0..R::len(buffer)));
                // the view type is invariant in its borrow, so each source is passed on its own instead of joined through Option::or
                match (fault_window.as_ref(), owned_view.as_ref()) {
                    (Some(window), _) => self.launch_route_prepass(&arena.buffers, kernel, compaction, device_uniforms, Some(window)),
                    (None, Some(window)) => self.launch_route_prepass(&arena.buffers, kernel, compaction, device_uniforms, Some(window)),
                    (None, None) => self.launch_route_prepass(&arena.buffers, kernel, compaction, device_uniforms, None),
                }
            }
            None => Ok(()),
        };
        let mut args = self.runtime.launch_arguments(function);
        for binding in &kernel.bindings[..output_index] {
            match binding {
                Binding::Input(node) | Binding::Indices(node) => match arena.buffers.get(node) {
                    Some(buffer) => push_buffer::<R>(&mut args, buffer),
                    None => {
                        drop(args);
                        arena
                            .buffers
                            .insert(output_node, CudaGraphBuffer::F32(device_output));
                        return Err(CudaDriverError::MissingBuffer(*node));
                    }
                },
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        args.push_mut(&mut device_output);
        for binding in &kernel.bindings[output_index + 1..] {
            match binding {
                Binding::Uniforms => args.push(device_uniforms),
                Binding::Fault => match (&fault_window, &fault_buffer) {
                    (Some(window), _) => args.push_view(window),
                    (None, Some(buffer)) => args.push(buffer),
                    (None, None) => return Err(CudaDriverError::BindingContract),
                },
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        for (_, buffer) in &mut extra_buffers {
            args.push_mut(buffer);
        }
        if let Some(buffer) = route_buffer.as_mut() {
            args.push_mut(buffer);
        }
        arena.launch_profile.mark(4, &mut lap);
        let launch_result = match prepass_result {
            Ok(()) => unsafe { self.runtime.launch(args, config) }.map_err(CudaDriverError::from),
            Err(error) => {
                drop(args);
                Err(error)
            }
        };
        if let Some(buffer) = route_buffer.take() {
            arena.route_compaction.insert(output_node, buffer);
        }
        arena.launch_profile.mark(5, &mut lap);
        Self::restore_extra_outputs(arena, extra_buffers);
        if let Err(error) = launch_result {
            arena
                .buffers
                .insert(output_node, CudaGraphBuffer::F32(device_output));
            return Err(error);
        }
        self.graph_launches.fetch_add(1, Ordering::Relaxed);
        if synchronize {
            if let Err(error) = self.runtime.synchronize() {
                arena
                    .buffers
                    .insert(output_node, CudaGraphBuffer::F32(device_output));
                return Err(error.into());
            }
            if let Some(fault_buffer) = fault_buffer.take() {
                let faults = self.runtime.download(&fault_buffer)?;
                if let Some(slot) = faults.iter().position(|value| *value != 0) {
                    let index = i32::from_ne_bytes(faults[slot].to_ne_bytes()) as i64;
                    arena
                        .buffers
                        .insert(output_node, CudaGraphBuffer::F32(device_output));
                    return Err(CudaDriverError::GatherIndexOutOfRange {
                        node: output_node,
                        slot,
                        index,
                    });
                }
            }
        }
        arena
            .buffers
            .insert(output_node, CudaGraphBuffer::F32(device_output));
        Ok(())
    }

    fn take_route_compaction(
        &self,
        held: &mut BTreeMap<NodeId, R::Buffer<u32>>,
        node: NodeId,
        words: usize,
    ) -> Result<Option<R::Buffer<u32>>, CudaDriverError> {
        if words == 0 {
            return Ok(None);
        }
        match held.remove(&node) {
            Some(buffer) if R::len(&buffer) == words => Ok(Some(buffer)),
            Some(_) | None => Ok(Some(self.runtime.alloc_zeros::<u32>(words)?)),
        }
    }

    /// Launches the grouped kernel's route prepass, its one prologue kernel, on the stream ahead
    /// of the kernel: it binds the route's index buffer, the shared uniforms, the fault window and
    /// the compaction buffer the grouped kernel reads next.
    fn launch_route_prepass(
        &self,
        buffers: &BTreeMap<NodeId, CudaGraphBuffer<R>>,
        kernel: &CudaKernel,
        compaction: &mut R::Buffer<u32>,
        uniforms: &R::Buffer<u8>,
        fault: Option<&R::View<'_, u32>>,
    ) -> Result<(), CudaDriverError> {
        let [prepass] = kernel.prologue.as_slice() else {
            return Err(CudaDriverError::BindingContract);
        };
        let function = self.runtime.build_function(prepass)?;
        let config = LaunchShape {
            grid_dim: (launch_blocks(&prepass.grid)?, 1, 1),
            block_dim: (prepass.grid.block_width_or_default() as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = self.runtime.launch_arguments(&function);
        for binding in &prepass.bindings {
            match binding {
                Binding::Indices(node) => match buffers.get(node) {
                    Some(buffer) => push_buffer::<R>(&mut args, buffer),
                    None => return Err(CudaDriverError::MissingBuffer(*node)),
                },
                Binding::Uniforms => args.push(uniforms),
                Binding::Fault => args.push_view(fault.ok_or(CudaDriverError::BindingContract)?),
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        args.push_mut(compaction);
        unsafe { self.runtime.launch(args, config) }?;
        self.graph_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn take_extra_output(
        &self,
        arena: &mut CudaF32Arena<R>,
        node: NodeId,
        length: usize,
    ) -> Result<R::Buffer<f32>, CudaDriverError> {
        match arena.buffers.remove(&node) {
            Some(CudaGraphBuffer::F32(buffer)) if R::len(&buffer) == length => Ok(buffer),
            Some(CudaGraphBuffer::Bytes(_)) => Err(CudaDriverError::BindingContract),
            Some(CudaGraphBuffer::F32(_)) | None => {
                arena.allocations += 1;
                arena.launch_profile.extra_allocs += 1;
                Ok(self.runtime.alloc_zeros::<f32>(length)?)
            }
        }
    }

    fn restore_extra_outputs(arena: &mut CudaF32Arena<R>, buffers: Vec<(NodeId, R::Buffer<f32>)>) {
        for (node, buffer) in buffers {
            arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        }
    }

    /// Copies one persistent node back to the host for an explicit oracle or
    /// final-result readback.
    pub fn read_f32(
        &self,
        arena: &CudaF32Arena<R>,
        node: NodeId,
    ) -> Result<Vec<f32>, CudaDriverError> {
        let buffer = arena
            .buffers
            .get(&node)
            .ok_or(CudaDriverError::MissingBuffer(node))?;
        let CudaGraphBuffer::F32(buffer) = buffer else {
            return Err(CudaDriverError::BindingContract);
        };
        Ok(self.runtime.download(buffer)?)
    }
}

/// Kernels of the previous binding, moved into the next one wherever the bound op is unchanged.
struct PriorKernels<'plan> {
    bound: &'plan [BoundOp],
    kernels: Vec<CudaKernel>,
}

#[derive(Debug, Default, Clone, Copy)]
struct LowerMicros {
    reused: usize,
    bind: u64,
    emit: u64,
    pack: u64,
}

/// The bound ops and packed-operand map a CUDA plan for `program` at `symbols` would
/// bind, by binding alone: no kernel is emitted and no device is touched. The packed map
/// is built from the weight blocks in `named` (a missing input binds unpacked), then
/// `bind_resolved` runs its steps: bind, prune dead ops, correct packed matmul layouts,
/// and derive the activation quantize nodes (`activation_storage` `None` is the default
/// q8_1 pair), so the returned map also holds each derived node's codec.
///
/// # Errors
/// Shape inference or binding fails at `symbols`, or the bound graph is not servable.
pub fn probe_bound_graph(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    named: &[(&str, QuantizedBlock<'_>)],
    numeric_policy: NumericPolicy,
    activation_storage: Option<NumericFormat>,
) -> Result<(Vec<BoundOp>, PackedOperands), CudaDriverError> {
    let packed_operands: PackedOperands = program
        .iter()
        .enumerate()
        .filter_map(|(index, op)| {
            let name = op.name().filter(|_| matches!(op, Op::Input { .. }))?;
            let (_, block) = named.iter().find(|(candidate, _)| *candidate == name)?;
            packed_codec(block).map(|codec| (NodeId(index as u32), codec))
        })
        .collect();
    let shapes = infer(program, symbols).map_err(|_| CudaDriverError::UnsupportedGraph {
        node: NodeId(0),
        reason: "shape inference failed",
    })?;
    let (resolved, packed, _) =
        bind_resolved(program, &shapes, outputs, &packed_operands, numeric_policy, false, activation_storage, &plan_grant())?;
    Ok((resolved, packed))
}

#[allow(clippy::too_many_arguments)] // one context per argument; a bundle would be a type that only relocates them
fn bind_resolved(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    unfused: bool,
    activation_storage: Option<NumericFormat>,
    grant: &Executes,
) -> Result<(Vec<BoundOp>, PackedOperands, LowerMicros), CudaDriverError> {
    let bind_started = Instant::now();
    let config = BindConfig::new(numeric_policy, !unfused, packed_operands).with_activation_storage(activation_storage);
    let bound = bind_plan(program, shapes, outputs, &config, grant).map_err(|_| CudaDriverError::UnsupportedGraph {
        node: NodeId(0),
        reason: "graph binding failed",
    })?;
    let (resolved, packed, numeric_policy) = (bound.resolved, bound.packed, bound.policy);
    validate_resolved(&resolved, numeric_policy, grant)?;
    let micros = LowerMicros {
        bind: bind_started.elapsed().as_micros() as u64,
        ..LowerMicros::default()
    };
    Ok((resolved, packed, micros))
}

fn validate_resolved(
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
    grant: &Executes,
) -> Result<(), CudaDriverError> {
    for bound in resolved {
        validate_cuda_dtype(bound)?;
    }
    reject_unsupported_kinds(resolved, numeric_policy, grant)
}

#[allow(clippy::too_many_arguments)] // one context per argument; a bundle would be a type that only relocates them
fn emit_and_pack(
    target: Target,
    resolved: &[BoundOp],
    prior: Option<PriorKernels<'_>>,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
    operand_dtypes: &OperandDtypes,
    micros: &mut LowerMicros,
) -> Result<(Vec<CudaKernel>, Vec<Vec<u8>>), CudaDriverError> {
    let emit_started = Instant::now();
    let (prior_bound, prior_kernels) =
        prior.map_or((&[][..], Vec::new()), |prior| (prior.bound, prior.kernels));
    let (kernels, reused) = crate::cuda::emit_reusing_on_target(
        target,
        prior_bound,
        prior_kernels,
        resolved,
        packed_operands,
        numeric_policy,
        device,
        operand_dtypes,
    )
    .map_err(|(node, error)| CudaDriverError::Emit {
        node,
        error: error.to_string(),
    })?;
    micros.emit = emit_started.elapsed().as_micros() as u64;
    micros.reused = reused;
    let pack_started = Instant::now();
    let uniforms = resolved
        .iter()
        .map(|bound| {
            crate::cuda::pack_cuda_uniforms(bound).map_err(|error| CudaDriverError::Emit {
                node: bound.node,
                error: error.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    micros.pack = pack_started.elapsed().as_micros() as u64;
    Ok((kernels, uniforms))
}

/// A runtime that cannot record events refuses with an error; the executor then skips timing
/// rather than failing the execute ([`GpuRuntime::record_event`]).
fn try_record_event<R: GpuRuntime>(runtime: &R) -> Option<R::Event> {
    runtime.record_event().ok()
}

fn report_event_timing<R: GpuRuntime>(
    origin: &R::Event,
    rows: &[(NodeId, String, usize, R::Event)],
) -> Result<(), CudaDriverError>
where
    CudaDriverError: From<R::Error>,
{
    let mut previous = origin;
    let mut by_kernel: BTreeMap<&str, (f64, usize, usize)> = BTreeMap::new();
    let mut total_ms = 0.0f64;
    for (_, label, output_len, event) in rows {
        let millis = f64::from(R::elapsed_ms(previous, event)?);
        let entry = by_kernel
            .entry(label.as_str())
            .or_insert((0.0, 0, *output_len));
        entry.0 += millis;
        entry.1 += 1;
        total_ms += millis;
        previous = event;
    }
    let mut ranked: Vec<_> = by_kernel.into_iter().collect();
    ranked.sort_by(|left, right| right.1.0.total_cmp(&left.1.0));
    eprintln!("cuda_event_timing: nodes={} gpu_ms={total_ms:.3}", rows.len());
    for (label, (millis, count, output_len)) in ranked.iter().take(24) {
        eprintln!(
            "cuda_event_timing_kernel gpu_ms={millis:.3} count={count} each_us={:.1} output_len={output_len} {label}",
            millis * 1000.0 / *count as f64
        );
    }
    Ok(())
}

fn refresh_refusal_name(refusal: RefreshRefusal) -> &'static str {
    match refusal {
        RefreshRefusal::PositionOutOfRange { .. } => "position out of range",
        RefreshRefusal::MatcherFusedKind { .. } => "matcher fused kind",
        RefreshRefusal::EpilogueFusionNotLocal { .. } => "epilogue fusion not local",
        RefreshRefusal::OpKindMismatch { .. } => "op kind mismatch",
        RefreshRefusal::RecomposedBodyMismatch { .. } => "recomposed body mismatch",
        RefreshRefusal::RecomposedOperandsDiffer { .. } => "recomposed operands differ",
        RefreshRefusal::ShapeError { .. } => "shape error",
    }
}

/// Re-derives only the KV-touched ops from `program` at `shapes`, keeping every
/// other retained op as-is. Returns the refreshed set and the touched count.
#[allow(clippy::too_many_arguments)] // one context per argument; a bundle would be a type that only relocates them
fn refresh_bound_resolved(
    (previous, previous_shapes): (&[BoundOp], &Shapes),
    program: &[Op],
    outputs: &[NodeId],
    shapes: &Shapes,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    activation_storage: Option<NumericFormat>,
    grant: &Executes,
) -> Result<(Vec<BoundOp>, PackedOperands, usize), RefreshRefusal> {
    let changed: BTreeSet<NodeId> = (0..program.len())
        .map(|position| NodeId(position as u32))
        .filter(|node| previous_shapes.of(*node) != shapes.of(*node))
        .collect();
    let base = underive_block_quantize(previous, program.len());
    let previous = base.as_slice();
    let masks: Vec<u64> = previous
        .iter()
        .map(|bound| u64::from(touches_changed(bound, program, &changed)))
        .collect();
    let touched = masks.iter().filter(|mask| **mask != 0).count();
    let refreshed = refresh_bound_ops(
        previous,
        program,
        outputs,
        shapes,
        &masks,
        1,
        numeric_policy,
    )?;
    let config = BindConfig::new(numeric_policy, true, packed_operands).with_activation_storage(activation_storage);
    let finished = finish_bound(refreshed, program, shapes, &config, grant);
    Ok((finished.resolved, finished.packed, touched))
}

fn cuda_admits(executes: &Executes, bound: &BoundOp, numeric_policy: NumericPolicy) -> bool {
    if executes.admit(bound).is_err() {
        return false;
    }
    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => {
            admit(numeric_policy, NumericRewrite::ContextChunkMerge).is_ok()
                && cuda_cached_attention_class(&bound.kind).is_some()
        }
        _ => true,
    }
}

/// Rejects the whole graph at once, naming every unsupported kind with its
/// node count, so a model's full gap list is one error rather than one node at a time.
fn reject_unsupported_kinds(
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
    executes: &Executes,
) -> Result<(), CudaDriverError> {
    let mut census: BTreeMap<&'static str, (usize, NodeId)> = BTreeMap::new();
    for bound in resolved
        .iter()
        .filter(|bound| !cuda_admits(executes, bound, numeric_policy))
    {
        census.entry(bound.kind.name()).or_insert((0, bound.node)).0 += 1;
    }
    let Some((_, (_, first))) = census.iter().next() else {
        return Ok(());
    };
    let first = *first;
    let kinds = census
        .iter()
        .map(|(name, (count, node))| format!("{name} x{count} (first {node:?})"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(CudaDriverError::UnsupportedOpKinds { node: first, kinds })
}

fn validate_cuda_dtype(bound: &BoundOp) -> Result<(), CudaDriverError> {
    // integer dtypes are carried as integer-valued f32 on device, exactly as the cpu interpreter's f32 pipeline holds them
    if bound.dtype == DType::FLOAT32 || cuda_integer_dtype(bound.dtype) {
        return Ok(());
    }
    Err(CudaDriverError::UnsupportedDtype {
        node: bound.node,
        dtype: bound.dtype,
    })
}

fn cuda_integer_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Bool | DType::INT8 | DType::UINT8 | DType::INT32 | DType::UINT32
    )
}

const CUDA_REQUIRED_HEADER: &str = "cuda_fp16.h";
const CUDA_SYSTEM_INCLUDE_DIRS: [&str; 2] = ["/usr/local/cuda/include", "/opt/cuda/include"];

fn cuda_include_candidates(system_dirs: &[&str]) -> Vec<std::path::PathBuf> {
    let explicit = std::env::var_os("PROXIMA_CUDA_INCLUDE_PATH").map(std::path::PathBuf::from);
    let rooted = ["CUDA_PATH", "CUDA_HOME"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(|root| std::path::PathBuf::from(root).join("include"));
    let system = system_dirs.iter().map(std::path::PathBuf::from);
    explicit.into_iter().chain(rooted).chain(system).collect()
}

fn resolve_cuda_include_dir(
    header: &str,
    system_dirs: &[&str],
) -> Result<std::path::PathBuf, CudaDriverError> {
    let candidates = cuda_include_candidates(system_dirs);
    match candidates.iter().find(|dir| dir.join(header).is_file()) {
        Some(dir) => Ok(dir.clone()),
        None => Err(CudaDriverError::CudaHeaderNotFound {
            header: header.to_owned(),
            tried: candidates
                .iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
}

fn require_nvrtc() -> Result<(), CudaDriverError> {
    if unsafe { nvrtc_sys::is_culib_present() } {
        return Ok(());
    }
    Err(CudaDriverError::Nvrtc("nvrtc library unavailable".into()))
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// NVRTC does not ship `cuda_fp16.h`, so the emitted source needs a toolkit
/// include directory.  Resolution order: `PROXIMA_CUDA_INCLUDE_PATH`,
/// `CUDA_PATH/include`, `CUDA_HOME/include`, `/usr/local/cuda/include`,
/// `/opt/cuda/include`; the first one that contains the header wins.
fn cuda_header_dir(source: &str) -> Result<Option<PathBuf>, CudaDriverError> {
    if !source.contains(CUDA_REQUIRED_HEADER) {
        return Ok(None);
    }
    resolve_cuda_include_dir(CUDA_REQUIRED_HEADER, &CUDA_SYSTEM_INCLUDE_DIRS).map(Some)
}

fn cuda_header_bytes(header_dir: Option<&Path>) -> Result<Option<Vec<u8>>, CudaDriverError> {
    let Some(dir) = header_dir else {
        return Ok(None);
    };
    let path = dir.join(CUDA_REQUIRED_HEADER);
    std::fs::read(&path)
        .map(Some)
        .map_err(|error| CudaDriverError::CudaHeaderUnreadable {
            path: path.display().to_string(),
            error: error.to_string(),
        })
}

fn nvrtc_options(header_dir: Option<&Path>) -> CompileOptions {
    let mut options = CompileOptions::default();
    if let Some(dir) = header_dir {
        options
            .include_paths
            .push(dir.to_string_lossy().into_owned());
    }
    options
}

fn compile_cubin(
    source: &str,
    header_dir: Option<&Path>,
    arch: &str,
) -> Result<Vec<u8>, CudaDriverError> {
    let refused = |what: &str, error: &dyn std::fmt::Debug| CudaDriverError::Nvrtc(format!("{what}: {error:?}"));
    let text = CString::new(source).map_err(|error| refused("kernel source", &error))?;
    let program = nvrtc_result::create_program(&text, None).map_err(|error| refused("create", &error))?;
    let mut flags = vec![format!("--gpu-architecture={arch}")];
    if let Some(dir) = header_dir {
        flags.push(format!("--include-path={}", dir.to_string_lossy()));
    }
    let image = unsafe { nvrtc_result::compile_program(program, &flags) }
        .map_err(|error| refused("compile", &error))
        .and_then(|()| unsafe { read_cubin(program) });
    unsafe { nvrtc_result::destroy_program(program) }.map_err(|error| refused("destroy", &error))?;
    image
}

unsafe fn read_cubin(program: nvrtc_sys::nvrtcProgram) -> Result<Vec<u8>, CudaDriverError> {
    let mut size = 0usize;
    let sized = unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut size) };
    if sized != nvrtc_sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(CudaDriverError::Nvrtc(format!("cubin size: {sized:?}")));
    }
    let mut image = vec![0u8; size];
    let read = unsafe { nvrtc_sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) };
    if read != nvrtc_sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(CudaDriverError::Nvrtc(format!("cubin read: {read:?}")));
    }
    Ok(image)
}

static NVRTC_VERSION: OnceLock<Result<(i32, i32), String>> = OnceLock::new();

fn query_nvrtc_version() -> Result<(i32, i32), String> {
    let mut major = 0;
    let mut minor = 0;
    match unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) } {
        nvrtc_sys::nvrtcResult::NVRTC_SUCCESS => Ok((major, minor)),
        status => Err(format!("nvrtcVersion failed: {status:?}")),
    }
}

// the first nvrtc call dlopens the library (~190 ms); starting it at driver creation overlaps it with weight upload
fn prefetch_nvrtc_version() {
    thread::spawn(|| NVRTC_VERSION.get_or_init(query_nvrtc_version).is_ok());
}

fn nvrtc_version() -> Result<(i32, i32), CudaDriverError> {
    NVRTC_VERSION
        .get_or_init(query_nvrtc_version)
        .clone()
        .map_err(CudaDriverError::Nvrtc)
}

fn kernel_disk_store() -> PtxStore {
    KernelCacheConfig::from_env().map_or_else(
        |error| {
            proxima_telemetry::warn!(
                error = %error,
                "kernel cache config rejected; kernels compile in memory only"
            );
            PtxStore::Disabled
        },
        |config| PtxStore::from_config(&config),
    )
}

fn cached_ptx(store: &PtxStore, key: &CacheKey) -> Option<String> {
    let PtxStore::Directory(dir) = store else {
        return None;
    };
    let name = key.to_hex();
    let path = dir.join(&name);
    let entry = match store.get(&name) {
        Ok(Some(entry)) => entry,
        Ok(None) => return None,
        Err(error) => {
            proxima_telemetry::warn!(path = %path.display(), error = %error, "kernel cache read failed; compiling");
            return None;
        }
    };
    match decode_entry(key, &entry) {
        Ok(text) => Some(text.to_owned()),
        Err(error) => {
            proxima_telemetry::warn!(path = %path.display(), error = %error, "kernel cache entry rejected; recompiling");
            None
        }
    }
}

fn store_ptx(store: &PtxStore, key: &CacheKey, ptx: &Ptx) {
    let name = key.to_hex();
    if let Err(error) = store.put(&name, &encode_entry(key, &ptx.to_src())) {
        proxima_telemetry::warn!(file = %name, error = %error, "kernel cache write failed; kernel stays uncached");
    }
}

/// The kernel disk store's lookup, compile, store sequence for one code object under `key`: a stored
/// entry that verifies is returned and counted as a disk hit, otherwise `compile` runs and its bytes
/// are stored. The binary framing is the cubin path's ([`encode_binary_entry`]).
fn stored_or_compiled<Failure>(
    counters: &CompileCounters,
    key: &CacheKey,
    compile: impl FnOnce() -> Result<Vec<u8>, Failure>,
) -> Result<Vec<u8>, Failure> {
    let name = key.to_hex();
    let started = Instant::now();
    let stored = counters
        .disk_store
        .get(&name)
        .ok()
        .flatten()
        .and_then(|entry| decode_binary_entry(key, &entry).ok().map(<[u8]>::to_vec));
    CompileCounters::add(&counters.disk_read_micros, started);
    if let Some(image) = stored {
        counters.disk_cache_hits.fetch_add(1, Ordering::Relaxed);
        return Ok(image);
    }
    let image = compile()?;
    if let Err(error) = counters.disk_store.put(&name, &encode_binary_entry(key, &image)) {
        proxima_telemetry::warn!(file = %name, error = %error, "kernel cache write failed; kernel stays uncached");
    }
    Ok(image)
}

impl<R: GpuRuntime> DevicePlan<R>
where CudaDriverError: From<R::Error> + From<R::BuildError>
{
    #[must_use]
    pub const fn exact_activations(&self) -> bool {
        self.exact_activations
    }

    /// Moves this plan to new symbol values when its graph structure is unchanged,
    /// re-binding extents and uniforms while keeping the device arena, resident
    /// weights and compiled modules. `Ok(false)` when a resource-constrained plan
    /// must be rebuilt. Teaching pointer: the same bind-and-emit step
    /// `plan_with_options` runs, minus input resolution and arena creation.
    pub fn refit_symbols(
        &mut self,
        symbols: &[u64],
        numeric_policy: NumericPolicy,
    ) -> Result<bool, CudaDriverError> {
        if self.constraints.is_some() {
            return Ok(false);
        }
        let infer_started = Instant::now();
        let shapes = self.shapes_for(symbols)?;
        let infer_micros = infer_started.elapsed().as_micros() as u64;
        let same_policy = numeric_policy == self.numeric_policy;
        let rows = symbols.first().copied().unwrap_or(0);
        if same_policy && rows != self.bound_rows {
            let repack_started = Instant::now();
            if let Some(op_count) = self.repack_rows(rows, &shapes, numeric_policy)? {
                self.refit_calls += 1;
                self.emit_reused += op_count as u64;
                self.pack_calls += op_count as u64;
                self.repack_calls += 1;
                self.shapes = shapes;
                proxima_telemetry::debug!(
                    infer_us = infer_micros,
                    repack_us = elapsed_micros(repack_started),
                    op_count = op_count,
                    bind_path = "repack",
                    refit_total = self.refit_calls,
                    "cuda plan refit phases"
                );
                return Ok(true);
            }
        }
        if same_policy {
            let patch_started = Instant::now();
            if let Some(touched) = self.patch_cached_attention(&shapes, numeric_policy)? {
                let op_count = self.resolved.len();
                self.refit_calls += 1;
                self.emit_reused += op_count.saturating_sub(touched) as u64;
                self.emit_calls += touched as u64;
                self.pack_calls += touched as u64;
                self.shapes = shapes;
                proxima_telemetry::debug!(
                    infer_us = infer_micros,
                    patch_us = elapsed_micros(patch_started),
                    op_count = op_count,
                    bind_path = "patch",
                    bind_touched = touched as u64,
                    refit_total = self.refit_calls,
                    "cuda plan refit phases"
                );
                return Ok(true);
            }
        }
        let previous_bound = std::mem::take(&mut self.resolved);
        let refresh_started = Instant::now();
        self.patch_memo = None;
        let refresh = if !same_policy {
            Err("numeric policy changed")
        } else {
            refresh_bound_resolved(
                (&previous_bound, &self.shapes),
                &self.program,
                &self.outputs,
                &shapes,
                &self.packed_operands,
                numeric_policy,
                self.activation_storage,
                &self.driver.runtime.executes(),
            )
            .map_err(refresh_refusal_name)
        };
        let (bind_path, touched, refused, resolved, kernels, uniforms, micros) = match refresh {
            Ok((resolved, packed, touched)) => {
                self.packed_operands = packed;
                let mut micros = LowerMicros {
                    bind: refresh_started.elapsed().as_micros() as u64,
                    ..LowerMicros::default()
                };
                validate_resolved(&resolved, numeric_policy, &self.driver.runtime.executes())?;
                let prior = PriorKernels {
                    bound: &previous_bound,
                    kernels: std::mem::take(&mut self.kernels),
                };
                let operand_dtypes = program_operand_dtypes(
                    &self.program,
                    &resolved,
                    &self.packed_operands,
                    numeric_policy,
                    &self.driver.runtime.executes(),
                );
                let (kernels, uniforms) = emit_and_pack(
                    self.driver.runtime.emit_target(),
                    &resolved,
                    Some(prior),
                    &self.packed_operands,
                    numeric_policy,
                    self.driver.runtime.emit_device(),
                    &operand_dtypes,
                    &mut micros,
                )?;
                ("refresh", touched, "none", resolved, kernels, uniforms, micros)
            }
            Err(refused) => {
                let prior = same_policy.then(|| PriorKernels {
                    bound: &previous_bound,
                    kernels: std::mem::take(&mut self.kernels),
                });
                let (resolved, packed, mut micros) = bind_resolved(
                    &self.program,
                    &shapes,
                    &self.outputs,
                    &self.packed_operands,
                    numeric_policy,
                    false,
                    self.activation_storage,
                    &self.driver.runtime.executes(),
                )?;
                self.packed_operands = packed;
                let operand_dtypes = program_operand_dtypes(
                    &self.program,
                    &resolved,
                    &self.packed_operands,
                    numeric_policy,
                    &self.driver.runtime.executes(),
                );
                let (kernels, uniforms) = emit_and_pack(
                    self.driver.runtime.emit_target(),
                    &resolved,
                    prior,
                    &self.packed_operands,
                    numeric_policy,
                    self.driver.runtime.emit_device(),
                    &operand_dtypes,
                    &mut micros,
                )?;
                ("full", 0, refused, resolved, kernels, uniforms, micros)
            }
        };
        let op_count = resolved.len();
        self.refit_calls += 1;
        self.emit_calls += op_count.saturating_sub(micros.reused) as u64;
        self.emit_reused += micros.reused as u64;
        self.pack_calls += uniforms.len() as u64;
        self.numeric_policy = numeric_policy;
        self.shapes = shapes;
        *self.pool = PoolSchedule::build(&resolved, &kernels, &self.outputs);
        self.resolved = resolved;
        self.functions = vec![None; kernels.len()];
        self.kernels = kernels;
        self.uniforms = uniforms;
        let previous_rows = std::mem::replace(&mut self.bound_rows, rows);
        if same_policy && previous_rows != rows && (*self.rows_template).is_none() {
            self.learn_row_template(&previous_bound, previous_rows, symbols, numeric_policy)?;
        }
        proxima_telemetry::debug!(
            infer_us = infer_micros,
            bind_us = micros.bind,
            emit_us = micros.emit,
            emit_reused = micros.reused,
            pack_us = micros.pack,
            op_count = op_count,
            bind_path = bind_path,
            bind_touched = touched as u64,
            bind_refused = refused,
            refit_total = self.refit_calls,
            emit_total = self.emit_calls,
            emit_reused_total = self.emit_reused,
            pack_total = self.pack_calls,
            "cuda plan refit phases"
        );
        Ok(true)
    }

    /// Moves the bound ops to `rows` through the proof learned at an earlier row change:
    /// every integer is re-evaluated, the launch grids and uniforms follow, and nothing is
    /// bound, fused or emitted. `Ok(None)` when no proof exists or either row count is
    /// outside the class the proof covers.
    fn repack_rows(
        &mut self,
        rows: u64,
        shapes: &Shapes,
        numeric_policy: NumericPolicy,
    ) -> Result<Option<usize>, CudaDriverError> {
        if rows < ROW_CLASS_MIN || self.bound_rows < ROW_CLASS_MIN {
            return Ok(None);
        }
        let Some(Some(template)) = &*self.rows_template else {
            return Ok(None);
        };
        let apply_started = Instant::now();
        let mut next = self.resolved.clone();
        let Some(attention) = apply_row_template(template, rows, &mut next) else {
            return Ok(None);
        };
        for position in attention {
            let Some(patched) = refresh_cached_attention(&next[position], shapes) else {
                return Ok(None);
            };
            next[position] = patched;
        }
        let apply_us = elapsed_micros(apply_started);
        let mut follow = FollowMicros::default();
        for (position, bound) in next.iter().enumerate() {
            let op_started = Instant::now();
            self.follow_rows(position, bound, numeric_policy, &mut follow)?;
            let op_us = elapsed_micros(op_started);
            if op_us >= 50 {
                proxima_telemetry::debug!(
                    position = position as u64,
                    kind = bound.kind.name(),
                    op_us = op_us,
                    prologue = self.kernels[position].prologue.len() as u64,
                    "cuda repack slow op"
                );
            }
        }
        let pool_started = Instant::now();
        self.pool.relength(&next);
        let count = next.len();
        self.resolved = next;
        self.bound_rows = rows;
        self.patch_memo = None;
        proxima_telemetry::debug!(
            apply_us = apply_us,
            grid_us = follow.grid,
            uniform_us = follow.uniform,
            attention_us = follow.attention,
            pool_us = elapsed_micros(pool_started),
            op_count = count as u64,
            "cuda repack phases"
        );
        Ok(Some(count))
    }

    fn follow_rows(
        &mut self,
        position: usize,
        next: &BoundOp,
        numeric_policy: NumericPolicy,
        follow: &mut FollowMicros,
    ) -> Result<(), CudaDriverError> {
        let emit_error = |error: crate::error::EmitError| CudaDriverError::Emit {
            node: next.node,
            error: error.to_string(),
        };
        let device = self.driver.runtime.emit_device();
        let grid_started = Instant::now();
        if matches!(next.kind, BoundOpKind::CachedAttention { .. }) {
            let kept = crate::cuda::refit_kernel_on_target(
                self.driver.runtime.emit_target(),
                &self.resolved[position],
                next,
                &mut self.kernels[position],
                &self.packed_operands,
                numeric_policy,
                device,
                &OperandDtypes::new(),
            )
            .map_err(emit_error)?;
            if !kept {
                self.functions[position] = None;
            }
            follow.attention += elapsed_micros(grid_started);
        } else {
            crate::cuda::refit_grid_on_target(
                self.driver.runtime.emit_target(),
                &mut self.kernels[position],
                next,
                &self.packed_operands,
                numeric_policy,
                device,
            )
            .map_err(emit_error)?;
            follow.grid += elapsed_micros(grid_started);
        }
        let uniform_started = Instant::now();
        self.uniforms[position] = crate::cuda::pack_cuda_uniforms(next).map_err(emit_error)?;
        follow.uniform += elapsed_micros(uniform_started);
        Ok(())
    }

    fn shapes_for(&self, symbols: &[u64]) -> Result<Shapes, CudaDriverError> {
        let rows = symbols.first().copied().unwrap_or(0);
        let moved = (*self.shapes_template)
            .as_ref()
            .filter(|(base, varying, _)| varying.first() != Some(&true) || rows >= ROW_CLASS_MIN || base.first() == Some(&rows))
            .and_then(|template| apply_symbols_template(template, symbols));
        if moved.is_none() {
            let template = (*self.shapes_template).as_ref();
            proxima_telemetry::debug!(
                has_template = template.is_some(),
                base = ?template.map(|(base, _, _)| base.clone()),
                varying = ?template.map(|(_, varying, _)| varying.clone()),
                symbols = ?symbols,
                "cuda shapes template did not apply"
            );
        }
        match moved {
            Some(shapes) => Ok(shapes),
            None => infer(&self.program, symbols).map_err(|_| CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "shape inference failed",
            }),
        }
    }

    fn bind_at_rows(
        &self,
        symbols: &[u64],
        rows: u64,
        numeric_policy: NumericPolicy,
    ) -> Result<(Shapes, Vec<BoundOp>), CudaDriverError> {
        let mut moved = symbols.to_vec();
        moved[0] = rows;
        let shapes = infer(&self.program, &moved).map_err(|_| CudaDriverError::UnsupportedGraph {
            node: NodeId(0),
            reason: "shape inference failed",
        })?;
        let (bound, _, _) = bind_resolved(
            &self.program,
            &shapes,
            &self.outputs,
            &self.packed_operands,
            numeric_policy,
            false,
            self.activation_storage,
            &self.driver.runtime.executes(),
        )?;
        Ok((shapes, bound))
    }

    /// Learns the row template when a wide-class plan is first built, by binding at two other
    /// in-class row counts, so the first row change already repacks. A refused or failed
    /// learning leaves the lazy learning at the first full refit in place.
    fn learn_rows_at_build(&mut self, symbols: &[u64], numeric_policy: NumericPolicy) {
        let rows = self.bound_rows;
        self.learn_shapes_template(symbols, learning_rows(rows));
        let Some(others) = learning_rows(rows).filter(|_| !symbols.is_empty() && rows_enumerable(&self.resolved)) else {
            return;
        };
        let started = Instant::now();
        let learned = self.learn_from_binds(symbols, others, numeric_policy);
        match learned {
            Ok(fitted) => proxima_telemetry::debug!(
                learn_us = elapsed_micros(started),
                rows = rows,
                fitted = fitted,
                "cuda plan learned its row template at build"
            ),
            Err(error) => proxima_telemetry::debug!(
                rows = rows,
                error = %error,
                "cuda row template not learned at build"
            ),
        }
    }

    /// Fits every extent affine in all symbols from shape inference at the build symbols, at two
    /// other row counts when the plan is wide, at each later symbol moved alone, and at every
    /// symbol moved together, so a refit at new row count and new kv skips inference.
    fn learn_shapes_template(&mut self, symbols: &[u64], wide_rows: Option<[u64; 2]>) {
        let moved_alone = |index: usize, value: u64| {
            let mut moved = symbols.to_vec();
            moved[index] = value;
            moved
        };
        let mut points = vec![symbols.to_vec()];
        points.extend(wide_rows.into_iter().flatten().map(|rows| moved_alone(0, rows)));
        points.extend((1..symbols.len()).map(|index| moved_alone(index, symbols[index] + 1)));
        let joint: Vec<u64> = symbols
            .iter()
            .enumerate()
            .map(|(index, value)| match (index, wide_rows) {
                (0, Some([_, second])) => second,
                (0, None) => *value,
                _ => value + 3,
            })
            .collect();
        points.push(joint);
        let inferred: Option<Vec<Shapes>> = points.iter().map(|point| infer(&self.program, point).ok()).collect();
        let binds: Option<Vec<(&[u64], &Shapes)>> = inferred
            .as_ref()
            .map(|all| points.iter().map(Vec::as_slice).zip(all).collect());
        *self.shapes_template = binds.as_deref().and_then(symbols_shapes_template);
        proxima_telemetry::debug!(
            fitted = self.shapes_template.is_some(),
            symbol_count = symbols.len() as u64,
            "cuda plan learned its shapes template at build"
        );
    }

    fn learn_from_binds(
        &mut self,
        symbols: &[u64],
        others: [u64; 2],
        numeric_policy: NumericPolicy,
    ) -> Result<bool, CudaDriverError> {
        let [first_rows, second_rows] = others;
        let (_, first) = self.bind_at_rows(symbols, first_rows, numeric_policy)?;
        let (_, second) = self.bind_at_rows(symbols, second_rows, numeric_policy)?;
        self.fit_row_template(&first, first_rows, &second, second_rows, numeric_policy)
    }

    /// After a row change that bound from scratch, proves from the old and new binds plus one
    /// more at a row count between them that every integer in the ops is affine in the row
    /// count and that every kernel text is the same, so the next change can be repacked.
    fn learn_row_template(
        &mut self,
        previous: &[BoundOp],
        previous_rows: u64,
        symbols: &[u64],
        numeric_policy: NumericPolicy,
    ) -> Result<(), CudaDriverError> {
        let rows = self.bound_rows;
        let middle = previous_rows / 2 + rows / 2 + (previous_rows & rows & 1);
        let distinct = middle != previous_rows && middle != rows;
        let in_class = previous_rows >= ROW_CLASS_MIN && rows >= ROW_CLASS_MIN && distinct;
        if !in_class || previous.len() != self.resolved.len() {
            return Ok(());
        }
        let (_, between) = self.bind_at_rows(symbols, middle, numeric_policy)?;
        self.fit_row_template(previous, previous_rows, &between, middle, numeric_policy)?;
        Ok(())
    }

    fn fit_row_template(
        &mut self,
        previous: &[BoundOp],
        previous_rows: u64,
        between: &[BoundOp],
        middle: u64,
        numeric_policy: NumericPolicy,
    ) -> Result<bool, CudaDriverError> {
        let rows = self.bound_rows;
        let same_text = self.text_is_row_independent(previous, between, numeric_policy)?;
        let template = same_text
            .then(|| {
                row_template(&[
                    (previous_rows, previous),
                    (rows, self.resolved.as_slice()),
                    (middle, between),
                ])
            })
            .flatten();
        let fitted = template.is_some();
        proxima_telemetry::debug!(
            same_text = same_text,
            fitted = fitted,
            rows = rows,
            previous_rows = previous_rows,
            middle_rows = middle,
            "cuda row template learned"
        );
        *self.rows_template = Some(template);
        Ok(fitted)
    }

    fn text_is_row_independent(
        &self,
        previous: &[BoundOp],
        between: &[BoundOp],
        numeric_policy: NumericPolicy,
    ) -> Result<bool, CudaDriverError> {
        let device = self.driver.runtime.emit_device();
        let packed = &self.packed_operands;
        let target = self.driver.runtime.emit_target();
        if between.len() != previous.len() || previous.len() != self.resolved.len() {
            return Ok(false);
        }
        for position in 0..previous.len() {
            let current = &self.resolved[position];
            let same_plan = |other: &BoundOp| {
                between.len() == previous.len()
                    && crate::cuda::same_text_modulo_extents_on_target(target, other, current, packed, numeric_policy, device)
            };
            let first = same_plan(&previous[position]);
            let second = same_plan(&between[position]);
            if !(first && second) {
                proxima_telemetry::debug!(position = position as u64, kind = current.kind.name(), "row template refused: kernel text plan differs");
                return Ok(false);
            }
            for other in [&previous[position], &between[position]] {
                if other.extents != current.extents && !self.emits_same_text(other, position, numeric_policy)? {
                    proxima_telemetry::debug!(position = position as u64, kind = current.kind.name(), "row template refused: emitted text differs");
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn emits_same_text(
        &self,
        previous: &BoundOp,
        position: usize,
        numeric_policy: NumericPolicy,
    ) -> Result<bool, CudaDriverError> {
        let operand_dtypes = program_operand_dtypes(
            &self.program,
            core::slice::from_ref(previous),
            &self.packed_operands,
            numeric_policy,
            &self.driver.runtime.executes(),
        );
        let before = crate::cuda::emit_on_target(
            self.driver.runtime.emit_target(),
            previous,
            &self.packed_operands,
            numeric_policy,
            self.driver.runtime.emit_device(),
            &operand_dtypes,
        )
        .map_err(|error| CudaDriverError::Emit {
            node: previous.node,
            error: error.to_string(),
        })?;
        let after = &self.kernels[position];
        let prologue_text = |kernel: &crate::cuda::CudaKernel| -> Vec<(String, String)> {
            kernel
                .prologue
                .iter()
                .map(|prologue| (prologue.entry.clone(), prologue.source.clone()))
                .collect()
        };
        Ok(before.entry == after.entry
            && before.source == after.source
            && prologue_text(&before) == prologue_text(after))
    }

    /// Moves the plan to `shapes` by patching only the cached-attention ops
    /// whose key range moved, leaving every other bound op, kernel and uniform
    /// block untouched. `Ok(None)` when the move reaches anything else, so the
    /// caller falls back to a full re-bind. The touched positions are a function
    /// of the changed-node set alone, so a repeat of the same set skips the walk.
    fn patch_cached_attention(
        &mut self,
        shapes: &Shapes,
        numeric_policy: NumericPolicy,
    ) -> Result<Option<usize>, CudaDriverError> {
        let changed = changed_nodes(&self.program, &self.shapes, shapes);
        let known = self.patch_memo.take().filter(|(seen, _)| *seen == changed);
        let (changed, positions) = known.unwrap_or_else(|| {
            let positions = touched_positions(&self.resolved, &self.program, &changed);
            (changed, positions)
        });
        let Some(patches) = refit_cached_attention_at(&self.resolved, &positions, shapes) else {
            return Ok(None);
        };
        let touched = patches.len();
        let mut lengths_moved = false;
        for (position, bound) in patches {
            let previous = std::mem::replace(&mut self.resolved[position], bound);
            lengths_moved |= previous.output_len() != self.resolved[position].output_len();
            let operand_dtypes = program_operand_dtypes(
                &self.program,
                core::slice::from_ref(&self.resolved[position]),
                &self.packed_operands,
                numeric_policy,
                &self.driver.runtime.executes(),
            );
            let text_kept = crate::cuda::refit_kernel_on_target(
                self.driver.runtime.emit_target(),
                &previous,
                &self.resolved[position],
                &mut self.kernels[position],
                &self.packed_operands,
                numeric_policy,
                self.driver.runtime.emit_device(),
                &operand_dtypes,
            )
            .map_err(|error| CudaDriverError::Emit {
                node: previous.node,
                error: error.to_string(),
            })?;
            if !text_kept {
                self.functions[position] = None;
            }
            self.uniforms[position] = crate::cuda::pack_cuda_uniforms(&self.resolved[position])
                .map_err(|error| CudaDriverError::Emit {
                    node: previous.node,
                    error: error.to_string(),
                })?;
        }
        self.patch_memo = Some((changed, positions));
        if lengths_moved {
            *self.pool = PoolSchedule::build(&self.resolved, &self.kernels, &self.outputs);
        }
        Ok(Some(touched))
    }

    /// Cumulative `[refit calls, ops re-emitted, ops reused, ops repacked, refits done as a row repack]` since this plan was built.
    #[must_use]
    pub const fn refit_counters(&self) -> [u64; 5] {
        [
            self.refit_calls,
            self.emit_calls,
            self.emit_reused,
            self.pack_calls,
            self.repack_calls,
        ]
    }

    pub fn scoped_reservation(&self) -> Result<(u64, u64, u64), CudaDriverError> {
        let mut total = 0_u64;
        let mut largest = 0_u64;
        let mut add = |bytes: u64| -> Result<(), CudaDriverError> {
            total = crate::backend::checked_resource_sum([total, bytes])
                .ok_or(CudaDriverError::ResourceOverflow)?;
            largest = largest.max(bytes);
            Ok(())
        };
        for node in &self.block_nodes {
            let bytes = cuda_float_bytes(self.shapes.of(*node))?;
            match self.arena.buffers.get(node) {
                Some(CudaGraphBuffer::F32(buffer)) => {
                    let current = (R::len(buffer) as u64)
                        .checked_mul(4)
                        .ok_or(CudaDriverError::ResourceOverflow)?;
                    add(current)?;
                    if current != bytes {
                        add(bytes)?;
                    }
                }
                Some(CudaGraphBuffer::Bytes(buffer)) => {
                    add(R::len(buffer.as_ref()) as u64)?;
                    add(bytes)?;
                }
                None => add(bytes)?,
            }
        }
        let mut host_workspace = 0_u64;
        let release_plan = &self.pool.plan;
        for (length, count) in &self.pool.peak {
            for _ in 0..*count {
                add(*length as u64 * 4)?;
            }
        }
        for (bound, uniform) in self.resolved.iter().zip(&self.uniforms) {
            let bytes = match &bound.kind {
                BoundOpKind::BlockQuantize { .. } => device_words(bound) as u64 * 4,
                _ => cuda_float_bytes(self.shapes.of(bound.node))?,
            };
            let pooled = release_plan.pooled.contains(&bound.node);
            match self.arena.buffers.get(&bound.node) {
                _ if pooled => {}
                Some(CudaGraphBuffer::F32(buffer)) => {
                    let current = (R::len(buffer) as u64)
                        .checked_mul(4)
                        .ok_or(CudaDriverError::ResourceOverflow)?;
                    add(current)?;
                    if current != bytes {
                        add(bytes)?;
                    }
                }
                Some(CudaGraphBuffer::Bytes(buffer)) => {
                    add(R::len(buffer.as_ref()) as u64)?;
                    add(bytes)?;
                }
                None => add(bytes)?,
            }
            if !matches!(bound.kind, BoundOpKind::Constant { .. }) {
                if let Some(existing) = self.arena.uniforms.get(&bound.node) {
                    add(R::len(existing) as u64)?;
                    if R::len(existing) != uniform.len() {
                        add(uniform.len() as u64)?;
                    }
                } else {
                    add(uniform.len() as u64)?;
                }
            } else {
                host_workspace = host_workspace.max(bytes);
            }
        }
        let host = self
            .outputs
            .iter()
            .try_fold(0_u64, |count, node| {
                count
                    .checked_add(cuda_float_bytes(self.shapes.of(*node))?)
                    .ok_or(CudaDriverError::ResourceOverflow)
            })?
            .max(host_workspace);
        Ok((host, total, largest))
    }

    fn validate_resources(
        &self,
        blocks: &[QuantizedBlock<'_>],
        resident_names: Option<&BTreeSet<&str>>,
    ) -> Result<(), CudaDriverError> {
        let Some(limits) = &self.constraints else {
            return Ok(());
        };
        if blocks.len() != self.block_contract.len() {
            return Err(CudaDriverError::InputContractChanged {
                node: NodeId(0),
                reason: "input count differs from the planned contract",
            });
        }
        for ((node, expected), block) in self
            .block_nodes
            .iter()
            .zip(&self.block_contract)
            .zip(blocks)
        {
            if cuda_input_contract(block)? != *expected {
                return Err(CudaDriverError::InputContractChanged {
                    node: *node,
                    reason: "codec or element count differs from the planned contract",
                });
            }
        }
        if limits.max_host_tensor_bytes.is_none()
            && limits.max_device_buffer_bytes.is_none()
            && limits.max_single_device_buffer_bytes.is_none()
        {
            return Ok(());
        }
        if resident_names.is_some_and(|names| !names.is_empty()) || !self.resident_nodes.is_empty()
        {
            return Err(CudaDriverError::UnsupportedConstraint {
                field: "constraints",
                reason: "shared resident cache has no per-plan reservation owner",
            });
        }
        if ["PROXIMA_CUDA_TRACE_OUTPUTS", "PROXIMA_CUDA_TRACE_VALUES"]
            .iter()
            .any(|key| std::env::var_os(key).is_some())
        {
            return Err(CudaDriverError::UnsupportedConstraint {
                field: "max_host_tensor_bytes",
                reason: "diagnostic tensor copies have no bounded workspace contract",
            });
        }
        if blocks
            .iter()
            .any(|block| !matches!(block, QuantizedBlock::Float32(_)))
            || self.resolved.iter().any(|bound| {
                bound.dtype != DType::FLOAT32
                    || gather_count(bound) != 0
                    || !matches!(
                        bound.kind,
                        BoundOpKind::Elementwise { .. }
                            | BoundOpKind::Reduce { .. }
                            | BoundOpKind::Iota
                            | BoundOpKind::Constant { .. }
                    )
            })
        {
            return Err(CudaDriverError::UnsupportedConstraint {
                field: "constraints",
                reason: "packed, gathered or specialized operations have no bounded CUDA workspace contract",
            });
        }
        let (host, total, largest) = self.scoped_reservation()?;
        for (field, requested, limit) in [
            ("max_host_tensor_bytes", host, limits.max_host_tensor_bytes),
            (
                "max_device_buffer_bytes",
                total,
                limits.max_device_buffer_bytes,
            ),
            (
                "max_single_device_buffer_bytes",
                largest,
                limits.max_single_device_buffer_bytes,
            ),
        ] {
            if let Some(limit) = limit
                && requested > limit
            {
                return Err(CudaDriverError::ResourceLimit {
                    field,
                    requested,
                    limit,
                });
            }
        }
        let retained_graph = self.arena.buffers.values().try_fold(0_u64, |sum, buffer| {
            let bytes = match buffer {
                CudaGraphBuffer::F32(values) => (R::len(values) as u64)
                    .checked_mul(4)
                    .ok_or(CudaDriverError::ResourceOverflow)?,
                CudaGraphBuffer::Bytes(values) => R::len(values.as_ref()) as u64,
            };
            sum.checked_add(bytes)
                .ok_or(CudaDriverError::ResourceOverflow)
        })?;
        let retained_uniforms = self
            .arena
            .uniforms
            .values()
            .try_fold(0_u64, |sum, buffer| {
                sum.checked_add(R::len(buffer) as u64)
                    .ok_or(CudaDriverError::ResourceOverflow)
            })?;
        let retained = retained_graph
            .checked_add(retained_uniforms)
            .and_then(|sum| sum.checked_add(self.arena.pool_bytes()))
            .ok_or(CudaDriverError::ResourceOverflow)?;
        let additional = total.saturating_sub(retained);
        // this is an observed free-capacity check; other contexts can allocate after it.
        let (free, _) = self.driver.memory_info()?;
        if additional > free as u64 {
            return Err(CudaDriverError::ResourceLimit {
                field: "observed_free_device_bytes",
                requested: additional,
                limit: free as u64,
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn kernel_launches(&self) -> Vec<String> {
        self.kernels
            .iter()
            .map(|kernel| {
                format!(
                    "entry={} grid_threads={} block_width={:?}",
                    kernel.entry,
                    kernel.grid.threads(),
                    kernel.grid.block_width_or_default()
                )
            })
            .collect()
    }

    #[must_use]
    pub fn replace_kernel_source(&mut self, position: usize, source: String) -> bool {
        let Some(kernel) = self.kernels.get_mut(position) else {
            return false;
        };
        kernel.source = source;
        if let Some(function) = self.functions.get_mut(position) {
            *function = None;
        }
        true
    }

    #[must_use]
    pub fn kernel_identities(&self) -> Vec<String> {
        self.kernels
            .iter()
            .flat_map(|kernel| kernel.prologue.iter().chain(std::iter::once(kernel)))
            .map(|kernel| format!("{}\n{}", kernel.entry, kernel.source))
            .collect()
    }
    /// The operations this plan's runtime executes: the grant its graph was bound under.
    #[must_use]
    pub fn executes(&self) -> Executes {
        self.driver.runtime.executes()
    }

    /// Whether every reader of the named K/V cache inputs is a fused `CachedAttention`, whose
    /// runtime bound excludes the pad rows of a bucketed range. A plan with no such input, no
    /// fused reader, a reader of another kind, or a K/V input that is itself an output must
    /// bind its cached range exact.
    #[must_use]
    pub fn pads_kv_safely(&self, is_kv_input: impl Fn(&str) -> bool) -> bool {
        kv_read_only_by_fused_attention(&self.program, &self.resolved, &self.outputs, is_kv_input)
    }

    #[must_use]
    pub fn resolved_op_count(&self) -> usize {
        self.resolved.len()
    }
    pub fn device_identity(&self) -> Result<String, CudaDriverError> {
        Ok(self.driver.runtime.device_identity()?)
    }
    /// Marks the inputs named in `resident_names` as immutable for the life of
    /// their source bytes, so any plan on this driver reuses the device buffer
    /// another plan uploaded for the same (address, length). Same contract as
    /// the Metal plan's `mark_resident`: a scan over the block-input nodes, not
    /// a device operation.
    pub fn mark_resident(&mut self, resident_names: &BTreeSet<&str>) {
        let mut marked: Vec<NodeId> = self
            .block_nodes
            .iter()
            .filter(|node| {
                self.program[node.0 as usize]
                    .name()
                    .is_some_and(|name| resident_names.contains(name))
            })
            .copied()
            .collect();
        marked.sort_unstable();
        self.resident_nodes = marked.into_boxed_slice();
    }

    /// Keeps each `(input, output)` pair's state on the device: after a call the output buffer
    /// becomes the next call's input buffer and is neither read back nor re-uploaded. Returns
    /// `false` (carrying nothing) when a resource-constrained plan or a pair whose input and
    /// output differ in element count cannot swap buffers. Pairs equal to the current set keep
    /// their live device state; a different set requires [`Self::drain_carried`] first.
    pub fn carry_state(&mut self, pairs: &[(NodeId, NodeId)]) -> bool {
        let wanted = pairs.to_vec();
        if self.carry.state.selects(&wanted) {
            return self.drive_state(StateEvent::Select { pairs: wanted }).is_ok();
        }
        let swappable = self.constraints.is_none()
            && pairs.iter().all(|(input, output)| {
                self.outputs.contains(output)
                    && self.shapes.of(*input).iter().product::<u64>()
                        == self.shapes.of(*output).iter().product::<u64>()
            });
        if !swappable || self.carry.state.has_live() {
            return false;
        }
        self.drive_state(StateEvent::Select { pairs: wanted }).is_ok()
    }

    fn drive_state(&mut self, event: StateEvent<NodeId>) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        let commands = self.carry.state.step(event)?;
        let mut drained = Vec::new();
        for command in commands {
            match command {
                StateCommand::Adopt { input } => self.adopt_parked_state(input)?,
                StateCommand::Swap { input, output } => self.swap_buffers(input, output),
                StateCommand::Park { input, output } => self.park_state(input, output),
                StateCommand::Drain { input, output } => {
                    drained.extend(self.drain_state(input, output)?);
                }
            }
        }
        Ok(drained)
    }

    fn adopt_parked_state(&mut self, input: NodeId) -> Result<(), CudaDriverError> {
        let Some((_, buffer)) = self.driver.parked_state.lock().remove(&input) else {
            return Ok(());
        };
        self.arena.buffers.insert(input, buffer);
        self.arena.resident_sources.remove(&input);
        self.carry.state.step(StateEvent::Resident { input })?;
        Ok(())
    }

    fn park_state(&mut self, input: NodeId, output: NodeId) {
        if let Some(buffer) = self.arena.buffers.remove(&input) {
            self.driver.parked_state.lock().insert(input, (output, buffer));
        }
    }

    fn drain_state(&mut self, input: NodeId, output: NodeId) -> Result<Option<(NodeId, Vec<f32>)>, CudaDriverError> {
        let Some(buffer) = self.arena.buffers.remove(&input) else {
            return Ok(None);
        };
        self.read_state(output, buffer).map(Some)
    }

    fn read_state(&mut self, output: NodeId, buffer: CudaGraphBuffer<R>) -> Result<(NodeId, Vec<f32>), CudaDriverError> {
        let CudaGraphBuffer::F32(slice) = buffer else {
            return Err(CudaDriverError::BindingContract);
        };
        let values = self.driver.runtime.download(&slice)?;
        self.carry.host_bytes += (values.len() * std::mem::size_of::<f32>()) as u64;
        Ok((output, values))
    }

    /// Hands live carried state to the driver so the plan that replaces it adopts it on
    /// the device instead of re-uploading a stale host copy.
    pub fn park_carried(&mut self) {
        self.park_appended();
        for command in self.carry.state.park() {
            if let StateCommand::Park { input, output } = command {
                self.park_state(input, output);
            }
        }
    }

    /// Reads every live carried state back to the host, keyed by the output node that produces
    /// it, and marks the device copies stale so the next call uploads its host inputs again.
    pub fn drain_carried(&mut self) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        self.park_appended();
        let mut drained = self.drive_state(StateEvent::Ended)?;
        let parked = std::mem::take(&mut *self.driver.parked_state.lock());
        for (_, (output, buffer)) in parked {
            drained.push(self.read_state(output, buffer)?);
        }
        Ok(drained)
    }

    /// Keeps each `(cache input, new-rows output, cache format)` triple's cache on the device: the
    /// input leaf has shape `[extent_rows, ..row]`, the output `[new_rows, ..row]`. The first call
    /// uploads the host block (holding `cached_rows` valid rows); afterwards the input is never
    /// uploaded and each call's output rows are encoded into the cache's format at the end of the
    /// valid rows through [`CudaDriver::encode_rows_into`], and the output is not read back. The
    /// graph-output format is the identity encode. Returns `false` (carrying nothing) for a
    /// resource-constrained plan, an output that is not a plan output, differing row widths, a
    /// format the encode render refuses, or a format whose host block upload is not encoded yet
    /// (anything but the graph-output format). The same set keeps live state; a different set
    /// requires [`Self::drain_appended_rows`] first. Every decision comes from [`KvCarry`]; this
    /// method only runs the commands it returns.
    pub fn carry_rows(&mut self, pairs: &[(NodeId, NodeId, NumericFormat)], cached_rows: u64) -> bool {
        let encodes: Option<BTreeMap<NodeId, KvEncode>> = pairs
            .iter()
            .map(|(input, _, format)| {
                let encode = kv_encode(self.driver.runtime.emit_target(), format).filter(|_| format.is_identity_of(&NumericFormat::default()));
                encode.map(|encode| (*input, encode))
            })
            .collect();
        let Some(encodes) = encodes else {
            return false;
        };
        let wanted: Vec<RowPair<NodeId>> = pairs
            .iter()
            .map(|(input, output, _)| RowPair {
                input: *input,
                output: *output,
                row_width: self.row_width(*input),
            })
            .collect();
        self.carry.cached_rows = cached_rows;
        self.carry.fresh_cached_rows = Some(cached_rows);
        if self.carry.rows.selects(&wanted) {
            if !wanted.is_empty() && encodes != self.carry.encodes {
                return false;
            }
            return self.drive_rows(KvEvent::Select { pairs: wanted }).is_ok();
        }
        let appendable = self.constraints.is_none()
            && pairs
                .iter()
                .all(|(input, output, _)| self.appendable(*input, *output));
        if !appendable || self.carry.rows.has_live() {
            return false;
        }
        self.carry.encodes = encodes;
        self.drive_rows(KvEvent::Select { pairs: wanted }).is_ok()
    }

    fn appendable(&self, input: NodeId, output: NodeId) -> bool {
        !self.shapes.of(input).is_empty()
            && !self.shapes.of(output).is_empty()
            && self.outputs.contains(&output)
            && !self.carry.state.is_pair(&input)
            && self.row_width(input) == self.row_width(output)
    }

    fn row_width(&self, node: NodeId) -> u64 {
        self.shapes.of(node).iter().skip(1).product()
    }

    fn drive_rows(&mut self, event: KvEvent<NodeId>) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        let commands = self.carry.rows.step(event)?;
        let mut drained = Vec::new();
        for command in commands {
            match command {
                KvCommand::Adopt { input } => self.adopt_parked_rows(input)?,
                KvCommand::Park { input, valid_rows } => self.park_rows(input, valid_rows),
                KvCommand::Grow { input, to_elements, keep_elements } => {
                    self.grow_cache(input, to_elements as usize, keep_elements as usize)?;
                }
                KvCommand::AppendCopy { input, output, offset_elements, count_elements } => {
                    self.copy_rows(input, output, offset_elements as usize, count_elements as usize)?;
                }
                KvCommand::Truncate { .. } => {}
                KvCommand::Drain { input, valid_rows, row_width } => {
                    drained.extend(self.drain_rows(input, valid_rows, row_width)?);
                }
            }
        }
        Ok(drained)
    }

    fn adopt_parked_rows(&mut self, input: NodeId) -> Result<(), CudaDriverError> {
        let Some((valid_rows, buffer)) = self.driver.parked_appends.lock().remove(&input) else {
            return Ok(());
        };
        let capacity_elements = buffer_elements(&buffer);
        self.arena.buffers.insert(input, buffer);
        self.arena.resident_sources.remove(&input);
        self.carry
            .rows
            .step(KvEvent::Resident { input, valid_rows, capacity_elements })?;
        Ok(())
    }

    fn park_rows(&mut self, input: NodeId, valid_rows: u64) {
        if let Some(buffer) = self.arena.buffers.remove(&input) {
            self.driver.parked_appends.lock().insert(input, (valid_rows, buffer));
        }
    }

    fn park_appended(&mut self) {
        for command in self.carry.rows.park() {
            if let KvCommand::Park { input, valid_rows } = command {
                self.park_rows(input, valid_rows);
            }
        }
    }

    fn drain_rows(
        &mut self,
        input: NodeId,
        valid_rows: u64,
        row_width: u64,
    ) -> Result<Option<(NodeId, Vec<f32>)>, CudaDriverError> {
        let Some(buffer) = self.arena.buffers.remove(&input) else {
            return Ok(None);
        };
        self.read_valid_rows(input, valid_rows, row_width, buffer).map(Some)
    }

    fn read_valid_rows(
        &mut self,
        input: NodeId,
        valid_rows: u64,
        row_width: u64,
        buffer: CudaGraphBuffer<R>,
    ) -> Result<(NodeId, Vec<f32>), CudaDriverError> {
        let CudaGraphBuffer::F32(slice) = buffer else {
            return Err(CudaDriverError::BindingContract);
        };
        let valid = (valid_rows * row_width) as usize;
        let values = self.driver.runtime.download_range(&slice, 0..valid)?;
        self.carry.host_bytes += (values.len() * std::mem::size_of::<f32>()) as u64;
        Ok((input, values))
    }

    /// Reads the valid rows of every live appended cache back to the host, keyed by the cache
    /// INPUT node, and marks the device copies stale so the next call uploads its host block again.
    pub fn drain_appended_rows(&mut self) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        let mut drained = self.drive_rows(KvEvent::Ended)?;
        let parked = std::mem::take(&mut *self.driver.parked_appends.lock());
        for (input, (valid_rows, buffer)) in parked {
            let row_width = self.row_width(input);
            drained.push(self.read_valid_rows(input, valid_rows, row_width, buffer)?);
        }
        drained.sort_by_key(|(input, _)| *input);
        Ok(drained)
    }

    fn begin_rows_step(&mut self) -> Result<(), CudaDriverError> {
        let cached_rows = self.carry.fresh_cached_rows.take();
        let extents: Vec<Extent<NodeId>> = self
            .carry
            .rows
            .pairs()
            .filter(|pair| self.carry.rows.is_live(&pair.input))
            .map(|pair| Extent {
                input: pair.input,
                cached_rows,
                extent_elements: self.shapes.of(pair.input).iter().product::<u64>(),
            })
            .collect();
        self.drive_rows(KvEvent::StepBegan { extents }).map(|_| ())
    }

    fn append_new_rows(&mut self) -> Result<(), CudaDriverError> {
        let appended: Vec<Appended<NodeId>> = self
            .carry
            .rows
            .pairs()
            .filter(|pair| self.carry.rows.is_live(&pair.input))
            .map(|pair| Appended {
                input: pair.input,
                new_rows: self.shapes.of(pair.output).first().copied().unwrap_or(0),
            })
            .collect();
        self.drive_rows(KvEvent::RowsAppended { appended }).map(|_| ())
    }

    /// The verify step kept `kept` of the rows it appended; the rest are dropped by moving the
    /// valid-row count back, with no device copy.
    pub fn accept_rows(&mut self, kept: u64) -> Result<(), CudaDriverError> {
        self.drive_rows(KvEvent::Accepted { kept }).map(|_| ())
    }

    fn grow_cache(&mut self, input: NodeId, to_elements: usize, keep_elements: usize) -> Result<(), CudaDriverError> {
        let Some(CudaGraphBuffer::F32(current)) = self.arena.buffers.get(&input) else {
            return Err(CudaDriverError::MissingBuffer(input));
        };
        let mut grown = self.driver.runtime.alloc_zeros::<f32>(to_elements)?;
        let preserved = keep_elements.min(R::len(current));
        if preserved > 0 {
            self.driver.runtime.copy_range(current, 0..preserved, &mut grown, 0)?;
        }
        self.arena.allocations += 1;
        self.arena.buffers.insert(input, CudaGraphBuffer::F32(grown));
        Ok(())
    }

    fn copy_rows(
        &mut self,
        input: NodeId,
        output: NodeId,
        offset: usize,
        count: usize,
    ) -> Result<(), CudaDriverError> {
        let Some(encode) = self.carry.encodes.get(&input) else {
            return Err(CudaDriverError::MissingBuffer(input));
        };
        let Some(CudaGraphBuffer::F32(mut cache)) = self.arena.buffers.remove(&input) else {
            return Err(CudaDriverError::MissingBuffer(input));
        };
        let appended = match self.arena.buffers.get(&output) {
            Some(CudaGraphBuffer::F32(source)) => self.driver.encode_into_cache(encode, source, &mut cache, offset, count),
            _ => Err(CudaDriverError::MissingBuffer(output)),
        };
        self.arena.buffers.insert(input, CudaGraphBuffer::F32(cache));
        appended
    }

    /// Bytes of carried recurrent state that crossed PCIe in either direction so far.
    #[must_use]
    pub fn carried_host_bytes(&self) -> u64 {
        self.carry.host_bytes
    }

    fn swap_carried_buffers(&mut self) {
        let ready: Vec<NodeId> = self
            .carry
            .state
            .pairs()
            .filter(|(input, output)| self.arena.buffers.contains_key(*output) && self.arena.buffers.contains_key(*input))
            .map(|(input, _)| *input)
            .collect();
        if let Ok(commands) = self.carry.state.step(StateEvent::StepFinished { ready }) {
            for command in commands {
                if let StateCommand::Swap { input, output } = command {
                    self.swap_buffers(input, output);
                }
            }
        }
    }

    fn swap_buffers(&mut self, input: NodeId, output: NodeId) {
        let (Some(next), Some(previous)) = (self.arena.buffers.remove(&output), self.arena.buffers.remove(&input)) else {
            return;
        };
        self.arena.buffers.insert(input, next);
        self.arena.buffers.insert(output, previous);
        self.arena.resident_sources.remove(&input);
    }

    /// Executes the planned graph with name-resolved inputs and returns the
    /// requested outputs. Packed bytes stay on device in the persistent arena.
    pub fn execute_named(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&BTreeSet<&str>>,
    ) -> Result<Evaluated, CudaDriverError> {
        self.execute_named_launch_events(named, resident_names, None)
    }

    /// Diagnostic twin of [`Self::execute_named`]: the same launches on the same stream, each
    /// bracketed by a pair of CUDA events, returning one [`crate::OpGpuTiming`] per launched op in
    /// plan order -- the record Metal's `execute_plan_named_op_timed` returns. `gpu_ns` is the
    /// GPU-timeline gap between the op's start and end event (event resolution is about 0.5 us), so
    /// a 3 us kernel reads as a few ticks and host launch gaps between ops are not charged to it.
    /// Ops the plan never launches (idle grids, constants) have no record. Two event records per
    /// launch is the whole overhead, which is why serving never takes this entry.
    ///
    /// # Errors
    /// Propagates every error [`Self::execute_named`] can raise, plus event-timing failures.
    #[cfg(feature = "instrument")]
    pub fn execute_named_op_timed(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&BTreeSet<&str>>,
    ) -> Result<(Evaluated, Vec<crate::OpGpuTiming>), CudaDriverError> {
        let mut launches: Vec<LaunchEvents<R>> = Vec::with_capacity(self.resolved.len());
        let evaluated = self.execute_named_launch_events(named, resident_names, Some(&mut launches))?;
        let timings = launches
            .iter()
            .map(|launch| self.op_timing(launch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluated, timings))
    }

    #[cfg(feature = "instrument")]
    fn op_timing(&self, launch: &LaunchEvents<R>) -> Result<crate::OpGpuTiming, CudaDriverError> {
        let bound = &self.resolved[launch.position];
        let millis = R::elapsed_ms(&launch.start, &launch.end)?;
        Ok(crate::OpGpuTiming {
            node: bound.node,
            kind: bound.kind.name(),
            extents: bound.extents.clone(),
            output_axes: match &bound.kind {
                BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
                _ => Vec::new(),
            },
            operand_bytes: launch.operand_bytes,
            bound_buffer_bytes: launch.operand_bytes,
            gpu_ns: (millis * 1_000_000.0).max(0.0) as u64,
            weight_name: bound
                .operands()
                .iter()
                .find_map(|(source, _, _)| self.program[source.0 as usize].name())
                .map(str::to_owned),
            operand_count: bound.operands().len(),
            packed_codec: bound
                .operands()
                .iter()
                .find_map(|(source, _, _)| self.packed_operands.get(source).copied()),
            packed_kernel_variant: "",
            packed_row_block_rejection: None,
        })
    }

    /// Bytes of the device buffers `bound` reads, as the arena holds them right now.
    fn operand_buffer_bytes(&self, bound: &BoundOp) -> u64 {
        bound
            .operands()
            .iter()
            .filter_map(|(source, _, _)| self.arena.buffers.get(source))
            .map(|buffer| match buffer {
                CudaGraphBuffer::Bytes(values) => R::len(values.as_ref()) as u64,
                plain => cuda_float_bytes(&[buffer_elements(plain)]).unwrap_or(0),
            })
            .sum()
    }

    fn execute_named_launch_events(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&BTreeSet<&str>>,
        mut op_events: Option<&mut Vec<LaunchEvents<R>>>,
    ) -> Result<Evaluated, CudaDriverError> {
        let host_timing = std::env::var_os("PROXIMA_CUDA_HOST_TIMING").is_some();
        let upload_started = host_timing.then(std::time::Instant::now);
        let phase_started = Instant::now();
        let upload_bytes_before = self.driver.packed_upload_bytes();
        let compilations_before = self.driver.kernel_compilations();
        let nvrtc_before = self.driver.nvrtc_micros();
        let load_before = self.driver.load_micros();
        let lookup_before = self.driver.runtime.compile_counters().lookup_micros();
        let mut upload_call_us = 0u64;
        let mut resident_hits = 0usize;
        let mut resident_misses = 0usize;
        let blocks = resolve_named_blocks(&self.program, named).map_err(|_| {
            CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "named input resolution failed",
            }
        })?;
        if blocks.len() != self.block_nodes.len() {
            return Err(CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "input count mismatch",
            });
        }
        self.validate_resources(&blocks, resident_names)?;
        for (node, block) in self.block_nodes.iter().copied().zip(blocks.iter()) {
            let name = match self.program.get(node.0 as usize) {
                Some(Op::Input {
                    name: Some(name), ..
                }) => name.as_str(),
                _ => "",
            };
            let resident = self.resident_nodes.binary_search(&node).is_ok()
                || resident_names.is_some_and(|names| names.contains(name));
            if self.carry.state.is_live(&node) || self.carry.rows.is_live(&node) {
                continue;
            }
            let identity = ResidentIdentity::new(block_address(block), block_length(block))
                .with_generation(if resident { owner_generation(name) } else { 0 });
            if resident && self.arena.resident_sources.get(&node) == Some(&identity) {
                resident_hits += 1;
                continue;
            }
            let cacheable = resident && !name.is_empty();
            if cacheable {
                let packed =
                    !matches!(block, QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_));
                let cached = {
                    let mut ledger = self.driver.resident_buffers.lock();
                    let found = ledger.lookup(name, identity)?;
                    if found
                        .as_ref()
                        .is_some_and(|buffer| matches!(buffer, CudaGraphBuffer::Bytes(_)) != packed)
                    {
                        ledger.release_names([name]);
                        None
                    } else {
                        found
                    }
                };
                if let Some(buffer) = cached {
                    self.arena.buffers.insert(node, buffer);
                    self.arena.resident_sources.insert(node, identity);
                    resident_hits += 1;
                    continue;
                }
            }
            if resident {
                resident_misses += 1;
            }
            let upload_call_started = Instant::now();
            match block {
                QuantizedBlock::Float32(values) => {
                    self.driver
                        .upload_float_input(&mut self.arena, node, values)?
                }
                QuantizedBlock::Int32(values) => {
                    let widened: Vec<f32> = values.iter().map(|&value| value as f32).collect();
                    self.driver
                        .upload_float_input(&mut self.arena, node, &widened)?;
                }
                other => {
                    let Some(codec) = packed_codec(other) else {
                        return Err(CudaDriverError::UnsupportedGraph {
                            node,
                            reason: "unsupported input codec",
                        });
                    };
                    if self.packed_operands.get(&node).copied() != Some(codec) {
                        return Err(CudaDriverError::UnsupportedGraph {
                            node,
                            reason: "input codec differs from planned codec",
                        });
                    }
                    match other {
                        QuantizedBlock::PackedStrided { .. } => self
                            .driver
                            .upload_runs(&mut self.arena, node, other)?,
                        _ => self
                            .driver
                            .upload_bytes(&mut self.arena, node, packed_bytes(other))?,
                    }
                }
            }
            upload_call_us += elapsed_micros(upload_call_started);
            if self.carry.state.is_pair(&node) {
                self.carry.host_bytes += block_length(block) as u64;
            }
            if self.carry.rows.is_pair(&node) {
                self.carry.host_bytes += block_length(block) as u64;
                let capacity_elements = self.arena.buffers.get(&node).map_or(0, buffer_elements);
                self.carry.rows.step(KvEvent::Resident {
                    input: node,
                    valid_rows: self.carry.cached_rows,
                    capacity_elements,
                })?;
                self.arena.resident_sources.remove(&node);
                continue;
            }
            if resident {
                self.arena.resident_sources.insert(node, identity);
                if let (true, Some(buffer)) = (cacheable, self.arena.buffers.get(&node).cloned()) {
                    self.driver
                        .resident_buffers
                        .lock()
                        .insert(name, identity, buffer)?;
                }
            } else {
                self.arena.resident_sources.remove(&node);
            }
        }
        if let Some(started) = upload_started {
            eprintln!(
                "cuda_host_timing: phase=upload_ms value={:.3} resident_hits={} resident_misses={} packed_bytes={} upload_call_ms={:.3}",
                started.elapsed().as_secs_f64() * 1_000.0,
                resident_hits,
                resident_misses,
                self.driver.packed_upload_bytes() - upload_bytes_before,
                upload_call_us as f64 / 1_000.0
            );
        }
        self.begin_rows_step()?;
        let upload_phase_us = elapsed_micros(phase_started);
        let submit_started = Instant::now();
        let trace = std::env::var_os("PROXIMA_CUDA_TRACE").is_some();
        let timing = std::env::var_os("PROXIMA_CUDA_TIMING").is_some();
        let mut timing_rows: Vec<(NodeId, u128, usize, String)> = Vec::new();
        let trace_outputs = std::env::var_os("PROXIMA_CUDA_TRACE_OUTPUTS").is_some();
        let trace_values: Vec<NodeId> = std::env::var("PROXIMA_CUDA_TRACE_VALUES")
            .ok()
            .into_iter()
            .flat_map(|value| value.split(',').map(str::to_owned).collect::<Vec<_>>())
            .filter_map(|value| value.parse::<u32>().ok().map(NodeId))
            .collect();
        let trace_node = std::env::var("PROXIMA_CUDA_TRACE_NODE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(NodeId);
        if trace_outputs {
            eprintln!(
                "cuda_trace_outputs requested_count={} requested_first={:?} resolved_count={} resolved_first={:?} resolved_last={:?}",
                self.outputs.len(),
                self.outputs.first(),
                self.resolved.len(),
                self.resolved.first().map(|bound| bound.node),
                self.resolved.last().map(|bound| bound.node)
            );
        }
        let mut fault_windows: Vec<(NodeId, usize, usize)> = Vec::new();
        for (bound, kernel) in self.resolved.iter().zip(self.kernels.iter()) {
            if kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Fault))
            {
                let offset = fault_windows
                    .last()
                    .map_or(0, |(_, offset, slots)| offset + slots);
                fault_windows.push((bound.node, offset, gather_count(bound)));
            }
        }
        let fault_total = fault_windows
            .last()
            .map_or(0, |(_, offset, slots)| offset + slots);
        let fault_array = (fault_total > 0)
            .then(|| self.driver.runtime.alloc_zeros::<u32>(fault_total))
            .transpose()?;
        let mut event_rows: Vec<(NodeId, String, usize, R::Event)> = Vec::new();
        let event_origin = std::env::var_os("PROXIMA_CUDA_EVENT_TIMING")
            .and_then(|_| try_record_event(&self.driver.runtime));
        let event_timing = event_origin.is_some();
        self.arena.launch_profile = LaunchProfile {
            enabled: host_timing,
            ..LaunchProfile::default()
        };
        let mut kind_rows: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
        let dispatch_started = host_timing.then(std::time::Instant::now);
        let mut function_lookups = 0u64;
        let mut function_lookup_us = 0u64;
        let release_plan = &self.pool.plan;
        for (position, (bound, (kernel, uniforms))) in self
            .resolved
            .iter()
            .zip(self.kernels.iter().zip(self.uniforms.iter()))
            .enumerate()
        {
            let node_started = timing.then(std::time::Instant::now);
            if trace_node == Some(bound.node) {
                eprintln!(
                    "cuda_trace_source_begin node={:?}\n{}\ncuda_trace_source_end",
                    bound.node, kernel.source
                );
            }
            if trace_outputs
                && (self.outputs.contains(&bound.node) || trace_values.contains(&bound.node))
            {
                eprintln!(
                    "cuda_trace_output_kernel node={:?} entry={} source_len={} bindings={:?}",
                    bound.node,
                    kernel.entry,
                    kernel.source.len(),
                    kernel.bindings
                );
            }
            let output_len = device_words(bound);
            if trace {
                eprintln!(
                    "cuda_trace: node={:?} kind={} extents={:?} bindings={:?} grid={} output_len={} uniforms={} detail={}",
                    bound.node,
                    bound.kind.name(),
                    bound.extents,
                    kernel.bindings,
                    kernel.grid.threads(),
                    output_len,
                    uniforms.len(),
                    if trace_node == Some(bound.node) {
                        format!("{bound:?}")
                    } else {
                        String::new()
                    },
                );
            }
            // The emitter already decided this op dispatches nothing: a
            // zero-extent intermediate is a valid tensor, not a CUDA launch.
            if kernel.grid.is_idle() {
                let needs_zero_buffer = !matches!(
                    self.arena.buffers.get(&bound.node),
                    Some(CudaGraphBuffer::F32(buffer)) if R::len(buffer) == 0
                );
                if needs_zero_buffer {
                    self.arena.allocations += 1;
                    self.arena.buffers.insert(
                        bound.node,
                        CudaGraphBuffer::F32(self.driver.runtime.alloc_zeros::<f32>(0)?),
                    );
                }
            } else if let BoundOpKind::Constant { value } = bound.kind {
                self.driver
                    .materialize_constant(&mut self.arena, bound.node, value, output_len)?;
            } else {
                let deferred_fault = fault_array.as_ref().and_then(|array| {
                    fault_windows
                        .iter()
                        .find(|(node, _, _)| *node == bound.node)
                        .map(|(_, offset, slots)| (array, *offset, *slots))
                });
                if self.functions[position].is_none() && !timing {
                    let lookup_started = host_timing.then(Instant::now);
                    self.functions[position] = Some(self.driver.runtime.build_function(kernel)?);
                    function_lookups += 1;
                    function_lookup_us += lookup_started.map_or(0, elapsed_micros);
                }
                if release_plan.pooled.contains(&bound.node) {
                    self.arena.seat_pooled(bound.node, output_len);
                }
                let launch_started = host_timing.then(Instant::now);
                let operand_bytes_before = if op_events.is_some() {
                    self.operand_buffer_bytes(bound)
                } else {
                    0
                };
                let start_event = op_events
                    .as_ref()
                    .and_then(|_| try_record_event(&self.driver.runtime));
                let launch = if timing {
                    self.driver.launch_f32_persistent_sized(
                        &mut self.arena,
                        kernel,
                        uniforms,
                        output_len,
                    )
                } else {
                    self.driver.launch_f32_persistent_sized_async(
                        &mut self.arena,
                        kernel,
                        uniforms,
                        output_len,
                        deferred_fault,
                        self.functions[position].as_ref(),
                    )
                };
                if let Some(started) = launch_started {
                    let row = kind_rows.entry(bound.kind.name()).or_default();
                    row.0 += 1;
                    row.1 += u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                }
                launch.map_err(|error| match error {
                    typed @ CudaDriverError::GatherIndexOutOfRange { .. } => typed,
                    other => CudaDriverError::Execute {
                        node: bound.node,
                        error: other.to_string(),
                    },
                })?;
                if let (Some(events), Some(start)) = (op_events.as_deref_mut(), start_event)
                    && let Some(end) = try_record_event(&self.driver.runtime)
                {
                    events.push(LaunchEvents {
                        position,
                        start,
                        end,
                        operand_bytes: operand_bytes_before,
                    });
                }
                if let Some(started) = node_started {
                    timing_rows.push((
                        bound.node,
                        started.elapsed().as_micros(),
                        output_len,
                        kernel.entry.clone(),
                    ));
                }
                if event_timing && let Some(event) = try_record_event(&self.driver.runtime) {
                    event_rows.push((bound.node, format!("{}:{}", bound.kind.name(), kernel.entry), output_len, event));
                }
            }
            if trace_outputs && self.outputs.contains(&bound.node) {
                let values = self.driver.read_f32(&self.arena, bound.node)?;
                let checksum = values.iter().copied().sum::<f32>();
                let maximum = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
                eprintln!(
                    "cuda_trace_output_after_launch node={:?} len={} checksum={} max_abs={}",
                    bound.node,
                    values.len(),
                    checksum,
                    maximum
                );
            }
            if trace {
                eprintln!("cuda_trace: node={:?} complete", bound.node);
            }
            self.arena.release_to_pool(&release_plan.release[position]);
        }
        if let Some(started) = dispatch_started {
            eprintln!(
                "cuda_host_timing: phase=submit_ms value={:.3} nodes={} function_lookups={} function_lookup_ms={:.3} compilations={} nvrtc_ms={:.3} load_ms={:.3}",
                started.elapsed().as_secs_f64() * 1_000.0,
                self.resolved.len(),
                function_lookups,
                function_lookup_us as f64 / 1_000.0,
                self.driver.kernel_compilations() - compilations_before,
                (self.driver.nvrtc_micros() - nvrtc_before) as f64 / 1_000.0,
                (self.driver.load_micros() - load_before) as f64 / 1_000.0
            );
            let lookup_after = self.driver.runtime.compile_counters().lookup_micros();
            eprintln!(
                "cuda_host_timing: phase=lookup_split build_ms={:.3} key_ms={:.3} disk_read_ms={:.3} module_load_ms={:.3} function_load_ms={:.3}",
                (lookup_after[0] - lookup_before[0]) as f64 / 1_000.0,
                (lookup_after[1] - lookup_before[1]) as f64 / 1_000.0,
                (lookup_after[2] - lookup_before[2]) as f64 / 1_000.0,
                (lookup_after[3] - lookup_before[3]) as f64 / 1_000.0,
                (lookup_after[4] - lookup_before[4]) as f64 / 1_000.0
            );
            report_launch_profile(&self.arena.launch_profile, &kind_rows);
            self.report_arena_footprint();
        }
        self.append_new_rows()?;
        let submit_us = elapsed_micros(submit_started);
        let sync_phase_started = Instant::now();
        let sync_started = host_timing.then(std::time::Instant::now);
        self.driver.runtime.synchronize()?;
        let sync_us = elapsed_micros(sync_phase_started);
        if let Some(array) = &fault_array {
            let faults = self.driver.runtime.download(array)?;
            for (node, offset, slots) in &fault_windows {
                let window = &faults[*offset..*offset + *slots];
                if let Some(slot) = window.iter().position(|value| *value != 0) {
                    let index = i32::from_ne_bytes(window[slot].to_ne_bytes()) as i64;
                    return Err(CudaDriverError::GatherIndexOutOfRange {
                        node: *node,
                        slot,
                        index,
                    });
                }
            }
        }
        if let Some(origin) = event_origin {
            report_event_timing::<R>(&origin, &event_rows)?;
        }
        if let Some(started) = sync_started {
            eprintln!(
                "cuda_host_timing: phase=sync_ms value={:.3}",
                started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        if timing {
            timing_rows.sort_by_key(|row| std::cmp::Reverse(row.1));
            eprintln!(
                "cuda_timing: nodes={} total_sync_ms={:.3}",
                timing_rows.len(),
                timing_rows.iter().map(|row| row.1).sum::<u128>() as f64 / 1000.0
            );
            let top_rows = std::env::var("PROXIMA_CUDA_TIMING_ROWS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(12);
            for (rank, (node, micros, output_len, entry)) in timing_rows.iter().take(top_rows).enumerate() {
                eprintln!(
                    "cuda_timing_top rank={} node={:?} elapsed_ms={:.3} output_len={} entry={}",
                    rank + 1,
                    node,
                    *micros as f64 / 1000.0,
                    output_len,
                    entry
                );
            }
        }
        self.swap_carried_buffers();
        let readback_phase_started = Instant::now();
        let host_readback_started = host_timing.then(std::time::Instant::now);
        let mut results = Vec::with_capacity(self.outputs.len());
        let readback_started = timing.then(std::time::Instant::now);
        for node in &self.outputs {
            if self.carry.state.pairs().any(|(input, output)| output == node && self.carry.state.is_live(input)) {
                continue;
            }
            if self.carry.rows.pairs().any(|pair| pair.output == *node && self.carry.rows.is_live(&pair.input)) {
                continue;
            }
            if trace_outputs {
                let length = self.arena.buffers.get(node).map(|buffer| match buffer {
                    CudaGraphBuffer::F32(values) => R::len(values),
                    CudaGraphBuffer::Bytes(values) => R::len(values.as_ref()),
                });
                eprintln!("cuda_trace_output node={node:?} arena_len={length:?}");
            }
            results.push((
                *node,
                self.shapes.of(*node).to_vec(),
                self.driver.read_f32(&self.arena, *node)?,
            ));
        }
        if let Some(started) = host_readback_started {
            eprintln!(
                "cuda_host_timing: phase=readback_ms value={:.3} outputs={}",
                started.elapsed().as_secs_f64() * 1_000.0,
                self.outputs.len()
            );
        }
        crate::backend::record_stage_phases(crate::backend::StagePhases {
            block_upload_nanos: upload_phase_us.saturating_mul(1_000),
            encode_dispatch_nanos: submit_us.saturating_mul(1_000),
            gpu_exec_nanos: sync_us.saturating_mul(1_000),
            readback_nanos: elapsed_micros(readback_phase_started).saturating_mul(1_000),
        });
        proxima_telemetry::debug!(
            upload_phase_us = upload_phase_us,
            upload_call_us = upload_call_us,
            upload_bytes = self.driver.packed_upload_bytes() - upload_bytes_before,
            resident_hits = resident_hits as u64,
            resident_misses = resident_misses as u64,
            submit_us = submit_us,
            sync_us = sync_us,
            readback_us = elapsed_micros(readback_phase_started),
            compilations = self.driver.kernel_compilations() - compilations_before,
            nvrtc_us = self.driver.nvrtc_micros() - nvrtc_before,
            load_us = self.driver.load_micros() - load_before,
            node_count = self.resolved.len() as u64,
            "cuda execute phases"
        );
        if let Some(started) = readback_started {
            eprintln!(
                "cuda_timing_readback: outputs={} elapsed_ms={:.3}",
                self.outputs.len(),
                started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        Ok(Evaluated::from_parts(
            self.program
                .len()
                .checked_sub(1)
                .map_or(NodeId(0), |index| NodeId(index as u32)),
            results,
            None,
        ))
    }

    fn report_arena_footprint(&self) {
        let mut by_kind: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new();
        let kinds: BTreeMap<NodeId, &str> = self
            .resolved
            .iter()
            .map(|bound| (bound.node, bound.kind.name()))
            .collect();
        let mut total = 0_u64;
        for (node, buffer) in &self.arena.buffers {
            let bytes = match buffer {
                CudaGraphBuffer::F32(values) => R::len(values) as u64 * 4,
                CudaGraphBuffer::Bytes(values) => R::len(values.as_ref()) as u64,
            };
            let kind = kinds.get(node).copied().unwrap_or("input");
            let row = by_kind.entry(kind).or_default();
            row.0 += 1;
            row.1 += bytes;
            row.2 = row.2.max(bytes);
            total += bytes;
        }
        let free = self.driver.memory_info().map_or(0, |(free, _)| free);
        eprintln!(
            "cuda_host_timing: arena buffers={} total_bytes={} resolved_nodes={} free_bytes={}",
            self.arena.buffers.len(),
            total,
            self.resolved.len(),
            free
        );
        for (kind, (count, bytes, largest)) in by_kind {
            eprintln!(
                "cuda_host_timing: arena_kind={kind} buffers={count} bytes={bytes} largest={largest}"
            );
        }
    }

    /// Number of device allocations made by the persistent graph arena.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.arena.allocations()
    }
}

// `Q5_1` is excluded here the same way `Iq4Nl`/`Iq2Xs`/`Iq3Xxs` are --
// no CUDA unpack kernel exists for any of them yet, so all stay routed through
// `None`. Every unsupported codec is named so a new `Codec` variant forces a
// decision here instead of slipping through a wildcard.
fn packed_codec(block: &QuantizedBlock<'_>) -> Option<Codec> {
    crate::msl::codec_from_quantized_block(block)?;
    let (QuantizedBlock::Packed { codec, .. } | QuantizedBlock::PackedStrided { codec, .. }) =
        block
    else {
        return None;
    };
    match *codec {
        Codec::Q2K
        | Codec::Q3K
        | Codec::Q4K
        | Codec::Q5K
        | Codec::Q6K
        | Codec::Q8_0
        | Codec::Q4_0
        | Codec::Q5_0
        | Codec::Q1_0
        | Codec::Q2_0
        | Codec::Pq2_0
        | Codec::Ptq1_0
        | Codec::Float16
        | Codec::BFloat16 => Some(*codec),
        Codec::Q5_1
        | Codec::Q4_1
        | Codec::Q8_1
        | Codec::Q8_1Grouped
        | Codec::Q8K
        | Codec::Iq1S
        | Codec::Iq1M
        | Codec::Iq2Xxs
        | Codec::Iq2Xs
        | Codec::Iq2S
        | Codec::Iq3Xxs
        | Codec::Iq3S
        | Codec::Iq4Nl
        | Codec::Iq4Xs
        | Codec::Tq10
        | Codec::Tq20
        | Codec::Mxfp4
        | Codec::Nvfp4
        | Codec::BFloat8
        | Codec::Bf8E5M2
        | Codec::Bf4E2M1
        | Codec::Fp6E3M2 => None,
        // the list above is the whole wire set as of this commit; a format added later is
        // reported as having no CUDA packed kernel rather than silently admitted
        _ => None,
    }
}

fn packed_bytes<'a>(block: &QuantizedBlock<'a>) -> &'a [u8] {
    match block {
        QuantizedBlock::Float32(_)
        | QuantizedBlock::Int32(_)
        | QuantizedBlock::PackedStrided { .. } => &[],
        QuantizedBlock::Packed { bytes, .. } => packed_codec(block).map_or(&[], |_| bytes),
    }
}

fn block_address(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(values) => values.as_ptr() as usize,
        QuantizedBlock::Int32(values) => values.as_ptr() as usize,
        QuantizedBlock::PackedStrided { bytes, .. } => bytes.as_ptr() as usize,
        QuantizedBlock::Packed { .. } => packed_bytes(block).as_ptr() as usize,
    }
}

fn block_length(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(values) => std::mem::size_of_val(*values),
        QuantizedBlock::Int32(values) => std::mem::size_of_val(*values),
        QuantizedBlock::PackedStrided { .. } => block.packed_len().unwrap_or(0),
        QuantizedBlock::Packed { .. } => packed_bytes(block).len(),
    }
}

fn add_one_kernel(input: NodeId, output: NodeId, elements: usize) -> Result<CudaKernel, CudaDriverError> {
    Ok(CudaKernel {
        source: String::from(
            r#"
        extern "C" __global__ void proxima_add_one(
            const float *in, float *out, const int *u
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < u[0]) out[i] = in[i] + 1.0f;
        }
        "#,
        ),
        entry: String::from("proxima_add_one"),
        bindings: vec![
            Binding::Input(input),
            Binding::Output(output),
            Binding::Uniforms,
        ],
        grid: CudaGridSpec::launch(
            output,
            elements as u64,
            None,
            "add_one",
        )
        .map_err(|_| CudaDriverError::IdleLaunch)?,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
        route_compaction_words: 0,
    })
}

// PTX ISA 7.0 is accepted by the RTX 2070's driver JIT and is old enough to
// remain broadly loadable on CUDA-capable devices. Keep this artifact tiny:
// it is a driver-boundary oracle, not the generated graph-kernel catalogue.
const PRECOMPILED_ADD_ONE_PTX: &str = r#"
.version 7.3
.target sm_52
.address_size 64

.visible .entry proxima_add_one(
    .param .u64 input,
    .param .u64 output,
    .param .u64 uniforms
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u64 %rd3, [uniforms];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    ld.global.u32 %r5, [%rd3];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra $DONE;
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    mov.f32 %f2, 0f3F800000;
    add.f32 %f2, %f1, %f2;
    st.global.f32 [%rd6], %f2;
$DONE:
    ret;
}
"#;

fn encode_kernel(rendered: &RenderedEncode, threads: usize) -> Result<CudaKernel, CudaDriverError> {
    let node = NodeId(0);
    Ok(CudaKernel {
        source: rendered.source.clone(),
        entry: rendered.entry.clone(),
        bindings: vec![Binding::Input(node), Binding::Output(node)],
        grid: CudaGridSpec::launch(node, threads as u64, None, "encode_rows").map_err(|_| CudaDriverError::IdleLaunch)?,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
        route_compaction_words: 0,
    })
}

/// The x-dimension block count of `grid`'s launch, or an error rather than the
/// silent truncation of an `as u32` cast: `emit_cuda` already rejects a grid
/// past `gridDim.x`, so this only fires for a `CudaKernel` built by hand.
fn launch_blocks(grid: &CudaGridSpec) -> Result<u32, CudaDriverError> {
    if grid.is_idle() {
        return Err(CudaDriverError::IdleLaunch);
    }
    u32::try_from(grid.blocks()).map_err(|_| CudaDriverError::GridTooLarge {
        threads: grid.threads(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CudaDriver, CudaDriverError, CudaFunction, PRECOMPILED_ADD_ONE_PTX,
        add_one_kernel, cached_ptx, launch_blocks, packed_bytes, packed_codec,
        resolve_cuda_include_dir, store_ptx, validate_cuda_dtype,
    };
    use crate::cuda::CudaGridSpec;
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use crate::gpu_runtime::GpuRuntime;
    use crate::cuda_kernel_cache::CacheKey;
    use crate::cuda_ptx_store::PtxStore;
    use crate::lowering::Codec;
    use cudarc::nvrtc::{CompileOptions, Ptx};
    use proxima_numeric::block_layout::BlockLayout;
    use super::PoolSchedule;
    use proxima_tensor::{BoundOp, BoundOpKind, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, bind_with_fusion, infer, map};
    use proxima_numeric::{DType, EncodeRule, FormatCodec, NumericPolicy, ScaleArithmetic};

    const KERNEL_CACHE_DIR_VAR: &str = "PROXIMA_CUDA_KERNEL_CACHE_DIR";

    const CUDA_UNPACKABLE: [Codec; 18] = [
        Codec::Q5_1,
        Codec::Q4_1,
        Codec::Q8_1,
        Codec::Q8_1Grouped,
        Codec::Q8K,
        Codec::Iq1S,
        Codec::Iq1M,
        Codec::Iq2Xxs,
        Codec::Iq2Xs,
        Codec::Iq2S,
        Codec::Iq3Xxs,
        Codec::Iq3S,
        Codec::Iq4Nl,
        Codec::Iq4Xs,
        Codec::Tq10,
        Codec::Tq20,
        Codec::Mxfp4,
        Codec::Nvfp4,
    ];

    const CODECS_WITH_CUDA_KERNEL: [Codec; 14] = [
        Codec::Q2K,
        Codec::Q3K,
        Codec::Q4K,
        Codec::Q5K,
        Codec::Q6K,
        Codec::Q8_0,
        Codec::Q4_0,
        Codec::Q5_0,
        Codec::Q1_0,
        Codec::Q2_0,
        Codec::Pq2_0,
        Codec::Ptq1_0,
        Codec::Float16,
        Codec::BFloat16,
    ];

    #[test]
    fn codecs_with_a_cuda_unpack_kernel_keep_their_bytes() {
        let weights = [0x5au8; 64];

        for codec in CODECS_WITH_CUDA_KERNEL {
            let block = QuantizedBlock::Packed {
                codec,
                bytes: &weights,
            };

            assert_eq!(packed_codec(&block), Some(codec), "{codec:?}");
            assert_eq!(packed_bytes(&block), &weights, "{codec:?}");
        }
    }

    #[test]
    fn codecs_without_a_cuda_unpack_kernel_bind_nothing() {
        let weights = [0x5au8; 64];

        for codec in CUDA_UNPACKABLE {
            let block = QuantizedBlock::Packed {
                codec,
                bytes: &weights,
            };

            assert_eq!(packed_codec(&block), None, "{codec:?}");
            assert!(packed_bytes(&block).is_empty(), "{codec:?}");
        }
    }

    const QUANTIZE_ROWS: usize = 3;
    const QUANTIZE_WIDTH: usize = 512;

    fn activation_values() -> Vec<f32> {
        let mut random = proxima_tensor::test_support::Lcg(0xfeed);
        (0..QUANTIZE_ROWS * QUANTIZE_WIDTH)
            .map(|index| random.next_unit() * (1.0 + (index % 7) as f32))
            .collect()
    }

    fn quantize_on_gpu(driver: &CudaDriver, layout: BlockLayout, values: &[f32], arithmetic: ScaleArithmetic) -> Vec<u8> {
        let bytes = (QUANTIZE_WIDTH / usize::from(layout.elements) * usize::from(layout.block_bytes)) as u64;
        let strides = [QUANTIZE_WIDTH as i64, 1].into_iter().collect();
        let bound = BoundOp {
            node: NodeId(1),
            dtype: DType::UINT8,
            extents: vec![QUANTIZE_ROWS as u64, bytes],
            kind: BoundOpKind::BlockQuantize {
                operands: vec![(NodeId(0), proxima_tensor::Layout { base: 0, strides }, None)],
                layout: Box::new(layout),
                arithmetic,
            },
        };
        let kernel = crate::cuda::emit_cuda_with_policy(&bound, &Default::default(), NumericPolicy::default())
            .expect("a q8_1 run layout emits a quantize kernel");
        let groups = (values.len() / 32) as i64;
        let words = driver
            .launch_f32(&kernel, &[values], &groups.to_le_bytes())
            .expect("the quantize kernel launches");
        let written = (bytes as usize) * QUANTIZE_ROWS;
        words.iter().flat_map(|word| word.to_ne_bytes()).take(written).collect()
    }

    fn cpu_quantize(layout: &BlockLayout, values: &[f32]) -> Vec<u8> {
        let blocks = values.len() / usize::from(layout.elements);
        let mut packed = vec![0u8; blocks * usize::from(layout.block_bytes)];
        proxima_gguf::quant::block_layout_encode::quantize(layout, values, &mut packed).expect("the cpu encoder takes the layout");
        packed
    }

    fn regrouped(single: &[u8]) -> Vec<u8> {
        single
            .chunks(36 * 4)
            .flat_map(|group| {
                let headers = group.chunks(36).flat_map(|run| run[..4].to_vec());
                let codes = group.chunks(36).flat_map(|run| run[4..].to_vec());
                headers.chain(codes).collect::<Vec<u8>>()
            })
            .collect()
    }

    /// llama's CUDA quantizer divides by d and its CPU quantizer multiplies by 1/d, so the two encoders
    /// differ exactly where those two f32 roundings disagree.
    #[test]
    fn gpu_and_cpu_grouped_quantize_each_match_their_llama_rounding_exactly() {
        let driver = CudaDriver::new(0).expect("a cuda device");
        let values = activation_values();
        let single = quantize_on_gpu(&driver, BlockLayout::Q8_1, &values, ScaleArithmetic::Quotient);
        let grouped = quantize_on_gpu(&driver, BlockLayout::Q8_1_GROUPED, &values, ScaleArithmetic::Quotient);
        assert_eq!(grouped, regrouped(&single), "each warp writes its own header word and codes");

        let cpu = cpu_quantize(&BlockLayout::Q8_1_GROUPED, &values);
        assert_eq!(grouped.len(), cpu.len());
        for (block, (gpu_block, cpu_block)) in grouped.chunks(144).zip(cpu.chunks(144)).enumerate() {
            for run in 0..4 {
                let start = (block * 4 + run) * 32;
                let run_values = &values[start..start + 32];
                let amax = run_values.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
                let scale = amax / 127.0;
                let scale_bits = half::f16::from_f32(scale).to_bits().to_le_bytes();
                let codes = 16 + run * 32..16 + run * 32 + 32;
                let mut gpu_sum = 0i32;
                let mut cpu_sum = 0i32;
                for (offset, value) in run_values.iter().enumerate() {
                    let (gpu_expected, cpu_expected) = if amax == 0.0 {
                        (0.0, 0.0)
                    } else {
                        ((value / scale).round(), (value * (1.0 / scale)).round())
                    };
                    let gpu_code = gpu_block[codes.start + offset] as i8;
                    let cpu_code = cpu_block[codes.start + offset] as i8;
                    assert_eq!(i32::from(gpu_code), gpu_expected as i32, "block {block} run {run} element {offset}: gpu code is roundf(x / d)");
                    assert_eq!(i32::from(cpu_code), cpu_expected as i32, "block {block} run {run} element {offset}: cpu code is roundf(x * (1 / d))");
                    gpu_sum += i32::from(gpu_code);
                    cpu_sum += i32::from(cpu_code);
                }
                let header = run * 4..run * 4 + 4;
                for (side, block_bytes, sum) in [("gpu", gpu_block, gpu_sum), ("cpu", cpu_block, cpu_sum)] {
                    let stored_sum = half::f16::from_f32(scale * sum as f32).to_bits().to_le_bytes();
                    assert_eq!(block_bytes[header.start..header.start + 2], scale_bits, "block {block} run {run}: {side} f16 scale");
                    assert_eq!(block_bytes[header.start + 2..header.end], stored_sum, "block {block} run {run}: {side} stored sum is f16(d * sum of its codes)");
                }
            }
        }
    }

    const NEAR_TIE_UNIT_PEAK_BITS: [u32; 6] = [0x3d11_2244, 0x3d31_62c5, 0x3d51_a346, 0x3d71_e3c7, 0x3f30_60c1, 0x3f50_a142];
    const NEAR_TIE_UNIT_PEAK_RECIPROCAL_CODES: [i8; 6] = [4, 5, 6, 7, 87, 103];
    const NEAR_TIE_UNIT_PEAK_QUOTIENT_CODES: [i8; 6] = [5, 6, 7, 8, 88, 104];
    const NEAR_TIE_WIDE_PEAK_BITS: [u32; 6] = [0x3d32_ff98, 0x3d95_2a54, 0x3dd0_d4dc, 0x3e06_3fb2, 0x3e24_14f6, 0x3e41_ea3a];
    const NEAR_TIE_WIDE_PEAK_RECIPROCAL_CODES: [i8; 6] = [2, 3, 4, 5, 6, 7];
    const NEAR_TIE_WIDE_PEAK_QUOTIENT_CODES: [i8; 6] = [1, 2, 3, 4, 5, 6];

    fn near_tie_values() -> Vec<f32> {
        let mut values = activation_values();
        values[..64].fill(0.0);
        values[0] = 1.0;
        values[32] = 3.7;
        for (first, ties) in [(1, NEAR_TIE_UNIT_PEAK_BITS), (33, NEAR_TIE_WIDE_PEAK_BITS)] {
            for (lane, bits) in ties.iter().enumerate() {
                values[first + lane] = f32::from_bits(*bits);
            }
        }
        values
    }

    /// The host loop of `block_codec` run under the same bound arithmetic the device kernel renders.
    fn host_quantize(layout: BlockLayout, values: &[f32], arithmetic: ScaleArithmetic) -> Vec<u8> {
        let rule = EncodeRule { arithmetic, ..layout.encode_rule().expect("q8_1 has a rule") };
        let codec = FormatCodec::block_with_rule(&layout, rule).expect("q8_1 runs under an integer rule");
        let mut out = vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
        codec.encode(values, &mut out).expect("the host encodes the groups");
        out
    }

    fn codes_at(packed: &[u8], group: usize, lanes: std::ops::Range<usize>) -> Vec<i8> {
        packed[group * 36 + 4..group * 36 + 36][lanes].iter().map(|byte| *byte as i8).collect()
    }

    #[test]
    fn gpu_reciprocal_multiply_quantize_equals_the_llama_mmq_port_on_near_ties_and_quotient_does_not() {
        let driver = CudaDriver::new(0).expect("a cuda device");
        let values = near_tie_values();
        let reciprocal = quantize_on_gpu(&driver, BlockLayout::Q8_1, &values, ScaleArithmetic::ReciprocalMultiply);
        let quotient = quantize_on_gpu(&driver, BlockLayout::Q8_1, &values, ScaleArithmetic::Quotient);
        assert_eq!(reciprocal, host_quantize(BlockLayout::Q8_1, &values, ScaleArithmetic::ReciprocalMultiply), "device reciprocal-multiply bytes against block_codec under the same rule");
        assert_eq!(quotient, host_quantize(BlockLayout::Q8_1, &values, ScaleArithmetic::Quotient), "device quotient bytes against block_codec under the same rule");
        assert_eq!(codes_at(&reciprocal, 0, 1..7), NEAR_TIE_UNIT_PEAK_RECIPROCAL_CODES);
        assert_eq!(codes_at(&reciprocal, 1, 1..7), NEAR_TIE_WIDE_PEAK_RECIPROCAL_CODES);
        assert_eq!(codes_at(&quotient, 0, 1..7), NEAR_TIE_UNIT_PEAK_QUOTIENT_CODES);
        assert_eq!(codes_at(&quotient, 1, 1..7), NEAR_TIE_WIDE_PEAK_QUOTIENT_CODES);
        assert_ne!(reciprocal, quotient);
    }

    #[test]
    fn gpu_quantize_with_no_configuration_equals_the_layout_rule_on_near_ties() {
        let driver = CudaDriver::new(0).expect("a cuda device");
        let values = near_tie_values();
        let own = BlockLayout::Q8_1.encode_rule().expect("q8_1 has a rule").arithmetic;
        let default = ScaleArithmetic::for_rows(1, NumericPolicy::bit_exact());
        assert_eq!(default, own);
        assert_eq!(quantize_on_gpu(&driver, BlockLayout::Q8_1, &values, default), host_quantize(BlockLayout::Q8_1, &values, own));
    }

    #[test]
    fn gpu_reciprocal_multiply_grouped_quantize_is_four_flat_runs_behind_one_header() {
        let driver = CudaDriver::new(0).expect("a cuda device");
        let values = near_tie_values();
        let flat = quantize_on_gpu(&driver, BlockLayout::Q8_1, &values, ScaleArithmetic::ReciprocalMultiply);
        let grouped = quantize_on_gpu(&driver, BlockLayout::Q8_1_GROUPED, &values, ScaleArithmetic::ReciprocalMultiply);
        assert_eq!(grouped, regrouped(&flat));
    }

    #[test]
    fn cuda_idle_grid_is_refused_before_any_device_call() {
        let grid = CudaGridSpec::idle();
        assert!(matches!(launch_blocks(&grid), Err(CudaDriverError::IdleLaunch)));
    }

    #[test]
    fn launch_blocks_covers_a_grid_at_the_default_block_width() {
        let grid = CudaGridSpec::launch(NodeId(1), 1_000_000, None, "test").expect("nonzero threads");

        assert!(matches!(launch_blocks(&grid), Ok(3907)));
    }

    #[test]
    fn launch_blocks_rejects_a_grid_past_the_launch_index_instead_of_truncating() {
        let threads = (u64::from(u32::MAX) + 1) * 256;
        let grid = CudaGridSpec::launch(NodeId(1), threads, None, "test").expect("nonzero threads");

        assert!(matches!(
            launch_blocks(&grid),
            Err(CudaDriverError::GridTooLarge { threads: rejected }) if rejected == threads
        ));
    }

    #[test]
    fn non_f32_bound_op_reports_node_and_dtype() {
        let bound = BoundOp {
            node: NodeId(157),
            dtype: DType::FLOAT16,
            extents: vec![1],
            kind: BoundOpKind::Constant { value: 0.0 },
        };

        assert!(
            matches!(
                validate_cuda_dtype(&bound),
                Err(CudaDriverError::UnsupportedDtype {
                    node: NodeId(157),
                    dtype: DType::FLOAT16,
                })
            ),
            "half output must be rejected"
        );
    }

    #[test]
    fn integer_bound_ops_are_accepted_as_integer_valued_f32() {
        for dtype in [
            DType::Bool,
            DType::INT8,
            DType::UINT8,
            DType::INT32,
            DType::UINT32,
        ] {
            let bound = BoundOp {
                node: NodeId(109),
                dtype,
                extents: vec![1],
                kind: BoundOpKind::Constant { value: 0.0 },
            };

            assert!(
                validate_cuda_dtype(&bound).is_ok(),
                "{dtype:?} must be accepted"
            );
        }
    }

    const ALTERNATIONS: u64 = 8;
    const INPUT: [f32; 3] = [1.0, 2.0, 3.0];

    fn add_two_kernel() -> crate::cuda::CudaKernel {
        let mut kernel = add_one_kernel(NodeId(0), NodeId(1), INPUT.len())
            .expect("the add-one probe has three elements");
        kernel.source = kernel.source.replace("+ 1.0f", "+ 2.0f");
        kernel
    }

    fn run_ptx(driver: &CudaDriver, function: &CudaFunction) -> Vec<f32> {
        let device_input = driver.runtime.stream().clone_htod(&INPUT).expect("upload input");
        let mut device_output = driver
            .runtime.stream()
            .alloc_zeros::<f32>(INPUT.len())
            .expect("alloc output");
        let device_uniforms = driver
            .runtime.stream()
            .clone_htod(&3u32.to_ne_bytes())
            .expect("upload uniforms");
        let mut args = driver.runtime.stream().launch_builder(function);
        args.arg(&device_input);
        args.arg(&mut device_output);
        args.arg(&device_uniforms);
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }.expect("ptx launch");
        driver.runtime.stream().synchronize().expect("ptx sync");
        driver
            .runtime.stream()
            .clone_dtoh(&device_output)
            .expect("ptx readback")
    }

    #[test]
    fn nvrtc_sources_sharing_an_entry_name_compile_once_each_and_keep_their_own_function() {
        let cache = tempfile::tempdir().expect("tempdir");
        let driver = temp_env::with_var(KERNEL_CACHE_DIR_VAR, Some(cache.path()), || {
            CudaDriver::new(0)
        })
        .expect("a cuda device is required for the module cache gate");
        let add_one = add_one_kernel(NodeId(0), NodeId(1), INPUT.len())
            .expect("the add-one probe has three elements");
        let add_two = add_two_kernel();
        assert_eq!(
            add_one.entry, add_two.entry,
            "the gate needs a shared entry name"
        );
        assert_ne!(
            add_one.source, add_two.source,
            "the gate needs distinct sources"
        );
        let uniforms = 3i32.to_ne_bytes();

        for _ in 0..ALTERNATIONS {
            let first = driver
                .launch_f32(&add_one, &[&INPUT], &uniforms)
                .expect("add_one launch");
            let second = driver
                .launch_f32(&add_two, &[&INPUT], &uniforms)
                .expect("add_two launch");

            assert_eq!(
                first,
                [2.0, 3.0, 4.0],
                "an entry-name hit returned the wrong source's function"
            );
            assert_eq!(
                second,
                [3.0, 4.0, 5.0],
                "an entry-name hit returned the wrong source's function"
            );
        }

        assert_eq!(
            driver.kernel_compilations(),
            2,
            "one compile per distinct source, not per alternation"
        );
    }

    #[test]
    fn a_second_runtime_sharing_the_cache_dir_loads_the_first_runtimes_kernel_without_nvrtc() {
        let cache = tempfile::tempdir().expect("tempdir");
        let runtime = || {
            temp_env::with_var(KERNEL_CACHE_DIR_VAR, Some(cache.path()), || {
                CudaDriver::new(0)
            })
            .expect("a cuda device is required for the persistent kernel cache gate")
        };
        let kernel = add_one_kernel(NodeId(0), NodeId(1), INPUT.len())
            .expect("the add-one probe has three elements");

        let first = runtime();
        let first_function = first.runtime.build_function(&kernel).expect("first runtime build");
        assert_eq!(first.kernel_compilations(), 1, "the first runtime compiles the kernel once");
        assert_eq!(first.disk_cache_hits(), 0, "the first runtime finds an empty cache");

        let second = runtime();
        let second_function = second.runtime.build_function(&kernel).expect("second runtime build");
        assert_eq!(
            second.kernel_compilations(),
            0,
            "the second runtime must load the stored image, not call nvrtc"
        );
        assert_eq!(second.disk_cache_hits(), 1, "the second runtime reads one stored entry");
        assert_eq!(
            run_ptx(&second, &second_function),
            run_ptx(&first, &first_function),
            "a cached image computes what the compiled one did"
        );
    }

    #[test]
    fn precompiled_ptx_sources_sharing_a_module_key_load_once_each_and_keep_their_own_function() {
        let driver =
            CudaDriver::new(0).expect("a cuda device is required for the module cache gate");
        let add_two_ptx = PRECOMPILED_ADD_ONE_PTX.replace("0f3F800000", "0f40000000");
        assert_ne!(
            add_two_ptx, PRECOMPILED_ADD_ONE_PTX,
            "the gate needs distinct sources"
        );

        for _ in 0..ALTERNATIONS {
            let first = driver
                .precompiled_function("smoke_add_one", "proxima_add_one", PRECOMPILED_ADD_ONE_PTX)
                .expect("add_one ptx");
            let second = driver
                .precompiled_function("smoke_add_one", "proxima_add_one", &add_two_ptx)
                .expect("add_two ptx");

            assert_eq!(
                run_ptx(&driver, &first),
                [2.0, 3.0, 4.0],
                "an entry-name hit returned the wrong module's function"
            );
            assert_eq!(
                run_ptx(&driver, &second),
                [3.0, 4.0, 5.0],
                "an entry-name hit returned the wrong module's function"
            );
        }

        assert_eq!(
            driver.kernel_compilations(),
            2,
            "one load per distinct ptx source, not per alternation"
        );
    }

    #[test]
    fn kernel_cache_key_is_stable_for_identical_inputs() {
        let header = b"#define HALF 1";
        let key = |source: &str| {
            CacheKey::new(
                source,
                Some(header),
                (13, 0),
                &CompileOptions::default(),
                (8, 9),
            )
        };
        assert_eq!(
            key("__global__ void add_one() {}"),
            key("__global__ void add_one() {}"),
            "identical kernel inputs must map to one cache entry"
        );
        assert_ne!(
            key("__global__ void add_one() {}"),
            key("__global__ void add_two() {}"),
            "a source change must map to a different cache entry"
        );
    }

    #[test]
    fn cached_ptx_returns_the_stored_text_and_none_for_a_corrupt_entry() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = PtxStore::Directory(root.path().to_path_buf());
        let key = CacheKey::new(
            "__global__ void add_one() {}",
            None,
            (13, 0),
            &CompileOptions::default(),
            (8, 9),
        );
        assert_eq!(cached_ptx(&store, &key), None, "an empty store is a miss");

        store_ptx(&store, &key, &Ptx::from_src(PRECOMPILED_ADD_ONE_PTX));
        assert_eq!(
            cached_ptx(&store, &key).as_deref(),
            Some(PRECOMPILED_ADD_ONE_PTX),
            "a stored entry must return the exact ptx text"
        );

        std::fs::write(root.path().join(key.to_hex()), b"PXKC truncated")
            .expect("corrupt the entry");
        assert_eq!(
            cached_ptx(&store, &key),
            None,
            "a corrupt entry must be a miss, never a load"
        );
    }

    fn toolkit_with_header(root: &std::path::Path) -> std::path::PathBuf {
        let include = root.join("include");
        std::fs::create_dir_all(&include).expect("create toolkit include dir");
        std::fs::write(include.join("cuda_fp16.h"), b"// header").expect("write header");
        include
    }

    fn resolve(system: &[&str]) -> Result<std::path::PathBuf, CudaDriverError> {
        resolve_cuda_include_dir("cuda_fp16.h", system)
    }

    #[test]
    fn explicit_proxima_path_wins_over_cuda_path() {
        let proxima = tempfile::tempdir().expect("tempdir");
        let cuda = tempfile::tempdir().expect("tempdir");
        let proxima_include = toolkit_with_header(proxima.path());
        toolkit_with_header(cuda.path());
        temp_env::with_vars(
            [
                (
                    "PROXIMA_CUDA_INCLUDE_PATH",
                    Some(proxima_include.as_os_str()),
                ),
                ("CUDA_PATH", Some(cuda.path().as_os_str())),
                ("CUDA_HOME", None),
            ],
            || assert_eq!(resolve(&[]).expect("resolves"), proxima_include),
        );
    }

    #[test]
    fn cuda_home_is_used_when_cuda_path_lacks_the_header() {
        let empty = tempfile::tempdir().expect("tempdir");
        let home = tempfile::tempdir().expect("tempdir");
        let home_include = toolkit_with_header(home.path());
        temp_env::with_vars(
            [
                ("PROXIMA_CUDA_INCLUDE_PATH", None),
                ("CUDA_PATH", Some(empty.path().as_os_str())),
                ("CUDA_HOME", Some(home.path().as_os_str())),
            ],
            || assert_eq!(resolve(&[]).expect("resolves"), home_include),
        );
    }

    #[test]
    fn system_dir_is_the_last_resort() {
        let system = tempfile::tempdir().expect("tempdir");
        let system_include = toolkit_with_header(system.path());
        let system_str = system_include.to_string_lossy().into_owned();
        temp_env::with_vars(
            [
                ("PROXIMA_CUDA_INCLUDE_PATH", None::<&str>),
                ("CUDA_PATH", None),
                ("CUDA_HOME", None),
            ],
            || assert_eq!(resolve(&[&system_str]).expect("resolves"), system_include),
        );
    }

    #[test]
    fn missing_header_error_names_every_path_tried() {
        let empty = tempfile::tempdir().expect("tempdir");
        let home = tempfile::tempdir().expect("tempdir");
        temp_env::with_vars(
            [
                ("PROXIMA_CUDA_INCLUDE_PATH", Some(empty.path().as_os_str())),
                ("CUDA_PATH", Some(empty.path().as_os_str())),
                ("CUDA_HOME", Some(home.path().as_os_str())),
            ],
            || {
                let message = resolve(&["/nonexistent/cuda/include"])
                    .expect_err("no dir holds the header")
                    .to_string();
                assert!(message.contains("cuda_fp16.h"), "{message}");
                assert!(
                    message.contains(&empty.path().display().to_string()),
                    "{message}"
                );
                assert!(
                    message.contains(&home.path().join("include").display().to_string()),
                    "{message}"
                );
                assert!(message.contains("/nonexistent/cuda/include"), "{message}");
            },
        );
    }

    type RelengthFixture = (Vec<BoundOp>, Vec<crate::cuda::CudaKernel>, Vec<NodeId>);

    fn relength_fixture(rows: u64) -> Result<RelengthFixture, String> {
        let mut program = Vec::new();
        let input = |shape: Vec<Extent>| Op::Input { dtype: DType::FLOAT32, shape, name: None };
        let lhs = append(&mut program, input(vec![Extent::Symbolic(0), Extent::Static(8)]));
        let rhs = append(&mut program, input(vec![Extent::Static(8), Extent::Static(4)]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = infer(&program, &[rows]).map_err(|error| format!("infer: {error:?}"))?;
        let bound = bind_with_fusion(&program, &shapes, &[sum], true, NumericPolicy::bit_exact())
            .map_err(|error| format!("bind: {error:?}"))?;
        let kernels = bound
            .iter()
            .map(|op| crate::cuda::emit_cuda(op, &Default::default()).map_err(|error| format!("emit: {error:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((bound, kernels, vec![sum]))
    }

    #[test]
    fn relength_gives_the_schedule_a_rebuild_at_the_new_rows_would() -> Result<(), String> {
        let (small, kernels, outputs) = relength_fixture(16)?;
        let (large, _, _) = relength_fixture(64)?;
        let mut moved = PoolSchedule::build(&small, &kernels, &outputs);

        moved.relength(&large);

        let rebuilt = PoolSchedule::build(&large, &kernels, &outputs);
        assert_eq!(moved.peak, rebuilt.peak);
        assert_eq!(format!("{:?}", moved.plan), format!("{:?}", rebuilt.plan));
        Ok(())
    }

    #[test]
    fn every_cuda_encode_the_neutral_renderer_accepts_compiles_with_nvrtc() {
        use crate::cuda::target::Target;
        use crate::cuda_encode::CudaEncode;
        use crate::encode_render::format_encode_kernel;

        let dialect = CudaEncode { target: Target::CUDA };
        let mut compiled = 0usize;
        for format in proxima_numeric::NumericFormat::every() {
            let Ok(rendered) = format_encode_kernel(&dialect, &format, "probe_encode") else { continue };
            let header_dir = super::cuda_header_dir(&rendered.source).expect("a cuda include dir");
            super::compile_ptx_with_opts(&rendered.source, super::nvrtc_options(header_dir.as_deref()))
                .unwrap_or_else(|error| panic!("{format:?} does not compile: {error}\n{}", rendered.source));
            compiled += 1;
        }
        assert!(compiled > 0, "no encode kernel was compiled");
    }

    #[test]
    fn every_cuda_encode_matches_the_host_encoder_byte_for_byte_at_an_offset() {
        use crate::cuda::target::Target;
        use crate::cuda_encode::CudaEncode;
        use crate::encode_render::{EncodeRows, format_encode_kernel};
        use proxima_numeric::{FormatCodec, NumericFormat};

        let driver = CudaDriver::new(0).expect("a cuda device");
        let dialect = CudaEncode { target: Target::CUDA };
        let guard_byte = u8::MAX;
        let offset = core::mem::size_of::<u64>();
        let mut covered = Vec::new();
        for format in NumericFormat::every() {
            let Ok(rendered) = format_encode_kernel(&dialect, &format, "probe_encode") else { continue };
            let NumericFormat::Scalar(layout) = &format else { panic!("{format:?} rendered without a scalar layout") };
            let maximum = layout.max_normal();
            let half_ulp = (0..=layout.mantissa_bits).fold(1.0, |scale, _| scale * 0.5);
            let infinity = maximum / (maximum * 0.0);
            let nan = infinity - infinity;
            let magnitudes = [
                maximum * 0.0,
                layout.min_subnormal(),
                layout.min_subnormal() * 0.5,
                layout.min_subnormal() * 1.5,
                layout.min_normal(),
                1.0,
                1.0 + half_ulp,
                1.0 + 3.0 * half_ulp,
                2.0 + 2.0 * half_ulp,
                0.1,
                maximum / 3.0,
                maximum,
                maximum * 1.5,
                maximum * 2.0,
                maximum * maximum * maximum * maximum,
                infinity,
            ];
            let mut values: Vec<_> = magnitudes.iter().flat_map(|value| [*value, -*value]).collect();
            values.extend([nan, -nan]);
            while values.len() % rendered.elements_per_thread != 0 {
                values.push(1.0);
            }
            let codec = FormatCodec::for_format(&format).expect("a rendered format resolves");
            let mut expected = vec![0u8; codec.packed_len(values.len()).expect("whole elements")];
            codec.encode(&values, &mut expected).expect("the host encodes");

            let rows = EncodeRows { element_count: values.len(), byte_offset: offset };
            let source = driver.runtime.stream().clone_htod(&values).expect("upload rows");
            let mut destination = driver.runtime.stream().clone_htod(&vec![guard_byte; offset + expected.len() + offset]).expect("upload guard");
            driver.encode_rows(&rendered, &format, &source, &mut destination, rows).expect("the encode launches");
            driver.runtime.stream().synchronize().expect("the encode completes");
            let device = driver.runtime.stream().clone_dtoh(&destination).expect("read back");

            assert!(device[..offset].iter().all(|byte| *byte == guard_byte), "{format:?}: bytes before the offset changed");
            assert!(device[offset + expected.len()..].iter().all(|byte| *byte == guard_byte), "{format:?}: bytes past the range changed");
            let width = rendered.bytes_per_thread;
            for (index, (device_code, host_code)) in device[offset..offset + expected.len()].chunks(width).zip(expected.chunks(width)).enumerate() {
                let first = index * rendered.elements_per_thread;
                assert_eq!(device_code, host_code, "{format:?} element {first}: input bits {:#010x}, device {device_code:?}, host {host_code:?}", values[first].to_bits());
            }
            covered.push(format);
        }
        eprintln!("device encode covered {covered:?}");
        assert!(!covered.is_empty(), "no format was encoded on the device");
    }

    fn carry_cache_bytes<Row: cudarc::driver::DeviceRepr>(driver: &CudaDriver, cache: &cudarc::driver::CudaSlice<Row>) -> Vec<u8> {
        let bytes = cache.len() * core::mem::size_of::<Row>();
        let view = unsafe { cache.transmute::<u8>(bytes) }.expect("the cache reads as bytes");
        driver.runtime.stream().clone_dtoh(&view).expect("read the cache back")
    }

    fn carry_sentinel_cache<Row: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        driver: &CudaDriver,
        bytes: usize,
    ) -> cudarc::driver::CudaSlice<Row> {
        let words = bytes.div_ceil(core::mem::size_of::<Row>());
        let mut cache = driver.runtime.stream().alloc_zeros::<Row>(words).expect("allocate the sentinel cache");
        let total = words * core::mem::size_of::<Row>();
        let mut view = unsafe { cache.transmute_mut::<u8>(total) }.expect("the cache writes as bytes");
        driver.runtime.stream().memcpy_htod(&vec![u8::MAX; total], &mut view).expect("fill the sentinel");
        cache
    }

    #[test]
    fn carrying_graph_output_rows_leaves_the_bytes_of_the_old_device_copy() {
        use super::kv_encode;
        use proxima_numeric::NumericFormat;

        let driver = CudaDriver::new(0).expect("a cuda device");
        let format = NumericFormat::default();
        let encode = kv_encode(driver.runtime.emit_target(), &format).expect("the graph output format renders an encode");
        let NumericFormat::Scalar(layout) = &format else { panic!("the graph output format is a scalar") };
        let maximum = layout.max_normal();
        let infinity = maximum / (maximum * 0.0);
        let nan = infinity - infinity;
        let rows = [maximum * 0.0, -maximum * 0.0, 1.0, 0.1, maximum, layout.min_subnormal() * 0.5, infinity, -infinity, nan, -nan];
        let source = driver.runtime.stream().clone_htod(&rows).expect("upload rows");
        let offset = 3;
        let filler = vec![rows[2] * 7.5; offset + rows.len() + offset];
        let mut encoded = driver.runtime.stream().clone_htod(&filler).expect("upload cache");
        let mut copied = driver.runtime.stream().clone_htod(&filler).expect("upload cache");
        driver.encode_into_cache(&encode, &source, &mut encoded, offset, rows.len()).expect("the carry encodes");
        driver.runtime.stream().memcpy_dtod(&source.slice(0..rows.len()), &mut copied.slice_mut(offset..offset + rows.len())).expect("the old copy");
        driver.runtime.stream().synchronize().expect("the carry completes");
        assert_eq!(carry_cache_bytes(&driver, &encoded), carry_cache_bytes(&driver, &copied));
    }

    #[test]
    fn carrying_into_a_narrow_cache_format_writes_the_host_codec_bytes_at_the_row_offset() {
        use super::kv_encode;
        use proxima_numeric::{FormatCodec, NumericFormat};

        let driver = CudaDriver::new(0).expect("a cuda device");
        let mut covered = Vec::new();
        for format in NumericFormat::every() {
            let Some(encode) = kv_encode(driver.runtime.emit_target(), &format) else { continue };
            if format.is_identity_of(&NumericFormat::default()) {
                continue;
            }
            let NumericFormat::Scalar(layout) = &format else { continue };
            let maximum = layout.max_normal();
            let mut values = vec![1.0, -1.0, 0.1, maximum / 3.0, maximum, -maximum * 2.0, layout.min_subnormal(), maximum * 0.0];
            while values.len() % encode.rendered.elements_per_thread != 0 {
                values.push(0.5);
            }
            let codec = FormatCodec::for_format(&format).expect("a rendered format resolves");
            let mut expected = vec![0u8; codec.packed_len(values.len()).expect("whole elements")];
            codec.encode(&values, &mut expected).expect("the host encodes");
            let skipped = encode.rendered.elements_per_thread * 2;
            let byte_offset = format.packed_len(skipped).expect("whole blocks");
            let source = driver.runtime.stream().clone_htod(&values).expect("upload rows");
            let mut cache = carry_sentinel_cache(&driver, byte_offset + expected.len() + byte_offset);
            let before = carry_cache_bytes(&driver, &cache);
            driver.encode_into_cache(&encode, &source, &mut cache, skipped, values.len()).expect("the carry encodes");
            driver.runtime.stream().synchronize().expect("the carry completes");
            let after = carry_cache_bytes(&driver, &cache);
            assert_eq!(after[byte_offset..byte_offset + expected.len()], expected[..], "{format:?}: the rows are not the host codec's bytes");
            assert_eq!(after[..byte_offset], before[..byte_offset], "{format:?}: rows before the offset changed");
            assert_eq!(after[byte_offset + expected.len()..], before[byte_offset + expected.len()..], "{format:?}: rows past the range changed");
            covered.push(format);
        }
        assert!(!covered.is_empty(), "no narrow format was carried");
    }
}

// ---- ROCm runtime boundary -------------------------------------------------

pub mod rocm {
    //! The ROCm runtime boundary over the HIP half of the [`crate::cuda`] emitter.
    //!
    //! `crate::cuda` is deliberately usable without ROCm: it emits HIP C++ text
    //! and links nothing. This module is the separate runtime half. It resolves
    //! `libamdhip64` and `libhiprtc` at run time — no link-time dependency, no
    //! headers, no build-script probing — compiles emitted source to a code
    //! object with hipRTC, loads the code object, and launches kernels through the
    //! HIP module API.
    //!
    //! # Why `dlopen` and not a `-sys` crate
    //!
    //! A ROCm build must stay buildable on a host with no ROCm installed — the
    //! same property `cuda` (emitter) keeps against `cuda-driver`, and what lets
    //! this crate's macOS CI gate compile the lane. A `hip-sys` dependency
    //! requires headers and a link path at BUILD time; resolving symbols at RUN
    //! time turns "no ROCm here" into a typed
    //! [`HipDriverError::RuntimeNotInstalled`] instead of a build failure.
    //!
    //! # What is proven where
    //!
    //! The device path is only reachable when `libamdhip64` resolves and a GPU is
    //! present. This host has neither, so what is exercised here is: the
    //! symbol-resolution contract (an absent runtime is a typed error naming the
    //! sonames tried), the element-window arithmetic and the kernel-parameter
    //! order.
    //! Numerical parity on a real device is a separate gate that needs the
    //! hardware; [`CudaDriver::smoke_add_one`](super::CudaDriver::smoke_add_one) on a
    //! `CudaDriver<HipRuntime>` (from [`shared_driver`]) is that gate's entry point.
    //!
    //! # Scope
    //!
    //! [`HipRuntime`] is the second [`GpuRuntime`] of the one CUDA executor: a `DevicePlan<HipRuntime>`
    //! lowers and launches exactly as a `CudaPlan` does, with the kernels rendered as HIP C++. This
    //! module is the device boundary beneath it: library loading, hipRTC, allocation, copies and launch.

    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::cell::Cell;
    use core::ffi::{c_char, c_int, c_void};

    use std::ffi::{CStr, CString};
    use std::ptr;
    use std::sync::{Arc, OnceLock};

    use std::collections::BTreeMap;

    use cudarc::nvrtc::CompileOptions;

    use super::CacheKey;
    use crate::config::RuntimeConfig;
    use crate::cuda::{EmitDevice, HipKernel, HipWave};
    use crate::kernel_params::{KernelArgument, KernelParams};
    use proxima_numeric::NumericPolicy;
    use proxima_tensor::{NodeId, Op, QuantizedBlock};
    use crate::lowering::Binding;
    use crate::cuda::CudaKernel;
    use crate::gpu_runtime::{DeviceElement, EventMillis, GpuRuntime, LaunchArguments, LaunchShape};

    /// Shared objects the runtime half probes, in order. The versioned names come
    /// first so a host with several ROCm installs resolves the one on its linker
    /// path rather than an arbitrary sibling.
    #[cfg(not(windows))]
    pub const RUNTIME_LIBRARIES: [&str; 4] = [
        "libamdhip64.so.6",
        "libamdhip64.so",
        "libamdhip64.so.5",
        "libamdhip64.dylib",
    ];

    /// The HIP runtime DLL on Windows, newest first: `amdhip64_7.dll` (HIP 7),
    /// `amdhip64_6.dll` (HIP 6, renamed from `amdhip64.dll` as of ROCm 6.1.2) and
    /// `amdhip64.dll` (5.x). Source: the ROCm on Windows release notes,
    /// <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/about/releasenotes.html>
    /// and <https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/>.
    #[cfg(windows)]
    pub const RUNTIME_LIBRARIES: [&str; 3] = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];

    /// hipRTC — the runtime compiler the emitter's source needs. On Linux its
    /// symbols are also exported by `libamdhip64.so`.
    #[cfg(not(windows))]
    pub const RTC_LIBRARIES: [&str; 3] = ["libhiprtc.so.6", "libhiprtc.so", "libhiprtc.dylib"];

    /// Windows hipRTC: `amdhip64.dll` does not carry the HIPRTC symbols, so
    /// `hiprtc.dll` (or the versioned `hiprtcXXYY.dll`, XX major and YY minor,
    /// two digits each: HIP 5.3 is `hiprtc0503.dll`) must be loaded. The
    /// versioned name is derived from the runtime version by
    /// [`hiprtc_windows_candidates`]; this is the unversioned fallback. Source:
    /// <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>.
    #[cfg(windows)]
    pub const RTC_LIBRARIES: [&str; 1] = ["hiprtc.dll"];

    /// The Windows hipRTC names to probe for a runtime that reported `version`
    /// from `hipRuntimeGetVersion` (HIP's `HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000
    /// + HIP_VERSION_PATCH` encoding, per
    /// <https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.0/how-to/faq.html>): the
    /// versioned DLL first, then plain `hiprtc.dll`.
    #[must_use]
    pub fn hiprtc_windows_candidates(version: c_int) -> Vec<String> {
        let major = version / 10_000_000;
        let minor = (version / 100_000) % 100;
        let mut names = Vec::new();
        if major > 0 {
            names.push(format!("hiprtc{major:02}{minor:02}.dll"));
        }
        names.push(String::from("hiprtc.dll"));
        names
    }

    /// `hipMemcpyHostToDevice`.
    const MEMCPY_HOST_TO_DEVICE: c_int = 1;
    /// `hipMemcpyDeviceToHost`.
    const MEMCPY_DEVICE_TO_HOST: c_int = 2;
    /// `hipMemcpyDeviceToDevice`.
    const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
    const HIP_SUCCESS: c_int = 0;
    const RTC_SUCCESS: c_int = 0;

    /// The R0600 `hipDeviceProp_t` is about 1.5 KB; the margin keeps a later
    /// runtime from writing past this buffer in `hipGetDevicePropertiesR0600`.
    const HIP_DEVICE_PROPERTIES_BYTES: usize = 4096;
    const HIP_DEVICE_PROPERTIES_WORDS: usize = HIP_DEVICE_PROPERTIES_BYTES / core::mem::size_of::<u64>();
    const _: () = assert!(HIP_DEVICE_PROPERTIES_BYTES >= 2048);

    /// Errors from loading the ROCm runtime, compiling HIP C++, or executing.
    #[derive(Debug, thiserror::Error)]
    pub enum HipDriverError {
        #[error("ROCm runtime is not installed; tried: {tried}")]
        RuntimeNotInstalled { tried: String },
        #[error("hipRTC is not installed; tried: {tried}")]
        RtcNotInstalled { tried: String },
        #[error("ROCm library `{library}` does not export `{symbol}`")]
        MissingSymbol {
            library: &'static str,
            symbol: &'static str,
        },
        #[error("HIP `{call}` failed with code {code}: {message}")]
        Runtime {
            call: &'static str,
            code: c_int,
            message: String,
        },
        #[error("hipRTC `{call}` failed with code {code}: {message}")]
        Rtc {
            call: &'static str,
            code: c_int,
            message: String,
        },
        #[error("hipRTC rejected the emitted source:\n{log}")]
        CompileLog { log: String },
        #[error("kernel binding contract is not inputs, output, uniforms")]
        BindingContract,
        #[error("launch argument count does not match the kernel's bindings")]
        ArgumentCountMismatch,
        #[error("device ordinal {ordinal} is out of range; this host exposes {count}")]
        DeviceOutOfRange { ordinal: c_int, count: c_int },
        #[error("hip device ordinal {0} does not fit HIP's integer device index")]
        DeviceOrdinalTooLarge(u32),
        #[error("a launch of {threads} threads needs more blocks than the launch index holds")]
        GridTooLarge { threads: u64 },
        #[error("refused to launch an idle grid: an op with no live output dispatches nothing")]
        IdleLaunch,
        #[error("smoke input is too large for the launch ABI: {0} elements")]
        InputTooLarge(usize),
        #[error("hip wave width {0} is not supported; expected 32 or 64")]
        UnsupportedWaveWidth(u32),
        #[error("configured hip wave width {configured} does not match the device's warpSize {device}")]
        WaveMismatch { configured: u32, device: u32 },
        #[error("hip graph path rejected the graph: {reason}")]
        UnsupportedGraph { reason: &'static str },
        #[error("hip graph node {node:?} could not be emitted: {error}")]
        Emit { node: NodeId, error: String },
        #[error("hip graph buffer for node {0:?} is missing")]
        MissingBuffer(NodeId),
        #[error("a {len}-byte view at offset {offset} passes the end of a {bytes}-byte allocation")]
        ViewOutOfRange { offset: usize, len: usize, bytes: usize },
        #[error("hip events are not recorded by this runtime, so launch timing is skipped")]
        EventsUnsupported,
        #[error("this hip runtime does not export hipMemGetInfo")]
        MemoryInfoUnavailable,
    }

    type HipInit = unsafe extern "C" fn(u32) -> c_int;
    type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type HipSetDevice = unsafe extern "C" fn(c_int) -> c_int;
    type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type HipGetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
    type HipDeviceSynchronize = unsafe extern "C" fn() -> c_int;
    type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
    type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
    type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
    type HipMemset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
    type HipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
    type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
    type HipModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
    type HipModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
    type HipModuleLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int;
    type HipModuleUnload = unsafe extern "C" fn(*mut c_void) -> c_int;

    type RtcCreateProgram = unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int;
    type RtcCompileProgram =
        unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
    type RtcSizeQuery = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
    type RtcWriteText = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
    type RtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
    type RtcDestroyProgram = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
    type RtcGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
    #[cfg(windows)]
    type HipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;

    /// One loaded shared object (`dlopen` on unix, `LoadLibrary` on Windows,
    /// through `libloading`). Kept alive for as long as any symbol resolved from
    /// it is callable, which is why every device, module and buffer holds an
    /// `Arc` of the [`HipApi`] that owns it.
    #[derive(Debug)]
    struct Library(libloading::Library);

    fn open_library<S: AsRef<str>>(
        candidates: &[S],
        absent: fn(String) -> HipDriverError,
    ) -> Result<Library, HipDriverError> {
        let mut tried = Vec::new();
        for name in candidates {
            let name = name.as_ref();
            // SAFETY: loading a library runs its initializers; the names are the
            // vendor runtime's own, and a failure is reported, never ignored.
            match unsafe { libloading::Library::new(name) } {
                Ok(library) => return Ok(Library(library)),
                Err(error) => tried.push(format!("{name} ({error})")),
            }
        }
        Err(absent(tried.join(", ")))
    }

    fn resolve<T: Copy>(
        library: &Library,
        library_name: &'static str,
        symbol: &'static str,
    ) -> Result<T, HipDriverError> {
        // SAFETY: the caller names `T` as the exported signature of `symbol`, and
        // the copied function pointer is only called while the `Library` is held.
        unsafe { library.0.get::<T>(symbol.as_bytes()) }
            .map(|address| *address)
            .map_err(|_| HipDriverError::MissingSymbol {
                library: library_name,
                symbol,
            })
    }

    #[cfg(windows)]
    fn open_rtc(runtime: &Library) -> Result<Library, HipDriverError> {
        let version = resolve::<HipRuntimeGetVersion>(runtime, "amdhip64", "hipRuntimeGetVersion")
            .ok()
            .and_then(|query| {
                let mut version: c_int = 0;
                // SAFETY: `version` is a valid out-parameter.
                (unsafe { query(&mut version) } == HIP_SUCCESS).then_some(version)
            })
            .unwrap_or(0);
        open_library(&hiprtc_windows_candidates(version), |tried| {
            HipDriverError::RtcNotInstalled { tried }
        })
    }

    #[cfg(not(windows))]
    fn open_rtc(_runtime: &Library) -> Result<Library, HipDriverError> {
        open_library(&RTC_LIBRARIES, |tried| HipDriverError::RtcNotInstalled { tried })
    }

    /// The resolved ROCm entry points, shared by every device, module and buffer.
    ///
    /// Holding the two library handles is what keeps the resolved pointers
    /// callable for the lifetime of the `Arc`.
    #[derive(Debug)]
    pub struct HipApi {
        _runtime: Library,
        _rtc: Library,
        hip_init: HipInit,
        hip_get_device_count: HipGetDeviceCount,
        hip_set_device: HipSetDevice,
        hip_device_get_name: HipDeviceGetName,
        hip_get_device_properties: Option<HipGetDeviceProperties>,
        hip_device_synchronize: HipDeviceSynchronize,
        hip_malloc: HipMalloc,
        hip_free: HipFree,
        hip_memcpy: HipMemcpy,
        hip_memset: HipMemset,
        hip_mem_get_info: Option<HipMemGetInfo>,
        hip_get_error_string: HipGetErrorString,
        hip_module_load_data: HipModuleLoadData,
        hip_module_get_function: HipModuleGetFunction,
        hip_module_launch_kernel: HipModuleLaunchKernel,
        hip_module_unload: HipModuleUnload,
        rtc_create_program: RtcCreateProgram,
        rtc_compile_program: RtcCompileProgram,
        rtc_get_program_log_size: RtcSizeQuery,
        rtc_get_program_log: RtcWriteText,
        rtc_get_code_size: RtcSizeQuery,
        rtc_get_code: RtcWriteText,
        rtc_destroy_program: RtcDestroyProgram,
        rtc_get_error_string: RtcGetErrorString,
        rtc_version: Option<RtcVersion>,
    }

    impl HipApi {
        /// Resolves the HIP runtime and hipRTC. A host without ROCm installed
        /// returns [`HipDriverError::RuntimeNotInstalled`] naming every soname
        /// probed, never a panic and never a link failure.
        ///
        /// # Errors
        /// [`HipDriverError::RuntimeNotInstalled`] or
        /// [`HipDriverError::RtcNotInstalled`] when a library is absent, and
        /// [`HipDriverError::MissingSymbol`] when a library resolves but does not
        /// export a function this boundary needs.
        pub fn load() -> Result<Arc<Self>, HipDriverError> {
            let runtime = open_library(&RUNTIME_LIBRARIES, |tried| {
                HipDriverError::RuntimeNotInstalled { tried }
            })?;
            let rtc = open_rtc(&runtime)?;
            let runtime_name = "libamdhip64";
            let rtc_name = "libhiprtc";
            let hip_get_device_properties = resolve::<HipGetDeviceProperties>(
                &runtime,
                runtime_name,
                "hipGetDevicePropertiesR0600",
            )
            .or_else(|_| resolve(&runtime, runtime_name, "hipGetDeviceProperties"))
            .ok();
            Ok(Arc::new(Self {
                hip_init: resolve(&runtime, runtime_name, "hipInit")?,
                hip_get_device_count: resolve(&runtime, runtime_name, "hipGetDeviceCount")?,
                hip_set_device: resolve(&runtime, runtime_name, "hipSetDevice")?,
                hip_device_get_name: resolve(&runtime, runtime_name, "hipDeviceGetName")?,
                hip_get_device_properties,
                hip_device_synchronize: resolve(&runtime, runtime_name, "hipDeviceSynchronize")?,
                hip_malloc: resolve(&runtime, runtime_name, "hipMalloc")?,
                hip_free: resolve(&runtime, runtime_name, "hipFree")?,
                hip_memcpy: resolve(&runtime, runtime_name, "hipMemcpy")?,
                hip_memset: resolve(&runtime, runtime_name, "hipMemset")?,
                hip_mem_get_info: resolve(&runtime, runtime_name, "hipMemGetInfo").ok(),
                hip_get_error_string: resolve(&runtime, runtime_name, "hipGetErrorString")?,
                hip_module_load_data: resolve(&runtime, runtime_name, "hipModuleLoadData")?,
                hip_module_get_function: resolve(&runtime, runtime_name, "hipModuleGetFunction")?,
                hip_module_launch_kernel: resolve(&runtime, runtime_name, "hipModuleLaunchKernel")?,
                hip_module_unload: resolve(&runtime, runtime_name, "hipModuleUnload")?,
                rtc_create_program: resolve(&rtc, rtc_name, "hiprtcCreateProgram")?,
                rtc_compile_program: resolve(&rtc, rtc_name, "hiprtcCompileProgram")?,
                rtc_get_program_log_size: resolve(&rtc, rtc_name, "hiprtcGetProgramLogSize")?,
                rtc_get_program_log: resolve(&rtc, rtc_name, "hiprtcGetProgramLog")?,
                rtc_get_code_size: resolve(&rtc, rtc_name, "hiprtcGetCodeSize")?,
                rtc_get_code: resolve(&rtc, rtc_name, "hiprtcGetCode")?,
                rtc_destroy_program: resolve(&rtc, rtc_name, "hiprtcDestroyProgram")?,
                rtc_get_error_string: resolve(&rtc, rtc_name, "hiprtcGetErrorString")?,
                rtc_version: resolve(&rtc, rtc_name, "hiprtcVersion").ok(),
                _runtime: runtime,
                _rtc: rtc,
            }))
        }

        /// Whether the ROCm runtime resolves on this host. A caller choosing
        /// between drivers before any device work reads this.
        #[must_use]
        pub fn is_available() -> bool {
            Self::load().is_ok()
        }

        fn runtime_error(&self, call: &'static str, code: c_int) -> HipDriverError {
            // SAFETY: `hipGetErrorString` returns a NUL-terminated static string
            // for any code, including one it does not recognize.
            let message = unsafe { CStr::from_ptr((self.hip_get_error_string)(code)) }
                .to_string_lossy()
                .into_owned();
            HipDriverError::Runtime { call, code, message }
        }

        fn rtc_error(&self, call: &'static str, code: c_int) -> HipDriverError {
            // SAFETY: same contract as `hipGetErrorString`.
            let message = unsafe { CStr::from_ptr((self.rtc_get_error_string)(code)) }
                .to_string_lossy()
                .into_owned();
            HipDriverError::Rtc { call, code, message }
        }

        fn initialize(&self) -> Result<(), HipDriverError> {
            // SAFETY: `hipInit` takes a flags word that must be 0.
            let status = unsafe { (self.hip_init)(0) };
            if status != HIP_SUCCESS {
                return Err(self.runtime_error("hipInit", status));
            }
            Ok(())
        }

        fn device_count(&self) -> Result<c_int, HipDriverError> {
            let mut count: c_int = 0;
            // SAFETY: `count` is a valid out-parameter.
            let status = unsafe { (self.hip_get_device_count)(&mut count) };
            if status != HIP_SUCCESS {
                return Err(self.runtime_error("hipGetDeviceCount", status));
            }
            Ok(count)
        }

        fn device_name(&self, ordinal: c_int) -> Result<String, HipDriverError> {
            let mut buffer = [0 as c_char; 256];
            // SAFETY: `buffer` is 256 bytes, which is the length passed.
            let status = unsafe { (self.hip_device_get_name)(buffer.as_mut_ptr(), 256, ordinal) };
            if status != HIP_SUCCESS {
                return Err(self.runtime_error("hipDeviceGetName", status));
            }
            // SAFETY: `hipDeviceGetName` NUL-terminates within the buffer.
            Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        }

        fn device_arch(&self, ordinal: c_int) -> Option<String> {
            let query = self.hip_get_device_properties?;
            let mut properties = [0_u64; HIP_DEVICE_PROPERTIES_WORDS];
            // SAFETY: the buffer is HIP_DEVICE_PROPERTIES_BYTES and the runtime
            // writes at most one R0600 hipDeviceProp_t (~1.5 KB) into it.
            let status = unsafe { query(properties.as_mut_ptr().cast::<c_void>(), ordinal) };
            if status != HIP_SUCCESS {
                return None;
            }
            // SAFETY: the call filled the initialized buffer; the scan reads only bytes.
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    properties.as_ptr().cast::<u8>(),
                    HIP_DEVICE_PROPERTIES_BYTES,
                )
            };
            gcn_arch_name(bytes).map(str::to_owned)
        }
    }

    /// The `gcnArchName` token inside a `hipDeviceProp_tR0600` image: the first
    /// NUL-terminated `gfx...` run of architecture-name bytes. Scanning rather
    /// than declaring the ~1.5 KB struct keeps this independent of the header's
    /// versioned layout.
    fn gcn_arch_name(properties: &[u8]) -> Option<&str> {
        properties
            .windows(3)
            .position(|window| window == b"gfx")
            .and_then(|start| {
                let rest = &properties[start..];
                let end = rest.iter().position(|byte| *byte == 0)?;
                let token = core::str::from_utf8(&rest[..end]).ok()?;
                let suffix = token.get(3..)?;
                let shaped = !suffix.is_empty()
                    && suffix
                        .chars()
                        .all(|value| value.is_ascii_alphanumeric() || matches!(value, ':' | '-' | '+' | '_'));
                shaped.then_some(token)
            })
    }

    /// A device allocation with the runtime handle that must free it.
    pub struct HipBuffer {
        api: Arc<HipApi>,
        pointer: *mut c_void,
        bytes: usize,
        ordinal: c_int,
    }

    impl core::fmt::Debug for HipBuffer {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter
                .debug_struct("HipBuffer")
                .field("bytes", &self.bytes)
                .finish_non_exhaustive()
        }
    }

    impl HipBuffer {
        #[must_use]
        pub fn bytes(&self) -> usize {
            self.bytes
        }

        /// A borrowed window of `len` bytes starting `offset` bytes into this
        /// allocation. The view cannot outlive the buffer, and the window is
        /// checked against the allocation here, so no later copy has to trust it.
        ///
        /// # Errors
        /// [`HipDriverError::ViewOutOfRange`] when `offset + len` passes the allocation.
        pub fn view(&self, offset: usize, len: usize) -> Result<HipView<'_>, HipDriverError> {
            window_end(self.bytes, offset, len).ok_or(HipDriverError::ViewOutOfRange {
                offset,
                len,
                bytes: self.bytes,
            })?;
            Ok(HipView {
                pointer: offset_pointer(self.pointer, offset),
                len,
                owner: core::marker::PhantomData,
            })
        }
    }

    /// The end offset of a `len`-byte window at `offset` inside `total` bytes,
    /// or `None` when it overflows or passes the end.
    fn window_end(total: usize, offset: usize, len: usize) -> Option<usize> {
        offset.checked_add(len).filter(|end| *end <= total)
    }

    /// `base` advanced `offset` bytes. The result is only dereferenced by the
    /// device, and only after [`window_end`] proved it stays inside the allocation.
    fn offset_pointer(base: *mut c_void, offset: usize) -> *mut c_void {
        base.cast::<u8>().wrapping_add(offset).cast::<c_void>()
    }

    /// A bounds-checked window into a [`HipBuffer`], borrowed from it.
    #[derive(Debug, Clone, Copy)]
    pub struct HipView<'buffer> {
        pointer: *mut c_void,
        len: usize,
        owner: core::marker::PhantomData<&'buffer HipBuffer>,
    }

    impl KernelArgument for HipBuffer {
        fn slot(&self) -> *mut c_void {
            (&raw const self.pointer).cast_mut().cast::<c_void>()
        }
    }

    impl KernelArgument for HipView<'_> {
        fn slot(&self) -> *mut c_void {
            (&raw const self.pointer).cast_mut().cast::<c_void>()
        }
    }

    impl HipView<'_> {
        #[must_use]
        pub fn len(&self) -> usize {
            self.len
        }

        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }
    }

    impl Drop for HipBuffer {
        fn drop(&mut self) {
            if !self.pointer.is_null() {
                // SAFETY: the pointer came from `hipMalloc` on this ordinal's device, bound
                // first because the dropping thread may not have it current, and is freed once.
                release_on_device(&self.api, self.ordinal, || {
                    unsafe { (self.api.hip_free)(self.pointer) };
                });
            }
        }
    }

    /// A loaded code object. Dropping it unloads the module, so a function handle
    /// obtained from it must not outlive it.
    pub struct HipModule {
        api: Arc<HipApi>,
        handle: *mut c_void,
        ordinal: c_int,
    }

    impl core::fmt::Debug for HipModule {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("HipModule")
        }
    }

    impl HipModule {
        /// The entry point `name` inside this module.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when the module has no such symbol.
        pub fn function(&self, name: &str) -> Result<*mut c_void, HipDriverError> {
            let Ok(name_c) = CString::new(name) else {
                return Err(HipDriverError::BindingContract);
            };
            let mut function = ptr::null_mut();
            // SAFETY: `self.handle` is a live module; `name_c` is NUL-terminated.
            let status =
                unsafe { (self.api.hip_module_get_function)(&mut function, self.handle, name_c.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipModuleGetFunction", status));
            }
            Ok(function)
        }
    }

    impl Drop for HipModule {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                // SAFETY: the handle came from `hipModuleLoadData` on this ordinal's device, bound
                // first, and is unloaded once.
                release_on_device(&self.api, self.ordinal, || {
                    unsafe { (self.api.hip_module_unload)(self.handle) };
                });
            }
        }
    }

    /// The options hipRTC receives: one `--offload-arch` naming the opened device
    /// unless the caller already supplied an arch, so a code object never targets
    /// a device other than the one the plan selected.
    fn compile_options(arch: Option<&str>, options: &[&str]) -> Vec<String> {
        let explicit = options.iter().any(|option| {
            option.starts_with("--offload-arch") || option.starts_with("--gpu-architecture")
        });
        let mut resolved = Vec::with_capacity(options.len() + 1);
        if !explicit
            && let Some(arch) = arch.filter(|arch| !arch.is_empty())
        {
            resolved.push(format!("--offload-arch={arch}"));
        }
        resolved.extend(options.iter().map(|option| (*option).to_owned()));
        resolved
    }

    /// Converts the configured ordinal to HIP's integer device index before any
    /// runtime call.
    pub fn open_ordinal(device: u32) -> Result<c_int, HipDriverError> {
        c_int::try_from(device).map_err(|_| HipDriverError::DeviceOrdinalTooLarge(device))
    }

    /// The HIP lane's numeric policy: the conflaguration/proxima-numeric policy
    /// with the accumulator and flash narrowing every GPU lane applies.
    #[must_use]
    pub fn numeric_policy(numeric: NumericPolicy) -> NumericPolicy {
        crate::bind_plan::binding_policy(numeric, &crate::cuda::executes(&crate::cuda::target::Target::HIP))
    }

    /// One ROCm device, resolved through the runtime API.
    #[derive(Debug)]
    pub struct HipDevice {
        api: Arc<HipApi>,
        ordinal: c_int,
        name: String,
        arch: Option<String>,
    }

    impl HipDevice {
        /// The devices this host exposes. An empty vector means the runtime loaded
        /// but sees no GPU.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when `hipInit` or `hipGetDeviceCount` fails.
        pub fn devices() -> Result<Vec<String>, HipDriverError> {
            let api = HipApi::load()?;
            api.initialize()?;
            let count = api.device_count()?;
            (0..count).map(|ordinal| api.device_name(ordinal)).collect()
        }

        /// Opens `ordinal`, making it the current device for every launch issued
        /// through this value.
        ///
        /// # Errors
        /// [`HipDriverError::RuntimeNotInstalled`] or
        /// [`HipDriverError::RtcNotInstalled`] when ROCm is absent,
        /// [`HipDriverError::DeviceOutOfRange`] past the device count, and
        /// [`HipDriverError::Runtime`] when the device cannot be selected.
        pub fn open(ordinal: c_int) -> Result<Self, HipDriverError> {
            let api = HipApi::load()?;
            api.initialize()?;
            let count = api.device_count()?;
            if ordinal < 0 || ordinal >= count {
                return Err(HipDriverError::DeviceOutOfRange { ordinal, count });
            }
            bind_ordinal(ordinal, |ordinal| api.set_device(ordinal))?;
            let name = api.device_name(ordinal)?;
            let arch = api.device_arch(ordinal);
            Ok(Self {
                api,
                ordinal,
                name,
                arch,
            })
        }

        #[must_use]
        pub fn name(&self) -> &str {
            &self.name
        }

        /// The device's `gcnArchName` when the runtime exposes
        /// `hipGetDevicePropertiesR0600`; `None` lets hipRTC target the current
        /// device itself.
        #[must_use]
        pub fn arch(&self) -> Option<&str> {
            self.arch.as_deref()
        }

        /// The `(major, minor)` of the hipRTC that compiles for this device, or `None` when the
        /// library does not export `hiprtcVersion` or the call fails.
        #[must_use]
        pub fn compiler_version(&self) -> Option<(i32, i32)> {
            let query = self.api.rtc_version?;
            let (mut major, mut minor) = (0, 0);
            // SAFETY: both out-parameters are valid for the call.
            let status = unsafe { query(&mut major, &mut minor) };
            (status == RTC_SUCCESS).then_some((major, minor))
        }

        #[must_use]
        pub fn ordinal(&self) -> c_int {
            self.ordinal
        }

        #[must_use]
        pub fn api(&self) -> &Arc<HipApi> {
            &self.api
        }

        /// Makes this device the calling thread's current HIP device; a no-op when the thread already
        /// bound it. Every entry that allocates, copies, sets memory, compiles, loads, launches,
        /// synchronizes or queries memory starts here, whichever thread calls it.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when `hipSetDevice` fails.
        pub fn bind(&self) -> Result<(), HipDriverError> {
            bind_ordinal(self.ordinal, |ordinal| self.api.set_device(ordinal))
        }

        /// Compiles HIP C++ to a loadable code object.
        ///
        /// # Errors
        /// [`HipDriverError::CompileLog`] with the compiler's own log when the
        /// source is rejected, [`HipDriverError::Rtc`] when hipRTC itself fails,
        /// and [`HipDriverError::Runtime`] when the object cannot be loaded.
        pub fn compile(&self, source: &str, options: &[&str]) -> Result<HipModule, HipDriverError> {
            self.load_code_object(&self.compile_code_object(source, options)?)
        }

        /// Compiles HIP C++ to the code-object bytes [`Self::load_code_object`] loads, so a caller can
        /// keep the bytes between processes.
        ///
        /// # Errors
        /// [`HipDriverError::CompileLog`] with the compiler's own log when the source is rejected, and
        /// [`HipDriverError::Rtc`] when hipRTC itself fails.
        pub fn compile_code_object(&self, source: &str, options: &[&str]) -> Result<Vec<u8>, HipDriverError> {
            self.bind()?;
            let Ok(source_c) = CString::new(source) else {
                return Err(HipDriverError::BindingContract);
            };
            let Ok(name_c) = CString::new("proxima_hip") else {
                return Err(HipDriverError::BindingContract);
            };
            let mut program = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call; there
            // are no header sources, so those arrays are NULL/0.
            let status = unsafe {
                (self.api.rtc_create_program)(
                    &mut program,
                    source_c.as_ptr(),
                    name_c.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            };
            if status != RTC_SUCCESS {
                return Err(self.api.rtc_error("hiprtcCreateProgram", status));
            }

            let options = compile_options(self.arch(), options);
            let options_c = options
                .iter()
                .filter_map(|option| CString::new(option.as_str()).ok())
                .collect::<Vec<_>>();
            let option_pointers = options_c
                .iter()
                .map(|option| option.as_ptr())
                .collect::<Vec<_>>();
            let option_count = c_int::try_from(option_pointers.len()).unwrap_or(c_int::MAX);
            // SAFETY: `program` is live; every option pointer is NUL-terminated
            // and outlives the call.
            let status = unsafe {
                (self.api.rtc_compile_program)(program, option_count, option_pointers.as_ptr())
            };
            if status != RTC_SUCCESS {
                let log = self.program_log(program);
                // SAFETY: `program` is live and is destroyed exactly once.
                unsafe { (self.api.rtc_destroy_program)(&mut program) };
                return Err(HipDriverError::CompileLog { log });
            }

            let mut code_size: usize = 0;
            // SAFETY: `program` is live and `code_size` is a valid out-parameter.
            let status = unsafe { (self.api.rtc_get_code_size)(program, &mut code_size) };
            if status != RTC_SUCCESS {
                // SAFETY: `program` is live and is destroyed exactly once.
                unsafe { (self.api.rtc_destroy_program)(&mut program) };
                return Err(self.api.rtc_error("hiprtcGetCodeSize", status));
            }
            let mut code = vec![0u8; code_size];
            // SAFETY: the buffer is exactly `code_size` bytes, as the API expects.
            let status = unsafe { (self.api.rtc_get_code)(program, code.as_mut_ptr().cast::<c_char>()) };
            // SAFETY: `program` is live and is destroyed exactly once.
            unsafe { (self.api.rtc_destroy_program)(&mut program) };
            if status != RTC_SUCCESS {
                return Err(self.api.rtc_error("hiprtcGetCode", status));
            }
            Ok(code)
        }

        /// Loads a code object produced by [`Self::compile_code_object`] for this device's arch.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when the object cannot be loaded.
        pub fn load_code_object(&self, code: &[u8]) -> Result<HipModule, HipDriverError> {
            self.bind()?;
            let mut handle = ptr::null_mut();
            // SAFETY: `code` holds a complete code object; HIP copies it.
            let status =
                unsafe { (self.api.hip_module_load_data)(&mut handle, code.as_ptr().cast::<c_void>()) };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipModuleLoadData", status));
            }
            Ok(HipModule {
                api: Arc::clone(&self.api),
                handle,
                ordinal: self.ordinal,
            })
        }

        fn program_log(&self, program: *mut c_void) -> String {
            let mut size: usize = 0;
            // SAFETY: `program` is live and `size` is a valid out-parameter.
            let status = unsafe { (self.api.rtc_get_program_log_size)(program, &mut size) };
            if status != RTC_SUCCESS || size == 0 {
                return String::from("hipRTC produced no log");
            }
            let mut log = vec![0 as c_char; size];
            // SAFETY: the buffer is exactly `size` bytes, which is what the API writes.
            let status = unsafe { (self.api.rtc_get_program_log)(program, log.as_mut_ptr()) };
            if status != RTC_SUCCESS {
                return String::from("hipRTC produced an unreadable log");
            }
            // SAFETY: the log is NUL-terminated within `size` bytes.
            unsafe { CStr::from_ptr(log.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        }

        /// Uploads raw bytes (uniforms, packed weight blocks).
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when the allocation or copy fails.
        pub fn upload_bytes(&self, bytes: &[u8]) -> Result<HipBuffer, HipDriverError> {
            self.bind()?;
            let buffer = self.allocate_bytes(bytes.len())?;
            let pointer = buffer.pointer;
            if !bytes.is_empty() {
                // SAFETY: `pointer` holds `buffer.bytes >= bytes.len()` bytes.
                let status = unsafe {
                    (self.api.hip_memcpy)(
                        pointer,
                        bytes.as_ptr().cast::<c_void>(),
                        bytes.len(),
                        MEMCPY_HOST_TO_DEVICE,
                    )
                };
                if status != HIP_SUCCESS {
                    return Err(self.api.runtime_error("hipMemcpy", status));
                }
            }
            Ok(buffer)
        }

        /// Allocates `bytes` of uninitialized device memory (at least one byte,
        /// so an empty request still owns a valid pointer).
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when the allocation fails.
        pub fn allocate_bytes(&self, bytes: usize) -> Result<HipBuffer, HipDriverError> {
            self.bind()?;
            let requested = bytes.max(1);
            let mut pointer = ptr::null_mut();
            // SAFETY: `pointer` is a valid out-parameter.
            let status = unsafe { (self.api.hip_malloc)(&mut pointer, requested) };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipMalloc", status));
            }
            Ok(HipBuffer {
                api: Arc::clone(&self.api),
                pointer,
                bytes: requested,
                ordinal: self.ordinal,
            })
        }

        /// Allocates `bytes` of device memory and zeroes all of it on the device,
        /// with no host-side zero buffer: the device twin of `alloc_zeros`.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when the allocation or the memset fails.
        pub fn allocate_zeroed_bytes(&self, bytes: usize) -> Result<HipBuffer, HipDriverError> {
            self.bind()?;
            let buffer = self.allocate_bytes(bytes)?;
            // SAFETY: `buffer.pointer` owns `buffer.bytes` bytes of device memory.
            let status = unsafe { (self.api.hip_memset)(buffer.pointer, 0, buffer.bytes) };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipMemset", status));
            }
            Ok(buffer)
        }

        /// Copies `source`'s bytes into the start of `destination` on the device.
        ///
        /// # Errors
        /// [`HipDriverError::ViewOutOfRange`] when `source` is longer than `destination`,
        /// [`HipDriverError::Runtime`] when the copy fails.
        pub fn copy_device(&self, destination: &HipView<'_>, source: &HipView<'_>) -> Result<(), HipDriverError> {
            if source.len > destination.len {
                return Err(HipDriverError::ViewOutOfRange {
                    offset: 0,
                    len: source.len,
                    bytes: destination.len,
                });
            }
            if source.is_empty() {
                return Ok(());
            }
            self.bind()?;
            // SAFETY: both views were bounds-checked against live allocations
            // they borrow, and `source.len <= destination.len` by the check above.
            let status = unsafe {
                (self.api.hip_memcpy)(
                    destination.pointer,
                    source.pointer.cast_const(),
                    source.len,
                    MEMCPY_DEVICE_TO_DEVICE,
                )
            };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipMemcpy", status));
            }
            Ok(())
        }

        /// Blocks until every queued launch on this device has completed.
        ///
        /// # Errors
        /// [`HipDriverError::Runtime`] when synchronization fails.
        pub fn synchronize(&self) -> Result<(), HipDriverError> {
            self.bind()?;
            // SAFETY: no arguments beyond the implicit current device, bound just above.
            let status = unsafe { (self.api.hip_device_synchronize)() };
            if status != HIP_SUCCESS {
                return Err(self.api.runtime_error("hipDeviceSynchronize", status));
            }
            Ok(())
        }
    }

    /// Probe kernel: writes the device's `warpSize`, so the wave width the
    /// emitter is built for comes from the hardware, never from a card table.
    fn warp_size_kernel() -> Result<HipKernel, HipDriverError> {
        Ok(HipKernel {
            source: String::from(
                "extern \"C\" __global__ void proxima_warp_size(unsigned int *out, const unsigned int *u) {\n\
                 \x20   if (threadIdx.x == 0) out[0] = warpSize;\n\
                 }\n",
            ),
            entry: String::from("proxima_warp_size"),
            bindings: vec![
                Binding::Output(proxima_tensor::NodeId(0)),
                Binding::Uniforms,
            ],
            grid: crate::cuda::HipGridSpec::launch(
                proxima_tensor::NodeId(0),
                1,
                None,
                "warp_size_probe",
            )
            .map_err(|_| HipDriverError::IdleLaunch)?,
            extra_outputs: Vec::new(),
            prologue: Vec::new(),
            route_compaction_words: 0,
        })
    }

    /// The wave `RuntimeConfig::hip_wave` selects: `None` for 0, which means the
    /// device reports it. The one place the config value becomes a [`HipWave`].
    #[must_use]
    pub fn hip_wave_of(config: &RuntimeConfig) -> Option<HipWave> {
        HipWave::from_lanes(u64::from(config.hip_wave))
    }

    /// The wave a runtime emits for: the device's `probed` width, which a `configured` width must equal
    /// because the emitted kernels trap on any other `warpSize`.
    ///
    /// # Errors
    /// [`HipDriverError::WaveMismatch`] naming both widths.
    fn resolve_wave(configured: Option<HipWave>, probed: HipWave) -> Result<HipWave, HipDriverError> {
        let width = |wave: HipWave| u32::try_from(wave.lanes()).unwrap_or(u32::MAX);
        match configured {
            Some(wave) if wave != probed => Err(HipDriverError::WaveMismatch { configured: width(wave), device: width(probed) }),
            _ => Ok(probed),
        }
    }

    type HipDriverKey = (u32, Option<u64>);
    type HipDriverTable = proxima_primitives::sync::blocking::Mutex<BTreeMap<HipDriverKey, super::CudaDriver<HipRuntime>>>;

    static HIP_DRIVERS: OnceLock<HipDriverTable> = OnceLock::new();

    /// The process's one driver for `(config.hip_device, wave)`, opened on first use: the HIP
    /// counterpart of the single CUDA driver, so every plan shares its device context, weight ledger
    /// and kernel cache. Clones share state (every field is an `Arc`). A failed open is not cached,
    /// so a missing runtime stays the typed [`HipDriverError`] on each call.
    ///
    /// # Errors
    /// The refusals of [`HipRuntime::open`].
    pub fn shared_driver(config: &RuntimeConfig, wave: Option<HipWave>) -> Result<super::CudaDriver<HipRuntime>, HipDriverError> {
        let drivers = HIP_DRIVERS.get_or_init(|| proxima_primitives::sync::blocking::Mutex::new(BTreeMap::new()));
        let key = (config.hip_device, wave.map(HipWave::lanes));
        let mut open = drivers.lock();
        if let Some(driver) = open.get(&key) {
            return Ok(driver.clone());
        }
        let driver = super::CudaDriver::<HipRuntime>::open(config, wave)?;
        open.insert(key, driver.clone());
        Ok(driver)
    }

    /// Plans `program` on the shared executor over [`shared_driver`]: the same `plan_with_config`
    /// path a CUDA device takes, under the HIP grant of the resolved wave.
    ///
    /// # Errors
    /// The refusals of [`shared_driver`] and of `plan_with_config`.
    pub fn plan(
        config: &RuntimeConfig,
        wave: Option<HipWave>,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
    ) -> Result<super::HipPlan, super::CudaDriverError> {
        let driver = shared_driver(config, wave)?;
        driver.plan_with_config(program, symbols, outputs, named, config)
    }

    // SAFETY (the four impls below): a HIP allocation, code object and function handle are process-wide
    // runtime state addressed by pointer; the host never dereferences them, so moving or sharing the
    // owning handle across threads cannot alias host memory, and the device serializes on its queue.
    // HIP's current device is per host thread, so the handle alone does not pin its device: every
    // `HipDevice`/`HipRuntime` entry calls `bind` first, which makes the device it was opened for the
    // calling thread's current device, so a thread other than the opener still acts on that device.
    unsafe impl Send for HipBuffer {}
    unsafe impl Sync for HipBuffer {}
    unsafe impl Send for HipModule {}
    unsafe impl Sync for HipModule {}

    /// The element window `range` names inside a buffer of `total` elements of `size` bytes each, as
    /// `(byte offset, byte length)`. A range past the end is refused, never clamped: a kernel gets
    /// only the pointer, so a clamped start would hand it an end-of-allocation pointer with no
    /// length. The refusal rides the window and surfaces when the window is pushed to a launch.
    fn element_window(total: usize, range: &core::ops::Range<usize>, size: usize) -> Result<(usize, usize), WindowRefusal> {
        if range.start <= range.end && range.end <= total {
            return Ok((range.start * size, (range.end - range.start) * size));
        }
        Err(WindowRefusal {
            offset: range.start.saturating_mul(size),
            len: range.end.saturating_sub(range.start).saturating_mul(size),
            bytes: total.saturating_mul(size),
        })
    }

    /// An out-of-range window request, kept on the window until a launch can return it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct WindowRefusal {
        offset: usize,
        len: usize,
        bytes: usize,
    }

    impl WindowRefusal {
        fn error(self) -> HipDriverError {
            HipDriverError::ViewOutOfRange { offset: self.offset, len: self.len, bytes: self.bytes }
        }
    }

    /// A HIP device allocation of `Element`s: the untyped [`HipBuffer`] plus the element count the
    /// shared executor's typed buffers carry.
    pub struct HipSlice<Element> {
        buffer: Arc<HipBuffer>,
        len: usize,
        element: core::marker::PhantomData<fn() -> Element>,
    }

    impl<Element> Clone for HipSlice<Element> {
        fn clone(&self) -> Self {
            Self { buffer: Arc::clone(&self.buffer), len: self.len, element: core::marker::PhantomData }
        }
    }

    impl<Element> core::fmt::Debug for HipSlice<Element> {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.debug_struct("HipSlice").field("len", &self.len).finish_non_exhaustive()
        }
    }

    /// A shared window of `Element`s onto a [`HipSlice`].
    pub struct HipWindow<'buffer, Element> {
        view: HipView<'buffer>,
        refusal: Option<WindowRefusal>,
        element: core::marker::PhantomData<fn() -> Element>,
    }

    /// A writable window of `Element`s onto a [`HipSlice`], exclusive for its lifetime.
    pub struct HipWindowMut<'buffer, Element> {
        view: HipView<'buffer>,
        refusal: Option<WindowRefusal>,
        exclusive: core::marker::PhantomData<&'buffer mut HipBuffer>,
        element: core::marker::PhantomData<fn() -> Element>,
    }

    impl<Element> KernelArgument for HipWindow<'_, Element> {
        fn slot(&self) -> *mut c_void {
            self.view.slot()
        }
    }

    impl<Element> KernelArgument for HipWindowMut<'_, Element> {
        fn slot(&self) -> *mut c_void {
            self.view.slot()
        }
    }

    /// A kernel's source, awaiting [`GpuRuntime::load_module`].
    #[derive(Debug)]
    pub struct HipImage(String);

    /// A compiled entry point; cloning shares the loaded module.
    #[derive(Debug, Clone)]
    pub struct HipFunction {
        // keeps the code object loaded while any handle to its function lives
        _module: Arc<HipModule>,
        handle: *mut c_void,
    }

    // SAFETY: see the impls above; the handle is only passed to the launch call, which binds the
    // runtime's device on the launching thread first.
    unsafe impl Send for HipFunction {}
    unsafe impl Sync for HipFunction {}

    /// The borrowed parameter array of one HIP launch: every argument is borrowed by the
    /// [`KernelParams`] until the launch returns.
    pub struct HipLaunch<'launch> {
        function: &'launch HipFunction,
        parameters: KernelParams<'launch>,
        refusal: Option<WindowRefusal>,
    }

    impl<'launch> LaunchArguments<'launch, HipRuntime> for HipLaunch<'launch> {
        fn push<Element: DeviceElement>(&mut self, buffer: &'launch HipSlice<Element>) -> &mut Self {
            self.parameters.push(&*buffer.buffer);
            self
        }

        fn push_mut<Element: DeviceElement>(&mut self, buffer: &'launch mut HipSlice<Element>) -> &mut Self {
            self.parameters.push(&*buffer.buffer);
            self
        }

        fn push_view<Element: DeviceElement>(&mut self, view: &'launch HipWindow<'_, Element>) -> &mut Self {
            self.refusal = self.refusal.or(view.refusal);
            self.parameters.push(view);
            self
        }

        fn push_view_mut<Element: DeviceElement>(&mut self, view: &'launch mut HipWindowMut<'_, Element>) -> &mut Self {
            self.refusal = self.refusal.or(view.refusal);
            self.parameters.push(&*view);
            self
        }

        fn push_value<Element: DeviceElement>(&mut self, value: &'launch Element) -> &mut Self {
            self.parameters.push_value(value);
            self
        }
    }

    thread_local! {
        static BOUND_ORDINAL: Cell<Option<c_int>> = const { Cell::new(None) };
    }

    /// Makes `ordinal` this host thread's current HIP device through `set_device`, skipping the call
    /// when the thread already bound it. HIP's current device is per host thread, so every entry that
    /// touches the device binds first. A failed bind clears the memo: the thread's device is unknown.
    fn bind_ordinal<Failure>(ordinal: c_int, set_device: impl FnOnce(c_int) -> Result<(), Failure>) -> Result<(), Failure> {
        if BOUND_ORDINAL.with(Cell::get) == Some(ordinal) {
            return Ok(());
        }
        let result = set_device(ordinal);
        BOUND_ORDINAL.with(|bound| bound.set(result.is_ok().then_some(ordinal)));
        result
    }

    /// Runs `release` on `ordinal`'s device. A failed bind skips the release: a free on the wrong
    /// device is worse than a leak, and a `Drop` cannot report, so the runtime reclaims it at teardown.
    fn release_on_device(api: &HipApi, ordinal: c_int, release: impl FnOnce()) {
        release_after_bind(ordinal, |bound| api.set_device(bound), release);
    }

    fn release_after_bind<Failure>(
        ordinal: c_int,
        set_device: impl FnOnce(c_int) -> Result<(), Failure>,
        release: impl FnOnce(),
    ) {
        if bind_ordinal(ordinal, set_device).is_ok() {
            release();
        }
    }

    impl HipApi {
        fn set_device(&self, ordinal: c_int) -> Result<(), HipDriverError> {
            // SAFETY: callers pass an ordinal bounds-checked against the device count at open.
            let status = unsafe { (self.hip_set_device)(ordinal) };
            if status != HIP_SUCCESS {
                return Err(self.runtime_error("hipSetDevice", status));
            }
            Ok(())
        }

        fn copy(
            &self,
            call: &'static str,
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
            direction: c_int,
        ) -> Result<(), HipDriverError> {
            if bytes == 0 {
                return Ok(());
            }
            // SAFETY: every caller bounds-checked both sides against live allocations or host slices of `bytes`.
            let status = unsafe { (self.hip_memcpy)(destination, source, bytes, direction) };
            if status != HIP_SUCCESS {
                return Err(self.runtime_error(call, status));
            }
            Ok(())
        }
    }

    /// Options every kernel compiles with: ROCm 6.2 hides the `__shfl_*_sync` builtins the emitted text
    /// calls unless this is defined; it is harmless on 6.3 and later.
    const HIPRTC_OPTIONS: &[&str] = &["-DHIP_ENABLE_WARP_SYNC_BUILTINS"];

    /// The kernel disk store's name for the code object `source` compiles to on `arch` under `options`.
    /// The resolved hipRTC flags carry the `--offload-arch`, so another arch or option set is another
    /// key; `compiler` is the hipRTC `(major, minor)` in the key's compiler-version field, so a code
    /// object is reused only by the compiler version that built it. The schema version and framing
    /// are the CUDA path's ([`CacheKey`]).
    fn code_object_key(source: &str, arch: &str, compiler: (i32, i32), options: &[&str]) -> CacheKey {
        let flags = CompileOptions { options: compile_options(Some(arch), options), ..CompileOptions::default() };
        CacheKey::new(source, None, compiler, &flags, (0, 0))
    }

    /// Built functions by entry name, each bucket holding `(source, function)` pairs.
    type FunctionCache<Function> = BTreeMap<String, Vec<(String, Function)>>;

    /// The function built from exactly `source` under `entry`, running `build` only on a miss. Mirrors
    /// the CUDA runtime's `cached_function`: a bucket per entry, matched on source text.
    fn cached_by_source<Function: Clone, Failure>(
        cache: &proxima_primitives::sync::blocking::Mutex<FunctionCache<Function>>,
        entry: &str,
        source: &str,
        build: impl FnOnce() -> Result<Function, Failure>,
    ) -> Result<Function, Failure> {
        let found = cache
            .lock()
            .get(entry)
            .and_then(|bucket| bucket.iter().find(|(built_from, _)| built_from == source))
            .map(|(_, function)| function.clone());
        if let Some(function) = found {
            return Ok(function);
        }
        let function = build()?;
        cache
            .lock()
            .entry(entry.to_owned())
            .or_default()
            .push((source.to_owned(), function.clone()));
        Ok(function)
    }

    /// The shared executor's runtime on a ROCm device: the [`GpuRuntime`] a `DevicePlan` runs on when
    /// the lowering is CUDA's and the kernels render as HIP C++ at `wave`.
    ///
    /// Teaching pointer: the cudarc runtime is the other [`GpuRuntime`]; both are built from the same
    /// executor bodies. Launch parameters come from [`KernelParams`] built over borrowed arguments,
    /// and the function cache is keyed by the kernel's entry and then its exact source text, as the
    /// CUDA runtime's is, so two sources that share an entry name never share a function.
    #[derive(Debug, Clone)]
    pub struct HipRuntime {
        device: Arc<HipDevice>,
        wave: HipWave,
        functions: Arc<proxima_primitives::sync::blocking::Mutex<FunctionCache<HipFunction>>>,
        compile_counters: Arc<super::CompileCounters>,
    }

    impl HipRuntime {
        /// A runtime over an opened `device`, emitting for `wave`.
        #[must_use]
        pub fn new(device: HipDevice, wave: HipWave) -> Self {
            Self {
                device: Arc::new(device),
                wave,
                functions: Arc::new(proxima_primitives::sync::blocking::Mutex::new(FunctionCache::new())),
                compile_counters: Arc::new(super::CompileCounters::new(super::kernel_disk_store())),
            }
        }

        /// The device this runtime launches on.
        #[must_use]
        pub fn device(&self) -> &HipDevice {
            &self.device
        }

        /// The module for `source`, its code object read from the kernel disk store when one built for
        /// this device's arch, this hipRTC version and these options is there. Without a known arch or
        /// compiler version no key can name the target, so the object is compiled and never stored.
        fn code_object_module(&self, source: &str) -> Result<HipModule, HipDriverError> {
            let compile = || -> Result<Vec<u8>, HipDriverError> {
                let code = self.device.compile_code_object(source, HIPRTC_OPTIONS)?;
                self.compile_counters.compilations.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(code)
            };
            let arch = self.device.arch().filter(|arch| !arch.is_empty());
            let code = match (arch, self.device.compiler_version()) {
                (Some(arch), Some(compiler)) => {
                    let key = code_object_key(source, arch, compiler, HIPRTC_OPTIONS);
                    super::stored_or_compiled(&self.compile_counters, &key, compile)?
                }
                _ => compile()?,
            };
            self.device.load_code_object(&code)
        }

        /// Opens `ordinal`, probes the width the device reports (the `warpSize` kernel launched through
        /// this runtime's own launch path), and emits for it. A `configured` width must equal the probe.
        ///
        /// # Errors
        /// The ROCm refusals of [`HipDevice::open`], [`HipDriverError::UnsupportedWaveWidth`]
        /// for a reported width other than 32 or 64, and [`HipDriverError::WaveMismatch`] when
        /// `configured` differs from the reported width.
        pub fn open(ordinal: c_int, configured: Option<HipWave>) -> Result<Self, HipDriverError> {
            let probe = Self::new(HipDevice::open(ordinal)?, HipWave::Wave64);
            let lanes = probe.warp_size()?;
            let probed = HipWave::from_lanes(u64::from(lanes)).ok_or(HipDriverError::UnsupportedWaveWidth(lanes))?;
            let wave = resolve_wave(configured, probed)?;
            Ok(Self { wave, ..probe })
        }

        /// The device's wavefront width, read by launching a one-thread kernel that stores `warpSize`.
        ///
        /// # Errors
        /// Any build, launch or copy error.
        pub fn warp_size(&self) -> Result<u32, HipDriverError> {
            let function = self.build_function(&warp_size_kernel()?)?;
            let mut lanes = self.alloc_zeros::<u32>(1)?;
            let uniforms = self.upload::<u8>(&[0u8; 4])?;
            let mut arguments = self.launch_arguments(&function);
            arguments.push_mut(&mut lanes).push(&uniforms);
            let shape = LaunchShape { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: the probe kernel's parameters are its output buffer then its uniforms, as pushed.
            unsafe { self.launch(arguments, shape) }?;
            self.synchronize()?;
            Ok(self.download(&lanes)?.first().copied().unwrap_or(0))
        }

        fn slice_from<Element: DeviceElement>(buffer: HipBuffer, len: usize) -> HipSlice<Element> {
            HipSlice { buffer: Arc::new(buffer), len, element: core::marker::PhantomData }
        }

        fn byte_len<Element>(len: usize) -> Result<usize, HipDriverError> {
            len.checked_mul(core::mem::size_of::<Element>()).ok_or(HipDriverError::InputTooLarge(len))
        }

        fn check_range<Element>(buffer: &HipSlice<Element>, range: &core::ops::Range<usize>) -> Result<(), HipDriverError> {
            let size = core::mem::size_of::<Element>();
            (range.start <= range.end && range.end <= buffer.len).then_some(()).ok_or(HipDriverError::ViewOutOfRange {
                offset: range.start * size,
                len: range.end.saturating_sub(range.start) * size,
                bytes: buffer.len * size,
            })
        }

        fn window_parts(window: Result<(usize, usize), WindowRefusal>) -> (usize, usize, Option<WindowRefusal>) {
            match window {
                Ok((offset, len)) => (offset, len, None),
                Err(refusal) => (0, 0, Some(refusal)),
            }
        }

        fn sub_view<'any>(pointer: *mut c_void, offset: usize, len: usize) -> HipView<'any> {
            HipView { pointer: offset_pointer(pointer, offset), len, owner: core::marker::PhantomData }
        }
    }

    impl super::CudaDriver<HipRuntime> {
        /// Opens the configured ROCm device and builds the shared executor's driver over it: the same
        /// [`super::CudaDriver`] a CUDA device gets, so `plan_with_config` lowers and runs HIP plans.
        ///
        /// # Errors
        /// The refusals of [`HipRuntime::open`].
        pub fn open(config: &RuntimeConfig, wave: Option<HipWave>) -> Result<Self, HipDriverError> {
            let runtime = HipRuntime::open(open_ordinal(config.hip_device)?, wave)?;
            let baseline_used_bytes = runtime.memory_info().map_or(0, |(free, total)| total.saturating_sub(free) as u64);
            Ok(Self {
                runtime,
                resident_buffers: Arc::new(super::Mutex::new(super::ResidentLedger::default())),
                packed_upload_bytes: Arc::new(core::sync::atomic::AtomicU64::new(0)),
                graph_uploads: Arc::new(core::sync::atomic::AtomicU64::new(0)),
                graph_launches: Arc::new(core::sync::atomic::AtomicU64::new(0)),
                parked_state: Arc::default(),
                parked_appends: Arc::default(),
                baseline_used_bytes,
            })
        }

        /// The device name this driver runs on.
        #[must_use]
        pub fn device_name(&self) -> &str {
            self.runtime.device().name()
        }
    }

    impl GpuRuntime for HipRuntime {
        type Error = HipDriverError;
        type BuildError = HipDriverError;
        type Buffer<Element: DeviceElement> = HipSlice<Element>;
        type View<'a, Element: DeviceElement> = HipWindow<'a, Element>;
        type ViewMut<'a, Element: DeviceElement> = HipWindowMut<'a, Element>;
        type Event = core::convert::Infallible;
        type Function = HipFunction;
        type Module = HipModule;
        type Image = HipImage;
        type Launch<'a> = HipLaunch<'a>;

        fn alloc_zeros<Element: DeviceElement>(&self, len: usize) -> Result<HipSlice<Element>, HipDriverError> {
            let buffer = self.device.allocate_zeroed_bytes(Self::byte_len::<Element>(len)?)?;
            Ok(Self::slice_from(buffer, len))
        }

        fn upload<Element: DeviceElement>(&self, values: &[Element]) -> Result<HipSlice<Element>, HipDriverError> {
            // SAFETY: the elements are plain device-representable values, read as their own bytes.
            let bytes = unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values)) };
            let buffer = self.device.upload_bytes(bytes)?;
            Ok(Self::slice_from(buffer, values.len()))
        }

        fn upload_into<Element: DeviceElement>(&self, values: &[Element], buffer: &mut HipSlice<Element>) -> Result<(), HipDriverError> {
            self.upload_at(values, buffer, 0)
        }

        fn upload_at<Element: DeviceElement>(&self, values: &[Element], buffer: &mut HipSlice<Element>, offset: usize) -> Result<(), HipDriverError> {
            let range = offset..offset + values.len();
            Self::check_range(buffer, &range)?;
            self.device.bind()?;
            let size = core::mem::size_of::<Element>();
            let destination = offset_pointer(buffer.buffer.pointer, offset * size);
            self.device.api.copy("hipMemcpy", destination, values.as_ptr().cast::<c_void>(), core::mem::size_of_val(values), MEMCPY_HOST_TO_DEVICE)
        }

        fn download<Element: DeviceElement>(&self, buffer: &HipSlice<Element>) -> Result<Vec<Element>, HipDriverError> {
            self.download_range(buffer, 0..buffer.len)
        }

        fn download_range<Element: DeviceElement>(&self, buffer: &HipSlice<Element>, range: core::ops::Range<usize>) -> Result<Vec<Element>, HipDriverError> {
            Self::check_range(buffer, &range)?;
            self.device.bind()?;
            let size = core::mem::size_of::<Element>();
            let mut values = Vec::<Element>::with_capacity(range.len());
            let source = offset_pointer(buffer.buffer.pointer, range.start * size).cast_const();
            self.device.api.copy("hipMemcpy", values.as_mut_ptr().cast::<c_void>(), source, range.len() * size, MEMCPY_DEVICE_TO_HOST)?;
            // SAFETY: the copy initialized `range.len()` elements, and every bit pattern is a valid element.
            unsafe { values.set_len(range.len()) };
            Ok(values)
        }

        fn copy_range<Element: DeviceElement>(
            &self,
            source: &HipSlice<Element>,
            range: core::ops::Range<usize>,
            destination: &mut HipSlice<Element>,
            offset: usize,
        ) -> Result<(), HipDriverError> {
            Self::check_range(source, &range)?;
            Self::check_range(destination, &(offset..offset + range.len()))?;
            self.device.bind()?;
            let size = core::mem::size_of::<Element>();
            let from = offset_pointer(source.buffer.pointer, range.start * size).cast_const();
            let to = offset_pointer(destination.buffer.pointer, offset * size);
            self.device.api.copy("hipMemcpy", to, from, range.len() * size, MEMCPY_DEVICE_TO_DEVICE)
        }

        fn build_function(&self, kernel: &CudaKernel) -> Result<HipFunction, HipDriverError> {
            cached_by_source(&self.functions, &kernel.entry, &kernel.source, || {
                let module = Arc::new(self.code_object_module(&kernel.source)?);
                Ok(HipFunction { handle: module.function(&kernel.entry)?, _module: module })
            })
        }

        fn image_from_source(source: &str) -> HipImage {
            HipImage(source.to_owned())
        }

        fn load_module(&self, image: HipImage) -> Result<Arc<HipModule>, HipDriverError> {
            Ok(Arc::new(self.code_object_module(&image.0)?))
        }

        fn load_function(&self, module: &Arc<HipModule>, entry: &str) -> Result<HipFunction, HipDriverError> {
            self.device.bind()?;
            Ok(HipFunction { handle: module.function(entry)?, _module: Arc::clone(module) })
        }

        fn launch_arguments<'a>(&'a self, function: &'a HipFunction) -> HipLaunch<'a> {
            HipLaunch { function, parameters: KernelParams::new(), refusal: None }
        }

        unsafe fn launch(&self, mut arguments: HipLaunch<'_>, shape: LaunchShape) -> Result<(), HipDriverError> {
            if let Some(refusal) = arguments.refusal {
                return Err(refusal.error());
            }
            self.device.bind()?;
            let api = &self.device.api;
            // SAFETY: the caller matched the arguments to the kernel's parameter list; the params borrow
            // every argument until this call returns, and the launch is on the default stream.
            let status = unsafe {
                (api.hip_module_launch_kernel)(
                    arguments.function.handle,
                    shape.grid_dim.0,
                    shape.grid_dim.1,
                    shape.grid_dim.2,
                    shape.block_dim.0,
                    shape.block_dim.1,
                    shape.block_dim.2,
                    shape.shared_mem_bytes,
                    ptr::null_mut(),
                    arguments.parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if status != HIP_SUCCESS {
                return Err(api.runtime_error("hipModuleLaunchKernel", status));
            }
            Ok(())
        }

        fn executes(&self) -> proxima_tensor::Executes {
            crate::cuda::executes(&self.emit_target())
        }

        fn emit_target(&self) -> crate::cuda::target::Target {
            crate::cuda::target::Target::hip_wave(self.wave.lanes())
        }

        fn emit_device(&self) -> EmitDevice {
            EmitDevice::default()
        }

        fn len<Element: DeviceElement>(buffer: &HipSlice<Element>) -> usize {
            buffer.len
        }

        fn view<Element: DeviceElement>(buffer: &HipSlice<Element>, range: core::ops::Range<usize>) -> HipWindow<'_, Element> {
            let (offset, len, refusal) = Self::window_parts(element_window(buffer.len, &range, core::mem::size_of::<Element>()));
            HipWindow { view: Self::sub_view(buffer.buffer.pointer, offset, len), refusal, element: core::marker::PhantomData }
        }

        fn view_mut<Element: DeviceElement>(buffer: &mut HipSlice<Element>, range: core::ops::Range<usize>) -> HipWindowMut<'_, Element> {
            let (offset, len, refusal) = Self::window_parts(element_window(buffer.len, &range, core::mem::size_of::<Element>()));
            HipWindowMut {
                view: Self::sub_view(buffer.buffer.pointer, offset, len),
                refusal,
                exclusive: core::marker::PhantomData,
                element: core::marker::PhantomData,
            }
        }

        fn view_mut_len<Element: DeviceElement>(view: &HipWindowMut<'_, Element>) -> usize {
            view.view.len / core::mem::size_of::<Element>()
        }

        fn view_mut_range<'view, Element: DeviceElement>(view: &'view mut HipWindowMut<'_, Element>, range: core::ops::Range<usize>) -> HipWindowMut<'view, Element> {
            let (offset, len, refusal) =
                Self::window_parts(element_window(Self::view_mut_len(view), &range, core::mem::size_of::<Element>()));
            HipWindowMut {
                view: Self::sub_view(view.view.pointer, offset, len),
                refusal: view.refusal.or(refusal),
                exclusive: core::marker::PhantomData,
                element: core::marker::PhantomData,
            }
        }

        fn bytes_mut<Element: DeviceElement>(buffer: &mut HipSlice<Element>) -> Option<HipWindowMut<'_, u8>> {
            let bytes = buffer.len * core::mem::size_of::<Element>();
            Some(HipWindowMut {
                view: Self::sub_view(buffer.buffer.pointer, 0, bytes),
                refusal: None,
                exclusive: core::marker::PhantomData,
                element: core::marker::PhantomData,
            })
        }

        fn record_event(&self) -> Result<core::convert::Infallible, HipDriverError> {
            Err(HipDriverError::EventsUnsupported)
        }

        fn elapsed_ms(_start: &core::convert::Infallible, _end: &core::convert::Infallible) -> Result<EventMillis, HipDriverError> {
            Err(HipDriverError::EventsUnsupported)
        }

        fn synchronize(&self) -> Result<(), HipDriverError> {
            self.device.synchronize()
        }

        fn memory_info(&self) -> Result<(usize, usize), HipDriverError> {
            self.device.bind()?;
            let api = &self.device.api;
            let query = api.hip_mem_get_info.ok_or(HipDriverError::MemoryInfoUnavailable)?;
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: both out-parameters are valid for the call.
            let status = unsafe { query(&mut free, &mut total) };
            if status != HIP_SUCCESS {
                return Err(api.runtime_error("hipMemGetInfo", status));
            }
            Ok((free, total))
        }

        fn device_identity(&self) -> Result<String, HipDriverError> {
            Ok(format!("hip:{}:{}", self.device.ordinal, self.device.name))
        }

        fn compile_counters(&self) -> &super::CompileCounters {
            &self.compile_counters
        }
    }

    #[cfg(test)]
    mod runtime_tests {
        use core::convert::Infallible;
        use core::sync::atomic::{AtomicUsize, Ordering};

        use proxima_primitives::sync::blocking::Mutex;

        use super::super::{CompileCounters, stored_or_compiled};
        use super::{FunctionCache, HIPRTC_OPTIONS, HipDriverError, WindowRefusal, cached_by_source, code_object_key, compile_options, element_window};
        use crate::cuda_ptx_store::PtxStore;

        const SOURCE: &str = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void touch_first() { }\n";
        const COMPILER: (i32, i32) = (6, 2);
        const UPGRADED_COMPILER: (i32, i32) = (6, 3);
        const CODE_OBJECT: &[u8] = b"\x7fELF\x02\x01\x01\x40amdgcn-amd-amdhsa--gfx1100 code object bytes";

        fn compiled(compiles: &AtomicUsize) -> Result<Vec<u8>, Infallible> {
            compiles.fetch_add(1, Ordering::Relaxed);
            Ok(CODE_OBJECT.to_vec())
        }

        #[test]
        fn a_code_object_compiles_once_then_reads_back_from_the_disk_store() {
            let root = tempfile::tempdir().expect("tempdir");
            let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
            let compiles = AtomicUsize::new(0);
            let key = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);

            let first = stored_or_compiled(&counters, &key, || compiled(&compiles));
            let second = stored_or_compiled(&counters, &key, || compiled(&compiles));

            assert_eq!(first, Ok(CODE_OBJECT.to_vec()));
            assert_eq!(second, Ok(CODE_OBJECT.to_vec()), "the stored bytes are the compiled bytes");
            assert_eq!(compiles.load(Ordering::Relaxed), 1, "the second call reads the store and compiles nothing");
            assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 1, "the read-back is counted as a disk hit");
        }

        #[test]
        fn a_different_compiler_version_compiles_again_instead_of_reading_the_stored_object() {
            let root = tempfile::tempdir().expect("tempdir");
            let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
            let compiles = AtomicUsize::new(0);
            let before = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);
            let after = code_object_key(SOURCE, "gfx1100", UPGRADED_COMPILER, HIPRTC_OPTIONS);

            stored_or_compiled(&counters, &before, || compiled(&compiles)).expect("compiles");
            stored_or_compiled(&counters, &after, || compiled(&compiles)).expect("compiles");
            stored_or_compiled(&counters, &after, || compiled(&compiles)).expect("reads back");

            assert_eq!(compiles.load(Ordering::Relaxed), 2, "the upgraded compiler compiles once, then reads its own entry");
            assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn another_arch_or_another_source_compiles_again() {
            let root = tempfile::tempdir().expect("tempdir");
            let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
            let compiles = AtomicUsize::new(0);
            let keys = [
                code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS),
                code_object_key(SOURCE, "gfx90a", COMPILER, HIPRTC_OPTIONS),
                code_object_key("extern \"C\" __global__ void other() {}\n", "gfx1100", COMPILER, HIPRTC_OPTIONS),
                code_object_key(SOURCE, "gfx1100", COMPILER, &[]),
                code_object_key(SOURCE, "gfx1100", UPGRADED_COMPILER, HIPRTC_OPTIONS),
            ];

            for key in &keys {
                stored_or_compiled(&counters, key, || compiled(&compiles)).expect("compiles");
            }

            assert_eq!(compiles.load(Ordering::Relaxed), keys.len(), "no key reads another key's code object");
            assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 0);
        }

        #[test]
        fn a_corrupt_stored_code_object_is_recompiled_not_loaded() {
            let root = tempfile::tempdir().expect("tempdir");
            let counters = CompileCounters::new(PtxStore::Directory(root.path().to_path_buf()));
            let compiles = AtomicUsize::new(0);
            let key = code_object_key(SOURCE, "gfx1100", COMPILER, HIPRTC_OPTIONS);
            std::fs::write(root.path().join(key.to_hex()), b"PXKC truncated").expect("corrupt the entry");

            let image = stored_or_compiled(&counters, &key, || compiled(&compiles));

            assert_eq!(image, Ok(CODE_OBJECT.to_vec()));
            assert_eq!(compiles.load(Ordering::Relaxed), 1, "a rejected entry compiles");
            assert_eq!(counters.disk_cache_hits.load(Ordering::Relaxed), 0);
        }

        fn build_counted(builds: &AtomicUsize, tag: &str) -> Result<String, Infallible> {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok(tag.to_owned())
        }

        #[test]
        fn two_sources_sharing_an_entry_get_distinct_functions() {
            let cache = Mutex::new(FunctionCache::new());
            let builds = AtomicUsize::new(0);

            let first = cached_by_source(&cache, "k_add", "source a", || build_counted(&builds, "built from a"));
            let second = cached_by_source(&cache, "k_add", "source b", || build_counted(&builds, "built from b"));
            let first_again = cached_by_source(&cache, "k_add", "source a", || build_counted(&builds, "rebuilt"));

            assert_eq!(first, Ok(String::from("built from a")));
            assert_eq!(second, Ok(String::from("built from b")));
            assert_eq!(first_again, Ok(String::from("built from a")), "the first source still resolves to its own function");
            assert_eq!(builds.load(Ordering::Relaxed), 2, "a repeated source builds nothing");
        }

        #[test]
        fn every_kernel_compiles_with_the_warp_sync_builtins_enabled() {
            let options = compile_options(Some("gfx1100"), HIPRTC_OPTIONS);

            assert_eq!(options, vec![String::from("--offload-arch=gfx1100"), String::from("-DHIP_ENABLE_WARP_SYNC_BUILTINS")]);
        }

        #[test]
        fn a_window_inside_the_buffer_is_scaled_to_bytes() {
            assert_eq!(element_window(100, &(10..30), 4), Ok((40, 80)));
        }

        #[test]
        fn a_window_past_the_end_is_refused_with_its_byte_extent() {
            let refusal = element_window(100, &(90..130), 4).expect_err("a window past the end is refused, not clamped");

            assert_eq!(refusal, WindowRefusal { offset: 360, len: 160, bytes: 400 });
            assert!(
                matches!(refusal.error(), HipDriverError::ViewOutOfRange { offset: 360, len: 160, bytes: 400 }),
                "the refusal is the typed view error"
            );
        }

        #[test]
        fn a_window_starting_past_the_end_is_refused() {
            assert!(element_window(100, &(120..130), 4).is_err(), "a start past the end would hand the kernel an end pointer");
        }

        #[test]
        fn a_reversed_range_is_refused() {
            let reversed = core::ops::Range { start: 30, end: 10 };

            assert!(element_window(100, &reversed, 4).is_err(), "a reversed range names no window");
        }

        #[test]
        fn an_empty_range_is_an_empty_window_at_its_start() {
            assert_eq!(element_window(100, &(30..30), 1), Ok((30, 0)));
        }
    }

    #[cfg(test)]
    mod rocm_tests {
        use super::*;

        use crate::OperandDtypes;
        use crate::cuda::emit_hip_at_wave;
        use proxima_numeric::{DType, NumericPolicy};
        use proxima_tensor::{
            Extent, IndexMap, Keep, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
        };

        #[test]
        fn hip_compile_options_name_the_open_devices_gcn_arch() {
            assert_eq!(
                compile_options(Some("gfx90a:sramecc-:xnack-"), &[]),
                vec![String::from("--offload-arch=gfx90a:sramecc-:xnack-")]
            );
        }

        #[test]
        fn hip_compile_options_let_an_explicit_arch_override_the_device() {
            assert_eq!(
                compile_options(Some("gfx942"), &["--offload-arch=gfx90a"]),
                vec![String::from("--offload-arch=gfx90a")]
            );
            assert_eq!(
                compile_options(Some("gfx942"), &["--gpu-architecture=gfx90a"]),
                vec![String::from("--gpu-architecture=gfx90a")]
            );
        }

        #[test]
        fn hip_compile_options_omit_the_arch_when_the_property_is_unreadable() {
            assert!(compile_options(None, &[]).is_empty());
            assert_eq!(compile_options(Some(""), &["-O3"]), vec![String::from("-O3")]);
        }

        #[test]
        fn hip_properties_scan_finds_the_gcn_arch_name() {
            let token = b"gfx90a:sramecc-:xnack-";
            let mut properties = [0_u8; HIP_DEVICE_PROPERTIES_BYTES];
            properties[512..512 + token.len()].copy_from_slice(token);
            properties[512 + token.len()] = 0;

            assert_eq!(gcn_arch_name(&properties), Some("gfx90a:sramecc-:xnack-"));
            assert_eq!(gcn_arch_name(b"no architecture token here\0"), None);
            assert_eq!(gcn_arch_name(b"gfx\0"), None, "a bare gfx is not an arch name");
        }

        #[test]
        fn hip_numeric_policy_is_the_conflaguration_policy_with_the_shared_narrowing() {
            let numeric = RuntimeConfig::default().numeric;

            assert_eq!(
                numeric_policy(numeric),
                numeric.with_accumulator_narrowing().with_flash_narrowing(true)
            );
        }

        #[test]
        fn hip_open_ordinal_converts_the_configured_device() {
            assert!(matches!(open_ordinal(1), Ok(1)));
            assert!(matches!(open_ordinal(i32::MAX as u32), Ok(i32::MAX)));
            assert!(matches!(
                open_ordinal(i32::MAX as u32 + 1),
                Err(HipDriverError::DeviceOrdinalTooLarge(_))
            ));
            assert!(matches!(
                open_ordinal(u32::MAX),
                Err(HipDriverError::DeviceOrdinalTooLarge(u32::MAX))
            ));
        }

        #[test]
        fn a_view_window_ends_inside_the_allocation_or_is_refused() {
            assert_eq!(window_end(64, 0, 64), Some(64), "the whole allocation is a window");
            assert_eq!(window_end(64, 16, 48), Some(64), "a tail window ends on the last byte");
            assert_eq!(window_end(64, 64, 0), Some(64), "an empty window at the end is legal");
            assert_eq!(window_end(64, 16, 49), None, "one byte past the end is refused");
            assert_eq!(window_end(64, 65, 0), None, "an offset past the end is refused");
            assert_eq!(window_end(64, usize::MAX, 2), None, "an overflowing window is refused, not wrapped");
        }

        #[test]
        fn a_view_pointer_advances_by_bytes_not_elements() {
            let base = 0x1000 as *mut c_void;

            assert_eq!(offset_pointer(base, 0), base);
            assert_eq!(offset_pointer(base, 12) as usize, 0x100c);
            assert_eq!(offset_pointer(base, 4096) as usize, 0x2000);
        }

        #[test]
        fn availability_is_whether_the_runtime_loads() {
            assert_eq!(HipApi::is_available(), HipApi::load().is_ok());
            if !HipApi::is_available() {
                assert!(matches!(
                    HipApi::load(),
                    Err(HipDriverError::RuntimeNotInstalled { .. } | HipDriverError::RtcNotInstalled { .. })
                ));
            }
        }

        #[test]
        fn an_absent_runtime_is_a_typed_error_naming_the_sonames_tried() {
            match HipApi::load() {
                // a host with ROCm installed: the absent-runtime contract is
                // unexercised and the device test below is the live one
                Ok(_) => {}
                Err(HipDriverError::RuntimeNotInstalled { tried }) => {
                    assert!(tried.contains("amdhip64"), "tried: {tried}");
                }
                Err(other) => panic!("unexpected load error: {other:?}"),
            }
        }

        #[test]
        fn the_windows_hiprtc_name_is_derived_from_the_runtime_version_with_a_plain_fallback() {
            assert_eq!(
                hiprtc_windows_candidates(50_300_000),
                vec![String::from("hiprtc0503.dll"), String::from("hiprtc.dll")]
            );
            assert_eq!(
                hiprtc_windows_candidates(60_241_133),
                vec![String::from("hiprtc0602.dll"), String::from("hiprtc.dll")]
            );
            assert_eq!(hiprtc_windows_candidates(0), vec![String::from("hiprtc.dll")]);
        }

        fn count_binds(ordinals: &[c_int]) -> Vec<c_int> {
            let ordinals = ordinals.to_vec();
            std::thread::spawn(move || {
                let mut calls = Vec::new();
                for ordinal in ordinals {
                    bind_ordinal(ordinal, |bound| {
                        calls.push(bound);
                        Ok::<(), HipDriverError>(())
                    })
                    .expect("the stand-in bind succeeds");
                }
                calls
            })
            .join()
            .expect("the bind thread runs")
        }

        #[test]
        fn a_thread_binds_its_device_once_and_skips_the_repeat() {
            assert_eq!(count_binds(&[1, 1, 1]), vec![1]);
        }

        #[test]
        fn a_different_ordinal_binds_again_and_a_return_rebinds() {
            assert_eq!(count_binds(&[0, 0, 1, 1, 0]), vec![0, 1, 0]);
        }

        #[test]
        fn a_failed_bind_is_not_remembered() {
            let calls = std::thread::spawn(|| {
                let mut calls = 0;
                let mut attempt = |ok: bool| {
                    bind_ordinal(2, |_| {
                        calls += 1;
                        if ok { Ok(()) } else { Err(HipDriverError::BindingContract) }
                    })
                };
                assert!(attempt(false).is_err());
                assert!(attempt(true).is_ok());
                assert!(attempt(true).is_ok());
                calls
            })
            .join()
            .expect("the bind thread runs");
            assert_eq!(calls, 2);
        }

        #[test]
        fn a_release_runs_only_after_its_device_is_bound() {
            let (bound, released_after_bind, released_after_failure) = std::thread::spawn(|| {
                let mut bound = Vec::new();
                let mut released_after_bind = false;
                release_after_bind(
                    3,
                    |ordinal| {
                        bound.push(ordinal);
                        Ok::<(), HipDriverError>(())
                    },
                    || released_after_bind = true,
                );
                let mut released_after_failure = false;
                release_after_bind(4, |_| Err(HipDriverError::BindingContract), || released_after_failure = true);
                (bound, released_after_bind, released_after_failure)
            })
            .join()
            .expect("the release thread runs");
            assert_eq!(bound, vec![3], "the drop binds the object's own ordinal");
            assert!(released_after_bind, "a bound device runs the release");
            assert!(!released_after_failure, "a failed bind skips the release");
        }

        #[test]
        fn an_unconfigured_wave_takes_the_probed_width() {
            assert_eq!(resolve_wave(None, HipWave::Wave32).expect("probe wins"), HipWave::Wave32);
            assert_eq!(resolve_wave(None, HipWave::Wave64).expect("probe wins"), HipWave::Wave64);
        }

        #[test]
        fn a_configured_wave_equal_to_the_probe_is_accepted() {
            assert_eq!(resolve_wave(Some(HipWave::Wave32), HipWave::Wave32).expect("equal"), HipWave::Wave32);
        }

        #[test]
        fn a_configured_wave_that_differs_from_the_device_names_both_widths() {
            let error = resolve_wave(Some(HipWave::Wave64), HipWave::Wave32).expect_err("mismatch refused");
            assert!(matches!(error, HipDriverError::WaveMismatch { configured: 64, device: 32 }), "{error:?}");
            let message = error.to_string();
            assert!(message.contains("64") && message.contains("32"), "{message}");
        }

        #[test]
        fn the_smoke_matches_the_host_oracle_when_a_device_is_present() {
            let Ok(driver) = shared_driver(&RuntimeConfig::default(), None) else {
                return;
            };
            let input = [0.0f32, 1.0, -2.5, 42.0, 1e6];
            let output = driver.smoke_add_one(&input).expect("the shared smoke kernel runs through hiprtc");
            assert_eq!(output.len(), input.len());
            for (index, (found, expected)) in output.iter().zip(input.iter()).enumerate() {
                assert!(
                    (found - (expected + 1.0)).abs() < 1e-6,
                    "element {index}: {found} vs {}",
                    expected + 1.0
                );
            }
        }

        /// hiprtc 9.x static-asserts that a `_sync` mask is 8 bytes at every wavefront width. This
        /// compiles the emitted reduce at both widths through the same runtime path a plan uses;
        /// a host without ROCm names the skip instead of pretending to have compiled.
        #[test]
        fn hiprtc_compiles_the_emitted_wave32_and_wave64_masks() {
            let mut program = Vec::new();
            let input = append(
                &mut program,
                Op::Input {
                    dtype: DType::FLOAT32,
                    shape: vec![Extent::Static(4), Extent::Static(4096)],
                    name: None,
                },
            );
            let root = append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::FLOAT32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: input,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[0])),
                    keep: Keep::Reduce,
                    name: Some("hiprtc_mask_reduce".into()),
                }),
            );
            let shapes = infer(&program, &[]).expect("the reduce infers");
            let bound = bind(&program, &shapes, &[root], NumericPolicy::llama_relaxed())
                .expect("the reduce binds under the reassociation policy");
            let packed = BTreeMap::new();
            let device = match HipDevice::open(0) {
                Ok(device) => device,
                Err(HipDriverError::RuntimeNotInstalled { tried }) => {
                    eprintln!("skipped: rocm runtime not installed ({tried})");
                    return;
                }
                Err(error) => panic!("a ROCm host failed to open device 0: {error}"),
            };
            for wave in [HipWave::Wave32, HipWave::Wave64] {
                let kernel = emit_hip_at_wave(
                    wave,
                    &bound[0],
                    &packed,
                    NumericPolicy::llama_relaxed(),
                    EmitDevice::default(),
                    &OperandDtypes::new(),
                )
                .expect("the reduce emits");
                device
                    .compile(&kernel.source, &[])
                    .unwrap_or_else(|error| panic!("{wave:?} mask did not compile: {error}"));
            }
        }
    }
}

/// A [`GpuRuntime`] that moves no data and runs no kernel: it appends every device call to a shared
/// log, so a test can prove what the shared executor uploads, downloads and launches with no GPU.
///
/// Teaching pointer: this is the third runtime of the one executor, beside the cudarc and HIP ones.
/// It binds the program CUDA binds ([`CudarcRuntime::cuda_grant`], [`CudarcRuntime::cuda_target`]), so
/// the plan it builds is the plan a CUDA device would run.
#[cfg(test)]
mod recording_runtime {
    use super::*;

    use core::convert::Infallible;
    use core::marker::PhantomData;
    use core::mem::zeroed;

    use crate::gpu_runtime::EventMillis;

    use proxima_tensor::{Extent, IndexMap, Keep, Reduce, ReduceInit, ScalarOp, append, map};

    /// One launch argument: a buffer the kernel reads, a buffer it writes, or a scalar.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Argument {
        Read(u64),
        Write(u64),
        Scalar,
    }

    /// One device call, with the buffer identity and the byte count it moved.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Event {
        Alloc {
            buffer: u64,
            bytes: usize,
        },
        Upload {
            buffer: u64,
            bytes: usize,
        },
        UploadInPlace {
            buffer: u64,
            bytes: usize,
        },
        Download {
            buffer: u64,
            bytes: usize,
        },
        DeviceCopy {
            source: u64,
            destination: u64,
            bytes: usize,
        },
        Launch {
            entry: String,
            source: String,
            arguments: Vec<Argument>,
        },
        Synchronize,
    }

    #[derive(Debug, thiserror::Error)]
    enum RecordingFailure {
        #[error("the recording runtime keeps no device events")]
        EventsUnsupported,
        #[error("window {start}..{end} lies outside a buffer of {len} elements")]
        OutOfRange {
            start: usize,
            end: usize,
            len: usize,
        },
    }

    impl From<RecordingFailure> for CudaDriverError {
        fn from(failure: RecordingFailure) -> Self {
            Self::Driver(failure.to_string())
        }
    }

    #[derive(Debug, Clone)]
    struct RecordingRuntime {
        log: Arc<Mutex<Vec<Event>>>,
        next_buffer: Arc<AtomicU64>,
        compile_counters: Arc<CompileCounters>,
        target: Target,
    }

    impl RecordingRuntime {
        fn new() -> Self {
            Self::on_target(CudarcRuntime::cuda_target())
        }

        fn on_target(target: Target) -> Self {
            Self {
                target,
                log: Arc::default(),
                next_buffer: Arc::new(AtomicU64::new(1)),
                compile_counters: Arc::new(CompileCounters::new(PtxStore::Disabled)),
            }
        }

        fn record(&self, event: Event) {
            self.log.lock().push(event);
        }

        fn fresh<Element: DeviceElement>(&self, len: usize) -> RecordingBuffer<Element> {
            let buffer = self.next_buffer.fetch_add(1, Ordering::Relaxed);
            RecordingBuffer {
                id: buffer,
                len,
                element: PhantomData,
            }
        }

        fn check(range: &Range<usize>, len: usize) -> Result<(), RecordingFailure> {
            if range.start <= range.end && range.end <= len {
                return Ok(());
            }
            Err(RecordingFailure::OutOfRange {
                start: range.start,
                end: range.end,
                len,
            })
        }

        fn bytes_of<Element>(elements: usize) -> usize {
            elements * size_of::<Element>()
        }

        fn zeros<Element: DeviceElement>(len: usize) -> Vec<Element> {
            // SAFETY: `DeviceElement` requires `ValidAsZeroBits`, so the all-zero pattern is a valid element.
            vec![unsafe { zeroed::<Element>() }; len]
        }
    }

    /// A buffer's identity and length; the bytes are never kept, since no kernel reads them.
    #[derive(Debug, Clone)]
    struct RecordingBuffer<Element> {
        id: u64,
        len: usize,
        element: PhantomData<fn() -> Element>,
    }

    struct RecordingView<'buffer, Element> {
        id: u64,
        borrow: PhantomData<&'buffer ()>,
        element: PhantomData<fn() -> Element>,
    }

    struct RecordingViewMut<'buffer, Element> {
        id: u64,
        len: usize,
        borrow: PhantomData<&'buffer mut ()>,
        element: PhantomData<fn() -> Element>,
    }

    #[derive(Debug, Clone)]
    struct RecordingFunction {
        entry: String,
        source: String,
    }

    struct RecordingModule {
        source: String,
    }

    struct RecordingLaunch<'launch> {
        function: &'launch RecordingFunction,
        arguments: Vec<Argument>,
    }

    impl<'launch> LaunchArguments<'launch, RecordingRuntime> for RecordingLaunch<'launch> {
        fn push<Element: DeviceElement>(
            &mut self,
            buffer: &'launch RecordingBuffer<Element>,
        ) -> &mut Self {
            self.arguments.push(Argument::Read(buffer.id));
            self
        }

        fn push_mut<Element: DeviceElement>(
            &mut self,
            buffer: &'launch mut RecordingBuffer<Element>,
        ) -> &mut Self {
            self.arguments.push(Argument::Write(buffer.id));
            self
        }

        fn push_view<Element: DeviceElement>(
            &mut self,
            view: &'launch RecordingView<'_, Element>,
        ) -> &mut Self {
            self.arguments.push(Argument::Read(view.id));
            self
        }

        fn push_view_mut<Element: DeviceElement>(
            &mut self,
            view: &'launch mut RecordingViewMut<'_, Element>,
        ) -> &mut Self {
            self.arguments.push(Argument::Write(view.id));
            self
        }

        fn push_value<Element: DeviceElement>(&mut self, _value: &'launch Element) -> &mut Self {
            self.arguments.push(Argument::Scalar);
            self
        }
    }

    impl GpuRuntime for RecordingRuntime {
        type Error = RecordingFailure;
        type BuildError = RecordingFailure;
        type Buffer<Element: DeviceElement> = RecordingBuffer<Element>;
        type View<'a, Element: DeviceElement> = RecordingView<'a, Element>;
        type ViewMut<'a, Element: DeviceElement> = RecordingViewMut<'a, Element>;
        type Event = Infallible;
        type Function = RecordingFunction;
        type Module = RecordingModule;
        type Image = String;
        type Launch<'a> = RecordingLaunch<'a>;

        fn alloc_zeros<Element: DeviceElement>(
            &self,
            len: usize,
        ) -> Result<RecordingBuffer<Element>, RecordingFailure> {
            let buffer = self.fresh::<Element>(len);
            self.record(Event::Alloc {
                buffer: buffer.id,
                bytes: Self::bytes_of::<Element>(len),
            });
            Ok(buffer)
        }

        fn upload<Element: DeviceElement>(
            &self,
            values: &[Element],
        ) -> Result<RecordingBuffer<Element>, RecordingFailure> {
            let buffer = self.fresh::<Element>(values.len());
            self.record(Event::Upload {
                buffer: buffer.id,
                bytes: size_of_val(values),
            });
            Ok(buffer)
        }

        fn upload_into<Element: DeviceElement>(
            &self,
            values: &[Element],
            buffer: &mut RecordingBuffer<Element>,
        ) -> Result<(), RecordingFailure> {
            self.upload_at(values, buffer, 0)
        }

        fn upload_at<Element: DeviceElement>(
            &self,
            values: &[Element],
            buffer: &mut RecordingBuffer<Element>,
            offset: usize,
        ) -> Result<(), RecordingFailure> {
            Self::check(&(offset..offset + values.len()), buffer.len)?;
            self.record(Event::UploadInPlace {
                buffer: buffer.id,
                bytes: size_of_val(values),
            });
            Ok(())
        }

        fn download<Element: DeviceElement>(
            &self,
            buffer: &RecordingBuffer<Element>,
        ) -> Result<Vec<Element>, RecordingFailure> {
            self.download_range(buffer, 0..buffer.len)
        }

        fn download_range<Element: DeviceElement>(
            &self,
            buffer: &RecordingBuffer<Element>,
            range: Range<usize>,
        ) -> Result<Vec<Element>, RecordingFailure> {
            Self::check(&range, buffer.len)?;
            self.record(Event::Download {
                buffer: buffer.id,
                bytes: Self::bytes_of::<Element>(range.len()),
            });
            Ok(Self::zeros(range.len()))
        }

        fn copy_range<Element: DeviceElement>(
            &self,
            source: &RecordingBuffer<Element>,
            range: Range<usize>,
            destination: &mut RecordingBuffer<Element>,
            offset: usize,
        ) -> Result<(), RecordingFailure> {
            Self::check(&range, source.len)?;
            Self::check(&(offset..offset + range.len()), destination.len)?;
            self.record(Event::DeviceCopy {
                source: source.id,
                destination: destination.id,
                bytes: Self::bytes_of::<Element>(range.len()),
            });
            Ok(())
        }

        fn build_function(
            &self,
            kernel: &CudaKernel,
        ) -> Result<RecordingFunction, RecordingFailure> {
            Ok(RecordingFunction {
                entry: kernel.entry.clone(),
                source: kernel.source.clone(),
            })
        }

        fn image_from_source(source: &str) -> String {
            source.to_owned()
        }

        fn load_module(&self, image: String) -> Result<Arc<RecordingModule>, RecordingFailure> {
            Ok(Arc::new(RecordingModule { source: image }))
        }

        fn load_function(
            &self,
            module: &Arc<RecordingModule>,
            entry: &str,
        ) -> Result<RecordingFunction, RecordingFailure> {
            Ok(RecordingFunction {
                entry: entry.to_owned(),
                source: module.source.clone(),
            })
        }

        fn launch_arguments<'a>(&'a self, function: &'a RecordingFunction) -> RecordingLaunch<'a> {
            RecordingLaunch {
                function,
                arguments: Vec::new(),
            }
        }

        unsafe fn launch(
            &self,
            arguments: RecordingLaunch<'_>,
            _shape: LaunchShape,
        ) -> Result<(), RecordingFailure> {
            self.record(Event::Launch {
                entry: arguments.function.entry.clone(),
                source: arguments.function.source.clone(),
                arguments: arguments.arguments,
            });
            Ok(())
        }

        fn executes(&self) -> Executes {
            cuda_executes(&self.target)
        }

        fn emit_target(&self) -> Target {
            self.target
        }

        fn emit_device(&self) -> EmitDevice {
            EmitDevice::default()
        }

        fn len<Element: DeviceElement>(buffer: &RecordingBuffer<Element>) -> usize {
            buffer.len
        }

        fn view<Element: DeviceElement>(
            buffer: &RecordingBuffer<Element>,
            _range: Range<usize>,
        ) -> RecordingView<'_, Element> {
            RecordingView {
                id: buffer.id,
                borrow: PhantomData,
                element: PhantomData,
            }
        }

        fn view_mut<Element: DeviceElement>(
            buffer: &mut RecordingBuffer<Element>,
            range: Range<usize>,
        ) -> RecordingViewMut<'_, Element> {
            RecordingViewMut {
                id: buffer.id,
                len: range.len(),
                borrow: PhantomData,
                element: PhantomData,
            }
        }

        fn view_mut_len<Element: DeviceElement>(view: &RecordingViewMut<'_, Element>) -> usize {
            view.len
        }

        fn view_mut_range<'view, Element: DeviceElement>(
            view: &'view mut RecordingViewMut<'_, Element>,
            range: Range<usize>,
        ) -> RecordingViewMut<'view, Element> {
            RecordingViewMut {
                id: view.id,
                len: range.len(),
                borrow: PhantomData,
                element: PhantomData,
            }
        }

        fn bytes_mut<Element: DeviceElement>(
            buffer: &mut RecordingBuffer<Element>,
        ) -> Option<RecordingViewMut<'_, u8>> {
            Some(RecordingViewMut {
                id: buffer.id,
                len: Self::bytes_of::<Element>(buffer.len),
                borrow: PhantomData,
                element: PhantomData,
            })
        }

        fn record_event(&self) -> Result<Infallible, RecordingFailure> {
            Err(RecordingFailure::EventsUnsupported)
        }

        fn elapsed_ms(
            _start: &Infallible,
            _end: &Infallible,
        ) -> Result<EventMillis, RecordingFailure> {
            Err(RecordingFailure::EventsUnsupported)
        }

        fn synchronize(&self) -> Result<(), RecordingFailure> {
            self.record(Event::Synchronize);
            Ok(())
        }

        fn memory_info(&self) -> Result<(usize, usize), RecordingFailure> {
            Ok((1 << 40, 1 << 40))
        }

        fn device_identity(&self) -> Result<String, RecordingFailure> {
            Ok(String::from("recording:0"))
        }

        fn compile_counters(&self) -> &CompileCounters {
            &self.compile_counters
        }
    }

    impl CudaDriver<RecordingRuntime> {
        /// The shared executor's driver over the recording runtime: the literal
        /// `CudaDriver::<HipRuntime>::open` builds, with a log in place of a device.
        fn recording() -> Self {
            Self::recording_on(RecordingRuntime::new())
        }

        fn recording_on(runtime: RecordingRuntime) -> Self {
            Self {
                runtime,
                resident_buffers: Arc::new(Mutex::new(ResidentLedger::default())),
                packed_upload_bytes: Arc::new(AtomicU64::new(0)),
                graph_uploads: Arc::new(AtomicU64::new(0)),
                graph_launches: Arc::new(AtomicU64::new(0)),
                parked_state: Arc::default(),
                parked_appends: Arc::default(),
                baseline_used_bytes: 0,
            }
        }

        fn events(&self) -> Vec<Event> {
            self.runtime.log.lock().clone()
        }
    }

    const INPUT: u32 = 8;
    const HIDDEN: u32 = 5;
    const OUTPUT: u32 = 3;

    fn input_node(program: &mut Vec<Op>, rows: u32, columns: u32, name: &str) -> NodeId {
        append(
            program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(rows), Extent::Static(columns)],
                name: Some(name.into()),
            },
        )
    }

    fn matmul(program: &mut Vec<Op>, weight: NodeId, activation: NodeId, name: &str) -> NodeId {
        let product = append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some(name.into()),
            }),
        )
    }

    /// `out = (w1 . (w0 . x)) + (w1 . (w0 . x))`: two matmuls over named weights and an elementwise add.
    fn two_layer_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight_first = input_node(&mut program, HIDDEN, INPUT, "w0");
        let weight_second = input_node(&mut program, OUTPUT, HIDDEN, "w1");
        let activation = input_node(&mut program, INPUT, 1, "x");
        let hidden = matmul(&mut program, weight_first, activation, "hidden");
        let logits = matmul(&mut program, weight_second, hidden, "logits");
        let out = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                operands: vec![
                    (logits, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (logits, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: Some("out".into()),
            },
        );
        (program, out)
    }

    fn weight_values(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as f32 + seed) * 0.37).sin())
            .collect()
    }

    fn named_blocks<'block, Value>(
        weight_first: &'block [Value],
        weight_second: &'block [Value],
        activation: &'block [Value],
    ) -> [(&'static str, QuantizedBlock<'block>); 3]
    where
        QuantizedBlock<'block>: From<&'block [Value]>,
    {
        [
            ("w0", QuantizedBlock::from(weight_first)),
            ("w1", QuantizedBlock::from(weight_second)),
            ("x", QuantizedBlock::from(activation)),
        ]
    }

    fn uploads_of(events: &[Event], bytes: usize) -> usize {
        events.iter().filter(|event| matches!(event, Event::Upload { bytes: found, .. } | Event::UploadInPlace { bytes: found, .. } if *found == bytes)).count()
    }

    fn launches_of(events: &[Event]) -> Vec<(usize, &Vec<Argument>)> {
        events
            .iter()
            .enumerate()
            .filter_map(|(position, event)| match event {
                Event::Launch { arguments, .. } => Some((position, arguments)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_shared_executor_uploads_each_weight_once_and_downloads_only_outputs() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(&weight_first, &weight_second, &activation);
        let driver = CudaDriver::<RecordingRuntime>::recording();
        let mut plan = driver
            .plan_with_config(&program, &[], &[out], &named, &RuntimeConfig::default())
            .expect("the recording driver plans the two-layer program");
        plan.mark_resident(&BTreeSet::from(["w0", "w1"]));
        let kernel_count = plan.kernel_launches().len();
        let weight_bytes = [
            size_of_val(weight_first.as_slice()),
            size_of_val(weight_second.as_slice()),
        ];
        let activation_bytes = size_of_val(activation.as_slice());
        assert_eq!(
            plan.executes(),
            CudarcRuntime::cuda_grant(),
            "the plan binds the grant CUDA binds"
        );

        let mut windows = Vec::new();
        for _ in 0..2 {
            let before = driver.events().len();
            plan.execute_named(&named, None)
                .expect("the recording runtime executes the plan");
            windows.push(driver.events()[before..].to_vec());
        }
        let all = driver.events();

        for bytes in weight_bytes {
            assert_eq!(
                uploads_of(&all, bytes),
                1,
                "a {bytes}-byte weight uploads once across both executes"
            );
            assert_eq!(
                uploads_of(&windows[1], bytes),
                0,
                "the second execute re-uploads no weight"
            );
        }
        assert_eq!(
            uploads_of(&windows[0], activation_bytes),
            1,
            "the activation uploads on the first execute"
        );
        assert_eq!(
            uploads_of(&windows[1], activation_bytes),
            1,
            "the activation refreshes on the second execute"
        );

        for (index, window) in windows.iter().enumerate() {
            let launches = launches_of(window);
            assert_eq!(
                launches.len(),
                kernel_count,
                "execute {index} launches the plan's kernel count"
            );
            let downloads: Vec<&Event> = window
                .iter()
                .filter(|event| matches!(event, Event::Download { .. }))
                .collect();
            assert_eq!(
                downloads.len(),
                1,
                "execute {index} downloads exactly the plan's one output: {downloads:?}"
            );
            let Event::Download {
                buffer: downloaded,
                bytes,
            } = downloads[0]
            else {
                unreachable!()
            };
            assert_eq!(
                *bytes,
                OUTPUT as usize * 4,
                "the download is the output's byte count"
            );
            let (last_position, last_arguments) = launches[launches.len() - 1];
            assert!(
                last_arguments.contains(&Argument::Write(*downloaded)),
                "the downloaded buffer is written by the final launch"
            );
            for (_, arguments) in &launches[..launches.len() - 1] {
                assert!(
                    !arguments.contains(&Argument::Write(*downloaded))
                        || arguments.contains(&Argument::Read(*downloaded)),
                    "no earlier launch owns the downloaded buffer: an intermediate is never read back"
                );
            }
            let first_position = launches[0].0;
            let between = &window[first_position..=last_position];
            assert!(
                between.iter().all(|event| !matches!(
                    event,
                    Event::Download { .. } | Event::DeviceCopy { .. }
                )),
                "execute {index} reads back or copies between launches: {between:?}"
            );
            let between_uploads: Vec<usize> = between
                .iter()
                .filter_map(|event| match event {
                    Event::Upload { bytes, .. } | Event::UploadInPlace { bytes, .. } => {
                        Some(*bytes)
                    }
                    _ => None,
                })
                .collect();
            let later_uniforms: Vec<usize> = if index == 0 {
                plan.uniforms.iter().skip(1).map(Vec::len).collect()
            } else {
                Vec::new()
            };
            assert_eq!(
                between_uploads, later_uniforms,
                "execute {index}: the first execute uploads only the later kernels uniform blocks between launches (the second reuses them), never a weight or activation"
            );
        }
        eprintln!(
            "recorded uploads: first execute {:?}, second execute {:?}",
            windows[0]
                .iter()
                .filter(|event| matches!(event, Event::Upload { .. } | Event::UploadInPlace { .. }))
                .collect::<Vec<_>>(),
            windows[1]
                .iter()
                .filter(|event| matches!(event, Event::Upload { .. } | Event::UploadInPlace { .. }))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "recorded downloads: {:?}",
            all.iter()
                .filter(|event| matches!(event, Event::Download { .. }))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn two_named_plans_on_one_driver_upload_each_weight_once_in_total() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(weight_first.as_slice(), weight_second.as_slice(), activation.as_slice());
        let driver = CudaDriver::<RecordingRuntime>::recording();
        let resident = BTreeSet::from(["w0", "w1"]);
        for _ in 0..2 {
            let mut plan = driver
                .plan(&program, &[], &[out], NumericPolicy::default(), &named)
                .expect("the shared named-plan entry plans the two-layer program");
            plan.mark_resident(&resident);
            plan.execute_named(&named, None)
                .expect("the recording runtime executes the plan");
        }
        let all = driver.events();
        for bytes in [size_of_val(weight_first.as_slice()), size_of_val(weight_second.as_slice())] {
            assert_eq!(uploads_of(&all, bytes), 1, "a {bytes}-byte weight uploads once across both plans");
        }
    }

    #[test]
    fn refit_symbols_on_a_named_plan_returns_true_and_uploads_no_weight() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(weight_first.as_slice(), weight_second.as_slice(), activation.as_slice());
        let driver = CudaDriver::<RecordingRuntime>::recording();
        let mut plan = driver
            .plan(&program, &[], &[out], NumericPolicy::default(), &named)
            .expect("the shared named-plan entry plans the two-layer program");
        plan.mark_resident(&BTreeSet::from(["w0", "w1"]));
        plan.execute_named(&named, None)
            .expect("the recording runtime executes the plan");
        let before = driver.events().len();
        let refitted = plan
            .refit_symbols(&[], NumericPolicy::default())
            .expect("refit shape inference succeeds");
        let after = driver.events();
        assert!(refitted, "a named plan carries no constraints, so it refits in place");
        for bytes in [size_of_val(weight_first.as_slice()), size_of_val(weight_second.as_slice())] {
            assert_eq!(uploads_of(&after[before..], bytes), 0, "refit uploads no {bytes}-byte weight");
        }
    }

    fn launched_sources(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Launch { source, .. } => Some(source.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_kernel_source_replaced_after_the_first_execute_is_the_one_the_next_execute_launches() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(&weight_first, &weight_second, &activation);
        let driver = CudaDriver::<RecordingRuntime>::recording();
        let mut plan = driver
            .plan_with_config(&program, &[], &[out], &named, &RuntimeConfig::default())
            .expect("the recording driver plans the two-layer program");
        let original = plan.kernels[0].source.clone();
        let replacement = format!("{original}\n// refit text");

        plan.execute_named(&named, None).expect("the first execute runs");
        let first = driver.events();
        assert!(plan.replace_kernel_source(0, replacement.clone()), "position 0 exists");
        plan.execute_named(&named, None).expect("the second execute runs");
        let all = driver.events();
        let second = &all[first.len()..];

        assert_eq!(launched_sources(&first)[0], original, "the first execute launches the planned source");
        assert_eq!(
            launched_sources(second)[0],
            replacement,
            "the execute after a replacement launches a function built from the new source"
        );
        assert!(!plan.replace_kernel_source(usize::MAX, String::new()), "an absent position replaces nothing");
    }

    #[test]
    fn a_runtime_that_refuses_events_skips_timing_instead_of_failing_the_execute() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(&weight_first, &weight_second, &activation);
        let driver = CudaDriver::<RecordingRuntime>::recording();
        let mut plan = driver
            .plan_with_config(&program, &[], &[out], &named, &RuntimeConfig::default())
            .expect("the recording driver plans the two-layer program");
        let mut launches: Vec<LaunchEvents<RecordingRuntime>> = Vec::new();

        let evaluated = temp_env::with_var("PROXIMA_CUDA_EVENT_TIMING", Some("1"), || {
            plan.execute_named_launch_events(&named, None, Some(&mut launches))
        });

        assert!(evaluated.is_ok(), "event timing on a runtime without events must not fail the execute: {:?}", evaluated.err());
        assert!(launches.is_empty(), "a runtime that records no events returns no timings");
        assert!(!launches_of(&driver.events()).is_empty(), "the execute still launched its kernels");
    }

    #[test]
    fn a_hip_target_runtime_emits_refits_and_compares_text_in_its_own_dialect() {
        let (program, out) = two_layer_program();
        let weight_first = weight_values((HIDDEN * INPUT) as usize, 1.0);
        let weight_second = weight_values((OUTPUT * HIDDEN) as usize, 7.0);
        let activation = weight_values(INPUT as usize, 3.0);
        let named = named_blocks(&weight_first, &weight_second, &activation);
        let driver = CudaDriver::<RecordingRuntime>::recording_on(RecordingRuntime::on_target(Target::hip_wave(64)));
        let mut plan = driver
            .plan_with_config(&program, &[], &[out], &named, &RuntimeConfig::default())
            .expect("the recording driver plans the two-layer program under a HIP target");
        let policy = plan.numeric_policy;
        let device = driver.runtime.emit_device();
        let cuda_differs = (0..plan.resolved.len()).any(|position| {
            let cuda = crate::cuda::emit_on_target(Target::CUDA, &plan.resolved[position], &plan.packed_operands, policy, device, &OperandDtypes::new())
                .expect("the op emits as CUDA too");
            cuda.source != plan.kernels[position].source
        });

        assert!(cuda_differs, "the plan's kernels are HIP text, which CUDA emission does not reproduce");
        for position in 0..plan.resolved.len() {
            let previous = plan.resolved[position].clone();
            assert!(
                plan.emits_same_text(&previous, position, policy).expect("the op re-emits"),
                "position {position}: the comparison emits under the runtime's HIP target, so it matches the plan's own text"
            );
        }
        let before: Vec<String> = plan.kernels.iter().map(|kernel| kernel.source.clone()).collect();
        let bound = plan.resolved.clone();
        for (position, next) in bound.iter().enumerate() {
            plan.follow_rows(position, next, policy, &mut FollowMicros::default())
                .expect("a HIP plan refits its grid under its own target");
        }
        let after: Vec<String> = plan.kernels.iter().map(|kernel| kernel.source.clone()).collect();
        assert_eq!(before, after, "refitting an unchanged op keeps its HIP text");
    }
}
