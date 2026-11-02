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
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::sys as nvrtc_sys;
use cudarc::nvrtc::result as nvrtc_result;
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use proxima_primitives::sync::blocking::Mutex;

use crate::config::{ResourceConstraints, RuntimeConfig};
use crate::cuda::{CudaGridSpec, CudaKernel, EmitDevice, cuda_cached_attention_class};
use crate::cuda_kernel_cache::{
    CacheKey, decode_binary_entry, decode_entry, encode_binary_entry, encode_entry, header_digest,
};
use crate::kernel_cache_config::KernelCacheConfig;
use crate::cuda_ptx_store::PtxStore;
use crate::lowering::{Binding, Codec, PackedOperands, gather_count};
use proxima_tensor::bind::{
    RefreshRefusal, changed_nodes, kv_read_only_by_fused_attention, refit_cached_attention_at,
    RowTemplate, SymbolsTemplate, apply_row_template, apply_symbols_template, refresh_bound_ops,
    refresh_cached_attention, row_template, rows_enumerable, symbols_shapes_template,
    touched_positions, touches_changed,
};
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Evaluated, Keep, NodeId, NumericPolicy, NumericRewrite, Op,
    QuantizedBlock, Shapes, admit, bind_unfused, bind_with_fusion, block_node_ids,
    correct_packed_matmul_layouts, infer, prune_dead, resolve_named_blocks,
};

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
    #[error("CUDA kernel binding contract is not inputs, output, uniforms")]
    BindingContract,
    #[error("CUDA graph buffer for node {0:?} is missing")]
    MissingBuffer(NodeId),
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
}

impl From<DriverError> for CudaDriverError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

// keyed by the caller's (address, length) so a fresh plan or arena finds the weight another plan uploaded; the name only drives eviction.
type ResidentBuffers = BTreeMap<(usize, usize), (String, CudaGraphBuffer)>;

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
struct CompileCounters {
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
enum Completion<'fault> {
    Synchronize,
    Defer(Option<(&'fault CudaSlice<u32>, usize, usize)>),
}

/// One CUDA context and stream owned by a caller's inference runtime.
#[derive(Debug, Clone)]
pub struct CudaDriver {
    context: Arc<CudaContext>,
    device: EmitDevice,
    stream: Arc<CudaStream>,
    // synchronous ffi callers share these short cache mutations; compilation and dispatch happen outside the guard.
    modules: Arc<Mutex<BTreeMap<String, Vec<CachedCudaModule>>>>,
    resident_buffers: Arc<Mutex<ResidentBuffers>>,
    compile_counters: Arc<CompileCounters>,
    packed_upload_bytes: Arc<AtomicU64>,
    graph_uploads: Arc<AtomicU64>,
    graph_launches: Arc<AtomicU64>,
    // carried recurrent state whose plan was evicted: keyed by state input node, valued by its output node and device buffer, so the replacement plan adopts it without a host round trip.
    parked_state: Arc<Mutex<BTreeMap<NodeId, (NodeId, CudaGraphBuffer)>>>,
    // appended-row caches whose plan was evicted: keyed by cache input node, valued by its valid row count and device buffer.
    parked_appends: Arc<Mutex<BTreeMap<NodeId, (u64, CudaGraphBuffer)>>>,
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
struct CachedCudaModule {
    source: String,
    module: Arc<CudaModule>,
    functions: BTreeMap<String, CudaFunction>,
}

/// Persistent f32 storage for a graph's named inputs and intermediate nodes.
/// The map owns allocations across dispatches; a shape change is the only
/// event that replaces a node's device buffer.
#[derive(Debug, Default)]
pub struct CudaF32Arena {
    buffers: BTreeMap<NodeId, CudaGraphBuffer>,
    uniforms: BTreeMap<NodeId, CudaSlice<u8>>,
    uniform_host: BTreeMap<NodeId, Vec<u8>>,
    resident_sources: BTreeMap<NodeId, (usize, usize)>,
    pool: BTreeMap<usize, Vec<CudaSlice<f32>>>,
    allocations: u64,
    launch_profile: LaunchProfile,
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

impl PoolSchedule {
    fn build(resolved: &[BoundOp], kernels: &[CudaKernel], outputs: &[NodeId]) -> Self {
        let plan = ReleasePlan::build(resolved, kernels, outputs);
        let produced: Vec<(NodeId, usize)> = resolved
            .iter()
            .map(|bound| (bound.node, bound_output_len(bound)))
            .collect();
        let peak = plan.pool_shape(&produced);
        Self { plan, peak }
    }

    fn relength(&mut self, resolved: &[BoundOp]) {
        let produced: Vec<(NodeId, usize)> = resolved
            .iter()
            .map(|bound| (bound.node, bound_output_len(bound)))
            .collect();
        self.peak = self.plan.pool_shape(&produced);
    }
}

fn bound_output_len(bound: &BoundOp) -> usize {
    match &bound.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            epilogue_broadcast_axes,
            ..
        } if epilogue_broadcast_axes.is_empty() => output_axes
            .iter()
            .map(|axis| bound.extents[*axis as usize] as usize)
            .product(),
        _ => bound.extents.iter().map(|extent| *extent as usize).product(),
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

impl CudaF32Arena {
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
                self.pool.entry(buffer.len()).or_default().push(buffer);
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
enum CudaGraphBuffer {
    F32(CudaSlice<f32>),
    // shared with the driver's resident cache, so a clone is a refcount bump rather than a device copy.
    Bytes(Arc<CudaSlice<u8>>),
}

/// A reusable CUDA graph for the currently connected dense/reduction CUDA ABI.
/// Planning resolves shapes, fusion, dead nodes, and kernel structure once;
/// execution refreshes only named inputs and reuses the device arena.
#[derive(Debug)]
pub struct CudaPlan {
    constraints: Option<ResourceConstraints>,
    block_contract: Vec<(Option<Codec>, bool, usize)>,
    driver: CudaDriver,
    program: Vec<Op>,
    shapes: Shapes,
    resolved: Vec<BoundOp>,
    kernels: Vec<CudaKernel>,
    uniforms: Vec<Vec<u8>>,
    block_nodes: Vec<NodeId>,
    outputs: Box<[NodeId]>,
    // boxed: the arena is three maps (72 bytes) that only move when the plan is built, keeping CudaPlan inside clippy's large_enum_variant gap against the cpu plan
    arena: Box<CudaF32Arena>,
    numeric_policy: NumericPolicy,
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
    functions: Vec<Option<CudaFunction>>,
    emit_calls: u64,
    emit_reused: u64,
    pack_calls: u64,
    repack_calls: u64,
    carry: Box<StateCarry>,
    pool: Box<PoolSchedule>,
}

/// Recurrent state that stays on the device between calls: `pairs` maps a state input node to the
/// output node that produces its next value, `live` holds the inputs whose arena buffer is already
/// current, and `host_bytes` counts every carried state byte that crossed PCIe in either direction.
/// `appends` maps a cache input node to the output node whose rows are appended to it on the
/// device, `append_rows` holds the valid row count of each live cache input.
#[derive(Debug, Default)]
struct StateCarry {
    pairs: BTreeMap<NodeId, NodeId>,
    live: BTreeSet<NodeId>,
    host_bytes: u64,
    appends: BTreeMap<NodeId, NodeId>,
    append_rows: BTreeMap<NodeId, u64>,
    append_cached_rows: u64,
}

/// Numeric and activation policy for one plan, plus the runtime config when the
/// plan is built from one. Not a `RuntimeConfig`: the config-less entry points
/// plan with fusion enabled, which `RuntimeConfig::default()` (fusion off) would
/// silently change.
struct PlanPolicy<'config> {
    numeric_policy: NumericPolicy,
    exact_activations: bool,
    config: Option<&'config RuntimeConfig>,
}

impl CudaF32Arena {
    /// Number of device allocations performed by this arena.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }
}

impl CudaDriver {
    /// Removes device copies for resident input nodes after their host source changes.
    pub fn invalidate_resident_buffers(&self, nodes: &[NodeId]) -> Result<(), CudaDriverError> {
        let mut buffers = self.resident_buffers.lock();
        if !nodes.is_empty() {
            // graph-local node ids cannot identify entries keyed by host source identity.
            buffers.clear();
        }
        Ok(())
    }

    /// Initializes the CUDA primary context for `ordinal`.
    ///
    /// This is the first real device boundary: on a machine without a
    /// visible NVIDIA device it returns a typed `Driver` error rather than
    /// pretending that emitted CUDA source is executable.
    pub fn new(ordinal: usize) -> Result<Self, CudaDriverError> {
        prefetch_nvrtc_version();
        // the dynamic loader otherwise panics before it can return a driver error.
        if !unsafe { cuda_sys::is_culib_present() } {
            return Err(CudaDriverError::Driver(
                "cuda driver library unavailable".into(),
            ));
        }
        let context = CudaContext::new(ordinal)?;
        let device = EmitDevice { multiprocessors: multiprocessor_count(&context)? };
        // every launch runs on this one default stream, so per-buffer read/write events order nothing and cost two cuEventCreate per buffer
        unsafe { context.disable_event_tracking() };
        let stream = context.default_stream();
        Ok(Self {
            context,
            stream,
            device,
            modules: Arc::new(Mutex::new(BTreeMap::new())),
            resident_buffers: Arc::new(Mutex::new(BTreeMap::new())),
            compile_counters: Arc::new(CompileCounters::new(kernel_disk_store())),
            packed_upload_bytes: Arc::new(AtomicU64::new(0)),
            graph_uploads: Arc::new(AtomicU64::new(0)),
            graph_launches: Arc::new(AtomicU64::new(0)),
            parked_state: Arc::default(),
            parked_appends: Arc::default(),
        })
    }

    /// Reads the selected CUDA device and the driver's graph activity counters.
    pub fn device_snapshot(&self) -> Result<CudaDeviceSnapshot, CudaDriverError> {
        Ok(CudaDeviceSnapshot {
            ordinal: self.context.ordinal(),
            name: self.context.name()?,
            graph_uploads: self.graph_uploads.load(Ordering::Relaxed),
            graph_launches: self.graph_launches.load(Ordering::Relaxed),
        })
    }

    /// Creates an empty persistent buffer arena owned by this driver.
    #[must_use]
    pub fn new_arena(&self) -> CudaF32Arena {
        CudaF32Arena::default()
    }

    /// Number of NVRTC/module compilations performed by this driver. A
    /// steady-state plan should stop increasing this after its first dispatch
    /// of each distinct kernel identity.
    #[must_use]
    pub fn kernel_compilations(&self) -> u64 {
        self.compile_counters.compilations.load(Ordering::Relaxed)
    }

    /// Kernels whose PTX was loaded from the on-disk cache instead of NVRTC.
    /// A warm run with unchanged keys stops increasing `kernel_compilations`
    /// and grows this instead.
    #[must_use]
    pub fn disk_cache_hits(&self) -> u64 {
        self.compile_counters
            .disk_cache_hits
            .load(Ordering::Relaxed)
    }

    /// Cumulative wall-clock microseconds spent in NVRTC source-to-PTX
    /// compilation across this driver's `kernel_compilations`.
    #[must_use]
    pub fn nvrtc_micros(&self) -> u64 {
        self.compile_counters.nvrtc_micros.load(Ordering::Relaxed)
    }

    /// Cumulative wall-clock microseconds spent in `cuModuleLoadData` (PTX to
    /// device module, including driver JIT) across this driver's
    /// `kernel_compilations`.
    #[must_use]
    pub fn load_micros(&self) -> u64 {
        self.compile_counters.load_micros.load(Ordering::Relaxed)
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
        let released: BTreeSet<&str> = names.into_iter().collect();
        self.resident_buffers
            .lock()
            .retain(|_, (name, _)| !released.contains(name.as_str()));
    }

    /// Evicts every device buffer cached for a host block inside one of these
    /// host address ranges. The cache keys a weight by its host `(address,
    /// length)`, which is sound only while that host allocation lives: the
    /// owner of the host bytes calls this as it drops them, so a later
    /// allocation at the same address can never resolve to a stale buffer.
    pub fn release_resident_ranges(&self, ranges: &[Range<usize>]) {
        let mut buffers = self.resident_buffers.lock();
        let doomed: Vec<(usize, usize)> = ranges
            .iter()
            .flat_map(|range| {
                buffers
                    .range((range.start, 0)..(range.end, 0))
                    .map(|(key, _)| *key)
            })
            .collect();
        for key in doomed {
            buffers.remove(&key);
        }
    }

    /// Number of host blocks currently holding a cached resident device buffer.
    #[must_use]
    pub fn resident_buffer_count(&self) -> usize {
        self.resident_buffers.lock().len()
    }

    /// Returns the CUDA driver's current `(free_bytes, total_bytes)` view.
    /// This is the authoritative device-memory measurement used by a tier
    /// fit check; host RSS and checkpoint file size are only proxies.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaDriverError> {
        Ok(self.context.mem_get_info()?)
    }

    fn function_for(&self, kernel: &CudaKernel) -> Result<CudaFunction, CudaDriverError> {
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
        let module = self.context.load_module(ptx)?;
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
        let module = self.context.load_module(image);
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
        let function = module.load_function(entry);
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
    pub fn precompiled_function(
        &self,
        module_key: &str,
        entry: &str,
        ptx_source: &str,
    ) -> Result<CudaFunction, CudaDriverError> {
        let cache_key = format!("ptx:{module_key}");
        self.cached_function(&cache_key, entry, ptx_source, || {
            let module = self.context.load_module(Ptx::from_src(ptx_source))?;
            self.compile_counters
                .compilations
                .fetch_add(1, Ordering::Relaxed);
            Ok(module)
        })
    }

    /// Resolves a dense/reduction graph into a persistent CUDA plan.
    pub fn plan(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        numeric_policy: NumericPolicy,
        named: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<CudaPlan, CudaDriverError> {
        let policy = PlanPolicy {
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
    ) -> Result<CudaPlan, CudaDriverError> {
        let policy = PlanPolicy {
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
    ) -> Result<CudaPlan, CudaDriverError> {
        let policy = PlanPolicy {
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
    ) -> Result<CudaPlan, CudaDriverError> {
        let PlanPolicy {
            numeric_policy,
            exact_activations,
            config,
        } = policy;
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
        let (resolved, mut micros) =
            bind_resolved(program, &shapes, outputs, &packed_operands, numeric_policy, unfused)?;
        let (kernels, uniforms) =
            emit_and_pack(&resolved, None, &packed_operands, numeric_policy, self.device, &mut micros)?;
        let pool = Box::new(PoolSchedule::build(&resolved, &kernels, outputs));
        let mut plan = CudaPlan {
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
                self.function_for(kernel)?;
            }
        } else {
            plan.learn_rows_at_build(symbols, numeric_policy);
        }
        Ok(plan)
    }

    /// Uploads or refreshes one named f32 input, retaining its allocation for
    /// later dispatches when the element count is unchanged.
    pub fn upload_f32(
        &self,
        arena: &mut CudaF32Arena,
        node: NodeId,
        values: &[f32],
    ) -> Result<(), CudaDriverError> {
        if let Some(CudaGraphBuffer::F32(buffer)) = arena.buffers.get_mut(&node)
            && buffer.len() == values.len()
        {
            self.stream.memcpy_htod(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
        self.graph_uploads.fetch_add(1, Ordering::Relaxed);
        arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        arena.allocations += 1;
        Ok(())
    }

    fn upload_bytes(
        &self,
        arena: &mut CudaF32Arena,
        node: NodeId,
        values: &[u8],
    ) -> Result<(), CudaDriverError> {
        self.packed_upload_bytes
            .fetch_add(values.len() as u64, Ordering::Relaxed);
        // a buffer shared with the resident cache is never overwritten in place
        if let Some(CudaGraphBuffer::Bytes(shared)) = arena.buffers.get_mut(&node)
            && shared.len() == values.len()
            && let Some(buffer) = Arc::get_mut(shared)
        {
            self.stream.memcpy_htod(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
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
        arena: &mut CudaF32Arena,
        node: NodeId,
        values: &[u8],
    ) -> Result<(), CudaDriverError> {
        if let Some(buffer) = arena.uniforms.get_mut(&node)
            && buffer.len() == values.len()
        {
            if arena.uniform_host.get(&node).is_some_and(|held| held == values) {
                return Ok(());
            }
            self.stream.memcpy_htod(values, buffer)?;
            self.graph_uploads.fetch_add(1, Ordering::Relaxed);
            arena.launch_profile.uniform_uploads += 1;
            arena.uniform_host.insert(node, values.to_vec());
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
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
        arena: &mut CudaF32Arena,
        node: NodeId,
        values: &[f32],
    ) -> Result<(), CudaDriverError> {
        if values.is_empty() {
            arena.allocations += 1;
            arena.buffers.insert(
                node,
                CudaGraphBuffer::F32(self.stream.alloc_zeros::<f32>(0)?),
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
        arena: &mut CudaF32Arena,
        node: NodeId,
        value: f32,
        output_len: usize,
    ) -> Result<(), CudaDriverError> {
        if matches!(
            arena.buffers.get(&node),
            Some(CudaGraphBuffer::F32(buffer)) if buffer.len() == output_len
        ) {
            return Ok(());
        }
        let buffer = if output_len == 0 {
            self.stream.alloc_zeros::<f32>(0)?
        } else {
            let values = vec![value; output_len];
            let buffer = self.stream.clone_htod(&values)?;
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
        let kernel = add_one_kernel(NodeId(0), NodeId(1), input.len());
        self.launch_f32(&kernel, &[input], &n.to_ne_bytes())
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
        let device_input = self.stream.clone_htod(input)?;
        let mut device_output = self.stream.alloc_zeros::<f32>(input.len())?;
        let device_uniforms = self.stream.clone_htod(&n.to_ne_bytes())?;
        let mut args = self.stream.launch_builder(&function);
        args.arg(&device_input);
        args.arg(&mut device_output);
        args.arg(&device_uniforms);
        unsafe {
            args.launch(LaunchConfig {
                grid_dim: ((input.len() as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&device_output)?)
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
            CudaGraphBuffer::F32(self.stream.clone_htod(input)?),
        );
        arena.allocations = 1;
        let first = add_one_kernel(NodeId(0), NodeId(1), input.len());
        self.launch_f32_persistent(&mut arena, &first, &n.to_ne_bytes())?;
        let second = add_one_kernel(NodeId(1), NodeId(2), input.len());
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
        let output_len = usize::try_from(kernel.grid.threads)
            .map_err(|_| CudaDriverError::InputTooLarge(usize::MAX))?;
        let function = self.function_for(kernel)?;
        let device_inputs = inputs
            .iter()
            .map(|input| self.stream.clone_htod(*input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut device_output = self.stream.alloc_zeros::<f32>(output_len)?;
        let device_uniforms = self.stream.clone_htod(uniforms)?;
        let config = LaunchConfig {
            grid_dim: (launch_blocks(&kernel.grid)?, 1, 1),
            block_dim: (kernel.grid.block_width_or_default() as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = self.stream.launch_builder(&function);
        for input in &device_inputs {
            args.arg(input);
        }
        args.arg(&mut device_output);
        args.arg(&device_uniforms);
        unsafe { args.launch(config) }?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&device_output)?)
    }

    /// Dispatches a kernel against persistent arena buffers. Inputs and the
    /// output are selected from `CudaKernel::bindings`; no host tensor copy
    /// occurs for an intermediate node already resident in the arena.
    pub fn launch_f32_persistent(
        &self,
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
    ) -> Result<(), CudaDriverError> {
        self.launch_f32_persistent_sized(arena, kernel, uniforms, kernel.grid.threads as usize)
    }

    /// Persistent launch with an output allocation sized independently from
    /// the dispatch thread count (reductions dispatch over output*reduction
    /// elements but materialize only the output shape).
    pub fn launch_f32_persistent_sized(
        &self,
        arena: &mut CudaF32Arena,
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
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        deferred_fault: Option<(&CudaSlice<u32>, usize, usize)>,
        function: Option<&CudaFunction>,
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
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
    ) -> Result<(), CudaDriverError> {
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
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        completion: Completion<'_>,
        function: Option<&CudaFunction>,
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
                looked_up = self.function_for(kernel)?;
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
                self.stream.alloc_zeros::<f32>(output_len)?
            }
        };
        if device_output.len() != output_len {
            let replacement = match self.stream.alloc_zeros::<f32>(output_len) {
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
        let config = LaunchConfig {
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
            .then(|| self.stream.alloc_zeros::<u32>(gather_slots))
            .transpose()?;
        let fault_window =
            deferred_fault.map(|(array, offset, slots)| array.slice(offset..offset + slots));
        arena.launch_profile.mark(3, &mut lap);
        let mut args = self.stream.launch_builder(function);
        for binding in &kernel.bindings[..output_index] {
            match binding {
                Binding::Input(node) | Binding::Indices(node) => match arena.buffers.get(node) {
                    Some(CudaGraphBuffer::F32(input)) => args.arg(input),
                    Some(CudaGraphBuffer::Bytes(input)) => args.arg(input.as_ref()),
                    None => {
                        arena
                            .buffers
                            .insert(output_node, CudaGraphBuffer::F32(device_output));
                        return Err(CudaDriverError::MissingBuffer(*node));
                    }
                },
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        args.arg(&mut device_output);
        for binding in &kernel.bindings[output_index + 1..] {
            match binding {
                Binding::Uniforms => args.arg(device_uniforms),
                Binding::Fault => match (&fault_window, &fault_buffer) {
                    (Some(window), _) => args.arg(window),
                    (None, Some(buffer)) => args.arg(buffer),
                    (None, None) => return Err(CudaDriverError::BindingContract),
                },
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        for (_, buffer) in &mut extra_buffers {
            args.arg(buffer);
        }
        arena.launch_profile.mark(4, &mut lap);
        let launch_result = unsafe { args.launch(config) };
        arena.launch_profile.mark(5, &mut lap);
        Self::restore_extra_outputs(arena, extra_buffers);
        if let Err(error) = launch_result {
            arena
                .buffers
                .insert(output_node, CudaGraphBuffer::F32(device_output));
            return Err(error.into());
        }
        self.graph_launches.fetch_add(1, Ordering::Relaxed);
        if synchronize {
            if let Err(error) = self.stream.synchronize() {
                arena
                    .buffers
                    .insert(output_node, CudaGraphBuffer::F32(device_output));
                return Err(error.into());
            }
            if let Some(fault_buffer) = fault_buffer.take() {
                let faults = self.stream.clone_dtoh(&fault_buffer)?;
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

    fn take_extra_output(
        &self,
        arena: &mut CudaF32Arena,
        node: NodeId,
        length: usize,
    ) -> Result<CudaSlice<f32>, CudaDriverError> {
        match arena.buffers.remove(&node) {
            Some(CudaGraphBuffer::F32(buffer)) if buffer.len() == length => Ok(buffer),
            Some(CudaGraphBuffer::Bytes(_)) => Err(CudaDriverError::BindingContract),
            Some(CudaGraphBuffer::F32(_)) | None => {
                arena.allocations += 1;
                arena.launch_profile.extra_allocs += 1;
                Ok(self.stream.alloc_zeros::<f32>(length)?)
            }
        }
    }

    fn restore_extra_outputs(arena: &mut CudaF32Arena, buffers: Vec<(NodeId, CudaSlice<f32>)>) {
        for (node, buffer) in buffers {
            arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        }
    }

    fn record_event(&self, stream: &CudaStream) -> Result<CudaEvent, CudaDriverError> {
        let event = stream
            .context()
            .new_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        event.record(stream)?;
        Ok(event)
    }

    /// Copies one persistent node back to the host for an explicit oracle or
    /// final-result readback.
    pub fn read_f32(
        &self,
        arena: &CudaF32Arena,
        node: NodeId,
    ) -> Result<Vec<f32>, CudaDriverError> {
        let buffer = arena
            .buffers
            .get(&node)
            .ok_or(CudaDriverError::MissingBuffer(node))?;
        let CudaGraphBuffer::F32(buffer) = buffer else {
            return Err(CudaDriverError::BindingContract);
        };
        Ok(self.stream.clone_dtoh(buffer)?)
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

/// Whether a plan built for `program` at `symbols` would bind its cached K/V range
/// padded safely (see [`CudaPlan::pads_kv_safely`]), decided by binding alone: no
/// kernel is emitted and no device is touched. `named` need only carry the weights;
/// a missing input is not an error here, it just binds unpacked.
///
/// # Errors
/// Shape inference or binding fails at `symbols`.
pub fn probe_pads_kv(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    named: &[(&str, QuantizedBlock<'_>)],
    numeric_policy: NumericPolicy,
    is_kv_input: impl Fn(&str) -> bool,
) -> Result<bool, CudaDriverError> {
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
    let (resolved, _) = bind_resolved(program, &shapes, outputs, &packed_operands, numeric_policy, false)?;
    Ok(kv_read_only_by_fused_attention(program, &resolved, outputs, is_kv_input))
}

fn bind_resolved(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    unfused: bool,
) -> Result<(Vec<BoundOp>, LowerMicros), CudaDriverError> {
    let bind_started = Instant::now();
    let bound = if unfused {
        bind_unfused(program, shapes, outputs, numeric_policy)
    } else {
        bind_with_fusion(
            program,
            shapes,
            outputs,
            numeric_policy.reassociation,
            numeric_policy,
        )
    }
    .map_err(|_| CudaDriverError::UnsupportedGraph {
        node: NodeId(0),
        reason: "graph binding failed",
    })?;
    let mut resolved = prune_dead(bound, outputs);
    correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());
    validate_resolved(&resolved, numeric_policy)?;
    let micros = LowerMicros {
        bind: bind_started.elapsed().as_micros() as u64,
        ..LowerMicros::default()
    };
    Ok((resolved, micros))
}

fn validate_resolved(
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
) -> Result<(), CudaDriverError> {
    for bound in resolved {
        validate_cuda_dtype(bound)?;
    }
    reject_unsupported_kinds(resolved, numeric_policy)
}

fn emit_and_pack(
    resolved: &[BoundOp],
    prior: Option<PriorKernels<'_>>,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
    micros: &mut LowerMicros,
) -> Result<(Vec<CudaKernel>, Vec<Vec<u8>>), CudaDriverError> {
    let emit_started = Instant::now();
    let (prior_bound, prior_kernels) =
        prior.map_or((&[][..], Vec::new()), |prior| (prior.bound, prior.kernels));
    let (kernels, reused) = crate::emit_cuda_reusing(
        prior_bound,
        prior_kernels,
        resolved,
        packed_operands,
        numeric_policy,
        device,
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

fn report_event_timing(
    origin: &CudaEvent,
    rows: &[(NodeId, String, usize, CudaEvent)],
) -> Result<(), CudaDriverError> {
    let mut previous = origin;
    let mut by_kernel: BTreeMap<&str, (f64, usize, usize)> = BTreeMap::new();
    let mut total_ms = 0.0f64;
    for (_, label, output_len, event) in rows {
        let millis = f64::from(previous.elapsed_ms(event)?);
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
fn refresh_bound_resolved(
    previous: &[BoundOp],
    previous_shapes: &Shapes,
    program: &[Op],
    outputs: &[NodeId],
    shapes: &Shapes,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<(Vec<BoundOp>, usize), RefreshRefusal> {
    let changed: BTreeSet<NodeId> = (0..program.len())
        .map(|position| NodeId(position as u32))
        .filter(|node| previous_shapes.of(*node) != shapes.of(*node))
        .collect();
    let masks: Vec<u64> = previous
        .iter()
        .map(|bound| u64::from(touches_changed(bound, program, &changed)))
        .collect();
    let touched = masks.iter().filter(|mask| **mask != 0).count();
    let mut refreshed = refresh_bound_ops(
        previous,
        program,
        outputs,
        shapes,
        &masks,
        1,
        numeric_policy,
    )?;
    correct_packed_matmul_layouts(&mut refreshed, &packed_operands.keys().copied().collect());
    Ok((refreshed, touched))
}

fn cuda_supports_kind(kind: &BoundOpKind, numeric_policy: NumericPolicy) -> bool {
    match kind {
        BoundOpKind::CachedAttention { .. } => {
            admit(numeric_policy, NumericRewrite::ContextChunkMerge).is_ok()
                && cuda_cached_attention_class(kind).is_some()
        }
        BoundOpKind::Elementwise { .. }
        | BoundOpKind::Reduce { .. }
        | BoundOpKind::Iota
        | BoundOpKind::Constant { .. }
        | BoundOpKind::MoeTopK { .. } => true,
        _ => false,
    }
}

/// Rejects the whole graph at once, naming every unsupported kind with its
/// node count, so a model's full gap list is one error rather than one node at a time.
fn reject_unsupported_kinds(
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
) -> Result<(), CudaDriverError> {
    let mut census: BTreeMap<&'static str, (usize, NodeId)> = BTreeMap::new();
    for bound in resolved
        .iter()
        .filter(|bound| !cuda_supports_kind(&bound.kind, numeric_policy))
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

impl CudaPlan {
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
                &previous_bound,
                &self.shapes,
                &self.program,
                &self.outputs,
                &shapes,
                &self.packed_operands,
                numeric_policy,
            )
            .map_err(refresh_refusal_name)
        };
        let (bind_path, touched, refused, resolved, kernels, uniforms, micros) = match refresh {
            Ok((resolved, touched)) => {
                let mut micros = LowerMicros {
                    bind: refresh_started.elapsed().as_micros() as u64,
                    ..LowerMicros::default()
                };
                validate_resolved(&resolved, numeric_policy)?;
                let prior = PriorKernels {
                    bound: &previous_bound,
                    kernels: std::mem::take(&mut self.kernels),
                };
                let (kernels, uniforms) = emit_and_pack(
                    &resolved,
                    Some(prior),
                    &self.packed_operands,
                    numeric_policy,
                    self.driver.device,
                    &mut micros,
                )?;
                ("refresh", touched, "none", resolved, kernels, uniforms, micros)
            }
            Err(refused) => {
                let prior = same_policy.then(|| PriorKernels {
                    bound: &previous_bound,
                    kernels: std::mem::take(&mut self.kernels),
                });
                let (resolved, mut micros) = bind_resolved(
                    &self.program,
                    &shapes,
                    &self.outputs,
                    &self.packed_operands,
                    numeric_policy,
                    false,
                )?;
                let (kernels, uniforms) = emit_and_pack(
                    &resolved,
                    prior,
                    &self.packed_operands,
                    numeric_policy,
                    self.driver.device,
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
            self.follow_rows(position, bound, numeric_policy, &mut follow)?;
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
        let device = self.driver.device;
        let grid_started = Instant::now();
        if matches!(next.kind, BoundOpKind::CachedAttention { .. }) {
            let kept = crate::cuda::refit_kernel(
                &self.resolved[position],
                next,
                &mut self.kernels[position],
                &self.packed_operands,
                numeric_policy,
                device,
            )
            .map_err(emit_error)?;
            if !kept {
                self.functions[position] = None;
            }
            follow.attention += elapsed_micros(grid_started);
        } else {
            crate::cuda::refit_grid(&mut self.kernels[position], next, &self.packed_operands, numeric_policy, device)
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
        let (bound, _) = bind_resolved(
            &self.program,
            &shapes,
            &self.outputs,
            &self.packed_operands,
            numeric_policy,
            false,
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
        let device = self.driver.device;
        let packed = &self.packed_operands;
        for position in 0..previous.len() {
            let current = &self.resolved[position];
            let same_plan = |other: &BoundOp| {
                between.len() == previous.len()
                    && crate::cuda::same_text_modulo_extents(other, current, packed, numeric_policy, device)
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
        let before = crate::emit_cuda_on_device(previous, &self.packed_operands, numeric_policy, self.driver.device)
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
            lengths_moved |= bound_output_len(&previous) != bound_output_len(&self.resolved[position]);
            let text_kept = crate::cuda::refit_kernel(
                &previous,
                &self.resolved[position],
                &mut self.kernels[position],
                &self.packed_operands,
                numeric_policy,
                self.driver.device,
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
                    let current = (buffer.len() as u64)
                        .checked_mul(4)
                        .ok_or(CudaDriverError::ResourceOverflow)?;
                    add(current)?;
                    if current != bytes {
                        add(bytes)?;
                    }
                }
                Some(CudaGraphBuffer::Bytes(buffer)) => {
                    add(buffer.len() as u64)?;
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
            let bytes = cuda_float_bytes(self.shapes.of(bound.node))?;
            let pooled = release_plan.pooled.contains(&bound.node);
            match self.arena.buffers.get(&bound.node) {
                _ if pooled => {}
                Some(CudaGraphBuffer::F32(buffer)) => {
                    let current = (buffer.len() as u64)
                        .checked_mul(4)
                        .ok_or(CudaDriverError::ResourceOverflow)?;
                    add(current)?;
                    if current != bytes {
                        add(bytes)?;
                    }
                }
                Some(CudaGraphBuffer::Bytes(buffer)) => {
                    add(buffer.len() as u64)?;
                    add(bytes)?;
                }
                None => add(bytes)?,
            }
            if !matches!(bound.kind, BoundOpKind::Constant { .. }) {
                if let Some(existing) = self.arena.uniforms.get(&bound.node) {
                    add(existing.len() as u64)?;
                    if existing.len() != uniform.len() {
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
                CudaGraphBuffer::F32(values) => (values.len() as u64)
                    .checked_mul(4)
                    .ok_or(CudaDriverError::ResourceOverflow)?,
                CudaGraphBuffer::Bytes(values) => values.len() as u64,
            };
            sum.checked_add(bytes)
                .ok_or(CudaDriverError::ResourceOverflow)
        })?;
        let retained_uniforms = self
            .arena
            .uniforms
            .values()
            .try_fold(0_u64, |sum, buffer| {
                sum.checked_add(buffer.len() as u64)
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
    pub fn kernel_identities(&self) -> Vec<String> {
        self.kernels
            .iter()
            .map(|kernel| format!("{}\n{}", kernel.entry, kernel.source))
            .collect()
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
        Ok(format!(
            "cuda:{}:{}",
            self.driver.context.ordinal(),
            self.driver.context.name()?
        ))
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
        let wanted: BTreeMap<NodeId, NodeId> = pairs.iter().copied().collect();
        if wanted == self.carry.pairs {
            return true;
        }
        let swappable = self.constraints.is_none()
            && pairs.iter().all(|(input, output)| {
                self.outputs.contains(output)
                    && self.shapes.of(*input).iter().product::<u64>()
                        == self.shapes.of(*output).iter().product::<u64>()
            });
        if !swappable || !self.carry.live.is_empty() {
            return false;
        }
        self.carry.pairs = wanted;
        self.adopt_parked();
        true
    }

    fn adopt_parked(&mut self) {
        let mut parked = self.driver.parked_state.lock();
        for input in self.carry.pairs.keys() {
            if let Some((_, buffer)) = parked.remove(input) {
                self.arena.buffers.insert(*input, buffer);
                self.arena.resident_sources.remove(input);
                self.carry.live.insert(*input);
            }
        }
    }

    /// Hands live carried state to the driver so the plan that replaces this one adopts it on
    /// the device instead of re-uploading a stale host copy.
    pub fn park_carried(&mut self) {
        self.park_appended();
        let live = std::mem::take(&mut self.carry.live);
        let mut parked = self.driver.parked_state.lock();
        for input in live {
            let (Some(output), Some(buffer)) =
                (self.carry.pairs.get(&input).copied(), self.arena.buffers.remove(&input))
            else {
                continue;
            };
            parked.insert(input, (output, buffer));
        }
    }

    /// Reads every live carried state back to the host, keyed by the output node that produces
    /// it, and marks the device copies stale so the next call uploads its host inputs again.
    pub fn drain_carried(&mut self) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        self.park_carried();
        let parked = std::mem::take(&mut *self.driver.parked_state.lock());
        let mut drained = Vec::with_capacity(parked.len());
        for (_, (output, buffer)) in parked {
            let CudaGraphBuffer::F32(slice) = buffer else {
                return Err(CudaDriverError::BindingContract);
            };
            let values = self.driver.stream.clone_dtoh(&slice)?;
            self.carry.host_bytes += (values.len() * std::mem::size_of::<f32>()) as u64;
            drained.push((output, values));
        }
        Ok(drained)
    }

    /// Keeps each `(cache input, new-rows output)` pair's cache on the device: the input leaf has
    /// shape `[extent_rows, ..row]`, the output `[new_rows, ..row]`. The first call uploads the host
    /// block (holding `cached_rows` valid rows); afterwards the input is never uploaded and each
    /// call's output rows are copied device-to-device onto the end of the valid rows, and the output
    /// is not read back. Returns `false` (carrying nothing) for a resource-constrained plan, an
    /// output that is not a plan output, or differing row widths. The same set keeps live state; a
    /// different set requires [`Self::drain_appended_rows`] first.
    pub fn carry_rows(&mut self, pairs: &[(NodeId, NodeId)], cached_rows: u64) -> bool {
        let wanted: BTreeMap<NodeId, NodeId> = pairs.iter().copied().collect();
        if wanted == self.carry.appends {
            return true;
        }
        let appendable = self.constraints.is_none()
            && pairs
                .iter()
                .all(|(input, output)| self.appendable(*input, *output));
        if !appendable || !self.carry.append_rows.is_empty() {
            return false;
        }
        self.carry.appends = wanted;
        self.carry.append_cached_rows = cached_rows;
        self.adopt_parked_appends();
        true
    }

    fn appendable(&self, input: NodeId, output: NodeId) -> bool {
        !self.shapes.of(input).is_empty()
            && !self.shapes.of(output).is_empty()
            && self.outputs.contains(&output)
            && !self.carry.pairs.contains_key(&input)
            && self.row_width(input) == self.row_width(output)
    }

    fn row_width(&self, node: NodeId) -> u64 {
        self.shapes.of(node).iter().skip(1).product()
    }

    fn adopt_parked_appends(&mut self) {
        let mut parked = self.driver.parked_appends.lock();
        for input in self.carry.appends.keys() {
            if let Some((rows, buffer)) = parked.remove(input) {
                self.arena.buffers.insert(*input, buffer);
                self.arena.resident_sources.remove(input);
                self.carry.append_rows.insert(*input, rows);
            }
        }
    }

    fn park_appended(&mut self) {
        let live = std::mem::take(&mut self.carry.append_rows);
        let mut parked = self.driver.parked_appends.lock();
        for (input, rows) in live {
            if let Some(buffer) = self.arena.buffers.remove(&input) {
                parked.insert(input, (rows, buffer));
            }
        }
    }

    /// Reads the valid rows of every live appended cache back to the host, keyed by the cache
    /// INPUT node, and marks the device copies stale so the next call uploads its host block again.
    pub fn drain_appended_rows(&mut self) -> Result<Vec<(NodeId, Vec<f32>)>, CudaDriverError> {
        self.park_appended();
        let parked = std::mem::take(&mut *self.driver.parked_appends.lock());
        let mut drained = Vec::with_capacity(parked.len());
        for (input, (rows, buffer)) in parked {
            let CudaGraphBuffer::F32(slice) = buffer else {
                return Err(CudaDriverError::BindingContract);
            };
            let valid = (rows * self.row_width(input)) as usize;
            let values = self.driver.stream.clone_dtoh(&slice.slice(0..valid))?;
            self.carry.host_bytes += (values.len() * std::mem::size_of::<f32>()) as u64;
            drained.push((input, values));
        }
        Ok(drained)
    }

    fn grow_appended_inputs(&mut self) -> Result<(), CudaDriverError> {
        let live: Vec<(NodeId, u64)> = self.carry.append_rows.iter().map(|(node, rows)| (*node, *rows)).collect();
        for (input, rows) in live {
            let needed = self.shapes.of(input).iter().product::<u64>() as usize;
            let valid = (rows * self.row_width(input)) as usize;
            let Some(CudaGraphBuffer::F32(current)) = self.arena.buffers.get(&input) else {
                return Err(CudaDriverError::MissingBuffer(input));
            };
            if current.len() >= needed {
                continue;
            }
            if valid > current.len() {
                return Err(CudaDriverError::AppendOverflow {
                    node: input,
                    valid_elements: valid,
                    new_elements: 0,
                    capacity_elements: current.len(),
                });
            }
            let mut grown = self.driver.stream.alloc_zeros::<f32>(needed)?;
            if valid > 0 {
                self.driver
                    .stream
                    .memcpy_dtod(&current.slice(0..valid), &mut grown.slice_mut(0..valid))?;
            }
            self.arena.allocations += 1;
            self.arena.buffers.insert(input, CudaGraphBuffer::F32(grown));
        }
        Ok(())
    }

    fn append_new_rows(&mut self) -> Result<(), CudaDriverError> {
        let pairs: Vec<(NodeId, NodeId)> = self.carry.appends.iter().map(|(input, output)| (*input, *output)).collect();
        for (input, output) in pairs {
            let Some(rows) = self.carry.append_rows.get(&input).copied() else {
                continue;
            };
            let new_rows = self.shapes.of(output).first().copied().unwrap_or(0);
            let width = self.row_width(input);
            self.append_output(input, output, (rows * width) as usize, (new_rows * width) as usize)?;
            self.carry.append_rows.insert(input, rows + new_rows);
        }
        Ok(())
    }

    fn append_output(
        &mut self,
        input: NodeId,
        output: NodeId,
        offset: usize,
        count: usize,
    ) -> Result<(), CudaDriverError> {
        let Some(CudaGraphBuffer::F32(mut cache)) = self.arena.buffers.remove(&input) else {
            return Err(CudaDriverError::MissingBuffer(input));
        };
        if offset + count > cache.len() {
            match self.grown_cache(&cache, offset, offset + count) {
                Ok(grown) => cache = grown,
                Err(error) => {
                    self.arena.buffers.insert(input, CudaGraphBuffer::F32(cache));
                    return Err(error);
                }
            }
        }
        let copied = match self.arena.buffers.get(&output) {
            Some(CudaGraphBuffer::F32(source)) if source.len() >= count => {
                if count == 0 {
                    Ok(())
                } else {
                    self.driver
                        .stream
                        .memcpy_dtod(&source.slice(0..count), &mut cache.slice_mut(offset..offset + count))
                        .map_err(CudaDriverError::from)
                }
            }
            _ => Err(CudaDriverError::MissingBuffer(output)),
        };
        self.arena.buffers.insert(input, CudaGraphBuffer::F32(cache));
        copied
    }

    fn grown_cache(
        &mut self,
        cache: &CudaSlice<f32>,
        valid: usize,
        required: usize,
    ) -> Result<CudaSlice<f32>, CudaDriverError> {
        let mut grown = self.driver.stream.alloc_zeros::<f32>(required.next_power_of_two())?;
        let preserved = valid.min(cache.len());
        if preserved > 0 {
            self.driver
                .stream
                .memcpy_dtod(&cache.slice(0..preserved), &mut grown.slice_mut(0..preserved))?;
        }
        self.arena.allocations += 1;
        Ok(grown)
    }

    /// Bytes of carried recurrent state that crossed PCIe in either direction so far.
    #[must_use]
    pub fn carried_host_bytes(&self) -> u64 {
        self.carry.host_bytes
    }

    fn swap_carried_buffers(&mut self) {
        for (input, output) in &self.carry.pairs {
            if !(self.arena.buffers.contains_key(output) && self.arena.buffers.contains_key(input)) {
                continue;
            }
            let (Some(next), Some(previous)) = (
                self.arena.buffers.remove(output),
                self.arena.buffers.remove(input),
            ) else {
                continue;
            };
            self.arena.buffers.insert(*input, next);
            self.arena.buffers.insert(*output, previous);
            self.arena.resident_sources.remove(input);
            self.carry.live.insert(*input);
        }
    }

    /// Executes the planned graph with name-resolved inputs and returns the
    /// requested outputs. Packed bytes stay on device in the persistent arena.
    pub fn execute_named(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&BTreeSet<&str>>,
    ) -> Result<Evaluated, CudaDriverError> {
        let host_timing = std::env::var_os("PROXIMA_CUDA_HOST_TIMING").is_some();
        let upload_started = host_timing.then(std::time::Instant::now);
        let phase_started = Instant::now();
        let upload_bytes_before = self.driver.packed_upload_bytes();
        let compilations_before = self.driver.kernel_compilations();
        let nvrtc_before = self.driver.nvrtc_micros();
        let load_before = self.driver.load_micros();
        let lookup_before = self.driver.compile_counters.lookup_micros();
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
            let source = (block_address(block), block_length(block));
            let resident = self.resident_nodes.binary_search(&node).is_ok()
                || resident_names.is_some_and(|names| names.contains(name));
            if self.carry.live.contains(&node) || self.carry.append_rows.contains_key(&node) {
                continue;
            }
            if resident && self.arena.resident_sources.get(&node) == Some(&source) {
                resident_hits += 1;
                continue;
            }
            if resident {
                let packed =
                    !matches!(block, QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_));
                let cached = self
                    .driver
                    .resident_buffers
                    .lock()
                    .get(&source)
                    .filter(|(_, buffer)| matches!(buffer, CudaGraphBuffer::Bytes(_)) == packed)
                    .map(|(_, buffer)| buffer.clone());
                if let Some(buffer) = cached {
                    self.arena.buffers.insert(node, buffer);
                    self.arena.resident_sources.insert(node, source);
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
                    self.driver
                        .upload_bytes(&mut self.arena, node, packed_bytes(other))?;
                }
            }
            upload_call_us += elapsed_micros(upload_call_started);
            if self.carry.pairs.contains_key(&node) {
                self.carry.host_bytes += block_length(block) as u64;
            }
            if self.carry.appends.contains_key(&node) {
                self.carry.host_bytes += block_length(block) as u64;
                self.carry.append_rows.insert(node, self.carry.append_cached_rows);
                self.arena.resident_sources.remove(&node);
                continue;
            }
            if resident {
                self.arena.resident_sources.insert(node, source);
                if let Some(buffer) = self.arena.buffers.get(&node).cloned() {
                    self.driver
                        .resident_buffers
                        .lock()
                        .insert(source, (name.to_owned(), buffer));
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
        self.grow_appended_inputs()?;
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
            .then(|| self.driver.stream.alloc_zeros::<u32>(fault_total))
            .transpose()?;
        let event_timing = std::env::var_os("PROXIMA_CUDA_EVENT_TIMING").is_some();
        let mut event_rows: Vec<(NodeId, String, usize, CudaEvent)> = Vec::new();
        let event_origin = event_timing
            .then(|| self.driver.record_event(&self.driver.stream))
            .transpose()?;
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
            let output_len = bound_output_len(bound);
            if trace {
                eprintln!(
                    "cuda_trace: node={:?} kind={} extents={:?} bindings={:?} grid={} output_len={} uniforms={} detail={}",
                    bound.node,
                    bound.kind.name(),
                    bound.extents,
                    kernel.bindings,
                    kernel.grid.threads,
                    output_len,
                    uniforms.len(),
                    if trace_node == Some(bound.node) {
                        format!("{bound:?}")
                    } else {
                        String::new()
                    },
                );
            }
            // A zero-extent intermediate is a valid tensor, not a valid CUDA
            // launch.  Skip the dispatch; downstream reductions still see
            // their identity value over the empty axis.
            if output_len == 0 {
                let needs_zero_buffer = !matches!(
                    self.arena.buffers.get(&bound.node),
                    Some(CudaGraphBuffer::F32(buffer)) if buffer.is_empty()
                );
                if needs_zero_buffer {
                    self.arena.allocations += 1;
                    self.arena.buffers.insert(
                        bound.node,
                        CudaGraphBuffer::F32(self.driver.stream.alloc_zeros::<f32>(0)?),
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
                    self.functions[position] = Some(self.driver.function_for(kernel)?);
                    function_lookups += 1;
                    function_lookup_us += lookup_started.map_or(0, elapsed_micros);
                }
                if release_plan.pooled.contains(&bound.node) {
                    self.arena.seat_pooled(bound.node, output_len);
                }
                let launch_started = host_timing.then(Instant::now);
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
                if let Some(started) = node_started {
                    timing_rows.push((
                        bound.node,
                        started.elapsed().as_micros(),
                        output_len,
                        kernel.entry.clone(),
                    ));
                }
                if event_timing {
                    let event = self.driver.record_event(&self.driver.stream)?;
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
            let lookup_after = self.driver.compile_counters.lookup_micros();
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
        self.driver.stream.synchronize()?;
        let sync_us = elapsed_micros(sync_phase_started);
        if let Some(array) = &fault_array {
            let faults = self.driver.stream.clone_dtoh(array)?;
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
            report_event_timing(&origin, &event_rows)?;
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
            if self.carry.pairs.iter().any(|(input, output)| output == node && self.carry.live.contains(input)) {
                continue;
            }
            if self.carry.appends.iter().any(|(input, output)| output == node && self.carry.append_rows.contains_key(input)) {
                continue;
            }
            if trace_outputs {
                let length = self.arena.buffers.get(node).map(|buffer| match buffer {
                    CudaGraphBuffer::F32(values) => values.len(),
                    CudaGraphBuffer::Bytes(values) => values.len(),
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
                CudaGraphBuffer::F32(values) => values.len() as u64 * 4,
                CudaGraphBuffer::Bytes(values) => values.len() as u64,
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
    let QuantizedBlock::Packed { codec, .. } = block else {
        return None;
    };
    match codec {
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
    }
}

fn packed_bytes<'a>(block: &QuantizedBlock<'a>) -> &'a [u8] {
    match block {
        QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => &[],
        QuantizedBlock::Packed { bytes, .. } => packed_codec(block).map_or(&[], |_| bytes),
    }
}

fn block_address(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(values) => values.as_ptr() as usize,
        QuantizedBlock::Int32(values) => values.as_ptr() as usize,
        _ => packed_bytes(block).as_ptr() as usize,
    }
}

fn block_length(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(values) => std::mem::size_of_val(*values),
        QuantizedBlock::Int32(values) => std::mem::size_of_val(*values),
        _ => packed_bytes(block).len(),
    }
}

fn add_one_kernel(input: NodeId, output: NodeId, elements: usize) -> CudaKernel {
    CudaKernel {
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
        grid: CudaGridSpec {
            threads: elements as u64,
            block_width: None,
        },
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
    }
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

/// The x-dimension block count of `grid`'s launch, or an error rather than the
/// silent truncation of an `as u32` cast: `emit_cuda` already rejects a grid
/// past `gridDim.x`, so this only fires for a `CudaKernel` built by hand.
fn launch_blocks(grid: &CudaGridSpec) -> Result<u32, CudaDriverError> {
    u32::try_from(grid.blocks()).map_err(|_| CudaDriverError::GridTooLarge {
        threads: grid.threads,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CudaDriver, CudaDriverError, CudaFunction, LaunchConfig, PRECOMPILED_ADD_ONE_PTX,
        PushKernelArg, add_one_kernel, cached_ptx, launch_blocks, packed_bytes, packed_codec,
        resolve_cuda_include_dir, store_ptx, validate_cuda_dtype,
    };
    use crate::cuda::CudaGridSpec;
    use crate::cuda_kernel_cache::CacheKey;
    use crate::cuda_ptx_store::PtxStore;
    use crate::lowering::Codec;
    use cudarc::nvrtc::{CompileOptions, Ptx};
    use proxima_tensor::{
        BoundOp, BoundOpKind, DType, NodeId, QuantizedBlock,
    };

    const KERNEL_CACHE_DIR_VAR: &str = "PROXIMA_CUDA_KERNEL_CACHE_DIR";

    const CUDA_UNPACKABLE: [Codec; 17] = [
        Codec::Q5_1,
        Codec::Q4_1,
        Codec::Q8_1,
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

    #[test]
    fn launch_blocks_covers_a_grid_at_the_default_block_width() {
        let grid = CudaGridSpec {
            threads: 1_000_000,
            block_width: None,
        };

        assert!(matches!(launch_blocks(&grid), Ok(3907)));
    }

    #[test]
    fn launch_blocks_rejects_a_grid_past_the_launch_index_instead_of_truncating() {
        let threads = (u64::from(u32::MAX) + 1) * 256;
        let grid = CudaGridSpec {
            threads,
            block_width: None,
        };

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
        let mut kernel = add_one_kernel(NodeId(0), NodeId(1), INPUT.len());
        kernel.source = kernel.source.replace("+ 1.0f", "+ 2.0f");
        kernel
    }

    fn run_ptx(driver: &CudaDriver, function: &CudaFunction) -> Vec<f32> {
        let device_input = driver.stream.clone_htod(&INPUT).expect("upload input");
        let mut device_output = driver
            .stream
            .alloc_zeros::<f32>(INPUT.len())
            .expect("alloc output");
        let device_uniforms = driver
            .stream
            .clone_htod(&3u32.to_ne_bytes())
            .expect("upload uniforms");
        let mut args = driver.stream.launch_builder(function);
        args.arg(&device_input);
        args.arg(&mut device_output);
        args.arg(&device_uniforms);
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }.expect("ptx launch");
        driver.stream.synchronize().expect("ptx sync");
        driver
            .stream
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
        let add_one = add_one_kernel(NodeId(0), NodeId(1), INPUT.len());
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
}
