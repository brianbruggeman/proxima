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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::sys as nvrtc_sys;
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use proxima_primitives::sync::blocking::Mutex;

use crate::config::{ResourceConstraints, RuntimeConfig};
use crate::cuda::{CudaGridSpec, CudaKernel};
use crate::msl::{Binding, Codec, PackedOperands, gather_count};
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Evaluated, Keep, NodeId, NumericPolicy, Op, QuantizedBlock,
    Shapes, bind_unfused, bind_with_fusion, block_node_ids, correct_packed_matmul_layouts, infer,
    prune_dead, resolve_named_blocks,
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

/// One CUDA context and stream owned by a caller's inference runtime.
#[derive(Debug, Clone)]
pub struct CudaDriver {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // synchronous ffi callers share these short cache mutations; compilation and dispatch happen outside the guard.
    modules: Arc<Mutex<BTreeMap<String, Vec<CachedCudaModule>>>>,
    resident_buffers: Arc<Mutex<ResidentBuffers>>,
    kernel_compilations: Arc<AtomicU64>,
    packed_upload_bytes: Arc<AtomicU64>,
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
    resident_sources: BTreeMap<NodeId, (usize, usize)>,
    allocations: u64,
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
    outputs: Vec<NodeId>,
    arena: CudaF32Arena,
    packed_operands: PackedOperands,
    // sorted; boxed slice keeps CudaPlan under clippy's large_enum_variant gap against the cpu plan
    resident_nodes: Box<[NodeId]>,
}

impl CudaF32Arena {
    /// Number of device allocations performed by this arena.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }
}

impl CudaDriver {
    /// Initializes the CUDA primary context for `ordinal`.
    ///
    /// This is the first real device boundary: on a machine without a
    /// visible NVIDIA device it returns a typed `Driver` error rather than
    /// pretending that emitted CUDA source is executable.
    pub fn new(ordinal: usize) -> Result<Self, CudaDriverError> {
        // the dynamic loader otherwise panics before it can return a driver error.
        if !unsafe { cuda_sys::is_culib_present() } {
            return Err(CudaDriverError::Driver(
                "cuda driver library unavailable".into(),
            ));
        }
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        Ok(Self {
            context,
            stream,
            modules: Arc::new(Mutex::new(BTreeMap::new())),
            resident_buffers: Arc::new(Mutex::new(BTreeMap::new())),
            kernel_compilations: Arc::new(AtomicU64::new(0)),
            packed_upload_bytes: Arc::new(AtomicU64::new(0)),
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
        self.kernel_compilations.load(Ordering::Relaxed)
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

    /// Returns the CUDA driver's current `(free_bytes, total_bytes)` view.
    /// This is the authoritative device-memory measurement used by a tier
    /// fit check; host RSS and checkpoint file size are only proxies.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaDriverError> {
        Ok(self.context.mem_get_info()?)
    }

    fn function_for(&self, kernel: &CudaKernel) -> Result<CudaFunction, CudaDriverError> {
        self.cached_function(&kernel.entry, &kernel.entry, &kernel.source, || {
            // nvrtc has the same panic-on-missing-library loader contract.
            if !unsafe { nvrtc_sys::is_culib_present() } {
                return Err(CudaDriverError::Nvrtc("nvrtc library unavailable".into()));
            }
            let ptx = compile_cuda_source(&kernel.source)?;
            Ok(self.context.load_module(ptx)?)
        })
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
            None => {
                let module = build_module()?;
                self.kernel_compilations.fetch_add(1, Ordering::Relaxed);
                module
            }
        };
        let function = module.load_function(entry)?;
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
            Ok(self.context.load_module(Ptx::from_src(ptx_source))?)
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
        self.plan_with_options(program, symbols, outputs, numeric_policy, named, None)
    }

    pub fn plan_with_config(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        named: &[(&str, QuantizedBlock<'_>)],
        config: &RuntimeConfig,
    ) -> Result<CudaPlan, CudaDriverError> {
        self.plan_with_options(
            program,
            symbols,
            outputs,
            config.numeric,
            named,
            Some(config),
        )
    }

    fn plan_with_options(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        numeric_policy: NumericPolicy,
        named: &[(&str, QuantizedBlock<'_>)],
        config: Option<&RuntimeConfig>,
    ) -> Result<CudaPlan, CudaDriverError> {
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
        let mut resolved = prune_dead(
            if config.is_some_and(|value| !value.allow_fusion) {
                bind_unfused(program, &shapes, outputs, numeric_policy)
            } else {
                bind_with_fusion(program, &shapes, outputs, false, numeric_policy)
            }
            .map_err(|_| CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "graph binding failed",
            })?,
            outputs,
        );
        correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());
        for bound in &resolved {
            validate_cuda_dtype(bound)?;
        }
        reject_unsupported_kinds(&resolved)?;
        let mut kernels = Vec::with_capacity(resolved.len());
        let mut uniforms = Vec::with_capacity(resolved.len());
        for bound in &resolved {
            let kernel = crate::emit_cuda_with_policy(bound, &packed_operands, numeric_policy)
                .map_err(|error| CudaDriverError::Emit {
                    node: bound.node,
                    error: error.to_string(),
                })?;
            let packed_uniforms =
                crate::cuda::pack_cuda_uniforms(bound).map_err(|error| CudaDriverError::Emit {
                    node: bound.node,
                    error: error.to_string(),
                })?;
            kernels.push(kernel);
            uniforms.push(packed_uniforms);
        }
        let plan = CudaPlan {
            constraints: config.map(|value| value.constraints.clone()),
            block_contract: blocks
                .iter()
                .map(cuda_input_contract)
                .collect::<Result<Vec<_>, _>>()?,
            driver: self.clone(),
            program: program.to_vec(),
            shapes,
            resolved,
            kernels,
            uniforms,
            block_nodes,
            outputs: outputs.to_vec(),
            arena: self.new_arena(),
            packed_operands,
            resident_nodes: Box::default(),
        };
        if config.is_some() {
            plan.validate_resources(&blocks, None)?;
            for kernel in &plan.kernels {
                self.function_for(kernel)?;
            }
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
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
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
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
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
            self.stream.memcpy_htod(values, buffer)?;
            return Ok(());
        }
        let buffer = self.stream.clone_htod(values)?;
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
            self.stream.clone_htod(&values)?
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
        self.launch_f32_persistent_sized_with_sync(arena, kernel, uniforms, output_len, true, None)
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
            false,
            deferred_fault,
        )
    }

    fn launch_f32_persistent_sized_with_sync(
        &self,
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        synchronize: bool,
        deferred_fault: Option<(&CudaSlice<u32>, usize, usize)>,
    ) -> Result<(), CudaDriverError> {
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
        let input_nodes: Vec<NodeId> = kernel
            .bindings
            .iter()
            .filter_map(|binding| match binding {
                Binding::Input(node) | Binding::Indices(node) => Some(*node),
                Binding::Output(_) | Binding::Uniforms | Binding::Fault => None,
                Binding::ExpertPayloads(_) | Binding::ExpertDescriptors(_) | Binding::Scratch => {
                    None
                }
            })
            .collect();
        for node in &input_nodes {
            if !arena.buffers.contains_key(node) {
                return Err(CudaDriverError::MissingBuffer(*node));
            }
        }
        let fault_count = kernel
            .bindings
            .iter()
            .filter(|binding| matches!(binding, Binding::Fault))
            .count();
        if fault_count > 1 {
            return Err(CudaDriverError::BindingContract);
        }
        let function = self.function_for(kernel)?;
        self.upload_uniform(arena, output_node, uniforms)?;
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
        let device_uniforms = arena
            .uniforms
            .get(&output_node)
            .ok_or(CudaDriverError::BindingContract)?;
        let mut device_output = match arena.buffers.remove(&output_node) {
            Some(CudaGraphBuffer::F32(buffer)) => buffer,
            Some(CudaGraphBuffer::Bytes(_)) => return Err(CudaDriverError::BindingContract),
            None => {
                arena.allocations += 1;
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
        }
        if input_nodes
            .iter()
            .any(|node| !arena.buffers.contains_key(node))
        {
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
        let mut args = self.stream.launch_builder(&function);
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
        let launch_result = unsafe { args.launch(config) };
        Self::restore_extra_outputs(arena, extra_buffers);
        if let Err(error) = launch_result {
            arena
                .buffers
                .insert(output_node, CudaGraphBuffer::F32(device_output));
            return Err(error.into());
        }
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
                Ok(self.stream.alloc_zeros::<f32>(length)?)
            }
        }
    }

    fn restore_extra_outputs(arena: &mut CudaF32Arena, buffers: Vec<(NodeId, CudaSlice<f32>)>) {
        for (node, buffer) in buffers {
            arena.buffers.insert(node, CudaGraphBuffer::F32(buffer));
        }
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

fn cuda_supports_kind(kind: &BoundOpKind) -> bool {
    matches!(
        kind,
        BoundOpKind::Elementwise { .. }
            | BoundOpKind::Reduce { .. }
            | BoundOpKind::Iota
            | BoundOpKind::Constant { .. }
            | BoundOpKind::MoeTopK { .. }
    )
}

/// Rejects the whole graph at once, naming every unsupported kind with its
/// node count, so a model's full gap list is one error rather than one node at a time.
fn reject_unsupported_kinds(resolved: &[BoundOp]) -> Result<(), CudaDriverError> {
    let mut census: BTreeMap<&'static str, (usize, NodeId)> = BTreeMap::new();
    for bound in resolved
        .iter()
        .filter(|bound| !cuda_supports_kind(&bound.kind))
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
    if bound.dtype == DType::Float32 || cuda_integer_dtype(bound.dtype) {
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
        DType::Bool | DType::Int8 | DType::UInt8 | DType::Int32 | DType::UInt32
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

/// NVRTC does not ship `cuda_fp16.h`, so the emitted source needs a toolkit
/// include directory.  Resolution order: `PROXIMA_CUDA_INCLUDE_PATH`,
/// `CUDA_PATH/include`, `CUDA_HOME/include`, `/usr/local/cuda/include`,
/// `/opt/cuda/include`; the first one that contains the header wins.
fn compile_cuda_source(source: &str) -> Result<Ptx, CudaDriverError> {
    let mut options = CompileOptions::default();
    if source.contains(CUDA_REQUIRED_HEADER) {
        let include_dir =
            resolve_cuda_include_dir(CUDA_REQUIRED_HEADER, &CUDA_SYSTEM_INCLUDE_DIRS)?;
        options
            .include_paths
            .push(include_dir.to_string_lossy().into_owned());
    }
    compile_ptx_with_opts(source, options)
        .map_err(|error| CudaDriverError::Nvrtc(error.to_string()))
}

impl CudaPlan {
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
        for (bound, uniform) in self.resolved.iter().zip(&self.uniforms) {
            let bytes = cuda_float_bytes(self.shapes.of(bound.node))?;
            match self.arena.buffers.get(&bound.node) {
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
                bound.dtype != DType::Float32
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
            .ok_or(CudaDriverError::ResourceOverflow)?;
        let additional = total
            .checked_sub(retained)
            .ok_or(CudaDriverError::ResourceOverflow)?;
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

    /// Executes the planned graph with name-resolved inputs and returns the
    /// requested outputs. Packed bytes stay on device in the persistent arena.
    pub fn execute_named(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&BTreeSet<&str>>,
    ) -> Result<Evaluated, CudaDriverError> {
        let host_timing = std::env::var_os("PROXIMA_CUDA_HOST_TIMING").is_some();
        let upload_started = host_timing.then(std::time::Instant::now);
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
                "cuda_host_timing: phase=upload_ms value={:.3} resident_hits={} resident_misses={}",
                started.elapsed().as_secs_f64() * 1_000.0,
                resident_hits,
                resident_misses
            );
        }
        let trace = std::env::var_os("PROXIMA_CUDA_TRACE").is_some();
        let timing = std::env::var_os("PROXIMA_CUDA_TIMING").is_some();
        let mut timing_rows: Vec<(NodeId, u128, usize)> = Vec::new();
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
        let dispatch_started = host_timing.then(std::time::Instant::now);
        for (bound, (kernel, uniforms)) in self
            .resolved
            .iter()
            .zip(self.kernels.iter().zip(self.uniforms.iter()))
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
            let output_len = match &bound.kind {
                BoundOpKind::Reduce {
                    keep: Keep::Reduce,
                    output_axes,
                    epilogue_broadcast_axes,
                    ..
                } if epilogue_broadcast_axes.is_empty() => output_axes
                    .iter()
                    .map(|axis| bound.extents[*axis as usize] as usize)
                    .product(),
                _ => bound
                    .extents
                    .iter()
                    .map(|extent| *extent as usize)
                    .product(),
            };
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
                    )
                };
                launch.map_err(|error| match error {
                    typed @ CudaDriverError::GatherIndexOutOfRange { .. } => typed,
                    other => CudaDriverError::Execute {
                        node: bound.node,
                        error: other.to_string(),
                    },
                })?;
                if let Some(started) = node_started {
                    timing_rows.push((bound.node, started.elapsed().as_micros(), output_len));
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
        }
        if let Some(started) = dispatch_started {
            eprintln!(
                "cuda_host_timing: phase=submit_ms value={:.3} nodes={}",
                started.elapsed().as_secs_f64() * 1_000.0,
                self.resolved.len()
            );
        }
        let sync_started = host_timing.then(std::time::Instant::now);
        self.driver.stream.synchronize()?;
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
            for (rank, (node, micros, output_len)) in timing_rows.iter().take(12).enumerate() {
                eprintln!(
                    "cuda_timing_top rank={} node={:?} elapsed_ms={:.3} output_len={}",
                    rank + 1,
                    node,
                    *micros as f64 / 1000.0,
                    output_len
                );
            }
        }
        let host_readback_started = host_timing.then(std::time::Instant::now);
        let mut results = Vec::with_capacity(self.outputs.len());
        let readback_started = timing.then(std::time::Instant::now);
        for node in &self.outputs {
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

    /// Number of device allocations made by the persistent graph arena.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.arena.allocations()
    }
}

// `Q5_1`/`Q5_0` are excluded here the same way `Iq4Nl`/`Iq2Xs`/`Iq3Xxs` are --
// no CUDA unpack kernel exists for any of them yet, so all stay routed through
// `None`. Every unsupported codec is named so a new `Codec` variant forces a
// decision here instead of slipping through a wildcard.
fn packed_codec(block: &QuantizedBlock<'_>) -> Option<Codec> {
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
        | Codec::Float16
        | Codec::BFloat16 => Some(*codec),
        Codec::Q5_1
        | Codec::Q5_0
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
        | Codec::Q1_0
        | Codec::Q2_0
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
        PushKernelArg, add_one_kernel, launch_blocks, packed_bytes, packed_codec,
        resolve_cuda_include_dir, validate_cuda_dtype,
    };
    use crate::cuda::CudaGridSpec;
    use crate::msl::Codec;
    use proxima_tensor::{BoundOp, BoundOpKind, DType, NodeId, QuantizedBlock};

    const CUDA_UNPACKABLE: [Codec; 20] = [
        Codec::Q5_1,
        Codec::Q5_0,
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
        Codec::Q1_0,
        Codec::Q2_0,
    ];

    const CODECS_WITH_CUDA_KERNEL: [Codec; 9] = [
        Codec::Q2K,
        Codec::Q3K,
        Codec::Q4K,
        Codec::Q5K,
        Codec::Q6K,
        Codec::Q8_0,
        Codec::Q4_0,
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
            dtype: DType::Float16,
            extents: vec![1],
            kind: BoundOpKind::Constant { value: 0.0 },
        };

        assert!(
            matches!(
                validate_cuda_dtype(&bound),
                Err(CudaDriverError::UnsupportedDtype {
                    node: NodeId(157),
                    dtype: DType::Float16,
                })
            ),
            "half output must be rejected"
        );
    }

    #[test]
    fn integer_bound_ops_are_accepted_as_integer_valued_f32() {
        for dtype in [
            DType::Bool,
            DType::Int8,
            DType::UInt8,
            DType::Int32,
            DType::UInt32,
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
        let driver =
            CudaDriver::new(0).expect("a cuda device is required for the module cache gate");
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
