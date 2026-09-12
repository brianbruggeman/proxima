//! Minimal real CUDA driver boundary over the [`crate::cuda`] emitter.
//!
//! The emitter is deliberately usable without NVIDIA libraries. This module
//! is the separate runtime half: cudarc initializes the CUDA Driver API,
//! NVRTC compiles CUDA C into PTX, and a CUDA stream launches the resulting
//! kernel. The smoke method is intentionally tiny and numerical, so a host
//! with a CUDA device can prove initialization, compilation, argument
//! marshalling, launch, synchronization, and device-to-host readback before
//! the full graph binding layer is connected.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::cuda::{CudaGridSpec, CudaKernel};
use crate::msl::{Binding, PackedCodec, PackedOperands};
use proxima_tensor::{
    BoundOp, DType, Evaluated, NodeId, NumericPolicy, Op, QuantizedBlock, Shapes, bind_with_fusion,
    block_node_ids, infer, prune_dead, resolve_named_blocks,
};

/// Errors from CUDA initialization, NVRTC compilation, or execution.
#[derive(Debug, thiserror::Error)]
pub enum CudaDriverError {
    #[error("CUDA driver: {0}")]
    Driver(String),
    #[error("CUDA NVRTC: {0}")]
    Nvrtc(String),
    #[error("CUDA smoke input is too large for the launch ABI: {0} elements")]
    InputTooLarge(usize),
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

/// One CUDA context and stream owned by a caller's inference runtime.
#[derive(Debug, Clone)]
pub struct CudaDriver {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: Arc<std::sync::Mutex<BTreeMap<String, CachedCudaModule>>>,
    resident_buffers: Arc<std::sync::Mutex<BTreeMap<NodeId, ((usize, usize), CudaGraphBuffer)>>>,
    kernel_compilations: Arc<AtomicU64>,
}

#[derive(Debug)]
struct CachedCudaModule {
    source: String,
    module: Arc<CudaModule>,
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
    Bytes(CudaSlice<u8>),
}

/// A reusable CUDA graph for the currently connected dense/reduction CUDA ABI.
/// Planning resolves shapes, fusion, dead nodes, and kernel structure once;
/// execution refreshes only named inputs and reuses the device arena.
#[derive(Debug)]
pub struct CudaPlan {
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
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        Ok(Self {
            context,
            stream,
            modules: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            resident_buffers: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            kernel_compilations: Arc::new(AtomicU64::new(0)),
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

    /// Returns the CUDA driver's current `(free_bytes, total_bytes)` view.
    /// This is the authoritative device-memory measurement used by a tier
    /// fit check; host RSS and checkpoint file size are only proxies.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaDriverError> {
        Ok(self.context.mem_get_info()?)
    }

    fn function_for(&self, kernel: &CudaKernel) -> Result<CudaFunction, CudaDriverError> {
        let cached = self
            .modules
            .lock()
            .map_err(|_| CudaDriverError::Driver("CUDA module cache poisoned".into()))?
            .get(&kernel.entry)
            .filter(|cached| cached.source == kernel.source)
            .map(|cached| cached.module.clone());
        let module = match cached {
            Some(module) => module,
            None => {
                let ptx = compile_cuda_source(&kernel.source)
                    .map_err(|error| CudaDriverError::Nvrtc(error.to_string()))?;
                let module = self.context.load_module(ptx)?;
                self.modules
                    .lock()
                    .map_err(|_| CudaDriverError::Driver("CUDA module cache poisoned".into()))?
                    .insert(
                        kernel.entry.clone(),
                        CachedCudaModule {
                            source: kernel.source.clone(),
                            module: module.clone(),
                        },
                    );
                self.kernel_compilations.fetch_add(1, Ordering::Relaxed);
                module
            }
        };
        Ok(module.load_function(&kernel.entry)?)
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
        let cached = self
            .modules
            .lock()
            .map_err(|_| CudaDriverError::Driver("CUDA module cache poisoned".into()))?
            .get(&cache_key)
            .filter(|cached| cached.source == ptx_source)
            .map(|cached| cached.module.clone());
        let module = match cached {
            Some(module) => module,
            None => {
                let module = self.context.load_module(Ptx::from_src(ptx_source))?;
                self.modules
                    .lock()
                    .map_err(|_| CudaDriverError::Driver("CUDA module cache poisoned".into()))?
                    .insert(
                        cache_key,
                        CachedCudaModule {
                            source: ptx_source.to_owned(),
                            module: module.clone(),
                        },
                    );
                self.kernel_compilations.fetch_add(1, Ordering::Relaxed);
                module
            }
        };
        Ok(module.load_function(entry)?)
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
        let blocks = resolve_named_blocks(program, named).map_err(|_| {
            CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "named input resolution failed",
            }
        })?;
        let block_nodes = block_node_ids(program);
        let packed_operands = block_nodes
            .iter()
            .zip(blocks.iter())
            .filter_map(|(node, block)| packed_codec(block).map(|codec| (*node, codec)))
            .collect();
        let shapes =
            infer(program, symbols).map_err(|error| CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: if error.to_string().is_empty() {
                    "shape inference failed"
                } else {
                    "shape inference failed"
                },
            })?;
        let resolved = prune_dead(
            bind_with_fusion(program, &shapes, outputs, false, numeric_policy).map_err(|_| {
                CudaDriverError::UnsupportedGraph {
                    node: NodeId(0),
                    reason: "graph binding failed",
                }
            })?,
            outputs,
        );
        for bound in &resolved {
            validate_cuda_dtype(bound)?;
            if !matches!(
                bound.kind,
                proxima_tensor::BoundOpKind::Elementwise { .. }
                    | proxima_tensor::BoundOpKind::Reduce { .. }
                    | proxima_tensor::BoundOpKind::Iota
                    | proxima_tensor::BoundOpKind::Constant { .. }
            ) {
                return Err(CudaDriverError::UnsupportedGraph {
                    node: bound.node,
                    reason: "unsupported op kind",
                });
            }
        }
        let mut kernels = Vec::with_capacity(resolved.len());
        let mut uniforms = Vec::with_capacity(resolved.len());
        for bound in &resolved {
            let kernel = crate::emit_cuda_with_policy(bound, &packed_operands, numeric_policy)
                .map_err(|error| CudaDriverError::Emit {
                    node: bound.node,
                    error: error.to_string(),
                })?;
            let packed_uniforms = crate::cuda::pack_cuda_uniforms(bound).map_err(|error| {
                CudaDriverError::Emit {
                    node: bound.node,
                    error: error.to_string(),
                }
            })?;
            kernels.push(kernel);
            uniforms.push(packed_uniforms);
        }
        Ok(CudaPlan {
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
        })
    }

    /// Uploads or refreshes one named f32 input, retaining its allocation for
    /// later dispatches when the element count is unchanged.
    pub fn upload_f32(
        &self,
        arena: &mut CudaF32Arena,
        node: NodeId,
        values: &[f32],
    ) -> Result<(), CudaDriverError> {
        if let Some(buffer) = arena.buffers.get_mut(&node) {
            if let CudaGraphBuffer::F32(buffer) = buffer {
                if buffer.len() == values.len() {
                    self.stream.memcpy_htod(values, buffer)?;
                    return Ok(());
                }
            }
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
        if let Some(CudaGraphBuffer::Bytes(buffer)) = arena.buffers.get_mut(&node) {
            if buffer.len() == values.len() {
                self.stream.memcpy_htod(values, buffer)?;
                return Ok(());
            }
        }
        let buffer = self.stream.clone_htod(values)?;
        arena.buffers.insert(node, CudaGraphBuffer::Bytes(buffer));
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
        if let Some(buffer) = arena.uniforms.get_mut(&node) {
            if buffer.len() == values.len() {
                self.stream.memcpy_htod(values, buffer)?;
                return Ok(());
            }
        }
        let buffer = self.stream.clone_htod(values)?;
        arena.uniforms.insert(node, buffer);
        arena.allocations += 1;
        Ok(())
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
    /// allocation count is an observable reuse oracle: three allocations are
    /// expected (input plus two distinct output nodes), not one new allocation
    /// per dispatch plus a host round trip.
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
        let block_width = kernel.grid.block_width.unwrap_or(256).max(1);
        let config = LaunchConfig {
            grid_dim: (
                (kernel.grid.threads as u32).div_ceil(block_width as u32),
                1,
                1,
            ),
            block_dim: (block_width as u32, 1, 1),
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
        self.launch_f32_persistent_sized_with_sync(arena, kernel, uniforms, output_len, true)
    }

    /// Enqueues a kernel on the driver's stream without synchronizing.
    ///
    /// CUDA stream order already preserves the graph's dependency order. The
    /// graph plan uses this form for ordinary kernels and performs one stream
    /// synchronization after the whole evaluation, avoiding a host/device
    /// round trip after every node. Fault-buffer kernels remain on the
    /// synchronous path because their diagnostic readback must be observed
    /// before the next operation can be trusted.
    pub fn launch_f32_persistent_sized_async(
        &self,
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
    ) -> Result<(), CudaDriverError> {
        if kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Fault))
        {
            return Err(CudaDriverError::BindingContract);
        }
        self.launch_f32_persistent_sized_with_sync(arena, kernel, uniforms, output_len, false)
    }

    fn launch_f32_persistent_sized_with_sync(
        &self,
        arena: &mut CudaF32Arena,
        kernel: &CudaKernel,
        uniforms: &[u8],
        output_len: usize,
        synchronize: bool,
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
        let block_width = kernel.grid.block_width.unwrap_or(256).max(1);
        let config = LaunchConfig {
            grid_dim: (
                (kernel.grid.threads as u32).div_ceil(block_width as u32),
                1,
                1,
            ),
            block_dim: (block_width as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut fault_buffer = (fault_count > 0)
            .then(|| self.stream.alloc_zeros::<u32>(fault_count))
            .transpose()?;
        let mut args = self.stream.launch_builder(&function);
        for binding in &kernel.bindings[..output_index] {
            match binding {
                Binding::Input(node) | Binding::Indices(node) => match arena.buffers.get(node) {
                    Some(CudaGraphBuffer::F32(input)) => args.arg(input),
                    Some(CudaGraphBuffer::Bytes(input)) => args.arg(input),
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
                Binding::Fault => args.arg(
                    fault_buffer
                        .as_ref()
                        .ok_or(CudaDriverError::BindingContract)?,
                ),
                _ => return Err(CudaDriverError::BindingContract),
            };
        }
        let launch_result = unsafe { args.launch(config) };
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

fn validate_cuda_dtype(bound: &BoundOp) -> Result<(), CudaDriverError> {
    if bound.dtype != DType::Float32 {
        return Err(CudaDriverError::UnsupportedDtype {
            node: bound.node,
            dtype: bound.dtype,
        });
    }
    Ok(())
}

/// Compiles generated CUDA C with an explicit header search path when the
/// runtime distribution does not install headers in NVRTC's default search
/// list.  `PROXIMA_CUDA_INCLUDE_PATH` is the deployment escape hatch for
/// toolkit/container layouts; `CUDA_PATH` and the conventional Linux path are
/// harmless fallbacks.
fn compile_cuda_source(source: &str) -> Result<Ptx, cudarc::nvrtc::CompileError> {
    let mut options = CompileOptions::default();
    let include_path = std::env::var_os("PROXIMA_CUDA_INCLUDE_PATH")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("CUDA_PATH").map(|path| std::path::PathBuf::from(path).join("include"))
        })
        .or_else(|| {
            let path = std::path::Path::new("/usr/local/cuda/include");
            path.is_dir().then(|| path.to_owned())
        });
    if let Some(path) = include_path {
        options
            .include_paths
            .push(path.to_string_lossy().into_owned());
    }
    compile_ptx_with_opts(source, options)
}

impl CudaPlan {
    /// Executes the planned graph with name-resolved inputs and returns the
    /// requested outputs. Packed bytes stay on device in the persistent arena.
    pub fn execute_named(
        &mut self,
        named: &[(&str, QuantizedBlock<'_>)],
        resident_names: Option<&std::collections::BTreeSet<&str>>,
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
        for (node, block) in self.block_nodes.iter().copied().zip(blocks.iter()) {
            let name = match self.program.get(node.0 as usize) {
                Some(Op::Input { name: Some(name), .. }) => name.as_str(),
                _ => "",
            };
            let source = (block_address(block), block_length(block));
            let resident = resident_names.is_some_and(|names| names.contains(name));
            if resident && self.arena.resident_sources.get(&node) == Some(&source) {
                resident_hits += 1;
                continue;
            }
            if resident {
                let cached = self
                    .driver
                    .resident_buffers
                    .lock()
                    .map_err(|_| CudaDriverError::Driver("CUDA resident cache poisoned".into()))?
                    .get(&node)
                    .filter(|(cached_source, _)| *cached_source == source)
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
                    if values.is_empty() {
                        self.arena.allocations += 1;
                        self.arena.buffers.insert(
                            node,
                            CudaGraphBuffer::F32(self.driver.stream.alloc_zeros::<f32>(0)?),
                        );
                    } else {
                        self.driver.upload_f32(&mut self.arena, node, values)?;
                    }
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
                        .map_err(|_| CudaDriverError::Driver("CUDA resident cache poisoned".into()))?
                        .insert(node, (source, buffer));
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
                proxima_tensor::BoundOpKind::Reduce {
                    keep: proxima_tensor::Keep::Reduce,
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
                    (trace_node == Some(bound.node))
                        .then(|| format!("{bound:?}"))
                        .unwrap_or_default(),
                );
            }
            // A zero-extent intermediate is a valid tensor, not a valid CUDA
            // launch.  Skip the dispatch; downstream reductions still see
            // their identity value over the empty axis.
            if output_len == 0 {
                let needs_zero_buffer = !matches!(
                    self.arena.buffers.get(&bound.node),
                    Some(CudaGraphBuffer::F32(buffer)) if buffer.len() == 0
                );
                if needs_zero_buffer {
                    self.arena.allocations += 1;
                    self.arena.buffers.insert(
                        bound.node,
                        CudaGraphBuffer::F32(self.driver.stream.alloc_zeros::<f32>(0)?),
                    );
                }
            } else if let proxima_tensor::BoundOpKind::Constant { value } = bound.kind {
                self.driver.materialize_constant(
                    &mut self.arena,
                    bound.node,
                    value,
                    output_len,
                )?;
            } else {
                let launch = if timing
                    || kernel
                    .bindings
                    .iter()
                    .any(|binding| matches!(binding, Binding::Fault))
                {
                    self.driver.launch_f32_persistent_sized(
                        &mut self.arena,
                        &kernel,
                        &uniforms,
                        output_len,
                    )
                } else {
                    self.driver.launch_f32_persistent_sized_async(
                        &mut self.arena,
                        &kernel,
                        &uniforms,
                        output_len,
                    )
                };
                launch.map_err(|error| CudaDriverError::Execute {
                    node: bound.node,
                    error: error.to_string(),
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

fn packed_codec(block: &QuantizedBlock<'_>) -> Option<PackedCodec> {
    match block {
        QuantizedBlock::Q2K(_) => Some(PackedCodec::Q2K),
        QuantizedBlock::Q3K(_) => Some(PackedCodec::Q3K),
        QuantizedBlock::Q4K(_) => Some(PackedCodec::Q4K),
        QuantizedBlock::Q5K(_) => Some(PackedCodec::Q5K),
        QuantizedBlock::Q6K(_) => Some(PackedCodec::Q6K),
        QuantizedBlock::Q8_0(_) => Some(PackedCodec::Q8_0),
        QuantizedBlock::Q4_0(_) => Some(PackedCodec::Q4_0),
        QuantizedBlock::Float16(_) => Some(PackedCodec::Float16),
        QuantizedBlock::BFloat16(_) => Some(PackedCodec::BFloat16),
        QuantizedBlock::Float32(_)
        | QuantizedBlock::Int32(_)
        | QuantizedBlock::Q5_1(_)
        | QuantizedBlock::Iq4Nl(_)
        | QuantizedBlock::Iq2Xs(_)
        | QuantizedBlock::Iq3Xxs(_) => None,
    }
}

fn packed_bytes<'a>(block: &QuantizedBlock<'a>) -> &'a [u8] {
    match block {
        QuantizedBlock::Q2K(bytes)
        | QuantizedBlock::Q3K(bytes)
        | QuantizedBlock::Q4K(bytes)
        | QuantizedBlock::Q5K(bytes)
        | QuantizedBlock::Q6K(bytes)
        | QuantizedBlock::Q8_0(bytes)
        | QuantizedBlock::Q4_0(bytes)
        | QuantizedBlock::Float16(bytes)
        | QuantizedBlock::BFloat16(bytes) => bytes,
        QuantizedBlock::Float32(_)
        | QuantizedBlock::Int32(_)
        | QuantizedBlock::Q5_1(_)
        | QuantizedBlock::Iq4Nl(_)
        | QuantizedBlock::Iq2Xs(_)
        | QuantizedBlock::Iq3Xxs(_) => &[],
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
        QuantizedBlock::Float32(values) => values.len() * core::mem::size_of::<f32>(),
        QuantizedBlock::Int32(values) => values.len() * core::mem::size_of::<i32>(),
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

#[cfg(test)]
mod tests {
    use super::{CudaDriverError, validate_cuda_dtype};
    use proxima_tensor::{BoundOp, BoundOpKind, DType, NodeId};

    #[test]
    fn non_f32_bound_op_reports_node_and_dtype() {
        let bound = BoundOp {
            node: NodeId(157),
            dtype: DType::Int32,
            extents: vec![1],
            kind: BoundOpKind::Constant { value: 0.0 },
        };

        let error = validate_cuda_dtype(&bound).expect_err("integer output must be rejected");
        assert!(matches!(
            error,
            CudaDriverError::UnsupportedDtype {
                node: NodeId(157),
                dtype: DType::Int32,
            }
        ));
    }
}
