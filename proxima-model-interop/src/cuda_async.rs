//! Async CUDA service integration for owned graph evaluations.

use core::fmt::{Debug, Formatter};
use core::future::Future;
#[cfg(test)]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::string::String;
use std::sync::Arc;
#[cfg(test)]
use std::sync::mpsc::{Receiver, Sender, channel};

use conflaguration::Validate;
use omega::config::{RuntimeConfig, SelectionMode};
use omega::owner_config::OwnerConfig;
use omega::device_owner::{
    OwnerCommand as DeviceOwnerCommand, OwnerFlow, PlanLease, PlanRegistry, PlanRegistryError,
    execute_registered_plan, run_owner,
};
use prime::os::background::ProximaBackgroundPool;
use proxima_core::ProximaError;
use proxima_numeric::codec::Codec;
use proxima_primitives::pipe::PipeExt;
use proxima_primitives::pipe::{
    CreditAcquireError, CreditCapacityError, CreditGate, CreditLease, Pipe,
};
#[cfg(test)]
use proxima_primitives::sync::blocking::Mutex;
use proxima_primitives::sync::mpsc::{
    Receiver as CommandReceiver, Sender as CommandSender, channel as command_channel,
};
use proxima_primitives::sync::oneshot;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::{NodeId, Op};
use proxima_numeric::{NumericPolicy};


#[cfg(test)]
mod owner_failure_tests;
#[cfg(test)]
mod prime_progress_tests;

type OwnerCommand = DeviceOwnerCommand<
    OwnedGraph,
    omega::CudaPlan,
    AdmittedGpuJob,
    CompletedGpuJob,
    CudaPlanRegistrationError,
>;

#[derive(Debug, thiserror::Error)]
pub enum CudaPlanRegistrationError {
    #[error("async cuda planning does not support {mode}")]
    UnsupportedMode { mode: &'static str },
    #[error("cuda service closed after execution failure")]
    ServiceClosed,
    #[error(transparent)]
    Registry(#[from] PlanRegistryError),
    #[error(transparent)]
    Driver(#[from] omega::CudaDriverError),
}

#[derive(Debug, thiserror::Error)]
pub enum CudaExecutionError {
    #[error(transparent)]
    Plan(#[from] PlanRegistryError),
    #[error(transparent)]
    Driver(#[from] omega::CudaDriverError),
    #[error("cuda service closed after execution failure")]
    ServiceClosed,
    #[error("cuda service closed before accepting the job")]
    ServiceClosedWithInput(OwnedGpuJob),
    #[error("cuda owner stopped before returning the result")]
    OwnerStopped,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuAdmissionError {
    #[error("cuda service closed while the job waited for capacity")]
    Closed(OwnedGpuJob),
    #[error("cuda job exceeds the service capacity")]
    OverCapacity(OwnedGpuJob),
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPipeError {
    #[error(transparent)]
    Admission(#[from] GpuAdmissionError),
    #[error(transparent)]
    Execution(#[from] CudaExecutionError),
}

/// Owned plan reference and all input data accepted for one graph evaluation.
pub struct OwnedGpuJob {
    lease: PlanLease,
    inputs: OwnedGpuInputs,
}

impl Debug for OwnedGpuJob {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedGpuJob")
            .field("lease", &self.lease)
            .field("owned_bytes", &self.inputs.owned_bytes)
            .finish()
    }
}

impl OwnedGpuJob {
    /// Recover the plan lease and owned input payload after admission fails.
    pub fn into_parts(self) -> (PlanLease, OwnedGpuInputs) {
        (self.lease, self.inputs)
    }
}

struct AdmittedGpuJob {
    job: OwnedGpuJob,
    credit: CreditLease,
}

impl Debug for AdmittedGpuJob {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("AdmittedGpuJob")
            .field("job", &self.job)
            .finish_non_exhaustive()
    }
}

struct OwnedGraph {
    program: Vec<Op>,
    symbols: Vec<u64>,
    outputs: Vec<NodeId>,
    numeric_policy: NumericPolicy,
    exact_activations: bool,
    omega_runtime: Option<RuntimeConfig>,
    inputs: OwnedGpuInputs,
}

/// The plan-shaping options that travel together into `register_graph`.
#[derive(Clone, Copy)]
pub struct CudaPlanOptions<'config> {
    pub numeric_policy: NumericPolicy,
    pub exact_activations: bool,
    pub omega_runtime: Option<&'config RuntimeConfig>,
}

pub(crate) fn validate_cuda_plan_options(
    numeric_policy: NumericPolicy,
    exact_activations: bool,
    omega_runtime: Option<&RuntimeConfig>,
) -> Result<(), CudaPlanRegistrationError> {
    let Some(runtime) = omega_runtime else {
        return Ok(());
    };
    if runtime.engine != "gpu" {
        return Err(CudaPlanRegistrationError::UnsupportedMode {
            mode: "non-gpu engine",
        });
    }
    if runtime.gpu_driver != "cuda" && !selects_hip(runtime) {
        return Err(CudaPlanRegistrationError::UnsupportedMode {
            mode: "non-cuda driver selection",
        });
    }
    if runtime.selection.mode != SelectionMode::Configured {
        return Err(CudaPlanRegistrationError::UnsupportedMode {
            mode: "candidate measurement selection",
        });
    }
    if runtime.numeric != numeric_policy || runtime.exact_activations != exact_activations {
        return Err(CudaPlanRegistrationError::UnsupportedMode {
            mode: "conflicting numeric or activation policy",
        });
    }
    runtime
        .validate()
        .map_err(|_| CudaPlanRegistrationError::UnsupportedMode {
            mode: "invalid runtime configuration",
        })
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FakePlanInputs {
    pub(crate) numeric_policy: NumericPolicy,
    pub(crate) exact_activations: bool,
    pub(crate) omega_runtime: Option<RuntimeConfig>,
}

enum RegisteredPlan {
    // boxed: the plan is 352 bytes against the fake variant's few words, which trips clippy's large_enum_variant.
    Cuda(Box<omega::CudaPlan>),
    // boxed for the same reason as the cuda plan.
    Rocm(Box<omega::backend::hip::runtime::HipPlan>),
    #[cfg(test)]
    Fake {
        program: Vec<Op>,
        outputs: Vec<NodeId>,
        symbols: Vec<u64>,
    },
}

struct OwnerState {
    plans: PlanRegistry<RegisteredPlan>,
    driver: Option<omega::CudaDriver>,
    completed_jobs: u64,
    #[cfg(test)]
    fake_owner_hold: Option<(Sender<()>, Receiver<()>)>,
}

struct CompletedGpuJob {
    result: Result<proxima_tensor::cpu::Evaluated, CudaExecutionError>,
    completion_sequence: u64,
    _credit: CreditLease,
}

fn reject_after_execution_failure(command: OwnerCommand, completion_sequence: u64) {
    match command {
        OwnerCommand::ExecutePlan { job, reply } => {
            let AdmittedGpuJob { job, credit } = job;
            drop(job);
            let _ = reply.send(CompletedGpuJob {
                result: Err(CudaExecutionError::ServiceClosed),
                completion_sequence,
                _credit: credit,
            });
        }
        OwnerCommand::RegisterGraph { reply, .. } => {
            let _ = reply.send(Err(CudaPlanRegistrationError::ServiceClosed));
        }
        OwnerCommand::RegisterPlan { reply, .. } => {
            let _ = reply.send(Err(PlanRegistryError::ServiceClosed));
        }
        OwnerCommand::RetirePlan { reply, .. } => {
            let _ = reply.send(Err(PlanRegistryError::ServiceClosed));
        }
        OwnerCommand::DeviceSnapshot { reply } => drop(reply),
        OwnerCommand::InvalidateResidents { reply, .. } => drop(reply),
    }
}

/// An evaluated graph with its completion position on the CUDA owner.
pub struct CudaExecutionReceipt {
    pub evaluated: proxima_tensor::cpu::Evaluated,
    pub completion_sequence: u64,
}

struct GpuCapacityGate<'gate> {
    gate: &'gate CreditGate<ReservationSink>,
}

impl Pipe for GpuCapacityGate<'_> {
    type In = OwnedGpuJob;
    type Out = AdmittedGpuJob;
    type Err = GpuAdmissionError;

    fn call(&self, job: OwnedGpuJob) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            match self.gate.acquire(1).await {
                Ok(credit) => Ok(AdmittedGpuJob { job, credit }),
                Err(CreditAcquireError::Closed) => Err(GpuAdmissionError::Closed(job)),
                Err(CreditAcquireError::OverCapacity) => Err(GpuAdmissionError::OverCapacity(job)),
            }
        }
    }
}

struct CudaExecutionPipe {
    commands: Arc<CommandSender<OwnerCommand>>,
}

struct CudaExecutionReceiptPipe {
    commands: Arc<CommandSender<OwnerCommand>>,
}

impl Pipe for CudaExecutionPipe {
    type In = AdmittedGpuJob;
    type Out = proxima_tensor::cpu::Evaluated;
    type Err = GpuPipeError;

    fn call(&self, job: AdmittedGpuJob) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        submit_cuda_job(Arc::clone(&self.commands), job)
    }
}

impl Pipe for CudaExecutionReceiptPipe {
    type In = AdmittedGpuJob;
    type Out = CudaExecutionReceipt;
    type Err = GpuPipeError;

    fn call(&self, job: AdmittedGpuJob) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        submit_cuda_job_recorded(Arc::clone(&self.commands), job)
    }
}

async fn submit_cuda_job(
    commands: Arc<CommandSender<OwnerCommand>>,
    job: AdmittedGpuJob,
) -> Result<proxima_tensor::cpu::Evaluated, GpuPipeError> {
    Ok(submit_cuda_job_recorded(commands, job).await?.evaluated)
}

async fn submit_cuda_job_recorded(
    commands: Arc<CommandSender<OwnerCommand>>,
    job: AdmittedGpuJob,
) -> Result<CudaExecutionReceipt, GpuPipeError> {
    let (reply, response) = oneshot::channel();
    let command = OwnerCommand::ExecutePlan { job, reply };
    match commands.send(command).await {
        Ok(()) => match response.await {
            Ok(completed) => completed
                .result
                .map(|evaluated| CudaExecutionReceipt {
                    evaluated,
                    completion_sequence: completed.completion_sequence,
                })
                .map_err(GpuPipeError::Execution),
            Err(_) => Err(GpuPipeError::Execution(CudaExecutionError::OwnerStopped)),
        },
        Err(error) => match error.into_inner() {
            OwnerCommand::ExecutePlan { job, .. } => Err(GpuPipeError::Execution(
                CudaExecutionError::ServiceClosedWithInput(job.job),
            )),
            OwnerCommand::DeviceSnapshot { .. }
            | OwnerCommand::RegisterGraph { .. }
            | OwnerCommand::RegisterPlan { .. }
            | OwnerCommand::RetirePlan { .. }
            | OwnerCommand::InvalidateResidents { .. } => {
                unreachable!("execute send returns its execute command")
            }
        },
    }
}

fn borrow_owned_block(block: &OwnedBlock) -> QuantizedBlock<'_> {
    match block {
        OwnedBlock::Float32(values) => QuantizedBlock::Float32(values),
        OwnedBlock::Int32(values) => QuantizedBlock::Int32(values),
        OwnedBlock::Packed { codec, bytes } => QuantizedBlock::Packed {
            codec: **codec,
            bytes,
        },
    }
}

fn selects_hip(runtime: &RuntimeConfig) -> bool {
    matches!(runtime.gpu_driver.as_str(), "hip" | "rocm")
}

fn execute_cuda_plan(
    plan: &mut RegisteredPlan,
    inputs: &OwnedGpuInputs,
) -> Result<proxima_tensor::cpu::Evaluated, omega::CudaDriverError> {
    let named = inputs
        .blocks
        .iter()
        .map(|input| (input.name.as_ref(), borrow_owned_block(&input.block)))
        .collect::<Vec<_>>();
    let resident_names = inputs
        .resident_names
        .iter()
        .map(AsRef::as_ref)
        .collect::<BTreeSet<&str>>();
    match plan {
        RegisteredPlan::Rocm(plan) => plan.execute_named(
            &named,
            (!resident_names.is_empty()).then_some(&resident_names),
        ),
        RegisteredPlan::Cuda(plan) => plan.execute_named(
            &named,
            (!resident_names.is_empty()).then_some(&resident_names),
        ),
        #[cfg(test)]
        RegisteredPlan::Fake {
            program,
            outputs,
            symbols,
        } => proxima_tensor::cpu::evaluate_quantized_named(program, symbols, &named, outputs)
            .map_err(|_| omega::CudaDriverError::UnsupportedGraph {
                node: outputs.last().copied().unwrap_or(NodeId(0)),
                reason: "fake graph evaluation failed",
            }),
    }
}

struct ReservationSink;

impl Pipe for ReservationSink {
    type In = ();
    type Out = ();
    type Err = core::convert::Infallible;

    fn call(&self, _input: ()) -> impl Future<Output = Result<(), Self::Err>> {
        async { Ok(()) }
    }
}

/// Fixed byte ceiling whose owned buffers retain their reservation until drop.
pub struct OwnedByteBudget {
    gate: CreditGate<ReservationSink>,
}

impl OwnedByteBudget {
    /// Build a budget whose credit unit is one owned host byte.
    pub fn new(max_bytes: usize) -> Result<Self, CreditCapacityError> {
        Ok(Self {
            gate: CreditGate::new(ReservationSink, max_bytes)?,
        })
    }

    /// Return a snapshot of unreserved host bytes.
    #[must_use]
    pub fn available_bytes(&self) -> usize {
        self.gate.available()
    }

    /// Copy only after reserving the exact source length.
    pub async fn copy(&self, source: &[u8]) -> Result<OwnedByteBlock, CreditAcquireError> {
        let lease = self.gate.acquire(source.len()).await?;
        Ok(OwnedByteBlock {
            bytes: Arc::from(source),
            _lease: lease,
        })
    }

    /// Own named tensor blocks after reserving their combined copied byte size.
    pub async fn copy_named_blocks(
        &self,
        source: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<OwnedGpuInputs, OwnedInputError> {
        let owned_bytes = source.iter().try_fold(0usize, |total, (name, block)| {
            total
                .checked_add(name.len())
                .and_then(|next| next.checked_add(block_byte_len(*block)?))
        });
        let owned_bytes = owned_bytes.ok_or(OwnedInputError::SizeOverflow)?;
        let lease = self.gate.acquire(owned_bytes).await?;
        let blocks = source
            .iter()
            .map(|(name, block)| OwnedNamedBlock {
                name: Arc::from(*name),
                block: own_block(*block),
            })
            .collect::<Vec<_>>();

        Ok(OwnedGpuInputs {
            blocks: Arc::from(blocks),
            owned_bytes,
            _lease: Arc::new(lease),
            _resident_lease: None,
            resident_names: Arc::from([]),
        })
    }
}

fn block_byte_len(block: QuantizedBlock<'_>) -> Option<usize> {
    match block {
        QuantizedBlock::Float32(values) => values.len().checked_mul(size_of::<f32>()),
        QuantizedBlock::Int32(values) => values.len().checked_mul(size_of::<i32>()),
        QuantizedBlock::Packed { .. } | QuantizedBlock::PackedStrided { .. } => block.packed_len(),
    }
}

fn own_block(block: QuantizedBlock<'_>) -> OwnedBlock {
    match block {
        QuantizedBlock::Float32(values) => OwnedBlock::Float32(Arc::from(values)),
        QuantizedBlock::Int32(values) => OwnedBlock::Int32(Arc::from(values)),
        QuantizedBlock::Packed { codec, bytes } => OwnedBlock::Packed {
            codec: Arc::new(codec),
            bytes: Arc::from(bytes),
        },
        QuantizedBlock::PackedStrided { codec, .. } => OwnedBlock::Packed {
            codec: Arc::new(codec),
            bytes: block
                .packed_runs()
                .into_iter()
                .flatten()
                .flatten()
                .copied()
                .collect(),
        },
    }
}

/// Owned tensor payload with its source-neutral packed codec identity.
#[derive(Clone)]
pub enum OwnedBlock {
    /// Owned floating point values.
    Float32(Arc<[f32]>),
    /// Owned integer values.
    Int32(Arc<[i32]>),
    /// Owned packed bytes and their codec identity.
    Packed {
        /// shared so the 688-byte codec value does not size every variant
        codec: Arc<Codec>,
        bytes: Arc<[u8]>,
    },
}

/// An owned block name and its payload.
#[derive(Clone)]
pub struct OwnedNamedBlock {
    /// Name used by Omega's named-input resolver.
    pub name: Arc<str>,
    /// Tensor data copied from the caller's borrowed input.
    pub block: OwnedBlock,
}

/// Owned input set whose single lease covers every copied name and payload.
#[derive(Clone)]
pub struct OwnedGpuInputs {
    /// All named inputs in their original order.
    pub blocks: Arc<[OwnedNamedBlock]>,
    /// Exact number of source bytes copied into names and tensor data.
    pub owned_bytes: usize,
    _lease: Arc<CreditLease>,
    _resident_lease: Option<Arc<CreditLease>>,
    resident_names: Arc<[Arc<str>]>,
}

impl OwnedGpuInputs {
    /// Reuses immutable resident payloads and preserves the caller's named-block order.
    pub fn with_resident_blocks(
        source: &[(&str, QuantizedBlock<'_>)],
        resident_names: &BTreeSet<&str>,
        resident: &Self,
        dynamic: Self,
    ) -> Result<Self, OwnedInputError> {
        let owned_bytes = resident
            .owned_bytes
            .checked_add(dynamic.owned_bytes)
            .ok_or(OwnedInputError::SizeOverflow)?;
        let mut resident_blocks = resident.blocks.iter();
        let mut dynamic_blocks = dynamic.blocks.iter();
        let blocks = source
            .iter()
            .map(|(name, _)| {
                let block = if resident_names.contains(name) {
                    resident_blocks.next().cloned()
                } else {
                    dynamic_blocks.next().cloned()
                };
                block.filter(|block| block.name.as_ref() == *name)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(OwnedInputError::InputPartitionMismatch)?;
        Ok(Self {
            blocks: Arc::from(blocks),
            owned_bytes,
            _lease: dynamic._lease,
            _resident_lease: Some(Arc::clone(&resident._lease)),
            resident_names: Arc::from(
                resident
                    .blocks
                    .iter()
                    .map(|block| Arc::clone(&block.name))
                    .collect::<Vec<_>>(),
            ),
        })
    }
}

/// Failure to account the source sizes for an owned input set.
#[derive(Debug, thiserror::Error)]
pub enum OwnedInputError {
    /// The total byte size cannot be represented by `usize`.
    #[error("owned input byte size overflow")]
    SizeOverflow,
    /// The resident and dynamic subsets did not match the source names.
    #[error("owned input partition does not match source blocks")]
    InputPartitionMismatch,
    /// Byte admission failed before any input copy was made.
    #[error(transparent)]
    Budget(#[from] CreditAcquireError),
}

/// Owned host bytes with their byte reservation attached.
pub struct OwnedByteBlock {
    bytes: Arc<[u8]>,
    _lease: CreditLease,
}

impl OwnedByteBlock {
    /// Borrow the copied bytes while retaining their budget reservation.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Number of bytes owned and reserved by this block.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the owned block contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Owns the dedicated worker pool until its shutdown future has completed.
pub struct CudaGpuService<Owner> {
    pool: Option<ProximaBackgroundPool>,
    commands: Arc<CommandSender<OwnerCommand>>,
    job_gate: Arc<CreditGate<ReservationSink>>,
    byte_budget: Arc<OwnedByteBudget>,
    owner: Owner,
    #[cfg(test)]
    fake_resident_sources: Arc<Mutex<BTreeMap<NodeId, usize>>>,
    #[cfg(test)]
    fake_plan_inputs: Arc<Mutex<Vec<FakePlanInputs>>>,
    #[cfg(test)]
    fake_execute_count: Arc<AtomicUsize>,
}

#[cfg(test)]
pub(crate) struct FakeOwnerControl {
    pub(crate) accepted: Receiver<()>,
    pub(crate) release: Sender<()>,
}

impl CudaGpuService<()> {
    /// Starts one dedicated owner worker sized by [`OwnerConfig`] and retains its typed completion.
    pub fn start(
        config: &OwnerConfig,
    ) -> Result<
        CudaGpuService<impl Future<Output = Result<(), ProximaError>> + Send + use<>>,
        ProximaError,
    > {
        Self::start_inner(
            config,
            #[cfg(test)]
            false,
            #[cfg(test)]
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn start_fake()
    -> Result<CudaGpuService<impl Future<Output = Result<(), ProximaError>> + Send>, ProximaError>
    {
        Self::start_inner(&OwnerConfig::default(), true, None)
    }

    #[cfg(test)]
    pub(crate) fn start_fake_held() -> Result<
        (
            CudaGpuService<impl Future<Output = Result<(), ProximaError>> + Send>,
            FakeOwnerControl,
        ),
        ProximaError,
    > {
        let (accepted_sender, accepted) = channel();
        let (release, release_receiver) = channel();
        let service = Self::start_inner(
            &OwnerConfig::default(),
            true,
            Some((accepted_sender, release_receiver)),
        )?;
        Ok((service, FakeOwnerControl { accepted, release }))
    }

    fn start_inner(
        config: &OwnerConfig,
        #[cfg(test)] fake: bool,
        #[cfg(test)] fake_owner_hold: Option<(Sender<()>, Receiver<()>)>,
    ) -> Result<
        CudaGpuService<impl Future<Output = Result<(), ProximaError>> + Send + use<>>,
        ProximaError,
    > {
        let job_gate = Arc::new(
            CreditGate::new(ReservationSink, config.queue_depth)
                .map_err(|_| ProximaError::Config(String::from("invalid cuda job capacity")))?,
        );
        let byte_budget =
            Arc::new(OwnedByteBudget::new(config.owned_host_bytes).map_err(|_| {
                ProximaError::Config(String::from("invalid cuda owned byte capacity"))
            })?);
        let pool = ProximaBackgroundPool::with_threads(1)?;
        let (commands, owner_commands) = command_channel(config.queue_depth);
        let plan_capacity = config.plan_capacity;
        let commands = Arc::new(commands);
        let owner_job_gate = Arc::clone(&job_gate);
        let owner_byte_budget = Arc::clone(&byte_budget);
        let identity = Arc::new(());
        let owner_identity = Arc::clone(&identity);
        #[cfg(test)]
        let fake_resident_sources = Arc::new(Mutex::new(BTreeMap::new()));
        #[cfg(test)]
        let owner_fake_resident_sources = Arc::clone(&fake_resident_sources);
        #[cfg(test)]
        let fake_plan_inputs = Arc::new(Mutex::new(Vec::new()));
        #[cfg(test)]
        let owner_fake_plan_inputs = Arc::clone(&fake_plan_inputs);
        #[cfg(test)]
        let fake_execute_count = Arc::new(AtomicUsize::new(0));
        #[cfg(test)]
        let owner_fake_execute_count = Arc::clone(&fake_execute_count);
        let owner = pool.spawn_async(async move {
            let state = OwnerState {
                plans: PlanRegistry::new(owner_identity, plan_capacity),
                driver: None,
                completed_jobs: 0,
                #[cfg(test)]
                fake_owner_hold,
            };
            let handle = move |state: &mut OwnerState,
                               owner_commands: &mut CommandReceiver<OwnerCommand>,
                               command: OwnerCommand| {
                let OwnerState {
                    plans,
                    driver,
                    completed_jobs,
                    #[cfg(test)]
                    fake_owner_hold,
                } = state;
                let mut flow = OwnerFlow::Continue;
                match command {
                    OwnerCommand::DeviceSnapshot { reply } => {
                        let result =
                            match driver.take().map_or_else(|| omega::CudaDriver::new(0), Ok) {
                                Ok(created) => {
                                    let snapshot = created.device_snapshot();
                                    *driver = Some(created);
                                    snapshot
                                }
                                Err(error) => Err(error),
                            };
                        let _ = reply.send(result);
                    }
                    OwnerCommand::RegisterGraph { graph, reply } => {
                        let result = (|graph: OwnedGraph| {
                            #[cfg(test)]
                            if fake {
                                owner_fake_plan_inputs.lock().push(FakePlanInputs {
                                    numeric_policy: graph.numeric_policy,
                                    exact_activations: graph.exact_activations,
                                    omega_runtime: graph.omega_runtime.clone(),
                                });
                                return plans
                                    .register(RegisteredPlan::Fake {
                                        program: graph.program,
                                        outputs: graph.outputs,
                                        symbols: graph.symbols,
                                    })
                                    .map_err(CudaPlanRegistrationError::from);
                            }
                            let named = graph
                                .inputs
                                .blocks
                                .iter()
                                .map(|input| {
                                    (input.name.as_ref(), borrow_owned_block(&input.block))
                                })
                                .collect::<Vec<_>>();
                            if let Some(config) = graph.omega_runtime.as_ref().filter(|config| selects_hip(config)) {
                                let plan = omega::backend::hip::shared::plan(
                                    config,
                                    omega::backend::hip::shared::hip_wave_of(config),
                                    &graph.program,
                                    &graph.symbols,
                                    &named,
                                    &graph.outputs,
                                )?;
                                return plans
                                    .register(RegisteredPlan::Rocm(Box::new(plan)))
                                    .map_err(CudaPlanRegistrationError::from);
                            }
                            if driver.is_none() {
                                *driver = Some(omega::CudaDriver::new(0)?);
                            }
                            let Some(driver) = driver.as_ref() else {
                                unreachable!("cuda driver was created above")
                            };
                            let plan = match &graph.omega_runtime {
                                Some(config) => driver.plan_with_config(
                                    &graph.program,
                                    &graph.symbols,
                                    &graph.outputs,
                                    &named,
                                    config,
                                )?,
                                None => driver.plan_with_activation_policy(
                                    &graph.program,
                                    &graph.symbols,
                                    &graph.outputs,
                                    graph.numeric_policy,
                                    graph.exact_activations,
                                    &named,
                                )?,
                            };
                            plans
                                .register(RegisteredPlan::Cuda(Box::new(plan)))
                                .map_err(CudaPlanRegistrationError::from)
                        })(*graph);
                        if let Err(Ok(lease)) = reply.send(result) {
                            let _ = plans.retire(&lease);
                        }
                    }
                    OwnerCommand::RegisterPlan { plan, reply } => {
                        let result = plans.register(RegisteredPlan::Cuda(plan));
                        if let Err(Ok(lease)) = reply.send(result) {
                            let _ = plans.retire(&lease);
                        }
                    }
                    OwnerCommand::RetirePlan { lease, reply } => {
                        let _ = reply.send(plans.retire(&lease));
                    }
                    OwnerCommand::InvalidateResidents { nodes, reply } => {
                        #[cfg(test)]
                        if fake {
                            let mut sources = owner_fake_resident_sources.lock();
                            for node in &nodes {
                                sources.remove(node);
                            }
                        }
                        let result = match driver.as_ref() {
                            Some(driver) => driver.invalidate_resident_buffers(&nodes),
                            None => Ok(()),
                        };
                        let _ = reply.send(result);
                    }
                    OwnerCommand::ExecutePlan { job, reply } => {
                        #[cfg(test)]
                        if fake {
                            owner_fake_execute_count.fetch_add(1, Ordering::Relaxed);
                            if let Some((accepted_sender, release_receiver)) =
                                fake_owner_hold.take()
                            {
                                let _ = accepted_sender.send(());
                                let _ = release_receiver.recv();
                            }
                        }
                        let AdmittedGpuJob {
                            job: OwnedGpuJob { lease, inputs },
                            credit,
                        } = job;
                        let result = execute_registered_plan(plans, &lease, |plan| {
                            execute_cuda_plan(plan, &inputs)
                        })
                        .map_err(CudaExecutionError::Plan)
                        .and_then(|result| result.map_err(CudaExecutionError::Driver));
                        let execution_failed =
                            matches!(&result, Err(CudaExecutionError::Driver(_)));
                        drop(inputs);
                        let completion_sequence = *completed_jobs;
                        *completed_jobs += 1;
                        if execution_failed {
                            owner_job_gate.close();
                            owner_byte_budget.gate.close();
                            owner_commands.close();
                        }
                        let _ = reply.send(CompletedGpuJob {
                            result,
                            completion_sequence,
                            _credit: credit,
                        });
                        if execution_failed {
                            flow = OwnerFlow::Drain;
                        }
                    }
                }
                flow
            };
            let reject = |state: &mut OwnerState, command: OwnerCommand| {
                reject_after_execution_failure(command, state.completed_jobs);
            };
            run_owner(owner_commands, state, handle, reject).await;
            Ok(())
        });

        Ok(CudaGpuService {
            pool: Some(pool),
            commands,
            job_gate,
            byte_budget,
            owner,
            #[cfg(test)]
            fake_resident_sources,
            #[cfg(test)]
            fake_plan_inputs,
            #[cfg(test)]
            fake_execute_count,
        })
    }
}

impl<Owner> CudaGpuService<Owner>
where
    Owner: Future<Output = Result<(), ProximaError>> + Send,
{
    /// Reads the selected device and graph activity on the service owner.
    pub async fn device_snapshot(
        &self,
    ) -> Result<omega::CudaDeviceSnapshot, CudaPlanRegistrationError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(OwnerCommand::DeviceSnapshot { reply })
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?;
        response
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?
            .map_err(CudaPlanRegistrationError::from)
    }

    /// Reserve and copy borrowed named inputs before handing them to the worker.
    pub async fn copy_named_blocks(
        &self,
        source: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<OwnedGpuInputs, OwnedInputError> {
        self.byte_budget.copy_named_blocks(source).await
    }

    /// Remaining credits for copied named input bytes.
    #[must_use]
    pub fn available_owned_host_bytes(&self) -> usize {
        self.byte_budget.available_bytes()
    }

    /// Build and register a graph wholly on the service owner.
    pub async fn register_graph(
        &self,
        program: &[Op],
        symbols: &[u64],
        outputs: &[NodeId],
        options: CudaPlanOptions<'_>,
        inputs: &OwnedGpuInputs,
    ) -> Result<PlanLease, CudaPlanRegistrationError> {
        let CudaPlanOptions {
            numeric_policy,
            exact_activations,
            omega_runtime,
        } = options;
        validate_cuda_plan_options(numeric_policy, exact_activations, omega_runtime)?;
        let (reply, response) = oneshot::channel();
        let graph = OwnedGraph {
            program: program.to_vec(),
            symbols: symbols.to_vec(),
            outputs: outputs.to_vec(),
            numeric_policy,
            exact_activations,
            omega_runtime: omega_runtime.cloned(),
            inputs: inputs.clone(),
        };
        self.commands
            .send(OwnerCommand::RegisterGraph {
                graph: Box::new(graph),
                reply,
            })
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?;
        response
            .await
            .unwrap_or(Err(PlanRegistryError::OwnerStopped.into()))
    }

    #[cfg(test)]
    pub(crate) fn fake_plan_inputs(&self) -> Vec<FakePlanInputs> {
        self.fake_plan_inputs.lock().clone()
    }

    #[cfg(test)]
    pub(crate) fn fake_execute_count(&self) -> usize {
        self.fake_execute_count.load(Ordering::Relaxed)
    }

    /// Clears changed resident device copies on the service owner.
    pub async fn invalidate_resident_sources(
        &self,
        program: &[Op],
        names: &BTreeSet<&str>,
    ) -> Result<(), CudaPlanRegistrationError> {
        let nodes = program
            .iter()
            .enumerate()
            .filter_map(|(index, op)| match op {
                Op::Input {
                    name: Some(name), ..
                } if names.contains(name.as_str()) => Some(
                    u32::try_from(index)
                        .map(NodeId)
                        .map_err(|_| PlanRegistryError::ProgramTooLarge),
                ),
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (reply, response) = oneshot::channel();
        self.commands
            .send(OwnerCommand::InvalidateResidents { nodes, reply })
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?;
        response
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?
            .map_err(CudaPlanRegistrationError::from)
    }

    /// Retire a plan generation after previously queued owner work reaches it.
    ///
    /// Cancelling before the command is queued leaves the plan resident; after
    /// it is queued the owner retires it regardless, and a repeat call reports
    /// `Stale`.
    pub async fn retire_cuda_plan(&self, lease: &PlanLease) -> Result<(), PlanRegistryError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(OwnerCommand::RetirePlan {
                lease: lease.clone(),
                reply,
            })
            .await
            .map_err(|_| PlanRegistryError::OwnerStopped)?;
        response
            .await
            .unwrap_or(Err(PlanRegistryError::OwnerStopped))
    }

    /// Await bounded owner execution through the composed admission and execution Pipes.
    pub async fn execute_cuda(
        &self,
        lease: &PlanLease,
        inputs: OwnedGpuInputs,
    ) -> Result<proxima_tensor::cpu::Evaluated, GpuPipeError> {
        let pipeline = GpuCapacityGate {
            gate: &self.job_gate,
        }
        .and_then(CudaExecutionPipe {
            commands: Arc::clone(&self.commands),
        });
        pipeline
            .call(OwnedGpuJob {
                lease: lease.clone(),
                inputs,
            })
            .await
    }

    /// Await owner execution and return the owner's completion sequence.
    pub async fn execute_cuda_recorded(
        &self,
        lease: &PlanLease,
        inputs: OwnedGpuInputs,
    ) -> Result<CudaExecutionReceipt, GpuPipeError> {
        let pipeline = GpuCapacityGate {
            gate: &self.job_gate,
        }
        .and_then(CudaExecutionReceiptPipe {
            commands: Arc::clone(&self.commands),
        });
        pipeline
            .call(OwnedGpuJob {
                lease: lease.clone(),
                inputs,
            })
            .await
    }

    /// Closes both admission gates and wakes callers waiting for copied bytes or job credit.
    pub fn close_admission(&self) {
        self.job_gate.close();
        self.byte_budget.gate.close();
    }

    /// Closes the owner loop, awaits its typed result, and then joins its pool.
    pub async fn shutdown(self) -> Result<(), ProximaError> {
        self.close_admission();
        self.commands.close();
        let result = self.owner.await;
        drop(self.pool);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdmittedGpuJob, CommandSender, CudaExecutionError, CudaGpuService,
        GpuAdmissionError, GpuCapacityGate, GpuPipeError, OwnedBlock,
        OwnedByteBudget, OwnedGpuJob, ReservationSink, command_channel,
    };
    use omega::device_owner::{PlanRegistry, execute_registered_plan};
    use omega::owner_config::OwnerConfig;
    use core::future::{Future, poll_fn};
    use core::pin::pin;
    use core::task::Poll;
    use prime::os::background::ProximaBackgroundPool;
    use proxima_numeric::codec::Codec;
    use proxima_primitives::pipe::{CreditAcquireError, CreditGate, Pipe, PipeExt};
    use proxima_primitives::sync::oneshot::Sender;
    use proxima_tensor::cpu::QuantizedBlock;
    use proxima_tensor::{Extent, NodeId, Op};
    use proxima_numeric::{DType};
    use std::collections::BTreeSet;
    use std::sync::Arc;

    #[proxima::test]
    async fn test_omega_async_serving_owner_invalidates_only_named_resident_node() {
        let service = CudaGpuService::start_fake().expect("fake owner should start");
        let program = [
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Symbolic(0)],
                name: Some(String::from("changed")),
            },
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Symbolic(0)],
                name: Some(String::from("retained")),
            },
        ];
        {
            let mut sources = service.fake_resident_sources.lock();
            sources.insert(NodeId(0), 11);
            sources.insert(NodeId(1), 22);
        }
        service
            .invalidate_resident_sources(&program, &BTreeSet::from(["changed"]))
            .await
            .expect("owner should process invalidation");
        assert_eq!(
            *service.fake_resident_sources.lock(),
            [(NodeId(1), 22)].into(),
        );
        service.shutdown().await.expect("owner should drain");
    }

    #[proxima::test]
    async fn test_omega_async_owned_strided_block_equals_its_contiguous_runs() {
        let experts = 3usize;
        let run = 144usize;
        let source: Vec<u8> = (0..experts * 2 * run).map(|index| (index % 251) as u8).collect();
        let span = &source[run..run + (experts - 1) * 2 * run + run];
        let strided = QuantizedBlock::PackedStrided {
            codec: Codec::Q4K,
            bytes: span,
            expert_bytes: run,
            stride_bytes: 2 * run,
        };
        let byte_budget = OwnedByteBudget::new(1 << 20).expect("valid test byte budget");

        let owned = byte_budget
            .copy_named_blocks(&[("up_exps", strided)])
            .await
            .expect("strided block should be admitted");

        let expected: Vec<u8> = (0..experts)
            .flat_map(|expert| source[(expert * 2 + 1) * run..(expert * 2 + 2) * run].to_vec())
            .collect();
        let OwnedBlock::Packed { codec, bytes } = &owned.blocks[0].block else {
            panic!("expected an owned packed block");
        };
        assert_eq!(**codec, Codec::Q4K);
        assert_eq!(bytes.as_ref(), expected.as_slice());
        assert_eq!(owned.owned_bytes, "up_exps".len() + experts * run);
    }

    type CaptureQueue = CommandSender<(AdmittedGpuJob, Sender<Result<usize, GpuPipeError>>)>;

    struct CaptureExecution {
        queue: Arc<CaptureQueue>,
    }

    impl Pipe for CaptureExecution {
        type In = AdmittedGpuJob;
        type Out = usize;
        type Err = GpuPipeError;

        fn call(&self, job: AdmittedGpuJob) -> impl Future<Output = Result<usize, GpuPipeError>> {
            let queue = Arc::clone(&self.queue);
            async move {
                let (reply, response) = super::oneshot::channel();
                match queue.send((job, reply)).await {
                    Ok(()) => response.await.unwrap_or(Err(GpuPipeError::Execution(
                        CudaExecutionError::OwnerStopped,
                    ))),
                    Err(error) => {
                        let (job, _) = error.into_inner();
                        Err(GpuPipeError::Execution(
                            CudaExecutionError::ServiceClosedWithInput(job.job),
                        ))
                    }
                }
            }
        }
    }

    async fn test_owned_gpu_job() -> OwnedGpuJob {
        let byte_budget = OwnedByteBudget::new(1).expect("valid test byte budget");
        let inputs = byte_budget
            .copy_named_blocks(&[])
            .await
            .expect("empty job should be admitted");
        OwnedGpuJob {
            lease: PlanRegistry::new(Arc::new(()), 1)
                .register(())
                .expect("empty registry should accept a plan"),
            inputs,
        }
    }

    #[proxima::test]
    async fn test_omega_async_pipe_response_waits_for_job_capacity() {
        let gate = CreditGate::new(ReservationSink, 1).expect("valid job capacity");
        let held = gate.acquire(1).await.expect("hold job capacity");
        let (queue, mut receiver) = command_channel(1);
        let queue = Arc::new(queue);
        let pipeline = GpuCapacityGate { gate: &gate }.and_then(CaptureExecution {
            queue: Arc::clone(&queue),
        });
        let mut call = pin!(pipeline.call(test_owned_gpu_job().await));

        let pending = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        drop(held);
        let pending = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        assert_eq!(gate.available(), 0);

        let (job, reply) = receiver
            .recv()
            .await
            .expect("admitted job should reach execution");
        drop(reply);
        drop(job);
        assert_eq!(gate.available(), 1);

        let closed_gate = CreditGate::new(ReservationSink, 1).expect("valid job capacity");
        let held = closed_gate.acquire(1).await.expect("hold job capacity");
        let (closed_queue, _closed_receiver) = command_channel(1);
        let closed_pipeline = GpuCapacityGate { gate: &closed_gate }.and_then(CaptureExecution {
            queue: Arc::new(closed_queue),
        });
        let mut waiting = pin!(closed_pipeline.call(test_owned_gpu_job().await));
        let pending = poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        closed_gate.close();
        let rejected = poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await;
        let rejected_job = match rejected {
            Poll::Ready(Err(GpuPipeError::Admission(GpuAdmissionError::Closed(job)))) => job,
            _ => panic!("closed admission should return the owned job"),
        };
        let (_, inputs) = rejected_job.into_parts();
        assert_eq!(inputs.owned_bytes, 0);
        drop(held);
    }

    #[proxima::test]
    async fn test_omega_async_pipe_response_awaits_result_and_wake_orders() {
        let gate = CreditGate::new(ReservationSink, 1).expect("valid job capacity");
        let (queue, mut receiver) = command_channel(1);
        let queue = Arc::new(queue);
        let pipeline = GpuCapacityGate { gate: &gate }.and_then(CaptureExecution {
            queue: Arc::clone(&queue),
        });
        let mut call = pin!(pipeline.call(test_owned_gpu_job().await));

        let pending = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        let (job, reply) = receiver
            .recv()
            .await
            .expect("job should be queued before response wait");
        assert_eq!(gate.available(), 0);
        reply
            .send(Ok(101))
            .expect("response receiver should still wait");
        let result = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(result, Poll::Ready(Ok(101))));
        assert_eq!(gate.available(), 0);
        drop(job);
        assert_eq!(gate.available(), 1);

        let (reply, response) = super::oneshot::channel();
        reply
            .send(Ok::<_, GpuPipeError>(107))
            .expect("receiver may start late");
        assert!(matches!(response.await, Ok(Ok(107))));
    }

    #[proxima::test]
    async fn test_omega_async_pipe_response_retains_admitted_job_after_caller_drop() {
        let gate = CreditGate::new(ReservationSink, 1).expect("valid job capacity");
        let (queue, mut receiver) = command_channel(1);
        let queue = Arc::new(queue);
        let pipeline = GpuCapacityGate { gate: &gate }.and_then(CaptureExecution {
            queue: Arc::clone(&queue),
        });
        let mut call = pin!(pipeline.call(test_owned_gpu_job().await));

        let pending = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        assert_eq!(gate.available(), 0);

        let (job, reply) = receiver
            .recv()
            .await
            .expect("accepted job must outlive caller future");
        assert_eq!(gate.available(), 0);
        drop(reply);
        drop(job);
        assert_eq!(gate.available(), 1);
    }

    #[proxima::test]
    async fn test_omega_async_pipe_response_returns_job_when_queue_is_full() {
        let gate = CreditGate::new(ReservationSink, 1).expect("valid job capacity");
        let (queue, mut receiver) = command_channel(1);
        let queue = Arc::new(queue);
        let (held_reply, _held_response) = super::oneshot::channel();
        let held_gate = CreditGate::new(ReservationSink, 1).expect("valid held capacity");
        let held_job = AdmittedGpuJob {
            job: test_owned_gpu_job().await,
            credit: held_gate
                .acquire(1)
                .await
                .expect("hold channel slot credit"),
        };
        queue
            .try_send((held_job, held_reply))
            .expect("first command fills the channel");
        let pipeline = GpuCapacityGate { gate: &gate }.and_then(CaptureExecution {
            queue: Arc::clone(&queue),
        });
        let mut call = pin!(pipeline.call(test_owned_gpu_job().await));
        assert!(matches!(
            poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await,
            Poll::Pending
        ));

        let (held_job, held_reply) = receiver
            .recv()
            .await
            .expect("accepted command remains queued");
        drop(held_reply);
        drop(held_job);
        assert!(matches!(
            poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await,
            Poll::Pending
        ));
        let (accepted_job, reply) = receiver.recv().await.expect("waiting send should wake");
        reply
            .send(Ok(109))
            .expect("waiting caller receives completion");
        let result = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
        assert!(matches!(result, Poll::Ready(Ok(109))));
        drop(accepted_job);
        assert_eq!(gate.available(), 1);
    }

    #[proxima::test]
    async fn test_omega_async_owner_queue_is_fifo_and_drains_on_close() {
        let (queue, mut receiver) = command_channel(2);
        queue.try_send(11).expect("first value should fit");
        queue.try_send(22).expect("second value should fit");
        assert!(queue.close());
        assert_eq!(receiver.recv().await, Some(11));
        assert_eq!(receiver.recv().await, Some(22));
        assert_eq!(receiver.recv().await, None);
    }

    #[proxima::test]
    async fn test_omega_async_owner_queue_waits_for_capacity() {
        let (queue, mut receiver) = command_channel(1);
        queue.try_send(11).expect("first value should fit");
        let mut waiting = pin!(queue.send(22));
        assert!(matches!(
            poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await,
            Poll::Pending
        ));
        assert_eq!(receiver.recv().await, Some(11));
        assert!(matches!(
            poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await,
            Poll::Ready(Ok(()))
        ));
        assert_eq!(receiver.recv().await, Some(22));
    }

    #[proxima::test]
    async fn test_omega_async_owner_queue_returns_command_after_close() {
        let (queue, mut receiver) = command_channel(2);
        queue.try_send(11).expect("value should fit before close");
        assert!(queue.close());
        match queue.send(22).await {
            Err(error) => assert_eq!(error.into_inner(), 22),
            Ok(()) => panic!("closed channel should return later commands"),
        }
        assert_eq!(receiver.recv().await, Some(11));
        assert_eq!(receiver.recv().await, None);
    }

    #[proxima::test]
    async fn test_omega_async_byte_credit_waits_for_exact_capacity() {
        let budget = OwnedByteBudget::new(4).expect("valid byte budget");
        let first = budget.copy(&[1, 2, 3]).await.expect("first copy");
        assert_eq!(first.len(), 3);
        assert_eq!(budget.available_bytes(), 1);

        let mut waiting = pin!(budget.copy(&[4, 5]));
        let pending = poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await;
        assert!(matches!(pending, Poll::Pending));
        drop(first);

        let second = match poll_fn(|context| Poll::Ready(waiting.as_mut().poll(context))).await {
            Poll::Ready(Ok(block)) => block,
            _ => panic!("released byte capacity should admit the waiting copy"),
        };
        assert_eq!(second.as_slice(), &[4, 5]);
        assert_eq!(budget.available_bytes(), 2);
        drop(second);
        assert_eq!(budget.available_bytes(), 4);
    }

    #[proxima::test]
    async fn test_omega_async_byte_credit_rejects_over_budget_before_copy() {
        let budget = OwnedByteBudget::new(2).expect("valid byte budget");
        assert!(matches!(
            budget.copy(&[1, 2, 3]).await,
            Err(CreditAcquireError::OverCapacity)
        ));
        assert_eq!(budget.available_bytes(), 2);
    }

    #[proxima::test]
    async fn test_omega_async_byte_credit_lives_through_queue_retirement() {
        let budget = OwnedByteBudget::new(3).expect("valid byte budget");
        let block = budget.copy(&[1, 2, 3]).await.expect("owned copy");
        let (queue, mut receiver) = command_channel(2);
        assert!(queue.try_send(block).is_ok());
        assert_eq!(budget.available_bytes(), 0);

        let retired = receiver.recv().await.expect("queued block");
        assert_eq!(retired.as_slice(), &[1, 2, 3]);
        assert_eq!(budget.available_bytes(), 0);
        drop(retired);
        assert_eq!(budget.available_bytes(), 3);
    }

    #[proxima::test]
    async fn test_omega_async_payload_owns_f32_i32_and_packed_forms() {
        let budget = OwnedByteBudget::new(64).expect("valid byte budget");
        let float_values = [1.25, 2.5];
        let integer_values = [7, -4];
        let packed_values = [0xaa, 0x55, 0x00];
        let source = [
            ("float", QuantizedBlock::Float32(&float_values)),
            ("int", QuantizedBlock::Int32(&integer_values)),
            (
                "packed",
                QuantizedBlock::Packed {
                    codec: Codec::Q4K,
                    bytes: &packed_values,
                },
            ),
        ];

        let owned = budget
            .copy_named_blocks(&source)
            .await
            .expect("all blocks should be admitted");
        assert!(
            matches!(&owned.blocks[0].block, OwnedBlock::Float32(values) if values.as_ref() == float_values)
        );
        assert!(
            matches!(&owned.blocks[1].block, OwnedBlock::Int32(values) if values.as_ref() == integer_values)
        );
        assert!(matches!(
            &owned.blocks[2].block,
            OwnedBlock::Packed { codec, bytes } if **codec == Codec::Q4K && bytes.as_ref() == packed_values
        ));
    }

    #[proxima::test]
    async fn test_omega_async_payload_outlives_borrowed_source() {
        let budget = OwnedByteBudget::new(64).expect("valid byte budget");
        let owned = {
            let source_values = vec![3.5_f32, 8.25];
            let source = [("activation", QuantizedBlock::Float32(&source_values))];
            budget
                .copy_named_blocks(&source)
                .await
                .expect("borrowed source should be copied")
        };

        assert!(matches!(
            &owned.blocks[0].block,
            OwnedBlock::Float32(values) if values.as_ref() == [3.5, 8.25]
        ));
    }

    #[proxima::test]
    async fn test_omega_async_payload_accounts_each_copied_byte() {
        let budget = OwnedByteBudget::new(32).expect("valid byte budget");
        let values = [5_i32, 9];
        let packed = [0x12, 0x34, 0x56];
        let source = [
            ("ids", QuantizedBlock::Int32(&values)),
            (
                "weights",
                QuantizedBlock::Packed {
                    codec: Codec::Q5K,
                    bytes: &packed,
                },
            ),
        ];

        let owned = budget
            .copy_named_blocks(&source)
            .await
            .expect("input bytes should fit the budget");
        assert_eq!(owned.owned_bytes, 8 + 3 + 3 + 7);
        assert_eq!(budget.available_bytes(), 32 - owned.owned_bytes);
        drop(owned);
        assert_eq!(budget.available_bytes(), 32);
    }

    #[proxima::test]
    async fn test_omega_async_cuda_stage_runs_executor_on_worker_thread() {
        let service = Arc::new(());
        let mut registry = PlanRegistry::new(service, 1);
        let lease = registry.register(11_u64).expect("plan should register");
        let pool = ProximaBackgroundPool::with_threads(1).expect("worker pool should start");
        let worker = pool.spawn(move || {
            let worker_id = std::thread::current().id();
            let result = execute_registered_plan(&mut registry, &lease, |plan| {
                assert_eq!(std::thread::current().id(), worker_id);
                *plan += 1;
                Ok::<_, &'static str>(*plan)
            });
            Ok(result
                .expect("registry should resolve")
                .expect("execution should pass"))
        });

        assert_eq!(worker.await.expect("worker result should arrive"), 12);
        drop(pool);
    }

    #[proxima::test]
    async fn test_omega_async_owner_lifecycle_starts_one_worker() {
        let service =
            CudaGpuService::start(&OwnerConfig::default()).expect("service should start");
        assert_eq!(service.pool.as_ref().map(|pool| pool.workers()), Some(1));
        service
            .shutdown()
            .await
            .expect("owner should report successful shutdown");
    }

    #[proxima::test]
    async fn test_omega_async_owner_lifecycle_closes_and_joins() {
        let service =
            CudaGpuService::start(&OwnerConfig::default()).expect("service should start");
        service
            .shutdown()
            .await
            .expect("shutdown should await typed owner completion");
    }
}
