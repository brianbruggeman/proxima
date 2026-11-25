//! The single-owner GPU device registry and its command vocabulary.
//!
//! One task owns the device and every plan registered on it; callers talk to it
//! through a bounded [`proxima_primitives::sync::mpsc`] queue of
//! [`OwnerCommand`]s and receive replies on [`proxima_primitives::sync::oneshot`]
//! channels. The registry is generic over the plan type the owner drives.

use std::collections::BTreeSet;
use std::future::Future;
use std::ops::Range;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle, Thread};
use std::vec::Vec;

use proxima_core::ring::Ring;
use proxima_numeric::NumericPolicy;
use proxima_primitives::sync::mpsc::{Receiver, Sender, channel};
use proxima_primitives::sync::{Notify, oneshot};
use proxima_tensor::{Evaluated, NodeId, Op, QuantizedBlock};

use crate::config::RuntimeConfig;
use crate::cuda_driver::{CudaDriver, DevicePlan};
use crate::gpu_runtime::GpuRuntime;
use crate::owner_config::OwnerConfig;
use crate::{CudaDeviceSnapshot, CudaDriverError};

/// Service-scoped identity for a registered plan generation.
#[derive(Clone, Debug)]
pub struct PlanLease {
    service: Arc<()>,
    slotindex: usize,
    generation: u64,
}

impl PlanLease {
    /// Generation assigned by this service when the plan was registered.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanRegistryError {
    #[error("cuda plan registry is full")]
    Full,
    #[error("cuda plan lease belongs to another service")]
    WrongService,
    #[error("cuda plan lease is stale")]
    Stale,
    #[error("cuda plan generation has been retired")]
    Retired,
    #[error("cuda plan already has the maximum number of accepted jobs")]
    InFlightCountExhausted,
    #[error("cuda owner command queue is full")]
    QueueFull,
    #[error("cuda owner stopped before registering the plan")]
    OwnerStopped,
    #[error("cuda service closed after execution failure")]
    ServiceClosed,
    #[error("cuda graph has more input nodes than the node id can represent")]
    ProgramTooLarge,
}

struct PlanSlot<Plan> {
    generation: u64,
    plan: Option<Plan>,
    in_flight: usize,
    retired: bool,
}

/// Proof that one job was accepted against a plan generation.
pub struct PlanJobLease {
    plan: PlanLease,
}

pub struct PlanRegistry<Plan> {
    service: Arc<()>,
    capacity: usize,
    slots: Vec<PlanSlot<Plan>>,
    free: Vec<usize>,
}

impl<Plan> PlanRegistry<Plan> {
    #[must_use]
    pub fn new(service: Arc<()>, capacity: usize) -> Self {
        Self {
            service,
            capacity,
            slots: Vec::with_capacity(capacity),
            free: Vec::with_capacity(capacity),
        }
    }

    pub fn register(&mut self, plan: Plan) -> Result<PlanLease, PlanRegistryError> {
        let slotindex = match self.free.pop() {
            Some(slotindex) => slotindex,
            None if self.slots.len() < self.capacity => {
                let slotindex = self.slots.len();
                self.slots.push(PlanSlot {
                    generation: 0,
                    plan: None,
                    in_flight: 0,
                    retired: false,
                });
                slotindex
            }
            None => return Err(PlanRegistryError::Full),
        };
        let Some(slot) = self.slots.get_mut(slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        slot.plan = Some(plan);
        slot.in_flight = 0;
        slot.retired = false;
        Ok(PlanLease {
            service: Arc::clone(&self.service),
            slotindex,
            generation: slot.generation,
        })
    }

    pub fn begin_job(&mut self, lease: &PlanLease) -> Result<PlanJobLease, PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &lease.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(lease.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != lease.generation || slot.plan.is_none() {
            return Err(PlanRegistryError::Stale);
        }
        if slot.retired {
            return Err(PlanRegistryError::Retired);
        }
        let Some(in_flight) = slot.in_flight.checked_add(1) else {
            return Err(PlanRegistryError::InFlightCountExhausted);
        };
        slot.in_flight = in_flight;
        Ok(PlanJobLease {
            plan: lease.clone(),
        })
    }

    pub fn resolve_job_mut(&mut self, job: &PlanJobLease) -> Result<&mut Plan, PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &job.plan.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(job.plan.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != job.plan.generation || slot.in_flight == 0 {
            return Err(PlanRegistryError::Stale);
        }
        slot.plan.as_mut().ok_or(PlanRegistryError::Stale)
    }

    pub fn complete_job(&mut self, job: PlanJobLease) -> Result<(), PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &job.plan.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(job.plan.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != job.plan.generation || slot.in_flight == 0 {
            return Err(PlanRegistryError::Stale);
        }
        slot.in_flight -= 1;
        let release = slot.retired && slot.in_flight == 0;
        if release {
            self.release_slot(job.plan.slotindex);
        }
        Ok(())
    }

    pub fn retire(&mut self, lease: &PlanLease) -> Result<(), PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &lease.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(lease.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != lease.generation || slot.plan.is_none() {
            return Err(PlanRegistryError::Stale);
        }
        slot.retired = true;
        let release = slot.in_flight == 0;
        if release {
            self.release_slot(lease.slotindex);
        }
        Ok(())
    }

    fn release_slot(&mut self, slotindex: usize) {
        let Some(slot) = self.slots.get_mut(slotindex) else {
            return;
        };
        slot.plan = None;
        slot.retired = false;
        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.free.push(slotindex);
        }
    }
}

/// Runs `execute` against the plan behind `lease`, bracketed by begin and complete.
pub fn execute_registered_plan<Plan, Output, Error>(
    registry: &mut PlanRegistry<Plan>,
    lease: &PlanLease,
    execute: impl FnOnce(&mut Plan) -> Result<Output, Error>,
) -> Result<Result<Output, Error>, PlanRegistryError> {
    let job = registry.begin_job(lease)?;
    let execution = match registry.resolve_job_mut(&job) {
        Ok(plan) => execute(plan),
        Err(error) => {
            registry.complete_job(job)?;
            return Err(error);
        }
    };
    registry.complete_job(job)?;
    Ok(execution)
}

/// Every message the device owner accepts.
///
/// `Graph` is the owned graph a registration plans, `Plan` the device plan an
/// owner registers directly, `Job` an admitted execution, `Completed` its
/// reply, and `GraphError` the failure of planning a graph.
pub enum OwnerCommand<Graph, Plan, Job, Completed, GraphError> {
    DeviceSnapshot {
        reply: oneshot::Sender<Result<CudaDeviceSnapshot, CudaDriverError>>,
    },
    RegisterGraph {
        // boxed: the owned graph is ~300 bytes and would size every command
        graph: Box<Graph>,
        reply: oneshot::Sender<Result<PlanLease, GraphError>>,
    },
    RegisterPlan {
        // boxed: the plan is ~1.2 KB and would size every command; the owner boxes it into its registered plan anyway
        plan: Box<Plan>,
        reply: oneshot::Sender<Result<PlanLease, PlanRegistryError>>,
    },
    RetirePlan {
        lease: PlanLease,
        reply: oneshot::Sender<Result<(), PlanRegistryError>>,
    },
    InvalidateResidents {
        nodes: Vec<NodeId>,
        reply: oneshot::Sender<Result<(), CudaDriverError>>,
    },
    ExecutePlan {
        job: Job,
        reply: oneshot::Sender<Completed>,
    },
}

/// What the owner loop does after one command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerFlow {
    Continue,
    /// Stop serving: every command still queued is rejected, then the loop ends.
    Drain,
}

/// The owner loop: awaits each command in FIFO order and lets `handle` serve it.
///
/// It yields the task while the queue is empty, so the owner shares its worker
/// with whatever else that worker polls. `handle` may close the receiver and
/// return [`OwnerFlow::Drain`]; every command already queued then goes to
/// `reject`.
pub async fn run_owner<Command, State>(
    mut commands: Receiver<Command>,
    mut state: State,
    mut handle: impl FnMut(&mut State, &mut Receiver<Command>, Command) -> OwnerFlow,
    mut reject: impl FnMut(&mut State, Command),
) {
    while let Some(command) = commands.recv().await {
        if handle(&mut state, &mut commands, command) == OwnerFlow::Drain {
            while let Some(queued) = commands.recv().await {
                reject(&mut state, queued);
            }
            break;
        }
    }
}

/// Names one device an owner serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceKey(pub u32);

/// A weight already on the device: the registered block `slot`, its device `address` and `len` in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentRef {
    pub slot: u32,
    pub address: u64,
    pub len: usize,
}

/// One step's inputs: resident refs plus owned dynamic activations. The owner hands it back
/// in the reply so the caller refills the same vectors and a step grows no capacity.
#[derive(Debug)]
pub struct Frame<Block> {
    pub residents: Vec<ResidentRef>,
    pub dynamic: Vec<(Arc<str>, Block)>,
}

impl<Block> Default for Frame<Block> {
    fn default() -> Self {
        Self {
            residents: Vec::new(),
            dynamic: Vec::new(),
        }
    }
}

impl<Block> Frame<Block> {
    /// Empties both lists and keeps their capacity.
    pub fn clear(&mut self) {
        self.residents.clear();
        self.dynamic.clear();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceOwnerError {
    #[error("device driver failed: {0}")]
    Driver(#[from] CudaDriverError),
    #[error("device plan registry refused the command: {0}")]
    Registry(#[from] PlanRegistryError),
    /// The command was accepted, then the owner stopped or drained before answering it.
    #[error("device owner stopped")]
    Stopped,
    /// The owner is gone, so the command was never accepted.
    #[error("device owner closed")]
    Closed,
}

/// What one step produced: the outputs, or the resident slots the owner does not hold.
pub enum StepOutcome<Block> {
    Done {
        evaluated: Evaluated,
        frame: Frame<Block>,
        /// The owner's completion order for this step, counted from zero.
        sequence: u64,
    },
    NeedResident {
        slots: Vec<u32>,
        frame: Frame<Block>,
    },
}

pub type StepReply<Block> = Result<StepOutcome<Block>, DeviceOwnerError>;

/// One reusable reply slot: a ring and a [`Notify`] permit, so a reply is a push and a wake.
///
/// Composes [`proxima_core::ring::Ring`] and [`proxima_primitives::sync::notify::Notify`].
/// `Ring` rounds capacity up to 2 (its Vyukov protocol is degenerate at 1), so the slot holds
/// two; the protocol is one outstanding step per slot.
pub struct Completion<Value> {
    ring: Ring<Value>,
    ready: Notify,
}

impl<Value> Completion<Value> {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            ring: Ring::with_capacity(1),
            ready: Notify::new(),
        })
    }

    /// Publishes the reply and wakes the waiter; the owner side of the slot.
    pub fn complete(&self, value: Value) {
        if self.ring.push(value).is_ok() {
            self.ready.notify_one();
        }
    }

    /// Awaits the owner's reply.
    pub async fn wait(&self) -> Value {
        loop {
            if let Some(value) = self.ring.dequeue() {
                return value;
            }
            self.ready.notified().await;
        }
    }
}

/// Everything a registration plans from; `blocks` are owned, as interop registers today.
pub struct RegisterRequest<Block> {
    pub program: Vec<Op>,
    pub symbols: Vec<u64>,
    pub outputs: Vec<NodeId>,
    pub blocks: Vec<(Arc<str>, Block)>,
    pub resident_names: Vec<Arc<str>>,
    pub numeric_policy: NumericPolicy,
    pub exact_activations: bool,
    /// `Some` plans through `plan_with_config`; `None` through `plan_with_activation_policy`.
    pub config: Option<RuntimeConfig>,
}

/// Every message a [`DeviceOwner`] accepts; each carries its own typed reply.
pub enum Command<Block> {
    Register {
        // boxed: the request holds the program and config and would size every command; registration is setup
        request: Box<RegisterRequest<Block>>,
        reply: oneshot::Sender<Result<PlanLease, DeviceOwnerError>>,
    },
    Step {
        lease: PlanLease,
        frame: Frame<Block>,
        completion: Arc<Completion<StepReply<Block>>>,
    },
    Retire {
        lease: PlanLease,
        reply: oneshot::Sender<Result<(), DeviceOwnerError>>,
    },
    Snapshot {
        reply: oneshot::Sender<Result<CudaDeviceSnapshot, DeviceOwnerError>>,
    },
    ReleaseNames {
        names: Vec<Arc<str>>,
        reply: oneshot::Sender<()>,
    },
    ReleaseRanges {
        ranges: Vec<Range<usize>>,
        reply: oneshot::Sender<()>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

struct OwnedPlan<R: GpuRuntime, Block> {
    plan: DevicePlan<R>,
    blocks: Vec<(Arc<str>, Block)>,
}

/// The sans-IO owner of one device: a driver, the plans registered on it, and `serve`.
///
/// `Block` is the caller's owned payload; the executor reads it through
/// `QuantizedBlock: From<&Block>`. Composes [`CudaDriver`] and [`PlanRegistry`].
pub struct DeviceOwner<R: GpuRuntime, Block> {
    key: DeviceKey,
    driver: CudaDriver<R>,
    registry: PlanRegistry<OwnedPlan<R, Block>>,
    completed: u64,
}

impl<R: GpuRuntime, Block> DeviceOwner<R, Block>
where
    CudaDriverError: From<R::Error> + From<R::BuildError>,
    for<'block> QuantizedBlock<'block>: From<&'block Block>,
{
    #[must_use]
    pub fn new(key: DeviceKey, driver: CudaDriver<R>, plan_capacity: usize) -> Self {
        Self {
            key,
            driver,
            registry: PlanRegistry::new(Arc::new(()), plan_capacity),
            completed: 0,
        }
    }

    #[must_use]
    pub const fn key(&self) -> DeviceKey {
        self.key
    }

    pub fn serve(&mut self, command: Command<Block>) -> OwnerFlow {
        match command {
            Command::Register { request, reply } => {
                if let Err(Ok(lease)) = reply.send(self.register(*request)) {
                    let _ = self.registry.retire(&lease);
                }
            }
            Command::Step {
                lease,
                frame,
                completion,
            } => {
                let reply = self.step(&lease, frame);
                let flow = match &reply {
                    Err(DeviceOwnerError::Driver(_)) => OwnerFlow::Drain,
                    _ => OwnerFlow::Continue,
                };
                completion.complete(reply);
                return flow;
            }
            Command::Retire { lease, reply } => {
                let _ = reply.send(self.registry.retire(&lease).map_err(Into::into));
            }
            Command::Snapshot { reply } => {
                let _ = reply.send(self.driver.device_snapshot().map_err(Into::into));
            }
            Command::ReleaseNames { names, reply } => {
                self.driver
                    .release_resident_names(names.iter().map(AsRef::as_ref));
                let _ = reply.send(());
            }
            Command::ReleaseRanges { ranges, reply } => {
                self.driver.release_resident_ranges(&ranges);
                let _ = reply.send(());
            }
            Command::Shutdown { reply } => {
                let _ = reply.send(());
                return OwnerFlow::Drain;
            }
        }
        OwnerFlow::Continue
    }

    fn register(&mut self, request: RegisterRequest<Block>) -> Result<PlanLease, DeviceOwnerError> {
        let mut plan = {
            let named = request
                .blocks
                .iter()
                .map(|(name, block)| (name.as_ref(), QuantizedBlock::from(block)))
                .collect::<Vec<_>>();
            match &request.config {
                Some(config) => self.driver.plan_with_config(
                    &request.program,
                    &request.symbols,
                    &request.outputs,
                    &named,
                    config,
                )?,
                None => self.driver.plan_with_activation_policy(
                    &request.program,
                    &request.symbols,
                    &request.outputs,
                    request.numeric_policy,
                    request.exact_activations,
                    &named,
                )?,
            }
        };
        let resident = request
            .resident_names
            .iter()
            .map(AsRef::as_ref)
            .collect::<BTreeSet<&str>>();
        plan.mark_resident(&resident);
        let blocks = request
            .blocks
            .into_iter()
            .filter(|(name, _)| resident.contains(name.as_ref()))
            .collect();
        let owned = OwnedPlan { plan, blocks };
        Ok(self.registry.register(owned)?)
    }

    fn step(&mut self, lease: &PlanLease, frame: Frame<Block>) -> StepReply<Block> {
        let outcome =
            execute_registered_plan(&mut self.registry, lease, |owned| Self::run(owned, &frame))??;
        let sequence = self.completed;
        self.completed = self.completed.wrapping_add(1);
        Ok(match outcome {
            Ok(evaluated) => StepOutcome::Done {
                evaluated,
                frame,
                sequence,
            },
            Err(slots) => StepOutcome::NeedResident { slots, frame },
        })
    }

    fn run(
        owned: &mut OwnedPlan<R, Block>,
        frame: &Frame<Block>,
    ) -> Result<Result<Evaluated, Vec<u32>>, CudaDriverError> {
        let missing = frame
            .residents
            .iter()
            .filter(|resident| owned.blocks.get(resident.slot as usize).is_none())
            .map(|resident| resident.slot)
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Ok(Err(missing));
        }
        let named = frame
            .residents
            .iter()
            .filter_map(|resident| owned.blocks.get(resident.slot as usize))
            .chain(frame.dynamic.iter())
            .map(|(name, block)| (name.as_ref(), QuantizedBlock::from(block)))
            .collect::<Vec<_>>();
        owned.plan.execute_named(&named, None).map(Ok)
    }

    /// Answers a command the owner will not run: replies carry `Stopped`, releases and shutdown just ack.
    pub(crate) fn reject(command: Command<Block>) {
        match command {
            Command::Register { reply, .. } => {
                let _ = reply.send(Err(DeviceOwnerError::Stopped));
            }
            Command::Step { completion, .. } => completion.complete(Err(DeviceOwnerError::Stopped)),
            Command::Retire { reply, .. } => {
                let _ = reply.send(Err(DeviceOwnerError::Stopped));
            }
            Command::Snapshot { reply } => {
                let _ = reply.send(Err(DeviceOwnerError::Stopped));
            }
            Command::ReleaseNames { reply, .. }
            | Command::ReleaseRanges { reply, .. }
            | Command::Shutdown { reply } => {
                let _ = reply.send(());
            }
        }
    }
}

enum Lifecycle<R: GpuRuntime, Block, Opener> {
    Unopened {
        open: Opener,
        key: DeviceKey,
        plan_capacity: usize,
    },
    Open(DeviceOwner<R, Block>),
}

impl<R: GpuRuntime, Block, Opener> Lifecycle<R, Block, Opener>
where
    CudaDriverError: From<R::Error> + From<R::BuildError>,
    for<'block> QuantizedBlock<'block>: From<&'block Block>,
    Opener: FnMut() -> Result<CudaDriver<R>, CudaDriverError>,
{
    fn serve(&mut self, command: Command<Block>) -> OwnerFlow {
        match self {
            Self::Open(owner) => owner.serve(command),
            Self::Unopened {
                open,
                key,
                plan_capacity,
            } => match command {
                Command::ReleaseNames { reply, .. } | Command::ReleaseRanges { reply, .. } => {
                    let _ = reply.send(());
                    OwnerFlow::Continue
                }
                Command::Shutdown { reply } => {
                    let _ = reply.send(());
                    OwnerFlow::Drain
                }
                command => match open() {
                    Ok(driver) => {
                        let mut owner = DeviceOwner::new(*key, driver, *plan_capacity);
                        let flow = owner.serve(command);
                        *self = Self::Open(owner);
                        flow
                    }
                    Err(error) => {
                        Self::refuse(command, error);
                        OwnerFlow::Continue
                    }
                },
            },
        }
    }

    fn refuse(command: Command<Block>, error: CudaDriverError) {
        match command {
            Command::Register { reply, .. } => {
                let _ = reply.send(Err(DeviceOwnerError::Driver(error)));
            }
            Command::Step { completion, .. } => {
                completion.complete(Err(DeviceOwnerError::Driver(error)));
            }
            Command::Retire { reply, .. } => {
                let _ = reply.send(Err(DeviceOwnerError::Driver(error)));
            }
            Command::Snapshot { reply } => {
                let _ = reply.send(Err(DeviceOwnerError::Driver(error)));
            }
            Command::ReleaseNames { reply, .. }
            | Command::ReleaseRanges { reply, .. }
            | Command::Shutdown { reply } => {
                let _ = reply.send(());
            }
        }
    }
}

/// The owner loop as a future for the caller to spawn on any executor.
///
/// Composes [`run_owner`] over a private lifecycle: the device opens lazily, by calling `open`
/// on the first `Register`, `Step`, `Retire` or `Snapshot`. A failed open answers that one command with the
/// open error and the next command retries; `ReleaseNames`, `ReleaseRanges` and `Shutdown` never
/// open the device. `Shutdown`, or a `Step` that fails in the driver, closes the queue and every
/// queued command is rejected with [`DeviceOwnerError::Stopped`].
pub async fn run_device_owner<R: GpuRuntime, Block, Opener>(
    key: DeviceKey,
    config: &OwnerConfig,
    open: Opener,
    commands: Receiver<Command<Block>>,
) where
    CudaDriverError: From<R::Error> + From<R::BuildError>,
    for<'block> QuantizedBlock<'block>: From<&'block Block>,
    Opener: FnMut() -> Result<CudaDriver<R>, CudaDriverError> + Send,
{
    let lifecycle = Lifecycle::Unopened {
        open,
        key,
        plan_capacity: config.plan_capacity,
    };
    run_owner(
        commands,
        lifecycle,
        |lifecycle, commands, command| {
            let flow = lifecycle.serve(command);
            if flow == OwnerFlow::Drain {
                commands.close();
            }
            flow
        },
        |_, command| DeviceOwner::<R, Block>::reject(command),
    )
    .await;
}

/// A cloneable, `Send + Sync` sender of commands to one device owner.
pub struct DeviceHandle<Block> {
    key: DeviceKey,
    commands: Sender<Command<Block>>,
}

impl<Block> Clone for DeviceHandle<Block> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            commands: self.commands.clone(),
        }
    }
}

impl<Block> DeviceHandle<Block> {
    #[must_use]
    pub fn new(key: DeviceKey, commands: Sender<Command<Block>>) -> Self {
        Self { key, commands }
    }

    #[must_use]
    pub const fn key(&self) -> DeviceKey {
        self.key
    }

    async fn ask<Reply>(
        &self,
        build: impl FnOnce(oneshot::Sender<Reply>) -> Command<Block>,
    ) -> Result<Reply, DeviceOwnerError> {
        let (reply, answer) = oneshot::channel();
        self.commands
            .send(build(reply))
            .await
            .map_err(|_| DeviceOwnerError::Closed)?;
        answer.await.map_err(|_| DeviceOwnerError::Stopped)
    }

    pub async fn register(
        &self,
        request: RegisterRequest<Block>,
    ) -> Result<PlanLease, DeviceOwnerError> {
        self.ask(|reply| Command::Register {
            request: Box::new(request),
            reply,
        })
        .await?
    }

    /// Runs one step; `completion` is the plan's reusable reply slot.
    pub async fn step(
        &self,
        lease: &PlanLease,
        frame: Frame<Block>,
        completion: &Arc<Completion<StepReply<Block>>>,
    ) -> StepReply<Block> {
        let command = Command::Step {
            lease: lease.clone(),
            frame,
            completion: Arc::clone(completion),
        };
        self.commands
            .send(command)
            .await
            .map_err(|_| DeviceOwnerError::Closed)?;
        completion.wait().await
    }

    pub async fn retire(&self, lease: &PlanLease) -> Result<(), DeviceOwnerError> {
        self.ask(|reply| Command::Retire {
            lease: lease.clone(),
            reply,
        })
        .await?
    }

    /// Reads the device identity and graph activity from the owner, opening the device if it is not open yet.
    pub async fn snapshot(&self) -> Result<CudaDeviceSnapshot, DeviceOwnerError> {
        self.ask(|reply| Command::Snapshot { reply }).await?
    }

    pub async fn release_names(&self, names: Vec<Arc<str>>) -> Result<(), DeviceOwnerError> {
        self.ask(|reply| Command::ReleaseNames { names, reply })
            .await
    }

    pub async fn release_ranges(&self, ranges: Vec<Range<usize>>) -> Result<(), DeviceOwnerError> {
        self.ask(|reply| Command::ReleaseRanges { ranges, reply })
            .await
    }

    pub async fn shutdown(&self) -> Result<(), DeviceOwnerError> {
        self.ask(|reply| Command::Shutdown { reply }).await
    }
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives `future` to completion on the calling thread, parking between wakes.
pub fn block_on_thread<Output>(future: impl Future<Output = Output>) -> Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        thread::park();
    }
}

/// Starts the owner loop on one dedicated std thread: `open` runs on that thread, lazily and
/// retried per command, so the driver opens there and every device call is made there. For tests
/// and thread-per-device hosts. `config` sizes the command queue and the plan registry.
pub fn launch_on_thread<R, Block, Opener>(
    key: DeviceKey,
    config: &OwnerConfig,
    open: Opener,
) -> (DeviceHandle<Block>, JoinHandle<()>)
where
    R: GpuRuntime,
    CudaDriverError: From<R::Error> + From<R::BuildError>,
    Block: Send + 'static,
    for<'block> QuantizedBlock<'block>: From<&'block Block>,
    Opener: FnMut() -> Result<CudaDriver<R>, CudaDriverError> + Send + 'static,
{
    let (commands, receiver) = channel(config.queue_depth);
    let owner_config = config.clone();
    let joined = thread::spawn(move || {
        block_on_thread(run_device_owner(key, &owner_config, open, receiver));
    });
    (DeviceHandle::new(key, commands), joined)
}

#[cfg(test)]
mod tests {
    use super::{PlanRegistry, PlanRegistryError, execute_registered_plan};
    use std::sync::Arc;

    #[test]
    fn test_omega_async_plan_register_returns_service_lease() {
        fn assert_send<Type: Send>() {}
        assert_send::<crate::CudaPlan>();

        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 2);
        let lease = registry.register(41_u64).expect("plan should register");
        let job = registry.begin_job(&lease).expect("valid lease");
        assert_eq!(*registry.resolve_job_mut(&job).expect("valid job"), 41);
        registry.complete_job(job).expect("job should complete");
        assert!(Arc::ptr_eq(&lease.service, &service));
    }

    #[test]
    fn test_omega_async_plan_register_rejects_stale_and_foreign_leases() {
        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 1);
        let stale = registry.register(7_u64).expect("plan should register");
        registry.retire(&stale).expect("plan should retire");
        let replacement = registry.register(9_u64).expect("slot should be reused");
        assert_eq!(replacement.slotindex, stale.slotindex);
        assert_ne!(replacement.generation, stale.generation);
        assert_eq!(
            registry.begin_job(&stale).err(),
            Some(PlanRegistryError::Stale)
        );

        let mut other_service = PlanRegistry::<u64>::new(Arc::new(()), 1);
        assert_eq!(
            other_service.begin_job(&replacement).err(),
            Some(PlanRegistryError::WrongService)
        );
    }

    #[test]
    fn test_omega_async_plan_register_owner_access_is_serial() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(0_u64).expect("plan should register");
        let job = registry.begin_job(&lease).expect("job should start");
        *registry.resolve_job_mut(&job).expect("valid plan") += 1;
        *registry.resolve_job_mut(&job).expect("valid plan") += 1;
        assert_eq!(*registry.resolve_job_mut(&job).expect("valid plan"), 2);
        registry.complete_job(job).expect("job should complete");
    }

    #[test]
    fn test_omega_async_plan_retire_shape_miss_rejects_new_jobs() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(17_u64).expect("plan should register");
        registry
            .retire(&lease)
            .expect("shape miss should retire plan");
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
    }

    #[test]
    fn test_omega_async_plan_retire_stream_invalidation_rejects_new_jobs() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(23_u64).expect("plan should register");
        registry
            .retire(&lease)
            .expect("stream invalidation should retire plan");
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
    }

    #[test]
    fn test_omega_async_plan_retire_keeps_accepted_job_until_completion() {
        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 1);
        let lease = registry.register(31_u64).expect("plan should register");
        let accepted = registry.begin_job(&lease).expect("job should be accepted");
        registry.retire(&lease).expect("plan should retire");

        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Retired)
        );
        assert_eq!(
            *registry
                .resolve_job_mut(&accepted)
                .expect("accepted job retains retired plan"),
            31
        );
        assert!(registry.slots[lease.slotindex].plan.is_some());

        registry
            .complete_job(accepted)
            .expect("accepted job should retire");
        assert!(registry.slots[lease.slotindex].plan.is_none());
        let replacement = registry.register(37_u64).expect("slot should be reused");
        assert_eq!(replacement.slotindex, lease.slotindex);
        assert_ne!(replacement.generation, lease.generation);
    }

    #[test]
    fn test_omega_async_cuda_stage_rejects_before_executor_call() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(5_u64).expect("plan should register");
        registry.retire(&lease).expect("plan should retire");
        let mut invoked = false;
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            invoked = true;
            Ok::<_, &'static str>(())
        });

        assert_eq!(result, Err(PlanRegistryError::Stale));
        assert!(!invoked);
    }

    #[test]
    fn test_omega_async_cuda_stage_publishes_after_executor_finishes() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(13_u64).expect("plan should register");
        let mut completed = false;
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            completed = true;
            Ok::<_, &'static str>(19_u64)
        })
        .expect("registry should resolve");

        assert!(completed);
        assert_eq!(result, Ok(19));
        assert_eq!(registry.slots[lease.slotindex].in_flight, 0);
    }

    #[test]
    fn test_omega_async_cuda_stage_preserves_executor_failure_and_retires_job() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(29_u64).expect("plan should register");
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            Err::<(), _>("fake driver failure")
        })
        .expect("registry should resolve");

        assert_eq!(result, Err("fake driver failure"));
        assert_eq!(registry.slots[lease.slotindex].in_flight, 0);
        let job = registry
            .begin_job(&lease)
            .expect("lease should remain live");
        assert_eq!(
            *registry.resolve_job_mut(&job).expect("job should resolve"),
            29
        );
        registry.complete_job(job).expect("job should complete");
    }
}
