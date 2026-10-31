use core::convert::Infallible;
use core::future::Future;
use core::mem::ManuallyDrop;
use std::io;
use std::os::fd::OwnedFd;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, Thread};

use futures::FutureExt;
use proxima_core::ring::{EnqueueOutcome, FailMode, HeapBoundedQueue};
use proxima_primitives::pipe::Pipe;

use super::error::{KvmError, KvmResult};
use super::memory::{GuestRam, HostMapping, NamedRegion, close_checked};
use super::vcpu::{Machine, Vcpu};

const QUEUE_CAPACITY: usize = 16;
const WORKERS: usize = 4;
const FAILURE_CAPACITY: usize = 8;

/// Everything one run owns: the VM and its vCPU, guest RAM, and the host
/// mappings registered as memory slots. Dropping a session releases all of
/// it, in the order the kernel needs: vCPU, then VM, then memory.
pub struct Session {
    vcpu: Option<Vcpu>,
    machine: ManuallyDrop<Machine>,
    vm_descriptor: Option<OwnedFd>,
    named_regions: Vec<NamedRegion>,
    mappings: Vec<HostMapping>,
}

impl Session {
    pub fn open() -> KvmResult<Self> {
        let machine = Machine::create()?;
        let vm_descriptor = machine.duplicate_vm_descriptor()?;
        Ok(Self {
            vcpu: None,
            machine: ManuallyDrop::new(machine),
            vm_descriptor: Some(vm_descriptor),
            named_regions: Vec::new(),
            mappings: Vec::new(),
        })
    }

    #[must_use]
    pub fn machine(&self) -> &Machine {
        &self.machine
    }

    pub fn create_vcpu(&mut self) -> KvmResult<()> {
        self.vcpu = Some(self.machine.create_vcpu()?);
        Ok(())
    }

    pub fn vcpu(&self) -> KvmResult<&Vcpu> {
        self.vcpu.as_ref().ok_or_else(|| KvmError::guest("the vcpu is created before the run starts"))
    }

    pub fn attach_named_region(&mut self, region: NamedRegion) -> GuestRam {
        let ram = GuestRam::of(region.primary());
        self.named_regions.push(region);
        ram
    }

    pub fn attach_mapping(&mut self, mapping: HostMapping) {
        self.mappings.push(mapping);
    }

    fn release(&mut self) -> Vec<(&'static str, io::Error)> {
        let mut failures = Vec::new();
        drop(self.vcpu.take());
        // SAFETY: the machine is dropped exactly once, here, and never touched again.
        unsafe { ManuallyDrop::drop(&mut self.machine) };
        if let Some(descriptor) = self.vm_descriptor.take()
            && let Err(error) = close_checked(descriptor)
        {
            failures.push(("close vm fd", error));
        }
        for mapping in self.mappings.drain(..) {
            if let Err(error) = mapping.release() {
                failures.push(("munmap guest mapping", error));
            }
        }
        for region in self.named_regions.drain(..) {
            failures.extend(region.release().into_iter().map(|error| ("release guest memory", error)));
        }
        failures
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for (operation, error) in self.release() {
            record_failure(operation, &error);
        }
    }
}

struct Teardown {
    sessions: HeapBoundedQueue<Session>,
    workers: Vec<Thread>,
    failures: HeapBoundedQueue<String>,
    reported_dropped: AtomicU64,
    next_worker: AtomicUsize,
}

fn worker_main(teardown: &'static Teardown) {
    loop {
        match teardown.sessions.dequeue() {
            Some(session) => drop(session),
            None => thread::park(),
        }
    }
}

fn start_teardown() -> Teardown {
    let mut teardown = Teardown {
        sessions: HeapBoundedQueue::new(QUEUE_CAPACITY, FailMode::FailClosed),
        workers: Vec::new(),
        failures: HeapBoundedQueue::new(FAILURE_CAPACITY, FailMode::DropNewest),
        reported_dropped: AtomicU64::new(0),
        next_worker: AtomicUsize::new(0),
    };
    let mut started = Vec::new();
    for index in 0..WORKERS {
        let spawned = thread::Builder::new()
            .name(format!("proxima-vm-teardown-{index}"))
            .spawn(|| worker_main(teardown_state()));
        match spawned {
            Ok(handle) => started.push(handle.thread().clone()),
            Err(error) => teardown_failure("spawn teardown worker", &error, &teardown.failures),
        }
    }
    teardown.workers = started;
    teardown
}

fn teardown_state() -> &'static Teardown {
    static TEARDOWN: OnceLock<Teardown> = OnceLock::new();
    TEARDOWN.get_or_init(start_teardown)
}

fn teardown_failure(operation: &'static str, error: &io::Error, failures: &HeapBoundedQueue<String>) {
    proxima_telemetry::error!(operation, %error, "vm teardown failed to release a resource");
    failures.enqueue(format!("{operation}: {error}"));
}

pub(crate) fn record_failure(operation: &'static str, error: &io::Error) {
    teardown_failure(operation, error, &teardown_state().failures);
}

/// Drains what teardown failed to release, one line per failure, plus a line
/// naming how many further failures found the record full.
#[must_use]
pub fn take_teardown_failures() -> Vec<String> {
    let teardown = teardown_state();
    let mut lines = Vec::new();
    while let Some(line) = teardown.failures.dequeue() {
        lines.push(line);
    }
    let dropped = teardown.failures.dropped();
    let unreported = dropped - teardown.reported_dropped.swap(dropped, Ordering::AcqRel);
    if unreported > 0 {
        lines.push(format!("{unreported} further teardown failures were not recorded"));
    }
    lines
}

/// Sink: hands a finished run's session to the bounded queue the teardown
/// workers drain, so the slow kernel VM destruction is off the thread about
/// to answer. A full queue is the backpressure: the session is refused by the
/// queue and released inline, on the caller, never leaked.
pub struct TeardownSink;

impl Pipe for TeardownSink {
    type In = Session;
    type Out = ();
    type Err = Infallible;

    fn call(&self, session: Session) -> impl Future<Output = Result<(), Infallible>> {
        let teardown = teardown_state();
        if teardown.workers.is_empty() {
            drop(session);
        } else if teardown.sessions.enqueue(session) == EnqueueOutcome::Enqueued {
            let index = teardown.next_worker.fetch_add(1, Ordering::Relaxed) % teardown.workers.len();
            teardown.workers[index].unpark();
        }
        core::future::ready(Ok(()))
    }
}

/// Hands `session` to the teardown workers. Ready on its first poll.
pub fn release_off_the_return_path(session: Session) {
    let _ = Pipe::call(&TeardownSink, session).now_or_never();
}
