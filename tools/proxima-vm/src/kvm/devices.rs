use core::cell::{Cell, RefCell};
use core::future::Future;

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::process::{ChildRequest, ChildResponse};
use proxima_protocols::virtio::{MmioAccess, MmioEffect};

use super::error::KvmError;
use super::memory::GuestRam;
use super::pipes::ExitLedger;
use super::trap::{End, Reply, RunError, Trap, VcpuExit, Writeback};
use crate::dispatch::dispatch_captured;
use crate::virtio_blk::BlkTransport;
use crate::virtio_console::ConsoleTransport;
use crate::virtio_net::NetTransport;

pub const EMIT_VERB: u16 = 0xfffe;
pub const HALT_VERB: u16 = 0xffff;
const NO_QUEUE_NOTIFIED: u16 = 0xffff;
const DISPATCH_RESPONSE_SCRATCH_CAPACITY: usize = 512;
const LANE_BYTES: u64 = 4;

#[derive(Debug)]
pub struct Channel {
    bytes: Vec<u8>,
    capacity: usize,
}

impl Channel {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { bytes: Vec::new(), capacity }
    }

    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity - self.bytes.len()
    }

    pub fn push(&mut self, bytes: &[u8]) -> bool {
        let fits = bytes.len() <= self.remaining();
        if fits {
            self.bytes.extend_from_slice(bytes);
        }
        fits
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

pub type Verdict = core::future::Ready<Result<Reply, RunError>>;

pub fn answered(reply: Reply) -> Verdict {
    core::future::ready(Ok(reply))
}

pub fn failed(message: impl Into<String>) -> Verdict {
    core::future::ready(Err(RunError::Failed(KvmError::guest(message))))
}

pub fn declined(trap: Trap) -> Verdict {
    core::future::ready(Err(RunError::Declined(trap)))
}

/// The exits every lane treats alike: a signal stops `KVM_RUN` early and the
/// run simply resumes, a failed `KVM_RUN` is the run's error, and the
/// watchdog's expiry ends it.
pub struct Lifecycle {
    ends_on_shutdown: bool,
}

impl Lifecycle {
    #[must_use]
    pub const fn new(ends_on_shutdown: bool) -> Self {
        Self { ends_on_shutdown }
    }
}

impl Pipe for Lifecycle {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match trap {
            Trap::Exit(VcpuExit::Interrupted) => answered(Reply::ACK),
            Trap::Exit(VcpuExit::Shutdown) if self.ends_on_shutdown => answered(Reply::End(End::Shutdown)),
            Trap::Exit(VcpuExit::RunFailed { errno }) => core::future::ready(Err(KvmError::syscall(
                "KVM_RUN",
                std::io::Error::from_raw_os_error(errno),
            )
            .into())),
            Trap::WatchdogFired => core::future::ready(Err(RunError::Watchdog)),
            other => declined(other),
        }
    }
}

/// The hypercall endpoint: `out dx, al` carries the verb in the port and the
/// payload's location in `rdi`/`rsi`, which the exit source has already
/// captured into the exit. A tail device of the hypercall-only lane too: an
/// `in` is refused by name rather than declined.
pub struct Hypercall<'run, Dispatcher> {
    dispatcher: &'run Dispatcher,
    emitted: &'run RefCell<Channel>,
    sentinel: &'run Cell<i64>,
    calls: Cell<usize>,
    budget: usize,
}

impl<'run, Dispatcher> Hypercall<'run, Dispatcher>
where
    Dispatcher: Pipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
{
    #[must_use]
    pub fn new(
        dispatcher: &'run Dispatcher,
        emitted: &'run RefCell<Channel>,
        sentinel: &'run Cell<i64>,
        budget: usize,
    ) -> Self {
        Self { dispatcher, emitted, sentinel, calls: Cell::new(0), budget }
    }

    fn serve(&self, verb: u16, pointer: u64, length: u64, payload: Option<Bytes>) -> Result<Reply, RunError> {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        if calls > self.budget {
            return Err(KvmError::guest("guest exceeded hypercall budget without halting").into());
        }
        match verb {
            HALT_VERB => Ok(Reply::End(End::Halted)),
            EMIT_VERB => self.emit(payload.as_deref()),
            _ => self.dispatch(verb, pointer, length, payload),
        }
    }

    fn emit(&self, payload: Option<&[u8]>) -> Result<Reply, RunError> {
        let first = payload.and_then(|bytes| bytes.first());
        match first.map(|byte| self.emitted.borrow_mut().push(&[*byte])) {
            Some(true) => Ok(Reply::ACK),
            _ => Err(KvmError::guest("emit hypercall pointer or output capacity out of range").into()),
        }
    }

    fn dispatch(&self, verb: u16, pointer: u64, length: u64, payload: Option<Bytes>) -> Result<Reply, RunError> {
        let mut response = [0_u8; DISPATCH_RESPONSE_SCRATCH_CAPACITY];
        let declared = usize::try_from(length).unwrap_or(usize::MAX);
        let payload = payload.map(|bytes| bytes.slice(..declared.min(bytes.len())));
        let dispatched = dispatch_captured(self.dispatcher, verb, pointer, length, payload.as_deref(), &mut response);
        let Ok(response_length) = usize::try_from(dispatched) else {
            self.sentinel.set(dispatched);
            return Err(KvmError::guest(format!("hypercall dispatch failed with sentinel {dispatched}")).into());
        };
        let writable = response_length.min(usize::try_from(length).unwrap_or(usize::MAX));
        Ok(Reply::Hypercall {
            rax: response_length as u64,
            writeback: Some(Writeback { address: pointer, bytes: Bytes::copy_from_slice(&response[..writable]) }),
        })
    }
}

impl<Dispatcher> Pipe for Hypercall<'_, Dispatcher>
where
    Dispatcher: Pipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
{
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match trap {
            Trap::Exit(VcpuExit::PortIn { .. }) => {
                failed("guest read an io port; only `out dx, al` hypercalls are modeled")
            }
            Trap::Exit(VcpuExit::PortOut { port, arguments, payload, .. }) => {
                let (pointer, length) = arguments.map_or((0, 0), |arguments| (arguments.pointer, arguments.length));
                core::future::ready(self.serve(port, pointer, length, payload))
            }
            other => declined(other),
        }
    }
}

#[derive(Debug)]
pub enum Transport {
    Console(ConsoleTransport),
    Net(NetTransport),
    Blk(BlkTransport),
}

impl Transport {
    fn apply(&mut self, access: MmioAccess) -> Option<(u32, u16)> {
        let effect = match self {
            Self::Console(transport) => transport.apply(access),
            Self::Net(transport) => transport.apply(access),
            Self::Blk(transport) => transport.apply(access),
        }
        .ok()?;
        Some(match effect {
            MmioEffect::ReadValue(word) => (word, NO_QUEUE_NOTIFIED),
            MmioEffect::QueueNotify { queue } => (0, queue),
            _ => (0, NO_QUEUE_NOTIFIED),
        })
    }

    fn drain(&mut self, queue: u16, ram: &mut [u8]) -> Option<Vec<u8>> {
        match self {
            Self::Console(transport) => transport.drain_tx(queue, ram).ok(),
            Self::Net(transport) => {
                let mut frames = Vec::new();
                let mut collect = |drained: crate::virtio_net::DrainedFrame| frames.extend_from_slice(&drained.frame);
                transport.drain_tx(queue, ram, &mut collect).ok()?;
                Some(frames)
            }
            Self::Blk(transport) => {
                let serviced = transport.service_queue(queue, ram).ok()?;
                let mut encoded = Vec::new();
                for request in serviced {
                    encoded.extend_from_slice(&request.sector.to_le_bytes());
                    encoded.push(request.status);
                    encoded.extend_from_slice(&request.data);
                }
                Some(encoded)
            }
        }
    }
}

/// A virtio-mmio window on the bus: register accesses run through the
/// transport's register-block codec, and a notified queue is drained from
/// guest RAM into the window's own byte channel. An 8-byte access walks two
/// 32-bit register lanes (`offset` and `offset + 4`), the model's convention
/// for every 64-bit register it exposes.
pub struct VirtioWindow<'run> {
    name: &'static str,
    base: u64,
    size: u64,
    transport: RefCell<Option<Transport>>,
    emitted: &'run RefCell<Channel>,
    ram: GuestRam,
    ledger: &'run ExitLedger,
}

impl<'run> VirtioWindow<'run> {
    #[must_use]
    pub fn new(
        name: &'static str,
        base: u64,
        size: u64,
        transport: Option<Transport>,
        emitted: &'run RefCell<Channel>,
        ram: GuestRam,
        ledger: &'run ExitLedger,
    ) -> Self {
        Self { name, base, size, transport: RefCell::new(transport), emitted, ram, ledger }
    }

    fn owns(&self, address: u64) -> bool {
        address >= self.base && address - self.base < self.size
    }

    fn access(&self, address: u64, is_write: bool, written: &[u8], read_length: usize) -> Result<Reply, RunError> {
        let mut guard = self.transport.borrow_mut();
        let Some(transport) = guard.as_mut() else {
            return Err(KvmError::guest("mmio exit against a window with no transport").into());
        };
        let length = if is_write { written.len() } else { read_length };
        if length > 8 {
            return Err(KvmError::guest("mmio access wider than 8 bytes").into());
        }
        self.ledger.count_virtio_exit();
        let mut write_value = [0_u8; 8];
        write_value[..written.len().min(8)].copy_from_slice(&written[..written.len().min(8)]);
        let write_value = u64::from_le_bytes(write_value);
        let offset = address - self.base;
        let lanes = if length == 8 { 2 } else { 1 };
        let (mut read_value, mut notified) = (0_u64, NO_QUEUE_NOTIFIED);
        for lane in 0..lanes {
            let access = MmioAccess {
                offset: offset + lane * LANE_BYTES,
                is_write,
                value: (write_value >> (lane * 32)) as u32,
            };
            let Some((lane_value, lane_queue)) = transport.apply(access) else {
                return Err(KvmError::guest(format!(
                    "mmio register access rejected: window={} offset=0x{offset:x} is_write={}",
                    self.name,
                    u8::from(is_write)
                ))
                .into());
            };
            read_value |= u64::from(lane_value) << (lane * 32);
            if lane_queue != NO_QUEUE_NOTIFIED {
                notified = lane_queue;
            }
        }
        if notified != NO_QUEUE_NOTIFIED {
            self.drain(transport, notified)?;
        }
        Ok(match is_write {
            true => Reply::ACK,
            false => Reply::value(Bytes::copy_from_slice(&read_value.to_le_bytes()[..length])),
        })
    }

    fn drain(&self, transport: &mut Transport, queue: u16) -> Result<(), RunError> {
        let drained = self.ram.with_mut(|ram| transport.drain(queue, ram));
        let stored = drained.is_some_and(|bytes| self.emitted.borrow_mut().push(&bytes));
        match stored {
            true => Ok(()),
            false => Err(KvmError::guest("mmio queue drain failed").into()),
        }
    }
}

impl Pipe for VirtioWindow<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        let outcome = match &trap {
            Trap::Exit(VcpuExit::MmioWrite { address, data }) if self.owns(*address) => {
                self.access(*address, true, data, 0)
            }
            Trap::Exit(VcpuExit::MmioRead { address, length }) if self.owns(*address) => {
                self.access(*address, false, &[], usize::from(*length))
            }
            _ => Err(RunError::Declined(trap)),
        };
        core::future::ready(outcome)
    }
}

/// Tail of the elf lane's bus: an MMIO exit no window owns is the guest's bug.
pub struct RejectUnmapped;

impl Pipe for RejectUnmapped {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match trap {
            Trap::Exit(VcpuExit::MmioRead { .. } | VcpuExit::MmioWrite { .. }) => {
                failed("mmio exit outside any virtio window")
            }
            other => declined(other),
        }
    }
}
