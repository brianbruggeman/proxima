use core::cell::{Cell, RefCell};
use core::future::Future;

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::process::{ChildRequest, ChildResponse};
use proxima_protocols::virtio::{CONFIG_SPACE_BASE, MmioAccess, MmioEffect, MmioError};

use super::error::KvmError;
use super::memory::GuestRam;
use super::pipes::ExitLedger;
use super::trap::{End, IrqEdge, Reply, RunError, Trap, VcpuExit, Writeback};
use crate::dispatch::dispatch_captured;
use crate::virtio_blk::BlkTransport;
use crate::virtio_console::ConsoleTransport;
use crate::virtio_net::{HostNetTransport, RX_QUEUE};

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

    /// A zero-capacity channel declares the caller wants no byte stream: whatever
    /// the device drained already went out its own pipe.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.capacity == 0
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

/// Each frame behind its own little-endian u32 length, so a notify that drains
/// several chains keeps their boundaries in the byte stream.
fn length_prefixed<'frame>(frames: impl Iterator<Item = &'frame [u8]>) -> Vec<u8> {
    frames.fold(Vec::new(), |mut stream, frame| {
        stream.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        stream.extend_from_slice(frame);
        stream
    })
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
    Net(HostNetTransport),
    Blk(BlkTransport),
}

impl Transport {
    fn apply(&mut self, access: MmioAccess) -> Result<(u32, u16), MmioError> {
        let effect = match self {
            Self::Console(transport) => transport.apply(access),
            Self::Net(transport) => transport.apply(access),
            Self::Blk(transport) => transport.apply(access),
        }?;
        Ok(match effect {
            MmioEffect::ReadValue(word) => (word, NO_QUEUE_NOTIFIED),
            MmioEffect::QueueNotify { queue } => (0, queue),
            _ => (0, NO_QUEUE_NOTIFIED),
        })
    }

    fn interrupt_pending(&self) -> bool {
        match self {
            Self::Net(transport) => transport.interrupt_pending(),
            Self::Console(_) | Self::Blk(_) => false,
        }
    }

    fn config_read(&self, offset: u64, length: usize) -> Option<Result<u64, MmioError>> {
        match self {
            Self::Net(transport) => Some(transport.config_read(offset, length)),
            Self::Console(_) | Self::Blk(_) => None,
        }
    }

    fn drain(&mut self, queue: u16, ram: &mut [u8]) -> Result<Vec<u8>, String> {
        match self {
            Self::Console(transport) => transport.drain_tx(queue, ram).map_err(|error| error.to_string()),
            Self::Net(transport) if queue == RX_QUEUE => {
                let delivered = transport.fill_rx(ram).map_err(|error| error.to_string())?;
                proxima_telemetry::debug!(queue, delivered, "net rx queue filled");
                Ok(Vec::new())
            }
            Self::Net(transport) => {
                let drained = transport.drain_tx(queue, ram).map_err(|error| error.to_string())?;
                let lengths: Vec<usize> = drained.iter().map(|frame| frame.frame.len()).collect();
                proxima_telemetry::debug!(queue, chains = drained.len(), lengths = ?lengths, "net tx queue drained");
                Ok(length_prefixed(drained.iter().map(|frame| frame.frame.as_slice())))
            }
            Self::Blk(transport) => {
                let serviced = transport.service_queue(queue, ram).map_err(|error| error.to_string())?;
                let mut encoded = Vec::new();
                for request in serviced {
                    encoded.extend_from_slice(&request.sector.to_le_bytes());
                    encoded.push(request.status);
                    encoded.extend_from_slice(&request.data);
                }
                Ok(encoded)
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
    transport: &'run RefCell<Option<Transport>>,
    emitted: &'run RefCell<Channel>,
    ram: GuestRam,
    ledger: &'run ExitLedger,
    irq: Option<u32>,
    line: Cell<bool>,
}

impl<'run> VirtioWindow<'run> {
    #[must_use]
    pub fn new(
        name: &'static str,
        base: u64,
        size: u64,
        transport: &'run RefCell<Option<Transport>>,
        emitted: &'run RefCell<Channel>,
        ram: GuestRam,
        ledger: &'run ExitLedger,
    ) -> Self {
        Self { name, base, size, transport, emitted, ram, ledger, irq: None, line: Cell::new(false) }
    }

    /// The same window raising `line` while its device has an interrupt
    /// pending: every reply that changes the pending state carries the level change.
    #[must_use]
    pub fn with_irq(self, line: u32) -> Self {
        Self { irq: Some(line), ..self }
    }

    fn owns(&self, address: u64) -> bool {
        address >= self.base && address - self.base < self.size
    }

    fn edge(&self, transport: &Transport) -> Option<IrqEdge> {
        let line = self.irq?;
        let level = transport.interrupt_pending();
        (level != self.line.replace(level)).then_some(IrqEdge { line, level })
    }

    /// Delivers whatever the host has queued for the guest into the receive
    /// ring, returning the interrupt-line change that results.
    pub fn service_rx(&self) -> Result<Option<IrqEdge>, RunError> {
        let mut guard = self.transport.borrow_mut();
        let Some(transport) = guard.as_mut() else {
            return Err(KvmError::guest("rx service against a window with no transport").into());
        };
        self.drain(transport, RX_QUEUE)?;
        let irq = self.edge(transport);
        proxima_telemetry::debug!(window = self.name, pending = transport.interrupt_pending(), edge = ?irq, "rx serviced on doorbell");
        Ok(irq)
    }

    fn registers(
        &self,
        transport: &mut Transport,
        offset: u64,
        is_write: bool,
        write_value: u64,
        length: usize,
    ) -> Result<(u64, u16), RunError> {
        let narrow_config = !is_write && length < 4 && offset >= CONFIG_SPACE_BASE;
        if let Some(config) = narrow_config.then(|| transport.config_read(offset, length)).flatten() {
            let value = config.map_err(|error| {
                KvmError::guest(format!("mmio config read rejected: window={} offset=0x{offset:x} {error}", self.name))
            })?;
            return Ok((value, NO_QUEUE_NOTIFIED));
        }
        let lanes = if length == 8 { 2 } else { 1 };
        let (mut read_value, mut notified) = (0_u64, NO_QUEUE_NOTIFIED);
        for lane in 0..lanes {
            let access = MmioAccess {
                offset: offset + lane * LANE_BYTES,
                is_write,
                value: (write_value >> (lane * 32)) as u32,
            };
            let (lane_value, lane_queue) = transport.apply(access).map_err(|error| {
                KvmError::guest(format!(
                    "mmio register access rejected: window={} offset=0x{:x} is_write={} value=0x{:x} {error:?}",
                    self.name,
                    access.offset,
                    u8::from(is_write),
                    access.value
                ))
            })?;
            read_value |= u64::from(lane_value) << (lane * 32);
            if lane_queue != NO_QUEUE_NOTIFIED {
                notified = lane_queue;
            }
        }
        Ok((read_value, notified))
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
        let (read_value, notified) = self.registers(transport, offset, is_write, write_value, length)?;
        if notified != NO_QUEUE_NOTIFIED {
            proxima_telemetry::debug!(window = self.name, queue = notified, "queue notify");
            self.drain(transport, notified)?;
        }
        let value = (!is_write).then(|| Bytes::copy_from_slice(&read_value.to_le_bytes()[..length]));
        let irq = self.edge(transport);
        if matches!(offset, 0x50 | 0x60 | 0x64) || irq.is_some() {
            proxima_telemetry::debug!(window = self.name, offset, is_write, write_value, read_value, pending = transport.interrupt_pending(), edge = ?irq, "mmio interrupt-path access");
        }
        Ok(Reply::Io { value, irq })
    }

    fn drain(&self, transport: &mut Transport, queue: u16) -> Result<(), RunError> {
        let bytes = self
            .ram
            .with_mut(|ram| transport.drain(queue, ram))
            .map_err(|error| KvmError::guest(format!("mmio queue drain failed: window={} queue={queue} {error}", self.name)))?;
        let mut channel = self.emitted.borrow_mut();
        match channel.is_disabled() || channel.push(&bytes) {
            true => Ok(()),
            false => Err(KvmError::guest(format!(
                "mmio emitted channel full: window={} queue={queue} bytes={} remaining={}",
                self.name,
                bytes.len(),
                channel.remaining()
            ))
            .into()),
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

impl Pipe for &VirtioWindow<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        VirtioWindow::call(self, trap)
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use futures::FutureExt;
    use proxima_protocols::virtio::mmio::{
        REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_INTERRUPT_ACK, REG_INTERRUPT_STATUS, REG_QUEUE_DESC_HIGH,
        REG_QUEUE_DESC_LOW, REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH, REG_QUEUE_DRIVER_LOW,
        REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL, REG_STATUS,
    };
    use proxima_protocols::virtio::status::{STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK};
    use proxima_protocols::virtio::DriverQueue;

    use super::*;
    use crate::frame_pipe::{FrameSink, frame_channel};
    use crate::kvm::memory::HostMapping;
    use crate::virtio_net::NetTransport;

    const BASE: u64 = 0xd000_0000;
    const LINE: u32 = 5;
    const MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    const RX_TABLE: usize = 0x4000;
    const RX_AVAIL: usize = 0x5000;
    const RX_USED: u64 = 0x6000;
    const BUFFER: u64 = 0x9000;
    const MEMORY: usize = 0x10000;

    struct Rig {
        mapping: HostMapping,
        transport: RefCell<Option<Transport>>,
        emitted: RefCell<Channel>,
        ledger: ExitLedger,
        to_guest: FrameSink,
    }

    fn rig() -> Rig {
        let (to_guest, ingress) = frame_channel(8);
        let (egress, _from_guest) = frame_channel(8);
        let transport = Transport::Net(NetTransport::with_pipes(MAC, ingress, egress));
        Rig {
            mapping: HostMapping::anonymous(MEMORY, false).expect("anonymous guest memory maps"),
            transport: RefCell::new(Some(transport)),
            emitted: RefCell::new(Channel::with_capacity(0)),
            ledger: ExitLedger::begin(),
            to_guest,
        }
    }

    fn window(rig: &Rig) -> VirtioWindow<'_> {
        VirtioWindow::new("net", BASE, 0x1000, &rig.transport, &rig.emitted, GuestRam::of(&rig.mapping), &rig.ledger)
    }

    fn write(window: &VirtioWindow<'_>, offset: u64, value: u32) -> Reply {
        let trap = Trap::Exit(VcpuExit::MmioWrite { address: BASE + offset, data: Bytes::copy_from_slice(&value.to_le_bytes()) });
        window.call(trap).now_or_never().expect("ready").expect("window answers")
    }

    fn read(window: &VirtioWindow<'_>, offset: u64, length: u8) -> Reply {
        let trap = Trap::Exit(VcpuExit::MmioRead { address: BASE + offset, length });
        window.call(trap).now_or_never().expect("ready").expect("window answers")
    }

    fn bring_up_rx(window: &VirtioWindow<'_>, rig: &Rig) {
        write(window, REG_STATUS, u32::from(STATUS_ACKNOWLEDGE));
        write(window, REG_STATUS, u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER));
        for (select, word) in [(0, 0), (1, 1)] {
            write(window, REG_DRIVER_FEATURES_SEL, select);
            write(window, REG_DRIVER_FEATURES, word);
        }
        write(window, REG_STATUS, u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK));
        for (offset, value) in [
            (REG_QUEUE_SEL, 0),
            (REG_QUEUE_NUM, 4),
            (REG_QUEUE_DESC_LOW, RX_TABLE as u32),
            (REG_QUEUE_DESC_HIGH, 0),
            (REG_QUEUE_DRIVER_LOW, RX_AVAIL as u32),
            (REG_QUEUE_DRIVER_HIGH, 0),
            (REG_QUEUE_DEVICE_LOW, RX_USED as u32),
            (REG_QUEUE_DEVICE_HIGH, 0),
            (REG_QUEUE_READY, 1),
        ] {
            write(window, offset, value);
        }
        let all = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK;
        write(window, REG_STATUS, u32::from(all));
        GuestRam::of(&rig.mapping).with_mut(|memory| {
            let mut queue = DriverQueue::<4>::new(&mut memory[RX_TABLE..]).expect("queue lays out");
            let (before, after) = memory.split_at_mut(RX_AVAIL);
            queue.post(&mut before[RX_TABLE..], after, &[], &[(BUFFER, 2048)]).expect("buffer posts");
        });
    }

    #[test]
    fn mac_read_bytewise_through_a_one_byte_mmio_read_returns_each_mac_byte() {
        let rig = rig();
        let window = window(&rig);

        let bytes: Vec<u8> = (0..6)
            .map(|index| match read(&window, 0x100 + index, 1) {
                Reply::Io { value: Some(value), .. } => value[0],
                other => panic!("expected a value reply, got {other:?}"),
            })
            .collect();

        assert_eq!(bytes, MAC);
    }

    #[test]
    fn status_reads_as_sixteen_bits_at_offset_0x106() {
        let rig = rig();
        let window = window(&rig);

        let reply = read(&window, 0x106, 2);

        assert_eq!(reply, Reply::value(Bytes::from_static(&[1, 0])));
    }

    #[test]
    fn a_rejected_narrow_config_read_is_the_runs_error() {
        let rig = rig();
        let window = window(&rig);
        let trap = Trap::Exit(VcpuExit::MmioRead { address: BASE + 0x10b, length: 2 });

        let outcome = window.call(trap).now_or_never().expect("ready");

        assert!(matches!(outcome, Err(RunError::Failed(_))));
    }

    #[test]
    fn irq_edge_rides_the_rx_delivery_then_the_ack_lowers_it_and_nothing_repeats() {
        let rig = rig();
        let window = window(&rig).with_irq(LINE);
        bring_up_rx(&window, &rig);
        rig.to_guest.call(vec![0xab; 60]).now_or_never().expect("ready").expect("frame queues");

        let raised = window.service_rx().expect("rx services");
        let repeated = window.service_rx().expect("rx services again");
        let status = read(&window, REG_INTERRUPT_STATUS, 4);
        let acked = write(&window, REG_INTERRUPT_ACK, 1);
        let after = read(&window, REG_INTERRUPT_STATUS, 4);

        assert_eq!(raised, Some(IrqEdge { line: LINE, level: true }));
        assert_eq!(repeated, None, "the level has not changed");
        assert_eq!(status, Reply::Io { value: Some(Bytes::from_static(&[1, 0, 0, 0])), irq: None });
        assert_eq!(acked, Reply::Io { value: None, irq: Some(IrqEdge { line: LINE, level: false }) });
        assert_eq!(after, Reply::Io { value: Some(Bytes::from_static(&[0, 0, 0, 0])), irq: None });
    }

    #[test]
    fn a_window_without_an_irq_line_never_reports_an_edge() {
        let rig = rig();
        let window = window(&rig);
        bring_up_rx(&window, &rig);
        rig.to_guest.call(vec![0xab; 60]).now_or_never().expect("ready").expect("frame queues");

        let edge = window.service_rx().expect("rx services");

        assert_eq!(edge, None);
    }
}
