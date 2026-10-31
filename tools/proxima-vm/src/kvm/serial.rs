use core::cell::{Cell, RefCell};
use core::future::Future;

use bytes::Bytes;
use proxima_primitives::pipe::Pipe;

use super::devices::{answered, declined};
use super::pipes::ExitLedger;
use super::trap::{End, IrqEdge, Reply, RunError, Trap, VcpuExit};
use crate::cmos_rtc::{CmosRtc, DATA_PORT, INDEX_PORT};
use crate::uart16550::{PORT_COUNT, Uart16550};

pub const COM1_BASE: u16 = 0x3f8;
pub const COM1_IRQ: u32 = 4;
pub const COM2_BASE: u16 = 0x2f8;
pub const COM2_IRQ: u32 = 3;
pub const RESET_PORT: u16 = 0x64;
pub const RESET_VALUE: u8 = 0xfe;
pub const MARKER_PORT: u16 = 0xe9;
pub const MARKER_INVOCATION_READY: u8 = 2;
pub const MARKER_ENTRYPOINT: u8 = 3;
pub const MAX_UNHANDLED_PORTS: usize = 32;

#[derive(Debug)]
pub struct Transmit {
    bytes: Vec<u8>,
    capacity: usize,
    dropped: usize,
}

impl Transmit {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { bytes: Vec::new(), capacity, dropped: 0 }
    }

    fn push(&mut self, byte: u8) {
        match self.bytes.len() < self.capacity {
            true => self.bytes.push(byte),
            false => self.dropped += 1,
        }
    }

    #[must_use]
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// One 16550 on the port bus: port accesses run through the sans-IO codec,
/// transmitted bytes land in the lane-owned buffer, and a change of the
/// interrupt line rides back in the reply.
pub struct SerialPort<'run> {
    base: u16,
    irq: u32,
    uart: RefCell<Uart16550<'run>>,
    transmit: &'run RefCell<Transmit>,
    line: Cell<bool>,
    ends_on_overflow: bool,
}

impl<'run> SerialPort<'run> {
    #[must_use]
    pub fn new(
        base: u16,
        irq: u32,
        receive: &'run [u8],
        transmit: &'run RefCell<Transmit>,
        ends_on_overflow: bool,
    ) -> Self {
        Self {
            base,
            irq,
            uart: RefCell::new(Uart16550::new(receive)),
            transmit,
            line: Cell::new(false),
            ends_on_overflow,
        }
    }

    fn owns(&self, port: u16) -> bool {
        port >= self.base && port - self.base < PORT_COUNT
    }

    fn edge(&self) -> Option<IrqEdge> {
        let level = self.uart.borrow().irq_level();
        (level != self.line.replace(level)).then_some(IrqEdge { line: self.irq, level })
    }

    pub fn enable_receive(&self) -> Option<IrqEdge> {
        self.uart.borrow_mut().enable_receive();
        self.edge()
    }

    fn write(&self, port: u16, data: &[u8], size: usize) -> Reply {
        let offset = (port - self.base) as u8;
        for chunk in data.chunks(size.max(1)) {
            if let Some(byte) = self.uart.borrow_mut().write(offset, chunk[0]) {
                self.transmit.borrow_mut().push(byte);
            }
        }
        match self.ends_on_overflow && self.transmit.borrow().dropped() > 0 {
            true => Reply::End(End::OutputOverflow),
            false => Reply::Io { value: None, irq: self.edge() },
        }
    }

    fn read(&self, port: u16, size: usize, count: usize) -> Reply {
        let offset = (port - self.base) as u8;
        let mut value = vec![0xff_u8; size * count];
        for chunk in value.chunks_mut(size.max(1)) {
            chunk[0] = self.uart.borrow_mut().read(offset);
        }
        Reply::Io { value: Some(Bytes::from(value)), irq: self.edge() }
    }
}

impl Pipe for SerialPort<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port, size, data, .. }) if self.owns(*port) => {
                answered(self.write(*port, data, usize::from(*size)))
            }
            Trap::Exit(VcpuExit::PortIn { port, size, count }) if self.owns(*port) => {
                answered(self.read(*port, usize::from(*size), *count as usize))
            }
            _ => declined(trap),
        }
    }
}

/// The invocation channel is on the bus and also the target of the init marker,
/// so the bus holds it by reference.
impl Pipe for &SerialPort<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        SerialPort::call(self, trap)
    }
}

/// The CMOS RTC at ports 0x70 and 0x71.
pub struct RtcPorts {
    rtc: RefCell<CmosRtc>,
}

impl RtcPorts {
    #[must_use]
    pub fn at(seconds_since_epoch: i64) -> Self {
        Self { rtc: RefCell::new(CmosRtc::at(seconds_since_epoch)) }
    }
}

impl Pipe for RtcPorts {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port, data, .. }) if matches!(*port, INDEX_PORT | DATA_PORT) => {
                self.rtc.borrow_mut().write(*port, data[0]);
                answered(Reply::ACK)
            }
            Trap::Exit(VcpuExit::PortIn { port, size, count }) if matches!(*port, INDEX_PORT | DATA_PORT) => {
                let mut value = vec![0xff_u8; usize::from(*size) * *count as usize];
                value[0] = self.rtc.borrow().read(*port);
                answered(Reply::value(Bytes::from(value)))
            }
            _ => declined(trap),
        }
    }
}

/// The i8042 reset port: the 0xfe command is the guest asking to power off.
pub struct ResetPort;

impl Pipe for ResetPort {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port: RESET_PORT, data, .. }) if data[0] == RESET_VALUE => {
                answered(Reply::End(End::ResetPort))
            }
            _ => declined(trap),
        }
    }
}

/// The init marker port: the first write stamps when `/init` began, and the
/// invocation-ready value releases the held input stream to COM2.
pub struct InitMarker<'run> {
    invocation: &'run SerialPort<'run>,
    ledger: &'run ExitLedger,
    stamp: &'run Cell<Option<u64>>,
    entrypoint_stamp: &'run Cell<Option<u64>>,
}

impl<'run> InitMarker<'run> {
    #[must_use]
    pub fn new(
        invocation: &'run SerialPort<'run>,
        ledger: &'run ExitLedger,
        stamp: &'run Cell<Option<u64>>,
        entrypoint_stamp: &'run Cell<Option<u64>>,
    ) -> Self {
        Self { invocation, ledger, stamp, entrypoint_stamp }
    }
}

impl Pipe for InitMarker<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port: MARKER_PORT, data, .. }) => {
                if self.stamp.get().is_none() {
                    self.stamp.set(Some(self.ledger.elapsed_nanos()));
                }
                if data[0] == MARKER_ENTRYPOINT && self.entrypoint_stamp.get().is_none() {
                    self.entrypoint_stamp.set(Some(self.ledger.elapsed_nanos()));
                }
                let irq = (data[0] == MARKER_INVOCATION_READY).then(|| self.invocation.enable_receive()).flatten();
                answered(Reply::Io { value: None, irq })
            }
            _ => declined(trap),
        }
    }
}

#[derive(Debug, Default)]
pub struct PortTally {
    pub entries: Vec<(u16, u64, u64)>,
    pub overflow: u64,
}

impl PortTally {
    fn count(&mut self, port: u16, is_write: bool) {
        let (reads, writes) = if is_write { (0, 1) } else { (1, 0) };
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.0 == port) {
            entry.1 += reads;
            entry.2 += writes;
        } else if self.entries.len() < MAX_UNHANDLED_PORTS {
            self.entries.push((port, reads, writes));
        } else {
            self.overflow += 1;
        }
    }
}

/// Tail of the Linux bus: whatever nothing answered. Reads see all ones, writes
/// vanish, and each port is tallied so a guest probing hardware this model
/// lacks is visible in the report rather than silent.
pub struct Unhandled<'run> {
    tally: &'run RefCell<PortTally>,
}

impl<'run> Unhandled<'run> {
    #[must_use]
    pub const fn new(tally: &'run RefCell<PortTally>) -> Self {
        Self { tally }
    }
}

impl Pipe for Unhandled<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port, .. }) => {
                self.tally.borrow_mut().count(*port, true);
                answered(Reply::ACK)
            }
            Trap::Exit(VcpuExit::PortIn { port, size, count }) => {
                self.tally.borrow_mut().count(*port, false);
                answered(Reply::value(Bytes::from(vec![0xff_u8; usize::from(*size) * *count as usize])))
            }
            Trap::Exit(VcpuExit::MmioRead { length, .. }) => {
                answered(Reply::value(Bytes::from(vec![0xff_u8; usize::from(*length).max(8)])))
            }
            Trap::Exit(VcpuExit::MmioWrite { .. }) => answered(Reply::ACK),
            _ => declined(trap),
        }
    }
}
