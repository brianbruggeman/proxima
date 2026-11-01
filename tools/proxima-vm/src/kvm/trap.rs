use arrayvec::ArrayVec;
use bytes::Bytes;

use super::error::KvmError;
use super::slots::SlotOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HypercallArguments {
    pub pointer: u64,
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VcpuExit {
    PortOut {
        port: u16,
        size: u8,
        count: u32,
        data: Bytes,
        arguments: Option<HypercallArguments>,
        payload: Option<Bytes>,
    },
    PortIn {
        port: u16,
        size: u8,
        count: u32,
    },
    MmioRead {
        address: u64,
        length: u8,
    },
    MmioWrite {
        address: u64,
        data: Bytes,
    },
    Hlt,
    Shutdown,
    Interrupted,
    RunFailed {
        errno: i32,
    },
    FailEntry {
        hardware_reason: u64,
    },
    InternalError {
        suberror: u32,
    },
    Other {
        reason: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trap {
    Exit(VcpuExit),
    WatchdogFired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IrqEdge {
    pub line: u32,
    pub level: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Writeback {
    pub address: u64,
    pub bytes: Bytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Halted,
    ResetPort,
    Shutdown,
    OutputOverflow,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    Io { value: Option<Bytes>, irq: Option<IrqEdge> },
    Hypercall { rax: u64, writeback: Option<Writeback> },
    Remap(ArrayVec<SlotOp, 4>),
    End(End),
}

impl Reply {
    pub const ACK: Self = Self::Io { value: None, irq: None };

    #[must_use]
    pub fn value(bytes: Bytes) -> Self {
        Self::Io { value: Some(bytes), irq: None }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("the trap sources are exhausted")]
    Exhausted,
    #[error("no device claimed {0:?}")]
    Declined(Trap),
    #[error("the run ended: {0:?}")]
    Ended(End),
    #[error("the watchdog expired")]
    Watchdog,
    #[error("{0}")]
    Failed(#[from] KvmError),
}

impl From<proxima_primitives::pipe::fan_in::Exhausted> for RunError {
    fn from(_: proxima_primitives::pipe::fan_in::Exhausted) -> Self {
        Self::Exhausted
    }
}
