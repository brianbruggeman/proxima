use core::cell::Cell;
use core::future::Future;
use std::time::Instant;

use proxima_primitives::pipe::Pipe;

use super::error::KvmError;
use super::memory::GuestRam;
use super::trap::{Reply, RunError, Trap, VcpuExit};
use super::vcpu::{Machine, Vcpu};

pub const MAX_TOTAL_EXITS: u64 = 5_000_000;

/// One-of-N routing by claim: `head` answers a trap it owns and declines
/// the rest, handing the untouched trap on to `tail`. A bus is nested
/// `FirstClaim`s, each device naming its own ports or addresses.
pub struct FirstClaim<Head, Tail> {
    head: Head,
    tail: Tail,
}

impl<Head, Tail> FirstClaim<Head, Tail> {
    #[must_use]
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }
}

impl<Head, Tail> Pipe for FirstClaim<Head, Tail>
where
    Head: Pipe<In = Trap, Out = Reply, Err = RunError>,
    Tail: Pipe<In = Trap, Out = Reply, Err = RunError>,
{
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        async move {
            match self.head.call(trap).await {
                Err(RunError::Declined(unclaimed)) => self.tail.call(unclaimed).await,
                answered => answered,
            }
        }
    }
}

/// What the meter has seen of the run so far; the lane reads it back after the
/// run ends.
#[derive(Debug)]
pub struct ExitLedger {
    started: Instant,
    exits: Cell<u64>,
    mmio_exits: Cell<u64>,
    virtio_exits: Cell<u64>,
    first_exit_nanos: Cell<Option<u64>>,
}

impl ExitLedger {
    #[must_use]
    pub fn begin() -> Self {
        Self {
            started: Instant::now(),
            exits: Cell::new(0),
            mmio_exits: Cell::new(0),
            virtio_exits: Cell::new(0),
            first_exit_nanos: Cell::new(None),
        }
    }

    #[must_use]
    pub fn exits(&self) -> u64 {
        self.exits.get()
    }

    #[must_use]
    pub fn mmio_exits(&self) -> u64 {
        self.mmio_exits.get()
    }

    #[must_use]
    pub fn virtio_exits(&self) -> u64 {
        self.virtio_exits.get()
    }

    #[must_use]
    pub fn first_exit_nanos(&self) -> u64 {
        self.first_exit_nanos.get().unwrap_or(0)
    }

    #[must_use]
    pub fn elapsed_nanos(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    pub fn count_virtio_exit(&self) {
        self.virtio_exits.set(self.virtio_exits.get() + 1);
    }
}

/// Observe: hands every trap back untouched, counting exits and stamping the
/// first, and fails the run once it has seen more than `budget`.
pub struct Meter<'run> {
    ledger: &'run ExitLedger,
    budget: u64,
    exceeded: &'static str,
}

impl<'run> Meter<'run> {
    #[must_use]
    pub const fn new(ledger: &'run ExitLedger, budget: u64, exceeded: &'static str) -> Self {
        Self { ledger, budget, exceeded }
    }
}

impl Pipe for Meter<'_> {
    type In = Trap;
    type Out = Trap;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Trap, RunError>> {
        let counted = matches!(&trap, Trap::Exit(exit) if !matches!(exit, VcpuExit::Interrupted | VcpuExit::RunFailed { .. }));
        if counted {
            let seen = self.ledger.exits.get() + 1;
            self.ledger.exits.set(seen);
            if seen == 1 {
                self.ledger.first_exit_nanos.set(Some(self.ledger.elapsed_nanos()));
            }
            if matches!(&trap, Trap::Exit(VcpuExit::MmioRead { .. } | VcpuExit::MmioWrite { .. })) {
                self.ledger.mmio_exits.set(self.ledger.mmio_exits.get() + 1);
            }
        }
        let over_budget = counted && self.ledger.exits.get() > self.budget;
        let exceeded = self.exceeded;
        async move {
            match over_budget {
                true => Err(RunError::Failed(KvmError::guest(exceeded))),
                false => Ok(trap),
            }
        }
    }
}

/// Sink: applies a device's reply to the stopped vCPU and its VM, so the next
/// `KVM_RUN` resumes with the answer in place. A terminal reply ends the run
/// through the error channel, the way `FanIn` ends a merge.
pub struct Resume<'run> {
    machine: &'run Machine,
    vcpu: &'run Vcpu,
    ram: GuestRam,
}

impl<'run> Resume<'run> {
    #[must_use]
    pub const fn new(machine: &'run Machine, vcpu: &'run Vcpu, ram: GuestRam) -> Self {
        Self { machine, vcpu, ram }
    }

    fn apply(&self, reply: Reply) -> Result<(), RunError> {
        match reply {
            Reply::Io { value, irq } => {
                if let Some(bytes) = value {
                    self.vcpu.complete_read(&bytes);
                }
                irq.map_or(Ok(()), |edge| self.machine.set_irq_line(edge.line, edge.level))?;
                Ok(())
            }
            Reply::Hypercall { rax, writeback } => {
                if let Some(write) = writeback
                    && !self.ram.write(write.address, &write.bytes)
                {
                    return Err(KvmError::guest("dispatch response write-back would overrun guest memory").into());
                }
                Ok(self.vcpu.set_rax(rax)?)
            }
            Reply::Remap(ops) => Ok(self.machine.apply(&ops, "KVM_SET_USER_MEMORY_REGION delta page")?),
            Reply::End(end) => Err(RunError::Ended(end)),
        }
    }
}

impl Pipe for Resume<'_> {
    type In = Reply;
    type Out = ();
    type Err = RunError;

    fn call(&self, reply: Reply) -> impl Future<Output = Result<(), RunError>> {
        let outcome = self.apply(reply);
        async move { outcome }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use bytes::Bytes;
    use futures::executor::block_on;

    use super::*;
    use crate::kvm::devices::{answered, declined};

    struct PortRange {
        first: u16,
        last: u16,
        tag: u8,
    }

    impl Pipe for PortRange {
        type In = Trap;
        type Out = Reply;
        type Err = RunError;

        fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
            match &trap {
                Trap::Exit(VcpuExit::PortIn { port, .. }) if (self.first..=self.last).contains(port) => {
                    answered(Reply::value(Bytes::from(vec![self.tag])))
                }
                _ => declined(trap),
            }
        }
    }

    fn read(port: u16) -> Trap {
        Trap::Exit(VcpuExit::PortIn { port, size: 1, count: 1 })
    }

    fn bus() -> impl Pipe<In = Trap, Out = Reply, Err = RunError> {
        FirstClaim::new(
            PortRange { first: 0x3f8, last: 0x3ff, tag: 1 },
            FirstClaim::new(PortRange { first: 0x2f8, last: 0x2ff, tag: 2 }, PortRange { first: 0x70, last: 0x71, tag: 3 }),
        )
    }

    #[test]
    fn the_device_owning_a_port_range_answers_and_the_others_are_never_asked() {
        for (port, tag) in [(0x3fd, 1), (0x2f8, 2), (0x71, 3)] {
            let reply = block_on(bus().call(read(port))).expect("a device owns the port");

            assert_eq!(reply, Reply::value(Bytes::from(vec![tag])), "port {port:#x}");
        }
    }

    #[test]
    fn a_port_nobody_owns_comes_back_declined_with_the_trap_intact() {
        let error = block_on(bus().call(read(0x80))).expect_err("no device owns port 0x80");

        assert!(matches!(error, RunError::Declined(trap) if trap == read(0x80)));
    }

    #[test]
    fn the_meter_fails_the_run_on_the_exit_past_its_budget_and_counts_only_exits() {
        let ledger = ExitLedger::begin();
        let meter = Meter::new(&ledger, 2, "over budget");

        block_on(meter.call(read(1))).expect("first exit");
        block_on(meter.call(Trap::WatchdogFired)).expect("a watchdog expiry is not an exit");
        block_on(meter.call(Trap::Exit(VcpuExit::Interrupted))).expect("a signal is not an exit");
        block_on(meter.call(read(2))).expect("second exit");
        let error = block_on(meter.call(read(3))).expect_err("third exit is over budget");

        assert_eq!(ledger.exits(), 3);
        assert_eq!(error.to_string(), "over budget");
    }
}
