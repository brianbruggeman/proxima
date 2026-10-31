use core::future::Future;
use core::task::Poll;

use proxima_core::markers::DropSafe;
use proxima_primitives::pipe::UnpinPipe;
use proxima_primitives::pipe::fan_in::Exhausted;

use super::memory::GuestRam;
use super::trap::{HypercallArguments, Trap, VcpuExit};
use super::vcpu::Vcpu;
use super::watchdog::Watchdog;

/// One scan slot of the trap merge. `FanIn` merges `[S; N]` of a single `S`,
/// so the two sources of a run, the vCPU's exit stream and the watchdog's
/// expiry, are the two arms of one type.
pub enum TrapSource<'run> {
    Watchdog(Option<&'run Watchdog>),
    Exit { vcpu: &'run Vcpu, hypercall_memory: Option<GuestRam> },
}

impl DropSafe for TrapSource<'_> {}

fn capture_hypercall(vcpu: &Vcpu, ram: GuestRam, exit: VcpuExit) -> VcpuExit {
    let VcpuExit::PortOut { port, size, count, data, .. } = exit else {
        return exit;
    };
    match vcpu.registers() {
        Ok(registers) => VcpuExit::PortOut {
            port,
            size,
            count,
            data,
            arguments: Some(HypercallArguments { pointer: registers.rdi, length: registers.rsi }),
            payload: ram
                .read(registers.rdi, registers.rsi.max(1))
                .or_else(|| ram.read(registers.rdi, registers.rsi)),
        },
        Err(error) => VcpuExit::RunFailed { errno: error.errno() },
    }
}

impl TrapSource<'_> {
    fn next_trap(&self) -> Option<Trap> {
        match self {
            Self::Watchdog(watchdog) => watchdog.filter(|armed| armed.fired()).map(|_| Trap::WatchdogFired),
            Self::Exit { vcpu, hypercall_memory } => {
                let exit = vcpu.run();
                Some(Trap::Exit(match hypercall_memory {
                    Some(ram) => capture_hypercall(vcpu, *ram, exit),
                    None => exit,
                }))
            }
        }
    }
}

impl UnpinPipe for TrapSource<'_> {
    type In = ();
    type Out = Trap;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Trap, Exhausted>> + Unpin {
        let mut ready = self.next_trap();
        core::future::poll_fn(move |_context| match ready.take() {
            Some(trap) => Poll::Ready(Ok(trap)),
            None => Poll::Pending,
        })
    }
}
