//! The Linux/x86_64 KVM lane as pipes.
//!
//! The leaf is `kvm-ioctls` (the rust-vmm incumbent) plus the few `libc` calls
//! it does not cover (`mmap`, `memfd_create`, `sigaction`, `pthread_kill`). Above
//! it everything is a pipe:
//!
//! - the vCPU's exits and the watchdog's expiry are sources ([`TrapSource`]),
//!   merged by [`FanIn`](proxima_primitives::pipe::fan_in::FanIn) with
//!   `Select::Fifo`, watchdog first;
//! - each device is a transform from a [`Trap`] to a [`Reply`] over a sans-IO
//!   codec, composed onto a bus by [`FirstClaim`];
//! - the write-back and resume of a reply is a sink ([`Resume`]);
//! - teardown is a sink into a bounded queue ([`TeardownSink`]).
//!
//! One function, [`drive`], steps the composed pipe until a reply ends the run.

pub mod describe;
pub mod devices;
pub mod dirty;
pub mod drive;
pub mod elf_lane;
pub mod error;
pub mod exit_source;
pub mod layered;
pub mod linux_lane;
pub mod memory;
pub mod pipes;
pub mod probe;
pub mod scratch;
pub mod serial;
pub mod session;
pub mod slots;
pub mod trap;
pub mod vcpu;
pub mod watchdog;
pub mod windows;

pub use drive::drive;
pub use elf_lane::{
    Capacities, ElfOutcome, ElfReport, ElfRun, GuestSegment, Transports, run_elf, run_elf_observed,
};
pub use error::{KvmError, KvmResult};
pub use exit_source::TrapSource;
pub use pipes::{FirstClaim, Resume};
pub use session::{TeardownSink, take_teardown_failures};
pub use trap::{Reply, RunError, Trap, VcpuExit};
