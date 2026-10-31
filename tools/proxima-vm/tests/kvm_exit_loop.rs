//! KVM lane sad paths for the exit loop in `src/kvm`, driven through
//! [`proxima_vm::kvm::run_elf`] directly with hand-assembled x86_64 guests —
//! the lambda guest only ever halts cleanly, so every named error the loop
//! can raise needs a guest written to raise it. Each case asserts the error
//! text, never just a failure: a failure cannot tell a guest the loop rejected
//! from one that crashed the host.
//!
//! KVM needs no entitlement, so these run in-process.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::time::{Duration, Instant};

use proxima_core::ProximaError;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::process::{ChildRequest, ChildResponse};
use proxima_vm::kvm::{Capacities, ElfOutcome, ElfRun, GuestSegment, Transports, run_elf};

const GUEST_MEMORY_SIZE: u64 = 2 * 1024 * 1024;

struct NoDispatcher;

impl Pipe for NoDispatcher {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;

    fn call(&self, _request: ChildRequest) -> impl std::future::Future<Output = Result<ChildResponse, ProximaError>> {
        async { Ok(ChildResponse::Close) }
    }
}

fn run_code(code: &[u8], segment_address: u64, max_hypercalls: usize, watchdog_millis: u64) -> ElfOutcome {
    let segment = GuestSegment { guest_address: segment_address, data: code, memory_size: code.len() as u64, writable: false };
    run_elf(ElfRun {
        segments: &[segment],
        memory_size: GUEST_MEMORY_SIZE,
        entry: 0,
        boot_argument: 0,
        dispatcher: &NoDispatcher,
        transports: Transports { console: None, net: None, blk: None },
        max_hypercalls,
        watchdog_millis,
        capacities: Capacities { emitted: 16, console: 0, net: 0, blk: 0 },
    })
}

fn run_hypercall_loop(code: &[u8], segment_address: u64, max_hypercalls: usize) -> ElfOutcome {
    run_code(code, segment_address, max_hypercalls, 0)
}

#[proxima::test]
#[case::hlt_without_the_halt_verb(&[0xf4], 1, "unexpected KVM exit reason 5 at rip=0x1")]
#[case::reading_an_io_port(&[0xec], 1, "guest read an io port")]
#[case::virtio_window_with_no_transport(
    &[0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x01, 0x00, 0x00, 0x00],
    1,
    "mmio exit against a window with no transport"
)]
#[case::unmapped_address_triple_faults_with_no_idt(
    &[0x48, 0xb8, 0x00, 0x00, 0x20, 0x00, 0x10, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x01, 0x00, 0x00, 0x00],
    1,
    "unexpected KVM exit reason 8"
)]
#[case::hypercall_budget_exhausted(
    &[0x66, 0xba, 0xfe, 0xff, 0xee, 0xeb, 0xfd],
    3,
    "guest exceeded hypercall budget without halting"
)]
async fn the_exit_loop_names_the_error_for_a_misbehaving_guest(
    #[case] code: &[u8],
    #[case] max_hypercalls: usize,
    #[case] expected: &str,
) {
    let outcome = run_hypercall_loop(code, 0, max_hypercalls);

    let error = outcome.failure.expect("the loop must fail, not report success");
    assert!(
        error.contains(expected),
        "expected the error to contain {expected:?}, got {error:?}"
    );
}

fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("list /proc/self/fd")
        .count()
}

/// More runs than the teardown queue holds (16) force the inline-release
/// fallback as well as the worker path; either way every KVM, vcpu and memfd
/// descriptor must come back, and no `close` may have failed.
#[proxima::test]
async fn teardown_off_the_return_path_releases_every_descriptor() {
    const HALT: [u8; 5] = [0x66, 0xba, 0xff, 0xff, 0xee];
    const RUNS: usize = 64;
    const SETTLE_LIMIT: Duration = Duration::from_secs(10);
    let baseline = open_fd_count();

    for run in 0..RUNS {
        let outcome = run_hypercall_loop(&HALT, 0, 1);
        assert_eq!(outcome.failure, None, "run {run} must halt cleanly");
    }

    let deadline = Instant::now() + SETTLE_LIMIT;
    while open_fd_count() > baseline && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(
        open_fd_count(),
        baseline,
        "descriptors leaked after {RUNS} runs"
    );
    assert_eq!(proxima_vm::dispatch::take_teardown_failures(), Vec::<String>::new());
}

#[proxima::test]
async fn a_segment_past_the_memory_reservation_is_rejected_before_any_vm_exists() {
    let outcome = run_hypercall_loop(&[0xf4], GUEST_MEMORY_SIZE, 1);

    assert_eq!(outcome.failure.as_deref(), Some("guest segment exceeds guest memory reservation"));
}

/// `jmp $` never traps, so without the watchdog `KVM_RUN` would block
/// forever. The error carries `rip`, the positional evidence that makes a
/// stuck guest debuggable.
#[proxima::test]
async fn a_guest_that_never_traps_is_stopped_by_the_watchdog_and_reports_its_rip() {
    const WATCHDOG_MILLIS: u64 = 300;
    const JUMP_TO_SELF: [u8; 2] = [0xeb, 0xfe];

    let started = Instant::now();
    let outcome = run_code(&JUMP_TO_SELF, 0, 1, WATCHDOG_MILLIS);
    let elapsed = started.elapsed();

    let error = outcome.failure.expect("the watchdog must end the run");
    assert!(
        error.starts_with("watchdog forced exit") && error.contains("rip=0x0"),
        "the watchdog error must name itself and the stuck rip, got {error:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(WATCHDOG_MILLIS),
        "the run ended in {elapsed:?}, before the {WATCHDOG_MILLIS} ms watchdog window"
    );
}
