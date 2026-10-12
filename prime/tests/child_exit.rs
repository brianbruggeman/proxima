#![cfg(unix)]

use std::future::poll_fn;
#[cfg(target_os = "macos")]
use std::io;
use std::num::NonZeroI32;
use std::os::fd::{AsRawFd, RawFd};
#[cfg(target_os = "macos")]
use std::os::unix::net::UnixStream;
#[cfg(target_os = "linux")]
use std::os::unix::process::ExitStatusExt;
#[cfg(target_os = "macos")]
use std::process::ExitStatus;
use std::task::Poll;

use prime::os::child;
use prime::os::core_shard::run_on_this_thread;
use prime::os::readiness::{Readiness, ReadyState};
use proxima_runtime::CoreId;

const RELEASE_BYTE: u8 = b'R';
const HELD_CHILD_CODE: i32 = 0;
#[cfg(target_os = "macos")]
const FAST_CHILD_CODE: i32 = 7;
#[cfg(target_os = "macos")]
const STEP_BOUND: u32 = 8;

struct HeldChild {
    pid: NonZeroI32,
    release_write: RawFd,
}

fn fork_child(exit_code: i32, wait_for_release: bool) -> HeldChild {
    let mut release_pipe = [-1; 2];
    // SAFETY: the array has room for both descriptors returned by pipe.
    assert_eq!(unsafe { libc::pipe(release_pipe.as_mut_ptr()) }, 0, "pipe for release");
    // SAFETY: the child performs only close, read, and _exit after fork.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork child fixture");
    if pid == 0 {
        unsafe {
            libc::close(release_pipe[1]);
            let mut byte = 0_u8;
            if wait_for_release {
                libc::read(release_pipe[0], (&raw mut byte).cast(), 1);
            }
            libc::_exit(exit_code);
        }
    }
    // SAFETY: only the parent closes its copy of the read end.
    assert_eq!(unsafe { libc::close(release_pipe[0]) }, 0, "close release read end");
    let Some(pid) = NonZeroI32::new(pid) else {
        panic!("fork returned a non-positive pid {pid}");
    };
    HeldChild {
        pid,
        release_write: release_pipe[1],
    }
}

impl HeldChild {
    fn release(&self) {
        let byte = RELEASE_BYTE;
        // SAFETY: release_write is open and owned by this fixture.
        let written = unsafe { libc::write(self.release_write, (&raw const byte).cast(), 1) };
        assert_eq!(written, 1, "write release byte");
    }
}

impl Drop for HeldChild {
    fn drop(&mut self) {
        // SAFETY: release_write is open and owned by this fixture.
        unsafe { libc::close(self.release_write) };
    }
}

#[cfg(target_os = "macos")]
fn thread_count() -> usize {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: info is a proc_taskinfo of the size passed.
    let written = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            (&raw mut info).cast(),
            size,
        )
    };
    assert_eq!(written, size, "proc_pidinfo returns a full proc_taskinfo");
    info.pti_threadnum as usize
}

#[cfg(target_os = "linux")]
fn thread_count() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .expect("read /proc/self/status")
        .lines()
        .find_map(|line| line.strip_prefix("Threads:"))
        .expect("Threads line present")
        .trim()
        .parse()
        .expect("thread count parses")
}

fn watch_exit_without_reaping(pid: NonZeroI32) {
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: info outlives the call; WNOWAIT leaves the zombie for the test to reap.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid.get() as libc::id_t,
            &raw mut info,
            libc::WEXITED | libc::WNOWAIT,
        )
    };
    assert_eq!(result, 0, "waitid WNOWAIT sees the child exit");
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    pendings: u32,
    wakes: u32,
    exits: u32,
    reap_attempts: u32,
    esrch: u32,
    steps: u32,
}

#[cfg(target_os = "macos")]
fn make_gate(pid: NonZeroI32) -> Readiness {
    Readiness::process_exit(pid)
}

#[cfg(target_os = "macos")]
fn reap_when_gate_opened(pid: NonZeroI32) -> io::Result<Option<ExitStatus>> {
    child::reap_reported(pid).map(Some)
}

#[cfg(target_os = "macos")]
fn reap_before_first_park(pid: NonZeroI32) -> io::Result<Option<ExitStatus>> {
    child::try_reap(pid)
}

#[cfg(target_os = "macos")]
// polls the exit gate once per call; parked-then-retry means the kernel reported the exit.
fn exit_step(
    gate: &mut Readiness,
    pid: NonZeroI32,
    counts: &mut Counts,
    context: &std::task::Context<'_>,
) -> Poll<io::Result<ExitStatus>> {
    counts.steps += 1;
    if counts.pendings > 0 && counts.wakes < counts.pendings {
        counts.wakes += 1;
    }
    loop {
        let fired = counts.pendings > 0;
        match gate.poll(context) {
            Ok(ReadyState::Retry) => {
                counts.reap_attempts += 1;
                let reaped = if fired {
                    reap_when_gate_opened(pid)
                } else {
                    reap_before_first_park(pid)
                };
                match reaped {
                    Ok(Some(status)) => {
                        counts.exits += 1;
                        return Poll::Ready(Ok(status));
                    }
                    Ok(None) => continue,
                    Err(error) => return Poll::Ready(Err(error)),
                }
            }
            Ok(ReadyState::Parked) => {
                counts.pendings += 1;
                return Poll::Pending;
            }
            Ok(ReadyState::OffWorker) => {
                return Poll::Ready(Err(io::Error::other("exit gate polled off a worker")));
            }
            Err(error) if error.raw_os_error() == Some(libc::ESRCH) => {
                counts.esrch += 1;
                counts.reap_attempts += 1;
                counts.exits += 1;
                return Poll::Ready(child::reap_reported(pid));
            }
            Err(error) => return Poll::Ready(Err(error)),
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macos_exit_probe() {
    let held = fork_child(HELD_CHILD_CODE, true);
    let pid = held.pid;

    let (status, counts, threads_before, threads_after) = run_on_this_thread(CoreId(0), async move {
        let threads_before = thread_count();
        let mut gate = make_gate(pid);
        let mut counts = Counts::default();
        let status = poll_fn(|context| {
            let outcome = exit_step(&mut gate, pid, &mut counts, context);
            if counts.pendings == 1 && counts.steps == 1 {
                held.release();
            }
            outcome
        })
        .await
        .expect("exit gate resolves to a status");
        (status, counts, threads_before, thread_count())
    })
    .expect("root completes");

    println!(
        "probe=macos_exit_probe pending={} wakes={} exits={} helper_threads={}",
        counts.pendings,
        counts.wakes,
        counts.exits,
        threads_after.abs_diff(threads_before)
    );
    assert_eq!(status.code(), Some(HELD_CHILD_CODE), "held child exits cleanly");
    assert_eq!((counts.pendings, counts.wakes, counts.exits), (1, 1, 1));
    assert_eq!(threads_after, threads_before, "no helper thread is spawned");
}

#[cfg(target_os = "macos")]
#[test]
fn macos_esrch_reaps_through_reap_reported() {
    let finished = fork_child(FAST_CHILD_CODE, false);
    let pid = finished.pid;
    watch_exit_without_reaping(pid);

    let (status, counts) = run_on_this_thread(CoreId(0), async move {
        let mut gate = make_gate(pid);
        let mut counts = Counts::default();
        let status = poll_fn(|context| exit_step(&mut gate, pid, &mut counts, context))
            .await
            .expect("zombie child resolves to a status");
        (status, counts)
    })
    .expect("root completes");

    println!(
        "probe=macos_esrch_reaps esrch={} reaped_code={} reap_attempts={}",
        counts.esrch,
        status.code().expect("normal exit has a code"),
        counts.reap_attempts
    );
    assert_eq!(status.code(), Some(FAST_CHILD_CODE));
    assert!(counts.reap_attempts > 0, "the exit was reaped, not skipped");
    assert_eq!(counts.esrch, 1, "registration on a zombie reports ESRCH");
}

// a source that stays readable the whole run keeps the reactor from ever idling;
// the exit must still be delivered in a bounded number of steps.
#[cfg(target_os = "macos")]
#[test]
fn macos_exit_delivered_while_another_source_stays_readable() {
    use std::io::Write;

    let finished = fork_child(FAST_CHILD_CODE, false);
    let pid = finished.pid;
    watch_exit_without_reaping(pid);
    let (mut writer, reader) = UnixStream::pair().expect("socketpair");
    writer.write_all(b"never drained").expect("make reader readable");

    let (status, counts, busy_polls) = run_on_this_thread(CoreId(0), async move {
        let mut exit_gate = make_gate(pid);
        let mut busy_gate = Readiness::new(reader.as_raw_fd());
        let mut counts = Counts::default();
        let mut busy_polls = 0_u32;
        let status = poll_fn(|context| {
            busy_polls += 1;
            let _ = busy_gate.poll(context).expect("busy source arms");
            context.waker().wake_by_ref();
            exit_step(&mut exit_gate, pid, &mut counts, context)
        })
        .await
        .expect("exit resolves");
        drop(reader);
        (status, counts, busy_polls)
    })
    .expect("root completes");

    println!(
        "probe=macos_exit_with_readable_source steps={} reap_attempts={} busy_polls={busy_polls}",
        counts.steps, counts.reap_attempts
    );
    assert_eq!(status.code(), Some(FAST_CHILD_CODE));
    assert!(counts.reap_attempts > 0, "reap was attempted");
    assert!(counts.steps <= STEP_BOUND, "exit delivered within {STEP_BOUND} steps");
}

#[cfg(target_os = "linux")]
fn make_gate(pid: NonZeroI32) -> (Readiness, std::os::fd::OwnedFd) {
    let pidfd = child::open_pidfd(pid).expect("pidfd_open on a live child");
    (Readiness::new(pidfd.as_raw_fd()), pidfd)
}

#[cfg(target_os = "linux")]
#[test]
fn linux_pidfd_exit_probe() {
    use std::os::fd::AsFd;

    let held = fork_child(HELD_CHILD_CODE, true);
    let pid = held.pid;

    let (status, counts, threads_before, threads_after) = run_on_this_thread(CoreId(0), async move {
        let threads_before = thread_count();
        let (mut gate, pidfd) = make_gate(pid);
        let mut counts = Counts::default();
        let status = poll_fn(|context| {
            counts.steps += 1;
            if counts.pendings > 0 && counts.wakes < counts.pendings {
                counts.wakes += 1;
            }
            loop {
                match gate.poll(context) {
                    Ok(ReadyState::Retry) => {
                        counts.reap_attempts += 1;
                        match child::try_reap_pidfd(pidfd.as_fd()) {
                            Ok(Some(status)) => {
                                counts.exits += 1;
                                return Poll::Ready(status);
                            }
                            Ok(None) => continue,
                            Err(error) => panic!("try_reap_pidfd failed: {error}"),
                        }
                    }
                    Ok(ReadyState::Parked) => {
                        counts.pendings += 1;
                        if counts.pendings == 1 {
                            held.release();
                        }
                        return Poll::Pending;
                    }
                    Ok(ReadyState::OffWorker) => panic!("polled off a worker"),
                    Err(error) => panic!("pidfd gate failed: {error}"),
                }
            }
        })
        .await;
        (status, counts, threads_before, thread_count())
    })
    .expect("root completes");

    println!(
        "probe=linux_pidfd_exit_probe pending={} wakes={} exits={} helper_threads={}",
        counts.pendings,
        counts.wakes,
        counts.exits,
        threads_after.abs_diff(threads_before)
    );
    assert_eq!(status.code(), Some(HELD_CHILD_CODE));
    assert_eq!((counts.pendings, counts.wakes, counts.exits), (1, 1, 1));
    assert_eq!(threads_after, threads_before);
}

#[cfg(target_os = "linux")]
#[test]
fn linux_pidfd_reports_signal_death() {
    use std::os::fd::AsFd;

    let held = fork_child(HELD_CHILD_CODE, true);
    let pid = held.pid;
    let pidfd = child::open_pidfd(pid).expect("pidfd_open");
    // SAFETY: pid is our live child.
    assert_eq!(unsafe { libc::kill(pid.get(), libc::SIGKILL) }, 0, "kill child");
    watch_exit_without_reaping(pid);

    let status = child::try_reap_pidfd(pidfd.as_fd())
        .expect("waitid on pidfd")
        .expect("child has exited");

    assert_eq!(status.signal(), Some(libc::SIGKILL));
}
