#![cfg(target_os = "linux")]

use std::fs;
use std::mem::MaybeUninit;
use std::os::fd::RawFd;

const CHILD_EXIT_STATUS: i32 = 0;
const EPOLL_EVENT_TOKEN: u64 = 0x4348_494c_4450_4944;
const RELEASE_BYTE: u8 = b'R';

#[test]
fn linux_exit_probe() {
    let threads_before = process_thread_count();
    let mut release_pipe = [-1; 2];
    // SAFETY: the array has room for both descriptors returned by pipe.
    assert_eq!(unsafe { libc::pipe(release_pipe.as_mut_ptr()) }, 0);

    // SAFETY: the child performs only close, read, and _exit after fork.
    let child_pid = unsafe { libc::fork() };
    assert!(child_pid >= 0, "fork held child fixture");
    if child_pid == 0 {
        unsafe {
            libc::close(release_pipe[1]);
            let mut release_byte = 0_u8;
            let read_count = libc::read(
                release_pipe[0],
                (&mut release_byte as *mut u8).cast(),
                core::mem::size_of::<u8>(),
            );
            let exit_status = if read_count == 1 && release_byte == RELEASE_BYTE {
                CHILD_EXIT_STATUS
            } else {
                17
            };
            libc::_exit(exit_status);
        }
    }

    // SAFETY: only the parent closes the child's pipe read end.
    assert_eq!(unsafe { libc::close(release_pipe[0]) }, 0);
    let mut held_child = HeldChild::new(child_pid, release_pipe[1]);
    let child_pid_descriptor = match open_pidfd(child_pid) {
        Ok(descriptor) => descriptor,
        Err(error_number) if unsupported_pidfd_error(error_number) => {
            held_child.release_and_reap();
            println!(
                "probe=linux_exit_probe record=unsupported-source source=pidfd errno={error_number}"
            );
            return;
        }
        Err(error_number) => {
            held_child.release_and_reap();
            panic!("pidfd_open failed with errno {error_number}");
        }
    };

    // SAFETY: epoll_create1 returns an owned descriptor or -1.
    let epoll_descriptor = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    if epoll_descriptor < 0 {
        let error_number = std::io::Error::last_os_error()
            .raw_os_error()
            .expect("epoll_create1 error has OS code");
        // SAFETY: child_pid_descriptor is open and owned by this test.
        unsafe { libc::close(child_pid_descriptor) };
        held_child.release_and_reap();
        if unsupported_pidfd_error(error_number) {
            println!(
                "probe=linux_exit_probe record=unsupported-source source=epoll errno={error_number}"
            );
            return;
        }
        panic!("epoll_create1 failed with errno {error_number}");
    }
    let mut registration = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLET) as u32,
        u64: EPOLL_EVENT_TOKEN,
    };
    // SAFETY: both descriptors are open and registration is initialized.
    let registration_result = unsafe {
        libc::epoll_ctl(
            epoll_descriptor,
            libc::EPOLL_CTL_ADD,
            child_pid_descriptor,
            &mut registration,
        )
    };
    if registration_result != 0 {
        let error_number = std::io::Error::last_os_error()
            .raw_os_error()
            .expect("epoll_ctl error has OS code");
        unsafe {
            libc::close(child_pid_descriptor);
            libc::close(epoll_descriptor);
        }
        held_child.release_and_reap();
        if unsupported_pidfd_error(error_number) {
            println!(
                "probe=linux_exit_probe record=unsupported-source source=pidfd_epoll errno={error_number}"
            );
            return;
        }
        panic!("epoll_ctl pidfd registration failed with errno {error_number}");
    }

    let mut event = MaybeUninit::<libc::epoll_event>::uninit();
    let before_release_events = epoll_wait(epoll_descriptor, event.as_mut_ptr(), 0);
    if before_release_events == 0 {
        println!("probe=linux_exit_probe record=pending pending=1 events_before_release=0");
    }

    held_child.release();

    let wake_count = epoll_wait(epoll_descriptor, event.as_mut_ptr(), 5_000);
    assert_eq!(wake_count, 1, "pidfd produces one epoll wake");
    // SAFETY: epoll_wait filled one event.
    let wake_event = unsafe { event.assume_init() };
    let wake_token = wake_event.u64;
    let wake_flags = wake_event.events;
    assert_eq!(wake_token, EPOLL_EVENT_TOKEN, "kernel wake matches pidfd");
    assert_ne!(wake_flags & libc::EPOLLIN as u32, 0, "pidfd is readable");
    println!("probe=linux_exit_probe record=kernel_wake wakes=1 source=pidfd+epoll");

    let wait_status = held_child.reap();
    assert!(libc::WIFEXITED(wait_status), "child exits normally");
    assert_eq!(libc::WEXITSTATUS(wait_status), CHILD_EXIT_STATUS);
    println!("probe=linux_exit_probe record=exit observations=1 status=0 reaped=1");

    assert_eq!(before_release_events, 0, "held child has no exit event");

    // Closing the last pidfd removes the source from epoll before another turn.
    // SAFETY: child_pid_descriptor is the only open pidfd and epoll remains open.
    assert_eq!(unsafe { libc::close(child_pid_descriptor) }, 0);
    assert_eq!(
        epoll_wait(epoll_descriptor, event.as_mut_ptr(), 0),
        0,
        "closed pidfd source does not wake twice"
    );
    // SAFETY: epoll_descriptor remains owned by this test.
    assert_eq!(unsafe { libc::close(epoll_descriptor) }, 0);

    let threads_after = process_thread_count();
    assert_eq!(
        threads_after, threads_before,
        "probe creates no helper threads"
    );
    println!("probe=linux_exit_probe record=helper_threads count=0");
}

fn open_pidfd(child_pid: libc::pid_t) -> Result<RawFd, i32> {
    // SAFETY: pidfd_open accepts a child pid and flags; success returns a new fd.
    let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, child_pid, 0_u32) as RawFd };
    if descriptor >= 0 {
        Ok(descriptor)
    } else {
        Err(std::io::Error::last_os_error()
            .raw_os_error()
            .expect("pidfd_open error has OS code"))
    }
}

fn unsupported_pidfd_error(error_number: i32) -> bool {
    matches!(error_number, libc::ENOSYS | libc::EINVAL | libc::EPERM)
}

struct HeldChild {
    pid: libc::pid_t,
    release_descriptor: RawFd,
    reaped: bool,
}

impl HeldChild {
    fn new(pid: libc::pid_t, release_descriptor: RawFd) -> Self {
        Self {
            pid,
            release_descriptor,
            reaped: false,
        }
    }

    fn release(&mut self) {
        if self.release_descriptor < 0 {
            return;
        }
        // SAFETY: one byte releases the direct child blocked on this pipe.
        let written = unsafe {
            libc::write(
                self.release_descriptor,
                (&RELEASE_BYTE as *const u8).cast(),
                core::mem::size_of::<u8>(),
            )
        };
        // SAFETY: this object owns the parent's pipe write descriptor.
        let closed = unsafe { libc::close(self.release_descriptor) };
        self.release_descriptor = -1;
        assert_eq!(written, 1, "release held child");
        assert_eq!(closed, 0, "close release pipe");
    }

    fn reap(&mut self) -> i32 {
        let mut wait_status = 0;
        loop {
            // SAFETY: pid is the unreaped direct child of this process.
            let result = unsafe { libc::waitpid(self.pid, &mut wait_status, 0) };
            if result >= 0 {
                assert_eq!(result, self.pid);
                self.reaped = true;
                return wait_status;
            }
            let error_number = std::io::Error::last_os_error()
                .raw_os_error()
                .expect("waitpid error has OS code");
            assert_eq!(
                error_number,
                libc::EINTR,
                "waitpid only retries interruption"
            );
        }
    }

    fn release_and_reap(&mut self) {
        self.release();
        let wait_status = self.reap();
        assert!(libc::WIFEXITED(wait_status), "child exits normally");
        assert_eq!(libc::WEXITSTATUS(wait_status), CHILD_EXIT_STATUS);
    }
}

impl Drop for HeldChild {
    fn drop(&mut self) {
        self.release();
        if !self.reaped {
            let mut wait_status = 0;
            loop {
                // SAFETY: pid is this process's direct child and has not been reaped.
                let result = unsafe { libc::waitpid(self.pid, &mut wait_status, 0) };
                if result >= 0
                    || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR)
                {
                    self.reaped = true;
                    break;
                }
            }
        }
    }
}

fn epoll_wait(epoll_descriptor: RawFd, event: *mut libc::epoll_event, timeout_ms: i32) -> i32 {
    loop {
        // SAFETY: event points to one writable epoll_event slot.
        let result = unsafe { libc::epoll_wait(epoll_descriptor, event, 1, timeout_ms) };
        if result >= 0 {
            return result;
        }
        let error_number = std::io::Error::last_os_error()
            .raw_os_error()
            .expect("epoll_wait error has OS code");
        assert_eq!(
            error_number,
            libc::EINTR,
            "epoll_wait only retries interruption"
        );
    }
}

fn process_thread_count() -> usize {
    fs::read_dir("/proc/self/task")
        .expect("read Linux process thread directory")
        .count()
}
