# Q1 native Linux child exit wake evidence

The probe source is prime/tests/child_reactor.rs. This run was performed on
native Linux through SSH to the versailles host. It is not Wine or a
cross-compiled run.

## Execution record

- Host: versailles
- Kernel: Linux 7.2.6-zen2-1-zen
- Acceptance command: cargo test -p prime --test child_reactor -- linux_exit_probe
- Captured command: cargo test -p prime --test child_reactor linux_exit_probe -- --nocapture
- Process exit code: 0
- Test result: 1 passed; 0 failed

Exact captured output from the native Linux --nocapture run:

    host=versailles
    kernel=Linux 7.2.6-zen2-1-zen
       Compiling prime v0.1.0 (/tmp/proxima-cargo-val-q1/prime)
        Finished test profile [unoptimized + debuginfo] target(s) in 0.26s
         Running tests/child_reactor.rs (/tmp/cargo-val-q1-target/debug/deps/child_reactor-289746a55be50080)

    running 1 test
    probe=linux_exit_probe record=pending pending=1 events_before_release=0
    probe=linux_exit_probe record=kernel_wake wakes=1 source=pidfd+epoll
    probe=linux_exit_probe record=exit observations=1 status=0 reaped=1
    probe=linux_exit_probe record=helper_threads count=0
    test linux_exit_probe ... ok

    test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

    exit_code=0

## Source pointers

- Held child waits on its release pipe: prime/tests/child_reactor.rs:11-37.
- pidfd open syscall and descriptor result: prime/tests/child_reactor.rs:149-159.
- pidfd registered in epoll: prime/tests/child_reactor.rs:74-86.
- Pending observation occurs before release: prime/tests/child_reactor.rs:105-111.
- One kernel wake is checked against the registered pidfd token: prime/tests/child_reactor.rs:113-121.
- Child is reaped with waitpid; exit status and record are checked: prime/tests/child_reactor.rs:123-126, with the wait loop at 199-218.
- Closing the pidfd is followed by a zero-event poll: prime/tests/child_reactor.rs:130-137.
- Helper-thread count is compared before and after the probe: prime/tests/child_reactor.rs:141-146; the count reads /proc/self/task at 265-269.
- Explicit unsupported-source outputs cover pidfd open, epoll creation, and pidfd registration: prime/tests/child_reactor.rs:42-55, 59-72, and 87-103; classified errno values are at 161-163.
