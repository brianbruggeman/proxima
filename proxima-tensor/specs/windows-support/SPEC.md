# windows-support

status: admitted
owner: brian
created: 2026-10-02

execution: source-v3 native runtime, examples, CRT observations, and a native
full-workspace compile are recorded in EVIDENCE.md. The direct-command workflow
has not yet run in CI.

### native C runtime selection refinement

The native root debug build emitted LNK4098. Inspection of zlibstatic-ngd.lib
found DEFAULTLIB:MSVCRTD; its CMake cache selected Debug without an explicit
MSVC runtime property. Preserve debug symbols and optimization settings while
selecting the same C runtime as Rust: MultiThreadedDLL normally, MultiThreaded
when Cargo's target features include crt-static. Bind this through a repo-owned
CMake toolchain and the x86_64-pc-windows-msvc target-specific Cargo environment
key. Compose any caller-supplied host/target/generic toolchain in cmake-rs
precedence order; never silently replace its compiler or sysroot. An explicitly
provided target-specific toolchain remains authoritative. Do not suppress linker
warnings or change compression backends.

Validation: inspect native zlib archive directives and generated compiler/runtime
settings before and after; expected debug-CRT directives/imports 0 after the
change, runtime-conflict warnings 0, native selected test executions 48 with
failed/ignored/missing 0. Re-run full MSVC cross compilation (missing/excluded
members 0, compiler errors 0, successful build-finished records 1), native
workspace compilation, and examples. The existing recorded host and native CMake
policy checks cover normal/static selection and supplied-toolchain composition;
the current native build also verifies the selected runtime through generated
settings, archive directives, and executable imports. Static selection does not
claim a complete static-CRT product build.

## problem

On `x86_64-pc-windows-msvc`, Proxima must compile every current workspace member with default features and expose its default prime runtime with functioning TCP and UDP IO, measured by compiler records and named native loopback test assertions against the same payloads on Windows and the host.

## refutation condition

The port is refuted if the default Windows build omits a portable workspace member or TCP/UDP facade, if a pending socket cannot resume after real readiness, if received bytes differ from sent bytes, or if the gate accepts a zero-test or deliberately failing run. Cross-compilation alone cannot establish runtime behavior.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Compile every workspace member with defaults for x86_64-pc-windows-msvc, including all Cargo targets and the root default feature closure. Preserve the member set from Cargo metadata; no package exclusions to make the gate pass. | yes |
| R2 | Implement Windows reactor registration, readiness rearming, external wake, deregistration, and stale-generation rejection while preserving worker ownership and source lifetime rules. Real OS errors propagate; successful no-op IO is forbidden. | yes |
| R3 | Default prime TCP bind/connect/accept/read/write/EOF and UDP send/receive operate on Windows through the existing futures-IO and datagram surfaces. Portable facade exports stay present. | yes |
| R4 | Keep intrinsically Unix APIs (POSIX descriptors, signals, fork, PTY, interposition) at narrow Unix cfg boundaries; preserve portable logic in those crates. Inventory each such boundary with rationale and source path; no blanket removal of portable functionality. | yes |
| R5 | Add a native Windows CI job that runs direct Cargo workspace compilation, tests, doctests, and the runnable example. Preserve the command output in the CI job log. | yes |
| R6 | Run the same focused network/runtime contract on the host and retain platform-specific build regressions, doctests, and runnable examples as separate evidence. | yes |

## architecture

Keep prime's executor and `SourceKey` generation lifecycle shared. Select the OS reactor backend below `prime::os::reactor`; Windows uses owned native socket handles and real OS readiness/wake operations, never POSIX integer descriptors or a busy-loop substitute. Preserve `Reactor::register`, `reregister`, `deregister`, waker registration, and `turn` semantics so TCP/UDP and runtime callers need only platform handle extraction at the IO boundary. Runtime pending transitions register the relevant waker before returning `Pending`; subsequent readiness resumes the task; dropping a source invalidates its generation and unregisters its OS state before socket destruction. Retained wake handles must not reference a destroyed OS object.

Windows TCP/UDP retain `socket2` socket creation and nonblocking IO. Keep Unix-domain socket types and descriptor-specific syscalls behind `cfg(unix)`; keep common network code and facade exports available on Windows. Process/interposition build scripts must skip only unsupported Unix artifacts on Windows; they must not fabricate an artifact path or report a successful unsupported operation. Other compile failures discovered under full-workspace checking receive the same narrow treatment.

The native Windows workflow runs Cargo directly. Cross-target checking is a separate Cargo xwin command; it does not satisfy native execution. Cargo and nextest command output remains in the CI job log, and native Windows, host, Wine, and cross-target observations retain separate environment labels in EVIDENCE.md.

The executable example uses independent `std::net` peers and the literal request/datagram byte vectors in the fixed test contract. Its shared payload checker also runs against a standard-library-only TCP/UDP exchange before the prime exchange, expecting both incumbent payloads to pass. The example reports `incumbent_payload_matches=2`, `tcp_payload_matches=1`, and `udp_payload_matches=1`; the gate requires all three counts. Thus an incorrect or vacuous checker cannot be accepted solely through agreement between prime endpoints.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| initial target | x86_64-pc-windows-msvc | binds the first Windows ABI; GNU/ARM64 require additional validation |
| portable surface | full default workspace and root default runtime | excluding prime/network or whole portable crates would compile a different product |
| reactor seam | native backend behind existing reactor operations | duplicating the executor or substituting Tokio would change the default runtime contract |
| feature boundaries | target dependencies and narrow cfgs | a successful stub or blanket module exclusion conceals missing behavior |
| test oracle | byte assertions plus forced Pending/wake transitions | an immediately ready socket roundtrip alone does not test the reactor |
| evidence | native execution separate from cross compilation | a cross compiler cannot observe Windows socket or wake behavior |

### fixed test contract

These are specified acceptance fixtures, not observed test counts. Add `prime/tests/windows_port.rs` with exactly these eight tests, compiled on both Windows and the host with `--features runtime-prime-full,runtime-prime-virtual-clock,runtime-prime-bgpool`:

1. `windows_port_tcp_loopback_payload`: bind ephemeral IPv4 loopback; connect and accept through prime; exchange `GET /windows-port HTTP/1.1\r\nHost: localhost\r\n\r\n`; assert exact request and response bytes and peer addresses.
2. `windows_port_udp_loopback_payload`: send and receive `proxima windows udp probe` between ephemeral loopback sockets; assert bytes, byte count, and source address.
3. `windows_port_tcp_pending_read_wakes`: force and assert an initial read `Pending` before peer write using explicit channel/barrier coordination; observe wake and exact bytes after readiness.
4. `windows_port_udp_pending_receive_wakes`: force and assert initial receive `Pending`; send only after the registration handshake; observe wake and payload.
5. `windows_port_tcp_peer_close_reports_eof`: complete a connection, close the peer, and assert read result is zero bytes rather than Pending forever.
6. `windows_port_tcp_connect_refused_reports_error`: connect to a local unlistened endpoint held bound for the test lifetime; require an error, never successful readiness alone. Run an independent standard-library connection against the same endpoint in parallel and compare error kinds; a bound non-listening endpoint can time out instead of refusing on macOS.
7. `windows_port_reactor_external_wake`: arm wake and coordinate an external producer firing it; require reactor return and task progress, including deterministic virtual-clock advancement that expires a registered timer, and a retained wake handle used after reactor drop without a stale-handle access.
8. `windows_port_reactor_deregister_rejects_stale_generation`: unregister and reuse a slot; old key cannot wake or mutate its replacement; repeated registration/readiness exercises rearm without losing a pending reader. Transfer Readiness between workers whose local keys collide; reject polling on the wrong owner and preserve that owner's registration on foreign drop.

The EOF fixture also transfers a registered Prime stream across threads, requires
foreign polling to return an error, and observes peer EOF after foreign drop both
before and after the owning worker shuts down. Retained wake handles are exercised
after reactor destruction on every supported host.

The root TCP fixture additionally builds an App with a two-worker Prime runtime,
loads a configured HTTP listener at an ephemeral address, and uses an independent
standard-library client to assert status 200 and the exact body
`proxima windows two workers`. This covers the real configuration-to-listener
path: direct factory tests alone do not exercise per-worker binding topology.
Windows lacks Unix SO_REUSEPORT semantics; listener ownership must be explicit,
while HTTP handlers can use the existing spread-to-peers dispatch path.

Add root `tests/windows_port.rs` with exactly two tests built with the root defaults: `windows_port_default_runtime_tcp_payload` and `windows_port_default_runtime_udp_payload`. Both fixture files use independent `std::net` peers as the byte oracle; identical defects in two prime endpoints must not satisfy the assertion. They exercise public root runtime/network exports and assert the corresponding payloads above, so removing exports or silently changing the default executor cannot satisfy the tests. Use owned buffers, deterministic coordination, bounded runner timeouts, and no test sleeps. Unit-level helpers may live beside their implementation; the integration tests cover module composition.

The native workflow runs the named Cargo nextest commands in
`.github/workflows/windows.yml` on Windows, each with `--no-tests=fail` and
retries disabled. Host executions use the same named inventory and are labeled
with the actual host target. Wine remains a separate environment and is not
native Windows evidence.


### execution order and filesystem contract

The owner's 2026-10-02 direction orders implementation and measurement as Prime first,
including filesystem/OS handles, then proxima-net. Broader workspace edits are paused
until those layers have build and execution records. Prime currently offloads file IO
through `ProximaBackgroundPool::spawn`; do not introduce a redundant filesystem facade.
Windows `OwnedHandle` and socket `RawSocket` remain distinct. Native Windows `FileExt`
seek operations advance the file cursor; Unix positional operations preserve it.

Prime's `windows_fs` fixture has four named tests: `windows_fs_background_round_trip_unicode_path`,
`windows_fs_owned_handle_rename_and_remove`, `windows_fs_os_file_extensions_preserve_platform_cursor_contract`,
and `windows_fs_errors_survive_background_completion`. The oracle is the literal
`b"proxima windows file io\n"` file payload, independently read with std::fs; native
FileExt behavior follows Rust's platform API contract. The fixture must assert a
Pending background future before releasing its worker, then exact bytes, handle closure,
rename/delete and native error kinds. All files are created beneath tempfile roots.

Supporting R4/R6 command: `cargo nextest run -p prime --features runtime-prime-full,runtime-prime-virtual-clock,runtime-prime-bgpool,runtime-prime-config --test windows_fs`.
Expected: exactly the four named tests executed, four passed, zero skipped, zero failed.
Record host and Windows-binary execution separately. Wine is labeled as Wine, never native Windows.
The runtime and host gates execute this filesystem fixture before the Prime networking
fixture and retain a separate named result record for each binary. All fixtures remain
in the existing crates; the worktree directory name is not a new Cargo package.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo xwin check --cross-compiler clang-cl --workspace --all-targets --target x86_64-pc-windows-msvc --message-format json` | Cargo reports all 47 workspace members, 0 compiler errors, and a successful build-finished record; no package exclusions |
| AC2 | R2, R3 | run the named `cargo nextest` commands in `.github/workflows/windows.yml` on native Windows | the 48 named tests execute; nextest reports zero tests per missing filter, failures, and skipped tests; retain the counts and native environment label in EVIDENCE.md |
| AC3 | R4 | inspect `ai_docs/windows-surface.jsonl` and follow each listed source path | every boundary is classified supported or Unix-only with a reason; root defaults, Prime reactor, TCP, and UDP are supported entries |
| AC4 | R5 | inspect `.github/workflows/windows.yml` | one native Windows job is triggered on pull requests and runs direct Cargo compile, test, doctest, and example commands without a script runner |
| AC5 | R6 | run the same named `cargo nextest` commands from `.github/workflows/windows.yml` on the host | the 38 named host tests execute; nextest reports zero tests per missing filter, failures, and skipped tests; evidence labels the host triple |
| AC6 | R6 | `cargo test -p proxima --doc` and `cargo run -p proxima --example windows_port --features http1-native` | doctests report passed/failed/ignored counts; the example reports incumbent/TCP/UDP/multiworker payload matches |

An AC result is recorded only after its command runs. Installing a toolchain, writing CI, compiling tests, or capturing a cross compile does not fulfill native execution. No fallback from native test to host test is allowed under the same evidence label.

The current host record contains 38 named executions. The source-v3 native
runtime record contains all 48 named executions. Native full-workspace compile
records are separate from selected runtime execution. Host and native records
remain separately labeled in EVIDENCE.md.

## out of scope

- Windows GNU and ARM64 targets, GPU backend enablement, and performance claims.
- Emulating inherently POSIX interfaces on Windows (fork, POSIX descriptor passing, libc loader interposition, POSIX PTY and signals). Their portable siblings and shared data types remain in scope.
- Triggering paid remote runners, publishing, committing, or claiming native output where no native host is available.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| Windows readiness and handle ownership differ from Unix | observed source uses RawFd | lost wakes or invalid handle use | assert Pending-to-wake, rearm, stale generation, EOF and post-drop wake behavior |
| Optional Unix facilities enter the default workspace graph | observed process build script panics outside macOS/Linux | workspace compile failure | preserve package membership and narrow implementation/dependency cfgs |
| MSVC dependency tools are unavailable on a cross host | environment-dependent | compiler stops before project source | retain compiler output and configure native CI; do not turn missing tools into passing evidence |
| Native Windows hardware is unavailable in the current session | unmeasured until access inventory | native execution evidence cannot be collected locally | make native CI runnable and distinguish executable code from unexecuted native acceptance cells |
| Portable defaults are accidentally removed during cfg changes | implementation risk | different product passes compilation | exact public-facade tests plus full workspace artifact coverage |

## context

- `prime/src/os/reactor.rs:16`: raw descriptor import; `:993`: unsupported-target cfg and following compile error.
- `prime/src/os/net.rs:49`: feature/platform gate; `:111`: real socket creation/bind; `:155`: accept operation and readiness registration after a nonblocking return.
- `prime/src/os.rs:70`: network module platform boundary.
- `proxima-net/src/lib.rs:37`: prime facade platform boundary.
- `proxima-process/build.rs:23`: OS match; `:26`: unsupported-target panic.
- `Cargo.toml:1`: workspace membership; member counts are extracted by the gate rather than copied from an earlier draft.
- `.github/workflows/prime-serve.yml:35`: Ubuntu/macOS matrix; `.github/workflows/affected.yml:1`: affected-area workflow.

## struck

- ~~Wait for native build output before defining implementation scope.~~ The user requested the port now; portable defaults, the runtime boundary and complete workspace are specified before code, with native execution left as an explicit measurement requirement.
- ~~Exclude POSIX process/network modules wholesale until a Windows implementation is specified.~~ Only intrinsically Unix operations may remain Unix; TCP, UDP, reactor and portable functionality must remain available.
- ~~`workspace_members=47` copied from setup narrative.~~ Extract and compare actual metadata membership; hardcoded history can silently mask a removed crate.
- ~~A nonzero test set (`executed_tests>=1`) establishes the portable runtime.~~ Exact named IO and reactor assertions replace this weak gate.

### superseded draft acceptance text


~~These Windows-port criteria are draft and unadmitted until native Windows output exists and the spec auditor reviews them. This worktree setup did not run or claim any Windows build.~~

~~| id | discharges | command | expected |~~
~~|---|---|---|---|~~
~~| AC1 | R1, R2 | `cargo check --workspace --target x86_64-pc-windows-msvc` on a Windows runner | `workspace_members=47`, `compiler_errors=0`; unsupported platform-bound features are listed in the gate output rather than silently omitted. |~~
~~| AC3 | R3 | `rg -c '^\s*runs-on: windows-latest\s*$' .github/workflows/windows.yml` (workflow is a planned port artifact) | `1` Windows runner job; its pull-request trigger and affected-path selection are present in the same workflow. |~~
~~| AC4 | R4 | `cargo nextest run --workspace --target x86_64-pc-windows-msvc` on the Windows runner, with a machine-readable test summary | `executed_tests>=1`, `failed_tests=0`; a zero-test run fails the gate. |~~

## supporting proxima-net facade contract

After the Prime runtime and filesystem fixtures execute, validate the public
`proxima-net::prime` adapters in the existing crate. Add
`proxima-net/tests/windows_port.rs` with exactly these four tests, enabled by
`--features prime,runtime-prime-inbox-alloc`:

1. `windows_port_acceptor_factory_payload`: bind `PrimeAcceptorFactory` on
   ephemeral loopback, accept an independent `std::net::TcpStream`, assert the
   exact `GET /windows-port HTTP/1.1\r\nHost: localhost\r\n\r\n` request, send
   `HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok`, and assert `PeerInfo::Tcp`
   matches the std peer's local address.
2. `windows_port_tcp_upstream_payload`: dial an independent std TCP listener
   through `PrimeTcpUpstream::connect`, exchange the same request/response
   byte literals, and assert the connection peer matches the listener address.
3. `windows_port_datagram_factory_payload`: bind `PrimeDatagramFactory`, exchange
   `proxima windows udp probe` with an independent std UDP socket, and assert
   payload, count, and source address. Exercise `poll_recv_batch` and
   `poll_send_batch` with distinct `proxima batch first` / `proxima batch second`
   literals, checking every received payload/address and every sent datagram
   rather than assuming that a batch fills in one poll. Also assert that an
   undersized receive buffer exposes the expected payload prefix.
4. `windows_port_packet_listener_factory_payload`: bind
   `PrimePacketListenerFactory`, receive `proxima windows udp probe` from a std
   UDP peer, assert `Packet.src`, `Packet.dst`, and `Packet.data`, then send the
   received packet as a reply and assert its destination and byte payload at the
   std peer. This tests the documented reply contract: `Packet.src` is the send
   destination; `Packet.dst` is the local address.

All four execute on Prime CoreShard workers with independent std peers and
bounded harness timeouts. No sleeps, new crate, or feature removal is permitted.
The same fixture must compile for MSVC and execute on the native Windows gate;
host execution and Wine execution retain their distinct environment labels.
The supporting acceptance command is:

```sh
cargo nextest run -p proxima-net --features prime,runtime-prime-inbox-alloc --test windows_port
```

Expected fixture cardinality: `executed_tests=4`, `failed_tests=0`. Also run the
existing `prime::` unit-test subset; record its actual executed/skipped counts
separately from this fixed four-test fixture. A successful compile alone does
not discharge either execution requirement.

## supporting tensor instrumentation contract

Windows `instrument` uses QueryPerformanceCounter raw ticks and a cached
QueryPerformanceFrequency only at `ticks_to_nanos`, preserving deferred unit
conversion. Thread CPU time comes from GetThreadTimes kernel + user FILETIME
counters (100 ns units), not wall time. The existing infallible API must fail
explicitly if a Windows timing call fails, never report fabricated zero.
`ru_minflt` stays Unix-only: Windows total page faults are a different metric.
No external caller of instrument::ru_minflt was found in the source scan.

Add exactly two Windows-only tests in
`proxima-tensor/tests/windows_instrument.rs`:

- `windows_instrument_qpc_ticks_and_conversion`: bracket `read_ticks` between
  independent direct QPC calls and assert ordering; query QPF directly and
  assert one frequency-unit converts to 1,000,000,000 ns. Check zero, fractional
  units, multiple seconds, and saturating large input against u128 arithmetic.
- `windows_instrument_thread_cpu_matches_os`: bracket `thread_cpu_nanos` by
  independent GetThreadTimes calls and assert kernel + user accounting bounds
  and 100 ns units; execute real black-box arithmetic until OS CPU time advances
  within a bounded wall guard, then assert the exposed thread counter advances.

Command on Windows: `cargo nextest run -p proxima-tensor --features instrument
--test windows_instrument`; expected `executed_tests=2`, `failed_tests=0`.
Wine execution keeps its environment label and does not stand in for native
Windows. The platform branches must also compile with the existing root
feature graph; no timing or throughput performance claim follows from tests.

Authoritative API contracts read before implementation:
https://learn.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter
https://learn.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancefrequency
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadtimes

## cli platform boundary

Keep HTTP serving, stdio, SSH pipeline transport, parsing and offline commands
available on Windows. Unix-domain requests must return a configuration error
naming the unsupported transport. Only Unix transport enum variants, helpers,
and Unix-specific integration tests are gated. Windows HTTP serving waits for
Ctrl-C through Tokio's Windows signal implementation; Unix retains TERM/INT.
Compile every CLI target in the full workspace gate. Validate exactly three new
Windows CLI tests across two binaries:
- proxima: `pipeline::windows_tests::windows_cli_local_uds_reports_unsupported`
  and `windows_tests::windows_cli_daemon_uds_reports_unsupported` reject local
  and daemon-client UDS requests (executed_tests=2, failed_tests=0).
- proximad: `windows_tests::windows_cli_server_uds_rejects_before_state_creation`
  rejects both HTTP and MCP Unix socket modes before runtime or state-directory
  creation, using a nonexistent child of a tempfile directory
  (executed_tests=1, failed_tests=0).

Run `cargo nextest run -p proxima-cli --bin proxima windows_cli_` and
`cargo nextest run -p proxima-cli --bin proximad windows_cli_` on Windows.
Parsing unit tests remain available on Windows. No silent success or package exclusion.

The supplemental root fixture runs the same two facade tests with
`--features http1-native`; its TCP case additionally drives a two-worker HTTP
listener. Default-feature facade execution remains a separate two-test cell.
The runnable Windows example enables `http1-native` for this supplemental path.

## supporting process and host-ground contract

The existing Windows Command/Child adapter must execute real children through
std::process, retain native OS-string arguments/current directories, collect
both output streams, feed stdin, expose exit/try_wait/kill, and propagate spawn
errors. POSIX dispatch-fd, interposition, controlling terminal, umask and raw-fd
stdio remain explicitly unsupported on Windows. Command SendPipe must shuttle
its input/output bytes through the existing dedicated blocking IO threads.

HostRead/HostWrite execute real filesystem effects at the existing dedicated
blocking dispatch boundary (dispatched.rs dispatch_thread_body), not reactor
polling. Read honors max_bytes/offset and EOF. Write appends because protocol
Write has no offset, and acknowledges only actual OS-written bytes. File errors
retain the OS error in ProximaError; unsupported request kinds return explicit
protocol errors. HostRead Stat uses real Unix metadata; Windows rejects the
Unixmode-shaped Stat instead of inventing POSIX permission bits.
OsEntropy fills via getrandom with error propagation. RealClock emits decimal
UNIX epoch seconds, matching FixedClock's documented epoch_seconds, with read
bounds/EOF. FixedClock gains the same bounds and loses its false AllocFree
marker because the protocol response owns allocated decimal bytes.

Add exactly five Windows-only tests in proxima-process/tests/windows_process.rs:
- windows_process_output_and_exit: independent std Command oracle and fixed
  stdout/stderr literals from cmd.exe, nonzero exit 23, stdio config restoration.
- windows_process_stdin_round_trip: fixed stdin line echoed by child; assert
  stdout bytes, wait/try_wait exit, and actual nonzero PID.
- windows_process_environment_directory_and_kill: Unicode tempfile directory,
  child-written environment payload read independently from disk; live child's
  piped stdin keeps it pending until kill, then wait observes termination.
- windows_process_errors_and_unsupported: missing executable and embedded NUL
  fail; all named unsupported Windows knobs reject rather than succeed.
- windows_process_pipe_round_trip_and_error: Request body through Command's
  SendPipe to actual stdin/stdout; exact response bytes and missing-program error.

Add exactly four cross-platform tests in proxima-process/tests/host_grounds.rs:
- host_ground_filesystem_payloads: two actual writes append fixed literals to a
  Unicode tempfile path; independent std read compares bytes, bounded offset
  reads/EOF compare literal slices, and Stat follows its platform contract.
- host_ground_filesystem_errors: missing read, directory write, and unsupported
  protocol operations report errors; failed writes do not acknowledge bytes.
- host_ground_entropy_reads_os: requested lengths and writable bytes from real
  OS entropy, with zero/canned-buffer rejection sanity assertions (not a claim
  about cryptographic quality); unsupported writes reject.
- host_ground_clock_epoch_and_slices: RealClock seconds fall within independent
  SystemTime brackets; FixedClock literal slices honor bounds and EOF.

Windows commands: cargo nextest run -p proxima-process --test windows_process
(expected executed_tests=5, failed_tests=0); cargo nextest run -p proxima-process
--test host_grounds (expected executed_tests=4, failed_tests=0). Host executes
host_grounds plus existing library/type-system tests, with existing-test counts
recorded separately. Wine and native Windows results retain distinct labels.

## model-interop dependency boundary

The root's test dependency graph enables model interop and Metal features on
Windows. Metal-only callers remain macOS-only even when the feature is enabled.
Ordinary portable mmap and CPU paths remain compiled. Unix mapping advice is
an optional kernel hint at mmap construction and is applied only on Unix;
requests for the separately exposed Unix expert-prefetch operation return an
explicit unsupported error elsewhere. Fixed-address expert windows keep their
explicit Unix-only restriction. Do not invent bytes-prefetched counts or
replace native fixed mapping with heap copies. Full workspace all-targets
compilation must include these packages rather than excluding them.

Process pipe fixture refinement (parent admitted): configure piped stderr and
make the child write repeated stderr bytes before its stdin/stdout echo. The
stdout literal must still arrive; the configured stderr pipe must be drained
concurrently on the blocking IO owner so pipe capacity cannot deadlock output.
The Windows process fixture cardinality remains five.

Windows listener ownership: a single lane owns each bound address because
Windows SO_REUSEADDR does not provide Unix reuseport load balancing. HTTP uses
the existing SpreadToPeers handler dispatch over configured workers. Other
protocols bind once and retain their existing inline handler semantics. The
supplemental root HTTP fixture forces PROXIMA_HTTP_HANDLER_SPREAD=0 through
temp_env, so passing cannot depend on an ambient opt-in. Preserve the captured
AddrInUse failure before the topology change and compare the same payload
fixture after it; do not reduce the configured worker count or weaken binds.

Process lifecycle refinement (parent admitted): within the existing pipe
fixture, a never-ready request body keeps a real silent child alive. An
independently opened OS process handle must be unsignaled before response
drop and signaled after drop; the pending body must also report destruction.
Windows process fixtures run in nextest's per-test process isolation; direct
Wine/libtest execution uses --test-threads=1 for the parent-PID child oracle.
Within the existing environment fixture, mixed-case set/set/set and removal
are compared to std Command. Missing executable/file checks compare retained
OS error kinds and raw codes to independent std operations. Counts stay 5+4.

Process ownership decision: abandon output-reader-owned Child followed by
joining the input writer, because an unread response and a pending request
strand both threads and the child. A duplicated process HANDLE preserves
identity for a response/startup RAII cancellation guard; AbortHandle releases
the pending request, and a separate waiter owns Child. Configured stderr drains
concurrently. Errors retain ProximaError::Io. No new output cap is introduced:
output() retains std wait_with_output collection semantics; streaming uses its
bounded channel and fixed-size IO buffer. Native Command preserves OS strings;
the CString descriptor's Windows conversion remains explicitly UTF-8-only.

Prime completion assertions (R2/R3/R6, existing eight names): extend the TCP
EOF fixture so Prime writes a fixed request, closes its write direction, and
an independent std peer reads through EOF before sending a fixed response;
Prime must still receive that response. Async close propagates the actual
shutdown result rather than discarding an OS error. Extend TCP pending-read
fixture with a separate connection whose peer reads nothing until Prime
actually reports write Pending; bounded repeated-pattern bytes fill the send
buffer, then the peer drains and compares every byte through EOF. Completion
must follow a registered write wake, and bounded capacity exhaustion without
Pending is a fixture failure. Extend TCP loopback with two accept operations
whose independent clients connect only after an observed accept Pending.
Extend UDP loopback with 40 independently sent, prequeued distinct datagrams;
64 caller slots cannot make any batch exceed DATAGRAM_BATCH=32. Receive all
40 and assert each payload, source address, and untouched slots beyond the
returned batch. The first batch must return 32, and remaining datagrams must
remain available. No sleeps or fixture-count changes. Validation remains the
existing host and native/Wine-labeled Prime windows_port command with
executed=8, failed=0; raw assertion failures remain evidence.

### upstream call ownership refinement

Each StreamUpstream connection call owns its pending operation. Replace the
shared-reference poll_connect hook with object-safe connect_future returning
ConnectFuture<'_, Self::Conn>; retain StreamUpstreamExt::connect and its named
Connect future. The boxed future is required at the existing open transport
erasure boundary; concrete Prime TCP dialing uses an inherent async method.
Do not identify calls by Waker equality or store dial state in a shared mutex.
Dropping one call releases its registration without cancelling or waking a
different call. Wrappers and other runtime adapters create fresh state too.

Prime host resolution runs on the existing ProximaBackgroundPool primitive,
behind a process-lifetime std-only resolver pool. This lifetime prevents a
cancelled DNS call from waiting on a synchronous system resolver on the reactor thread.
Dial all returned addresses in resolver order until one connects; retain the
actual connected peer and propagate the last real error if none connect.
Construction does not resolve. Each future owns its DNS completion receiver;
cancellation does not block waiting for getaddrinfo to return.

Extend the existing proxima-net upstream fixture (total remains exactly four
focused tests) with fixed payloads from independent std TCP peers for hostname
dialing, simultaneous calls sharing one upstream on one and two workers, and
dropped-call isolation. Hostname fallback additionally uses a deterministic
ordered address vector with loopback port zero (independent std refusal oracle)
before a live address,
exercising the same dial helper; failed resolution and all-address failure must
retain io::Error. A controlled blocking resolver seam verifies another task can
run while resolution is held, and releasing it resumes the exact payload path.
The existing host and Windows proxima-net windows_port commands still require
executed_tests=4, failed_tests=0; helper/unit tests record separate cardinalities.

The half-close fixture also compares retained OS error kind/raw code against
an independent std client after both observe an abortive peer close (linger
zero). Each client consumes the reset with read before shutdown(Write)/close;
compare actual results rather than prescribing a platform-specific error.

TLS/QUIC ownership assertions: the existing TLS loopback fixture accepts two
TCP connections before starting either handshake; concurrent calls against one
upstream must independently return hello/world bytes. The existing QUIC stream
fixture opens two concurrent calls, exchanges distinct doq-first/doq-other
requests and ack-first/ack-other responses, and retains the configured pool
bound. Existing fixture counts remain unchanged; run the connector module and
QUIC stream_listener test filters on host and MSVC binaries on native Windows.

The two additional unit cases are named
`upstream_address_fallback_preserves_payload_and_errors` and
`upstream_resolver_yields_and_cancels_without_waiting_for_work`; filter
`prime::tests::upstream_` requires executed_tests=2, failed_tests=0.
The first compares fallback-request/fallback-response bytes and actual peer,
last error kind/raw code against std, and the empty-address error. The second
holds resolver work behind a channel while an unrelated reactor task runs,
then checks both receiver completion and cancellation before work is released.
A cancelled call's worker must join while the resolver remains held. Embedded
NUL hostname errors must match the system resolver kind/raw code. No sleeps.

### final gate fixture inventory

The runtime/host gates also require exact named subsets for the caller-owned
upstream repair: net resolver/fallback 2, TLS connector 4, QUIC stream listener 1.
These are additional assertion-bearing executions, not replacements for the
Prime filesystem/runtime or adapter tests. The host gate therefore expects 38
executions: Prime fs 4, Prime runtime 8, net adapters 4, net resolver 2, CONNECT tunnel 6, fixed-tuple lease 1, TLS 4,
QUIC 1, default root 2, HTTP-enabled root 2, host grounds 4. Native Windows
adds timing 2, CLI 2, daemon 1, children 5, for 48 executions. Every subset
requires failed=0, ignored=0 and its exact name set; the native guard stays in
place. AC2/AC5 include this full inventory in addition to their original counts.

Write-Pending fixture refinement (parent admitted after execution): the Wine
run in /private/tmp/proxima-windows-evidence/prime-sourcefreeze-wine-windows-port.log
accepted the whole repeated-pattern payload before returning Pending; the
assertion failed rather than passing without readiness coverage. A bounded
payload and peer receive-buffer setting do not establish the sender capacity.
Expose ordinary TcpStream send-buffer setter/getter wrappers over socket2,
request 64 KiB in the fixture, and read back the effective SO_SNDBUF value.
Require positive capacity below the fixed payload length before writing.
Retain the actual Pending handshake and complete byte comparison; do not
replace the readiness assertion with the option value. Keep the failed log.
Each individual send attempt is capped at 16 KiB, preserving the complete
pattern while avoiding a single oversized Winsock staging operation.

Fixed-port DPDK/XDP migration: each upstream retains its configured local port.
A second call for an occupied peer tuple returns AddrInUse rather than replacing
that tuple's TCP state. A bounded free/pending/cancelled lease publishes Drop
cancellation with release/acquire ordering; the next shared-stack acquisition
reclaims only the dial's matching initial sequence generation. Completion
transfers the lease into the delivered stream; its Drop publishes cancellation. No multiport or hardware
execution claim is implied. One sans-IO unit case
`tcp_dial::tests::fixed_tuple_preserves_payload_and_cancels_only_owned_generation`
asserts the original SYN identity and payload survive a rejected second call,
pending and completed-before-delivery cancellation reclaim owned state, and a
stale cancellation cannot remove a replacement generation. Run this filter with
executed_tests=1, failed_tests=0, plus Linux adapter compilation.
The fixed-tuple cancellation unit also reconstructs passive replacement with
exactly the cancelled active call's initial sequence; passive! bytes and accept
identity must survive reclamation. Open entries retain their active/passive
origin for this check. The TCP initial sequence is not a globally unique
identifier; active-owner exclusivity plus origin and sequence matching define
the cancellation boundary.
Delivered optional-driver active streams retain the same tuple lease until
stream Drop; RST does not permit a second active call to reuse that tuple while
an old stream exists. Active reads/writes/flush/close and delivery check origin
and initial sequence before accessing the stack. Listener-origin connections
keep their current behavior. Extend the same lease test with an old handle
retained across RST and passive replacement: old I/O returns ConnectionReset,
replacement bytes remain untouched, and old Drop cannot cancel replacement.
An old queued completion cannot deliver a stream while its own active entry
has not completed the handshake. Guard completion transfers ownership rather
than releasing the tuple. Counts remain unchanged.

CONNECT input bounds refinement (R2/R3/R6): keep the existing constructor and
add with_max_response_header_bytes, default 16 KiB including status line and
terminating CRLFCRLF. The HTTP codec has per-line and field-count policy but
no shared cumulative response-head bound exposed to this crate; this explicit
per-upstream setting avoids importing another transport layer. Reject zero
bound and invalid target host/nonzero-port combinations before proxy dialing.
Hosts support RFC 3986 reg-name (unreserved, sub-delimiters, and valid percent
escapes), IPv4 and raw or bracketed IPv6; emit bracketed IPv6 authority. Reject
controls, whitespace, userinfo, paths/query/fragment, malformed percent escapes
and embedded ports with InvalidInput. Raw Unicode requires URI encoding.
Read only remaining header capacity; an unterminated head reaching the bound
returns InvalidData without reading more bytes. A complete head exactly at
the limit remains accepted. Preserve fixed CONNECT wire bytes and subsequent
tunnel payload. New synchronous scripted-I/O fixture names are
connect_tunnel_rejects_invalid_authority_before_dial,
connect_tunnel_bounds_fragmented_proxy_headers, and
connect_tunnel_preserves_fragmented_response_and_payload; they assert actual
dial counters, read bytes, written request bytes, response fragments, exact
limit success, oversized/incomplete failure, and transparent payload. Run
cargo nextest run -p proxima-net --features prime -E
'test(prime::connect_tunnel::tests::)':
executed_tests=6, failed_tests=0, including the existing parser cases.

Native gate provenance includes OS system/release/build/machine and checks
ntdll for Wine exports. A Windows-target Python/Rust running under Wine must
not satisfy the native runtime guard. Cross/host and Wine records retain their
own environment labels.

CONNECT coalesced payload refinement (parent requested): retain bytes following
the successful response head using futures::io::Chain<Cursor<Vec<u8>>, C>.
The returned ConnectTunnelConnection<C> delegates writes/flush/close and peer
identity to C and owns its disposal; the upstream associated Conn becomes
this wrapper. Existing HTTP/TLS call sites are generic over StreamConnection.
Extend the same normal-payload fixture with a 2xx head and banner in one
underlying read, followed by another fragment; the caller reads exact
banner-plus-tail bytes, writes reach C unchanged, and Drop reaches C once.
Keep executed_tests=6. Reject the prior assumption that a proxy cannot forward
server-first data beside its response head (RFC9110 section9.3.6).
The existing successful-status parser fixture also rejects unsupported HTTP
versions and non-three-digit/non-ASCII-digit status tokens with InvalidData,
while accepting HTTP/1.0 and HTTP/1.1 responses. Counts remain six.
The status-line assertion additionally requires the SP after status digits
and permits only HTAB/SP/VCHAR/obs-text in the reason; missing separator, NUL,
bare LF, and DEL are InvalidData. Empty reason with its separator is valid
(RFC9112 section4).
Fragment processing scans only new bytes plus the prior three-byte delimiter
overlap, invoking the complete-head parser once CRLFCRLF is found. Existing
fragmented-limit assertions also deliver one byte per read for over-limit
and exact-limit heads; no throughput or latency result is claimed.
