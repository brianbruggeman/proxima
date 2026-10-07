# windows port execution evidence

## client TLS roots, target worktree

Worktree: `proxima-tls-native-roots`, branch `feat/tls-root-settings`.
These are current-worktree observations from Darwin arm64 plus an MSVC
cross-target check; none is native Windows runtime evidence.

| contract | observation | raw artifact |
|---|---|---|
| AC7/AC9 target defaults and root policy | 9 selected tests passed, 0 failed, 27 skipped by filter | `/private/tmp/proxima-windows-evidence/tls-native-roots/card-6-host.stdout` |
| AC7 layered file/env configuration | 2 passed, 0 failed, 34 skipped by filter | `/private/tmp/proxima-windows-evidence/tls-native-roots/card-7-host.stdout` |
| AC8 portable trust handshakes | 4 passed, 0 failed, 32 skipped by filter; native-store test is Windows-only and not selected on Darwin | `/private/tmp/proxima-windows-evidence/tls-native-roots/card-8-handshakes-host.stdout` |
| AC8 malformed custom bundles | 1 passed, 0 failed, 35 skipped by filter | `/private/tmp/proxima-windows-evidence/tls-native-roots/card-8-bad-bundles-host.stdout` |
| complete Proxima TLS library suite | 36 passed, 0 failed, 0 skipped | `/private/tmp/proxima-windows-evidence/tls-native-roots/final-host.stdout` |
| complete TLS connector filter | 14 passed, 0 failed, 0 skipped | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-10-host.stdout` |
| Windows MSVC compile including Windows-only store fixture | `cargo xwin check -p proxima-tls --features futures-io --all-targets --target x86_64-pc-windows-msvc` finished successfully with no compiler errors | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-10-windows-xwin.stdout` |
| AC11 universal client spec and canonical forwarding | 2 tests passed: the fluent universal client carries `tls_client`, and canonical HTTP forwarding preserves native/custom roots | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-client.stdout` |
| AC11 HTTP TLS factory propagation | 2 tests passed: prime and Hyper factories each fail before dialing on a configured missing custom CA and include its path | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-http-factory.stdout` |
| AC11 gRPC TLS factory propagation | 1 test passed: configured missing custom CA fails before dialing and the error includes its path | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-grpc-factory.stdout` |
| AC11 listener mTLS custom roots | 1 test passed: required client-CA bundle path round-trips through the listener TLS spec and builds a client-certificate verifier | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-listener.stdout` |
| AC7–AC11 full TLS library after facade wiring | 37 passed, 0 failed, 0 skipped | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-tls-full.stdout` |
| AC11 Proxima Windows MSVC all-target check | `cargo xwin check -p proxima --all-targets --target x86_64-pc-windows-msvc` finished with exit 0 in 5m10s | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-windows-xwin.stdout` |
| AC11 Hyper feature Windows MSVC all-target check | `cargo xwin check -p proxima-http --all-targets --features http1-stream-client,http1-tls --target x86_64-pc-windows-msvc` finished with exit 0 in 1m51s | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-http-windows-xwin.stdout` |
| AC11 universal Listener TLS forwarding | 1 test passed, 0 failed/skipped; `TlsListenProtocol` forwards `required_files` and `client-ca.pem` to the wrapped protocol spec | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-11-listener-facade.stdout` |

## hyper alias and ALPN corrections

The Windows alias had asymmetric cfgs: `AliasFactory` and its impl used `unix`
at `src/load.rs` while the registration already allowed `any(unix, windows)`.
The type therefore disappeared in the Windows both-wires feature set. The
definition now shares the registration's platform predicate.

The Hyper TLS path built a normal `TlsClientConfig`, whose default ALPN list is
non-empty, then passed it to `HttpsConnectorBuilder`; Hyper owns ALPN based on
`.enable_http1()` / `.enable_http2()`, and rejects a preset list. The connector
now clears only `rustls_config.alpn_protocols` before handing it to Hyper.
Root policy, verifier, and protocol versions remain the configured rustls values.

| contract | observation | raw artifact |
|---|---|---|
| AC12a Windows `http-hyper` + Prime alias cfg | `cargo xwin check -p proxima --lib --features http-hyper --target x86_64-pc-windows-msvc` finished with exit 0; compiles the Windows alias registration and its factory type | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-12-alias-xwin.stdout` |
| AC12b Hyper TLS ALPN ownership | 1 regression test passed; URL-aware TLS config starts with ALPN set and `with_client_tls_config` constructs `hyper-rustls` successfully after clearing it | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-12-hyper-alpn.stdout` |
| AC12b Hyper without TLS feature | `cargo check -p proxima-http --lib --features http1` finished with exit 0, with no unused-field or unused-mut diagnostics | `/private/tmp/proxima-windows-evidence/tls-native-roots/ac-12-hyper-no-tls.stdout` |
| Native Windows CI for `f2e26b1bf0080fcad8080e3102a744c1e501084d` | GitHub Actions run [37542940589](https://github.com/brianbruggeman/proxima/actions/runs/37542940589) completed with failure in `windows-port` at “compile every workspace target” (exit 1). Public annotations identify the failed step but provide no compiler diagnostic; fetching job logs returned HTTP 403 (`Must have admin rights to Repository`), and the Actions page requires sign-in. The compile failure mechanism is therefore unexplained by accessible artifacts. | https://api.github.com/repos/brianbruggeman/proxima/actions/runs/37542940589/jobs |

The universal facade acceptance commands are:

```text
cargo nextest run -p proxima --lib --features http-prime-deps -E 'test(tls_client_settings_are_carried_by_the_universal_client_spec) + test(canonical_http_forwards_transport_now)' --no-tests=fail --test-threads 1 --retries 0
cargo nextest run -p proxima --lib --features http-prime,http2 -E 'test(grpc_factory_applies_client_tls_root_settings)' --no-tests=fail --test-threads 1 --retries 0
cargo nextest run -p proxima-http --lib --features http1-stream-client,http1-tls -E 'test(factory_applies_client_tls_root_settings_before_dial) + test(hyper_factory_applies_client_tls_root_settings)' --no-tests=fail --test-threads 1 --retries 0
cargo nextest run -p proxima-tls --lib --features futures-io -E 'test(listener_client_auth_bundle_paths_round_trip_and_build)' --no-tests=fail --test-threads 1 --retries 0
```

The portable round-trip assertions compare exact `b"tls root probe"` replies;
the untrusted certificate is rejected and malformed bundle errors retain the
input path. The cross-target command proves compilation only. The
`tls_trust_native_store_round_trip` fixture has compiled for Windows but has
not executed on a Windows host. Native Windows runtime remains required before
AC8/AC10 can be closed.

Worktree: `proxima-windows`, branch `port/windows-support`. Cargo workspace
membership remained at 47 entries; no new crate or Cargo manifest was created.
Changes remain uncommitted. Extracted records are in
`ai_docs/windows-evidence.jsonl`; raw logs are under
`/private/tmp/proxima-windows-evidence`.

The checked-in Windows workflow uses direct Cargo commands. The temporary
Python evidence runners used to capture the raw records are not included in the
patch; earlier report paths below preserve their output as historical evidence.

## source archive association

The staged native source archive is `/private/tmp/proxima-windows-evidence/native-source-v2.zip`
(SHA-256 `7a715197b0e304b477a11f4c7f1779e141cfe561646f854c9416b8032498f035`,
18,079,538 bytes). Its manifest is
`/private/tmp/proxima-windows-evidence/native-source-v2-manifest.json`
(SHA-256 `23deaa895ecaa71d546823d0e68550de246d8bd6b27e031f9f5a1cc27c503867`);
it lists 2,961 files. The file
hash/length comparison against the staged tree recorded 2,961 present, zero
missing, zero mismatched before later evidence and CRT-policy edits. The
source-v2 archive identifies the payload used for the preliminary native rows
below; it predates the target-specific MSVC CRT policy.

The updated source-v3 archive is
`/private/tmp/proxima-windows-evidence/native-source-v3.zip` (SHA-256
`5ec562114ea2c3b29e7182f9e56ea4180f7dc55c50dddb401a4296574119543f`,
16,628,351 bytes). Its manifest lists 2,963 files. The native guest verified the source-v3 manifest against the staged tree:
`native-vm/manifest-verify-v3.log` records 2,963 checked files and zero
mismatches. Source-v3 is the Rust implementation used by the final native
runtime, CRT, and example observations below. Native full-workspace compilation
remains pending. `ai_docs/windows-evidence.jsonl` associates both archive hashes
with their manifest records and records the native v3 manifest verification.

## compiler and host gates

`gate-msvc-sourcev3/compile.json` records the final `--xwin` run of
`cargo xwin check --cross-compiler clang-cl --workspace --all-targets --target
x86_64-pc-windows-msvc --message-format json`: 47 workspace members, 565
artifact packages, zero missing members, zero excluded members, zero compiler
errors, and one successful build-finished record. The compiler host is macOS
`aarch64-apple-darwin`; this record is cross-compilation, not Windows execution.

`gate-msvc-sourcev3/cmake-crt-policy.jsonl` records 13 counted CMake policy
checks, all passed, including composition with the actual xwin clang-cl
toolchain. `native_compilation_executed=false` is part of that result. On the native
Windows guest, `native-vm/cmake-runtime-v3-absolute.log` records 12 CMake CRT
policy checks passed and zero failed; conflicting-runtime and wrong-target
negative cases returned 1 with their expected error text. That run records
`xwin_composition_executed=false` and `native_compilation_executed=false`, so it
does not establish a native build. The earlier
`native-vm/cmake-runtime-v3.log` attempt failed before any policy case because
Python could not find `cmake` (`WinError 2`); the absolute-executable run is
recorded separately.

The native zlib build cache also needs explicit invalidation when the CRT
selection inputs change. In the inspected `cmake` 0.1.58 source,
`src/lib.rs:936-943` reads and prints environment values but does not emit
`cargo:rerun-if-env-changed`; its C/C++ helper probes disable Cargo metadata at
`src/lib.rs:519` and `src/lib.rs:531`. The `libz-ng-sys` 1.1.29
`zng/cmake.rs` path likewise configures CMake without tracking the CRT-related
environment. As a result, changing the CMake policy alone can leave Cargo's
existing `libz-ng-sys` build output reusable. Before the first source-v3 native
build in an existing Windows target directory, run:

The cache was invalidated with `cargo clean -p libz-ng-sys --target
x86_64-pc-windows-msvc`; the source-v3 runtime path then rebuilt and linked the
native dependencies under the selected CRT policy.

`native-vm/zlib-baseline-preserve.log` records the pre-clean generated cache: the
CMake toolchain/generator/prefix variables were `None`, generated project files
contained both `MultiThreadedDebugDLL` and `MultiThreadedDLL` runtime entries,
and the captured rerun-environment-directives section was empty.
`native-vm/zlib-clean-result.log` records `cargo clean -p libz-ng-sys --target
x86_64-pc-windows-msvc` removing 189 files (26.6 MiB reported) and exit 0.
After cleanup, `native-vm/zlib-v3-final-inspection2.log` records the target-specific
CMake policy path in the build-script environment, `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`,
`-MD` in both observed C flag values, and four generated project runtime entries
set to `MultiThreadedDLL`. The associated
`native-vm/zlib-v3-directives-raw.log` dumpbin output, counted only from the
`Dump of file` section to avoid the repeated header, records 50 `/DEFAULTLIB:MSVCRT`,
50 `/DEFAULTLIB:OLDNAMES`, and 0 `/DEFAULTLIB:MSVCRTD` directives. These artifacts
link the cache invalidation to the regenerated CRT selection and archive
metadata. `native-vm/root-imports.log` separately records dumpbin exit 0 for the
default and HTTP-enabled root runtime test executables. Each lists one
`VCRUNTIME140.dll`, one release CRT DLL, and zero debug CRT DLLs. The additional
`api-ms-win-crt-*` and Windows system imports in those outputs are not counted as
MSVC CRT DLLs. The final native run's linker stderr has no `LNK4098` or
`MSVCRTD` matches. Native runtime, example execution, and full-workspace
compilation are recorded below.
The preceding cross-cache setup attempt is retained in
`gate-msvc-crt-policy/compile-x86_64-pc-windows-msvc.stderr`: cargo-xwin could
not remove the shared cache's `clang-cl` symlink (`EPERM`). The final run used
an isolated `XWIN_CACHE_DIR=/private/tmp/proxima-windows-evidence/xwin-cache-sourcev3`
copied from the existing cache; the shared cache was preserved.

`gate-host-sourcefinal-nextest/host.json` records 38 selected host executions
with zero failures, ignored cases, and missing tests. The entries are Prime
filesystem 4, Prime runtime 8, net adapters 4, resolver ownership 2, CONNECT
tunnel 6, fixed-tuple ownership 1, TLS connector 4, QUIC listener 1, root
default 2, HTTP-enabled root 2, and host grounds 4. Its final workspace
all-targets check returned 0. Raw per-command records are in the same directory.

`gate-examples-sourcefinal/examples.json` records 21 doctests passed, 0 failed,
14 ignored, and one executed example. Payload match records are incumbent 2,
TCP 1, UDP 1, and multiworker listener 1. The negative-control report
`gate-control-sourcefinal/control.json` records five rejected invalid fixtures
and zero accepted; the intentionally corrupted payload command returned 1.
`gate-surface-sourcefinal/surface.json` records four required supported
entries, zero missing sources, and zero unclassified changed boundaries.
`gate-ci-sourcefinal/ci.json` records the earlier workflow inventory. The
landing workflow now runs direct Cargo commands and does not use a Python gate
or artifact-upload step.

## native Windows guest observations

`native-vm/gate-native-crt-runtime-final-raw/gate-native-crt-runtime-final/runtime.json`
records the final source-v3 native runtime run. Its
environment is Windows Server 2025, version 10.0.26100, AMD64, with
`wine_export_present=false`; the separate fingerprint log identifies Microsoft
Windows Server 2025 Standard Evaluation. The report lists 15 test groups with
48 selected tests, all passed, zero failed, ignored, or missing; its 31 recorded
commands have return code 0. The corresponding per-group libtest JSON outputs
are under `native-vm/gate-native-crt-runtime-final-raw/gate-native-crt-runtime-final/`.

| fixture | observed assertions | raw artifact |
|---|---|---|
| Prime filesystem | 4 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/prime-windows_fs-tests.stdout` |
| Prime runtime | 8 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/prime-windows_port-tests.stdout` |
| proxima-net adapters | 4 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/proxima-net-windows_port-tests.stdout` |
| upstream resolver | 2 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/net-upstream-ownership-tests.stdout` |
| CONNECT tunnel | 6 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/net-connect-tunnel-tests.stdout` |
| fixed-tuple lease | 1 test passed, 0 failed/ignored | `native-vm/gate-native-runtime/net-fixed-tuple-ownership-tests.stdout` |
| TLS connector | 4 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/tls-upstream-ownership-tests.stdout` |
| QUIC stream listener | 1 test passed, 0 failed/ignored | `native-vm/gate-native-runtime/quic-upstream-ownership-tests.stdout` |
| root default facade | 2 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/proxima-windows_port-tests.stdout` |
| HTTP-enabled root facade | 2 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/proxima-windows_port-http1-native-tests.stdout` |
| host grounds | 4 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/process-host-grounds-tests.stdout` |
| tensor timing | 2 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/tensor-windows-instrument-tests.stdout` |
| CLI refusals | 2 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/cli-windows-refusals-tests.stdout` |
| daemon refusal | 1 test passed, 0 failed/ignored | `native-vm/gate-native-runtime/daemon-windows-refusal-tests.stdout` |
| process children | 5 tests passed, 0 failed/ignored | `native-vm/gate-native-runtime/process-windows-child-tests.stdout` |

The raw-path cells above point to the source-v2 baseline. The source-v3 files
with the same per-group names are in
`native-vm/gate-native-crt-runtime-final-raw/gate-native-crt-runtime-final/`.

The source-v3 report records 31 commands with return code 0 and 48 selected
tests passed, 0 failed/ignored/missing across 15 groups. The final raw record
includes each group's libtest events. The native control report records five
invalid fixtures rejected, including the deliberately corrupted payload.

`native-vm/native-crt-examples-raw/examples.json` records the source-v3 native
example run: 21 doctests passed, 0 failed, 14 ignored, and one example executed.
The example output records incumbent/TCP/UDP/multiworker payload matches of
2/1/1/1. Its stdout and stderr are retained alongside the report. The separate
`native-vm/control-native-final.json` report records the five rejected controls.

The guest's native runtime environment is Windows Server 2025 x64 with MSVC C++
x64 build tools and a Windows SDK, Rust stable `x86_64-pc-windows-msvc`,
`cargo-nextest`, and NASM on `PATH` for native dependency builds. The current
workflow's exact focused Cargo commands are listed in
`.github/workflows/windows.yml`; each test filter uses `--no-tests=fail` and
retries are disabled.

Do not label Wine execution as native Windows execution. The native report
records the OS build and checks for Wine exports.

## Windows-target executables under Wine

Wine 11.18 runs on Linux; these records use `LANG=C.UTF-8 LC_ALL=C.UTF-8` and
are not native Windows evidence.

| fixture | observed assertions | raw artifact |
|---|---|---|
| Prime runtime, bounded-buffer revision | 8 tests passed, 0 failed/ignored | `prime-bufferbound-wine-windows-port.log` |
| Prime filesystem | 4 tests passed, 0 failed/ignored | `prime-sourcefreeze-wine-windows-fs.log` |
| proxima-net adapters | 4 tests passed, 0 failed/ignored | `proxima-net-windows-port-wine-current.log` |
| final-source-v2 CONNECT tunnel | 6 tests passed, 0 failed/ignored | `proxima-net-tunnel-final-wine.log` |
| final-source-v2 upstream resolver | 2 tests passed, 0 failed/ignored | `proxima-net-resolver-final-sourcev2-wine.log` |
| final-source-v2 tuple lease | 1 test passed, 0 failed/ignored | `proxima-net-lease-final-sourcev2-wine.log` |
| TLS connector | 4 tests passed, 0 failed/ignored | `tls-connector-wine-tests.log` |
| QUIC stream listener | 0 passed, 1 failed; listener setup returned OS error 10045 (`FormatMessageW()` error 317) | `quic-stream-listener-wine-tests.log` |

The QUIC test passed on the native Server 2025 guest in
`native-vm/proxima-quic-native.log`; the Wine error is retained without
inferring its cause. Earlier root, CLI, daemon, process, and timing Wine records
remain in the JSONL ledger and retain their recorded source snapshots.

## failures retained with their corrections

- Under Wine's SSH POSIX locale, both direct std::fs and the Prime background
  path failed on the same Unicode filename. The identical executable ran the
  byte assertions with UTF-8 locale. See `wine-windows-fs-std-oracle-default.log`
  and `wine-windows-fs-std-oracle-c-utf8.log`.
- Windows UDP truncation filled the buffer but reported a zero count through
  WSAEMSGSIZE. Prime now exposes the filled prefix when RecvFlags reports
  truncation; the test compares the literal prefix and a std socket oracle.
  See `wine-windows-port.log` and `prime-guard-wine-runtime.log`.
- The two-worker Windows HTTP listener failed with OS error 10048/AddrInUse
  when each lane bound the same address. A single Windows bind owner now uses
  existing HTTP SpreadToPeers dispatch. The fixture keeps two workers and
  disables the optional spread environment flag. See
  `root-multiworker-wine-final.log` and `root-multiworker-topology-http1-wine.log`.
- The child stderr fixture initially omitted cmd.exe's redirection-adjacent
  space. Grouped redirection and a fixed literal are now compared against an
  independent std::process invocation. Earlier failing output is retained in
  `process-wine-windows-process.log`; current output is in the sourcefreeze log.

The retained-waker fixture verifies the remote wake callback after executor
destruction and preservation of a newer executor when an older one is dropped.
CoreShard pending-task destructors also assert that runtime TLS is unavailable.
See `prime/tests/windows_port.rs` and `prime-final-host-port.log`.

## optional Linux driver checks and limits

`linux-dpdk-final-sourcev2.jsonl` records `cargo check -p proxima-net
--features dpdk --message-format=json`; it contains five `proxima-net`
compiler-artifact records, zero compiler errors, and one successful
build-finished record. `linux-xdp-final-sourcev2.jsonl` records
`cargo check -p proxima-net --features xdp --all-targets --message-format=json`;
it contains thirteen `proxima-net` artifact records, zero compiler errors, and
one successful build-finished record. Both ran on Linux in the isolated
`/mnt/data1/proxima-windows-linux-check-20261002` copy with one Cargo job and
debug info disabled. The checked copies of `tcp_dial.rs`, `tcp_stack.rs`,
`lib.rs`, DPDK/XDP stream listeners, `prime/mod.rs`, and `prime/connect_tunnel.rs`
matched the final worktree SHA-256 values. These are compile observations only;
no DPDK/XDP device execution is recorded.

The native Windows CI workflow exists but was not published or triggered.
POSIX PTY, fd passing/interposition, Unix sockets, hardware VM backends and
Metal remain platform-bound. Collected child output retains std's unbounded
buffer contract; streaming uses bounded chunks, and cancellation covers the
direct child rather than a descendant process tree. These records do not
support performance claims.

## full-workspace compile observations

`native-vm/crt-runtime-final-root-build.log` records native Windows
`cargo check --workspace --all-targets` completion with process status 0 and
Cargo's `Finished test profile` line. A fresh cross-target check on the current
worktree ran `cargo xwin check --workspace --all-targets --target
x86_64-pc-windows-msvc --message-format short`; it exited 0 after 9m 48s and
printed `Finished dev profile`. Cargo emitted a future-incompatibility notice
for `proc-macro-error2 v2.0.1`, without a compiler error. These checks do not
record execution of the revised direct-command Windows workflow; that workflow
has not been triggered.

## current host workflow commands

On Darwin arm64, the current worktree ran the workflow's
`cargo test -p proxima --doc`: 21 passed, 0 failed, 14 ignored. It also ran
`cargo run -p proxima --example windows_port --features http1-native` with
process exit 0. The example printed incumbent/TCP/UDP/multiworker payload
matches of 2/1/1/1. These are host execution observations; they do not stand in
for the Windows runner's execution.
