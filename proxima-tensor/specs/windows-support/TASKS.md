# windows-support -- slices

Each slice is one coherent behavior change with its validation command. No commit is authorized. Fixed fixture counts below are acceptance targets, not measured results. Run an independent spec-auditor before source implementation.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Implement Windows reactor backend and handle ownership at the existing reactor seam | AC2 | Run Prime filesystem and reactor steps in `.github/workflows/windows.yml` on native Windows | exact native source-v3 inventory is recorded in EVIDENCE.md; native selected runtime record: 48 passed, 0 failed/ignored/missing across 15 groups | [ ] | Native selected tests are recorded; exact direct-command CI execution remains pending. |
| 2 | Enable shared TCP/UDP implementation and default public runtime facades on Windows | AC2, AC5 | Run the network, TLS, QUIC, root facade, and process steps in `.github/workflows/windows.yml` on Windows and host | host record contains 38 named executions; native record contains Prime, proxima-net, and root facade payload assertions | [ ] | Native selected tests are recorded; exact direct-command CI execution remains pending. |
| 3 | Repair remaining workspace Windows compile boundaries and record Unix-only facilities | AC1, AC3 | `cargo xwin check --cross-compiler clang-cl --workspace --all-targets --target x86_64-pc-windows-msvc --message-format json`; inspect `ai_docs/windows-surface.jsonl` source paths | source-v3 cross record covers 47 workspace members with 0 missing/excluded members, 0 compiler errors, and 1 successful build record; surface inventory records the supported and Unix-only boundaries | [ ] | Cross-target check and native full-workspace compile are recorded; exact direct-command CI execution remains pending. |
| 4 | Add native Windows workflow and executable example | AC4, AC6 | inspect `.github/workflows/windows.yml`; run `cargo test -p proxima --doc` and `cargo run -p proxima --example windows_port --features http1-native` | host example record: 21 doctests passed (14 ignored), one example, payload matches 2/1/1/1; native examples record is retained in EVIDENCE.md | [ ] | The workflow now runs direct Cargo commands and does not use a script gate. |
| 5 | Execute native Windows and host acceptance commands and attach raw evidence | AC1–AC7 | Run the direct Cargo commands in `.github/workflows/windows.yml` on the declared host | source-v3 native selected runtime: 48 passed, 0 failed/ignored/missing; examples, controls, and native full-workspace compile are recorded | [ ] | Keep open until the direct-command Windows workflow completes. |

## resume

Last observed execution: source-v3 native Windows Server 2025 runtime evidence records 48 selected tests across 15 groups, 48 passed, 0 failed/ignored/missing. Native examples, CRT selection, and negative controls are also recorded in `EVIDENCE.md` and `ai_docs/windows-evidence.jsonl`.
Next action: run the direct-command Windows workflow and record its full-workspace compile and test output. Keep slice 5 open until that workflow completes.
Open acceptance: execution of the updated direct-command Windows workflow.

## struck

- ~~Finalize portable scope only after native output exists.~~ Scope is now bound to every workspace member and default prime TCP/UDP before implementation.
- ~~A single nonzero Windows test count closes runtime validation.~~ Replaced by the exact 48-test runtime inventory, payload and readiness assertions, negative controls, and separately recorded example execution.

## execution order

User steering binds the implementation order: prime filesystem and handle boundaries plus runtime first; proxima-net second; broader workspace portability third. That ordering has been followed. Process portability follows the executed Prime/filesystem and proxima-net fixtures. Source-v3 native runtime is observed as 48/48 on Windows Server 2025 AMD64; native and current-worktree cross-target full-workspace compiles are also recorded. Cross-compilation, Wine, host, and native evidence remain separately labeled in EVIDENCE.md.


## current AC evidence map

This map records the latest opened artifacts; it does not mark final acceptance complete. Source-v3 native runtime, CRT, examples, controls, and full-workspace compile results are recorded; execution of the updated direct-command workflow remains pending.

| AC | latest observed artifact | recorded fields | remaining native/final evidence |
|---|---|---|---|
| AC1 | `/private/tmp/proxima-windows-evidence/gate-msvc-sourcev3/compile.json`; `native-vm/crt-runtime-final-root-build.log`; current `cargo xwin check` output | 47 workspace members; 565 artifact packages; 0 missing/excluded members; 0 compiler errors; native and fresh cross-target workspace checks exited 0 | run the updated direct-command Windows workflow |
| AC2 | `/private/tmp/proxima-windows-evidence/native-vm/gate-native-crt-runtime-final-raw/gate-native-crt-runtime-final/runtime.json`; raw groups in its sibling `*-tests.stdout` files | source-v3, Windows Server 2025 AMD64, Wine export absent; 48 passed; 0 failed/ignored/missing; 31 recorded command return codes 0 | run updated full-workspace native workflow |
| AC3 | `/private/tmp/proxima-windows-evidence/gate-surface-sourcefinal/surface.json` | 4 required supported entries; 0 missing sources; 0 unclassified changed boundaries | final native surface mode |
| AC4 | `.github/workflows/windows.yml` | one native Windows pull-request job; direct Cargo commands; no Python setup or gate | execute the updated workflow |
| AC5 | `/private/tmp/proxima-windows-evidence/gate-host-sourcefinal-nextest/host.json` | 38 selected host tests; 0 failed/ignored/missing; host workspace all-target compile record has 0 compiler errors | final native runtime evidence is tracked under AC2 |
| AC6 | `/private/tmp/proxima-windows-evidence/native-vm/control-native-final.json` | 5 invalid fixtures rejected; 0 accepted, including corrupted payload | none |
| AC7 | `/private/tmp/proxima-windows-evidence/native-vm/native-crt-examples-raw/` | host and source-v3 Windows: 21 doctests passed, 0 failed, 14 ignored; 1 example; payload matches incumbent/TCP/UDP/multiworker = 2/1/1/1 | execute through updated Windows workflow |
