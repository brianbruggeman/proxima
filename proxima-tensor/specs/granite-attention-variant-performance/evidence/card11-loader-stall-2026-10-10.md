# Card 11 focused test launch record

The focused test execution did not report a test result. `cargo test` built the test binary, printed `Running unittests`, then remained without test output. The observed process was at 0.0% CPU with a 96 KB physical footprint. `sample` recorded the main thread at `_dyld_start` for all 771 samples and no application frames (`/tmp/omega-ed45597f1f927059_2026-10-10_073635_v62o.sample.txt`). The process was interrupted after the sample; it was no longer present in `ps` afterward.

After the final dispatch changes, `cargo test -p omega --features metal,metal-attn-variants,metal-attn-split-rows --lib --no-run` rebuilt the test binary successfully. This verifies test compilation only; it does not record execution results.

The feature build `cargo check -p omega --features metal,metal-attn-variants,metal-attn-split-rows` completed successfully after the schedule was threaded through partial uniforms, merge admission and generation, merge uniforms, scratch sizing, and the one-split output path. `git diff --check` returned no diagnostics.

No Metal dispatch or GPU benchmark ran during this card.

A bounded direct launch of the already-built native test binary with `DYLD_PRINT_LIBRARIES=1` was run for the one-split admission test. It timed out after 25 seconds with zero stdout/stderr lines; the child was killed and its transcript is `/tmp/card11-native-dyld-libraries.log`. The x86_64 Rosetta route was checked with `arch -x86_64 /usr/bin/true` (exit 0), then its serialized Cargo build was stopped during first-time dependencies before a test binary existed; it had reached `aws-lc-sys`/project dependency compilation. No x86_64 test result was produced.

After updating the emitted scratch/direct-output branch and the source assertion, the final checks completed:

- `cargo check -p omega --features metal,metal-attn-variants,metal-attn-split-rows`: exit 0.
- `cargo test -p omega --features metal,metal-attn-variants,metal-attn-split-rows --lib --no-run`: exit 0 and rebuilt the unit-test binary.
- `git diff --check`: no diagnostics.

The original test binary path still does not enter the harness. A copied
binary at `/tmp/omega-card11-tests-final` did enter the harness and ran the
focused cases successfully; this is recorded below. The original launch
failure remains unexplained.

After adding a source-level composition assertion for F16 MMA, SharedKv,
Rows16, SimdgroupRows, and Groups4, the scoped unit-test target compiled with:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p omega --features std,metal,metal-attn-split-rows,metal-attn-variants --lib --no-run
```

It exited 0 and produced `/tmp/cargo_target/debug/deps/omega-6a0a2fca2e15db9c`.
No test process was launched, so this records compilation only. A broader
`cargo check --tests` was interrupted after it began compiling every omega
integration-test target; it was not used as acceptance evidence.

A direct launch of the newly built test binary for only
`msl::attn_rows_tests::explicit_four_simdgroup_dispatch_preserves_row_and_key_tiles`
was bounded by a 15-second process alarm. At 13 seconds the process was in
state `SN`, at 0.0% CPU, with 32 KB RSS and no output; the alarm killed it
(exit 142) before the harness reported a test result. `file` identifies the
binary as Mach-O arm64, `codesign --verify --verbose=2` reports it valid, and
`xattr -l` shows `com.apple.provenance` without a quarantine attribute. This
narrows the repeated stall to launch/runtime behavior but does not identify
its cause. No test result is claimed.

After correcting the test fixtures and assertions, the unit-test binary was
rebuilt (exit 0) and copied to `/tmp/omega-card11-tests-final`. Running the
five exact selectors from the acceptance command individually against that
copy produced `focused_tests=5 passed=5 failed=0`. All five are CPU-only
dispatch/source/uniform checks; no Metal replay or timing was run. The test
binary under `/tmp/cargo_target/debug/deps` continued to stall at launch under
a 15-second alarm, while its copy entered the harness in under one second.
The path-dependent startup difference is observed; its cause is unexplained.

The passing selectors were:

- `msl::attn_rows_tests::explicit_four_simdgroup_dispatch_preserves_row_and_key_tiles`
- `msl::attn_rows_tests::explicit_simdgroup_count_rejects_a_non_row_tiled_shape`
- `msl::attn_rows_tests::explicit_simdgroup_count_updates_merge_admission_at_one_split`
- `metal::arena_encode_dispatch_finish::attention_scratch_len_tests::explicit_simdgroup_count_keeps_partial_scratch_and_both_uniforms_aligned`
- `msl::attn_rows_tests::card_20_dispatch_matrix_exposes_legacy_and_eight_one_factor_flips`

The feature-only check
`CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check -p omega --features metal,metal-attn-variants,metal-attn-split-rows`
also exited 0 after the final source changes. `git diff --check` returned no
diagnostics.


## Path-context diagnosis follow-up

A later bounded probe compared the same executable bytes in the Cargo `deps` directory and child directories. Direct executables in `/private/tmp/cargo_target/debug/deps` stalled in `--list` and `codesign --verify`; the identical executable in a new child directory under that target completed `--list` in 0.427 seconds and code-sign verification in 0.063 seconds. A fresh `/private/tmp` child completed `--list` in 0.455 seconds. `/tmp` resolves to `/private/tmp`, all locations are on the same device, and the `deps` directory had 65,535 entries. This isolates the direct flat `deps` directory context as the trigger; the dyld/macOS sub-operation remains unexplained. Detailed command output is recorded in `evidence/card12-revalidation-2026-10-10.md`.
