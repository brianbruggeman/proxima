# Card 18 configuration test retry and loader trace

The configuration assertion was already executed in the Card 12 revalidation: `serving_settings::tests::simdgroup_count_setting_round_trips_and_reaches_serving_config` reported `1 passed; 0 failed; 727 filtered out`. Its assertions cover builder/TOML/environment parity for Groups2/4/8, Legacy default with no explicit attention variant, lowering each explicit count, Legacy MMA precision, and rejection of `groups16` (`evidence/card12-revalidation-2026-10-10.md:22-25,66`). Card18's six Granite process runs independently exercised `ServingSettings::from_env` through serving lowering and captured the selected count-specific entry.

A fresh retry used:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,metal-attn-variants --lib simdgroup_count_setting_round_trips_and_reaches_serving_config -- --exact --nocapture
```

Cargo compiled the test binary in 18.62 seconds, then the runner under `/tmp/cargo_target/debug/deps` remained in startup without printing a test result. The same executable copied to `/tmp/card18-lib-tests` also remained in startup. It was then copied to `/tmp/cargo_target/debug/deps/card18-runner`, where a 4-second `sample` observed 3,564 main-thread samples at `_dyld_start`, 96 KB physical footprint, and no binary images; the sample is `/tmp/card18-lib-test.sample`. The process had not reached the Rust test harness. That attempt was interrupted. `sample` reported it could not find `_sigtramp` in expected dylibs; that diagnostic does not identify the dyld operation causing the stall.

The earlier child-directory test execution and path probes are retained in `evidence/card12-revalidation-2026-10-10.md` and `evidence/card16-cargo-target-dyld-sample-2026-10-10.txt`. The cause of this launch's dyld stall remains unidentified. No pass is attributed to this retry; the prior recorded config test result is the execution evidence.
