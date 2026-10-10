# Card 21 bounded nextest compile attempt

Command:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= timeout 180 cargo nextest run -p proxima-model-interop --features std,metal,metal-attn-split-rows,metal-attn-variants --lib -E 'test(simdgroup_count_setting_round_trips_and_reaches_serving_config)' -- --nocapture
```

The process compiled workspace crates serially, including `prime`, `proxima-protocols`, `proxima-net`, `proxima-tensor`, `omega`, `proxima-process`, `proxima-recording`, `proxima-http`, `proxima-patterns`, `proxima-test`, and `proxima`. It reached the 180-second timeout before reporting test execution. The copied library-test executable had separately run the exact setting test and reported 1 passed, 0 failed, 734 filtered out in `card21-serving-setting-tests-2026-10-10.log`. Nextest discovery was independently measured by the copied binary and cached `cargo nextest list`; see `card21-nextest-list-2026-10-10.log`.

A second uncapped run after this compile populated the target built the focused test in 42.18 seconds and reported `1 test run: 1 passed, 727 skipped`. Its full output is in `card21-nextest-setting-test-2026-10-10.log`; its discovery-process sample is in `card21-nextest-dyld-sample-2026-10-10.txt`.
