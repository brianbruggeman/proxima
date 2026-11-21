# agentic-config-formats -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice. Every cargo command runs as
`timeout -k 30s 25m git wt cargo ...`, `CARGO_BUILD_JOBS=2`, one build at a time. Dependencies change
only through `cargo add` (D6). `<L>` is the proxima-lib feature set defined in SPEC.md, "acceptance
criteria". `<E>` is `<tickets>/PROXIMA-50/evidence`. The conflaguration pin is 2.1.0 on crates.io,
published 2026-11-20 (CONFLAG-7).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | baseline counts at main `fb38d30fa`, before any change: AC8 (a)-(i), the `<L>` compile, the prefix-collision grep, and a read of each C1x type's derives | AC8 | AC8 (a)-(i) with `--list`-free runs, logs to `<E>/baseline-{a..i}.log`; `cargo check -p proxima-lib <L> --lib`; `git grep -c 'fn config_format_\|fn client_config_format_\|fn pipeline_format_\|fn pipeline_xml_text_\|fn mcp_config_format_\|fn mcp_config_xml_text_'` | nine nonzero counts, 0 failed, recorded in this note; `<L>` compiles (`Finished`); the grep prints nothing (0 collisions); the 8 C1x types' `PartialEq` / `Serialize` listed | [ ] | |
| 1 | the dependency bump, one `cargo add` per crate below; then `Cargo.toml:153` and `Cargo.lock` read 2.1.0 | none (entry condition for 2-17) | `git grep -n '^conflaguration = ' Cargo.toml` ; `git wt cargo check -p proxima-config -p proxima-lib -p proxima-intercept --features proxima-lib/tokio` | the grep prints exactly 1 line, `Cargo.toml:153`, with `2.1.0`; `Finished`, 0 errors | [ ] | commands below; record whether `cargo add` bumped the workspace table or the line needed the member form |
| 2 | proxima-config: five std-tier arms and the JSON arm parse through `conflaguration::from_<fmt>_str::<Tree>` + `Value::from`; `toml_to_json` at `:402` goes; `doc.<ext>` fixtures | AC1, AC1b (first half) | `cargo val -p proxima-config --features std --lib -E 'test(config_format_arm_)'` ; `cargo check -p proxima-config --no-default-features --features alloc` | 8 passed; alloc build `Finished` | [ ] | |
| 3 | client config `from_path` reads by extension through the registry; `handle.rs:835` `toml_to_json` goes; `client.<ext>` fixtures | AC2, AC1b | `cargo val -p proxima-lib <L> --lib -E 'test(client_config_format_)'` ; `git grep -c 'fn toml_to_json' -- proxima-config/src src/client` | 7 passed; the grep prints nothing | [ ] | |
| 4 | R4 on proxima-primitives: `lenient` on DelayConfig (4 fields), TransformConfig (8), FilterConfig (4), ValidateConfig inline `Schema` (12); their XML fixtures and 28 buffered tests | AC4 (28 of 33), AC4b (4 of 8) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_buffered_delay_) or test(config_format_buffered_transform_) or test(config_format_buffered_filter_) or test(config_format_buffered_validate_) or test(config_format_ctl_delay) or test(config_format_ctl_transform) or test(config_format_ctl_filter) or test(config_format_ctl_validate)'` | 32 passed (28 + 4) | [ ] | |
| 5 | R4 on the rest: ClientAuthConfig `refresh_ahead_ms` (1), KvCacheConfig and KvFileConfig `max_bytes` (2), ProcessConfig `ready_probe.timeout_ms` (2); 5 buffered tests | AC4 (5 of 33), AC4b (4 of 8) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_buffered_) or test(config_format_ctl_)'` | 41 passed (33 + 8, the whole of AC4 and AC4b) | [ ] | |
| 6 | R3 on proxima-primitives and proxima-patterns: the 10 sites of RetryConfig, TransformConfig, RateLimitConfig, ClientAuthConfig (2), AuthConfig, ValidateConfig, FilterConfig, DelayConfig, IsolateConfig use the D1 expression; 9 fixture pairs and twin tests | AC3 (9 of 33) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_retry_config) or test(config_format_twin_transform_config) or test(config_format_twin_rate_limit_config) or test(config_format_twin_client_auth_config) or test(config_format_twin_auth_config) or test(config_format_twin_validate_config) or test(config_format_twin_filter_config) or test(config_format_twin_delay_config) or test(config_format_twin_isolate_config)'` | 9 passed | [ ] | |
| 7 | R3 on HttpConfig (2 sites), H3NativeConfig, GrpcConfig, SynthConfig (`synth.rs:393`), CallbackConfig; 5 pairs; payload fields as JSON text in both twins (D5) | AC3 (5 of 33) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_http_config) or test(config_format_twin_h3_native_config) or test(config_format_twin_grpc_config) or test(config_format_twin_synth_config) or test(config_format_twin_callback_config)'` | 5 passed | [ ] | |
| 8 | R3 on ProcessConfig, ProcessRpcConfig (`process_rpc.rs:420`), FsConfig, KvCacheConfig (`:367`, `:377`), KvFileConfig (`:540`, `:550`): 7 sites; 5 pairs | AC3 (5 of 33) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_process_config) or test(config_format_twin_process_rpc_config) or test(config_format_twin_fs_config) or test(config_format_twin_kv_cache_config) or test(config_format_twin_kv_file_config)'` | 5 passed | [ ] | |
| 9 | R3 on StreamPassthroughSettings, VmConfig, RecordConfig and the seven client configs (`mqtt.rs:64`, `dns.rs:76`, `pgwire.rs:166`, `amqp.rs:64`, `redis.rs:69`, `memcached.rs:65`, `kafka.rs:70`): 10 sites; 10 pairs | AC3 (10 of 33) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_stream_passthrough_settings) or test(config_format_twin_vm_config) or test(config_format_twin_record_config) or test(config_format_twin_mqtt_client_config) or test(config_format_twin_dns_resolver_config) or test(config_format_twin_pg_client_config) or test(config_format_twin_amqp_client_config) or test(config_format_twin_redis_client_config) or test(config_format_twin_memcached_client_config) or test(config_format_twin_kafka_client_config)'` | 10 passed | [ ] | |
| 10 | R3 on StageSpec (`load.rs:914`), ConfiguredTool (`mcp.rs:344`), `[[schema]]` (`app.rs:1186`), ProximaSettings (`settings/mod.rs:153`); 4 pairs; `config_format_members` (33 names, 33 + 33 fixture files) | AC3 (4 of 33, and the members test) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_) or test(config_format_members)'` | 34 passed (33 twins + 1 members) | [ ] | |
| 11 | R3 on InterceptConfig (`proxima-intercept/src/pipe.rs:1419`): 1 site, 1 pair | AC3b | `cargo val -p proxima-intercept --features intercept-config --lib -E 'test(config_format_twin_)'` | 1 passed | [ ] | |
| 12 | R3 on the seven server override reads (C1x), 7 sites, 7 pairs, in one commit because each is the same one-line change | AC3c, AC3d | for `<c>` in mqtt amqp kafka memcached redis dns pgwire: `cargo val -p proxima-<c> --features listen --lib -E 'test(config_format_twin_)'` ; `git grep -n 'and_then(from_tree::<' -- . ':!*tests*' ':!*/tests/*' \| wc -l` | 1 passed in each of 7 crates; the grep prints `45` | [ ] | |
| 13 | R5 in `src/load.rs`: middleware (`:1095`), miss_on (`:1133`), write_back (`:1168`), list_mode (`:657`), weight (`:812`) get absent / XML / wrong-shape; upstreams (`:783`) and write_back index (`:1206`) accept the XML form | AC5 (17 of 36) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_read_middleware) or test(config_format_read_miss_on) or test(config_format_read_write_back) or test(config_format_read_list_mode) or test(config_format_read_weight) or test(config_format_read_upstreams) or test(config_format_read_write_back_index)'` | 17 passed (5 x 3 + 2) | [ ] | the error texts are the SPEC table after the acceptance criteria |
| 14 | R5 in `src/app.rs`: schema (`:1173`), pipe (`:1194`), methods (`:1307`), host (`:1317`), chain (`:1693`) get absent / XML / wrong-shape; listen (`:1214`) and mount (`:1275`) accept the XML form | AC5 (17 of 36) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_read_schema) or test(config_format_read_pipe) or test(config_format_read_methods) or test(config_format_read_host) or test(config_format_read_chain) or test(config_format_read_listen) or test(config_format_read_mount)'` | 17 passed (5 x 3 + 2) | [ ] | |
| 15 | R5 in `src/listeners/mcp.rs:326` tools and `proxima-primitives/src/pipe/interval_pipe.rs:184` period_ms accept the XML form | AC5 (2 of 36, whole of AC5 after this) | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_read_)'` | 36 passed (17 + 17 + 2) | [ ] | |
| 16 | R6 pipeline: `triage.{yaml,json5,ron,xml}`; XML carries its two payloads as JSON text (D4); 8 case tests and the text-leaf test | AC6 (pipe), AC7 (pipe) | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_format_) or test(pipeline_xml_text_)'` | 9 passed (8 + 1) | [ ] | |
| 17 | R6 CLI: `review_ticket.{yaml,json5,ron,xml}`; XML carries its 6 payloads as JSON text; 4 session tests and the text-leaf test | AC6 (cli), AC7 (cli) | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_format_) or test(mcp_config_xml_text_)'` | 5 passed (4 + 1) | [ ] | |
| 18 | incumbents and lints | AC8, AC9 | AC8 (a)-(i); AC9 (clippy, then the control) | each count equals its slice-0 count plus this spec's new tests in that file, 0 failed; clippy prints `exit=0`, `0`, `1`; control prints `2` | [ ] | |

Slice 1 commands, run from the repo root; each states the crate and its own features (the features
restate what the crate had, `derive` and `toml`, so the pin keeps them; `json` is added at every site
crate for `conflaguration::Error: From<serde_json::Error>`, `src/lib.rs:409-411`; `xml` where a test
reads XML):

```
cargo add conflaguration@2.1.0 --package proxima-config --optional --features toml,json,yaml,xml,ron,json5
cargo add conflaguration@2.1.0 --package proxima-lib --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-http --features derive,toml,json
cargo add conflaguration@2.1.0 --package proxima-primitives --optional --features derive,toml,json
cargo add conflaguration@2.1.0 --package proxima-patterns --optional --features derive,toml,json
cargo add conflaguration@2.1.0 --package proxima-intercept --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-core --optional --features derive,toml
cargo add conflaguration@2.1.0 --package proxima-mqtt --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-amqp --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-kafka --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-memcached --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-redis --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-dns --optional --features derive,toml,json,xml
cargo add conflaguration@2.1.0 --package proxima-pgwire --features derive,toml,json,xml
```

`proxima-config` is new to conflaguration: after its `cargo add --optional`, `std = [...]`
(`proxima-config/Cargo.toml:18-27`) gains `"dep:conflaguration"` by hand (a feature-list edit, D6).
proxima-pgwire (`Cargo.toml:26`) is a non-optional dependency, so it takes no `--optional`.

## resume

Last landed slice: none (spec drafted, ready for the spec-auditor)
Next action: run the `spec-auditor` agent on this folder; then slice 0 (the AC8 baselines at main `fb38d30fa`), which needs no conflaguration release
Open question, if any: none; slice 1 waits on conflaguration 2.1.0 on crates.io (owner's go)

Keep these three lines current. They are what a fresh conversation reads instead of
re-deriving the session.

## struck

-
