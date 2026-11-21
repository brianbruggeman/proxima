# agentic-config-formats -- every config kind in any format, through conflaguration

status: draft, ready for audit; conflaguration 2.1.0 published to crates.io 2026-11-20 (`5bb4b91`, tag `v2.1.0`)
owner: brian
created: 2026-11-20

Owner, 2026-11-20: "conflaguration should let us do anything: json, toml, yaml, xml, etc."; the
owner chose conflaguration as the single format layer ("conflaguration is nested config using
Setting"). PROXIMA-50. Follows `agentic-config-payloads` (PROXIMA-49, landed), which makes step
payloads values. Every `file:line` is in proxima at main `fb38d30fa` (the PROXIMA-49 landing), read
2026-11-20; the PROXIMA-49 censuses were taken at `285b519e5` and are re-cited where a line moved
(`synth.rs` 366 -> 393, `process_rpc.rs` 409 -> 420; every other cited line is unchanged).
conflaguration lines are in `/Users/brianbruggeman/repos/slot-0/conflaguration-publish` at `f7d698e`
(its `xml-lenient` spec); the 2.1.0 release is `5bb4b91`, `f7d698e` plus the version bump in
`Cargo.toml` / `Cargo.lock` only (CONFLAG-7). `<tickets>` is
`/Users/brianbruggeman/repos/obsidian/slot-0/slot-0/20 - Collaboration and Records/22 - Investigations/22.1 - Tickets`;
the PM topic is `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`, "Split".

## problem

proxima parses six formats itself (`proxima-config/src/lib.rs:262-392`) into `serde_json::Value`,
then reads configs from that value strictly: XML keeps one of each repeated element and gives every
scalar as text, so an XML config loses pipes, stages and tools or fails on its first number; and a
second TOML converter serves the client config (`src/client/handle.rs:176-186,835`) -- measured by
every config type, hand-read key and end-to-end fixture loading the same from XML as from TOML.

## refutation condition

If a config type cannot load from XML without changing its TOML / JSON shape (beyond a field
attribute), or without proxima keeping a parser conflaguration also has, the single-layer frame is
wrong and this spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | proxima-config's six `ConfigFormatFactory::parse` arms parse through conflaguration (`from_<fmt>_str::<Tree>`, then `serde_json::Value::from`), keeping the YAML / XML top-level checks (`:332-344,380-392`); `toml_to_json` (`:402`) goes; serializers stay (conflaguration does not write) | yes |
| R2 | the client config's `from_path` (`src/client/handle.rs:176-186`) reads any format through the registry by extension; `handle.rs:835` `toml_to_json` goes | yes |
| R3 | every non-test typed read of a config tree goes through `conflaguration::from_tree` (the exact expression is in the architecture): the 33 types of census C1 plus the 8 members C1 missed (C1x), 45 sites, so XML text scalars and one-element lists load into typed fields | yes |
| R4 | each buffered non-string field (census C2, 33 fields in 8 types) carries conflaguration's `lenient::deserialize` / `lenient::option` | yes |
| R5 | each hand read of a list or scalar on the load path (census C3, 14 list reads and 5 scalar reads) accepts XML's form: a single element where a list is read, text where a number or bool is read; the 10 silent skips / defaults among them keep their default when the key is absent and become errors that name the key when it is present in the wrong shape | yes |
| R6 | the triage pipeline and the CLI review config, written in YAML, JSON5, RON and XML (beside the JSON and TOML twins that exist), answer the same as the JSON forms; the XML twins carry the free-form payloads as JSON text (decision D4) | yes |
| R7 | incumbents: every existing config-loading test passes with its slice-0 count; a config that is not XML loads to the same value as before (JSON duplicate keys, TOML datetimes and `inf` / `nan` named as the only differences, per conflaguration R5) | yes |

## census

C1 -- typed config reads (type | kind | deserialize site), at main `fb38d30fa`:
HttpConfig http `proxima-http/src/http1/upstream.rs:309`, `prime_upstream.rs:76`; H3NativeConfig
h3-native `src/upstreams/h3_native.rs:56`; GrpcConfig grpc `grpc_h2.rs:118`; SynthConfig synth
`synth.rs:393`; CallbackConfig callback `callback.rs:139`; ProcessConfig process `process.rs:689`;
ProcessRpcConfig process_rpc `process_rpc.rs:420`; FsConfig fs `fs.rs:225`; KvCacheConfig kv:cache
`kv_cache.rs:367`; KvFileConfig kv:file `kv_file.rs:540`; StreamPassthroughSettings stream
`stream_passthrough.rs:296`; VmConfig vm `vm.rs:769`; RecordConfig record `record.rs:925` (its
`SinkConfig` is a field of it, not a second read); InterceptConfig intercept
`proxima-intercept/src/pipe.rs:1419`; the seven client configs, all in proxima-lib's `src/upstreams/`:
MqttClientConfig `mqtt.rs:64`, DnsResolverConfig `dns.rs:76`, PgClientConfig `pgwire.rs:166`,
AmqpClientConfig `amqp.rs:64`, RedisClientConfig `redis.rs:69`, MemcachedClientConfig
`memcached.rs:65`, KafkaClientConfig `kafka.rs:70`; StageSpec pipeline `src/load.rs:914`; RetryConfig
`proxima-primitives/src/pipe/retry.rs:720`; TransformConfig `transform.rs:262`; RateLimitConfig
`rate_limit.rs:381`; ClientAuthConfig `proxima-patterns/src/middleware/client_auth.rs:51,399`;
AuthConfig `auth.rs:42`; ValidateConfig `validate.rs:224`; FilterConfig `filter.rs:477`; DelayConfig
`delay.rs:214`; IsolateConfig `isolate.rs:70`; plus ConfiguredTool `src/listeners/mcp.rs:344` and
`[[schema]]` blocks `src/app.rs:1186`. That is 33 types, 35 sites (the draft's "31 types" is the
first 31; ConfiguredTool and Schema are the other two).

C1x -- typed reads the PROXIMA-50 census missed, found at main (each is a non-test
`serde_json::from_value` on a config tree): `src/upstreams/kv_cache.rs:377` (`build_kv_cache`, a
second site for KvCacheConfig) and `kv_file.rs:550` (`build_kv_file`, a second site for KvFileConfig);
`src/settings/mod.rs:153` (`ProximaSettings`, read from `registry.parse_with_hint`, `:151-153`); and
the seven server-side override reads in the client crates, each `spec.get("<proto>")` then
`serde_json::from_value(overrides.clone())` in a private `resolve_config` /
`resolve_<x>_config`: `proxima-mqtt/src/any_protocol.rs:112` (MqttServerConfig),
`proxima-amqp/src/any_protocol.rs:83` (AmqpServerConfig), `proxima-kafka/src/any_protocol.rs:89`
(KafkaServerConfig), `proxima-memcached/src/any_protocol.rs:124` (MemcachedServerConfig),
`proxima-redis/src/any_protocol.rs:82` (RedisServerConfig), `proxima-dns/src/config.rs:30`
(DnsServerConfig), `proxima-pgwire/src/spec.rs:29` (PgServerConfig). The 7 server crates hold the
only C1 sites that are in client crates; the seven client configs above are read in proxima-lib.
C1 + C1x = 33 + 8 = 41 members and 35 + 10 = 45 sites. Generic layer helpers
(`src/app_config.rs:430`, `src/cassette_config.rs:647`, `proxima-telemetry/src/config_merge.rs:137`,
`proxima-process/src/command_config.rs:801`) read the output of conflaguration's own layered
builder, which since `84b7a26` reads XML leniently; they are out of scope.

C2 -- buffered non-string fields (33): ClientAuthConfig `scheme(oauth).refresh_ahead_ms` (tag
`client_auth.rs:68`, flatten `:109`) 1; DelayConfig `dist(const).ms`, `dist(range).min_ms`, `max_ms`,
`seed` (`delay.rs:52`) 4; TransformConfig request and response `mutate` `bits`, `seed`, `when.prob`,
`when.seed` (`transform.rs:135,149,186,196`, `mutate.rs:15,80,84`) 8; FilterConfig
`predicate(when | unless).gate.prob`, `seed` (`filter.rs:69,80,88`) 4; ValidateConfig inline `Schema`
`int.min/max`, `uint.min/max`, `float.finite`, `string.min_len/max_len`, `bytes.max_len`,
`seq.min_items/max_items`, `struct.fields[].flags.optional/deprecated` (`validate.rs:236`,
`proxima-config/src/schema/mod.rs:47`) 12; KvCacheConfig and KvFileConfig `max_bytes(bytes)`
(`kv_cache.rs:223`) 2; ProcessConfig `ready_probe(stdout_line | tcp).timeout_ms` (`process.rs:494`) 2.

C3 -- hand reads: lists `src/load.rs:783` upstreams (errors), `:1095` middleware (skips), `:1133`
miss_on (defaults), `:1168` write_back (defaults); `src/app.rs:1173` schema (skips), `:1194` pipe
(skips), `:1214` listen (errors), `:1275` mount (errors), `:1307` methods (defaults), `:1317` host
(defaults), `:1693` chain (drops); `src/listeners/mcp.rs:326` tools (errors); unaffected:
`load.rs:1182`, `:882`. Scalars: `load.rs:657` list_mode (silently false), `:812` weight (silently
1), `:1206` write_back index (errors), `kv_cache.rs:384` max_bytes (handled),
`proxima-primitives/src/pipe/interval_pipe.rs:184` period_ms (errors). Confirmed at main: 8 silent
list reads (`:1095`, `:1133`, `:1168`, `:1173`, `:1194`, `:1307`, `:1317`, `:1693`) and 2 silent
scalar reads (`:657`, `:812`) = 10 sites that skip, default, drop or silently fall back; the 9 other
sites (783, 1214, 1275, 326, 1206, interval 184 error; 1182, 882, kv 384 need nothing) are not
silent.

## measured for the audit (PROXIMA-50 sweep, 2026-11-20, rechecked at main `fb38d30fa`)

- Every crate holding a C1 or C1x site already depends on `conflaguration` directly with features
  `["derive", "toml"]`: proxima-lib `Cargo.toml:381`, proxima-http `:61` (both non-optional);
  proxima-primitives `:144`, proxima-patterns `:133`, proxima-intercept `:33`, and the seven client
  crates (dns `:31`, pgwire `:26`, mqtt `:23`, amqp `:22`, redis `:23`, memcached `:23`, kafka `:36`),
  optional but enabled by the features their factories need (`listen`). All inherit one version from
  the workspace table, `Cargo.toml:153` (`version = "2.0.0"`). The one crate they all share is
  `proxima-core`, whose optional `config` feature already pulls conflaguration
  (`proxima-core/Cargo.toml:64,73`). proxima-config does not depend on conflaguration and is shared
  by only proxima-lib and proxima-primitives.
- Every C1 site maps `serde_json::Error` to `ProximaError::Config("<kind> config: {err}")` on the
  next line (`<kind>` per site, e.g. `src/upstreams/synth.rs:394`, `proxima-primitives/src/pipe/retry.rs:721`);
  the three non-factory sites read `pipeline \`stages\`: ` (`src/load.rs:915`), `mcp tools[{index}]: `
  (`src/listeners/mcp.rs:345`), `[[schema]] \`{name}\` decode: ` (`src/app.rs:1187`); the seven server
  reads map to `"<proto> spec: {error}"`; `src/settings/mod.rs:154` maps to `decode ProximaSettings: `.
- Comparison per C1 type: 15 derive `PartialEq` (StreamPassthroughSettings, VmConfig, the seven
  client configs, StageSpec, DelayConfig, ConfiguredTool, Schema, the recording SinkConfig; FilterConfig
  has a manual impl over `predicate` and `on_reject` only, `filter.rs:354-355`), the rest do not; all
  derive `Serialize` except `ConfiguredTool` (`src/listeners/mcp.rs:295`, `Deserialize` only, which
  has `PartialEq`). So a twin comparison is typed equality where `PartialEq` covers every field, and
  `serde_json::to_value` equality otherwise. Whether the 8 C1x types derive either is read by slice 0.
- conflaguration at `f7d698e` (read, not assumed), `<c>` = `/Users/brianbruggeman/repos/slot-0/conflaguration-publish`:
  `Tree` implements `Deserialize` (`src/tree.rs:36-40`), so `serde_json::Value`, which is a
  `Deserializer` with `Error = serde_json::Error`, goes to `Tree` with
  `serde_json::from_value::<Tree>(value)`; `from_tree::<T>(tree: Tree) -> Result<T, conflaguration::Error>`
  is `Ok(T::deserialize(tree.into_lenient())?)` (`src/file.rs:264-266`), re-exported at
  `src/lib.rs:1243`; `Tree` at `:1233-1237`, `pub mod lenient` at `:1239-1240`, all gated on a format
  feature (`toml`, `json`, `yaml`, `xml`, `ron`, `json5`) plus `std`. `conflaguration::Error`
  has `From<serde_json::Error>`: `Json(#[from] serde_json::Error)` at `src/lib.rs:409-411`, behind
  feature `json`, displayed `json: {0}`; `from_tree`'s own failure is
  `ValueDeserialize(#[from] serde_value::DeserializerError)` at `src/lib.rs:419-421`, displayed
  `value deserialize: {0}`. `From<Tree> for serde_json::Value` is `src/tree.rs:149-154`, behind
  feature `json`. `lenient::deserialize` and `lenient::option` are `src/lenient.rs:96-101,104-109`.
  `from_toml_str` / `from_json_str` / `from_yaml_str` / `from_xml_str` / `from_ron_str` /
  `from_json5_str` are `src/file.rs:12-14,18-20,24-26,45-47,51-53,57-59`, each generic over
  `DeserializeOwned`, so `Tree` is a valid `T`.

## architecture

Reuse map: conflaguration's `Tree`, `from_tree`, the six `from_<fmt>_str` readers and `lenient`
(above); proxima's format registry (`proxima-config/src/lib.rs:262-392`, `default_config_format_registry`);
the twin fixtures PROXIMA-49 landed (`tests/fixtures/pipeline_pipe/triage.toml`,
`tools/proxima-cli/tests/fixtures/mcp_config_tool/review_ticket.toml`) with their rigs
(`tests/pipeline_pipe.rs`, `tools/proxima-cli/tests/integration/mcp_config_tool.rs`, helpers
`check_seen`, `json_text_leaves`, `check_triage`).

The expression every C1 / C1x site uses to go from the `spec: serde_json::Value` it already holds to
`T`, and the error type of each step:

```rust
use conflaguration::{Tree, from_tree};

let config: T = serde_json::from_value::<Tree>(spec.clone())   // Result<Tree, serde_json::Error>
    .map_err(conflaguration::Error::from)                      // Error::Json, src/lib.rs:409-411
    .and_then(from_tree::<T>)                                  // Result<T, conflaguration::Error>, file.rs:264
    .map_err(|err| ProximaError::Config(format!("<kind> config: {err}")))?;   // the site's own line
```

Two error types meet in one `map_err` because `conflaguration::Error: From<serde_json::Error>`, which
needs conflaguration's `json` feature in each site crate (the slice-1 `cargo add` lines add it). The
site's existing message keeps its prefix; the cause gains `json: ` (a `Value` that is not a `Tree`)
or `value deserialize: ` (a field that is not `T`), and the text of the serde error follows.

R1 is the registry side of the same pipe: each arm becomes
`serde_json::Value::from(conflaguration::from_<fmt>_str::<Tree>(raw)?)` with the existing
`ProximaError::Config("<fmt>: {err}")` mapping.

R4 field shape: `#[serde(deserialize_with = "conflaguration::lenient::deserialize")]` on a buffered
field, `lenient::option` on an `Option` field (`default` stays where it is today).

R5 helpers, private to `src/load.rs` / `src/app.rs` / `src/listeners/mcp.rs` (three sites of reuse
inside proxima-lib; no shared module, the same reason as D1): a list read takes an array, or a
present non-array value as one element (XML's one-element form), a `Null` or absent key as absent;
a scalar read takes a typed value or text that parses (`"3"`, `"true"`).

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| D1 where the Value -> `T` call lives | each C1 / C1x site calls conflaguration directly, no shared helper | every site crate already depends on conflaguration directly (measured above) and already maps its own error on the next line; a helper in `proxima-core`'s `config` feature would add a `proxima-core/config` feature edge to 12 crates for one expression |
| D2 R3 fixtures | one TOML and one XML fixture per member (41 pairs), every non-string field set, compared as the comparison bullet says; one test per crate that reaches the members (proxima-lib 33, proxima-intercept 1, one in each of 7 client crates) | proxima-lib's `Cargo.toml` reaches 32 of the 33 C1 types (proxima-http `:300`, proxima-primitives `:295`, proxima-patterns `:312`, `proxima-recording` `:393`, the seven client crates behind `<proto>-client` features `:665-939`) and `ProximaSettings`, but no `proxima-intercept` dependency exists anywhere outside that crate (`Cargo.toml:48` is the workspace member list; `proxima-intercept/Cargo.toml:41-42` has proxima-lib only as a dev-dependency), and the seven server reads sit in private fns of crates that proxima-lib reaches only through `<proto>-listener` features; one test per crate is the only layout that exercises every member without a feature set no crate enables today |
| D3 R5 silent sites | absent key keeps its default; present key in the wrong shape is an error naming the key, in the style of the error nearest the site (`` `miss_on` entries must be strings ``, `load.rs:1136`; `` `upstreams` must be an array ``, `:785`) | keeping the silent skip would let an XML typo load a pipeline with no middleware; making an absent key an error would break every existing config |
| D4 XML free-form payloads | `StageRequest.body`, `ProcessRpcInitFrame.send`, `SynthConfig.body` and `SynthConfig.body_template` (all `serde_json::Value`) are written in XML as JSON text; a string keeps the meaning PROXIMA-49 gave it | the target is `Value`, so lenient reading has no type to aim at: `<id>1</id>` would arrive as `"1"` and a one-element list as a scalar, silently changing the payload |
| D5 JSON text in TOML twins of C1 fixtures | the 4 payload fields in a C1 twin pair are JSON-text strings in both the TOML and the XML fixture | a TOML table next to XML text would differ by design (D4); the PROXIMA-49 table form is covered by its own ACs |
| D6 how a dependency changes | `cargo add conflaguration@2.1.0 --features ...` per crate (TASKS slice 1); no hand edit of a `[dependencies]` line | repo rule: dependencies change only through `cargo add`; a `[features]` list edit (`std = [..., "dep:conflaguration"]`) is not a dependency edit |
| D7 JSON arm of the registry | all six arms move under `std`; the alloc-tier `JsonConfigFormat` (`proxima-config/Cargo.toml:14-16`, serde_json only) is unchanged | conflaguration is std-only (`src/lib.rs:1233`), so the alloc build cannot reach it |

## acceptance criteria

Runner `cargo val`, always `--lib` or `--test`, each run wrapped `timeout -k 30s 25m git wt cargo
val ...`, `CARGO_BUILD_JOBS=2`, one build at a time. `<L>` is the proxima-lib feature set
`--features tokio,http1,http-prime,http2,h3-native-upstream,recording,vm,amqp-client,dns-client,kafka-client,memcached-client,mqtt-client,pgwire-client,redis-client`
(the cfg gates of `src/upstreams/mod.rs:1-76`; slice 0 compiles it and records the result).
Test-name prefixes: `config_format_arm_` (proxima-config), `client_config_format_` (lib),
`config_format_twin_` and `config_format_members` (each crate named below),
`config_format_buffered_` and `config_format_ctl_` (lib), `config_format_read_` (lib hand reads),
`pipeline_format_` and `pipeline_xml_text_` (`tests/pipeline_pipe.rs`), `mcp_config_format_` and
`mcp_config_xml_text_` (CLI integration). None is a substring of an existing name in its file; slice
0 asserts `git grep -c 'fn config_format_\|fn client_config_format_\|fn pipeline_format_\|fn pipeline_xml_text_\|fn mcp_config_format_\|fn mcp_config_xml_text_'` prints 0 for each touched file (the one place a 0 is the expected count).

The 41 members, by crate. Fixture pair = `<dir>/<name>.toml` and `<dir>/<name>.xml`; `<dir>` is
`tests/fixtures/config_formats/types` for proxima-lib, `<crate>/tests/fixtures/config_formats` for
the others. Each fixture sets every non-string field of the type (D5 for payload fields).

| crate | members (name = snake_case of the type) | n |
|---|---|---|
| proxima-lib | http_config, h3_native_config, grpc_config, synth_config, callback_config, process_config, process_rpc_config, fs_config, kv_cache_config, kv_file_config, stream_passthrough_settings, vm_config, record_config, mqtt_client_config, dns_resolver_config, pg_client_config, amqp_client_config, redis_client_config, memcached_client_config, kafka_client_config, stage_spec, retry_config, transform_config, rate_limit_config, client_auth_config, auth_config, validate_config, filter_config, delay_config, isolate_config, configured_tool, schema, proxima_settings | 33 |
| proxima-intercept | intercept_config | 1 |
| proxima-mqtt, -amqp, -kafka, -memcached, -redis, -dns, -pgwire | mqtt_server_config, amqp_server_config, kafka_server_config, memcached_server_config, redis_server_config, dns_server_config, pg_server_config | 7 |

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-config --features std --lib -E 'test(config_format_arm_)'` | 8 passed: 6 formats (toml, json, yaml, json5, ron, xml) each parse `proxima-config/tests/fixtures/formats/doc.<ext>` (one logical document with a number, a bool, a nested table and a 2-element list) to the same expected `Value`, XML with its repeated element a list and its scalars text; plus the YAML and XML top-level rejections (`:332-344,380-392`) with their existing reasons |
| AC1b | R1, R2 | `git grep -c 'fn toml_to_json' -- proxima-config/src src/client` ; `git grep -c 'conflaguration::from_.*_str' -- proxima-config/src/lib.rs` | first prints nothing and exits 1 (slice 0: 2 matches, `lib.rs:402`, `handle.rs:835`); second prints `6` |
| AC2 | R2 | `cargo val -p proxima-lib <L> --lib -E 'test(client_config_format_)'` | 7 passed: `ClientConfig::from_path` over `tests/fixtures/client_config/client.<ext>` for toml, json, yaml, json5, ron, xml gives one equal `ClientConfig` (6), and a path with extension `.cfg` returns the registry's unknown-format error (1) |
| AC3 | R3 | (lib) `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_)'` and `... -E 'test(config_format_members)'` | 33 passed (one per lib member, a failure names the member) and 1 passed (`config_format_members` asserts its const list holds 33 names and the fixture directory holds exactly 33 `.toml` and 33 `.xml`); each twin test loads its TOML and XML fixture through `Tree` and the D1 expression and asserts equality |
| AC3b | R3 | (intercept) `cargo val -p proxima-intercept --features intercept-config --lib -E 'test(config_format_twin_)'` | 1 passed (`intercept_config`) |
| AC3c | R3 | (clients) for each `<c>` in mqtt, amqp, kafka, memcached, redis, dns, pgwire: `cargo val -p proxima-<c> --features listen --lib -E 'test(config_format_twin_)'` | 1 passed per crate, 7 crates, 7 total; each calls the crate's own `resolve_config` with a base config and the `<proto>` override read from the XML fixture, and the same from TOML, and asserts equal |
| AC3d | R3 | `git grep -n 'and_then(from_tree::<' -- . ':!*tests*' ':!*/tests/*' \| wc -l` ; `ls tests/fixtures/config_formats/types/*.toml \| wc -l` ; `ls tests/fixtures/config_formats/types/*.xml \| wc -l` | `45` (every site replaced: 35 C1 + 10 C1x); `33`; `33` |
| AC4 | R4 | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_buffered_)'` | 33 passed, one per C2 field (client_auth 1, delay 4, transform 8, filter 4, validate 12, kv_cache 1, kv_file 1, process 2 = 33 over 8 types), each loading a fixture with only that field set as XML text and equal to the TOML-typed value; field names listed in the test file's const table |
| AC4b | R4 | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_ctl_)'` | 8 passed, one per C2 type: its XML twin through today's `serde_json::from_value` returns Err (the degenerate control: without the lenient read the field fails, so AC4 is not passing for another reason) |
| AC5 | R5 | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_read_)'` | 36 passed: 16 XML-form loads (the 12 list sites 783, 1095, 1133, 1168, 1173, 1194, 1214, 1275, 1307, 1317, 1693, mcp 326 and the 4 scalar sites 657, 812, 1206 plus `interval_pipe.rs:184` period_ms), 10 absent-key defaults, 10 wrong-shape errors (one per silent site, each asserting the key named in the text; the table below) |
| AC6 | R6 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_format_)'` ; `cargo val -p proxima-cli --test integration -E 'test(mcp_config_format_)'` | pipe 8 passed (yaml, json5, ron, xml each as `tests/fixtures/pipeline_pipe/triage.<ext>`, each with PROXIMA-24 -> `check_triage` G8 and `check_seen(llm, T1)`, and PROXIMA-25 -> G9 and `check_seen(brief, T2)`); cli 4 passed (`review_ticket.<ext>` for the same four, each reproducing `TOOL_TEXT_OK` through `proxima serve`) |
| AC7 | R6 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_xml_text_)'` ; `cargo val -p proxima-cli --test integration -E 'test(mcp_config_xml_text_)'` | pipe 1 passed: `triage.xml` carries its two payloads (`review`, `summary` bodies) as JSON text, `json_text_leaves` is 2, and the same file answers as the TOML twin (PROXIMA-24 and PROXIMA-25 checks, 2 inside the one test); cli 1 passed: `review_ticket.xml`'s `json_text_leaves` is 6 and the served session equals the TOML twin's 3 lines |
| AC8 | R7 | (a) lib `--lib` over the touched modules; (b) `--test pipeline_pipe`; (c) CLI `mcp_config_tool_` and `ctl_serve_`; (d) `load::`; (e) `proxima-http` `templates::`; (f) `proxima-config --lib`; (g) `proxima-primitives --lib`; (h) `proxima-patterns --lib`; (i) each client crate and `proxima-intercept` `--lib` | 0 failed in each, each count equal to its slice-0 count plus this spec's new tests in that file (sibling baseline at main `88e07443d`: (a) 126, (b) 31, (c) 5, (d) 10, (e) 13; slice 0 re-measures all at `fb38d30fa`, logs `<tickets>/PROXIMA-50/evidence/baseline-{a..i}.log`) |
| AC9 | R7 | `cargo clippy -p proxima-lib -p proxima-http -p proxima-config -p proxima-primitives -p proxima-patterns -p proxima-intercept -p proxima-cli --all-targets --features tokio -- -D warnings > <E>/clippy.log 2>&1; echo "exit=$?"; grep -cE '^(error\|warning)' <E>/clippy.log; grep -cE '^ *Finished ' <E>/clippy.log` ; the C8x control `printf 'warning: x\nerror: y\n' \| grep -cE '^(error\|warning)'` | `exit=0`, `0`, `1`; control prints `2` |

AC5's ten silent sites, the wrong-shape fixture and the error each moves to (`<dir>` =
`tests/fixtures/config_formats/reads`; each site has `<site>_absent.toml`, `<site>_xml.xml`,
`<site>_wrong.toml`, and the error asserts the key by name):

| site | key | `<site>` | wrong shape and the error text |
|---|---|---|---|
| `load.rs:1095` | `middleware` | `middleware` | `middleware = 5` -> `` `middleware` must be an array of tables `` |
| `load.rs:1133` | `miss_on` | `miss_on` | `miss_on = 5` -> `` `miss_on` must be an array `` (entries keep `` `miss_on` entries must be strings ``, `:1136`) |
| `load.rs:1168` | `write_back` | `write_back` | `write_back = "x"` -> `` `write_back` must be an array `` |
| `load.rs:657` | `list_mode` | `list_mode` | `list_mode = "maybe"` -> `` `list_mode` must be a bool `` |
| `load.rs:812` | `weight` | `weight` | `weight = "heavy"` -> `` `weight` must be an unsigned integer `` |
| `app.rs:1173` | `schema` | `schema` | `schema = 5` -> `` `schema` must be an array of tables `` |
| `app.rs:1194` | `pipe` | `pipe` | `pipe = 5` -> `` `pipe` must be an array of tables `` |
| `app.rs:1307` | `methods` | `methods` | `methods = [1]` -> `` `methods` entries must be strings `` (today the 1 is dropped) |
| `app.rs:1317` | `host` | `host` | `host = [1]` -> `` `host` entries must be strings `` |
| `app.rs:1693` | `chain` | `chain` | `chain = [1]` -> `` `chain` entries must be strings `` |

The XML form of each: `<methods>GET</methods>` (one element) reads as the one-element list `["GET"]`,
`<weight>3</weight>` as 3, `<list_mode>true</list_mode>` as true; absent keeps today's default
(no middleware, `fallthrough_default`, no write-back, `list_mode` false, weight 1, no schema blocks,
no pipes, any method, any host, a pipe entry standing alone).

Counts: AC1 8 + AC2 7 + AC3 34 + AC3b 1 + AC3c 7 + AC4 33 + AC4b 8 + AC5 36 + AC6 12 + AC7 2 = 148
new tests, over 41 members (33 + 1 + 7), 33 C2 fields, 16 + 10 + 10 hand-read cases.

## out of scope

- Json.NET's `json:Array="true"` convention for XML payloads: the four free-form payloads are
  `serde_json::Value`, so lenient reading has no type to aim at (D4); the XML form is JSON text
- a shared helper crate or module for the Value -> `T` call (D1)
- writing configs in any format (conflaguration reads only; serializers stay in proxima-config)
- the generic layer helpers `app_config.rs:430`, `cassette_config.rs:647`, `config_merge.rs:137`,
  `command_config.rs:801` (they read conflaguration's own layered output)
- XML attributes as config keys, XML namespaces, comments-as-data (whatever conflaguration's
  `xml-lenient` spec decided stands)
- releasing conflaguration 2.1.0 (done 2026-11-20, CONFLAG-7)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| ~~conflaguration 2.1.0 is not on crates.io when slice 1 starts~~ struck 2026-11-20: published (`max_version` 2.1.0, CONFLAG-7) | -- | -- | -- |
| error text gains a `json: ` or `value deserialize: ` cause prefix at 45 sites | high | an existing test asserting a `serde_json` message breaks | AC8 counts incumbents per file; slice 1's grep lists every test asserting `config:` text before slice 5 starts |
| `cargo add conflaguration@2.1.0` does not bump a `workspace = true` entry | medium | the 2.0.0 pin at `Cargo.toml:153` stays | slice 1 validates `Cargo.toml:153` reads 2.1.0; the slice note records which `cargo add` form moved it |
| a Value holding one key `$text` collapses when read as a `Tree` (`src/tree.rs:138-143`) | low | a config with a literal `$text` key changes | none of the 41 fixtures use it; named, not fixed here |
| an XML scalar element with attributes or children where text is expected | medium | `<ms unit="s">5</ms>` fails | out of scope above; the failing error names the key (R5 style) |
| the 8 C1x types lack `Serialize` or `PartialEq` | medium | the twin comparison has no equality | slice 0 reads each; the fallback is comparing the `Value` the site's expression produces from both fixtures |
| `JsonConfigFormat` alloc build diverges from the std build | low | R1's cfg split | D7; AC1 runs at `std`, slice 2 also runs `cargo check -p proxima-config --no-default-features --features alloc` |
| the shared machine's load guard refuses runs | high | delay | one build at a time, coordinated with peer sessions |

## context

- `proxima-tensor/specs/agentic-config-payloads/SPEC.md`, `TASKS.md` (landed; AC style, fixtures, rigs)
- `<tickets>/PROXIMA-49` (the payloads card and its evidence), `<tickets>/PROXIMA-50` (this card)
- `/Users/brianbruggeman/repos/slot-0/conflaguration-publish` at `f7d698e`: `src/file.rs`, `src/tree.rs`, `src/lenient.rs`, `src/lib.rs`, and its `specs/xml-lenient`
- `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` (goal, boundary, "Split")

## open before audit

- resolved: where the one helper lives -- there is none; each site calls conflaguration directly (see
  architecture, D1)
- resolved: per-type fixtures for R3 -- 41 TOML / XML pairs, one test per reaching crate: proxima-lib
  33, proxima-intercept 1, seven client crates 1 each (see architecture, D2, AC3-AC3d)
- resolved: per-site fixtures for R5 and the error text each silent site moves to -- the 10 sites,
  fixtures and texts are the table after the acceptance criteria table (see architecture, D3, AC5)
- resolved: XML free-form payloads with one-element lists -- they stay JSON text in XML; Json.NET's
  `json:Array="true"` is out of scope (see architecture, D4, AC7)
