# agentic-config-formats -- every config kind in any format, through conflaguration

status: draft, pre-audit (blocked on conflaguration `xml-lenient` released to crates.io: owner's go)
owner: brian
created: 2026-11-20

Owner, 2026-11-20: "conflaguration should let us do anything: json, toml, yaml, xml, etc."; the
owner chose conflaguration as the single format layer ("conflaguration is nested config using
Setting"). PROXIMA-50. Follows `agentic-config-payloads` (PROXIMA-49), which makes step payloads
values. Every `file:line` is in `~/repos/mine/proxima` at `285b519e5`, from the PROXIMA-49 censuses
(`/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`, "Split"; vault ticket PROXIMA-50).

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

## requirements (draft)

| id | requirement |
|---|---|
| R1 | proxima-config's six `ConfigFormatFactory::parse` arms parse through conflaguration (`from_<fmt>_str::<Tree>`, then `serde_json::Value::from`), keeping the YAML / XML top-level checks (`:332-344,380-392`); `toml_to_json` (`:402`) goes; serializers stay (conflaguration does not write) |
| R2 | the client config's `from_path` (`src/client/handle.rs:176-186`) reads any format through the registry by extension; `handle.rs:835` `toml_to_json` goes |
| R3 | every non-test typed read of a pipe or middleware config (`serde_json::from_value::<Config>`, 31 types: census table C1) goes through `conflaguration::from_tree`, so XML text scalars and one-element lists load into typed fields |
| R4 | each buffered non-string field (census C2, 33 fields in 8 types) carries conflaguration's `lenient::deserialize` / `lenient::option` |
| R5 | each hand read of a list or scalar on the load path (census C3, 14 list reads and 5 scalar reads) accepts XML's form: a single element where a list is read, text where a number or bool is read; the 10 silent skips / defaults among them become loads or errors that name the key |
| R6 | the triage pipeline and the CLI review config, written in all six formats, answer the same |
| R7 | incumbents: every existing config-loading test passes with its slice-0 count; a config that is not XML loads to the same value as before (JSON duplicate keys, TOML datetimes and `inf` / `nan` named as the only differences, per conflaguration R5) |

## census

C1 -- typed config reads (type | kind | deserialize site):
HttpConfig http `proxima-http/src/http1/upstream.rs:309`, `prime_upstream.rs:76`; H3NativeConfig
h3-native `src/upstreams/h3_native.rs:56`; GrpcConfig grpc `grpc_h2.rs:118`; SynthConfig synth
`synth.rs:366`; CallbackConfig callback `callback.rs:139`; ProcessConfig process `process.rs:689`;
ProcessRpcConfig process_rpc `process_rpc.rs:409`; FsConfig fs `fs.rs:225`; KvCacheConfig kv:cache
`kv_cache.rs:367`; KvFileConfig kv:file `kv_file.rs:540`; StreamPassthroughSettings stream
`stream_passthrough.rs:296`; VmConfig vm `vm.rs:769`; RecordConfig / SinkConfig record
`record.rs:925`; InterceptConfig intercept `proxima-intercept/src/pipe.rs:1419`; MqttClientConfig
`mqtt.rs:64`; DnsResolverConfig `dns.rs:76`; PgClientConfig `pgwire.rs:166`; AmqpClientConfig
`amqp.rs:64`; RedisClientConfig `redis.rs:69`; MemcachedClientConfig `memcached.rs:65`;
KafkaClientConfig `kafka.rs:70`; StageSpec pipeline `src/load.rs:914`; RetryConfig
`proxima-primitives/src/pipe/retry.rs:720`; TransformConfig `transform.rs:262`; RateLimitConfig
`rate_limit.rs:381`; ClientAuthConfig `proxima-patterns/src/middleware/client_auth.rs:51,399`;
AuthConfig `auth.rs:42`; ValidateConfig `validate.rs:224`; FilterConfig `filter.rs:477`; DelayConfig
`delay.rs:214`; IsolateConfig `isolate.rs:70`; plus ConfiguredTool `src/listeners/mcp.rs:344` and
`[[schema]]` blocks `src/app.rs:1186`.

C2 -- buffered non-string fields: ClientAuthConfig `scheme(oauth).refresh_ahead_ms` (tag
`client_auth.rs:68`, flatten `:109`); DelayConfig `dist(const).ms`, `dist(range).min_ms`, `max_ms`,
`seed` (`delay.rs:52`); TransformConfig request and response `mutate` `bits`, `seed`, `when.prob`,
`when.seed` (`transform.rs:135,149,186,196`, `mutate.rs:15,80,84`); FilterConfig `predicate(when |
unless).gate.prob`, `seed` (`filter.rs:69,80,88`); ValidateConfig inline `Schema` `int.min/max`,
`uint.min/max`, `float.finite`, `string.min_len/max_len`, `bytes.max_len`, `seq.min_items/max_items`,
`struct.fields[].flags.optional/deprecated` (`validate.rs:236`, `proxima-config/src/schema/mod.rs:47`);
KvCacheConfig and KvFileConfig `max_bytes(bytes)` (`kv_cache.rs:223`); ProcessConfig
`ready_probe(stdout_line | tcp).timeout_ms` (`process.rs:494`).

C3 -- hand reads: lists `src/load.rs:783` upstreams (errors), `:1095` middleware (skips), `:1133`
miss_on (defaults), `:1168` write_back (defaults); `src/app.rs:1173` schema (skips), `:1194` pipe
(skips), `:1214` listen (errors), `:1275` mount (errors), `:1307` methods (defaults), `:1317` host
(defaults), `:1693` chain (drops); `src/listeners/mcp.rs:326` tools (errors); unaffected:
`load.rs:1182`, `:882`. Scalars: `load.rs:657` list_mode (silently false), `:812` weight (silently
1), `:1206` write_back index (errors), `kv_cache.rs:384` max_bytes (handled),
`proxima-primitives/src/pipe/interval_pipe.rs:184` period_ms (errors).

## measured for the audit (PROXIMA-50 sweep, 2026-11-20)

- Every crate holding a C1 site already depends on `conflaguration` directly: proxima-lib
  `Cargo.toml:381`, proxima-http `:61` (both non-optional); proxima-primitives `:144`,
  proxima-patterns `:133`, proxima-intercept `:33`, and the seven client crates (dns `:31`, pgwire
  `:26`, mqtt `:23`, amqp `:22`, redis `:23`, memcached `:23`, kafka `:36`), optional but enabled by
  the features their factories need. The one crate they all share is `proxima-core`, whose optional
  `config` feature already pulls conflaguration (`proxima-core/Cargo.toml:64,73`). proxima-config
  does not depend on conflaguration and is shared by only proxima-lib and proxima-primitives.
- Every C1 site maps `serde_json::Error` to `ProximaError::Config("<kind> config: {err}")` on the
  next line (`<kind>` per site, e.g. `src/upstreams/synth.rs:367`, `proxima-primitives/src/pipe/retry.rs:721`);
  the three non-factory sites read `pipeline \`stages\`: ` (`src/load.rs:915`), `mcp tools[{index}]: `
  (`src/listeners/mcp.rs:345`), `[[schema]] \`{name}\` decode: ` (`src/app.rs:1187`).
- Comparison per C1 type: 15 derive `PartialEq` (StreamPassthroughSettings, VmConfig, the seven
  client configs, StageSpec, DelayConfig, ConfiguredTool, Schema, the recording SinkConfig; FilterConfig
  has a manual impl over `predicate` and `on_reject` only, `filter.rs:354-355`), the rest do not; all
  derive `Serialize` except `ConfiguredTool` (`src/listeners/mcp.rs:295`, `Deserialize` only, which
  has `PartialEq`). So a twin comparison is typed equality where `PartialEq` covers every field, and
  `serde_json::to_value` equality otherwise.

## open before audit

- where the one helper lives: `proxima-core`'s `config` feature (shared by all, conflaguration
  already optional there) versus a direct `conflaguration::from_tree(Tree::deserialize(value)?)` at
  each of the 35 sites; the first keeps one error mapping, the second adds no feature edges
- per-type fixtures for R3 (one TOML and one XML twin per C1 type, every non-string field set)
- per-site fixtures for R5 and the error text each silent site moves to
- XML free-form payloads with one-element lists: stay JSON text in XML unless the owner picks a
  convention (Json.NET's `json:Array="true"` is the offered option)
