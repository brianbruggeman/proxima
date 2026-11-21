# agentic-config-formats -- every config kind in any format, through conflaguration

status: audited (spec-auditor ADMIT, round 6, at `ed4d91653`); conflaguration 2.1.0 published to crates.io 2026-11-20 (`5bb4b91`, tag `v2.1.0`)
amendment: 2026-11-20 after spec-auditor round 1 (REFUSE): independent oracles (`.expected.json`) and `ctl_format_` controls, R5 sites 1182 / 882 / kv 384, R7 value ACs, runnable AC8, JSON stays on serde_json (D7), `schema-std` wiring for the 12 Schema fields; counts below are the amended ones
amendment 2: 2026-11-20 after spec-auditor round 2 (REFUSE): `write_back` gains a `{ from, to }` table form (D10, the single permitted shape extension), `doc.xml.expected.json` (AC1), AC2 is behavioural through `Client::from_path` (slice 7b), PASS controls for the C2 fields and the AC5 XML-form literals, AC4b evidence only from slice 0b, full wrong-shape error texts, `G(<args>, <name>)` and `<E>` defined here; counts below are the amended ones
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
the PM topic is `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`, "Split". `<E>` is
`/Users/brianbruggeman/repos/obsidian/slot-0/slot-0/20 - Collaboration and Records/22 - Investigations/22.1 - Tickets/PROXIMA-50/evidence`
(the same path as `<tickets>/PROXIMA-50/evidence`; TASKS.md defines it too).

## problem

proxima parses six formats itself (`proxima-config/src/lib.rs:262-392`) into `serde_json::Value`,
then reads configs from that value strictly: XML keeps one of each repeated element and gives every
scalar as text, so an XML config loses pipes, stages and tools or fails on its first number; and a
second TOML converter serves the client config (`src/client/handle.rs:176-186,835`) -- measured by
every config type, hand-read key and end-to-end fixture loading the same from XML as from TOML.

## refutation condition

If a config type cannot load from XML without changing its TOML / JSON shape (beyond a field
attribute), or without proxima keeping a parser conflaguration also has, the single-layer frame is
wrong and this spec is struck. One shape extension is permitted and is the only one (D10): a
`write_back` entry also accepts a `{ from, to }` table, because a positional `[from, to]` pair has no
XML form; the array form that TOML and JSON use today is unchanged, so no existing TOML / JSON config
changes shape.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | ~~proxima-config's six `ConfigFormatFactory::parse` arms parse through conflaguration~~ struck: the JSON arm serves the alloc tier and conflaguration is std-only, and conflaguration's own JSON read is `serde_json::from_str` (`src/file.rs:18-20`, `from_json_str`), so moving it buys nothing. proxima-config's five std-tier `ConfigFormatFactory::parse` arms (toml, yaml, xml, ron, json5) parse through conflaguration (`from_<fmt>_str::<Tree>`, then `serde_json::Value::from`), keeping the YAML / XML top-level checks (`:332-344,380-392`); the JSON arm stays on serde_json; `toml_to_json` (`:402`) goes; serializers stay (conflaguration does not write). Entry condition: proxima-lib resolves conflaguration 2.1.0 (AC0) | yes |
| R2 | the client config's `from_path` (`src/client/handle.rs:176-186`) reads any format through the registry by extension; `handle.rs:835` `toml_to_json` goes | yes |
| R3 | every non-test typed read of a config tree goes through `conflaguration::from_tree` (the exact expression is in the architecture): the 33 types of census C1 plus the 8 members C1 missed (C1x), 45 sites, so XML text scalars and one-element lists load into typed fields | yes |
| R4 | ~~each buffered non-string field (census C2, 33 fields in 8 types) carries conflaguration's `lenient::deserialize` / `lenient::option`~~ struck: the two `max_bytes` fields (KvCacheConfig, KvFileConfig) already accept XML text today: `SizeValue` is `#[serde(untagged)] enum { Bytes(u64), Sized(String) }` (`src/upstreams/kv_cache.rs:222-227`), the fields are `Option<SizeValue>` (`kv_cache.rs:263`, `kv_file.rs:433`), text lands in `Sized` and `parse_size` resolves it (`kv_cache.rs:396-417`), so they carry NO `lenient` attribute (it would change nothing). R4's attributed set is 31 of the census's 33 C2 fields: 19 in the lib run (client_auth 1, delay 4, transform 8, filter 4, process 2) plus the 12 Schema fields; each carries conflaguration's `lenient::deserialize` / `lenient::option`; the 12 ValidateConfig `Schema` fields live in `proxima-config/src/schema/mod.rs:46-86,131,134` (~~`46-86`~~ gains `:131` `FieldFlags.optional` and `:134` `deprecated`, the two flag fields; feature `schema`, which proxima-primitives enables as `schema-std` without `std`, `proxima-primitives/Cargo.toml:54-55,140`, `proxima-config/Cargo.toml:38`), so they carry it under `cfg_attr(feature = "schema-std", ...)` and `schema-std` gains `dep:conflaguration` (D8) | yes |
| R5 | each hand read of a list or scalar on the load path (census C3, 14 list reads and 5 scalar reads) accepts XML's form: a single element where a list is read, text where a number or bool is read; the 10 silent skips / defaults among them keep their default when the key is absent and become errors that name the key when it is present in the wrong shape; `write_back` entries also accept a `{ from, to }` table and `write_back` references a string of digits as an index (D10) | yes |
| R6 | the triage pipeline and the CLI review config, written in YAML, JSON5, RON and XML (beside the JSON and TOML twins that exist), answer the same as the JSON forms; the XML twins carry the free-form payloads as JSON text (decision D4) | yes |
| R7 | incumbents: every existing config-loading test passes with its slice-0 count; a config that is not XML loads to the same value as before (~~JSON duplicate keys, TOML datetimes and `inf` / `nan` named as the only differences~~ struck: JSON stays on serde_json (D7), so duplicate keys do not change; the TOML datetime and `inf` / `nan` rules are the two cases where the mechanism changes, and the value is pinned by AC1d: conflaguration R5 gives the same RFC 3339 string and `null` that `toml_to_json` gives today, `proxima-config/src/lib.rs:406-408,410`) | yes |

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
(`kv_cache.rs:223`) 2, which already read XML text (`SizeValue::Sized`, `kv_cache.rs:222-227,263,396-417`, `kv_file.rs:433`) and carry no `lenient` attribute (R4); ProcessConfig `ready_probe(stdout_line | tcp).timeout_ms` (`process.rs:494`) 2.

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
| D2 R3 fixtures | one TOML and one XML fixture per member (41 pairs) plus a hand-written `<member>.expected.json` (41 more files, D9), every non-string field set, each fixture compared to the expected value as the comparison bullet says; one test per crate that reaches the members (proxima-lib 33, proxima-intercept 1, one in each of 7 client crates) | proxima-lib's `Cargo.toml` reaches 32 of the 33 C1 types (proxima-http `:300`, proxima-primitives `:295`, proxima-patterns `:312`, `proxima-recording` `:393`, the seven client crates behind `<proto>-client` features `:665-939`) and `ProximaSettings`, but no `proxima-intercept` dependency exists anywhere outside that crate (`Cargo.toml:48` is the workspace member list; `proxima-intercept/Cargo.toml:41-42` has proxima-lib only as a dev-dependency), and the seven server reads sit in private fns of crates that proxima-lib reaches only through `<proto>-listener` features; one test per crate is the only layout that exercises every member without a feature set no crate enables today |
| D3 R5 silent sites | absent key keeps its default; present key in the wrong shape is an error naming the key, in the style of the error nearest the site or, where a one-element reading reaches an existing entry check, that check's own text (the AC5 wrong-shape table) (`` `miss_on` entries must be strings ``, `load.rs:~~1136~~1138` (drift: the quoted text is at `:1138`); `` `upstreams` must be an array ``, `:785`) | keeping the silent skip would let an XML typo load a pipeline with no middleware; making an absent key an error would break every existing config; the one exception to "no shape change" is `write_back` (D10): a positional pair has no XML form, so its entries also take a `{ from, to }` table |
| D4 XML free-form payloads | `StageRequest.body`, `ProcessRpcInitFrame.send`, `SynthConfig.body` and `SynthConfig.body_template` (all `serde_json::Value`) are written in XML as JSON text; a string keeps the meaning PROXIMA-49 gave it | the target is `Value`, so lenient reading has no type to aim at: `<id>1</id>` would arrive as `"1"` and a one-element list as a scalar, silently changing the payload |
| D5 JSON text in TOML twins of C1 fixtures | the 4 payload fields in a C1 twin pair are JSON-text strings in both the TOML and the XML fixture | a TOML table next to XML text would differ by design (D4); the PROXIMA-49 table form is covered by its own ACs |
| D6 how a dependency changes | ~~`cargo add conflaguration@2.1.0 --features ...` per crate~~ struck: the members inherit `workspace = true` and `cargo add` cannot edit `[workspace.dependencies]`. The workspace table (`Cargo.toml:153`, `version = "2.1.0"`) is hand-edited (allowed: it is the one place the version lives); each member takes `git wt cargo add conflaguration --package <crate> --features <missing>` with no version, which keeps `workspace = true` (TASKS slice 1); no hand edit of a member's `[dependencies]` line | repo rule: dependencies change only through `cargo add`; a `[features]` list edit (`std = [..., "dep:conflaguration"]`) is not a dependency edit |
| D7 JSON arm of the registry | ~~all six arms move under `std`~~ struck (it contradicted the unchanged alloc-tier `JsonConfigFormat`): the JSON arm stays on serde_json; the five std-tier arms (toml, yaml, xml, ron, json5) move; `JsonConfigFormat` (`proxima-config/Cargo.toml:14-16`, serde_json only) is unchanged | conflaguration is std-only (`src/lib.rs:1233`), so the alloc build cannot reach it, and its own JSON read is `serde_json::from_str` (`src/file.rs:18-20`, `from_json_str`), so a move changes no behaviour |
| D8 the 12 `Schema` fields | `schema-std = [..., "dep:conflaguration"]` (a `[features]` edit, D6) and `#[cfg_attr(feature = "schema-std", serde(deserialize_with = "conflaguration::lenient::..."))]` on the 12 fields of `proxima-config/src/schema/mod.rs:46-86,131,134` (~~`46-86`~~ gains `:131` `FieldFlags.optional` and `:134` `deprecated`) | the module is built under `schema`, and proxima-primitives enables `schema-std` without `std` (`proxima-primitives/Cargo.toml:54-55,140`), so hanging it on `std` would leave the fields strict in exactly that build; AC4-schema runs without `std` so feature unification in a lib run cannot hide the gap |
| D9 oracles | every new-code test compares the XML fixture and the TOML fixture each to a hand-written expected value, written by the fixture author and produced by no code: `<member>.expected.json` (AC3, AC3b, AC3c), `doc.expected.json` (AC1 for toml, yaml, ron and json5; AC1c) and `doc.xml.expected.json` (AC1 for xml: the same document with every scalar as a JSON string, since XML reads every scalar as text, conflaguration `src/file.rs:44-47`, `src/tree.rs:25-27`, and the 2-element list as a list), ~~`client.expected.json` (AC2)~~ struck: there is no `ClientConfig` to serialise; AC2 asserts the literal status and body of a synth upstream the client config declares, a literal per C2 field in the test's const table (AC4, AC4-schema), a literal per site (AC5), the sibling spec's `TOOL_TEXT_OK` (AC7). Comparison form (the comparison bullet above): typed equality where `PartialEq` covers every field, otherwise `serde_json::to_value(&config) == expected`; `ConfiguredTool` has no `Serialize`, so it compares by `PartialEq` against a struct literal in the test | the draft compared the XML result to the TOML result, both through the changed code, so a bug common to both passed; each `ctl_format_` control (slice 0b) runs today's code on the TOML / non-XML side against the same expected value and passes before any change, so a wrong expected value is caught by the control, not by the new code; ~~every literal that is not an `.expected.json` has its control too~~ struck: the 6 new wrong-shape texts are emitted by no incumbent, so no control can pass on today's code; every literal that today's code can produce has a PASS control: one `ctl_format_buffered_` control per C2 field (33) and one `ctl_format_read_<site>_typed` control per AC5 XML-form literal (19), plus `ctl_format_read_1206_label_typed` for site 1206's second literal `Ok(0)` (20 in all), each running today's code on the TOML form of the same config against the same literal, plus one `ctl_format_read_<site>_wrong_typed` control for each of the 4 wrong-shape texts that exist today (middleware, miss_on, schema, pipe; the TOML array form `<key> = [5]`, which today's entry check already iterates, asserting the same FULL `config: ` text as the row); the 6 NEW wrong-shape texts (`write_back`, `list_mode`, `weight`, `methods`, `host`, `chain`) are R5's new behaviour, so no incumbent emits them and the literal itself is their oracle, and each of those 6 rows names today's observable instead (a silent default, a dropped entry, or a different text), which is the before line; the proxima-config rejection and unknown-extension literals have `ctl_format_reject_yaml`, `ctl_format_reject_xml` and `ctl_format_unknown_ext` |
| D10 `write_back` entry shape | a `write_back` entry is a `[from, to]` array (unchanged) or a `{ from = <ref>, to = <ref> }` table (`parse_write_back_entry`, `src/load.rs:1178-1200`); in XML that is `<write_back><from>primary</from><to>cache</to></write_back>`. `resolve_upstream_index` (`:1202-1217`) also accepts a string of ASCII digits as an index when no upstream has that label; the label match comes first (XML reads `<idx>1</idx>` as the text `"1"`, which the typed-integer branch `as_u64` refuses). The TOML / JSON array form is unchanged | a positional tuple has no XML form (XML has no nameless pair and no nested list), so naming its two positions is the one extension; keeping the array-only reading would leave `write_back` unloadable from XML, which breaks the refutation condition, and a shape change anywhere else would be a second extension, which the refutation condition forbids |

## acceptance criteria

Runner `cargo val`, always `--lib` or `--test`, each run wrapped `timeout -k 30s 25m git wt cargo
val ...`, `CARGO_BUILD_JOBS=2`, one build at a time. `<L>` is the proxima-lib feature set
`--features tokio,tcp,http1,http-prime,http2,h3-native-upstream,recording,vm,amqp-client,dns-client,kafka-client,memcached-client,mqtt-client,pgwire-client,redis-client`
(the cfg gates of `src/upstreams/mod.rs:1-78`; ~~without `tcp`, `mod.rs:1-76`~~ struck 2026-11-20: `stream_passthrough` is gated `any(feature = "tcp", feature = "unix")` at `mod.rs:77-78`, so without `tcp` the StreamPassthroughSettings member is compiled out of every `<L>` run (slice 0b batch 3: 28 passed under the old set, 29 with `tcp`, `"<E>/slice0b-batch3-with-tcp.log"`); slice 0 compiled the old set, and the `tcp` set compiles in `"<E>/slice0b-batch3-with-tcp.log"`).
Test-name prefixes: `config_format_arm_`, `config_format_same_`, `config_format_diff_` and
`config_format_buffered_schema_` (proxima-config), `client_config_format_` (lib),
`config_format_twin_` and `config_format_members` (each crate named below),
`config_format_buffered_` and `config_format_ctl_` (lib), `config_format_read_` (lib hand reads),
`ctl_format_` (the controls, AC10; each crate that holds fixtures), `pipeline_format_` and
`pipeline_xml_text_` (`tests/pipeline_pipe.rs`), `mcp_config_format_` and `mcp_config_xml_text_`
(CLI integration). All new proxima-config tests and controls live in one new file, `proxima-config/src/config_formats_tests.rs`, declared `#[cfg(test)] mod config_formats_tests;` in `proxima-config/src/lib.rs` (AC1b-config counts distinct arm calls in `lib.rs`, which that file cannot affect). The new proxima-lib tests live in one new module, `src/config_formats_tests.rs`
(each test gated on its member's feature), so none matches AC8's `pipelines::`,
`upstreams::process_rpc::`, `upstreams::synth::` or `load::` filters. The prefix-collision check, one
command that matches a prefix anywhere in a fn name, run in slice 0 only:
`git grep -n -E 'fn [a-z0-9_]*(config_format_|client_config_format_|pipeline_format_|pipeline_xml_text_|mcp_config_format_|mcp_config_xml_text_|ctl_format_)' -- '*.rs' | grep -v 'fn default_config_format_registry' | wc -l | tr -d ' '`
prints `0` (the one place a 0 is the expected count; every later run prints the count of new test
fns). ~~the draft's unanchored form `git grep -n -E '(config_format_|...)' -- '*.rs' | wc -l`~~ struck:
it prints 26 at `fb38d30fa` because `default_config_format_registry` (`proxima-config/src/lib.rs:250`
and 21 call sites) contains `config_format_`; anchoring on `fn` and removing that one pub fn makes the
0 true. Build gate `G(<args>, <name>)` (~~`G(<args>)`~~ struck: `<name>` had no value; every use below gives it, e.g. `G(-p proxima-lib <L> --lib, baseline-L)` writes `"<E>/baseline-L.log"`) used below: `timeout -k 30s 25m git wt cargo check <args> > "<E>/<name>.log" 2>&1; grep -c '^error' "<E>/<name>.log"; grep -c 'Finished' "<E>/<name>.log"`,
expected two counts, `0` then `1` (a run killed by the timeout or failing to compile prints no
`Finished`). `<B_x>` below is a placeholder filled from slice 0's log `"<E>/baseline-<x>.log"`.

The 41 members, by crate. Fixture triple = `<dir>/<name>.toml`, `<dir>/<name>.xml` and the hand-written `<dir>/<name>.expected.json` (D9); `<dir>` is
`tests/fixtures/config_formats/types` for proxima-lib, `<crate>/tests/fixtures/config_formats` for
the others. Each fixture sets every non-string field of the type (D5 for payload fields).

| crate | members (name = snake_case of the type) | n |
|---|---|---|
| proxima-lib | http_config, h3_native_config, grpc_config, synth_config, callback_config, process_config, process_rpc_config, fs_config, kv_cache_config, kv_file_config, stream_passthrough_settings, vm_config, record_config, mqtt_client_config, dns_resolver_config, pg_client_config, amqp_client_config, redis_client_config, memcached_client_config, kafka_client_config, stage_spec, retry_config, transform_config, rate_limit_config, client_auth_config, auth_config, validate_config, filter_config, delay_config, isolate_config, configured_tool, schema, proxima_settings | 33 |
| proxima-intercept | intercept_config | 1 |
| proxima-mqtt, -amqp, -kafka, -memcached, -redis, -dns, -pgwire | mqtt_server_config, amqp_server_config, kafka_server_config, memcached_server_config, redis_server_config, dns_server_config, pg_server_config | 7 |

| id | discharges | command | expected |
|---|---|---|---|
| AC0 | R1 (entry condition) | `git wt cargo tree -p proxima-lib -i conflaguration --depth 0 2>/dev/null \| head -1` ; G(`-p proxima-config -p proxima-lib -p proxima-intercept --features proxima-lib/tokio`, `ac0`) | `conflaguration v2.1.0`; then `0` and `1` (~~"exactly 1 line" of `git grep -n '^conflaguration = ' Cargo.toml`~~ struck: root `Cargo.toml` holds the pin at `:153` and the proxima-lib dependency at `:381`, so the grep cannot print one line) |
| AC1 | R1 | `cargo val -p proxima-config --features std --lib -E 'test(config_format_arm_)'` | ~~8 passed: 6 formats each parse `doc.<ext>` to the same expected `Value`~~ struck (both sides went through the changed arms; JSON is no longer an arm, D7). 7 passed: the five std-tier arms (toml, yaml, xml, ron, json5) each parse `proxima-config/tests/fixtures/formats/doc.<ext>` (one logical document with a number, a bool, a nested table and a 2-element list): toml, yaml, ron and json5 each equal the hand-written `doc.expected.json` (4), and xml equals the hand-written `doc.xml.expected.json` (1; its independent oracle is `ctl_format_doc_xml` (AC10), which runs the released library directly, before slice 2 rewires the XML arm, so this case then checks the wiring; ~~"to the hand-written `doc.expected.json` (5 ...)"~~ struck: XML reads every scalar as text, conflaguration `src/file.rs:44-47`, `src/tree.rs:25-27`, so the XML value is the same document with every scalar as a JSON string and the 2-element list as a list; the XML test also asserts its repeated element is a list), plus the YAML and XML top-level rejections (`:332-344,380-392`) with their existing reasons (2), each asserting the FULL error text, the `ProximaError::Config` display (`#[error("config: {0}")]`, `proxima-core/src/lib.rs:322`) of the arm's message: a bare YAML scalar returns Err with exactly `config: yaml: top-level must be a mapping or sequence` (`lib.rs:339-341`), an XML document with no element returns Err with exactly `config: xml: top-level must be an element` (`lib.rs:387-389`); their controls `ctl_format_reject_yaml` and `ctl_format_reject_xml` (AC10) run today's arm on the same inputs and expect the same texts |
| AC1b-config | R1 | (row 2) `git grep -n 'fn toml_to_json' -- proxima-config/src ; echo "exit=$?"` ; `git grep -o -E 'conflaguration::from_(toml\|yaml\|xml\|ron\|json5)_str' -- proxima-config/src/lib.rs \| sort -u \| wc -l \| tr -d ' '` | first prints only `exit=1` (slice 0: `proxima-config/src/lib.rs:402:...`, exit 0); second prints `5`: five DISTINCT arm calls in `lib.rs` (slice 0: `0`; ~~`git grep -c 'conflaguration::from_.*_str' -- proxima-config/src/lib.rs` printing `proxima-config/src/lib.rs:5`~~ struck: a count of matching lines, which a control in the same file or a duplicate call inflates; ~~`6`~~ struck earlier, five arms; the JSON arm stays on serde_json). All proxima-config controls and new tests live in `proxima-config/src/config_formats_tests.rs`, so no control adds a match to `lib.rs` |
| AC1b-client | R2 | (row 3) `git grep -n 'fn toml_to_json' -- src/client ; echo "exit=$?"` | prints only `exit=1` (slice 0: `src/client/handle.rs:835:...`, exit 0) |
| AC1c | R7 | `cargo val -p proxima-config --features std --lib -E 'test(config_format_same_)'` | 5 passed: `default_config_format_registry()?.parse_with_hint(text, Some("<ext>"))` on `doc.toml`, `doc.json`, `doc.yaml`, `doc.json5`, `doc.ron` each equals `doc.expected.json` (the controls AC10 run today's parser on the same five files, expecting the same value) |
| AC1d | R7 | `cargo val -p proxima-config --features std --lib -E 'test(config_format_diff_)'` | 2 passed, each value explicit: (1) `diff_datetime.toml` (`when = 1979-05-27T07:32:00Z`) through the TOML arm equals `{"when":"1979-05-27T07:32:00Z"}` (conflaguration R5: a TOML datetime becomes its RFC 3339 string, `src/tree.rs:13,140`, `specs/xml-lenient/SPEC.md` R5; the same string `toml_to_json` gives today); (2) `diff_nonfinite.toml` (`x = inf`, `y = nan`) equals `{"x":null,"y":null}` (R5: a non-finite float converts to JSON `null`; today's `Number::from_f64(..).unwrap_or(Null)`, `lib.rs:406-408`) |
| AC1e | R1 | G(`-p proxima-config --no-default-features --features alloc`, `ac1e`) | `0` and `1` (the alloc tier still builds with `JsonConfigFormat` only) |
| AC2 | R2 | `cargo val -p proxima-lib <L> --lib -E 'test(client_config_format_)'` | ~~7 passed: `ClientConfig::from_path` over `client.<ext>` each equals the hand-written `client.expected.json` by `serde_json::to_value` equality (6) and a `.cfg` path errors (1)~~ struck: there is no `ClientConfig`. `Client::from_path` (`src/client/handle.rs:176-186`) keeps the spec in a private `Inner.spec: Value` (`:49-50`) and hands it to `load(Spec::Inline(spec), ...)` (`:233-245`), the loader the server uses, so no typed value exists to compare; its C1 / C3 sites read the spec. 7 passed, behavioural: each `tests/fixtures/client_config/client.<ext>` (toml, json, yaml, json5, ron, xml: 6 tests) declares a synth upstream (`SynthConfig`, `synth.rs:393`) with the hand-written status `201` and body `created`; the test (`#[proxima::test]`, in `src/config_formats_tests.rs`) builds `Client::from_path(path).await`, sends `client.call("GET", "/").send().await` (the pattern of the client tests at `handle.rs:1354,1369-1391`) and asserts `response.status() == 201` and `response.bytes().await` equals `b"created"`; plus 1 test that a path with extension `.cfg` returns the registry's unknown-format error, asserting the FULL text ``registry: no config format for extension `.cfg` `` (``ProximaError::Registry(format!("no config format for extension `.{ext}`"))`` at `proxima-config/src/lib.rs:198`, shown through `#[error("registry: {0}")]`, `proxima-core/src/lib.rs:353`, with `ext` = `cfg`); its control `ctl_format_unknown_ext` (AC10) runs today's `get_by_extension("cfg")` (`lib.rs:195-205`) on the default registry (`:249-253`) and expects the same text. Split by slice: 6 land in slice 3 (toml, json, yaml, json5, ron and the `.cfg` error), 1 lands in slice 7b (`client.xml`, after slice 7: XML gives `status` as the text `"201"`, and `SynthConfig`'s read at `synth.rs:393` takes text scalars only once R3 lands there). The control `ctl_format_client_toml` (AC10) runs the same test on today's code with `client.toml` |
| AC3 | R3 | (lib) `cargo val -p proxima-lib <L> --lib -E 'test(config_format_twin_)'` ; `cargo val -p proxima-lib <L> --lib -E 'test(config_format_members)'` | 33 passed (one per lib member, a failure names the member) ; 1 passed (`config_format_members` asserts its const list holds 33 names and the fixture directory holds exactly 33 `.toml`, 33 `.xml` and 33 `.expected.json`); each twin test builds its type through the production entry point (the factory build / resolve function, never the D1 expression, so AC3d's count is not inflated) from its TOML fixture and from its XML fixture and asserts each equals `<member>.expected.json` (comparison form: D9) |
| AC3b | R3 | (intercept) `cargo val -p proxima-intercept --features intercept-config --lib -E 'test(config_format_twin_)'` | 1 passed (`intercept_config`: the TOML and the XML fixture each equal `intercept_config.expected.json`) |
| AC3c | R3 | (clients) for each `<c>` in mqtt, amqp, kafka, memcached, redis, dns, pgwire: `cargo val -p proxima-<c> --features listen --lib -E 'test(config_format_twin_)'` | 1 passed per crate, 7 crates, 7 total; each calls the crate's own `resolve_config` with a base config and the `<proto>` override read from the XML fixture, and again from the TOML fixture, and asserts each resolved config equals `<proto>_server_config.expected.json` (~~"the two are equal"~~ struck, not an oracle) |
| AC3d | R3 | `git grep -n 'and_then(from_tree::<' -- '*.rs' ':!*/tests/*' ':!tests/*' \| wc -l \| tr -d ' '` ; `ls tests/fixtures/config_formats/types/*.toml \| wc -l` ; `ls tests/fixtures/config_formats/types/*.xml \| wc -l` | `45` after all sites change (every site replaced: 35 C1 + 10 C1x; slice 0 prints `0`, the "before" line); `33`; `33`. The pathspec excludes markdown and test dirs (~~`':!*tests*' ':!*/tests/*'`~~ struck: it matched this spec and TASKS.md; `':!tests/*'` is added because `*/tests/*` needs a leading directory and misses the root `tests/`); the twin tests call production entry points (AC3), so in-file `#[cfg(test)]` code adds none |
| AC4 | R4 | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_buffered_)'` | ~~33 passed (validate 12 among them)~~ struck: the 12 Schema fields are not reachable from a lib run without feature unification (D8). 21 passed, one per C2 field (client_auth 1, delay 4, transform 8, filter 4, kv_cache 1, kv_file 1, process 2 = 21 over 7 types), each loading a fixture with only that field set as XML text and asserting the field equals the hand-written typed literal in the test file's const table (one literal per field). The kv_cache and kv_file tests are regression guards, not flips: both the XML fixture and the TOML fixture use the string form `"64MB"` and the literal is `Some(SizeValue::Sized("64MB".into()))`, because text already lands in `Sized` today (`kv_cache.rs:222-227`, `kv_file.rs:433`); each has a PASS control (AC10, `ctl_format_buffered_<type>_<field>`): today's code on that field's TOML fixture (`<field>.toml`, typed; for kv_cache and kv_file the same string form `"64MB"`) equals the same literal, 21 in the lib run |
| AC4-schema | R4 | `cargo val -p proxima-config --no-default-features --features schema-std --lib -E 'test(config_format_buffered_schema_)'` | 12 passed, one per ValidateConfig inline `Schema` field (`int.min/max`, `uint.min/max`, `float.finite`, `string.min_len/max_len`, `bytes.max_len`, `seq.min_items/max_items`, `struct.fields[].flags.optional/deprecated`), each loading an XML fixture with only that field set and asserting the const-table literal; run without `std`, so `std` unification cannot supply the `lenient` dependency; each has a PASS control (AC10, `ctl_format_buffered_schema_<field>`, 12 in this run): `toml::from_str::<Schema>` (`toml` is a `schema-std` dependency) on that field's typed TOML fixture equals the same literal on today's code |
| AC4b | R4 | slice 0b only: `timeout -k 30s 25m git wt cargo val -p proxima-lib <L> --lib -E 'test(config_format_ctl_)' > "<E>/slice0b-ac4b.log" 2>&1` | ~~8 passed, one per C2 type, run again in slices 4 and 5~~ struck: after R4 each control flips (conflaguration `src/lenient.rs:45-47,96-101`; `delay.rs:51-56,195-197`: the lenient field reads text), so none can stay in a passing suite. ~~8 passed in slice 0b, one per C2 type (client_auth, delay, transform, filter, validate, kv_cache, kv_file, process)~~ struck: kv_cache and kv_file are not failing-side cases, their `max_bytes` text already loads through `SizeValue::Sized` (`kv_cache.rs:222-227,263,396-417`, `kv_file.rs:433`) and gets no `lenient` (R4), so the control would pass and flips nothing. 6 passed in slice 0b, one per flipping C2 type (client_auth, delay, transform, filter, validate, process): its hand-written `<type>_c2.xml` (every C2 field of the type as text) through today's XML read and `serde_json::from_value` returns Err (the failing-side control: without the lenient read the field fails, so AC4 is not passing for another reason). They are evidence from slice 0b only, log `"<E>/slice0b-ac4b.log"`; each is deleted, with its `_c2.xml`, in the slice that flips it: slice 4 (delay, transform, filter, validate) and slice 5 (client_auth, process; ~~kv_cache, kv_file~~ struck, see above); that slice's note names the deletion, and no later run counts them |
| AC5 | R5 | `cargo val -p proxima-lib <L> --lib -E 'test(config_format_read_)'` | ~~36 passed: 16 XML-form loads~~ struck: R5 counts 19 hand reads (14 lists, 5 scalars) and the draft loaded 16. 39 passed: 19 XML-form loads (the 14 list sites 783, 1095, 1133, 1168, 1173, 1194, 1214, 1275, 1307, 1317, 1693, mcp 326, 1182 (changed by D10: the `{ from, to }` table entry; ~~"the unaffected 1182"~~) and the unaffected 882; the 5 scalar sites 657, 812, 1206, `kv_cache.rs:384` max_bytes and `interval_pipe.rs:184` period_ms), each asserting the literal in the XML-form table below, 10 absent-key defaults, 10 wrong-shape errors (one per silent site, each asserting the FULL error text of the table below, ~~the key named in the text~~ struck: a key-only assertion passes on any message that contains the key), and 20 PASS controls (AC10, `ctl_format_read_<site>_typed`, one per XML-form literal, ~~19~~ plus `ctl_format_read_1206_label_typed` for the `Ok(0)` literal) and 4 more (AC10, `ctl_format_read_<site>_wrong_typed`, one per existing wrong-shape text) outside this count |
| AC6 | R6 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_format_)'` ; `cargo val -p proxima-cli --test integration -E 'test(mcp_config_format_)'` | pipe 8 passed (yaml, json5, ron, xml each as `tests/fixtures/pipeline_pipe/triage.<ext>`, each with PROXIMA-24 -> `check_triage` G8 and `check_seen(llm, T1)`, and PROXIMA-25 -> G9 and `check_seen(brief, T2)`); cli 4 passed (`review_ticket.<ext>` for the same four, each reproducing `TOOL_TEXT_OK` through `proxima serve`) |
| AC7 | R6 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_xml_text_)'` ; `cargo val -p proxima-cli --test integration -E 'test(mcp_config_xml_text_)'` | pipe 1 passed: `triage.xml` carries its two payloads (`review`, `summary` bodies) as JSON text, `json_text_leaves` is 2, and answers the literals (~~"as the TOML twin"~~ struck, not an oracle): PROXIMA-24 -> status 200, calls `[1, 1, 1, 0]` and PROXIMA-25 -> status 200, calls `[1, 0, 0, 1]` (`tests/pipeline_pipe.rs:710` `BRANCH_PR` calls `[1, 1, 1, 0]` and `:639` `BODY_SUMMARY_FRESH` calls `[1, 0, 0, 1]`, the sibling spec's AC4; ~~`:605-609`~~ moved); cli 1 passed: `review_ticket.xml`'s `json_text_leaves` is 6 and the served session is the sibling spec's three lines: `result.protocolVersion` `2024-11-05`; `check_tool_list(.., &[configured_tool()])` Ok; `check_tool_text(line, TOOL_TEXT_OK)` Ok with the literal `TOOL_TEXT_OK` = `{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"content\":[{\"text\":\"review of PROXIMA-24: adds StageBody\",\"type\":\"text\"}]}"}]}}` (`tools/proxima-cli/tests/integration/mcp_config_tool.rs:22`) |
| AC8 | R7 | the 16 commands of the table after this one, each with package and features | 0 failed in each run, and each count equals `<B_x> + n_x` (the table; `<B_x>` is the run's slice-0 count, logged to `"<E>/baseline-<x>.log"`; `n_x` is this spec's new tests that land in that run's filter) |
| AC9 | R7 | `cargo clippy -p proxima-lib -p proxima-http -p proxima-config -p proxima-primitives -p proxima-patterns -p proxima-intercept -p proxima-cli --all-targets --features tokio -- -D warnings > "<E>/clippy.log" 2>&1; echo "exit=$?"; grep -cE '^(error\|warning)' "<E>/clippy.log"; grep -cE '^ *Finished ' "<E>/clippy.log"` ; the C8x control `printf 'warning: x\nerror: y\n' \| grep -cE '^(error\|warning)'` | `exit=0`, `0`, `1`; control prints `2` |
| AC10 | R1-R5, R7 (controls, `ctl_format_`) | slice 0b, before any code change, each crate: lib `cargo val -p proxima-lib <L> --lib -E 'test(ctl_format_)'` ; `cargo val -p proxima-config --features std --lib -E 'test(ctl_format_)'` ; `cargo val -p proxima-config --no-default-features --features schema-std --lib -E 'test(ctl_format_buffered_schema_)'` ; `cargo val -p proxima-intercept --features intercept-config --lib -E 'test(ctl_format_)'` ; for each `<c>` in the seven client crates `cargo val -p proxima-<c> --features listen --lib -E 'test(ctl_format_)'` | ~~59 controls~~ ~~111 controls~~ ~~112 controls~~ ~~116 controls~~ 120 controls, each run on today's code, each PASS, against the same expected value the new-code tests use: lib 89 (33 `ctl_format_member_<name>`, in `src/config_formats_tests.rs`, which read their fixtures with `include_str!`: today's `serde_json::from_value::<T>` on `<name>.toml` read by today's TOML parser equals `<name>.expected.json` by the D9 form; 1 `ctl_format_client_toml`: today's `Client::from_path` on `client.toml` declares the synth upstream, one request returns status `201` and body `created` (AC2); 10 `ctl_format_read_<site>_absent`: today's hand read on `<site>_absent.toml` gives the listed default; 21 `ctl_format_buffered_<type>_<field>`, one per C2 field of AC4: today's code on that field's typed TOML fixture equals the AC4 const-table literal; 19 `ctl_format_read_<site>_typed`, one per XML-form literal of AC5: today's code on `<site>_typed.toml`, the TOML form of the same config, equals the same literal, with `write_back` (1168) on the TOML array form `write_back = [["primary", "cache"]]` and the scalar sites on TOML typed values; 1 `ctl_format_read_1206_label_typed`: the TOML string `idx = "1"` with the upstream at position 0 labelled `"1"` returns `Ok(0)`, the `idx = 1` integer control of `ctl_format_read_1206_typed` giving `Ok(1)`; 4 `ctl_format_read_<site>_wrong_typed` for site in middleware, miss_on, schema, pipe: today's entry check on the TOML array form `<key> = [5]` returns the same FULL `config: ` text as the AC5 wrong-shape row of that site, fixture `<site>_wrong_typed.toml`), proxima-config 11 in the `std` run, all in `proxima-config/src/config_formats_tests.rs` (declared `#[cfg(test)] mod config_formats_tests;` in `proxima-config/src/lib.rs`) (5 `ctl_format_doc_<ext>`: today's `ConfigFormatFactory::parse` on `doc.toml`, `doc.json`, `doc.yaml`, `doc.json5`, `doc.ron` equals `doc.expected.json`; 1 `ctl_format_doc_xml`: the released library directly, `conflaguration::from_xml_str::<conflaguration::Tree>(doc.xml)` then `serde_json::Value::from`, equals `doc.xml.expected.json` (conflaguration 2.1.0 is already a proxima-config `std` dependency, slice 1, `4669610c6`; `src/file.rs:44-47`, and the json-gated `From<Tree> for serde_json::Value`, `src/tree.rs:149-154`, at `f7d698e`): an oracle independent of proxima's XML arm, checked before slice 2 rewires it; 2 `ctl_format_diff_*`: today's parse of `diff_datetime.toml` and `diff_nonfinite.toml` gives AC1d's values; 2 `ctl_format_reject_<yaml|xml>`: today's arm on a bare YAML scalar and on an XML document with no element returns Err with the exact AC1 texts; 1 `ctl_format_unknown_ext`: today's registry lookup by extension `cfg` returns Err with the exact AC2 text) and 12 in the `schema-std` run (`ctl_format_buffered_schema_<field>`, one per Schema field of AC4-schema), intercept 1 (`ctl_format_member_intercept_config`), each of the 7 client crates 1 (`ctl_format_member_<proto>_server_config`) = 89 + 11 + 12 + 1 + 7 = 120. The `json5` and `ron` doc controls and the 2 diff controls go beyond the four named parsers because AC1c and AC1d cover them. Passed counts: `89`, `11`, `12`, `1`, and `1` in each of the 7 client crates |

AC8's 16 commands (`<tickets>/PROXIMA-50/evidence/baseline-{a..h,i1..i8}.log` hold slice 0's runs). (a)-(e) are the sibling spec's AC7 (a)-(e) (`proxima-tensor/specs/agentic-config-payloads/SPEC.md:183`) verbatim except (c), which gains `or test(mcp_config_payload_)`:

| run | command | expected |
|---|---|---|
| (a) | `cargo val -p proxima-lib --features tokio --lib -E 'test(pipelines::) or test(upstreams::process_rpc::) or test(upstreams::synth::)'` | `126 + 0` |
| (b) | `cargo val -p proxima-lib --features tokio --test pipeline_pipe` | `31 + 9` (AC6 pipe 8, AC7 pipe 1) |
| (c) | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_tool_) or test(ctl_serve_) or test(mcp_config_payload_)'` | `7 + 0` (this spec's `mcp_config_format_` and `mcp_config_xml_text_` match none of the three) |
| (d) | `cargo val -p proxima-lib --features tokio --lib -E 'test(load::)'` | `10 + 0` |
| (e) | `cargo val -p proxima-http --features http1 --lib -E 'test(templates::)'` | `13 + 0` |
| (f) | `cargo val -p proxima-config --features std --lib` | `17 + 25` (AC1 7, AC1c 5, AC1d 2, ctl 11; the 12 AC4-schema tests and the 12 `ctl_format_buffered_schema_` controls need `schema-std`, not `std`) |
| (g) | `cargo val -p proxima-primitives --features std --lib` | `434 + 0` |
| (h) | `cargo val -p proxima-patterns --features std --lib` | `63 + 0` |
| (i1) | `cargo val -p proxima-mqtt --features listen --lib` | `37 + 2` (twin 1, ctl 1) |
| (i2) | `cargo val -p proxima-amqp --features listen --lib` | `63 + 2` |
| (i3) | `cargo val -p proxima-kafka --features listen --lib` | `55 + 2` |
| (i4) | `cargo val -p proxima-memcached --features listen --lib` | `20 + 2` |
| (i5) | `cargo val -p proxima-redis --features listen --lib` | `33 + 2` |
| (i6) | `cargo val -p proxima-dns --features listen --lib` | `33 + 2` |
| (i7) | `cargo val -p proxima-pgwire --features listen --lib` | `129 + 2` |
| (i8) | `cargo val -p proxima-intercept --features intercept-config --lib` | `122 + 2` |

Each client crate's `listen` and intercept's `intercept-config` feature is in the command because the new tests need it (AC3b, AC3c); slice 0 runs the same feature set so `<B_i*>` counts what the later run counts.

AC5's ten silent sites, the wrong-shape fixture and the error each moves to (`<dir>` =
`tests/fixtures/config_formats/reads`; each site has `<site>_absent.toml`, `<site>_typed.toml` (the
TOML form of the XML-form literal, for the PASS control), `<site>_xml.xml`, `<site>_wrong.toml`, the 4 sites with an existing text also `<site>_wrong_typed.toml` (`<key> = [5]`, for the PASS control `ctl_format_read_<site>_wrong_typed`), and
the test asserts the FULL error text of the table, ~~the key by name~~ struck: a key-only assertion
passes on any message that contains the key). Every text in the table is a `ProximaError::Config`, so the FULL text a test asserts is its display, `config: ` followed by the table text (`#[error("config: {0}")]`, `proxima-core/src/lib.rs:322`); each text below is written with that prefix. The four existing texts are each a `ProximaError::Config` at `fb38d30fa` (`load.rs:1098`, `:1138`, `app.rs:1178`, `:1199`), each returned with a plain `?` and no wrapper (~~`app.rs:1206` passes `load(...)?` through unwrapped~~ struck, audit round 6: the pipe text is raised at `:1199`, before the `load(...)?` call at `:1206`, so `:1206` is not on its path). Each row says whether the text is an existing one
(file:line read at `fb38d30fa`) or new:

| site | key | `<site>` | wrong shape and the error text |
|---|---|---|---|
| `load.rs:1095` | `middleware` | `middleware` | `middleware = 5` -> `` config: [[middleware]] entries require `type` `` (existing, `load.rs:1098`, a `ProximaError::Config`, control `ctl_format_read_middleware_wrong_typed` on `middleware = [5]`: D3's one-element reading hands `5` to the entry check, `5.get("type")` is `None`; ~~`` `middleware` entries must be tables ``~~ struck: that text exists nowhere in the code, and the existing check already fires) |
| `load.rs:1133` | `miss_on` | `miss_on` | `miss_on = 5` -> `` config: `miss_on` entries must be strings `` (existing, `load.rs:1138`, a `ProximaError::Config`, control `ctl_format_read_miss_on_wrong_typed` on `miss_on = [5]`, confirmed unchanged: the one-element reading hands `5` to the `as_str` check; ~~`` `miss_on` must be an array ``~~, same reason) |
| `load.rs:1168` | `write_back` | `write_back` | `write_back = "x"` -> `` config: write_back entries must be `[from, to]` arrays or `{ from, to }` tables `` (new, D10; today's observable for `write_back = "x"` is a silent default: `load.rs:1168-1170` returns `Ok(targets)` empty, because `"x"` is not an array (audit round 6); ~~today's `` write_back entries must be `[from_index, to_index]` arrays ``, `load.rs:1183`~~ struck: the entry check now names both forms, and this text is what separates a wrong shape from the XML-form case, which is an accepted load; a `{ from, to }` table missing a key gets the same text) |
| `load.rs:657` | `list_mode` | `list_mode` | `list_mode = "maybe"` -> `` config: `list_mode` must be a bool `` (new; today `.and_then(Value::as_bool).unwrap_or(false)`, `load.rs:657-659`, has no text) |
| `load.rs:812` | `weight` | `weight` | `weight = "heavy"` -> `` config: `weight` must be an unsigned integer `` (new; today `.and_then(Value::as_u64).unwrap_or(1)`, `load.rs:812-815`, has no text) |
| `app.rs:1173` | `schema` | `schema` | `schema = 5` -> `` config: [[schema]] requires `name` `` (existing, `app.rs:1178`, a `ProximaError::Config`, control `ctl_format_read_schema_wrong_typed` on `schema = [5]`, the entry check D3's one-element reading reaches; ~~`` `schema` entries must be tables ``~~ struck: that text exists nowhere in the code) |
| `app.rs:1194` | `pipe` | `pipe` | `pipe = 5` -> `` config: [[pipe]] requires `name` `` (existing, `app.rs:1199`, a `ProximaError::Config`, passed through by `load(...)?` at `app.rs:1206`, control `ctl_format_read_pipe_wrong_typed` on `pipe = [5]`; ~~`` `pipe` entries must be tables ``~~, same reason) |
| `app.rs:1307` | `methods` | `methods` | `methods = [1]` -> `` config: `methods` entries must be strings `` (new; today `filter_map` drops the 1, `app.rs:1308-1312`) |
| `app.rs:1317` | `host` | `host` | `host = [1]` -> `` config: `host` entries must be strings `` (new; today `filter_map` drops the 1, `app.rs:1318-1322`) |
| `app.rs:1693` | `chain` | `chain` | `chain = [1]` -> `` config: `chain` entries must be strings `` (new; today `filter_map(Value::as_str)` drops the 1, `app.rs:1693-1700`) |

Of the 10 rows, 4 reuse an existing text (`middleware`, `miss_on`, `schema`, `pipe`; all four are reached
through D3's one-element reading) and 6 are new (`write_back`, `list_mode`,
`weight`, `methods`, `host`, `chain`).

The XML form of each of the 19 loads, and the hand-written literal it asserts (`<dir>/<site>_xml.xml`;
the four sites that are not whole-config reads, 1182, 1206, 882 and `kv_cache.rs:384`, parse the file
through the registry and hand the named element to the site's function; 1168, 1182 and 1206 are the
`write_back` functions `parse_write_back_targets`, `parse_write_back_entry` and
`resolve_upstream_index`, private to `src/load.rs`, so slice 13 widens them and `BuiltUpstream`
(`load.rs:574`) to `pub(crate)`, a visibility edit with no behaviour change, and the test builds a
two-element `Vec<BuiltUpstream>` (labels `primary` with no kv backend and `cache` with a kv backend)
and calls them). Absent keeps today's default (no middleware, `fallthrough_default`, no write-back,
`list_mode` false, weight 1, no schema blocks, no pipes, any method, any host, a pipe entry standing
alone), which the `ctl_format_read_<site>_absent` controls assert today. The last column is the TOML
form of the same config, which the `ctl_format_read_<site>_typed` control reads through today's code
and compares to the same literal:

| site | XML form | expected literal | TOML form (control) |
|---|---|---|---|
| `load.rs:783` upstreams | one `<upstreams>` element (label `a`) | built upstream labels `["a"]` | `upstreams = [{ ..., name = "a" }]` |
| `load.rs:1095` middleware | one `<middleware>` element, `<type>retry</type>` | applied middleware types `["retry"]` | `[[middleware]]`, `type = "retry"` |
| `load.rs:1133` miss_on | `<miss_on>no_data</miss_on>` | `on_no_data` true, `on_status` `[]`, `on_error` false | `miss_on = ["no_data"]` |
| `load.rs:1168` write_back | `<write_back><from>primary</from><to>cache</to></write_back>` (one element, a `{ from, to }` table, D10) | accepted load: `parse_write_back_targets` returns the same `WriteBackTarget` set as TOML `write_back = [["primary", "cache"]]`, observed as `targets.len() == 1`, `targets[0].label == "cache"` and `Arc::ptr_eq(&targets[0].backend, <the cache upstream's backend>)` (`WriteBackTarget`, `proxima-patterns/src/middleware/write_back.rs:16-20`, has public fields and no `PartialEq`); ~~`Err` containing `` write_back entries must be `[from_index, to_index]` arrays `` (one element is a one-entry list; a nested pair has no XML form)~~ struck: D10 gives the pair an XML form | `write_back = [["primary", "cache"]]` (the TOML array form) |
| `load.rs:1182` write_back entry | ~~`<from>a</from><to>kv</to>`~~ struck: the test's `Vec<BuiltUpstream>` declares the labels `primary` and `cache`, not `a` / `kv`. `<from>primary</from><to>cache</to>` handed over as the entry (D10) | target label `"cache"`, backend ptr-equal to the `cache` upstream's (~~`<w>a</w><w>kv</w>`, "unaffected today"~~ struck: a sibling pair read as an array is not a form an author writes, and D10 supersedes it) | the entry `["primary", "cache"]` |
| `load.rs:657` list_mode | `<list_mode>true</list_mode>` | `list_mode` true | `list_mode = true` |
| `load.rs:812` weight | `<weight>3</weight>` | weight `3` | `weight = 3` |
| `load.rs:1206` write_back index | `<idx>1</idx>`, read as the text `"1"` | `resolve_upstream_index` returns `Ok(1)` (D10: no upstream is labelled `"1"`); the same test asserts `Ok(0)` when the upstream at position 0 is labelled `"1"` (the label match comes first, `load.rs:1209-1213`; its control `ctl_format_read_1206_label_typed`: the TOML string `idx = "1"` with the upstream at position 0 labelled `"1"` returns `Ok(0)` on today's code, and the `idx = 1` integer control gives `Ok(1)`) | `idx = 1` (a TOML integer) |
| `load.rs:882` stage output | `<out>b</out>` | `PipelineOutput::Stage("b")` (unaffected today) | `out = "b"` |
| `app.rs:1173` schema | one `<schema>` element named `s` | one schema block named `s` | `[[schema]]`, `name = "s"` |
| `app.rs:1194` pipe | one `<pipe>` element named `p` | one named pipe `p` | `[[pipe]]`, `name = "p"` |
| `app.rs:1214` listen | one `<listen>` element | one bound listener | `[[listen]]` |
| `app.rs:1275` mount | one `<mount>` (`path` `/x`, `pipe` `p`) | one mount at `/x` | `[[listen.mount]]`, `path = "/x"`, `pipe = "p"` |
| `app.rs:1307` methods | `<methods>GET</methods>` | `MethodFilter::only(["GET"])` | `methods = ["GET"]` |
| `app.rs:1317` host | `<host>a.example</host>` | `HostFilter::only(["a.example"])` | `host = ["a.example"]` |
| `app.rs:1693` chain | `<chain>a</chain>` | chain names `["a"]` | `chain = ["a"]` |
| `listeners/mcp.rs:326` tools | one `<tools>` element | one `ConfiguredTool` equal to the struct literal in the test | `[[tools]]` |
| `interval_pipe.rs:184` period_ms | `<period_ms>250</period_ms>` | period 250 ms | `period_ms = 250` |
| `kv_cache.rs:384` max_bytes | `<max_bytes>64MB</max_bytes>` | `Some(Ok(67108864))` (handled today by the `as_str` arm) | `max_bytes = "64MB"` |

Counts (amended, after round 2): AC1 7 + AC1c 5 + AC1d 2 + AC2 7 + AC3 34 + AC3b 1 + AC3c 7 + AC4 21 +
AC4-schema 12 + AC5 39 + AC6 12 + AC7 2 + AC10 120 = 269 (~~265~~ struck: AC10 gained the 4 `ctl_format_read_<site>_wrong_typed` controls; ~~261~~ struck: AC10 gained 4 controls, `ctl_format_reject_yaml`, `ctl_format_reject_xml`, `ctl_format_unknown_ext` and `ctl_format_read_1206_label_typed`; ~~260~~ struck earlier: AC10 gained `ctl_format_doc_xml`) new tests that remain after slice 5
(~~216~~ struck: it counted AC4b's 8 as permanent and AC10 as 59 controls; ~~148~~ struck earlier: the
draft, before the oracles, controls and the added sites). AC4b's 6 failing-side controls (~~8~~ struck: kv_cache and kv_file are not flips, AC4b) are written
in slice 0b and deleted in slices 4 and 5 (AC4b), so 275 (~~267~~ ~~268~~ ~~271~~) are written in all (269 + 6) and 269 are left. The 269
cover 41 members (33 + 1 + 7), 33 C2 fields (21 + 12), 19 + 10 + 10 = 39 hand-read cases, and 120
controls (89 + 11 + 12 + 1 + 7, where lib 89 = 33 member + 1 client + 10 absent + 21 C2 + 20 AC5
XML-form and label-match controls + 4 wrong-shape controls). AC0, AC1b-config, AC1b-client, AC1e, AC3d, AC8 and AC9 are commands with no new test.
Per-run split of the 120 (AC10): lib 89, proxima-config `std` 11, proxima-config `schema-std` 12,
intercept 1, client crates 7 x 1. No `n_x` of AC8 changes except (f)'s: the 89 lib controls live in
`src/config_formats_tests.rs`, which matches none of (a)-(e); the 12 schema controls run only under
`schema-std`, which is no AC8 run; (f) becomes `+ 25` (AC1 7, AC1c 5, AC1d 2, ctl 11; ~~`+ 22`~~ ~~`+ 21`~~ struck: the controls are 11, adding 2 reject and 1 unknown_ext to the 8); (g) and (h) read
no new test; (i1)-(i8) keep `+ 2`.

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
| ~~`cargo add conflaguration@2.1.0` does not bump a `workspace = true` entry~~ struck 2026-11-20: `cargo add` cannot edit `[workspace.dependencies]`; the table at `Cargo.toml:153` is hand-edited to `version = "2.1.0"` and members take `cargo add` with no version (D6) | -- | -- | -- |
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
