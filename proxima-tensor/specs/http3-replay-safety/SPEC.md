# native HTTP/3 ambiguous replay safety

status: draft
owner: brian
created: 2026-10-27

## problem

PROXIMAHTTP-3 must stop the native H3 upstream's implicit replay of an ambiguously delivered non-idempotent request on a reused QUIC connection, measured by three native facade/peer payload cases including unfixed reproduction and retained bounded GET recovery.

## refutation condition

A second native connection/stream receives the same consumed POST body after the original peer closes before response headers, or eligible GET recovery is disabled/unbounded, refutes this contract. Method predicate tests or proto-only drivers without H3NativeUpstream::call cannot discharge it.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | After reused native H3 ambiguous delivery, non-idempotent requests return the original failure, clear cached connection and do not resend the consumed body; subsequent independent request reconnects safely. | yes |
| R2 | Canonical idempotent methods retain at most one automatic recovery on reused-connection failure, including GET success and failed-retry termination. | yes |
| R3 | Fresh-connection failures and non-idempotent/unknown method classes do not automatically replay; test actual native path, preserve peer/request/error records and account the final class site. | yes |

## architecture

Owner PM item01a17b9f-1187-7410-bf51-dc3c4f82fda7 (`http_replay_h3`), PROXIMAHTTP-3, belongs to class parentPROXIMAHTTP-1. PROXIMAHTTP-2 H1/canonical prerequisite is recorded done in actual PM (`/Users/brianbruggeman/.local/share/ragd/pm-exports/hierarchical-intelligence/execution/http_replay_h1-record-readback.json`, item01a17b9f-1187-7aec-ab53-d004a24654c3, stage=done). H3 remains required before class closure. Source/spec audit and independent QA ticket review of this exact revision precede H3 implementation. The canonical API is already available and bound below; do not duplicate its method classifier or change its H1-owned source. PM remains slice tracker; no repo TASKS file. All writes occur only in isolated /private/tmp/hierarchy-live/proxima-replay; shared original is read-only.

The sole production site is proxima-http/src/http3/native/upstream.rs:211–247. The loop has two iterations, obtains cached connection through existing AsyncMutex, records was_reused and drives the full request. drive_one_request (:368–424) opens native request stream, sends actual payload and FIN, then pumps actual native datagrams/events. A close before status yields ProximaError::Upstream; there is no typed proof that the request was unprocessed. Error branch clears cache and retries once whenever was_reused, regardless of method. Commented “likely idle-closed” is not a delivery guarantee. Native QUIC retransmitting stream packets is reliability and distinct from opening a new connection/request after ambiguous application delivery.

Add only `request.method.is_idempotent()` to the existing reused-connection recovery decision: a failed request may reconnect only when `was_reused && request.method.is_idempotent()`. The exact canonical method is `pub fn Method::is_idempotent(&self) -> bool` at `proxima-primitives/src/pipe/method.rs:113`; it delegates known variants to `pub fn Method::is_idempotent_token(token: &[u8]) -> bool` at `:119` and refuses `Other`. `Method` is exported at `proxima-primitives/src/pipe/mod.rs:175`, with import path `use proxima_primitives::pipe::Method;` for fixture construction when needed; the typed request guard calls the inherent method directly and needs no trait import or reparse/allocation. H3 already retains `request` in its async call (`proxima-http/src/http3/native/upstream.rs:191,205`), so preserve that ownership and use its method field. Existing H1 uses the identical canonical condition at `proxima-http/src/http1/client.rs:1070`. Reuse its standard GET/HEAD/PUT/DELETE/OPTIONS/TRACE semantics and default refusal for POST/PATCH/CONNECT/Other/case-mismatched unknowns. Do not add a fresh public API, configuration knob, new client framework, error-string classification, QUIC replay detector, request ID cache or transport strategy. Non-idempotent path clears failed cached connection as now, returns the first actual request error unchanged, and performs no second connection establishment or stream/body submission. Eligible GET still has at most one automatic reconnect; failure of that retry returns without a third attempt. Fresh failure returns immediately. Preserve existing async ownership; no new locks, blocking mutex, spin, boxed futures or production heap allocation.

RFC9114 section4.1.1 specifically distinguishes H3_REQUEST_REJECTED (no application processing, may be treated as never sent) from partially/fully processed cancellation. This is a real protocol unprocessed signal, not a close/error string heuristic. The current generic error branch carries no such typed evidence, and this card invents none or expands replay to support it; ordinary ambiguous close remains gated by canonical idempotence. EOF/connection close/timeout/no response/header absence/connection reuse cannot authorize non-idempotent replay. A timeout after consumption is ambiguous; peer/caller work beyond observed records stays unknown. No inference, privacy permission, model routing or egress policy is introduced.

### native peer fixture ownership

This card owns one minimal test-only native loopback peer fixture shared by all three cases, beside upstream.rs unit tests where its cache state is visible. Reuse native rustls TLS/config helpers and canonical QUIC/H3 driver/listener machinery: tests/native_round_trip.rs:30,34,61,81,94,111 documents certificate/config/datagram pumping; native/server.rs:75,96 exposes native server facade/drive; proxima-quic/src/native/listener.rs:317 exposes connection_mut for deterministic native close. Existing protocol helpers can be lifted/reused privately only as needed; no new mock-transport framework or production transport abstraction. Existing pure-proto driver tests alone cannot prove facade replay behavior, since the defective outer loop lives in H3NativeUpstream::call.

The actual client under test must be H3NativeUpstream::with_client_config + SendPipe::call, using a native rustls QUIC peer over local ephemeral UDP (CPU-only, no external network/hardware/model). The peer drives native listener/server connection state and captures decoded RequestHeaders, complete DATA payload and request FIN, with connection identity, stream identity and ordered raw packet/stream payload records. After a warmup/preflight completes and the upstream cache is populated, read the entire second POST body/FIN; only then close that native connection before writing response HEADERS. Use a native close event or explicit channel-driven server action, not elapsed sleeps. Keep accepting fresh native connections long enough to observe an erroneous replay during the unfixed baseline. Completion/close coordination must use owned channels or existing async gates; no synchronous mutex is added. Typed codec/driver errors are retained and not relabeled as zero native work.

Fixture complexity boundary: source policy is one branch, native peer evidence is part of this same card. If existing native listener primitives cannot express the consumed-body/close transition without a new production abstraction or a separate fixture project, report the exact missing API/ownership to root before code; root must name/register that independent prerequisite before implementation, not replace the desired payload observation with a Method-only mock. No protocol codec/reliability rewrite is permitted to simplify the fixture.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| policy | canonical H1-owned Method semantics at existing retry branch | discarded duplicate POST list and new RetryFramework: semantics already have an owner |
| delivery proof | actual native peer consumes body/FIN before close | “idle stale” or missing response cannot establish non-processing |
| fixture | existing native TLS/listener/driver with real facade call | pure proto or fake method evaluator never reaches the defective replay loop |
| ownership | one test-only peer fixture and three cases | no new transport/mock framework, production lock or global disable |

## acceptance criteria

Prospective/unmeasured. Exactly three named non-parameterized async cases in namespace `http3_replay_safety`; additional method/failure vectors remain assertions inside these cases. Use #[proxima::test] with native prime-capable runtime, ephemeral loopback ports, existing native rustls trust fixture and deterministic event/channel progress. No sleeps. Tests retain actual consumed bytes/headers/FIN, connection/stream IDs, close/error phase and returned error/response plus subsequent state. Test-owned packet buffers may allocate; no hot-path performance claim.

Mandatory baseline before policy edit: run desired ambiguous_post_not_replayed case on UNFIXED upstream with full native fixture. Expected desired assertion failure with two peer-consumed identical POST bodies on distinct connection identities. Capture exact transcript and failure, including warmup and both deliveries. If this is not observed, instrument/narrow the native path before editing the guard; a source guess does not replace reproduction. After repair, rerun identical desired fixture and GET semantic control. No native model call or public service is needed.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-http --lib --no-default-features --features http3-native-upstream -E 'test(http3_replay_safety) & test(ambiguous_post_not_replayed)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; complete warmup then consumed POST/FIN before native close has exactly1 original body,0 replay bodies/second request connections, original failure preserved; next independent call succeeds on fresh connection |
| AC2 | R2 | `cargo val -p proxima-http --lib --no-default-features --features http3-native-upstream -E 'test(http3_replay_safety) & test(idempotent_recovery_bounded)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; same native fixture GET recovers once and returns200; failing recovery stops after exactly2 request attempts, no third connection/request; exact method/head/body records retained |
| AC3 | R3 | `cargo val -p proxima-http --lib --no-default-features --features http3-native-upstream -E 'test(http3_replay_safety) & test(fresh_and_unknown_not_replayed)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; fresh native failure has1 attempt and original error, reused POST/PATCH/CONNECT/custom/lowercase methods never resubmit ambiguous body; canonical idempotent classification is reused, actual native observations accompany method matrix |

Combined: `cargo val -p proxima-http --lib --no-default-features --features http3-native-upstream -E 'test(http3_replay_safety)' --no-tests fail --output plain` expects3passed,0failed,0ignored,0not_run. Explicit feature selects the native branch (Cargo.toml:200), not `http3` umbrella which also enables quinn compatibility. Existing local native TLS/mock fixtures are CPU-capable; unavailable external hardware/model is not an explanation for skipping this contract.

Selection control: `cargo val -p proxima-http --lib --no-default-features --features http3-native-upstream -E 'test(http3_replay_safety_does_not_exist)' --no-tests fail --output plain` expects selected0/run0/passed0 with explicit no-tests refusal and nonzero exit. Empty selection is not success. Retain actual commands, selected counts/stdout/stderr, native payload traces and source hashes in external evidence; no counters-only or README claims.

## out of scope

- Changes to completed H1/Method/canonical classifier sources; consume existing `Method::is_idempotent()` directly without redefining its semantics.
- QUIC codec/TLS/reliability/scheduler changes, quinn compatibility rewrite, new mock framework/client backend, general retry configuration or service product.
- Native model calls, paid API, privacy/egress grants, performance/compute utility claims, commits/pushes/PM mutation.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| fixture misses real outer replay | possible | predicate test masquerades as delivery proof | require actual H3NativeUpstream::call and peer payload duplicates on unfixed code |
| close occurs before consumption | possible | wrong delivery class tested | peer body/FIN receipt barrier precedes close and raw records retain order |
| reuse docs differ from source | observed | incorrect design premise | source cache/was_reused branch is authority, not outdated per-call docs |
| fixture becomes new infrastructure | possible | unbounded card | reuse native listener/driver/TLS; exact missing seam routed to root before separate prerequisite |

## context

- proxima-http/src/http3/native/upstream.rs:65 cached native client; :211 two-attempt loop; :240 error/cache cleanup; :312 establishment; :368 request open/body/FIN; :418 missing-status error; :914 existing unit tests.
- proxima-http/src/http3/native/server.rs:75 constructor; :96 drive; proxima-quic/src/native/listener.rs:317 connection_mut.
- proxima-http/tests/native_round_trip.rs:30 TLS provider, :34 config, :81 datagram drain, :94 pump, :111 establishment, :128 real native proto round trip (reuse helpers, not an outer replay oracle).
- proxima-http/Cargo.toml:174 http3-native, :200 http3-native-upstream; no http3-quinn-compat required.
- docs/proxima-quic/guiding-principles.md:27 reuse/ownership, :60 RFC compliance; docs/proxima-h3/guiding-principles.md:34 source-faithful tests, :47 typestate; ai_docs/invariants.jsonl relevant QUIC/H3/sans-IO records.
- proxima-primitives/src/pipe/method.rs:113 exact `Method::is_idempotent(&self) -> bool`; :119 borrowed-token canonical classification; proxima-primitives/src/pipe/mod.rs:175 exports `proxima_primitives::pipe::Method`. proxima-http/src/http1/client.rs:1070 uses `reused && request.method.is_idempotent()`; H3 consumes the same inherent API, without modifying its owner.
- proxima-tensor/specs/http1-replay-safety/SPEC.md:1 canonical predicate ownership and required class split; external `/Users/brianbruggeman/.local/share/ragd/pm-exports/hierarchical-intelligence/execution/http_replay_h1-record-readback.json` records PROXIMAHTTP-2 stage=done. H3 remains a separately required class site; this document makes no whole-class completion claim.
- RFC9110 section9.2.2 https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2 and RFC9114 section4.1 https://www.rfc-editor.org/rfc/rfc9114.html#section-4.1; primary method semantics already opened during class preparation.
