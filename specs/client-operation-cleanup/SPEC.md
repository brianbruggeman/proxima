# client-operation-cleanup

status: proposed v5 — immutable external author revision; separate source card and fresh critic/QA/spec-auditor admission required before code
owner: brian / existing Proxima Client
created: 2026-11-05
role: existing proxima-architect AUTHOR; root-selected Sol, not Opus
supersedes for review: external proposal-v4/SPEC.md and TASKS.md; all earlier bytes preserved

## problem

Consumers of an observed Proxima Client call need a read-only operation-cleanup receipt that cannot fire from cancellation lineage or an earlier replayed request and resolves only after request-owned I/O/streams drop, measured by literal actual-worker event traces including pending initialization, header, body and forwarding phases.

## refutation condition

A receipt that resolves while its call still owns an upload, pending factory/request I/O future or actual upstream response stream, or a worker cancellation that needs more remote bytes or relay capacity before releasing those resources, refutes this contract.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Observed-call APIs mint a private independent root completion Signal and expose only existing read-only Fired; no completion field/setter exists on RequestContext, so cloned/replayed contexts cannot contain a stale receipt. | yes: lineage/replay can fail while drop ordering holds |
| R2 | Observed send/call future construction arms completion synchronously and retains input/upload before completion guard in one drop-ordered state; unpolled drop and all local errors destroy local resources before receipt completion, with zero sends. | yes: dispatched behavior may pass while unpolled upload ordering fails |
| R3 | Explicit cancellation races local handle initialization and worker request I/O; cancellation destroys the actual initializer/wait/request future before completing its receipt. | yes: body behavior can pass while Pending factory construction hangs |
| R4 | Actual worker response/body destruction or cancellation interrupts pending upstream next and blocked forwarding without remote bytes/channel capacity, drops upstream resources, then completes. | yes: head cancellation can work while body/forward remains pending |
| R5 | Inline cleanup is drive-or-drop: cancellation alone does not promise cleanup of a retained unpolled response; dropping the owned response/stream or driving its cancellation branch completes after resource destruction. Preserve full body bytes and the explicitly stated baseline trailer-conversion behavior. | yes: worker behavior can pass while retained-inline or conversions fail |
| R6 | One receipt covers one Client dispatch including its configured inner retry attempts; distinct observed cloned calls mint independent receipts, while an external Retry<Client> exposes no invented logical receipt from inner attempts. B's existing owned task holds its existing gate through full-body drop and receipt. | yes: basic call completion may pass while replay or consumer ownership releases early |
| R7 | Independent literal checker accepts compliant worker/inline/pooled traces and rejects ancestor-fire, stale-attempt, upload-after-ack, old-relay, cancel-as-complete and early-release controls; prove actual default-prime worker and native TCP paths. | yes: an implementation can satisfy its own assertions with a permissive checker |
| R8 | Real normal library graph compiles without std, public docs execute nonempty observed-call examples, and the existing finite signal example executes the new API; lib-test feature unification and list-only checks do not discharge these gates. | yes: unit tests may pass while no_std/docs/example consumers fail |

## architecture

### selected API: private root completion, existing read-only observer

No new type or primitive is proposed. The existing pipe expression is `SendPipe::call(&client, request): impl Future<Output=Result<Response<Bytes>,ProximaError>> + Send`. It already owns a complete call. The missing observable fact is whether that operation's owned I/O is gone; attach an existing privately owned SignalGuard to that operation and return existing `proxima_core::signal::Fired`. This is a tuple of existing values, not a generic Lease or a relocated handler trait.

Before: `let response = client.get(path).send().await?;` cannot observe worker cleanup after cancellation. After: `let (operation, cleanup) = client.get(path).send_with_cleanup();` owns the concrete call future and an independent read-only cleanup future; the owner drops/drives the full body and then awaits cleanup before releasing its resource. A caller can now wait for actual operation cleanup without being given the ability to fire it. The two lines differ in capability, not in a wrapper's name.

```rust
// src/client/request.rs: existing Client RequestBuilder
#[must_use]
pub fn context(mut self, context: RequestContext) -> Self;

pub fn send_with_cleanup(self) -> (
    impl Future<Output = Result<Response, ProximaError>> + Send + 'static,
    Fired,
);

// src/client/handle.rs: existing Client; lower typed Request/Response form
pub fn call_with_cleanup(&self, request: Request<Bytes>) -> (
    impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'static,
    Fired,
);
```

Import Fired from `proxima_core::signal` at module scope; its concrete public type is already exported there and needs no wrapper/re-export. The builder's existing `send().await` and Client's existing SendPipe::call retain their public signatures and return types. Both observed methods are synchronous constructors of concrete futures, arming before first poll. Both call one private lifecycle implementation shared with ordinary dispatch. `context` stores the existing RequestContext and forwards it through the already existing lower RequestBuilder::context; default remains RequestContext::default. There is NO operation_cleanup field or caller-provided completion Signal on RequestContext. No changes to its production layout, serde, Clone, replay or no_std representation are needed.

Each observed API call executes `let completion = Signal::new(); let observer = completion.fired();` privately, creates its SignalGuard, and exposes only observer. Neither a caller's cancel Signal nor a child/ancestor of it is ever used as completion. Fired has poll/await observation but no fire/child method. No caller freshness/lineage precondition substitutes for that construction. A cloned Request passed to another call_with_cleanup gets a new private root and a new observer. Ignoring/dropping the observer does not cancel the operation or change cleanup ownership. Dropping a builder without invoking send_with_cleanup starts no observed operation; invoking it and dropping the returned future unpolled does start a cleanup obligation.

Completion means Client's call no longer owns a live request upload, pending initialization/request I/O future or actual upstream response stream. It does not mean response success, complete delivery of buffered relay bytes, closed pooled TCP socket, remote EOF already observed, or completion of arbitrary independently detached work by a custom factory/handler. The actual request factory/future/stream must own the I/O covered by this contract; document that boundary. No trace is labeled TCP-closed solely from this receipt.

### synchronous input-drop ownership

For typed call_with_cleanup, build ONE captured tuple `(request, completion_guard)` synchronously, with Request first. Capture that tuple intact in the returned concrete async future until its first poll. Never capture upload and completion guard as independent async captures whose order is left implicit. Unpolled future destruction drops Request and its RequestStream before SignalGuard fires completion. The builder variant similarly captures `(builder, completion_guard)`; lower request construction errors drop the builder/request state before firing completion. Tests record actual upload-stream destructor, not merely zero send count.

After first poll, transfer ownership explicitly: request/input before completion, pending init/call future before completion, actual upstream response stream before completion. Use lexical scopes and an intact tuple or explicit drop ordering; an Either returned by select can own its losing future, so drop the entire scoped selection state before completing. No new pinned wrapper/state type is necessary. Existing SignalGuard/tuple and the concrete async block supply this ownership. No Box<dyn Future> is added except the existing object-safe Runtime handoff discussed below.

### cancellation, initialization and worker phases

Each operation derives a fresh child from existing request.context.cancel, assigns that child only to the operation's request, and retains an existing cancellation SignalGuard for disposable caller/response lifetimes. Never fire the original parent or shared NEVER_CANCEL. A parent canceled before first poll causes zero actual sends and no initialization work beyond local teardown. This child-scope allocation is not advertised as zero-cost or allocation-free.

Cancellation is selected left-first when already ready. Before any worker submission, race `child.fired()` against `self.handle()` INCLUDING OnceCell get_or_try_init and factory load/build await. The branch owns the handle-initialization future and drops it before completing the call receipt. If this call is the initializer, its pending build future and its owned resources must drop. If this call only waits behind another operation's initializer, drop this call's wait; the unrelated initializer may continue and is not declared cleaned up by this call's receipt. A canceled initialization must not leave OnceCell permanently occupied: another uncanceled call can subsequently initialize through the existing OnceCell contract. No shared global cancellation is sent to unrelated callers.

After a successful handle load, move the operation's completion guard into the existing worker task BEFORE Runtime submission. If spawn rejects and consumes the task, its owned input drops before completion and actual sends remain zero. After accepted submission the disposable caller owns a child-cancel guard and receiver, while the worker owns request/completion. Explicit parent/child cancellation and the existing pre-head receiver-canceled observation race the actual SendPipe call. Exit/destroy the scope containing pinned call and select/Either before firing completion. Preserve existing local/load/spawn/transport error classification. Explicit pre-header cancellation retains `ProximaError::Upstream("client dispatch cancelled".to_owned())`; an absent dropped caller receives no fabricated output.

After headers, the worker owns the actual upstream ChunkStream and completion guard. Independently race child cancellation against BOTH a pending upstream next() and pending bounded channel forward send. Do not depend on a new remote byte, a free channel slot or an unrelated task barrier to notice cancellation. Drop losing read/send futures and actual stream before completion. Keep RELAY_CHUNKS_IN_FLIGHT unchanged and retain the original metadata/status/bounded byte stream.

The consumer's relayed ResponseStream is composed with an existing cancellation guard; dropping whole Response, ResponseStream or its converted chunk stream fires child cancellation. The consumer wrapper watches the child: explicit cancellation with a live consumer yields existing `ProximaError::Body("body cancelled".to_owned())` once and terminates, without requiring the worker to enqueue that error into a full queue. Normal EOS is not an error. Completion may precede consumption of already-buffered relay chunks because they own no request I/O. Channel cancellation is not completion: only the worker drops its actual stream and then the private completion guard.

### inline drive-or-drop and baseline conversion behavior

Inline dispatch DOES NOT spawn an extra background driver or change runtime/wire. Its actual response stream is lazy and remains caller-owned. Compose existing futures::stream::unfold with an intact state whose actual stream precedes completion guard in drop order; retain that state across cancellation/next awaits and carry existing trailers_slot when rebuilding ResponseStream. Inline explicit cancellation is processed only when that owned wrapper is polled or dropped. If caller retains an unpolled response, or drops only a borrowing next() future while retaining the underlying stream, the cleanup observer MUST remain Pending. General runtime progress elsewhere is insufficient. Public docs and tests state this precondition.

B's supported sequence is stronger and explicit: its existing owned request task races its ORIGINAL budget/cancel against a future that OWNS the complete response/body stream. When cancellation wins, destroy that entire losing future and its owned stream (not merely a borrowed next()), fire cancel, then await cleanup. This discharges inline drive-or-drop without an added driver. The retain-unpolled case is a required negative state test: cancellation does not spuriously acknowledge; full stream drop then makes cleanup ready. No poisoned owner, leak or time-based guess substitutes for drop.

Preserve full bounded body bytes, headers and status in every conversion. Preserve the existing trailers slot where it exists on lower Response.stream and in the worker's relay/wrapper. The existing public client Response::into_body calls lower into_chunk_stream, which explicitly discards trailers; lower Response::into_chunk_stream also discards them. v2 NORMATIVELY preserves that baseline behavior and documents it; it does not promise new trailer preservation through those conversions or change body representation. Tests assert the original slot remains accessible before conversion, converted stream has no slot, and all payload bytes remain exact. bytes/text/json consume the original content unchanged. This is not permission to drop any body bytes, to erase Retry-After headers at the HTTP head, or to conceal conversion behavior from Ragd. A separate trailer API repair is outside this lifecycle card.

### clone/retry semantics: receipt belongs to outer Client call

Completion is not request context data, so build-once/re-send-many Request::clone and HttpReplay context cloning cannot duplicate or inherit an observed-call receipt. Existing Retry<Client> continues calling Client's ordinary SendPipe path with existing semantics and has no access to a per-attempt observer. The new Client-specific observed methods are not methods of arbitrary Retry<Inner>. Consequently an observer returned by an inner Client call is never described as completion of an external enclosing Retry<Client>.

To observe a configured retry sequence as ONE logical operation, compose the EXISTING retry middleware inside the Client spec and call the outer Client's send_with_cleanup or call_with_cleanup. The one outer completion owner spans handle dispatch, every internally configured retry attempt, and the final actual response stream. It does not fire at first attempt completion. Test a literal retryable first response and final streamed response while final read is Pending: outer observer stays Pending until all owned intermediate outcomes are destroyed and final body ends/drops. This repairs receipt semantics; it does not add/enable retries in Ragd B. External Retry<Client> remains usable but offers no logical cleanup receipt through this Client-only API; obtaining that distinct outer capability is outside this bounded public surface, not a hidden stale-ACK precondition.

### owned B task and exact Send/runtime boundary

Use the existing `Runtime::spawn_on_core(CoreId(0), Box::pin(operation))`; its actual signature requires Future<Output=()> + Send + 'static and returns Result<(),SpawnError>. This sole box is required at the EXISTING object-safe runtime boundary and carries a one-line reason; no extra erased future is introduced. Do not call RuntimeExt::core(...).spawn from an arbitrary off-worker caller because its placement logic evaluates current_core. No new runtime/service/global.

The owned async move task captures the caller-owned Arc<AsyncMutex<u64>>, owned Client/request data, cancellation Signal and result sender. Acquire the borrowing guard INSIDE that task. Construct `(request_future, cleanup)` through the new observed API only after admission; on cancel/deadline destroy the complete owned request/body scope, request cancellation and await the retained Fired before releasing guard. Do not move a caller-borrowing guard into a static task. For B's injected clock require `C: Clock + Clone + Send + Sync + 'static`, `C::Delay: Send`; compile the actual composed owned operation at the existing Runtime boundary. Spawn rejection acquires/sends nothing; cancellation before constructing an observed future also sends nothing and owes no unarmed receipt await. After future construction even an unpolled drop has a receipt. Later callers retain their finite budgets during cleanup. No send/retry occurs after original cancellation/deadline; cleanup has no guessed grace cutoff.

The runtime must continue driving an accepted worker task. Inline owner drive-or-drop is a separate precondition, not implied by runtime progress. Crash/abort and a third-party handler deliberately detaching unrelated I/O are outside this async ownership contract.

### physical connection and feature boundary

Receipt is operation cleanup, not physical socket closure. H1 finish can park a connection at client.rs:849–864; Prime Drop can defer owner-reactor release at net.rs:619–624,1332–1342; poll_close is a write half-close. B still needs a reviewed physical pooling/closure policy if it promises at most one OPEN connection rather than at most one in-flight request operation. Global controlled-machine egress remains unmeasured. Do not grow this card into TCP/reactor/pool policy work.

Actual loader source src/load.rs:329–358 makes prime the default http backend when available even with http-hyper enabled; explicit wire tokio chooses the separate alias. Test default-prime worker and native TCP paths plus explicit tokio/inline compatibility. No production wire/runtime switch. Candidate source is HEAD 9bb9ce6984b9db616b5e79eb79dc26f49233e89e plus dirty overlay, pinned under this immutable v4 directory.

### worked trace and shared controls

Hand-derived oracle, not captured runtime evidence:

| order | event | assertion |
|---|---|---|
| 1 | observed API creates private completion root and captures (upload request, guard); B task acquires service gate | cancel cannot fire that root; request owns upload |
| 2 | handle factory registers Pending; cancel fires | same call's initializer future must drop before receipt; no actual send |
| 3 | a separate observed call loads successfully, worker sends, returns 429/Retry-After 3 and body prefix, upstream read Pending | header/body prefix literal, actual active I/O recorded |
| 4 | cleanup observer Pending; caller drops complete response | child cancel requested, receipt still Pending until worker destroys I/O |
| 5 | worker cancellation wakes; actual read future/stream drop while remote tail gate remains closed | no remote byte or forward capacity required |
| 6 | private completion fires; observer wakes | after actual upstream drop |
| 7 | B owner observes cleanup and releases service guard; next permitted call may send | no operation overlap |

Also accept a normal pooled-idle trace: active I/O ends and stream returns connection to pool before receipt, while socket remains open. Also accept the explicit inline retained-unpolled state: parent cancel, observer Pending, full stream Drop, observer Ready. The same checker rejects ancestor-derived completion firing before worker drop, previous-attempt observer used for a later call, upload_drop after completion despite zero sends, old relay requiring remote bytes, cancellation-as-completion and early gate release. Each rejected fixture differs in its named invariant; known compliant literal fixtures must first pass the checker. Observer Pending/wake is driven by actual waker gates/MockClock; an unrelated worker barrier alone does not prove this operation was polled. Keep complete source/attempt/event records, not only aggregate counts.

## acceptance criteria

Prospective only: no compilation, test or provider/model execution occurred. Run from the isolated candidate worktree. Exact new umbrella physical bodies remain six; one primitive compatibility body is added in existing request.rs test module. Table subcases do not inflate physical counts. Gated docs/example additions are described below and must exist before admission of source hunks.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1,R2,R3,R4,R5,R6,R7 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest list --offline -p proxima-lib --lib --features http-prime -E 'test(client_cleanup_)' --run-ignored all --ignore-default-filter --message-format json` | exactly 6 physical bodies listed, 0 ignored |
| AC2 | R1,R2,R3,R4,R5,R6,R7 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-lib --lib --features http-prime -E 'test(client_cleanup_)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | 6 passed, 0 failed, 0 skipped; G4-M2 inherited-output PASS and G4-T literal-positive/negative checks below are mandatory |
| AC3 | R1,R3,R4,R5,R6,R7 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-lib --lib --features http-prime,http-hyper,runtime-tokio -E 'test(client_cleanup_)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | same 6 passed, 0 failed, 0 skipped; G4-M3 wire-labeled inherited-output PASS and G4-T checks, including explicit tokio/inline table |
| AC4 | R6 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest list --offline -p proxima-primitives --lib -E 'test(client_cleanup_request_clone_preserves_cancel_without_completion_state)' --run-ignored all --ignore-default-filter --message-format json` | 1 physical compatibility body listed |
| AC5 | R6 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-primitives --lib -E 'test(client_cleanup_request_clone_preserves_cancel_without_completion_state)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | 1 passed, 0 failed, 0 skipped; G4-M5 literal clone/cancel check with its B3 incumbent PASS control |
| AC6 | R3,R4,R5 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-lib --lib --features http-prime -E 'test(client_send_auto_dispatches_when_off_worker) | test(off_worker_send_reads_body_larger_than_head_scratch) | test(off_worker_send_dropped_by_caller_closes_)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | 4 existing regression bodies passed, 0 failed, 0 skipped; G4-M6 same existing literal/peer checks with B1/B2 incumbent PASS controls |
| AC7 | R8 | `python3 /Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/proposal-v5/validation-gates.py no-std .` | 1 normal LIB artifact with std absent accepted; 1 std-enabled normal LIB control rejected by same checker; 0 dev/test graph used |
| AC8 | R1,R2,R5,R6,R8 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo test --offline -p proxima-lib --doc --features http-prime send_with_cleanup -- --nocapture` | exactly 1 new runnable public-method doctest passed, 0 failed, 0 ignored; buffered body plus receipt assertion; G4-M8 same byte check as B4 incumbent, G4-T receipt check |
| AC9 | R1,R2,R6,R8 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo test --offline -p proxima-lib --doc --features http-prime call_with_cleanup -- --nocapture` | exactly 1 new runnable public-method doctest passed, 0 failed, 0 ignored; owned typed request unpolled-drop receipt assertion; G4-M9 same resource check as B4 incumbent, G4-T receipt ordering |
| AC10 | R1,R6,R8 | `python3 /Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/proposal-v5/validation-gates.py signal-example .` | existing signal example exits after own assertions; exactly 1 new literal `client cleanup: body=cleanup, receipts=1` line; existing late-observer assertion still runs; G4-M10 same B4 inherited checks, G4-T new receipt check |

### G4 explicit incumbent PASS controls and predicate boundary (v5)

This subsection is normative for AC2, AC3, AC5, AC6, AC8, AC9 and AC10. It does not turn compatibility output from modified candidate production code into an incumbent observation. There are two distinct independent checks:

- **G4-I, inherited facts:** literal fixture equality (status/body bytes, explicitly asserted labels/request fields, upload destruction/zero upload polls, or sticky Signal observations). Compare actual output against the independently written fixture literals, never against another SUT output. The SAME equality predicate and SAME fixture values are required on the incumbent and candidate for each matched subcase. An additional candidate fixture such as a new header is checked against its own independent literal; a baseline row that did not observe it supplies no incumbent coverage of that field.
- **G4-T, new lifecycle facts:** the hand-derived worked trace above and its explicit compliant normal/pooled/retained-inline vectors are the oracle for resource-drop-before-receipt-before-release and Pending states. Before applying a trace predicate to any observed call, run its compliant literal vector through that identical predicate and expect acceptance; run each applicable ancestor-fire, stale-attempt, upload-after-ack, old-relay, cancel-as-complete and early-release vector through it and expect rejection. Fixtures and checker must not call the cleanup implementation to compute the expected result. No receipt is inserted into an incumbent record.

**Commanded incumbent control phase — before the first production edit.** Use the CURRENT isolated `/Users/brianbruggeman/repos/slot-0/proxima-client-cleanup-wt`, whose only admitted source-card changes so far are spec Markdown. Verify the pinned production baseline/dirty overlay against source-ledger.json before this phase. No second checkout, helper crate, capture producer or new target is required. Baseline-only instrumentation may add ordinary-API fixture assertions to the EXISTING `examples/signal/main.rs` and EXISTING `request_context_clone_and_peer_label_work_without_std` body; it must not change Client/Signal/RequestContext/Request/HTTP/runtime production statements or add observed methods. Record and independently inspect that test/example-only diff before running. Later candidate changes retain those literal assertions unchanged and add the new observed cases. This is a prospective test-fixture step on the separately registered source card, not permission to edit source in this author revision.

The baseline ordinary fixture in the existing signal example uses `Client::from_value` with a synth response body literal `cleanup`, ordinary builder `.send()`, and separately the existing typed `SendPipe::call`. The typed request's public `stream` owns an existing RequestStream built from a chunk stream whose state owns an existing SignalGuard; drop of that state fires a private upload-destruction probe. Construct the ordinary typed call and drop it UNPOLLED, then check actual upload-stream poll count `0` and that destruction probe `true`. The probe observes upload destruction only; it is neither an operation cleanup receipt nor a substitute for one. Existing stream composition/Signal values suffice; no new probe/lease type. The ordinary fixture has no new observed API dependency. Its exact marker `client incumbent: body=cleanup, upload_polls=0, upload_dropped=1` is emitted once ONLY after the byte/resource assertions. Retain the existing sticky Signal/late observer assertions and their actual poll counts. In the existing primitive context test, construct a buffered request with literal method `POST`, path `/items`, payload `created`, pipe label `my-pipe` and upstream label `my-upstream`, clone it, and check each cloned field against those literals; assert the existing cancellation relationship using existing Signal, without any completion field. These test-only assertions give baseline outputs the specific shared checks actually require; they add no physical body.

Run B1–B4 from that worktree, before production changes; every expected PASS is prospective, not an author runtime claim:

| incumbent control | exact command | expected counted assertions/output |
|---|---|---|
| B1 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-lib --lib --features http-prime -E 'test(client_send_auto_dispatches_when_off_worker) | test(off_worker_send_reads_body_larger_than_head_scratch) | test(off_worker_send_dropped_by_caller_closes_)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | exactly the 4 AC6 incumbent functions PASS, 0 failed/skipped/ignored; their actual literal payload/peer checks execute |
| B2 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-lib --lib --features http-prime,http-hyper,runtime-tokio -E 'test(client_send_auto_dispatches_when_off_worker) | test(off_worker_send_reads_body_larger_than_head_scratch) | test(off_worker_send_dropped_by_caller_closes_)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | same 4 incumbent functions PASS under the compatibility feature graph; 0 failed/skipped/ignored; default-prime execution is not relabeled explicit tokio wire |
| B3 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo nextest run --offline -p proxima-primitives --lib -E 'test(request_context_clone_and_peer_label_work_without_std) | test(cancel_signal_child_derivation_under_std)' --run-ignored all --ignore-default-filter --status-level pass --final-status-level none` | exactly 2 existing bodies PASS, 0 failed/skipped/ignored; literal request/context fields and cancellation assertions execute |
| B4 | `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo run --offline -p proxima-lib --example signal --features tokio` | exactly 1 incumbent marker after the ordinary byte/input-destruction checks; existing consumer poll count 2 and late-observer poll count 1 asserted; 0 observed-API receipt claims |

Before each B1–B3 run obtain ignored-inclusive JSON inventory with the same package/features/filter via `cargo nextest list` and `--run-ignored all --ignore-default-filter --message-format json`; require exactly the named functions, zero ignored, then bind PASS identities to that inventory using the v4 parser rules. Freeze the actual executed binary paths/hashes, complete source/fixture diff identity, features, argv/cwd, raw stdout/stderr and failure status in an exclusive external per-run evidence directory under validation-runs; use v4 evidence-ownership requirements. A failed or missing control is not a baseline PASS. No build or source test is run in this author pass. This phase is separate from the candidate full-contract ten-record ledger, and cannot be substituted by running ordinary APIs AFTER production edits. Candidate completion evidence must link this phase's immutable raw records; source review checks that only the declared fixture assertions changed before it and that shared check expressions/fixtures remain identical afterward.

| mapping | candidate AC/check | explicit incumbent-output PASS required | newly introduced facts and boundary |
|---|---|---|---|
| G4-M2 | AC2: inherited G4-I assertions in the six existing proposed bodies | B1 actual status `200`, exact `hello` bytes, exact independently generated large-body bytes, and peer `eof`; B3 literal clone/cancel facts; B4 ordinary `cleanup` bytes and unpolled upload-poll/destruction facts pass the identical corresponding literal/resource predicates used by candidate subcases | G4-T literal compliant vectors PASS before observed receipt traces; all applicable named bad vectors reject. Headers, trailer conversion, factory/backpressure and receipt predicates not present in B1–B4 retain their independent literal oracle; do not claim old tests observed them. |
| G4-M3 | AC3: same G4-I/G4-T under selected features | B2 plus B3/B4 cover only their named inherited facts; each candidate inherited matched fixture passes the same check | B2 is a compatibility FEATURES control, not an explicit tokio-wire proof. Explicit tokio/inline new cases use independent literal positive/negative vectors; old wire-specific receipt output does not exist. |
| G4-M5 | AC5: cloned fields equal `POST`, `/items`, `created`, `my-pipe`, `my-upstream`; existing cancellation relationship | B3 actual old Request/context clone output must PASS the SAME field-literal/cancel predicates; comparing clone only to original is insufficient | No cleanup receipt field or predicate is added to this primitive path. |
| G4-M6 | AC6: inherited exact status/body and peer-EOF checks | B1/B2 incumbent outputs must PASS the SAME existing independent equality predicates, unchanged in candidate | Source assertions are handle.rs:1005–1025, :1050–1115, :1220–1253. These do not observe new pending-body receipt ordering or equate peer EOF with receipt. |
| G4-M8 | AC8: body bytes equal literal `cleanup` | B4 ordinary baseline send's actual bytes PASS the SAME `actual_bytes == b"cleanup"` check as observed send doctest | Doctest separately runs G4-T buffered/drop-before-receipt literal positive and wrong-order negative before checking its actual new receipt. No old send_with_cleanup exists. |
| G4-M9 | AC9: unpolled owned request gives `(actual_upload_polls, upload_probe_fired) == (0,true)` | B4 ordinary typed-call actual observations PASS the SAME resource predicate as new typed-call doctest | Receipt readiness/order uses its G4-T literal positive and upload-after-ack negative, then actual observed call. Zero actual sends remains a separate new-call assertion from the selected trace fixture; the old probe signals only upload destruction and its poll counter, not an instrumented backend send count. |
| G4-M10 | AC10: existing sticky Signal/count assertions and ordinary `cleanup` body check | B4 actual old Signal/ordinary-call observations PASS the SAME literal/count/byte predicates retained in candidate example | New receipt marker occurs only after independent G4-T positive/negative and actual observed receipt assertions. It is a sentinel, never an incumbent receipt. |

**Exact G4 rule-domain residual for re-audit.** spec-auditor.md:60–64 allows a hand-derived worked example and independently also requires incumbent output to PASS the same check. The incumbent Client has no send_with_cleanup/call_with_cleanup or operation receipt (pinned request.rs:80–98 and handle.rs:484–496); its pending old relay is intentionally a negative control for the new contract. A requirement that this incumbent PASS the FULL new ACK predicate is therefore unsatisfiable: it has no ACK observation, and adding/fabricating one would alter the subject. V5 supplies explicit real incumbent PASS for every matched inherited projection and independent compliant-literal PASS for the new predicate, while exposing which additional new-only facts lack an incumbent counterpart. It does NOT silently waive the conjunct, supply a synthetic incumbent receipt, or assert that formal G4 now passes. The formal auditor must resolve that application boundary on these exact bytes; the existing REFUSE remains in force until re-audit. This residual changes no selected API, requirement, physical test count or lifecycle assertion.

### v4 evidence ownership and ignored-aware enumeration

Every runner invocation exclusively creates a fresh `run-*` directory under external `/Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/validation-runs`, or an explicit external CLIENT_CLEANUP_EVIDENCE_ROOT. Reject an evidence root inside the candidate worktree. The unique directory is allocated with mkdtemp before the first command; the initial manifest is running and its counted ledger is an empty array. No prior run file is reused. Raw command artifacts use exclusive creation. The per-run manifest binds run ID, worktree absolute/git/common paths, HEAD, tracked dirty-overlay hash, untracked input-byte hashes, proposal/runner input hashes, actual cargo executable hash, exact argv/cwd, raw output paths/hashes, return status, counted AC records and terminal completion/failure. The full-contract run cannot be completed if source/worktree identity differs at the end. Evidence assumes exclusive candidate-worktree ownership; the endpoint hash comparison is not a concurrent mutation detector. Interrupted runs lacking completed status are not complete evidence.

Before each command, persist its running command record. On nonzero command or parse/count failure, persist failed run status, the failing phase/error and current partial ledger; raw outputs remain bound only to that run. Finally persist finish time and ledger hash even on failure. A successful earlier run and an early-failed later run must have distinct exclusive directories, and the failed run must expose neither old AC outputs nor an old complete ledger. Read only each run manifest's referenced outputs; do not merge files across directories or choose a successful older run as current evidence.

Nextest 0.9.146 installed help and the opened existing real JSON list are frozen under format-evidence. The JSON schema source has rust-suites keyed by binary identity; each suite has package-name/binary-id/kind/status and testcases keyed by full module-qualified test name, with explicit ignored boolean and filter-match.status. AC1/AC4 use `--run-ignored all --ignore-default-filter --message-format json`. Require exactly the declared full-name set in exactly one expected package library binary, zero ignored matching records and matches filter state. A matching family body hidden by a mismatching filter state is rejection, not absence. Never infer zero ignored from default human listing.

Every run uses that same ignored-inclusive selection. Default body runs reuse their identical prior JSON inventory; compatibility-feature and regression runs obtain their OWN JSON inventory before execution, validate their exact full-name set/ignored state, then check full binary/name PASS identities against that inventory. This adds enumeration evidence without changing feature or physical-body scope. Require status-level pass/final-status-level none from the installed help to make executed identities available without duplicated final PASS rows. An extra ignored matching body must be rejected by the same parser before any source test execution.

Only bounded SYNTHETIC runner controls are authorized in this proposal pass. Their manifests must say synthetic-runner-control and identify the fake command executable, never actual-cargo-validation. They test parser/evidence behavior, not Client cleanup, compiled function counts, actual no_std, docs/example execution or provider behavior. Required controls: literal compliant JSON inventory accepted; same inventory plus extra ignored matching body rejected; a completed synthetic full-contract run followed by an early-failed second synthetic run yields separate directories, failed terminal manifest and no stale later outputs/complete ledger in the latter.

### complete-behavior task validation and final contract gate

The implementation is ONE complete behavior unit in TASKS.md, with no partial slice claiming a broader AC. Its actual validation command is `python3 /Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/proposal-v5/validation-gates.py full-contract .`. That command runs each AC1–AC10 cargo invocation separately, saves each raw stdout/stderr and a counted record, and rejects any missing/duplicate physical name, missing PASS record, zero doctest match or missing example assertion marker. Process exit alone never supplies a count. No source/test command was executed by the author.

The final whole-contract gate is the same full-contract command, run after the final candidate hunks/docs/example are ready. It supplies candidate AC1–AC10 evidence for the single implementation row. Closure additionally requires the independently labeled pre-production B1–B4 incumbent-control evidence and its fixture/source-diff review; this runner does not execute or verify that earlier phase. If independent review changes hunks afterward, run it again on the changed candidate; an earlier run cannot close new bytes. Independent source/QA/spec review remains separate from this measured execution record.

| runner phase | AC executed | independently asserted output |
|---|---|---|
| compiled Client enumeration | AC1 | exact set of 6 Client physical names, each listed once |
| default-prime Client bodies | AC2 | exact same 6 PASS records, no failed/skipped selected record |
| both-wire compatibility bodies | AC3 | exact same 6 PASS records under compatibility features |
| compiled primitive enumeration | AC4 | exact 1 primitive physical name |
| primitive compatibility body | AC5 | that 1 PASS record |
| existing named regressions | AC6 | exact 4 named PASS records |
| normal no_std LIB graph and std control | AC7 | 1 accepted library artifact and 1 rejected std-enabled control using same checker |
| public observed-builder doc | AC8 | 1 runnable doctest passed, 0 ignored, method name present |
| public typed observed-call doc | AC9 | 1 runnable doctest passed, 0 ignored, method name present |
| existing finite signal example | AC10 | exact receipt marker count 1 after example assertions |

All ten records must be present and associated with their own raw command output. Enumeration and test execution remain separate records. Table subcases and control fixtures remain asserted inside the physical bodies and do not inflate these counts. The runner does not infer behavior from a test name; independent review must inspect the authored assertions and full event traces specified above.

AC7 invokes cargo check --lib, not --lib tests, and inspects compiler-artifact features for the actual selected proxima_primitives target. Its std-positive control must be rejected. A unit test compiled with dev dependency feature unification cannot replace it. No production primitives context field is added in v2; this remains an actual normal-graph compatibility gate.

### six plus one physical bodies

| function / existing file | mandatory subcases |
|---|---|
| client_cleanup_observed_calls_and_unpolled_upload / src/client/request.rs | readonly Fired/private-root creation; cancellation ancestor fired while observed call remains Pending; builder context forwarding; unpolled typed upload destructor before receipt; independent observed cloned Requests, ordinary send/SendPipe compatibility; compliant literal and upload-after-ack negative checker |
| client_cleanup_worker_initialization_and_headers / src/client/handle.rs | actual Client builder factory Pending initialization; explicit cancel destroys initializer while its gate stays closed, cancels waiting caller without touching other initializer, next caller can initialize; pre-head connect/send/read pending; spawn rejection and local error; actual drop-before-ack traces |
| client_cleanup_worker_pending_body_and_backpressure / src/client/handle.rs | actual default-prime worker H1 pipeline via existing test factory/StreamConnection probe; actual read Pending and bounded forward Pending; response/full-body drop and explicit cancel; private receipt not canceled by ancestry; old-relay/cancel-as-complete controls |
| client_cleanup_inline_eos_errors_and_conversions / src/client/handle.rs | actual on-worker inline plus explicit tokio under AC3; retain response after borrowing next future drops, cancel parent, receipt stays Pending, then full stream Drop completes; normal EOS/errors/buffered, full literal bytes/status/header preserved; existing accessible trailer slot before conversion, absent after into_body/into_chunk_stream per baseline |
| client_cleanup_owned_permit_retry_and_controls / src/client/handle.rs | existing Runtime::spawn_on_core concrete Send+'static owned task/Arc gate/MockClock; full-body drop then receipt then gate release; internal configured retry first attempt then final pending body retains one logical receipt; external Retry<Client> ordinary call unchanged and no inner receipt claim; same checker rejects stale-attempt and early release |
| client_cleanup_default_prime_pending_tcp_body / src/client/handle.rs | actual Client::from_value default http loader, no wire override/from_pipe; native local TCP head and partial body, withheld tail; cancellation and complete-response Drop, cleanup and peer EOF observed separately while Client alive; complete body/pooled-idle control; actual socket observation never inferred from fake destructor |
| client_cleanup_request_clone_preserves_cancel_without_completion_state / proxima-primitives/src/pipe/request.rs | existing buffered Request::clone and HttpReplay-compatible context remain unchanged; cancellation parent shared as before, labels/payload exact, no completion state/API on context; actual Client receipt freshness is tested in umbrella body, not inferred here |

Async/parameterized tests use proxima::test. Plain sync bodies deliberately drive the Client off-worker from libtest threads. Actual fake I/O is mounted via existing factory + real Client runtime dispatch, never Client::from_pipe; default native TCP proof remains separate. Gates/Signal/MockClock replace sleeps; record actual resource drop, pending wake and stream-poll phases. The original marker tests alone do not prove behavior.

Public docs add exactly one executable Rust block to each new observed method. send_with_cleanup's block uses existing synth Client to assert exact buffered `cleanup` bytes, drops/drives full body and awaits receipt; call_with_cleanup's block constructs a typed upload/request future, drops unpolled and observes receipt. No no_run/ignore fences. Docs explicitly teach operation vs TCP closure, inline drive-or-drop, private root and logical Client retry scope. Extend EXISTING finite examples/signal/main.rs with a new concluding observed Client synth call/body/receipt assertion and the exact AC10 marker; preserve its existing sticky Signal/late-observer demonstration. No new example target or network/provider call. Its source name is bound by current Cargo.toml:1824–1827. These are prospective additions, not claims that the methods/docs exist now.

## isolated baseline and integration

Use the same isolated `/Users/brianbruggeman/repos/slot-0/proxima-client-cleanup-wt` plan: detached exact HEAD 9bb9ce6984b9db616b5e79eb79dc26f49233e89e plus reconstructed full relevant dirty overlay, including both owner-added ai_docs question records/routes; verify v4 source-ledger hashes. Do not edit the shared checkout or reconstruct from clean HEAD alone. No worktree is created in this author task.

Source scope: existing src/client/{request,handle}.rs plus necessary method docs in those files, existing examples/signal/main.rs, and one compatibility test in proxima-primitives/src/pipe/request.rs. Existing response conversions and production RequestContext/Signal/Retry/Runtime/HTTP/TCP representations remain unchanged. Existing response-stream composition is private helper code inside the Client ownership seam; no generic Lease, new primitive, service or transport.

After separate source card and fresh critic/QA/spec admission, author full bodies and list them, then run AC2–AC10. Preserve trace artifacts and actual feature graph. No performance claim; new root/child scope setup allocation is unmeasured. Appropriate existing four functional regressions are included; inbound serve benches cannot settle request cleanup. Independently review actual hunks and guards before root applies any scoped patch back to shared dirty Proxima. Verify original target hashes, use git apply --check, re-review drift; do not overwrite whole files, edit dependency Cargo.toml by hand, commit, publish or switch wire. Ragd patch still points to shared /slot-0/proxima; candidate-only result is not Ragd integration evidence. B's separately admitted integration tests follow guarded shared-source integration, not this proposal.

## out of scope

- source implementation, compilation/tests/provider/model calls in this revision pass
- caller-supplied completion Signal/context field, generic outer Retry cleanup API or arbitrary detached I/O guarantees
- background inline driver, new runtime/service/Lease/semaphore, pool/HTTP/TCP/reactor redesign or transport switch
- stronger trailer conversion semantics; baseline discarding is explicitly documented while full body remains exact
- physical socket/global policy verdict, performance/zero-allocation claim, Ragd A/B production changes, commits/publish

## risks

| risk | source mechanism | counter |
|---|---|---|
| public observer could inherit cancellation | Signal children observe ancestors | private Signal::new root, caller receives only Fired |
| retained inline response cannot self-clean | lazy stream no longer polled | explicit drive-or-drop contract and retained-unpolled negative state |
| initialization hangs before worker race | handle OnceCell/factory Pending | race entire local handle future and destroy initializer/wait before receipt |
| unpolled upload drops after completion | separate future captures have unspecified intended order | one intact Request-before-guard tuple and destructor trace |
| retry clones stale receipt | HttpReplay clones Context | no receipt in Context; one private root per observed Client dispatch |
| no_std test graph enables std | facade dev dependency | normal --lib check artifact-features gate + std control |
| method docs/examples drift | nextest excludes doctests/example execution | separate count-bound AC8–AC10 |

## context

Opened source authority is pinned in source-ledger.json, including critic source readback comparisons; no runtime result is asserted. Context references resolve to these actual files:

- `/Users/brianbruggeman/repos/slot-0/proxima/.claude/agents/proxima-architect.md:1`
- `/Users/brianbruggeman/repos/slot-0/proxima/AGENTS.md:163`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/request.rs:80`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/handle.rs:223`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/handle.rs:258`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/handle.rs:378`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/handle.rs:428`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/client/response.rs:39`
- `/Users/brianbruggeman/repos/slot-0/proxima/src/load.rs:329`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-core/src/signal.rs:130`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-core/src/signal.rs:193`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-primitives/src/pipe/request.rs:284`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-primitives/src/pipe/retry.rs:262`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-primitives/src/pipe/retry.rs:430`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-primitives/src/pipe/body.rs:130`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-primitives/Cargo.toml:205`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-runtime/src/lib.rs:293`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-runtime/src/ext.rs:178`
- `/Users/brianbruggeman/repos/slot-0/proxima/examples/signal/main.rs:1`
- `/Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/critique/CRITIQUE.md`
- `/Users/brianbruggeman/.local/share/ragd/pm-exports/chat-capabilities/client-lifecycle-discovery/proposal-v4/source-ledger.json`

Guiding principles changed the proposal: existing Fired replaced writable injected completion; explicit ownership replaced a cloned-context flag; lazy inline scope prevented an unearned autonomous-cleanup promise; input tuple order supplied the unpolled-drop invariant; actual normal feature graph replaced a dev-test proxy; existing docs/example targets supply executable public teaching. The request/response operation remains the existing SendPipe. Signals and read-only observer are existing completion values, not another pipe/type or public abstraction. No author ADMIT or behavior verdict is attached.

## v3 review provenance

The owning critic stable v2 review is `../critique/v2-independent-critic/REVIEW.md` (SHA-256 `740b4dc51b1027e6900e62f5d618d8380a2ef839feb8e55b2a3c62330a168f1b`), which accounts for all prior findings and identifies no additional grounded counterexample in its bounded pass; it grants no admission or runtime result. Independent QA `../proposal-v2/qa-lifecycle-planning-panel.json` supplies QA-LIFECYCLE-1: previous TASKS rows claimed ACs not executed by their row commands. v3 preserves all ACs and the selected design, and changes the task partition plus actual counted whole-contract command. Fresh independent review/spec admission and source-card registration remain required.

## v4 narrow review merge

QA-V3-1 and the owning critic v3 sole finding both concern reused in-tree evidence. QA-V3-2 concerns default ignored-test filtering. V4 changes only runner evidence ownership, machine-readable ignored-inclusive exact enumeration, associated command flags and synthetic controls. Selected v2 lifecycle API, physical body/feature scope, AC1–AC10 and physical-socket residual are preserved. No production/source edit or Cargo build/test is authorized here.

## v5 narrow formal-audit revision

The v1 formal audit passed G1/G2/G3/G5/G6/G7 and refused G4 for the missing explicit incumbent-PASS mapping in AC2/3/5/6/8/9/10. V5 adds the named inherited predicates, commanded pre-production B1–B4 controls, per-AC mapping, and exact novel-receipt rule-domain residual. Independent bounded interpretation is `../critique/g4-barrier-interpretation/INTERPRETATION.md` (SHA-256 `eb13a13f49690df3f15e7798e0fd8ecdf3e1a6f81714280b0b3f321e38c1a429`). No source, Cargo gate or provider/model call ran. The v4 runner bytes, selected lifecycle API, all eight requirements, ten candidate ACs and seven new physical bodies remain unchanged. Registered PROXIMA-18 receives updated bound artifacts through normal intake; no new card is proposed.
