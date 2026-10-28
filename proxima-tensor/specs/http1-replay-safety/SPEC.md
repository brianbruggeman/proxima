# HTTP/1 ambiguous replay safety

status: draft
owner: brian
created: 2026-10-27

## problem

A reused native HTTP/1 connection currently reconnects and resends any failed request, including a fully consumed decision POST whose response was lost; constrain both structured/raw replay to canonical idempotent method semantics, measured by four payload-capturing cases before and after repair.

## refutation condition

If a non-idempotent or unknown request body appears again on a new connection after its original bytes were consumed without a response, or a permitted idempotent recovery exceeds one retry, this contract is refuted. Refusing every request also fails the positive GET and post-error subsequent-call controls.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Structured H1 never automatically replays ambiguous non-idempotent requests; return the original failure and discard unsafe connection state. | yes |
| R2 | Preserve bounded one-reconnect recovery for canonical idempotent methods; no method-specific POST exception or global retry disable. | yes |
| R3 | Raw H1 follows the same method semantics, defaults unknown/malformed methods to no replay, preserves original bytes and cannot leak partially failed state into next call. | yes |
| R4 | Fresh-connection errors and failed automatic retry return without another replay; class inventory accounts all sibling paths and canonical predicate has one owner. | yes |

## architecture

Preparation only. Implementation waits for owner-namespaced PM registration, spec-auditor admission, independent QA ticket review and source/build admission of the isolated /private/tmp/hierarchy-live/proxima-replay checkout. Never modify the broken-admin/shared /Users/brianbruggeman/repos/slot-0/proxima original. Source descriptor proxima-replay-source-admission.json now records the isolated overlay source match; this spec was transplanted only after reading that descriptor and verifying the primary source digest. Nominal Git index differences are overlay history, not product deletion instructions. PM is slice tracker; no repository TASKS file.

Source sites: proxima-http/src/http1/client.rs:1036,1070 structured exchange reconnects after any reused-connection error; :1208,1220 send_raw does the same. write_and_read_head may finish writing a POST before EOF while waiting for the response; no response is not evidence of non-processing. Existing ExchangeSlot Drop (:226) discards unsettled connections. Preserve that ownership, encoded/read/write buffers and original failure; non-replay branch must not attempt reconnection that could replace the actual error. Subsequent independent call reconnects normally with its own bytes, not the abandoned body's tail. No success-path serialization/layout change is needed.

Canonical semantics already exist through pipe::capabilities::Idempotent for Request<Bytes> (proxima-primitives/src/pipe/retry.rs:483) and its private method predicate (:781). Do not introduce a RetryFramework, global disable flag, new wrapper type, bespoke POST policy, second method list or native decision-specific transport. Consolidate idempotence with the existing Method type (pipe/method.rs:12) and have the existing request capability and H1 structured/raw paths consume that one rule. Standard GET/HEAD/OPTIONS/TRACE/PUT/DELETE are idempotent by RFC9110; POST/PATCH/CONNECT/Other default false without an explicit proven unprocessed delivery signal. TRACE is present in Method but absent from the older private retry matcher: account this correction and enumerate every standard variant plus unknown/lowercase. Method names are case sensitive. No new hot-path heap allocation for raw parsing: classify a borrowed first method token exactly, refusing unknown/malformed tokens for replay; reuse canonical method semantics instead of allocating Other just to decide retry. Placement of a borrowed-token helper belongs beside canonical Method only if required, with no gratuitous public API.

Two H1 sites are one coherent bounded card. Native H3 upstream.rs:211–247 contains the same ambiguous reused-connection replay loop and is an explicitly REQUIRED separate owner-namespaced sibling card before class closure, because its QUIC execution fixture/protocol ownership differs. This primary card fixes2of3 source-observed replay sites, not the whole class. Root owns PROXIMAHTTP-1 class parent, PROXIMAHTTP-2 H1/canonical primary and PROXIMAHTTP-3 H3 sibling registration/dependencies before implementation. Both are required before class closure and CORE16 dispatch. H2-native client returns write/read/parse errors; its output-flush/read loop is not a whole-request reconnect. H3 quinn compatibility sends one stream and propagates errors. Hyper legacy wrappers delegate cancellation handling: pinned hyper-util0.1.20 client retries only when try_send_request returns the original unstarted message (legacy/client.rs:250–270,325–340), or a checkout closes before request starts (:383); distinguish that proven-not-started mechanism from absent-response ambiguity. Keep its provenance and do not disable it merely to hide the native H1 defect. Caller-configured Retry middleware is a separate explicit policy (retry.rs:254); it is not the implicit pool replay being repaired.

Protocol knowledge may allow non-idempotent replay when the request was provably not processed, but these native H1 generic error branches expose no such proof. Do not infer proof from idle age, EOF, zero response bytes, a likely stale connection, model name or local URL. This card adds no retry configuration, disclosure grant, routing change or HTTP2/3 protocol rewrite. If another ambiguous replay site is discovered, update the class inventory and allocate its owner before dispatch; do not silently exclude it.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| replay gate | existing canonical method idempotence | discarded indiscriminate reused-connection replay: original operation may already have applied |
| rule owner | Method plus existing Idempotent consumer | discarded new RetryFramework/duplicate POST list: semantics would drift |
| proof | exact transport input/output traces before and after | a counter alone cannot show which body was duplicated |
| scope | H1 structured/raw together; H3 tracked sibling | QUIC fixture is distinct work, not permission to omit its observed defect |

## acceptance criteria

Four non-parameterized named cases, with multiple literal subfixtures inside each case. Use existing StreamUpstream/StreamConnection and in-memory async read/write seams (http1/client.rs:350–415) or existing loopback helpers when actual capture is needed; no sleeps, blocking/new mutexes, spinlocks or boxed futures. Existing async gates/channels and test-owned buffers suffice. Async tests use #[proxima::test]. Capture actual connection ordinal, full HTTP request head/body bytes, peer read completion, EOF/write error phase, and client returned error/response; retain payload records externally. Test fixtures are cold allocation-permitted; production method guard introduces no new heap work.

Before production edits, add the desired no-replay fixture and run it against UNFIXED source, expecting assertion failure on the desired single POST dispatch; retain both original and repeated body records. This is required causal reproduction, not a passing gate. Then change the policy and run the identical fixture; preserve original failure, bytes and connection state observations. A server handler signal after reading full body triggers deterministic close-before-response, not timer/race guesses. Preflight keep-alive response is completely consumed before decision exchange.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety) & test(ambiguous_post_not_replayed)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; preflight keepalive then consumed structured POST and lost head yields exactly1 original POST body,0 replay bodies, original failure; subsequent independent valid request succeeds |
| AC2 | R2 | `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety) & test(idempotent_recovery_bounded)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; same ambiguous fixture GET retries once and succeeds; second failure never retries again; literal canonical classification covers all standard variants plus lowercase/unknown |
| AC3 | R3 | `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety) & test(raw_unknown_not_replayed)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; raw POST/PATCH/unknown/lowercase or malformed method inputs are not resent; raw GET recovery remains bounded; exact original wire bytes retained and next request state is independent |
| AC4 | R4 | `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety) & test(fresh_failure_not_replayed)' --no-tests fail --output plain` | 1passed,0failed,0ignored,0not_run; fresh transport error returns once without reconnect for structured/raw; original error class/context preserved; class records account2H1sites,1requiredH3site and all inspected non-class paths |

Combined: `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety)' --no-tests fail --output plain` expects4passed,0failed,0ignored,0not_run. No default-feature assumption: proxima-http/Cargo.toml default/http1 provides native H1 client under the explicit http1 selection. Nonexistent selection control: `cargo val -p proxima-http --lib --features http1 -E 'test(http_replay_safety_nonexistent)' --no-tests fail --output plain` expects selected0, run0 and a nonzero runner refusal. Existing target/core tier checks appropriate to canonical Method change are delegated to the worker after source admission; no new performance benchmark claimed by this error-path correctness repair.

External proxima-replay-validation-boundary.md binds this one primary slice and the mandatory unfixed payload reproduction. H3 sibling must receive its own protocol-faithful worked fixture and exact gate before implementation; primary H1 acceptance must explicitly retain unresolved H3 item rather than claim entire-class completion.

## out of scope

- Native H3 implementation in this H1 slice (explicitly tracked REQUIRED sibling, not dropped), HTTP2 protocol changes, QUIC rewrite, new retry framework/configuration.
- Native model calls, CORE16 adapter workarounds, privacy/egress grants, routing, paid APIs, commits/pushes.
- Performance/zero-allocation verdicts, greenfield component benches or unrelated shared-tree repairs.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| peer consumed request despite no head | source-visible branch, runtime reproduction pending | repeated non-idempotent operation | require unfixed real wire-body records and identical fixed fixture |
| guard leaves pooled state corrupt | possible | next request leaks/resends old bytes | assert original error and independent next request bytes/state |
| raw parser invents method semantics | possible | same bug via raw API | exact borrowed token classification through canonical predicate; unknown defaults false |
| one path is fixed while sibling remains | source-observed H3 site | class still unsafe | explicit class inventory, separate owned H3 card before implementation |

## context

- proxima-http/src/http1/client.rs:1036,1070,1208,1220 source branches; :226 ExchangeSlot cleanup; :350 StreamUpstream example; :1283 existing test helpers.
- proxima-http/src/http2/client.rs:233 read/flush loop and error exits; proxima-http/src/http3/upstream.rs:205 one stream send/error; proxima-http/src/http3/native/upstream.rs:211 required sibling replay.
- proxima-primitives/src/pipe/method.rs:12 Method; :113 is_read is only GET/HEAD and is not full idempotence. pipe/retry.rs:483 Idempotent Request impl; :781 private matcher.
- proxima-http/src/http1/shared_http.rs:113 hyper builder; pinned Cargo.lock hyper-util0.1.20; registry hyper-util0.1.20 src/client/legacy/client.rs:250,325 unstarted-message recovery.
- AGENTS.md; ai_docs/AGENT.md; ai_docs/index.jsonl/task-routes.jsonl/invariants.jsonl read; no existing specific HTTP replay route was found, new knowledge records are prepared externally for admitted clone.
- RFC9110 sections9.2.1/9.2.2: https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2 (primary source opened during preparation).
