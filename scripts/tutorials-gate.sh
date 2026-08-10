#!/usr/bin/env bash
# tutorials-gate.sh — compile-check every ```rust fenced block under
# docs/tutorials/*.md against today's API.
#
# This gate exists because docs/tutorials/*.md carried Rust snippets that
# nothing compiled: `cargo test --doc` never saw them (they are prose
# markdown, not a doc comment on any item), so a snippet could cite a type
# or a method that was renamed or deleted and the docs would keep teaching a
# system that no longer exists. `00-foundations.md`'s own opening claims
# "every code block below is copied verbatim ... either a doctest that
# `cargo test` compiles, a unit test, or a runnable `examples/*/main.rs`" —
# nothing enforced that claim before this script existed.
#
# Mechanism: extend `cargo test --doc`, the pattern this repo already
# mandates (see AGENTS.md's Testing section) instead of inventing a second
# harness. `src/tutorial_doctests.rs` carries one `#[cfg(doctest)]
# #[doc = include_str!(...)] mod ... {}` per tutorial file, each pointing at
# a copy this script writes into `.tutorial-gate-generated/` (gitignored,
# regenerated every run — never a second, driftable copy of the docs). The
# per-line transform (fence retagging, hidden prelude, `ignore`/`no_run`
# classification) lives in tutorials-gate-transform.awk — see that file's
# header for the full mechanism and why it exists.
#
# TWO PASSES, because same-file sibling context (`struct Increment` in one
# block, `Increment.and_then(Halve)` three paragraphs later) is common, but
# feeding every earlier block forward UNCONDITIONALLY was measured to make
# things WORSE, not better: one broken early block (a stale API citation, a
# bare associated-fn excerpt) cascades into every later block in the same
# file that inherits it, and the pass count went DOWN when this was tried.
# Pass 1 compiles every block in isolation and records which ones stand on
# their own; pass 2 regenerates, this time feeding each block only the
# EARLIER blocks pass 1 proved compile standalone. A later block that
# redefines an earlier one's name (the tutorials' own "Before (...) / Today
# (...)" comparisons) retires the earlier definition from later context
# instead of leaving both in scope.
#
# The original docs/tutorials/*.md files are never modified by this script.
#
# This does NOT require every block to compile — see the FLOOR comment
# near the bottom of this file for why a floor, not zero failures, is the
# honest bar today, and what growing it actually takes.
#
# Usage:  bash scripts/tutorials-gate.sh
# Exits 0 if at least FLOOR blocks compile (regression-detecting); non-zero
# if the compiling count drops below FLOOR, or if zero tutorial doctests
# ran at all (the empty-match false-green AGENTS.md warns about).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TUTORIALS_DIR=docs/tutorials
GEN_DIR=.tutorial-gate-generated
GLUE_FILE=src/tutorial_doctests.rs
AWK_SCRIPT="$(dirname "$0")/tutorials-gate-transform.awk"

# same feature union examples-gate.sh uses to build every example in one
# pass — the tutorials cite the same examples, so the same union covers
# every symbol they reference.
ROOT_UNION="http1,http1-native,http2,http3,http-prime-deps,tracing-init,tokio,runtime-tokio,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,macros,instrument-metrics,histogram,otlp-http,serve-prime,pgwire,redis-listener,memcached-listener,memcached-client,dns-listener,dns-client,kafka-listener,kafka-client,mqtt-listener,mqtt-client,amqp-listener,amqp-client,tls,h3-native-upstream"

file_count=0
block_count=0
for f in "$TUTORIALS_DIR"/*.md; do
  base=$(basename "$f" .md)
  [ "$base" = "README" ] && continue
  file_count=$((file_count + 1))
  n=$(grep -c '^```rust' "$f")
  block_count=$((block_count + n))
done

echo "tutorials-gate: $file_count tutorial files, $block_count rust fenced blocks"
if [ "$block_count" -eq 0 ]; then
  echo "tutorials-gate error: zero rust blocks found under $TUTORIALS_DIR — extraction regressed silently"
  exit 1
fi

glue_count=$(grep -c '#\[doc = include_str!("\.\./\.tutorial-gate-generated/' "$GLUE_FILE")
if [ "$glue_count" -ne "$file_count" ]; then
  echo "tutorials-gate error: $GLUE_FILE wires $glue_count tutorial file(s) but $file_count exist under $TUTORIALS_DIR — a tutorial file landed ungated. Add a matching #[cfg(doctest)] mod to $GLUE_FILE."
  exit 1
fi

run_doctests() {
  LOG="$1"
  RUSTDOCFLAGS="${RUSTDOCFLAGS:-} --cfg tutorial_gate" \
  cargo test -p proxima --doc --features "$ROOT_UNION" > "$LOG" 2>&1
  return $?
}

# a test's own reported line number is the generated file's OPEN FENCE
# line — verified empirically against a throwaway crate before relying on
# it (`test FILE - NAME (line N) ... ok`, N == the ```rust line). rustdoc
# appends "- compile" to a test's name for a block with no executable
# statement (only item declarations, e.g. the algebra_claims fn defs) —
# still compiled and still reported ok/FAILED, just an optional suffix
# before " ... ".
result_key() {
  # "<file_base>\t<generated_line>\t<status>" per test result line in $1
  grep -oE '^test src/\.\./\.tutorial-gate-generated/[^ ]+\.md - [A-Za-z0-9_:]+ \(line [0-9]+\)( - [a-z]+)? \.\.\. (ok|FAILED|ignored)$' "$1" \
    | sed -E 's#^test src/\.\./\.tutorial-gate-generated/([^ ]+)\.md - [A-Za-z0-9_:]+ \(line ([0-9]+)\)( - [a-z]+)? \.\.\. (ok|FAILED|ignored)$#\1\t\2\t\4#'
}

rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"
MANIFEST="$(mktemp)"
: > "$MANIFEST"
for f in "$TUTORIALS_DIR"/*.md; do
  base=$(basename "$f" .md)
  [ "$base" = "README" ] && continue
  awk -v FILEBASE="$base" -v MANIFEST="$MANIFEST" -f "$AWK_SCRIPT" "$f" > "$GEN_DIR/$base.md"
done

LOG1="$(mktemp)"
run_doctests "$LOG1"
pass1_code=$?

GOODFILE="$(mktemp)"
: > "$GOODFILE"
result_key "$LOG1" | while IFS=$'\t' read -r rbase rline rstatus; do
  [ "$rstatus" = "ok" ] || continue
  awk -F'\t' -v B="$rbase" -v L="$rline" '$1==B && $3==L {print $1"\t"$2}' "$MANIFEST"
done > "$GOODFILE"

rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"
for f in "$TUTORIALS_DIR"/*.md; do
  base=$(basename "$f" .md)
  [ "$base" = "README" ] && continue
  awk -v FILEBASE="$base" -v GOODFILE="$GOODFILE" -f "$AWK_SCRIPT" "$f" > "$GEN_DIR/$base.md"
done

LOG2="$(mktemp)"
run_doctests "$LOG2"
code=$?
cat "$LOG2"

# `cargo test --doc` exits 0 on an empty match (AGENTS.md's own warning) —
# count PER-TEST result lines naming a generated tutorial fixture, not just
# grep hits, since a FAILED test's diagnostic block repeats the path and
# would double-count. `ignore` (the bare-associated-fn-signature class) is
# counted separately from a pass — it is a documented, mechanical exclusion,
# not silently dropped from the totals.
tutorial_ok=$(grep -cE '^test .*\.tutorial-gate-generated/.* \.\.\. ok$' "$LOG2")
tutorial_failed=$(grep -cE '^test .*\.tutorial-gate-generated/.* \.\.\. FAILED$' "$LOG2")
tutorial_ignored=$(grep -cE '^test .*\.tutorial-gate-generated/.* \.\.\. ignored$' "$LOG2")
tutorial_ran=$((tutorial_ok + tutorial_failed + tutorial_ignored))
rm -f "$LOG1" "$LOG2" "$MANIFEST" "$GOODFILE"

echo
echo "tutorials-gate: pass 1 (isolated) exit=$pass1_code; pass 2 (accumulated-known-good) is authoritative"
echo "tutorials-gate: $tutorial_ran tutorial doctests found ($tutorial_ok ok, $tutorial_failed failed, $tutorial_ignored ignored) of $block_count blocks in docs/tutorials/*.md"

if [ "$tutorial_ran" -eq 0 ]; then
  echo "tutorials-gate error: cargo test --doc ran zero tutorial doctests — an empty match is a false green, not a pass"
  exit 1
fi
if [ "$tutorial_ok" -eq 0 ]; then
  echo "tutorials-gate error: zero blocks actually compiled and ran — every doctest was ignored, no_run, or failed"
  exit 1
fi

# FLOOR, not zero-failures. Two-pass accumulation (prelude + same-file
# known-good context) still leaves a large residual: most of the remaining
# failures cite a name that exists ONLY in that tutorial's own PROSE, never
# in any code block this generator can see — a local variable introduced
# mid-paragraph ("bind", "store", "app"), or a narrative type invented for
# one worked example (`Order`, `Overflow`, `Reply`) whose defining block
# itself needs context this gate has no way to reconstruct without either a
# real Rust name resolver or rewriting the snippets to be self-contained —
# out of scope here per this task's own framing ("before any tutorial is
# REWRITTEN, build the gate"). Requiring zero failures today would mean
# either a permanently-red required check or silently dropping 129 blocks
# from the count, and both are worse than the honest floor below: it is the
# reproducible ok-count measured across repeated runs of this exact
# mechanism (42, unchanged across three consecutive runs while iterating on
# the transform; raised to 44 on 2026-08-09 after 00-foundations.md's
# citations were resynced to current file:line and a stale `HalveError`
# doctest snippet was given back the `#[derive(Debug, ...)]` its `Pipe::Err`
# bound requires; raised to 47 later the same day after 01-ergonomics.md's
# own citation resync, which also fixed two real bugs in this transform's
# `is_bare_self_fn` detector — the container regex missed a generic `impl<T>
# ..` header (no space before `<`) and a generic `fn foo<A, B>(&self, ..)`
# signature (generics between the name and the paren) — that were silently
# mis-marking 01-ergonomics.md's own `App::mount` and `Timer<ClockImpl>`
# excerpts, plus added the `,compile_fail`/explicit-`,ignore` fence-attribute
# passthrough a tutorial author can now reach for instead of an unexplained
# FAILED block — 47 unchanged across two consecutive runs; raised to 49 later
# the same day after 04-listener-hello.md's own rewrite turned its two
# excerpted `main`-body fragments (needing a `?` inside a function that
# actually returns `Result`, and a `server` to call `.run_until_signal()` on)
# into small, honestly-labeled, self-contained wrappers around the same real
# lines instead of leaving them silently FAILED); raised to 52 later the same
# day after 05-listener-universal.md's own rewrite turned all 5 of its blocks
# (all 5 previously FAILED — an undefined-`bind`/`my_handler` fragment, two
# further copies of the same undefined-variable shape, and a bare
# `.accept("h2")` one-liner that isn't a legal standalone expression) into 3
# real, self-contained, compiling excerpts of `examples/any_listener.rs`
# (the redundant/broken fragments were cut, not replaced 1:1) — the rewrite
# also caught a real bug the old prose taught: `Listener::builder().handle()`
# takes an `impl Into<PipeHandle>` (a `Handler`-shaped VALUE), not a bare
# `async fn` the way `App::mount` does (`App::mount`'s bare-fn adapter,
# `FnHandler`, is private to `src/app.rs`, never reachable through
# `Listener::builder()`) — confirmed by compiling `into_handle(bare_fn)`
# directly and reading the resulting E0277. Raised to 61 later the same day
# after 06-listener-production.md's own rewrite turned all 10 of its blocks
# (9 previously FAILED — undefined-variable fragments excerpted straight from
# a runnable example with no enclosing fn, plus one library-internals `match`
# over names nothing in the block ever defined) into 10 real, self-contained,
# compiling `fn`/`struct` excerpts; the rewrite also caught the tutorial
# teaching an actual defect as still-open — h1 silently not enforcing
# `max_in_flight_requests` and a body-carrying h2 shed request getting
# `RST_STREAM` instead of the documented 503 — that commit `8a12a93b5`
# ("fix: enforce request admission on h1 and drain h2 shed bodies") had
# already fixed the same day the tutorial was first written, proven by two
# real, currently-green tests (`proxima-http/src/http1/serve.rs`'s
# `h1_in_flight_shed_renders_503_then_recovers_after_release` and
# `tests/e2e/listener_h2.rs`'s
# `native_h2_listener_body_carrying_shed_request_receives_in_band_503_not_reset`)
# rather than left asserted from stale prose. Raised to 67 later the same
# day after 07-sugar-composition.md's own rewrite turned all 6 of its
# blocks (5 previously FAILED — `main`-body fragments referencing
# `bind_1`/`bind_2`/`bind_3`/`bind`/`bind_bad`, `FixedOk`, and
# `stub_handle()`, none defined within the excerpt, plus a bare `json!`
# macro call with no `use serde_json::json` in scope) into 7 real,
# self-contained, compiling `fn`/`struct` excerpts (`FixedOk` defined once
# and reused via same-file accumulation everywhere after; one new block
# added for the `.grpc().quic()` rejection, which the prose already taught
# but no code block ever proved) — the rewrite also caught two stale
# `file:line` citations (the `prelude` module and `TlsConfig::self_signed`
# had both drifted since the tutorial was written) and a dangling internal
# forward-reference to a client-side `.tls()` section that was never
# written, fixed by adding the real `ClientSecurityExt::tls()` composition
# `examples/sugar_composition.rs` itself was missing, not just the prose
# describing it. A regression — any currently-compiling block citing an
# API that gets renamed or removed — drops the count below the floor and
# fails the build; growing the floor by fixing a currently-failing block
# (prelude gap, tutorial content, or the awk transform's own reach) is
# always welcome and never blocked by this check. Raised to 73 later the
# same day after 08-protocol-fleet.md's own rewrite turned all 10 of its
# blocks (all 10 previously FAILED — a bare `let server = ...await?;`
# fragment with no enclosing fn using an undefined `bind`, a DNS section
# whose "Listener:" prose described `.dns(handler)` but showed no
# `Listener::builder()` code block at all, and three more handler-only
# impl blocks with no round trip) into 6 real, self-contained, compiling
# `async fn`/`struct` excerpts mirroring `examples/protocol_fleet.rs`'s
# own per-protocol functions — one shared `NullHttp` fallback-handler
# block plus one full listener+client round-trip block per protocol
# (memcached/DNS/Kafka/MQTT/AMQP); the rewrite also fixed two stale
# `file:line`/section citations (`.dns(dsn)` dialing UDP was cited to a
# nonexistent DNS module doc, corrected to `src/upstreams/dns.rs`'s own
# matching doc comment; the DNS `.quic()` config-error citation pointed at
# 07-sugar-composition.md's own `.dns(handler)` section instead of its
# failure-mode section that actually demonstrates a rejected composition).
# Raised to 77 after 10-conflaguration.md's own rewrite turned all 6 of its
# blocks (all 6 previously FAILED — every one referenced a name from
# nowhere in its own file: `toml_path`/`bind`/`built`/`handler` never
# defined, `default_host`/`default_max_message` helper fns never excerpted
# alongside the `#[serde(default = ..)]` that cites them, and a
# `Validate::validate` stub returning bare `()` against a `conflaguration::
# Result<()>` signature) into 4 real, self-contained, compiling excerpts:
# the full `ServerConfig` struct+`Default`+`Validate` from
# `examples/config/main.rs`, `ListenTuningConfig`'s and `BlacklistConfig`'s
# own crate-doctest builder-vs-TOML parity assertions verbatim from
# `proxima-listen/src/config.rs`/`src/admission/blacklist.rs`, and — the
# rewrite's real find — `examples/protocol_fleet.rs`'s own `KafkaServerConfig`
# section, whose doc comment already CLAIMED "wired into a real listener"
# while its body only ever compared two structs for equality and never
# touched `Listener::builder()` at all; extended that function itself (not
# just the tutorial prose) into an actual `.protocol(KafkaAnyProtocol::new(..)
# .with_config(config))` listener serving one real PRODUCE round trip, which
# is what closed the tutorial's own "illustrative, not run" hedge for good.
# Two structural bugs surfaced only by making every block compile in the SAME
# file: a locally re-declared `struct KafkaServerConfig` (mirroring the real
# type's shape, teaching-style) collided (E0255 + orphan-rule E0116/E0117)
# with a LATER block's `use proxima_kafka::{.., KafkaServerConfig, ..}` of
# the real type, since same-file context accumulation carries a struct
# definition forward but has no way to know a later block's IMPORT of an
# identically-named real type should retire it — fixed by dropping the
# redundant local mirror (the struct's anatomy was already fully taught by
# `ServerConfig` in §1) rather than teaching the same shape twice; and two
# independent blocks each spelling `use std::io::Write;` (both borrowed
# verbatim from real crate doctests that don't know about each other)
# collided too (E0252) once accumulated into one scope — fixed with
# `use std::io::Write as _;` in the second, which still satisfies `write!`
# without binding a second `Write` name. Raised to 82 after
# 09-extend-your-own-protocol.md's own rewrite: all 4 of its blocks were
# previously FAILED — the `AnyProtocol` trait excerpt and the
# `PingPongProtocol` impl excerpt both cited `StreamConnection`/`PeerInfo`
# (and the trait excerpt also `AnyHandler`) bare, none of which
# `tutorial_gate_prelude` re-exported (fixed by adding all three re-exports
# to the prelude itself, `src/tutorial_doctests.rs` — a real gap, not a
# tutorial-content bug, since `examples/extend_protocol.rs` imports them
# from `proxima::{PeerInfo, StreamConnection}` /
# `proxima::listen::any::AnyHandler` same as the excerpt), and the
# `.protocol(candidate)`/`.ping_pong(candidate)` registration blocks were
# bare `let server = ...await?;` fragments referencing an undefined `bind`
# and `LegitOk` — fixed by turning both into real, self-contained
# `async fn start(bind: SocketAddr) -> Result<(), ProximaError>` excerpts
# with `LegitOk` defined inline, mirroring `examples/extend_protocol.rs`'s
# own `main` minus the `free_loopback_addr()?` plumbing (same pattern
# 05-listener-universal.md's own rewrite used for the identical shape).
# The rewrite also fixed two real drift bugs the prose asserted as fact:
# "that's the whole trait" showed 5 methods against a real trait that had
# grown a 6th (`wants_datagram`, added by `a39c1c9e8` without updating this
# page's own trait excerpt) — corrected to name and defer that method to
# part 8 rather than silently omit it; and "the three parameters this
# example doesn't use" undercounted by one — `peer: Option<PeerInfo>` was
# unused by `PingPongProtocol::drive` but never explained. Both `file:line`
# citations for the trait (`proxima-listen/src/any/probe.rs:134-195`, stale
# by the same doc-comment growth) and `ListenerBuilder::protocol`
# (`src/listener/handle.rs:277-289`, off by the same margin
# 02-listener-builder.md's own already-correct `278-292` citation for the
# identical function proved) were resynced. The shared prelude fix also
# incidentally raised 11-any-transport-agnostic.md's own passing count by
# one (3 failing blocks down to 2) since it cites the same two types.
# Raised to 84 after 11-any-transport-agnostic.md's own remaining two
# blocks were fixed with the identical `09-extend-your-own-protocol.md`
# pattern above: both `.protocol(candidate)` registration blocks (§4's
# worked example, §5's priority/ambiguity fleet) were bare
# `let server = ...await?;` fragments referencing an undefined `bind` and
# `LegitOk` — turned into real, self-contained
# `async fn ..(bind: SocketAddr) -> Result<(), ProximaError>` excerpts with
# `LegitOk` defined inline (repeated verbatim in both blocks, since §4's
# block itself needs §3's `LiteralUdpProtocol` as cross-block context and so
# never compiles standalone in pass 1, meaning it can never be fed forward
# as pass 2 context to §5's block).
# Raised to 87 after build-a-bare-metal-pipe.md's own rewrite: 2 of its 3
# blocks were previously FAILED — the `FrameStore` excerpt cited `StoreError`,
# `RingSink`, `RING_SLOTS`, and `RING_SLOT_BYTES` with none defined in the
# block, and the build-time `mod config { include!(...) }` excerpt can never
# compile inside this harness (its `include!` resolves `OUT_DIR` relative to
# `proxima-example-no-std`'s own build, which does not run here) — fixed by
# making the pipe excerpt self-contained (its real imports, `StoreError`
# inline, the two constants shown as the literal values `no-std.toml` bakes)
# and retagging the config excerpt `text` instead of leaving it silently
# FAILED. The rewrite also added 2 new passing blocks teaching
# `#[proxima_macros::piped]`'s auto-`Clone` holding at the same no-alloc
# floor (`ring_capacity`, `no-std/src/lib.rs:89-92`, previously untaught
# despite being tested in the answer key itself), and fixed 3 stale
# `file:line` citations drifted by the crate doc-comment and README both
# growing since this page was first written (`no-std/src/lib.rs:15`→`19`;
# `proxima-primitives/src/pipe/mod.rs:184`→`200`; `no-std/README.md:37-45`
# and `:47-52`, which had drifted onto an unrelated paragraph, →`:32-36`
# and `:64-66`).
# Raised to 90 after build-a-caching-reverse-proxy.md's own rewrite: all 4
# of its original blocks were previously FAILED — the tutorial never showed
# `CachedOriginDispatch`'s own struct definition at all (only its field-less
# construction and its `impl SendPipe`), so no block could ever resolve the
# type; `KvCache`/`KvCaps`/`KvUpstream`/`UpstreamRef`/`Fallthrough`/
# `Selection`/`KvHandle`/`WriteBack`/`SynthUpstream` were also missing from
# `tutorial_gate_prelude` entirely (a real gap, not a tutorial-content bug —
# `examples/cache/main.rs` imports them from `proxima::{..}` /
# `proxima::upstreams::KvUpstream` / `proxima::selection::Selection` the same
# way); and the origin-upstream line was a bare comment
# (`into_handle(/* ForwardPipe { client } | SynthUpstream::new(...) */)`),
# an `into_handle` call with zero real arguments. Fixed by adding the 9
# missing re-exports to `src/tutorial_doctests.rs`, replacing the comment
# with a real `SynthUpstream::new(..)` call (the real example wraps it one
# layer deeper in a call-counting `CountingOrigin` purely so its own
# assertions can prove the origin was hit exactly once —
# `cache/main.rs:190-209` — elided here as instrumentation, not concept),
# and restructuring into 3 self-contained blocks: two upstreams; the
# `CachedOriginDispatch` struct AND its `impl SendPipe` merged into ONE
# block (splitting them across two blocks compiled each individually but
# broke the chain — the impl block, needing the struct from a separate
# chunk, could never pass pass 1 standalone itself, so it could never donate
# both the struct AND the impl forward to the final wire-up block, which
# needs both to resolve `into_handle(dispatch): PipeHandle` — measured
# directly, E0277 "the trait bound `CachedOriginDispatch: SendPipe` is not
# satisfied" even though the impl block itself showed `... ok` two blocks
# earlier); and the `Fallthrough` construction + `WriteBack::single` wire-up.
# Raised to 94 after build-a-crud-origin-service.md's own rewrite: all 4 of
# its blocks were previously FAILED — the `Store` struct cited `BTreeMap`/
# `Mutex`/`MutexGuard`/`PoisonError`/`AtomicU64` with none imported and none
# in `tutorial_gate_prelude` (fixed with real `use` lines in the block
# itself, matching the tutorials' own established convention of showing a
# plain `std` import inline rather than adding a std collection type to the
# prelude); the routing block's `mount_with_methods` calls referenced
# `ReadItem`/`UpdateItem`/`DeleteItem`, none of which any earlier block ever
# defined (`ReadItem`'s prose described its 404 behavior but showed no code,
# `UpdateItem`/`DeleteItem` were never shown at all); and the serve block
# was a bare fragment (`app.build_listener(...)`, `run_crud_flow(bind)`,
# `listener.shutdown()`) with no enclosing fn and an undefined `app`, plus
# `ShutdownBarrier` — which the real file imports from `proxima::shutdown`
# — missing from `tutorial_gate_prelude` entirely (a real gap, fixed by
# adding it). Fixed by merging `CreateItem` and `ReadItem` (plus the
# `item_id` path-param helper and a duplicated `Store`) into ONE
# self-contained block — the same "impl needs the struct from a separate
# chunk" chaining trap `build-a-caching-reverse-proxy.md` hit, except here
# TWO downstream blocks (routing, serve) each needed BOTH handlers forwarded,
# so splitting `CreateItem`/`ReadItem` into two independently-standalone
# blocks would have made each redefine `Store` and mutually retire the
# other from later context (the awk transform's redefinition rule disables
# the WHOLE earlier chunk, not just the redefined name) — merging them into
# one chunk that defines `Store` exactly once was the only shape that
# survives forwarding into both downstream blocks; then turning the routing
# fragment into a real, self-contained `mount_routes(app: &App, store:
# Store) -> Result<(), ProximaError>` mounting the two taught handlers (UPDATE
# and DELETE described in prose with a real citation instead of shown code,
# preserving the tutorial's original scope rather than inflating it), and the
# serve fragment into a real, self-contained `async fn start(bind:
# SocketAddr) -> Result<(), ProximaError>` mirroring `crud/main.rs`'s own
# `main` (build, mount inline, `build_listener`, `shutdown`,
# `ShutdownBarrier::broadcast_drop`) minus the `run_crud_flow` HTTP-client
# harness, which is test plumbing, not concept — same elision precedent
# `build-a-caching-reverse-proxy.md` used for its own call-counting wrapper.
# The rewrite also resynced every `crud/main.rs` `file:line` citation, all of
# which had drifted by a small, non-constant margin (e.g. the `Store` struct
# cited `33-40` against a real `35-39`; the CREATE handler cited `67-90`
# against a real `66-89`) since this page was first written.
# Raised to 97 after build-a-kafka-style-partitioner.md's own rewrite: 4 of
# its blocks were FAILED for the "excerpt of private/module-relative
# internals" reason 01-ergonomics.md's own `Tier::plan` precedent already
# covers (`impl Pipe for FanIn`'s elided body returns the private
# `FanInCall`; both `algebra_claims` fn excerpts use `super::`/private
# helpers that only resolve inside that module; `FanIn::new`'s bare
# signature is excerpted from an impl over private fields) — all four given
# the same `,ignore` treatment with a one-line why. 2 more (`PartitionKey`,
# `route`) were FAILED because they cited `Record`/`fnv1a`/`PARTITIONS`/
# `block_on_ready` without defining them in-block — fixed by repeating the
# real definitions inline (each still cited separately) the same way
# `build-a-crud-origin-service.md`'s own rewrite merged handlers with the
# struct they need. The real gaps this surfaced: `DropSafe`, `ControlFlow`,
# `DrainState`, and `Waker` were all missing from `tutorial_gate_prelude`
# despite the tutorial's own real-source excerpts needing them (`impl Pipe
# for FanIn`'s trait bound, `DrainSource::drain_ready`'s signature, and
# `block_on_ready`'s manual poll loop) — added to
# `src/tutorial_doctests.rs`. Also caught a real drift bug: `fanout.rs`
# grew a base-tier `impl Pipe for FanOut` (commit `4e31e74f8`) that the
# tutorial's own compile-checked-proof paragraph never updated to mention,
# still asserting "there is no `impl Pipe for FanOut`" against source that
# had had one for months.
FLOOR=97
if [ "$tutorial_ok" -lt "$FLOOR" ]; then
  echo "tutorials-gate: FAIL — $tutorial_ok compiling blocks, below the floor of $FLOOR"
  exit 1
fi
if [ "$code" -ne 0 ] && [ "$tutorial_ok" -ge "$FLOOR" ]; then
  echo "tutorials-gate: $tutorial_failed block(s) below the floor still fail to compile (see full log above) — not a regression, tracked as residual coverage"
fi
echo "tutorials-gate: clean ($tutorial_ok >= floor $FLOOR)"
