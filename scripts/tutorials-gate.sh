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
# always welcome and never blocked by this check.
FLOOR=67
if [ "$tutorial_ok" -lt "$FLOOR" ]; then
  echo "tutorials-gate: FAIL — $tutorial_ok compiling blocks, below the floor of $FLOOR"
  exit 1
fi
if [ "$code" -ne 0 ] && [ "$tutorial_ok" -ge "$FLOOR" ]; then
  echo "tutorials-gate: $tutorial_failed block(s) below the floor still fail to compile (see full log above) — not a regression, tracked as residual coverage"
fi
echo "tutorials-gate: clean ($tutorial_ok >= floor $FLOOR)"
