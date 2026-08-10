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
# The bar is every rust block, with no tunable: compiled == counted. There is
# deliberately no threshold here. A threshold is blind to a swap — one block
# rots while another is fixed, the total holds, and the gate stays green over
# a broken page — and it has to be hand-bumped on every edit, which made it
# a standing merge conflict.
#
# `,ignore` is not an outcome either; it is the same blindness hidden per
# block. If a block cannot stand alone, either make it self-contained (repeat
# the real definitions and imports it elides, copied verbatim from source) or
# admit it is a quotation, not a program, and fence it as `text` — the docs
# already do this for terminal transcripts. The one exception is
# `,compile_fail`, which is a real assertion that the type system rejects
# something; rustdoc still compiles it and still enforces the failure.
#
# Usage:  bash scripts/tutorials-gate.sh
# Exits 0 only when every rust block compiled. Non-zero if any block failed,
# any block was ignored, the compiled count and the block count disagree, or
# zero tutorial doctests ran at all (the empty-match false-green AGENTS.md
# warns about).

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

# The residual this gate used to tolerate is gone: every rust block under
# docs/tutorials/ now compiles. The technique that closed the last 45 was
# always the same one — repeat, verbatim from source, the definitions and
# imports the excerpt elided, and where the quoted thing is not a program
# at all (a proc-macro crate's private internals), fence it as `text` and
# cite it instead of pretending it is compilable.
if [ "$tutorial_failed" -ne 0 ]; then
  echo "tutorials-gate: FAIL — $tutorial_failed block(s) fail to compile"
  exit 1
fi
if [ "$tutorial_ignored" -ne 0 ]; then
  echo "tutorials-gate: FAIL — $tutorial_ignored block(s) ignored; a block that cannot compile is not rust, fence it as \`text\`"
  exit 1
fi
if [ "$tutorial_ok" -ne "$block_count" ]; then
  echo "tutorials-gate: FAIL — $tutorial_ok of $block_count rust blocks compiled; the two must match exactly"
  exit 1
fi
echo "tutorials-gate: clean — all $block_count rust blocks compile"
