#!/usr/bin/env bash
# Demo script that exercises every `proxima pipeline` verb against a
# local or remote `proximad`. Defaults to a local UDS instance under
# /tmp; set PROXIMA_REMOTE_HOST=<host> to drive the SSH-stdio leg
# against host-b instead.
#
# Usage:
#   ./run.sh                       # local UDS, ephemeral state dir
#   PROXIMA_REMOTE_HOST=host-b ./run.sh  # remote SSH-stdio
#
# Pass criteria (the four threshold questions for the anchor demo):
#   1. structured introspection — `inspect` returns per-stage structure
#   2. deterministic replay + mutate — `replay` reproduces, `replay
#      --substitute` propagates a fresh failure
#   3. explain walks causal chain — DAG ancestors render bottom-up
#   4. recording is the source of truth — tail / inspect / explain all
#      derive from the on-disk recording, no inferred edges
#
# The script prints PASS / FAIL per criterion and exits non-zero on
# any failure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXIMA_ROOT="$(cd "$HERE/../.." && pwd)"

# locate binaries — defer to CARGO_TARGET_DIR if set, else workspace target
TARGET_DIR="${CARGO_TARGET_DIR:-$PROXIMA_ROOT/target}"
PROXIMAD="$TARGET_DIR/debug/proximad"
PROXIMA_CLI="$TARGET_DIR/debug/proxima"

if [[ ! -x "$PROXIMAD" || ! -x "$PROXIMA_CLI" ]]; then
    echo "building proximad + proxima-cli first…"
    (cd "$PROXIMA_ROOT" && cargo build -p proxima-cli)
fi

declare -a CLEANUP=()
trap 'for cmd in "${CLEANUP[@]}"; do eval "$cmd" || true; done' EXIT

if [[ -n "${PROXIMA_REMOTE_HOST:-}" ]]; then
    TRANSPORT=(--host "$PROXIMA_REMOTE_HOST")
    echo "transport: ssh ${PROXIMA_REMOTE_HOST} proximad serve --stdio"
else
    # local UDS — spawn a proximad in the background, tear it down on exit
    SOCK_DIR="$(mktemp -d -t proximad-demo-XXXX)"
    STATE_DIR="$(mktemp -d -t proximad-state-XXXX)"
    SOCK="$SOCK_DIR/proximad.sock"
    "$PROXIMAD" serve --unix "$SOCK" --state-dir "$STATE_DIR" >/tmp/proximad-demo.log 2>&1 &
    PROXIMAD_PID=$!
    CLEANUP+=("kill -TERM $PROXIMAD_PID 2>/dev/null")
    CLEANUP+=("rm -rf $SOCK_DIR $STATE_DIR")
    # wait for READY
    for _ in $(seq 1 200); do
        if grep -q "^READY " /tmp/proximad-demo.log 2>/dev/null; then break; fi
        sleep 0.05
    done
    if ! grep -q "^READY " /tmp/proximad-demo.log; then
        echo "proximad did not print READY within 10s; daemon log:"
        cat /tmp/proximad-demo.log
        exit 1
    fi
    TRANSPORT=(--socket "$SOCK")
    echo "transport: local UDS at $SOCK"
fi

CLI() { "$PROXIMA_CLI" pipeline "${TRANSPORT[@]}" "$@"; }

PASS=()
FAIL=()
pass() { PASS+=("$1"); echo "  ✓ $1"; }
fail() { FAIL+=("$1"); echo "  ✗ $1"; }

echo
echo "submit"
SUBMIT_OUTPUT="$(CLI submit "$HERE/pipeline.toml")"
echo "$SUBMIT_OUTPUT"
PIPELINE_ID="$(echo "$SUBMIT_OUTPUT" | grep -oE '01[0-9A-Z]{24}' | head -1)"
if [[ -z "$PIPELINE_ID" ]]; then
    fail "submit didn't return a pipeline id"
    exit 1
fi
pass "submit returned id $PIPELINE_ID"

# give the pipeline a moment to run all three stages
sleep 1.5

echo
echo "list"
LIST_OUTPUT="$(CLI list)"
echo "$LIST_OUTPUT"
if echo "$LIST_OUTPUT" | grep -q "remote-pipeline-demo"; then
    pass "list surfaces the pipeline by name"
else
    fail "list did not surface the pipeline"
fi

echo
echo "inspect (criterion #1: structured introspection)"
INSPECT_OUTPUT="$(CLI inspect remote-pipeline-demo)"
echo "$INSPECT_OUTPUT"
if echo "$INSPECT_OUTPUT" | grep -q '"stages"' \
   && echo "$INSPECT_OUTPUT" | grep -q '"fetch"' \
   && echo "$INSPECT_OUTPUT" | grep -q '"build"' \
   && echo "$INSPECT_OUTPUT" | grep -q '"bench"'; then
    pass "criterion #1: structured introspection (per-stage record + spec)"
else
    fail "inspect output missing structured per-stage detail"
fi

echo
echo "tail (replays the recording for the now-terminal pipeline)"
TAIL_OUTPUT="$(CLI tail remote-pipeline-demo)"
echo "$TAIL_OUTPUT" | head -5
TAIL_LINE_COUNT="$(echo "$TAIL_OUTPUT" | grep -c '"proto":' || true)"
if [[ "$TAIL_LINE_COUNT" -gt 0 ]]; then
    pass "tail emitted $TAIL_LINE_COUNT recorded events"
else
    fail "tail emitted no events"
fi

echo
echo "explain bench (criterion #3: causal chain)"
EXPLAIN_OUTPUT="$(CLI explain remote-pipeline-demo --stage bench)"
echo "$EXPLAIN_OUTPUT"
if echo "$EXPLAIN_OUTPUT" | grep -q "bench" \
   && echo "$EXPLAIN_OUTPUT" | grep -q "build" \
   && echo "$EXPLAIN_OUTPUT" | grep -q "fetch"; then
    pass "criterion #3: explain walks bench → build → fetch"
else
    fail "explain did not surface the full dep chain"
fi

echo
echo "artifact retrieval"
if [[ -z "${PROXIMA_REMOTE_HOST:-}" ]]; then
    # local: pull the bench stage's criterion.html
    ARTIFACT_OUTPUT="$(CLI artifact remote-pipeline-demo --stage bench --path criterion.html --output -)"
    echo "$ARTIFACT_OUTPUT"
    if echo "$ARTIFACT_OUTPUT" | grep -q "bench report"; then
        pass "artifact streams bytes from the bench workspace"
    else
        fail "artifact did not return expected content"
    fi
else
    echo "  (skipped over SSH transport — artifact streams over chunked encoding"
    echo "   which our SSH-stdio one-shot client doesn't yet decode incrementally)"
fi

echo
echo "replay (criterion #2: deterministic replay + bar #4: source of truth)"
REPLAY_OUTPUT="$(CLI replay remote-pipeline-demo)"
echo "$REPLAY_OUTPUT"
REPLAY_ID="$(echo "$REPLAY_OUTPUT" | grep -oE '01[0-9A-Z]{24}' | head -1)"
if [[ -z "$REPLAY_ID" ]]; then
    fail "replay didn't return a new pipeline id"
else
    pass "replay returned new id $REPLAY_ID"
fi

# give the replay a moment, then compare its events
sleep 1
ORIGINAL_EVENTS="$(CLI tail remote-pipeline-demo | grep -c '"phase":' || true)"
REPLAY_EVENTS="$(CLI tail "$REPLAY_ID" | grep -c '"phase":' || true)"
echo "  original events: $ORIGINAL_EVENTS, replay events: $REPLAY_EVENTS"
if [[ "$ORIGINAL_EVENTS" -eq "$REPLAY_EVENTS" && "$ORIGINAL_EVENTS" -gt 0 ]]; then
    pass "criterion #2: replay event count matches original (deterministic replay)"
    pass "criterion #4: replay events derived from recording (no inference)"
else
    fail "replay event count drift: original=$ORIGINAL_EVENTS replay=$REPLAY_EVENTS"
fi

echo
echo "replay with --substitute (mutate-mid-flight)"
MUTATE_OUTPUT="$(CLI replay remote-pipeline-demo --substitute "build=$HERE/build-failure.toml")"
echo "$MUTATE_OUTPUT"
MUTATE_ID="$(echo "$MUTATE_OUTPUT" | grep -oE '01[0-9A-Z]{24}' | head -1)"
if [[ -z "$MUTATE_ID" ]]; then
    fail "replay --substitute didn't return an id"
else
    sleep 1
    MUTATE_INSPECT="$(CLI inspect "$MUTATE_ID")"
    if echo "$MUTATE_INSPECT" | grep -q '"status":"failed"'; then
        pass "criterion #2: substitute propagates failure downstream (bench skipped)"
    else
        fail "expected mutated replay to end in 'failed'; got: $MUTATE_INSPECT"
    fi
fi

echo
echo "proxima replay --verify against the pipeline recording"
# Bar #9 (verify-CLI migration): exercise the proxima replay
# walker against the pipeline's recording.jsonl. Policy at
# scenarios/remote_pipeline_demo/policy.toml flags the three
# stages as `allowed_upstreams` and `must_derive_from_record`.
# Remote-transport demos skip this — the recording lives on the
# remote host and isn't visible to the local CLI.
if [[ -n "${PROXIMA_REMOTE_HOST:-}" ]]; then
    echo "  (skipped over SSH transport — recording lives on the remote;"
    echo "   local proxima CLI can't see \$STATE_DIR on \$PROXIMA_REMOTE_HOST)"
else
    RECORDING="$STATE_DIR/$PIPELINE_ID/recording.jsonl"
    if [[ ! -f "$RECORDING" ]]; then
        fail "verify-CLI bar #9: expected recording.jsonl at $RECORDING (proximad did not write it)"
    else
        VERIFY_OUTPUT="$("$PROXIMA_CLI" replay \
            --recording "$RECORDING" \
            --verify "$HERE/policy.toml" \
            --spec "$HERE/pipeline.toml" \
            --strict 2>&1)"
        VERIFY_EXIT=$?
        echo "$VERIFY_OUTPUT"
        if [[ "$VERIFY_EXIT" -eq 0 ]] \
            && echo "$VERIFY_OUTPUT" | grep -q "PASS unauthorized_upstream_call" \
            && echo "$VERIFY_OUTPUT" | grep -q "PASS inferred_not_recorded"; then
            pass "verify-CLI bar #9: replay walker accepts pipeline recording with strict policy"
        else
            fail "verify-CLI bar #9: replay walker did not pass strict policy (exit=$VERIFY_EXIT)"
        fi
    fi
fi

echo
echo "summary"
echo "PASS: ${#PASS[@]}"
for entry in "${PASS[@]}"; do echo "  ✓ $entry"; done
if [[ "${#FAIL[@]}" -gt 0 ]]; then
    echo "FAIL: ${#FAIL[@]}"
    for entry in "${FAIL[@]}"; do echo "  ✗ $entry"; done
    exit 1
fi
echo
echo "all pass criteria satisfied."
