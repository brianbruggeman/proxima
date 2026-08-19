#!/usr/bin/env bash
# examples-gate.sh — build AND RUN the workspace's examples.
#
# This gate exists because `cargo check --all-targets` compiles every example
# and executes none of them, and nothing else in CI runs them either. Two
# classes of breakage have reached public main through that hole and both were
# found again on 2026-08-05:
#
#   - an example that compiles and PANICS on a stale assertion
#     (quic_connection_state_walkthrough asserted IdleClosed where the FSM
#     returns HandshakeTimeout — a teaching walkthrough, wrong for a reader
#     and green for the compiler);
#   - an example that compiles and dies at startup on a feature its manifest
#     never enabled (vscode-proxy-capture: `Registry("no listen protocol
#     named 'http'")`, a runtime error by construction, never a compile one).
#
# A long-running server example is expected to still be alive at the timeout;
# that is a PASS and is reported separately from a clean exit, because
# "exited 0" and "served until killed" are different proofs and collapsing
# them would hide a server that exits early.
#
# Examples needing hardware this host does not have (DPDK, XDP, a real NVMe
# controller) are reported as SKIP-NO-HARDWARE by name. They are never
# silently dropped: a shrinking run set is exactly how this class hides.
#
# Usage:  bash scripts/examples-gate.sh [timeout-seconds]
# Exits 0 if every runnable example built and ran, non-zero otherwise.

set -u

TIMEOUT="${1:-25}"
cd "$(dirname "$0")/.." || exit 1

# one feature union per package, chosen so a single build covers that
# package's examples. keep in sync with the `required-features` in the
# manifests -- an example whose features are missing here fails the
# "never built" check below rather than vanishing.
ROOT_UNION="http1,http1-native,http2,http3,http-prime-deps,tracing-init,tokio,runtime-tokio,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,macros,instrument-metrics,histogram,otlp-http,serve-prime,pgwire,redis-listener,memcached-listener,memcached-client,dns-listener,dns-client,kafka-listener,kafka-client,mqtt-listener,mqtt-client,amqp-listener,amqp-client,tls,h3-native-upstream,sync-wrappers"

feats_for() {
  case "$1" in
    proxima)            echo "$ROOT_UNION" ;;
    prime)              echo "runtime-prime-inbox-alloc,runtime-prime-inbox-dynamic" ;;
    proxima-centauri)   echo "aead-chacha20poly1305" ;;
    proxima-http)       echo "http3-native" ;;
    proxima-intercept)  echo "intercept-capture,intercept-replay,quic-intercept" ;;
    proxima-patterns)   echo "alert,std,proto,json-shape" ;;
    proxima-pgwire)     echo "scram" ;;
    proxima-protocols)  echo "dns-codec-trait,pgwire_codec,quic-mock-tls,quic" ;;
    proxima-redis)      echo "client" ;;
    proxima-telemetry)  echo "instrument-metrics,macros,elevation" ;;
    rekt)               echo "scheduler" ;;
    *)                  echo "" ;;
  esac
}

# examples that cannot prove anything without an argument, an env var, or a
# capture file that a prior run produced. each is listed with WHY, so the list
# stays auditable rather than becoming a place to hide a failure.
needs_input() {
  case "$1" in
    # takes a dump path as argv[1] and prints usage without one
    decode-h2-dump|decode-ws-deflate-dump) return 0 ;;
    # requires PROXIMA_UPSTREAM_MAP / PROXIMA_UPSTREAM_ADDR
    transparent-capture) return 0 ;;
    # replays a capture a previous transparent-capture run wrote
    replay-capture) return 0 ;;
    # need a live postgres / redis on a known port
    capture_realpg|capture_realredis) return 0 ;;
    *) return 1 ;;
  esac
}

pass=0; fail=0; served=0; skipped=0; gated=0
failed_names=""

while IFS=$'\t' read -r pkg name reqfeats; do
  case "$reqfeats" in
    *dpdk*|*xdp*|*nvme-uio*)
      echo "SKIP-NO-HARDWARE  $pkg :: $name  ($reqfeats)"
      gated=$((gated + 1)); continue ;;
  esac

  if needs_input "$name"; then
    echo "SKIP-NEEDS-INPUT  $pkg :: $name"
    skipped=$((skipped + 1)); continue
  fi

  feats=$(feats_for "$pkg")
  if [ -n "$feats" ]; then
    set -- run -p "$pkg" --example "$name" --features "$feats"
  else
    set -- run -p "$pkg" --example "$name"
  fi

  timeout -s KILL "$TIMEOUT" cargo "$@" < /dev/null > /tmp/examples-gate-$$.log 2>&1
  code=$?
  case $code in
    0)
      echo "ok                $pkg :: $name"
      pass=$((pass + 1)) ;;
    124|137)
      echo "ok (served)       $pkg :: $name  — still serving at ${TIMEOUT}s"
      served=$((served + 1)) ;;
    *)
      echo "FAIL              $pkg :: $name  — exit $code"
      tail -20 /tmp/examples-gate-$$.log
      fail=$((fail + 1)); failed_names="$failed_names $pkg::$name" ;;
  esac
  rm -f /tmp/examples-gate-$$.log
done < <(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] as $p | $p.targets[] | select(.kind[]=="example")
           | [$p.name, .name, ((."required-features" // []) | join(","))] | @tsv' \
  | sort)

echo
echo "ran ok: $pass    served: $served    no-hardware: $gated    needs-input: $skipped    FAILED: $fail"
if [ "$fail" -ne 0 ]; then
  echo "failing examples:$failed_names"
  exit 1
fi

# a run set that silently shrank to nothing would report 0 failures and pass.
if [ "$((pass + served))" -lt 60 ]; then
  echo "gate error: only $((pass + served)) examples executed; expected at least 60"
  exit 1
fi
exit 0
