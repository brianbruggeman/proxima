#!/usr/bin/env bash
# rekt vs wrk vs hey, all against the same fixed target (examples/bench_target).
# rekt drives THREADS one-core prime runtimes, each with CONNECTIONS keep-alive
# clients; wrk uses -tTHREADS -c(THREADS*CONNECTIONS). measures completed
# requests/sec — load-generator efficiency, not the server.
#
# NOTE the localhost target caps near ~150k req/s (loopback + a userspace
# server); above that both tools are server-bound. for per-core generator
# efficiency read the THREADS=1 row.
#
#   scripts/bench_vs_wrk.sh [connections_per_thread] [duration_secs] [threads] [port]
set -euo pipefail

CONNECTIONS="${1:-25}"
DURATION="${2:-5}"
THREADS="${3:-1}"
PORT="${4:-8080}"
URL="http://127.0.0.1:${PORT}/"
TOTAL_CONNS=$((CONNECTIONS * THREADS))

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')"
SERVER_BIN="${TARGET_DIR}/release/examples/bench_target"
REKT_BIN="${TARGET_DIR}/release/examples/rekt_load"

echo "building (release)..."
cargo build --release --example bench_target >/dev/null 2>&1
cargo build --release --features scheduler --example rekt_load >/dev/null 2>&1

echo "starting target on ${URL} ..."
"${SERVER_BIN}" "127.0.0.1:${PORT}" >/tmp/bench_target.log 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
sleep 1

# warm up the target / connections
curl -s -o /dev/null "${URL}" || { echo "target not responding"; exit 1; }

echo
echo "${THREADS} thread(s) x ${CONNECTIONS} conns = ${TOTAL_CONNS} connections, ${DURATION}s"

echo "--- rekt ---"
"${REKT_BIN}" "${URL}" "${CONNECTIONS}" "${DURATION}" "${THREADS}" | sed 's/^/  /'

if command -v wrk >/dev/null 2>&1; then
  echo "--- wrk -t${THREADS} -c${TOTAL_CONNS} ---"
  wrk -t"${THREADS}" -c"${TOTAL_CONNS}" -d"${DURATION}s" "${URL}" 2>&1 \
    | grep -E "Requests/sec|Socket errors" | sed 's/^/  /'
fi

if command -v hey >/dev/null 2>&1; then
  echo "--- hey -c${TOTAL_CONNS} ---"
  hey -c "${TOTAL_CONNS}" -z "${DURATION}s" "${URL}" 2>&1 \
    | grep -E "Requests/sec" | sed 's/^/  /'
fi
