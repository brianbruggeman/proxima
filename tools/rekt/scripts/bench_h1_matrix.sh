#!/usr/bin/env bash
# h1 server matrix: proxima (examples/bench_server) vs axum
# (examples/bench_server_axum), each driven by wrk AND rekt, across server
# core counts. the generator load is FIXED across cells (GEN_THREADS x
# CONNS_PER_THREAD) so the server's core count is the only variable.
#
# per bench point: rps median + CoV over TRIALS, wrk p50/p99, a warm
# single-request TTFB from curl (the number the load client hides), server
# CPU% and peak RSS sampled during the run, and an error gate.
#
# on a small loopback host the top cells oversubscribe generator + server;
# treat local numbers as harness validation — the bench box is authoritative.
#
#   scripts/bench_h1_matrix.sh [duration_secs] [trials]
#
# env: CORES_LIST="1 2 4 8"  GEN_THREADS=4  CONNS_PER_THREAD=32  PORT_BASE=8180
set -euo pipefail

DURATION="${1:-10}"
TRIALS="${2:-3}"
CORES_LIST="${CORES_LIST:-1 2 4 8}"
GEN_THREADS="${GEN_THREADS:-4}"
CONNS_PER_THREAD="${CONNS_PER_THREAD:-32}"
PORT_BASE="${PORT_BASE:-8180}"
TOTAL_CONNS=$((GEN_THREADS * CONNS_PER_THREAD))

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/target}"
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
RESULTS="${ROOT}/benches/RESULTS_h1_matrix_${PLATFORM}.md"
WORK_DIR="$(mktemp -d)"
ROWS="${WORK_DIR}/rows.tsv"
SERVER_PID=""
SAMPLER_PID=""

cleanup() {
    [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null || true
    [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

command -v wrk >/dev/null || { echo "wrk not found"; exit 1; }

echo "building (release)..."
(cd "${ROOT}" && cargo build --release --features scheduler \
    --example bench_server --example rekt_load >/dev/null)
(cd "${ROOT}" && cargo build --release --example bench_server_axum >/dev/null)

server_bin() {
    case "$1" in
        proxima) echo "${TARGET_DIR}/release/examples/bench_server" ;;
        axum)    echo "${TARGET_DIR}/release/examples/bench_server_axum" ;;
    esac
}

# port-squat guard: a stale listener silently benches the wrong server
assert_port_free() {
    local port="$1"
    if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "port ${port} already has a listener; refusing to bench" >&2
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN >&2
        exit 1
    fi
}

wait_ready() {
    local url="$1"
    for _ in $(seq 1 50); do
        local body
        body="$(curl -s --max-time 1 "${url}" 2>/dev/null || true)"
        [[ "${body}" == "ok" ]] && return 0
        sleep 0.1
    done
    echo "server never became ready at ${url}" >&2
    return 1
}

assert_listener_is_server() {
    local port="$1" pid="$2"
    local owner
    owner="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | head -1)"
    if [[ "${owner}" != "${pid}" ]]; then
        echo "listener on port ${port} is pid ${owner}, not our server ${pid}" >&2
        exit 1
    fi
}

# %cpu on macOS sums threads and decays over ~seconds; sampling only inside the
# load window keeps it honest enough to separate compute-bound from idle
sample_usage() {
    local pid="$1" out="$2"
    while kill -0 "${pid}" 2>/dev/null && [[ -f "${WORK_DIR}/sampling" ]]; do
        ps -o %cpu=,rss= -p "${pid}" >> "${out}" 2>/dev/null || break
        sleep 0.5
    done
}

start_sampler() {
    : > "${WORK_DIR}/usage"
    touch "${WORK_DIR}/sampling"
    sample_usage "${SERVER_PID}" "${WORK_DIR}/usage" &
    SAMPLER_PID=$!
}

stop_sampler() {
    rm -f "${WORK_DIR}/sampling"
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=""
    awk '{ cpu += $1; rss = ($2 > rss) ? $2 : rss; n += 1 }
         END { if (n > 0) printf "%.0f %.1f\n", cpu / n, rss / 1024; else print "0 0" }' \
        "${WORK_DIR}/usage"
}

curl_ttfb_ms() {
    local url="$1"
    for _ in $(seq 1 5); do
        curl -s -o /dev/null -w '%{time_starttransfer}\n' "${url}"
    done | sort -n | awk 'NR == 3 { printf "%.2f", $1 * 1000 }'
}

to_ms() {
    awk -v raw="$1" 'BEGIN {
        if (raw ~ /us$/)      printf "%.2f", substr(raw, 1, length(raw) - 2) / 1000
        else if (raw ~ /ms$/) printf "%.2f", substr(raw, 1, length(raw) - 2) + 0
        else if (raw ~ /m$/)  printf "%.2f", substr(raw, 1, length(raw) - 1) * 60000
        else if (raw ~ /s$/)  printf "%.2f", substr(raw, 1, length(raw) - 1) * 1000
        else print "-"
    }'
}

median_and_cov() {
    printf '%s\n' "$@" | sort -n | awk '
        { values[NR] = $1; sum += $1 }
        END {
            median = (NR % 2) ? values[(NR + 1) / 2] \
                              : (values[NR / 2] + values[NR / 2 + 1]) / 2
            mean = sum / NR
            for (i = 1; i <= NR; i++) varsum += (values[i] - mean) ^ 2
            cov = (mean > 0 && NR > 1) ? sqrt(varsum / (NR - 1)) / mean * 100 : 0
            printf "%.0f %.1f\n", median, cov
        }'
}

run_wrk() {
    local url="$1" out="$2"
    wrk -t"${GEN_THREADS}" -c"${TOTAL_CONNS}" -d"${DURATION}s" --latency "${url}" > "${out}" 2>&1
    local rps p50 p99 errors
    rps="$(awk '/^Requests\/sec/ { print $2 }' "${out}")"
    p50="$(to_ms "$(awk '$1 == "50%" { print $2 }' "${out}")")"
    p99="$(to_ms "$(awk '$1 == "99%" { print $2 }' "${out}")")"
    errors="$(awk '
        /Socket errors/ { gsub(/,/, ""); total += $4 + $6 + $8 + $10 }
        /Non-2xx/       { total += $NF }
        END             { print total + 0 }' "${out}")"
    echo "${rps} ${p50} ${p99} ${errors}"
}

run_rekt() {
    local url="$1" out="$2"
    "${TARGET_DIR}/release/examples/rekt_load" \
        "${url}" "${CONNS_PER_THREAD}" "${DURATION}" "${GEN_THREADS}" > "${out}" 2>&1
    local rps errors
    rps="$(awk -F': ' '/Requests\/sec/ { print $NF }' "${out}")"
    errors="$(awk '/completed,/ { print $(NF - 1) }' "${out}")"
    echo "${rps} - - ${errors}"
}

echo "h1 matrix: servers={proxima,axum} cores={${CORES_LIST}} gens={wrk,rekt}"
echo "load: ${GEN_THREADS} threads x ${CONNS_PER_THREAD} conns = ${TOTAL_CONNS}, ${DURATION}s x ${TRIALS} trial(s)"
echo

cell=0
for server in proxima axum; do
    for cores in ${CORES_LIST}; do
        port=$((PORT_BASE + cell)); cell=$((cell + 1))
        url="http://127.0.0.1:${port}/"

        assert_port_free "${port}"
        "$(server_bin "${server}")" "127.0.0.1:${port}" "${cores}" \
            > "${WORK_DIR}/${server}-${cores}.log" 2>&1 &
        SERVER_PID=$!
        wait_ready "${url}"
        assert_listener_is_server "${port}" "${SERVER_PID}"

        ttfb="$(curl_ttfb_ms "${url}")"

        for gen in wrk rekt; do
            rps_list=() p50="-" p99="-" cpu="-" rss="-" errors=0
            for trial in $(seq 1 "${TRIALS}"); do
                start_sampler
                read -r rps trial_p50 trial_p99 trial_errors \
                    < <(run_${gen} "${url}" "${WORK_DIR}/${gen}-${server}-${cores}-${trial}.log")
                read -r cpu rss < <(stop_sampler)
                rps_list+=("${rps}")
                p50="${trial_p50}"; p99="${trial_p99}"
                errors=$((errors + trial_errors))
                sleep 1
            done
            read -r rps_median rps_cov < <(median_and_cov "${rps_list[@]}")
            printf '%-8s %d cores  %-4s  %8s rps (cov %s%%)  p50 %-8s p99 %-8s ttfb %sms  cpu %s%%  rss %sMB  errors %s\n' \
                "${server}" "${cores}" "${gen}" "${rps_median}" "${rps_cov}" \
                "${p50}" "${p99}" "${ttfb}" "${cpu}" "${rss}" "${errors}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${server}" "${cores}" "${gen}" "${rps_median}" "${rps_cov}" \
                "${p50}" "${p99}" "${ttfb}" "${cpu}" "${rss}" "${errors}" >> "${ROWS}"
        done

        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    done
done

{
    echo "# h1 matrix: proxima vs axum (${PLATFORM})"
    echo
    echo "- date: $(date '+%Y-%m-%d %H:%M')"
    echo "- load: ${GEN_THREADS} threads x ${CONNS_PER_THREAD} conns = ${TOTAL_CONNS}, ${DURATION}s, ${TRIALS} trial(s), median rps"
    echo "- host: $(sysctl -n hw.ncpu 2>/dev/null || nproc) cpus, loadavg $(uptime | awk -F'load averages?: ' '{ print $2 }')"
    echo "- latency p50/p99 from wrk --latency; ttfb = warm single-request curl median (5 shots)"
    echo
    echo "| server | cores | gen | rps | cov% | p50 ms | p99 ms | ttfb ms | cpu% | rss MB | errors |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|"
    awk -F'\t' '{ printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 }' "${ROWS}"
    echo
    echo "## proxima / axum rps ratio"
    echo
    echo "| cores | gen | proxima | axum | ratio |"
    echo "|---|---|---|---|---|"
    awk -F'\t' '
        $1 == "proxima" { prox[$2 "/" $3] = $4 }
        $1 == "axum"    { ax[$2 "/" $3] = $4 }
        END {
            split("'"${CORES_LIST}"'", cores_arr, " ")
            split("wrk rekt", gens_arr, " ")
            for (ci = 1; ci in cores_arr; ci++)
                for (gi = 1; gi in gens_arr; gi++) {
                    key = cores_arr[ci] "/" gens_arr[gi]
                    if (key in prox && key in ax && ax[key] > 0)
                        printf "| %s | %s | %s | %s | %.2fx |\n",
                            cores_arr[ci], gens_arr[gi], prox[key], ax[key], prox[key] / ax[key]
                }
        }' "${ROWS}"
} > "${RESULTS}"

echo
echo "results written to ${RESULTS}"
