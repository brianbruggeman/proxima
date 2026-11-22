#!/usr/bin/env bash
# usage: nova-bench.sh [--time-passes] FIXTURE...   (FIXTURE is M8, S8, S32 or M32)
# NOVA_BENCH_ARMS   comma list, default llvm,nova (llvm_b is a second stock-llvm arm, the control)
# NOVA_BENCH_RUNS   runs per arm per fixture, default 5
# NOVA_BENCH_DYLIB  nova backend dylib, default the release dylib
# every build gets its own empty directory (arm, fixture, run); runs go ABAB
# bench fixture=.. arm=.. n=.. maxrss_mb_median=.. real_s_median=.. instr_median=.. exit=..
# ratio fixture=.. maxrss=.. real=..        (second arm over first)
# tp fixture=.. arm=.. dylib=.. codegen_crate_s=.. codegen_crate_rss_delta_mb=.. incr_lines=.. nova_lower_mir_s=.. nova_lower_functions=..   (--time-passes; the nova fields come from -Cllvm-args=nova-time)
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cache="$HOME/.cache/proxima-nova"
work="$cache/nova-bench"
arms="${NOVA_BENCH_ARMS:-llvm,nova}"
runs="${NOVA_BENCH_RUNS:-5}"
toolchain="nightly-2026-11-15"
rustc_bin="$(rustup which --toolchain "$toolchain" rustc)" || { echo "no rustc for $toolchain" >&2; exit 1; }

case "$(uname -s)" in
    Darwin) ext=dylib ;;
    *) ext=so ;;
esac
dylib="${NOVA_BENCH_DYLIB:-${CARGO_TARGET_DIR:-$root/../target}/release/libproxima_nova_rustc.$ext}"

tp_mode=0
fixtures=()
for arg in "$@"; do
    if [[ "$arg" == --time-passes ]]; then tp_mode=1; else fixtures+=("$arg"); fi
done
[[ ${#fixtures[@]} -gt 0 ]] || { echo "no fixture given (M8, S8, S32, M32)" >&2; exit 1; }

fixture_path() {
    case "$1" in
        M8) echo "$cache/PROXIMANOVA-46/modules/big_8000.rs" ;;
        S8) echo "$cache/PROXIMANOVA-45/big_8000.rs" ;;
        S32) echo "$cache/PROXIMANOVA-46/big_32000.rs" ;;
        M32) echo "$cache/PROXIMANOVA-46/modules/big_32000.rs" ;;
        *) echo "unknown fixture $1" >&2; return 1 ;;
    esac
}

flags=(--edition 2024 -C panic=abort -C overflow-checks=off --crate-type bin -C link-arg=-lSystem -C opt-level=0 -C debuginfo=0)

# stock arms add no backend flag; every other arm name is the nova backend
arm_flags() {
    case "$1" in
        llvm | llvm_b) arm_extra=() ;;
        *) arm_extra=("-Zcodegen-backend=$dylib") ;;
    esac
}

# one exit-capture function for every arm
exit_of() {
    [[ -x "$1" ]] || { echo nobinary; return; }
    timeout 60 "$1" >/dev/null 2>&1
    echo $?
}

# median of the numbers on stdin, three decimals
median() {
    sort -n | awk '{ v[NR] = $1 } END { if (NR == 0) { print "nan"; exit } printf "%.3f\n", (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'
}

# field <file> <label>: the number in front of the label in /usr/bin/time -l output
field() { grep -E "$2" "$1" | head -n 1 | awk -v col="$3" '{ print $col }'; }

build_dir() { echo "$work/$1_$2_$3"; }

one_build() {
    local arm="$1" fixture="$2" run="$3" dir source
    dir="$(build_dir "$arm" "$fixture" "$run")"
    source="$(fixture_path "$fixture")" || return 1
    rm -rf "$dir"
    mkdir -p "$dir"
    arm_flags "$arm"
    timeout 300 /usr/bin/time -l -o "$dir/time.txt" "$rustc_bin" "${flags[@]}" "${arm_extra[@]}" \
        "$source" -o "$dir/bin" >"$dir/build.log" 2>&1
    echo "$?" >"$dir/build_exit.txt"
    exit_of "$dir/bin" >"$dir/exit.txt"
}

shared_exit() {
    local arm="$1" fixture="$2" run first value
    first=""
    for ((run = 1; run <= runs; run++)); do
        value="$(cat "$(build_dir "$arm" "$fixture" "$run")/exit.txt")"
        if [[ -z "$first" ]]; then first="$value"; elif [[ "$value" != "$first" ]]; then echo mixed; return; fi
    done
    echo "$first"
}

arm_median() {
    local arm="$1" fixture="$2" column="$3" label="$4" scale="$5" run
    for ((run = 1; run <= runs; run++)); do
        field "$(build_dir "$arm" "$fixture" "$run")/time.txt" "$label" "$column"
    done | awk -v scale="$scale" '{ printf "%.6f\n", $1 / scale }' | median
}

report_bench() {
    local arm="$1" fixture="$2" rss real instr
    rss="$(arm_median "$arm" "$fixture" 1 'maximum resident set size' 1000000)"
    real="$(arm_median "$arm" "$fixture" 1 ' real ' 1)"
    instr="$(arm_median "$arm" "$fixture" 1 'instructions retired' 1)"
    echo "bench fixture=$fixture arm=$arm n=$runs maxrss_mb_median=$rss real_s_median=$real instr_median=$instr exit=$(shared_exit "$arm" "$fixture")"
    echo "$rss $real" >"$work/med_${arm}_${fixture}.txt"
}

report_ratio() {
    local fixture="$1" first="$2" second="$3"
    read -r first_rss first_real <"$work/med_${first}_${fixture}.txt"
    read -r second_rss second_real <"$work/med_${second}_${fixture}.txt"
    awk -v fixture="$fixture" -v ar="$first_rss" -v br="$first_real" -v bs="$second_rss" -v bt="$second_real" \
        'BEGIN { printf "ratio fixture=%s maxrss=%.3f real=%.3f\n", fixture, bs / ar, bt / br }' \
        2>/dev/null || true
}

time_passes() {
    local arm="$1" fixture="$2" dir source label name
    dir="$work/tp_${arm}_${fixture}"
    source="$(fixture_path "$fixture")" || return 1
    rm -rf "$dir"
    mkdir -p "$dir"
    arm_flags "$arm"
    case "$arm" in
        llvm | llvm_b) label=none; mode_extra=() ;;
        *) label="$(basename "$dylib")"; mode_extra=("-Cllvm-args=nova-time") ;;
    esac
    timeout 300 "$rustc_bin" "${flags[@]}" "${arm_extra[@]}" "${mode_extra[@]}" -Z time-passes "$source" -o "$dir/bin" >"$dir/tp.txt" 2>&1
    nova_line="$(grep -E '^nova_time ' "$dir/tp.txt" | head -n 1)"
    lower_s="$(echo "$nova_line" | sed -nE 's/.*lower_mir_s=([0-9.]+).*/\1/p')"
    lower_n="$(echo "$nova_line" | sed -nE 's/.*functions=([0-9]+).*/\1/p')"
    line="$(grep -E 'codegen_crate$' "$dir/tp.txt" | head -n 1)"
    seconds="$(echo "$line" | sed -E 's/^time: *([0-9.]+);.*/\1/')"
    delta="$(echo "$line" | sed -E 's/.*\( *\+?(-?[0-9]+)MB\).*/\1/')"
    incr="$(grep -cE 'incr_comp|serialize_dep_graph' "$dir/tp.txt")"
    echo "tp fixture=$fixture arm=$arm dylib=$label codegen_crate_s=$seconds codegen_crate_rss_delta_mb=$delta incr_lines=$incr nova_lower_mir_s=${lower_s:-na} nova_lower_functions=${lower_n:-na}"
}

mkdir -p "$work"
IFS=',' read -r -a arm_list <<<"$arms"

for fixture in "${fixtures[@]}"; do
    if [[ $tp_mode -eq 1 ]]; then
        for arm in "${arm_list[@]}"; do time_passes "$arm" "$fixture"; done
        continue
    fi
    for ((run = 1; run <= runs; run++)); do
        for arm in "${arm_list[@]}"; do one_build "$arm" "$fixture" "$run"; done
    done
    for arm in "${arm_list[@]}"; do report_bench "$arm" "$fixture"; done
    [[ ${#arm_list[@]} -ge 2 ]] && report_ratio "$fixture" "${arm_list[0]}" "${arm_list[1]}"
done
