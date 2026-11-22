#!/usr/bin/env bash
# nova-compare.sh <ir|layout|list> <before.dylib> <after.dylib>: per tests/corpus member, the adapter's listing from each dylib must be byte-identical
# nova-compare.sh --control <ir|layout|list> <dylib>: a fixed pair of members (ir, list: arith and call_helper; layout: struct_fields and enum_copy_fields), both non-empty in that mode, must differ; prints <mode>_lines_a=<n> <mode>_lines_b=<n>, and an empty side is a vacuous control, exit 1
# output: <mode>_same=<n> of=<n>, then <mode>_lines=<n> (listing lines read; 0 is RED, identical emptiness proves nothing); exit 1 unless all members are the same and lines > 0 (control: unless it differs)
root="$(cd "$(dirname "$0")/.." && pwd)"
toolchain="nightly-2026-11-15"
cache="$HOME/.cache/proxima-nova/PROXIMANOVA-47"
mkdir -p "$cache"
run="$(mktemp -d "$cache/cmp.XXXXXX")"
trap 'rm -rf "$run"' EXIT
rustc_args=(--edition 2024 -C panic=abort -C overflow-checks=off --crate-type bin -C link-arg=-lSystem -C opt-level=0 -C debuginfo=0)

control=0
[[ "$1" == --control ]] && { control=1; shift; }
mode="$1"
case "$mode" in
    ir) flag=nova-ir; keep='^nova_ir ' ;;
    layout) flag=nova-layout; keep='^nova_(layout|deref) ' ;;
    list) flag=nova-list; keep='^mono_item=' ;;
    *) echo "usage: nova-compare.sh [--control] <ir|layout|list> <dylib> [dylib]" >&2; exit 2 ;;
esac

listing() {
    local dylib="$1" source="$2" out="$3"
    timeout -k 5s 60s rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" "-Cllvm-args=$flag" "$source" -o "$run/x.bin" 2>/dev/null | grep -E "$keep" >"$out"
}

if [[ "$control" == 1 ]]; then
    case "$mode" in
        ir | list) member_a=arith; member_b=call_helper ;;
        layout) member_a=struct_fields; member_b=enum_copy_fields ;;
    esac
    listing "$2" "$root/tests/corpus/$member_a.rs" "$run/a.txt"
    listing "$2" "$root/tests/corpus/$member_b.rs" "$run/b.txt"
    lines_a="$(wc -l <"$run/a.txt" | tr -d ' ')"
    lines_b="$(wc -l <"$run/b.txt" | tr -d ' ')"
    echo "${mode}_lines_a=$lines_a ${mode}_lines_b=$lines_b"
    [[ "$lines_a" -gt 0 && "$lines_b" -gt 0 ]] || { echo "vacuous control" >&2; exit 1; }
    same=0
    cmp -s "$run/a.txt" "$run/b.txt" && same=1
    echo "${mode}_same=$same of=1"
    [[ "$same" == 0 ]]
    exit
fi

total=0
same=0
lines=0
for source in "$root"/tests/corpus/*.rs; do
    total=$((total + 1))
    listing "$2" "$source" "$run/before.txt"
    listing "$3" "$source" "$run/after.txt"
    lines=$((lines + $(wc -l <"$run/before.txt")))
    cmp -s "$run/before.txt" "$run/after.txt" && same=$((same + 1))
done
echo "${mode}_same=$same of=$total"
echo "${mode}_lines=$lines"
[[ "$total" -gt 0 && "$lines" -gt 0 && "$same" -eq "$total" ]]
