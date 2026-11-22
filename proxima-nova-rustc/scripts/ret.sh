#!/usr/bin/env bash
# no flag: build both fixtures with the nova backend and stock rustc, run both, compare exit codes
# --list: print the mono items nova sees for each fixture
# --emit-obj: --emit=obj -o must leave the object at the -o path, as LLVM does
# --control: compare nova's ret42 against llvm's ret7 through the same compare; must report match=0
set -uo pipefail

mode="${1:-}"
root="$(cd "$(dirname "$0")/.." && pwd)"
toolchain="nightly-2026-11-15"
cache="$HOME/.cache/proxima-nova/PROXIMANOVA-8"
mkdir -p "$cache"

case "$(uname -s)" in
    Darwin) pattern='libproxima_nova_rustc\.dylib' ;;
    Linux) pattern='libproxima_nova_rustc\.so' ;;
    *) pattern='proxima_nova_rustc\.dll' ;;
esac

messages="$(mktemp "$cache/build.XXXXXX")"
CARGO_TARGET_DIR="$root/target" CARGO_BUILD_JOBS=2 \
    cargo "+$toolchain" build --manifest-path "$root/Cargo.toml" \
    --message-format=json-render-diagnostics >"$messages"
# cargo reports where it uplifted the dylib, so its location is never assumed
dylib="$(grep -m 1 -oE "[^\"]*$pattern" "$messages" || true)"
rm -f "$messages"
[[ -n "$dylib" ]] || { echo "cargo reported no dylib" >&2; exit 1; }

rustc_args=(--edition 2024 -C panic=abort --crate-type bin)
# the fixtures are no_std and rustc links with -nodefaultlibs, so dyld_stub_binder needs libSystem explicitly
[[ "$(uname -s)" == Darwin ]] && rustc_args+=(-C link-arg=-lSystem)

list_fixture() {
    local fixture="$1" out listing count
    out="$(mktemp -d "$cache/run.XXXXXX")"
    listing="$(rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" -Cllvm-args=nova-list "$fixture" -o "$out/a.out")"
    rm -rf "$out"
    echo "$listing" | grep '^mono_item=' || true
    count="$(echo "$listing" | grep -c '^mono_item=' || true)"
    echo "$(basename "$fixture" .rs) mono_items=$count"
}

# prints the exit code of the built binary; compiler output goes to stderr and a build log
build_and_run() {
    local backend="$1" fixture="$2" out code
    out="$(mktemp -d "$cache/run.XXXXXX")"
    if [[ "$backend" == nova ]]; then
        rustc "+$toolchain" "${rustc_args[@]}" "-Zcodegen-backend=$dylib" "$fixture" -o "$out/a.out" 2>"$out/rustc.log"
    else
        rustc "+$toolchain" "${rustc_args[@]}" "$fixture" -o "$out/a.out" 2>"$out/rustc.log"
    fi
    cat "$out/rustc.log" >>"$cache/rustc-${backend}.log"
    if [[ ! -x "$out/a.out" ]]; then
        cat "$out/rustc.log" >&2
        rm -rf "$out"
        echo "build-failed"
        return
    fi
    "$out/a.out"
    code=$?
    rm -rf "$out"
    echo "$code"
}

compare() {
    local label="$1" nova_fixture="$2" llvm_fixture="$3" nova llvm
    nova="$(build_and_run nova "$nova_fixture")"
    llvm="$(build_and_run llvm "$llvm_fixture")"
    echo "$label nova=$nova llvm=$llvm"
    [[ "$nova" =~ ^[0-9]+$ && "$nova" == "$llvm" ]]
}

nova_log="$cache/rustc-nova.log"

count_warnings() {
    grep -c '^warning' "$1" || true
}

# --emit=obj -o <dir>/<label>.o with one backend; the check reads only the file the compile left
emit_check() {
    local label="$1" dir symbols present=absent
    shift
    dir="$(mktemp -d "$cache/emit.XXXXXX")"
    rustc "+$toolchain" "${rustc_args[@]}" "$@" --emit=obj "$fixtures/ret42.rs" -o "$dir/$label.o" >"$dir/rustc.out" 2>"$dir/rustc.log"
    [[ -f "$dir/$label.o" ]] && present=present
    : >"$dir/nm.out"
    [[ "$present" == present ]] && nm "$dir/$label.o" >"$dir/nm.out" 2>&1
    symbols="$(grep -c ' T _main$' "$dir/nm.out" || true)"
    rm -rf "$dir"
    echo "$label emit_obj=$present main_symbol=$symbols"
}

fixtures="$root/tests/fixtures"
case "$mode" in
    --list)
        list_fixture "$fixtures/ret42.rs"
        list_fixture "$fixtures/ret7.rs"
        ;;
    --emit-obj)
        emit_check nova "-Zcodegen-backend=$dylib"
        emit_check llvm
        emit_check control "-Zcodegen-backend=$dylib" -Cllvm-args=nova-list
        ;;
    --control)
        : >"$nova_log"
        matched=0
        compare control "$fixtures/ret42.rs" "$fixtures/ret7.rs" && matched=1
        echo "control match=$matched of=1"
        # a fixture with a dead function must warn, so the counter is shown to see warnings
        dir="$(mktemp -d "$cache/control.XXXXXX")"
        cp "$fixtures/ret42.rs" "$dir/ret42_unused.rs"
        echo 'fn unused_helper() {}' >>"$dir/ret42_unused.rs"
        : >"$nova_log"
        build_and_run nova "$dir/ret42_unused.rs" >/dev/null
        echo "control_warnings=$(count_warnings "$nova_log")"
        rm -rf "$dir"
        ;;
    "")
        : >"$nova_log"
        matched=0
        for stem in ret42 ret7; do
            compare "$stem" "$fixtures/$stem.rs" "$fixtures/$stem.rs" && matched=$((matched + 1))
        done
        echo "match=$matched of=2"
        echo "nova_warnings=$(count_warnings "$nova_log")"
        ;;
    *)
        echo "usage: ret.sh [--list|--control|--emit-obj]" >&2
        exit 2
        ;;
esac
