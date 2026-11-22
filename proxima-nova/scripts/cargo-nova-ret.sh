#!/usr/bin/env bash
# installs cargo-nova into a tempdir root, then runs `cargo nova run` in a fresh copy of each fixture package
# prints `<fixture> nova=<exit> cargo_target_debug=<present|absent>` per fixture and `match=<n> of=2`
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
toolchain="nightly-2026-11-15"
cache="$HOME/.cache/proxima-nova/PROXIMANOVA-11"
mkdir -p "$cache"
workdir="$(mktemp -d "$cache/run.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

case "$(uname -s)" in
    Darwin) pattern='libproxima_nova_rustc\.dylib' ;;
    Linux) pattern='libproxima_nova_rustc\.so' ;;
    *) pattern='proxima_nova_rustc\.dll' ;;
esac

# build the backend the way proxima-nova-rustc/scripts/ret.sh does; cargo reports the dylib path
messages="$workdir/backend.json"
CARGO_TARGET_DIR="$root/proxima-nova-rustc/target" CARGO_BUILD_JOBS=2 \
    cargo "+$toolchain" build --manifest-path "$root/proxima-nova-rustc/Cargo.toml" \
    --message-format=json-render-diagnostics >"$messages"
NOVA_BACKEND="$(grep -m 1 -oE "[^\"]*$pattern" "$messages" || true)"
[[ -n "$NOVA_BACKEND" ]] || { echo "cargo reported no dylib" >&2; exit 1; }
export NOVA_BACKEND

install_root="$workdir/install"
CARGO_TARGET_DIR="$root/target" CARGO_BUILD_JOBS=2 cargo install --path "$root/proxima-nova" --features cli \
    --locked --offline --root "$install_root" --force >&2 \
    || { echo "cargo install failed" >&2; exit 1; }
export PATH="$install_root/bin:$PATH"

matched=0
run_fixture() {
    local stem="$1" expected="$2" scratch code debug
    scratch="$(mktemp -d "$workdir/fixture.XXXXXX")"
    cp -R "$root/proxima-nova/tests/fixtures/$stem-pkg/." "$scratch/"
    (cd "$scratch" && cargo nova run)
    code=$?
    debug=absent
    [[ -d "$scratch/target/debug" ]] && debug=present
    echo "$stem-pkg nova=$code cargo_target_debug=$debug"
    [[ "$code" == "$expected" ]] && matched=$((matched + 1))
}

run_fixture ret42 42
run_fixture ret7 7
echo "match=$matched of=2"
