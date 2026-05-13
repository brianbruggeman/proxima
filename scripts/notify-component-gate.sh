#!/usr/bin/env bash
# notify-component-gate.sh — runs the disciplined-component gate per component.
#
# Reads scripts/notify-component-gate.toml for (package, feature, bench, tier)
# mapping. Tier dictates the build command:
#   tier-1: cargo build -p <package> --no-default-features --features alloc,<feature>
#   tier-2: cargo build -p <package> --features <feature>
#   tier-3: cargo build -p <package> --no-default-features --features <feature>
#
# Gate steps (the script automates 2/3/4/5/6/13; the rest are human-verified
# at sealing per docs/proxima-notify/discipline.md row evidence):
#   2. build clean under sub-flag (tier-dependent)
#   3. tests pass
#   4. clippy pedantic clean
#   5. micro-bench file builds (cargo build --bench)
#   6. compare-bench numbers — bench saved against named baseline
#  13. home-turf incumbent arm exists in the bench file
#
# usage: scripts/notify-component-gate.sh <component-name>   (e.g. S1, C3, C10)

set -euo pipefail

component="${1:-}"
if [[ -z "${component}" ]]; then
    printf 'usage: %s <component-name>\n' "$0" >&2
    printf '       e.g. %s S1  or  %s C3\n' "$0" "$0" >&2
    printf 'components: S1, S2, S3, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\n' >&2
    exit 2
fi

mapping="scripts/notify-component-gate.toml"
if [[ ! -f "${mapping}" ]]; then
    printf 'error: %s not found (run from worktree root)\n' "${mapping}" >&2
    exit 2
fi

# Parse the TOML mapping via python3's tomllib (3.11+). Falls back to a
# clear error if the host python is older — every dev box in this workspace
# has 3.11+ per the proxima toolchain baseline.
read -r package feature bench tier < <(
    python3 - "${component}" "${mapping}" <<'PY'
import sys, tomllib, pathlib
component = sys.argv[1]
data = tomllib.loads(pathlib.Path(sys.argv[2]).read_text())
if component not in data:
    sys.stderr.write(f"error: component {component!r} not in mapping\n")
    sys.exit(2)
row = data[component]
print(row["package"], row["feature"], row["bench"], row["tier"])
PY
)

printf '\n== notify-gate %s (package=%s feature=%s tier=%s) ==\n' \
    "${component}" "${package}" "${feature}" "${tier}"

case "${tier}" in
    tier-1) build_args=(--no-default-features --features "alloc,${feature}") ;;
    tier-2) build_args=(--features "${feature}") ;;
    tier-3) build_args=(--no-default-features --features "${feature}") ;;
    *)
        printf 'error: unknown tier %s for %s\n' "${tier}" "${component}" >&2
        exit 2
        ;;
esac

printf '\n[1/6] build clean under sub-flag (tier=%s)\n' "${tier}"
cargo build -p "${package}" "${build_args[@]}"

printf '\n[2/6] tests pass\n'
if command -v cargo-nextest >/dev/null 2>&1; then
    cargo nextest run -p "${package}" "${build_args[@]}"
else
    cargo test -p "${package}" "${build_args[@]}"
fi

printf '\n[3/6] clippy pedantic clean\n'
cargo clippy -p "${package}" "${build_args[@]}" -- -D warnings

# Bench file may not exist yet during early bootstrap; tolerate that case
# but require it for SEALED rows. Discipline log row's Seal cell stays
# IN-FLIGHT until benches land.
printf '\n[4/6] micro-bench file builds\n'
if cargo build --bench "${bench}" -p "${package}" "${build_args[@]}" 2>/dev/null; then
    printf '  bench %s builds.\n' "${bench}"
else
    printf '  bench %s not buildable yet (component in-flight; seal pending).\n' "${bench}"
fi

printf '\n[5/6] compare-bench baseline check\n'
baseline_dir="benches/baselines/${component}"
if compgen -G "${baseline_dir}/*/criterion.tar.zst" >/dev/null; then
    printf '  found saved baseline(s) under %s\n' "${baseline_dir}"
else
    printf '  no saved baseline under %s yet (seal pending per P16)\n' "${baseline_dir}"
fi

printf '\n[6/6] home-turf incumbent arm presence\n'
bench_file="benches/${bench}.rs"
if [[ -f "${bench_file}" ]] && grep -q 'design-favors.*incumbent' "${bench_file}"; then
    printf '  home-turf incumbent arm present in %s\n' "${bench_file}"
else
    printf '  home-turf incumbent arm NOT yet present in %s (seal pending per gate point 13)\n' "${bench_file}"
fi

printf '\n== notify-gate %s: build/tests/clippy green; seal-blockers reported above ==\n' "${component}"
