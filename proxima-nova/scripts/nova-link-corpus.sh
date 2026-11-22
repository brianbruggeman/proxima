#!/usr/bin/env bash
# c13's gate: link the fixtures with nova's own linker, no platform-linker process.
# prints: linked=N of=N spawned=0 exits=N of=N codesign=N
# control: NOVA_LINK_ARM=system links through the incumbent path and prints spawned>=1
# RED until the L1 card delivers proxima-link; it prints linked=0 of=1 and exits 1 today.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$root/.." && pwd)"
arm="${NOVA_LINK_ARM:-nova}"
objdir="${NOVA_LINK_OBJ_DIR:-$HOME/.cache/proxima-nova/PROXIMANOVA-55}"
outdir="${NOVA_LINK_OUT:-}"
mkdir -p "$objdir"
fixtures=(ret42)
n=${#fixtures[@]}

linked=0
spawned=0
exits=0
codesign=0

work="$(mktemp -d "$HOME/.cache/proxima-nova/link-corpus.XXXXXX")"
trap 'rm -rf "$work"' EXIT
[[ -n "$outdir" ]] || outdir="$work"
mkdir -p "$outdir"
: >"$work/spawned"

# recording shims that DELEGATE, so a platform-linker exec is observed and the build still
# works. The real path is resolved before the shim directory enters PATH, so no recursion.
for tool in cc ld ld64; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [[ -n "$real" ]] || continue
    printf '#!/bin/sh\nprintf "%%s\\n" "$0 $*" >> "%s/spawned"\nexec "%s" "$@"\n' "$work" "$real" >"$work/$tool"
    chmod +x "$work/$tool"
done

if [[ "$arm" == "system" ]]; then
    # the incumbent: rustc with the nova backend, linked by cc, traced through the shims
    if PATH="$work:$PATH" "$repo/proxima-nova-rustc/scripts/ret.sh" >"$work/ret.log" 2>&1; then
        linked=1
        exits=1
    fi
    spawned=$(wc -l <"$work/spawned" | tr -d ' ')
    echo "linked=$linked of=$n spawned=$spawned exits=$exits of=$n codesign=0"
    [[ "$linked" -eq "$n" && "$spawned" -ge 1 && "$exits" -eq "$n" ]]
    exit $?
fi

link="${NOVA_LINK:-$repo/target/release/proxima-link}"
if [[ ! -x "$link" ]]; then
    echo "linked=0 of=$n spawned=0 exits=0 of=$n codesign=0"
    exit 1
fi

for stem in "${fixtures[@]}"; do
    obj="$objdir/$stem.o"
    [[ -f "$obj" ]] || continue
    out="$outdir/$stem.bin"
    PATH="$work:$PATH" "$link" "$obj" -o "$out" || continue
    linked=$((linked + 1))
    "$out"
    [[ $? -eq 42 ]] && exits=$((exits + 1))
    codesign -v "$out" >/dev/null 2>&1 && codesign=$((codesign + 1))
done

spawned=$(wc -l <"$work/spawned" | tr -d ' ')
echo "linked=$linked of=$n spawned=$spawned exits=$exits of=$n codesign=$codesign"
[[ "$linked" -eq "$n" && "$spawned" -eq 0 && "$exits" -eq "$n" && "$codesign" -eq "$n" ]]
