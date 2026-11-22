#!/usr/bin/env bash
# c14's gate: one op table drives omega and nova.
# prints: ops_shared=N of=1 results=N of=4 definitions=N of=1
# RED until the O2 card lands the table; it prints zeros and exits 1 today.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$root/.." && pwd)"
table="${NOVA_OPS_TABLE:-$repo/proxima-numeric/src/op_table.rs}"
ops=1
shared=0
results=0
definitions=0

if [[ -f "$table" ]]; then
    # exactly one definition of the shared entry point across every consumer
    defs=$(grep -rnE --include='*.rs' 'pub fn shared_add\b' "$repo" \
        --exclude-dir=target --exclude-dir=.git 2>/dev/null | wc -l | tr -d ' ')
    [[ "$defs" -eq 1 ]] && definitions=1

    # each consumer's own arm body must call the table, not merely import it
    work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
    awk '/ScalarOp::Add =>/{f=1} f{print; if(++n>=12) exit}' \
        "$repo/proxima-tensor/src/cpu/typed_eval.rs" >"$work/omega"
    awk '/Inst::Iadd/{f=1} f{print; if(++n>=12) exit}' \
        "$repo/proxima-nova/src/aarch64/lower.rs" >"$work/nova"
    omega_calls=$(grep -c 'shared_add' "$work/omega" || true)
    nova_calls=$(grep -c 'shared_add' "$work/nova" || true)
    [[ "$omega_calls" -ge 1 && "$nova_calls" -ge 1 ]] && shared=1

    # two tests per consumer, selected by the one module filter
    for pkg in proxima-tensor proxima-nova; do
        log="$HOME/.cache/proxima-nova/omega-ops.$pkg.log"
        if cargo val -p "$pkg" op_table::shared_add >"$log" 2>&1; then
            passed=$(grep -oE 'passed=[0-9]+' "$log" | head -1 | cut -d= -f2)
            results=$((results + ${passed:-0}))
        fi
    done
fi

echo "ops_shared=$shared of=$ops results=$results of=4 definitions=$definitions of=1"
[[ "$shared" -eq "$ops" && "$results" -eq 4 && "$definitions" -eq 1 ]]
