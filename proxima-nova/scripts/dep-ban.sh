#!/usr/bin/env bash
# AC6: no cranelift/llvm/inkwell/melior package in the nova or backend dependency trees
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
toolchain="nightly-2026-11-15"
cache="$HOME/.cache/proxima-nova/PROXIMANOVA-8"
mkdir -p "$cache"
banned='^(cranelift|llvm|inkwell|melior)[-_ ]'

cd "$root"
git wt cargo tree -p proxima-nova --prefix none >"$cache/tree-nova.log" 2>&1 || { echo "tree failed: $cache/tree-nova.log" >&2; exit 1; }
cargo "+$toolchain" tree --manifest-path proxima-nova-rustc/Cargo.toml --prefix none >"$cache/tree-rustc.log" 2>&1 || { echo "tree failed: $cache/tree-rustc.log" >&2; exit 1; }

banned_nova="$(grep -cE "$banned" "$cache/tree-nova.log" || true)"
banned_rustc="$(grep -cE "$banned" "$cache/tree-rustc.log" || true)"
control="$(echo 'cranelift-codegen v0.120.0' | grep -cE "$banned" || true)"
echo "banned_nova=$banned_nova banned_rustc=$banned_rustc control=$control"
