#!/usr/bin/env bash
# disciplined-component gate: build / test / clippy under a component sub-flag,
# plus the default build to prove the flag is a real firewall.
set -euo pipefail

flag="${1:-scheduler}"

echo "== default build (flag off, firewall intact) =="
cargo build --quiet

echo "== build --features $flag =="
cargo build --quiet --features "$flag"

echo "== test --features $flag =="
cargo test --quiet --features "$flag"

echo "== clippy --all-targets --features $flag -- -D warnings =="
cargo clippy --quiet --all-targets --features "$flag" -- -D warnings

echo "== fmt check =="
cargo fmt --check

echo "gate green for: $flag"
