#!/usr/bin/env bash
# runs the mechanical part of the protocol-gap Definition of Done for one
# component (sibling of component-gate.sh which handles runtime-prime).
# usage: scripts/protocol-gap-gate.sh <component>
#   <component>: pool | ws-upstream | redis | dns | h3-upstream | grpc-upstream | memcached | mqtt | amqp | kafka
#
# enforces:
#   - build clean under sub-flag (default-features off, only this sub-flag on)
#   - unit tests pass for that component's module path
#   - clippy pedantic clean
#   - the matching micro-bench compiles
#
# what this script does NOT enforce (you do these by hand and record in
# docs/protocol-gap/discipline.md):
#   - compare-bench numbers vs named alternatives (column 6)
#   - E2E bench arm (column 7)
#   - optimization sweep (column 8)
#   - SIMD / state-machine / no-Box / discriminated-enum pass (column 10)
#   - strict O(1) audit (column 11)

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <pool|ws-upstream|redis|dns|h3-upstream|grpc-upstream|memcached|mqtt|amqp|kafka>\n' "$0" >&2
  exit 64
fi

component="$1"

# component -> (feature flag, bench file name, module path)
case "$component" in
  pool)          feature="upstream-pool-tuning";   bench="bench_pool_tuning";  module="upstreams::pool" ;;
  ws-upstream)   feature="websocket-upstream";     bench="bench_ws_upstream";  module="upstreams::websocket" ;;
  redis)         feature="redis-listener";         bench="bench_redis";        module="listeners::redis" ;;
  dns)           feature="dns";                    bench="bench_dns";          module="dns" ;;
  h3-upstream)   feature="http3-upstream";         bench="bench_h3_upstream";  module="upstreams::h3" ;;
  grpc-upstream) feature="grpc-upstream";          bench="bench_grpc_upstream"; module="upstreams::grpc" ;;
  memcached)     feature="memcached-listener";     bench="bench_memcached";    module="listeners::memcached" ;;
  mqtt)          feature="mqtt-listener";          bench="bench_mqtt";         module="listeners::mqtt" ;;
  amqp)          feature="amqp-listener";          bench="bench_amqp";         module="listeners::amqp" ;;
  kafka)         feature="kafka-listener";         bench="bench_kafka";        module="listeners::kafka" ;;
  *) printf 'unknown component: %s\n' "$component" >&2; exit 64 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

printf 'protocol-gap-gate: %s\n' "$component"

printf -- '--- build (--no-default-features --features %s) ---\n' "$feature"
cargo build --no-default-features --features "$feature"

printf -- '--- test (-- %s::) ---\n' "$module"
cargo test --no-default-features --features "$feature" -- "$module"::

printf -- '--- clippy ---\n'
cargo clippy --no-default-features --features "$feature" --all-targets -- -D warnings

printf -- '--- bench compiles (%s, --no-run) ---\n' "$bench"
cargo bench --no-default-features --features "$feature" --bench "$bench" --no-run

printf 'PASS (mechanical): %s\n' "$component"
printf 'NEXT: fill in compare-bench / E2E / opt-sweep / SIMD/SM/no-Box / O(1) cells in docs/protocol-gap/discipline.md\n'
