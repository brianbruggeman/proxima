#!/usr/bin/env bash
# _floor-crate-matrix.sh — shared crate/feature matrix for the tokio-free
# floor gate (tokio-free-floor.sh) and the thumbv7m cliff gate
# (thumbv7m-cliff-gate.sh). Source this file; do not execute it directly.
#
# One source of truth (guiding-principle 1: RISC, no forked copies) for
# "which crates claim the no_std + alloc floor tier, and what feature
# string proves it" -- both gates check the exact same matrix from two
# different angles (tokio-free vs thumbv7m-none-eabi compiles).
#
# Per docs/pipe-to-metal/edges.md's "tokio/futures compat-layer sweep"
# scoping row #0 (2026-07-16): prime, proxima-primitives, proxima-net,
# proxima-runtime, proxima-core all expose a single `alloc` Cargo feature
# that is the floor tier (`--no-default-features --features alloc`).
# proxima-protocols has no such umbrella feature (it is per-protocol,
# see its Cargo.toml's own per-module `-alloc`/`-no-alloc` aliases) -- its
# entry below is a representative union of its documented no_std+alloc
# (tier-1) and no_std+no-alloc (tier-3) protocol features, standing in
# for "the crate's actual floor feature set" per this gate's own scope.

# label|crate|features (comma-separated, passed verbatim to --features)
# consumed by the sourcing gate scripts, not this file
# shellcheck disable=SC2034
declare -a FLOOR_CRATE_CELLS=(
    # NOT bare `alloc`: prime/src/core.rs:4-27 gates every submodule except
    # `sized` behind the runtime-prime-* features, so `--features alloc` alone
    # compiles only sized.rs (a 3-line build-script const include) -- a green
    # cell that proved NOTHING about the scheduler (false confidence, caught by
    # task #5's scoping pass, docs/pipe-to-metal/edges.md 2026-07-16). This is
    # the feature set that actually compiles the task table, hand-rolled waker,
    # timer wheel and alloc inbox on the floor, so the cell earns its keep.
    "prime|prime|alloc,runtime-prime-inbox-alloc,runtime-prime-executor,runtime-prime-timer"
    # the alloc-FREE sibling: inbox_const is a stack-backed SPSC inbox that
    # needs no `alloc` feature at all -- the strictest floor cell we have.
    "prime-inbox-const|prime|alloc,runtime-prime-inbox-const,runtime-prime-executor,runtime-prime-timer,runtime-prime-thread-identity"
    "proxima-primitives|proxima-primitives|alloc"
    "proxima-net|proxima-net|alloc"
    "proxima-runtime|proxima-runtime|alloc"
    "proxima-core|proxima-core|alloc"
    "proxima-protocols|proxima-protocols|tcp,mqtt,amqp,kafka,memcached,nvme,inet,pgwire_codec,process,jsonrpc,websocket_frame,proxy_protocol,redis,hpack,http1_codec,http2_codec,http3_codec-alloc,json_framing,quic-alloc,dns,grpc_framing,protobuf_wire,websocket_handshake,codec-pipe"
)

# Cells that are TOKIO-FREE-checkable but NOT thumbv7m-buildable, because they
# carry `std`. Consumed by tokio-free-floor.sh ONLY -- never by
# thumbv7m-cliff-gate.sh: a `std` build cannot compile for thumbv7m-none-eabi
# (there is no libstd for that target), so listing a std cell in
# FLOOR_CRATE_CELLS made the cliff gate fail on it.
#
# Deliberately OUT of FLOOR_CRATE_CELLS: that array's contract is "crates that
# claim the no_std + alloc FLOOR tier", and a std-tier cell is not one. The two
# gates ask different questions -- "is it tokio-free?" is meaningful at both
# tiers; "does it compile bare-metal?" is only meaningful for the floor.
# shellcheck disable=SC2034
declare -a TOKIO_FREE_EXTRA_CELLS=(
    # the realistic "assemble a working executor" combo (docs/pipe-to-metal/
    # edges.md, "prime-tokio-feature-split (task #8 remainder)") -- bare
    # `alloc` never compiles os/primitives.rs's Send RuntimeFactory impl at
    # all, so it could not have caught the C31 tokio leak this cell locks in
    # as fixed. `prime-tokio-compat` stays OFF here on purpose.
    "prime-default-std|prime|std,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool"
)
