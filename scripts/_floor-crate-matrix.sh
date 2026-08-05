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
    # proxima-primitives' FINEST tier is stricter than its `alloc` cell above:
    # pipe/mod.rs declares batch_source, capabilities, capture_surface,
    # drain_sink, drain_source, ext, fan_in, header_name, method, primitives,
    # resilience, retry_rules, sink_front, stream_bridge, upgrade and when with
    # NO cfg at all (lines 30-137), so the whole root form family compiles with
    # nothing turned on. Proving only `alloc` left the no-alloc half of every
    # one of those ungated -- the same shape as the proxima-core cell below.
    "proxima-primitives-bare-no-alloc|proxima-primitives|"
    "proxima-net|proxima-net|alloc"
    "proxima-runtime|proxima-runtime|alloc"
    # proxima-runtime's FINEST tier is stricter than its `alloc` cell above:
    # lib.rs gates ext, primitives, SpawnRequest, BackgroundPool and the
    # Runtime trait itself on `alloc`, but `CoreId` and `SpawnError` (the
    # cross-core dispatch vocabulary every caller matches on) carry no cfg at
    # all. Proving only `alloc` left that pair -- and its thiserror-derived
    # `core::error::Error` impl -- unproven on the cliff.
    "proxima-runtime-bare-no-alloc|proxima-runtime|"
    "proxima-core|proxima-core|alloc"
    # proxima-core's FINEST tier is stricter than its `alloc` cell above: arch,
    # datagram_batch, factory::Named, markers, per_core, ring and time all
    # compile with nothing turned on at all. Proving only `alloc` left the
    # no-alloc half of every one of those two-tier modules ungated.
    "proxima-core-bare-no-alloc|proxima-core|"
    # proxima-telemetry claims a no_std + no-alloc floor in lib.rs:17-19
    # (`sized` + `id` + `error` + `trace::{status,kind}`) and had NO cell here
    # at all until the 2026-08-04 consistency pass. Only the BARE cell is
    # honest: `--features alloc` does not reach the cliff, because `fastrand`'s
    # default `std` feature fails first and proxima-primitives' parking_lot
    # second. Adding an `alloc` cell would be a red gate, not a proof.
    "proxima-telemetry-bare-no-alloc|proxima-telemetry|"
    "proxima-protocols|proxima-protocols|tcp,mqtt,amqp,kafka,memcached,nvme,inet,pgwire_codec,process,jsonrpc,websocket_frame,proxy_protocol,redis,hpack,http1_codec,http2_codec,http3_codec-alloc,json_framing,quic-alloc,dns,grpc_framing,protobuf_wire,websocket_handshake,codec-pipe"
    # proxima-codec's own Cargo.toml calls `alloc` its tier-1 floor and four
    # crates consume it there (proxima-kafka, proxima-protocols' -codec-trait
    # features, proxima-listen's framed-any, and the root crate's bench
    # dev-dependency) -- but it was absent from this matrix, so neither gate
    # ever proved the claim. Everything reachable at this tier (the codec
    # traits, Addressed, FrameLimits, LengthDelimitedCodec, BytesPassthrough)
    # is `Vec`-and-slices; JsonCodec + the registry are std-gated above it.
    "proxima-codec|proxima-codec|alloc"
    # proxima-clock has no `alloc` feature at all -- every cell above proves
    # the no_std + alloc tier (`--features alloc`); this one proves the
    # strictly stricter no_std + NO-alloc floor (`--features ""`, i.e. bare
    # `--no-default-features` with nothing turned on). Ticks/UnixNanos/
    # AnchorCell/TickCell/ToUnixNanos never touch `alloc::`/`Box`/`Vec`/
    # `Arc`/`String`; only the optional `config` feature (std-only,
    # conflaguration/bon/serde) is excluded here by construction, matching
    # this array's own contract of the FLOOR tier, not every tier.
    "proxima-clock-bare-no-alloc|proxima-clock|"
    # proxima-config's lib.rs opens by claiming `ConfigFormatFactory`,
    # `DynConfigFormatFactory` and `JsonConfigFormat` hold at no_std + alloc,
    # and its `sugar` / `schema` / `store` features each declare `alloc` as
    # their floor too -- and NONE of it was in this matrix, so no gate had ever
    # compiled the crate for an embedded target. Bare `--features ""` is
    # excluded on purpose: with nothing on, every public item is cfg'd out and
    # the cell would be an empty crate proving nothing.
    "proxima-config|proxima-config|alloc"
    # the three folded-in modules are default-OFF, so the `alloc` cell above
    # does not reach any of them. Each names `alloc` as its own floor; this is
    # where that is checked. `schema` drags regex/time/url on, all three pinned
    # in the manifest with `default-features = false` precisely to survive here.
    "proxima-config-sugar|proxima-config|alloc,sugar"
    "proxima-config-schema|proxima-config|alloc,schema"
    "proxima-config-store|proxima-config|alloc,store"
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
