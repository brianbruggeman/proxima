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
    # proxima-net's FINEST tier is stricter than its `alloc` cell above: lib.rs
    # gates packet on `std` and tcp_listener/tcp_stack on `alloc`, but `stack`
    # (the sans-IO ARP/ICMP responder, lib.rs:29) carries NO cfg at all and the
    # crate doc calls it "the crate's bare no_std no-alloc floor". Proving only
    # `alloc` left that floor unproven -- same shape as the primitives/runtime/
    # core cells around it.
    "proxima-net-bare-no-alloc|proxima-net|"
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
    # the cell above turns on each protocol's BASE feature and `codec-pipe`
    # (the generic adapter), but none of the `-codec-trait` / `-frame-pipe` /
    # `-session` / `-alloc` features -- which is where this crate's tier claims
    # actually live: `dns-codec-trait`'s heapless no-alloc `DnsTcpQuery::name`
    # rung, `websocket_frame-session`'s "Tier-1: no_std + alloc", and every
    # per-codec `OwnFrame`/`Incomplete` impl the adapter composes against. None
    # of it had ever been compiled for an embedded target. Measured 2026-08-04:
    # it builds clean, so this cell locks in a claim that was true but unproven.
    "proxima-protocols-codec-seams|proxima-protocols|dns-codec-trait,memcached-codec-trait,kafka-codec-trait,redis-codec-trait,json_framing-codec-trait,grpc_framing-codec-trait,grpc_framing-frame-pipe,protobuf_wire-codec-trait,websocket_frame-codec-trait,websocket_frame-frame-pipe,websocket_frame-session,http1_codec-alloc,http1_codec-codec-trait,http1_codec-frame-pipe,http2_codec-alloc,http2_codec-codec-trait,http3_codec-codec-trait,http3_codec-part-source,hpack-alloc,hpack-codec-trait,proxy_protocol-codec-trait,quic-codec-trait,quic-mock-tls"
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
    # proxima-patterns' Cargo.toml has documented an `alert` tier-3 cell
    # ("#![no_std], no per-call alloc, heapless containers") since the fold,
    # and its `kv` / `control_plane` patterns each name `alloc` as their floor
    # — none of it was in this matrix, so the claim had never been compiled for
    # an embedded target. It did not hold: `ulid`'s manifest leaves
    # default-features on its own serde edge, so `features = ["serde"]` turned
    # `serde/std` on for the whole build. Fixed by pinning ulid locally with
    # `default-features = false` and hand-rolling the identical canonical-string
    # wire form (`alert::event::ulid_str`). No bare `--features ""` cell: lib.rs
    # cfg's out every module, so it would compile an empty crate.
    "proxima-patterns-alert-proto|proxima-patterns|alert,proto"
    "proxima-patterns-kv|proxima-patterns|kv"
    "proxima-patterns-control-plane|proxima-patterns|control_plane"
    # proxima-storage's pmem module is declared with NO cfg at all
    # (proxima-storage/src/lib.rs) and its module doc calls itself "Tier-3:
    # compiles under #![no_std] with no allocator" -- and the crate had no cell
    # here at all until the 2026-08-05 consistency pass, so nothing had ever
    # compiled it for an embedded target. The bare cell is the whole no-alloc
    # leaf: the core::arch cache-maintenance primitives plus the CoW
    # atomic-root-swap FSM over a borrowed &mut [u8]. Measured 2026-08-05: it
    # builds clean, so this locks in a claim that was true but unproven. No
    # `alloc` cell exists because the crate has no `alloc` feature -- std is the
    # only rung above bare.
    "proxima-storage-bare-no-alloc|proxima-storage|"
    # the sibling tier-3 claim (proxima-storage/src/nvme/mod.rs: "the engine is
    # #![no_std] + no-alloc"): the queue-pair engine over the sans-IO
    # proxima-protocols::nvme codec, ring cursors in atomics, Pipe + SendPipe.
    # `nvme` does not imply `std`, so this cell is a real second floor, not a
    # superset of the one above.
    "proxima-storage-nvme|proxima-storage|nvme"
    # proxima-listen/src/lib.rs:5 opens by claiming a "Base (no_std +
    # no_alloc)" tier -- the `admission` FSM (ungated) plus the `preface`
    # classifier (ungated) -- and the crate had no cell here at all until the
    # 2026-08-05 consistency pass, so nothing had ever compiled it for an
    # embedded target. It is not enough to run the crate's own
    # `--no-default-features` command on the host: proxima-listen dev-depends
    # on the `proxima` umbrella, so host feature unification turns `std` back
    # on and 122 tests still run with no std tier exercised at all. Only the
    # cliff proves the claim. Measured 2026-08-05: both cells build clean.
    "proxima-listen-bare-no-alloc|proxima-listen|"
    # the alloc rung of the same two modules: `alloc` swaps the admission
    # table's fixed-cap heapless::FnvIndexMap for a growable hashbrown::HashMap
    # (proxima-listen/src/admission/state.rs), so it is a genuinely different
    # compile, not a superset that the bare cell already covered.
    "proxima-listen-alloc|proxima-listen|alloc"
    # proxima-redis' Cargo.toml has advertised "`--no-default-features` leaves
    # the bare sans-IO codec ... for bare-metal embedding" since the fold, and
    # the crate had no cell here, so nothing ever compiled it for an embedded
    # target. The claim was FALSE: lib.rs never declared `#![no_std]`, so the
    # bare build still linked libstd and rustc rejected the cliff outright
    # ("`std` is required by `proxima_redis` because it does not declare
    # `#![no_std]`"). Fixed in the 2026-08-05 consistency pass; this cell is
    # what keeps it fixed. No `alloc` cell: the crate has no `alloc` feature --
    # `client` and `listen` both imply `std`, so bare is the only floor.
    "proxima-redis-bare|proxima-redis|"
    # proxima-dns carried the identical defect its redis sibling above did,
    # and for the identical reason: same `default = ["std"]` / `std = []`
    # shape, no `#![no_std]` in lib.rs, no cell here, so nothing ever tried
    # the cliff. Measured before the fix on 2026-08-05: `cargo build -p
    # proxima-dns --no-default-features --lib --target thumbv7m-none-eabi`
    # exits 101 with "`std` is required by `proxima_dns` because it does not
    # declare `#![no_std]`". Same shape, same single floor: `client` and
    # `listen` both imply `std`, so there is no `alloc` cell to add.
    "proxima-dns-bare|proxima-dns|"
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
