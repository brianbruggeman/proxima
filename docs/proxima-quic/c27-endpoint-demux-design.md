# C27 — Endpoint demux + datagram classification

The endpoint-level primitive that sits between the UDP datagram
source (Phase B) and the per-connection state machine (C11). Given
an inbound datagram, decides:

- which connection (if any) owns it — DCID table lookup;
- whether it's a new server-side connection attempt;
- whether the wire-format version is supported (→ trigger
  Version Negotiation reply);
- whether the datagram should be dropped.

Tagged per principle 13: `/algorithm-development` (the classification
state machine has 6 outcomes with strict ordering — version check
before CID lookup per RFC 9000 §6; CID-table lookup before
new-connection decision). No crypto material — `/security-review`
not invoked.

## Scope split

| Slice | Scope | Lands here |
|---|---|---|
| **C27.0** (this row) | `EndpointDemux<CAP>` table primitive — register/unregister/lookup connection IDs + `classify_datagram` returning a typed `DatagramClassification` enum. **Server accept policy + Retry-issue policy + actual connection construction defer to Phase B.** | tier-3 v1 |
| C27.1 | Server-side new-connection accept policy (retry-vs-accept decision, anti-DoS rate limits, address-validation token check via C19.1) | defers to Phase B / I/O facade |
| C27.2 | Stateless-reset token table (RFC 9000 §10.3) + per-connection stateless-reset key derivation | defers; tied to C8 CID issuance |

## C27.0 — DCID table + classification

### Data structure

```rust
pub struct EndpointDemux<const CAP: usize> {
    /// DCID → connection-handle table. Caller-opaque handle
    /// (u32 index into the caller's slab of connection structs).
    table: heapless::LinearMap<ConnectionIdBytes, ConnectionHandle, CAP>,
    /// Locally-supported wire-format versions, in preference order.
    /// First entry is our preferred version; subsequent are accepted.
    supported_versions: &'static [u32],
}

pub struct ConnectionHandle(pub u32);
```

LinearMap is O(N) lookup. Per principle-12 sizing the default cap
is small (`endpoint.dcid_table_cap = 64`), so cache-locality wins
over hash-map overhead. Production-scale endpoints (10k+
connections) supply their own DCID-table implementation via the
facade layer — C27.0 is the sans-IO primitive only.

### Classification

```rust
pub enum DatagramClassification<'a> {
    /// DCID found in table; route to that connection.
    Existing {
        handle: ConnectionHandle,
        first_byte_form: Form,    // Short vs Long
    },
    /// Long-header Initial with unknown DCID — server should
    /// accept a new connection (or trigger Retry per its
    /// address-validation policy in C27.1).
    NewInitial {
        dcid: &'a [u8],
        scid: &'a [u8],
        token: &'a [u8],          // RFC 9000 §17.2.5 Token field
        version: u32,
    },
    /// Long-header with a version we don't support — server
    /// should emit a Version Negotiation packet.
    UnsupportedVersion {
        dcid: &'a [u8],
        scid: &'a [u8],
        peer_version: u32,
    },
    /// Long-header non-Initial with unknown DCID, or short-header
    /// with unknown DCID, or malformed bytes — drop silently per
    /// RFC 9000 §10.3.
    Drop {
        reason: DropReason,
    },
}

pub enum DropReason {
    /// Header parse failed (truncation / fixed-bit / etc).
    MalformedHeader,
    /// Long-header Handshake / 0-RTT / Retry with unknown DCID.
    UnknownDcidLongHeader,
    /// Short-header with unknown DCID (RFC 9000 §10.3 — stateless
    /// reset is the proper response; this primitive just signals
    /// "drop" and leaves the SR decision to the caller).
    UnknownDcidShortHeader,
}
```

### Classification algorithm

Per RFC 9000 §17 (Packet Headers) + §5 (Connections) + §6 (Version
Negotiation):

```
classify(datagram):
  if datagram.is_empty(): return Drop(MalformedHeader)
  form = peek_form(datagram)
  if form == Long:
    if parse_long fails: return Drop(MalformedHeader)
    header = parse_long(datagram)
    # Version 0 = VersionNegotiation packet from a server we're a
    # client of — but endpoints in the SERVER role never receive
    # these. The classifier doesn't try to discriminate role here;
    # if a VN reaches us as a server we treat it as MalformedHeader
    # (RFC §6 says we MUST discard).
    if header is VersionNegotiation:
      return Drop(MalformedHeader)
    version = header.version
    if version not in supported_versions:
      return UnsupportedVersion { dcid, scid, peer_version: version }
    dcid = header.dcid
    if table.lookup(dcid):
      return Existing { handle, first_byte_form: Long }
    if header is Initial:
      return NewInitial { dcid, scid, token, version }
    # Long-header non-Initial with unknown DCID → drop.
    return Drop(UnknownDcidLongHeader)
  else:  # Short-header
    # Short-header has no version; DCID length is connection-specific
    # (caller-known). We delegate the SCID-len-aware parse to the
    # caller and only do a best-effort table scan: try lookup against
    # every key in the table as a prefix match.
    handle = table.lookup_short_header_dcid(datagram)
    if handle:
      return Existing { handle, first_byte_form: Short }
    return Drop(UnknownDcidShortHeader)
```

### Short-header DCID handling — design note

Short-header packets carry the DCID WITHOUT a length prefix; the
receiver must know the expected DCID length out-of-band (per
RFC 9000 §17.3). The endpoint demux therefore can't parse a
short-header packet from raw bytes alone — it must scan the table.

C27.0 implements this as: for each registered DCID, attempt
prefix-match against `datagram[1..]`. If any match (longest-first),
return Existing. Otherwise Drop.

For low-CAP tables this is O(CAP × max-CID-len) = O(64 × 20) =
1280 bytes scanned per packet — well within cache. For production-
scale this is an unacceptable hot-path cost; production deployments
implement their own DCID table keyed by hash + length-tagged entry.
Documented limitation.

## Worked example (multi-classification trace)

State at t=T0: empty EndpointDemux<8> with supported_versions=[1].

| step | inbound | Action | Outcome |
|------|---------|--------|---------|
| 1    | empty   | `classify(&[])` | `Drop(MalformedHeader)` |
| 2    | long-header v=0 (VN as if from server) | `classify(&[0xc0, 0,0,0,0, 0,0])` | `Drop(MalformedHeader)` (we're not a client; VN is unexpected) |
| 3    | long-header v=99 Initial with new DCID `[0xAA]` | `classify(...)` | `UnsupportedVersion { peer_version: 99, dcid: [0xAA], scid: ... }` |
| 4    | long-header v=1 Initial with new DCID `[0xAB; 8]` | `classify(...)` | `NewInitial { dcid: [0xAB; 8], scid: [...], token: [], version: 1 }` |
| 5    | `register(DCID=[0xAB; 8], handle=42)` then long-header v=1 Initial with same DCID | `classify(...)` | `Existing { handle: 42, first_byte_form: Long }` |
| 6    | short-header with byte 0 = `0x40` (Short, fixed=1) + DCID=[0xAB; 8] + payload | `classify(...)` | `Existing { handle: 42, first_byte_form: Short }` |
| 7    | short-header with unknown DCID prefix | `classify(...)` | `Drop(UnknownDcidShortHeader)` |
| 8    | `unregister(DCID=[0xAB; 8])` then same short-header as step 6 | `classify(...)` | `Drop(UnknownDcidShortHeader)` |

## Security review (per principle 13)

| Concern | Mitigation |
|---|---|
| Connection-ID enumeration attack (attacker probes random DCIDs to map active connections) | Short-header unknown-DCID returns Drop; caller MAY upgrade to Stateless Reset per RFC §10.3 (not this primitive's concern) |
| Long-header amplification (attacker spoofs source addr + sends valid Initial with new DCID to elicit large server response) | NewInitial outcome surfaces the classification; server's anti-amplification (C20) caps response volume; Retry token (C19.1) enforces address validation |
| DCID-table overflow (attacker creates many connections to fill the table) | TableFull error; caller decides accept policy (refuse new connections, evict idle entries, raise CAP). Anti-DoS lives at the I/O layer |
| Cross-connection DCID collision | DuplicatePathId-style DuplicateDcid error on register |
| Version-downgrade attack via spoofed inbound | Version check is observe-only; classifier surfaces the requested version; CC layer's version negotiation per RFC 9000 §6 enforces the real check |

## Tier

C27.0 is tier-3 (heapless::LinearMap + POD enum results; no alloc).

## Per principle 14 (incumbent wins)

Classification rules taken verbatim from RFC 9000:
- §17 (header forms) → form discrimination + per-form parse routing
- §6 (Version Negotiation) → version check before further processing
- §10.3 (Stateless Reset) → short-header unknown-DCID drop policy
- §5.1.1 (CID length cap) → DCID parsing reuses C2's parser

No invention. Outcomes are direct translations of RFC-mandated
behaviors.

## Sized constants (principle 12)

```toml
[endpoint]
# Per-endpoint cap on tracked DCIDs in the demux table. Low for
# embedded, high for server deployments. Default 64 — covers
# typical small-deployment workloads; production tune via env.
dcid_table_cap = 64
```

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes (8-row worked example).
- **Pass 3 — code maps step-by-step**: planned.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; skipping the version check before CID lookup, returning Existing for a short-header with mismatched DCID prefix, or accepting a new Initial on the same DCID twice would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
