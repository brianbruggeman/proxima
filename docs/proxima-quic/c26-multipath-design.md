# C26 — Multipath QUIC path-tracking (paper proof + state machine)

Per [draft-ietf-quic-multipath-21] (latest as of 2026-05-25; pinned in
[`rfc-reference.md`](./rfc-reference.md)). Specifies an extension that
lets a single QUIC connection use multiple network paths concurrently.

Tagged per principle 13: `/algorithm-development` (the per-path state
machine has at least 4 status transitions + a per-path closure
sequence), `/research-rigor` (path scheduling policy — when to use
which path — is explicitly out of scope of the draft and is its own
contested design space).

[draft-ietf-quic-multipath-21]: https://www.ietf.org/archive/id/draft-ietf-quic-multipath-21.txt

## Pinned revision

This implementation tracks `draft-ietf-quic-multipath-21` (dated 17
March 2026). Wire-format frames + state transitions follow that
revision verbatim. **If the draft moves** (it has revised ~21 times
to date) the version pin must be re-cut + a discipline-log changelog
row added documenting which fields/transitions changed.

## Scope split

C26 is decomposed into focused slices:

| Slice | Scope | Lands here |
|---|---|---|
| **C26.0** (this row) | Per-path state machine primitive + path-table + status transitions. **Wire-format frames + scheduler policy defer.** | tier-1 v1 |
| C26.1 | Wire-format frames (PATH_ACK, PATH_ABANDON, PATH_STATUS_*, PATH_NEW_CONNECTION_ID, PATH_RETIRE_CONNECTION_ID, MAX_PATH_ID, PATHS_BLOCKED, PATH_CIDS_BLOCKED) | defers until paired with a multipath-capable test peer for parity (principle 14) |
| C26.2 | Per-path packet-number space + nonce derivation per §2.4 | defers; tied to 1-RTT egress slice |
| C26.3 | `initial_max_path_id` transport parameter + negotiation | defers; tied to C4 transport-parameter codec extension |
| C26.4 | Path scheduling policy (round-robin / minRTT / redundant) | defers; explicit non-goal of the draft itself ("this document does not specify ... how multiple open paths are used for sending") |

## C26.0 — Per-path state primitive

### State machine

Per §3.1 (path initiation), §3.3 (status management), §3.4 (closure):

```
                    PATH_CHALLENGE issued, PATH_RESPONSE pending
                                  │
                                  ▼
                          ┌───────────────┐
                          │   Validating  │
                          └───────┬───────┘
                                  │ PATH_RESPONSE matches token
                                  ▼
            ┌──────────  ┌───────────────┐  ──────────┐
            │   peer's   │   Available   │  peer's    │
            │ PATH_STATUS│               │ PATH_STATUS│
            │ _BACKUP    │               │ _AVAILABLE │
            ▼            └───────┬───────┘            ▲
    ┌───────────────┐            │ PATH_ABANDON       │
    │     Backup    │◀───────────┴──────┐ rcvd/sent   │
    └───────┬───────┘                   │             │
            │   PATH_ABANDON rcvd/sent  │             │
            │                           ▼             │
            │                   ┌───────────────┐     │
            └──────────────────▶│    Closing    │     │
                                └───────┬───────┘     │
                                        │ drain done  │
                                        │ (3 × PTO)   │
                                        ▼             │
                                ┌───────────────┐     │
                                │   Abandoned   │─────┘ (terminal;
                                └───────────────┘       remove from table)
```

Notes:

- `Validating` is the entry state for any path other than the
  connection's initial path. The initial path is created in
  `Available` directly (it inherits the handshake's path validation).
- `Available ↔ Backup` is bidirectional via PATH_STATUS frames (§3.3).
- `Closing → Abandoned` is the only terminal transition. Once
  `Abandoned`, the path is removed from the table.
- `Abandoned` paths MUST NOT be re-used (RFC §3.4 / draft §3.4).

### Per-path state

```rust
pub struct MultipathPath {
    pub path_id: PathId,                      // newtype over u32 (§2.1 cap = 2^32-1)
    pub status: PathStatus,
    pub local_cid: ConnectionIdBytes,         // CID we use as src for this path
    pub remote_cid: ConnectionIdBytes,        // CID peer uses as src for this path
    pub last_active: Instant,                 // for drain timeout + idle tracking
    pub close_deadline: Option<Instant>,      // set when Closing
}
```

### Path table

`MultipathTable<const CAP: usize>` wraps `heapless::LinearMap<PathId, MultipathPath, CAP>`.
LinearMap is O(N) lookup; for low CAP (≤16) this is faster than
FnvIndexMap due to cache-locality. Discipline-log row honestly
documents the trade-off: the draft itself recommends low active path
counts (§2.1 — "the initial_max_path_id parameter [...] limits the
initial maximum number of open paths") so the LinearMap default is
the right starting point.

Sized constant `multipath.max_paths_per_connection` (default 4)
controls CAP. Env override `PROXIMA_QUIC_PROTO_MULTIPATH_MAX_PATHS_PER_CONNECTION`.

## Worked example (multipath path lifecycle)

Two-path scenario: client establishes a primary path P0 during the
handshake, then opens an alternate P1 over a different network.
Server eventually drops P0; client closes it; P1 takes over.

State at t=T0: handshake just confirmed. Path P0 = Available (CIDs
local=L0 / remote=R0). P1 not in table.

| t  | Event | Action | State after |
|----|-------|--------|-------------|
| T1 | Client opens P1 (issues PATH_CHALLENGE on a new 4-tuple) | `table.register_path(P1, local=L1, remote=R1, Validating)` | P0=Available, P1=Validating |
| T2 | PATH_RESPONSE matching P1's token arrives | `table.note_path_validated(P1)` → Validating → Available | P0=Available, P1=Available |
| T3 | Server sends PATH_STATUS_BACKUP for P0 | `table.set_remote_status_preference(P0, Backup)` → Available → Backup | P0=Backup, P1=Available |
| T4 | P0 connectivity breaks (no PATH_RESPONSE to our liveness probes) | `table.note_breakage(P0, now=T4)` → Backup → Closing; close_deadline = T4 + 3×PTO | P0=Closing, P1=Available |
| T5 | drain timer fires at T4 + 3×PTO | `table.tick(now=T5)` → Closing → Abandoned → removed | P1=Available (P0 gone) |

Every transition above maps to one named method on `MultipathTable`;
illegal transitions return typed `MultipathError`.

## Security review (per principle 13)

| Concern | Mitigation |
|---|---|
| Path-injection (attacker forges PATH_RESPONSE) | Path validation (C21) uses random 8-byte tokens; PATH_RESPONSE must match a recently-issued PATH_CHALLENGE token byte-for-byte |
| Path-status-flap DoS (peer flips Available↔Backup endlessly) | Per-status sequence number per draft §4.3 — old PATH_STATUS frames are silently discarded |
| Abandoned-path resurrection | Terminal Abandoned → removed from table; subsequent PATH_NEW_CONNECTION_ID for an abandoned PathId rejected as PROTOCOL_VIOLATION |
| CID exhaustion via path churn | Existing CidQueue cap (C8) bounds outstanding CIDs; PATH_CIDS_BLOCKED frame surfaces back-pressure |
| Nonce-collision across paths (§2.4) | Per-path PN space + path_id-mixed nonce — defers to C26.2 |

## Tier

C26.0 is tier-3 (POD path state + heapless::LinearMap). No alloc.

## Per principle 14 (incumbent wins)

Path-state nomenclature (Validating / Available / Backup / Closing /
Abandoned) and transition triggers (PATH_RESPONSE / PATH_STATUS_*
frames / PATH_ABANDON) are taken verbatim from draft-21 sections
3.1, 3.3, 3.4. No invention. The state-machine diagram above is a
restatement of those sections in graphical form for unambiguous code
mapping.

Wire-format frames (C26.1) defer until paired with a multipath-capable
test peer (picoquic or s2n-quic with multipath enabled) for
bit-exact parity — RFC §17.2.5-style "incumbent or proven-bug" gate.

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes (5-row example above).
- **Pass 3 — code maps step-by-step**: planned for the impl below.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; transitioning from Validating without a PATH_RESPONSE match, skipping the close_deadline, or re-using an Abandoned PathId would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
