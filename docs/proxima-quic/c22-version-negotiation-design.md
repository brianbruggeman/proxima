# C22 — Version negotiation (paper proof)

Per [RFC 9000 §6] + [§17.2.1]. The simplest A5 component: a Version
Negotiation (VN) packet from the server tells the client "I don't
support your version; here's a list of versions I do support."

[RFC 9000 §6]: https://www.rfc-editor.org/rfc/rfc9000#section-6
[§17.2.1]: https://www.rfc-editor.org/rfc/rfc9000#section-17.2.1

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Scope

**C22 v1 client-side**:
- Parse VN packet from inbound datagram in Initial state (the C2
  `Header::VersionNegotiation` parser already does the heavy lifting).
- Extract the list of offered versions.
- If we support any of them, surface to caller via a structured
  error: `ConnectionError::VersionNegotiationRequested { offered: ArrayVec<u32, MAX> }`.
- If we don't support any, surface as
  `ConnectionError::VersionNegotiationFailed { offered: ArrayVec<u32, MAX> }`.
- Caller decides whether to construct a new Connection with one of
  the offered versions OR fail.

**C22 v2 server-side** (defers to when server-side FSM lands):
- Generate a VN packet in response to an Initial with an unknown
  version. Trivial wire encode via the existing C2 `Header::encode`.

Versions WE support, today:
- `QUIC_VERSION_1 = 0x00000001` (RFC 9000) — the only one.

Future:
- v2 (RFC 9369) when we add it (not in v1).
- draft revisions only if needed for interop testing.

## Security

VN packets carry NO authentication or integrity protection (RFC 9000
§6.1) because no shared secret exists yet. **The off-path attacker
can forge VN packets.** Defense: the client treats a VN as advisory;
it doesn't restart the handshake until the user explicitly approves
(or the application policy explicitly authorises auto-restart).

`/security-review` lite: an attacker who can spoof the server's
address can DoS the client by repeatedly injecting VN packets with
no-overlapping versions. Mitigation: surface as a typed error;
caller's retry policy decides. Per RFC 9000 §6.1.2, even a successful
VN doesn't change anything about the path validation.

## Worked example

State: ConnectionState::Initial with `local_version = 0x00000001`.

| Event | Action | Outcome |
|---|---|---|
| Inbound VN packet: dcid=client.SCID, scid=server.chosen, versions=[0x00000001, 0x00000002] | parse `Header::VersionNegotiation`; check overlap | We support v1 → `Err(VersionNegotiationRequested { offered: [v1, v2] })` |
| Inbound VN packet: versions=[0xff00001d] | parse; no overlap with our supported set | `Err(VersionNegotiationFailed { offered: [0xff00001d] })` |
| Inbound VN packet: empty versions list | parse OK; no overlap | `Err(VersionNegotiationFailed { offered: [] })` |
| Inbound VN packet: malformed (non-mult-of-4 versions_raw) | C2 parse rejects | `Err(Header(MalformedVersionList))` |

## Code site

- `proxima-quic-proto/src/connection/mod.rs`:
  - Dispatcher: `handle_datagram` for Initial state peeks the
    header type AFTER `parse_long`; if it's `Header::VersionNegotiation`,
    route to `handle_version_negotiation_datagram`.
  - New helper `handle_version_negotiation_datagram` returns the
    structured error.
- `proxima-quic-proto/src/connection/error.rs`:
  - New variants `VersionNegotiationRequested { offered }` +
    `VersionNegotiationFailed { offered }`.
  - `MAX_VN_OFFERED_VERSIONS` const-generic cap from
    `proxima-quic-proto.toml [connection].vn_max_offered_versions`
    (default 16; RFC has no formal cap but practical impls limit
    around 8-16).

## Tier

Tier-1 (the connection module is alloc-required). The parsing primitive
inherits tier-3 from C2.

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes;
  4-row worked example covers overlap + no-overlap + empty + malformed.
- **Pass 3 — code maps step-by-step**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; swapping
  Requested/Failed on overlap detection, off-by-one in the versions
  list walk, forgetting the multiple-of-4 check would all break
  specific assertions.
- **Pass 6 — paper linked to test**: yes.

## Per principle 14 (incumbent wins on correctness)

VN packet wire format per RFC 9000 §17.2.1. Test vectors lifted from
RFC text directly (no draft / no memory). Cross-check against
quinn-proto's `accept` path if/when integration testing surfaces a
parity question.
