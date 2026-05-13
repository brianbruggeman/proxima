# proxima-quic + proxima-h3 RFC cross-reference

## Crate consolidation note (2026-07)

The sans-IO proto crates named below were folded into consolidated
crates:

- `proxima-quic-proto` → `proxima-protocols::quic`
- `proxima-h3-proto` → `proxima-protocols::http3_codec`

Section labels below use the pre-consolidation names as written at
the time. Full rename map:
[`docs/decomposition/consolidation.md`](../decomposition/consolidation.md).

Source of truth for "which RFC section does this implement". Every
component row in `discipline.md` cites RFC §x.y; this table is the
authoritative map.

## RFCs and drafts in scope (v1)

| RFC | Title | Scope |
|-----|-------|-------|
| RFC 9000 | QUIC: A UDP-Based Multiplexed and Secure Transport | wire format, streams, flow control, frames, ACK, address validation, migration, version negotiation |
| RFC 9001 | Using TLS to Secure QUIC | packet protection, header protection, key derivation, 0-RTT, key update |
| RFC 9002 | QUIC Loss Detection and Congestion Control | PTO, persistent congestion, NewReno baseline |
| RFC 9221 | An Unreliable Datagram Extension to QUIC | DATAGRAM frame, transport-parameter negotiation, send/recv queues |
| RFC 9438 | CUBIC for Fast and Long-Distance Networks | C16 congestion controller |
| draft-ietf-ccwg-bbr | BBRv2 congestion control | C17 congestion controller — pin draft revision in `edges.md` |
| draft-ietf-quic-multipath | Multipath Extension for QUIC | C26 — pin draft revision in `edges.md` |
| RFC 8311 | Relaxing Restrictions on Explicit Congestion Notification | ECN baseline; informs C18 |
| RFC 9114 | HTTP/3 | h3 frame codec, request/response state machine, server push, GOAWAY |
| RFC 9204 | QPACK: Field Compression for HTTP/3 | static / dynamic table, encoder, decoder |
| RFC 9220 | Bootstrapping WebSockets with HTTP/3 | extended CONNECT |
| RFC 9297 | HTTP Datagrams and the Capsule Protocol | H3-Datagrams over RFC 9221 QUIC DATAGRAM |
| RFC 8446 | TLS Protocol Version 1.3 | TLS 1.3 itself — composed via rustls + aws-lc-rs |

## Component → RFC section table

(`proxima-quic-proto`)

| Component | RFC | Section(s) |
|-----------|-----|-----------|
| C1 varint | 9000 | §16 |
| C2 packet header | 9000 | §17, §17.2 (long), §17.3 (short), §17.2.1 (VN), §17.2.5 (Retry) |
| C3 frame codec | 9000, 9221 | RFC 9000 §19 (all 28 types) + RFC 9221 §4 (DATAGRAM) |
| C4 transport parameters | 9000, 9221, multipath, ECN | RFC 9000 §18 + RFC 9221 §3 + multipath §10 + ECN params |
| C5 HKDF-Expand-Label | 9001 | §5.2 |
| C6 AEAD packet protection | 9001 | §5.1, §5.3 (nonce construction), §5.5 (header) |
| C7 header protection | 9001 | §5.4 |
| C8 connection ID | 9000 | §5.1, §19.15 (NEW_CONNECTION_ID), §19.16 (RETIRE_CONNECTION_ID) |
| C9 packet number spaces | 9000 | §12.3 |
| C10 TLS 1.3 sans-IO | 9001, 8446 | RFC 9001 §4 (handshake), RFC 8446 (the TLS protocol itself) |
| C11 connection state machine | 9000 | §10 (idle / closing / draining), §10.1 (idle timeout), §10.2 (immediate close) |
| C12 streams + flow control | 9000 | §4 (data flow control), §3 (stream states), §19.5–19.13 (stream frames) |
| C13 ACK generation | 9000 | §13.2, §19.3 (ACK frame format) |
| C14 loss detection | 9002 | §6 (loss detection), §6.2 (PTO), §6.4 (persistent congestion) |
| C15 NewReno | 9002 | §7 (congestion control baseline) |
| C16 CUBIC | 9438 | full |
| C17 BBRv2 | draft-ietf-ccwg-bbr | full — pin revision |
| C18 ECN | 9000, 8311 | RFC 9000 §13.4, §19.3.2 (ACK_ECN), RFC 8311 |
| C19 address validation | 9000 | §8 (address validation), §8.1 (anti-amplification), Retry (§17.2.5) |
| C20 anti-amplification | 9000 | §8.1 |
| C21 path migration | 9000 | §9, §19.17 (PATH_CHALLENGE), §19.18 (PATH_RESPONSE) |
| C22 version negotiation | 9000 | §6 + §17.2.1 |
| C23 key update | 9001 | §6 (key update), §6.1 (initiating), §6.2 (responding), §6.5 (timing) |
| C24 0-RTT | 9001 | §4.6 (0-RTT), §9.2 (replay considerations) |
| C25 RFC 9221 DATAGRAM | 9221 | §3 (transport param), §4 (frame), §5 (semantics) |
| C26 multipath | draft-ietf-quic-multipath | full — pin revision |
| C27 endpoint demux | 9000 | §5.2 (CID matching), §10.3 (stateless reset) |

(`proxima-h3-proto`)

| Component | RFC | Section(s) |
|-----------|-----|-----------|
| C32 H3 frame codec | 9114 | §7 (all frame types) |
| C33 QPACK encoder | 9204 | §4 (encoder), §5 (instructions) |
| C34 QPACK decoder | 9204 | §4 (decoder) |
| C35 H3 server state machine | 9114 | §4 (HTTP semantics over QUIC), §5 (connection setup), §10 (errors) |
| C36 H3 client state machine | 9114 | §4, §5, §7.2.3 (push) |
| C37 H3-Datagrams | 9297 | full + RFC 9114 §11.4 |
| C38 extended CONNECT | 9220 | full + RFC 9114 §6.4 |

## RFC compliance notes (deviations / interpretations)

(Empty until components land. Each deviation gets a row here citing
the RFC section, the proxima choice, and the rationale.)

| Component | RFC | Section | proxima choice | rationale |
|-----------|-----|---------|---------------|-----------|
| (none yet) | | | | |
