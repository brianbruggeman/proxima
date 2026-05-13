//! Endpoint-level datagram demux + classification primitive.
//!
//! Sits between the UDP datagram source and the per-connection state
//! machine. Given an inbound datagram + a registered DCID-to-handle
//! table, returns a typed [`DatagramClassification`] that drives the
//! caller's per-outcome action:
//!
//! - [`DatagramClassification::Existing`] → route to that connection's
//!   `handle_datagram`.
//! - [`DatagramClassification::NewInitial`] → server-side accept path
//!   (or trigger Retry; that policy is C27.1, not this row).
//! - [`DatagramClassification::UnsupportedVersion`] → emit Version
//!   Negotiation reply (RFC 9000 §6).
//! - [`DatagramClassification::Drop`] → discard silently per
//!   RFC 9000 §10.3.
//!
//! # Tier
//!
//! Two-tier by design (principle 3): the alloc tier routes through a
//! growable `BTreeMap` keyed by [`ConnectionIdBytes`] that scales with the
//! live connection count; the bare `no_std + no_alloc` tier uses a fixed-cap
//! `heapless::FnvIndexMap`. See [`DcidTable`].

use arrayvec::ArrayVec;
#[cfg(not(feature = "quic-alloc"))]
use heapless::index_map::FnvIndexMap;

use crate::quic::packet::header::{self, Form, Header, MAX_CID_LEN};
use crate::quic::sized;

/// Fixed cap on tracked connection IDs in the **no-alloc** demux table.
/// Sourced from `proxima-quic-proto.toml [endpoint].dcid_table_cap`. Only the
/// `no_std + no_alloc` tier is bounded by this — the alloc tier grows.
#[cfg(not(feature = "quic-alloc"))]
pub const DCID_TABLE_CAP: usize = sized::ENDPOINT_DCID_TABLE_CAP;

/// Largest UDP payload (whole datagram) this endpoint advertises it will
/// receive — RFC 9000 §18.2 transport parameter 0x03 (`max_udp_payload_size`).
/// The SINGLE source of truth for three sizes that must never drift apart:
/// the advertised transport parameter, the I/O recv buffers, and the
/// per-packet unprotect scratch. A recv buffer smaller than the advertised
/// value silently truncates the peer's datagrams (mangled AEAD tag ->
/// `DecryptFailed` mid-connection); advertising the spec default (65527)
/// without sizing buffers to match is the canonical loopback footgun. Sourced
/// from `proxima-quic-proto.toml [endpoint].max_udp_payload_size`; tune down
/// for memory-constrained (no_alloc) targets via env override.
pub const MAX_UDP_PAYLOAD_SIZE: usize = sized::ENDPOINT_MAX_UDP_PAYLOAD_SIZE;

/// Connection-id → handle routing table, **two-tier by design** (principle 3).
///
/// On the `alloc` tier it is a growable [`hashbrown::HashMap`] — O(1) exact
/// lookup on the per-datagram hot path (the SwissTable that backs std's
/// `HashMap`, no_std-compatible) that scales with the live connection count:
/// a std server never overflows a fixed cap, so a handshake burst cannot drop
/// connections into a multi-second PTO stall. On the bare `no_std + no_alloc`
/// tier it is a fixed-cap `heapless::FnvIndexMap` sized by [`DCID_TABLE_CAP`]
/// for bounded embedded memory. Bounding the connection count against DoS is
/// admission control and belongs at the accept layer, NOT this routing table.
#[cfg(feature = "quic-alloc")]
type DcidTable = hashbrown::HashMap<ConnectionIdBytes, ConnectionHandle>;
#[cfg(not(feature = "quic-alloc"))]
type DcidTable = FnvIndexMap<ConnectionIdBytes, ConnectionHandle, DCID_TABLE_CAP>;

/// Caller-opaque connection handle. The endpoint demux stores
/// these against DCIDs; the meaning (slab index, pointer, etc.)
/// is the caller's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConnectionHandle(pub u32);

/// Inline connection-ID byte string (matches
/// [`crate::quic::connection::state::ConnectionIdBytes`] but reproduced
/// here so this module compiles tier-3 without the connection FSM).
pub type ConnectionIdBytes = ArrayVec<u8, MAX_CID_LEN>;

/// Errors from [`EndpointDemux`] table operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DemuxError {
    /// DCID already registered with a (possibly different) handle.
    DuplicateDcid,
    /// CID byte length exceeded [`MAX_CID_LEN`].
    DcidTooLong,
    /// Table at capacity. Only the `no_std + no_alloc` tier is bounded
    /// (the alloc tier grows), so this is never returned under `alloc`.
    TableFull,
}

/// Why a datagram was discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DropReason {
    /// Datagram bytes failed header parsing (truncated, fixed-bit
    /// clear, malformed varint, etc.).
    MalformedHeader,
    /// Long-header non-Initial packet (Handshake / 0-RTT / Retry /
    /// Version Negotiation) referenced an unknown DCID — caller
    /// has no use for it.
    UnknownDcidLongHeader,
    /// Short-header packet referenced an unknown DCID — RFC 9000
    /// §10.3 allows the caller to upgrade this to a Stateless
    /// Reset; the classifier only signals the drop.
    UnknownDcidShortHeader,
}

/// Result of [`EndpointDemux::classify_datagram`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatagramClassification<'a> {
    /// DCID found in the table; route to the named connection.
    Existing {
        handle: ConnectionHandle,
        first_byte_form: Form,
    },
    /// Long-header Initial with a DCID NOT in the table. The
    /// server-side caller may accept a new connection or trigger
    /// Retry per its address-validation policy (C27.1).
    NewInitial {
        dcid: &'a [u8],
        scid: &'a [u8],
        token: &'a [u8],
        version: u32,
    },
    /// Long-header with a version not in `supported_versions`.
    /// Server should emit a Version Negotiation packet per
    /// RFC 9000 §6.
    UnsupportedVersion {
        dcid: &'a [u8],
        scid: &'a [u8],
        peer_version: u32,
    },
    /// Discard silently.
    Drop { reason: DropReason },
}

/// Per-endpoint DCID-to-handle table + datagram classifier.
#[derive(Debug, Clone)]
pub struct EndpointDemux {
    /// DCID → handle routing table. See [`DcidTable`]: a growable
    /// `BTreeMap` on the alloc tier (scales with load), a fixed-cap
    /// `heapless::FnvIndexMap` on the no-alloc tier.
    table: DcidTable,
    supported_versions: &'static [u32],
    /// When `Some(N)`, the endpoint issues fixed-length local DCIDs
    /// (always N bytes); `classify_short` slices `datagram[1..1+N]`
    /// as the exact lookup key and avoids the variable-length scan
    /// entirely. `register` enforces N=dcid.len() to keep the table
    /// internally consistent. When `None`, the demux falls back to
    /// scanning every registered key for a prefix match — the slow
    /// path the discipline log called out for production scale.
    local_cid_len: Option<u8>,
}

impl EndpointDemux {
    /// Construct with the given list of supported wire-format
    /// versions (most-preferred first). Variable-length DCID mode —
    /// `classify_short` falls back to the O(N) prefix scan when
    /// dispatching short-header packets. Use
    /// [`Self::with_local_cid_len`] for the O(1) production path.
    #[must_use]
    pub fn new(supported_versions: &'static [u32]) -> Self {
        Self {
            table: DcidTable::new(),
            supported_versions,
            local_cid_len: None,
        }
    }

    /// Construct an endpoint that issues fixed-length local DCIDs
    /// (one byte length declared at construction). `classify_short`
    /// then dispatches in O(1) via a hash lookup keyed by the exact
    /// `datagram[1..1+N]` slice.
    ///
    /// `local_cid_len` MUST be `>=4` (per RFC 9000 §5.1.1's lower bound
    /// for non-zero CIDs) and `<= MAX_CID_LEN` (20 bytes).
    #[must_use]
    pub fn with_local_cid_len(supported_versions: &'static [u32], local_cid_len: u8) -> Self {
        assert!(
            local_cid_len as usize <= MAX_CID_LEN,
            "local_cid_len {local_cid_len} exceeds MAX_CID_LEN={MAX_CID_LEN}"
        );
        Self {
            table: DcidTable::new(),
            supported_versions,
            local_cid_len: Some(local_cid_len),
        }
    }

    /// Register a new DCID → handle binding.
    ///
    /// # Errors
    ///
    /// See [`DemuxError`].
    pub fn register(&mut self, dcid: &[u8], handle: ConnectionHandle) -> Result<(), DemuxError> {
        if dcid.len() > MAX_CID_LEN {
            return Err(DemuxError::DcidTooLong);
        }
        if let Some(expected) = self.local_cid_len
            && dcid.len() != expected as usize
        {
            return Err(DemuxError::DcidTooLong);
        }
        let mut key = ConnectionIdBytes::new();
        // try_extend_from_slice cannot fail — bounds-checked above.
        key.try_extend_from_slice(dcid).ok();
        if self.table.contains_key(&key) {
            return Err(DemuxError::DuplicateDcid);
        }
        // alloc tier: BTreeMap grows, insert never fails. no-alloc tier:
        // heapless is bounded by DCID_TABLE_CAP and returns the entry back on
        // overflow → TableFull.
        #[cfg(feature = "quic-alloc")]
        {
            self.table.insert(key, handle);
        }
        #[cfg(not(feature = "quic-alloc"))]
        {
            self.table
                .insert(key, handle)
                .map_err(|_| DemuxError::TableFull)?;
        }
        Ok(())
    }

    /// Remove a DCID → handle binding. Returns the handle if found.
    pub fn unregister(&mut self, dcid: &[u8]) -> Option<ConnectionHandle> {
        if dcid.len() > MAX_CID_LEN {
            return None;
        }
        let mut key = ConnectionIdBytes::new();
        key.try_extend_from_slice(dcid).ok();
        // BTreeMap::remove on alloc; heapless swap_remove (O(1), no shift) on
        // no-alloc.
        #[cfg(feature = "quic-alloc")]
        {
            self.table.remove(&key)
        }
        #[cfg(not(feature = "quic-alloc"))]
        {
            self.table.swap_remove(&key)
        }
    }

    /// Look up a DCID in the table.
    #[must_use]
    pub fn lookup(&self, dcid: &[u8]) -> Option<ConnectionHandle> {
        if dcid.len() > MAX_CID_LEN {
            return None;
        }
        let mut key = ConnectionIdBytes::new();
        key.try_extend_from_slice(dcid).ok();
        self.table.get(&key).copied()
    }

    /// Number of registered DCIDs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// `true` if no DCIDs are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Classify an inbound datagram per the algorithm in
    /// `docs/proxima-quic/c27-endpoint-demux-design.md`.
    ///
    /// **Ordering matters**: version check before DCID lookup
    /// per RFC 9000 §6.
    #[must_use]
    pub fn classify_datagram<'a>(&self, datagram: &'a [u8]) -> DatagramClassification<'a> {
        let form = match header::peek_form(datagram) {
            Some(form) => form,
            None => {
                return DatagramClassification::Drop {
                    reason: DropReason::MalformedHeader,
                };
            }
        };

        match form {
            Form::Long => self.classify_long(datagram),
            Form::Short => self.classify_short(datagram),
        }
    }

    fn classify_long<'a>(&self, datagram: &'a [u8]) -> DatagramClassification<'a> {
        let parsed = match header::parse_long(datagram) {
            Ok(header) => header,
            Err(_) => {
                return DatagramClassification::Drop {
                    reason: DropReason::MalformedHeader,
                };
            }
        };

        match parsed {
            Header::VersionNegotiation { .. } => DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            },
            Header::Initial {
                version,
                dcid,
                scid,
                token,
                ..
            } => {
                if !self.supported_versions.contains(&version) {
                    return DatagramClassification::UnsupportedVersion {
                        dcid,
                        scid,
                        peer_version: version,
                    };
                }
                if let Some(handle) = self.lookup(dcid) {
                    return DatagramClassification::Existing {
                        handle,
                        first_byte_form: Form::Long,
                    };
                }
                DatagramClassification::NewInitial {
                    dcid,
                    scid,
                    token,
                    version,
                }
            }
            Header::ZeroRtt {
                version,
                dcid,
                scid,
                ..
            }
            | Header::Handshake {
                version,
                dcid,
                scid,
                ..
            } => {
                if !self.supported_versions.contains(&version) {
                    return DatagramClassification::UnsupportedVersion {
                        dcid,
                        scid,
                        peer_version: version,
                    };
                }
                if let Some(handle) = self.lookup(dcid) {
                    return DatagramClassification::Existing {
                        handle,
                        first_byte_form: Form::Long,
                    };
                }
                DatagramClassification::Drop {
                    reason: DropReason::UnknownDcidLongHeader,
                }
            }
            Header::Retry {
                version,
                dcid,
                scid,
                ..
            } => {
                if !self.supported_versions.contains(&version) {
                    return DatagramClassification::UnsupportedVersion {
                        dcid,
                        scid,
                        peer_version: version,
                    };
                }
                if let Some(handle) = self.lookup(dcid) {
                    return DatagramClassification::Existing {
                        handle,
                        first_byte_form: Form::Long,
                    };
                }
                DatagramClassification::Drop {
                    reason: DropReason::UnknownDcidLongHeader,
                }
            }
            Header::Short { .. } => {
                // peek_form said Long; parse_long returned Short.
                // Programmer-error path — surface as malformed.
                DatagramClassification::Drop {
                    reason: DropReason::MalformedHeader,
                }
            }
        }
    }

    fn classify_short<'a>(&self, datagram: &'a [u8]) -> DatagramClassification<'a> {
        if datagram.len() < 2 {
            return DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            };
        }
        // Short header: byte 0 is type, then DCID with caller-known length.
        // Fast path: when the endpoint commits to a fixed local CID
        // length (via with_local_cid_len), slice datagram[1..1+N] as
        // the exact lookup key and do a single hash get — O(1) per
        // datagram regardless of how many connections are registered.
        if let Some(cid_len) = self.local_cid_len {
            let cid_len_usize = cid_len as usize;
            if datagram.len() < 1 + cid_len_usize {
                return DatagramClassification::Drop {
                    reason: DropReason::MalformedHeader,
                };
            }
            let mut key = ConnectionIdBytes::new();
            // Length already validated; ignore the (impossible) overflow.
            key.try_extend_from_slice(&datagram[1..1 + cid_len_usize])
                .ok();
            if let Some(handle) = self.table.get(&key) {
                return DatagramClassification::Existing {
                    handle: *handle,
                    first_byte_form: Form::Short,
                };
            }
            return DatagramClassification::Drop {
                reason: DropReason::UnknownDcidShortHeader,
            };
        }
        // Slow path: variable-length DCIDs. The library cannot
        // determine the length from the wire bytes alone, so it scans
        // every registered key for a prefix match. Longest-match wins
        // per RFC 9000 §5.1 (CIDs are not guaranteed unique-prefix
        // across endpoints, but locally we choose them so a longer
        // match is the more specific binding).
        let body = &datagram[1..];
        let mut best: Option<(usize, ConnectionHandle)> = None;
        for (key, handle) in self.table.iter() {
            let len = key.len();
            if body.len() >= len && &body[..len] == key.as_slice() {
                match best {
                    Some((best_len, _)) if best_len >= len => {}
                    _ => best = Some((len, *handle)),
                }
            }
        }
        if let Some((_, handle)) = best {
            DatagramClassification::Existing {
                handle,
                first_byte_form: Form::Short,
            }
        } else {
            DatagramClassification::Drop {
                reason: DropReason::UnknownDcidShortHeader,
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::packet::header::Header;

    const V1: &[u32] = &[0x0000_0001];

    fn build_initial_datagram(version: u32, dcid: &[u8], scid: &[u8]) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; 256];
        let pn_and_payload = alloc::vec![0u8; 20];
        let header = Header::Initial {
            version,
            dcid,
            scid,
            token: &[],
            length: 20,
            pn_and_payload: &pn_and_payload,
        };
        let written = header.encode(&mut buf).expect("encode");
        buf.truncate(written);
        buf
    }

    fn build_short_datagram(dcid: &[u8]) -> alloc::vec::Vec<u8> {
        // byte0 = 0x40 (long=0, fixed=1) + raw DCID + a few payload bytes.
        let mut buf = alloc::vec![0x40];
        buf.extend_from_slice(dcid);
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        buf
    }

    extern crate alloc;

    #[test]
    fn empty_datagram_returns_drop_malformed() {
        let demux = EndpointDemux::new(V1);
        let outcome = demux.classify_datagram(&[]);
        assert_eq!(
            outcome,
            DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            }
        );
    }

    #[test]
    fn long_header_unknown_version_returns_unsupported() {
        let demux = EndpointDemux::new(V1);
        let dcid = [0xAA, 0xAB, 0xAC, 0xAD];
        let scid = [0xBA, 0xBB];
        let datagram = build_initial_datagram(99, &dcid, &scid);
        let outcome = demux.classify_datagram(&datagram);
        match outcome {
            DatagramClassification::UnsupportedVersion { peer_version, .. } => {
                assert_eq!(peer_version, 99);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn long_header_v1_initial_unknown_dcid_returns_new_initial() {
        let demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 8];
        let scid = [0xBA; 4];
        let datagram = build_initial_datagram(1, &dcid, &scid);
        let outcome = demux.classify_datagram(&datagram);
        match outcome {
            DatagramClassification::NewInitial {
                dcid: parsed_dcid,
                version,
                ..
            } => {
                assert_eq!(version, 1);
                assert_eq!(parsed_dcid, &dcid[..]);
            }
            other => panic!("expected NewInitial, got {other:?}"),
        }
    }

    #[test]
    fn registered_dcid_long_header_returns_existing_long() {
        let mut demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 8];
        demux
            .register(&dcid, ConnectionHandle(42))
            .expect("register ok");
        let datagram = build_initial_datagram(1, &dcid, &[0xBA, 0xBB]);
        let outcome = demux.classify_datagram(&datagram);
        assert_eq!(
            outcome,
            DatagramClassification::Existing {
                handle: ConnectionHandle(42),
                first_byte_form: Form::Long,
            }
        );
    }

    #[test]
    fn short_header_known_dcid_returns_existing_short() {
        let mut demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 8];
        demux
            .register(&dcid, ConnectionHandle(42))
            .expect("register ok");
        let datagram = build_short_datagram(&dcid);
        let outcome = demux.classify_datagram(&datagram);
        assert_eq!(
            outcome,
            DatagramClassification::Existing {
                handle: ConnectionHandle(42),
                first_byte_form: Form::Short,
            }
        );
    }

    #[test]
    fn short_header_unknown_dcid_returns_drop_unknown_short() {
        let demux = EndpointDemux::new(V1);
        let datagram = build_short_datagram(&[0xCC; 8]);
        let outcome = demux.classify_datagram(&datagram);
        assert_eq!(
            outcome,
            DatagramClassification::Drop {
                reason: DropReason::UnknownDcidShortHeader,
            }
        );
    }

    #[test]
    fn unregister_removes_short_header_routing() {
        let mut demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 8];
        demux
            .register(&dcid, ConnectionHandle(42))
            .expect("register");
        let removed = demux.unregister(&dcid).expect("removed");
        assert_eq!(removed, ConnectionHandle(42));
        let datagram = build_short_datagram(&dcid);
        let outcome = demux.classify_datagram(&datagram);
        assert_eq!(
            outcome,
            DatagramClassification::Drop {
                reason: DropReason::UnknownDcidShortHeader,
            }
        );
    }

    #[test]
    fn register_rejects_duplicate_dcid() {
        let mut demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 4];
        demux.register(&dcid, ConnectionHandle(1)).expect("first");
        let err = demux.register(&dcid, ConnectionHandle(2)).unwrap_err();
        assert_eq!(err, DemuxError::DuplicateDcid);
    }

    #[test]
    fn register_rejects_dcid_too_long() {
        let mut demux = EndpointDemux::new(V1);
        let too_long = [0u8; MAX_CID_LEN + 1];
        let err = demux.register(&too_long, ConnectionHandle(1)).unwrap_err();
        assert_eq!(err, DemuxError::DcidTooLong);
    }

    #[cfg(not(feature = "quic-alloc"))]
    #[test]
    fn register_rejects_when_table_full() {
        let mut demux = EndpointDemux::new(V1);
        for i in 0..DCID_TABLE_CAP as u32 {
            let dcid = i.to_be_bytes();
            demux
                .register(&dcid, ConnectionHandle(i))
                .expect("under cap");
        }
        let extra = (DCID_TABLE_CAP as u32 + 1).to_be_bytes();
        let err = demux.register(&extra, ConnectionHandle(99)).unwrap_err();
        assert_eq!(err, DemuxError::TableFull);
    }

    #[test]
    fn with_local_cid_len_classify_short_uses_fixed_length_lookup() {
        // Fast path: endpoint commits to 8-byte local CIDs. Any
        // datagram[1..9] slice is the lookup key — no linear scan.
        let mut demux = EndpointDemux::with_local_cid_len(V1, 8);
        let dcid = [0xAB; 8];
        demux
            .register(&dcid, ConnectionHandle(42))
            .expect("register fixed-length");
        let datagram = build_short_datagram(&dcid);
        let outcome = demux.classify_datagram(&datagram);
        assert_eq!(
            outcome,
            DatagramClassification::Existing {
                handle: ConnectionHandle(42),
                first_byte_form: Form::Short,
            }
        );
    }

    #[test]
    fn with_local_cid_len_rejects_register_of_wrong_length() {
        // Endpoint declared 8-byte fixed length — a 4-byte register
        // must be rejected so the fast-path lookup invariant holds.
        let mut demux = EndpointDemux::with_local_cid_len(V1, 8);
        let too_short = [0xAA; 4];
        let err = demux.register(&too_short, ConnectionHandle(1)).unwrap_err();
        assert_eq!(err, DemuxError::DcidTooLong);
    }

    #[test]
    fn with_local_cid_len_short_too_small_returns_drop_malformed() {
        // Datagram has fewer than 1 + cid_len bytes — drop as malformed
        // BEFORE attempting the lookup (otherwise we'd alloc a key
        // from out-of-bounds data).
        let demux = EndpointDemux::with_local_cid_len(V1, 8);
        let runt = [0x40, 0xAB, 0xAB, 0xAB]; // only 4 bytes of "CID"
        let outcome = demux.classify_datagram(&runt);
        assert_eq!(
            outcome,
            DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            }
        );
    }

    #[test]
    fn classify_routes_version_check_before_dcid_lookup() {
        // Even with the DCID registered, an unsupported version
        // must surface UnsupportedVersion — not Existing — so the
        // caller can emit a VN reply.
        let mut demux = EndpointDemux::new(V1);
        let dcid = [0xAB; 8];
        demux
            .register(&dcid, ConnectionHandle(42))
            .expect("register");
        let datagram = build_initial_datagram(99, &dcid, &[0xBA, 0xBB]);
        let outcome = demux.classify_datagram(&datagram);
        match outcome {
            DatagramClassification::UnsupportedVersion { peer_version, .. } => {
                assert_eq!(peer_version, 99);
            }
            other => panic!("expected UnsupportedVersion (version checked first), got {other:?}"),
        }
    }
}
