//! [`Classifier`] — the per-connection arbitration state machine that picks
//! one [`AnyProtocol`] out of a candidate slice as bytes accumulate.
//! Generalizes [`crate::preface::classify_preface`] (a closed h1-or-h2
//! choice) to an open, registry-driven candidate set.
//!
//! # The priority-ordered-wait rule
//!
//! Each candidate is one of three states across the connection's lifetime —
//! see [`CandidateState`]: `Live { at_least }` (still deciding), `Dead`
//! (ruled out for good), or `Matched { consumed }` (resolved to its own
//! wire). [`Classifier::advance`] probes every still-`Live` candidate
//! against the accumulated prefix, then arbitrates: a `Matched` candidate's
//! win is HELD (`NeedMoreBytes`) as long as any STRICTLY higher-priority
//! candidate is still `Live` — a lower-priority match must never be
//! observable while a higher-priority candidate could still beat it. Once
//! no higher-priority candidate remains live, the highest-priority
//! `Matched` candidate wins outright, unless more than one candidate tied
//! at that priority, in which case the classifier reports
//! [`ClassifyOutcome::AmbiguousMatch`] rather than silently picking one.
//!
//! Three invariants this guarantees: **termination** (every candidate
//! resolves to `Dead` or `Matched` by the time the prefix reaches
//! `min(global_cap, candidate.max_prefix_bytes())` — the classifier forces
//! a candidate whose `probe` still hasn't decided past its own declared
//! ceiling to `Dead`, rather than trusting every implementor to
//! self-terminate); **no-misroute** (a lower-priority match is
//! unobservable while a strictly-higher-priority candidate is still
//! `Live`); and **flat-equal-priority never spuriously waits** (the hold
//! only triggers on a STRICT priority ordering, so candidates sharing a
//! priority resolve as soon as one of them wins).

use std::sync::Arc;

use super::probe::{AnyProtocol, ProbeVerdict};

/// Outcome of one [`Classifier::advance`] call. FINAL shape — see this
/// module's doc for the arbitration rule that produces it.
#[derive(Clone)]
#[non_exhaustive]
pub enum ClassifyOutcome {
    /// Exactly one candidate won: either the sole `Match`, or the
    /// highest-priority `Match` once every strictly-higher-priority
    /// candidate has resolved.
    Matched(Arc<dyn AnyProtocol>),
    /// Every candidate answered `No` before any one candidate could win.
    /// `bytes_examined` is the prefix length at the moment of rejection.
    Rejected { bytes_examined: usize },
    /// No winner yet: either a candidate is still deciding, or a `Matched`
    /// candidate's win is being HELD because a strictly-higher-priority
    /// candidate is still `Live` and could still out-rank it. `at_least`
    /// is the smallest total-prefix-length any still-`Live` candidate
    /// asked for. The caller reads more and calls `advance` again with the
    /// larger buffer.
    NeedMoreBytes { at_least: usize },
    /// The accumulated prefix has reached the classifier's `global_cap`
    /// without any candidate resolving to `Match` or every candidate
    /// resolving to `No` — a peer that will never send enough bytes to
    /// decide. The caller should reject the connection rather than read
    /// forever.
    PrefixBoundExceeded,
    /// More than one candidate resolved to `Matched` at the SAME winning
    /// priority — the classifier never silently picks one; the caller
    /// decides (log a configuration conflict, reject, or whatever policy
    /// fits). `priority` is the tied winning priority; `matches` lists
    /// every tied candidate with its own `consumed` length.
    AmbiguousMatch {
        priority: u16,
        matches: Vec<(Arc<dyn AnyProtocol>, usize)>,
    },
}

// Manual impl: `Arc<dyn AnyProtocol>` doesn't (and shouldn't) require
// `Debug` on every implementor just to satisfy a derive here — print the
// candidate's own `name()` instead of its trait-object address.
impl core::fmt::Debug for ClassifyOutcome {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Matched(protocol) => formatter
                .debug_tuple("Matched")
                .field(&protocol.name())
                .finish(),
            Self::Rejected { bytes_examined } => formatter
                .debug_struct("Rejected")
                .field("bytes_examined", bytes_examined)
                .finish(),
            Self::NeedMoreBytes { at_least } => formatter
                .debug_struct("NeedMoreBytes")
                .field("at_least", at_least)
                .finish(),
            Self::PrefixBoundExceeded => formatter.write_str("PrefixBoundExceeded"),
            Self::AmbiguousMatch { priority, matches } => formatter
                .debug_struct("AmbiguousMatch")
                .field("priority", priority)
                .field(
                    "matches",
                    &matches
                        .iter()
                        .map(|(protocol, consumed)| (protocol.name(), *consumed))
                        .collect::<Vec<_>>(),
                )
                .finish(),
        }
    }
}

/// Per-candidate progress, persisted in [`Classifier`] across `advance`
/// calls — `Dead` and `Matched` are STICKY (a candidate in either state is
/// never probed again on this connection): a `No` verdict can only get
/// more definite as more bytes arrive, never reverse, and a `Match`
/// verdict is the candidate's own final word on the wire it already
/// recognized.
#[derive(Clone, Copy)]
enum CandidateState {
    /// Still deciding; `at_least` is the smallest total prefix length this
    /// candidate has said it needs so far.
    Live { at_least: usize },
    /// Ruled out for the remaining lifetime of the connection.
    Dead,
    /// Resolved to this candidate's own wire; `consumed` is how many
    /// leading bytes are its own fixed framing (informational only — see
    /// [`ProbeVerdict::Match`]).
    Matched { consumed: usize },
}

/// Per-connection classification state. Sized once per connection —
/// `states` tracks, per index into `candidates`, that candidate's
/// [`CandidateState`].
pub struct Classifier {
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    states: Vec<CandidateState>,
    global_cap: usize,
}

impl Classifier {
    /// `candidates` is typically [`crate::any::AnyRegistry::snapshot`] (or
    /// `snapshot_named`). `global_cap` bounds how many prefix bytes this
    /// classifier will ever be advanced with before reporting
    /// `PrefixBoundExceeded` — see [`crate::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT`]
    /// for the build-time default.
    #[must_use]
    pub fn new(candidates: Arc<[Arc<dyn AnyProtocol>]>, global_cap: usize) -> Self {
        let states = vec![CandidateState::Live { at_least: 0 }; candidates.len()];
        Self {
            candidates,
            states,
            global_cap,
        }
    }

    /// Advance classification with the FULL accumulated prefix (from byte
    /// zero of the connection) — callers grow `prefix` across reads and
    /// re-call, mirroring [`crate::preface::classify_preface`]'s contract.
    /// See this module's doc for the priority-ordered-wait rule this
    /// implements.
    #[must_use]
    pub fn advance(&mut self, prefix: &[u8]) -> ClassifyOutcome {
        if prefix.len() > self.global_cap {
            return ClassifyOutcome::PrefixBoundExceeded;
        }

        self.probe_live_candidates(prefix);

        let live = self.live_candidates();
        let min_live_at_least = live.iter().map(|(_, at_least)| *at_least).min();

        if let Some(winning_priority) = self.winning_priority() {
            let higher_priority_live = live
                .iter()
                .any(|(priority, _)| *priority > winning_priority);
            if !higher_priority_live {
                return self.resolve_matched_winners(winning_priority);
            }
        }

        match min_live_at_least {
            Some(at_least) => ClassifyOutcome::NeedMoreBytes { at_least },
            None => ClassifyOutcome::Rejected {
                bytes_examined: prefix.len(),
            },
        }
    }

    /// Probes every still-`Live` candidate against the accumulated prefix
    /// and updates its state. `probe` is always called — a candidate that
    /// can decide from fewer bytes than its own declared
    /// [`AnyProtocol::max_prefix_bytes`] must still resolve on the first
    /// call it sees, even one carrying far more bytes than it needs (a
    /// single read can hand the classifier an entire small request at
    /// once). The one override: a candidate that STILL reports `NeedMore`
    /// despite the prefix having already passed its own declared ceiling
    /// is forced to `Dead` — the classifier enforces that bound itself
    /// rather than trusting every implementor to self-terminate. `Dead`/
    /// `Matched` candidates are skipped entirely (sticky, never
    /// re-probed).
    fn probe_live_candidates(&mut self, prefix: &[u8]) {
        for (candidate, state) in self.candidates.iter().zip(self.states.iter_mut()) {
            if !matches!(state, CandidateState::Live { .. }) {
                continue;
            }
            *state = match candidate.probe(prefix) {
                ProbeVerdict::No => CandidateState::Dead,
                ProbeVerdict::Match { consumed } => CandidateState::Matched { consumed },
                ProbeVerdict::NeedMore { at_least } => {
                    if prefix.len() > candidate.max_prefix_bytes() {
                        CandidateState::Dead
                    } else {
                        CandidateState::Live { at_least }
                    }
                }
            };
        }
    }

    /// `(priority, at_least)` for every candidate still `Live`.
    fn live_candidates(&self) -> Vec<(u16, usize)> {
        self.candidates
            .iter()
            .zip(self.states.iter())
            .filter_map(|(candidate, state)| match state {
                CandidateState::Live { at_least } => Some((candidate.priority(), *at_least)),
                CandidateState::Dead | CandidateState::Matched { .. } => None,
            })
            .collect()
    }

    /// Highest priority among candidates that have resolved to `Matched`,
    /// or `None` if nothing has matched yet.
    fn winning_priority(&self) -> Option<u16> {
        self.candidates
            .iter()
            .zip(self.states.iter())
            .filter_map(|(candidate, state)| match state {
                CandidateState::Matched { .. } => Some(candidate.priority()),
                CandidateState::Live { .. } | CandidateState::Dead => None,
            })
            .max()
    }

    /// Every `Matched` candidate at exactly `priority` — exactly one is a
    /// clean `Matched`, two or more is an `AmbiguousMatch` the classifier
    /// never resolves by picking one.
    fn resolve_matched_winners(&self, priority: u16) -> ClassifyOutcome {
        let winners: Vec<(Arc<dyn AnyProtocol>, usize)> = self
            .candidates
            .iter()
            .zip(self.states.iter())
            .filter_map(|(candidate, state)| match state {
                CandidateState::Matched { consumed } if candidate.priority() == priority => {
                    Some((Arc::clone(candidate), *consumed))
                }
                _ => None,
            })
            .collect();
        if let [(protocol, _consumed)] = winners.as_slice() {
            return ClassifyOutcome::Matched(Arc::clone(protocol));
        }
        ClassifyOutcome::AmbiguousMatch {
            priority,
            matches: winners,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::any::probe::AnyHandler;
    use proxima_core::ProximaError;
    use proxima_primitives::stream::{PeerInfo, StreamConnection};
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    /// A candidate that matches a fixed literal byte string exactly, the
    /// same shape as h2's fixed preface — real production shape (RFC 9113
    /// §3.4 is itself a fixed literal), not a synthetic byte pattern.
    struct LiteralAny {
        label: &'static str,
        literal: &'static [u8],
    }

    impl AnyProtocol for LiteralAny {
        fn name(&self) -> &str {
            self.label
        }

        fn max_prefix_bytes(&self) -> usize {
            self.literal.len()
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            let compare_len = prefix.len().min(self.literal.len());
            if prefix[..compare_len] != self.literal[..compare_len] {
                return ProbeVerdict::No;
            }
            if prefix.len() < self.literal.len() {
                return ProbeVerdict::NeedMore {
                    at_least: self.literal.len(),
                };
            }
            ProbeVerdict::Match {
                consumed: self.literal.len(),
            }
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    fn h1_like() -> Arc<dyn AnyProtocol> {
        Arc::new(LiteralAny {
            label: "h1-like",
            literal: b"GET ",
        })
    }

    fn h2_like() -> Arc<dyn AnyProtocol> {
        Arc::new(LiteralAny {
            label: "h2-like",
            literal: b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n",
        })
    }

    fn candidates() -> Arc<[Arc<dyn AnyProtocol>]> {
        Arc::from(vec![h1_like(), h2_like()])
    }

    #[test]
    fn resolves_h1_like_candidate_on_first_bytes() {
        let mut classifier = Classifier::new(candidates(), 256);
        let outcome = classifier.advance(b"GET / HTTP/1.1\r\n");
        match outcome {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h1-like"),
            other => panic!("expected Matched(h1-like), got {other:?}"),
        }
    }

    #[test]
    fn needs_more_bytes_before_the_h2_preface_completes() {
        let mut classifier = Classifier::new(candidates(), 256);
        let outcome = classifier.advance(b"PRI *");
        assert!(
            matches!(outcome, ClassifyOutcome::NeedMoreBytes { at_least: 24 }),
            "expected NeedMoreBytes{{at_least: 24}}, got {outcome:?}"
        );
    }

    #[test]
    fn resolves_h2_like_candidate_once_the_full_preface_arrives() {
        let mut classifier = Classifier::new(candidates(), 256);
        let full_preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        // first call narrows to "still could be h2, still could be h1
        // until the divergent byte" — feed the full preface in one shot,
        // mirroring a client that writes it in a single syscall.
        let outcome = classifier.advance(full_preface);
        match outcome {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h2-like"),
            other => panic!("expected Matched(h2-like), got {other:?}"),
        }
    }

    #[test]
    fn rejects_once_every_candidate_answers_no() {
        let mut classifier = Classifier::new(candidates(), 256);
        let outcome = classifier.advance(b"XYZQ");
        assert!(
            matches!(outcome, ClassifyOutcome::Rejected { bytes_examined: 4 }),
            "expected Rejected{{bytes_examined: 4}}, got {outcome:?}"
        );
    }

    #[test]
    fn reports_prefix_bound_exceeded_once_the_global_cap_is_hit() {
        let mut classifier = Classifier::new(candidates(), 4);
        // 5 bytes exceeds a 4-byte cap, even though the bytes are a live
        // prefix of the h2-like literal.
        let outcome = classifier.advance(b"PRI *");
        assert!(matches!(outcome, ClassifyOutcome::PrefixBoundExceeded));
    }

    #[test]
    fn a_dead_candidate_stays_dead_across_subsequent_advances() {
        let mut classifier = Classifier::new(candidates(), 256);
        // "PRI " eliminates h1-like (mismatches "GET ") but keeps h2-like alive.
        let first = classifier.advance(b"PRI *");
        assert!(matches!(first, ClassifyOutcome::NeedMoreBytes { .. }));
        // even if the next read somehow looked like it could satisfy
        // h1-like again, h1-like must not be resurrected.
        let full_preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        let second = classifier.advance(full_preface);
        match second {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h2-like"),
            other => panic!("expected Matched(h2-like), got {other:?}"),
        }
    }

    // --- proven priority-ordered-wait rule: real-shaped candidates ---

    /// Fixed 8-byte header check shared by `PGWIRE` and its priority-100
    /// collision partner `CUSTOM_RPC` in the algorithm's worked traces:
    /// `bytes[0..4]` as a big-endian i32 total length in `[8, 10000]` AND
    /// `bytes[4..8] == [0x00, 0x03, 0x00, 0x00]` (pgwire's protocol-version
    /// 3.0 word) together mean `Match { consumed: 8 }`; anything else once
    /// 8 bytes have arrived is `No`; fewer than 8 bytes is always
    /// `NeedMore { at_least: 8 }` — the header is checked all at once,
    /// never incrementally.
    struct FixedHeaderAny {
        label: &'static str,
        priority: u16,
    }

    impl AnyProtocol for FixedHeaderAny {
        fn name(&self) -> &str {
            self.label
        }

        fn priority(&self) -> u16 {
            self.priority
        }

        fn max_prefix_bytes(&self) -> usize {
            8192
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            if prefix.len() < 8 {
                return ProbeVerdict::NeedMore { at_least: 8 };
            }
            let length = i32::from_be_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]);
            let version_word = prefix[4..8] == [0x00, 0x03, 0x00, 0x00];
            if (8..=10_000).contains(&length) && version_word {
                ProbeVerdict::Match { consumed: 8 }
            } else {
                ProbeVerdict::No
            }
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// `FRAMED_RPC`: a 4-byte big-endian total length (inclusive of
    /// itself), bounded to `[5, 65536]`, followed by a 1-byte tag once the
    /// full frame has arrived. `tag_is_legal` parameterizes the two
    /// variants the worked traces need — one where the pgwire startup
    /// bytes' tag (`0x00`) is illegal, one where it is legal — without
    /// duplicating the framing logic.
    struct FramedRpcAny {
        priority: u16,
        tag_is_legal: fn(u8) -> bool,
    }

    impl AnyProtocol for FramedRpcAny {
        fn name(&self) -> &str {
            "framed-rpc"
        }

        fn priority(&self) -> u16 {
            self.priority
        }

        fn max_prefix_bytes(&self) -> usize {
            65536
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            if prefix.len() < 4 {
                return ProbeVerdict::NeedMore { at_least: 4 };
            }
            let length = u32::from_be_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
            if !(5..=65536).contains(&length) {
                return ProbeVerdict::No;
            }
            if prefix.len() < length {
                return ProbeVerdict::NeedMore { at_least: length };
            }
            if (self.tag_is_legal)(prefix[4]) {
                ProbeVerdict::Match { consumed: length }
            } else {
                ProbeVerdict::No
            }
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// `MC_HANDSHAKE`: a Minecraft-shaped LEB128 varint length prefix,
    /// bounded to `[6, 2048]` once decoded (a length outside that range is
    /// `No` immediately) — this is the candidate the worked traces use to
    /// prove a higher-priority candidate that dies early never gets to
    /// hold a lower-priority match.
    struct McHandshakeAny {
        priority: u16,
    }

    impl McHandshakeAny {
        fn decode_varint(prefix: &[u8]) -> Option<(u64, usize)> {
            let mut value: u64 = 0;
            for (index, byte) in prefix.iter().enumerate().take(5) {
                value |= u64::from(byte & 0x7f) << (7 * index);
                if byte & 0x80 == 0 {
                    return Some((value, index + 1));
                }
            }
            None
        }
    }

    impl AnyProtocol for McHandshakeAny {
        fn name(&self) -> &str {
            "mc-handshake"
        }

        fn priority(&self) -> u16 {
            self.priority
        }

        fn max_prefix_bytes(&self) -> usize {
            2048
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            match Self::decode_varint(prefix) {
                Some((length, varint_len)) => {
                    if !(6..=2048).contains(&length) {
                        return ProbeVerdict::No;
                    }
                    let total = varint_len + length as usize;
                    if prefix.len() < total {
                        return ProbeVerdict::NeedMore { at_least: total };
                    }
                    ProbeVerdict::Match { consumed: total }
                }
                None if prefix.len() >= 5 => ProbeVerdict::No,
                None => ProbeVerdict::NeedMore {
                    at_least: prefix.len() + 1,
                },
            }
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// Real pgwire (PostgreSQL wire protocol) `StartupMessage` bytes: a
    /// 4-byte big-endian total length (self-inclusive), the protocol
    /// version word `3.0` (`0x00030000`), then null-terminated
    /// `key\0value\0` pairs and a final `0x00` terminator — RFC-shaped
    /// production bytes, not a synthetic pattern.
    fn pgwire_startup_bytes() -> Vec<u8> {
        let mut bytes = vec![0x00, 0x00, 0x00, 0x25, 0x00, 0x03, 0x00, 0x00];
        bytes.extend_from_slice(b"user\0");
        bytes.extend_from_slice(b"alice\0");
        bytes.extend_from_slice(b"database\0");
        bytes.extend_from_slice(b"proxima\0");
        bytes.push(0x00);
        bytes
    }

    #[test]
    fn pgwire_startup_bytes_are_37_bytes_with_a_self_inclusive_length() {
        let bytes = pgwire_startup_bytes();
        assert_eq!(bytes.len(), 37);
        let length = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(length, 37);
    }

    #[test]
    fn higher_priority_candidate_that_dies_early_never_engages() {
        let mc: Arc<dyn AnyProtocol> = Arc::new(McHandshakeAny { priority: 150 });
        let pgwire: Arc<dyn AnyProtocol> = Arc::new(FixedHeaderAny {
            label: "pgwire",
            priority: 100,
        });
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![mc, pgwire]);
        let bytes = pgwire_startup_bytes();
        let mut classifier = Classifier::new(candidates, 4096);

        for length in 1..8 {
            let outcome = classifier.advance(&bytes[..length]);
            assert!(
                matches!(outcome, ClassifyOutcome::NeedMoreBytes { at_least: 8 }),
                "prefix len {length}: expected NeedMoreBytes{{at_least: 8}}, got {outcome:?}"
            );
        }
        let outcome = classifier.advance(&bytes[..8]);
        match outcome {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "pgwire"),
            other => panic!("expected Matched(pgwire) at byte 8, got {other:?}"),
        }
    }

    #[test]
    fn a_matched_lower_priority_candidate_is_held_while_a_higher_priority_candidate_stays_live() {
        let framed_rpc: Arc<dyn AnyProtocol> = Arc::new(FramedRpcAny {
            priority: 150,
            tag_is_legal: |tag| tag != 0x00,
        });
        let pgwire: Arc<dyn AnyProtocol> = Arc::new(FixedHeaderAny {
            label: "pgwire",
            priority: 100,
        });
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![framed_rpc, pgwire]);
        let bytes = pgwire_startup_bytes();
        let mut classifier = Classifier::new(candidates, 4096);

        for length in 1..4 {
            let outcome = classifier.advance(&bytes[..length]);
            assert!(
                matches!(outcome, ClassifyOutcome::NeedMoreBytes { at_least: 4 }),
                "prefix len {length}: expected NeedMoreBytes{{at_least: 4}}, got {outcome:?}"
            );
        }
        for length in 4..8 {
            let outcome = classifier.advance(&bytes[..length]);
            assert!(
                matches!(outcome, ClassifyOutcome::NeedMoreBytes { at_least: 8 }),
                "prefix len {length}: expected NeedMoreBytes{{at_least: 8}}, got {outcome:?}"
            );
        }
        for length in 8..37 {
            let outcome = classifier.advance(&bytes[..length]);
            assert!(
                matches!(outcome, ClassifyOutcome::NeedMoreBytes { at_least: 37 }),
                "prefix len {length}: expected the pgwire match HELD as \
                 NeedMoreBytes{{at_least: 37}}, got {outcome:?}"
            );
        }
        let outcome = classifier.advance(&bytes);
        match outcome {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "pgwire"),
            other => panic!("expected Matched(pgwire) at byte 37, got {other:?}"),
        }
    }

    #[test]
    fn a_tied_priority_is_broken_toward_the_higher_priority_matched_candidate() {
        // Same header/framing as the held-match case, but the pgwire
        // startup bytes' tag (0x00) is now LEGAL for framed-rpc, so
        // framed-rpc (150) wins outright once it resolves at byte 37.
        let framed_rpc: Arc<dyn AnyProtocol> = Arc::new(FramedRpcAny {
            priority: 150,
            tag_is_legal: |_tag| true,
        });
        let pgwire: Arc<dyn AnyProtocol> = Arc::new(FixedHeaderAny {
            label: "pgwire",
            priority: 100,
        });
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![framed_rpc, pgwire]);
        let bytes = pgwire_startup_bytes();
        let mut classifier = Classifier::new(candidates, 4096);

        for length in 1..37 {
            let _ = classifier.advance(&bytes[..length]);
        }
        let outcome = classifier.advance(&bytes);
        match outcome {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "framed-rpc"),
            other => panic!(
                "expected Matched(framed-rpc) at byte 37 (150 beats pgwire's 100), got {other:?}"
            ),
        }
    }

    #[test]
    fn two_candidates_matching_at_the_same_priority_report_ambiguous_match() {
        let pgwire: Arc<dyn AnyProtocol> = Arc::new(FixedHeaderAny {
            label: "pgwire",
            priority: 100,
        });
        let custom_rpc: Arc<dyn AnyProtocol> = Arc::new(FixedHeaderAny {
            label: "custom-rpc",
            priority: 100,
        });
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![pgwire, custom_rpc]);
        let bytes = pgwire_startup_bytes();
        let mut classifier = Classifier::new(candidates, 4096);

        for length in 1..8 {
            let outcome = classifier.advance(&bytes[..length]);
            assert!(matches!(
                outcome,
                ClassifyOutcome::NeedMoreBytes { at_least: 8 }
            ));
        }
        let outcome = classifier.advance(&bytes[..8]);
        match outcome {
            ClassifyOutcome::AmbiguousMatch { priority, matches } => {
                assert_eq!(priority, 100);
                let mut names: Vec<&str> =
                    matches.iter().map(|(protocol, _)| protocol.name()).collect();
                names.sort_unstable();
                assert_eq!(names, vec!["custom-rpc", "pgwire"]);
                assert!(matches.iter().all(|(_, consumed)| *consumed == 8));
            }
            other => panic!("expected AmbiguousMatch{{priority: 100, ..}}, got {other:?}"),
        }
    }

    #[test]
    fn flat_equal_priority_never_spuriously_waits_for_a_dead_sibling() {
        let mut h1_classifier = Classifier::new(candidates(), 256);
        match h1_classifier.advance(b"GET / HTTP/1.1\r\n") {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h1-like"),
            other => panic!("expected Matched(h1-like) with no priority-wait, got {other:?}"),
        }

        let mut h2_classifier = Classifier::new(candidates(), 256);
        let full_preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        match h2_classifier.advance(&full_preface[..10]) {
            ClassifyOutcome::NeedMoreBytes { at_least: 24 } => {}
            other => panic!("expected NeedMoreBytes{{at_least: 24}} mid-way, got {other:?}"),
        }
        match h2_classifier.advance(full_preface) {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h2-like"),
            other => panic!("expected Matched(h2-like) with no priority-wait, got {other:?}"),
        }
    }
}
