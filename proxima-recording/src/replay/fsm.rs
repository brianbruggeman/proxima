//! Sans-IO cassette FSM, setup through load.
//!
//! Each state is a struct that owns only the data legal in that state; each
//! transition consumes `self` and returns an outcome enum whose variant is the
//! decision the edge executes (read the file, truncate and re-record, refuse).
//! There is no top-level state enum, so a (state, event) pair the protocol
//! forbids has no method to call (guiding-principles P11).
//!
//! Everything here is `Copy` and allocation-free. What grows with the cassette
//! lives in the caller-owned [`IndexBuilder`], passed to [`Loading::on_event`].
//! Composes no pipe: the loader edge turns recorded events into [`HeaderEvent`]
//! and [`BodyEvent`]s and executes the outcomes; see `proxima-recording`'s
//! `ReplayUpstream` for the transform pipe that serves the sealed index.

use serde::{Deserialize, Serialize};

use crate::replay::index::{CassetteIndex, IndexBuilder, IndexEntry, SlotId};
use crate::replay::keying::{
    KeyDigest, KeySchemaVersion, KeySpecDigest, RequestDigest, ResponseDigest,
};
use crate::replay::ledger::{RecordLedger, Unopened};

/// Cassette header `format` field; equals `RECORDING_FORMAT_VERSION`. A
/// distinct type because it sits beside line numbers and segment ids, all `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatVersion(u32);

impl FormatVersion {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Digest of the declared upstream identity. A distinct type from
/// [`KeySpecDigest`]: both are `u64` fields of one header and swapping them
/// would compare the wrong thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpstreamDigest(u64);

impl UpstreamDigest {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One recording session within a cassette; strictly increasing in file order.
/// Distinct from line numbers in [`LineFault`], which carries both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentId(u32);

impl SegmentId {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(raw) => Some(Self(raw)),
            None => None,
        }
    }
}

/// Wall-clock milliseconds since the Unix epoch. Distinct from [`Millis`]:
/// subtracting two instants gives a duration, adding two instants is a bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnixMillis(u64);

impl UnixMillis {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A duration in milliseconds (an age or an age limit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Millis(u64);

impl Millis {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CassetteMode {
    Strict,
    RecordMissing,
    Verify,
    Rerecord,
}

/// What happens to an existing cassette that is being rebuilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Disposition {
    Truncate,
    Backup,
}

/// What to do when one key is recorded twice with different responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DuplicatePolicy {
    RejectDivergent,
    LastWins,
    FirstWins,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OnStale {
    Fail,
    Warn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Staleness {
    Unbounded,
    MaxAge { max_age: Millis, on_stale: OnStale },
}

/// The identity a cassette is stamped with. One type is both the expectation
/// (what [`Policy`] requires) and the observation (what a file declares), so
/// [`CassetteHeader::mismatch`] compares like with like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassetteHeader {
    pub format: FormatVersion,
    pub key_schema: KeySchemaVersion,
    pub key_spec: KeySpecDigest,
    pub upstream: UpstreamDigest,
}

/// Everything the edge decides before the FSM runs. `now` is read by the edge
/// once, so the FSM never touches a clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub mode: CassetteMode,
    pub duplicates: DuplicatePolicy,
    pub staleness: Staleness,
    pub replace: Disposition,
    pub expect: CassetteHeader,
    pub now: UnixMillis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Absent,
    Present,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordedAt {
    At(UnixMillis),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentStamp {
    pub segment: SegmentId,
    pub recorded_at: RecordedAt,
}

/// What the loader read off a `cassette-entry` stamp and its interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedEntry {
    pub key: KeyDigest,
    pub request: RequestDigest,
    pub segment: SegmentId,
    pub response: ResponseDigest,
}

/// A cassette line the loader could not turn into an event. `line` is the
/// 1-based line in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineFault {
    Malformed { line: u32 },
    FormatVersion { line: u32, found: FormatVersion },
    SegmentOrder { line: u32, segment: SegmentId },
    UnknownSegment { line: u32, segment: SegmentId },
    DanglingInteraction { line: u32 },
    SegmentsExhausted { newest: SegmentId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderEvent {
    Header(CassetteHeader),
    Headerless,
    Fault(LineFault),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BodyEvent<R> {
    Segment {
        line: u32,
        stamp: SegmentStamp,
    },
    Entry {
        line: u32,
        entry: LoadedEntry,
        payload: R,
    },
    Fault(LineFault),
    End,
}

/// First header field that differs from the expectation, in the order format,
/// key schema, key spec, upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mismatch {
    Unversioned,
    Format {
        expected: FormatVersion,
        found: FormatVersion,
    },
    KeySchema {
        expected: KeySchemaVersion,
        found: KeySchemaVersion,
    },
    KeySpec {
        expected: KeySpecDigest,
        found: KeySpecDigest,
    },
    Upstream {
        expected: UpstreamDigest,
        found: UpstreamDigest,
    },
}

impl CassetteHeader {
    /// `self` is the expectation, `found` the observation.
    #[must_use]
    pub fn mismatch(&self, found: &CassetteHeader) -> Option<Mismatch> {
        if self.format != found.format {
            return Some(Mismatch::Format {
                expected: self.format,
                found: found.format,
            });
        }
        if self.key_schema != found.key_schema {
            return Some(Mismatch::KeySchema {
                expected: self.key_schema,
                found: found.key_schema,
            });
        }
        if self.key_spec != found.key_spec {
            return Some(Mismatch::KeySpec {
                expected: self.key_spec,
                found: found.key_spec,
            });
        }
        if self.upstream != found.upstream {
            return Some(Mismatch::Upstream {
                expected: self.upstream,
                found: found.upstream,
            });
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Age {
    Known(Millis),
    Unknown,
}

/// Why the FSM will not proceed. Typed and `Copy`; the edge renders it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    MissingCassette,
    Version(Mismatch),
    Unreadable(LineFault),
    Miss {
        key: KeyDigest,
    },
    Stale {
        key: KeyDigest,
        age: Age,
        max_age: Millis,
    },
    DivergentDuplicate {
        key: KeyDigest,
        segment: SegmentId,
        prior: ResponseDigest,
        incoming: ResponseDigest,
    },
    KeyCollision {
        key: KeyDigest,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordScope {
    Missing,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayServing {
    staleness: Staleness,
    now: UnixMillis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordServing {
    scope: RecordScope,
    staleness: Staleness,
    now: UnixMillis,
    session: SegmentId,
    duplicates: DuplicatePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyServing {
    staleness: Staleness,
    now: UnixMillis,
}

impl RecordServing {
    #[must_use]
    pub const fn scope(&self) -> RecordScope {
        self.scope
    }

    #[must_use]
    pub const fn session(&self) -> SegmentId {
        self.session
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ready {
    Replay(ReplayServing),
    Record(RecordServing),
    Verify(VerifyServing),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshStart {
    Create,
    Replace(Disposition),
}

/// Why a cassette is being rebuilt. Keeps "the file was absent" and "the
/// caller asked for a re-record" apart from "the file was invalid".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildCause {
    Missing,
    Requested,
    Invalid(Mismatch),
}

/// The edge rewrites the cassette: dispose of the old file per `fresh`, write
/// a header, then serve with `ready`. Reused by every outcome that can end in
/// a rebuild so the edge has one executor for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rebuild {
    pub cause: RebuildCause,
    pub fresh: FreshStart,
    pub ready: RecordServing,
}

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opened {
    ReadCassette(ExpectingHeader),
    Rebuild(Rebuild),
    Refuse(Refusal),
}

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderOutcome {
    Accepted(Loading),
    Rebuild(Rebuild),
    Refuse(Refusal),
}

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOutcome {
    Continue(Loading),
    Ready(Ready),
    Rebuild(Rebuild),
    Refuse(Refusal),
}

/// What one duplicate key does to the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Insert,
    Supersede,
    Replace,
    KeepPrior,
    Refuse(Refusal),
}

impl DuplicatePolicy {
    /// The one duplicate rule, used at load and (in the ledger) at record.
    /// Newer segments supersede, older ones lose, an identical digest in the
    /// same segment is a no-op under every policy, and only a divergent
    /// same-segment pair consults the policy.
    #[must_use]
    pub fn resolve(
        self,
        key: KeyDigest,
        prior: Option<(SegmentId, ResponseDigest)>,
        incoming: (SegmentId, ResponseDigest),
    ) -> Resolution {
        let Some((prior_segment, prior_response)) = prior else {
            return Resolution::Insert;
        };
        let (incoming_segment, incoming_response) = incoming;
        if incoming_segment > prior_segment {
            return Resolution::Supersede;
        }
        if incoming_segment < prior_segment || incoming_response == prior_response {
            return Resolution::KeepPrior;
        }
        match self {
            Self::RejectDivergent => Resolution::Refuse(Refusal::DivergentDuplicate {
                key,
                segment: incoming_segment,
                prior: prior_response,
                incoming: incoming_response,
            }),
            Self::LastWins => Resolution::Replace,
            Self::FirstWins => Resolution::KeepPrior,
        }
    }
}

impl Policy {
    fn record_scope(&self) -> RecordScope {
        match self.mode {
            CassetteMode::Rerecord => RecordScope::All,
            CassetteMode::Strict | CassetteMode::RecordMissing | CassetteMode::Verify => {
                RecordScope::Missing
            }
        }
    }

    fn record_serving(&self, session: SegmentId) -> RecordServing {
        RecordServing {
            scope: self.record_scope(),
            staleness: self.staleness,
            now: self.now,
            session,
            duplicates: self.duplicates,
        }
    }

    fn rebuild(&self, cause: RebuildCause, fresh: FreshStart) -> Rebuild {
        Rebuild {
            cause,
            fresh,
            ready: self.record_serving(SegmentId(0)),
        }
    }

    // strict and verify never write, so an invalid cassette is a refusal
    fn on_invalid(&self, cause: Mismatch) -> Result<Rebuild, Refusal> {
        match self.mode {
            CassetteMode::Strict | CassetteMode::Verify => Err(Refusal::Version(cause)),
            CassetteMode::RecordMissing | CassetteMode::Rerecord => Ok(self.rebuild(
                RebuildCause::Invalid(cause),
                FreshStart::Replace(self.replace),
            )),
        }
    }

    fn ready(&self, session: SegmentId) -> Ready {
        match self.mode {
            CassetteMode::Strict => Ready::Replay(ReplayServing {
                staleness: self.staleness,
                now: self.now,
            }),
            CassetteMode::Verify => Ready::Verify(VerifyServing {
                staleness: self.staleness,
                now: self.now,
            }),
            CassetteMode::RecordMissing | CassetteMode::Rerecord => {
                Ready::Record(self.record_serving(session))
            }
        }
    }
}

/// Entry state: nothing is known except the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwaitingPresence {
    policy: Policy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectingHeader {
    policy: Policy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loading {
    policy: Policy,
    newest: Option<SegmentId>,
}

impl AwaitingPresence {
    #[must_use]
    pub const fn new(policy: Policy) -> Self {
        Self { policy }
    }

    pub fn on_presence(self, presence: Presence) -> Opened {
        let policy = self.policy;
        match (policy.mode, presence) {
            (CassetteMode::Strict | CassetteMode::Verify, Presence::Absent) => {
                Opened::Refuse(Refusal::MissingCassette)
            }
            (
                CassetteMode::Strict | CassetteMode::Verify | CassetteMode::RecordMissing,
                Presence::Present,
            ) => Opened::ReadCassette(ExpectingHeader { policy }),
            (CassetteMode::RecordMissing, Presence::Absent) => {
                Opened::Rebuild(policy.rebuild(RebuildCause::Missing, FreshStart::Create))
            }
            (CassetteMode::Rerecord, Presence::Absent) => {
                Opened::Rebuild(policy.rebuild(RebuildCause::Requested, FreshStart::Create))
            }
            (CassetteMode::Rerecord, Presence::Present) => Opened::Rebuild(
                policy.rebuild(RebuildCause::Requested, FreshStart::Replace(policy.replace)),
            ),
        }
    }
}

impl ExpectingHeader {
    pub fn on_header(self, event: HeaderEvent) -> HeaderOutcome {
        let policy = self.policy;
        let cause = match event {
            HeaderEvent::Header(found) => match policy.expect.mismatch(&found) {
                Some(mismatch) => mismatch,
                None => {
                    return HeaderOutcome::Accepted(Loading {
                        policy,
                        newest: None,
                    });
                }
            },
            HeaderEvent::Headerless => Mismatch::Unversioned,
            HeaderEvent::Fault(LineFault::FormatVersion { found, .. }) => Mismatch::Format {
                expected: policy.expect.format,
                found,
            },
            HeaderEvent::Fault(other) => {
                return HeaderOutcome::Refuse(Refusal::Unreadable(other));
            }
        };
        match policy.on_invalid(cause) {
            Ok(rebuild) => HeaderOutcome::Rebuild(rebuild),
            Err(refusal) => HeaderOutcome::Refuse(refusal),
        }
    }
}

impl Loading {
    /// Feeds one cassette line into the caller-owned `builder`.
    ///
    /// A `Rebuild` or `Refuse` outcome leaves `builder` holding the entries
    /// read so far; the edge drops it.
    pub fn on_event<R>(self, event: BodyEvent<R>, builder: &mut IndexBuilder<R>) -> LoadOutcome {
        match event {
            BodyEvent::Segment { line, stamp } => self.on_segment(line, stamp, builder),
            BodyEvent::Entry {
                line,
                entry,
                payload,
            } => self.on_entry(line, entry, payload, builder),
            BodyEvent::Fault(LineFault::FormatVersion { found, .. }) => {
                let cause = Mismatch::Format {
                    expected: self.policy.expect.format,
                    found,
                };
                match self.policy.on_invalid(cause) {
                    Ok(rebuild) => LoadOutcome::Rebuild(rebuild),
                    Err(refusal) => LoadOutcome::Refuse(refusal),
                }
            }
            BodyEvent::Fault(other) => LoadOutcome::Refuse(Refusal::Unreadable(other)),
            BodyEvent::End => self.on_end(),
        }
    }

    fn on_segment<R>(
        self,
        line: u32,
        stamp: SegmentStamp,
        builder: &mut IndexBuilder<R>,
    ) -> LoadOutcome {
        if self.newest.is_some_and(|newest| stamp.segment <= newest) {
            return LoadOutcome::Refuse(Refusal::Unreadable(LineFault::SegmentOrder {
                line,
                segment: stamp.segment,
            }));
        }
        builder.declare(stamp);
        LoadOutcome::Continue(Self {
            newest: Some(stamp.segment),
            ..self
        })
    }

    fn on_entry<R>(
        self,
        line: u32,
        entry: LoadedEntry,
        payload: R,
        builder: &mut IndexBuilder<R>,
    ) -> LoadOutcome {
        if !builder.is_declared(entry.segment) {
            return LoadOutcome::Refuse(Refusal::Unreadable(LineFault::UnknownSegment {
                line,
                segment: entry.segment,
            }));
        }
        let prior = builder.get(entry.key);
        if prior.is_some_and(|held| held.request != entry.request) {
            return LoadOutcome::Refuse(Refusal::KeyCollision { key: entry.key });
        }
        let held = prior.map(|held| (held.segment, held.response));
        match self
            .policy
            .duplicates
            .resolve(entry.key, held, (entry.segment, entry.response))
        {
            Resolution::Insert | Resolution::Supersede | Resolution::Replace => {
                builder.put(IndexEntry {
                    key: entry.key,
                    request: entry.request,
                    segment: entry.segment,
                    response: entry.response,
                    payload,
                });
                LoadOutcome::Continue(self)
            }
            Resolution::KeepPrior => LoadOutcome::Continue(self),
            Resolution::Refuse(refusal) => LoadOutcome::Refuse(refusal),
        }
    }

    fn on_end(self) -> LoadOutcome {
        let Some(newest) = self.newest else {
            return LoadOutcome::Ready(self.policy.ready(SegmentId(0)));
        };
        match newest.next() {
            Some(session) => LoadOutcome::Ready(self.policy.ready(session)),
            None => {
                LoadOutcome::Refuse(Refusal::Unreadable(LineFault::SegmentsExhausted { newest }))
            }
        }
    }
}

/// How old a served entry is against the staleness policy. Keeps the age and
/// the limit, which a bare `bool` would destroy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    Fresh,
    Expired { age: Age, max_age: Millis },
}

/// A replay hit: the entry's slot in the caller's [`CassetteIndex`] and how
/// fresh it is. The edge reads the payload with `index.get(slot)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub slot: SlotId,
    pub freshness: Freshness,
}

/// What the cassette already holds for a key; the ledger's duplicate rule and
/// verification both consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prior {
    Unrecorded,
    Recorded {
        slot: SlotId,
        segment: SegmentId,
        response: ResponseDigest,
        freshness: Freshness,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveRoute {
    Serve(SlotId),
    CallLive(Prior),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyFinding {
    Matches {
        key: KeyDigest,
        freshness: Freshness,
    },
    Unrecorded {
        key: KeyDigest,
    },
    Diverged {
        key: KeyDigest,
        recorded: ResponseDigest,
        live: ResponseDigest,
        freshness: Freshness,
    },
}

impl Staleness {
    fn classify(self, now: UnixMillis, recorded_at: RecordedAt) -> Freshness {
        let Self::MaxAge { max_age, .. } = self else {
            return Freshness::Fresh;
        };
        let age = match recorded_at {
            RecordedAt::At(at) => Age::Known(Millis(now.0.saturating_sub(at.0))),
            RecordedAt::Unknown => Age::Unknown,
        };
        match age {
            Age::Known(known) if known <= max_age => Freshness::Fresh,
            Age::Known(_) | Age::Unknown => Freshness::Expired { age, max_age },
        }
    }
}

// the key picks the slot; the request digest proves the slot belongs to this
// request, so a 64-bit key collision is refused rather than served (R1)
fn recorded<R>(
    index: &CassetteIndex<R>,
    key: KeyDigest,
    request: RequestDigest,
) -> Result<Option<(SlotId, &IndexEntry<R>)>, Refusal> {
    let found = index
        .find(key)
        .and_then(|slot| index.get(slot).map(|entry| (slot, entry)));
    match found {
        Some((_, entry)) if entry.request != request => Err(Refusal::KeyCollision { key }),
        other => Ok(other),
    }
}

fn freshness_of<R>(
    staleness: Staleness,
    now: UnixMillis,
    index: &CassetteIndex<R>,
    entry: &IndexEntry<R>,
) -> Freshness {
    let segments = index.segments();
    let recorded_at = segments
        .binary_search_by_key(&entry.segment, |stamp| stamp.segment)
        .ok()
        .and_then(|position| segments.get(position))
        .map_or(RecordedAt::Unknown, |stamp| stamp.recorded_at);
    staleness.classify(now, recorded_at)
}

fn prior_of<R>(
    staleness: Staleness,
    now: UnixMillis,
    index: &CassetteIndex<R>,
    slot: SlotId,
    entry: &IndexEntry<R>,
) -> Prior {
    Prior::Recorded {
        slot,
        segment: entry.segment,
        response: entry.response,
        freshness: freshness_of(staleness, now, index, entry),
    }
}

impl ReplayServing {
    /// Per-request read; allocates nothing. `request` is the live request's
    /// [`RequestDigest`]: a hit whose stored digest differs is `KeyCollision`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Miss`] when the key is unrecorded, [`Refusal::KeyCollision`]
    /// on a digest mismatch, [`Refusal::Stale`] when the hit is expired under
    /// `OnStale::Fail`.
    pub fn lookup<R>(
        &self,
        index: &CassetteIndex<R>,
        key: KeyDigest,
        request: RequestDigest,
    ) -> Result<Hit, Refusal> {
        let (slot, entry) = recorded(index, key, request)?.ok_or(Refusal::Miss { key })?;
        let freshness = freshness_of(self.staleness, self.now, index, entry);
        match (freshness, self.staleness) {
            (
                Freshness::Expired { age, max_age },
                Staleness::MaxAge {
                    on_stale: OnStale::Fail,
                    ..
                },
            ) => Err(Refusal::Stale { key, age, max_age }),
            _ => Ok(Hit { slot, freshness }),
        }
    }
}

impl RecordServing {
    /// Serves a fresh hit; an expired hit or a miss goes live, carrying what
    /// the cassette held so the ledger can apply the duplicate rule.
    ///
    /// # Errors
    ///
    /// [`Refusal::KeyCollision`] when the indexed entry's request digest
    /// differs from the live request's; a collision is never served.
    pub fn route<R>(
        &self,
        index: &CassetteIndex<R>,
        key: KeyDigest,
        request: RequestDigest,
    ) -> Result<LiveRoute, Refusal> {
        if self.scope == RecordScope::All {
            return Ok(LiveRoute::CallLive(Prior::Unrecorded));
        }
        let Some((slot, entry)) = recorded(index, key, request)? else {
            return Ok(LiveRoute::CallLive(Prior::Unrecorded));
        };
        Ok(
            match prior_of(self.staleness, self.now, index, slot, entry) {
                Prior::Recorded {
                    freshness: Freshness::Fresh,
                    ..
                } => LiveRoute::Serve(slot),
                other => LiveRoute::CallLive(other),
            },
        )
    }

    #[must_use]
    pub fn open_ledger(&self) -> RecordLedger<Unopened> {
        RecordLedger::new(self.session, self.duplicates, self.now)
    }
}

impl VerifyServing {
    /// Looks up the recorded side of a verify comparison.
    ///
    /// # Errors
    ///
    /// [`Refusal::KeyCollision`] when the indexed entry's request digest
    /// differs from the live request's; a collision is never compared.
    pub fn route<R>(
        &self,
        index: &CassetteIndex<R>,
        key: KeyDigest,
        request: RequestDigest,
    ) -> Result<Prior, Refusal> {
        Ok(match recorded(index, key, request)? {
            Some((slot, entry)) => prior_of(self.staleness, self.now, index, slot, entry),
            None => Prior::Unrecorded,
        })
    }

    #[must_use]
    pub fn verify(&self, key: KeyDigest, prior: Prior, live: ResponseDigest) -> VerifyFinding {
        match prior {
            Prior::Unrecorded => VerifyFinding::Unrecorded { key },
            Prior::Recorded {
                response,
                freshness,
                ..
            } if response == live => VerifyFinding::Matches { key, freshness },
            Prior::Recorded {
                response,
                freshness,
                ..
            } => VerifyFinding::Diverged {
                key,
                recorded: response,
                live,
                freshness,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::index::IndexBuilder;
    use crate::replay::keying::{HeaderAllowlist, request_digest, request_key};
    use bytes::Bytes;
    use proxima_primitives::pipe::method::Method;
    use proxima_primitives::pipe::request::Request;

    const NOW: u64 = 1_791_676_800_000;
    const SEGMENT_ZERO_AT: u64 = 1_790_812_800_000;
    const THIRTY_DAYS: u64 = 2_592_000_000;

    fn accept_allowlist() -> HeaderAllowlist {
        HeaderAllowlist::new(["accept"]).expect("accept is a valid header name")
    }

    fn forecast(city: &str) -> Request<Bytes> {
        let mut request = Request::builder()
            .method(Method::Get)
            .path("/v1/forecast")
            .build()
            .expect("a method and path are enough to build a request");
        request.query = alloc::vec![(
            Bytes::from_static(b"city"),
            Bytes::copy_from_slice(city.as_bytes())
        )]
        .into();
        request
    }

    fn expected_header() -> CassetteHeader {
        CassetteHeader {
            format: FormatVersion(3),
            key_schema: crate::replay::keying::KEY_SCHEMA_VERSION,
            key_spec: accept_allowlist().digest(),
            upstream: UpstreamDigest(0),
        }
    }

    fn policy(mode: CassetteMode) -> Policy {
        Policy {
            mode,
            duplicates: DuplicatePolicy::RejectDivergent,
            staleness: Staleness::MaxAge {
                max_age: Millis(THIRTY_DAYS),
                on_stale: OnStale::Fail,
            },
            replace: Disposition::Backup,
            expect: expected_header(),
            now: UnixMillis(NOW),
        }
    }

    fn open(mode: CassetteMode, presence: Presence) -> Opened {
        AwaitingPresence::new(policy(mode)).on_presence(presence)
    }

    fn expecting(mode: CassetteMode) -> ExpectingHeader {
        match open(mode, Presence::Present) {
            Opened::ReadCassette(state) => state,
            other => panic!("present cassette under {mode:?} must be read, got {other:?}"),
        }
    }

    fn loading(mode: CassetteMode) -> Loading {
        match expecting(mode).on_header(HeaderEvent::Header(expected_header())) {
            HeaderOutcome::Accepted(state) => state,
            other => panic!("matching header must be accepted, got {other:?}"),
        }
    }

    fn rebuild_for(
        mode: CassetteMode,
        cause: RebuildCause,
        fresh: FreshStart,
        scope: RecordScope,
    ) -> Rebuild {
        Rebuild {
            cause,
            fresh,
            ready: RecordServing {
                scope,
                staleness: policy(mode).staleness,
                now: UnixMillis(NOW),
                session: SegmentId(0),
                duplicates: DuplicatePolicy::RejectDivergent,
            },
        }
    }

    fn stamp(segment: u32) -> SegmentStamp {
        SegmentStamp {
            segment: SegmentId(segment),
            recorded_at: RecordedAt::At(UnixMillis(SEGMENT_ZERO_AT)),
        }
    }

    fn loaded(city: &str, segment: u32, body: &[u8]) -> LoadedEntry {
        let allowlist = accept_allowlist();
        let request = forecast(city);
        LoadedEntry {
            key: request_key(&request, &allowlist),
            request: request_digest(&request, &allowlist),
            segment: SegmentId(segment),
            response: crate::replay::keying::response_digest(200, &[body]),
        }
    }

    fn entry_event(city: &str, segment: u32, body: &[u8]) -> BodyEvent<u32> {
        BodyEvent::Entry {
            line: 5,
            entry: loaded(city, segment, body),
            payload: 0,
        }
    }

    fn continued(outcome: LoadOutcome) -> Loading {
        match outcome {
            LoadOutcome::Continue(state) => state,
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    fn after_segment(mode: CassetteMode, builder: &mut IndexBuilder<u32>) -> Loading {
        let event = BodyEvent::Segment {
            line: 4,
            stamp: stamp(0),
        };
        continued(loading(mode).on_event(event, builder))
    }

    fn end_of(mode: CassetteMode, newest: Option<u32>) -> LoadOutcome {
        let mut builder = IndexBuilder::<u32>::new();
        let state = match newest {
            None => loading(mode),
            Some(segment) => {
                let event = BodyEvent::Segment {
                    line: 4,
                    stamp: stamp(segment),
                };
                continued(loading(mode).on_event(event, &mut builder))
            }
        };
        state.on_event(BodyEvent::End, &mut builder)
    }

    // setup: 8 rows

    #[test]
    fn setup_strict_absent_refuses_missing_cassette() {
        assert_eq!(
            open(CassetteMode::Strict, Presence::Absent),
            Opened::Refuse(Refusal::MissingCassette)
        );
    }

    #[test]
    fn setup_strict_present_reads_the_cassette() {
        let outcome = open(CassetteMode::Strict, Presence::Present);
        assert!(matches!(outcome, Opened::ReadCassette(_)), "{outcome:?}");
    }

    #[test]
    fn setup_verify_absent_refuses_missing_cassette() {
        assert_eq!(
            open(CassetteMode::Verify, Presence::Absent),
            Opened::Refuse(Refusal::MissingCassette)
        );
    }

    #[test]
    fn setup_verify_present_reads_the_cassette() {
        let outcome = open(CassetteMode::Verify, Presence::Present);
        assert!(matches!(outcome, Opened::ReadCassette(_)), "{outcome:?}");
    }

    #[test]
    fn setup_record_missing_absent_creates_and_records_missing_in_session_zero() {
        assert_eq!(
            open(CassetteMode::RecordMissing, Presence::Absent),
            Opened::Rebuild(rebuild_for(
                CassetteMode::RecordMissing,
                RebuildCause::Missing,
                FreshStart::Create,
                RecordScope::Missing
            ))
        );
    }

    #[test]
    fn setup_record_missing_present_reads_the_cassette() {
        let outcome = open(CassetteMode::RecordMissing, Presence::Present);
        assert!(matches!(outcome, Opened::ReadCassette(_)), "{outcome:?}");
    }

    #[test]
    fn setup_rerecord_absent_creates_and_records_all_in_session_zero() {
        assert_eq!(
            open(CassetteMode::Rerecord, Presence::Absent),
            Opened::Rebuild(rebuild_for(
                CassetteMode::Rerecord,
                RebuildCause::Requested,
                FreshStart::Create,
                RecordScope::All
            ))
        );
    }

    #[test]
    fn setup_rerecord_present_replaces_without_reading_the_file() {
        assert_eq!(
            open(CassetteMode::Rerecord, Presence::Present),
            Opened::Rebuild(rebuild_for(
                CassetteMode::Rerecord,
                RebuildCause::Requested,
                FreshStart::Replace(Disposition::Backup),
                RecordScope::All
            ))
        );
    }

    // header: 11 rows

    #[test]
    fn header_equal_is_accepted_in_every_read_mode() {
        for mode in [
            CassetteMode::Strict,
            CassetteMode::Verify,
            CassetteMode::RecordMissing,
        ] {
            let outcome = expecting(mode).on_header(HeaderEvent::Header(expected_header()));
            assert!(
                matches!(outcome, HeaderOutcome::Accepted(_)),
                "{mode:?}: {outcome:?}"
            );
        }
    }

    fn stale_key_spec_header() -> (CassetteHeader, Mismatch) {
        let found = CassetteHeader {
            key_spec: HeaderAllowlist::new(["accept", "x-api-version"])
                .expect("valid header names")
                .digest(),
            ..expected_header()
        };
        let mismatch = Mismatch::KeySpec {
            expected: expected_header().key_spec,
            found: found.key_spec,
        };
        (found, mismatch)
    }

    #[test]
    fn header_differs_under_strict_refuses_with_the_first_difference() {
        let (found, mismatch) = stale_key_spec_header();
        let outcome = expecting(CassetteMode::Strict).on_header(HeaderEvent::Header(found));
        assert_eq!(outcome, HeaderOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn header_differs_under_verify_refuses_with_the_first_difference() {
        let (found, mismatch) = stale_key_spec_header();
        let outcome = expecting(CassetteMode::Verify).on_header(HeaderEvent::Header(found));
        assert_eq!(outcome, HeaderOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn header_differs_under_record_missing_rebuilds_with_replace() {
        let (found, mismatch) = stale_key_spec_header();
        let outcome = expecting(CassetteMode::RecordMissing).on_header(HeaderEvent::Header(found));
        assert_eq!(
            outcome,
            HeaderOutcome::Rebuild(rebuild_for(
                CassetteMode::RecordMissing,
                RebuildCause::Invalid(mismatch),
                FreshStart::Replace(Disposition::Backup),
                RecordScope::Missing
            ))
        );
    }

    #[test]
    fn header_headerless_under_strict_refuses_unversioned() {
        let outcome = expecting(CassetteMode::Strict).on_header(HeaderEvent::Headerless);
        assert_eq!(
            outcome,
            HeaderOutcome::Refuse(Refusal::Version(Mismatch::Unversioned))
        );
    }

    #[test]
    fn header_headerless_under_verify_refuses_unversioned() {
        let outcome = expecting(CassetteMode::Verify).on_header(HeaderEvent::Headerless);
        assert_eq!(
            outcome,
            HeaderOutcome::Refuse(Refusal::Version(Mismatch::Unversioned))
        );
    }

    #[test]
    fn header_headerless_under_record_missing_rebuilds_unversioned() {
        let outcome = expecting(CassetteMode::RecordMissing).on_header(HeaderEvent::Headerless);
        assert_eq!(
            outcome,
            HeaderOutcome::Rebuild(rebuild_for(
                CassetteMode::RecordMissing,
                RebuildCause::Invalid(Mismatch::Unversioned),
                FreshStart::Replace(Disposition::Backup),
                RecordScope::Missing
            ))
        );
    }

    fn future_format_fault() -> (HeaderEvent, Mismatch) {
        let found = FormatVersion(4);
        let mismatch = Mismatch::Format {
            expected: FormatVersion(3),
            found,
        };
        (
            HeaderEvent::Fault(LineFault::FormatVersion { line: 1, found }),
            mismatch,
        )
    }

    #[test]
    fn header_format_fault_under_strict_refuses_format() {
        let (event, mismatch) = future_format_fault();
        let outcome = expecting(CassetteMode::Strict).on_header(event);
        assert_eq!(outcome, HeaderOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn header_format_fault_under_verify_refuses_format() {
        let (event, mismatch) = future_format_fault();
        let outcome = expecting(CassetteMode::Verify).on_header(event);
        assert_eq!(outcome, HeaderOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn header_format_fault_under_record_missing_rebuilds_format() {
        let (event, mismatch) = future_format_fault();
        let outcome = expecting(CassetteMode::RecordMissing).on_header(event);
        assert_eq!(
            outcome,
            HeaderOutcome::Rebuild(rebuild_for(
                CassetteMode::RecordMissing,
                RebuildCause::Invalid(mismatch),
                FreshStart::Replace(Disposition::Backup),
                RecordScope::Missing
            ))
        );
    }

    #[test]
    fn header_other_fault_refuses_unreadable_in_every_mode_so_corruption_never_auto_destroys() {
        let fault = LineFault::Malformed { line: 1 };
        for mode in [
            CassetteMode::Strict,
            CassetteMode::Verify,
            CassetteMode::RecordMissing,
        ] {
            let outcome = expecting(mode).on_header(HeaderEvent::Fault(fault));
            assert_eq!(
                outcome,
                HeaderOutcome::Refuse(Refusal::Unreadable(fault)),
                "{mode:?}"
            );
        }
    }

    // load: 14 rows (12 designed + key collision + segment-id exhaustion)

    #[test]
    fn load_segment_newer_than_newest_continues() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let event = BodyEvent::Segment {
            line: 9,
            stamp: stamp(1),
        };
        let outcome = state.on_event(event, &mut builder);
        assert!(matches!(outcome, LoadOutcome::Continue(_)), "{outcome:?}");
        assert!(builder.is_declared(SegmentId(1)));
    }

    #[test]
    fn load_segment_not_greater_refuses_segment_order() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let event = BodyEvent::Segment {
            line: 9,
            stamp: stamp(0),
        };
        assert_eq!(
            state.on_event(event, &mut builder),
            LoadOutcome::Refuse(Refusal::Unreadable(LineFault::SegmentOrder {
                line: 9,
                segment: SegmentId(0)
            }))
        );
    }

    #[test]
    fn load_entry_naming_undeclared_segment_refuses_unknown_segment() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let outcome = state.on_event(entry_event("oslo", 3, b"{\"temp\":12}"), &mut builder);
        assert_eq!(
            outcome,
            LoadOutcome::Refuse(Refusal::Unreadable(LineFault::UnknownSegment {
                line: 5,
                segment: SegmentId(3)
            }))
        );
    }

    #[test]
    fn load_entry_is_inserted_and_continues() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let outcome = state.on_event(entry_event("oslo", 0, b"{\"temp\":12}"), &mut builder);
        assert!(matches!(outcome, LoadOutcome::Continue(_)), "{outcome:?}");
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn load_divergent_duplicate_under_reject_refuses() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let state =
            continued(state.on_event(entry_event("oslo", 0, b"{\"temp\":12}"), &mut builder));
        let outcome = state.on_event(entry_event("oslo", 0, b"{\"temp\":13}"), &mut builder);
        let first = loaded("oslo", 0, b"{\"temp\":12}");
        let second = loaded("oslo", 0, b"{\"temp\":13}");
        assert_eq!(
            outcome,
            LoadOutcome::Refuse(Refusal::DivergentDuplicate {
                key: first.key,
                segment: SegmentId(0),
                prior: first.response,
                incoming: second.response
            })
        );
    }

    #[test]
    fn load_key_collision_with_different_request_digest_refuses() {
        let mut builder = IndexBuilder::<u32>::new();
        let state = after_segment(CassetteMode::Strict, &mut builder);
        let state =
            continued(state.on_event(entry_event("oslo", 0, b"{\"temp\":12}"), &mut builder));
        let genuine = loaded("oslo", 0, b"{\"temp\":12}");
        let impostor = LoadedEntry {
            request: loaded("bergen", 0, b"{\"temp\":12}").request,
            ..genuine
        };
        let event = BodyEvent::Entry {
            line: 6,
            entry: impostor,
            payload: 1,
        };
        assert_eq!(
            state.on_event(event, &mut builder),
            LoadOutcome::Refuse(Refusal::KeyCollision { key: genuine.key })
        );
    }

    #[test]
    fn load_format_fault_under_strict_refuses_format() {
        let mut builder = IndexBuilder::<u32>::new();
        let (HeaderEvent::Fault(fault), mismatch) = future_format_fault() else {
            panic!("future_format_fault yields a fault event");
        };
        let outcome = loading(CassetteMode::Strict).on_event(BodyEvent::Fault(fault), &mut builder);
        assert_eq!(outcome, LoadOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn load_format_fault_under_verify_refuses_format() {
        let mut builder = IndexBuilder::<u32>::new();
        let (HeaderEvent::Fault(fault), mismatch) = future_format_fault() else {
            panic!("future_format_fault yields a fault event");
        };
        let outcome = loading(CassetteMode::Verify).on_event(BodyEvent::Fault(fault), &mut builder);
        assert_eq!(outcome, LoadOutcome::Refuse(Refusal::Version(mismatch)));
    }

    #[test]
    fn load_format_fault_under_record_missing_rebuilds() {
        let mut builder = IndexBuilder::<u32>::new();
        let (HeaderEvent::Fault(fault), mismatch) = future_format_fault() else {
            panic!("future_format_fault yields a fault event");
        };
        let outcome =
            loading(CassetteMode::RecordMissing).on_event(BodyEvent::Fault(fault), &mut builder);
        assert_eq!(
            outcome,
            LoadOutcome::Rebuild(rebuild_for(
                CassetteMode::RecordMissing,
                RebuildCause::Invalid(mismatch),
                FreshStart::Replace(Disposition::Backup),
                RecordScope::Missing
            ))
        );
    }

    #[test]
    fn load_other_fault_refuses_unreadable() {
        let mut builder = IndexBuilder::<u32>::new();
        let fault = LineFault::DanglingInteraction { line: 7 };
        let outcome =
            loading(CassetteMode::RecordMissing).on_event(BodyEvent::Fault(fault), &mut builder);
        assert_eq!(outcome, LoadOutcome::Refuse(Refusal::Unreadable(fault)));
    }

    #[test]
    fn load_end_under_strict_is_ready_to_replay() {
        let outcome = end_of(CassetteMode::Strict, Some(0));
        assert!(
            matches!(outcome, LoadOutcome::Ready(Ready::Replay(_))),
            "{outcome:?}"
        );
    }

    #[test]
    fn load_end_under_verify_is_ready_to_verify() {
        let outcome = end_of(CassetteMode::Verify, Some(0));
        assert!(
            matches!(outcome, LoadOutcome::Ready(Ready::Verify(_))),
            "{outcome:?}"
        );
    }

    #[test]
    fn load_end_under_record_missing_records_into_the_session_after_newest() {
        let LoadOutcome::Ready(Ready::Record(serving)) =
            end_of(CassetteMode::RecordMissing, Some(2))
        else {
            panic!("record-missing end must be ready to record");
        };
        assert_eq!(serving.session(), SegmentId(3));
        assert_eq!(serving.scope(), RecordScope::Missing);
    }

    #[test]
    fn load_end_with_no_segment_records_into_session_zero() {
        let LoadOutcome::Ready(Ready::Record(serving)) = end_of(CassetteMode::RecordMissing, None)
        else {
            panic!("record-missing end must be ready to record");
        };
        assert_eq!(serving.session(), SegmentId(0));
    }

    #[test]
    fn load_end_after_the_last_segment_id_refuses_exhausted() {
        let newest = SegmentId(u32::MAX);
        assert_eq!(
            end_of(CassetteMode::RecordMissing, Some(u32::MAX)),
            LoadOutcome::Refuse(Refusal::Unreadable(LineFault::SegmentsExhausted { newest }))
        );
    }

    // duplicates: 7 rows

    fn digest(body: &[u8]) -> ResponseDigest {
        crate::replay::keying::response_digest(200, &[body])
    }

    fn resolve(
        policy: DuplicatePolicy,
        prior: Option<(u32, &[u8])>,
        incoming: (u32, &[u8]),
    ) -> Resolution {
        let key = request_key(&forecast("oslo"), &accept_allowlist());
        policy.resolve(
            key,
            prior.map(|(segment, body)| (SegmentId(segment), digest(body))),
            (SegmentId(incoming.0), digest(incoming.1)),
        )
    }

    const TWELVE: &[u8] = b"{\"temp\":12}";
    const THIRTEEN: &[u8] = b"{\"temp\":13}";

    #[test]
    fn duplicates_no_prior_inserts() {
        assert_eq!(
            resolve(DuplicatePolicy::RejectDivergent, None, (0, TWELVE)),
            Resolution::Insert
        );
    }

    #[test]
    fn duplicates_newer_segment_supersedes() {
        assert_eq!(
            resolve(
                DuplicatePolicy::RejectDivergent,
                Some((0, TWELVE)),
                (1, THIRTEEN)
            ),
            Resolution::Supersede
        );
    }

    #[test]
    fn duplicates_older_segment_keeps_prior() {
        assert_eq!(
            resolve(DuplicatePolicy::LastWins, Some((2, TWELVE)), (1, THIRTEEN)),
            Resolution::KeepPrior
        );
    }

    #[test]
    fn duplicates_same_segment_identical_digest_keeps_prior_under_every_policy() {
        for policy in [
            DuplicatePolicy::RejectDivergent,
            DuplicatePolicy::LastWins,
            DuplicatePolicy::FirstWins,
        ] {
            assert_eq!(
                resolve(policy, Some((1, TWELVE)), (1, TWELVE)),
                Resolution::KeepPrior,
                "{policy:?}"
            );
        }
    }

    #[test]
    fn duplicates_same_segment_divergent_reject_refuses_with_both_digests() {
        let key = request_key(&forecast("oslo"), &accept_allowlist());
        assert_eq!(
            resolve(
                DuplicatePolicy::RejectDivergent,
                Some((1, TWELVE)),
                (1, THIRTEEN)
            ),
            Resolution::Refuse(Refusal::DivergentDuplicate {
                key,
                segment: SegmentId(1),
                prior: digest(TWELVE),
                incoming: digest(THIRTEEN)
            })
        );
    }

    #[test]
    fn duplicates_same_segment_divergent_last_wins_replaces() {
        assert_eq!(
            resolve(DuplicatePolicy::LastWins, Some((1, TWELVE)), (1, THIRTEEN)),
            Resolution::Replace
        );
    }

    #[test]
    fn duplicates_same_segment_divergent_first_wins_keeps_prior() {
        assert_eq!(
            resolve(DuplicatePolicy::FirstWins, Some((1, TWELVE)), (1, THIRTEEN)),
            Resolution::KeepPrior
        );
    }

    // serving fixtures

    const SEVEN_DAYS: u64 = 604_800_000;
    const TEN_DAYS: u64 = NOW - SEGMENT_ZERO_AT;

    fn max_age(limit: u64, on_stale: OnStale) -> Staleness {
        Staleness::MaxAge {
            max_age: Millis(limit),
            on_stale,
        }
    }

    fn index_of(recorded_at: RecordedAt, cities: &[(&str, &[u8])]) -> CassetteIndex<u32> {
        let mut builder = IndexBuilder::new();
        builder.declare(SegmentStamp {
            segment: SegmentId(0),
            recorded_at,
        });
        for (position, (city, body)) in cities.iter().enumerate() {
            let held = loaded(city, 0, body);
            builder.put(IndexEntry {
                key: held.key,
                request: held.request,
                segment: held.segment,
                response: held.response,
                payload: u32::try_from(position).expect("a handful of entries fits u32"),
            });
        }
        builder.seal()
    }

    fn recorded_then(cities: &[(&str, &[u8])]) -> CassetteIndex<u32> {
        index_of(RecordedAt::At(UnixMillis(SEGMENT_ZERO_AT)), cities)
    }

    fn key_of(city: &str) -> KeyDigest {
        request_key(&forecast(city), &accept_allowlist())
    }

    fn request_of(city: &str) -> RequestDigest {
        request_digest(&forecast(city), &accept_allowlist())
    }

    fn slot_of(index: &CassetteIndex<u32>, city: &str) -> SlotId {
        index
            .find(key_of(city))
            .expect("the city was put into the index")
    }

    fn replay(staleness: Staleness) -> ReplayServing {
        ReplayServing {
            staleness,
            now: UnixMillis(NOW),
        }
    }

    fn record(scope: RecordScope, session: u32) -> RecordServing {
        RecordServing {
            scope,
            staleness: max_age(THIRTY_DAYS, OnStale::Fail),
            now: UnixMillis(NOW),
            session: SegmentId(session),
            duplicates: DuplicatePolicy::RejectDivergent,
        }
    }

    fn verifying() -> VerifyServing {
        VerifyServing {
            staleness: max_age(THIRTY_DAYS, OnStale::Warn),
            now: UnixMillis(NOW),
        }
    }

    fn expired_known() -> Freshness {
        Freshness::Expired {
            age: Age::Known(Millis(TEN_DAYS)),
            max_age: Millis(SEVEN_DAYS),
        }
    }

    // replay lookup: 7 rows + 1 collision row

    #[test]
    fn lookup_miss_refuses_with_the_key() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        assert_eq!(
            replay(Staleness::Unbounded).lookup(&index, key_of("bergen"), request_of("bergen")),
            Err(Refusal::Miss {
                key: key_of("bergen")
            })
        );
    }

    #[test]
    fn lookup_hit_unbounded_is_fresh_however_old() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        assert_eq!(
            replay(Staleness::Unbounded).lookup(&index, key_of("oslo"), request_of("oslo")),
            Ok(Hit {
                slot: slot_of(&index, "oslo"),
                freshness: Freshness::Fresh
            })
        );
    }

    #[test]
    fn lookup_hit_within_max_age_is_fresh() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = replay(max_age(THIRTY_DAYS, OnStale::Fail));
        assert_eq!(
            serving.lookup(&index, key_of("oslo"), request_of("oslo")),
            Ok(Hit {
                slot: slot_of(&index, "oslo"),
                freshness: Freshness::Fresh
            })
        );
    }

    #[test]
    fn lookup_hit_older_than_max_age_under_fail_refuses_stale_with_the_known_age() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = replay(max_age(SEVEN_DAYS, OnStale::Fail));
        assert_eq!(
            serving.lookup(&index, key_of("oslo"), request_of("oslo")),
            Err(Refusal::Stale {
                key: key_of("oslo"),
                age: Age::Known(Millis(TEN_DAYS)),
                max_age: Millis(SEVEN_DAYS)
            })
        );
    }

    #[test]
    fn lookup_hit_older_than_max_age_under_warn_serves_expired() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = replay(max_age(SEVEN_DAYS, OnStale::Warn));
        assert_eq!(
            serving.lookup(&index, key_of("oslo"), request_of("oslo")),
            Ok(Hit {
                slot: slot_of(&index, "oslo"),
                freshness: expired_known()
            })
        );
    }

    #[test]
    fn lookup_hit_with_unknown_recorded_at_under_fail_refuses_stale_unknown() {
        let index = index_of(RecordedAt::Unknown, &[("oslo", TWELVE)]);
        let serving = replay(max_age(SEVEN_DAYS, OnStale::Fail));
        assert_eq!(
            serving.lookup(&index, key_of("oslo"), request_of("oslo")),
            Err(Refusal::Stale {
                key: key_of("oslo"),
                age: Age::Unknown,
                max_age: Millis(SEVEN_DAYS)
            })
        );
    }

    #[test]
    fn lookup_hit_with_unknown_recorded_at_under_warn_serves_expired_unknown() {
        let index = index_of(RecordedAt::Unknown, &[("oslo", TWELVE)]);
        let serving = replay(max_age(SEVEN_DAYS, OnStale::Warn));
        assert_eq!(
            serving.lookup(&index, key_of("oslo"), request_of("oslo")),
            Ok(Hit {
                slot: slot_of(&index, "oslo"),
                freshness: Freshness::Expired {
                    age: Age::Unknown,
                    max_age: Millis(SEVEN_DAYS)
                }
            })
        );
    }

    // an index whose "oslo" key carries bergen's request digest: the stored
    // witness disagrees with the live request that hashes to the same key
    fn colliding_index() -> CassetteIndex<u32> {
        let mut builder = IndexBuilder::new();
        builder.declare(stamp(0));
        let held = loaded("oslo", 0, TWELVE);
        builder.put(IndexEntry {
            key: held.key,
            request: request_of("bergen"),
            segment: held.segment,
            response: held.response,
            payload: 0,
        });
        builder.seal()
    }

    #[test]
    fn lookup_key_hit_with_a_different_request_digest_refuses_key_collision() {
        assert_eq!(
            replay(Staleness::Unbounded).lookup(
                &colliding_index(),
                key_of("oslo"),
                request_of("oslo")
            ),
            Err(Refusal::KeyCollision {
                key: key_of("oslo")
            })
        );
    }

    // live route: 6 rows + 2 collision rows

    #[test]
    fn route_record_missing_serves_a_fresh_hit() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = record(RecordScope::Missing, 1);
        assert_eq!(
            serving.route(&index, key_of("oslo"), request_of("oslo")),
            Ok(LiveRoute::Serve(slot_of(&index, "oslo")))
        );
    }

    #[test]
    fn route_record_missing_sends_an_expired_hit_live_carrying_the_recorded_prior() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = RecordServing {
            staleness: max_age(SEVEN_DAYS, OnStale::Warn),
            ..record(RecordScope::Missing, 1)
        };
        assert_eq!(
            serving.route(&index, key_of("oslo"), request_of("oslo")),
            Ok(LiveRoute::CallLive(Prior::Recorded {
                slot: slot_of(&index, "oslo"),
                segment: SegmentId(0),
                response: digest(TWELVE),
                freshness: expired_known()
            }))
        );
    }

    #[test]
    fn route_record_missing_sends_a_miss_live_as_unrecorded() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = record(RecordScope::Missing, 1);
        assert_eq!(
            serving.route(&index, key_of("bergen"), request_of("bergen")),
            Ok(LiveRoute::CallLive(Prior::Unrecorded))
        );
    }

    #[test]
    fn route_verify_hit_is_the_recorded_prior() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        assert_eq!(
            verifying().route(&index, key_of("oslo"), request_of("oslo")),
            Ok(Prior::Recorded {
                slot: slot_of(&index, "oslo"),
                segment: SegmentId(0),
                response: digest(TWELVE),
                freshness: Freshness::Fresh
            })
        );
    }

    #[test]
    fn route_verify_miss_is_unrecorded() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        assert_eq!(
            verifying().route(&index, key_of("bergen"), request_of("bergen")),
            Ok(Prior::Unrecorded)
        );
    }

    #[test]
    fn route_record_all_goes_live_as_unrecorded_even_when_the_index_holds_the_key() {
        let index = recorded_then(&[("oslo", TWELVE)]);
        let serving = record(RecordScope::All, 0);
        assert_eq!(
            serving.route(&index, key_of("oslo"), request_of("oslo")),
            Ok(LiveRoute::CallLive(Prior::Unrecorded))
        );
    }

    #[test]
    fn route_record_missing_key_collision_refuses_instead_of_serving() {
        let serving = record(RecordScope::Missing, 1);
        assert_eq!(
            serving.route(&colliding_index(), key_of("oslo"), request_of("oslo")),
            Err(Refusal::KeyCollision {
                key: key_of("oslo")
            })
        );
    }

    #[test]
    fn route_verify_key_collision_refuses_instead_of_comparing() {
        assert_eq!(
            verifying().route(&colliding_index(), key_of("oslo"), request_of("oslo")),
            Err(Refusal::KeyCollision {
                key: key_of("oslo")
            })
        );
    }

    // ledger: 4 rows + 2 unopened rows

    fn prior_at(segment: u32, body: &[u8]) -> Prior {
        let index = recorded_then(&[("oslo", body)]);
        Prior::Recorded {
            slot: slot_of(&index, "oslo"),
            segment: SegmentId(segment),
            response: digest(body),
            freshness: Freshness::Fresh,
        }
    }

    fn opened(
        admission: crate::replay::ledger::Admission,
    ) -> RecordLedger<crate::replay::ledger::Open> {
        match admission {
            crate::replay::ledger::Admission::Opened { ledger, .. } => ledger,
            other => panic!("first append must open the session, got {other:?}"),
        }
    }

    #[test]
    fn ledger_first_append_opens_the_session_and_stamps_its_segment() {
        let ledger = record(RecordScope::Missing, 3).open_ledger();
        let admission = ledger.admit(key_of("oslo"), Prior::Unrecorded, digest(TWELVE));
        match admission {
            crate::replay::ledger::Admission::Opened { stamp, .. } => assert_eq!(
                stamp,
                SegmentStamp {
                    segment: SegmentId(3),
                    recorded_at: RecordedAt::At(UnixMillis(NOW))
                }
            ),
            other => panic!("expected Opened, got {other:?}"),
        }
    }

    #[test]
    fn ledger_later_appends_need_no_second_stamp() {
        let ledger = record(RecordScope::Missing, 3).open_ledger();
        let mut open = opened(ledger.admit(key_of("oslo"), Prior::Unrecorded, digest(TWELVE)));
        assert_eq!(
            open.admit(key_of("bergen"), Prior::Unrecorded, digest(THIRTEEN)),
            crate::replay::ledger::AppendDecision::Append
        );
    }

    #[test]
    fn ledger_same_key_same_response_in_one_session_skips_identical() {
        let ledger = record(RecordScope::Missing, 3).open_ledger();
        let mut open = opened(ledger.admit(key_of("oslo"), Prior::Unrecorded, digest(TWELVE)));
        assert_eq!(
            open.admit(key_of("oslo"), Prior::Unrecorded, digest(TWELVE)),
            crate::replay::ledger::AppendDecision::SkipIdentical
        );
    }

    #[test]
    fn ledger_same_key_divergent_response_in_one_session_refuses_under_reject() {
        let ledger = record(RecordScope::Missing, 3).open_ledger();
        let mut open = opened(ledger.admit(key_of("oslo"), Prior::Unrecorded, digest(TWELVE)));
        assert_eq!(
            open.admit(key_of("oslo"), Prior::Unrecorded, digest(THIRTEEN)),
            crate::replay::ledger::AppendDecision::Refuse(Refusal::DivergentDuplicate {
                key: key_of("oslo"),
                segment: SegmentId(3),
                prior: digest(TWELVE),
                incoming: digest(THIRTEEN)
            })
        );
    }

    #[test]
    fn ledger_unopened_skip_hands_back_the_unopened_ledger() {
        let ledger = record(RecordScope::Missing, 1).open_ledger();
        let admission = ledger.admit(key_of("oslo"), prior_at(1, TWELVE), digest(TWELVE));
        assert!(matches!(
            admission,
            crate::replay::ledger::Admission::Skipped(_)
        ));
    }

    #[test]
    fn ledger_unopened_refusal_hands_back_the_unopened_ledger() {
        let ledger = record(RecordScope::Missing, 1).open_ledger();
        let admission = ledger.admit(key_of("oslo"), prior_at(1, TWELVE), digest(THIRTEEN));
        match admission {
            crate::replay::ledger::Admission::Refused { refusal, .. } => assert_eq!(
                refusal,
                Refusal::DivergentDuplicate {
                    key: key_of("oslo"),
                    segment: SegmentId(1),
                    prior: digest(TWELVE),
                    incoming: digest(THIRTEEN)
                }
            ),
            other => panic!("expected Refused, got {other:?}"),
        }
    }

    // verify: 3 rows

    #[test]
    fn verify_unrecorded_prior_reports_unrecorded() {
        assert_eq!(
            verifying().verify(key_of("bergen"), Prior::Unrecorded, digest(TWELVE)),
            VerifyFinding::Unrecorded {
                key: key_of("bergen")
            }
        );
    }

    #[test]
    fn verify_equal_digest_matches_with_the_priors_freshness() {
        assert_eq!(
            verifying().verify(key_of("oslo"), prior_at(0, TWELVE), digest(TWELVE)),
            VerifyFinding::Matches {
                key: key_of("oslo"),
                freshness: Freshness::Fresh
            }
        );
    }

    #[test]
    fn verify_different_digest_diverges_with_both_digests() {
        assert_eq!(
            verifying().verify(key_of("oslo"), prior_at(0, TWELVE), digest(THIRTEEN)),
            VerifyFinding::Diverged {
                key: key_of("oslo"),
                recorded: digest(TWELVE),
                live: digest(THIRTEEN),
                freshness: Freshness::Fresh
            }
        );
    }
}
