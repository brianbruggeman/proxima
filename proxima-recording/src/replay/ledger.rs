//! Record-session ledger as a typestate: [`RecordLedger<Unopened>`] becomes
//! [`RecordLedger<Open>`] on the first append, so "has this session stamped its
//! segment yet" is a type, not a flag (guiding-principles P11).
//!
//! Single owner by construction: `admit` takes `self` or `&mut self`, so the
//! recorder's driving task is the only holder and no lock exists (P21 level 1).
//! Alloc tier; the session's appended keys live in a `BTreeMap`, off the
//! per-request lookup path. Obtain one from [`RecordServing::open_ledger`];
//! the duplicate rule is the same [`DuplicatePolicy::resolve`] the loader uses.

use alloc::collections::BTreeMap;
use core::marker::PhantomData;

use crate::replay::fsm::{
    DuplicatePolicy, Prior, RecordedAt, Refusal, Resolution, SegmentId, SegmentStamp, UnixMillis,
};
use crate::replay::keying::{KeyDigest, ResponseDigest};

/// Marker: this session has not appended, so its segment stamp is unwritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unopened;

/// Marker: the session's segment stamp has been handed to the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Open;

#[derive(Debug, Clone)]
pub struct RecordLedger<State> {
    session: SegmentId,
    duplicates: DuplicatePolicy,
    now: UnixMillis,
    appended: BTreeMap<KeyDigest, ResponseDigest>,
    state: PhantomData<State>,
}

/// What the edge does with one exchange once the session is open.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendDecision {
    Append,
    SkipIdentical,
    Refuse(Refusal),
}

/// Outcome of the first `admit`. Only an append moves the ledger to `Open`;
/// a skip or refusal hands back the still-`Unopened` ledger.
#[must_use]
#[derive(Debug, Clone)]
pub enum Admission {
    Opened {
        ledger: RecordLedger<Open>,
        stamp: SegmentStamp,
    },
    Skipped(RecordLedger<Unopened>),
    Refused {
        ledger: RecordLedger<Unopened>,
        refusal: Refusal,
    },
}

impl<State> RecordLedger<State> {
    fn retype<Next>(self) -> RecordLedger<Next> {
        RecordLedger {
            session: self.session,
            duplicates: self.duplicates,
            now: self.now,
            appended: self.appended,
            state: PhantomData,
        }
    }

    fn decide(&mut self, key: KeyDigest, prior: Prior, response: ResponseDigest) -> AppendDecision {
        let held = match (self.appended.get(&key), prior) {
            (Some(appended), _) => Some((self.session, *appended)),
            (
                None,
                Prior::Recorded {
                    segment, response, ..
                },
            ) => Some((segment, response)),
            (None, Prior::Unrecorded) => None,
        };
        match self.duplicates.resolve(key, held, (self.session, response)) {
            Resolution::Insert | Resolution::Supersede | Resolution::Replace => {
                self.appended.insert(key, response);
                AppendDecision::Append
            }
            Resolution::KeepPrior => AppendDecision::SkipIdentical,
            Resolution::Refuse(refusal) => AppendDecision::Refuse(refusal),
        }
    }
}

impl RecordLedger<Unopened> {
    pub(crate) fn new(session: SegmentId, duplicates: DuplicatePolicy, now: UnixMillis) -> Self {
        Self {
            session,
            duplicates,
            now,
            appended: BTreeMap::new(),
            state: PhantomData,
        }
    }

    pub fn admit(mut self, key: KeyDigest, prior: Prior, response: ResponseDigest) -> Admission {
        match self.decide(key, prior, response) {
            AppendDecision::Append => {
                let stamp = SegmentStamp {
                    segment: self.session,
                    recorded_at: RecordedAt::At(self.now),
                };
                Admission::Opened {
                    ledger: self.retype(),
                    stamp,
                }
            }
            AppendDecision::SkipIdentical => Admission::Skipped(self),
            AppendDecision::Refuse(refusal) => Admission::Refused {
                ledger: self,
                refusal,
            },
        }
    }
}

impl RecordLedger<Open> {
    pub fn admit(
        &mut self,
        key: KeyDigest,
        prior: Prior,
        response: ResponseDigest,
    ) -> AppendDecision {
        self.decide(key, prior, response)
    }
}
