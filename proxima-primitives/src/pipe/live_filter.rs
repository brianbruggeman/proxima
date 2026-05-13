//! [`LiveFilter`] — a decision pipe whose predicate is swapped live through a
//! control handle. The pub/sub filter whose membership changes at runtime
//! without tearing down the subscription.
//!
//! It is a thin [`SendPipe`] face over [`proxima_core::live::Live`]: the general
//! "hot readers, rare control-plane writes" cell, specialised to answer
//! pass/reject (`Ok(item)` = admit, `Err(item)` = reject — see
//! `fan_in.rs`'s module doc for why a data-path decision is a pipe, not a
//! `bool`-returning capability trait). [`live_filter`] splits into the two
//! shared-state halves:
//!
//! - [`LiveFilter`] — the filtering (read) half. It implements [`SendPipe`], so
//!   it drops straight into `.and_then(inner)`; the hot data path does one
//!   lock-free load and delegates to the current predicate. For a pure
//!   membership query that never forwards the item on (a control-plane
//!   question, not a data-path decision — [`FilterRegistry::matches`]'s
//!   fan-out lookup is the example), [`LiveFilter::contains`] answers
//!   synchronously without the pipe detour.
//! - [`FilterControl`] — the control (write) half. The "pipe pumping in":
//!   [`replace`](FilterControl::replace) / [`update`](FilterControl::update)
//!   swap the predicate, and for id subscriptions
//!   [`apply`](FilterControl::apply) folds a [`FilterUpdate`] into it. Cheap to
//!   clone; drive it directly or from a stream of updates drained through an
//!   [`UnpinPipe`](crate::pipe::UnpinPipe) source.
//!
//! Both halves (and every clone) share one cell, so a control mutation is
//! visible to all filter clones — this is the "split" of a pub/sub subscription:
//! data-out ([`LiveFilter`]) and control-in ([`FilterControl`]) are the two ends
//! of one object. The swap is per-call monotonic (see
//! [`proxima_core::live`](proxima_core::live) for the ordering contract): a call
//! sees either the pre-swap or post-swap predicate, which is the right semantics
//! for id filtering ("start matching id X soon after I add it"). A filter that
//! must change at a precise item needs the update carried *in* the item stream
//! (a [`FanIn`](crate::pipe::FanIn) select over data + control) instead.

use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use core::future::Future;

use proxima_core::live::{Live, LiveControl, live};
use serde::{Deserialize, Serialize};

use crate::pipe::SendPipe;

/// The filtering (read) half of a [`live_filter`] split: a [`SendPipe`] whose
/// predicate is swapped live by the paired [`FilterControl`]. Clone is a single
/// `Arc` bump — hand a clone to every subscriber.
pub struct LiveFilter<Predicate> {
    inner: Live<Predicate>,
}

/// The control (write) half of a [`live_filter`] split — the "pipe pumping in"
/// that swaps the predicate the paired [`LiveFilter`] reads.
pub struct FilterControl<Predicate> {
    control: LiveControl<Predicate>,
}

/// Split an initial predicate into a live filtering half and its control half,
/// sharing one cell. `initial` is the only input (e.g.
/// `IdSet::from_ids(config.subscribed_ids)`).
#[must_use]
pub fn live_filter<Predicate>(
    initial: Predicate,
) -> (LiveFilter<Predicate>, FilterControl<Predicate>) {
    let (inner, control) = live(initial);
    (LiveFilter { inner }, FilterControl { control })
}

/// Sugar for the id-subscription case: split a live [`IdSet`] filter seeded from
/// `ids`. Equivalent to `live_filter(IdSet::from_ids(ids))`.
#[must_use]
pub fn live_filter_ids<Id: Ord + Clone>(
    ids: impl IntoIterator<Item = Id>,
) -> (LiveFilter<IdSet<Id>>, FilterControl<IdSet<Id>>) {
    live_filter(IdSet::from_ids(ids))
}

impl<Predicate> Clone for LiveFilter<Predicate> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<Predicate> LiveFilter<Predicate> {
    /// Snapshot the current predicate (introspection).
    #[must_use]
    pub fn snapshot(&self) -> Arc<Predicate> {
        self.inner.snapshot()
    }
}

impl<Id: Ord> LiveFilter<IdSet<Id>> {
    /// Synchronous membership check — for a control-plane query that only
    /// asks "does `id` match" and never forwards the item onward (unlike the
    /// [`SendPipe`] form, which admits or rejects an item flowing through it).
    #[must_use]
    pub fn contains(&self, id: &Id) -> bool {
        self.inner.read(|set| set.contains(id))
    }
}

impl<Item, Predicate> SendPipe for LiveFilter<Predicate>
where
    Predicate: SendPipe<In = Item, Out = Item> + Send + Sync + 'static,
    Item: Send + 'static,
{
    type In = Item;
    type Out = Item;
    type Err = Predicate::Err;

    fn call(&self, input: Item) -> impl Future<Output = Result<Item, Predicate::Err>> + Send {
        // snapshot outside the lock, then await the call — `Live::read`'s
        // closure is synchronous and cannot itself hold an in-flight future.
        let predicate = self.snapshot();
        async move { predicate.call(input).await }
    }
}

impl<Predicate> Clone for FilterControl<Predicate> {
    fn clone(&self) -> Self {
        Self {
            control: self.control.clone(),
        }
    }
}

impl<Predicate> FilterControl<Predicate> {
    /// Replace the predicate wholesale. Every subsequent read of the paired
    /// [`LiveFilter`] sees `next`.
    pub fn replace(&self, next: Predicate) {
        self.control.replace(next);
    }

    /// Read-modify-write the predicate. `mutate` derives the next predicate from
    /// the current one and may be retried under contention, so it must be pure.
    pub fn update(&self, mutate: impl Fn(&Predicate) -> Predicate) {
        self.control.update(mutate);
    }

    /// Snapshot the current predicate (introspection).
    #[must_use]
    pub fn snapshot(&self) -> Arc<Predicate> {
        self.control.snapshot()
    }
}

/// A membership predicate over a set of ids — the canonical pub/sub filter.
/// [`contains`](Self::contains) is `true` when the set holds `id`; as a
/// [`SendPipe`] it admits (`Ok(id)`) or rejects (`Err(id)`) the id flowing
/// through it. Build one as the `initial` for [`live_filter`] and grow it
/// live with [`FilterUpdate`].
///
/// Serde-transparent: it round-trips as a bare array of ids, so a subscription's
/// set is exactly what you load from config or ship over the wire (the config
/// mirror; the layered `conflaguration` loader attaches at the service that
/// composes this, per the workspace config principle's layering caveat).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IdSet<Id: Ord> {
    members: BTreeSet<Id>,
}

impl<Id: Ord> IdSet<Id> {
    /// An empty set — matches nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            members: BTreeSet::new(),
        }
    }

    /// A set seeded from `ids`.
    pub fn from_ids(ids: impl IntoIterator<Item = Id>) -> Self {
        ids.into_iter().collect()
    }

    /// Whether `id` is a member.
    #[must_use]
    pub fn contains(&self, id: &Id) -> bool {
        self.members.contains(id)
    }

    /// The number of ids in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the set is empty (matches nothing).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Iterate the member ids in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &Id> + '_ {
        self.members.iter()
    }
}

impl<Id: Ord + Clone> IdSet<Id> {
    /// A copy with `id` added — the copy-on-write step for
    /// [`FilterControl::update`].
    #[must_use]
    pub fn with(&self, id: Id) -> Self {
        let mut members = self.members.clone();
        members.insert(id);
        Self { members }
    }

    /// A copy with `id` removed.
    #[must_use]
    pub fn without(&self, id: &Id) -> Self {
        let mut members = self.members.clone();
        members.remove(id);
        Self { members }
    }
}

impl<Id: Ord> Default for IdSet<Id> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: Ord> FromIterator<Id> for IdSet<Id> {
    fn from_iter<Iter: IntoIterator<Item = Id>>(iter: Iter) -> Self {
        Self {
            members: iter.into_iter().collect(),
        }
    }
}

// `Id` doubles as both the admitted `Out` and the rejected `Err` — there is
// no richer rejection reason than "this id was not a member", so reusing the
// id itself needs no new error type.
impl<Id: Ord + Clone + core::fmt::Debug + Send + Sync + 'static> SendPipe for IdSet<Id> {
    type In = Id;
    type Out = Id;
    type Err = Id;

    fn call(&self, id: Id) -> impl Future<Output = Result<Id, Id>> + Send {
        let admitted = self.contains(&id);
        async move { if admitted { Ok(id) } else { Err(id) } }
    }
}

/// The control-plane update algebra for an [`IdSet`] filter, folded in through
/// [`FilterControl::apply`]. `Add`/`Remove` grow or shrink the live set
/// incrementally; `Replace`/`Clear` swap it wholesale. Serializable, so it is
/// also the wire form of a control message ("add id X" over the network).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterUpdate<Id> {
    Add(Id),
    Remove(Id),
    Replace(Vec<Id>),
    Clear,
}

// fluent control handle: chaining is the ergonomic point, and the returned
// `&Self` is optional at each call, so `#[must_use]` would wrongly warn on the
// common statement form (`control.add(a);`).
#[allow(clippy::return_self_not_must_use)]
impl<Id: Ord + Clone> FilterControl<IdSet<Id>> {
    /// Apply an [`IdSet`] update to the live filter.
    pub fn apply(&self, update: FilterUpdate<Id>) -> &Self {
        match update {
            FilterUpdate::Add(id) => self.update(move |set| set.with(id.clone())),
            FilterUpdate::Remove(id) => self.update(move |set| set.without(&id)),
            FilterUpdate::Replace(members) => self.replace(members.into_iter().collect()),
            FilterUpdate::Clear => self.replace(IdSet::new()),
        }
        self
    }

    /// Add an id to the live set (sugar over [`apply`](Self::apply)).
    pub fn add(&self, id: Id) -> &Self {
        self.apply(FilterUpdate::Add(id))
    }

    /// Remove an id from the live set.
    pub fn remove(&self, id: Id) -> &Self {
        self.apply(FilterUpdate::Remove(id))
    }

    /// Clear the live set (match nothing).
    pub fn clear(&self) -> &Self {
        self.apply(FilterUpdate::Clear)
    }

    /// Replace the live set wholesale from an id iterator.
    pub fn replace_ids(&self, ids: impl IntoIterator<Item = Id>) -> &Self {
        self.apply(FilterUpdate::Replace(ids.into_iter().collect()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::ptr_arg)]
mod tests {
    use super::*;

    // realistic request-correlation ids (hex trace-id prefixes).
    fn ids() -> (String, String, String) {
        (
            "req-7f3a9c2e1b4d".to_string(),
            "req-0f43d9c0aa18".to_string(),
            "req-c06db8bb5521".to_string(),
        )
    }

    #[test]
    fn decides_membership_against_the_current_set() {
        let (subscribed, other, _) = ids();
        let (filter, _control) = live_filter(IdSet::from_ids([subscribed.clone()]));
        assert!(filter.contains(&subscribed));
        assert!(!filter.contains(&other));
    }

    #[test]
    fn replace_swaps_the_predicate_seen_by_an_existing_filter() {
        let (first, second, _) = ids();
        let (filter, control) = live_filter(IdSet::from_ids([first.clone()]));
        assert!(filter.contains(&first));
        control.replace(IdSet::from_ids([second.clone()]));
        assert!(
            !filter.contains(&first),
            "old id no longer matches after replace"
        );
        assert!(filter.contains(&second), "new id matches after replace");
    }

    #[test]
    fn add_grows_and_remove_shrinks_the_live_set() {
        let (first, second, third) = ids();
        let (filter, control) = live_filter(IdSet::from_ids([first.clone()]));

        control.apply(FilterUpdate::Add(second.clone()));
        assert!(filter.contains(&first));
        assert!(filter.contains(&second));
        assert!(!filter.contains(&third));

        control.apply(FilterUpdate::Remove(first.clone()));
        assert!(!filter.contains(&first), "removed id stops matching");
        assert!(filter.contains(&second), "untouched id still matches");
    }

    #[test]
    fn clear_matches_nothing_then_replace_reseeds() {
        let (first, second, _) = ids();
        let (filter, control) = live_filter(IdSet::from_ids([first.clone()]));
        control.apply(FilterUpdate::Clear);
        assert!(!filter.contains(&first));
        assert_eq!(filter.snapshot().len(), 0);
        control.apply(FilterUpdate::Replace(
            [second.clone()].into_iter().collect(),
        ));
        assert!(filter.contains(&second));
    }

    #[test]
    fn all_clones_share_one_live_predicate() {
        // the split: many subscribers (filter clones), one control — an update
        // through the control is visible to every subscriber.
        let (first, second, _) = ids();
        let (filter, control) = live_filter(IdSet::from_ids([first.clone()]));
        let subscriber_a = filter.clone();
        let subscriber_b = filter.clone();
        control.apply(FilterUpdate::Add(second.clone()));
        assert!(subscriber_a.contains(&second));
        assert!(subscriber_b.contains(&second));
    }

    #[test]
    fn live_filter_satisfies_the_pipe_seam() {
        // proves the filtering half is exactly what `.and_then(inner)` wants
        // (the RISC-reuse claim), at the type level: any `SendPipe<In = Out =
        // Item>` composes, and `LiveFilter<IdSet<Item>>` is one.
        fn admits<Predicate>(predicate: &Predicate, id: String) -> bool
        where
            Predicate: SendPipe<In = String, Out = String>,
        {
            let mut call = core::pin::pin!(SendPipe::call(predicate, id));
            let waker = core::task::Waker::noop();
            let mut context = core::task::Context::from_waker(waker);
            loop {
                if let core::task::Poll::Ready(outcome) = call.as_mut().poll(&mut context) {
                    return outcome.is_ok();
                }
            }
        }
        let (subscribed, other, _) = ids();
        let (filter, _control) = live_filter(IdSet::from_ids([subscribed.clone()]));
        assert!(admits(&filter, subscribed));
        assert!(!admits(&filter, other));
    }

    #[test]
    fn update_derives_next_from_current() {
        // generic update path (not id-set specific): mutate reads the current set.
        let (first, second, _) = ids();
        let (filter, control) = live_filter(IdSet::from_ids([first.clone()]));
        control.update(|set| set.with(second.clone()));
        assert!(filter.contains(&first));
        assert!(filter.contains(&second));
    }

    #[test]
    fn fluent_control_sugars_chain() {
        let (first, second, third) = ids();
        let (filter, control) = live_filter_ids([first.clone()]);
        control
            .add(second.clone())
            .add(third.clone())
            .remove(first.clone());
        assert!(!filter.contains(&first), "removed id gone");
        assert!(filter.contains(&second), "added id present");
        assert!(filter.contains(&third), "added id present");
    }

    #[test]
    fn id_set_config_mirror_round_trips() {
        let (first, second, _) = ids();
        let set = IdSet::from_ids([first, second]);
        let serialized = serde_json::to_string(&set).unwrap();
        assert!(
            serialized.starts_with('['),
            "an id-set is a bare array of ids"
        );
        let restored: IdSet<String> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored, set, "config mirror round-trips");
    }

    #[test]
    fn filter_update_is_a_serializable_control_message() {
        let (_, _, third) = ids();
        let update = FilterUpdate::Add(third);
        let serialized = serde_json::to_string(&update).unwrap();
        let restored: FilterUpdate<String> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            restored, update,
            "control message round-trips over the wire"
        );
    }
}
