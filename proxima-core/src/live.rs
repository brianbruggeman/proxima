//! [`Live<T>`] — a lock-free, live-swappable value split into a read half and a
//! control half that share one cell.
//!
//! The "hot readers, rare control-plane writes" pattern: the data path reads the
//! current value on every call through a lock-free
//! [`ArcSwap`](arc_swap::ArcSwap) load, while a control plane swaps the value out
//! of band. [`live`] hands back the two ends:
//!
//! - [`Live`] — the read half. [`read`](Live::read) borrows the current value
//!   for a closure (the hot path, no clone); cheap to clone, so every reader
//!   holds its own [`Live`] over the same cell.
//! - [`LiveControl`] — the control half. [`replace`](LiveControl::replace) swaps
//!   wholesale; [`update`](LiveControl::update) reads-modifies-writes.
//!
//! ## Ordering
//!
//! The swap is *per-read monotonic*: a [`read`](Live::read) sees either the
//! pre-swap or the post-swap value, never a torn state, and a swap applies to
//! every read after the store — but not at an exact call boundary. That is the
//! right contract for a control plane retuning a data plane on the fly.
//!
//! ## Sibling
//!
//! This is deliberately the *read-latest, no-notify* variant. When a reader must
//! be woken on change (react-to-update rather than sample-current), reach for the
//! `watch` channel in `proxima-sync` instead — it guards the value and notifies,
//! at the cost of the lock-free read this primitive keeps.

use alloc::sync::Arc;

use arc_swap::ArcSwap;

/// The read half of a [`live`] split. Clone is a single [`Arc`] bump — hand a
/// clone to every reader; they all sample the same live value.
pub struct Live<T> {
    cell: Arc<ArcSwap<T>>,
}

/// The control half of a [`live`] split — the out-of-band writer that swaps the
/// value the paired [`Live`] reads.
pub struct LiveControl<T> {
    cell: Arc<ArcSwap<T>>,
}

/// Split an initial value into a read half and a control half sharing one cell.
///
/// `initial` is the only input; the handles are runtime state, so there is no
/// builder (the workspace config principle's single-parameter exception).
#[must_use]
pub fn live<T>(initial: T) -> (Live<T>, LiveControl<T>) {
    let cell = Arc::new(ArcSwap::from_pointee(initial));
    (
        Live {
            cell: Arc::clone(&cell),
        },
        LiveControl { cell },
    )
}

impl<T> Clone for Live<T> {
    fn clone(&self) -> Self {
        Self {
            cell: Arc::clone(&self.cell),
        }
    }
}

impl<T> Live<T> {
    /// Read the current value under a lock-free load, borrowing it for `with`.
    /// The hot path — no `Arc` clone, no lock.
    pub fn read<R>(&self, with: impl FnOnce(&T) -> R) -> R {
        let guard = self.cell.load();
        with(&guard)
    }

    /// Snapshot the current value as an owned [`Arc`] — for introspection or to
    /// escape the [`read`](Live::read) borrow.
    #[must_use]
    pub fn snapshot(&self) -> Arc<T> {
        self.cell.load_full()
    }
}

impl<T> Clone for LiveControl<T> {
    fn clone(&self) -> Self {
        Self {
            cell: Arc::clone(&self.cell),
        }
    }
}

impl<T> LiveControl<T> {
    /// Swap the value wholesale. Every subsequent [`Live::read`] sees `next`.
    pub fn replace(&self, next: T) {
        self.cell.store(Arc::new(next));
    }

    /// Derive the next value from the current one. `mutate` may be retried under
    /// write contention, so it must be pure (no observable side effects).
    pub fn update(&self, mutate: impl Fn(&T) -> T) {
        self.cell.rcu(|current| Arc::new(mutate(current)));
    }

    /// Snapshot the current value as an owned [`Arc`] (introspection).
    #[must_use]
    pub fn snapshot(&self) -> Arc<T> {
        self.cell.load_full()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn read_sees_the_initial_value() {
        let (live_value, _control) = live(41_u64);
        assert_eq!(live_value.read(|value| *value), 41);
    }

    #[test]
    fn replace_is_visible_to_an_existing_reader() {
        let (live_value, control) = live(41_u64);
        control.replace(108_500);
        assert_eq!(live_value.read(|value| *value), 108_500);
    }

    #[test]
    fn update_derives_next_from_current() {
        let (live_value, control) = live(100_u64);
        control.update(|current| current + 25);
        assert_eq!(live_value.read(|value| *value), 125);
    }

    #[test]
    fn all_clones_share_one_cell() {
        let (live_value, control) = live(1_u64);
        let reader_a = live_value.clone();
        let reader_b = live_value.clone();
        control.replace(420_000);
        assert_eq!(reader_a.read(|value| *value), 420_000);
        assert_eq!(reader_b.read(|value| *value), 420_000);
    }

    #[test]
    fn snapshot_escapes_the_read_borrow() {
        let (live_value, control) = live(64_800_u64);
        let before = live_value.snapshot();
        control.replace(64_801);
        assert_eq!(*before, 64_800, "snapshot is a stable owned view");
        assert_eq!(
            *live_value.snapshot(),
            64_801,
            "a fresh snapshot sees the swap"
        );
    }
}
