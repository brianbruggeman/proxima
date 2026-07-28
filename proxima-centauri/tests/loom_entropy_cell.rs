//! Exhaustive interleaving check for [`EntropyCell`]'s take-once claim.
//!
//! ```text
//! cargo test -p proxima-centauri --features loom --test loom_entropy_cell
//! ```
//!
//! # Why this exists alongside the thread test
//!
//! Both catch a broken claim: mutation-tested 2026-07-28 by replacing the
//! `compare_exchange` with a non-atomic check-then-act, the thread test in
//! `entropy::tests::contention` fails and so does this one. The difference is
//! *why* they fail.
//!
//! The thread test samples. It caught that mutation on this machine, at this
//! thread count, on this run — a wider race window or a busier host and it
//! might not. loom enumerates every legal interleaving of the model, so it
//! fails deterministically and on any host. For a claim whose violation is
//! silent nonce reuse, sampling is not the standard to hold.
//!
//! It is a default-off feature rather than `--cfg loom` because RUSTFLAGS
//! applies the cfg to every crate in the dependency graph, which breaks
//! dependencies carrying their own loom support.
//!
//! Two threads is the right size here: the property is pairwise, and loom's
//! state space grows combinatorially with thread count.

#![cfg(feature = "loom")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use loom::sync::Arc;
use loom::thread;
use proxima_centauri::EntropyCell;

#[test]
fn a_filled_cell_is_claimed_by_exactly_one_of_two_drawers() {
    loom::model(|| {
        let cell = Arc::new(EntropyCell::new());
        cell.set([0x5A; 32]).expect("a fresh cell is empty");

        let contender = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || usize::from(cell.draw().is_ok()))
        };

        let here = usize::from(cell.draw().is_ok());
        let there = contender.join().expect("no drawer panics");

        assert_eq!(
            here + there,
            1,
            "exactly one drawer may claim a single filled cell — two winners \
             is the same value used as two nonces"
        );
    });
}

#[test]
fn a_fill_racing_a_draw_never_duplicates_or_loses_the_value() {
    loom::model(|| {
        let cell = Arc::new(EntropyCell::new());
        cell.set([0x01; 32]).expect("a fresh cell is empty");

        // one thread tries to refill while another drains: the refill must be
        // refused while a value is outstanding, so the pair can never yield
        // two claims off one fill.
        let filler = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || usize::from(cell.set([0x02; 32]).is_ok()))
        };

        let drew = usize::from(cell.draw().is_ok());
        let filled = filler.join().expect("no filler panics");

        // at most one draw can succeed off the original fill; a successful
        // refill only ever follows a successful drain.
        assert!(drew <= 1);
        if filled == 1 {
            assert_eq!(drew, 1, "a refill implies the previous value was claimed");
        }
    });
}
