#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/block_keys.rs"]
mod block_keys;

use block_keys::chained_keys;

fn sixteen_ids() -> Vec<u32> {
    (0..16).collect()
}

#[test]
fn blockfile_chained_key_depends_on_every_earlier_block() {
    let ids = sixteen_ids();
    let keys = chained_keys(&ids, 4, 0);
    assert_eq!(keys.len(), 4);

    let mut first_changed = ids.clone();
    first_changed[0] = 999;
    let from_first = chained_keys(&first_changed, 4, 0);
    assert!(keys.iter().zip(&from_first).all(|(original, changed)| original != changed));

    let mut middle_changed = ids.clone();
    middle_changed[5] = 999;
    let from_middle = chained_keys(&middle_changed, 4, 0);
    assert_eq!(keys[0], from_middle[0]);
    assert!(keys[1..].iter().zip(&from_middle[1..]).all(|(original, changed)| original != changed));

    assert_eq!(chained_keys(&ids[..7], 4, 0).len(), 1);
    assert!(chained_keys(&ids, 0, 0).is_empty());
    assert!(chained_keys(&ids[..3], 4, 0).is_empty());
}

#[test]
fn blockfile_chained_key_is_bound_to_the_seed() {
    let ids = sixteen_ids();
    let under_zero = chained_keys(&ids, 4, 0);
    let under_one = chained_keys(&ids, 4, 1);
    assert!(under_zero.iter().zip(&under_one).all(|(zero, one)| zero != one));
    assert_eq!(under_zero, chained_keys(&ids, 4, 0));
}
