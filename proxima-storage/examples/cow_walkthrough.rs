//! Teaching walkthrough for the copy-on-write atomic-root-swap update FSM.
//!
//! Drives [`CowRoot`] through every legal transition of [`UpdateState`], printing
//! the persistent byte image after each step, then shows that `recover` reads
//! only the root. Run with: `cargo run -p proxima-storage --example cow_walkthrough`.
//!
//! This composes the two primitives the module exposes: [`proxima_storage::pmem::persist`]
//! (the durability barrier) and [`CowRoot`] (the crash-consistent FSM). The
//! `persist` closure here is the real [`proxima_storage::pmem::persist`] function — on a
//! dev host without pmem it is a documented no-op; the FSM logic is identical.

use proxima_storage::pmem::cow::{CowRoot, UpdateState};
use proxima_storage::pmem::error::PmemError;
use proxima_storage::pmem::persist;

fn show(label: &str, region: &[u8], layout: &CowRoot) {
    let root = layout.live_index(region);
    println!("{label:<22} root={root}  bytes={region:02x?}");
}

fn main() -> Result<(), PmemError> {
    let layout = CowRoot::new(8)?;
    let old = [0xAAu8; 8];
    let new = [0xBBu8; 8];

    let mut region = vec![0u8; layout.region_len()];
    layout.init(&mut region, &old, &persist::persist)?;
    show("after init", &region, &layout);

    let plan = layout.prepare(&region, &new)?;
    let mut state = UpdateState::Idle;
    println!("\ndriving the update FSM, NEW = {new:02x?}:");
    while state != UpdateState::Committed {
        state = layout.step(&mut region, &plan, state, &persist::persist);
        show(&format!("{state:?}"), &region, &layout);
    }

    let live = layout.recover(&region)?;
    println!("\nrecover() returns the live slot: {live:02x?}");
    assert_eq!(live, &new, "after a clean commit the live value is NEW");
    println!("recovery is a single root read — no log, no replay.");
    Ok(())
}
