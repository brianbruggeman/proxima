//! Human-readable driver for the no-std example. Requires `--features std`
//! (see the crate's `[[bin]] required-features`); the default `cargo build`
//! never compiles this binary, because a `#![no_std]` build has no `main`
//! entry point and no `println!` to link.
//!
//! The pipe logic this calls into (`FrameStore`, `block_on`) lives in
//! `lib.rs` and is unchanged from the `#![no_std]` build — only this driver
//! and its std-only `println!` wrapper are new.

fn main() {
    proxima_example_no_std::run_demo();
}
