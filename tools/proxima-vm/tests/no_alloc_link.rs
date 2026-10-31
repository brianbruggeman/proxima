//! The tier-3 claim at binary level: every guest links with no global
//! allocator and carries no allocator shim symbol, so a dependency that
//! starts linking the alloc crate fails the guest build here.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

use object::{Object, ObjectSymbol};

const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");
const GUEST_BINS: [&str; 3] = [
    "proxima-vm-guest-lambda",
    "uppercase",
    "proxima-vm-guest-lambda-netstack",
];
const ALLOCATOR_SYMBOLS: [&str; 4] = [
    "__rust_alloc",
    "__rust_dealloc",
    "__rust_realloc",
    "__rust_no_alloc_shim_is_unstable",
];

fn build_guest(target: &str, bin: &str) -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"), PathBuf::from)
        .join("proxima-vm-no-alloc-guests");
    let status = Command::new("cargo")
        .args(["build", "--release", "--target", target, "--bin", bin])
        .args(["--manifest-path", &format!("{GUEST_MANIFEST_DIR}/Cargo.toml")])
        .env("CARGO_TARGET_DIR", &target_dir)
        .status()
        .expect("run cargo build for a guest");
    assert!(status.success(), "{bin} failed to link for {target} with no global allocator");
    target_dir.join(target).join("release").join(bin)
}

fn allocator_symbols_in(path: &PathBuf) -> Vec<String> {
    let bytes = std::fs::read(path).expect("read the linked ELF");
    let file = object::File::parse(bytes.as_slice()).expect("parse the linked ELF");
    file.symbols()
        .filter_map(|symbol| symbol.name().ok().map(str::to_owned))
        .filter(|name| ALLOCATOR_SYMBOLS.iter().any(|banned| name.contains(banned)))
        .collect()
}

fn assert_guests_carry_no_allocator(target: &str) {
    for bin in GUEST_BINS {
        let found = allocator_symbols_in(&build_guest(target, bin));
        assert!(found.is_empty(), "{bin} for {target} links allocator symbols: {found:?}");
    }
}

#[test]
fn guests_link_without_a_global_allocator_on_x86_64() {
    assert_guests_carry_no_allocator("x86_64-unknown-none");
}

#[test]
fn guests_link_without_a_global_allocator_on_aarch64() {
    assert_guests_carry_no_allocator("aarch64-unknown-none");
}

#[test]
fn the_symbol_scan_flags_a_binary_that_does_allocate() {
    let this_test_binary = std::env::current_exe().expect("locate the running test binary");
    assert!(!allocator_symbols_in(&this_test_binary).is_empty());
}
