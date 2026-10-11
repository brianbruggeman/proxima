//! The MIL lane holds no lock: `proxima-mil` names no lock type and no Apple
//! framework, and the door that owns the plan cache is not `Sync`,
//! the single-writer cache that the Core ML handle will live beside.

#![cfg(all(feature = "onnx-door", feature = "mil-writer"))]
// fixtures read this repository's own files; a failed read IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::marker::PhantomData;
use std::path::Path;

use omega::door::{Door, MilSink};
use omega::onnx_door::OnnxReader;

const MIL_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../proxima-mil");
const LOCK_NAMES: [&str; 4] = ["Mutex", "RwLock", "parking_lot", "spin"];
const FRAMEWORK_NAME: &str = "objc2";

struct Probe<Subject>(PhantomData<Subject>);

trait WhenSync {
    fn is_sync(&self) -> bool;
}

impl<Subject: Sync> WhenSync for Probe<Subject> {
    fn is_sync(&self) -> bool {
        true
    }
}

trait Otherwise {
    fn is_sync(&self) -> bool;
}

impl<Subject> Otherwise for &Probe<Subject> {
    fn is_sync(&self) -> bool {
        false
    }
}

macro_rules! is_sync {
    ($subject:ty) => {
        (&Probe::<$subject>(PhantomData)).is_sync()
    };
}

fn rust_files(directory: &Path) -> Vec<String> {
    fs::read_dir(directory)
        .expect("the source directory exists")
        .flat_map(|entry| {
            let path = entry.expect("a directory entry reads").path();
            match path.is_dir() {
                true => rust_files(&path),
                false => read_if_rust(&path),
            }
        })
        .collect()
}

fn read_if_rust(path: &Path) -> Vec<String> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("rs") => vec![fs::read_to_string(path).expect("a source file reads")],
        _ => Vec::new(),
    }
}

fn occurrences(text: &str, names: &[&str]) -> usize {
    names.iter().map(|name| text.matches(name).count()).sum()
}

fn mil_sources() -> Vec<String> {
    rust_files(&Path::new(MIL_ROOT).join("src"))
}

#[test]
fn lock_free_no_lock_types_in_proxima_mil_sources() {
    let sources = mil_sources();

    let found: usize = sources
        .iter()
        .map(|text| occurrences(text, &LOCK_NAMES))
        .sum();

    assert!(
        !sources.is_empty(),
        "the scan must read at least one source file"
    );
    assert_eq!(found, 0, "proxima-mil/src names a lock type");
}

#[test]
fn lock_free_counter_control_counts_a_lock_name() {
    let text = "struct Cache { slots: Mutex<Vec<u8>> }";

    assert_eq!(occurrences(text, &LOCK_NAMES), 1);
    assert_eq!(
        occurrences("struct Cache { slots: Vec<u8> }", &LOCK_NAMES),
        0
    );
    assert_eq!(occurrences("use parking_lot::RwLock;", &LOCK_NAMES), 2);
}

#[test]
fn lock_free_door_is_not_sync_single_writer_plan_cache() {
    assert!(is_sync!(u64), "the probe must report a Sync type as Sync");
    assert!(!is_sync!(Door<OnnxReader, MilSink>));
}

#[test]
fn lock_free_no_apple_framework_in_proxima_mil() {
    let manifest = fs::read_to_string(Path::new(MIL_ROOT).join("Cargo.toml"))
        .expect("the proxima-mil manifest reads");
    let sources = mil_sources();

    let found = occurrences(&manifest, &[FRAMEWORK_NAME])
        + sources
            .iter()
            .map(|text| occurrences(text, &[FRAMEWORK_NAME]))
            .sum::<usize>();

    assert_eq!(found, 0, "proxima-mil names an Apple framework crate");
}
