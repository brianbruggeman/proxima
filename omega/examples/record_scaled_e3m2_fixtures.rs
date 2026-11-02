//! Rewrites the row-blocked scaled-e3m2 kernel fixtures under
//! `tests/fixtures/` from the current emitter, through the same
//! `matvec_source` the gate tests compare against.
//!
//! `cargo run -p proxima-omega --features test-support --example record_scaled_e3m2_fixtures`

use std::path::PathBuf;
use std::process::ExitCode;

use omega::msl::golden_support::matvec_source;

const FIXTURES: [(&str, u32, u32); 2] = [
    ("scaled_e3m2_row_blocked_1024x512.msl", 1024, 512),
    ("scaled_e3m2_row_blocked_1536x6144.msl", 1536, 6144),
];

fn main() -> ExitCode {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (file_name, rows, reduction) in FIXTURES {
        let path = directory.join(file_name);
        if let Err(error) = std::fs::write(&path, matvec_source(rows, reduction)) {
            eprintln!("cannot write fixture {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
        println!("wrote {}", path.display());
    }
    ExitCode::SUCCESS
}
