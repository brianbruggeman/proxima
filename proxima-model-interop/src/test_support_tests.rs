use std::path::Path;

use crate::checkpoint_resolver::{fixture_manifest, fixture_path};

#[test]
fn fixture_constant_unset_variable_names_variable_and_row() {
    let error = temp_env::with_vars(
        [
            ("PROXIMA_FIXTURES", None::<&str>),
            ("PROXIMA_MODEL_DIRS", Some("/nonexistent/model/dir")),
        ],
        || fixture_path(5),
    )
    .expect_err("an unset PROXIMA_FIXTURES with no manifest in the model dirs must be an error");

    assert!(error.contains("PROXIMA_FIXTURES"), "error was: {error}");
    assert!(error.contains("row 5"), "error was: {error}");
}

#[test]
fn fixture_constant_rows_resolve_or_name_the_row() {
    let manifest_path =
        fixture_manifest().expect("PROXIMA_FIXTURES or a model dir must hold manifest.jsonl");
    let manifest = std::fs::read_to_string(&manifest_path).expect("manifest.jsonl readable");
    let row_five_file = manifest
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("manifest line is json"))
        .find(|entry| entry["row"] == 5)
        .and_then(|entry| entry["file"].as_str().map(str::to_owned))
        .expect("manifest has a row 5 with a file");

    let granite = fixture_path(5).expect("row 5 resolves through the manifest");
    let outside = fixture_path(17).expect_err("row 17 is outside the 16 manifest rows");

    assert!(
        Path::new(&granite).exists(),
        "row 5 path does not exist: {}",
        granite.display()
    );
    assert!(
        granite.ends_with(&row_five_file),
        "row 5 path {} does not end with {row_five_file}",
        granite.display()
    );
    assert!(outside.contains("17"), "error was: {outside}");
}
