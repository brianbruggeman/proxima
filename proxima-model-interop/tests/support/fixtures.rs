use std::path::PathBuf;

use serde_json::Value;

pub fn fixture_path(row: u8) -> Result<PathBuf, String> {
    let root = std::env::var("PROXIMA_FIXTURES")
        .map_err(|error| format!("row {row}: PROXIMA_FIXTURES is not usable: {error}"))?;
    let root = PathBuf::from(root);
    let manifest = root.join("manifest.jsonl");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("row {row}: cannot read {}: {error}", manifest.display()))?;
    for line in text.lines() {
        let entry: Value = serde_json::from_str(line)
            .map_err(|error| format!("row {row}: bad line in {}: {error}", manifest.display()))?;
        if entry["row"].as_u64() == Some(u64::from(row)) {
            let file = entry["file"]
                .as_str()
                .ok_or_else(|| format!("row {row}: no `file` key in {}", manifest.display()))?;
            return Ok(root.join(file));
        }
    }
    Err(format!("row {row}: not present in {}", manifest.display()))
}
