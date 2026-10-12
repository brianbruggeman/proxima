//! `protoc` as the independent decoder of an emitted `Model.proto`, shared by
//! the MIL writer tests; it shares no code with the writer under test.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn proto_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/coreml-lane/proto/coreml")
}

/// The text `protoc --decode` prints for `model`, or what it said on stderr.
pub fn protoc_decode(model: &[u8]) -> Result<String, String> {
    let mut child = Command::new("protoc")
        .arg("--decode=CoreML.Specification.Model")
        .arg(format!("-I{}", proto_directory().display()))
        .arg("Model.proto")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| format!("the protoc binary did not start: {source}"))?;
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(model)
        .map_err(|source| format!("writing the model to protoc failed: {source}"))?;
    let output = child
        .wait_with_output()
        .map_err(|source| format!("waiting for protoc failed: {source}"))?;
    match output.status.success() {
        true => Ok(String::from_utf8_lossy(&output.stdout).into_owned()),
        false => Err(String::from_utf8_lossy(&output.stderr).into_owned()),
    }
}

/// The `type` of every operation in a decoded model, in block order.
pub fn operator_sequence(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.trim_start().strip_prefix("type: \""))
        .filter_map(|rest| rest.strip_suffix('"'))
        .map(str::to_owned)
        .collect()
}

/// The `Model.proto` bytes `protoc --encode` writes for a text-format model.
pub fn protoc_encode(text: &str) -> Vec<u8> {
    let mut child = Command::new("protoc")
        .arg("--encode=CoreML.Specification.Model")
        .arg(format!("-I{}", proto_directory().display()))
        .arg("Model.proto")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the protoc binary starts");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(text.as_bytes())
        .expect("protoc reads the text model");
    let output = child.wait_with_output().expect("protoc finishes");
    assert!(output.status.success(), "protoc refused the text model");
    output.stdout
}
