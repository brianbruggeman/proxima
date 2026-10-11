#![allow(clippy::unwrap_used, clippy::expect_used)]

use tempfile::tempdir;
use tokio::process::Command;

fn proxima_bin() -> &'static str {
    env!("CARGO_BIN_EXE_proxima")
}

#[proxima::test]
async fn export_mlx_of_a_missing_gguf_exits_1_and_writes_nothing() {
    let dir = tempdir().expect("tempdir");
    let missing = dir.path().join("absent.gguf");
    let out = dir.path().join("out");

    let output = Command::new(proxima_bin())
        .args(["model", "export-mlx"])
        .arg(&missing)
        .arg("--out")
        .arg(&out)
        .output()
        .await
        .expect("spawn proxima");

    assert_eq!(output.status.code(), Some(1));
    assert!(!out.exists(), "no output directory for a missing source");
}

#[proxima::test]
async fn export_mlx_with_seven_bit_quantization_exits_2_naming_the_bits() {
    let dir = tempdir().expect("tempdir");

    let output = Command::new(proxima_bin())
        .args(["model", "export-mlx"])
        .arg(dir.path().join("model.gguf"))
        .arg("--out")
        .arg(dir.path().join("out"))
        .args(["--quantize", "7,64"])
        .output()
        .await
        .expect("spawn proxima");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains('7'), "{stderr}");
}
