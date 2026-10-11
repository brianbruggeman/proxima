use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const WINDOWS_TARGET: &str = "x86_64-pc-windows-msvc";
const CHILD_RECORD: &[u8] = b"cargo-val child record v1\r\n";

#[test]
fn unix_only_negative_control() -> io::Result<()> {
    let fixture_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/windows_target_gate");
    let temporary = tempfile::tempdir()?;
    let target_directory = temporary.path().join("target");
    let windows_manifest = copy_fixture(
        &fixture_root.join("windows_target"),
        &temporary.path().join("windows_target"),
    )?;
    let unix_manifest = copy_fixture(
        &fixture_root.join("unix_only"),
        &temporary.path().join("unix_only"),
    )?;

    let windows_gate = cargo_check(&windows_manifest, WINDOWS_TARGET, &target_directory)?;
    assert!(
        windows_gate.status.success(),
        "hand-authored Windows target gate must compile for {WINDOWS_TARGET}: {}",
        String::from_utf8_lossy(&windows_gate.stderr)
    );

    let unix_only = cargo_check(&unix_manifest, WINDOWS_TARGET, &target_directory)?;
    assert!(
        !unix_only.status.success(),
        "Unix-only compile control unexpectedly compiled for {WINDOWS_TARGET}"
    );
    assert!(
        String::from_utf8_lossy(&unix_only.stderr).contains("unix"),
        "negative control failed for an unrelated reason: {}",
        String::from_utf8_lossy(&unix_only.stderr)
    );

    assert_std_process_child_record(temporary.path())
}

fn copy_fixture(source: &Path, destination: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(destination.join("src"))?;
    for filename in ["Cargo.toml", "src/lib.rs"] {
        let content = fs::read(source.join(filename))?;
        fs::write(destination.join(filename), content)?;
    }
    Ok(destination.join("Cargo.toml"))
}

fn cargo_check(manifest: &Path, target: &str, target_directory: &Path) -> io::Result<Output> {
    Command::new(env!("CARGO"))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--target")
        .arg(target)
        .env("CARGO_TARGET_DIR", target_directory)
        .output()
}

fn assert_std_process_child_record(temporary_directory: &Path) -> io::Result<()> {
    let source_path = temporary_directory.join("child_record.rs");
    let executable_path =
        temporary_directory.join(format!("child_record{}", std::env::consts::EXE_SUFFIX));
    fs::write(
        &source_path,
        "fn main() { print!(\"cargo-val child record v1\\r\\n\"); std::process::exit(23); }\n",
    )?;

    let compile = Command::new("rustc")
        .arg("--edition=2024")
        .arg(&source_path)
        .arg("-o")
        .arg(&executable_path)
        .output()?;
    assert!(
        compile.status.success(),
        "child-record fixture must compile: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let child = Command::new(&executable_path).output()?;
    assert_eq!(child.status.code(), Some(23));
    assert_eq!(child.stdout, CHILD_RECORD);
    assert!(child.stderr.is_empty());
    Ok(())
}
