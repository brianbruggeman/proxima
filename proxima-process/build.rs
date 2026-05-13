//! Build the libc-interpose shim as a side-output `.dylib`/`.so`.
//!
//! The shim is a separate compilation unit (a C source file linked
//! as a shared library), so its `#[no_mangle]`-equivalent libc
//! exports stay isolated from the rlib that rust consumers link.
//! The artifact's absolute path is threaded into the rlib via the
//! `PROXIMA_LIBC_SHIM_PATH` cargo-rustc env var so call-site code
//! can hand it to children through `DYLD_INSERT_LIBRARIES` /
//! `LD_PRELOAD`.

use std::env;
use std::path::PathBuf;

#[allow(clippy::expect_used)]
fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("src").join("interpose.c");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");

    println!("cargo:rerun-if-changed={}", source.display());

    let (artifact_name, link_flag) = match target_os.as_str() {
        "macos" => ("libproxima_process_shim.dylib", "-dynamiclib"),
        "linux" => ("libproxima_process_shim.so", "-shared"),
        other => panic!("proxima-process libc shim: unsupported target_os {other}"),
    };

    let artifact_path = out_dir.join(artifact_name);

    // cc::Build's get_compiler() picks the right portable compiler
    // (clang on macOS, gcc/cc on Linux, honours CC env var). We
    // drive it manually because cc::Build::compile only produces a
    // static lib; we need a shared one.
    let compiler = cc::Build::new().file(&source).get_compiler();
    let status = compiler
        .to_command()
        .arg(link_flag)
        .arg("-fPIC")
        .arg("-O2")
        .arg("-Wall")
        .arg("-o")
        .arg(&artifact_path)
        .arg(&source)
        .status()
        .expect("failed to invoke C compiler for interpose shim");
    assert!(
        status.success(),
        "C compiler exited non-zero while building interpose shim"
    );

    println!(
        "cargo:rustc-env=PROXIMA_LIBC_SHIM_PATH={}",
        artifact_path.display()
    );
}
