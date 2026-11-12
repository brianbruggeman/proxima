//! regenerates src/bindings.rs from the vendored `onnxruntime_c_api.h` (onnxruntime 1.20.1, `ORT_API_VERSION` 20).
//! run: `cargo run --features bindgen --bin regen_bindings`

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

const LIBCLANG_DIR: &str = "/Library/Developer/CommandLineTools/usr/lib";

fn main() -> ExitCode {
    // safety: first statement of a single-threaded bin, before bindgen spawns anything or reads the env
    unsafe { env::set_var("LIBCLANG_PATH", LIBCLANG_DIR) };
    match generate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("regen_bindings failed: {message}");
            ExitCode::FAILURE
        }
    }
}

fn generate() -> Result<(), String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let headers = root.join("ort-headers");
    let bindings = bindgen::Builder::default()
        .header(headers.join("onnxruntime_c_api.h").to_string_lossy())
        .clang_arg(format!("-I{}", headers.display()))
        .dynamic_library_name("OrtLib")
        .dynamic_link_require_all(true)
        .layout_tests(true)
        .derive_default(false)
        .derive_debug(true)
        .generate_comments(false)
        .allowlist_function("OrtGetApiBase")
        .allowlist_type("OrtApi")
        .allowlist_type("OrtApiBase")
        .generate()
        .map_err(|error| error.to_string())?;
    bindings
        .write_to_file(root.join("src/bindings.rs"))
        .map_err(|error| error.to_string())
}
