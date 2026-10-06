use std::env;
use std::fs;
use std::path::PathBuf;

#[path = "src/build_defaults.rs"]
mod build_defaults;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .expect("cargo must provide CARGO_CFG_TARGET_OS to proxima-tls build.rs");
    let output_directory = PathBuf::from(
        env::var_os("OUT_DIR").expect("cargo must provide OUT_DIR to proxima-tls build.rs"),
    );
    let native_roots_default = build_defaults::native_roots_default(&target_os);
    let generated =
        format!("pub const TLS_CLIENT_NATIVE_ROOTS_DEFAULT: bool = {native_roots_default};\n");

    fs::write(output_directory.join("tls_client_defaults.rs"), generated)
        .expect("write generated TLS target defaults");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
}
