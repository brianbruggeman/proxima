use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

#[path = "src/build_defaults.rs"]
mod build_defaults;

fn missing_env(name: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("cargo must provide {name} to proxima-tls build.rs"),
    )
}

fn main() -> io::Result<()> {
    let target_os =
        env::var("CARGO_CFG_TARGET_OS").map_err(|_| missing_env("CARGO_CFG_TARGET_OS"))?;
    let output_directory =
        PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| missing_env("OUT_DIR"))?);
    let native_roots_default = build_defaults::native_roots_default(&target_os);
    let generated =
        format!("pub const TLS_CLIENT_NATIVE_ROOTS_DEFAULT: bool = {native_roots_default};\n");
    let generated_path = output_directory.join("tls_client_defaults.rs");

    fs::write(&generated_path, generated).map_err(|source| {
        io::Error::new(
            source.kind(),
            format!(
                "write generated TLS target defaults to {}: {source}",
                generated_path.display()
            ),
        )
    })?;
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    Ok(())
}
