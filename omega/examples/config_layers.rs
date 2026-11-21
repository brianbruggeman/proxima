//! Show which source wins when a `RuntimeConfig` is layered: defaults, then a
//! file, then the environment, then explicit setters.
//!
//! ```text
//! cargo run -p proxima-omega --example config_layers
//! OMEGA_ENGINE=gpu OMEGA_GPU_DRIVER=metal cargo run -p proxima-omega --example config_layers
//! ```

use anyhow::Result;
use omega::config::RuntimeConfig;

const FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/omega_hello_tensor.toml"
);

fn main() -> Result<()> {
    let defaults = RuntimeConfig::default();
    let from_file = RuntimeConfig::from_file(FILE)?;
    let from_file_then_env = RuntimeConfig::builder().from_file(FILE)?.from_env()?.build()?;
    let then_setter = RuntimeConfig::builder()
        .from_file(FILE)?
        .from_env()?
        .with_engine("cpu")
        .build()?;

    report("defaults", &defaults);
    report("file", &from_file);
    report("file + env", &from_file_then_env);
    report("file + env + with_engine(\"cpu\")", &then_setter);
    Ok(())
}

fn report(source: &str, config: &RuntimeConfig) {
    println!(
        "{source:<34} engine={:<4} gpu_driver={}",
        config.engine, config.gpu_driver
    );
}
