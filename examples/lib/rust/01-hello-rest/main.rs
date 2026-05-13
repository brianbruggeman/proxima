#![allow(clippy::unwrap_used, clippy::expect_used, clippy::field_reassign_with_default, clippy::type_complexity, clippy::useless_vec, clippy::needless_range_loop, clippy::default_constructed_unit_structs)]
//! Hello world REST via proxima as a library.
//!
//! Loads the same `proxima.toml` the CLI example uses; mounts it at
//! `/{*path}`; binds to 127.0.0.1:8080. Curl any path to get the
//! synth response.

use std::path::PathBuf;

use proxima::{App, RunConfig};

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> proxima::ProximaResult<()> {
    let mut app = App::new()?;
    let config = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|parent| parent.parent())
        .and_then(|examples| examples.parent())
        .map(|root| root.join("config/01-hello-rest/proxima.toml"))
        .expect("find config relative to manifest");
    let hello = app.pipe("hello", config).await?;
    app.mount("/{*path}", hello)?;
    let shutdown = app
        .run_until_signal(RunConfig::http("127.0.0.1:8080".parse().expect("addr")))
        .await?;
    tokio::signal::ctrl_c().await.ok();
    shutdown.stop();
    Ok(())
}
