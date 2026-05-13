//! Drive the wrk-beating throughput run through a FIRST-CLASS config surface.
//! The SAME `LoadPlan` (and the same `send_raw` hot loop) reached three
//! equivalent ways — none the lesser:
//!
//!   rekt_plan examples/rekt.toml         # conflaguration TOML (+ REKT_* env overlay)
//!   rekt_plan http://127.0.0.1:8080/     # fluent builder
//!   REKT_TARGET=http://127.0.0.1:8080/ rekt_plan   # typed, per-key env
//!
//! The config resolves ONCE here; the run then drives the identical
//! monomorphized `H1ClientUpstream::send_raw` loop the CLI bench uses — the
//! first-class config surface costs nothing on the hot path.

use std::fs;

// the typed-env loader is the `Settings` trait's `from_env`; bring it in scope.
use conflaguration::Settings as _;
use rekt::error::Error;
use rekt::plan::LoadPlan;

fn main() -> Result<(), Error> {
    let plan = match std::env::args().nth(1).as_deref() {
        // a TOML path → the file half of the config surface.
        Some(path) if path.ends_with(".toml") => {
            let text = fs::read_to_string(path).map_err(|err| Error::Engine(err.to_string()))?;
            LoadPlan::from_toml(&text)?
        }
        // a bare URL → the fluent builder surface (defaults fill the rest).
        Some(url) => LoadPlan::builder()
            .target(url.to_string())
            .build(),
        // nothing → pure typed env (REKT_TARGET / REKT_CONNECTIONS_PER_CORE / ...).
        None => LoadPlan::from_env().map_err(|err| Error::Engine(err.to_string()))?,
    };

    println!("plan: {plan:?}");
    let throughput = plan.run()?;
    println!("rekt: {} completed, {} errors", throughput.completed, throughput.errors);
    println!("rekt: Requests/sec: {:.2}", throughput.per_sec());
    Ok(())
}
