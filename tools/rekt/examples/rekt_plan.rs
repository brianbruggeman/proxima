//! Drive a load plan through a FIRST-CLASS config surface.
//!
//!   rekt_plan examples/rekt.toml   # conflaguration TOML (+ REKT_* env overlay)
//!   rekt_plan                      # typed, per-key env for the machine knobs
//!
//! `LoadPlan` and `Scenario` used to be two config types with an overlapping
//! `target`. They collapsed: the plan carries the machine knobs and the gates,
//! and holds as many scenarios as the file names. A scenario stage that names no
//! `rate` runs flat out — which is what the old standalone throughput config was.

use std::fs;

// the typed-env loader is the `Settings` trait's `from_env`; bring it in scope.
use conflaguration::Settings as _;
use rekt::error::Error;
use rekt::scenario::LoadPlan;

fn main() -> Result<(), Error> {
    let plan = match std::env::args().nth(1) {
        Some(path) => {
            let text = fs::read_to_string(&path).map_err(|err| Error::Engine(err.to_string()))?;
            LoadPlan::from_toml(&text)?
        }
        None => LoadPlan::from_env().map_err(|err| Error::Engine(err.to_string()))?,
    };

    println!("plan: {plan:?}");
    let metrics = rekt::engine::run(&plan)?;
    let (text, passed) = rekt::report::render(&metrics, &plan);
    print!("{text}");
    if passed { Ok(()) } else { std::process::exit(1) }
}
