use std::process::ExitCode;

use rekt::error::Error;
use rekt::scenario::LoadPlan;

fn main() -> ExitCode {
    match run() {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("rek: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, Error> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rek.toml".into());
    let text = std::fs::read_to_string(&path)?;
    let plan = LoadPlan::from_toml(&text)?;

    #[cfg(feature = "scheduler")]
    let metrics = rekt::engine::run(&plan)?;

    // without `scheduler` there is no engine to drive; the run is a parse check.
    #[cfg(not(feature = "scheduler"))]
    let metrics = rekt::report::store();

    let (text, passed) = rekt::report::render(&metrics, &plan);
    print!("{text}");
    Ok(passed)
}
