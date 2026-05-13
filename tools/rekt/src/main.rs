use std::process::ExitCode;

use rekt::error::Error;
use rekt::scenario::Scenario;

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
    let scenario = Scenario::from_toml(&text)?;

    println!("target: {}", scenario.target_url);

    #[cfg(feature = "scheduler")]
    let recorder = rekt::engine::run(&scenario)?;

    #[cfg(not(feature = "scheduler"))]
    let recorder = {
        // mock engine: the default-off build has no proxima to drive.
        let mut target = rekt::driver::MockTarget::new();
        rekt::driver::run(&scenario, &mut target)
    };

    let report = recorder.report(&scenario.thresholds);
    print!("{}", report.render());
    Ok(report.passed)
}
