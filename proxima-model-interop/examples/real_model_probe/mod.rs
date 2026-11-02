#![allow(dead_code)]
use std::fs::File;

use memmap2::{Mmap, MmapOptions};
use proxima_model_interop::prefault;

pub fn map_checkpoint(path: &str, env_var: &str) -> Result<Mmap, String> {
    let file = File::open(path).map_err(|error| {
        format!("no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path ({error})")
    })?;
    // SAFETY: read-only private mapping; the probe is the only process touching the fixture.
    unsafe { MmapOptions::new().map(&file) }.map_err(|error| format!("mmap {path}: {error}"))
}

pub fn prefault_if_requested(file_bytes: &[u8]) -> Result<bool, String> {
    let enabled = std::env::var("PROXIMA_PREFAULT").is_ok_and(|value| value == "1");
    if enabled {
        prefault(file_bytes).map_err(|error| format!("prefault the gguf mapping: {error}"))?;
    }
    Ok(enabled)
}

pub fn env_prompt(default_prompt: &str) -> String {
    std::env::var("PROXIMA_PROMPT").unwrap_or_else(|_| default_prompt.to_string())
}

pub fn env_max_tokens(default_tokens: usize) -> usize {
    std::env::var("PROXIMA_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_tokens)
}

#[cfg(all(feature = "metal", feature = "instrument"))]
pub mod telemetry {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use proxima_telemetry::export::Exporter;
    use proxima_telemetry::recorder::Recorder;

    pub struct TelemetryProbe {
        recorder: Arc<Recorder>,
        drained_total: Arc<AtomicUsize>,
    }

    impl TelemetryProbe {
        pub fn drain_and_total(&self) -> usize {
            let final_pass = self.recorder.drain();
            self.drained_total.fetch_add(final_pass, Ordering::Relaxed) + final_pass
        }
    }

    pub fn install_stdout_telemetry() -> Result<TelemetryProbe, String> {
        proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
            "debug",
        ));
        let recorder = Recorder::builder()
            .ring_capacity(65536)
            .export(Exporter::std())
            .map_err(|error| format!("console exporter: {error:?}"))?
            .install()
            .map_err(|error| format!("stdout telemetry recorder: {error:?}"))?;
        let drained_total = Arc::new(AtomicUsize::new(0));
        let pump_recorder = Arc::clone(&recorder);
        let pump_total = Arc::clone(&drained_total);
        std::thread::Builder::new()
            .name("telemetry-console-drain".to_string())
            .spawn(move || {
                loop {
                    let drained = pump_recorder.drain();
                    pump_total.fetch_add(drained, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
            .map_err(|error| format!("spawn the telemetry drain thread: {error}"))?;
        Ok(TelemetryProbe {
            recorder,
            drained_total,
        })
    }
}
