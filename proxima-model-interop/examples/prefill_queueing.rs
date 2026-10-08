//! Where the milliseconds between a prefill command buffer's `commit` and its
//! `scheduled` callback go: one process runs the stages named in `PQ_STAGES`
//! (comma list, in order) and the host phase events of every generation land
//! in `PQ_OUT/telemetry.log` for `queueing_summary` to split.
//!
//! Stages: `dry` runs three empty command buffers (a one-thread kernel on a 16-byte buffer, no checkpoint buffer bound); `pretouch` reads one byte of every page of the checkpoint mapping
//! (weights resident in this process before any GPU use); `small` generates two
//! tokens from the 25-token prompt in `PQ_SMALL_PROMPT` (the first command buffer
//! of the process, with shapes the long prefill does not share); `prefill` is a
//! one-token generation from the 1000-token prompt in `PROXIMA_PROMPT_FILE`,
//! its first character changed per occurrence so the prompt cache cannot skip the
//! prefill. Every stage prints a `cell` line (wall, process CPU, RSS, footprint,
//! Metal bytes, load).
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "cell_resources/cell.rs"]
mod cell;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use memmap2::{Mmap, MmapOptions};
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};
    use proxima_telemetry::export::Exporter;
    use proxima_telemetry::recorder::Recorder;

    use super::cell::Cell;

    const PAGE_BYTES: usize = 16384;

    fn serving_config() -> ServingConfig<'static> {
        ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            reasoning_budget: 0,
            dispatch_type: omega::DispatchType::Serial,
            ..ServingConfig::default()
        }
    }

    fn install_telemetry(path: &PathBuf) -> Arc<Recorder<proxima_telemetry::clock::GlobalClock>> {
        proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
            "proxima_model_interop=info,omega::metal::placements_execute_named=debug",
        ));
        let recorder = Recorder::builder()
            .ring_capacity(262_144)
            .export(Exporter::file(path))
            .expect("file exporter installs")
            .install()
            .expect("telemetry recorder installs");
        let pump_recorder = Arc::clone(&recorder);
        let pumped = Arc::new(AtomicUsize::new(0));
        thread::Builder::new()
            .name("pq-telemetry-drain".to_string())
            .spawn(move || {
                loop {
                    pumped.fetch_add(pump_recorder.drain(), Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(5));
                }
            })
            .expect("spawn telemetry drain thread");
        recorder
    }

    fn pretouch(bytes: &[u8]) -> u64 {
        let mut sum = 0u64;
        for offset in (0..bytes.len()).step_by(PAGE_BYTES) {
            sum = sum.wrapping_add(u64::from(bytes[offset]));
        }
        std::hint::black_box(sum)
    }

    fn generate(model: &LoadedModel<'_>, prompt: &str, tokens: usize) -> usize {
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        let (token_ids, _text, _stopped) = model
            .generate_streaming(prompt, tokens, serving_config(), &mut on_token)
            .expect("greedy decode");
        token_ids.len()
    }

    pub fn run() {
        let stages = std::env::var("PQ_STAGES").expect("PQ_STAGES");
        let out_dir = PathBuf::from(std::env::var("PQ_OUT").expect("PQ_OUT"));
        std::fs::create_dir_all(&out_dir).expect("create PQ_OUT");
        let recorder = install_telemetry(&out_dir.join("telemetry.log"));
        let path = std::env::var("PROXIMA_GEMMA4_E2B_GGUF").expect("PROXIMA_GEMMA4_E2B_GGUF");
        let file = File::open(path).expect("open checkpoint");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map checkpoint");
        let parsed = parse_complete(&bytes).expect("parse header");
        let load_cell = Cell::begin();
        let model = LoadedModel::load(&parsed, &bytes).expect("bind model");
        println!("pq {}", load_cell.end("load"));
        let long_prompt = std::fs::read_to_string(
            std::env::var("PROXIMA_PROMPT_FILE").expect("PROXIMA_PROMPT_FILE"),
        )
        .expect("read prompt file");
        let small_prompt = std::fs::read_to_string(
            std::env::var("PQ_SMALL_PROMPT").expect("PQ_SMALL_PROMPT"),
        )
        .expect("read small prompt");
        let mut prefills = 0usize;
        for (index, stage) in stages.split(',').enumerate() {
            let cell = Cell::begin();
            let detail = match stage {
                "pretouch" => format!("checksum={}", pretouch(&bytes)),
                "dry" => {
                    let spans: Vec<String> = (0..3)
                        .map(|_| {
                            let started = std::time::Instant::now();
                            let span = omega::time_empty_command_buffer_gpu_ns().expect("empty command buffer");
                            format!("gpu_ns={span:.0}/wall_ms={:.3}", started.elapsed().as_secs_f64() * 1e3)
                        })
                        .collect();
                    format!("empty_command_buffers=[{}]", spans.join(" "))
                }
                "small" => format!("generated={}", generate(&model, &small_prompt, 2)),
                "prefill" => {
                    let letters = ['p', 'q', 'r', 's', 't', 'u'];
                    let prompt = long_prompt.replacen("proxima", &format!("{}roxima", letters[prefills % letters.len()]), 1);
                    prefills += 1;
                    format!("generated={}", generate(&model, &prompt, 1))
                }
                other => panic!("PQ_STAGES: unknown stage `{other}`"),
            };
            println!("pq stage={stage} index={index} {detail} {}", cell.end(stage));
        }
        thread::sleep(Duration::from_millis(100));
        recorder.drain();
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    harness::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("unsupported target for this example");
    std::process::exit(1);
}
