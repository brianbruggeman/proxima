//! What the host encode costs one decode step: the live step against the same
//! captured step replayed with no host encoding on the critical path.
//!
//! One decode runs with `PROXIMA_CAPTURE_LIVE` on a chosen step; its dispatches
//! are kept. Then `SEB_ROUNDS` rounds, each running three arms in rotating
//! order inside this one process so box state hits every arm alike:
//! - `live`: a whole `generate_streaming` of `SEB_TOKENS` tokens with capture
//!   off; the figure is the median gap between token events from the fourth
//!   token on, and the cell covers the decode window (first token to last).
//! - `replay_one`: the captured step as ONE command buffer
//!   (`CapturedDispatch::time_gpu_sequence_ns`); the figure is the GPU span, the
//!   step with no host encode and no inter-chunk gap.
//! - `replay_chunks`: the captured step as one command buffer per original chunk
//!   (`time_gpu_chunk_sequences_ns`); the figure is the sum of the chunk spans,
//!   the live chunking with the host out of the way.
//!
//! Every cell prints wall, process CPU, CPU%, peak RSS, physical footprint,
//! Metal allocated bytes and load before and after (`cell_resources/cell.rs`).
//! Knobs: `SEB_STEP` (23), `SEB_ROUNDS` (21), `SEB_TOKENS` (24),
//! `PROXIMA_GEMMA4_E2B_GGUF`, `PROXIMA_PROMPT_FILE`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "cell_resources/cell.rs"]
mod cell;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::fs::File;
    use std::time::Instant;

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};

    use super::cell::Cell;

    const WARMUP_TOKENS: usize = 3;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        sorted[sorted.len() / 2]
    }

    fn cov_percent(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / (values.len() as f64 - 1.0).max(1.0);
        100.0 * variance.sqrt() / mean
    }

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

    struct Live {
        per_token_ms: f64,
        cell_line: String,
    }

    fn live_generation(model: &LoadedModel<'_>, prompt: &str, tokens: usize) -> Live {
        let mut stamps: Vec<Instant> = Vec::new();
        let mut cell: Option<Cell> = None;
        let mut cell_line = String::new();
        let mut on_token = |_event: TokenEvent<'_>| {
            stamps.push(Instant::now());
            if stamps.len() == 1 {
                cell = Some(Cell::begin());
            }
            if stamps.len() == tokens
                && let Some(open) = cell.take()
            {
                cell_line = open.end("live_decode_window");
            }
            ControlFlow::Continue(())
        };
        model
            .generate_streaming(prompt, tokens, serving_config(), &mut on_token)
            .expect("greedy decode");
        let gaps: Vec<f64> = stamps
            .windows(2)
            .skip(WARMUP_TOKENS)
            .map(|pair| (pair[1] - pair[0]).as_secs_f64() * 1e3)
            .collect();
        assert!(!gaps.is_empty(), "N==0: no decode gaps timed");
        Live { per_token_ms: median(&gaps), cell_line }
    }

    pub fn run() {
        let step = env_usize("SEB_STEP", 23);
        let rounds = env_usize("SEB_ROUNDS", 21);
        let tokens = env_usize("SEB_TOKENS", 24);
        let path = std::env::var("PROXIMA_GEMMA4_E2B_GGUF").expect("PROXIMA_GEMMA4_E2B_GGUF");
        let file = File::open(path).expect("open checkpoint");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map checkpoint");
        let parsed = parse_complete(&bytes).expect("parse header");
        let model = LoadedModel::load(&parsed, &bytes).expect("bind model");
        let prompt = std::fs::read_to_string(
            std::env::var("PROXIMA_PROMPT_FILE").expect("PROXIMA_PROMPT_FILE"),
        )
        .expect("read prompt file");
        // SAFETY: called before any thread other than the telemetry-free main one exists.
        unsafe {
            std::env::set_var("PROXIMA_CAPTURE_NODES", "all");
            std::env::set_var("PROXIMA_CAPTURE_STEPS", step.to_string());
            std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
        }
        live_generation(&model, &prompt, step + 1);
        let captured: Vec<CapturedDispatch> = omega::take_captured_dispatches()
            .into_iter()
            .filter(|dispatch| dispatch.grid.threads > 0)
            .collect();
        let replayable: Vec<CapturedDispatch> = captured
            .into_iter()
            .filter(|dispatch| dispatch.unreplayable.is_none())
            .collect();
        assert!(!replayable.is_empty(), "N==0: nothing captured");
        let chunk_count = replayable
            .iter()
            .map(|dispatch| dispatch.chunk_index)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        println!(
            "seb capture step={step} dispatches={} chunks={chunk_count} rounds={rounds} tokens={tokens}",
            replayable.len()
        );
        // SAFETY: single-threaded here; later generations must not capture again.
        unsafe {
            std::env::remove_var("PROXIMA_CAPTURE_NODES");
            std::env::remove_var("PROXIMA_CAPTURE_STEPS");
            std::env::remove_var("PROXIMA_CAPTURE_LIVE");
        }
        let mut live_ms = Vec::new();
        let mut one_ms = Vec::new();
        let mut chunk_ms = Vec::new();
        for round in 0..rounds {
            for offset in 0..3 {
                match (round + offset) % 3 {
                    0 => {
                        let live = live_generation(&model, &prompt, tokens);
                        println!("seb live round={round} per_token_ms={:.4}", live.per_token_ms);
                        println!("{}", live.cell_line);
                        live_ms.push(live.per_token_ms);
                    }
                    1 => {
                        let cell = Cell::begin();
                        let span = CapturedDispatch::time_gpu_sequence_ns(&replayable)
                            .expect("one-buffer replay");
                        println!("seb replay_one round={round} gpu_span_ms={:.4}", span / 1e6);
                        println!("{}", cell.end("replay_one"));
                        one_ms.push(span / 1e6);
                    }
                    _ => {
                        let cell = Cell::begin();
                        let spans = CapturedDispatch::time_gpu_chunk_sequences_ns(&replayable)
                            .expect("chunked replay");
                        let sum: f64 = spans.iter().sum();
                        println!(
                            "seb replay_chunks round={round} chunk_span_sum_ms={:.4} chunks={}",
                            sum / 1e6,
                            spans.len()
                        );
                        println!("{}", cell.end("replay_chunks"));
                        chunk_ms.push(sum / 1e6);
                    }
                }
            }
        }
        for (label, values) in [
            ("live_per_token_ms", &live_ms),
            ("replay_one_gpu_span_ms", &one_ms),
            ("replay_chunks_span_sum_ms", &chunk_ms),
        ] {
            println!(
                "seb summary arm={label} n={} median={:.4} min={:.4} max={:.4} cov_pct={:.2}",
                values.len(),
                median(values),
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                cov_percent(values)
            );
        }
        println!(
            "seb bound live_minus_replay_one_ms={:.4} live_minus_replay_chunks_ms={:.4}",
            median(&live_ms) - median(&one_ms),
            median(&live_ms) - median(&chunk_ms)
        );
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
