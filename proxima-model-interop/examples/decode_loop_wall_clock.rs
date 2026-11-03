//! Wall-clock of the cached greedy decode loop over a host-local openchat
//! checkpoint, through the public `Pipe::call` path: one `LoadedModel::load`,
//! one call of `max_tokens` (default 24), the per-token mean printed.
//!
//! `PROXIMA_OPENCHAT_GGUF` names the checkpoint (default: `ServingConfig`'s
//! own model path); `PROXIMA_PROMPT`, `PROXIMA_MAX_TOKENS` and
//! `PROXIMA_PREFAULT=1` shape the run. With `--features instrument` it also
//! prints the matmul-arm and cohort breakdown the run accumulated.
//!
//! `cargo run -p proxima-model-interop --release --features std --example decode_loop_wall_clock`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "real_model_probe/mod.rs"]
mod real_model_probe;

use std::process::ExitCode;
use std::time::Instant;

use proxima_model_interop::{LoadedModel, ServingConfig};

const DEFAULT_PROMPT: &str = "GPT4 Correct User: Write a Python function that returns the nth Fibonacci number.<|end_of_turn|>GPT4 Correct Assistant:";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("decode_loop_wall_clock: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let model_path = std::env::var("PROXIMA_OPENCHAT_GGUF")
        .unwrap_or_else(|_| ServingConfig::default().model_path.to_string());
    let mapped = real_model_probe::map_checkpoint(&model_path, "PROXIMA_OPENCHAT_GGUF")?;
    let file_bytes: &[u8] = &mapped;
    let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
        .map_err(|error| format!("parse the openchat gguf: {error}"))?;
    real_model_probe::prefault_if_requested(file_bytes)?;

    let model = LoadedModel::load(&parsed, file_bytes)
        .map_err(|error| format!("load the openchat checkpoint: {error}"))?;
    let prompt = real_model_probe::env_prompt(DEFAULT_PROMPT);
    let max_tokens = real_model_probe::env_max_tokens(24);

    println!("prompt={prompt:?} max_tokens={max_tokens}");
    let decode_start = Instant::now();
    let serving_config = ServingConfig {
        exact_activations: std::env::var("PROXIMA_EXACT_ACTIVATIONS")
            .map_or(true, |value| value != "0"),
        ..ServingConfig::default()
    };
    let generated = model
        .generate_with_serving_config(&prompt, max_tokens, serving_config)
        .map_err(|error| format!("generate with the serving config: {error}"))?;
    let total_elapsed = decode_start.elapsed();
    let tokens_generated = generated.0.len();
    let mean_ms_per_token = total_elapsed.as_secs_f64() * 1000.0 / tokens_generated.max(1) as f64;

    println!(
        "decode_summary tokens_generated={tokens_generated} stopped_by_eos={} total_wall_clock_ms={:.3} mean_ms_per_token={mean_ms_per_token:.3} generated_text={:?} ids={:?}",
        generated.2,
        total_elapsed.as_secs_f64() * 1000.0,
        generated.1,
        generated.0
    );
    if let Ok(forced_text) = std::env::var("PROXIMA_FORCED_IDS") {
        let forced: Vec<u32> = forced_text
            .split(',')
            .filter_map(|id| id.trim().parse().ok())
            .collect();
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
            .map_err(|error| format!("vocab: {error:?}"))?;
        let wants_bos = vocab
            .add_bos_token()
            .unwrap_or_else(|| vocab.bos_token_id().is_some());
        let prompt_ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, wants_bos, false)
            .map_err(|error| format!("tokenize: {error:?}"))?;
        let probe_config = ServingConfig {
            exact_activations: std::env::var("PROXIMA_EXACT_ACTIVATIONS")
                .map_or(true, |value| value != "0"),
            ..ServingConfig::default()
        };
        let rows = model
            .teacher_forced_logits_from_ids(&prompt_ids, &forced, &probe_config)
            .map_err(|error| format!("teacher forced logits: {error:?}"))?;
        for (index, row) in rows.iter().enumerate().take(forced.len()) {
            let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
            ranked.truncate(5);
            println!(
                "forced_top5 index={index} top5={ranked:?} forced_id={} forced_logit={}",
                forced[index], row[forced[index] as usize]
            );
        }
    }

    #[cfg(feature = "instrument")]
    {
        use core::sync::atomic::Ordering;
        use proxima_tensor::instrument::cohort as cohort_diag;
        // execution witness for WHICH quantized matmul arm ran: q4k_macs
        // is only ever incremented inside the packed-int8 `Q4K` branch,
        // so a nonzero count is proof the dequantize-then-fold codec did
        // not carry the forward.
        let quant = proxima_tensor::instrument::matmul_dispatch_totals();
        let q4k_ns = proxima_tensor::instrument::ticks_to_nanos(quant.q4k_call_ticks);
        std::println!(
            "quant_arm q4k_macs={} q4k_ms={:.3} q4k_ns_per_mac={:.5} q5k_macs={} q6k_macs={} q5k_f32_calls={} q6k_f32_calls={} reduce_quantized_calls={} workers_calls={} workers_none={}",
            quant.q4k_macs,
            q4k_ns as f64 / 1e6,
            if quant.q4k_macs == 0 {
                0.0
            } else {
                q4k_ns as f64 / quant.q4k_macs as f64
            },
            quant.q5k_macs,
            quant.q6k_macs,
            quant.q5k_f32_calls,
            quant.q6k_f32_calls,
            quant.reduce_quantized_calls,
            // `workers_calls == workers_none` is the mechanical proof
            // `PROXIMA_MATMUL_WORKERS=1` actually took effect
            // (`cpu::quantized_matmul_workers`'s own `workers > 1`
            // gate) -- a prior session's single-threaded comparison
            // was invalidated by never checking this.
            quant.workers_calls,
            quant.workers_none
        );
        // is the matmul bucket actually KERNEL, or is it orchestration?
        // reduce_quantized is the whole bucket; q4k/q5k/q6k_call are the
        // packed kernels inside it; the rest is dispatch the bucket also
        // pays for. printed cumulatively -- subtract a MAX_TOKENS=1 run
        // to isolate decode.
        let ms = |ticks: u64| proxima_tensor::instrument::ticks_to_nanos(ticks) as f64 / 1e6;
        std::println!(
            "matmul_split bucket_ms={:.3} kernel_ms={:.3} quantize_activation_ms={:.3} transpose_ms={:.3} setup_ms={:.3} spawn_ms={:.3} own_chunk_ms={:.3} recv_wait_ms={:.3} dispatch_calls={} position_loop_iters={}",
            ms(quant.reduce_quantized_ticks),
            ms(quant.q4k_call_ticks) + ms(quant.q5k_call_ticks) + ms(quant.q6k_call_ticks),
            ms(quant.quantize_activation_ticks),
            ms(quant.q4k_transpose_ticks),
            ms(quant.setup_ticks),
            ms(quant.spawn_ticks),
            ms(quant.own_chunk_ticks),
            ms(quant.recv_wait_ticks),
            quant.calls,
            quant.position_loop_iters
        );
        // `matmul_split` above is the UNBATCHED population only
        // (`node_kind=reduce_matmul_quantized`, `dispatch_calls`'s own
        // denominator, every field a nested subset of that line's own
        // `bucket_ms`); this line is the `cohort-staged-graph` folded
        // population (`node_kind=staged_batch`) -- ROW97/98's dominant
        // bucket, previously invisible below its own outer wall-clock
        // total. Its own `quantize_ms` is a dedicated counter
        // (`STAGED_MATMUL_QUANTIZE_TICKS`), not shared with the line
        // above's `quantize_activation_ms`, so each line's fields stay
        // internally additive.
        std::println!(
            "matmul_split_staged round_ms={:.3} quantize_ms={:.3} transpose_ms={:.3} macs={} nodes={} ns_per_mac={:.5}",
            ms(quant.staged_round_ticks),
            ms(quant.staged_quantize_ticks),
            ms(quant.staged_transpose_ticks),
            quant.staged_macs,
            quant.staged_nodes,
            if quant.staged_macs == 0 {
                0.0
            } else {
                proxima_tensor::instrument::ticks_to_nanos(quant.staged_round_ticks) as f64
                    / quant.staged_macs as f64
            },
        );
        let rounds = cohort_diag::ROUNDS.load(Ordering::Relaxed);
        let parks = cohort_diag::PARKS.load(Ordering::Relaxed);
        let unpark_rounds = cohort_diag::UNPARK_ROUNDS.load(Ordering::Relaxed);
        let spin_hits = cohort_diag::SPIN_HITS.load(Ordering::Relaxed);
        let immediate_hits = cohort_diag::IMMEDIATE_HITS.load(Ordering::Relaxed);
        std::println!(
            "cohort_summary rounds={rounds} parks={parks} unpark_rounds={unpark_rounds} spin_hits={spin_hits} immediate_hits={immediate_hits}"
        );
        for slot in 0..cohort_diag::MAX_SLOTS {
            let slot_rounds = cohort_diag::SLOT_ROUNDS[slot].load(Ordering::Relaxed);
            if slot_rounds == 0 {
                continue;
            }
            let first_claim_ns = cohort_diag::SLOT_FIRST_CLAIM_NANOS[slot].load(Ordering::Relaxed);
            let tail_ns = cohort_diag::SLOT_TAIL_NANOS[slot].load(Ordering::Relaxed);
            let compute_ns = cohort_diag::SLOT_COMPUTE_NANOS[slot].load(Ordering::Relaxed);
            let chunks = cohort_diag::SLOT_CHUNKS[slot].load(Ordering::Relaxed);
            std::println!(
                "cohort_slot slot={slot} rounds={slot_rounds} chunks={chunks} first_claim_ms={:.4} compute_ms={:.4} tail_ms={:.4}",
                first_claim_ns as f64 / slot_rounds as f64 / 1e6,
                compute_ns as f64 / slot_rounds as f64 / 1e6,
                tail_ns as f64 / slot_rounds as f64 / 1e6,
            );
        }
    }

    if tokens_generated > max_tokens {
        return Err(format!(
            "decode loop exceeded the requested budget: {tokens_generated} > {max_tokens}"
        ));
    }
    if generated.2 && tokens_generated >= max_tokens {
        return Err("an eos stop must produce strictly fewer ids than the full budget".to_string());
    }
    if !generated.2 && tokens_generated != max_tokens {
        return Err(format!(
            "budget exhaustion must produce exactly one id per step: {tokens_generated} != {max_tokens}"
        ));
    }
    if generated.1.is_empty() {
        return Err("degenerate control: decode loop produced no text".to_string());
    }
    Ok(())
}
