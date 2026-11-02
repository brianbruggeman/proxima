//! Time-to-first-token of an ~850-950-token qwen3 prompt (`PREFIX.repeat(5)`),
//! three runs against a fresh `BackendRuntime` each so no run benefits from
//! another's warm dispatch-plan cache. `max_tokens: 1` isolates prefill's own
//! step-0 evaluation from any decode-step cost. An untimed 8-token pass
//! prints the greedy id sequence so two builds with different feature sets
//! can be diffed byte-for-byte: a speedup that changes the greedy decode is a
//! correctness regression, not a win.
//!
//! `PROXIMA_QWEN3_GGUF` names the checkpoint.
//!
//! `cargo run -p proxima-model-interop --release --features std,metal --example prefill_ttft_850`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "real_model_probe/mod.rs"]
mod real_model_probe;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod ttft {
    use std::time::Instant;

    use proxima_model_interop::probes;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, PromptCacheConfig, ServingConfig};

    use super::real_model_probe;

    const PREFIX: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";
    const QWEN3_BLOB: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6";
    const TIMED_RUNS: usize = 3;

    fn math_mode_from_env() -> Result<omega::MathMode, String> {
        match std::env::var("PROXIMA_MATH_MODE").as_deref() {
            Ok("safe") => Ok(omega::MathMode::Safe),
            Ok("fast") => Ok(omega::MathMode::Fast),
            Ok("relaxed") | Err(_) => Ok(omega::MathMode::Relaxed),
            Ok(other) => Err(format!(
                "PROXIMA_MATH_MODE={other}: expected safe, relaxed, or fast"
            )),
        }
    }

    fn greedy_config() -> Result<ServingConfig<'static>, String> {
        Ok(ServingConfig {
            kv_cache_key_quant: proxima_gguf::GgmlType::F32,
            kv_cache_value_quant: proxima_gguf::GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            math_mode: math_mode_from_env()?,
            temperature: 0.0,
            prompt_cache: PromptCacheConfig::off(),
            ..ServingConfig::default()
        })
    }

    pub fn run() -> Result<(), String> {
        let model_path =
            std::env::var("PROXIMA_QWEN3_GGUF").unwrap_or_else(|_| QWEN3_BLOB.to_string());
        let mapped = real_model_probe::map_checkpoint(&model_path, "PROXIMA_QWEN3_GGUF")?;
        let file_bytes: &[u8] = &mapped;
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .map_err(|error| format!("parse the qwen3 gguf: {error}"))?;
        let model = LoadedModel::load(&parsed, file_bytes)
            .map_err(|error| format!("load the qwen3 checkpoint: {error}"))?;
        let serving_config = greedy_config()?;
        let prompt = PREFIX.repeat(5);

        let mut ttft_ms: Vec<f64> = Vec::with_capacity(TIMED_RUNS);
        let mut prompt_token_count = 0usize;
        for _run in 0..TIMED_RUNS {
            let start = Instant::now();
            let probe = probes::decode(&model, &prompt, 1, &serving_config)
                .map_err(|error| format!("prefill for one timed TTFT run: {error}"))?;
            ttft_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            prompt_token_count = probe.prompt_token_count;
        }
        ttft_ms.sort_by(f64::total_cmp);
        let (min_ms, med_ms, max_ms) = (ttft_ms[0], ttft_ms[1], ttft_ms[2]);
        let tokens_per_sec_at_median = prompt_token_count as f64 / (med_ms / 1000.0);
        println!(
            "prefill_ttft_850 prompt_token_count={prompt_token_count} ttft_ms_min={min_ms:.1} ttft_ms_med={med_ms:.1} ttft_ms_max={max_ms:.1} tokens_per_sec_at_median={tokens_per_sec_at_median:.1}"
        );
        if prompt_token_count <= 700 {
            return Err(format!(
                "fixture prompt must tokenize past 700 rows (got {prompt_token_count})"
            ));
        }

        let identity = probes::decode(&model, &prompt, 8, &serving_config).map_err(|error| {
            format!("greedy-decode 8 tokens for the identity comparison: {error}")
        })?;
        println!("prefill_ttft_850 greedy_eight_token_ids={:?}", identity.ids);
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> std::process::ExitCode {
    match ttft::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("prefill_ttft_850: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("prefill_ttft_850 requires --features std,metal on macOS; skipped");
}
