//! Per-op GPU attribution of ONE decode or prefill step: arms
//! `PROXIMA_METAL_OP_PROFILE_STEP` so `run_decode_loop` swaps that step from
//! the batched `execute_plan` to one command buffer per `BoundOp`, with a
//! console telemetry recorder installed so the `op_profile_*` events print.
//!
//! `op_profile_step <probe>` with probe one of `openchat-decode-step3`
//! (`PROXIMA_OPENCHAT_GGUF`, step 3, 5 tokens), `openchat-prefill-step0`
//! (`PROXIMA_OPENCHAT_GGUF`, the whole prompt in one prefill) or
//! `qwen3-prefill-step0` (`PROXIMA_QWEN3_GGUF`, an ~850-token prompt).
//!
//! `cargo run -p proxima-model-interop --release --features std,metal,instrument --example op_profile_step -- <probe>`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#[path = "real_model_probe/mod.rs"]
mod real_model_probe;

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
mod profile {
        use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, PromptCacheConfig, ServingConfig};
    use proxima_numeric::NumericFormat;

    use super::real_model_probe;

    const OPENCHAT_PROMPT: &str = "GPT4 Correct User: Write a Python function that returns the nth Fibonacci number.<|end_of_turn|>GPT4 Correct Assistant:";
    const PREFIX: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";
    const QWEN3_BLOB: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6";

    fn openchat_config() -> ServingConfig<'static> {
        ServingConfig {
            kv_cache_key_quant: NumericFormat::Float32,
            kv_cache_value_quant: NumericFormat::Float32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        }
    }

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

    fn qwen3_config() -> Result<ServingConfig<'static>, String> {
        Ok(ServingConfig {
            kv_cache_key_quant: NumericFormat::Float32,
            kv_cache_value_quant: NumericFormat::Float32,
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

    fn arm_profile_step(step: &str) {
        // SAFETY: set before any decode work starts; the only other thread is
        // the telemetry drain pump, which never reads the environment.
        unsafe { std::env::set_var("PROXIMA_METAL_OP_PROFILE_STEP", step) };
    }

    fn disarm_profile_step() {
        // SAFETY: same single-writer argument as `arm_profile_step`.
        unsafe { std::env::remove_var("PROXIMA_METAL_OP_PROFILE_STEP") };
    }

    fn require_flushed(flushed: usize) -> Result<(), String> {
        if flushed == 0 {
            return Err("the op-profile step emitted no op_profile telemetry".to_string());
        }
        Ok(())
    }

    fn openchat(step: &str, max_tokens: usize) -> Result<(), String> {
        let model_path = std::env::var("PROXIMA_OPENCHAT_GGUF")
            .unwrap_or_else(|_| ServingConfig::default().model_path.to_string());
        let telemetry = real_model_probe::telemetry::install_stdout_telemetry()?;
        let mapped = real_model_probe::map_checkpoint(&model_path, "PROXIMA_OPENCHAT_GGUF")?;
        let file_bytes: &[u8] = &mapped;
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .map_err(|error| format!("parse the openchat gguf: {error}"))?;
        real_model_probe::prefault_if_requested(file_bytes)?;
        let model = LoadedModel::load_with_serving(&parsed, file_bytes, &openchat_config())
            .map_err(|error| format!("load the openchat checkpoint: {error}"))?;
        let prompt = real_model_probe::env_prompt(OPENCHAT_PROMPT);

        arm_profile_step(step);
        let generated = model.generate_with_serving_config(&prompt, max_tokens, openchat_config())
            .map_err(|error| format!("generate through the metal backend: {error}"))?;
        disarm_profile_step();

        require_flushed(telemetry.drain_and_total())?;
        println!(
            "op_profile_run tokens_generated={} stopped_by_eos={} generated_text={:?}",
            generated.0.len(),
            generated.2,
            generated.1
        );
        if generated.1.is_empty() {
            return Err("degenerate control: metal decode loop produced no text".to_string());
        }
        Ok(())
    }

    fn qwen3_prefill() -> Result<(), String> {
        let model_path =
            std::env::var("PROXIMA_QWEN3_GGUF").unwrap_or_else(|_| QWEN3_BLOB.to_string());
        let mapped = real_model_probe::map_checkpoint(&model_path, "PROXIMA_QWEN3_GGUF")?;
        let file_bytes: &[u8] = &mapped;
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .map_err(|error| format!("parse the qwen3 gguf: {error}"))?;
        let serving_config = qwen3_config()?;
        let model = LoadedModel::load_with_serving(&parsed, file_bytes, &serving_config)
            .map_err(|error| format!("load the qwen3 checkpoint: {error}"))?;
        let prompt = PREFIX.repeat(5);
        let telemetry = real_model_probe::telemetry::install_stdout_telemetry()?;

        arm_profile_step("0");
        let probe = real_model_probe::stream_decode(&model, &prompt, 1, serving_config)
            .map_err(|error| format!("prefill the ~850-token prompt with step 0 armed: {error}"))?;
        disarm_profile_step();

        let flushed = telemetry.drain_and_total();
        println!(
            "prefill_step_zero_op_profile prompt_token_count={} telemetry_records_flushed={flushed}",
            probe.prompt_token_count
        );
        if probe.prompt_token_count <= 700 {
            return Err(format!(
                "fixture prompt must tokenize past 700 rows (got {})",
                probe.prompt_token_count
            ));
        }
        require_flushed(flushed)
    }

    pub fn run(probe: &str) -> Result<(), String> {
        match probe {
            "openchat-decode-step3" => openchat("3", 5),
            "openchat-prefill-step0" => openchat("0", 1),
            "qwen3-prefill-step0" => qwen3_prefill(),
            other => Err(format!(
                "unknown probe {other:?}; expected openchat-decode-step3, openchat-prefill-step0 or qwen3-prefill-step0"
            )),
        }
    }
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn main() -> std::process::ExitCode {
    let probe = std::env::args().nth(1).unwrap_or_default();
    match profile::run(&probe) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("op_profile_step: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", feature = "instrument", target_os = "macos")))]
fn main() {
    println!("op_profile_step requires --features std,metal,instrument on macOS; skipped");
}
