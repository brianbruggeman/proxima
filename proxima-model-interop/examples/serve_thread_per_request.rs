//! PROXIMAOMEGA-61: what a downstream server's thread-per-request serving shape costs. The
//! model loads once on the main thread (and is warmed there, as
//! `decode_gbps_baseline` does); then `PROXIMA_REQUESTS` identical requests
//! are served. Arm A (default) gives each request a fresh `std::thread::spawn`
//! joined before the next, the shape of a downstream server's proxima provider. Arm B
//! (`PROXIMA_SAME_THREAD=1`) serves every request on one long-lived spawned
//! thread. Model loading and the `run=done` fields follow
//! `decode_gbps_baseline.rs`; this example still reads its own legacy
//! `PROXIMA_*` env surface rather than `ServingSettings`.
//!
//! `footprint_in_thread_bytes` is read on the serving thread right after the
//! request returns, `footprint_after_join_bytes` on main after the thread has
//! exited and its thread-local destructors have run (arm B reads it once,
//! after the single join, on a `run=after_join` line). The upload counters
//! are omega's process-global `RESIDENT_BUFFER_UPLOADS`,
//! `COPYING_BUFFER_UPLOADS`, `NOCOPY_BUFFER_UPLOADS` and `BLOCK_COPIED_BYTES`,
//! cumulative since process start.

use core::ops::ControlFlow;
use std::fs::File;
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use conflaguration::Settings as _;
use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_model_interop::{
    GPU_LAYERS_ALL, LoadWarm, LoadedModel, Phase, PromptCacheConfig, ServingConfig, SpeculativeSettings,
    TokenEvent,
};
use proxima_numeric::NumericFormat;
use proxima_tensor::process_usage;

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const DEFAULT_MAX_TOKENS: usize = 48;
const DEFAULT_REQUESTS: usize = 3;
const DEFAULT_PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";

fn footprint_bytes() -> u64 {
    process_usage::resident_and_footprint_bytes()
        .map(|(_, footprint)| footprint)
        .unwrap_or(0)
}

fn fnv64(text: &str) -> u64 {
    text.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn env_parsed<T: std::str::FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| anyhow!("{name}={value}: not parseable")),
        Err(_) => Ok(default),
    }
}

fn format_from_env(name: &str, default: NumericFormat) -> Result<NumericFormat> {
    match std::env::var(name).as_deref() {
        Ok(value) => NumericFormat::from_name(value)
            .ok_or_else(|| anyhow!("{name}={value}: not a NumericFormat name")),
        Err(_) => Ok(default),
    }
}

fn serving_config_from_env(speculative_settings: &SpeculativeSettings) -> Result<ServingConfig<'_>> {
    let defaults = ServingConfig::default();
    let kv_cache_type = format_from_env("PROXIMA_KV_CACHE_TYPE", defaults.kv_cache_value_quant)?;
    let kv_cache_key_type = format_from_env("PROXIMA_KV_CACHE_KEY_TYPE", kv_cache_type)?;
    let speculative = speculative_settings.as_speculative_config();
    let ignore_eos = match std::env::var("PROXIMA_IGNORE_EOS").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(other) => return Err(anyhow!("PROXIMA_IGNORE_EOS={other}: expected `0` or `1`")),
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let dispatch_type = match std::env::var("PROXIMA_DISPATCH").as_deref() {
        Ok("concurrent") => omega::DispatchType::Concurrent,
        Ok("serial") | Err(_) => omega::DispatchType::Serial,
        Ok(other) => {
            return Err(anyhow!("PROXIMA_DISPATCH={other}: expected `serial` or `concurrent`"));
        }
    };
    Ok(ServingConfig {
        ignore_eos,
        speculative,
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: kv_cache_key_type,
        kv_cache_value_quant: kv_cache_type,
        flash_attention: false,
        batch_size: env_parsed("PROXIMA_BATCH_SIZE", 0)?,
        ubatch_size: env_parsed("PROXIMA_UBATCH_SIZE", 0)?,
        reasoning_budget: 0,
        kv_bucket_tokens: env_parsed("PROXIMA_KV_BUCKET_TOKENS", defaults.kv_bucket_tokens)?,
        resident_prefill_plan_bytes: env_parsed(
            "PROXIMA_RESIDENT_PREFILL_PLAN_BYTES",
            defaults.resident_prefill_plan_bytes,
        )?,
        resident_kv_bytes: env_parsed("PROXIMA_RESIDENT_KV_BYTES", defaults.resident_kv_bytes)?,
        prompt_cache: PromptCacheConfig::off(),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        dispatch_type,
        ..defaults
    })
}

fn upload_counters() -> String {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        format!(
            "resident_uploads={} copying_uploads={} nocopy_uploads={} block_copied_bytes={}",
            omega::metal::RESIDENT_BUFFER_UPLOADS.get(),
            omega::metal::COPYING_BUFFER_UPLOADS.get(),
            omega::metal::NOCOPY_BUFFER_UPLOADS.get(),
            omega::metal::BLOCK_COPIED_BYTES.get(),
        )
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        "resident_uploads=0 copying_uploads=0 nocopy_uploads=0 block_copied_bytes=0".to_string()
    }
}

fn serve_request(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    serving_config: ServingConfig<'_>,
    request_index: usize,
) -> Result<()> {
    let thread_name = thread::current().name().unwrap_or("unnamed").to_string();
    let mut prefill_elapsed_ms: u64 = 0;
    let mut ttft_ms_precise: f64 = 0.0;
    let mut prompt_token_count: usize = 0;
    let mut previous_tick = Instant::now();
    let mut decode_step_ns_total: u64 = 0;
    let mut on_token = |event: TokenEvent<'_>| {
        let is_prefill = matches!(event.phase, Phase::Prefill { .. });
        if let Phase::Prefill { prompt_tokens } = event.phase {
            prefill_elapsed_ms = event.elapsed_ms;
            prompt_token_count = prompt_tokens;
            ttft_ms_precise = event.elapsed_ns as f64 / 1_000_000.0;
        }
        let now_tick = Instant::now();
        let step_ns = now_tick.duration_since(previous_tick).as_nanos() as u64;
        previous_tick = now_tick;
        if !is_prefill && event.step >= 1 {
            decode_step_ns_total += step_ns;
        }
        ControlFlow::Continue(())
    };
    let start = Instant::now();
    let (token_ids, text, stopped_by_eos) = model
        .generate_streaming(prompt, max_tokens, serving_config, &mut on_token)
        .map_err(|error| anyhow!("greedy decode on the real checkpoint: {error:?}"))?;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tokens_generated = token_ids.len();
    let decode_ms_per_token = if tokens_generated > 1 {
        (wall_ms - prefill_elapsed_ms as f64) / (tokens_generated - 1) as f64
    } else {
        0.0
    };
    let decode_step_ms_per_token =
        decode_step_ns_total as f64 / 1_000_000.0 / tokens_generated.saturating_sub(1).max(1) as f64;
    eprintln!(
        "serve_thread_per_request run=done request_index={request_index} thread={thread_name} \
         wall_ms={wall_ms:.3} tokens_generated={tokens_generated} prompt_token_count={prompt_token_count} \
         text_hash={:016x} stopped_by_eos={stopped_by_eos} decode_ms_per_token={decode_ms_per_token:.3} \
         decode_step_ms_per_token={decode_step_ms_per_token:.6} ttft_ms={prefill_elapsed_ms} \
         ttft_ms_precise={ttft_ms_precise:.6} footprint_in_thread_bytes={} {}",
        fnv64(&text),
        footprint_bytes(),
        upload_counters(),
    );
    Ok(())
}

fn main() -> Result<()> {
    let model_file =
        std::env::var("PROXIMA_DECODE_MODEL_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string());
    let prompt_override = std::env::var("PROXIMA_PROMPT").ok();
    let prompt: &str = prompt_override.as_deref().unwrap_or(DEFAULT_PROMPT);
    let max_tokens: usize = env_parsed("PROXIMA_MAX_TOKENS", DEFAULT_MAX_TOKENS)?;
    let requests: usize = env_parsed("PROXIMA_REQUESTS", DEFAULT_REQUESTS)?;
    let same_thread = std::env::var("PROXIMA_SAME_THREAD").is_ok_and(|value| value == "1");
    let speculative_settings = SpeculativeSettings::from_env().map_err(|error| {
        anyhow!("PROXIMA_SPECULATIVE_* env parses as the serving speculative settings: {error:?}")
    })?;
    let serving_config = serving_config_from_env(&speculative_settings)?;
    let arm = if same_thread { "same_thread" } else { "thread_per_request" };

    eprintln!(
        "serve_thread_per_request run=start arm={arm} requests={requests} max_tokens={max_tokens} \
         prompt_bytes={} batch_size={} ubatch_size={} footprint_before_load_bytes={}",
        prompt.len(),
        serving_config.batch_size,
        serving_config.ubatch_size,
        footprint_bytes(),
    );

    let file = File::open(&model_file).with_context(|| format!("open {model_file}"))?;
    // SAFETY: the mapping outlives every use below and POSIX keeps it valid after `file` drops
    let bytes: Mmap =
        unsafe { MmapOptions::new().map(&file) }.with_context(|| format!("map {model_file}"))?;
    let parsed = parse_complete(&bytes)
        .map_err(|error| anyhow!("parse gguf header of {model_file}: {error:?}"))?;
    let model = LoadedModel::load_with_serving(&parsed, &bytes, &serving_config)
        .map_err(|error| anyhow!("bind {model_file}: {error:?}"))?;
    let warm_started = Instant::now();
    let LoadWarm { buffers: warmed_buffers, rows: warmed_rows } = model
        .warm_at_load(&serving_config)
        .map_err(|error| anyhow!("warm the model at load: {error:?}"))?;
    eprintln!(
        "serve_thread_per_request warmed_buffers={warmed_buffers} warmed_rows={warmed_rows} warm_ms={:.3}",
        warm_started.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!(
        "serve_thread_per_request run=loaded footprint_after_load_bytes={} {}",
        footprint_bytes(),
        upload_counters(),
    );

    let model_ref = &model;
    if same_thread {
        thread::scope(|scope| -> Result<()> {
            thread::Builder::new()
                .name("serve-0".to_string())
                .spawn_scoped(scope, move || -> Result<()> {
                    for request_index in 0..requests {
                        serve_request(model_ref, prompt, max_tokens, serving_config, request_index)?;
                    }
                    Ok(())
                })
                .context("spawn the long-lived serving thread")?
                .join()
                .map_err(|_| anyhow!("serving thread must not panic"))?
        })?;
        eprintln!(
            "serve_thread_per_request run=after_join requests={requests} footprint_after_join_bytes={} {}",
            footprint_bytes(),
            upload_counters(),
        );
    } else {
        for request_index in 0..requests {
            thread::scope(|scope| -> Result<()> {
                thread::Builder::new()
                    .name(format!("req-{request_index}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        serve_request(model_ref, prompt, max_tokens, serving_config, request_index)
                    })
                    .context("spawn the request thread")?
                    .join()
                    .map_err(|_| anyhow!("request thread must not panic"))?
            })?;
            eprintln!(
                "serve_thread_per_request run=after_join request_index={request_index} footprint_after_join_bytes={} {}",
                footprint_bytes(),
                upload_counters(),
            );
        }
    }
    Ok(())
}
