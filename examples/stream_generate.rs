//! ONE real verification for [`proxima_model_interop::LoadedModel::generate_streaming`]:
//! does a real checkpoint's decode loop actually hand `on_token` a
//! [`proxima_model_interop::TokenEvent`] per step, with `text_piece`s that
//! concatenate to the same string the non-streaming
//! `generate_with_serving_config` would return? Composes exactly the same
//! two primitives `examples/gguf_generate.rs` does
//! ([`proxima_gguf::pipe::parse_complete`] +
//! [`proxima_model_interop::LoadedModel::load`]), the CPU backend only
//! (`gpu_layers = 0`) so this example needs no Metal device to prove the
//! callback itself works.
//!
//! Run: `cargo run --example stream_generate -- <gguf-path> <prompt>
//! <max-tokens>` (all three positional arguments optional; defaults print
//! below).

use std::env;
use std::time::Instant;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::Control;
use proxima_model_interop::LoadedModel;
use proxima_model_interop::Phase;
use proxima_model_interop::ServingConfig;
use proxima_model_interop::TokenEvent;

fn main() {
    let args: Vec<String> = env::args().collect();
    let gguf_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| ServingConfig::default().model_path.to_string());
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox".to_string());
    let max_tokens: usize = args
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);

    println!("gguf_path = {gguf_path}");
    println!("prompt = {prompt:?}");
    println!("max_tokens = {max_tokens}");

    let gguf_file = match std::fs::File::open(&gguf_path) {
        Ok(gguf_file) => gguf_file,
        Err(error) => {
            eprintln!("open the gguf file: {error}");
            std::process::exit(1);
        }
    };
    // SAFETY: the checkpoint file is not written or truncated by any
    // process while this mapping is alive for the duration of this run.
    let file_map = match unsafe { memmap2::Mmap::map(&gguf_file) } {
        Ok(file_map) => file_map,
        Err(error) => {
            eprintln!("mmap the gguf file: {error}");
            std::process::exit(1);
        }
    };
    let file_bytes: &[u8] = &file_map;
    println!("file_bytes = {} bytes", file_bytes.len());

    let parsed = match parse_complete(file_bytes) {
        Ok(parsed) => parsed,
        Err(error) => {
            println!("GGUF PARSE FAILED: {error}");
            std::process::exit(1);
        }
    };

    let model = match LoadedModel::load(&parsed, file_bytes) {
        Ok(model) => model,
        Err(error) => {
            println!("WEIGHT LOAD FAILED: {error}");
            std::process::exit(1);
        }
    };
    println!(
        "model_name = {:?}, layer_count = {}, checkpoint_bytes = {}",
        model.model_name(),
        model.layer_count(),
        model.checkpoint_bytes()
    );

    let serving_config = ServingConfig {
        model_path: &gguf_path,
        kv_cache_key_quant: proxima_gguf::types::GgmlType::F32,
        kv_cache_value_quant: proxima_gguf::types::GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    let mut event_count = 0usize;
    let mut streamed_text = String::new();
    let started = Instant::now();
    let mut on_token = |event: TokenEvent<'_>| {
        event_count += 1;
        if event_count <= 5 {
            match event.phase {
                Phase::Prefill { prompt_tokens } => println!(
                    "event[{event_count}] PREFILL prompt_tokens={prompt_tokens} \
                     step={} elapsed_ms={}",
                    event.step, event.elapsed_ms
                ),
                Phase::Token => println!(
                    "event[{event_count}] TOKEN token_id={} text_piece={:?} step={} \
                     elapsed_ms={}",
                    event.token_id, event.text_piece, event.step, event.elapsed_ms
                ),
            }
        }
        if let Phase::Token = event.phase {
            streamed_text.push_str(event.text_piece);
        }
        Control::Continue
    };

    let outcome = model.generate_streaming(&prompt, max_tokens, serving_config, &mut on_token);
    let generate_ms = started.elapsed().as_secs_f64() * 1000.0;

    match outcome {
        Ok((ids, text, stopped_by_eos)) => {
            println!(
                "generated {} tokens in {generate_ms:.3} ms ({:.3} tok/s), stopped_by_eos={stopped_by_eos}",
                ids.len(),
                ids.len() as f64 / (generate_ms / 1000.0)
            );
            println!("returned text = {text:?}");
            println!("streamed text = {streamed_text:?}");
            println!("streamed text == returned text: {}", streamed_text == text);
        }
        Err(error) => {
            println!("GENERATE FAILED: {error}");
            std::process::exit(1);
        }
    }
}
