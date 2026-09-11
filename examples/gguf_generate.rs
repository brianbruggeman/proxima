//! Gate experiment: does proxima's own tensor stack load an arbitrary GGUF
//! checkpoint and generate text end to end, or does it stop at one of the
//! three real seams -- parse, hparam/weight bind, forward compile? Composes
//! exactly two primitives, no new type: [`proxima_gguf::pipe::parse_complete`]
//! (sans-IO metadata/tensor-directory parse) and
//! [`proxima_model_interop::generate::LoadedModel::load`] +
//! `generate_with_serving_config` (the same load/decode path
//! `proxima-model-interop/tests/real_lfm2_checkpoint.rs` and
//! `proxima-model-interop/src/bind.rs`'s own `real_openchat_file` acceptance
//! test already exercise against other real checkpoints).
//!
//! `architecture_from_metadata` (`proxima-model-interop/src/bind.rs:333`)
//! reads `general.architecture` as a plain string key PREFIX -- it never
//! matches on the architecture name itself, so an architecture this crate
//! has never seen is not rejected up front. [`LoadedModel::load`]
//! (`proxima-model-interop/src/generate.rs`) is the one architecture-routing
//! seam: every checkpoint but `qwen35` still compiles
//! `mistral_cached_forward_program_with_experts`, a dense-attention-plus-MoE
//! shape with no state-space/Mamba path; `qwen35`'s own hybrid
//! attention+state-space checkpoint routes to
//! [`proxima_tensor::spec::qwen35_forward_program`] instead, interleaving
//! `append_mistral_cached_layer`'s dense-attention shape with
//! `append_qwen35_ssm_mixer`'s gated-DeltaNet mixer per
//! `crate::qwen35::Qwen35LayerKind`. Its dense-attention layers still run
//! `append_mistral_cached_layer`'s single-section RoPE rather than this
//! checkpoint's real 4-section MRoPE (`qwen35.rope.dimension_sections`) --
//! a known, documented correctness gap on those layers, not a crash -- this
//! example exists to observe which outcome a real checkpoint gets, not to
//! assume it.
//!
//! Run: `cargo run --example gguf_generate -- <gguf-path> <prompt>
//! <max-tokens> [cpu|gpu]`
//!
//! The 4th positional argument picks the backend and defaults to `gpu`.
//! Set `PROXIMA_GPU_MEMORY_LIMIT_BYTES` to make the load-time device budget
//! explicit; an over-budget GPU request is rejected before Metal allocates
//! the checkpoint (for example, `4294967296` for a 4 GiB ceiling).
//! `PROXIMA_QWEN35MOE_PRE_GATHER=1` enables the per-layer router/residency
//! boundary when an expert sidecar is attached; without it the ordinary
//! monolithic decode path is used.
//! `cpu` sets `gpu_layers = 0` and runs CPU-only, no Metal attempt, no
//! fallback. `gpu` sets `gpu_layers = GPU_LAYERS_ALL`
//! (`proxima-model-interop/src/generate.rs:856`'s `select_backend` reads
//! that exact sentinel); on a non-metal build or a build with no working
//! Metal device, `apply_serving_config`'s own rejection
//! (`proxima-model-interop/src/serving.rs:270-276`) or the runtime Metal
//! failure surfaces as an explicit, unambiguous CPU fallback below -- never
//! a silent one.

use std::env;
use std::sync::Arc;
use std::time::Instant;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::ArchitectureRegistry;
use proxima_model_interop::Control;
use proxima_model_interop::GPU_LAYERS_ALL;
use proxima_model_interop::LoadedModel;
use proxima_model_interop::Phase;
use proxima_model_interop::ServingConfig;
use proxima_model_interop::TokenEvent;

fn nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    let rank = (percentile * sorted.len()).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn print_decode_latency(token_elapsed_ms: &[u64]) {
    let Some(ttft_ms) = token_elapsed_ms.first() else {
        println!("ttft_ms = unavailable");
        println!("ttnt_sample_count = 0");
        return;
    };
    println!("ttft_ms = {ttft_ms}");

    let mut ttnt_ms: Vec<u64> = token_elapsed_ms
        .windows(2)
        .map(|window| window[1].saturating_sub(window[0]))
        .collect();
    if ttnt_ms.is_empty() {
        println!("ttnt_sample_count = 0");
        return;
    }

    let sample_count = ttnt_ms.len();
    let sum_ms: u64 = ttnt_ms.iter().sum();
    let mean_ms = sum_ms as f64 / sample_count as f64;
    let variance_ms2 = ttnt_ms
        .iter()
        .map(|latency_ms| {
            let difference_ms = *latency_ms as f64 - mean_ms;
            difference_ms * difference_ms
        })
        .sum::<f64>()
        / sample_count as f64;
    let two_standard_deviations_ms = 2.0 * variance_ms2.sqrt();
    let decode_tokens_per_second = 1000.0 / mean_ms;
    ttnt_ms.sort_unstable();

    println!("ttnt_sample_count = {sample_count}");
    println!("ttnt_min_ms = {}", ttnt_ms[0]);
    println!("ttnt_max_ms = {}", ttnt_ms[sample_count - 1]);
    println!("ttnt_mean_ms = {mean_ms:.3}");
    println!("ttnt_two_standard_deviations_ms = {two_standard_deviations_ms:.3}");
    println!("ttnt_p90_ms = {}", nearest_rank(&ttnt_ms, 90));
    println!("ttnt_p99_ms = {}", nearest_rank(&ttnt_ms, 99));
    println!("decode_tokens_per_sec = {decode_tokens_per_second:.3}");
}

#[cfg(unix)]
fn print_peak_rss() {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes the caller-owned structure on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result == 0 {
        // Darwin reports bytes; Linux reports KiB.
        let bytes = unsafe { usage.assume_init().ru_maxrss as u64 }
            * if cfg!(target_os = "macos") { 1 } else { 1024 };
        println!("peak_rss_bytes = {bytes}");
    } else {
        println!("peak_rss_bytes = unavailable");
    }
}

#[cfg(not(unix))]
fn print_peak_rss() {
    println!("peak_rss_bytes = unavailable");
}

fn print_architecture_metadata(parsed: &proxima_gguf::pipe::ParsedGguf) {
    let name = parsed
        .metadata_value("general.name")
        .and_then(proxima_gguf::value::MetadataValue::as_str)
        .unwrap_or("<missing general.name>");
    println!("general.name = {name}");

    let architecture = parsed
        .metadata_value("general.architecture")
        .and_then(proxima_gguf::value::MetadataValue::as_str)
        .unwrap_or("<missing general.architecture>");
    println!("general.architecture = {architecture}");

    let prefix = format!("{architecture}.");
    let mut matched = 0usize;
    for (key, value) in &parsed.metadata {
        if key.starts_with(&prefix) {
            println!("  {key} = {value:?}");
            matched += 1;
        }
    }
    println!("matched {matched} keys under prefix {prefix:?}");
    println!("tensor_count = {}", parsed.tensor_count);

    // `architecture_from_metadata`'s (`proxima-model-interop/src/bind.rs:334`)
    // own five hard-required keys, echoed back with whatever this file
    // resolved them to -- proves whether the bind path found them before
    // load, rather than inferring it from load succeeding or failing.
    for suffix in [
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "block_count",
    ] {
        let key = format!("{prefix}{suffix}");
        match parsed.metadata_value(&key) {
            Some(value) => println!("  required_key {key} = {value:?}"),
            None => println!("  required_key {key} = <MISSING>"),
        }
    }

    // dedup by tensor-name SUFFIX (strip the per-block `blk.N.` prefix) --
    // this is what shows which distinct tensor roles the checkpoint carries
    // without a per-architecture branch: the same generic dedup regardless
    // of what those roles turn out to be.
    let mut suffix_pattern: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for tensor in &parsed.tensors {
        let name = tensor.name.as_str();
        let generic = match name.find('.') {
            Some(first_dot) if name[..first_dot] == *"blk" => {
                match name[first_dot + 1..].find('.') {
                    Some(second_dot) => {
                        format!("blk.N.{}", &name[first_dot + 1 + second_dot + 1..])
                    }
                    None => name.to_string(),
                }
            }
            _ => name.to_string(),
        };
        suffix_pattern.insert(generic);
    }
    println!("distinct_tensor_name_patterns = {}", suffix_pattern.len());
    for pattern in &suffix_pattern {
        println!("  tensor_name_pattern: {pattern}");
    }

    for name in [
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ] {
        match parsed.tensors.iter().find(|tensor| tensor.name == name) {
            Some(tensor) => {
                println!(
                    "PROBE_DIMS {name} dims={:?} ggml_type={:?} offset={} bytes={:?}",
                    tensor.dims,
                    tensor.ggml_type,
                    tensor.offset,
                    tensor.nbytes()
                );
            }
            None => println!("PROBE_DIMS {name} MISSING"),
        }
    }

    let ssm_or_conv_tensors: Vec<&str> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .filter(|name| name.contains("ssm") || name.contains("conv"))
        .collect();
    println!("ssm_or_conv_tensor_count = {}", ssm_or_conv_tensors.len());
    for name in &ssm_or_conv_tensors {
        println!("  ssm_or_conv_tensor: {name}");
    }
}

fn supported_serving_config(model_path: &str, gpu_layers: i32) -> ServingConfig<'_> {
    let gpu_memory_limit_bytes = env::var("PROXIMA_GPU_MEMORY_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let residency_budget_bytes = env::var("PROXIMA_QWEN35MOE_RESIDENCY_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let qwen35moe_pre_gather = env::var("PROXIMA_QWEN35MOE_PRE_GATHER")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        || (env::var_os("PROXIMA_EXPERT_SIDECAR").is_some() && residency_budget_bytes > 0);
    ServingConfig {
        model_path,
        kv_cache_key_quant: proxima_gguf::types::GgmlType::F32,
        kv_cache_value_quant: proxima_gguf::types::GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        // caller-selected: 0 (cpu-only) or `GPU_LAYERS_ALL` (`-ngl all`,
        // whole-model offload onto `omega::backend::Engine::Gpu` --
        // `generate.rs:856`'s `select_backend` reads this exact sentinel).
        gpu_layers,
        gpu_memory_limit_bytes,
        qwen35moe_pre_gather,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

#[derive(Clone, Copy)]
enum RequestedBackend {
    Cpu,
    Gpu,
}

impl RequestedBackend {
    fn gpu_layers(self) -> i32 {
        match self {
            RequestedBackend::Cpu => 0,
            RequestedBackend::Gpu => GPU_LAYERS_ALL,
        }
    }
}

fn parse_requested_backend(argument: Option<&String>) -> RequestedBackend {
    match argument.map(String::as_str) {
        None | Some("gpu") => RequestedBackend::Gpu,
        Some("cpu") => RequestedBackend::Cpu,
        Some(other) => {
            eprintln!("argv[4] must be 'cpu' or 'gpu', got {other:?}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(gguf_path) = args.get(1) else {
        eprintln!("argv[1]: path to a .gguf checkpoint");
        std::process::exit(1);
    };
    let Some(prompt) = args.get(2) else {
        eprintln!("argv[2]: prompt string");
        std::process::exit(1);
    };
    let max_tokens: usize = match args.get(3) {
        Some(value) => match value.parse() {
            Ok(max_tokens) => max_tokens,
            Err(error) => {
                eprintln!("argv[3] must be a non-negative integer: {error}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!("argv[3]: max token count");
            std::process::exit(1);
        }
    };

    let requested_backend = parse_requested_backend(args.get(4));

    println!("gguf_path = {gguf_path}");
    println!("prompt = {prompt:?}");
    println!("max_tokens = {max_tokens}");
    println!(
        "requested_backend = {}",
        match requested_backend {
            RequestedBackend::Cpu => "cpu",
            RequestedBackend::Gpu => "gpu",
        }
    );

    // bind.rs's gguf_tensor_as_packed_block borrows quantized weights straight out of
    // this buffer for the model's whole lifetime, so it must stay file-backed and
    // kernel-reclaimable rather than a private anonymous heap copy.
    let gguf_file = match std::fs::File::open(gguf_path) {
        Ok(gguf_file) => gguf_file,
        Err(error) => {
            eprintln!("open the gguf file: {error}");
            std::process::exit(1);
        }
    };
    // SAFETY: the checkpoint file is not written or truncated by any process while this
    // mapping is alive for the duration of this run, so the mapped bytes stay valid.
    let file_map = match unsafe { memmap2::Mmap::map(&gguf_file) } {
        Ok(file_map) => file_map,
        Err(error) => {
            eprintln!("mmap the gguf file: {error}");
            std::process::exit(1);
        }
    };
    let file_bytes: &[u8] = &file_map;
    #[cfg(target_os = "macos")]
    if env::var_os("PROXIMA_MMAP_RANDOM").is_some() {
        // SAFETY: the mapping remains alive and read-only for this process.
        let result = unsafe {
            libc::madvise(
                file_bytes.as_ptr().cast_mut().cast(),
                file_bytes.len(),
                libc::MADV_RANDOM,
            )
        };
        if result != 0 {
            eprintln!("madvise checkpoint mapping random failed: {result}");
            return;
        }
        println!("checkpoint_mmap_random = true");
    }
    println!("file_bytes = {} bytes", file_bytes.len());

    let parse_started = Instant::now();
    let parsed = match parse_complete(file_bytes) {
        Ok(parsed) => parsed,
        Err(error) => {
            println!("GGUF PARSE FAILED: {error}");
            println!("GGUF PARSE FAILED (debug): {error:?}");
            return;
        }
    };
    let parse_ms = parse_started.elapsed().as_secs_f64() * 1000.0;
    println!("gguf_parse_ms = {parse_ms:.3}");

    print_architecture_metadata(&parsed);

    let load_started = Instant::now();
    let registry = ArchitectureRegistry::with_builtin();
    let mut model = match LoadedModel::load_with_registry(&parsed, file_bytes, &registry) {
        Ok(model) => model,
        Err(error) => {
            println!("WEIGHT LOAD FAILED: {error}");
            println!("WEIGHT LOAD FAILED (debug): {error:?}");
            return;
        }
    };
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    println!("weight_load_ms = {load_ms:.3}");

    if let Some(sidecar_path) = env::var_os("PROXIMA_EXPERT_SIDECAR") {
        let sidecar_file = match std::fs::File::open(&sidecar_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("open expert sidecar: {error}");
                return;
            }
        };
        // SAFETY: the sidecar is read-only for the lifetime of the model.
        let sidecar_mapping = match unsafe { memmap2::Mmap::map(&sidecar_file) } {
            Ok(mapping) => Arc::new(mapping),
            Err(error) => {
                eprintln!("mmap expert sidecar: {error}");
                return;
            }
        };
        #[cfg(target_os = "macos")]
        if env::var_os("PROXIMA_MMAP_RANDOM").is_some() {
            // SAFETY: the mapping remains alive and read-only for this process.
            let result = unsafe {
                libc::madvise(
                    sidecar_mapping.as_ptr().cast_mut().cast(),
                    sidecar_mapping.len(),
                    libc::MADV_RANDOM,
                )
            };
            if result != 0 {
                eprintln!("madvise sidecar mapping random failed: {result}");
                return;
            }
        }
        match model.attach_expert_sidecar(sidecar_mapping) {
            Ok(()) => println!("expert_sidecar_attached = true"),
            Err(error) => {
                eprintln!("attach expert sidecar: {error}");
                return;
            }
        }
    }

    #[cfg(target_os = "macos")]
    if env::var_os("PROXIMA_CHECKPOINT_DISCARD_BEFORE_GENERATE").is_some() {
        match omega::discard_checkpoint_mmap_range(file_bytes) {
            Ok(()) => println!("checkpoint_discard_before_generate = true"),
            Err(error) => {
                eprintln!("discard checkpoint mapping before generate: {error}");
                return;
            }
        }
    }

    println!(
        "backend_evidence_note = look for a 'token_breakdown_metal ... gpu_exec_calls=' line \
         per decode step below -- that is the mechanical proof Metal ran, not this flag"
    );

    // A standalone logits probe is a full forward pass.  Running it before
    // generation would allocate the complete expert stack on the GPU before
    // `ServingConfig`'s memory-fit gate can reject an over-budget request.
    // Keep it for an explicitly CPU diagnostic run only; GPU serving must go
    // straight through the guarded generation path.
    if matches!(requested_backend, RequestedBackend::Cpu)
        && env::var_os("PROXIMA_SKIP_LOGITS_PROBE").is_none()
    {
        match model.forward_logits(prompt) {
            Ok(logits) => {
                let len = logits.len();
                let nan_count = logits.iter().filter(|value| value.is_nan()).count();
                let inf_count = logits.iter().filter(|value| value.is_infinite()).count();
                let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = logits.iter().map(|value| f64::from(*value)).sum();
                let mean = sum / len as f64;
                let all_equal = logits.iter().all(|value| *value == logits[0]);
                let mut ranked: Vec<usize> = (0..len).collect();
                ranked.sort_by(|left, right| {
                    logits[*right]
                        .total_cmp(&logits[*left])
                        .then_with(|| left.cmp(right))
                });
                let top5: Vec<(usize, f32)> = ranked
                    .iter()
                    .take(5)
                    .map(|index| (*index, logits[*index]))
                    .collect();
                println!(
                    "PROBE logits_len={len} nan_count={nan_count} inf_count={inf_count} min={min} max={max} mean={mean} all_equal={all_equal} top5={top5:?}"
                );
            }
            Err(error) => println!("PROBE forward_logits FAILED: {error:?}"),
        }
    }
    let generate_started = Instant::now();
    let mut token_elapsed_ms = Vec::with_capacity(max_tokens);
    let mut on_token = |event: TokenEvent<'_>| {
        match event.phase {
            Phase::Prefill { prompt_tokens } => println!(
                "prefill_event prompt_tokens={prompt_tokens} step={} elapsed_ms={}",
                event.step, event.elapsed_ms
            ),
            Phase::Token => {
                println!(
                    "token_event token_id={} text_piece={:?} step={} elapsed_ms={}",
                    event.token_id, event.text_piece, event.step, event.elapsed_ms
                );
                token_elapsed_ms.push(event.elapsed_ms);
            }
        }
        Control::Continue
    };
    let (outcome, backend_label) = match requested_backend {
        RequestedBackend::Cpu => {
            // requested CPU: never attempts Metal, so there is nothing to
            // fall back from -- a CPU request always produces a CPU run.
            let config = supported_serving_config(gguf_path, RequestedBackend::Cpu.gpu_layers());
            let outcome = model.generate_streaming(prompt, max_tokens, config, &mut on_token);
            (outcome, "CPU (requested)")
        }
        RequestedBackend::Gpu => {
            let config = supported_serving_config(gguf_path, RequestedBackend::Gpu.gpu_layers());
            let mut outcome = model.generate_streaming(prompt, max_tokens, config, &mut on_token);
            let mut backend_label = "GPU/METAL (requested, gpu_layers = GPU_LAYERS_ALL)";
            if let Err(error) = &outcome {
                println!("METAL RUN FAILED: {error}");
                println!("falling back to CPU (gpu_layers = 0), labeled explicitly below");
                let cpu_config =
                    supported_serving_config(gguf_path, RequestedBackend::Cpu.gpu_layers());
                outcome = model.generate_streaming(prompt, max_tokens, cpu_config, &mut on_token);
                backend_label = "CPU (fallback: gpu was requested but the metal run failed, \
                                  see METAL RUN FAILED above -- gpu_exec_calls below is the \
                                  mechanical proof this did NOT run on gpu)";
            }
            (outcome, backend_label)
        }
    };
    let generate_ms = generate_started.elapsed().as_secs_f64() * 1000.0;
    println!("backend_requested = {backend_label}");

    match outcome {
        Ok((ids, text, stopped_by_eos)) => {
            let token_count = ids.len();
            let tokens_per_sec = if generate_ms > 0.0 {
                (token_count as f64) / (generate_ms / 1000.0)
            } else {
                0.0
            };
            println!("generate_ms = {generate_ms:.3}");
            println!("tokens_generated = {token_count}");
            println!("tokens_per_sec = {tokens_per_sec:.3}");
            println!("stopped_by_eos = {stopped_by_eos}");
            println!("generated_ids = {ids:?}");
            println!("generated_text = {text:?}");
            print_decode_latency(&token_elapsed_ms);
        }
        Err(error) => {
            println!("generate_ms = {generate_ms:.3}");
            println!("GENERATION FAILED: {error}");
            println!("GENERATION FAILED (debug): {error:?}");
        }
    }

    // device-byte census by upload path -- proves the checkpoint-mapping
    // no-copy path (`omega::metal::checkpoint_mapping_offset`) is what
    // replaced `upload_resident_copy`'s per-tensor device copy, not merely
    // a relabeling of the same bytes.
    println!(
        "nocopy_buffer_uploads = {}",
        omega::metal::NOCOPY_BUFFER_UPLOADS.get()
    );
    println!(
        "nocopy_buffer_reuses = {}",
        omega::metal::NOCOPY_BUFFER_REUSES.get()
    );
    println!(
        "mapping_offset_uploads = {}",
        omega::metal::MAPPING_OFFSET_UPLOADS.get()
    );
    println!(
        "resident_buffer_uploads = {}",
        omega::metal::RESIDENT_BUFFER_UPLOADS.get()
    );
    println!(
        "resident_buffer_reuses = {}",
        omega::metal::RESIDENT_BUFFER_REUSES.get()
    );
    println!(
        "copying_buffer_uploads = {}",
        omega::metal::COPYING_BUFFER_UPLOADS.get()
    );
    println!(
        "device_current_allocated_size = {:?}",
        omega::metal::current_allocated_size()
    );
    println!("nocopy_cache_len = {}", omega::metal::nocopy_cache_len());
    print_peak_rss();
}
