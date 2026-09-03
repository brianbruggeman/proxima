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
//! <max-tokens>`

use std::env;
use std::time::Instant;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::GPU_LAYERS_ALL;
use proxima_model_interop::LoadedModel;
use proxima_model_interop::ServingConfig;

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
    let mut suffix_pattern: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
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
    ] {
        match parsed.tensors.iter().find(|tensor| tensor.name == name) {
            Some(tensor) => println!(
                "PROBE_DIMS {name} dims={:?} ggml_type={:?}",
                tensor.dims, tensor.ggml_type
            ),
            None => println!("PROBE_DIMS {name} MISSING"),
        }
    }

    let ssm_or_conv_tensors: Vec<&str> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .filter(|name| name.contains("ssm") || name.contains("conv"))
        .collect();
    println!(
        "ssm_or_conv_tensor_count = {}",
        ssm_or_conv_tensors.len()
    );
    for name in &ssm_or_conv_tensors {
        println!("  ssm_or_conv_tensor: {name}");
    }
}

fn supported_serving_config(model_path: &str) -> ServingConfig<'_> {
    ServingConfig {
        model_path,
        kv_cache_key_quant: proxima_gguf::types::GgmlType::F32,
        kv_cache_value_quant: proxima_gguf::types::GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        // `-ngl all` (`serving.rs:55`): whole-model offload onto
        // `omega::backend::Backend::Metal` -- `generate.rs:544-551`'s
        // `select_backend` reads this exact sentinel.
        gpu_layers: GPU_LAYERS_ALL,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let gguf_path = args.get(1).expect("argv[1]: path to a .gguf checkpoint");
    let prompt = args.get(2).expect("argv[2]: prompt string");
    let max_tokens: usize = args
        .get(3)
        .expect("argv[3]: max token count")
        .parse()
        .expect("argv[3] must be a non-negative integer");

    println!("gguf_path = {gguf_path}");
    println!("prompt = {prompt:?}");
    println!("max_tokens = {max_tokens}");

    // bind.rs's gguf_tensor_as_packed_block borrows quantized weights straight out of
    // this buffer for the model's whole lifetime, so it must stay file-backed and
    // kernel-reclaimable rather than a private anonymous heap copy.
    let gguf_file = std::fs::File::open(gguf_path).expect("open the gguf file");
    // SAFETY: the checkpoint file is not written or truncated by any process while this
    // mapping is alive for the duration of this run, so the mapped bytes stay valid.
    let file_map = unsafe { memmap2::Mmap::map(&gguf_file).expect("mmap the gguf file") };
    let file_bytes: &[u8] = &file_map;
    println!("file_bytes = {} bytes", file_bytes.len());

    let parse_started = Instant::now();
    let parsed = match parse_complete(&file_bytes) {
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
    let model = match LoadedModel::load(&parsed, &file_bytes) {
        Ok(model) => model,
        Err(error) => {
            println!("WEIGHT LOAD FAILED: {error}");
            println!("WEIGHT LOAD FAILED (debug): {error:?}");
            return;
        }
    };
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    println!("weight_load_ms = {load_ms:.3}");

    println!(
        "backend_evidence_note = look for a 'token_breakdown_metal ... gpu_exec_calls=' line \
         per decode step below -- that is the mechanical proof Metal ran, not this flag"
    );

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
            ranked.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]).then_with(|| left.cmp(right)));
            let top5: Vec<(usize, f32)> = ranked.iter().take(5).map(|index| (*index, logits[*index])).collect();
            println!(
                "PROBE logits_len={len} nan_count={nan_count} inf_count={inf_count} min={min} max={max} mean={mean} all_equal={all_equal} top5={top5:?}"
            );
        }
        Err(error) => println!("PROBE forward_logits FAILED: {error:?}"),
    }
    let generate_started = Instant::now();
    let mut outcome =
        model.generate_with_serving_config(prompt, max_tokens, supported_serving_config(gguf_path));
    let mut backend_label = "METAL (gpu_layers = GPU_LAYERS_ALL)";

    if let Err(error) = &outcome {
        println!("METAL RUN FAILED: {error}");
        println!("falling back to CPU (gpu_layers = 0), labeled explicitly below");
        let mut cpu_config = supported_serving_config(gguf_path);
        cpu_config.gpu_layers = 0;
        outcome = model.generate_with_serving_config(prompt, max_tokens, cpu_config);
        backend_label = "CPU (fallback: metal run failed, see METAL RUN FAILED above)";
    }
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
}
